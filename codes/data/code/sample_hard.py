"""
Generate a curated code dataset.

Pipeline:
- Normalize problems from HumanEval and OPC.
- Filter out problems the student already solves (pass@k ≥ threshold).
- Keep only problems where the teacher passes the tests.
- Save curated set to JSON, supporting resume.

Notes
-----
- Hydra + OmegaConf (flat config). No argparse, no dotenv.
- Resumable: reads an existing JSON (new structured or legacy list-only) and
  continues without duplicating work.
- Token usage is tracked approximately from generation utilities.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Set, Tuple, TypedDict

import hydra
from datasets import load_dataset
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from openai import OpenAI
from tqdm import tqdm

from codes.evaluation.code import evaluate_code_completion
from codes.evaluation.code import generate_code_completions
from codes.evaluation.utils import compute_pass_at_k
from codes.llm_utils import build_client

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# ---------------------------------------------------------------------
# MACROS
# ---------------------------------------------------------------------
GOOD_PROBLEM = 1
TEACHER_BLACKLIST = 2
STUDENT_BLACKLIST = 3


# ---------------------------------------------------------------------
# Typing helpers
# ---------------------------------------------------------------------


class ProblemDict(TypedDict, total=False):
    """Internal normalized problem format."""
    id: int
    prompt: str
    solution: str
    entry_point: str
    test: str


class ProcessSummary(TypedDict):
    """Return value shape of the main curation routine."""
    count: int
    tokens: Dict[str, int]  # {"prompt": int, "completion": int}
    output: str


# ---------------------------------------------------------------------
# Problem normalization
# ---------------------------------------------------------------------


def organize_humaneval_problem(problem_dict: Dict[str, Any]) -> ProblemDict:
    """
    Normalize a HumanEval problem into the internal format.

    Parameters
    ----------
    problem_dict : Dict[str, Any]
        A single item from the "openai_humaneval" dataset.

    Returns
    -------
    ProblemDict
        {
          "id": int,
          "prompt": str,
          "solution": str,      # prompt + canonical_solution
          "entry_point": str,   # function name
          "test": str           # unit test code (ends with check(entry_point))
        }
    """
    id_ = int(problem_dict["task_id"].split("/")[-1])
    prompt = problem_dict["prompt"]
    solution = prompt + problem_dict["canonical_solution"]
    entry_point = problem_dict["entry_point"]
    test_code = problem_dict["test"]
    if not test_code.endswith("\n"):
        test_code += "\n"
    test_code += f"check({entry_point})"
    return {
        "id": id_,
        "prompt": prompt,
        "solution": solution,
        "entry_point": entry_point,
        "test": test_code,
    }


def organize_opc_problem(problem_dict: Dict[str, Any]) -> ProblemDict:
    """
    Normalize an OPC (OpenCoder) item into the internal format.

    Parameters
    ----------
    problem_dict : Dict[str, Any]
        A single item from OpenCoder-LLM/opc-sft-stage2 (educational_instruct).

    Returns
    -------
    ProblemDict
        {
          "id": int,
          "prompt": str,        # instruction converted into a single-line task
          "solution": str,      # reference code
          "entry_point": str,   # function name
          "test": str           # joined assertions from provided testcases
        }
    """
    id_ = int(problem_dict["seq_id"])
    solution = problem_dict["code"]
    entry_point = problem_dict["entry_point"]
    testcases = problem_dict.get("testcase", []) or []
    first_tc = testcases[0] if testcases else "assert True"

    # Build a concise instruction prompt; keep only one function definition.
    prompt = (
        "# "
        + problem_dict["instruction"].replace("\n", " ")
        + f" Function should be titled: {entry_point}, and pass this test case: {first_tc}."
          " The response should only include one function definition. "
          "Infer the correct input arguments and output type from test case example.\n"
    )
    test_code = "\n".join(testcases) if testcases else "assert True"
    return {
        "id": id_,
        "prompt": prompt,
        "solution": solution,
        "entry_point": entry_point,
        "test": test_code,
    }


def organize_mbpp_problem(problem_dict: Dict[str, Any]) -> ProblemDict:
    """
    Normalize an OPC (OpenCoder) item into the internal format.

    Parameters
    ----------
    problem_dict : Dict[str, Any]
        A single item from OpenCoder-LLM/opc-sft-stage2 (educational_instruct).

    Returns
    -------
    ProblemDict
        {
          "id": int,
          "prompt": str,        # instruction converted into a single-line task
          "solution": str,      # reference code
          "entry_point": str,   # function name
          "test": str           # joined assertions from provided testcases
        }
    """
    id_ = int(problem_dict["task_id"])
    solution = problem_dict["code"]
    entry_point = solution.split("def")[1].split("(")[0].strip()
    testcases = problem_dict.get("testlist", []) or []
    first_tc = testcases[0] if testcases else "assert True"

    # Build a concise instruction prompt; keep only one function definition.
    prompt = (
        "# "
        + problem_dict["prompt"].replace("\n", " ")
        + f" Function should be titled: {entry_point}, and pass this test case: {first_tc}."
          " The response should only include one function definition. "
          "Infer the correct input arguments and output type from test case example.\n"
    )
    test_code = "\n".join(testcases) if testcases else "assert True"
    return {
        "id": id_,
        "prompt": prompt,
        "solution": solution,
        "entry_point": entry_point,
        "test": test_code,
    }


# ---------------------------------------------------------------------
# Safe evaluation wrapper
# ---------------------------------------------------------------------


def _safe_eval_code(args: Tuple[str, ProblemDict, bool, int]) -> Tuple[Any, bool]:
    """
    Safe wrapper for evaluate_code_completion to prevent executor crashes.

    Args
    ----
    args: (completion_code, problem_dict, sanity_check, timeout_sec)

    Returns
    -------
    (info, passed)
      - info: evaluator return (arbitrary) or {"error": "..."} on failure/timeout
      - passed: bool
    """
    code, problem, sanity_check, timeout_sec = args
    try:
        # Try passing a timeout kwarg if the evaluator supports it.
        try:
            return evaluate_code_completion(
                code, problem, sanity_check=sanity_check, timeout=timeout_sec  # type: ignore[call-arg]
            )
        except TypeError:
            # Fallback: evaluator has no timeout kwarg
            return evaluate_code_completion(code, problem, sanity_check=sanity_check)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout>{timeout_sec}s"}, False
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}, False


# ---------------------------------------------------------------------
# Main curation routine
# ---------------------------------------------------------------------
def process_problem(
    problem: ProblemDict,
    teacher_client: OpenAI,
    teacher_name: str,
    teacher_temperature: float,
    student_client: OpenAI,
    student_name: str,
    student_temperature: float,
    k_values: List[int],
    num_samples: int,
    pass_k: int,
    pass_at_k_threshold: float
):
    timeout_sec = int(os.getenv("EVAL_TIMEOUT_SEC", "12"))              # bump default from 3s → 12s

    # Function returns 1 for successful case, 2 for teacher blacklist, 3 for student blacklist
    pid = int(problem["id"])

    # 1) Sanity check ground truth / tests
    _, gt_ok = _safe_eval_code(("", problem, True, timeout_sec))
    if not gt_ok:
        # If tests/solution are broken, blacklist and skip
        logger.debug("Sanity check failed; blacklisting pid=%d", pid)
        return TEACHER_BLACKLIST, 0, 0

    # 2) Probe student model with N completions and compute pass@k
    completions, p_tks, c_tks = generate_code_completions(
        previous_conversation_str="",
        client=student_client,
        model_name=student_name,
        temperature=student_temperature,
        problem_dict=problem,   # API expects problem_dict for code path
        num_samples=num_samples,
        max_tokens=2048,
        show_gt=False
    )

    # Evaluate each completion in parallel (safe wrapper)
    results: List[bool] = []
    timeouts = errors = 0
    with ThreadPoolExecutor() as ex:
        futs = [ex.submit(_safe_eval_code, (comp, problem, False, timeout_sec)) for comp in completions]
        for fut in as_completed(futs):
            info, passed = fut.result()
            if isinstance(info, dict) and "error" in info:
                if "timeout" in info["error"]:
                    timeouts += 1
                else:
                    errors += 1
            results.append(bool(passed))
    if timeouts or errors:
        logger.warning("[student eval pid=%d] timeouts=%d, errors=%d", pid, timeouts, errors)

    scores = compute_pass_at_k(results, k_values)

    # If student already reaches threshold at pass_k, we do NOT keep it
    if scores[pass_k] >= pass_at_k_threshold:
        logger.debug("Student passes threshold for pid=%d (pass@%d=%.3f)", pid, pass_k, scores[pass_k])
        return STUDENT_BLACKLIST, p_tks, c_tks

    # 3) Probe teacher once; keep only if it passes tests
    t_texts, tp, tc = generate_code_completions(
        previous_conversation_str="",
        client=teacher_client,
        model_name=teacher_name,
        temperature=teacher_temperature,
        problem_dict=problem,
        num_samples=1,
        max_tokens=2048,
        show_gt=False,
    )

    t_code = t_texts[0] if t_texts else ""
    info, teacher_pass = _safe_eval_code((t_code, problem, False, timeout_sec))
    if isinstance(info, dict) and "error" in info:
        logger.warning("[teacher eval pid=%d] %s", pid, info["error"])
    if not teacher_pass:
        logger.debug("Teacher failed pid=%d; added to blacklist", pid)
        return TEACHER_BLACKLIST, p_tks + tp, c_tks + tc

    return GOOD_PROBLEM, p_tks + tp, c_tks + tc


# ---------------------------------------------------------------------
# Hydra entrypoint (flat config)
# ---------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../../../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """
    Hydra entrypoint for code data curation.

    Expected config keys (illustrative):
    - paths.openai_api_key_fpath : str
    - paths.data                 : str
    - teacher.model              : str
    - teacher.temperature        : float
    - student.model              : str
    - student.temperature        : float
    - experiment.dataset         : str
    - gen.k_values               : List[int]
    - gen.num_samples            : int
    - gen.pass_k                 : int
    - gen.pass_at_k_threshold    : float
    """
    # Read API key path and export for the SDK
    key_path = to_absolute_path(str(cfg.paths.openai_api_key_fpath))
    with open(key_path, "r") as f:
        openai_key = f.read().strip()
    os.environ["OPENAI_API_KEY"] = openai_key

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(g) for g in cfg.train.gpus])

    # Build teacher and student clients (names may be normalized inside)
    teacher_name, teacher_client = build_client(
        cfg.teacher.fpath if cfg.teacher.fpath else str(cfg.teacher.model)
    )
    student_name, student_client = build_client(
        cfg.student.fpath if cfg.student.fpath else str(cfg.student.model)
    )
    logger.info("Teacher=%s | Student=%s", teacher_name, student_name)

    # Output path: data/<teacher.model>/<dataset>/<student.model>.json
    teacher_dir = os.path.join(cfg.paths.data, cfg.teacher.model, cfg.experiment.dataset)
    os.makedirs(teacher_dir, exist_ok=True)
    output_fpath = os.path.join(teacher_dir, f"test_{cfg.student.model}.json")

    # Resume (supports new dict format and legacy list format)
    collected: List[ProblemDict] = []
    teacher_blacklist: Set[int] = set()
    student_ids: Set[int] = set()

    if os.path.exists(output_fpath):
        with open(output_fpath) as f:
            data = json.load(f)
        if isinstance(data, list):
            # Legacy format: only the collected problems
            collected = data  # type: ignore[assignment]
        else:
            # New structured format
            collected = data.get("collected", [])
            teacher_blacklist = set(data.get("teacher_blacklist", []))
            student_ids = set(data.get("student_ids", []))
        logger.info(
            "Resuming from %d problems in %s | teacher_blacklist=%d, student_processed=%d",
            len(collected), output_fpath, len(teacher_blacklist), len(student_ids)
        )

    num_samples = min(cfg.gen.num_samples, max(list(cfg.gen.k_values)))         # ensure pass@k is defined

    # Load datasets (HE test split; OPC sample)
    datasets = []
    if cfg.experiment.dataset == "humaneval_opc":
        he_dataset = load_dataset("openai_humaneval")["test"]
        datasets.append((he_dataset, organize_humaneval_problem))

        opc_dataset = (
            load_dataset("OpenCoder-LLM/opc-sft-stage2", split="train", name="educational_instruct")
            .shuffle(seed=2025)
        )
        datasets.append((opc_dataset, organize_opc_problem))

    elif cfg.experiment.dataset == "mbpp":
        mbpp_dataset = load_dataset("Muennighoff/mbpp", "sanitized")["test"]
        datasets.append((mbpp_dataset, organize_mbpp_problem))
    
    else:
        raise ValueError(f"dataset for code type should be one of humaneval_opc/mbpp")

    # total prompt and completion tokens
    total_prompt_tks, total_completion_tks = 0, 0

    # Parallel execution
    with ThreadPoolExecutor(max_workers=cfg.experiment.num_workers) as executor:
        future_to_pid = {}
        for dataset, normalizer in datasets:
            for raw in dataset:
                # Normalize problem into internal structure
                problem: ProblemDict = normalizer(raw)

                if problem["id"] in teacher_blacklist or problem["id"] in student_ids:
                    continue

                future_to_pid[
                    executor.submit(
                        process_problem,
                        problem,
                        teacher_client,
                        teacher_name,
                        cfg.teacher.temperature,
                        student_client,
                        student_name,
                        cfg.student.temperature,
                        list(cfg.gen.k_values),
                        num_samples,
                        int(cfg.gen.pass_k),
                        float(cfg.gen.pass_at_k_threshold)
                    )
                ] = problem

        for future in tqdm(as_completed(future_to_pid), total=len(future_to_pid), desc="Curating"):
            result, p_tks, c_tks = future.result()
            problem = future_to_pid[future]

            if result == GOOD_PROBLEM:
                collected.append(problem)
                student_ids.add(problem["id"])

            elif result == STUDENT_BLACKLIST:
                student_ids.add(problem["id"])

            elif result == TEACHER_BLACKLIST:
                teacher_blacklist.add(problem["id"])

            total_prompt_tks += p_tks
            total_completion_tks += c_tks

            if len(collected) >= cfg.data.data_amount:
                # Cancel pending futures
                for fut in future_to_pid:
                    fut.cancel()
                break

    # Final save
    with open(output_fpath, "w") as f:
        json.dump(
            {
                "collected": collected,
                "teacher_blacklist": sorted(teacher_blacklist),
                "student_ids": sorted(student_ids),
            },
            f,
            indent=2,
        )

    logger.info("Saved %d problems to %s", len(collected), output_fpath)
    logger.info("Token usage (approx): prompt=%d, completion=%d", total_prompt_tks, total_completion_tks)


if __name__ == "__main__":
    # Enable full Hydra tracebacks for debugging
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
