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

from codes.evaluation.code import evaluate_code_completion, generate_code_completions
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


def process_code(
    teacher_client: OpenAI,
    teacher_name: str,
    teacher_temperature: float,
    student_client: OpenAI,
    student_name: str,
    student_temperature: float,
    k_values: List[int],
    num_samples: int,
    data_amount: int,
    pass_k: int,
    pass_at_k_threshold: float,
    opc_select_n: int,
    opc_seed: int,
    output_fpath: str,
) -> ProcessSummary:
    """
    Build a curated code dataset via student/teacher probing.

    Steps
    -----
    1) Normalize problems from HumanEval and OPC.
    2) Sanity-check GT tests via `evaluate_code_completion("", problem, sanity_check=True)`.
    3) Probe student with up to `num_samples` completions; compute pass@k.
       - If pass@pass_k >= threshold: skip (student already solves it).
    4) Otherwise, probe teacher once; keep the problem only if teacher passes tests.
    5) Save/Resume from `output_fpath` periodically and at the end.

    Returns
    -------
    ProcessSummary
        {
          "count": int,
          "tokens": {"prompt": int, "completion": int},
          "output": str
        }
    """
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

    total_prompt_tks, total_completion_tks = 0, 0
    num_samples = min(num_samples, max(k_values))  # ensure pass@k is defined
    timeout_sec = int(os.getenv("EVAL_TIMEOUT_SEC", "12"))  # bump default from 3s → 12s
    max_workers = os.cpu_count() or 1

    # Load datasets (HE test split; OPC sample)
    he_dataset = load_dataset("openai_humaneval")["test"]
    opc_dataset = (
        load_dataset("OpenCoder-LLM/opc-sft-stage2", split="train", name="educational_instruct")
        .shuffle(seed=opc_seed)
        .select(range(opc_select_n))
    )

    # Iterate across HumanEval first, then OPC (order can be changed if desired)
    for dataset, normalizer in [(he_dataset, organize_humaneval_problem), (opc_dataset, organize_opc_problem)]:
        for raw in tqdm(dataset, desc="Curating"):
            if len(collected) >= data_amount:
                break

            # Normalize problem into internal structure
            problem: ProblemDict = normalizer(raw)
            pid = int(problem["id"])

            # Skip previously handled problems
            if pid in student_ids or pid in teacher_blacklist:
                continue

            # 1) Sanity check ground truth / tests
            _, gt_ok = evaluate_code_completion("", problem, sanity_check=True)
            if not gt_ok:
                # If tests/solution are broken, blacklist and skip
                teacher_blacklist.add(pid)
                logger.debug("Sanity check failed; blacklisting pid=%d", pid)
                continue

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
            total_prompt_tks += p_tks
            total_completion_tks += c_tks

            # Evaluate each completion in parallel (safe wrapper)
            results: List[bool] = []
            timeouts = errors = 0
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
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
                student_ids.add(pid)
                logger.debug("Student passes threshold for pid=%d (pass@%d=%.3f)", pid, pass_k, scores[pass_k])
                continue

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
            total_prompt_tks += tp
            total_completion_tks += tc

            t_code = t_texts[0] if t_texts else ""
            info, teacher_pass = _safe_eval_code((t_code, problem, False, timeout_sec))
            if isinstance(info, dict) and "error" in info:
                logger.warning("[teacher eval pid=%d] %s", pid, info["error"])
            if not teacher_pass:
                teacher_blacklist.add(pid)
                logger.debug("Teacher failed pid=%d; added to blacklist", pid)
                continue

            # Keep this problem
            collected.append(problem)
            student_ids.add(pid)

            # Periodic checkpoint every 10 kept problems
            if len(collected) % 10 == 0:
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
                logger.info("Checkpoint saved: kept=%d -> %s", len(collected), output_fpath)

        if len(collected) >= data_amount:
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
    - data.data_amount           : int
    - data.opc_select_n          : int
    - data.opc_seed              : int
    """
    # Read API key path and export for the SDK
    key_path = to_absolute_path(str(cfg.paths.openai_api_key_fpath))
    with open(key_path, "r") as f:
        openai_key = f.read().strip()
    os.environ["OPENAI_API_KEY"] = openai_key

    # Build teacher and student clients (names may be normalized inside)
    teacher_name, teacher_client = build_client(str(cfg.teacher.model))
    student_name, student_client = build_client(str(cfg.student.model))
    logger.info("Teacher=%s | Student=%s", teacher_name, student_name)

    # Output path: data/<teacher.model>/<dataset>/<student.model>.json
    teacher_dir = os.path.join(cfg.paths.data, cfg.teacher.model, cfg.experiment.dataset)
    os.makedirs(teacher_dir, exist_ok=True)
    output_fpath = os.path.join(teacher_dir, f"{cfg.student.model}.json")

    # Run the curation pipeline
    process_code(
        teacher_client=teacher_client,
        teacher_name=teacher_name,
        teacher_temperature=float(cfg.teacher.temperature),
        student_client=student_client,
        student_name=student_name,
        student_temperature=float(cfg.student.temperature),
        k_values=list(cfg.gen.k_values),
        num_samples=int(cfg.gen.num_samples),
        data_amount=int(cfg.data.data_amount),
        pass_k=int(cfg.gen.pass_k),
        pass_at_k_threshold=float(cfg.gen.pass_at_k_threshold),
        opc_select_n=int(cfg.data.opc_select_n),
        opc_seed=int(cfg.data.opc_seed),
        output_fpath=output_fpath,
    )


if __name__ == "__main__":
    # Enable full Hydra tracebacks for debugging
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
