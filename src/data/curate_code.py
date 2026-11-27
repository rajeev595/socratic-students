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

import os
import time
import logging
import subprocess
from random import shuffle
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Set, Tuple, TypedDict

import hydra
from datasets import load_dataset
from omegaconf import DictConfig
from openai import OpenAI
from tqdm import tqdm

from codes.evaluation.code import evaluate_code_completion
from codes.evaluation.code import generate_code_completions
from codes.evaluation.utils import compute_pass_at_k
from codes.utils.llm.server import build_client
from codes.utils.vars import set_envs
from codes.utils.file_ops import save_json_atomic, load_json_resilient

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
    num_samples: int
):
    timeout_sec = int(os.getenv("EVAL_TIMEOUT_SEC", "12"))              # bump default from 3s → 12s

    # Function returns 1 for successful case, 2 for teacher blacklist, 3 for student blacklist
    pid = int(problem["id"])

    # 1) Sanity check ground truth / tests
    _, gt_ok = _safe_eval_code(("", problem, True, timeout_sec))
    if not gt_ok:
        # If tests/solution are broken, blacklist and skip
        logger.debug("Sanity check failed; blacklisting pid=%d", pid)
        return TEACHER_BLACKLIST, {}, 0, 0

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
    with ThreadPoolExecutor(max_workers=4) as ex:
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
        return TEACHER_BLACKLIST, {}, p_tks + tp, c_tks + tc

    return GOOD_PROBLEM, scores, p_tks + tp, c_tks + tc


def count_examples(data):
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ks = ["1", "5", "10"]

    # Initialize counters
    counts = {k: {th: 0 for th in thresholds} for k in ks}

    for ex in data:
        for k in ks:
            try:
                score = ex["scores"][k]
            except:
                score = ex["scores"][int(k)]

            for th in thresholds:
                if score < th:
                    counts[k][th] += 1

    # Pretty print
    for k in ks:
        logger.info(f"Pass@{k}:")
        for th in thresholds:
            logger.info(f"  < {th}: {counts[k][th]}\n" if th == thresholds[-1] else f"  < {th}: {counts[k][th]}")


def save_train_test_val_split(
    data,
    train_size: float,
    val_size: float,
    dir: str,
    name: str,
    pass_k: int,
    pass_at_k_threshold: float
):
    # Sample examples with pass_at_k_threshold
    hard = []
    for d in data:
        try:
            s = d["scores"][str(pass_k)]
        except:
            s = d["scores"][int(pass_k)]
        
        if s < pass_at_k_threshold:
            hard.append(d["problem"])

    if len(hard) >= 10000:
        hard = hard[: len(hard)//10]

    # Shuffle before split to train-test
    shuffle(hard)
    total_num = len(hard)
    num_train = int(train_size * total_num)
    num_val = int(val_size * total_num)

    # Save train examples
    train_fpath = os.path.join(dir, f"splits_{pass_at_k_threshold}", f"{name}_train.json")
    save_json_atomic(
        path=train_fpath,
        obj=hard[: num_train],
        indent=2
    )
    logger.info("Train-split: %d problems at %s", num_train, train_fpath)

    # Save val examples
    val_fpath = os.path.join(dir, f"splits_{pass_at_k_threshold}", f"{name}_val.json")
    save_json_atomic(
        path=val_fpath,
        obj=hard[num_train: num_train + num_val],
        indent=2
    )
    logger.info("Val-split: %d problems at %s", num_val, val_fpath)

    # Save test examples
    test_fpath = os.path.join(dir, f"splits_{pass_at_k_threshold}", f"{name}_test.json")
    save_json_atomic(
        path=test_fpath,
        obj=hard[num_train + num_val: ],
        indent=2
    )
    logger.info("Test-split: %d problems at %s", total_num - num_train - num_val, test_fpath)


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
    start_time = time.perf_counter()  # ---- start timer ----

    # Set OpenAI API key and HF HOME envs
    set_envs(cfg.paths.openai_api_key_fpath, cfg.paths.hf_home)

    # Build teacher and student clients (names may be normalized inside)
    teacher_name, student_name = cfg.teacher.model, cfg.student.model
    teacher_client = build_client(
        model=teacher_name, gpus=cfg.teacher.gpus, server=cfg.teacher.server
    )
    student_client = build_client(
        model=student_name, gpus=cfg.student.gpus, server=cfg.student.server
    )
    logger.info("Teacher=%s | Student=%s", teacher_name, student_name)

    # Output path: data/<teacher.model>/<dataset>/<student.model>.json
    teacher_dir = os.path.join(cfg.paths.data, cfg.teacher.model, cfg.experiment.dataset)
    os.makedirs(teacher_dir, exist_ok=True)
    output_fpath = os.path.join(teacher_dir, f"{cfg.student.model}.json")

    # Resume (supports new dict format and legacy list format)
    collected: List[ProblemDict] = []
    teacher_blacklist: Set[int] = set()
    student_ids: Set[int] = set()

    # No. of examples with higher pass@k rate
    num_passk_examples = 0

    if os.path.exists(output_fpath):
        data = load_json_resilient(output_fpath)

        collected = data.get("collected", [])
        teacher_blacklist = set(data.get("teacher_blacklist", []))
        student_ids = set(data.get("student_ids", []))
        logger.info(
            "Resuming from %d problems in %s | teacher_blacklist=%d, student_processed=%d",
            len(collected), output_fpath, len(teacher_blacklist), len(student_ids)
        )

        for ex in collected:
            try:
                if ex["scores"][cfg.gen.pass_k] < cfg.gen.pass_at_k_threshold:
                    num_passk_examples += 1
            except:
                if ex["scores"][str(cfg.gen.pass_k)] < cfg.gen.pass_at_k_threshold:
                    num_passk_examples += 1

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

    # Collect all problems
    problems = []
    for dataset, normalizer in datasets:
        for raw in dataset:
            # Normalize problem into internal structure
            problem: ProblemDict = normalizer(raw)
            if problem["id"] in teacher_blacklist or problem["id"] in student_ids:
                continue
            problems.append(problem)

    pbar = tqdm(total=len(problems), desc="Problems done", dynamic_ncols=True)

    # Parallel execution
    with ThreadPoolExecutor(max_workers=cfg.experiment.num_workers) as executor:
        for i in range(0, len(problems), cfg.experiment.chunk_size):
            if num_passk_examples >= cfg.data.data_amount:
                break

            # Problem chunks in this iter
            problems_chunk = problems[i: i + cfg.experiment.chunk_size]

            # Prepare jobs
            future_to_pid = {}
            for problem in problems_chunk:
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
                        cfg.gen.num_samples
                    )
                ] = problem

            for future in as_completed(future_to_pid):
                result, scores, p_tks, c_tks = future.result()
                problem = future_to_pid[future]

                if result == GOOD_PROBLEM:
                    collected.append({"problem": problem, "scores": scores})
                    student_ids.add(problem["id"])
                    try:
                        if scores[cfg.gen.pass_k] < cfg.gen.pass_at_k_threshold:
                            num_passk_examples += 1
                    except:
                        if scores[str(cfg.gen.pass_k)] < cfg.gen.pass_at_k_threshold:
                            num_passk_examples += 1

                elif result == TEACHER_BLACKLIST:
                    teacher_blacklist.add(problem["id"])

                total_prompt_tks += p_tks
                total_completion_tks += c_tks
                pbar.update(1)
                logger.info("Collected examples: total %d | passed: %d/%d samples!", len(collected), num_passk_examples, cfg.data.data_amount)                
                logger.info("%s", pbar.format_meter(n=pbar.n, total=pbar.total, elapsed=pbar.format_dict["elapsed"], prefix=pbar.desc + ": "))

            save_json_atomic(
                output_fpath, 
                {
                    "collected": collected,
                    "teacher_blacklist": list(teacher_blacklist),
                    "student_ids": list(student_ids)
                },
                indent=2
            )

    logger.info("Total problems done: %d/%d", pbar.n, len(problems))
    logger.info("Saved examples: total %d | passed %d to %s", len(collected), num_passk_examples, output_fpath)
    logger.info("Token usage (approx): prompt=%d, completion=%d", total_prompt_tks, total_completion_tks)
    pbar.close()

    # Count the no. of examples
    count_examples(collected)

    # Train-test split
    save_train_test_val_split(
        data=collected,
        train_size=cfg.train.train_size,
        val_size=cfg.train.val_size,
        dir=teacher_dir,
        name=student_name,
        pass_k=cfg.gen.pass_k,
        pass_at_k_threshold=cfg.gen.pass_at_k_threshold
    )

    elapsed = time.perf_counter() - start_time
    logger.info("Total time elapsed: %s", timedelta(seconds=int(elapsed)))


if __name__ == "__main__":
    # Enable full Hydra tracebacks for debugging
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
