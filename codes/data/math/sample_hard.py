"""
Create curated math datasets (GSM-style).
Hydra + OmegaConf (flat config). No argparse, no dotenv. Self-contained.

Pipeline:
  1) Stream GSM-like problems from a dataset
  2) Probe student model with N samples -> compute pass@k (k=1)
  3) Keep only problems where student struggles (pass@1 < threshold)
  4) Verify teacher model solves it
  5) Save problems + maintain IDs (GPT blacklist + student-kept ids)
"""

from __future__ import annotations

import os
import time
import logging
from datetime import timedelta
from typing import Any, Dict, List, Set, TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import json
import hydra
from datasets import load_dataset
from omegaconf import DictConfig
from openai import OpenAI
from tqdm import tqdm

from codes.evaluation.math import generate_math_completions, evaluate_math_completion
from codes.evaluation.utils import compute_pass_at_k
from codes.utils.llm.server import build_client
from codes.utils.vars import set_envs
from codes.data.code.sample_hard import count_examples, save_train_test_split
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

class MathProblem(TypedDict, total=False):
    """Internal GSM-style math problem format."""
    id: int
    prompt: str
    solution: str


# =========================
# Row normalization
# =========================

def organize_gsm_row(row: Dict[str, Any]) -> MathProblem:
    """
    Normalize a GSM-style dataset row to the internal problem format.

    Parameters
    ----------
    row : Dict[str, Any]
        Input row with fields like {"id": "123-0", "question": str, "answer": str}.

    Returns
    -------
    MathProblem
        {"id": int, "prompt": str, "solution": str}
    """
    # Original logic: id like "123-0" and assert alt==0
    if "-" in row["id"]:
        pid_str, alt_str = row["id"].split("-")
        pid, alt = int(pid_str), int(alt_str)
        assert alt == 0
    else:
        pid = int(row["id"])

    return {"id": pid, "prompt": row["question"], "solution": row["answer"]}


def organize_math_row(row: Dict[str, Any]) -> MathProblem:
    # Original logic: id like "123-0" and assert alt==0
    pid_str, alt_str = row["id"].split("-")
    pid, alt = int(pid_str), int(alt_str)
    assert alt == 0
    return {"id": pid, "prompt": row["problem"], "solution": row["solution"]}


# =========================
# Main curation
# =========================
def process_problem(
    problem: MathProblem,
    teacher_client: OpenAI,
    teacher_name: str,
    teacher_temperature: float,
    student_client: OpenAI,
    student_name: str,
    student_temperature: float,
    k_values: List[int],
    num_samples: int
):
    pid = int(problem["id"])

    # -------------------------
    # 1) Probe student model
    # -------------------------
    completions, p, c = generate_math_completions(
        previous_conversation_str="",
        client=student_client,
        model_name=student_name,
        problem=problem,
        num_samples=num_samples,
        temperature=student_temperature,
        show_gt=False,
        is_ood=False,
    )

    # Evaluate student results
    results: List[bool] = []
    for comp in completions:
        _, ok = evaluate_math_completion(
            model_response=comp,
            problem=problem,
            client=teacher_client,
            model_name=teacher_name,
            temperature=0.0
        )
        results.append(ok)

    scores = compute_pass_at_k(results, k_values)

    # -------------------------
    # 2) Probe teacher model
    # -------------------------
    t_comps, tp, tc = generate_math_completions(
        previous_conversation_str="",
        client=teacher_client,
        model_name=teacher_name,
        problem=problem,
        num_samples=1,
        temperature=teacher_temperature,
        show_gt=False,
        is_ood=False,
    )

    _, teacher_pass = evaluate_math_completion(
        model_response=t_comps[0] if t_comps else "",
        problem=problem,
        client=teacher_client,
        model_name=teacher_name,
        temperature=0.0
    )

    if not teacher_pass:
        logger.debug("Teacher failed pid=%d; added to blacklist", pid)
        return TEACHER_BLACKLIST, {}, p + tp, c + tc

    return GOOD_PROBLEM, scores, p + tp, c + tc


# =========================
# Hydra entry
# =========================

@hydra.main(version_base=None, config_path="../../../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """
    Hydra entrypoint to curate GSM-style math problems.

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
    - gen.pass_at_k_threshold    : float
    - gen.take_every             : int
    - data.data_amount           : int
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

    # Output path under data/<teacher.model>/<dataset>/<student.model>.json
    teacher_dir = os.path.join(cfg.paths.data, cfg.teacher.model, cfg.experiment.dataset)
    os.makedirs(teacher_dir, exist_ok=True)
    output_fpath = os.path.join(teacher_dir, f"{cfg.student.model}.json")

    collected: List[MathProblem] = []
    teacher_blacklist: Set[int] = set()
    student_ids: Set[int] = set()

    # Resume from prior run if file exists (supports legacy list-only format)
    if os.path.exists(output_fpath):
        data = load_json_resilient(output_fpath)

        collected = data.get("collected", [])
        teacher_blacklist = set(data.get("teacher_blacklist", []))
        student_ids = set(data.get("student_ids", []))
        logger.info(
            "Resuming from %d problems | blacklist=%d | student_seen=%d | path=%s",
            len(collected), len(teacher_blacklist), len(student_ids), output_fpath
        )

    # Dataset (GSM-like). Adjust split/name as needed.
    gsm_ds = load_dataset("openai/gsm8k", "main", split="train")
    if "id" not in gsm_ds.column_names:
        gsm_ds = gsm_ds.add_column("id", [str(i) for i in range(len(gsm_ds))])

    total_prompt_tks = 0
    total_completion_tks = 0
    # Parallel execution
    with ThreadPoolExecutor(max_workers=cfg.experiment.num_workers) as executor:
        future_to_pid = {}
        for idx, row in enumerate(gsm_ds):
            # Optionally subsample rows (e.g., take every Nth example)
            if int(cfg.gen.take_every) > 1 and (idx % int(cfg.gen.take_every)) != 0:
                continue

            problem = organize_gsm_row(row)

            # Skip already seen or blacklisted IDs
            if problem["id"] in student_ids or problem["id"] in teacher_blacklist:
                continue

            future_to_pid[
                executor.submit(
                    process_problem,
                    problem,
                    teacher_client,
                    teacher_name,
                    float(cfg.teacher.temperature),
                    student_client,
                    student_name,
                    float(cfg.student.temperature),
                    list(cfg.gen.k_values),
                    int(cfg.gen.num_samples)
                )
            ] = problem

        for future in tqdm(as_completed(future_to_pid), total=len(future_to_pid), desc="Curating"):
            result, scores, p_tks, c_tks = future.result()
            problem = future_to_pid[future]

            if result == GOOD_PROBLEM:
                collected.append({"problem": problem, "scores": scores})
                student_ids.add(problem["id"])

            elif result == TEACHER_BLACKLIST:
                teacher_blacklist.add(problem["id"])

            total_prompt_tks += p_tks
            total_completion_tks += c_tks

            # Save every 100 examples
            if len(collected) > 0 and len(collected) % 100 == 0:
                save_json_atomic(
                    output_fpath, 
                    {
                        "collected": collected,
                        "teacher_blacklist": list(teacher_blacklist),
                        "student_ids": list(student_ids)
                    },
                    indent=2
                )
                logger.info("Collected %d/%d samples!", len(collected), cfg.data.data_amount)

            if len(collected) >= cfg.data.data_amount:
                # Cancel pending futures
                for fut in future_to_pid:
                    fut.cancel()
                executor.shutdown(cancel_futures=True)
                break

    # Final save
    save_json_atomic(
        output_fpath, 
        {
            "collected": collected,
            "teacher_blacklist": list(teacher_blacklist),
            "student_ids": list(student_ids)
        },
        indent=2
    )

    logger.info("Saved %d problems to %s", len(collected), output_fpath)
    logger.info("Token usage (approx): prompt=%d, completion=%d", total_prompt_tks, total_completion_tks)

    elapsed = time.perf_counter() - start_time
    logger.info("Total time elapsed: %s", timedelta(seconds=int(elapsed)))

    # Train-test split
    save_train_test_split(
        data=collected,
        dir=teacher_dir,
        name=student_name,
        pass_k=cfg.gen.pass_k,
        pass_at_k_threshold=cfg.gen.pass_at_k_threshold
    )

    # Count the no. of examples
    count_examples(collected)


if __name__ == "__main__":
    # Helpful for debugging Hydra tracebacks
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    main()
