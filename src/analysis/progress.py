"""
Progress analysis: how far has the student's reasoning progressed at each turn?

An LLM judge reads the problem, its gold solution and the student's message of every turn and
assigns a progress score in [0, 1]. Scores are averaged per turn (bootstrapped over problems).
Plot the result with `python -m src.plots.ablations progress`.

Output: progress_overall.json, progress_aggregate.json (and progress_improved_only.json when
analysis.baseline differs from analysis.compare).

Usage:
    python -m src.analysis.progress student.model=qwen7 data.type=math experiment.dataset=gsm8k \\
        data.split=test data.data_amount=1000 judge.model=qwen72 \\
        analysis.compare=scaffolded analysis.baseline=scaffolded
"""

import os
import time
import logging
from datetime import timedelta
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import hydra
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig

from src.utils.env import set_envs
from src.utils.llm.server import build_client
from src.utils.llm.chat import chat_complete
from src.utils.io import save_json_atomic, load_json_resilient
from src.analysis.common import (
    _safe_parse_json, _extract_student_texts, _transcripts_dir, _per_turn_bootstrap,
    get_improvement_matrix,
)

from src.prompts.math import judge_progress_system_prompt as progress_system_prompt_math
from src.prompts.math import get_judge_progress_user_prompt as get_progress_user_prompt_math
from src.prompts.code import judge_progress_system_prompt as progress_system_prompt_code
from src.prompts.code import get_judge_progress_user_prompt as get_progress_user_prompt_code

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)


def analyze_problem(
    judge_client,
    judge_model_name: str,
    problem: Dict[str, Any],
    transcripts_dir: str,
    system_prompt: str,
    get_user_prompt: str,
) -> Dict[str, Any]:
    """
    For one problem:
      - Load transcript
      - Extract student reasoning turns
      - Judge progress per turn
    Returns:
      {
        "problem_id": str,
        "turns": [{"turn": int, "progress": float|None, "confidence": str|None, "raw": str}],
        "token_usage": {"prompt": int, "completion": int},
        "error": optional str
      }
    """
    pid = str(problem["id"])
    transcript_fpath = os.path.join(transcripts_dir, f"{pid}.json")
    transcript = load_json_resilient(transcript_fpath)

    if not transcript:
        return {"problem_id": pid, "turns": [], "token_usage": {"prompt": 0, "completion": 0},
                "error": f"Missing or empty transcript: {transcript_fpath}"}

    student_turns = _extract_student_texts(transcript)
    if not student_turns:
        return {"problem_id": pid, "turns": [], "token_usage": {"prompt": 0, "completion": 0},
                "error": "No student turns found in transcript"}

    per_turn: List[Dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for t_idx, student_text in enumerate(student_turns):
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": get_user_prompt(
                    problem=problem,
                    student_answer=student_text,
                ),
            },
        ]
        resp, p_tok, c_tok = chat_complete(
            judge_client, judge_model_name, messages, 1, 0.0, 1024
        )
        raw = resp[0]
        total_prompt_tokens += int(p_tok or 0)
        total_completion_tokens += int(c_tok or 0)

        parsed = _safe_parse_json(raw if isinstance(raw, str) else str(raw))
        progress = None
        confidence = None
        if isinstance(parsed, dict):
            val = parsed.get("progress", None)
            if isinstance(val, (int, float)):
                # clamp
                progress = max(0.0, min(1.0, float(val)))
            conf = parsed.get("confidence", None)
            if isinstance(conf, str):
                confidence = conf

        per_turn.append(
            {
                "turn": t_idx,
                "progress": progress,
                "confidence": confidence,
                "raw": raw,  # keep for audit; remove if large
            }
        )

    return {
        "problem_id": pid,
        "turns": per_turn,
        "token_usage": {"prompt": total_prompt_tokens, "completion": total_completion_tokens},
    }


def run_progress(cfg: DictConfig) -> None:
    """Run LLM-judge progress scoring for COMPARE experiment; compute overall and improved-only aggregates."""
    t0 = time.perf_counter()

    dataset = str(cfg.experiment.dataset)
    baseline_name = str(cfg.analysis.baseline)
    compare_name  = str(cfg.analysis.compare)
    max_workers   = int(cfg.experiment.num_workers)
    k_use         = int(getattr(cfg.analysis, "k", 5))

    # --- experiment paths ---
    def _exp_root(exp_name):
        return os.path.join(
            str(cfg.paths.experiments),
            exp_name,
            str(cfg.teacher.model),
            str(cfg.student.model),
            dataset,
        )

    baseline_root = _exp_root(baseline_name)
    compare_root  = _exp_root(compare_name)
    transcripts_dir = _transcripts_dir(compare_root)
    out_dir = os.path.join(compare_root, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    # --- load problems ---
    problems_fpath = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        dataset,
        f"splits_{cfg.gen.pass_at_k_threshold}",
        f"{cfg.student.model}_{cfg.data.split}.json",
    )
    logger.info(f"[Progress] Loading problems: {problems_fpath}")
    data = load_json_resilient(problems_fpath) or []
    interaction_problems = data[: cfg.data.data_amount] if cfg.data.get("data_amount") else data
    pids = [str(d["id"]) for d in interaction_problems]
    logger.info(f"[Progress] Total problems: {len(interaction_problems)}")

    # --- build judge ---
    judge_model_name = getattr(cfg, "judge", {}).get("model", None) or str(cfg.teacher.model)
    if cfg.data.type == "code":
        system_prompt = progress_system_prompt_code
        get_user_prompt = get_progress_user_prompt_code
    elif cfg.data.type == "math":
        system_prompt = progress_system_prompt_math
        get_user_prompt = get_progress_user_prompt_math
    else:
        raise ValueError(f"{cfg.data.type} should be either code/math")

    client = build_client(judge_model_name, cfg.judge.gpus, cfg.judge.server)

    # --- evaluate per problem ---
    all_results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                analyze_problem,
                client,
                judge_model_name,
                problem,
                transcripts_dir,
                system_prompt,
                get_user_prompt,
            ): str(problem["id"])
            for problem in interaction_problems
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="[Progress] Judging"):
            pid = futures[fut]
            all_results[pid] = fut.result()

    # --- build [N, T] progress array ---
    max_T = max(len(all_results[pid]["turns"]) for pid in pids if "turns" in all_results[pid])
    vals = np.full((len(pids), max_T), np.nan, dtype=float)
    for i, pid in enumerate(pids):
        for entry in all_results.get(pid, {}).get("turns", []):
            idx = int(entry["turn"])
            prog = entry.get("progress")
            if isinstance(prog, (int, float)):
                vals[i, idx] = float(prog)

    # --- get improvement mask ---
    if baseline_name != compare_name:
        mask = get_improvement_matrix(pids, baseline_root, compare_root, k=k_use).astype(bool)
        if mask.shape[1] < max_T:
            pad = np.zeros((mask.shape[0], max_T - mask.shape[1]), dtype=bool)
            mask = np.concatenate([mask, pad], axis=1)
        elif mask.shape[1] > max_T:
            mask = mask[:, :max_T]

        # --- bootstrap per turn (improved-only) ---
        per_turn_imp = _per_turn_bootstrap(vals, mask=mask)
        ausc_imp = sum(m["mean"] for m in per_turn_imp if m["mean"] is not None)

        # --- save ---
        improved_out = os.path.join(out_dir, "progress_improved_only.json")
        save_json_atomic(improved_out, {"per_turn": per_turn_imp, "AUSC": ausc_imp})

    # --- bootstrap per turn (overall) ---
    per_turn_all = _per_turn_bootstrap(vals)

    # --- aggregate stats ---
    ausc_all = sum(m["mean"] for m in per_turn_all if m is not None)

    # --- save ---
    overall_out = os.path.join(out_dir, "progress_overall.json")
    save_json_atomic(overall_out, {"per_turn": per_turn_all, "AUSC": ausc_all})

    combined_out = os.path.join(out_dir, "progress_aggregate.json")
    data = {
        "baseline": baseline_name,
        "compare": compare_name,
        "dataset": dataset,
        "judge_model": judge_model_name,
        "overall": {"AUSC": ausc_all, "per_turn": per_turn_all}
    }
    if baseline_name != compare_name:
        data["improved_only"] = {"AUSC": ausc_imp, "per_turn": per_turn_imp}
    save_json_atomic(combined_out, data)

    dt = timedelta(seconds=(time.perf_counter() - t0))
    logger.info(f"[Progress] Saved overall  → {overall_out}")
    if baseline_name != compare_name:    
        logger.info(f"[Progress] Saved improved → {improved_out}")
    logger.info(f"[Progress] Saved combined → {combined_out}")
    logger.info(f"[Progress] Elapsed: {dt}")

    # kill the server
    client.shutdown()

@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    set_envs(cfg.paths.openai_api_key_fpath, cfg.paths.hf_home)
    run_progress(cfg)


if __name__ == "__main__":
    main()
