"""
Teacher-leakage analysis: how much of the answer does each teacher reply reveal?

An LLM judge reads the problem, its gold solution, the student's question and the teacher's reply
and assigns a leakage score in [0, 1] (0 = purely Socratic, 1 = full answer). The reported number is
the mean across turns: the average of the per-turn means, with the average of the per-turn
bootstrap standard deviations.

Output: leakage_aggregate.json (overall.mean, overall.std), leakage_overall.json,
leakage_per_problem.json (and leakage_improved_only.json when analysis.baseline differs).

Usage:
    python -m src.analysis.leakage student.model=qwen7 data.type=math experiment.dataset=gsm8k \\
        data.split=test data.data_amount=1000 judge.model=llama70 \\
        analysis.compare=self_odqs_pre_assessment analysis.baseline=self_odqs_pre_assessment
"""

import os
import time
import logging
from datetime import timedelta
from typing import Any, Callable, Dict, List
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
    _safe_parse_json, _extract_teacher_texts, _transcripts_dir, _mean_across_turns,
    get_improvement_matrix, _align_mask,
)

from src.prompts.math import judge_leakage_system_prompt as leakage_system_prompt_math
from src.prompts.math import get_judge_leakage_user_prompt as get_leakage_user_prompt_math
from src.prompts.code import judge_leakage_system_prompt as leakage_system_prompt_code
from src.prompts.code import get_judge_leakage_user_prompt as get_leakage_user_prompt_code

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)


def analyze_leakage(
    judge_client,
    judge_model_name: str,
    problem: Dict[str, Any],
    transcripts_dir: str,
    system_prompt: str,
    get_user_prompt: Callable,
) -> Dict[str, Any]:
    """
    For one problem:
      - Load transcript
      - Extract the teacher's replies (one per turn)
      - Judge leakage per teacher turn
    Returns:
      {
        "problem_id": str,
        "turns": [{"turn": int, "leakage": float|None, "confidence": str|None, "raw": str}],
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

    teacher_turns = _extract_teacher_texts(transcript)
    if not teacher_turns:
        return {"problem_id": pid, "turns": [], "token_usage": {"prompt": 0, "completion": 0},
                "error": "No teacher turns found in transcript"}

    per_turn: List[Dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for t_idx, (teacher_reply, preceding_student) in enumerate(teacher_turns):
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": get_user_prompt(
                    problem=problem,
                    teacher_reply=teacher_reply,
                    preceding_student_turn=preceding_student,
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
        leakage = None
        confidence = None
        if isinstance(parsed, dict):
            val = parsed.get("leakage", None)
            if isinstance(val, (int, float)):
                leakage = max(0.0, min(1.0, float(val)))
            conf = parsed.get("confidence", None)
            if isinstance(conf, str):
                confidence = conf

        per_turn.append(
            {
                "turn": t_idx,  # reply to the student question of turn t_idx
                "leakage": leakage,
                "confidence": confidence,
                "raw": raw,
            }
        )

    return {
        "problem_id": pid,
        "turns": per_turn,
        "token_usage": {"prompt": total_prompt_tokens, "completion": total_completion_tokens},
    }


def run_leakage(cfg: DictConfig) -> None:
    """Run LLM-judge leakage scoring on teacher turns; report the mean across turns (overall and improved-only)."""
    t0 = time.perf_counter()

    dataset = str(cfg.experiment.dataset)
    compare_name  = str(cfg.analysis.compare)
    baseline_name = str(cfg.analysis.baseline)
    max_workers   = int(cfg.experiment.num_workers)
    k_use         = int(getattr(cfg.analysis, "k", 5))

    def _exp_root(exp_name):
        return os.path.join(
            str(cfg.paths.experiments),
            exp_name,
            str(cfg.teacher.model),
            str(cfg.student.model),
            dataset,
        )

    baseline_root   = _exp_root(baseline_name)
    compare_root    = _exp_root(compare_name)
    transcripts_dir = _transcripts_dir(compare_root)

    out_dir         = os.path.join(compare_root, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    # --- load problems ---
    problems_fpath = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        dataset,
        f"splits_{cfg.gen.pass_at_k_threshold}",
        f"{cfg.student.model}_{cfg.data.split}.json",
    )
    logger.info(f"[Leakage] Loading problems: {problems_fpath}")
    data = load_json_resilient(problems_fpath) or []
    interaction_problems = data[: cfg.data.data_amount] if cfg.data.get("data_amount") else data
    pids = [str(d["id"]) for d in interaction_problems]
    logger.info(f"[Leakage] Total problems: {len(interaction_problems)}")

    # --- build judge ---
    judge_model_name = getattr(cfg, "judge", {}).get("model", None) or str(cfg.teacher.model)
    if cfg.data.type == "code":
        system_prompt   = leakage_system_prompt_code
        get_user_prompt = get_leakage_user_prompt_code
    elif cfg.data.type == "math":
        system_prompt   = leakage_system_prompt_math
        get_user_prompt = get_leakage_user_prompt_math
    else:
        raise ValueError(f"{cfg.data.type} should be either code/math")

    client = build_client(judge_model_name, cfg.judge.gpus, cfg.judge.server)

    # --- evaluate per problem ---
    all_results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                analyze_leakage,
                client,
                judge_model_name,
                problem,
                transcripts_dir,
                system_prompt,
                get_user_prompt,
            ): str(problem["id"])
            for problem in interaction_problems
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="[Leakage] Judging"):
            pid = futures[fut]
            all_results[pid] = fut.result()

    # --- build [N, T] leakage array (one column per turn) ---
    max_T = max(len(all_results[pid]["turns"]) for pid in pids if "turns" in all_results.get(pid, {}))
    vals = np.full((len(pids), max_T), np.nan, dtype=float)
    for i, pid in enumerate(pids):
        for entry in all_results.get(pid, {}).get("turns", []):
            idx = int(entry["turn"])
            leak = entry.get("leakage")
            if isinstance(leak, (int, float)):
                vals[i, idx] = float(leak)

    # --- mean across teacher turns, improved-only subset ---
    if baseline_name != compare_name:
        mask = get_improvement_matrix(pids, baseline_root, compare_root, k=k_use).astype(bool)
        mask = _align_mask(mask, max_T)

        improved = _mean_across_turns(vals, mask=mask)

        improved_out = os.path.join(out_dir, "leakage_improved_only.json")
        save_json_atomic(improved_out, improved)

    # --- mean across teacher turns, all problems ---
    overall = _mean_across_turns(vals)

    overall_out = os.path.join(out_dir, "leakage_overall.json")
    save_json_atomic(overall_out, overall)
    logger.info(f"[Leakage] mean across turns = {overall['mean']} (std {overall['std']})")

    combined = {
        "compare": compare_name,
        "baseline": baseline_name,
        "dataset": dataset,
        "judge_model": judge_model_name,
        "overall": overall,
    }
    if baseline_name != compare_name:
        combined["improved_only"] = improved

    combined_out = os.path.join(out_dir, "leakage_aggregate.json")
    save_json_atomic(combined_out, combined)

    raw_out = os.path.join(out_dir, "leakage_per_problem.json")
    save_json_atomic(raw_out, all_results)

    dt = timedelta(seconds=(time.perf_counter() - t0))
    logger.info(f"[Leakage] Saved overall  → {overall_out}")
    if baseline_name != compare_name:
        logger.info(f"[Leakage] Saved improved → {improved_out}")
    logger.info(f"[Leakage] Saved combined → {combined_out}")
    logger.info(f"[Leakage] Saved raw      → {raw_out}")
    logger.info(f"[Leakage] Elapsed: {dt}")

    client.shutdown()

@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    set_envs(cfg.paths.openai_api_key_fpath, cfg.paths.hf_home)
    run_leakage(cfg)


if __name__ == "__main__":
    main()
