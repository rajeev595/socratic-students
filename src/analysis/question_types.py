"""
Question-type analysis: what kind of questions does the student ask at each turn?

An LLM judge labels every student message as Unfocused (A), Conceptual (B), Hypothesis (C) or
Grounded (D). The distribution per turn is computed for `analysis.compare` and `analysis.baseline`
and plotted side by side; the dominant type of a problem is also related to its Pass@k improvement.
Judge labels are cached, so re-running only redraws the figures (as does `python -m src.plots.question_types`).

Output: question_type_{all,dist,correlation}_<compare>_vs_<baseline>.json and
question_type_{distribution,outcome}_<compare>_vs_<baseline>.{pdf,png}

Usage:
    python -m src.analysis.question_types student.model=qwen7 data.type=math experiment.dataset=gsm8k \\
        data.split=test data.data_amount=1000 judge.model=llama70 \\
        analysis.compare=self_odqs_pre_assessment analysis.baseline=pre_assessment
"""

import os
import time
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import hydra
from tqdm import tqdm
from omegaconf import DictConfig

from src.utils.env import set_envs
from src.utils.llm.server import build_client
from src.utils.llm.chat import chat_complete
from src.utils.io import save_json_atomic, load_json_resilient
from src.analysis.common import _safe_parse_json, _transcripts_dir, _load_turns
from src.plots.question_types import plot_distribution, plot_outcome

from src.prompts.math import judge_question_type_system_prompt as qtype_system_prompt_math
from src.prompts.math import get_judge_question_type_user_prompt as get_qtype_user_prompt_math
from src.prompts.code import judge_question_type_system_prompt as qtype_system_prompt_code
from src.prompts.code import get_judge_question_type_user_prompt as get_qtype_user_prompt_code

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)


def _classify_qtypes_problem(
    judge_client,
    judge_model_name: str,
    problem: Dict[str, Any],
    transcripts_dir: str,
    system_prompt: str,
    get_user_prompt: Callable,
) -> Dict[str, Any]:
    """Classify question types (A-D) per student turn for one problem in one condition."""
    pid = str(problem["id"])
    turns = _load_turns(transcripts_dir, pid)
    if not turns:
        return {"problem_id": pid, "turns": [], "types": [], "justifications": [],
                "error": f"Missing transcript: {transcripts_dir}/{pid}.json"}

    types, justifications = [], []
    for student_text in turns:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": get_user_prompt(problem["prompt"], student_text)},
        ]
        resp, _, _ = chat_complete(judge_client, judge_model_name, messages, 1, 0.0, 256)
        raw = resp[0] if isinstance(resp, (list, tuple)) else resp
        parsed = _safe_parse_json(raw if isinstance(raw, str) else str(raw))
        q_type = "A"
        just = ""
        if isinstance(parsed, dict):
            v = str(parsed.get("type", "A")).upper()
            if v in ("A", "B", "C", "D"):
                q_type = v
            just = str(parsed.get("justification", ""))
        types.append(q_type)
        justifications.append(just)

    return {
        "problem_id": pid,
        "turns": [
            {"turn": i, "type": types[i], "justification": justifications[i]}
            for i in range(len(types))
        ],
    }


def run_question_type(cfg: DictConfig) -> None:
    """Classify student question types (A–D) for baseline vs compare; aggregate distribution and correlate with outcomes."""
    t0 = time.perf_counter()

    dataset       = str(cfg.experiment.dataset)
    compare_name  = str(cfg.analysis.compare)
    baseline_name = str(cfg.analysis.baseline)
    max_workers   = int(cfg.experiment.num_workers)
    k_use         = int(getattr(cfg.analysis, "k", 5))

    def _exp_root(name):
        return os.path.join(str(cfg.paths.experiments), name, str(cfg.teacher.model), str(cfg.student.model), dataset)

    baseline_root = _exp_root(baseline_name)
    compare_root  = _exp_root(compare_name)
    out_dir       = os.path.join(compare_root, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    baseline_transcripts = _transcripts_dir(baseline_root)
    compare_transcripts  = _transcripts_dir(compare_root)

    # Load problems
    problems_fpath = os.path.join(
        str(cfg.paths.data), str(cfg.teacher.model), dataset,
        f"splits_{cfg.gen.pass_at_k_threshold}",
        f"{cfg.student.model}_{cfg.data.split}.json",
    )
    data = load_json_resilient(problems_fpath) or []
    interaction_problems = data[: cfg.data.data_amount] if cfg.data.get("data_amount") else data
    pids = [str(d["id"]) for d in interaction_problems]
    logger.info(f"[QuestionType] {len(pids)} problems | baseline={baseline_name} | compare={compare_name}")

    TYPE_LABELS = ["A", "B", "C", "D"]
    tag = f"{compare_name}_vs_{baseline_name}"
    raw_out = os.path.join(out_dir, f"question_type_all_{tag}.json")

    if os.path.exists(raw_out):
        logger.info(f"[QuestionType] Loading cached classifications → {raw_out}")
        cached = load_json_resilient(raw_out) or {}
        baseline_results = cached.get("baseline", {})
        compare_results  = cached.get("compare", {})
    else:
        # Select domain-specific prompts
        if cfg.data.type == "code":
            system_prompt   = qtype_system_prompt_code
            get_user_prompt = get_qtype_user_prompt_code
        elif cfg.data.type == "math":
            system_prompt   = qtype_system_prompt_math
            get_user_prompt = get_qtype_user_prompt_math
        else:
            raise ValueError(f"cfg.data.type must be 'code' or 'math', got: {cfg.data.type}")

        judge_model_name = getattr(cfg, "judge", {}).get("model", None) or str(cfg.teacher.model)
        client = build_client(judge_model_name, cfg.judge.gpus, cfg.judge.server)

        def _run_classification(transcripts_dir: str, label: str) -> Dict[str, Any]:
            results: Dict[str, Any] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {
                    ex.submit(
                        _classify_qtypes_problem,
                        client, judge_model_name, problem,
                        transcripts_dir, system_prompt, get_user_prompt,
                    ): str(problem["id"])
                    for problem in interaction_problems
                }
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"[QuestionType] {label}"):
                    pid = futures[fut]
                    results[pid] = fut.result()
            return results

        baseline_results = _run_classification(baseline_transcripts, baseline_name)
        compare_results  = _run_classification(compare_transcripts,  compare_name)
        save_json_atomic(raw_out, {"baseline": baseline_results, "compare": compare_results})

    # Aggregate type distribution per turn per condition
    def _aggregate_dist(results: Dict) -> Dict:
        by_turn: Dict[int, List[str]] = {}
        for pid in pids:
            for entry in results.get(pid, {}).get("turns", []):
                by_turn.setdefault(int(entry["turn"]), []).append(entry["type"])
        dist = {}
        for turn_id, type_list in sorted(by_turn.items()):
            n = len(type_list)
            dist[turn_id] = {t: round(type_list.count(t) / n, 4) for t in TYPE_LABELS}
            dist[turn_id]["n"] = n
        return dist

    baseline_dist = _aggregate_dist(baseline_results)
    compare_dist  = _aggregate_dist(compare_results)
    dist_out = os.path.join(out_dir, f"question_type_dist_{tag}.json")
    save_json_atomic(dist_out, {baseline_name: baseline_dist, compare_name: compare_dist})

    # Correlate dominant question type with pass@k improvement
    def _load_itemized(root):
        for name in ("itemized_results.json", "itemized_results_0.json"):
            p = os.path.join(root, name)
            if os.path.exists(p):
                return load_json_resilient(p) or {}
        return {}

    itemized          = _load_itemized(compare_root)
    baseline_itemized = _load_itemized(baseline_root)

    correlation_data: Dict[str, Any] = {}
    for pid in pids:
        compare_turn_entries = compare_results.get(pid, {}).get("turns", [])
        if not compare_turn_entries:
            continue
        compare_types = [e["type"] for e in compare_turn_entries]
        dominant = max(TYPE_LABELS, key=lambda t: compare_types.count(t))
        compare_res  = (itemized.get(pid) or {}).get("results") or {}
        baseline_res = (baseline_itemized.get(pid) or {}).get("results") or {}
        if not compare_res:
            continue
        final_turn = str(max(int(t) for t in compare_res))
        compare_pass  = float((compare_res.get(final_turn)  or {}).get(str(k_use), 0.0) or 0.0)
        baseline_pass = float((baseline_res.get(final_turn) or {}).get(str(k_use), 0.0) or 0.0)
        correlation_data[pid] = {
            "dominant_type":      dominant,
            "type_counts":        {t: compare_types.count(t) for t in TYPE_LABELS},
            "compare_pass_at_k":  compare_pass,
            "baseline_pass_at_k": baseline_pass,
            "improvement":        compare_pass - baseline_pass,
        }

    corr_out = os.path.join(out_dir, f"question_type_correlation_{tag}.json")
    save_json_atomic(corr_out, correlation_data)

    # Plots
    plot_distribution(baseline_name, compare_name, baseline_dist, compare_dist, Path(out_dir), tag)
    plot_outcome(correlation_data, Path(out_dir), tag)

    dt = timedelta(seconds=(time.perf_counter() - t0))
    logger.info(f"[QuestionType] tag={tag}")
    logger.info(f"[QuestionType] Saved raw  → {raw_out}")
    logger.info(f"[QuestionType] Saved dist → {dist_out}")
    logger.info(f"[QuestionType] Saved corr → {corr_out}")
    logger.info(f"[QuestionType] Elapsed: {dt}")

    if "client" in dir():
        client.shutdown()

@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    set_envs(cfg.paths.openai_api_key_fpath, cfg.paths.hf_home)
    run_question_type(cfg)


if __name__ == "__main__":
    main()
