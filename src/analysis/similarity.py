"""
Question-similarity analysis: does one method ask the questions of another method earlier?

For every problem, an LLM judge rates the similarity (0, 0.25, 0.5, 0.75 or 1) between the question
that `analysis.compare` asks at turn i and the question that `analysis.baseline` asks at turn j, for
all pairs of turns. The matrices are averaged over problems and drawn as a heatmap.

Output: similarity_judge_<compare>_vs_<baseline>_overall.{json,png}, ..._improved_only.{json,png},
..._all_matrices.json

Usage:
    python -m src.analysis.similarity student.model=qwen7 data.type=math experiment.dataset=gsm8k \\
        data.split=test data.data_amount=1000 judge.model=qwen72 \\
        analysis.compare=self_odqs_pre_assessment analysis.baseline=unscaffolded
"""

import os
import logging
from typing import Any, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import hydra
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm
from omegaconf import DictConfig

# Use LaTeX for all text rendering
rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

from src.utils.env import set_envs
from src.utils.llm.server import build_client
from src.utils.llm.chat import chat_complete
from src.utils.io import save_json_atomic, load_json_resilient
from src.analysis.common import (
    _safe_parse_json, _transcripts_dir, _load_turns, bootstrap_similarity_matrix,
    get_improvement_matrix, _align_mask,
)

from src.prompts.math import judge_similarity_system_prompt as similarity_system_prompt_math
from src.prompts.math import get_judge_similarity_user_prompt as get_similarity_user_prompt_math
from src.prompts.code import judge_similarity_system_prompt as similarity_system_prompt_code
from src.prompts.code import get_judge_similarity_user_prompt as get_similarity_user_prompt_code

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)


def similarity_judge(
    client,
    model_name: str,
    problem: Dict[str, Any],
    baseline_root: str,
    compare_root: str,
    system_prompt: str,
    get_user_prompt: str,
) -> Dict[str, Any]:
    pid = problem["id"]

    # load turns and pad/truncate to T
    baseline_turns = _load_turns(_transcripts_dir(baseline_root), pid)
    compare_turns = _load_turns(_transcripts_dir(compare_root), pid)
    if not baseline_turns or not compare_turns:
        return

    # build T×T similarity via judge
    scores, justifications, responses = [], [], []
    for cturn in compare_turns:
        s, j, r = [], [], []
        for bturn in baseline_turns:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": get_user_prompt(
                    problem=problem["prompt"],
                    response_a=cturn,
                    response_b=bturn,
                )},
            ]
            raw, p_tok, c_tok = chat_complete(client, model_name, messages, 1, 0.0, 512)
            raw = raw[0] if isinstance(raw, (list, tuple)) else raw
            parsed = _safe_parse_json(raw if isinstance(raw, str) else str(raw)) or {}
            sim  = float(parsed.get("similarity", 0.0)) if isinstance(parsed.get("similarity", None), (int, float)) else 0.0
            sim = max(0.0, min(1.0, sim))
            just = parsed.get("justification", "")

            s.append(sim)
            j.append(just)
            r.append(raw)

        scores.append(s)
        justifications.append(j)
        responses.append(r)

    return baseline_turns, compare_turns, scores, justifications, responses


def run_similarity_judge(cfg: DictConfig) -> None:
    """
    LLM-judge similarity between compare vs baseline turns.
    Builds per-problem T x T matrices (soft similarity in [0,1]).
    Aggregates with improvement mask (row-wise) and plots heatmap.
    """
    dataset = str(cfg.experiment.dataset)
    baseline_name = str(cfg.analysis.baseline)
    compare_name  = str(cfg.analysis.compare)
    max_workers = int(cfg.experiment.num_workers)

    if baseline_name == compare_name:
        return

    # roots
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
    out_dir = os.path.join(compare_root, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    # problems list
    problems_fpath = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        dataset,
        f"splits_{cfg.gen.pass_at_k_threshold}",
        f"{cfg.student.model}_{cfg.data.split}.json",
    )
    interaction_problems = load_json_resilient(problems_fpath)[: cfg.data.data_amount]
    pids = [d["id"] for d in interaction_problems]

    # judge client/model (prefer cfg.judge for consistency)
    model_name = getattr(cfg, "judge", {}).get("model", None) or str(cfg.teacher.model)
    client = build_client(model_name, cfg.teacher.gpus, cfg.teacher.server)

    # Pick prompts by data type
    if cfg.data.type == "code":
        system_prompt = similarity_system_prompt_code
        get_user_prompt = get_similarity_user_prompt_code
    elif cfg.data.type == "math":
        system_prompt = similarity_system_prompt_math
        get_user_prompt = get_similarity_user_prompt_math
    else:
        raise ValueError(f"{cfg.data.type} should be either code/math")

    per_problem_dict = {}
    all_mats_dict = {}

    # Fan out
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_pid = {
            ex.submit(
                similarity_judge,
                client, 
                model_name,
                problem,
                baseline_root,
                compare_root,
                system_prompt,
                get_user_prompt,
            ): problem["id"]
            for problem in interaction_problems
        }
        for future in tqdm(as_completed(future_to_pid), total=len(future_to_pid), desc="[LLM-Sim] Matrices-Judge"):
            baseline_turns, compare_turns, scores, justifications, responses = future.result()
            pid = future_to_pid[future]

            all_mats_dict[pid] = np.array(scores)
            per_problem_dict[pid] = {
                "problem_id": future_to_pid[future],
                "compare_turns": compare_turns,
                "baseline_turns": baseline_turns,
                "similarity_llm": scores,
                "justifications": justifications,
                "responses": responses
            }

    if not all_mats_dict:
        logger.warning("[LLM-Sim] No matrices computed.")
        return

    # Ordering the results in the same order as problems
    per_problem, all_mats = [], []
    for problem in interaction_problems:
        pid = problem["id"]
        per_problem.append(per_problem_dict[pid])
        all_mats.append(all_mats_dict[pid])

    out_all = os.path.join(out_dir, f"similarity_judge_{compare_name}_vs_{baseline_name}_all_matrices.json")
    save_json_atomic(out_all, per_problem)

    mats = np.stack(all_mats, axis=0)  # [N, T, T]
    N, T, _ = mats.shape

    k_use = int(cfg.gen.pass_k)
    raw_mask = get_improvement_matrix(pids, baseline_root, compare_root, k=k_use)
    mask = _align_mask(raw_mask, T).astype(bool)

    # ---- OVERALL ----
    avg_M_all, lo_M_all, hi_M_all, lead_j_all, lead_support_all = bootstrap_similarity_matrix(
        mats=mats, mask=None, B=int(getattr(cfg.analysis, "bootstrap_B", 2000)), seed=42
    )
    out_avg_all = os.path.join(out_dir, f"similarity_judge_{compare_name}_vs_{baseline_name}_overall.json")
    save_json_atomic(out_avg_all, {
        "baseline": baseline_name,
        "compare": compare_name,
        "dataset": dataset,
        "judge_model": model_name,
        "mode": "overall",
        "average_matrix": avg_M_all.tolist(),
        "ci_lo": lo_M_all.tolist(),
        "ci_hi": hi_M_all.tolist(),
        "lead_by_row": lead_j_all.tolist(),
        "lead_support": lead_support_all.tolist(),
        "n_problems": int(mats.shape[0]),
    })

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(avg_M_all, vmin=0, vmax=1, cmap="YlGn")
    turns = list(range(avg_M_all.shape[0]))
    ax.set_xticks(np.arange(avg_M_all.shape[1])); ax.set_xticklabels(turns)
    ax.set_yticks(np.arange(avg_M_all.shape[0])); ax.set_yticklabels(turns)
    ax.set_xlabel(rf"\textbf{{{baseline_name}}} turn ($j$)")
    ax.set_ylabel(rf"\textbf{{{compare_name}}} turn ($i$)")
    ax.set_title(rf"\textbf{{Average Similarity (LLM Judge, Overall):}} {compare_name} vs {baseline_name}", pad=10)
    for i in range(avg_M_all.shape[0]):
        for j in range(avg_M_all.shape[1]):
            ax.text(j, i, f"{avg_M_all[i,j]:.2f}", ha="center", va="center",
                    color=("black" if avg_M_all[i,j] < 0.6 else "white"), fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(r"\textbf{Similarity}", rotation=270, labelpad=15)
    plt.tight_layout()
    out_png_all = os.path.join(out_dir, f"similarity_judge_{compare_name}_vs_{baseline_name}_overall.png")
    plt.savefig(out_png_all, dpi=600); plt.close()

    # ---- IMPROVED-ONLY ----
    avg_M_imp, lo_M_imp, hi_M_imp, lead_j_imp, lead_support_imp = bootstrap_similarity_matrix(
        mats=mats, mask=mask, B=int(getattr(cfg.analysis, "bootstrap_B", 2000)), seed=42
    )
    out_avg_imp = os.path.join(out_dir, f"similarity_judge_{compare_name}_vs_{baseline_name}_improved_only.json")
    save_json_atomic(out_avg_imp, {
        "baseline": baseline_name,
        "compare": compare_name,
        "dataset": dataset,
        "judge_model": model_name,
        "mode": "improved_only",
        "k_for_improvement": k_use,
        "average_matrix": avg_M_imp.tolist(),
        "ci_lo": lo_M_imp.tolist(),
        "ci_hi": hi_M_imp.tolist(),
        "lead_by_row": lead_j_imp.tolist(),
        "lead_support": lead_support_imp.tolist(),
        "n_problems": int(mats.shape[0]),
        "n_rows_used_per_turn": mask.sum(axis=0).astype(int).tolist(),
    })

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(avg_M_imp, vmin=0, vmax=1, cmap="YlGn")
    turns = list(range(avg_M_imp.shape[0]))
    ax.set_xticks(np.arange(avg_M_imp.shape[1])); ax.set_xticklabels(turns)
    ax.set_yticks(np.arange(avg_M_imp.shape[0])); ax.set_yticklabels(turns)
    ax.set_xlabel(rf"\textbf{{{baseline_name}}} turn ($j$)")
    ax.set_ylabel(rf"\textbf{{{compare_name}}} turn ($i$)")
    ax.set_title(rf"\textbf{{Average Similarity (LLM Judge, Improved-only):}} {compare_name} vs {baseline_name}", pad=10)
    for i in range(avg_M_imp.shape[0]):
        for j in range(avg_M_imp.shape[1]):
            ax.text(j, i, f"{avg_M_imp[i,j]:.2f}", ha="center", va="center",
                    color=("black" if avg_M_imp[i,j] < 0.6 else "white"), fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(r"\textbf{Similarity}", rotation=270, labelpad=15)
    plt.tight_layout()
    out_png_imp = os.path.join(out_dir, f"similarity_judge_{compare_name}_vs_{baseline_name}_improved_only.png")
    plt.savefig(out_png_imp, dpi=600); plt.close()

    logger.info(f"[LLM-Sim] Saved overall  → {out_avg_all} | {out_png_all}")
    logger.info(f"[LLM-Sim] Saved improved → {out_avg_imp} | {out_png_imp}")

    # kill the server
    client.shutdown()

@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    set_envs(cfg.paths.openai_api_key_fpath, cfg.paths.hf_home)
    run_similarity_judge(cfg)


if __name__ == "__main__":
    main()
