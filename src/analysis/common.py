import os
import gc
import json
import time
import logging
from datetime import timedelta
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import hydra
from tqdm import tqdm
from omegaconf import DictConfig

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch
# Use LaTeX for all text rendering
rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from codes.utils.vars import set_envs
from codes.utils.llm.server import build_client
from codes.utils.llm.chat import chat_complete
from codes.utils.file_ops import save_json_atomic, load_json_resilient

from codes.prompts.math import judge_progress_system_prompt as progress_system_prompt_math
from codes.prompts.math import get_judge_progress_user_prompt as get_progress_user_prompt_math
from codes.prompts.code import judge_progress_system_prompt as progress_system_prompt_code
from codes.prompts.code import get_judge_progress_user_prompt as get_progress_user_prompt_code

from codes.prompts.math import judge_similarity_system_prompt as similarity_system_prompt_math
from codes.prompts.math import get_judge_similarity_user_prompt as get_similarity_user_prompt_math
from codes.prompts.code import judge_similarity_system_prompt as similarity_system_prompt_code
from codes.prompts.code import get_judge_similarity_user_prompt as get_similarity_user_prompt_code

from codes.prompts.math import judge_rank_system_prompt_math, get_rank_user_prompt_math
from codes.prompts.code import judge_rank_system_prompt_code, get_rank_user_prompt_code

from codes.prompts.math import judge_leakage_system_prompt as leakage_system_prompt_math
from codes.prompts.math import get_judge_leakage_user_prompt as get_leakage_user_prompt_math
from codes.prompts.code import judge_leakage_system_prompt as leakage_system_prompt_code
from codes.prompts.code import get_judge_leakage_user_prompt as get_leakage_user_prompt_code

from codes.prompts.math import judge_question_type_system_prompt as qtype_system_prompt_math
from codes.prompts.math import get_judge_question_type_user_prompt as get_qtype_user_prompt_math
from codes.prompts.code import judge_question_type_system_prompt as qtype_system_prompt_code
from codes.prompts.code import get_judge_question_type_user_prompt as get_qtype_user_prompt_code


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)


# -------------------- Helpers --------------------
def _safe_parse_json(s: str) -> Optional[Dict[str, Any]]:
    try:
        s_stripped = s.strip()
        if s_stripped.startswith("```"):
            # remove fences and keep innermost JSON block
            s_stripped = s_stripped.strip("`")
            l = s_stripped.find("{")
            r = s_stripped.rfind("}")
            if l != -1 and r != -1 and r > l:
                s_stripped = s_stripped[l : r + 1]
        return json.loads(s_stripped)
    except Exception:
        return None

def _extract_student_texts(transcript: List[Dict[str, Any]]) -> List[str]:
    """
    Returns a list where item t-1 is the concatenation of student turns [1..t].
    Ignores teacher/assistant messages.
    """
    student_only: List[str] = []
    for message in transcript[1: -1: 2]:
        student_only.append(message["content"])

    cumulative: List[str] = []
    acc = []
    for s in student_only:
        acc.append(s.strip())
        cumulative.append("\n\n---\n\n".join(acc))
    return cumulative

def _extract_teacher_texts(transcript: List[Dict[str, Any]]) -> List[tuple]:
    """
    Returns a list of (teacher_reply, preceding_student_message) tuples.
    Entry t is the teacher's reply to the student question of turn t (messages 2, 4, 6, ... of the transcript).
    The preceding student message is always at index i-1.
    """
    result = []
    for i in range(2, len(transcript), 2):
        teacher_reply = transcript[i]["content"]
        preceding_student = transcript[i - 1]["content"] if i >= 1 else None
        result.append((teacher_reply, preceding_student))
    return result


def _transcripts_dir(root: str) -> str:
    """Transcripts folder of an experiment: transcripts_0 (codes.run_interact) or transcripts (codes.run_cross_eval)."""
    d = os.path.join(root, "transcripts_0")
    return d if os.path.exists(d) else os.path.join(root, "transcripts")


def _load_turns(transcripts_dir: str, pid: str) -> dict:
    """
    Loads transcripts_<iter> and extracts up to `turns` full student turns per problem.
    """
    fpath = os.path.join(transcripts_dir, f"{pid}.json")
    data = load_json_resilient(fpath)

    # pick only student turns (odd)
    data = data[1:-1:2]

    return [m["content"] for m in data]


def _bootstrap_mean_std(
    values: List[Optional[float]],
    num_samples: int = 10_000,
    seed: int = 42
) -> Dict[str, Any]:
    """Bootstrap the MEAN of 'values'; 'std' is std of bootstrap means."""
    arr = np.array(
        [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))],
        dtype=float
    )
    n = len(arr)
    if n == 0:
        return {"n": 0, "mean": None, "std": None}
    if n == 1:
        return {"n": 1, "mean": float(arr[0]), "std": 0.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(num_samples, n))
    boot_means = arr[idx].mean(axis=1)
    return {"n": n, "mean": float(arr.mean()), "std": float(boot_means.std(ddof=1))}


def _per_turn_bootstrap(
    values: np.ndarray,              # [N, T] float or np.nan
    mask: Optional[np.ndarray] = None,  # [N, T] bool/0-1; if None use all
    num_samples: int = 10_000,
    seed: int = 42
) -> List[Dict[str, Any]]:
    N, T = values.shape
    out: List[Dict[str, Any]] = []
    for t in range(T):
        col = values[:, t]
        use = ~np.isnan(col)
        if mask is not None:
            use = use & (mask[:, t] > 0)
        vals = [float(v) for v in col[use]]
        stats = _bootstrap_mean_std(vals, num_samples=num_samples, seed=seed)
        out.append({
            "turn": t,
            "n": stats["n"],
            "mean": stats["mean"],
            "std": stats["std"],
        })
    return out


def _mean_across_turns(
    values: np.ndarray,                 # [N, T] float or np.nan
    mask: Optional[np.ndarray] = None,  # [N, T] bool/0-1; if None use all
    num_samples: int = 10_000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Collapse an [N, T] score matrix to a single mean(std) across turns.

    mean: average of the per-turn means (every turn weighted equally).
    std : average of the per-turn bootstrap standard deviations.

    This is the statistic reported as mean(std) in the paper's teacher-leakage table.
    """
    per_turn = _per_turn_bootstrap(values, mask=mask, num_samples=num_samples, seed=seed)
    used = [m for m in per_turn if m["mean"] is not None]
    if not used:
        return {"mean": None, "std": None, "n_turns": 0, "n_scores": 0}
    return {
        "mean": float(np.mean([m["mean"] for m in used])),
        "std": float(np.mean([m["std"] for m in used])),
        "n_turns": len(used),
        "n_scores": int(sum(m["n"] for m in used)),
    }


def bootstrap_similarity_matrix(
    mats: np.ndarray,       # shape [N, T, T] similarity per problem
    mask: np.ndarray=None,  # optional [N, T] row-wise (compare-turn) mask; 1=use row i for this problem
    B: int=2000,
    seed: int=42
):
    """
    Returns:
      mean_M: [T, T] mean similarity
      lo_M:   [T, T] 2.5% CI
      hi_M:   [T, T] 97.5% CI
      lead_j: [T]    argmax column per row on mean_M
      lead_support: [T] bootstrap support (fraction of draws where that j won)
    """
    rng = np.random.default_rng(seed)
    N, T, _ = mats.shape
    use_mask = mask is not None

    # point estimate
    if use_mask:
        # broadcast mask [N,T] -> [N,T,1]
        w = mask.astype(float)[:, :, None]
        sum_mat = (mats * w).sum(axis=0)                # [T,T]
        counts  = (w > 0).sum(axis=0)                   # [T,1]
        mean_M  = sum_mat / np.clip(counts, 1e-9, None) # [T,T]
    else:
        mean_M = mats.mean(axis=0)

    # bootstrap
    boot_means = np.empty((B, T, T), dtype=float)
    leads = np.empty((B, T), dtype=int)

    for b in range(B):
        idx = rng.integers(0, N, size=N)  # resample problems
        mb = mats[idx]                    # [N,T,T]
        if use_mask:
            wb = mask[idx].astype(float)[:, :, None]   # [N,T,1]
            sb = (mb * wb).sum(axis=0)                 # [T,T]
            cb = (wb > 0).sum(axis=0)                  # [T,1]
            M  = sb / np.clip(cb, 1e-9, None)
        else:
            M  = mb.mean(axis=0)
        boot_means[b] = M
        leads[b] = np.argmax(M, axis=1)  # best column per row

    lo_M = np.quantile(boot_means, 0.025, axis=0)
    hi_M = np.quantile(boot_means, 0.975, axis=0)

    # lead support (how often the point-estimate winner wins in bootstrap)
    lead_j = np.argmax(mean_M, axis=1)           # [T]
    lead_support = np.mean(leads == lead_j[None, :], axis=0)  # [T]

    return mean_M, lo_M, hi_M, lead_j, lead_support


def get_improvement_matrix(
    problems: List[str],
    baseline_dir: str,
    compare_dir: str,
    k: int = 5
) -> np.ndarray:
    baseline_results_fpath = os.path.join(baseline_dir, "itemized_results.json")
    if not os.path.exists(baseline_results_fpath):
        baseline_results_fpath = os.path.join(baseline_dir, "itemized_results_0.json")

    compare_results_fpath = os.path.join(compare_dir, "itemized_results.json")
    if not os.path.exists(compare_results_fpath):
        compare_results_fpath = os.path.join(compare_dir, "itemized_results_0.json")

    baseline_results = load_json_resilient(baseline_results_fpath)
    compare_results  = load_json_resilient(compare_results_fpath)

    mask = []
    for pid in problems:
        pid = str(pid)
        # turns present in the results (keys are strings "0", "1", "2", ...)
        turns = sorted(
            [int(t) for t in baseline_results[pid]["results"].keys()],
            key=int
        )
        # Skip turn 0: no question has been asked yet, so nothing can have improved.
        # Column j therefore says whether the question of turn j improved Pass@k at turn j + 1.
        row = []
        for t in turns:
            if t == 0:
                continue
            tk = str(k)
            base_k = baseline_results[pid]["results"][str(t)][tk]
            comp_k = compare_results[pid]["results"][str(t)][tk]
            row.append(1.0 if (comp_k - base_k) >= 1.0 else 0.0)
        mask.append(row)

    # N x T' with T' = number of turns - 1
    return np.array(mask, dtype=float)


def _align_mask(mask: np.ndarray, T: int) -> np.ndarray:
    if mask.shape[1] < T:
        pad = np.zeros((mask.shape[0], T - mask.shape[1]), dtype=mask.dtype)
        return np.concatenate([mask, pad], axis=1)
    if mask.shape[1] > T:
        return mask[:, :T]
    return mask


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


def similarity_rank_judge(
    client,
    model_name: str, 
    problem: Dict[str, Any],
    baseline_root: str,
    compare_root: str,
    system_prompt: str,
    get_user_prompt
):
    """
    Builds one ranking row per compare-turn.
    Returns a dict with:
      - problem_id
      - baseline_ids: list[int]   # turn of each baseline question (0, 1, 2, ...)
      - rows: list[ {compare_turn, ranking, scores, top_id, top_justification, (raw?)} ]
      - top1: list[int]           # top_id per row (None filtered out)
      - implied_matrix: Optional[List[List[float]]]  # [T, T] if 'scores' present for all rows
    """
    pid = problem["id"]
    problem_prompt = problem["prompt"]

    baseline_turns = _load_turns(_transcripts_dir(baseline_root), pid)  # list[str]
    compare_turns  = _load_turns(_transcripts_dir(compare_root), pid)  # list[str]
    if not baseline_turns or not compare_turns:
        return {
            "problem_id": pid,
            "baseline_ids": [],
            "rows": []
        }

    # Baseline candidate ids = the turn each baseline question was asked at: 0, 1, 2, ...
    base_ids = list(range(len(baseline_turns)))
    candidates = list(zip(base_ids, baseline_turns))

    rows = []
    for idx, ctext in enumerate(compare_turns):
        msg_user = get_user_prompt(problem_prompt, ctext, candidates)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": msg_user},
        ]
        raw, _, _ = chat_complete(
            client=client, 
            model_name=model_name, 
            messages=messages, 
            n=1, 
            temperature=0.0, 
            max_tokens=1024
        )
        raw = raw[0] if isinstance(raw, (list, tuple)) else raw
        parsed = _safe_parse_json(raw) or {}

        ranking = parsed.get("ranking") or []
        just = parsed.get("justification", "")

        # Fallbacks
        if not ranking:
            ranking = list(base_ids)  # default order
        ranking = [int(x) for x in ranking if int(x) in base_ids]
        for bid in base_ids:
            if bid not in ranking:
                ranking.append(bid)
                logger.info(f"[LLM-Rank] Missing baseline id {bid} in ranking for problem {pid}, turn {idx}")

        row_payload = {
            "compare_turn": idx,
            "ranking": ranking,
            "justification": just,
        }
        rows.append(row_payload)

    return {
        "problem_id": pid,
        "baseline_ids": base_ids,
        "rows": rows,
    }


# -------------------- Progress & Similarity runners --------------------

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


def run_similarity_embedding(cfg: DictConfig) -> None:
    """Run compare method vs baseline question similarity and save per-problem + averaged matrices."""
    dataset = str(cfg.experiment.dataset)
    baseline_name = str(cfg.analysis.baseline)
    compare_name = str(cfg.analysis.compare)
    emb_model_name = str(cfg.analysis.model)
    batch_size = int(cfg.analysis.batch_size)

    if baseline_name == compare_name:
        return

    def _exp_root(exp_name):
        return os.path.join(
            str(cfg.paths.experiments),
            exp_name,
            str(cfg.teacher.model),
            str(cfg.student.model),
            dataset,
        )

    baseline_root = _exp_root(baseline_name)
    compare_root = _exp_root(compare_name)
    out_dir = os.path.join(compare_root, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    # Load problems
    problems_fpath = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        dataset,
        f"splits_{cfg.gen.pass_at_k_threshold}",
        f"{cfg.student.model}_{cfg.data.split}.json",
    )
    logger.info(f"[Progress] Loading interaction problems: {problems_fpath}")
    data: List[Dict[str, Any]] = load_json_resilient(problems_fpath)
    interaction_problems = [d["id"] for d in data][: cfg.data.data_amount]
    logger.info(f"[Progress] Total problems: {len(interaction_problems)}")

    # Sentence embedding model
    model = SentenceTransformer(emb_model_name, device="cuda:4")

    all_mats = []
    per_problem = []

    for pid in tqdm(interaction_problems, desc="[Similarity] Matrices"):
        baseline_turns = _load_turns(_transcripts_dir(baseline_root), pid)
        compare_turns = _load_turns(_transcripts_dir(compare_root), pid)

        E1 = model.encode(baseline_turns, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
        E2 = model.encode(compare_turns, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
        M = cosine_similarity(E2, E1)       # Shape: T x T
        M01 = (M + 1) / 2.0
        all_mats.append(M01)                # Shape: N * T x T

        per_problem.append({
            "problem_id": pid,
            "compare_turns": compare_turns,
            "baseline_turns": baseline_turns,
            "similarity": M01.tolist(),
        })

    # Save all results
    out_all = os.path.join(out_dir, f"similarity_embedding_{compare_name}_vs_{baseline_name}_all_matrices.json")
    save_json_atomic(out_all, per_problem)

    # Build mats array [N, T, T]
    mats = np.stack(all_mats, axis=0)  # [N, T, T]
    N, T, _ = mats.shape

    # Improvement mask [N, T’] → align to T
    k_use = int(getattr(cfg.analysis, "k", 5))
    raw_mask = get_improvement_matrix(interaction_problems, baseline_root, compare_root, k=k_use)
    mask = _align_mask(raw_mask, T).astype(bool)

    # ---- OVERALL (no mask) ----
    avg_M_all, lo_M_all, hi_M_all, lead_j_all, lead_support_all = bootstrap_similarity_matrix(
        mats=mats, mask=None, B=int(getattr(cfg.analysis, "bootstrap_B", 2000)), seed=42
    )
    out_avg_all = os.path.join(out_dir, f"similarity_embedding_{compare_name}_vs_{baseline_name}_overall.json")
    save_json_atomic(out_avg_all, {
        "baseline": baseline_name,
        "compare": compare_name,
        "dataset": dataset,
        "embedding_model": emb_model_name,
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
    ax.set_title(rf"\textbf{{Avg Similarity (Overall):}} {compare_name} vs {baseline_name}", pad=10)
    for i in range(avg_M_all.shape[0]):
        for j in range(avg_M_all.shape[1]):
            ax.text(j, i, f"{avg_M_all[i,j]:.2f}", ha="center", va="center",
                    color=("black" if avg_M_all[i,j] < 0.6 else "white"), fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(r"\textbf{Similarity}", rotation=270, labelpad=15)
    plt.tight_layout()
    out_png_all = os.path.join(out_dir, f"similarity_embedding_{compare_name}_vs_{baseline_name}_overall.png")
    plt.savefig(out_png_all, dpi=600); plt.close()

    # ---- IMPROVED-ONLY (masked) ----
    avg_M_imp, lo_M_imp, hi_M_imp, lead_j_imp, lead_support_imp = bootstrap_similarity_matrix(
        mats=mats, mask=mask, B=int(getattr(cfg.analysis, "bootstrap_B", 2000)), seed=42
    )
    out_avg_imp = os.path.join(out_dir, f"similarity_embedding_{compare_name}_vs_{baseline_name}_improved_only.json")
    save_json_atomic(out_avg_imp, {
        "baseline": baseline_name,
        "compare": compare_name,
        "dataset": dataset,
        "embedding_model": emb_model_name,
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
    ax.set_title(rf"\textbf{{Avg Similarity (Improved-only):}} {compare_name} vs {baseline_name}", pad=10)
    for i in range(avg_M_imp.shape[0]):
        for j in range(avg_M_imp.shape[1]):
            ax.text(j, i, f"{avg_M_imp[i,j]:.2f}", ha="center", va="center",
                    color=("black" if avg_M_imp[i,j] < 0.6 else "white"), fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(r"\textbf{Similarity}", rotation=270, labelpad=15)
    plt.tight_layout()
    out_png_imp = os.path.join(out_dir, f"similarity_embedding_{compare_name}_vs_{baseline_name}_improved_only.png")
    plt.savefig(out_png_imp, dpi=600); plt.close()

    logger.info(f"[Similarity] Saved overall  → {out_avg_all} | {out_png_all}")
    logger.info(f"[Similarity] Saved improved → {out_avg_imp} | {out_png_imp}")

    # kill the server
    # client.shutdown()


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


def run_similarity_judge_ranking(cfg: DictConfig) -> None:
    dataset       = str(cfg.experiment.dataset)
    baseline_name = str(cfg.analysis.baseline)
    compare_name  = str(cfg.analysis.compare)
    max_workers   = int(cfg.experiment.num_workers)

    if baseline_name == compare_name:
        return

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

    problems_fpath = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        dataset,
        f"splits_{cfg.gen.pass_at_k_threshold}",
        f"{cfg.student.model}_{cfg.data.split}.json",
    )
    interaction_problems = load_json_resilient(problems_fpath)[: cfg.data.data_amount]
    if not interaction_problems:
        logger.warning("[LLM-Rank] No problems found.")
        return
    pids = [d["id"] for d in interaction_problems]

    # Judge model + prompts
    model_name = str(cfg.teacher.model)
    client = build_client(model_name, cfg.teacher.gpus, cfg.teacher.server)
    if cfg.data.type == "code":
        system_prompt  = judge_rank_system_prompt_code
        get_user_prompt = get_rank_user_prompt_code
    elif cfg.data.type == "math":
        system_prompt  = judge_rank_system_prompt_math
        get_user_prompt = get_rank_user_prompt_math
    else:
        raise ValueError(f"{cfg.data.type} should be either code/math")

    # Collect per-problem ranking outputs
    per_problem_dict, all_mats_dict = {}, {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_pid = {
            ex.submit(
                similarity_rank_judge,
                client,
                model_name,
                problem,
                baseline_root,
                compare_root,
                system_prompt,
                get_user_prompt
            ): problem["id"]
            for problem in interaction_problems
        }
        for future in tqdm(as_completed(future_to_pid), total=len(future_to_pid), desc="[LLM-Rank]"):
            result = future.result()
            mats = [r["ranking"] for r in result["rows"]]
            pid = future_to_pid[future]

            per_problem_dict[pid] = result
            all_mats_dict[pid] = np.array(mats)

    # Ordering the results in the same order as problems
    per_problem, all_mats = [], []
    for problem in interaction_problems:
        pid = problem["id"]
        per_problem.append(per_problem_dict[pid])
        all_mats.append(all_mats_dict[pid])

    # Save raw per-problem
    out_all = os.path.join(out_dir, f"similarity_rank_{compare_name}_vs_{baseline_name}_all.json")
    save_json_atomic(out_all, per_problem)

    # Masks (overall vs improved-only)
    mats = np.stack(all_mats, axis=0)  # [N, T, T]
    T = mats.shape[1]

    k_use = int(cfg.gen.pass_k)
    raw_mask = get_improvement_matrix(pids, baseline_root, compare_root, k=k_use)
    mask = _align_mask(raw_mask, T).astype(bool)

    # ---- OVERALL ----
    avg_M_all, lo_M_all, hi_M_all, lead_j_all, lead_support_all = bootstrap_similarity_matrix(
        mats=mats, mask=None, B=int(getattr(cfg.analysis, "bootstrap_B", 2000)), seed=42
    )
    out_avg_all = os.path.join(out_dir, f"similarity_rank_{compare_name}_vs_{baseline_name}_overall.json")
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
    ax.set_title(rf"\textbf{{Average Similarity Rank (LLM Judge, Overall):}} {compare_name} vs {baseline_name}", pad=10)
    for i in range(avg_M_all.shape[0]):
        for j in range(avg_M_all.shape[1]):
            ax.text(j, i, f"{avg_M_all[i,j]:.2f}", ha="center", va="center",
                    color=("black" if avg_M_all[i,j] < 0.6 else "white"), fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(r"\textbf{Similarity Rank}", rotation=270, labelpad=15)
    plt.tight_layout()
    out_png_all = os.path.join(out_dir, f"similarity_rank_{compare_name}_vs_{baseline_name}_overall.png")
    plt.savefig(out_png_all, dpi=600); plt.close()

    # ---- IMPROVED-ONLY ----
    avg_M_imp, lo_M_imp, hi_M_imp, lead_j_imp, lead_support_imp = bootstrap_similarity_matrix(
        mats=mats, mask=mask, B=int(getattr(cfg.analysis, "bootstrap_B", 2000)), seed=42
    )
    out_avg_imp = os.path.join(out_dir, f"similarity_rank_{compare_name}_vs_{baseline_name}_improved_only.json")
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
    ax.set_title(rf"\textbf{{Average Similarity Rank (LLM Judge, Improved-only):}} {compare_name} vs {baseline_name}", pad=10)
    for i in range(avg_M_imp.shape[0]):
        for j in range(avg_M_imp.shape[1]):
            ax.text(j, i, f"{avg_M_imp[i,j]:.2f}", ha="center", va="center",
                    color=("black" if avg_M_imp[i,j] < 0.6 else "white"), fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(r"\textbf{Similarity Rank}", rotation=270, labelpad=15)
    plt.tight_layout()
    out_png_imp = os.path.join(out_dir, f"similarity_rank_{compare_name}_vs_{baseline_name}_improved_only.png")
    plt.savefig(out_png_imp, dpi=600); plt.close()

    logger.info(f"[LLM-Rank] Saved overall  → {out_avg_all} | {out_png_all}")
    logger.info(f"[LLM-Rank] Saved improved → {out_avg_imp} | {out_png_imp}")

    # kill the server
    client.shutdown()


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


# -------------------- Question Type Analysis --------------------

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


_QTYPE_PAPER_NAMES = {
    "unscaffolded":                        "Unscaffolded",
    "scaffolded":                             "Scaffolded",
    "pre_assessment":                         "Pre-Assessment",
    "mid_assessment":                         "Mid-Assessment",
    "self_odqs_pre_assessment":  "Self-ODQS",
    "self_odqs_mid_assessment":  "Self-ODQS",
    "self_odqs":          "Self-ODQS",
    "peer_odqs_pre_assessment":  "Peer-ODQS",
    "peer_odqs_mid_assessment":  "Peer-ODQS",
}

# Human-readable suptitles keyed by (compare, baseline)
_COMPARISON_TITLES: dict = {
    # Scaffolding effect
    ("scaffolded",                            "unscaffolded"):  "Scaffolded vs. Unscaffolded",
    # Assessment effect (scaffold vs scaffold + assessment)
    ("pre_assessment",                        "scaffolded"):       "Pre-Assessment vs. No Assessment",
    ("mid_assessment",                        "scaffolded"):       "Mid-Assessment vs. No Assessment",
    # Assessment vs unscaffolded (combined effect)
    ("pre_assessment",                        "unscaffolded"):  "Pre-Assessment vs. Unscaffolded",
    ("mid_assessment",                        "unscaffolded"):  "Mid-Assessment vs. Unscaffolded",
    # Training effect (same scaffold, trained vs untrained)
    ("self_odqs_pre_assessment", "pre_assessment"):   "Self-ODQS vs. No Training",
    ("self_odqs_mid_assessment", "mid_assessment"):   "Self-ODQS vs. No Training",
    ("peer_odqs_pre_assessment", "pre_assessment"):   "Peer-ODQS vs. No Training",
    ("peer_odqs_mid_assessment", "mid_assessment"):   "Peer-ODQS vs. No Training",
    # Combined effect (trained vs unscaffolded)
    ("self_odqs_pre_assessment", "unscaffolded"):  "Self-ODQS vs. Unscaffolded",
    ("self_odqs_mid_assessment", "unscaffolded"):  "Self-ODQS vs. Unscaffolded",
    ("peer_odqs_pre_assessment", "unscaffolded"):  "Peer-ODQS vs. Unscaffolded",
    ("peer_odqs_mid_assessment", "unscaffolded"):  "Peer-ODQS vs. Unscaffolded",
}


def _qtype_paper_name(name: str) -> str:
    return _QTYPE_PAPER_NAMES.get(name, name.replace("_", r"\_"))


# Context-specific subplot label overrides: (compare, baseline, condition) → label
_CONTEXT_SUBPLOT_NAMES: dict = {
    # pre_assessment/mid_assessment as the no-training baseline for ODQS comparisons
    ("self_odqs_pre_assessment", "pre_assessment",  "pre_assessment"):  "No Training",
    ("self_odqs_mid_assessment", "mid_assessment",  "mid_assessment"):  "No Training",
    ("peer_odqs_pre_assessment", "pre_assessment",  "pre_assessment"):  "No Training",
    ("peer_odqs_mid_assessment", "mid_assessment",  "mid_assessment"):  "No Training",
    # scaffolded as the no-assessment baseline for assessment comparisons
    ("pre_assessment",                        "scaffolded",      "scaffolded"):      "No Assessment",
    ("mid_assessment",                        "scaffolded",      "scaffolded"):      "No Assessment",
}


def _qtype_subplot_name(name: str, compare_name: str, baseline_name: str) -> str:
    return _CONTEXT_SUBPLOT_NAMES.get(
        (compare_name, baseline_name, name),
        _qtype_paper_name(name),
    )


def _qtype_comparison_title(compare_name: str, baseline_name: str) -> str:
    return _COMPARISON_TITLES.get(
        (compare_name, baseline_name),
        f"{_qtype_paper_name(compare_name)} vs. {_qtype_paper_name(baseline_name)}",
    )


# ── Question-type palette & labels ────────────────────────────────────────────
# Human-readable names for the four student-question types (see the
# judge_question_type_system_prompt in codes/prompts/{math,code}.py).
QTYPE_NAMES = {
    "A": "Unfocused",
    "B": "Conceptual",
    "C": "Hypothesis",
    "D": "Grounded",
}

# Soft pastel palette, one distinct hue per type (no red). Pastels sit close
# together, so the in-bar percentage labels and white segment gaps carry the
# distinction too.
QTYPE_COLORS = {
    "A": "#AFDDF7",  # sky blue
    "B": "#F4B784",  # peach
    "C": "#8ED3B4",  # mint
    "D": "#AE9DE0",  # lavender
}

# All fills are light, so value labels are black throughout.
QTYPE_TEXT_COLORS = {"A": "black", "B": "black", "C": "black", "D": "black"}


def _plot_qtype_distribution(
    baseline_name: str,
    compare_name: str,
    baseline_dist: Dict,
    compare_dist: Dict,
    out_dir: str,
    tag: str = "",
) -> None:
    TYPE_LABELS = ["A", "B", "C", "D"]
    turns = sorted(baseline_dist.keys())
    x = np.arange(len(turns))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=True)
    for ax, dist, name in [(axes[0], baseline_dist, baseline_name), (axes[1], compare_dist, compare_name)]:
        bottoms = np.zeros(len(turns))
        for q_type in TYPE_LABELS:
            vals = np.array([dist.get(t, {}).get(q_type, 0.0) for t in turns]) * 100
            # White edges give the surface gap between stacked segments, which
            # keeps neighbouring segments crisp.
            bars = ax.bar(
                x, vals, bottom=bottoms,
                color=QTYPE_COLORS[q_type], label=QTYPE_NAMES[q_type],
                width=0.6, edgecolor="white", linewidth=1.2,
            )
            for bar, bot, val in zip(bars, bottoms, vals):
                if val >= 6:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bot + val / 2,
                        rf"{val:.0f}\%",
                        ha="center", va="center",
                        fontsize=11, color=QTYPE_TEXT_COLORS[q_type],
                    )
            bottoms += vals
        ax.set_title(_qtype_subplot_name(name, compare_name, baseline_name), fontsize=22)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in range(len(turns))], fontsize=22)
        ax.set_xlabel(r"Turns", fontsize=22)
        ax.set_ylim(0, 103)
        ax.tick_params(axis="y", labelsize=22)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(r"Questions (\%)", fontsize=22)

    # One shared legend in its own strip under the panels, so it can never sit
    # on top of the bars (the stacks fill the full 0-100% height).
    handles = [
        Patch(facecolor=QTYPE_COLORS[t], edgecolor="white", label=QTYPE_NAMES[t])
        for t in TYPE_LABELS
    ]
    fig.subplots_adjust(left=0.085, right=0.995, top=0.90, bottom=0.23, wspace=0.06)
    fig.legend(
        handles=handles, loc="lower center", ncol=len(TYPE_LABELS),
        fontsize=19, frameon=False, bbox_to_anchor=(0.5, 0.0),
        handlelength=1.5, handleheight=1.1,
        columnspacing=1.8, handletextpad=0.6,
    )
    suffix = f"_{tag}" if tag else ""
    for ext in ("png", "pdf"):
        out = os.path.join(out_dir, f"question_type_distribution{suffix}.{ext}")
        plt.savefig(out, dpi=600, bbox_inches="tight")
        logger.info(f"[QuestionType] Plot -> {out}")
    plt.close()


def _plot_qtype_outcome(correlation_data: Dict, out_dir: str, tag: str = "") -> None:
    TYPE_LABELS = ["A", "B", "C", "D"]

    by_type: Dict[str, List[float]] = {t: [] for t in TYPE_LABELS}
    for info in correlation_data.values():
        dom = info.get("dominant_type")
        imp = info.get("improvement")
        if dom in TYPE_LABELS and imp is not None:
            by_type[dom].append(float(imp))

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    data_for_plot = [by_type[t] for t in TYPE_LABELS]
    # Name over count on two lines, so four long names fit without rotating.
    tick_labels = [
        f"{QTYPE_NAMES[t]}\n" + rf"($n={len(by_type[t])}$)" for t in TYPE_LABELS
    ]
    bp = ax.boxplot(data_for_plot, patch_artist=True, tick_labels=tick_labels)
    for patch, q_type in zip(bp["boxes"], TYPE_LABELS):
        patch.set_facecolor(QTYPE_COLORS[q_type])
        patch.set_alpha(0.85)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.6)
    for i, (q_type, vals) in enumerate(zip(TYPE_LABELS, data_for_plot)):
        if vals:
            med = float(np.median(vals))
            ax.text(i + 1, med, rf"{med:.2f}", ha="center", va="bottom", fontsize=11)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel(r"$\mathrm{Pass}@k$ Improvement", fontsize=22)
    ax.set_xlabel(r"Question Type", fontsize=22, labelpad=8)
    ax.tick_params(axis="y", labelsize=22)
    ax.tick_params(axis="x", labelsize=17)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    suffix = f"_{tag}" if tag else ""
    for ext in ("png", "pdf"):
        out = os.path.join(out_dir, f"question_type_outcome{suffix}.{ext}")
        plt.savefig(out, dpi=600, bbox_inches="tight")
        logger.info(f"[QuestionType] Plot -> {out}")
    plt.close()


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
    _plot_qtype_distribution(baseline_name, compare_name, baseline_dist, compare_dist, out_dir, tag)
    _plot_qtype_outcome(correlation_data, out_dir, tag)

    dt = timedelta(seconds=(time.perf_counter() - t0))
    logger.info(f"[QuestionType] tag={tag}")
    logger.info(f"[QuestionType] Saved raw  → {raw_out}")
    logger.info(f"[QuestionType] Saved dist → {dist_out}")
    logger.info(f"[QuestionType] Saved corr → {corr_out}")
    logger.info(f"[QuestionType] Elapsed: {dt}")

    if "client" in dir():
        client.shutdown()


# -------------------- Main --------------------
ANALYSIS_TASKS = {
    "progress": run_progress,                         # per-turn reasoning progress (LLM judge)
    "similarity_embedding": run_similarity_embedding, # question similarity via sentence embeddings
    "similarity_judge": run_similarity_judge,         # question similarity via LLM judge
    "similarity_rank": run_similarity_judge_ranking,  # question similarity via LLM ranking judge
    "leakage": run_leakage,                           # teacher answer leakage (LLM judge), mean across turns
    "question_type": run_question_type,               # question-type distribution (LLM judge)
}


@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    # envs
    set_envs(cfg.paths.openai_api_key_fpath, cfg.paths.hf_home)

    task = str(cfg.analysis.task)
    if task not in ANALYSIS_TASKS:
        raise ValueError(f"analysis.task must be one of {sorted(ANALYSIS_TASKS)}, got: {task!r}")
    ANALYSIS_TASKS[task](cfg)


if __name__ == "__main__":
    main()
