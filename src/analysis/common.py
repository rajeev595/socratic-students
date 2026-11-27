"""
Helpers shared by the LLM-judge analyses: reading transcripts, bootstrapped statistics, and the
"which problems improved" mask used for the improved-only variants of each analysis.

Every analysis looks at the experiment `analysis.compare` and, where it needs a reference,
`analysis.baseline`. Outputs go to <experiments>/<analysis.compare>/<teacher>/<student>/<dataset>/analysis/.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.io import load_json_resilient

logger = logging.getLogger(__name__)


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
    """Transcripts folder of an experiment: transcripts_0 (src.interact) or transcripts (src.cross_eval)."""
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
