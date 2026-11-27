"""
Analyze human leakage annotation results from Amazon Mechanical Turk.

Inputs:
  --human_results   : CSV downloaded from Amazon Mechanical Turk (one row per annotation)
  --samples         : leakage_samples.jsonl (from extract_leakage_samples.py)
  --llm_scores      : JSON file with LLM judge scores {item_id -> score}
                      (or compute from experiments/*/analysis/leakage_*.json)

Outputs:
  - Prints inter-rater reliability (Krippendorff's alpha)
  - Prints per-condition mean ± std (human vs LLM judge)
  - Prints Spearman correlation between human mean and LLM score
  - Saves summary table as leakage_human_eval_results.json

Usage:
  python -m codes.human_eval.analyze_leakage_results \
      --human_results experiments/human_eval/data/amt_results.csv \
      --samples experiments/human_eval/data/leakage_samples.jsonl
"""

import json
import csv
import argparse
import math
import numpy as np
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Krippendorff's alpha (ordinal)
# ---------------------------------------------------------------------------

def krippendorffs_alpha(ratings_by_item: dict, level="ordinal") -> float:
    """
    Compute Krippendorff's alpha.
    ratings_by_item: {item_id: [r1, r2, r3, ...]}  — ragged lists OK (missing = None)
    level: 'nominal' | 'ordinal' | 'interval'
    Returns alpha in [-1, 1].
    """
    # Flatten to pairable units
    items = list(ratings_by_item.keys())
    all_ratings = []
    for item in items:
        for r in ratings_by_item[item]:
            if r is not None:
                all_ratings.append(r)

    values = sorted(set(all_ratings))
    V = len(values)
    val_idx = {v: i for i, v in enumerate(values)}

    def metric(v1, v2):
        if level == "nominal":
            return 0 if v1 == v2 else 1
        elif level == "ordinal":
            # Sum of all values between v1 and v2 in rank order
            i1, i2 = sorted([val_idx[v1], val_idx[v2]])
            return (sum(values[i1 : i2 + 1]) - (values[i1] + values[i2]) / 2) ** 2
        elif level == "interval":
            return (v1 - v2) ** 2
        raise ValueError(level)

    # Observed disagreement (D_o)
    coincidence = defaultdict(float)
    n_c = defaultdict(float)  # n_u: number of ratings per unit
    for item in items:
        rs = [r for r in ratings_by_item[item] if r is not None]
        mu = len(rs)
        if mu < 2:
            continue
        n_c[item] = mu
        for k, v1 in enumerate(rs):
            for l, v2 in enumerate(rs):
                if k != l:
                    coincidence[(v1, v2)] += 1.0 / (mu - 1)

    total = sum(coincidence.values())
    if total == 0:
        return float("nan")

    D_o = sum(metric(v1, v2) * count for (v1, v2), count in coincidence.items()) / total

    # Expected disagreement (D_e)
    n_k = defaultdict(float)
    for (v1, v2), count in coincidence.items():
        n_k[v1] += count
        n_k[v2] += count

    n_total = sum(n_k.values())
    D_e = 0.0
    for v1 in values:
        for v2 in values:
            if n_total > 0:
                D_e += metric(v1, v2) * (n_k[v1] / n_total) * (n_k[v2] / n_total)

    if D_e == 0:
        return 1.0

    return 1.0 - D_o / D_e


def _bootstrap_mean_std(vals, n_boot=10_000, seed=42):
    """Bootstrap the mean; std is std of bootstrap means (standard error).
    Matches the same approach used in run_analysis.py _bootstrap_mean_std."""
    arr = np.array(vals, dtype=float)
    n = len(arr)
    if n == 0:
        return {"n": 0, "mean": None, "std": None}
    if n == 1:
        return {"n": 1, "mean": float(arr[0]), "std": 0.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)
    return {"n": n, "mean": float(arr.mean()), "std": float(boot_means.std(ddof=1))}


def bootstrap_delta_ci(vals_a, vals_b, n_boot=10_000, seed=42):
    """Bootstrap CI for mean(b) - mean(a) and two-tailed p-value."""
    a = np.array(vals_a, dtype=float)
    b = np.array(vals_b, dtype=float)
    observed_delta = b.mean() - a.mean()
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    boot_deltas = b[ib].mean(axis=1) - a[ia].mean(axis=1)
    lo = float(np.percentile(boot_deltas, 2.5))
    hi = float(np.percentile(boot_deltas, 97.5))
    p = float((np.abs(boot_deltas) >= abs(observed_delta)).mean())
    return observed_delta, lo, hi, p


def spearman_correlation(xs, ys):
    """Spearman rank correlation between two equal-length lists."""
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan")

    def rank(lst):
        sorted_vals = sorted(enumerate(lst), key=lambda x: x[1])
        ranks = [0] * n
        for rank_val, (orig_idx, _) in enumerate(sorted_vals):
            ranks[orig_idx] = rank_val + 1
        return ranks

    rx = rank(xs)
    ry = rank(ys)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    rho = 1 - 6 * d2 / (n * (n * n - 1))

    # t-statistic for significance
    if abs(rho) == 1.0:
        p = 0.0
    else:
        t = rho * math.sqrt(n - 2) / math.sqrt(1 - rho ** 2)
        # Approximate p-value from t-distribution (two-tailed)
        # Use normal approximation for large n
        from math import erfc, sqrt
        z = abs(t) / sqrt(1)  # simple approximation
        p = erfc(abs(t) / sqrt(2))

    return rho, p


# Label mapping: convert 1-5 Likert to 0-1 scale matching paper's rubric
LABEL_TO_SCORE = {1: 0.0, 2: 0.25, 3: 0.50, 4: 0.75, 5: 1.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_results", required=True,
                        help="AMT results CSV: columns item_id, worker_id, rating (1-5)")
    parser.add_argument("--samples", required=True,
                        help="leakage_samples.jsonl with condition and item_id")
    parser.add_argument("--llm_scores", default=None,
                        help="JSON {item_id: llm_score} — if None, skips LLM correlation")
    parser.add_argument("--out", default="codes/human_eval/data/leakage_human_eval_results.json")
    args = parser.parse_args()

    # Load sample metadata
    item_meta = {}
    with open(args.samples) as f:
        for line in f:
            item = json.loads(line)
            item_meta[item["item_id"]] = item

    # Load human ratings
    # Expected CSV columns: item_id, worker_id, rating (integer 1-5)
    ratings_by_item = defaultdict(list)    # item_id -> [scores 0-1]
    raw_by_item = defaultdict(list)        # item_id -> [raw 1-5 labels]

    with open(args.human_results) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("AssignmentStatus", "").strip() == "Rejected":
                continue
            # Support both preprocessed ("item_id","rating") and raw AMT columns
            item_id = (row.get("item_id") or row.get("Input.item_id") or "").strip()
            if not item_id:
                continue
            raw = row.get("rating") or row.get("Answer.r") or row.get("Answer.rating") or ""
            try:
                label = int(float(raw))
            except (ValueError, TypeError):
                continue
            score = LABEL_TO_SCORE.get(label, None)
            if score is None:
                continue
            ratings_by_item[item_id].append(score)
            raw_by_item[item_id].append(label)

    # Filter to non-gold items only for main analysis
    eval_items = {
        iid: ratings
        for iid, ratings in ratings_by_item.items()
        if item_meta.get(iid, {}).get("is_gold", False) is False
           and item_meta.get(iid, {}).get("condition", "gold") != "gold"
    }

    # Gold item accuracy (quality control check)
    gold_items_rated = {
        iid: raw_by_item[iid]
        for iid in raw_by_item
        if item_meta.get(iid, {}).get("is_gold", False)
    }
    gold_accuracy = {}
    for iid, labels in gold_items_rated.items():
        expected = item_meta[iid]["gold_label"]
        within_1 = sum(1 for l in labels if abs(l - expected) <= 1) / max(len(labels), 1)
        gold_accuracy[iid] = {"expected": expected, "received": labels, "within_1_accuracy": within_1}

    print("\n=== GOLD ITEM QUALITY CHECK ===")
    for iid, info in gold_accuracy.items():
        print(f"  {iid}: expected={info['expected']}, got={info['received']}, within±1={info['within_1_accuracy']:.2f}")

    # Inter-rater reliability
    alpha = krippendorffs_alpha(eval_items, level="ordinal")
    print(f"\n=== INTER-RATER RELIABILITY ===")
    print(f"  Krippendorff's alpha (ordinal): {alpha:.3f}")
    print("  Interpretation: >0.8 = good, 0.6-0.8 = moderate, <0.6 = low")

    # Per-condition aggregation
    condition_scores = defaultdict(list)
    for iid, scores in eval_items.items():
        cond = item_meta.get(iid, {}).get("condition", "unknown")
        if scores:
            mean_score = sum(scores) / len(scores)
            condition_scores[cond].append(mean_score)

    print("\n=== PER-CONDITION HUMAN LEAKAGE SCORES (mean ± bootstrap_std) ===")
    condition_summary = {}
    for cond in ["unscaffolded", "scaffolded", "pre_assessment", "mid_assessment", "self_odqs", "peer_odqs"]:
        vals = condition_scores.get(cond, [])
        if not vals:
            print(f"  {cond}: no data")
            continue
        stats = _bootstrap_mean_std(vals)
        print(f"  {cond}: {stats['mean']:.3f} ± {stats['std']:.3f}  (n={stats['n']})")
        condition_summary[cond] = {"mean": stats["mean"], "std": stats["std"], "n": stats["n"]}

    # Delta between conditions (unscaffolded vs each trained method)
    if "unscaffolded" in condition_scores:
        print("\n=== DELTA vs UNSCAFFOLDED (bootstrap, 95% CI, two-tailed p) ===")
        for cond in ["self_odqs", "peer_odqs", "scaffolded", "mid_assessment"]:
            vals = condition_scores.get(cond, [])
            if not vals:
                continue
            delta, lo, hi, p = bootstrap_delta_ci(condition_scores["unscaffolded"], vals)
            sig = "**" if p < 0.05 else ("*" if p < 0.10 else "ns")
            print(f"  {cond} - unscaffolded: Δ={delta:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  p={p:.3f} {sig}")
            condition_summary[cond]["delta_vs_unscaffolded"] = {"delta": delta, "ci_lo": lo, "ci_hi": hi, "p": p}

    # LLM vs human correlation
    llm_corr_results = None
    if args.llm_scores:
        with open(args.llm_scores) as f:
            llm_scores = json.load(f)

        # Per-condition correlation
        cond_pairs = defaultdict(lambda: ([], []))
        for iid, scores in eval_items.items():
            if iid in llm_scores and scores:
                cond = item_meta.get(iid, {}).get("condition", "unknown")
                cond_pairs[cond][0].append(sum(scores) / len(scores))
                cond_pairs[cond][1].append(llm_scores[iid])

        # Also compute pooled
        all_human = [v for h, _ in cond_pairs.values() for v in h]
        all_llm   = [v for _, l in cond_pairs.values() for v in l]

        print(f"\n=== HUMAN vs LLM JUDGE CORRELATION (per condition) ===")
        llm_corr_results = {}
        for cond, (h, l) in sorted(cond_pairs.items()):
            rho, p = spearman_correlation(h, l)
            print(f"  {cond}: Spearman rho = {rho:.3f}  (p ≈ {p:.4f}, n={len(h)})")
            llm_corr_results[cond] = {"spearman_rho": rho, "p_value": p, "n": len(h)}
        if all_human:
            rho, p = spearman_correlation(all_human, all_llm)
            print(f"  pooled:   Spearman rho = {rho:.3f}  (p ≈ {p:.4f}, n={len(all_human)})")
            llm_corr_results["pooled"] = {"spearman_rho": rho, "p_value": p, "n": len(all_human)}

    # Save
    out = {
        "gold_quality_check": gold_accuracy,
        "inter_rater_alpha": alpha,
        "per_condition": condition_summary,
        "llm_human_correlation": llm_corr_results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {args.out}")

    print("\n=== SUMMARY ===")
    print("Key claim to verify: leakage scores are CONSISTENT ACROSS METHODS.")
    if condition_summary:
        vals = [v["mean"] for v in condition_summary.values()]
        spread = max(vals) - min(vals)
        print(f"  Max spread across conditions: {spread:.3f}")
        print(f"  (If spread < 0.1, conditions are effectively equivalent)")
    if llm_corr_results and "pooled" in llm_corr_results:
        print(f"  Judge calibration (Spearman rho, pooled): {llm_corr_results['pooled']['spearman_rho']:.3f}")
    else:
        print("  (provide --llm_scores to compute judge calibration)")


if __name__ == "__main__":
    main()
