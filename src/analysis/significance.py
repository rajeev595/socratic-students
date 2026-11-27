"""
Statistical significance of the main comparisons (paired bootstrap over problems).

For a pair of methods (A, B) evaluated on the same problems:
  avg  : mean over problems of [mean over turns of Pass@k(A) - Pass@k(B)],
         with a two-sided paired-bootstrap p-value and 95% CI
  max  : largest per-turn mean difference

  RQ1  Unscaffolded - Static (gain of the final turn over turn 0), Scaffolded - Unscaffolded
  RQ2  Pre/Mid-Assessment vs. no assessment, and vs. each other
  RQ3  Self-ODQS / Peer-ODQS vs. no training

Reads  <experiments>/<experiment>/<teacher>/<student>/<dataset>/itemized_results_0.json
Writes <experiments>/plots/significance.json and prints the tables.

Usage:
    python -m src.analysis.significance [--experiments ./experiments] [--students qwen7 mistral7]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from src.plots.results import BEST_ASSESSMENT, DATASETS, STUDENTS, TEACHERS, TURNS, experiment_folder, find_results


def paired_bootstrap_test(
    xs: np.ndarray,
    ys: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Two-sided paired bootstrap test on the mean difference xs - ys."""
    if xs.shape != ys.shape:
        raise ValueError(f"Shape mismatch: xs={xs.shape}, ys={ys.shape}")
    rng = np.random.default_rng(seed)
    diffs = xs - ys
    n = len(diffs)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = diffs[idx].mean()
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(boot_means, 100 * alpha / 2)),
        "ci_high": float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
        "p_value": float(min((boot_means <= 0).mean(), (boot_means >= 0).mean()) * 2),
    }


def p_to_stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def load_pass_by_turn(fpath: Path, turns: Iterable[int], k: int = 5) -> Dict[str, Dict[int, float]]:
    """problem id -> {turn -> Pass@k}. Problems missing any requested turn are skipped."""
    data = json.loads(Path(fpath).read_text())
    out: Dict[str, Dict[int, float]] = {}
    for pid, item in data.items():
        try:
            out[pid] = {int(t): float(item["results"][str(t)][str(k)]) for t in turns}
        except Exception:
            continue
    return out


def _itemized(exp_root: Path, method: str, teacher: str, student: str, dataset: str) -> Path:
    exp_dir = exp_root / experiment_folder(method, dataset) / teacher / student / dataset
    fpath = find_results(exp_dir, name="itemized_results")
    if fpath is None:
        raise FileNotFoundError(f"No itemized results under {exp_dir}")
    return fpath


def compare(
    exp_root: Path,
    method_a: str,
    method_b: str,
    teacher: str,
    student: str,
    dataset: str,
    turns: Iterable[int],
    k: int = 5,
) -> Dict[str, float]:
    """Compare method_a against method_b over the given turns."""
    turns = sorted(int(t) for t in turns)
    A = load_pass_by_turn(_itemized(exp_root, method_a, teacher, student, dataset), turns, k)
    B = load_pass_by_turn(_itemized(exp_root, method_b, teacher, student, dataset), turns, k)
    common = sorted(set(A) & set(B))
    if not common:
        raise ValueError(f"No common problems between {method_a} and {method_b} ({teacher}/{student}/{dataset})")

    a_avg = np.array([np.mean([A[i][t] for t in turns]) for i in common], dtype=np.float64)
    b_avg = np.array([np.mean([B[i][t] for t in turns]) for i in common], dtype=np.float64)
    boot = paired_bootstrap_test(a_avg, b_avg)
    per_turn = [float(np.mean([A[i][t] - B[i][t] for i in common])) for t in turns]
    return {
        "n": len(common),
        "avg": boot["mean_diff"],
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "p_value": boot["p_value"],
        "max": float(np.max(per_turn)),
    }


def static_gain(exp_root: Path, teacher: str, student: str, dataset: str, k: int = 5) -> Dict[str, float]:
    """Unscaffolded - Static: Pass@k at the final turn minus Pass@k before any interaction (same problems)."""
    first, last = TURNS[0], TURNS[-1]
    A = load_pass_by_turn(_itemized(exp_root, "unscaffolded", teacher, student, dataset), [first, last], k)
    final = np.array([v[last] for v in A.values()])
    static = np.array([v[first] for v in A.values()])
    boot = paired_bootstrap_test(final, static)
    return {"n": len(A), "avg": boot["mean_diff"], "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
            "p_value": boot["p_value"], "max": boot["mean_diff"]}


def _fmt(r: Dict[str, float]) -> str:
    return (f"avg={r['avg']:+.3f} (p={r['p_value']:.3f} {p_to_stars(r['p_value'])}) "
            f"max={r['max']:+.3f} CI=[{r['ci_low']:+.3f},{r['ci_high']:+.3f}] n={r['n']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", default="./experiments", help="experiments root (paths.experiments)")
    parser.add_argument("--teacher", default=TEACHERS[0])
    parser.add_argument("--students", nargs="+", default=STUDENTS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    exp_root = Path(args.experiments)
    all_turns = TURNS
    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    early_turns = [t for t in all_turns if t <= 2]   # turns 0-2
    late_turns = [t for t in all_turns if t >= 2]    # turns 2-5 (from the Mid-Assessment onwards)

    for dataset in args.datasets:
        best = BEST_ASSESSMENT[dataset]
        other = "mid_assessment" if best == "pre_assessment" else "pre_assessment"
        # Mid-Assessment is identical to no-assessment before its insertion point, so it is compared from there on.
        vs_none_turns = all_turns if best == "pre_assessment" else late_turns
        comparisons: List[tuple] = [
            ("RQ1  scaffolded - unscaffolded",               "scaffolded", "unscaffolded", all_turns),
            ("RQ1  scaffolded - unscaffolded (turns 0-2)",   "scaffolded", "unscaffolded", early_turns),
            (f"RQ2  {best} - no assessment",                  best,         "scaffolded",   vs_none_turns),
            (f"RQ2  {best} - {other} (turns 2-5)",            best,         other,          late_turns),
            ("RQ3  self_odqs - no training",                 "self_odqs",  "no_training",  all_turns),
            ("RQ3  peer_odqs - no training",                 "peer_odqs",  "no_training",  all_turns),
        ]
        for student in args.students:
            key = f"{dataset}/{student}"
            results[key] = {}
            print(f"\n=== {dataset} | student={student} ===")
            try:
                r = static_gain(exp_root, args.teacher, student, dataset, args.k)
                results[key]["RQ1  unscaffolded - static"] = r
                print(f"{'RQ1  unscaffolded - static':46s} {_fmt(r)}")
            except FileNotFoundError as e:
                print(f"[skip] {e}")
            for title, a, b, turns in comparisons:
                try:
                    r = compare(exp_root, a, b, args.teacher, student, dataset, turns, args.k)
                except FileNotFoundError as e:
                    print(f"[skip] {title}: {e}")
                    continue
                results[key][title] = r
                print(f"{title:46s} {_fmt(r)}")

    out = exp_root / "plots" / "significance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
