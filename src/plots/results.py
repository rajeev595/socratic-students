"""
Pass@k vs. turns for the three research questions (main result figures).

  RQ1  Unscaffolded vs. Scaffolded (with the static baseline)
  RQ2  No-assessment vs. Pre-Assessment vs. Mid-Assessment
  RQ3  No-training vs. Self-ODQS vs. Peer-ODQS

Reads   <experiments>/<experiment>/<teacher>/<student>/<dataset>/aggregate_results_0.json
Writes  <experiments>/plots/results/{dataset}__T-{teacher}__S-{student}__k{k}__{rq}.pdf
        for k in {1, 5, 10} and rq in {rq1, rq2, rq3}

Usage:
    python -m src.plots.results [--experiments ./experiments] [--students qwen7 mistral7]
    python -m src.plots.results --tables      # print turn-wise Pass@5 mean(std) for every method
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 22,
    "axes.titlesize": 22,
    "axes.labelsize": 22,
    "legend.fontsize": 22,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
})

LINE_COLORS = ["#2D9CDB", "#EB5757", "#F2C94C", "#27AE60", "#9B51E0", "#F2994A"]
MARKERS     = ["o", "s", "^", "D", "P"]

TEACHERS = ["qwen72"]
STUDENTS = ["qwen7", "mistral7"]
DATASETS = ["gsm8k", "humaneval_opc"]
TURNS    = [0, 1, 2, 3, 4, 5]       # turn 0 = before any question (static baseline)
K_VALUES = [1, 5, 10]

# The assessment that works best per domain (RQ2); ODQS students are evaluated with it (RQ3).
BEST_ASSESSMENT = {"gsm8k": "pre_assessment", "humaneval_opc": "mid_assessment"}


# ── Experiment folder / results file resolution ──────────────────────────────

def experiment_folder(method: str, dataset: str) -> str:
    """Map a method to its experiment folder name."""
    if method in ("self_odqs", "peer_odqs"):
        return f"{method}_{BEST_ASSESSMENT[dataset]}"
    if method == "no_training":
        return BEST_ASSESSMENT[dataset]
    return method


def find_results(exp_dir: Path, name: str = "aggregate_results") -> Optional[Path]:
    """src.interact writes <name>_0.json; src.cross_eval writes <name>.json."""
    for fname in (f"{name}_0.json", f"{name}.json"):
        if (exp_dir / fname).exists():
            return exp_dir / fname
    return None


# ── Label mapping ─────────────────────────────────────────────────────────────

def _label(method: str, tag: str) -> str:
    labels = {
        "unscaffolded":   "unscaffolded",
        "scaffolded":     "scaffolded",
        "pre_assessment": "pre-assessment",
        "mid_assessment": "mid-assessment",
        "no_training":    "no-training",
        "self_odqs":      "self-ODQS",
        "peer_odqs":      "peer-ODQS",
    }
    if tag == "rq2" and method == "scaffolded":
        return "no-assessment"
    return labels.get(method, method)


# ── Core plotting function ────────────────────────────────────────────────────

def plot_rq(
    exp_root: Path,
    out_dir: Path,
    methods: List[str],
    teachers: List[str],
    students: List[str],
    datasets: List[str],
    kvalues: Iterable[int],
    tag: str,
    y_limits: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    turns = sorted(TURNS)

    for teacher in teachers:
        for student in students:
            for dataset in datasets:
                for k in kvalues:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    plotted = []

                    for i, method in enumerate(methods):
                        folder = experiment_folder(method, dataset)
                        fpath = find_results(exp_root / folder / teacher / student / dataset)
                        if fpath is None:
                            continue
                        try:
                            data  = json.loads(fpath.read_text())
                            means = [data[str(t)][str(k)][0] for t in turns]
                            stds  = [data[str(t)][str(k)][1] for t in turns]
                        except Exception as e:
                            print(f"[WARN] {fpath}: {e}")
                            continue

                        color  = LINE_COLORS[i % len(LINE_COLORS)]
                        marker = MARKERS[i % len(MARKERS)]
                        label  = _label(method, tag)

                        ax.plot(range(len(turns)), means, marker=marker,
                                markersize=5.5, linewidth=1.8, color=color, label=label)
                        ax.errorbar(range(len(turns)), means, yerr=stds,
                                    fmt="none", ecolor=color, elinewidth=1.1, capsize=3)

                        # Static baseline = unscaffolded student before any interaction (turn 0)
                        if method == "unscaffolded":
                            ax.axhline(means[0], linestyle="--", linewidth=1.5,
                                       color=color, alpha=0.8, label="static")

                        plotted.append(method)

                    if not plotted:
                        plt.close(fig)
                        continue

                    ax.set_xlabel(r"Turns")
                    ax.set_ylabel(rf"$\mathrm{{Pass}}@{k}$")
                    ax.set_xticks(range(len(turns)))
                    if y_limits:
                        ax.set_ylim(*y_limits)
                    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
                    ax.legend(frameon=False, ncol=1)
                    ax.margins(x=0.02)
                    fig.tight_layout()

                    out = out_dir / f"{dataset}__T-{teacher}__S-{student}__k{k}__{tag}.pdf"
                    fig.savefig(out, dpi=600, bbox_inches="tight")
                    plt.close(fig)
                    print(f"Saved: {out}")


# ── Result tables ─────────────────────────────────────────────────────────────

TABLE_METHODS = ["unscaffolded", "scaffolded", "rejection_sampling", "rejection_sampling_dpo",
                 "pre_assessment", "mid_assessment", "self_odqs", "peer_odqs"]


def print_tables(exp_root: Path, teachers: List[str], students: List[str], datasets: List[str], k: int = 5) -> None:
    """Turn-wise Pass@k as mean(std) for every method, one table per (student, dataset)."""
    for teacher in teachers:
        for dataset in datasets:
            for student in students:
                print(f"\n{dataset} | teacher={teacher} | student={student} | Pass@{k} mean(std) per turn")
                for method in TABLE_METHODS:
                    fpath = find_results(exp_root / experiment_folder(method, dataset) / teacher / student / dataset)
                    if fpath is None:
                        continue
                    data = json.loads(fpath.read_text())
                    cells = [f"{data[str(t)][str(k)][0]:.3f}({data[str(t)][str(k)][1]:.3f})" for t in TURNS]
                    print(f"  {method:24s}" + "  ".join(cells))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", default="./experiments", help="experiments root (paths.experiments)")
    parser.add_argument("--out", default=None, help="output dir (default: <experiments>/plots/results)")
    parser.add_argument("--teachers", nargs="+", default=TEACHERS)
    parser.add_argument("--students", nargs="+", default=STUDENTS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--k", nargs="+", type=int, default=K_VALUES)
    parser.add_argument("--tables", action="store_true", help="print turn-wise Pass@5 tables instead of plotting")
    args = parser.parse_args()

    exp_root = Path(args.experiments)
    if args.tables:
        print_tables(exp_root, args.teachers, args.students, args.datasets)
        return
    out_dir = Path(args.out) if args.out else exp_root / "plots" / "results"
    common = (exp_root, out_dir)
    sel = (args.teachers, args.students, args.datasets, args.k)

    # RQ1: does interaction help? (scaffolded vs unscaffolded vs static)
    plot_rq(*common, ["scaffolded", "unscaffolded"], *sel, "rq1")

    # RQ2: do assessments help? (no-assessment vs pre vs mid)
    plot_rq(*common, ["scaffolded", "pre_assessment", "mid_assessment"], *sel, "rq2")

    # RQ3: does ODQS training help? (no-training vs self vs peer)
    plot_rq(*common, ["no_training", "self_odqs", "peer_odqs"], *sel, "rq3")


if __name__ == "__main__":
    main()
