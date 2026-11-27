"""
Analysis figures that are not part of the three main research questions.

  progress             LLM-judge progress score per turn for every method
                       (needs: python -m src.analysis.progress ... for each method)
  assessment-position  Pass@5 when a single assessment is inserted at turn 0..5
                       (needs: experiments named assessment_turn_0 ... assessment_turn_5)
  teacher-size         Pass@5 of one student with teachers of different sizes
                       (needs: unscaffolded runs with teacher.model=<each teacher>)

Output: <experiments>/plots/ablations/

Usage:
    python -m src.plots.ablations progress            [--experiments ./experiments] [--student qwen7] [--dataset gsm8k]
    python -m src.plots.ablations assessment-position [--experiments ./experiments] [--student qwen7]
    python -m src.plots.ablations teacher-size        [--experiments ./experiments] [--student qwen7] \\
                                                       [--teachers qwen7 qwen14 qwen32 qwen72]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.plots.results import DATASETS, LINE_COLORS, MARKERS, TURNS, experiment_folder, find_results

PROGRESS_METHODS = [
    ("unscaffolded",   "unscaffolded"),
    ("scaffolded",     "scaffolded"),
    ("pre_assessment", "pre-assessment"),
    ("mid_assessment", "mid-assessment"),
    ("self_odqs",      "self-ODQS"),
    ("peer_odqs",      "peer-ODQS"),
]


def _find_analysis_file(exp_dir: Path, fname: str, judge: Optional[str] = None) -> Optional[Path]:
    """The analyses write to <exp_dir>/analysis; analysis_<judge> is also accepted."""
    candidates = ([exp_dir / f"analysis_{judge}"] if judge else []) + [exp_dir / "analysis"] + sorted(exp_dir.glob("analysis_*"))
    for d in candidates:
        if (d / fname).exists():
            return d / fname
    return None


def _finish(fig, ax, xs: Sequence[int], ylabel: str, out: Path, ylim: Tuple[float, float] = (0.0, 1.0), ncol: int = 1) -> None:
    ax.set_xlabel(r"Turns")
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(xs))
    ax.set_ylim(*ylim)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=False, ncol=ncol)
    ax.margins(x=0.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def _curve(ax, i: int, xs, means, stds, label: str) -> None:
    color, marker = LINE_COLORS[i % len(LINE_COLORS)], MARKERS[i % len(MARKERS)]
    ax.plot(xs, means, marker=marker, markersize=5.5, linewidth=1.8, color=color, label=label)
    if stds is not None:
        ax.errorbar(xs, means, yerr=stds, fmt="none", ecolor=color, elinewidth=1.1, capsize=3)


def _pass_at_k(fpath: Path, k: int) -> Tuple[List[float], List[float]]:
    data = json.loads(fpath.read_text())
    return [data[str(t)][str(k)][0] for t in TURNS], [data[str(t)][str(k)][1] for t in TURNS]


# ── progress ─────────────────────────────────────────────────────────────────

def load_progress(exp_root: Path, method: str, teacher: str, student: str, dataset: str, judge: Optional[str] = None):
    """Per-turn (mean, std) progress scores of one method, ordered by turn; None if the analysis was not run."""
    exp_dir = exp_root / experiment_folder(method, dataset) / teacher / student / dataset
    fpath = _find_analysis_file(exp_dir, "progress_overall.json", judge)
    if fpath is None:
        return None
    per_turn = sorted(json.loads(fpath.read_text())["per_turn"], key=lambda m: m["turn"])
    per_turn = [m for m in per_turn if m["mean"] is not None]
    return [m["mean"] for m in per_turn], [m["std"] for m in per_turn]


def plot_progress(exp_root: Path, out_dir: Path, teacher: str, student: str, dataset: str, judge: Optional[str]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    n = 0
    for i, (method, label) in enumerate(PROGRESS_METHODS):
        loaded = load_progress(exp_root, method, teacher, student, dataset, judge)
        if loaded is None:
            print(f"[skip] no progress scores for {method}")
            continue
        means, stds = loaded
        _curve(ax, i, range(len(means)), means, stds, label)
        n = max(n, len(means))
    if n == 0:
        plt.close(fig)
        return
    _finish(fig, ax, range(n), r"Progress", out_dir / f"{dataset}__T-{teacher}__S-{student}__progress.pdf")


# ── assessment position ──────────────────────────────────────────────────────

def plot_assessment_position(exp_root: Path, out_dir: Path, teacher: str, student: str, dataset: str, k: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = 0
    for pos in range(len(TURNS)):
        fpath = find_results(exp_root / f"assessment_turn_{pos}" / teacher / student / dataset)
        if fpath is None:
            print(f"[skip] no results for assessment_turn_{pos}")
            continue
        means, stds = _pass_at_k(fpath, k)
        _curve(ax, plotted, range(len(TURNS)), means, stds, str(pos))
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    _finish(fig, ax, range(len(TURNS)), rf"$\mathrm{{Pass}}@{k}$",
            out_dir / f"{dataset}__T-{teacher}__S-{student}__k{k}__assessment_position.pdf", ncol=2)


# ── teacher size ─────────────────────────────────────────────────────────────

def plot_teacher_size(exp_root: Path, out_dir: Path, teachers: List[str], student: str, dataset: str, k: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = 0
    for teacher in teachers:
        fpath = find_results(exp_root / "unscaffolded" / teacher / student / dataset)
        if fpath is None:
            print(f"[skip] no unscaffolded results with teacher {teacher}")
            continue
        means, _ = _pass_at_k(fpath, k)
        _curve(ax, plotted, range(len(TURNS)), means, None, teacher)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    _finish(fig, ax, range(len(TURNS)), rf"$\mathrm{{Pass}}@{k}$",
            out_dir / f"{dataset}__S-{student}__k{k}__teacher_size.pdf", ncol=2)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figure", choices=["progress", "assessment-position", "teacher-size"])
    parser.add_argument("--experiments", default="./experiments", help="experiments root (paths.experiments)")
    parser.add_argument("--teacher", default="qwen72")
    parser.add_argument("--teachers", nargs="+", default=["qwen7", "qwen14", "qwen32", "qwen72"], help="teacher-size only")
    parser.add_argument("--student", default="qwen7")
    parser.add_argument("--dataset", nargs="+", default=DATASETS)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--judge", default=None, help="progress only: read analysis_<judge>/ instead of analysis/")
    args = parser.parse_args()

    plt.rcParams.update({
        "text.usetex": True, "font.family": "serif", "font.size": 22, "axes.titlesize": 22,
        "axes.labelsize": 22, "legend.fontsize": 18, "xtick.labelsize": 22, "ytick.labelsize": 22,
    })

    exp_root = Path(args.experiments)
    out_dir = exp_root / "plots" / "ablations"
    for dataset in args.dataset:
        if args.figure == "progress":
            plot_progress(exp_root, out_dir, args.teacher, args.student, dataset, args.judge)
        elif args.figure == "assessment-position":
            plot_assessment_position(exp_root, out_dir, args.teacher, args.student, dataset, args.k)
        else:
            plot_teacher_size(exp_root, out_dir, args.teachers, args.student, dataset, args.k)


if __name__ == "__main__":
    main()
