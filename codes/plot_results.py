"""
Generate RQ figures for k=1, k=5, k=10 (Pass@k vs Turns).

Reproduces the plot_results_rqs() logic from analysis.ipynb as a standalone script
so all three k-values can be generated without a Jupyter kernel.

Output: experiments/plots/results_figures_rqs/
  {dataset}__T-{teacher}__S-{student}__k{k}__{rq}.pdf
  for k in {1, 5, 10} and rq in {rq1, rq2, rq3}
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Style (matches existing RQ figures from analysis.ipynb) ──────────────────

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

EXP_DIR  = Path("/playpen-ssd2/ambati/interact_plus/experiments")
TEACHERS = ["qwen72"]
STUDENTS = ["mistral7", "qwen7", "gemma9"]
DATASETS = ["gsm8k", "humaneval_opc"]
TURNS    = [1, 3, 5, 7, 9, 11]
K_VALUES = [1, 5, 10]

# ── Method name / file resolution (mirrors the notebook) ─────────────────────

def _resolve(method: str, student: str, dataset: str) -> Tuple[str, str]:
    """Return (folder_name, results_filename) for a given method."""
    results_file = "aggregate_results.json"
    if "gemma" in student:
        results_file = "aggregate_results_0.json"

    if method in ("pre", "mid"):
        return f"{method}_cot", "aggregate_results_0.json"

    if method in ("selfguided", "peerguided"):
        suffix = "pre_cot" if dataset == "gsm8k" else "mid_cot"
        return f"{method}_zero_margin_{suffix}", "aggregate_results_0.json"

    return method, results_file


# ── Label mapping ─────────────────────────────────────────────────────────────

def _label(method: str, tag: str) -> str:
    labels = {
        "unguided":   "unscaffolded",
        "cot":        "scaffolded",
        "selfguided": "self-ODQS",
        "peerguided": "peer-ODQS",
        "pre":        "pre-assessment",
        "mid":        "mid-assessment",
    }
    name = labels.get(method, method)
    if tag == "rq2" and method == "cot":
        name = "no-assessment"
    if tag == "rq3" and method in ("pre", "mid"):
        name = "no-training"
    return name


# ── Core plotting function ────────────────────────────────────────────────────

def plot_rq(
    methods: List[str],
    teachers: List[str],
    students: List[str],
    datasets: List[str],
    kvalues: Iterable[int],
    tag: str,
    y_limits: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> None:
    plot_dir = EXP_DIR / "plots" / "results_figures_rqs"
    plot_dir.mkdir(parents=True, exist_ok=True)
    turns = sorted(TURNS)

    for teacher in teachers:
        for student in students:
            for dataset in datasets:
                for k in kvalues:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    plotted = []

                    for i, method in enumerate(methods):
                        folder, fname = _resolve(method, student, dataset)
                        fpath = EXP_DIR / folder / teacher / student / dataset / fname
                        if not fpath.exists():
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

                        if method == "unguided":
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

                    out = plot_dir / f"{dataset}__T-{teacher}__S-{student}__k{k}__{tag}.pdf"
                    fig.savefig(out, dpi=600, bbox_inches="tight")
                    plt.close(fig)
                    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # RQ1: does interaction help? (unscaffolded vs scaffolded)
    plot_rq(["cot", "unguided"], TEACHERS, STUDENTS, DATASETS, K_VALUES, "rq1")

    # RQ2: does assessment help? (no-assessment vs pre vs mid)
    plot_rq(["cot", "pre", "mid"], TEACHERS, STUDENTS, DATASETS, K_VALUES, "rq2")

    # RQ3: does ODQS training help? (no-training vs self vs peer)
    plot_rq(["pre",  "selfguided", "peerguided"], TEACHERS, STUDENTS, ["gsm8k"],         K_VALUES, "rq3")
    plot_rq(["mid",  "selfguided", "peerguided"], TEACHERS, STUDENTS, ["humaneval_opc"], K_VALUES, "rq3")


if __name__ == "__main__":
    main()
