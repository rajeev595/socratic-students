"""
Plot the 2x2 generator-vs-solver ablation (who benefits from ODQS training?).

Four conditions per (student, task), question generator x answer solver:
  Frozen  untrained generator + untrained solver   experiments/unscaffolded
  Qgen    ODQS generator      + untrained solver   experiments/cross_eval/gen_<S>_odqs__eval_<S>
  Solver  untrained generator + ODQS solver        experiments/cross_eval/gen_<S>__eval_<S>_odqs
  Full    ODQS generator      + ODQS solver        experiments/self_odqs

The Qgen and Solver runs include an assessment (Pre for math, Mid for code). With
--matched-assessment, Frozen and Full are read from runs with that same assessment
(experiments/assessment_turn_<t> and experiments/self_odqs_assessment_turn_<t>).

Output: <experiments>/plots/cross_eval/cross_eval_ablation.pdf (+ one PDF per panel)

Usage:
    python -m src.plots.cross_eval [--experiments ./experiments]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Style (matches src/plots/results.py) ──────────────────────────────────────

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

_BLUE   = "#2D9CDB"
_RED    = "#EB5757"
_YELLOW = "#F2C94C"
_GREEN  = "#27AE60"

# Condition styles: (key, color, marker, label)
CONDITIONS = [
    ("C1_NT_NT",     _RED,    "s", "Frozen"),
    ("C3_ODQS_NT",   _YELLOW, "^", "Qgen"),
    ("C4_NT_ODQS",   _GREEN,  "D", "Solver"),
    ("C2_ODQS_ODQS", _BLUE,   "o", "Full"),
]

TURNS = [0, 1, 2, 3, 4, 5]
X = list(range(len(TURNS)))

TEACHER = "qwen72"
DATASET = {"math": "gsm8k", "code": "humaneval_opc"}
LORA_SUFFIX = "_odqs"   # LoRA name used for the trained student: <student>_odqs


# ── Data loading ──────────────────────────────────────────────────────────────

def _find(exp_dir: Path) -> Path:
    """src.interact writes aggregate_results_0.json; src.cross_eval writes aggregate_results.json."""
    for fname in ("aggregate_results_0.json", "aggregate_results.json"):
        if (exp_dir / fname).exists():
            return exp_dir / fname
    raise FileNotFoundError(f"No aggregate results under {exp_dir}")


def load(path: Path) -> dict:
    with open(path) as f:
        d = json.load(f)
    return {int(t): (d[t]["5"][0], d[t]["5"][1]) for t in d}


ASSESSMENT_TURN = {"math": 0, "code": 2}   # turn of the assessment used in the cross-eval runs


def condition_paths(base: Path, model: str, task: str, matched: bool = False) -> dict:
    """
    Result files of the four generator x solver conditions for one (student, task).

    The Qgen/Solver runs (src.cross_eval) include an assessment (Pre for math, Mid for code).
    matched=False  Frozen/Full are the plain runs without assessment (unscaffolded, self_odqs).
    matched=True   Frozen/Full are the runs with the same assessment
                   (assessment_turn_<t>, self_odqs_assessment_turn_<t>).
    """
    ds = DATASET[task]
    trained = f"{model}{LORA_SUFFIX}"
    t = ASSESSMENT_TURN[task]
    frozen = f"assessment_turn_{t}" if matched else "unscaffolded"
    full = f"self_odqs_assessment_turn_{t}" if matched else "self_odqs"
    return {
        "C1_NT_NT":     _find(base / frozen / TEACHER / model / ds),
        "C2_ODQS_ODQS": _find(base / full / TEACHER / model / ds),
        "C3_ODQS_NT":   _find(base / "cross_eval" / f"gen_{trained}__eval_{model}" / TEACHER / ds),
        "C4_NT_ODQS":   _find(base / "cross_eval" / f"gen_{model}__eval_{trained}" / TEACHER / ds),
    }


def _draw(ax, base, model, task, matched=False):
    paths = condition_paths(base, model, task, matched)
    for cond_key, color, marker, label in CONDITIONS:
        data    = load(paths[cond_key])
        means   = np.array([data[t][0] for t in TURNS])
        stderrs = np.array([data[t][1] for t in TURNS])

        ax.plot(X, means, linestyle="-", marker=marker,
                markersize=5.5, linewidth=1.8, color=color, label=label)
        ax.fill_between(X, means - stderrs, means + stderrs,
                        color=color, alpha=0.15)

    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(X)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.margins(x=0.02)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", default="./experiments", help="experiments root (paths.experiments)")
    parser.add_argument("--students", nargs="+", default=["qwen7", "mistral7"])
    parser.add_argument("--matched-assessment", action="store_true",
                        help="use Frozen/Full runs that include the same assessment as the Qgen/Solver runs")
    args = parser.parse_args()

    BASE = Path(args.experiments)
    models = args.students
    tasks  = ["math", "code"]

    out_dir = BASE / "plots" / "cross_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    col_labels = {"math": "Math", "code": "Code"}
    row_labels = {"qwen7": "Qwen-7B", "mistral7": "Mistral-7B"}

    # ── Combined 2×2 figure ──────────────────────────────────────────────────
    fig, axes = plt.subplots(len(models), 2, figsize=(12, 4 * len(models)), sharex=True, sharey=True, squeeze=False)

    for row, model in enumerate(models):
        for col, task in enumerate(tasks):
            ax = axes[row][col]
            _draw(ax, BASE, model, task, args.matched_assessment)
            if col == 0:
                ax.set_ylabel(r"$\mathrm{Pass}@5$")
            if row == len(models) - 1:
                ax.set_xlabel(r"Turns")
            # Column header on top row only; row label on right column only
            if row == 0:
                ax.set_title(col_labels[task])
            if col == 1:
                ax.annotate(row_labels.get(model, model), xy=(1.02, 0.5),
                            xycoords="axes fraction", rotation=270,
                            va="center", ha="left", fontsize=22)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               frameon=True, bbox_to_anchor=(0.5, -0.04), fontsize=18)
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    out_path = out_dir / "cross_eval_ablation.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)

    # ── Individual subplots ───────────────────────────────────────────────────
    for model in models:
        for task in tasks:
            fig_i, ax_i = plt.subplots(figsize=(6, 4))
            _draw(ax_i, BASE, model, task, args.matched_assessment)
            ax_i.set_xlabel(r"Turns")
            ax_i.set_ylabel(r"$\mathrm{Pass}@5$")
            ax_i.legend(ncol=2, loc="lower right", frameon=False, fontsize=16)
            fig_i.tight_layout()
            p = out_dir / f"cross_eval_{model}_{task}.pdf"
            fig_i.savefig(p, bbox_inches="tight")
            plt.close(fig_i)
            print(f"Saved: {p}")


if __name__ == "__main__":
    main()
