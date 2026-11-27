"""
Plot the 2x2 cross-evaluation ablation results.

Four conditions per (model, task):
  C1: NT Generator  + NT Evaluator   (baseline)
  C3: ODQS Generator + NT Evaluator  (generator effect)
  C4: NT Generator  + ODQS Evaluator (evaluator effect)
  C2: ODQS Generator + ODQS Evaluator (full system, reported result)

Layout: 2×2 grid — rows = models (Qwen7, Mistral7), columns = tasks (Math, Code)
Output: experiments/plots/cross_eval_ablation.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Style (matches plot_results_rqs.py) ──────────────────────────────────────

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

# Actual turn keys in the JSON files
TURNS = [1, 3, 5, 7, 9, 11]
# X-axis positions shown as 0, 1, 2, 3, 4, 5
X = list(range(len(TURNS)))

BASE = Path("/playpen-ssd2/ambati/interact_plus/experiments")


# ── Data loading ──────────────────────────────────────────────────────────────

def load(path: Path) -> dict:
    with open(path) as f:
        d = json.load(f)
    return {int(t): (d[t]["5"][0], d[t]["5"][1]) for t in d}


PATHS = {
    "qwen7": {
        "math": {
            "C1_NT_NT":     BASE / "unguided/qwen72/qwen7/gsm8k/aggregate_results.json",
            "C2_ODQS_ODQS": BASE / "selfguided_zero_margin/qwen72/qwen7/gsm8k/aggregate_results_0.json",
            "C3_ODQS_NT":   BASE / "cross_eval/gen_qwen7_dpo__eval_qwen7/qwen72/gsm8k/aggregate_results.json",
            "C4_NT_ODQS":   BASE / "cross_eval/gen_qwen7__eval_qwen7_dpo/qwen72/gsm8k/aggregate_results.json",
        },
        "code": {
            "C1_NT_NT":     BASE / "unguided/qwen72/qwen7/humaneval_opc/aggregate_results.json",
            "C2_ODQS_ODQS": BASE / "selfguided_zero_margin/qwen72/qwen7/humaneval_opc/aggregate_results_0.json",
            "C3_ODQS_NT":   BASE / "cross_eval/gen_qwen7_dpo__eval_qwen7/qwen72/humaneval_opc/aggregate_results.json",
            "C4_NT_ODQS":   BASE / "cross_eval/gen_qwen7__eval_qwen7_dpo/qwen72/humaneval_opc/aggregate_results.json",
        },
    },
    "mistral7": {
        "math": {
            "C1_NT_NT":     BASE / "unguided/qwen72/mistral7/gsm8k/aggregate_results.json",
            "C2_ODQS_ODQS": BASE / "selfguided_zero_margin/qwen72/mistral7/gsm8k/aggregate_results_0.json",
            "C3_ODQS_NT":   BASE / "cross_eval/gen_mistral7_dpo__eval_mistral7/qwen72/gsm8k/aggregate_results.json",
            "C4_NT_ODQS":   BASE / "cross_eval/gen_mistral7__eval_mistral7_dpo/qwen72/gsm8k/aggregate_results.json",
        },
        "code": {
            "C1_NT_NT":     BASE / "unguided/qwen72/mistral7/humaneval_opc/aggregate_results.json",
            "C2_ODQS_ODQS": BASE / "selfguided_zero_margin/qwen72/mistral7/humaneval_opc/aggregate_results_0.json",
            "C3_ODQS_NT":   BASE / "cross_eval/gen_mistral7_dpo__eval_mistral7/qwen72/humaneval_opc/aggregate_results.json",
            "C4_NT_ODQS":   BASE / "cross_eval/gen_mistral7__eval_mistral7_dpo/qwen72/humaneval_opc/aggregate_results.json",
        },
    },
}


def _draw(ax, model, task):
    for cond_key, color, marker, label in CONDITIONS:
        data    = load(PATHS[model][task][cond_key])
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
    models = ["qwen7", "mistral7"]
    tasks  = ["math", "code"]

    out_dir = BASE / "plots" / "cross_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    col_labels = {"math": "Math", "code": "Code"}
    row_labels = {"qwen7": "Qwen7", "mistral7": "Mistral7"}

    # ── Combined 2×2 figure ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    for row, model in enumerate(models):
        for col, task in enumerate(tasks):
            ax = axes[row][col]
            _draw(ax, model, task)
            if col == 0:
                ax.set_ylabel(r"$\mathrm{Pass}@5$")
            if row == 1:
                ax.set_xlabel(r"Turns")
            # Column header on top row only; row label on right column only
            if row == 0:
                ax.set_title(col_labels[task])
            if col == 1:
                ax.annotate(row_labels[model], xy=(1.02, 0.5),
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
            _draw(ax_i, model, task)
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
