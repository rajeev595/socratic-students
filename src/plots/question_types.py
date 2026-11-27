"""
Regenerate question-type distribution and outcome plots from existing JSON data.
Reads all question_type_dist_*.json and question_type_correlation_*.json files
under the experiments root and saves updated PDFs next to them.

Usage:
    python -m codes.plot_qtype [--experiments ./experiments]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "text.usetex":    True,
    "font.family":    "serif",
    "font.size":      22,
    "axes.labelsize": 22,
    "xtick.labelsize":22,
    "ytick.labelsize":22,
})

TYPE_LABELS = ["A", "B", "C", "D"]

# Human-readable names for the four student-question types (see the
# judge_question_type_system_prompt in codes/prompts/{math,code}.py).
TYPE_NAMES = {
    "A": "Unfocused",
    "B": "Conceptual",
    "C": "Hypothesis",
    "D": "Grounded",
}

# Soft pastel palette, one distinct hue per type (no red). Pastels sit close
# together, so the in-bar percentage labels and white segment gaps carry the
# distinction too.
TYPE_COLORS = {
    "A": "#AFDDF7",  # sky blue
    "B": "#F4B784",  # peach
    "C": "#8ED3B4",  # mint
    "D": "#AE9DE0",  # lavender
}

# All fills are light, so value labels are black throughout.
TYPE_TEXT_COLORS = {"A": "black", "B": "black", "C": "black", "D": "black"}


PAPER_NAMES = {
    "unscaffolded":                           "No Training",
    "pre_assessment":                            "No Training",
    "mid_assessment":                            "No Training",
    "scaffolded":                         "Scaffolded",
    "self_odqs":             "Self-ODQS",
    "self_odqs_pre_assessment":     "Self-ODQS",
    "self_odqs_mid_assessment":     "Self-ODQS",
    "peer_odqs":             "Peer-ODQS",
    "peer_odqs_pre_assessment":     "Peer-ODQS",
    "peer_odqs_mid_assessment":     "Peer-ODQS",
}

def _paper_name(name: str) -> str:
    return PAPER_NAMES.get(name, name.replace("_", r"\_"))


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_distribution(baseline_name: str, compare_name: str,
                      baseline_dist: Dict, compare_dist: Dict,
                      out_dir: Path, tag: str = "") -> None:
    turns = sorted(baseline_dist.keys(), key=int)
    x = np.arange(len(turns))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), sharey=True)
    compare_label  = _paper_name(compare_name)
    baseline_label = _paper_name(baseline_name)
    fig.suptitle(rf"{compare_label} vs. {baseline_label}",
                 fontsize=22, fontweight="bold", y=0.985)
    for ax, dist, label in [
        (axes[0], baseline_dist, baseline_label),
        (axes[1], compare_dist,  compare_label),
    ]:
        bottoms = np.zeros(len(turns))
        for q_type in TYPE_LABELS:
            vals = np.array([dist.get(t, {}).get(q_type, 0.0) for t in turns]) * 100
            # White edges give the 2px surface gap between stacked segments,
            # which keeps neighbouring segments crisp.
            bars = ax.bar(x, vals, bottom=bottoms,
                          color=TYPE_COLORS[q_type], label=TYPE_NAMES[q_type],
                          width=0.6, edgecolor="white", linewidth=1.2)
            for bar, bot, val in zip(bars, bottoms, vals):
                if val >= 6:
                    ax.text(bar.get_x() + bar.get_width() / 2, bot + val / 2,
                            rf"{val:.0f}\%", ha="center", va="center",
                            fontsize=11, color=TYPE_TEXT_COLORS[q_type])
            bottoms += vals
        ax.set_title(label, fontsize=22)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in range(len(turns))], fontsize=22)
        ax.set_xlabel(r"Turns", fontsize=22)
        ax.set_ylim(0, 103)
        ax.tick_params(axis="y", labelsize=22)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(r"No. of Questions (\%)", fontsize=22)

    # One shared legend in its own strip under the panels, so it can never sit
    # on top of the bars (the stacks fill the full 0-100% height).
    handles = [Patch(facecolor=TYPE_COLORS[t], edgecolor="white",
                     label=TYPE_NAMES[t]) for t in TYPE_LABELS]
    fig.subplots_adjust(left=0.085, right=0.995, top=0.855, bottom=0.215,
                        wspace=0.06)
    fig.legend(handles=handles, loc="lower center", ncol=len(TYPE_LABELS),
               fontsize=19, frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.5, handleheight=1.1,
               columnspacing=1.8, handletextpad=0.6)

    suffix = f"_{tag}" if tag else ""
    for ext in ("png", "pdf"):
        out = out_dir / f"question_type_distribution{suffix}.{ext}"
        fig.savefig(out, dpi=600, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def plot_outcome(correlation_data: Dict, out_dir: Path, tag: str = "") -> None:
    by_type: Dict[str, List[float]] = {t: [] for t in TYPE_LABELS}
    for info in correlation_data.values():
        dom = info.get("dominant_type")
        imp = info.get("improvement")
        if dom in TYPE_LABELS and imp is not None:
            by_type[dom].append(float(imp))

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    data_for_plot = [by_type[t] for t in TYPE_LABELS]
    # Name over count on two lines, so four long names fit without rotating.
    tick_labels = [f"{TYPE_NAMES[t]}\n" + rf"($n={len(by_type[t])}$)"
                   for t in TYPE_LABELS]
    bp = ax.boxplot(data_for_plot, patch_artist=True, tick_labels=tick_labels)
    for patch, q_type in zip(bp["boxes"], TYPE_LABELS):
        patch.set_facecolor(TYPE_COLORS[q_type])
        patch.set_alpha(0.85)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.6)
    for i, vals in enumerate(data_for_plot):
        if vals:
            med = float(np.median(vals))
            ax.text(i + 1, med, rf"{med:.2f}", ha="center", va="bottom",
                    fontsize=11)
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
        out = out_dir / f"question_type_outcome{suffix}.{ext}"
        fig.savefig(out, dpi=600, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", default="./experiments", help="experiments root (paths.experiments)")
    args = parser.parse_args()

    for dist_file in sorted(Path(args.experiments).rglob("question_type_dist_*.json")):
        out_dir = dist_file.parent
        data = json.loads(dist_file.read_text())
        names = list(data.keys())
        if len(names) != 2:
            print(f"[SKIP] unexpected keys in {dist_file}: {names}")
            continue

        baseline_name, compare_name = names[0], names[1]
        baseline_dist = {k: v for k, v in data[baseline_name].items()}
        compare_dist  = {k: v for k, v in data[compare_name].items()}

        # derive tag from filename: question_type_dist_{tag}.json
        stem = dist_file.stem  # e.g. question_type_dist_self_odqs_vs_unscaffolded
        tag  = stem[len("question_type_dist_"):]

        plot_distribution(baseline_name, compare_name,
                          baseline_dist, compare_dist, out_dir, tag)

        # regenerate outcome plot if correlation file exists
        corr_file = out_dir / f"question_type_correlation_{tag}.json"
        if corr_file.exists():
            corr_data = json.loads(corr_file.read_text())
            plot_outcome(corr_data, out_dir, tag)


if __name__ == "__main__":
    main()
