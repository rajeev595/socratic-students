#!/usr/bin/env python3
"""
Plot Pass@k vs. Turns across experiments.
(Docstrings remain the same)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

import hydra
from omegaconf import DictConfig

from codes.utils.llm.server import _MODEL_ALIASES  # Dict[str, Tuple[str, int | None]]

# --- Matplotlib defaults, Palettes, and Markers remain the same ---
plt.rcParams.update({
    "font.size": 24,
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "legend.fontsize": 18,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "lines.markersize": 12,
    "lines.linewidth": 2,
})

OKABE_ITO = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # yellow-green
    "#17becf",  # cyan
    "#000000",  # black
]

MARKERS = ["o", "P", "*", "s", "^", "D", "v", "h", ">", "<", "X"]
STYLES = [(c, m) for c, m in zip(OKABE_ITO, MARKERS)]

# --- Data structures & IO functions remain the same ---
@dataclass(frozen=True)
class Record:
    root: str
    experiment: str
    teacher: str
    student: str
    dataset: str
    turn: int
    k: int
    mean: float
    std: Optional[float]

def _iter_aggregate_files(root: Path) -> Iterable[Tuple[Path, str, str, str, str]]:
    # ... (code is unchanged)
    for exp_dir in root.iterdir():
        if not exp_dir.is_dir(): continue
        experiment = exp_dir.name
        for teacher_dir in exp_dir.iterdir():
            if not teacher_dir.is_dir(): continue
            teacher = teacher_dir.name
            for student_dir in teacher_dir.iterdir():
                if not student_dir.is_dir(): continue
                student = student_dir.name
                for dataset_dir in student_dir.iterdir():
                    if not dataset_dir.is_dir(): continue
                    dataset = dataset_dir.name
                    json_path = dataset_dir / "aggregate_results.json"
                    if json_path.is_file():
                        yield json_path, experiment, teacher, student, dataset

def _load_json_records(json_path: Path, meta: Tuple[str, str, str, str]) -> List[Record]:
    # ... (code is unchanged)
    experiment, teacher, student, dataset = meta
    try:
        with open(json_path, "r") as f: data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {json_path}: {e}"); return []
    recs: List[Record] = []
    for turn_str, k_map in data.items():
        try: turn = int(turn_str)
        except Exception: continue
        if not isinstance(k_map, dict): continue
        for k_str, val in k_map.items():
            try: k = int(k_str)
            except Exception: continue
            mean: Optional[float] = None; std: Optional[float] = None
            if isinstance(val, list) and len(val) > 0:
                mean = float(val[0])
                if len(val) > 1 and val[1] is not None:
                    try: std = float(val[1])
                    except Exception: std = None
            elif isinstance(val, (int, float)): mean = float(val)
            if mean is None: continue
            recs.append(Record(root=str(json_path), experiment=experiment, teacher=teacher, student=student, dataset=dataset, turn=turn, k=k, mean=mean, std=std))
    return recs

def _collect_dataframe(root: Path, allow_experiments: Optional[List[str]] = None, allow_teachers: Optional[List[str]] = None, allow_students: Optional[List[str]] = None, allow_datasets: Optional[List[str]] = None) -> pd.DataFrame:
    # ... (code is unchanged)
    rows: List[Record] = []
    for json_path, experiment, teacher, student, dataset in _iter_aggregate_files(root):
        if allow_experiments and experiment not in allow_experiments: continue
        if allow_teachers and teacher not in allow_teachers: continue
        if allow_students and student not in allow_students: continue
        if allow_datasets and dataset not in allow_datasets: continue
        rows.extend(_load_json_records(json_path, (experiment, teacher, student, dataset)))
    if not rows: return pd.DataFrame(columns=["experiment", "teacher", "student", "dataset", "turn", "k", "mean", "std"])
    df = pd.DataFrame([r.__dict__ for r in rows])
    df.sort_values(["student", "dataset", "k", "experiment", "teacher", "turn"], inplace=True)
    return df

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _full_model_name(student: str) -> str:
    # ... (code is unchanged)
    if student in _MODEL_ALIASES:
        full = _MODEL_ALIASES[student]
        return full
    return student

def _dataset_type(dataset: str) -> str:
    # ... (code is unchanged)
    dataset = dataset.lower()
    if "humaneval" in dataset: return "Code"
    if "gsm" in dataset or "math" in dataset or "mmlu" in dataset: return "Math"
    if "code" in dataset or "codex" in dataset: return "Code"
    return "General"


def plot_passk_vs_turns(
    df: pd.DataFrame,
    outdir: Path,
    ks: List[int],
    style_map: Dict[str, Tuple[str, str]],  # MODIFIED: Accept the style map
    include_teacher_in_label: bool = False,
    errorbars: bool = False,
    usetex: bool = False,
) -> List[Path]:
    """Create one plot per (student, dataset, k), overlaying lines for each experiment (and teacher).

    Returns list of saved image paths.
    """
    saved: List[Path] = []
    if df.empty:
        print("[WARN] No data to plot.")
        return saved

    df["turn"] = df["turn"].astype(int)
    df["k"] = df["k"].astype(int)

    for (student, dataset) in sorted(df.groupby(["student", "dataset"]).groups.keys()):
        sub_sd = df[(df["student"] == student) & (df["dataset"] == dataset)]
        full_name = _full_model_name(student)
        for k in ks:
            sub = sub_sd[sub_sd["k"] == k]
            if sub.empty:
                continue

            fig = plt.figure(figsize=(9.6, 5.2))
            ax = fig.add_subplot(111)

            group_cols = ["experiment", "teacher"] if include_teacher_in_label else ["experiment"]

            for idx, (key, g) in enumerate(sub.groupby(group_cols)):
                label = "/".join(str(x) for x in key) if isinstance(key, tuple) else str(key)
                if label == "pre_assessment":
                    label = "pre"
                elif label == "mid_assessment":
                    label = "mid"
                elif label == "selfguided":
                    label = "self"
                elif label == "peerguided":
                    label = "peer"

                g = g.sort_values("turn")

                # The key from groupby is a tuple. The experiment name is always the first element.
                # We check isinstance for robustness in case the groupby key is ever a string.
                if isinstance(key, tuple):
                    style_key = key[0]
                else:
                    style_key = key
                
                # Look up the style from the map, with a fallback for safety
                if style_key in style_map:
                    col, mrk = style_map[style_key]
                else:
                    # Fallback for experiments not in the config list
                    print(f"[WARN] Experiment '{style_key}' not in style map. Assigning temporary style.")
                    col, mrk = STYLES[idx % len(STYLES)]

                if errorbars and g["std"].notna().any():
                    ax.errorbar(
                        g["turn"].values, g["mean"].values, yerr=g["std"].fillna(0.0).values,
                        linestyle="-", marker=mrk, color=col, ecolor=col,
                        capsize=3, elinewidth=1, label=label,
                    )
                else:
                    ax.plot(
                        g["turn"].values, g["mean"].values,
                        linestyle="-", marker=mrk, color=col, label=label,
                    )

            # Axes & labels (LaTeX-style mathtext)
            if usetex:
                plt.rcParams["text.usetex"] = True
            ax.set_xlabel(r"Turn $t$")
            ax.set_ylabel(rf"$\mathrm{{Pass}}@{k}$")
            ax.set_ylim(0.0, 1.0)
            ax.set_xlim(left=max(0, sub["turn"].min() - 0.25))
            ax.set_xticks(sorted(sub["turn"].unique()))
            ax.set_title(f"{full_name} — {_dataset_type(dataset)}")

            # Niceties
            ax.grid(True, which='major', linestyle='--', alpha=0.6)
            ax.legend(ncol=2, loc='best', frameon=True)
            fig.tight_layout()

            out_path = outdir / f"{student}__{dataset}__pass_at_{k}.pdf"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path)
            plt.close(fig)
            saved.append(out_path)
    return saved


# -----------------------------------------------------------------------------
# Main (Hydra)
# -----------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="../configs/", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    # Defaults if keys are missing in config
    experiments = cfg.get("experiments", ["cot", "mid_assessment", "pre_assessment", "unguided", "selfguided", "peerguided"])
    teachers = cfg.get("teachers", ["gpt-4o"])
    students = cfg.get("students", ["mistral7", "llama8", "qwen7"])
    datasets = cfg.get("datasets", ["gsm8k", "humaneval_opc"])

    root = Path(cfg.paths.experiments).expanduser().resolve()
    outdir = Path(os.path.join(root, "plots")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    
    # --- MODIFIED: Create the persistent style map ---
    # Sort experiments to ensure deterministic style assignment
    all_experiments = sorted(experiments)
    style_map = {exp: STYLES[i % len(STYLES)] for i, exp in enumerate(all_experiments)}
    print("[INFO] Style map created for experiments:")
    for exp, (color, marker) in style_map.items():
        print(f"  - {exp:<20} -> Color: {color}, Marker: {marker}")
    # --- End of modification ---

    df = _collect_dataframe(
        root, allow_experiments=experiments, allow_teachers=teachers,
        allow_students=students, allow_datasets=datasets,
    )

    if df.empty:
        print("[WARN] No records found with the given filters.")
        return

    csv_fpath = os.path.join(root, "aggregate_results.csv")
    try:
        df.to_csv(csv_fpath, index=False)
        print(f"[INFO] Saved dataframe to {csv_fpath}")
    except Exception as e:
        print(f"[WARN] Failed to save CSV: {e}")

    k_values = cfg.get("gen", {}).get("k_values", [1, 5, 10])
    include_teacher = cfg.get("plot", {}).get("include_teacher_in_label", False)
    errorbars = cfg.get("plot", {}).get("errorbars", False)
    usetex = cfg.get("plot", {}).get("usetex", False)

    # MODIFIED: Pass the style_map to the plotting function
    saved = plot_passk_vs_turns(
        df, outdir, ks=list(map(int, k_values)),
        style_map=style_map,
        include_teacher_in_label=include_teacher,
        errorbars=bool(errorbars), usetex=bool(usetex),
    )

    if saved:
        print("[INFO] Saved charts:")
        for pth in saved:
            print(f" - {pth}")
    else:
        print("[WARN] No charts saved (no matching (student, dataset, k) groups).")


if __name__ == "__main__":
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    main()
