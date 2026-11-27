"""
Rebuild the curated problem sets used in the paper from the released problem ids.

`problem_ids/<teacher>/<dataset>/<student>.json` lists, in order, the ids of the problems in the
train / val / test split of every student. This script downloads the source datasets and writes the
full problem files that the rest of the pipeline reads:

    <data>/<teacher>/<dataset>/splits_<threshold>/<student>_{train,val,test}.json

Ids:
  gsm8k           row index in the train split of `openai/gsm8k` (config `main`)
  humaneval_opc   "HumanEval/<n>" = task id in `openai_humaneval`
                  "opc/<seq_id>"  = seq_id in `OpenCoder-LLM/opc-sft-stage2` (config `educational_instruct`)

Usage:
    python -m codes.data.build_splits [--data ./data] [--teacher qwen72] [--students qwen7 mistral7]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List

from datasets import load_dataset

from codes.data.code.sample_hard import organize_humaneval_problem, organize_opc_problem
from codes.data.math.sample_hard import organize_gsm_row

IDS_ROOT = Path(__file__).resolve().parents[2] / "problem_ids"


def _gsm8k_lookup() -> Callable[[Any], Dict[str, Any]]:
    ds = load_dataset("openai/gsm8k", "main", split="train")

    def get(pid: Any) -> Dict[str, Any]:
        row = dict(ds[int(pid)])
        row["id"] = str(pid)
        return organize_gsm_row(row)

    return get


def _humaneval_opc_lookup() -> Callable[[Any], Dict[str, Any]]:
    humaneval = {int(r["task_id"].split("/")[-1]): r for r in load_dataset("openai_humaneval")["test"]}
    opc_ds = load_dataset("OpenCoder-LLM/opc-sft-stage2", split="train", name="educational_instruct")
    opc_row = {int(seq_id): i for i, seq_id in enumerate(opc_ds["seq_id"])}

    def get(pid: Any) -> Dict[str, Any]:
        source, num = str(pid).split("/")
        if source == "HumanEval":
            return organize_humaneval_problem(humaneval[int(num)])
        if source == "opc":
            return organize_opc_problem(opc_ds[opc_row[int(num)]])
        raise ValueError(f"Unknown problem id: {pid!r}")

    return get


LOOKUPS = {"gsm8k": _gsm8k_lookup, "humaneval_opc": _humaneval_opc_lookup}


def build(data_root: Path, teacher: str, dataset: str, students: List[str], threshold: float) -> None:
    get = None
    for student in students:
        ids_file = IDS_ROOT / teacher / dataset / f"{student}.json"
        if not ids_file.exists():
            print(f"[skip] no problem ids for {teacher}/{dataset}/{student}")
            continue
        get = get or LOOKUPS[dataset]()
        out_dir = data_root / teacher / dataset / f"splits_{threshold}"
        os.makedirs(out_dir, exist_ok=True)
        for split, ids in json.loads(ids_file.read_text()).items():
            problems = [get(pid) for pid in ids]
            out = out_dir / f"{student}_{split}.json"
            out.write_text(json.dumps(problems, indent=2))
            print(f"{len(problems):5d} problems -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="./data", help="data root (paths.data)")
    parser.add_argument("--teacher", default="qwen72")
    parser.add_argument("--students", nargs="+", default=["qwen7", "mistral7"])
    parser.add_argument("--datasets", nargs="+", default=list(LOOKUPS), choices=list(LOOKUPS))
    parser.add_argument("--threshold", type=float, default=0.5, help="gen.pass_at_k_threshold (names the output folder)")
    args = parser.parse_args()

    for dataset in args.datasets:
        build(Path(args.data), args.teacher, dataset, args.students, args.threshold)


if __name__ == "__main__":
    main()
