"""
Extract samples for blind A/B preference study: No-Training vs Self-ODQS.

Both conditions use the identical scaffold; the only variable is whether DPO
training was applied.  Human annotators see a problem statement and two Turn-0
student questions (randomly labeled A/B) and judge which question requires the
tutor to explain a mathematical/programming idea to answer it.

Scaffold per task:
  math   (gsm8k)        : pre_cot  vs selfguided_zero_margin_pre_cot
  coding (humaneval_opc): mid_cot  vs selfguided_zero_margin_mid_cot

Outputs (all under --out_dir):
  - preference_samples_{student}_{dataset}.jsonl  -- one item per line (includes label_a, label_b)
  - preference_samples_{student}_{dataset}.csv    -- for AMT/Prolific upload (ASCII-safe)

Usage:
  python -m codes.human_eval.extract_question_preference_samples \
      --task math   --student mistral7 [--n_problems 50] [--seed 42] [--out_dir experiments/human_eval/data]
  python -m codes.human_eval.extract_question_preference_samples \
      --task coding --student mistral7 [--n_problems 50] [--seed 42] [--out_dir experiments/human_eval/data]
"""

import json
import csv
import argparse
import hashlib
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

TASK_CONFIGS = {
    "math": {
        "dataset":      "gsm8k",
        "split_dir":    "splits_0.5",
        "baseline_exp": "pre_cot",
        "compare_exp":  "selfguided_zero_margin_pre_cot",
        "trans_subdir": "transcripts_0",
    },
    "coding": {
        "dataset":      "humaneval_opc",
        "split_dir":    "splits_0.3",
        "baseline_exp": "mid_cot",
        "compare_exp":  "selfguided_zero_margin_mid_cot",
        "trans_subdir": "transcripts_0",
    },
}

BASELINE_LABEL   = "no_training"
COMPARE_LABEL    = "self_odqs"
STUDENT_TURN_IDX = 1  # index 0 = teacher greeting, index 1 = first student turn (Turn 0)

CSV_FIELDNAMES = ["item_id", "problem_id", "student", "dataset", "turn_idx",
                  "label_a", "label_b", "problem_prompt", "question_a", "question_b"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_for_amt(text: str) -> str:
    """Replace common Unicode punctuation/symbols with ASCII equivalents."""
    if not isinstance(text, str):
        return text
    replacements = {
        "‘": "'", "’": "'",
        "“": '"', "”": '"',
        "–": "-", "—": "--",
        "…": "...",
        "×": "x", "÷": "/", "−": "-",
        "≥": ">=", "≤": "<=", "≠": "!=",
        "π": "pi", "√": "sqrt",
    }
    for uni, asc in replacements.items():
        text = text.replace(uni, asc)
    return text.encode("ascii", errors="replace").decode("ascii")


def _item_id(task: str, problem_id: str) -> str:
    key = f"pref_{task}_{problem_id}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _load_turn(transcript_path: Path, turn_idx: int) -> str | None:
    try:
        turns = json.loads(transcript_path.read_text())
        if len(turns) <= turn_idx:
            return None
        return turns[turn_idx]["content"].strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",          choices=["math", "coding"], default="math")
    parser.add_argument("--student",       default="mistral7")
    parser.add_argument("--n_problems",    type=int, default=50)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--out_dir",       default="experiments/human_eval/data")
    parser.add_argument("--exclude_jsonl", default=None,
                        help="JSONL from a previous run; problem_ids in it are excluded")
    args = parser.parse_args()

    cfg     = TASK_CONFIGS[args.task]
    dataset = cfg["dataset"]
    teacher = "qwen72"

    baseline_dir  = Path(f"experiments/{cfg['baseline_exp']}/{teacher}/{args.student}/{dataset}/{cfg['trans_subdir']}")
    compare_dir   = Path(f"experiments/{cfg['compare_exp']}/{teacher}/{args.student}/{dataset}/{cfg['trans_subdir']}")
    problems_file = Path(f"data/{teacher}/{dataset}/{cfg['split_dir']}/{args.student}_test.json")

    print(f"Task     : {args.task}  |  Student: {args.student}  |  Dataset: {dataset}")
    print(f"Baseline : {baseline_dir}")
    print(f"Compare  : {compare_dir}")

    rng = random.Random(args.seed)

    problems_data = json.loads(problems_file.read_text())
    problem_map   = {str(d["id"]): d["prompt"] for d in problems_data}

    baseline_ids = {p.stem for p in baseline_dir.glob("*.json")}
    compare_ids  = {p.stem for p in compare_dir.glob("*.json")}
    common_ids   = sorted(baseline_ids & compare_ids & set(problem_map.keys()))
    print(f"Common problems: {len(common_ids)}")

    if args.exclude_jsonl:
        used_ids = {json.loads(l)["problem_id"] for l in open(args.exclude_jsonl)}
        common_ids = [pid for pid in common_ids if pid not in used_ids]
        print(f"After excluding {len(used_ids)} already-sampled: {len(common_ids)} remain")

    sampled_ids = rng.sample(common_ids, min(args.n_problems, len(common_ids)))
    print(f"Sampled: {len(sampled_ids)} problems")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for pid in sampled_ids:
        baseline_q = _load_turn(baseline_dir / f"{pid}.json", STUDENT_TURN_IDX)
        compare_q  = _load_turn(compare_dir  / f"{pid}.json", STUDENT_TURN_IDX)
        if not baseline_q or not compare_q:
            print(f"  Skipping {pid}: missing transcript")
            continue

        item_id = _item_id(args.task, pid)
        flip    = rng.random() < 0.5
        if flip:
            question_a, question_b = compare_q,  baseline_q
            label_a,    label_b    = COMPARE_LABEL, BASELINE_LABEL
        else:
            question_a, question_b = baseline_q, compare_q
            label_a,    label_b    = BASELINE_LABEL, COMPARE_LABEL

        items.append({
            "item_id":        item_id,
            "problem_id":     pid,
            "student":        args.student,
            "dataset":        dataset,
            "turn_idx":       STUDENT_TURN_IDX,
            "label_a":        label_a,
            "label_b":        label_b,
            "problem_prompt": problem_map[pid],
            "question_a":     question_a,
            "question_b":     question_b,
        })

    print(f"Final items: {len(items)}")

    batch_suffix = f"_batch{args.seed}" if args.exclude_jsonl else ""
    tag = f"{args.student}_{dataset}{batch_suffix}"

    jsonl_path = out_dir / f"preference_samples_{tag}.jsonl"
    with open(jsonl_path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    print(f"Saved JSONL → {jsonl_path}")

    csv_path = out_dir / f"preference_samples_{tag}.csv"
    with open(csv_path, "w", newline="", encoding="ascii", errors="replace") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for item in items:
            row = {k: sanitize_for_amt(v) if isinstance(v, str) else v
                   for k, v in item.items()}
            writer.writerow(row)
    print(f"Saved CSV  → {csv_path}")


if __name__ == "__main__":
    main()
