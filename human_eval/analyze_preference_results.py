"""
Analyze human preference annotation results from Amazon Mechanical Turk.

For each item, 3 workers judged which of two student questions (A or B) requires
more mathematical explanation from the tutor.  label_a / label_b record which
condition was assigned to each slot (no_training or self_odqs).

Inputs:
  --scores   : CSV downloaded from AMT (one row per assignment)
  --samples  : preference_samples_{student}_{dataset}.jsonl
               (from extract_question_preference_samples.py — has label_a/label_b)

Outputs:
  - Per-item majority vote and agreement
  - Win-rate table: no_training vs self_odqs
  - Inter-rater agreement (% exact match across 3 workers)
  - Summary JSON saved to same dir as --scores

Usage:
  python -m human_eval.analyze_preference_results \
      --scores  experiments/human_eval/data/scores_amt_preference_samples_qwen7_gsm8k.csv \
      --samples experiments/human_eval/data/preference_samples_qwen7_gsm8k.jsonl
"""

import json
import csv
import argparse
from collections import defaultdict, Counter
from pathlib import Path


def majority_vote(votes: list[str]) -> str:
    """Return majority label; 'tie' if no majority (all different)."""
    c = Counter(votes)
    top = c.most_common(1)[0]
    if top[1] >= 2:
        return top[0]
    return "no_majority"


def pct(n, d):
    return 100 * n / d if d else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores",  required=True)
    parser.add_argument("--samples", required=True)
    args = parser.parse_args()

    # ── Load label map from JSONL ──────────────────────────────────────────
    label_map = {}   # item_id -> {label_a, label_b, problem_id}
    for line in open(args.samples):
        item = json.loads(line)
        label_map[item["item_id"]] = {
            "label_a":    item["label_a"],
            "label_b":    item["label_b"],
            "problem_id": item["problem_id"],
        }

    # ── Load AMT scores ────────────────────────────────────────────────────
    # AMT result column for the preference is Answer.preference
    rows = list(csv.DictReader(open(args.scores)))
    submitted = [r for r in rows if r.get("AssignmentStatus", "") != "Rejected"]
    print(f"Loaded {len(submitted)} submitted assignments across "
          f"{len(set(r['Input.item_id'] for r in submitted))} items\n")

    # ── Build label map: JSONL first, fall back to Input.label_a/b in scores ─
    # (Fallback handles the case where item_ids changed after CSV was uploaded)
    for r in submitted:
        iid = r["Input.item_id"]
        if iid not in label_map and r.get("Input.label_a") and r.get("Input.label_b"):
            label_map[iid] = {
                "label_a":    r["Input.label_a"],
                "label_b":    r["Input.label_b"],
                "problem_id": r.get("Input.problem_id", ""),
            }

    # ── Group votes by item ────────────────────────────────────────────────
    votes_by_item = defaultdict(list)   # item_id -> [raw votes: "A"/"B"/"tie"]
    for r in submitted:
        iid  = r["Input.item_id"]
        vote = r["Answer.preference"].strip()
        if not vote:
            vote = r.get("Answer._pref", "").strip()
        votes_by_item[iid].append(vote)

    # ── Resolve to condition-level preference ──────────────────────────────
    condition_votes = defaultdict(list)   # item_id -> condition chosen per worker

    for iid, votes in votes_by_item.items():
        info = label_map.get(iid)
        if not info:
            print(f"  WARNING: item {iid} not found in samples JSONL or scores CSV")
            continue
        la, lb = info["label_a"], info["label_b"]
        for v in votes:
            if v == "A":
                condition_votes[iid].append(la)
            elif v == "B":
                condition_votes[iid].append(lb)
            else:
                condition_votes[iid].append("tie")

    # ── Agreement ──────────────────────────────────────────────────────────
    unanimous   = sum(1 for vs in condition_votes.values() if len(set(vs)) == 1)
    majority_2  = sum(1 for vs in condition_votes.values()
                      if Counter(vs).most_common(1)[0][1] == 2)
    no_majority = sum(1 for vs in condition_votes.values()
                      if Counter(vs).most_common(1)[0][1] == 1)
    n_items = len(condition_votes)

    print("── Inter-rater agreement ──────────────────────────────────")
    print(f"  Unanimous (3/3 agree) : {unanimous:3d} / {n_items}  ({pct(unanimous, n_items):.1f}%)")
    print(f"  Majority  (2/3 agree) : {majority_2:3d} / {n_items}  ({pct(majority_2, n_items):.1f}%)")
    print(f"  No majority           : {no_majority:3d} / {n_items}  ({pct(no_majority, n_items):.1f}%)\n")

    # ── Win-rate breakdown ─────────────────────────────────────────────────
    # Count majority votes per condition
    mv_counter = Counter()
    for iid, vs in condition_votes.items():
        mv_counter[majority_vote(vs)] += 1

    conditions = sorted(set(
        c for vs in condition_votes.values() for c in vs if c != "tie"
    ))

    print("── Majority-vote win rates ────────────────────────────────")
    for cond in conditions:
        n = mv_counter.get(cond, 0)
        print(f"  {cond:40s}: {n:3d} / {n_items}  ({pct(n, n_items):.1f}%)")
    n_tie = mv_counter.get("tie", 0) + mv_counter.get("no_majority", 0)
    print(f"  {'tie / no majority':40s}: {n_tie:3d} / {n_items}  ({pct(n_tie, n_items):.1f}%)\n")

    # ── Raw vote breakdown (all 150 votes) ─────────────────────────────────
    all_cond_votes = [c for vs in condition_votes.values() for c in vs]
    raw_counter = Counter(all_cond_votes)
    n_total = len(all_cond_votes)

    print("── Raw vote breakdown (all worker votes) ──────────────────")
    for cond in conditions:
        n = raw_counter.get(cond, 0)
        print(f"  {cond:40s}: {n:3d} / {n_total}  ({pct(n, n_total):.1f}%)")
    n_tie_raw = raw_counter.get("tie", 0)
    print(f"  {'tie':40s}: {n_tie_raw:3d} / {n_total}  ({pct(n_tie_raw, n_total):.1f}%)\n")

    # ── Per-item detail ────────────────────────────────────────────────────
    print("── Per-item detail ────────────────────────────────────────")
    print(f"  {'item_id':14s}  {'label_a':25s}  {'label_b':25s}  votes          majority")
    print(f"  {'-'*14}  {'-'*25}  {'-'*25}  {'-'*14}  {'-'*12}")
    for iid in sorted(condition_votes):
        info = label_map[iid]
        vs   = condition_votes[iid]
        mv   = majority_vote(vs)
        print(f"  {iid:14s}  {info['label_a']:25s}  {info['label_b']:25s}  {str(vs):14s}  {mv}")

    # ── Save summary ───────────────────────────────────────────────────────
    summary = {
        "n_items":    n_items,
        "n_votes":    n_total,
        "agreement":  {
            "unanimous":   unanimous,
            "majority_2":  majority_2,
            "no_majority": no_majority,
        },
        "majority_vote_wins": dict(mv_counter),
        "raw_vote_counts":    dict(raw_counter),
        "per_item": {
            iid: {
                "label_a":  label_map[iid]["label_a"],
                "label_b":  label_map[iid]["label_b"],
                "votes":    condition_votes[iid],
                "majority": majority_vote(condition_votes[iid]),
            }
            for iid in condition_votes
        },
    }
    out_path = Path(args.scores).parent / (
        Path(args.scores).stem.replace("scores_amt_", "analysis_preference_") + ".json"
    )
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary → {out_path}")


if __name__ == "__main__":
    main()
