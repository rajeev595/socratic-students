"""
Extract teacher-response samples for human leakage evaluation.

Covers all four (student, task) combinations of the teacher-leakage table in the paper.
Each task has five conditions (the assessment type differs between tasks):

  condition        paper name         experiment folder (under experiments/)
  ---------------  -----------------  ------------------------------------------
  unscaffolded     UNSCAFFOLDED       unscaffolded
  scaffolded       SCAFFOLDED         scaffolded
  pre_assessment   +PRE-ASSESSMENT    pre_assessment              (math only)
  mid_assessment   +MID-ASSESSMENT    mid_assessment              (coding only)
  self_odqs        +self-ODQS         self_odqs_{pre,mid}_assessment
  peer_odqs        +peer-ODQS         peer_odqs_{pre,mid}_assessment

Transcripts are read from <experiment folder>/qwen72/<student>/<dataset>/transcripts_0.

Output (one CSV/JSONL pair per (student, task) combination):
  leakage_samples_{student}_{task}.jsonl / .csv   — task items + gold QC items
  gold_items_{task}.jsonl                         — QC anchor items (per task type)
  leakage_samples_all.jsonl / .csv                — merged across all four combinations

Usage:
  python -m human_eval.extract_leakage_samples \\
      --n_per_condition 10 \\
      --seed 42 \\
      --out_dir experiments/human_eval/data
"""

import json
import random
import csv
import argparse
import hashlib
from pathlib import Path


def sanitize_for_amt(text: str) -> str:
    """Replace common Unicode punctuation with ASCII; strip remaining non-ASCII.

    AMT CSV upload rejects non-ASCII characters (emojis, smart quotes, etc.).
    """
    if not isinstance(text, str):
        return text
    replacements = {
        "’": "'",    # right single quotation mark
        "‘": "'",    # left single quotation mark
        "“": '"',    # left double quotation mark
        "”": '"',    # right double quotation mark
        "–": "-",    # en dash
        "—": "--",   # em dash
        "…": "...",  # horizontal ellipsis
        "×": "x",    # multiplication sign
        "·": "*",    # middle dot
        "−": "-",    # minus sign
        "≥": ">=",   # greater-than or equal to
        "≤": "<=",   # less-than or equal to
        "≠": "!=",   # not equal to
        "π": "pi",   # greek small letter pi
        "√": "sqrt", # square root
        "é": "e",    # e with acute
        "à": "a",    # a with grave
        "è": "e",    # e with grave
        "ê": "e",    # e with circumflex
        "ñ": "n",    # n with tilde
    }
    for uni, ascii_eq in replacements.items():
        text = text.replace(uni, ascii_eq)
    # Strip any remaining non-ASCII (emojis, etc.)
    return text.encode("ascii", errors="ignore").decode("ascii")

BASE = Path(__file__).resolve().parents[1]  # repo root

# ---------------------------------------------------------------------------
# Condition maps: (exp_dir_name, transcript_subdir)
# ---------------------------------------------------------------------------
_MATH_CONDITIONS = {
    "unscaffolded":   ("unscaffolded",                         "transcripts_0"),
    "scaffolded":        ("scaffolded",                              "transcripts_0"),
    "pre_assessment":    ("pre_assessment",                          "transcripts_0"),
    "self_odqs":  ("self_odqs_pre_assessment",   "transcripts_0"),
    "peer_odqs":  ("peer_odqs_pre_assessment",   "transcripts_0"),
}
_CODING_CONDITIONS = {
    "unscaffolded":   ("unscaffolded",                         "transcripts_0"),
    "scaffolded":        ("scaffolded",                              "transcripts_0"),
    "mid_assessment":    ("mid_assessment",                          "transcripts_0"),
    "self_odqs":  ("self_odqs_mid_assessment",   "transcripts_0"),
    "peer_odqs":  ("peer_odqs_mid_assessment",   "transcripts_0"),
}

CONDITION_LABELS = {
    "unscaffolded":  "UNSCAFFOLDED",
    "scaffolded":       "SCAFFOLDED",
    "pre_assessment":   "+PRE-ASSESSMENT",
    "mid_assessment":   "+MID-ASSESSMENT",
    "self_odqs": "+self-ODQS",
    "peer_odqs": "+peer-ODQS",
}

# (student, task_type, task_dir_name, problem_file, teacher_role, condition_map)
STUDENT_TASK_MATRIX = [
    (
        "mistral7", "math", "gsm8k",
        BASE / "data/qwen72/gsm8k/splits_0.5/mistral7_test.json",
        "qwen72", _MATH_CONDITIONS,
    ),
    (
        "mistral7", "coding", "humaneval_opc",
        BASE / "data/qwen72/humaneval_opc/splits_0.5/mistral7_test.json",
        "qwen72", _CODING_CONDITIONS,
    ),
    (
        "qwen7", "math", "gsm8k",
        BASE / "data/qwen72/gsm8k/splits_0.5/qwen7_test.json",
        "qwen72", _MATH_CONDITIONS,
    ),
    (
        "qwen7", "coding", "humaneval_opc",
        BASE / "data/qwen72/humaneval_opc/splits_0.5/qwen7_test.json",
        "qwen72", _CODING_CONDITIONS,
    ),
]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_problems(fpath: Path) -> dict:
    """Return {problem_id (int) -> problem_dict} from a splits JSON file."""
    with open(fpath) as f:
        problems = json.load(f)
    return {p["id"]: p for p in problems}


def load_transcript(fpath: Path) -> list:
    with open(fpath) as f:
        return json.load(f)


def transcript_to_pairs(turns: list, teacher_role: str) -> list[dict]:
    """
    Extract (student_question, teacher_response) pairs, one per turn.
    Skips the teacher greeting. Matches any non-teacher preceding message as
    the student question — handles LoRA role names such as mistral7_odqs.
    Returns list of dicts: turn, student_q, teacher_response.
    """
    pairs = []
    for i, turn in enumerate(turns):
        if turn["role"] == teacher_role and turn["turn"] is not None:
            student_turn = next(
                (t for t in reversed(turns[:i]) if t["role"] != teacher_role), None
            )
            if student_turn is None:
                continue
            pairs.append({
                "turn": turn["turn"],
                "student_q": student_turn["content"],
                "teacher_response": turn["content"],
            })
    return pairs


def make_item_id(student: str, task_type: str, condition: str,
                 problem_id: int, turn: int) -> str:
    raw = f"{student}_{task_type}_{condition}_{problem_id}_{turn}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_condition(
    student: str,
    task_type: str,
    condition_name: str,
    transcript_dir: Path,
    problems: dict,
    teacher_role: str,
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Sample n items from transcript_dir, one pair per problem."""
    transcript_files = list(transcript_dir.glob("*.json"))
    rng.shuffle(transcript_files)

    items = []
    seen_problems = set()

    for fpath in transcript_files:
        if len(items) >= n:
            break
        try:
            problem_id = int(fpath.stem)
        except ValueError:
            continue
        if problem_id not in problems or problem_id in seen_problems:
            continue

        try:
            turns = load_transcript(fpath)
        except Exception:
            continue

        pairs = transcript_to_pairs(turns, teacher_role)
        if not pairs:
            continue

        # Use only the teacher replies of the first two turns (0 and 1): the student hasn't
        # solved the problem yet, so the teacher's response is a genuine hint/reveal decision.
        # Later replies are often just confirmation after the student already stated the
        # correct answer — not real leakage.
        early_pairs = [p for p in pairs if p["turn"] <= 1]
        if not early_pairs:
            continue
        chosen = rng.choice(early_pairs)

        problem = problems[problem_id]
        items.append({
            "item_id": make_item_id(student, task_type, condition_name,
                                    problem_id, chosen["turn"]),
            "student": student,
            "task_type": task_type,
            "condition": condition_name,
            "condition_label": CONDITION_LABELS[condition_name],
            "problem_id": str(problem_id),
            "turn": chosen["turn"],
            "problem_prompt": problem["prompt"].strip(),
            "gold_solution": problem["solution"].strip(),
            "student_question": chosen["student_q"],
            "teacher_response": chosen["teacher_response"],
        })
        seen_problems.add(problem_id)

    return items


# ---------------------------------------------------------------------------
# Gold / QC anchor items (task-type-specific so examples match the domain)
# ---------------------------------------------------------------------------

GOLD_ITEMS_CODING = [
    {
        "item_id": "gold_coding_1",
        "condition": "gold", "is_gold": True, "gold_label": 1,
        "student": "GOLD", "task_type": "coding",
        "problem_id": "GOLD", "turn": 1,
        "problem_prompt": "Write a Python function `add(a, b)` that returns the sum of two numbers.",
        "gold_solution": "def add(a, b):\n    return a + b",
        "student_question": "Should I use + or is there a built-in function for this?",
        "teacher_response": (
            "Good thinking! What built-in operations does Python provide for numeric values? "
            "Try experimenting — what happens when you type `3 + 5` in the Python shell?"
        ),
    },
    {
        "item_id": "gold_coding_5",
        "condition": "gold", "is_gold": True, "gold_label": 5,
        "student": "GOLD", "task_type": "coding",
        "problem_id": "GOLD", "turn": 1,
        "problem_prompt": "Write a Python function `add(a, b)` that returns the sum of two numbers.",
        "gold_solution": "def add(a, b):\n    return a + b",
        "student_question": "Can you show me how to write this?",
        "teacher_response": (
            "Sure! Here is the complete solution:\n\n"
            "```python\ndef add(a, b):\n    return a + b\n```\n\n"
            "This defines a function that takes `a` and `b` as arguments "
            "and returns their sum using the `+` operator."
        ),
    },
    {
        "item_id": "gold_coding_3",
        "condition": "gold", "is_gold": True, "gold_label": 3,
        "student": "GOLD", "task_type": "coding",
        "problem_id": "GOLD", "turn": 1,
        "problem_prompt": "Write a Python function `add(a, b)` that returns the sum of two numbers.",
        "gold_solution": "def add(a, b):\n    return a + b",
        "student_question": "What exactly should the return statement say?",
        "teacher_response": (
            "The return statement should be `return a + b`. "
            "You use the `+` operator between the two parameters to compute their sum."
        ),
    },
]

GOLD_ITEMS_MATH = [
    {
        "item_id": "gold_math_1",
        "condition": "gold", "is_gold": True, "gold_label": 1,
        "student": "GOLD", "task_type": "math",
        "problem_id": "GOLD", "turn": 1,
        "problem_prompt": "A store sells apples for $2 each. Tom buys 5 apples. How much does he spend?",
        "gold_solution": "Tom buys 5 apples at $2 each.\n5 x 2 = $10\n#### 10",
        "student_question": "But how do I set up the equation?",
        "teacher_response": (
            "Good! Think about what quantity × price gives you. "
            "Can you write that out using the numbers in the problem?"
        ),
    },
    {
        "item_id": "gold_math_5",
        "condition": "gold", "is_gold": True, "gold_label": 5,
        "student": "GOLD", "task_type": "math",
        "problem_id": "GOLD", "turn": 1,
        "problem_prompt": "A store sells apples for $2 each. Tom buys 5 apples. How much does he spend?",
        "gold_solution": "Tom buys 5 apples at $2 each.\n5 x 2 = $10\n#### 10",
        "student_question": "Can you just tell me the answer?",
        "teacher_response": (
            "Sure -- Tom spends 5 x $2 = $10. "
            "You multiply the number of apples (5) by the price per apple ($2) "
            "to get the total cost of $10."
        ),
    },
    {
        "item_id": "gold_math_3",
        "condition": "gold", "is_gold": True, "gold_label": 3,
        "student": "GOLD", "task_type": "math",
        "problem_id": "GOLD", "turn": 1,
        "problem_prompt": "A store sells apples for $2 each. Tom buys 5 apples. How much does he spend?",
        "gold_solution": "Tom buys 5 apples at $2 each.\n5 x 2 = $10\n#### 10",
        "student_question": "So do I multiply those two numbers?",
        "teacher_response": (
            "Yes, exactly -- multiply 5 by 2 to get the total. "
            "The equation is 5 x 2, which gives you the answer."
        ),
    },
]

GOLD_ITEMS_BY_TASK = {"coding": GOLD_ITEMS_CODING, "math": GOLD_ITEMS_MATH}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

CSV_FIELDNAMES = [
    "item_id", "student", "task_type", "condition", "condition_label",
    "problem_id", "turn",
    "problem_prompt", "gold_solution",
    "student_question", "teacher_response",
]


def write_jsonl(items: list, path: Path):
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def write_csv(items: list, path: Path):
    with open(path, "w", newline="", encoding="ascii", errors="replace") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for item in items:
            row = dict(item)
            row.setdefault("condition_label",
                           CONDITION_LABELS.get(item.get("condition", ""), ""))
            # Sanitize all string fields to ASCII for AMT compatibility
            row = {k: sanitize_for_amt(v) if isinstance(v, str) else v
                   for k, v in row.items()}
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_condition", type=int, default=10,
                        help="Items per condition per (student, task) combination")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="experiments/human_eval/data")
    parser.add_argument("--conditions", type=str, default=None,
                        help="Comma-separated conditions to include, e.g. self_odqs,unscaffolded "
                             "(default: all conditions for each task)")
    parser.add_argument("--students", type=str, default=None,
                        help="Comma-separated students to include, e.g. mistral7,qwen7 "
                             "(default: both)")
    parser.add_argument("--gold", action="store_true",
                        help="Append gold QC items to each AMT batch CSV")
    args = parser.parse_args()
    filter_conditions = set(args.conditions.split(",")) if args.conditions else None
    filter_students = set(args.students.split(",")) if args.students else None

    out_dir = BASE / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    items_by_task = {"coding": [], "math": []}

    for (student, task_type, task_dir, prob_file,
         teacher_role, condition_map) in STUDENT_TASK_MATRIX:

        if filter_students and student not in filter_students:
            continue
        print(f"\n=== {student} / {task_type} ===")
        problems = load_problems(prob_file)
        print(f"  Loaded {len(problems)} problems from {prob_file.name}")

        combo_items = []
        for cond_key, (exp_dir, trans_subdir) in condition_map.items():
            if filter_conditions and cond_key not in filter_conditions:
                continue
            tdir = (BASE / "experiments" / exp_dir / "qwen72"
                    / student / task_dir / trans_subdir)
            if not tdir.exists():
                print(f"  WARNING: missing {tdir}")
                continue
            items = sample_condition(
                student, task_type, cond_key, tdir, problems,
                teacher_role, args.n_per_condition, rng,
            )
            print(f"  {CONDITION_LABELS[cond_key]:20s}: {len(items)} items")
            combo_items.extend(items)

        # Per-combination JSONL for analysis (join target for AMT results)
        tag = f"{student}_{task_type}"
        write_jsonl(combo_items, out_dir / f"leakage_samples_{tag}.jsonl")
        print(f"  → leakage_samples_{tag}.jsonl  ({len(combo_items)} items)")

        items_by_task[task_type].extend(combo_items)

    # AMT upload CSVs — one per task type, gold items optionally appended once
    # Filename includes condition/student suffix when filters are used, to avoid overwriting
    cond_suffix = ("_" + "_".join(sorted(filter_conditions))) if filter_conditions else ""
    stud_suffix = ("_" + "_".join(sorted(filter_students))) if filter_students else ""
    total_hits = 0
    for task_type in ("coding", "math"):
        task_items = items_by_task[task_type]
        gold = GOLD_ITEMS_BY_TASK[task_type] if args.gold else []
        rng.shuffle(task_items)
        fname = f"amt_{task_type}{cond_suffix}{stud_suffix}.csv"
        out_path = out_dir / fname
        write_csv(task_items + gold, out_path)
        n_hits = len(task_items) + len(gold)
        total_hits += n_hits
        gold_note = f" + {len(gold)} gold" if gold else ""
        print(f"\n→ {fname}  ({len(task_items)} task{gold_note} = {n_hits} HITs)")

    print(f"\nGrand total HITs : {total_hits}")
    print(f"Output dir       : {out_dir}")


if __name__ == "__main__":
    main()
