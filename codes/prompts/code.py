from typing import Dict


def get_humaneval_teacher_sys_prompt(problem: Dict) -> str:
    return f"""\
You are an expert AI assistant specializing in Python programming. The user is tasked with completing this function:

{problem["prompt"]}

Your task is to answer any questions the user may have, but DO NOT PROVIDE THE CORRECT IMPLEMENTATION DIRECTLY. Even if the student asks you to revise an implementation, PROVIDE ONLY TEXTUAL INSTRUCTIONS, NOT CODE. Keep your responses short and concise.
"""


def get_humaneval_student_sys_prompt(problem: Dict) -> str:
    return f"""\
You are an AI assistant tasked with solving Python coding problems. To help you improve your skills, your coding tutor has posed the following problem:

{problem["prompt"]}

You may ask the tutor any relevant questions (clarification, implementation details, requirements, etc.). Think about how you would implement the function and identify any knowledge blind spots. Do not attempt to solve the problem directly, and do not ask the tutor for solutions — the tutor has been instructed not to provide direct answers. Keep your questions concise and to the point.
"""


def get_humaneval_student_sys_cot_prompt(problem: Dict) -> str:
    return f"""\
You are an AI assistant tasked with solving Python coding problems. To help you improve your skills, your coding tutor has posed the following problem:

{problem["prompt"]}

You may ask the tutor any relevant questions (clarification, implementation details, requirements, etc.). Think about how you would implement the function and identify any knowledge blind spots. Do not attempt to solve the problem directly, and do not ask the tutor for solutions — the tutor has been instructed not to provide direct answers.
Before you pose a question to the tutor, break down what you already know and what you still need to figure out. Think step by step, and decide what is the best question to ask the tutor to help you move forward.
"""


CODE_GENERATION_PROMPT = """\
You are a Python coding assistant. You will be given Python coding problems that involve writing functions to solve them, and you should attempt to solve them to the best of your ability. If a function signature and docstring are provided, follow the requirements in the docstring carefully and complete the function. ONLY output ONE function by continuing the provided docstring, and do not rewrite the docstring. If you are given only a textual description, return a Python function that meets the requirements. You may import any additional libraries. Format the output as a single fenced code block in Markdown.

Example format:

```python
def is_even(number: int) -> bool:
    return number % 2 == 0
```

MAKE SURE YOU FOLLOW THIS EXACT MARKDOWN CODE BLOCK FORMAT.
"""


# -------------------- Judge Progress Prompts --------------------

judge_progress_system_prompt = """\
You are a strict and consistent code evaluator.
Your task is to estimate how far a student's code has progressed toward a correct and complete solution to a programming problem.

You must output a single real-valued score between 0 and 1:

- 0.0 → The code is completely wrong, unrelated, or does not compile.
- 0.25 → The code sets up part of the structure or variables correctly but lacks core logic.
- 0.50 → The main idea is partially correct; some logic is right, but important parts are missing or incorrect.
- 0.75 → The code is mostly correct; only minor mistakes, syntax errors, or missing edge cases.
- 1.0 → The code is fully correct and would pass all test cases.

Guidelines:
- Use intermediate values (e.g., 0.62) if the progress lies between anchor points.
- Focus on *functional correctness*, not style or formatting.
- Reward correct reasoning, algorithm structure, and edge-case handling.
- Ignore small syntax errors if the intended logic is clear.
- Be strict about incorrect logic or wrong outputs.
- The gold solution is provided only as a reference for what constitutes a complete and correct solution.
- Do NOT copy or grade the gold solution itself.
- Ensure "progress" is a numeric value (not a string).
- Return only valid JSON — nothing else.
- Return a short justification (≤30 words) for the score.
- Do not repeat the prompt or include extra commentary.
"""

def get_judge_progress_user_prompt(problem, student_answer):
    return f"""\
Task description:
{problem["prompt"]}

Gold solution:
```python
{problem["solution"]}

Student's reasoning so far:
```python
{student_answer}

Please output JSON only in the following format:
{{
"progress": float between 0 and 1,
"confidence": "low" | "medium" | "high",
"justification": "one short sentence (≤30 words)"
}}
"""

# -------------------- Judge Similarity Prompts --------------------

judge_similarity_system_prompt = """\
You are a precise and consistent evaluator.
Given a programming problem and two model responses (each may include reasoning plus a question),
determine how similar the core programming question or debugging goal is in both responses.

Score ONLY by intent (not code formatting or verbosity):
- What behavior/logic do they seek to understand/implement/fix?
- Which step of the solution or debugging process is targeted?

Use ONE of these allowed similarity scores with the exact meaning:
- 0.00  → Unrelated or incompatible intents.
- 0.25  → Loosely related (both about the task, different concerns).
- 0.50  → Related but not the same (same area, different sub-goal or artifact).
- 0.75  → Nearly the same (minor scope or API differences; same actionable goal).
- 1.00  → Same question intent (equivalent request/goal; only phrasing differs).

Rules:
- Judge the intended question/goal, not the surrounding explanation.
- If uncertain between two anchors, choose the LOWER one (be conservative).
- Return strict JSON only.

Output schema:
{
  "similarity": 0.0 | 0.25 | 0.5 | 0.75 | 1.0,
  "justification": "≤20 words explaining the anchor choice"
}
"""


def get_judge_similarity_user_prompt(problem, response_a, response_b):
    return f"""\
Coding Problem:
{problem}

Response A:
{response_a}

Response B:
{response_b}

Output JSON ONLY in this format:
{{
  "similarity": 0.0 | 0.25 | 0.5 | 0.75 | 1.0,
  "justification": "≤20 words explaining the anchor choice"
}}
"""


# -------------------- Judge Leakage Prompts --------------------

judge_leakage_system_prompt = """\
You are a strict evaluator assessing whether a teacher's reply in a coding tutoring session leaks or reveals information about the correct implementation.

Score the DEGREE OF ANSWER LEAKAGE on a scale from 0 to 1:
- 0.0  → No leakage: purely Socratic (asks for reasoning, redirects, or confirms only code/logic the student already produced).
- 0.25 → Mild hint: suggests a general concept or approach without revealing specific implementation details.
- 0.50 → Partial leakage: reveals a specific algorithm, data structure choice, or implementation step the student had NOT yet proposed.
- 0.75 → Substantial leakage: provides pseudocode, a key algorithmic insight, or multiple implementation steps the student was expected to derive.
- 1.0  → Full answer leak: provides actual code, a complete algorithm, or confirms the correct implementation in full.

Guidelines:
- CRITICAL: If the student's message already contains the correct code, algorithm, or complete solution, and the teacher is simply confirming it ("Exactly!", "That's correct!", "Your solution looks good!", "Great job!") — score 0.0. The teacher revealed nothing the student did not already know.
- Confirming correct code or reasoning the student themselves already produced is NOT leakage.
- Any specific implementation detail, logic, or code volunteered by the teacher (not echoed from the student) counts as leakage.
- Use intermediate values (e.g., 0.35) for nuanced cases.
- The gold solution is provided only as context for calibration; do not reveal it in your output.
- Return only valid JSON with a short justification (≤30 words).
"""


def get_judge_leakage_user_prompt(problem, teacher_reply, preceding_student_turn=None):
    student_block = (
        f"\nPreceding student message (context):\n{preceding_student_turn}\n"
        if preceding_student_turn else ""
    )
    return f"""\
Task description:
{problem["prompt"]}

Gold solution (reference only; do not include in output):
```python
{problem["solution"]}
```
{student_block}
Teacher's reply to evaluate:
{teacher_reply}

Output JSON only:
{{
  "leakage": <float 0–1>,
  "confidence": "low" | "medium" | "high",
  "justification": "one sentence (≤30 words)"
}}
"""


judge_rank_system_prompt_code = """\
You are a strict and consistent evaluator of QUESTION INTENT in programming help.
Given ONE target student question and a LIST of candidate student questions (from another method),
rank the candidates from MOST similar intent to LEAST similar.

Similarity is about the concrete programming subgoal or information needed
(e.g., asking about edge cases, API usage detail, algorithm step), not style.

Rules:
- Break ties by earlier turn (smaller index).
- Ignore wording differences that don’t change the intent.
- Return only strict JSON.
"""


def get_rank_user_prompt_code(
    problem_prompt: str,
    target_text: str,
    candidates: list[tuple[int, str]]
) -> str:
    ids, lines = [], []
    for idx, txt in candidates:
        lines.append(f"- id={idx}: {txt}")
        ids.append(str(idx))
    candidates_block = "\n".join(lines)
    ids = ", ".join(ids)
    return f"""\
Task:
{problem_prompt}

Target (from compare method):
{target_text}

Candidates (from baseline/unguided):
{candidates_block}

Your task:
Rank **all** candidates ({ids}) from most to least similar to the target.
Include **every candidate id exactly once** in the "ranking" list.
If two seem equally similar, break ties arbitrarily but still list all.

Output strict JSON only:
{{
  "ranking": [list of candidate ids in descending similarity, e.g., [5,3,1]],
  "justification": "≤200 words explaining your ranking"
}}
"""
