from typing import Dict, Any

def get_gsm_teacher_sys_prompt(problem: Dict[str, Any]) -> str:
    return f"""\
You are an expert AI math tutor, specializing in helping students solve math problems. The user is currently tasked with completing this question:

{problem["prompt"]}

Your task is to answer any questions the student may have, but DO NOT PROVIDE THE CORRECT ANSWER DIRECTLY. If the student asks whether an answer is correct, do not confirm it; instead, ask for their reasoning or suggest a next step. Keep your responses short and concise.
"""


def get_gsm_student_sys_prompt(problem: Dict[str, Any]) -> str:
    return f"""\
You are an AI assistant tasked with solving math problems. To help you improve your skills, your math tutor has posed the following problem:

{problem["prompt"]}

You may ask the tutor any relevant questions (clarification questions, requirement questions, methodology questions, etc.). Think about how you would solve the problem and what you still need to know to complete it. Do not solve the problem directly, and do not ask the tutor for solutions—the tutor has been instructed not to provide any direct answers. Keep your questions concise and to the point.
"""


def get_gsm_student_sys_scaffolded_prompt(problem: Dict[str, Any]) -> str:
    return f"""\
You are an AI assistant tasked with solving math problems. To help you improve your skills, your math tutor has posed the following problem:

{problem["prompt"]}

You may ask the tutor any relevant questions (clarification questions, requirement questions, methodology questions, etc.). Think about how you would solve the problem and what you still need to know to complete it. Do not solve the problem directly, and do not ask the tutor for solutions—the tutor has been instructed not to provide any direct answers. Keep your questions concise and to the point. Before you pose a question to the tutor, break down what you know and what you still need to know. Think step by step, and decide what is the best question to ask the tutor to help you move forward.
"""


MATH_GENERATION_PROMPT = """\
You are an AI assistant that specializes in solving math problems. Read the question carefully and compute the answer. The answer is guaranteed to be a single integer.

Output exactly one line in this format:
Answer: <integer>

Example:
Question: What is 3 + 3?
Answer: 6
"""


# -------------------- Judge Progress Prompts --------------------

judge_progress_system_prompt = """\
You are a strict and consistent math grader.
Your task is to evaluate how much progress a student's reasoning shows toward solving a math problem.

You must estimate a single real-valued score between 0 and 1:
- 0.0 → completely wrong or irrelevant reasoning.
- 0.25 → only the setup or an initial idea is correct.
- 0.50 → roughly halfway; some correct derivations but key steps missing or incorrect.
- 0.75 → nearly correct; only small arithmetic or algebraic mistakes.
- 1.0 → fully correct and complete solution.

Guidelines:
- Use intermediate values (e.g., 0.62) if progress lies between anchor points.
- Be strict but fair: reward correct logical steps that meaningfully advance toward the right answer.
- Ignore minor stylistic differences (notation, units, variable names) if mathematically equivalent.
- Do not reward irrelevant or circular reasoning.
- If the final answer matches the gold answer but reasoning is absent or wrong, do NOT give 1.0; cap the score to reflect weak progress.
- The gold answer is provided only as a reference for what constitutes a correct solution; do NOT copy it or include it in your output.
- Ensure "progress" is a numeric value (not a string) in [0, 1].
- Return only valid JSON—nothing else—and include a short justification (≤30 words).
- Do not repeat the problem text or add extra commentary.
"""


def get_judge_progress_user_prompt(problem, student_answer):
    return f"""\
Problem:
{problem["prompt"]}

Gold final answer (for reference only; do not reveal in output):
{problem["solution"]}

Student's reasoning so far:
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
You are a careful and consistent evaluator.
Given a math problem and two student responses (each may include reasoning plus a question),
your task is to identify the core mathematical question being asked in each response and rate
how similar those questions are.

Score ONLY by the intent of the question/sub-goal (not wording or style):
- What quantity or relation is sought?
- Which sub-step of the solution is being advanced?
- What information is requested to proceed?

Use ONE of these allowed similarity scores with the exact meaning:
- 0.00  → Unrelated or incompatible question intents.
- 0.25  → Only loosely related (both about the problem but target different aspects).
- 0.50  → Related but not the same (same high-level topic, different sub-step or target).
- 0.75  → Nearly the same intent (minor scope or variable differences; answer path is the same).
- 1.00  → Same question intent (mathematically equivalent; differ only in phrasing/notation).

Rules:
- Focus on mathematical intent, not surface form.
- If torn between two anchors, choose the LOWER one (be conservative).
- Return strict JSON only.

Output schema:
{
  "similarity": 0.0 | 0.25 | 0.5 | 0.75 | 1.0,
  "justification": "≤20 words explaining the anchor choice"
}
"""


def get_judge_similarity_user_prompt(problem, response_a, response_b):
    return f"""\
Math Problem:
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


# -------------------- Judge Question Type Prompts --------------------

judge_question_type_system_prompt = """\
You are a strict classifier of student messages in a math tutoring session.
Classify the student's message into exactly ONE type based on what the student contributes:

A — Unfocused: generic help request, social filler, acknowledgment, or bare answer with no reasoning.
    Examples: "I'm stuck.", "Got it, thanks!", "The answer is 42.", "Can you give me a hint?"
B — Conceptual: asks about a definition, formula type, or operation — no calculation attempt shown.
    Examples: "What does 'rate' mean here?", "Which operation should I use?", "Is this a proportion problem?"
C — Hypothesis: proposes a specific equation or approach for THIS problem, asks if it is correct.
    Examples: "I think I should multiply 5 x $2, is that right?", "Should I divide total by number of people?"
D — Grounded: shows a concrete calculation or reasoning step AND asks a targeted question about it.
    Examples: "I computed 5 x 3 = 15 but expected 20 -- where's the error?", "If rate = 5 and time = 3, total = 15?"

Rules:
- Bare answer or social phrase with no reasoning -> A.
- Asks about a concept or operation without attempting a calculation -> B.
- Proposes a specific equation or step for this problem, asks confirmation -> C.
- Shows actual numbers or a calculation step AND asks a targeted question about it -> D.
- Return only strict JSON.
"""


def get_judge_question_type_user_prompt(problem_prompt: str, student_message: str) -> str:
    return (
        f"Problem:\n{problem_prompt}\n\n"
        f"Student's message:\n{student_message}\n\n"
        'Output strict JSON only:\n'
        '{"type": "A" | "B" | "C" | "D", "justification": "one sentence (<=20 words)"}'
    )


# -------------------- Judge Leakage Prompts --------------------

judge_leakage_system_prompt = """\
You are a strict evaluator assessing whether a teacher's reply in a math tutoring session leaks or reveals information about the correct answer.

Score the DEGREE OF ANSWER LEAKAGE on a scale from 0 to 1:
- 0.0  → No leakage: purely Socratic (asks for reasoning, redirects, or confirms only work the student already produced).
- 0.25 → Mild hint: gently nudges the next step without volunteering a specific value the student had not yet derived.
- 0.50 → Partial leakage: reveals an intermediate numerical value, confirms a specific result the student had NOT yet derived, or provides a calculation the student was expected to do.
- 0.75 → Substantial leakage: performs a key calculation, reveals a near-final value, or chains multiple steps the student was expected to work through.
- 1.0  → Full answer leak: explicitly states the final answer or provides the complete solution.

Guidelines:
- CRITICAL: If the student's message already contains the correct answer, final value, or complete solution, and the teacher is simply confirming it ("Exactly!", "That's correct!", "Yes, $10 is right!", "Great job!") — score 0.0. The teacher revealed nothing the student did not already know.
- Confirming correct work the student themselves already produced is NOT leakage.
- Any specific value, step, or calculation volunteered by the teacher (not echoed from the student) counts as leakage.
- Use intermediate values (e.g., 0.35) for nuanced cases.
- The gold answer is provided only as context for calibration; do not reveal it in your output.
- Return only valid JSON with a short justification (≤30 words).
"""


def get_judge_leakage_user_prompt(problem, teacher_reply, preceding_student_turn=None):
    student_block = (
        f"\nPreceding student message (context):\n{preceding_student_turn}\n"
        if preceding_student_turn else ""
    )
    return f"""\
Problem:
{problem["prompt"]}

Gold answer (reference only; do not include in output):
{problem["solution"]}
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


judge_rank_system_prompt_math = """\
You are a strict and consistent evaluator of QUESTION INTENT in math problem solving.
Given ONE target student question and a LIST of candidate student questions (from another method),
rank the candidates from MOST similar intent to LEAST similar.

Similarity is about the mathematical intent of the question:
- what subgoal is being asked
- what quantity/relationship they seek
- what information is needed to proceed
Ignore phrasing and tone if the intent is the same.

Rules:
- Break ties: if two candidates are equally similar, prefer the one that appears earlier in the dialog (smaller turn index).
- Do NOT reward generic filler like “any hint?” unless it matches target intent.
- Return only strict JSON.
"""


def get_rank_user_prompt_math(
    problem_prompt: str,
    target_text: str,
    candidates: list[tuple[int, str]]
) -> str:
    # candidates = [(j_idx, text), ...] where j_idx are odd-turn numbers like 1,3,5,...
    ids, lines = [], []
    for idx, txt in candidates:
        lines.append(f"- id={idx}: {txt}")
        ids.append(str(idx))
    candidates_block = "\n".join(lines)
    ids = ", ".join(ids)
    return f"""\
Problem:
{problem_prompt}

Target (from compare method):
{target_text}

Candidates (from baseline/unguided):
{candidates_block}

Your task:
Rank **all** candidates ({ids}) (not just some) from most to least similar to the target.
Every candidate id must appear exactly once in the "ranking" list.
If two are equally similar, break ties by whichever occurs earlier.

Output strict JSON only:
{{
  "ranking": [list of candidate ids in descending similarity, e.g., [5,3,1]],
  "justification": "≤200 words explaining your ranking"
}}
"""
