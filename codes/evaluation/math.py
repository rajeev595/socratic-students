"""
Math evaluation utilities.

This module provides:

1) `generate_math_completions`:
   - Builds a system prompt from `MATH_GENERATION_PROMPT`.
   - If a prior conversation is provided:
       * and `is_ood=True`: explicitly states it's a *similar* question with
         a *different* numeric answer.
       * and `is_ood=False`: says it's the *same* question context.
   - Optionally appends a GT hint when `show_gt=True`.
   - Calls `chat_complete` (OpenAI-compatible) with `n=num_samples`.
   - Post-processes each response using `extract_math_response` to pull the
     final numeric answer (or return the stripped text if none is found).

2) `evaluate_math_completion`:
   - Stage A: strict numeric equality against the ground-truth using
     regex extraction (no try/except).
   - Stage B: if not equal (or non-numeric), queries a lightweight
     LLM judge (`gpt-4o-mini`) to answer “YES/NO” for semantic equivalence.

Notes
-----
- We keep the `system` message; `chat_complete` already handles models
  that don’t support system role (e.g., Gemma) by folding it into user.
- No `try/except` is used for numeric parsing; we rely on regex gates.
"""

import os
import re
from typing import List, Dict, Any, Tuple
from openai import OpenAI

from codes.utils.llm.chat import chat_complete
from codes.data.utils import extract_math_response
from codes.prompts.math import MATH_GENERATION_PROMPT


# ---------------------------
# Generation
# ---------------------------

def generate_math_completions(
    previous_conversation_str: str,
    client: OpenAI,
    model_name: str,
    temperature: float,
    problem: Dict[str, Any],
    num_samples: int,
    max_tokens: int = 512,
    show_gt: bool = False,
    is_ood: bool = False,
) -> Tuple[List[str], int, int]:
    """
    Generate `num_samples` math answers and extract concise numeric responses.

    Behavior:
      - If `previous_conversation_str` is nonempty:
          * `is_ood=True`  -> mention it's a similar but different question (answers differ).
          * `is_ood=False` -> mention it's the same question context.
      - Else use the base math prompt.
      - When `show_gt=True`, append a brief hint with the canonical solution.

    Args:
        previous_conversation_str: Prior dialogue transcript (may be empty).
        client: OpenAI-compatible client.
        model_name: Target model id.
        problem: Dict containing at least {"prompt", "solution"}.
        num_samples: Number of completions to request in one call.
        temperature: Sampling temperature.
        show_gt: If True, append a GT hint string.
        is_ood: If True, craft the “similar but different” guidance.

    Returns:
        (answers, prompt_tokens, completion_tokens)
        where `answers` contains post-processed strings (typically numbers).
    """
    sys_prompt = MATH_GENERATION_PROMPT
    if previous_conversation_str and is_ood:
        sys_prompt += (
            "\n\nIn addition, you have previously conversed with your math tutor about a very similar question. "
            "You may reference the provided dialogue to aid solving the current question. Keep in mind the answers "
            "will be different between the two!"
        )
        previous_conversation_str = "**PREVIOUS CONVERSATION**\n\n" + previous_conversation_str + "\n\n**NEW QUESTION**"
    elif previous_conversation_str and not is_ood:
        sys_prompt += (
            "\n\nIn addition, you have previously conversed with your math tutor about this question. "
            "You reference the provided conversation to aid you in solving the question."
        )

    if show_gt:
        sys_prompt += f" The tutor secretly showed a hint: {problem['solution']}."

    user_content = (
        f"{previous_conversation_str}\n\n{problem['prompt']}"
        if previous_conversation_str else problem["prompt"]
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
    ]

    texts, p, c = chat_complete(
        client=client,
        model_name=model_name,
        messages=messages,
        n=num_samples,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    # Pull final numeric answer if present; otherwise return a clean string.
    outs = [extract_math_response((t or "").strip()) for t in texts]
    return outs, p, c


# ---------------------------
# Evaluation
# ---------------------------

_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _extract_float(s: str) -> Tuple[bool, float]:
    """
    Extract the *last* floating-point number from `s` and convert to float.

    Returns:
        (True, value) if a numeric token is found and convertible,
        (False, 0.0) otherwise.

    No try/except is needed because we only call float() on a regex-captured
    token that matches a valid float pattern.
    """
    matches = _NUMBER_RE.findall(s or "")
    if not matches:
        return False, 0.0
    return True, float(matches[-1])


def evaluate_math_completion(
    model_response: str, 
    problem: Dict[str, Any], 
    client: OpenAI,
    model_name: str,
    temperature: float
) -> Tuple[str, bool]:
    """
    Evaluate a math answer by strict numeric equality first, then by an LLM judge.

    Stage A (numeric):
      - Extract last float from both `model_response` and `problem['solution']`.
      - If both numeric -> exact float equality check.

    Stage B (LLM judge):
      - If Stage A fails or is non-numeric, ask a small LLM (“gpt-4o-mini”)
        to answer YES/NO for equivalence.

    Args:
        model_response: The model's raw text answer.
        problem: Dict with keys {"solution", ...} where solution may be numeric text.

    Returns:
        (original_response, passed_bool)
    """
    # Strict numeric check
    have_pred, pred_val = _extract_float(model_response)
    have_gt, gt_val = _extract_float(str(problem.get("solution", "")))
    if have_pred and have_gt and (pred_val == gt_val):
        return model_response, True

    # LLM judge fallback
    prompt = (
        "You are a teaching assistant for determining whether a student's answer is equivalent to the correct answer.\n\n"
        f"Student answer:\n{model_response}\n\n"
        f"Correct answer:\n{problem.get('solution', '')}\n\n"
        "Respond with a single YES or NO (your reply must start with YES or NO)."
    )
    messages = [{"role": "user", "content": prompt}]
    reply, _, _ = chat_complete(
        client=client,
        model_name=model_name,
        messages=messages,
        n=1,
        temperature=temperature,
        max_tokens=16
    )

    return model_response, reply[0].startswith("yes")
