from typing import Any, Dict


from src.prompts.code import (
    get_humaneval_student_sys_prompt,
    get_humaneval_student_sys_scaffolded_prompt,
    get_humaneval_teacher_sys_prompt,
)
from src.prompts.math import (
    get_gsm_student_sys_prompt,
    get_gsm_student_sys_scaffolded_prompt,
    get_gsm_teacher_sys_prompt,
)


def _select_sys_prompt(dataset: str, use_scaffold: bool, is_student_turn: bool, problem: Dict[str, Any]) -> str:
    """Choose the correct system prompt function and embed the problem text (already handled in your prompt fns)."""
    if dataset == "code":
        if is_student_turn:
            return (get_humaneval_student_sys_scaffolded_prompt if use_scaffold else get_humaneval_student_sys_prompt)(problem)
        else:
            return get_humaneval_teacher_sys_prompt(problem)
    else:
        if is_student_turn:
            return (get_gsm_student_sys_scaffolded_prompt if use_scaffold else get_gsm_student_sys_prompt)(problem)
        else:
            return get_gsm_teacher_sys_prompt(problem)


def _problem_greeting(dataset: str) -> str:
    """First message content (index 0 of the text timeline)."""
    return (
        "Hi there! I'm your coding tutor. How can I help you today?"
        if dataset == "code"
        else "Hi there! I'm your math tutor. How can I help you today?"
    )
