"""
Lightweight data utilities for code/math prompt processing.

Functions
---------
- extract_code_blocks(text): Return fenced code blocks, preferring ```python ...```.
- remove_docstring(text): Strip a single leading triple-quoted docstring from code.
- remove_think_tags(text): Drop everything up to and including the first </think>.
- msgs_to_string(texts): Convert a list of raw turns into a TUTOR/YOU transcript.
- extract_math_response(text): Extract the final numeric value after 'Answer:'.
- format_messages(texts, starting_with_user): Build OpenAI-style chat messages
  alternating user/assistant roles starting from the specified role.
"""

from __future__ import annotations

import re
from typing import List, Dict


# ---------------------------------------------------------------------------
# Code text helpers
# ---------------------------------------------------------------------------

def extract_code_blocks(text: str) -> List[str]:
    """
    Extract fenced code blocks from `text`.

    Preference order:
      1) Blocks explicitly tagged as Python: ```python ... ```
      2) Any fenced block: ``` ... ```

    If no fenced blocks are found, return a single-element list containing
    the original `text`.

    Returns:
        A list of block strings, stripped of surrounding whitespace.
    """
    py_blocks = re.findall(r"```python\s+(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if py_blocks:
        return [b.strip() for b in py_blocks]

    any_blocks = re.findall(r"```\s*(.*?)```", text, re.DOTALL)
    if any_blocks:
        return [b.strip() for b in any_blocks]

    return [text]


def remove_docstring(text: str) -> str:
    """
    Remove a single leading triple-quoted docstring (module- or function-level),
    preserving the indentation of the following code.

    Only the *first* top-of-file or immediately-in-block docstring is removed.

    Examples
    --------
    >>> remove_docstring('\"\"\"desc\"\"\"\\nprint(1)')
    'print(1)'
    >>> remove_docstring('def f():\\n    \"\"\"x\"\"\"\\n    return 1')
    'def f():\\n    return 1'
    """
    # Match a triple-quoted string that starts at the beginning of a line
    # (allowing indentation), consume at most ONE trailing newline, and
    # crucially DO NOT consume the indentation of the next line.
    pattern = re.compile(
        r'^[ \t]*(?P<q>"""|\'\'\')[\s\S]*?(?P=q)(?:\r?\n|$)',
        flags=re.MULTILINE
    )
    return pattern.sub('', text, count=1)


def remove_think_tags(text: str) -> str:
    """
    Remove hidden reasoning preamble up to the first closing </think> tag.

    Everything from the start of the string through the first '</think>' is
    discarded. If the tag does not occur, the original text is returned.

    Example:
        '</think> final output' -> 'final output'
    """
    tag = "</think>"
    if tag in text:
        return text.split(tag, 1)[1].lstrip()
    return text


# ---------------------------------------------------------------------------
# Conversation formatting
# ---------------------------------------------------------------------------

def msgs_to_string(texts: List[str]) -> str:
    """
    Convert a list of raw turn texts into a readable alternating transcript.

    The first line is labeled 'TUTOR', the second 'YOU', and so on.

    Example:
        ['Hello', 'Hi'] ->
        'TUTOR: Hello\\n\\nYOU: Hi'

    Returns:
        A single string with blank lines between turns.
    """
    current_student_turn = False
    out: List[str] = []
    for t in texts:
        role = "YOU" if current_student_turn else "TUTOR"
        out.append(f"{role}: {t}\n\n")
        current_student_turn = not current_student_turn
    return "".join(out).strip()


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def extract_math_response(text: str) -> str:
    """
    Extract the final numeric value following 'Answer:' in `text`.

    If multiple 'Answer:' occurrences exist, the last numeric match is returned.
    If no numeric match is found, returns the stripped original `text`.

    Examples:
        '... Answer: 12'            -> '12'
        'Answer: -3.5'              -> '-3.5'
        'No explicit answer found'  -> 'No explicit answer found'
    """
    matches = re.findall(r'Answer:\s*([-+]?\d*\.?\d+)', text)
    return matches[-1] if matches else text.strip()


# ---------------------------------------------------------------------------
# Chat message construction
# ---------------------------------------------------------------------------

def format_messages(texts: List[str], starting_with_user: bool) -> List[Dict[str, str]]:
    """
    Convert a list of turn texts into OpenAI-style chat messages.

    Roles alternate between 'user' and 'assistant', starting with 'user' if
    `starting_with_user` is True, otherwise starting with 'assistant'.

    Args:
        texts: List of plain string turns.
        starting_with_user: Whether the first role should be 'user'.

    Returns:
        A list of dicts like [{'role': 'user', 'content': '...'}, ...].
    """
    msgs: List[Dict[str, str]] = []
    role = "user" if starting_with_user else "assistant"
    for text in texts:
        msgs.append({"role": role, "content": text})
        role = "assistant" if role == "user" else "user"
    return msgs
