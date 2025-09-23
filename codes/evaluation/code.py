"""
Code evaluation utilities.

This module provides:
- A prompt template and a generator to obtain multiple code completions
  from an OpenAI-compatible client, with post-processing that mirrors
  your original evaluation logic (strip <think>, pick last fenced block,
  remove leading docstring, and trim a duplicate `def {entry_point}...`
  header if the prompt already shows the signature).
- An executor that evaluates a completion against the provided tests by
  writing a temporary Python file and running it.

Changes vs. previous version
----------------------------
- `evaluate_code_completion` now accepts `timeout` and `raise_on_timeout`.
  By default, the timeout is read from env var EVAL_TIMEOUT_SEC (default 12),
  and timeouts are treated as failures (return passed=False) rather than
  raising (configurable with EVAL_RAISE_ON_TIMEOUT=1). This prevents pool
  crashes in callers that don't wrap the evaluation.

Notes
-----
* `generate_code_completions` is model-agnostic: it uses the passed
  `client` (OpenAI-compatible) and forwards `n=num_samples` in a single call.
"""

import os
import sys
import tempfile
import subprocess
from typing import List, Dict, Any, Tuple, Optional

from codes.data.utils import (
    extract_code_blocks,
    remove_docstring,
    remove_think_tags,
)
from codes.prompts.code import CODE_GENERATION_PROMPT
from codes.llm_utils import chat_complete


# ---------------------------
# Generation
# ---------------------------

def generate_code_completions(
    previous_conversation_str: str,
    client,
    model_name,
    temperature: float,
    problem_dict: Dict[str, Any],
    num_samples: int,
    max_tokens: int = 2048,
    show_gt: bool = False
) -> Tuple[List[str], int, int]:
    """
    Generate `num_samples` code completions and post-process to extract clean code.

    Behavior matches the original:
      - If prior conversation is present, mention it in the system prompt.
      - If `show_gt=True`, inject the canonical solution text as a hint.
      - Post-processing:
          * Remove <think>... tags if present.
          * Take the last fenced code block if any; else whole text.
          * Remove a leading triple-quoted docstring.
          * If the prompt already shows `def {entry_point}...`, strip the first
            line of the returned code when it also starts with `def ` to avoid
            duplicated signatures.

    Returns:
        (completions, prompt_tokens, completion_tokens)
    """
    # 1) Build system prompt
    sys_prompt = CODE_GENERATION_PROMPT
    if previous_conversation_str:
        sys_prompt += (
            "\n\nIn addition, you have previously conversed with your coding tutor about this question. "
            "You reference the provided conversation to aid you in solving the question."
        )
    if show_gt:
        sys_prompt += (
            " The tutor gave secretly showed me this snippet as a hint: "
            f"{problem_dict['solution']} I wonder what I can do with this?"
        )

    # 2) Messages
    user_content = (
        f"{previous_conversation_str}\n\n{problem_dict['prompt']}"
        if previous_conversation_str else problem_dict["prompt"]
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
    ]

    # 3) Single API call with n = num_samples
    raw_texts, prompt_tks, completion_tks = chat_complete(
        client=client,
        model_name=model_name,
        messages=messages,
        n=num_samples,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # 4) Post-process each completion into a code string
    outputs: List[str] = []
    entry_point = problem_dict.get("entry_point", "")
    prompt_has_signature = bool(entry_point) and (f"def {entry_point}" in problem_dict.get("prompt", ""))

    for t in raw_texts:
        t = remove_think_tags(t)
        blocks = extract_code_blocks(t)
        code = remove_docstring(blocks[-1]) if blocks else ""
        if code.startswith("def ") and prompt_has_signature and "\n" in code:
            # Avoid duplicate `def {entry_point}...` when signature is already shown in the prompt
            code = code.split("\n", 1)[1]
        outputs.append(code)

    return outputs, prompt_tks, completion_tks


# ---------------------------
# Execution-based evaluation
# ---------------------------

def evaluate_code_completion(
    code_block: str,
    problem: Dict[str, Any],
    sanity_check: bool = False,
    timeout: Optional[int] = None,
    raise_on_timeout: Optional[bool] = None,
) -> Tuple[str, bool]:
    """
    Evaluate a single completion by executing the provided tests.

    This writes a temporary Python file comprised of:
        - canonical solution (if sanity_check=True) OR
          (problem prompt + model completion)
        - a blank line
        - the test code in problem["test"]

    Then runs: `python <temp_file>.py` with a timeout (configurable).

    Args:
        code_block: The model's code (already cleaned by the generator).
        problem: Dict expected to contain keys "prompt", "solution", "test".
        sanity_check: If True, ignore `code_block` and run the canonical solution.
        timeout: Optional override (seconds). If None, uses env EVAL_TIMEOUT_SEC (default 12).
        raise_on_timeout: If True, re-raises TimeoutExpired; if None, uses env
            EVAL_RAISE_ON_TIMEOUT (default "0" → False).

    Returns:
        (original_code_block, passed)
        where `passed` is True iff the process exit code is 0.

    Behavior on timeout:
        - Default: return (code_block, False) so callers don't crash.
        - To restore old behavior (raise), set raise_on_timeout=True or EVAL_RAISE_ON_TIMEOUT=1.
    """
    # Resolve runtime options
    timeout_sec = int(timeout) if timeout is not None else int(os.getenv("EVAL_TIMEOUT_SEC", "12"))
    do_raise = (
        bool(raise_on_timeout)
        if raise_on_timeout is not None
        else (os.getenv("EVAL_RAISE_ON_TIMEOUT", "0") == "1")
    )

    prompt = problem["prompt"]
    if not prompt.endswith("\n"):
        prompt += "\n"

    full_code = (problem["solution"] if sanity_check else prompt + code_block) + "\n\n" + problem["test"]

    # Create temp file
    f = tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, 
        # dir="/playpen-hdd/ambati/temp/"
    )
    temp_file_path = f.name
    f.write(full_code)
    f.close()

    try:
        # Run with a timeout; returncode==0 -> pass
        result = subprocess.run(
            [sys.executable, temp_file_path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        passed = (result.returncode == 0)
        return code_block, passed

    except subprocess.TimeoutExpired:
        if do_raise:
            raise
        # Treat as fail without raising (default)
        return code_block, False

    finally:
        # Always clean up the temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError:
                # Best-effort cleanup; ignore if another process already removed it
                pass
