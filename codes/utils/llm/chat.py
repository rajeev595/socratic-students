"""
Lightweight LLM utilities.

- `chat_complete`: thin wrapper over Chat Completions that:
    * passes through temperature, n, max_tokens,
    * returns (list of texts, prompt_tokens, completion_tokens),
    * applies a Gemma-specific shim: if the first message is a system prompt,
      prepend it to the first user message and drop the system role (many Gemma backends
      reject the "system" role).
"""

from __future__ import annotations
from typing import List, Dict, Tuple

from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tiktoken import encoding_for_model

from codes.utils.llm.server import get_model_name


# ---------------------------------------------------------------------------
# Chat completion
# ---------------------------------------------------------------------------

def count_tokens(model_name: str, messages: List[Dict[str, str]]) -> int:
    enc = encoding_for_model(model_name)
    total = 0
    for m in messages:
        total += len(enc.encode(m["content"]))
    return total


def chat_complete_local(
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, str]],
    n: int,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> Tuple[List[str], int, int]:
    """
    Call Chat Completions and return text outputs with token usage.

    Gemma shim:
        If the backend model name contains "gemma" and the first message is a
        system prompt, many Gemma-serving stacks reject the "system" role.
        To maintain compatibility, we prepend the system content to the first
        user message and drop the system message.

    Args:
        client: OpenAI-compatible client.
        model_name: Resolved model id to pass to the API.
        messages: OpenAI-style chat messages.
        n: Number of completions to request.
        temperature: Sampling temperature.
        max_tokens: Max tokens for each completion.

    Returns:
        texts: List of generated message contents (length == n).
        prompt_tokens: Count reported by the server (0 if unavailable).
        completion_tokens: Count reported by the server (0 if unavailable).
    """
    # Apply Gemma system-role workaround without mutating caller's list.
    use_messages = list(messages)
    if "gemma" in model_name.lower() and use_messages and use_messages[0].get("role") == "system":
        sys_msg = use_messages[0].get("content", "")
        if len(use_messages) >= 2 and use_messages[1].get("role") == "user":
            merged_user = f"{sys_msg}\n\n{use_messages[1].get('content', '')}"
            use_messages = [dict(use_messages[1], content=merged_user)] + use_messages[2:]
        else:
            # No user message to merge into; degrade gracefully by converting system→user.
            use_messages = [{"role": "user", "content": sys_msg}] + use_messages[1:]

    # # Set the max tokens based on the model context length
    # input_tokens = count_tokens(model_name, use_messages)
    # ctx = 8192  # or detect from model_name

    # allowed_max = max(1, ctx - input_tokens)
    # if max_tokens > allowed_max:
    #     max_tokens = allowed_max

    resp = client.chat.completions.create(
        model=get_model_name(model_name),
        messages=use_messages,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    texts = [(choice.message.content or "").strip() for choice in resp.choices]
    p = int(getattr(resp.usage, "prompt_tokens", 0) or 0)
    c = int(getattr(resp.usage, "completion_tokens", 0) or 0)
    return texts, p, c


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chat_complete_gpt(
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, str]],
    n: int,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> Tuple[List[str], int, int]:
    return chat_complete_local(
        client=client,
        model_name=model_name,
        messages=messages,
        n=n,
        temperature=temperature,
        max_tokens=max_tokens
    )


def chat_complete(
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, str]],
    n: int,
    temperature: float = 0.3,
    max_tokens: int = 2048
):
    if "gpt" in model_name:
        return chat_complete_gpt(
            client=client,
            model_name=model_name,
            messages=messages,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens
        )

    # OpenAI client
    if isinstance(client, OpenAI):
        return chat_complete_local(
            client=client,
            model_name=model_name,
            messages=messages,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens
        )
