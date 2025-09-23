"""
Lightweight LLM utilities.

- `build_client`: resolve a model nickname → (fully qualified model id, OpenAI-compatible client).
  Known nicknames are served via local OpenAI-compatible servers at http://localhost:{port}/v1.
  Any other string is treated as a first-party OpenAI model id (e.g., "gpt-4o").

- `chat_complete`: thin wrapper over Chat Completions that:
    * passes through temperature, n, max_tokens,
    * returns (list of texts, prompt_tokens, completion_tokens),
    * applies a Gemma-specific shim: if the first message is a system prompt,
      prepend it to the first user message and drop the system role (many Gemma backends
      reject the "system" role).
"""

from __future__ import annotations

import os
import time
import psutil
import signal
import atexit
import requests
import logging
import threading
import subprocess
from typing import List, Dict, Tuple

from hydra.utils import to_absolute_path

from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from vllm import LLM, SamplingParams

_VLLM_LOCK = threading.Lock()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Client + model resolution
# ---------------------------------------------------------------------------

# Known local nicknames → (HF-style model id, port)
_MODEL_ALIASES: Dict[str, Tuple[str, int | None]] = {
    # local instruct models
    "llama8":   ("meta-llama/Llama-3.1-8B-Instruct", 6006),
    "qwen3":    ("Qwen/Qwen2.5-3B-Instruct",         6007),
    "qwen7":    ("Qwen/Qwen2.5-7B-Instruct",         6008),
    "qwen14":   ("Qwen/Qwen2.5-14B-Instruct",        6009),
    "qwen30":   ("Qwen/Qwen3-30B-A3B-Instruct-2507", 6023),
    "qwen32":   ("Qwen/Qwen2.5-32B-Instruct",        6010),
    "qwen72":   ("Qwen/Qwen2.5-72B-Instruct",        6020),
    "mistral7": ("mistralai/Mistral-7B-Instruct-v0.3", 6011),
    "mistral24": ("mistralai/Mistral-Small-3.2-24B-Instruct-2506", 6021),
    "phi4":     ("microsoft/phi-4",                  6013),
    # deepseek-r1 distilled
    "r1llama8": ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 6014),
    "r1qwen7":  ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",  6015),
    "r1qwen14": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", 6022),
    # gemma family (some backends reject 'system' role)
    "gemma9":   ("google/gemma-2-9b-it",             6012),
    "gemma4":   ("google/gemma-3-4b-it",             6016),
    "gemma27":  ("google/gemma-2-27b-it",            6017),
    "gemma1":   ("google/gemma-3-1b-it",             6018),
    "phi4mini": ("microsoft/Phi-4-mini-instruct",    6019)
}


def set_openai_key_env(openai_api_key_fpath: str) -> None:
    # Load API key and export for OpenAI SDK
    key_path = to_absolute_path(str(openai_api_key_fpath))
    with open(key_path, "r") as f:
        openai_key = f.read().strip()
    os.environ["OPENAI_API_KEY"] = openai_key


def free_port(port):
    for conn in psutil.net_connections(kind="inet"):
        if conn.laddr.port == port:
            if conn.pid:
                try:
                    p = psutil.Process(conn.pid)
                    print(f"Killing PID {conn.pid} ({p.name()}) using port {port}")
                    p.kill()
                except Exception as e:
                    print(f"Could not kill PID {conn.pid}: {e}")


def launch_vllm_server(model, fpath):
    model_name, port = _MODEL_ALIASES[model]

    # free port
    free_port(port)
    base_fpath = fpath if type(fpath) is str else fpath[0]

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(base_fpath),
        "--port", str(port),
        "--host", "127.0.0.1"
    ]
    if type(fpath) is not str:
        cmd += [
            "--lora-paths", f"lora1={str(fpath[1])}"
        ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    proc = subprocess.Popen(cmd, env=env)

    logger.info(f"vLLM server for {model_name} running on http://localhost:{port}/v1")

    return proc


def launch_sglang_server(model, fpath):
    """
    Start SGLang's OpenAI-compatible server for a local model dir.
    Returns a Popen handle. The server lazily loads on first request.
    """
    model_name, port = _MODEL_ALIASES[model]
    host = "127.0.0.1"

    # free port
    free_port(port)
    base_fpath = fpath if isinstance(fpath, str) else fpath[0]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", base_fpath,
        "--host", str(host),
        "--port", str(port),
        "--tp", "4"
    ]
    if not isinstance(fpath, str):
        cmd += [
            "--lora-paths", f"{str(fpath[1])}"
        ]

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Wait for the HTTP server to come up
    base = f"http://{host}:{port}"
    for _ in range(120):
        try:
            r = requests.get(f"{base}/v1/models", timeout=1)
            if r.ok:
                break
        except Exception:
            time.sleep(0.5)

    logger.info(f"SGLang server for {model_name} running on {base}/v1")

    return proc


def build_client(model: str, fpath: str, server: bool) -> Tuple[str, OpenAI]:
    """
    Resolve a model identifier and construct an OpenAI-compatible client.

    Args:
        model: Either a known nickname (e.g., "qwen7", "gemma9") or a full OpenAI model id (e.g., "gpt-4o").

    Returns:
        Tuple of:
            - resolved_model_name: Fully qualified model id to pass to Chat Completions.
            - client: OpenAI SDK client. If the nickname maps to a local port, the client
              points at http://localhost:{port}/v1; otherwise it uses the default OpenAI API.

    Environment:
        Requires OPENAI_API_KEY in the environment. For local servers that do not validate
        the key, the value may be any non-empty string.
    """
    if not server:
        llm = LLM(model=fpath, enforce_eager=True, disable_log_stats=True)
        setup_shutdown_hooks(llm)
        return fpath, llm

    openai_api_key = os.environ["OPENAI_API_KEY"]

    if model in _MODEL_ALIASES:
        # process = launch_sglang_server(model, fpath)
        # setup_shutdown_hooks(process)

        resolved_name, port = _MODEL_ALIASES[model]
        base_url = f"http://localhost:{port}/v1"
        # Some local servers ignore the key, but the SDK requires a non-empty value.
        client = OpenAI(api_key=openai_api_key or "no-key", base_url=base_url)
        return resolved_name, client

    # Treat unknown strings as first-party OpenAI model ids (e.g., "gpt-4o").
    if "gpt" in model:
        return model, OpenAI(api_key=openai_api_key)


# Global registry of things to shut down
_MANAGED_PROCS = set()
_HOOKS_INSTALLED = False
_SHUTDOWN_ONCE = threading.Event()

def setup_shutdown_hooks(process) -> None:
    """Register a process (LLM or Popen) for shutdown; installs hooks once."""
    global _HOOKS_INSTALLED

    if process is None:
        return
    _MANAGED_PROCS.add(process)

    if _HOOKS_INSTALLED:
        return
    _HOOKS_INSTALLED = True

    def _shutdown(*_):
        if _SHUTDOWN_ONCE.is_set():
            return
        _SHUTDOWN_ONCE.set()

        for proc in list(_MANAGED_PROCS):
            try:
                if isinstance(proc, LLM):
                    if hasattr(proc, "shutdown"):
                        proc.shutdown()
                    elif getattr(getattr(proc, "llm_engine", None), "shutdown", None):
                        proc.llm_engine.shutdown()  # type: ignore[attr-defined]
                elif isinstance(proc, subprocess.Popen):
                    try:
                        proc.terminate()
                        proc.wait(timeout=10)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
            except Exception:
                pass
        _MANAGED_PROCS.clear()

    atexit.register(_shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)


# ---------------------------------------------------------------------------
# Chat completion
# ---------------------------------------------------------------------------

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

    resp = client.chat.completions.create(
        model=model_name,
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


def chat_complete_vllm(
    llm: LLM,
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

    text = llm.get_tokenizer().apply_chat_template(
        use_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    sampling_params = SamplingParams(temperature=temperature, n=n, max_tokens=max_tokens)
    with _VLLM_LOCK:
        response = llm.generate(text, sampling_params, use_tqdm=False)

    texts = []
    p, c = 0, 0
    for req in response:
        p += len(req.prompt_token_ids)

        for output in req.outputs:
            texts.append((output.text or "").strip())
            c += len(output.token_ids)

    return texts, p, c


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

    # vLLM client
    if isinstance(client, LLM):
        return chat_complete_vllm(
            llm=client,
            model_name=model_name,
            messages=messages,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens
        )
