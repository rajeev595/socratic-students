import os
import sys
import json
import time
import socket
import signal
import atexit
import logging
import requests
import subprocess
from contextlib import contextmanager
from typing import List, Optional, Sequence, Dict


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)


# Known local nicknames → (HF-style model id, port)
_MODEL_ALIASES: Dict[str, str] = {
    # local instruct models
    "llama8": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3": "meta-llama/Llama-3.2-3B-Instruct",
    "llama1": "meta-llama/Llama-3.2-1B-Instruct",

    "qwen3": "Qwen/Qwen2.5-3B-Instruct",
    "qwen7": "Qwen/Qwen2.5-7B-Instruct",
    "qwen14": "Qwen/Qwen2.5-14B-Instruct",
    "qwen30": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "qwen32": "Qwen/Qwen2.5-32B-Instruct",
    "qwen72": "Qwen/Qwen2.5-72B-Instruct",
    "mistral7": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral24": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "phi4": "microsoft/phi-4",

    # deepseek-r1 distilled
    "r1llama8": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "r1qwen7": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "r1qwen14": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",

    # gemma family (some backends reject 'system' role)
    "gemma9": "google/gemma-2-9b-it",
    "gemma4": "google/gemma-3-4b-it",
    "gemma27": "google/gemma-2-27b-it",
    "gemma1": "google/gemma-3-1b-it",
    "phi4mini": "microsoft/Phi-4-mini-instruct"
}


def get_model_name(nickname: str):
    if nickname in _MODEL_ALIASES:
        return _MODEL_ALIASES[nickname]
    else:
        return nickname


# ---------------------------
# Shared helpers
# ---------------------------


def _pick_free_port(start: int = 30000, span: int = 4000) -> int:
    for p in range(start, start + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("No free port available in range.")


def _linux_pdeathsig_preexec():
    """Ensure child gets SIGTERM when parent dies (Linux)."""
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception:
        # best-effort; atexit+group-kill still in place
        pass


def _wait_ready(base_url: str, proc: subprocess.Popen, timeout: float, verbose: bool = False):
    """
    Polls until the OpenAI-compatible API is ready. Primary probe: /v1/models.
    Fallback probe: HEAD /v1/chat/completions (some builds expose this first).
    """
    deadline = time.time() + timeout
    url1 = f"{base_url}/models"
    url2 = f"{base_url}/chat/completions"  # fallback probe
    last_err = None

    while time.time() < deadline:
        if proc.poll() is not None:
            out = err = ""
            # Only try communicate if output is piped (not verbose)
            if proc.stdout is not None or proc.stderr is not None:
                try:
                    out, err = proc.communicate(timeout=1)
                except Exception:
                    pass
            raise RuntimeError(
                f"Server exited during startup (code={proc.returncode}).\n"
                f"STDOUT:\n{out}\nSTDERR:\n{err}"
            )

        try:
            r = requests.get(url1, timeout=2)
            if r.status_code == 200:
                if verbose:
                    print("[server] ready:", base_url)
                return
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            last_err = repr(e)

        # Fallback probe: if HEAD succeeds, we consider it ready
        try:
            requests.head(url2, timeout=1)
            if verbose:
                print("[server] ready (fallback):", base_url)
            return
        except Exception:
            pass

        time.sleep(0.5)

    raise TimeoutError(f"Server not ready after {int(timeout)}s. Last error: {last_err}")


def _start_subprocess(cmd: Sequence[str], env: Dict[str, str], verbose: bool) -> subprocess.Popen:
    if verbose:
        print("[server] launch:", " ".join(cmd))
        print("[server] CUDA_VISIBLE_DEVICES:", env.get("CUDA_VISIBLE_DEVICES", ""))
    # Avoid PIPE deadlocks by discarding output when not verbose.
    return subprocess.Popen(
        list(cmd),
        env=env,
        stdout=None if verbose else subprocess.DEVNULL,
        stderr=None if verbose else subprocess.DEVNULL,
        text=True,
        bufsize=1,
        start_new_session=True,  # own process group so we can kill children
        preexec_fn=_linux_pdeathsig_preexec if sys.platform.startswith("linux") else None,
    )


def _register_cleanup(proc: subprocess.Popen):
    def _cleanup():
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
    atexit.register(_cleanup)


def _shutdown(proc: subprocess.Popen, timeout: float = 15.0):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


# ---------------------------
# SGLang launcher (OpenAI API)
# ---------------------------

@contextmanager
def launch_sglang_client(
    *,
    model: str,                     # HF id or local path
    gpus: List[int],
    port: Optional[int] = None,
    host: str = "127.0.0.1",
    tp: Optional[int] = None,
    dtype: str = "auto",
    lora: Optional[str] = None,         # uses --lora-paths "path1,path2"
    allow_download: bool = True,
    trust_remote_code: bool = True,
    max_model_len: Optional[int] = None,
    extra_server_args: Optional[List[str]] = None,
    env_overrides: Optional[Dict[str, str]] = None,
    ready_timeout: float = 600.0,
    exe: Optional[str] = None,          # Python executable for `-m sglang.launch_server`
    verbose: bool = False,
):
    """Yields an OpenAI-compatible client for SGLang; server is auto-killed on exit/crash."""
    exe = exe or sys.executable
    tp = tp or max(1, len(gpus))
    port = port or _pick_free_port()
    base_url = f"http://{host}:{port}/v1"
    extra_server_args = extra_server_args or []

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    if not allow_download:
        env["HF_HUB_OFFLINE"] = "1"
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
    if env_overrides:
        env.update({str(k): str(v) for k, v in env_overrides.items()})

    cmd = [
        exe, "-m", "sglang.launch_server",
        "--model", model,
        "--host", host,
        "--port", str(port),
        "--tp", str(tp),
        "--dtype", dtype,
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if max_model_len:
        cmd += ["--max-model-len", str(max_model_len)]
    if lora and lora.name:
        cmd.append("--lora-paths")
        # Create the JSON schema that SGLang expects
        lora_spec = {
            "lora_name": lora.name,
            "lora_path": lora.path,
            "pinned": True
        }
        # Append to command as separate args
        cmd.append(json.dumps(lora_spec))
    if extra_server_args:
        cmd += list(extra_server_args)

    proc = _start_subprocess(cmd, env, verbose)
    _register_cleanup(proc)
    try:
        _wait_ready(base_url, proc, ready_timeout, verbose)
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key="sglang")
        try:
            yield client
        finally:
            _shutdown(proc)
    finally:
        if proc.poll() is None:
            _shutdown(proc, timeout=10.0)


# ---------------------------
# vLLM launcher (OpenAI API) — using `vllm serve`
# ---------------------------

@contextmanager
def launch_vllm_client(
    *,
    model: str,                     # HF id or local path
    gpus: List[int],
    port: Optional[int] = None,
    host: str = "127.0.0.1",
    tp: Optional[int] = None,       # tensor parallel (defaults to len(gpus))
    dtype: Optional[str] = None,    # "bfloat16" | "float16" | None (vLLM default; NO 'auto')
    lora: Optional[str] = None,     # if provided: uses --enable-lora --lora-modules default=<path>
    allow_download: bool = True,
    trust_remote_code: bool = True,
    max_model_len: Optional[int] = None,       # --max-model-len
    served_model_name: Optional[str] = None,   # --served-model-name (optional)
    extra_server_args: Optional[List[str]] = None,
    env_overrides: Optional[Dict[str, str]] = None,
    ready_timeout: float = 600.0,
    exe: Optional[str] = None,      # path to `vllm` CLI; defaults to "vllm"
    verbose: bool = False,
):
    """Yields an OpenAI-compatible client for vLLM using the `vllm serve` CLI; server auto-killed."""
    exe = exe or "vllm"
    tp = tp or max(1, len(gpus))
    port = port or _pick_free_port()
    base_url = f"http://{host}:{port}/v1"
    extra_server_args = extra_server_args or []

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    if not allow_download:
        env["HF_HUB_OFFLINE"] = "1"
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
    if env_overrides:
        env.update({str(k): str(v) for k, v in env_overrides.items()})

    # Build `vllm serve` command (OpenAI-compatible endpoint)
    cmd = [
        exe, "serve",
        model,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tp),
    ]
    if served_model_name:
        cmd += ["--served-model-name", served_model_name]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if dtype:
        cmd += ["--dtype", dtype]  # note: vLLM expects explicit dtype or omit entirely
    if max_model_len:
        cmd += ["--max-model-len", str(max_model_len)]
    if lora and lora.name:
        cmd += [
            "--enable-lora", "--lora-modules", f"{lora.name}={lora.path}",
            # "--quantization", "bitsandbytes", "--load-format", "bitsandbytes"
        ]

    if extra_server_args:
        cmd += list(extra_server_args)

    proc = _start_subprocess(cmd, env, verbose)
    _register_cleanup(proc)
    try:
        _wait_ready(base_url, proc, ready_timeout, verbose)
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key="vllm")
        try:
            yield client
        finally:
            _shutdown(proc)
    finally:
        if proc.poll() is None:
            _shutdown(proc, timeout=10.0)


# ------------------------------------------------------
# OpenAI Client to use SGLang/vLLM local servers
# ------------------------------------------------------
def build_client(
    model: str,
    gpus: List[int],
    server: str = "sglang",
    port: Optional[int] = None,
    host: str = "127.0.0.1",
    tp: Optional[int] = None,
    dtype: str = "auto",
    lora: Optional[str] = None,
    allow_download: bool = True,
    trust_remote_code: bool = True,
    max_model_len: Optional[int] = None,
    served_model_name: Optional[str] = None,
    extra_server_args: Optional[List[str]] = None,
    env_overrides: Optional[Dict[str, str]] = None,
    ready_timeout: float = 600.0,
    exe: Optional[str] = None,
    verbose: bool = True,
):
    if not os.path.exists(model):
        try:
            model = get_model_name(model)
        except Exception as e:
            logger.info(e)
            logger.info(f"{model} is NOT a valid path or model name")

    if server == "sglang":
        cm = launch_sglang_client(
            model=model,
            gpus=gpus,
            port=port,
            host=host,
            tp=tp,
            dtype=dtype,
            lora=lora,
            allow_download=allow_download,
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
            extra_server_args=extra_server_args,
            env_overrides=env_overrides,
            ready_timeout=ready_timeout,
            exe=exe,
            verbose=verbose,
        )
    elif server == "vllm":
        dtype_for_vllm = None if dtype == "auto" else dtype
        cm = launch_vllm_client(
            model=model,
            gpus=gpus,
            port=port,
            host=host,
            tp=tp,
            dtype=dtype_for_vllm,
            lora=lora,
            allow_download=allow_download,
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
            served_model_name=served_model_name,
            extra_server_args=extra_server_args,
            env_overrides=env_overrides,
            ready_timeout=ready_timeout,
            exe=exe,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown server type: {server!r}. Use 'sglang' or 'vllm'.")

    # Enter the context immediately and register cleanup
    client = cm.__enter__()
    atexit.register(lambda: cm.__exit__(None, None, None))
    return client
