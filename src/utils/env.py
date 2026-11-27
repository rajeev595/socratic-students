import os
from typing import Optional


def set_openai_key_env(openai_api_key_fpath: Optional[str]) -> None:
    """Export an OpenAI key from a text file. Optional: local vLLM/SGLang servers do not need one."""
    if not openai_api_key_fpath:
        return
    with open(os.path.expanduser(str(openai_api_key_fpath)), "r") as f:
        os.environ["OPENAI_API_KEY"] = f.read().strip()


def set_hf_env(hf_home: Optional[str]) -> None:
    """Point the Hugging Face cache at `hf_home`. If empty, keep whatever HF_HOME is already set."""
    if hf_home:
        os.environ["HF_HOME"] = os.path.expanduser(str(hf_home))


def hf_cache_dir() -> Optional[str]:
    """Hub cache dir under HF_HOME, or None to use the Hugging Face default location."""
    hf_home = os.environ.get("HF_HOME")
    return os.path.join(hf_home, "hub") if hf_home else None


def set_hydra_env() -> None:
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")


def set_envs(openai_api_key_fpath: Optional[str] = None, hf_home: Optional[str] = None) -> None:
    set_hydra_env()
    set_openai_key_env(openai_api_key_fpath)
    set_hf_env(hf_home)
