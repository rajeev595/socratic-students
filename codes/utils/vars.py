import os


def set_openai_key_env(openai_api_key_fpath: str) -> None:
    # Load API key and export for OpenAI SDK
    with open(openai_api_key_fpath, "r") as f:
        openai_key = f.read().strip()
    os.environ["OPENAI_API_KEY"] = openai_key


def set_hf_env(hf_home: str) -> None:
    os.environ["HF_HOME"] = hf_home


def set_envs(openai_api_key_fpath, hf_home) -> None:
    set_openai_key_env(openai_api_key_fpath)
    set_hf_env(hf_home)
