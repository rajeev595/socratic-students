import tempfile
import json
import os
from json import JSONDecodeError
from typing import Any


def save_json_atomic(path: str, obj: Any, *, backup: bool = True, indent: int = 2) -> None:
    """
    Write JSON atomically:
      - write to a temp file in the same dir
      - flush + fsync
      - optionally rotate a .bak
      - os.replace to target (atomic on POSIX)
    """
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)

    # Serialize first to catch errors before touching disk
    data = json.dumps(obj, indent=indent)

    # Write to temp
    with tempfile.NamedTemporaryFile("w", dir=d, delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name

    # Rotate backup
    if backup and os.path.exists(path):
        bak = f"{path}.bak"
        try:
            os.replace(path, bak)  # atomic move
        except OSError:
            pass  # best-effort

    # Atomic replace into place
    os.replace(tmp_path, path)


def load_json_resilient(path: str) -> Any:
    """
    Try loading JSON from path; if corrupt, attempt .bak.
    Raise the original error only if both fail.
    """
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, JSONDecodeError) as e1:
        bak = f"{path}.bak"
        try:
            with open(bak, "r") as f:
                return json.load(f)
        except Exception as e2:
            # Re-raise the first error with context
            raise RuntimeError(f"Failed to load JSON from '{path}' and backup '{bak}'") from e1
