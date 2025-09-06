"""
Create OOD math problems from GSM8K+ alt=1 variants.

Build an out-of-distribution (OOD) math split by mapping each curated
in-distribution GSM-style problem to its alt=1 variant from GSM8K+.

Assumptions
-----------
- Base (in-distribution) problems were previously curated and saved to:
  <data>/<teacher.model>/<dataset>/<student.model>.json
  with a key "student_ids" listing kept problem IDs (ints).
- GSM8K+ organizes variants such that examples cycle by id suffixes:
  ...-0 (alt=0), ...-1 (alt=1), ...-2 (alt=2). We exploit this to
  scan efficiently by selecting rows where idx % 3 == 1 for alt==1.
"""

from __future__ import annotations

import os
import time
import logging
from datetime import timedelta
from typing import Any, Dict, List, TypedDict

import json
import hydra
from datasets import load_dataset
from omegaconf import DictConfig
from tqdm import tqdm

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------
# Typing helpers
# ---------------------------------------------------------------------

class MathProblem(TypedDict, total=False):
    """Normalized GSM8K+ math problem format."""
    id: int
    prompt: str
    solution: str


# ---------------------------
# Row normalization
# ---------------------------

def organize_gsm_row_with_alt(row: Dict[str, Any]) -> MathProblem:
    """
    Normalize a GSM8K+ row expected to be an alt=1 variant.

    Parameters
    ----------
    row : Dict[str, Any]
        Dataset row with fields: {"id": "123-1", "question": str, "answer": str}.

    Returns
    -------
    MathProblem
        {"id": int, "prompt": str, "solution": str}

    Raises
    ------
    AssertionError
        If alt != 1 (this function is meant for alt=1 rows only).
    """
    pid_str, alt_str = row["id"].split("-")
    pid, alt = int(pid_str), int(alt_str)
    assert alt == 1
    return {"id": pid, "prompt": row["question"], "solution": row["answer"]}


# ---------------------------
# Build id -> OOD map (alt=1)
# ---------------------------

def build_id_to_ood_map(target_ids: List[int]) -> Dict[int, MathProblem]:
    """
    Scan GSM8K+ 'train' split and collect alt=1 (OOD) rows for the given ids.

    The dataset cycles id variants in a fixed order, so filtering by
    `idx % 3 == 1` is an efficient shortcut for alt==1 rows.

    Parameters
    ----------
    target_ids : List[int]
        Base problem IDs (ints) for which we want alt=1 variants.

    Returns
    -------
    Dict[int, MathProblem]
        Mapping from base id -> normalized OOD (alt=1) problem dict.

    Raises
    ------
    RuntimeError
        If one or more requested ids cannot be found as alt=1 in the dataset.
    """
    want = set(int(x) for x in target_ids)
    id2ood: Dict[int, MathProblem] = {}

    ds = load_dataset("flagopen/gsm8kplus", split="train")
    for idx, row in enumerate(tqdm(ds, desc="Collecting OOD (alt=1) rows")):
        if not want:
            break
        # Fast path: GSM8K+ variant layout implies alt==1 roughly at indices with idx % 3 == 1
        if idx % 3 != 1:
            continue

        pdict = organize_gsm_row_with_alt(row)
        pid = int(pdict["id"])
        if pid in want:
            id2ood[pid] = pdict
            want.remove(pid)

    missing = sorted(list(want))
    if missing:
        # Be explicit if some IDs were not found as alt=1 to catch skew early.
        raise RuntimeError(
            f"Missing alt=1 OOD rows for ids: {missing[:20]}{'...' if len(missing) > 20 else ''}"
        )

    return id2ood


# ---------------------------
# Hydra entry
# ---------------------------

@hydra.main(version_base=None, config_path="../../../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """
    Hydra entrypoint to build the OOD (alt=1) math split.

    Expected input
    --------------
    - File at: <data>/<teacher.model>/<dataset>/<student.model>.json
      containing key "student_ids": List[int]

    Output
    ------
    - File at: <data>/<teacher.model>/<dataset>/<student.model>_ood.json
      containing a flat list of OOD problem dicts (no extra metadata).
    """
    start_time = time.perf_counter()  # ---- start timer ----

    # Resolve paths under data/<teacher.model>/<dataset>/
    teacher_dir = os.path.join(cfg.paths.data, cfg.teacher.model, cfg.experiment.dataset)
    os.makedirs(teacher_dir, exist_ok=True)
    base_problems_path = os.path.join(teacher_dir, f"{cfg.student.model}.json")
    output_fpath = os.path.join(teacher_dir, f"{cfg.student.model}_ood.json")

    # Load student's curated (in-distribution) math problem IDs
    with open(base_problems_path, "r") as f:
        base_json = json.load(f)
    base_ids: List[int] = base_json["student_ids"]
    logger.info("Loaded %d base IDs from %s", len(base_ids), base_problems_path)

    # Build base id -> alt=1 OOD mapping
    id2ood = build_id_to_ood_map(base_ids)

    # Remap each base problem id -> its OOD (alt=1) variant
    ood_list: List[MathProblem] = [id2ood[int(pid)] for pid in base_ids]

    # One flat dump, no extras
    with open(output_fpath, "w") as f:
        json.dump(ood_list, f, indent=2)

    logger.info("Wrote %d OOD math problems to %s", len(ood_list), output_fpath)

    elapsed = time.perf_counter() - start_time
    logger.info("Total time elapsed: %s", timedelta(seconds=int(elapsed)))


if __name__ == "__main__":
    # Helpful for Hydra debugging
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    main()
