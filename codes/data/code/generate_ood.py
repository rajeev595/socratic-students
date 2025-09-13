"""
Generate Out-of-Distribution Coding Problems
--------------------------------------------

Generate out-of-distribution (OOD) coding problems by prompting a teacher
model to create a new question similar to a base (in-distribution) problem.
Each generated OOD item is validated via ground-truth tests, filtered by
student pass@k (skip if too easy), and finally checked by a teacher solution
(pass required). The kept set is saved to JSON and supports resume.

Notes
-----
- Hydra + OmegaConf (flat config). No argparse, no dotenv.
- The base problems are read from: <data>/<teacher.model>/<dataset>/<student.model>.json
- The OOD output is written to:   <data>/<teacher.model>/<dataset>/<student.model>_ood.json
- Token usage (prompt/completion) is tracked approximately.
"""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple, TypedDict

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from openai import OpenAI
from tqdm import tqdm

from codes.evaluation.code import generate_code_completions, evaluate_code_completion
from codes.evaluation.utils import compute_pass_at_k
from codes.data.utils import extract_code_blocks
from codes.llm_utils import build_client, chat_complete


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

class ProblemDict(TypedDict, total=False):
    """Internal normalized/base or OOD problem format."""
    id: int
    prompt: str
    solution: str
    test: str
    entry_point: str


class OODSummary(TypedDict):
    """Return value shape of the main OOD routine."""
    count: int
    tokens: Dict[str, int]  # {"prompt": int, "completion": int}


# ---------------------------------------------------------------------
# OOD generation logic
# ---------------------------------------------------------------------

def build_generation_prompt(problem: ProblemDict) -> str:
    """
    Build a prompt for the teacher model to generate a *similar* programming
    problem using the base problem's prompt/solution/tests/entry point.

    Parameters
    ----------
    problem : ProblemDict
        Base problem dict with keys: "prompt", "solution", "test", "entry_point".

    Returns
    -------
    str
        A single user message string instructing the teacher to output
        a look-alike problem in the same sectioned format.
    """
    return f"""You are an expert Python coding assistant. Your task is to generate a programming question that is very similar to the one given to you. The generated question should use roughly the same idea/concepts/libraries/algorithms as the given question. The ideal question would be one such that a student, upon seeing the solution to the first problem, should be able to easily solve the second problem. The given problem is: 

### Question Start ###

{problem["prompt"]}

### Question end ###
### Solution start ###

{problem["solution"]}

### Solution end ###
### Test cases start ###

{problem["test"]}

### Test cases end ###
### Entry point start ###

{problem["entry_point"]}

### Entry point end ###

Now generate the similar question following the same formatting above."""


def parse_generated(text: str) -> Tuple[str, str, str, str]:
    """
    Parse a teacher-generated block into (prompt, solution, tests, entry_point).

    The teacher is expected to return sections delimited by:
      - '### Question Start ###' ... '### Question end ###'
      - '### Solution start ###' ... '### Solution end ###'
      - '### Test cases start ###' ... '### Test cases end ###'
      - '### Entry point start ###' ... '### Entry point end ###'

    Code fences (```python ... ```) are stripped if present.

    Returns
    -------
    Tuple[str, str, str, str]
        (prompt, solution, tests, entry_point), each possibly empty if parsing fails.
    """
    def block(start: str, end: str) -> str:
        m = re.search(re.escape(start) + r"(.*?)" + re.escape(end), text, re.DOTALL)
        return (m.group(1).strip() if m else "").strip()

    prompt = block("### Question Start ###", "### Question end ###")
    solution = block("### Solution start ###", "### Solution end ###")
    tests = block("### Test cases start ###", "### Test cases end ###")
    entry = block("### Entry point start ###", "### Entry point end ###")

    # Strip final fenced code block if present; otherwise return raw
    def _strip_fence(s: str) -> str:
        blocks = extract_code_blocks(s)
        return blocks[-1] if blocks else s

    prompt, solution, tests, entry = map(_strip_fence, (prompt, solution, tests, entry))

    # Ensure trailing newlines where execution may rely on them
    if prompt and not prompt.endswith("\n"): prompt += "\n"
    if solution and not solution.endswith("\n"): solution += "\n"
    if tests and not tests.endswith("\n"): tests += "\n"

    return prompt, solution, tests, entry


# ---------------------------------------------------------------------
# Safe evaluation wrapper
# ---------------------------------------------------------------------

def _safe_eval_code(args: Tuple[str, ProblemDict, bool]) -> Tuple[Any, bool]:
    """
    Guard evaluate_code_completion to avoid killing threads on unexpected errors.

    Args
    ----
    args: (completion_code, problem_dict, sanity_check)

    Returns
    -------
    (info, passed)
      - info: evaluator return (arbitrary) or {"error": "..."} on failure
      - passed: bool
    """
    code, problem, sanity_check = args
    try:
        return evaluate_code_completion(code, problem, sanity_check=sanity_check)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}, False


def generate_ood(
    problem,
    teacher_client: OpenAI,
    teacher_name: str,
    teacher_temperature: float,
    student_client: OpenAI,
    student_name: str,
    student_temperature: float,
    k_values: List[int],
    n_samples: int,
    pass_k: int,
    pass_at_k_threshold: float,
    retry_on_fail: int
):
    pid = problem["id"]
    total_prompt_tks = total_completion_tks = 0

    # Let the teacher generate a similar problem; retry a few times if needed
    num_attempts = retry_on_fail
    while num_attempts > 0:
        is_first_attempt = num_attempts == retry_on_fail

        if not is_first_attempt:
            messages.append({"role": "user", "content": "Give me another alternative."})
        else:
            gen_prompt = build_generation_prompt(problem)
            messages = [{"role": "user", "content": gen_prompt}]

        # generate new problems
        texts, p, c = chat_complete(
            teacher_client, teacher_name, messages,
            n=1, temperature=teacher_temperature, max_tokens=2048
        )
        total_prompt_tks += p
        total_completion_tks += c
        raw = texts[0] if texts else ""

        messages.append({"role": "assistant", "content": raw})

        q_prompt, q_solution, q_tests, q_entry = parse_generated(raw)

        # Minimal validation
        if not (q_prompt and q_solution and q_tests and q_entry):
            logger.debug("Parse failed or missing fields for pid=%d", pid)
            continue

        ood_problem: ProblemDict = {
            "id": pid,
            "prompt": q_prompt,
            "solution": q_solution,
            "test": q_tests,
            "entry_point": q_entry,
        }

        # Check if the teacher can solve the questions
        _, teacher_ok = evaluate_code_completion("", ood_problem, sanity_check=True)
        if not teacher_ok:
            is_first_attempt = False
            num_attempts -= 1
            logger.debug("Teacher failed!")
            continue

        # Probe student with N attempts; skip if already easy
        student_codes, sp, sc = generate_code_completions(
            previous_conversation_str="",
            client=student_client,
            model_name=student_name,
            temperature=student_temperature,
            problem_dict=ood_problem,
            num_samples=n_samples
        )
        total_prompt_tks += sp
        total_completion_tks += sc

        results: List[bool] = []
        with ThreadPoolExecutor() as ex:
            futs = [ex.submit(_safe_eval_code, (code, ood_problem, False)) for code in student_codes]
            for fut in as_completed(futs):
                _, passed = fut.result()
                results.append(bool(passed))

        scores = compute_pass_at_k(results, k_values)
        if scores[pass_k] >= pass_at_k_threshold:
            is_first_attempt = False
            num_attempts -= 1
            logger.debug("Student pass threshold for pid=%d (pass@%d=%.3f)", pid, pass_k, scores[pass_k])
            continue
        
        break

    return ood_problem, total_prompt_tks, total_completion_tks


# ---------------------------------------------------------------------
# Hydra entry
# ---------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../../../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """
    Hydra entrypoint to launch OOD generation.

    Expected config keys (illustrative):
    - paths.openai_api_key_fpath : str
    - paths.data                 : str
    - teacher.model              : str
    - teacher.temperature        : float
    - student.model              : str
    - student.temperature        : float
    - experiment.dataset         : str
    - gen.k_values               : List[int]
    - gen.num_samples            : int
    - gen.pass_k                 : int
    - gen.pass_at_k_threshold    : float
    - gen.retry_on_fail          : int
    """
    # Export API key for the SDK
    key_path = to_absolute_path(str(cfg.paths.openai_api_key_fpath))
    with open(key_path, "r") as f:
        os.environ["OPENAI_API_KEY"] = f.read().strip()

    # Build clients (model IDs may be normalized by build_client)
    teacher_name, teacher_client = build_client(str(cfg.teacher.model))
    student_name, student_client = build_client(str(cfg.student.model))
    logger.info("Teacher=%s | Student=%s", teacher_name, student_name)

    # IO paths under data/<teacher.model>/<dataset>/
    teacher_dir = os.path.join(cfg.paths.data, str(cfg.teacher.model), cfg.experiment.dataset)
    os.makedirs(teacher_dir, exist_ok=True)

    base_problems_path = os.path.join(teacher_dir, f"{cfg.student.model}.json")
    output_fpath = os.path.join(teacher_dir, f"{cfg.student.model}_ood.json")

    with open(base_problems_path, "r") as f:
        base_json = json.load(f)
    base_problems: List[ProblemDict] = base_json["collected"]
    logger.info("Loaded %d base problems from %s", len(base_problems), base_problems_path)

    # Resume bookkeeping
    ood_list: List[ProblemDict] = []
    collected_ids: List[int] = []

    if os.path.exists(output_fpath):
        with open(output_fpath, "r") as f:
            ood_list = json.load(f)
        collected_ids = [ood["id"] for ood in ood_list]
        logger.info(
            "Resuming: OOD=%d", len(ood_list)
        )

    total_prompt_tks = total_completion_tks = 0
    n_samples = min(cfg.gen.num_samples, max(list(cfg.gen.k_values)))  # ensure pass@k is computable

    # Parallel execution
    with ThreadPoolExecutor(max_workers=cfg.experiment.num_workers) as executor:
        jobs = [executor.submit(
            generate_ood,
            base,
            teacher_client,
            teacher_name,
            float(cfg.teacher.temperature),
            student_client,
            student_name,
            float(cfg.student.temperature),
            list(cfg.gen.k_values),
            n_samples,
            int(cfg.gen.pass_k),
            float(cfg.gen.pass_at_k_threshold),
            int(cfg.gen.retry_on_fail)
        ) for base in base_problems if base["id"] not in collected_ids]

    for future in tqdm(as_completed(jobs), total=len(jobs), desc="Generating OOD"):
        ood_problem, p_tks, c_tks = future.result()

        # Update
        ood_list.append(ood_problem)
        total_prompt_tks += p_tks
        total_completion_tks += c_tks

    # Final save
    with open(output_fpath, "w") as f:
        json.dump(ood_list, f, indent=2)

    logger.info("Saved %d OOD problems to %s", len(ood_list), output_fpath)
    logger.info("Token usage (approx): prompt=%d, completion=%d", total_prompt_tks, total_completion_tks)



if __name__ == "__main__":
    # Helpful for debugging Hydra tracebacks
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
