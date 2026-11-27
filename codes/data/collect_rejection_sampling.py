"""
Rejection Sampling data collection (baseline).

Same candidate sampling as codes.data.collect_self_odqs, but the best candidate question at every
turn is picked by an LLM judge (the teacher model) instead of by the student's downstream Pass@k.

Output (under <data>/<teacher>/<dataset>/<data.name>/):
  <student>_sft_iter0_<split>.jsonl    judge-selected questions
  <student>_dpo_iter0_<split>.jsonl    (selected > rejected) preference pairs

Usage:
    python -m codes.data.collect_rejection_sampling student.model=qwen7 data.type=math experiment.dataset=gsm8k \
        data.name=rejection_sampling data.split=train data.data_amount=1000 experiment.name=rejection_sampling
"""

import os
import time
import json
import random
import logging
from datetime import timedelta
from typing import Any, List, Dict, Tuple

import hydra
from omegaconf import DictConfig
from openai import OpenAI

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from codes.data.utils import format_messages, msgs_to_string
from codes.prompts.utils import _select_sys_prompt, _problem_greeting
from codes.run_interact import interact
from codes.utils.llm.server import build_client
from codes.utils.llm.chat import chat_complete
from codes.utils.vars import set_envs
from codes.utils.file_ops import load_json_resilient

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)

# =========================================================
# Helpers
# =========================================================
def _extract_text(msg: Any) -> str:
    """Best-effort extraction of the question text from a message-like object."""
    if isinstance(msg, dict) and "content" in msg:
        return msg["content"]
    if isinstance(msg, str):
        return msg
    try:
        return json.dumps(msg, ensure_ascii=False)
    except Exception:
        return str(msg)

# -------------------------
# Judge selection (teacher-as-judge)
# -------------------------
def _build_judge_prompt(
    data_type: str,
    problem: Dict[str, Any],
    history_texts: List[Any],
    candidates: List[Any],
    extra_instructions: str = ""
) -> List[Dict[str, str]]:
    """
    System+User messages asking the Judge (teacher LLM) to pick the best candidate by index.
    Returns OpenAI-style messages list.
    """
    def _msg_to_str(m):
        return m["content"] if isinstance(m, dict) and "content" in m else str(m)

    history_str = "\n".join([_msg_to_str(x) for x in history_texts])

    candidate_block = []
    for i, c in enumerate(candidates, 1):
        c_text = c["content"] if isinstance(c, dict) and "content" in c else str(c)
        candidate_block.append(f"[{i}] {c_text.strip()}")

    sys = (
        "You are a careful, impartial Judge. "
        "Given the problem, the prior conversation, and multiple candidate student questions, "
        "choose the SINGLE best candidate that is most likely to elicit helpful guidance from the teacher "
        "and ultimately improve final task performance. "
        "Return ONLY the integer index of the best candidate (1..K)."
    )
    if extra_instructions:
        sys += " " + extra_instructions.strip()

    user = (
        f"Task type: {data_type}\n\n"
        f"Problem:\n{json.dumps(problem, ensure_ascii=False, indent=2)}\n\n"
        f"Conversation so far:\n{history_str}\n\n"
        f"Candidates:\n" + "\n".join(candidate_block) + "\n\n"
        "Answer with just the index (1..K). No text, no JSON."
    )

    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user}
    ]


def judge_select_best_index(
    client: OpenAI,
    model_name: str,
    data_type: str,
    problem: Dict[str, Any],
    history_texts: List[Any],
    candidates: List[Any],
    temperature: float = 0.0,
    extra_instructions: str = "",
) -> int:
    """
    Calls the teacher client to judge the best candidate among 'candidates'.
    Returns 0-based best index. Falls back to 0 on parsing failure.
    """
    messages = _build_judge_prompt(
        data_type=data_type,
        problem=problem,
        history_texts=history_texts,
        candidates=candidates,
        extra_instructions=extra_instructions,
    )
    try:
        resp, p_tks, c_tks = chat_complete(
            client=client,
            model_name=model_name,
            messages=messages,
            n=1,
            temperature=temperature,
            max_tokens=4
        )

        raw = resp[0]
        digits = "".join(ch for ch in raw if ch.isdigit())
        idx1 = int(digits) if digits else 1
        idx0 = max(0, min(len(candidates) - 1, idx1 - 1))
        return idx0
    except Exception as e:
        logger.warning(f"Judge parse failure ({type(e).__name__}): {e}. Falling back to index 0.")
        return 0

# =========================================================
# Core runner (Judge baseline): ONLY SFT + DPO
# =========================================================
def run_problem_judge(
    data_type: str,
    teacher_client: OpenAI,
    teacher_name: str,
    teacher_temperature: float,
    student_client: OpenAI,
    student_name: str,
    student_temperature: float,
    interaction_problem: Dict[str, Any],
    evaluation_problem: Dict[str, Any],
    # generation knobs
    num_candidates: int,
    start_from_turn: int,
    total_turns: int,
    eps: float,
    use_scaffold: bool,
    # judge knobs
    judge_temperature: float,
    judge_extra_instructions: str,
    judge_use_all_as_rejects: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int, int, int]:
    """
    Run a single problem with teacher-as-judge selection:
      - Winner → SFT
      - All other candidates → DPO rejected (paired with winner)
    Returns:
      chosen_list (SFT), pairs (DPO), and token usage tallies.
    """
    assert interaction_problem["id"] == evaluation_problem["id"], "pid of interaction and evaluation doesn't match!"

    pid = interaction_problem["id"]
    total_teacher_prompt_tks = total_teacher_completion_tks = 0
    total_student_prompt_tks = total_student_completion_tks = 0

    # Initialize / resume conversation
    if start_from_turn <= 1:
        texts: List[Any] = [_problem_greeting(data_type)]
        cur_start = 1
        logger.debug(f"[{pid}] starting fresh")
    else:
        texts = ["Hi there! Let's pick up the conversation."]
        cur_start = 1
        logger.warning(f"[{pid}] resume requested but no transcript found; starting fresh")

    pairs: List[Dict[str, Any]] = []
    chosen_list: List[Dict[str, Any]] = []

    for turn in range(cur_start, total_turns, 2):
        # ---- Student: sample K candidate questions
        student_replies, p_tks, c_tks = interact(
            True,
            texts,
            interaction_problem,
            num_candidates,
            data_type,
            student_client,
            student_name,
            student_temperature,
            use_scaffold
        )
        total_student_prompt_tks += p_tks
        total_student_completion_tks += c_tks

        if not student_replies:
            logger.warning(f"[{pid}] No student candidates at turn {turn}; skipping turn.")
            continue

        # ---- Judge picks best candidate (index)
        best_idx = judge_select_best_index(
            client=teacher_client,
            model_name=teacher_name,
            data_type=data_type,
            problem=evaluation_problem,
            history_texts=texts,
            candidates=student_replies,
            temperature=judge_temperature,
            extra_instructions=judge_extra_instructions,
        )

        # ---- Teacher responds ONLY to the appended branch (best or exploration)
        # Exploration vs exploitation (propensity)
        appended_idx = best_idx
        if random.random() < 1 - eps:
            propensity = 1 - eps
            chosen_student_msg = student_replies[best_idx]
        else:
            non_best = [i for i in range(num_candidates) if i != best_idx]
            idx = random.choice(non_best) if non_best else best_idx
            appended_idx = idx
            propensity = eps / max(1, num_candidates)
            chosen_student_msg = student_replies[idx]

        # Get teacher reply for the appended path
        teacher_reply, p_tks, c_tks = interact(
            False,
            texts + [chosen_student_msg],
            interaction_problem,
            1,
            data_type,
            teacher_client,
            teacher_name,
            teacher_temperature,
            use_scaffold
        )
        teacher_reply = teacher_reply[0]
        total_teacher_prompt_tks += p_tks
        total_teacher_completion_tks += c_tks

        # Roll conversation
        texts += [chosen_student_msg, teacher_reply]

        # ---- Prompt messages used to elicit candidate questions (for SFT/DPO storage)
        sys_prompt = _select_sys_prompt(data_type, use_scaffold, True, evaluation_problem)
        prompt_messages = [{"role": "system", "content": sys_prompt}] + format_messages(
            texts[:-2],  # messages BEFORE the chosen Q/A (same prompt format as the ODQS data)
            starting_with_user=True
        )

        # ---- SFT item: winner chosen by Judge
        chosen_for_sft = student_replies[best_idx]
        chosen_list.append({
            "prompt": prompt_messages,
            "response": chosen_for_sft,
            "problem": evaluation_problem,
            "turn": turn,
            "propensity": propensity
        })

        # ---- DPO pairs: winner vs everyone else (judge baseline)
        if judge_use_all_as_rejects:
            rejected_indices = [i for i in range(num_candidates) if i != best_idx]
        else:
            # Fallback: if someone wants fewer pairs, randomly pick 1 negative (simple and fast)
            rejected_indices = []
            non_best = [i for i in range(num_candidates) if i != best_idx]
            if non_best:
                rejected_indices.append(random.choice(non_best))

        for i in rejected_indices:
            pairs.append({
                "prompt": prompt_messages,
                "chosen_response": chosen_for_sft,     # student reply (winner)
                "rejected_response": student_replies[i],
                "chosen_reward": {},                   # kept for schema compatibility
                "rejected_reward": {},                 # no eval in this baseline
                "problem": evaluation_problem,
                "turn": turn
            })

        logger.debug(f'pid-{evaluation_problem["id"]} turn-{turn} judge_best_idx={best_idx}')

    return (
        chosen_list,
        pairs,
        total_teacher_prompt_tks, total_teacher_completion_tks,
        total_student_prompt_tks, total_student_completion_tks
    )

# =========================================================
# Hydra entry (Judge Baseline)
# =========================================================
@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    start_time = time.perf_counter()  # ---- start timer ----

    # set envs
    set_envs(cfg.paths.openai_api_key_fpath, cfg.paths.hf_home)

    # Output files (SFT, DPO) — no GRPO here
    base_dir = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        str(cfg.experiment.dataset)
    )
    name = cfg.data.name
    os.makedirs(os.path.join(base_dir, name), exist_ok=True)

    fpath_sft = os.path.join(
        base_dir, name,
        f"{cfg.student.model}_scaffolded_sft_iter{cfg.gen.iteration}_{cfg.data.split}.jsonl" if cfg.experiment.scaffolded else f"{cfg.student.model}_sft_iter{cfg.gen.iteration}_{cfg.data.split}.jsonl"
    )
    fpath_dpo = os.path.join(
        base_dir, name,
        f"{cfg.student.model}_scaffolded_dpo_iter{cfg.gen.iteration}_{cfg.data.split}.jsonl" if cfg.experiment.scaffolded else f"{cfg.student.model}_dpo_iter{cfg.gen.iteration}_{cfg.data.split}.jsonl"
    )

    existing_pids = set()
    count_dpo, count_sft = 0, 0

    # Resume dedupe by iteration for SFT/DPO
    if os.path.exists(fpath_sft):
        with open(fpath_sft, "r") as f:
            for line in f:
                ex = json.loads(line)
                count_sft += 1
                if ex.get("iteration") == cfg.gen.iteration:
                    existing_pids.add(ex["problem"]["id"])
        logger.info(f"Loaded {count_sft} existing SFT-{cfg.gen.iteration} examples from {fpath_sft}")

    if os.path.exists(fpath_dpo):
        with open(fpath_dpo, "r") as f:
            for line in f:
                ex = json.loads(line)
                count_dpo += 1
                if ex.get("iteration") == cfg.gen.iteration:
                    existing_pids.add(ex["problem"]["id"])
        logger.info(f"Loaded {count_dpo} existing DPO-{cfg.gen.iteration} examples from {fpath_dpo}")

    # Problems
    problems_fpath = os.path.join(
        base_dir,
        f"splits_{cfg.gen.pass_at_k_threshold}",
        f"{cfg.student.model}_{cfg.data.split}.json",
    )
    logger.info(f"Loading interaction problems: {problems_fpath}")
    data = load_json_resilient(problems_fpath)
    interaction_problems: List[Dict[str, Any]] = [
        p for p in data if p["id"] not in existing_pids and str(p["id"]) not in existing_pids
    ][: max(0, cfg.data.data_amount - len(existing_pids))]
    logger.info(f"Interaction problems: finished {len(existing_pids)} | pending {len(interaction_problems)}")

    # OOD evaluation set (unused by the judge)
    evaluation_problems = interaction_problems
    if getattr(cfg.experiment, "ood", False):
        ood_problems_path = os.path.join(
            base_dir,
            f"splits_{cfg.gen.pass_at_k_threshold}",
            f"{cfg.student.model}_ood_{cfg.data.split}.json",
        )
        logger.info(f"Loading evaluation (OOD) problems: {ood_problems_path}")
        data = load_json_resilient(ood_problems_path)
        evaluation_problems = [
            p for p in data if p["id"] not in existing_pids and str(p["id"]) not in existing_pids
        ][: max(0, cfg.data.data_amount - len(existing_pids))]
        logger.info(f"Evaluation (OOD) problems: finished {len(existing_pids)} | pending {len(evaluation_problems)}")

    # Clients
    teacher_name, student_name = cfg.teacher.model, cfg.student.model
    teacher_client = build_client(
        teacher_name, cfg.teacher.gpus, cfg.teacher.server
    )
    student_client = build_client(
        student_name, cfg.student.gpus, cfg.student.server, lora=getattr(cfg.student, "lora", None)
    )
    logger.info("Teacher=%s | Student=%s", teacher_name, student_name)

    total_teacher_prompt_tks = total_teacher_completion_tks = 0
    total_student_prompt_tks = total_student_completion_tks = 0

    # Files
    f_dpo = open(fpath_dpo, "a" if os.path.exists(fpath_dpo) else "w")
    f_sft = open(fpath_sft, "a" if os.path.exists(fpath_sft) else "w")

    # Judge knobs (Hydra)
    judge_temperature = float(getattr(cfg.gen, "judge_temperature", 0.0))
    judge_extra_instructions = str(getattr(cfg.gen, "judge_extra_instructions", ""))
    judge_use_all_as_rejects = bool(getattr(cfg.gen, "judge_use_all_as_rejects", True))

    pbar = tqdm(total=len(interaction_problems), desc="Problems done", dynamic_ncols=True)

    # Parallel run
    with ThreadPoolExecutor(max_workers=cfg.experiment.num_workers) as executor:
        for i in range(0, len(interaction_problems), cfg.experiment.chunk_size):
            # Problem chunks in this iter
            interaction_problems_chunk = interaction_problems[i: i + cfg.experiment.chunk_size]
            evaluation_problems_chunk = evaluation_problems[i: i + cfg.experiment.chunk_size]

            # Prepare jobs
            futures = []
            for interaction_problem, evaluation_problem in zip(interaction_problems_chunk, evaluation_problems_chunk):
                futures.append(executor.submit(
                    run_problem_judge,
                    cfg.data.type,
                    teacher_client,
                    teacher_name,
                    cfg.teacher.temperature,
                    student_client,
                    cfg.student.lora.name if getattr(cfg.student, "lora", None) and getattr(cfg.student.lora, "name", None) else student_name,
                    cfg.student.temperature,
                    interaction_problem,
                    evaluation_problem,
                    # generation knobs
                    cfg.gen.num_candidates,
                    cfg.experiment.start_from_turn,
                    cfg.experiment.total_turns,
                    cfg.gen.eps,
                    cfg.experiment.scaffolded,
                    # judge knobs
                    judge_temperature,
                    judge_extra_instructions,
                    judge_use_all_as_rejects,
                ))

            for future in as_completed(futures):
                (
                    chosen,
                    pairs,
                    teacher_prompt_tks,
                    teacher_completion_tks,
                    student_prompt_tks,
                    student_completion_tks
                ) = future.result()

                # Save DPO
                for ex in pairs:
                    ex["iteration"] = cfg.gen.iteration
                    f_dpo.write(json.dumps(ex, ensure_ascii=False) + "\n")
                count_dpo += len(pairs)

                # Save SFT
                for ex in chosen:
                    ex["iteration"] = cfg.gen.iteration
                    f_sft.write(json.dumps(ex, ensure_ascii=False) + "\n")
                count_sft += len(chosen)

                # Token usage
                total_teacher_prompt_tks += teacher_prompt_tks
                total_teacher_completion_tks += teacher_completion_tks
                total_student_prompt_tks += student_prompt_tks
                total_student_completion_tks += student_completion_tks

                pbar.update(1)
                logger.info(f"No. of examples collected DPO: {count_dpo} | SFT: {count_sft}")
                logger.info("%s", pbar.format_meter(n=pbar.n, total=pbar.total, elapsed=pbar.format_dict["elapsed"], prefix=pbar.desc + ": "))

    # Close files
    f_dpo.close()
    f_sft.close()
    pbar.close()

    # Log summary
    logger.info(f"------- {cfg.data.split} -------")
    logger.info(f"Saved {count_dpo} preference pairs to {fpath_dpo}")
    logger.info(f"Saved {count_sft} SFT examples to {fpath_sft}")
    logger.info(
        f"{cfg.teacher.model} token usage: prompt={total_teacher_prompt_tks}, completion={total_teacher_completion_tks}"
    )
    logger.info(
        f"{cfg.student.model} token usage: prompt={total_student_prompt_tks}, completion={total_student_completion_tks}"
    )

    elapsed = time.perf_counter() - start_time
    logger.info("Total time elapsed: %s", timedelta(seconds=int(elapsed)))


if __name__ == "__main__":
    main()
