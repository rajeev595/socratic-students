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
from codes.run_interact import interact, evaluate, get_assessment
from codes.utils.llm.server import build_client
from codes.utils.vars import set_envs
from codes.utils.file_ops import save_json_atomic, load_json_resilient


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)


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


def run_problem(
    data_type: str,
    teacher_client: OpenAI,
    teacher_name: str,
    teacher_temperature: float,
    student_client: OpenAI,
    student_name: str,
    student_temperature: float,
    interaction_problem: Dict[str, Any],
    evaluation_problem: Dict[str, Any],
    k_values: List[int],
    pass_k: int,
    num_samples: int,
    rlhf_samples: int,
    assessment_turn: int,
    start_from_turn: int = 1,
    total_turns: int = 10,
    eps: float = 0.2,
    use_cot: bool = False,
    is_ood: bool = False,
) -> Tuple[
    List[Dict[str, Any]],   # chosen_list (SFT)
    List[Dict[str, Any]],   # pairs (DPO)
    List[Dict[str, Any]],   # grpo_groups  <-- NEW
    int, int, int, int
]:
    """
    Run a single problem end-to-end: student↔teacher interaction and inline evaluation.
    Returns:
      - chosen_list: SFT items (unchanged)
      - pairs: DPO pairs (unchanged)
      - grpo_groups: list of grouped examples for GRPO/RRHF
      - token usage tallies
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

    pairs, chosen_list, grpo_groups = [], [], []
    pass_k_key = str(pass_k)  # keys in your pass@k dicts are strings

    for turn in range(cur_start, total_turns, 2):
        # Optional assessment injected into previous text
        if assessment_turn > 0 and turn == assessment_turn and len(texts) > 0:
            texts[-1] += get_assessment(
                data_type=data_type,
                client=student_client,
                model_name=student_name,
                student_temperature=student_temperature,
                problem_dict=interaction_problem,
                texts=texts,
            )
            logger.debug(f"[{pid}] assessment injected at turn {turn}")

        # Run the student: sample K=rlhf_samples candidate questions
        student_replies, p_tks, c_tks = interact(
            True,
            texts,
            interaction_problem,
            rlhf_samples,
            data_type,
            student_client,
            student_name,
            student_temperature,
            use_cot
        )
        total_student_prompt_tks += p_tks
        total_student_completion_tks += c_tks

        # Run the teacher for each candidate
        teacher_replies = []
        for student_reply in student_replies:
            teacher_reply, p_tks, c_tks = interact(
                False,
                texts + [student_reply],
                interaction_problem,
                1,
                data_type,
                teacher_client,
                teacher_name,
                teacher_temperature,
                use_cot
            )
            teacher_replies.append(teacher_reply[0])
            total_teacher_prompt_tks += p_tks
            total_teacher_completion_tks += c_tks

        # Evaluate each (student question, teacher answer) pair
        pass_at_k_scores = []
        for student_reply, teacher_reply in zip(student_replies, teacher_replies):
            prev_convo = msgs_to_string(texts + [student_reply, teacher_reply])
            pass_at_k_scores_sample, _, p_tks, c_tks = evaluate(
                prev_convo=prev_convo,
                data_type=data_type,
                student_client=student_client,
                student_name=student_name,
                student_temperature=0.3,
                teacher_client=teacher_client,
                teacher_name=teacher_name,
                teacher_temperature=teacher_temperature,
                problem=evaluation_problem,
                k_values=k_values,
                num_samples=num_samples,
                is_ood=is_ood,
            )
            pass_at_k_scores.append(pass_at_k_scores_sample)
            total_student_prompt_tks += p_tks
            total_student_completion_tks += c_tks

        # Build prompt messages used to elicit candidate questions
        sys_prompt = _select_sys_prompt(data_type, use_cot, True, evaluation_problem)
        prompt_messages = [{"role": "system", "content": sys_prompt}] + format_messages(
            texts, starting_with_user=True
        )

        # ---------- NEW: collect GRPO/RRHF grouped data for this turn ----------
        # We do NOT apply any threshold here; we keep the full K-set to avoid bias.
        group = {
            "ex_id": pid,
            "turn": turn,
            "prompt_messages": prompt_messages,
            "prompt_type": "messages",
            "k": rlhf_samples,
            "pass_k_key": pass_k,  # stored as int for clarity
            "candidates": [],
        }
        for reply, score_dict in zip(student_replies, pass_at_k_scores):
            q_text = _extract_text(reply)
            # Ensure numeric rewards (prefer the chosen pass@k key; fallback to max available)
            if str(pass_k) in score_dict:
                reward = float(score_dict[str(pass_k)])
            else:
                # fallback: use the largest available k
                if len(score_dict) == 0:
                    reward = 0.0
                else:
                    best_k = max(map(int, score_dict.keys()))
                    reward = float(score_dict[str(best_k)])
            group["candidates"].append({
                "text": q_text,
                "reward": reward,
                "metrics": {k: float(v) for k, v in score_dict.items()},
            })
        grpo_groups.append(group)
        # ----------------------------------------------------------------------

        # Existing selection logic for continuing the dialogue
        # best reply by highest pass@k
        # (we still use this to advance the conversation for future turns)
        max_value = max([s.get(pass_k_key, 0.0) for s in pass_at_k_scores])
        best_idx = [s.get(pass_k_key, 0.0) for s in pass_at_k_scores].index(max_value)
        best_pass_at_k_score = pass_at_k_scores[best_idx]

        # Exploration vs exploitation to advance convo context
        if random.random() < 1 - eps:
            texts += [student_replies[best_idx], teacher_replies[best_idx]]
            propensity = 1 - eps
        else:
            non_best = [i for i in range(rlhf_samples) if i != best_idx]
            idx = random.choice(non_best)
            texts += [student_replies[idx], teacher_replies[idx]]
            propensity = eps / max(1, rlhf_samples - 1)

        # SFT/DPO collection (unchanged) — you may keep or remove if you only want GRPO
        if max_value >= 0.85:
            chosen = student_replies[best_idx]
            chosen_list.append({
                "prompt": prompt_messages,
                "response": chosen,
                "problem": evaluation_problem,
                "turn": turn,
                "propensity": propensity
            })

            for score, reply in zip(pass_at_k_scores, student_replies):
                if best_pass_at_k_score.get(pass_k_key, 0.0) - score.get(pass_k_key, 0.0) >= 0.05 and reply != chosen:
                    pairs.append({
                        "prompt": prompt_messages,
                        "chosen_response": chosen,
                        "rejected_response": reply,
                        "chosen_reward": best_pass_at_k_score,
                        "rejected_reward": score,
                        "problem": evaluation_problem,
                        "turn": turn
                    })

        logger.debug(f'pid-{evaluation_problem["id"]} turn-{turn} pass@k={best_pass_at_k_score}')

    return (
        chosen_list,
        pairs,
        grpo_groups,  # NEW
        total_teacher_prompt_tks,
        total_teacher_completion_tks,
        total_student_prompt_tks,
        total_student_completion_tks
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    start_time = time.perf_counter()  # ---- start timer ----

    # set the OpenAI key env variable, HF HOME
    set_envs(cfg.paths.openai_api_key_fpath, cfg.paths.hf_home)

    # Output files (SFT, DPO, GRPO)
    train_fpath_sft = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        str(cfg.experiment.dataset),
        f"{cfg.student.model}_cot_sft_iter{cfg.gen.dagger_iter}.jsonl" if cfg.experiment.cot else f"{cfg.student.model}_sft_iter{cfg.gen.dagger_iter}.jsonl"
    )
    train_fpath_rlhf = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        str(cfg.experiment.dataset),
        f"{cfg.student.model}_cot_rlhf_iter{cfg.gen.dagger_iter}.jsonl" if cfg.experiment.cot else f"{cfg.student.model}_rlhf_iter{cfg.gen.dagger_iter}.jsonl"
    )
    train_fpath_grpo = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        str(cfg.experiment.dataset),
        f"{cfg.student.model}_cot_grpo_iter{cfg.gen.dagger_iter}.jsonl" if cfg.experiment.cot else f"{cfg.student.model}_grpo_iter{cfg.gen.dagger_iter}.jsonl"
    )

    existing_pids = set()
    count_rlhf, count_sft, count_grpo = 0, 0, 0

    # Resume dedupe by dagger_iter for SFT/DPO (unchanged)
    if os.path.exists(train_fpath_sft):
        with open(train_fpath_sft, "r") as f:
            for line in f:
                ex = json.loads(line)
                if ex.get("dagger_iter") == cfg.gen.dagger_iter:
                    existing_pids.add(ex["problem"]["id"])
        logger.info(f"Loaded {len(existing_pids)} existing SFT-{cfg.gen.dagger_iter} examples from {train_fpath_sft}")

    if os.path.exists(train_fpath_rlhf):
        with open(train_fpath_rlhf, "r") as f:
            for line in f:
                ex = json.loads(line)
                if ex.get("dagger_iter") == cfg.gen.dagger_iter:
                    existing_pids.add(ex["problem"]["id"])

    # Problems
    problems_fpath = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        str(cfg.experiment.dataset),
        f"{cfg.student.model}_train.json",
    )
    logger.info(f"Loading interaction problems: {problems_fpath}")
    data = load_json_resilient(problems_fpath)
    interaction_problems: List[Dict[str, Any]] = [
        p for p in data if p["id"] not in existing_pids
    ][: cfg.data.data_amount]
    logger.info(f"Total interaction problems: {len(interaction_problems)}")

    # OOD problems
    evaluation_problems = interaction_problems
    if cfg.experiment.ood:
        ood_problems_path = os.path.join(
            str(cfg.paths.data),
            str(cfg.teacher.model),
            str(cfg.experiment.dataset),
            f"{cfg.student.model}_ood_train.json",
        )
        logger.info(f"Loading evaluation (OOD) problems: {ood_problems_path}")
        data = load_json_resilient(ood_problems_path)
        evaluation_problems = [
            p for p in data if p["id"] not in existing_pids
        ][: cfg.data.data_amount]
        logger.info(f"Total evaluation (OOD) problems: {len(evaluation_problems)}")

    # Build teacher and student clients
    teacher_name, student_name = cfg.teacher.model, cfg.student.model
    teacher_client = build_client(teacher_name, cfg.teacher.gpus, cfg.teacher.server)
    student_client = build_client(student_name, cfg.student.gpus, cfg.student.server)
    logger.info("Teacher=%s | Student=%s", teacher_name, student_name)

    total_teacher_prompt_tks = total_teacher_completion_tks = 0
    total_student_prompt_tks = total_student_completion_tks = 0

    # Open files
    f_rlhf = open(train_fpath_rlhf, "a" if os.path.exists(train_fpath_rlhf) else "w")
    f_sft = open(train_fpath_sft, "a" if os.path.exists(train_fpath_sft) else "w")
    f_grpo = open(train_fpath_grpo, "a" if os.path.exists(train_fpath_grpo) else "w")

    # Build everything in parallel
    with ThreadPoolExecutor(max_workers=cfg.experiment.num_workers) as executor:
        futures = []
        for interaction_problem, evaluation_problem in zip(interaction_problems, evaluation_problems):
            futures.append(executor.submit(
                run_problem,
                cfg.data.type,
                teacher_client,
                teacher_name,
                cfg.teacher.temperature,
                student_client,
                student_name,
                cfg.student.temperature,
                interaction_problem,
                evaluation_problem,
                cfg.gen.k_values,
                cfg.gen.pass_k,
                cfg.gen.num_samples,
                cfg.gen.rlhf_samples,
                cfg.experiment.assessment_turn,
                cfg.experiment.start_from_turn,
                cfg.experiment.total_turns,
                cfg.gen.eps,
                cfg.experiment.cot,
                cfg.experiment.ood
            ))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing problems"):
            (
                chosen,
                pairs,
                grpo_groups,  # NEW
                teacher_prompt_tks,
                teacher_completion_tks,
                student_prompt_tks,
                student_completion_tks
            ) = future.result()

            # Save RLHF jsonl (pairs) incrementally (unchanged)
            for ex in pairs:
                ex["dagger_iter"] = cfg.gen.dagger_iter
                f_rlhf.write(json.dumps(ex, ensure_ascii=False) + "\n")
            count_rlhf += len(pairs)

            # Save SFT jsonl incrementally (unchanged)
            for ex in chosen:
                ex["dagger_iter"] = cfg.gen.dagger_iter
                f_sft.write(json.dumps(ex, ensure_ascii=False) + "\n")
            count_sft += len(chosen)

            # Save GRPO groups incrementally (NEW)
            for group in grpo_groups:
                group["dagger_iter"] = cfg.gen.dagger_iter
                f_grpo.write(json.dumps(group, ensure_ascii=False) + "\n")
            count_grpo += len(grpo_groups)

            # Token usage
            total_teacher_prompt_tks += teacher_prompt_tks
            total_teacher_completion_tks += teacher_completion_tks
            total_student_prompt_tks += student_prompt_tks
            total_student_completion_tks += student_completion_tks

    # Close files
    f_rlhf.close()
    f_sft.close()
    f_grpo.close()

    # Log summary
    logger.info(f"Saved {count_rlhf} preference pairs to {train_fpath_rlhf}")
    logger.info(f"Saved {count_sft} SFT examples to {train_fpath_sft}")
    logger.info(f"Saved {count_grpo} GRPO groups to {train_fpath_grpo}")
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
