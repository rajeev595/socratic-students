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
    eps: float=0.2,
    use_cot: bool = False,
    is_ood: bool = False,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[int, List[Dict[int, float]]],
    List[Dict[int, float]],
    int, int, int, int
]:
    """
    Run a single problem end-to-end: student↔teacher interaction and inline evaluation.
    Returns:
      - edited_msgs: message dicts with roles and turns
      - all_results: turn -> list of pass@k dicts (to aggregate later)
      - performance_tracker: per-student-turn pass@k dicts in sequence
      - total_teacher_prompt_tks, total_teacher_completion_tks
      - total_student_prompt_tks, total_student_completion_tks
    """
    assert interaction_problem["id"] == evaluation_problem["id"], "pid of interaction and evaluation doesn't match!"

    pid = interaction_problem["id"]
    total_teacher_prompt_tks = total_teacher_completion_tks = 0
    total_student_prompt_tks = total_student_completion_tks = 0

    # Initialize / resume conversation
    if start_from_turn <= 1:
        texts: List[str] = [_problem_greeting(data_type)]
        cur_start = 1
        logger.debug(f"[{pid}] starting fresh")
    else:
        texts = ["Hi there! Let's pick up the conversation."]
        cur_start = 1
        logger.warning(f"[{pid}] resume requested but no transcript found; starting fresh")

    pairs, chosen_list = [], []
    for turn in range(cur_start, total_turns, 2):
        # Optional assessment injected into previous text
        if assessment_turn > 0 and turn == assessment_turn and len(texts) > 0:
            texts[-1] += get_assessment(
                data_type=data_type,
                student_client=student_client,
                student_name=student_name,
                student_temperature=student_temperature,
                teacher_client=teacher_client,
                teacher_name=teacher_name,
                teacher_temperature=teacher_temperature,
                problem_dict=interaction_problem,
                texts=texts,
            )
            logger.debug(f"[{pid}] assessment injected at turn {turn}")

        # Run the student
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

        # Run the teacher
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

        # Evaluation on student turns
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

            # Update pass@k scores and token counts
            pass_at_k_scores.append(pass_at_k_scores_sample)
            total_student_prompt_tks += p_tks
            total_student_completion_tks += c_tks

        # Prepare data for training
        sys_prompt = _select_sys_prompt(data_type, use_cot, True, evaluation_problem)
        prompt = [{"role": "system", "content": sys_prompt}] + format_messages(
            texts, starting_with_user=True
        )

        # best reply by highest pass@k
        max_value = max([s[pass_k] for s in pass_at_k_scores])
        best_idx = [s[pass_k] for s in pass_at_k_scores].index(max_value)
        best_pass_at_k_score = pass_at_k_scores[best_idx]

        # Update the chat with the best reply for next turns
        if random.random() < 1 - eps:
            texts += [student_replies[best_idx], teacher_replies[best_idx]]
            propensity = 1 - eps
        else:
            non_best = [i for i in range(rlhf_samples) if i != best_idx]
            idx = random.choice(non_best)
            texts += [student_replies[idx], teacher_replies[idx]]
            propensity = eps / rlhf_samples

        if max_value < 0.85:
            continue

        # best question is chosen
        chosen = student_replies[best_idx]

        # Don't train SFT on sub-optimal trajectories
        chosen_list.append({
            "prompt": prompt,
            "response": chosen,
            "problem": evaluation_problem,
            "turn": turn,
            "propensity": propensity
        })

        # Append chosen and rejected pairs for DPO
        for score, reply in zip(pass_at_k_scores, student_replies):
            if best_pass_at_k_score[pass_k] - score[pass_k] >= 0.05 and reply != chosen:
                pairs.append({
                    "prompt": prompt,
                    "chosen_response": chosen,
                    "rejected_response": reply,
                    "chosen_reward": best_pass_at_k_score,
                    "rejected_reward": score,
                    "problem": evaluation_problem,
                    "turn": turn
                })

        logger.debug(f"pid-{evaluation_problem["id"]} turn-{turn} pass@k={best_pass_at_k_score}")

    return (
        chosen_list,
        pairs,
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

    # Check if data exists already and load it
    fpath_sft = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        str(cfg.experiment.dataset),
        f"{cfg.student.model}_cot_sft_iter{cfg.gen.dagger_iter}_{cfg.data.split}.jsonl" if cfg.experiment.cot else f"{cfg.student.model}_sft_iter{cfg.gen.dagger_iter}_{cfg.data.split}.jsonl"
    )
    fpath_rlhf = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        str(cfg.experiment.dataset),
        f"{cfg.student.model}_cot_rlhf_iter{cfg.gen.dagger_iter}_{cfg.data.split}.jsonl" if cfg.experiment.cot else f"{cfg.student.model}_rlhf_iter{cfg.gen.dagger_iter}_{cfg.data.split}.jsonl"
    )

    existing_pids = set()
    count_rlhf, count_sft = 0, 0
    if os.path.exists(fpath_sft):
        with open(fpath_sft, "r") as f:
            for line in f:
                ex = json.loads(line)
                count_sft += 1
                if ex["dagger_iter"] == cfg.gen.dagger_iter:
                    existing_pids.add(ex["problem"]["id"])
        logger.info(f"Loaded {count_sft} existing SFT-{cfg.gen.dagger_iter} examples from {fpath_sft}")

    if os.path.exists(fpath_rlhf):
        with open(fpath_rlhf, "r") as f:
            for line in f:
                ex = json.loads(line)
                count_rlhf += 1
                if ex["dagger_iter"] == cfg.gen.dagger_iter:
                    existing_pids.add(ex["problem"]["id"])
        logger.info(f"Loaded {count_rlhf} existing RLHF-{cfg.gen.dagger_iter} examples from {fpath_rlhf}")

    # Load problems
    problems_fpath = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        str(cfg.experiment.dataset),
        f"{cfg.student.model}_{cfg.data.split}.json",
    )
    logger.info(f"Loading interaction problems: {problems_fpath}")
    data = load_json_resilient(problems_fpath)
    interaction_problems: List[Dict[str, Any]] = [
        p for p in data if p["id"] not in existing_pids and str(p["id"]) not in existing_pids
    ][: max(0, cfg.data.data_amount - len(existing_pids))]
    logger.info(f"Interaction problems: finished {len(existing_pids)} | pending {len(interaction_problems)}")

    # OOD problems
    evaluation_problems = interaction_problems
    if cfg.experiment.ood:
        ood_problems_path = os.path.join(
            str(cfg.paths.data),
            str(cfg.teacher.model),
            str(cfg.experiment.dataset),
            f"{cfg.student.model}_ood_{cfg.data.split}.json",
        )

        logger.info(f"Loading evaluation (OOD) problems: {ood_problems_path}")
        data = load_json_resilient(ood_problems_path)
        evaluation_problems = [
            p for p in data if p["id"] not in existing_pids
        ][: max(0, cfg.data.data_amount - len(existing_pids))]
        logger.info(f"Evaluation (OOD) problems: finished {len(existing_pids)} | pending {len(evaluation_problems)}")

    # Build teacher and student clients (names may be normalized inside)
    teacher_name, student_name = cfg.teacher.model, cfg.student.model
    teacher_client = build_client(
        teacher_name, cfg.teacher.gpus, cfg.teacher.server
    )
    student_client = build_client(
        student_name, cfg.student.gpus, cfg.student.server
    )
    logger.info("Teacher=%s | Student=%s", teacher_name, student_name)

    total_teacher_prompt_tks = total_teacher_completion_tks = 0
    total_student_prompt_tks = total_student_completion_tks = 0

    f_rlhf = open(fpath_rlhf, "a" if os.path.exists(fpath_rlhf) else "w")
    f_sft = open(fpath_sft, "a" if os.path.exists(fpath_sft) else "w")
    pbar = tqdm(total=len(interaction_problems), desc="Problems done", dynamic_ncols=True)

    # Parallel execution
    with ThreadPoolExecutor(max_workers=cfg.experiment.num_workers) as executor:
        for i in range(0, len(interaction_problems), cfg.experiment.chunk_size):
            # Problem chunks in this iter
            interaction_problems_chunk = interaction_problems[i: i + cfg.experiment.chunk_size]
            evaluation_problems_chunk = evaluation_problems[i: i + cfg.experiment.chunk_size]

            # Prepare jobs
            futures = []
            for interaction_problem, evaluation_problem in zip(interaction_problems_chunk, evaluation_problems_chunk):
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

            for future in as_completed(futures):
                (
                    chosen,
                    pairs,
                    teacher_prompt_tks, 
                    teacher_completion_tks, 
                    student_prompt_tks, 
                    student_completion_tks
                ) = future.result()

                # Save RLHF jsonl (or json) incrementally
                count_rlhf += len(pairs)
                for ex in pairs:
                    ex["dagger_iter"] = cfg.gen.dagger_iter
                    f_rlhf.write(json.dumps(ex, ensure_ascii=False) + "\n")

                # Save SFT jsonl (or json) incrementally
                count_sft += len(chosen)
                for ex in chosen:
                    ex["dagger_iter"] = cfg.gen.dagger_iter
                    f_sft.write(json.dumps(ex, ensure_ascii=False) + "\n")

                total_teacher_prompt_tks += teacher_prompt_tks
                total_teacher_completion_tks += teacher_completion_tks
                total_student_prompt_tks += student_prompt_tks
                total_student_completion_tks += student_completion_tks
                pbar.update(1)
                logger.info(f"No. of examples collected DPO: {count_rlhf} | SFT: {count_sft}")
                logger.info("%s", pbar.format_meter(n=pbar.n, total=pbar.total, elapsed=pbar.format_dict["elapsed"], prefix=pbar.desc + ": "))

    # Close files
    f_rlhf.close()
    f_sft.close()
    pbar.close()

    # Log summary
    logger.info("Total problems done: %d/%d", pbar.n, len(evaluation_problems))    
    logger.info(f"Saved {count_rlhf} preference pairs to {fpath_rlhf}")
    logger.info(f"Saved {count_sft} sft examples to {fpath_sft}")
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
