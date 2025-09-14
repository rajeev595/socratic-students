import os
import json
import logging
from typing import Any, List, Dict, Tuple

import hydra
from omegaconf import DictConfig
from openai import OpenAI

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from codes.data.utils import format_messages, msgs_to_string
from codes.prompts.utils import _select_sys_prompt, _problem_greeting
from codes.run_interact import interact, evaluate, get_assessment
from codes.llm_utils import set_openai_key_env, build_client


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
    assessment_turn: int,
    start_from_turn: int = 1,
    total_turns: int = 10,
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

    pairs = []
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

        # Run the student
        student_replies, p_tks, c_tks = interact(
            True,
            texts,
            interaction_problem,
            num_samples,
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
                num_samples,
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
                prev_convo,
                data_type,
                student_client,
                student_name,
                student_temperature,
                evaluation_problem,
                k_values,
                num_samples,
                is_ood
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
        if max_value == 0:
            continue

        # Append chosen and rejected pairs
        chosen = student_replies[best_idx]
        for score, reply in zip(pass_at_k_scores, student_replies):
            if score[pass_k] < best_pass_at_k_score[pass_k] and reply != chosen:
                pairs.append({
                    "prompt": prompt,
                    "chosen_response": chosen,
                    "rejected_response": reply,
                    "chosen_reward": best_pass_at_k_score,
                    "rejected_reward": score,
                    "problem": evaluation_problem,
                    "turn": turn
                })

        # Update the chat with the best reply for next turns
        texts += [student_replies[best_idx], teacher_replies[best_idx]]

        logger.debug(f"pid-{evaluation_problem["id"]} turn-{turn} pass@k={best_pass_at_k_score}")

    return (
        pairs,
        total_teacher_prompt_tks,
        total_teacher_completion_tks,
        total_student_prompt_tks,
        total_student_completion_tks
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    # set the OpenAI key env variable
    set_openai_key_env(cfg.paths.openai_api_key_fpath)

    # Check if data exists already and load it
    train_fpath_sft = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        str(cfg.experiment.dataset),
        f"{cfg.data.filename}_sft.jsonl"
    )
    train_fpath_rlhf = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        str(cfg.experiment.dataset),
        f"{cfg.data.filename}_rlhf.jsonl"
    )

    existing_pids = set()

    if os.path.exists(train_fpath_sft):
        with open(train_fpath_sft, "r") as f:
            for line in f:
                ex = json.loads(line)
                existing_pids.add(ex["problem"]["id"])

        logger.info(f"Loaded {len(existing_pids)} existing SFT examples from {train_fpath_sft}")

    if os.path.exists(train_fpath_rlhf):
        with open(train_fpath_rlhf, "r") as f:
            for line in f:
                ex = json.loads(line)
                existing_pids.add(ex["problem"]["id"])

        logger.info(f"Loaded {len(existing_pids)} existing RLHF examples from {train_fpath_rlhf}")

    # Load problems
    problems_fpath = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        str(cfg.experiment.dataset),
        f"{cfg.data.filename}.json",
    )
    logger.info(f"Loading interaction problems: {problems_fpath}")
    with open(problems_fpath, "r") as f:
        data = json.load(f)["collected"][100:]  
    interaction_problems: List[Dict[str, Any]] = [
        p for p in data if p["id"] not in existing_pids
    ]
    interaction_problems = sorted(interaction_problems, key=lambda x: x["id"])[: cfg.data.data_amount]
    logger.info(f"Total interaction problems: {len(interaction_problems)}")

    # OOD problems
    evaluation_problems = interaction_problems
    if cfg.experiment.ood:
        ood_problems_path = os.path.join(
            str(cfg.paths.data),
            str(cfg.teacher.model),
            str(cfg.experiment.dataset),
            f"{cfg.data.filename}_ood.json",
        )

        logger.info(f"Loading evaluation (OOD) problems: {ood_problems_path}")
        with open(ood_problems_path, "r") as f:
            data = json.load(f)[100:]
            evaluation_problems = [p for p in data if p["id"] not in existing_pids]
        evaluation_problems = sorted(evaluation_problems, key=lambda x: x["id"])[: cfg.data.data_amount]
        logger.info(f"Total evaluation (OOD) problems: {len(evaluation_problems)}")

    # Build clients
    teacher_name, teacher_client = build_client(cfg.teacher.model)
    student_name, student_client = build_client(cfg.student.model)

    total_teacher_prompt_tks = total_teacher_completion_tks = 0
    total_student_prompt_tks = total_student_completion_tks = 0

    f_rlhf = open(train_fpath_rlhf, "a" if os.path.exists(train_fpath_rlhf) else "w")
    f_sft = open(train_fpath_sft, "a" if os.path.exists(train_fpath_sft) else "w")

    count_rlhf, count_sft = 0, 0
    # Build pairs
    with ThreadPoolExecutor(max_workers=cfg.experiment.num_workers) as executor:
        # collect all the jobs
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
                cfg.experiment.assessment_turn,
                cfg.experiment.start_from_turn,
                cfg.experiment.total_turns,
                cfg.experiment.cot,
                cfg.experiment.ood
            ))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing problems"):
            (
                pairs, 
                teacher_prompt_tks, 
                teacher_completion_tks, 
                student_prompt_tks, 
                student_completion_tks
            ) = future.result()

            # Save RLHF jsonl (or json) incrementally
            count_rlhf += len(pairs)
            for ex in pairs:
                f_rlhf.write(json.dumps(ex, ensure_ascii=False) + "\n")
            
            # Save SFT jsonl (or json) incrementally
            if pairs:
                data = {
                    "prompt": pairs[0]["prompt"],
                    "response": pairs[0]["chosen_response"],
                    "problem": pairs[0]["problem"],
                    "turn": pairs[0]["turn"]
                }
                count_sft += 1
                f_sft.write(json.dumps(data, ensure_ascii=False) + "\n")

            total_teacher_prompt_tks += teacher_prompt_tks
            total_teacher_completion_tks += teacher_completion_tks
            total_student_prompt_tks += student_prompt_tks
            total_student_completion_tks += student_completion_tks

    # Close files
    f_rlhf.close()
    f_sft.close()

    # Log summary
    logger.info(f"Saved {count_rlhf} preference pairs to {train_fpath_rlhf}")
    logger.info(f"Saved {count_sft} sft examples to {train_fpath_sft}")
    logger.info(
        f"{cfg.teacher.model} token usage: prompt={total_teacher_prompt_tks}, completion={total_teacher_completion_tks}"
    )
    logger.info(
        f"{cfg.student.model} token usage: prompt={total_student_prompt_tks}, completion={total_student_completion_tks}"
    )


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
