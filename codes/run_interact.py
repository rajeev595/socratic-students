# interact_eval.py
# Combined runner: INTERACT ↔ EVALUATE (inline, per student turn)
# Hydra-based, matching your repo-2 style.

import os
import time
import logging
from datetime import timedelta
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import json
import numpy as np
from tqdm import tqdm
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from openai import OpenAI

from codes.llm_utils import build_client, chat_complete
from codes.data.utils import remove_think_tags, msgs_to_string, format_messages
from codes.prompts.code import (
    get_humaneval_student_sys_prompt,
    get_humaneval_student_sys_cot_prompt,
    get_humaneval_teacher_sys_prompt,
)
from codes.prompts.math import (
    get_gsm_student_sys_prompt,
    get_gsm_student_sys_cot_prompt,
    get_gsm_teacher_sys_prompt,
)

from codes.evaluation.code import generate_code_completions, evaluate_code_completion
from codes.evaluation.math import generate_math_completions, evaluate_math_completion
from codes.evaluation.utils import compute_pass_at_k


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)


# ---------- helpers ----------
def _select_sys_prompt(dataset: str, use_cot: bool, is_student_turn: bool, problem: Dict[str, Any]) -> str:
    """Choose the correct system prompt function and embed the problem text (already handled in your prompt fns)."""
    if dataset == "code":
        if is_student_turn:
            return (get_humaneval_student_sys_cot_prompt if use_cot else get_humaneval_student_sys_prompt)(problem)
        else:
            return get_humaneval_teacher_sys_prompt(problem)
    else:
        if is_student_turn:
            return (get_gsm_student_sys_cot_prompt if use_cot else get_gsm_student_sys_prompt)(problem)
        else:
            return get_gsm_teacher_sys_prompt(problem)


def _problem_greeting(dataset: str) -> str:
    """First message content (index 0 of the text timeline)."""
    return (
        "Hi there! I'm your coding tutor. How can I help you today?"
        if dataset == "code"
        else "Hi there! I'm your math tutor. How can I help you today?"
    )


def get_assessment(
    dataset: str,
    client: OpenAI,
    model_name: str,
    student_temperature: float,
    problem_dict: Dict[str, Any],
    texts: List[str],
) -> str:
    """
    Sample two additional student attempts using the prior conversation context,
    evaluate them, and return a feedback string to append to the latest message.
    (Used only as an *in-convo* reflection block; never fed into evaluation logic.)
    """
    prev_convo = msgs_to_string(texts)

    # Pick generator/evaluator based on dataset
    gen = generate_code_completions if dataset == "code" else generate_math_completions
    efunc = evaluate_code_completion if dataset == "code" else evaluate_math_completion

    # IMPORTANT: math/code generators use different kw names; route accordingly
    gen_kwargs = dict(
        previous_conversation_str=prev_convo,
        client=client,
        model_name=model_name,
        temperature=float(student_temperature),
        num_samples=2,
        show_gt=False,
    )
    if dataset == "code":
        gen_kwargs["problem_dict"] = problem_dict
    else:
        gen_kwargs["problem"] = problem_dict

    completions, _, _ = gen(**gen_kwargs)

    results: List[bool] = []
    for comp in completions:
        _, ok = efunc(comp, problem_dict)
        results.append(ok)

    msg = (
        "\nFor your reference, here are two of your previous responses to the question, "
        "along with my feedback.\n"
    )
    for i, (comp, ok) in enumerate(zip(completions, results), start=1):
        msg += f"\n--- Your response {i} ---\n{comp}\n--- Evaluation result: {ok} ---\n"
    return msg


def interact(
    dataset: str,
    transcripts_dir: str,    
    teacher_client: OpenAI,
    teacher_name: str,
    teacher_temperature: float,
    student_client: OpenAI,
    student_name: str,
    student_temperature: float,
    problem: Dict[str, Any],
    start_from_turn: int,
    total_turns: int,
    assessment_turn: int,
    use_cot: bool = False,
) -> Tuple[str, int, int]:
    """
    Generate a single reply for the current turn.
    Returns: (reply_text, prompt_tokens, completion_tokens)
    """
    pid = problem["id"]

    total_teacher_prompt_tks = total_teacher_completion_tks = 0
    total_student_prompt_tks = total_student_completion_tks = 0

    # Initialize / resume conversation
    if start_from_turn <= 1:
        texts: List[str] = [_problem_greeting(dataset)]
        cur_start = 1
        logger.debug(f"[{pid}] starting fresh")
    else:
        tpath = os.path.join(transcripts_dir, f"{pid}.json")
        if os.path.exists(tpath):
            with open(tpath, "r") as f:
                prev = json.load(f)
            texts = [m["content"] for m in prev][:start_from_turn]
            cur_start = start_from_turn
            logger.info(f"[{pid}] resumed at turn {cur_start} from {tpath}")
        else:
            texts = ["Hi there! Let's pick up the conversation."]
            cur_start = 1
            logger.warning(f"[{pid}] resume requested but no transcript found; starting fresh")

    for turn in range(cur_start, total_turns, 1):
        is_student = (turn % 2 == 1)

        # Optional assessment injected into previous text
        if assessment_turn > 0 and turn == assessment_turn and len(texts) > 0:
            texts[-1] += get_assessment(
                dataset=dataset,
                client=student_client,
                model_name=student_name,
                student_temperature=student_temperature,
                problem_dict=problem,
                texts=texts,
            )
            logger.debug(f"[{pid}] assessment injected at turn {turn}")

        # Build messages for this turn
        sys_prompt = _select_sys_prompt(dataset, use_cot, is_student, problem)
        starting_with_user = True if is_student else False
        messages = [{"role": "system", "content": sys_prompt}] + format_messages(texts, starting_with_user=starting_with_user)

        # Choose model for this turn
        cur_client = student_client if is_student else teacher_client
        cur_model = student_name if is_student else teacher_name
        cur_temperature = float(student_temperature if is_student else teacher_temperature)

        # Generate a reply
        texts_list, p_tks, c_tks = chat_complete(
            cur_client, cur_model, messages, n=1, temperature=cur_temperature, max_tokens=2048
        )
        reply = (texts_list[0] if texts_list else "").strip()
        if ("<think>" in reply) or ("</think>" in reply):
            reply = remove_think_tags(reply)
        
        texts.append(reply)
        if is_student:
            total_student_prompt_tks += p_tks
            total_student_completion_tks += c_tks
        else:
            total_teacher_prompt_tks += p_tks
            total_teacher_completion_tks += c_tks

    # Save transcript with roles (model names as roles)
    edited_msgs: List[Dict[str, Any]] = []
    for idx, text in enumerate(texts):
        role = student_name if (idx % 2 == 1) else teacher_name
        edited_msgs.append({"role": role, "turn": idx, "content": text})

    # Save transcript
    with open(os.path.join(transcripts_dir, f"{pid}.json"), "w") as f:
        json.dump(edited_msgs, f, indent=2)

    return (
        texts, 
        total_student_prompt_tks, 
        total_student_completion_tks, 
        total_teacher_prompt_tks, 
        total_teacher_completion_tks
    )


def evaluate(
    texts: List[str],
    dataset: str,
    client: OpenAI,
    model_name: str,
    temperature: float,
    problem: Dict[str, Any],
    k_values: List[int],
    num_samples: int,
    start_from_turn: int,
    total_turns: int,
    is_ood: bool = False,
) -> Tuple[Dict[int, float], int, int]:
    """
    Inline evaluation for the current student turn.
    Returns: (pass_at_k_scores, prompt_tokens_used, completion_tokens_used)
    """
    pass_at_k_scores = {}
    total_prompt_tks = total_completion_tks = 0
    for turn in range(start_from_turn, total_turns + 1, 2):
        prev_convo = msgs_to_string(texts[: turn])

        if dataset == "code":
            completions, p_tks, c_tks = generate_code_completions(
                previous_conversation_str=prev_convo,
                client=client,
                model_name=model_name,
                problem_dict=problem,
                num_samples=num_samples,
                temperature=temperature,
                max_tokens=2048,
                show_gt=False,
            )
            eval_func = evaluate_code_completion
        else:
            completions, p_tks, c_tks = generate_math_completions(
                previous_conversation_str=prev_convo,
                client=client,
                model_name=model_name,
                problem=problem,
                num_samples=num_samples,
                temperature=temperature,
                max_tokens=512,
                show_gt=False,
                is_ood=is_ood,
            )
            eval_func = evaluate_math_completion

        # Evaluate completions in parallel
        flags: List[bool] = []
        with ThreadPoolExecutor() as ex:
            futures = [ex.submit(eval_func, comp, problem) for comp in completions]
            for fut in as_completed(futures):
                _, passed = fut.result()
                flags.append(passed)
        
        pass_at_k_scores[turn] = compute_pass_at_k(flags, k_values)
        total_prompt_tks += p_tks
        total_completion_tks += c_tks

        logger.debug(f"[{problem["id"]}] turn {turn} pass@k={pass_at_k_scores[turn]}")

    return pass_at_k_scores, total_prompt_tks, total_completion_tks


def run_problem(
    dataset: str,
    transcripts_dir: str,
    teacher_client: OpenAI,
    teacher_name: str,
    teacher_temperature: float,
    student_client: OpenAI,
    student_name: str,
    student_temperature: float,
    interaction_problem: Dict[str, Any],
    evaluation_problem: Dict[str, Any],
    k_values: List[int],
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

    # Interaction
    (
        texts,
        student_prompt_tks,
        student_completion_tks,
        teacher_prompt_tks,
        teacher_completion_tks
    ) = interact(
        dataset=dataset,
        transcripts_dir=transcripts_dir,
        teacher_client=teacher_client,
        teacher_name=teacher_name,
        teacher_temperature=teacher_temperature,
        student_client=student_client,
        student_name=student_name,
        student_temperature=student_temperature,
        problem=interaction_problem,
        start_from_turn=start_from_turn,
        total_turns=total_turns,
        assessment_turn=assessment_turn,
        use_cot=use_cot
    )

    # Evaluation
    (
        pass_at_k_scores, 
        prompt_tks, 
        completion_tks
    ) = evaluate(
        texts=texts,
        dataset=dataset,
        client=student_client,
        model_name=student_name,
        temperature=student_temperature,
        problem=evaluation_problem,
        k_values=k_values,
        num_samples=num_samples,
        start_from_turn=start_from_turn,
        total_turns=total_turns,
        is_ood=is_ood,
    )

    return (
        pass_at_k_scores,
        teacher_prompt_tks,
        teacher_completion_tks,
        student_prompt_tks + prompt_tks,
        student_completion_tks + completion_tks,
    )


# ---------- main combined loop ----------
@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """
    Hydra config expectations (mirrors your repo-2 style):
      experiment.dataset: "code" | "math"
      interact:
        total_turns: int
        start_from_turn: int
        assessment_turn: int
        use_cot: bool
      eval:
        k_values: [int, ...]
        num_samples: int
      paths:
        data: <root to curated sets>     (e.g., ./data)
        experiments: <output root>       (e.g., ./runs)
      teacher:
        model: str
        temperature: float
      student:
        model: str
        temperature: float
      data.data_amount: optional int to limit #problems
    """
    start_time = time.perf_counter()  # ---- start timer ----

    # Load API key and export for OpenAI SDK
    key_path = to_absolute_path(str(cfg.paths.openai_api_key_fpath))
    with open(key_path, "r") as f:
        openai_key = f.read().strip()
    os.environ["OPENAI_API_KEY"] = openai_key

    # Hydra config keys
    dataset: str = cfg.experiment.dataset
    total_turns: int = int(cfg.experiment.total_turns)
    start_from_turn: int = int(cfg.experiment.start_from_turn)
    assessment_turn: int = int(cfg.experiment.assessment_turn)
    use_cot: bool = bool(cfg.experiment.cot)
    k_values: List[int] = list(cfg.gen.k_values)
    num_samples: int = int(cfg.gen.num_samples)

    # Paths
    exp_dir = os.path.join(
        str(cfg.paths.experiments),
        str(cfg.experiment.name),
        str(cfg.teacher.model),
        str(cfg.student.model),
        dataset,
    )
    transcripts_dir = os.path.join(exp_dir, "transcripts")
    os.makedirs(transcripts_dir, exist_ok=True)

    problems_path = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        dataset,
        f"{cfg.student.model}.json",
    )

    # Load problems
    logger.info(f"Loading interaction problems: {problems_path}")
    with open(problems_path, "r") as f:
        data = json.load(f)
    interaction_problems: List[Dict[str, Any]] = data["collected"][: cfg.data.data_amount]
    interaction_problems = sorted(interaction_problems, key=lambda x: x["id"])
    logger.info(f"Total interaction problems: {len(interaction_problems)}")

    evaluation_problems = interaction_problems
    if cfg.experiment.ood:
        ood_problems_path = os.path.join(
            str(cfg.paths.data),
            str(cfg.teacher.model),
            dataset,
            f"{cfg.student.model}_ood.json",
        )

        logger.info(f"Loading evaluation (OOD) problems: {ood_problems_path}")
        with open(ood_problems_path, "r") as f:
            data = json.load(f)
        evaluation_problems: List[Dict[str, Any]] = data[: cfg.data.data_amount]
        evaluation_problems = sorted(evaluation_problems, key=lambda x: x["id"])
        logger.info(f"Total evaluation (OOD) problems: {len(evaluation_problems)}")

    # Build clients
    teacher_name, teacher_client = build_client(cfg.teacher.model)
    student_name, student_client = build_client(cfg.student.model)

    # Collect results across problems for aggregation
    turn_results: Dict[int, List[Dict[int, float]]] = {turn: [] for turn in range(start_from_turn, total_turns + 1, 2)}
    all_results: Dict[str, List[Dict[int, float]]] = {}

    total_teacher_prompt_tks = total_teacher_completion_tks = 0
    total_student_prompt_tks = total_student_completion_tks = 0

    # Parallel execution
    with ThreadPoolExecutor(max_workers=cfg.experiment.num_workers) as executor:
        future_to_pid = {executor.submit(
            run_problem,
            dataset,
            transcripts_dir,
            teacher_client,
            teacher_name,
            cfg.teacher.temperature,
            student_client,
            student_name,
            cfg.student.temperature,
            ip,
            ep,
            k_values,
            num_samples,
            assessment_turn,
            start_from_turn,
            total_turns,
            use_cot,
            cfg.experiment.ood
        ): ip["id"] for (ip, ep) in zip(interaction_problems, evaluation_problems)}

        for future in tqdm(as_completed(future_to_pid), total=len(future_to_pid)):
            (
                results,
                teacher_prompt_tks,
                teacher_completion_tks,
                student_prompt_tks,
                student_completion_tks,
            ) = future.result()

            # Update aggregates
            pid = future_to_pid[future]
            all_results[pid] = []
            for turn, res_list in results.items():
                turn_results[turn].append(res_list)
                all_results[pid].append(res_list)

            total_teacher_prompt_tks += teacher_prompt_tks
            total_teacher_completion_tks += teacher_completion_tks
            total_student_prompt_tks += student_prompt_tks
            total_student_completion_tks += student_completion_tks

    # Aggregate mean/std across problems
    avg_results: Dict[int, Dict[int, Tuple[float, float]]] = {
        turn: {k: (0.0, 0.0) for k in k_values} for turn in range(start_from_turn, total_turns + 1, 2)
    }
    for turn, turn_scores in turn_results.items():
        for k in k_values:
            vals = [d[k] for d in turn_scores] if turn_scores else [0.0]
            avg_k = float(round(np.mean(vals), 3))
            std_k = float(round(np.std(vals), 3))
            avg_results[turn][k] = (avg_k, std_k)

    # Save per-problem statistics and aggregate
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "itemized_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    with open(os.path.join(exp_dir, "aggregate_results.json"), "w") as f:
        json.dump(avg_results, f, indent=2)

    # Log aggregate summary
    for turn in sorted(avg_results.keys()):
        logger.info(f"Turn {turn:<3}: {avg_results[turn]}")
    logger.info(
        "Teacher %s | prompt_tks=%d completion_tks=%d",
        teacher_name, total_teacher_prompt_tks, total_teacher_completion_tks,
    )
    logger.info(
        "Student %s | prompt_tks=%d completion_tks=%d",
        student_name, total_student_prompt_tks, total_student_completion_tks,
    )

    elapsed = time.perf_counter() - start_time
    logger.info("Total time elapsed: %s", timedelta(seconds=int(elapsed)))


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"  # for debugging
    main()
