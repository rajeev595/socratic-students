"""
Student-teacher interaction with inline evaluation.

For every problem the conversation runs for N = `experiment.total_turns` questioning turns.
At every turn t = 0..N (as in the paper):
  1. if t == `experiment.assessment_turn`, the teacher's assessment of two student attempts is added,
  2. the student is asked for a direct answer `gen.num_samples` times and Pass@k is computed,
  3. the student asks a question and the teacher replies (the question of the last turn gets no reply).
Turn 0 is therefore the static baseline, before any question has been asked.

  experiment.scaffolded        False = Unscaffolded prompt, True = Scaffolded prompt
  experiment.assessment_turn   -1 = none, 0 = Pre-Assessment, 2 = Mid-Assessment
  student.lora.{name,path}     evaluate an ODQS-trained student

Output (under <experiments>/<experiment.name>/<teacher>/<student>/<dataset>/):
  transcripts_0/<problem id>.json   the conversation
  itemized_results_0.json           per-problem Pass@k at every turn and the sampled answers
  aggregate_results_0.json          bootstrapped mean/std of Pass@k at every turn

Usage:
    python -m src.interact student.model=qwen7 data.type=math experiment.dataset=gsm8k \
        experiment.name=scaffolded experiment.scaffolded=True
"""

import os
import time
import logging
from datetime import timedelta
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import hydra
from omegaconf import DictConfig

from openai import OpenAI

from src.utils.llm.server import build_client
from src.utils.llm.chat import chat_complete
from src.utils.env import set_envs
from src.utils.messages import remove_think_tags, msgs_to_string, format_messages, transcript_messages
from src.prompts.select import _select_sys_prompt, _problem_greeting
from src.evaluation.code import generate_code_completions, evaluate_code_completion, get_code_feedback_from_eval
from src.evaluation.math import generate_math_completions, evaluate_math_completion, get_math_feedback_from_eval
from src.evaluation.pass_at_k import compute_pass_at_k, bootstrap_aggregate
from src.utils.io import save_json_atomic, load_json_resilient


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)


# ---------- helpers ----------
def get_assessment(
    data_type: str,
    student_client: OpenAI,
    student_name: str,
    student_temperature: float,
    teacher_client: OpenAI,
    teacher_name: str,
    teacher_temperature: float,
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
    if data_type == "code":
        gen_fn = generate_code_completions
        eval_fn = evaluate_code_completion
        feedback_fn = get_code_feedback_from_eval

    elif data_type == "math":
        gen_fn = generate_math_completions
        eval_fn = lambda x, y: evaluate_math_completion(
            x, y, teacher_client, teacher_name, teacher_temperature
        )
        feedback_fn = get_math_feedback_from_eval

    # IMPORTANT: math/code generators use different kw names; route accordingly
    gen_kwargs = dict(
        previous_conversation_str=prev_convo,
        client=student_client,
        model_name=student_name,
        temperature=float(student_temperature),
        num_samples=2,
        show_gt=False,
    )
    if data_type == "code":
        gen_kwargs["problem_dict"] = problem_dict
    else:
        gen_kwargs["problem"] = problem_dict

    completions, _, _ = gen_fn(**gen_kwargs)

    results: List[bool] = []
    for comp in completions:
        _, ok = eval_fn(comp, problem_dict)
        results.append(ok)

    msg = (
        "\nFor your reference, here are two of your previous responses to the question, "
        "along with my feedback.\n"
    )
    for i, (comp, ok) in enumerate(zip(completions, results), start=1):
        status = "PASS" if ok else "FAIL"
        feedback, _, _ = feedback_fn(
            teacher_name, teacher_temperature, teacher_client, problem_dict, comp, status
        )
        msg += f"\n--- Your response {i} ---\n{comp}\n--- Evaluation result: {status} ---\n--- Feedback: {feedback} ---\n"
    return msg


def interact(
    is_student: bool,
    texts: List[str],
    problem: Dict[str, Any],
    num_samples,
    data_type: str,
    client: OpenAI,
    model_name: str,
    temperature: float,
    use_scaffold: bool = False,
):
    # Build messages for this turn
    sys_prompt = _select_sys_prompt(data_type, use_scaffold, is_student, problem)
    starting_with_user = True if is_student else False
    messages = [{"role": "system", "content": sys_prompt}] + format_messages(
        texts, starting_with_user=starting_with_user
    )

    # Generate a reply
    texts_list, p_tks, c_tks = chat_complete(
        client=client,
        model_name=model_name,
        messages=messages,
        n=num_samples,
        temperature=temperature,
        max_tokens=2048
    )
    replies = []
    for text in texts_list:
        reply = (text if text else "").strip()
        if ("<think>" in reply) or ("</think>" in reply):
            reply = remove_think_tags(reply)

        replies.append(reply)

    return replies, p_tks, c_tks


def evaluate(
    prev_convo: str,
    data_type: str,
    student_client: OpenAI,
    student_name: str,
    student_temperature: float,
    teacher_client: OpenAI,
    teacher_name: str,
    teacher_temperature: float,
    problem: Dict[str, Any],
    k_values: List[int],
    num_samples: int,
    is_ood: bool = False,        
):
    if data_type == "code":
        completions, p_tks, c_tks = generate_code_completions(
            previous_conversation_str=prev_convo,
            client=student_client,
            model_name=student_name,
            problem_dict=problem,
            num_samples=num_samples,
            temperature=student_temperature,
            max_tokens=2048,
            show_gt=False,
        )
        eval_func = evaluate_code_completion
    else:
        completions, p_tks, c_tks = generate_math_completions(
            previous_conversation_str=prev_convo,
            client=student_client,
            model_name=student_name,
            problem=problem,
            num_samples=num_samples,
            temperature=student_temperature,
            max_tokens=512,
            show_gt=False,
            is_ood=is_ood,
        )
        eval_func = lambda x, y : evaluate_math_completion(x, y, teacher_client, teacher_name, teacher_temperature)

    # Evaluate completions in parallel (flags[i] is the result of completions[i])
    flags: List[bool] = [False] * len(completions)
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(eval_func, comp, problem): i for i, comp in enumerate(completions)}
        for fut in as_completed(futures):
            _, passed = fut.result()
            flags[futures[fut]] = passed
    
    pass_at_k_scores = compute_pass_at_k(flags, k_values)
    generations = [(c, f) for c, f in zip(completions, flags)]

    return pass_at_k_scores, generations, p_tks, c_tks


def run_problem(
    data_type: str,
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
    start_from_turn: int = 0,
    total_turns: int = 5,
    use_scaffold: bool = False,
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
      - generations: turn -> sampled answers with their pass/fail flags
      - pass_at_k_scores: turn -> pass@k dict
      - total_teacher_prompt_tks, total_teacher_completion_tks
      - total_student_prompt_tks, total_student_completion_tks
    """
    try:
        assert interaction_problem["id"] == evaluation_problem["id"], "pid of interaction and evaluation doesn't match!"

        pid = interaction_problem["id"]
        transcript_fpath = os.path.join(transcripts_dir, f"{pid}.json")

        pass_at_k_scores = {}
        generations = {}
        total_teacher_prompt_tks = total_teacher_completion_tks = 0
        total_student_prompt_tks = total_student_completion_tks = 0

        # Initialize / resume conversation. A transcript holds the teacher greeting followed by one
        # (student question, teacher reply) pair per turn, so turn t starts after 2t + 1 messages.
        if start_from_turn <= 0:
            texts: List[str] = [_problem_greeting(data_type)]
            cur_start = 0
            logger.debug(f"[{pid}] starting fresh")
        else:
            if os.path.exists(transcript_fpath):
                prev = load_json_resilient(transcript_fpath)
                texts = [m["content"] for m in prev][: 2 * start_from_turn + 1]
                cur_start = start_from_turn
                logger.info(f"[{pid}] resumed at turn {cur_start} from {transcript_fpath}")
            else:
                texts = ["Hi there! Let's pick up the conversation."]
                cur_start = 0
                logger.warning(f"[{pid}] resume requested but no transcript found; starting fresh")

        for turn in range(cur_start, total_turns + 1):
            # Optional assessment, appended to the teacher's latest message
            if assessment_turn >= 0 and turn == assessment_turn and len(texts) > 0:
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

            # Evaluation: can the student solve the problem given the conversation so far?
            prev_convo = msgs_to_string(texts)

            pass_at_k_scores_turn, generations_turn, p_tks, c_tks = evaluate(
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

            pass_at_k_scores[turn] = pass_at_k_scores_turn
            total_student_prompt_tks += p_tks
            total_student_completion_tks += c_tks
            generations[turn] = generations_turn

            logger.debug(f"[{evaluation_problem['id']}] turn {turn} pass@k={pass_at_k_scores[turn]}")

            # Student question
            replies, p_tks, c_tks = interact(
                True,
                texts,
                interaction_problem,
                1,
                data_type,
                student_client,
                student_name,
                student_temperature,
                use_scaffold
            )
            texts.append(replies[0])
            total_student_prompt_tks += p_tks
            total_student_completion_tks += c_tks

            # Teacher reply (the question of the last turn is not answered)
            if turn < total_turns:
                replies, p_tks, c_tks = interact(
                    False,
                    texts,
                    interaction_problem,
                    1,
                    data_type,
                    teacher_client,
                    teacher_name,
                    teacher_temperature,
                    use_scaffold
                )
                texts.append(replies[0])
                total_teacher_prompt_tks += p_tks
                total_teacher_completion_tks += c_tks

        # Save transcript with roles (model names as roles)
        edited_msgs: List[Dict[str, Any]] = transcript_messages(texts, student_name, teacher_name)

        # Save transcript
        save_json_atomic(os.path.join(transcripts_dir, f"{pid}.json"), edited_msgs)

        return (
            generations,
            pass_at_k_scores,
            total_teacher_prompt_tks,
            total_teacher_completion_tks,
            total_student_prompt_tks,
            total_student_completion_tks
        )
    except Exception as e:
        logger.info(f"Problem {interaction_problem['id']} encountered error: {e}")
        return None


# ---------- main combined loop ----------
@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """Run the interaction loop over the problem set (see the module docstring and configs/config.yaml)."""
    start_time = time.perf_counter()  # ---- start timer ----

    # set the OpenAI key env variable, HF HOME
    set_envs(cfg.paths.openai_api_key_fpath, cfg.paths.hf_home)

    # Hydra config keys
    dataset: str = cfg.experiment.dataset
    total_turns: int = int(cfg.experiment.total_turns)
    start_from_turn: int = int(cfg.experiment.start_from_turn)
    assessment_turn: int = int(cfg.experiment.assessment_turn)
    use_scaffold: bool = bool(cfg.experiment.scaffolded)
    k_values: List[int] = list(cfg.gen.k_values)
    num_samples: int = min(int(cfg.gen.num_samples), max(k_values))

    # Paths
    exp_dir = os.path.join(
        str(cfg.paths.experiments),
        str(cfg.experiment.name),
        str(cfg.teacher.model),
        str(cfg.student.model),
        dataset,
    )
    os.makedirs(exp_dir, exist_ok=True)
    transcripts_dir = os.path.join(exp_dir, f"transcripts_{cfg.gen.iteration}")
    os.makedirs(transcripts_dir, exist_ok=True)

    # Collect results across problems for aggregation
    turn_results: Dict[int, List[Dict[int, float]]] = {turn: [] for turn in range(start_from_turn, total_turns + 1)}
    all_results = {}

    # Load existing results
    results_fpath = os.path.join(exp_dir, f"itemized_results_{cfg.gen.iteration}.json")
    if os.path.exists(results_fpath):
        all_results = load_json_resilient(results_fpath)
        for _, example in all_results.items():
            for turn, res_list in example["results"].items():
                turn_results[int(turn)].append({int(k): v for k, v in res_list.items()})

        logger.info(f"Loaded experiment results for {len(all_results)} problems from: {results_fpath}")

    # Load problems
    problems_fpath = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        dataset,
        f"splits_{cfg.gen.pass_at_k_threshold}",
        f"{cfg.student.model}_{cfg.data.split}.json",
    )
    logger.info(f"Loading interaction problems: {problems_fpath}")
    data: List[Dict[str, Any]] = load_json_resilient(problems_fpath)
    interaction_problems = [
        p for p in data if p["id"] not in all_results and str(p["id"]) not in all_results
    ][: max(0, cfg.data.data_amount - len(all_results))]
    logger.info(f"Total interaction problems: {len(interaction_problems)}")

    # OOD problems
    evaluation_problems = interaction_problems
    if cfg.experiment.ood:
        ood_problems_path = os.path.join(
            str(cfg.paths.data),
            str(cfg.teacher.model),
            dataset,
            f"{cfg.student.model}_ood_{cfg.data.split}.json",
        )

        logger.info(f"Loading evaluation (OOD) problems: {ood_problems_path}")
        data: List[Dict[str, Any]] = load_json_resilient(ood_problems_path)
        evaluation_problems = [
            p for p in data if p["id"] not in all_results and str(p["id"]) not in all_results
        ][: max(0, cfg.data.data_amount - len(all_results))]
        logger.info(f"Total evaluation (OOD) problems: {len(evaluation_problems)}")

    if len(interaction_problems) == 0 or len(evaluation_problems) == 0:
        return

    # Build teacher and student clients (names may be normalized inside)
    teacher_name, student_name = cfg.teacher.model, cfg.student.model
    teacher_client = build_client(
        teacher_name, cfg.teacher.gpus, cfg.teacher.server
    )
    student_client = build_client(
        student_name, cfg.student.gpus, cfg.student.server, lora=cfg.student.lora
    )
    logger.info("Teacher=%s | Student=%s", teacher_name, student_name)

    total_teacher_prompt_tks = total_teacher_completion_tks = 0
    total_student_prompt_tks = total_student_completion_tks = 0
    pbar = tqdm(total=len(interaction_problems), desc="Problems done", dynamic_ncols=True)

    # Parallel execution
    with ThreadPoolExecutor(max_workers=cfg.experiment.num_workers) as executor:
        for i in range(0, len(interaction_problems), cfg.experiment.chunk_size):
            # Problem chunks in this iter
            interaction_problems_chunk = interaction_problems[i: i + cfg.experiment.chunk_size]
            evaluation_problems_chunk = evaluation_problems[i: i + cfg.experiment.chunk_size]

            # Prepare jobs
            future_to_pid = {executor.submit(
                run_problem,
                cfg.data.type,
                transcripts_dir,
                teacher_client,
                teacher_name,
                cfg.teacher.temperature,
                student_client,
                cfg.student.lora.name or student_name,
                cfg.student.temperature,
                ip,
                ep,
                k_values,
                num_samples,
                assessment_turn,
                start_from_turn,
                total_turns,
                use_scaffold,
                cfg.experiment.ood
            ): ip["id"] for (ip, ep) in zip(interaction_problems_chunk, evaluation_problems_chunk)}

            for future in as_completed(future_to_pid):
                result = future.result()
                if not result:
                    continue

                (
                    generations,
                    results,
                    teacher_prompt_tks,
                    teacher_completion_tks,
                    student_prompt_tks,
                    student_completion_tks,
                ) = result

                # Update aggregates
                pid = future_to_pid[future]
                for turn, res_list in results.items():
                    turn_results[turn].append(res_list)

                all_results[pid] = {
                    "results": results,
                    "generations": generations
                }

                total_teacher_prompt_tks += teacher_prompt_tks
                total_teacher_completion_tks += teacher_completion_tks
                total_student_prompt_tks += student_prompt_tks
                total_student_completion_tks += student_completion_tks
                pbar.update(1)
                logger.info("%s", pbar.format_meter(n=pbar.n, total=pbar.total, elapsed=pbar.format_dict["elapsed"], prefix=pbar.desc + ": "))

            # Save per-problem statistics each chunk
            save_json_atomic(results_fpath, all_results)
    pbar.close()

    # compute bootstrapped aggregate results
    avg_results = bootstrap_aggregate(all_results)
    save_json_atomic(os.path.join(exp_dir, f"aggregate_results_{cfg.gen.iteration}.json"), avg_results)

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
    main()
