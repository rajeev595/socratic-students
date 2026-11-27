"""
Cross-evaluation script for the 2x2 Question Generator × Evaluator ablation.

Runs the full student↔teacher interaction using the GENERATOR model (so questions
are generated in their natural context), but evaluates Pass@k at each student turn
using a separate EVALUATOR model.  The two models can be the same (standard run)
or different (cross condition).

This ablation answers: do gains come from asking better questions (generator effect)
or from being better at extracting information from teacher responses (evaluator effect)?

Four conditions:
    Generator       Evaluator
    -----------     -----------
    No-Training     No-Training     baseline (= standard unscaffolded/pre_assessment run)
    ODQS            ODQS            current reported result
    ODQS            No-Training     do ODQS questions help a weaker evaluator?
    No-Training     ODQS            does ODQS advantage come from the evaluator?

Usage:
    python codes/run_cross_eval.py \
        student.model=mistral7 \
        student.lora.name=mistral7_dpo \
        student.lora.path=/path/to/lora \
        evaluator.model=mistral7 \
        evaluator.lora.name=mistral7_dpo \
        evaluator.lora.path=/path/to/lora \
        evaluator.gpus=[4,5,6,7] \
        experiment.dataset=gsm8k

Output saved to:
    experiments/cross_eval/
      gen_{generator_model}__eval_{evaluator_model}/
        {teacher_model}/{dataset}/
          transcripts/
          itemized_results.json
          aggregate_results.json
"""

import os
import time
import logging
from datetime import timedelta
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from codes.utils.llm.server import build_client
from codes.utils.vars import set_envs
from codes.data.utils import msgs_to_string, transcript_messages
from codes.prompts.utils import _select_sys_prompt, _problem_greeting
from codes.evaluation.utils import bootstrap_aggregate
from codes.utils.file_ops import save_json_atomic, load_json_resilient
from codes.run_interact import interact, evaluate   # reuse both directly


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)


def run_problem_cross(
    data_type: str,
    transcripts_dir: str,
    teacher_client,
    teacher_name: str,
    teacher_temperature: float,
    generator_client,
    generator_name: str,
    generator_temperature: float,
    evaluator_client,
    evaluator_name: str,
    evaluator_temperature: float,
    problem: Dict[str, Any],
    k_values: List[int],
    num_samples: int,
    total_turns: int,
    assessment_turn: int,
    use_scaffold: bool,
) -> Tuple[Dict, Dict] | None:
    """
    Run one problem end-to-end:
      - GENERATOR model drives the conversation (asks questions)
      - EVALUATOR model generates Pass@k completions at each student turn
    """
    try:
        pid = problem["id"]
        transcript_path = os.path.join(transcripts_dir, f"{pid}.json")

        # Resume if transcript already exists and is complete
        if os.path.exists(transcript_path):
            existing = load_json_resilient(transcript_path)
            if existing and len(existing) >= 2 * total_turns + 2:
                logger.info(f"[{pid}] transcript complete, skipping interaction")
                texts = [t["content"] for t in existing]
                # Still need to re-evaluate with evaluator model
                pass_at_k_scores, generations = {}, {}
                for turn in range(total_turns + 1):
                    # conversation as it stood at this turn: greeting + one (question, reply) pair per earlier turn
                    prev_convo = msgs_to_string(texts[: 2 * turn + 1])
                    scores, gens, _, _ = evaluate(
                        prev_convo=prev_convo,
                        data_type=data_type,
                        student_client=evaluator_client,
                        student_name=evaluator_name,
                        student_temperature=evaluator_temperature,
                        teacher_client=teacher_client,
                        teacher_name=teacher_name,
                        teacher_temperature=teacher_temperature,
                        problem=problem,
                        k_values=k_values,
                        num_samples=num_samples,
                        is_ood=False,
                    )
                    pass_at_k_scores[turn] = scores
                    generations[turn] = gens
                return pass_at_k_scores, generations

        texts: List[str] = [_problem_greeting(data_type)]
        pass_at_k_scores: Dict[int, Any] = {}
        generations: Dict[int, Any] = {}

        for turn in range(total_turns + 1):
            # Optional assessment injection (uses generator for assessment sampling)
            if assessment_turn >= 0 and turn == assessment_turn and len(texts) > 0:
                from codes.run_interact import get_assessment
                texts[-1] += get_assessment(
                    data_type=data_type,
                    student_client=generator_client,
                    student_name=generator_name,
                    student_temperature=generator_temperature,
                    teacher_client=teacher_client,
                    teacher_name=teacher_name,
                    teacher_temperature=teacher_temperature,
                    problem_dict=problem,
                    texts=texts,
                )

            # Pass@k evaluation -- uses the EVALUATOR
            prev_convo = msgs_to_string(texts)
            scores, gens, _, _ = evaluate(
                prev_convo=prev_convo,
                data_type=data_type,
                student_client=evaluator_client,
                student_name=evaluator_name,
                student_temperature=evaluator_temperature,
                teacher_client=teacher_client,
                teacher_name=teacher_name,
                teacher_temperature=teacher_temperature,
                problem=problem,
                k_values=k_values,
                num_samples=num_samples,
                is_ood=False,
            )
            pass_at_k_scores[turn] = scores
            generations[turn] = gens
            logger.debug(f"[{pid}] turn {turn} pass@k={scores}")

            # Student question -- uses the GENERATOR
            replies, _, _ = interact(
                True, texts, problem, 1, data_type,
                generator_client, generator_name, generator_temperature, use_scaffold,
            )
            texts.append(replies[0])

            # Teacher reply (the question of the last turn is not answered)
            if turn < total_turns:
                replies, _, _ = interact(
                    False, texts, problem, 1, data_type,
                    teacher_client, teacher_name, teacher_temperature, use_scaffold,
                )
                texts.append(replies[0])

        # Save transcript (labelled with generator name as student role)
        edited_msgs = transcript_messages(texts, generator_name, teacher_name)
        save_json_atomic(transcript_path, edited_msgs)

        return pass_at_k_scores, generations

    except Exception as e:
        logger.error(f"Error on problem {problem.get('id', '?')}: {e}")
        return None


@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    start_time = time.perf_counter()

    set_envs(cfg.paths.openai_api_key_fpath, cfg.paths.hf_home)

    dataset: str = cfg.experiment.dataset
    k_values: List[int] = list(cfg.gen.k_values)
    num_samples: int = min(int(cfg.gen.num_samples), max(k_values))
    data_type: str = cfg.data.type
    total_turns: int = int(cfg.experiment.total_turns)
    assessment_turn: int = int(cfg.experiment.assessment_turn)
    use_scaffold: bool = bool(cfg.experiment.scaffolded)

    teacher_name: str = str(cfg.teacher.model)
    generator_name: str = str(cfg.student.lora.name or cfg.student.model)
    evaluator_lora = cfg.evaluator.get("lora", None)
    evaluator_lora_name = evaluator_lora.name if evaluator_lora else None
    evaluator_name: str = str(evaluator_lora_name or cfg.evaluator.model)

    # Output directory encodes both generator and evaluator
    out_dir = os.path.join(
        str(cfg.paths.experiments),
        "cross_eval",
        f"gen_{generator_name}__eval_{evaluator_name}",
        teacher_name,
        dataset,
    )
    transcripts_dir = os.path.join(out_dir, "transcripts")
    os.makedirs(transcripts_dir, exist_ok=True)
    results_fpath = os.path.join(out_dir, "itemized_results.json")

    # Load partial results
    all_results: Dict[str, Any] = {}
    if os.path.exists(results_fpath):
        all_results = load_json_resilient(results_fpath)
        logger.info(f"Resuming: {len(all_results)} problems already done")

    # Load problems
    problems_fpath = os.path.join(
        str(cfg.paths.data),
        teacher_name,
        dataset,
        f"splits_{cfg.gen.pass_at_k_threshold}",
        f"{cfg.student.model}_{cfg.data.split}.json",
    )
    problems: List[Dict[str, Any]] = load_json_resilient(problems_fpath)
    pending = [
        p for p in problems
        if str(p["id"]) not in all_results and p["id"] not in all_results
    ][: max(0, cfg.data.data_amount - len(all_results))]
    logger.info(f"Problems to run: {len(pending)}")

    if not pending:
        logger.info("Nothing to do.")
        return

    # Build clients
    teacher_client = build_client(teacher_name, cfg.teacher.gpus, cfg.teacher.server)
    generator_client = build_client(
        cfg.student.model, cfg.student.gpus, cfg.student.server, lora=cfg.student.lora
    )
    evaluator_client = build_client(
        cfg.evaluator.model,
        cfg.evaluator.gpus,
        cfg.evaluator.get("server", "vllm"),
        lora=cfg.evaluator.get("lora", None),
    )

    logger.info(f"Generator : {generator_name}")
    logger.info(f"Evaluator : {evaluator_name}")
    logger.info(f"Teacher   : {teacher_name}")

    pbar = tqdm(total=len(pending), desc="Cross-eval problems", dynamic_ncols=True)

    with ThreadPoolExecutor(max_workers=cfg.experiment.num_workers) as executor:
        for i in range(0, len(pending), cfg.experiment.chunk_size):
            chunk = pending[i: i + cfg.experiment.chunk_size]

            future_to_pid = {
                executor.submit(
                    run_problem_cross,
                    data_type,
                    transcripts_dir,
                    teacher_client,
                    teacher_name,
                    float(cfg.teacher.temperature),
                    generator_client,
                    generator_name,
                    float(cfg.student.temperature),
                    evaluator_client,
                    evaluator_name,
                    float(cfg.evaluator.get("temperature", 0.3)),
                    problem,
                    k_values,
                    num_samples,
                    total_turns,
                    assessment_turn,
                    use_scaffold,
                ): str(problem["id"])
                for problem in chunk
            }

            for future in as_completed(future_to_pid):
                pid = future_to_pid[future]
                result = future.result()
                if result is None:
                    pbar.update(1)
                    continue

                pass_at_k_scores, generations = result
                all_results[pid] = {"results": pass_at_k_scores, "generations": generations}
                pbar.update(1)

            save_json_atomic(results_fpath, all_results)

    pbar.close()

    avg_results = bootstrap_aggregate(all_results)
    save_json_atomic(os.path.join(out_dir, "aggregate_results.json"), avg_results)

    for turn in sorted(avg_results.keys()):
        logger.info(f"Turn {turn:<3}: {avg_results[turn]}")

    elapsed = time.perf_counter() - start_time
    logger.info(f"Done in {timedelta(seconds=int(elapsed))}. Results: {out_dir}")


if __name__ == "__main__":
    main()
