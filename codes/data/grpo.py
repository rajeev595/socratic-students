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

# -------------------------
# Helpers
# -------------------------
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

def _get_score(score_dict: Dict[Any, float], k: int) -> float:
    """
    Access pass@k from a dict that may have int or str keys.
    Falls back to the largest available k if not present.
    """
    if score_dict is None:
        return 0.0
    if k in score_dict:
        return float(score_dict[k])
    ks = list(score_dict.keys())
    # try string form
    if str(k) in score_dict:
        return float(score_dict[str(k)])
    # try int-cast on all keys and take max
    try:
        int_keys = [int(x) for x in ks]
        best_k = int_keys[int_keys.index(max(int_keys))]
        return float(score_dict.get(best_k, score_dict.get(str(best_k), 0.0)))
    except Exception:
        # as a last resort, take any numeric-looking value
        try:
            return float(next(iter(score_dict.values())))
        except Exception:
            return 0.0

def _to_float_dict(d: Dict[Any, Any]) -> Dict[Any, float]:
    return { (int(k) if str(k).isdigit() else k): float(v) for k, v in d.items() }

# -------------------------
# Core runner
# -------------------------
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
    # NEW knobs (read from cfg in caller)
    use_delta_reward: bool = False,
    dpo_margin_min: float = 0.01,
    dpo_margin_max: float = 1.0,
    dpo_max_pairs_per_turn: int = None,
) -> Tuple[
    List[Dict[str, Any]],   # chosen_list (SFT)
    List[Dict[str, Any]],   # pairs (DPO)
    List[Dict[str, Any]],   # grpo_groups
    int, int, int, int
]:
    """
    Run a single problem end-to-end: student↔teacher interaction and inline evaluation.
    Returns:
      - chosen_list: SFT items
      - pairs: DPO pairs (now with delta-aware rewards if enabled)
      - grpo_groups: list of grouped examples for GRPO/RRHF (with metrics_current/baseline/delta)
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

    # --- Baseline reuse cache across turns ---
    # pass@k dict corresponding to the EXACT 'texts' at the *start* of a turn.
    last_baseline_scores: Dict[Any, float] = {}
    last_baseline_valid: bool = False

    for turn in range(cur_start, total_turns, 2):
        # Optional assessment injected into previous text
        assessment_injected = False
        if assessment_turn > 0 and turn == assessment_turn and len(texts) > 0:
            texts[-1] += get_assessment(
                data_type=data_type,
                client=student_client,
                model_name=student_name,
                student_temperature=student_temperature,
                problem_dict=interaction_problem,
                texts=texts,
            )
            assessment_injected = True
            logger.debug(f"[{pid}] assessment injected at turn {turn}")

        # ---- Student: sample K candidate questions
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

        # ---- Teacher for each candidate (serial, unchanged)
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

        # ---- Establish the baseline ONCE per turn (reuse previous turn if valid)
        if use_delta_reward:
            if last_baseline_valid and not assessment_injected:
                base_scores = last_baseline_scores
            else:
                convo_baseline = msgs_to_string(texts)
                base_scores_raw, _, p_b, c_b = evaluate(
                    prev_convo=convo_baseline,
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
                base_scores = _to_float_dict(base_scores_raw)
                total_student_prompt_tks += p_b
                total_student_completion_tks += c_b
        else:
            base_scores = {}

        # ---- Evaluate each (student Q, teacher A) pair (current scores only)
        pass_at_k_scores_current = []
        for student_reply, teacher_reply in zip(student_replies, teacher_replies):
            # current: history + candidate Q/A
            convo_current = msgs_to_string(texts + [student_reply, teacher_reply])
            cur_scores, _, p_tks, c_tks = evaluate(
                prev_convo=convo_current,
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
            pass_at_k_scores_current.append(_to_float_dict(cur_scores))
            total_student_prompt_tks += p_tks
            total_student_completion_tks += c_tks

        # Build a per-candidate baseline list (same baseline for all candidates this turn)
        pass_at_k_scores_baseline = (
            [base_scores for _ in range(len(student_replies))] if use_delta_reward else [{} for _ in range(len(student_replies))]
        )

        # ---- Prompt messages used to elicit candidate questions
        sys_prompt = _select_sys_prompt(data_type, use_cot, True, evaluation_problem)
        prompt_messages = [{"role": "system", "content": sys_prompt}] + format_messages(
            texts, starting_with_user=True
        )

        # ---------- Collect GRPO group ----------
        group = {
            "ex_id": pid,
            "problem": evaluation_problem,
            "turn": turn,
            "prompt_messages": prompt_messages,
            "prompt_type": "messages",
            "k": rlhf_samples,
            "pass_k_key": pass_k,  # int
            "candidates": [],
        }

        for reply, cur_scores, base_scores_i in zip(student_replies, pass_at_k_scores_current, pass_at_k_scores_baseline):
            q_text = _extract_text(reply)

            # reward definition
            if use_delta_reward:
                reward = _get_score(cur_scores, pass_k) - _get_score(base_scores_i, pass_k)
            else:
                reward = _get_score(cur_scores, pass_k)

            # metrics payloads
            metrics_current = {int(k) if str(k).isdigit() else k: float(v) for k, v in cur_scores.items()}
            metrics_baseline = (
                {int(k) if str(k).isdigit() else k: float(v) for k, v in base_scores_i.items()}
                if use_delta_reward else {}
            )
            metrics_delta = (
                {k: float(metrics_current.get(k, 0.0) - metrics_baseline.get(k, 0.0)) for k in metrics_current.keys()}
                if use_delta_reward else {}
            )

            group["candidates"].append({
                "text": q_text,
                "reward": float(reward),
                "metrics_current": metrics_current,
                "metrics_baseline": metrics_baseline,
                "metrics_delta": metrics_delta,
            })

        grpo_groups.append(group)
        # ---------------------------------------

        # ---- Choose best by chosen pass@k on CURRENT (not delta) to roll convo forward
        # (we want the dialogue itself to improve; policy learning can still use delta.)
        best_values = [_get_score(s, pass_k) for s in pass_at_k_scores_current]
        max_value = max(best_values) if best_values else 0.0
        best_idx = best_values.index(max_value) if best_values else 0
        best_pass_at_k_score = pass_at_k_scores_current[best_idx] if pass_at_k_scores_current else {}

        # ---- Exploration vs exploitation (propensity fix)
        appended_idx = best_idx
        if random.random() < 1 - eps:
            texts += [student_replies[best_idx], teacher_replies[best_idx]]
            propensity = 1 - eps
        else:
            non_best = [i for i in range(rlhf_samples) if i != best_idx]
            idx = random.choice(non_best) if non_best else best_idx
            texts += [student_replies[idx], teacher_replies[idx]]
            appended_idx = idx
            # probability mass spread over (K-1) non-best choices
            propensity = eps / max(1, rlhf_samples)

        # SFT item
        chosen = student_replies[best_idx]

        # ---- SFT/DPO collection gated by quality (unchanged gate)
        if max_value >= 0.85:
            chosen_list.append({
                "prompt": prompt_messages,
                "response": chosen,
                "problem": evaluation_problem,
                "turn": turn,
                "propensity": propensity
            })

        # ---- DPO pair mining (now delta-aware and hard-negative friendly)
        # Build list of candidates with margins vs best
        deltas_for_pairs = []
        for score_cur, reply in zip(pass_at_k_scores_current, student_replies):
            if reply is chosen:
                continue
            best_val = _get_score(best_pass_at_k_score, pass_k)
            cur_val = _get_score(score_cur, pass_k)
            margin = best_val - cur_val
            deltas_for_pairs.append((margin, reply, score_cur))

        # Filter by margin band
        min_m = dpo_margin_min if dpo_margin_min is not None else 0.05  # previous behavior ~0.05
        max_m = dpo_margin_max if dpo_margin_max is not None else 1.0
        filtered = [(m, r, s) for (m, r, s) in deltas_for_pairs if (m >= min_m and m <= max_m)]

        # If we cap pairs per turn, bias selection toward harder (smaller margin) examples
        if dpo_max_pairs_per_turn is not None and len(filtered) > dpo_max_pairs_per_turn:
            # sort by margin ascending (harder first)
            filtered.sort(key=lambda x: x[0])
            filtered = filtered[:dpo_max_pairs_per_turn]

        # Save pairs
        for margin, reply, score_cur in filtered:
            # rewards for DPO: current or delta to stay consistent with GRPO
            if use_delta_reward:
                # baseline is shared across candidates this turn
                chosen_reward_dict = {
                    "current": best_pass_at_k_score,
                    "baseline": base_scores,
                    "delta@pass_k": _get_score(best_pass_at_k_score, pass_k) - _get_score(base_scores, pass_k)
                }
                rejected_reward_dict = {
                    "current": score_cur,
                    "baseline": base_scores,
                    "delta@pass_k": _get_score(score_cur, pass_k) - _get_score(base_scores, pass_k)
                }
            else:
                chosen_reward_dict = best_pass_at_k_score
                rejected_reward_dict = score_cur

            pairs.append({
                "prompt": prompt_messages,
                "chosen_response": chosen,
                "rejected_response": reply,
                "chosen_reward": chosen_reward_dict,
                "rejected_reward": rejected_reward_dict,
                "problem": evaluation_problem,
                "turn": turn
            })

        logger.debug(f'pid-{evaluation_problem["id"]} turn-{turn} best pass@{pass_k}={_get_score(best_pass_at_k_score, pass_k):.3f}')

        # ---- Update reusable baseline for NEXT turn
        # The baseline for the next turn is the pass@k of the context we just appended (the *appended_idx* branch).
        if use_delta_reward and pass_at_k_scores_current:
            last_baseline_scores = pass_at_k_scores_current[appended_idx] or {}
            last_baseline_valid = True

    return (
        chosen_list,
        pairs,
        grpo_groups,
        total_teacher_prompt_tks,
        total_teacher_completion_tks,
        total_student_prompt_tks,
        total_student_completion_tks
    )

# -------------------------
# Hydra entry
# -------------------------
@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    start_time = time.perf_counter()  # ---- start timer ----

    # set envs
    set_envs(cfg.paths.openai_api_key_fpath, cfg.paths.hf_home)

    # Output files (SFT, DPO, GRPO)
    fpath_sft = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        str(cfg.experiment.dataset),
        f"{cfg.student.model}_cot_sft_iter{cfg.gen.dagger_iter}_{cfg.data.split}.jsonl" if cfg.experiment.cot else f"{cfg.student.model}_sft_iter{cfg.gen.dagger_iter}_{cfg.data.split}.jsonl"
    )
    fpath_dpo = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        str(cfg.experiment.dataset),
        f"{cfg.student.model}_cot_dpo_iter{cfg.gen.dagger_iter}_{cfg.data.split}.jsonl" if cfg.experiment.cot else f"{cfg.student.model}_dpo_iter{cfg.gen.dagger_iter}_{cfg.data.split}.jsonl"
    )
    fpath_grpo = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        str(cfg.experiment.dataset),
        f"{cfg.student.model}_cot_grpo_iter{cfg.gen.dagger_iter}_{cfg.data.split}.jsonl" if cfg.experiment.cot else f"{cfg.student.model}_grpo_iter{cfg.gen.dagger_iter}_{cfg.data.split}.jsonl"
    )

    existing_pids = set()
    count_dpo, count_sft, count_grpo = 0, 0, 0

    # Resume dedupe by dagger_iter for SFT/DPO
    if os.path.exists(fpath_sft):
        with open(fpath_sft, "r") as f:
            for line in f:
                ex = json.loads(line)
                count_sft += 1
                if ex.get("dagger_iter") == cfg.gen.dagger_iter:
                    existing_pids.add(ex["problem"]["id"])
        logger.info(f"Loaded {count_sft} existing SFT-{cfg.gen.dagger_iter} examples from {fpath_sft}")

    if os.path.exists(fpath_dpo):
        with open(fpath_dpo, "r") as f:
            for line in f:
                ex = json.loads(line)
                count_dpo += 1
                if ex.get("dagger_iter") == cfg.gen.dagger_iter:
                    existing_pids.add(ex["problem"]["id"])
        logger.info(f"Loaded {count_dpo} existing DPO-{cfg.gen.dagger_iter} examples from {fpath_dpo}")

    if os.path.exists(fpath_grpo):
        with open(fpath_grpo, "r") as f:
            for line in f:
                ex = json.loads(line)
                count_grpo += 1
                if ex.get("dagger_iter") == cfg.gen.dagger_iter:
                    existing_pids.add(ex["problem"]["id"])
        logger.info(f"Loaded {count_grpo} existing GRPO-{cfg.gen.dagger_iter} examples from {fpath_grpo}")

    # Problems
    problems_fpath = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        str(cfg.experiment.dataset),
        f"splits_{cfg.gen.pass_at_k_threshold}",
        f"{cfg.student.model}_{cfg.data.split}.json",
    )
    logger.info(f"Loading interaction problems: {problems_fpath}")
    data = load_json_resilient(problems_fpath)
    interaction_problems: List[Dict[str, Any]] = [
        p for p in data if p["id"] not in existing_pids
    ][: max(0, cfg.data.data_amount - len(existing_pids))]
    logger.info(f"Interaction problems: finished {len(existing_pids)} | pending {len(interaction_problems)}")

    # OOD problems
    evaluation_problems = interaction_problems
    if cfg.experiment.ood:
        ood_problems_path = os.path.join(
            str(cfg.paths.data),
            str(cfg.teacher.model),
            str(cfg.experiment.dataset),
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
    teacher_client = build_client(teacher_name, cfg.teacher.gpus, cfg.teacher.server)
    student_client = build_client(student_name, cfg.student.gpus, cfg.student.server)
    logger.info("Teacher=%s | Student=%s", teacher_name, student_name)

    total_teacher_prompt_tks = total_teacher_completion_tks = 0
    total_student_prompt_tks = total_student_completion_tks = 0

    # Files
    f_dpo = open(fpath_dpo, "a" if os.path.exists(fpath_dpo) else "w")
    f_sft = open(fpath_sft, "a" if os.path.exists(fpath_sft) else "w")
    f_grpo = open(fpath_grpo, "a" if os.path.exists(fpath_grpo) else "w")

    # NEW knobs from cfg (with defaults)
    use_delta_reward = bool(getattr(cfg.gen, "use_delta_reward", False))
    dpo_margin_min = float(getattr(cfg.gen, "dpo_margin_min", 0.01))
    dpo_margin_max = float(getattr(cfg.gen, "dpo_margin_max", 1.0))
    dpo_max_pairs_per_turn = getattr(cfg.gen, "dpo_max_pairs_per_turn", None)
    if dpo_max_pairs_per_turn is not None:
        dpo_max_pairs_per_turn = int(dpo_max_pairs_per_turn)
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
                    cfg.experiment.ood,
                    use_delta_reward,
                    dpo_margin_min,
                    dpo_margin_max,
                    dpo_max_pairs_per_turn,
                ))

            for future in as_completed(futures):
                (
                    chosen,
                    pairs,
                    grpo_groups,
                    teacher_prompt_tks,
                    teacher_completion_tks,
                    student_prompt_tks,
                    student_completion_tks
                ) = future.result()

                # Save DPO
                for ex in pairs:
                    ex["dagger_iter"] = cfg.gen.dagger_iter
                    f_dpo.write(json.dumps(ex, ensure_ascii=False) + "\n")
                count_dpo += len(pairs)

                # Save SFT
                for ex in chosen:
                    ex["dagger_iter"] = cfg.gen.dagger_iter
                    f_sft.write(json.dumps(ex, ensure_ascii=False) + "\n")
                count_sft += len(chosen)

                # Save GRPO
                for group in grpo_groups:
                    group["dagger_iter"] = cfg.gen.dagger_iter
                    f_grpo.write(json.dumps(group, ensure_ascii=False) + "\n")
                count_grpo += len(grpo_groups)

                # Token usage
                total_teacher_prompt_tks += teacher_prompt_tks
                total_teacher_completion_tks += teacher_completion_tks
                total_student_prompt_tks += student_prompt_tks
                total_student_completion_tks += student_completion_tks
                pbar.update(1)
                logger.info(f"No. of examples collected DPO: {count_dpo} | SFT: {count_sft} | GRPO: {count_grpo}")
                logger.info("%s", pbar.format_meter(n=pbar.n, total=pbar.total, elapsed=pbar.format_dict["elapsed"], prefix=pbar.desc + ": "))

    # Close files
    f_dpo.close()
    f_sft.close()
    f_grpo.close()
    pbar.close()

    # Log summary
    logger.info(f"------- {cfg.data.split} -------")
    logger.info(f"Saved {count_dpo} preference pairs to {fpath_dpo}")
    logger.info(f"Saved {count_sft} SFT examples to {fpath_sft}")
    logger.info(f"Saved {count_grpo} GRPO groups to {fpath_grpo}")
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
