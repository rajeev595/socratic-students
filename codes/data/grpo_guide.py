# grpo_guide.py

import os
import time
import json
import random
import logging
from datetime import timedelta
from typing import Any, List, Dict, Tuple, Optional

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
from codes.utils.file_ops import load_json_resilient


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)

# -------------------------
# Helpers
# -------------------------
def _extract_text(msg: Any) -> str:
    """Best-effort extraction of text content."""
    if isinstance(msg, dict) and "content" in msg:
        return msg["content"]
    if isinstance(msg, str):
        return msg
    try:
        return json.dumps(msg, ensure_ascii=False)
    except Exception:
        return str(msg)

def _get_score(score_dict: Optional[Dict[Any, float]], k: int) -> float:
    """
    Access pass@k from a dict that may have int or str keys.
    Falls back to the largest available k if not present.
    """
    if not score_dict:
        return 0.0
    if k in score_dict:
        return float(score_dict[k])
    if str(k) in score_dict:
        return float(score_dict[str(k)])
    try:
        int_keys = [int(x) for x in score_dict.keys()]
        if not int_keys:
            return float(next(iter(score_dict.values())))
        best_k = max(int_keys)
        return float(score_dict.get(best_k, score_dict.get(str(best_k), 0.0)))
    except Exception:
        try:
            return float(next(iter(score_dict.values())))
        except Exception:
            return 0.0

def _to_float_dict(d: Dict[Any, Any]) -> Dict[Any, float]:
    return {(int(k) if str(k).isdigit() else k): float(v) for k, v in d.items()}

# -------------------------
# Core runner
# -------------------------
def run_problem(
    data_type: str,
    # teacher
    teacher_client: OpenAI,
    teacher_name: str,
    teacher_temperature: float,
    # student
    student_client: OpenAI,
    student_name: str,
    student_temperature: float,
    # guide
    guide_client: OpenAI,
    guide_name: str,
    guide_temperature: float,
    # problems
    interaction_problem: Dict[str, Any],
    evaluation_problem: Dict[str, Any],
    # eval knobs
    k_values: List[int],
    pass_k: int,
    num_samples: int,
    # RLHF/GRPO sampling
    rlhf_samples_student: int,
    rlhf_samples_guide: int,
    # schedule
    assessment_turn: int,
    start_from_turn: int = 1,
    total_turns: int = 10,
    eps: float = 0.2,
    use_cot: bool = False,
    is_ood: bool = False,
    # GRPO / DPO knobs
    use_delta_reward: bool = False,
    dpo_margin_min: float = 0.01,
    dpo_margin_max: float = 1.0,
    dpo_max_pairs_per_turn: Optional[int] = None,
) -> Tuple[
    List[Dict[str, Any]],   # chosen_list (SFT)
    List[Dict[str, Any]],   # pairs (DPO)
    List[Dict[str, Any]],   # grpo_groups
    int, int,               # teacher prompt/completion tks
    int, int,               # student prompt/completion tks
    int, int,               # guide prompt/completion tks
]:
    """
    One problem end-to-end with a *guide* model contributing candidates alongside the student.
    """
    assert interaction_problem["id"] == evaluation_problem["id"], "pid of interaction and evaluation doesn't match!"
    pid = interaction_problem["id"]

    total_teacher_prompt_tks = total_teacher_completion_tks = 0
    total_student_prompt_tks = total_student_completion_tks = 0
    total_guide_prompt_tks = total_guide_completion_tks = 0

    # Initialize / resume
    if start_from_turn <= 1:
        texts: List[Any] = [_problem_greeting(data_type)]
        cur_start = 1
    else:
        texts = ["Hi there! Let's pick up the conversation."]
        cur_start = 1
        logger.warning(f"[{pid}] resume requested but no transcript found; starting fresh")

    pairs: List[Dict[str, Any]] = []
    chosen_list: List[Dict[str, Any]] = []
    grpo_groups: List[Dict[str, Any]] = []

    # Turn-baseline cache (for delta rewards)
    last_baseline_scores: Dict[Any, float] = {}
    last_baseline_valid = False

    for turn in range(cur_start, total_turns, 2):
        # Optional assessment injected into previous text
        assessment_injected = False
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
            assessment_injected = True

        # ---- Candidate generation: student + guide
        student_replies, sp_tks, sc_tks = interact(
            True, 
            texts, 
            interaction_problem, 
            rlhf_samples_student,
            data_type, 
            student_client, 
            student_name, 
            student_temperature, 
            use_cot
        )
        total_student_prompt_tks += sp_tks
        total_student_completion_tks += sc_tks

        guide_replies, gp_tks, gc_tks = interact(
            True, 
            texts, 
            interaction_problem, 
            rlhf_samples_guide,
            data_type, 
            guide_client, 
            guide_name, 
            guide_temperature, 
            use_cot
        )
        total_guide_prompt_tks += gp_tks
        total_guide_completion_tks += gc_tks

        # Label sources and concatenate
        # Each element: (student_reply, "student"/"guide")
        candidates = [(r, "student") for r in student_replies] + [(r, "guide") for r in guide_replies]

        # ---- Teacher answers for each candidate
        teacher_replies = []
        for student_reply, _src in candidates:
            teacher_reply, tp_tks, tc_tks = interact(
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
            total_teacher_prompt_tks += tp_tks
            total_teacher_completion_tks += tc_tks

        # ---- Baseline (reuse if allowed)
        if use_delta_reward:
            if last_baseline_valid and not assessment_injected:
                base_scores = last_baseline_scores
            else:
                convo_baseline = msgs_to_string(texts)
                base_scores_raw, _, pb, cb = evaluate(
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
                total_student_prompt_tks += pb
                total_student_completion_tks += cb
        else:
            base_scores = {}

        # ---- Evaluate each (candidate Q, teacher A)
        pass_at_k_scores_current: List[Dict[Any, float]] = []
        for (reply_msg, _src), teacher_reply in zip(candidates, teacher_replies):
            # current: history + candidate Q/A
            convo_current = msgs_to_string(texts + [reply_msg, teacher_reply])
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

        # ---- Prompt messages (used for SFT/DPO/GRPO payloads)
        sys_prompt = _select_sys_prompt(data_type, use_cot, True, evaluation_problem)
        prompt_messages = [{"role": "system", "content": sys_prompt}] + format_messages(texts, starting_with_user=True)

        # ---------- GRPO group ----------
        group = {
            "ex_id": pid,
            "problem": evaluation_problem,
            "turn": turn,
            "prompt_messages": prompt_messages,
            "prompt_type": "messages",
            "k": len(candidates),
            "pass_k_key": pass_k,
            "candidates": [],
        }
        # same baseline for all cands in this turn if enabled
        for (reply_msg, src), cur_scores in zip(candidates, pass_at_k_scores_current):
            q_text = _extract_text(reply_msg)
            if use_delta_reward:
                reward = _get_score(cur_scores, pass_k) - _get_score(base_scores, pass_k)
            else:
                reward = _get_score(cur_scores, pass_k)

            metrics_current = {int(k) if str(k).isdigit() else k: float(v) for k, v in cur_scores.items()}
            metrics_baseline = (
                {int(k) if str(k).isdigit() else k: float(v) for k, v in base_scores.items()} if use_delta_reward else {}
            )
            metrics_delta = (
                {k: float(metrics_current.get(k, 0.0) - metrics_baseline.get(k, 0.0)) for k in metrics_current.keys()}
                if use_delta_reward else {}
            )
            group["candidates"].append({
                "text": q_text,
                "source": src,  # <-- student | guide
                "reward": float(reward),
                "metrics_current": metrics_current,
                "metrics_baseline": metrics_baseline,
                "metrics_delta": metrics_delta,
            })
        grpo_groups.append(group)
        # --------------------------------

        # ---- Choose best by CURRENT pass@k to roll convo
        current_vals = [_get_score(s, pass_k) for s in pass_at_k_scores_current]
        max_value = max(current_vals) if current_vals else 0.0
        best_idx = current_vals.index(max_value) if current_vals else 0
        best_pass_at_k_score = pass_at_k_scores_current[best_idx] if pass_at_k_scores_current else {}
        # hstack the reply+teacher chosen to continue
        appended_idx = best_idx
        if random.random() < 1 - eps:
            texts += [candidates[best_idx][0], teacher_replies[best_idx]]
            propensity = 1 - eps
        else:
            non_best = [i for i in range(len(candidates)) if i != best_idx]
            idx = random.choice(non_best) if non_best else best_idx
            texts += [candidates[idx][0], teacher_replies[idx]]
            appended_idx = idx
            propensity = eps / max(1, len(candidates))

        # SFT item
        chosen = candidates[best_idx][0]

        # ---- SFT gating (quality)
        if max_value >= 0.85:
            chosen_list.append({
                "prompt": prompt_messages,
                "response": chosen,
                "problem": evaluation_problem,
                "turn": turn,
                "propensity": propensity
            })

        # ---- DPO pairs (margin band + optional capping)
        deltas_for_pairs = []
        best_val = _get_score(best_pass_at_k_score, pass_k)
        for idx, ((reply, _src), cur_scores) in enumerate(zip(candidates, pass_at_k_scores_current)):
            if reply is chosen:
                continue
            cur_val = _get_score(cur_scores, pass_k)
            margin = best_val - cur_val
            deltas_for_pairs.append((margin, reply, cur_scores))

        min_m = dpo_margin_min if dpo_margin_min is not None else 0.05
        max_m = dpo_margin_max if dpo_margin_max is not None else 1.0
        filtered = [(m, r, s) for (m, r, s) in deltas_for_pairs if (m >= min_m and m <= max_m)]

        if dpo_max_pairs_per_turn is not None and len(filtered) > dpo_max_pairs_per_turn:
            filtered.sort(key=lambda x: x[0])  # harder (smaller margin) first
            filtered = filtered[:dpo_max_pairs_per_turn]

        for margin, rejected_reply, rejected_score in filtered:
            if use_delta_reward:
                chosen_reward_dict = {
                    "current": best_pass_at_k_score,
                    "baseline": base_scores,
                    "delta@pass_k": _get_score(best_pass_at_k_score, pass_k) - _get_score(base_scores, pass_k)
                }
                rejected_reward_dict = {
                    "current": rejected_score,
                    "baseline": base_scores,
                    "delta@pass_k": _get_score(rejected_score, pass_k) - _get_score(base_scores, pass_k)
                }
            else:
                chosen_reward_dict = best_pass_at_k_score
                rejected_reward_dict = rejected_score

            pairs.append({
                "prompt": prompt_messages,
                "chosen_response": chosen,
                "rejected_response": rejected_reply,
                "chosen_reward": chosen_reward_dict,
                "rejected_reward": rejected_reward_dict,
                "problem": evaluation_problem,
                "turn": turn
            })

        # ---- Prepare baseline for next turn (the branch we actually appended)
        if use_delta_reward and pass_at_k_scores_current:
            last_baseline_scores = pass_at_k_scores_current[appended_idx] or {}
            last_baseline_valid = True

        logger.debug(f'pid-{evaluation_problem["id"]} turn-{turn} best pass@{pass_k}={_get_score(best_pass_at_k_score, pass_k):.3f}')

    return (
        chosen_list,
        pairs,
        grpo_groups,
        total_teacher_prompt_tks, total_teacher_completion_tks,
        total_student_prompt_tks, total_student_completion_tks,
        total_guide_prompt_tks, total_guide_completion_tks,
    )

# -------------------------
# Hydra entry
# -------------------------
@hydra.main(version_base=None, config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    start_time = time.perf_counter()

    # envs
    set_envs(cfg.paths.openai_api_key_fpath, cfg.paths.hf_home)

    # Output files (include guide in filename to avoid collisions)
    base_dir = os.path.join(str(cfg.paths.data), str(cfg.teacher.model), str(cfg.experiment.dataset))
    split_dir = os.path.join(base_dir, f"splits_{cfg.gen.pass_at_k_threshold}")

    # Prefer same naming convention as your rlhf_guide (student_guide_*)
    def _fname(stem: str, ext: str) -> str:
        return os.path.join(
            base_dir, "full",
            f"{cfg.student.model}__{cfg.guide.model}__{stem}_iter{cfg.gen.dagger_iter}_{cfg.data.split}.{ext}"
        )

    fpath_sft = _fname("sft", "jsonl")
    fpath_dpo = _fname("dpo", "jsonl")
    fpath_grpo = _fname("grpo", "jsonl")

    existing_pids = set()
    count_dpo = count_sft = count_grpo = 0

    # Resume dedupe by dagger_iter for SFT/DPO/GRPO
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
        logger.info(f"Loaded {count_grpo} existing GRPO-{cfg.gen.dagger_iter} groups from {fpath_grpo}")

    # Problems (match your grpo.py sourcing)
    problems_fpath = os.path.join(
        split_dir,
        f"{cfg.student.model}_{cfg.data.split}.json",
    )
    logger.info(f"Loading interaction problems: {problems_fpath}")
    data = load_json_resilient(problems_fpath)
    interaction_problems: List[Dict[str, Any]] = [
        p for p in data if p["id"] not in existing_pids and str(p["id"]) not in existing_pids
    ][: max(0, cfg.data.data_amount - len(existing_pids))]
    logger.info(f"Interaction problems: finished {len(existing_pids)} | pending {len(interaction_problems)}")

    # OOD (optional)
    evaluation_problems = interaction_problems
    if cfg.experiment.ood:
        ood_problems_path = os.path.join(
            split_dir,
            f"{cfg.student.model}_ood_{cfg.data.split}.json",
        )
        logger.info(f"Loading evaluation (OOD) problems: {ood_problems_path}")
        data = load_json_resilient(ood_problems_path)
        evaluation_problems = [
            p for p in data if p["id"] not in existing_pids and str(p["id"]) not in existing_pids
        ][: max(0, cfg.data.data_amount - len(existing_pids))]
        logger.info(f"Evaluation (OOD) problems: finished {len(existing_pids)} | pending {len(evaluation_problems)}")

    # Clients
    teacher_name, student_name, guide_name = cfg.teacher.model, cfg.student.model, cfg.guide.model
    teacher_client = build_client(teacher_name, cfg.teacher.gpus, cfg.teacher.server)
    student_client = build_client(student_name, cfg.student.gpus, cfg.student.server, lora=cfg.student.lora)
    guide_client = build_client(guide_name, cfg.guide.gpus, cfg.guide.server)
    logger.info("Teacher=%s | Student=%s | Guide=%s", teacher_name, student_name, guide_name)

    # Files
    f_dpo = open(fpath_dpo, "a" if os.path.exists(fpath_dpo) else "w")
    f_sft = open(fpath_sft, "a" if os.path.exists(fpath_sft) else "w")
    f_grpo = open(fpath_grpo, "a" if os.path.exists(fpath_grpo) else "w")

    # Knobs
    use_delta_reward = bool(getattr(cfg.gen, "use_delta_reward", False))
    dpo_margin_min = float(getattr(cfg.gen, "dpo_margin_min", 0.01))
    dpo_margin_max = float(getattr(cfg.gen, "dpo_margin_max", 1.0))
    dpo_max_pairs_per_turn = getattr(cfg.gen, "dpo_max_pairs_per_turn", None)
    if dpo_max_pairs_per_turn is not None:
        dpo_max_pairs_per_turn = int(dpo_max_pairs_per_turn)

    # K per source (defaults: both = cfg.gen.rlhf_samples)
    rlhf_samples_student = int(getattr(cfg.gen, "rlhf_samples_student", getattr(cfg.gen, "rlhf_samples", 5)))
    rlhf_samples_guide = int(getattr(cfg.gen, "rlhf_samples_guide", getattr(cfg.gen, "rlhf_samples", 5)))

    # Token tallies
    total_teacher_prompt_tks = total_teacher_completion_tks = 0
    total_student_prompt_tks = total_student_completion_tks = 0
    total_guide_prompt_tks = total_guide_completion_tks = 0

    pbar = tqdm(total=len(interaction_problems), desc="Problems done", dynamic_ncols=True)

    # Parallel run (chunked)
    with ThreadPoolExecutor(max_workers=cfg.experiment.num_workers) as executor:
        for i in range(0, len(interaction_problems), cfg.experiment.chunk_size):
            chunk_inter = interaction_problems[i: i + cfg.experiment.chunk_size]
            chunk_eval = evaluation_problems[i: i + cfg.experiment.chunk_size]

            futures = []
            for interaction_problem, evaluation_problem in zip(chunk_inter, chunk_eval):
                futures.append(executor.submit(
                    run_problem,
                    cfg.data.type,
                    # teacher
                    teacher_client, teacher_name, cfg.teacher.temperature,
                    # student
                    student_client, cfg.student.lora.name or student_name, cfg.student.temperature,
                    # guide
                    guide_client, guide_name, cfg.guide.temperature,
                    # problems
                    interaction_problem, evaluation_problem,
                    # eval
                    cfg.gen.k_values, cfg.gen.pass_k, cfg.gen.num_samples,
                    # RLHF K
                    rlhf_samples_student, rlhf_samples_guide,
                    # schedule
                    cfg.experiment.assessment_turn,
                    cfg.experiment.start_from_turn,
                    cfg.experiment.total_turns,
                    cfg.gen.eps,
                    cfg.experiment.cot,
                    cfg.experiment.ood,
                    # GRPO/DPO knobs
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
                    t_pp, t_cp,
                    s_pp, s_cp,
                    g_pp, g_cp
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

                # Tokens
                total_teacher_prompt_tks += t_pp
                total_teacher_completion_tks += t_cp
                total_student_prompt_tks += s_pp
                total_student_completion_tks += s_cp
                total_guide_prompt_tks += g_pp
                total_guide_completion_tks += g_cp
                pbar.update(1)
                logger.info(f"Collected — DPO: {count_dpo} | SFT: {count_sft} | GRPO: {count_grpo} ")
                logger.info("%s", pbar.format_meter(n=pbar.n, total=pbar.total, elapsed=pbar.format_dict["elapsed"], prefix=pbar.desc + ": "))

    # Close files
    f_dpo.close()
    f_sft.close()
    f_grpo.close()
    pbar.close()

    # Summary
    logger.info(f"------- {cfg.data.split} -------")
    logger.info(f"Saved {count_dpo} preference pairs to {fpath_dpo}")
    logger.info(f"Saved {count_sft} SFT examples to {fpath_sft}")
    logger.info(f"Saved {count_grpo} GRPO groups to {fpath_grpo}")
    logger.info(
        f"Teacher-{cfg.teacher.model} tokens — prompt={total_teacher_prompt_tks}, completion={total_teacher_completion_tks}"
    )
    logger.info(
        f"Student-{cfg.student.model} tokens — prompt={total_student_prompt_tks}, completion={total_student_completion_tks}"
    )
    logger.info(
        f"Guide-{cfg.guide.model} tokens — prompt={total_guide_prompt_tks}, completion={total_guide_completion_tks}"
    )

    elapsed = time.perf_counter() - start_time
    logger.info("Total time elapsed: %s", timedelta(seconds=int(elapsed)))


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
