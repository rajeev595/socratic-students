"""
EXPERIMENTAL -- not used in the paper and not maintained.

Variant of codes.data.collect_self_odqs that scores a candidate question by a short look-ahead
rollout (gen.rollout_horizon further turns, discounted by gen.rollout_gamma) instead of by the
Pass@k right after the teacher's reply.

Usage:
    python -m codes.experimental.collect_self_odqs_rollout student.model=qwen7 data.type=math \
        experiment.dataset=gsm8k data.name=odqs_rollout data.split=train experiment.name=self_odqs_rollout
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
    # try string form
    if str(k) in score_dict:
        return float(score_dict[str(k)])
    ks = list(score_dict.keys())
    # try int-cast on all keys and take max
    try:
        int_keys = [int(x) for x in ks]
        best_k = max(int_keys)
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
# k-step rollout (max over discounted evals)
# -------------------------
def rollout_max_reward(
    *,
    data_type: str,
    texts: List[Any],
    initial_student_reply: Any,
    initial_teacher_reply: Any,
    cur_scores_s0: Dict[Any, float],   # pass@k dict at s=0 (already computed)
    evaluation_problem: Dict[str, Any],
    student_client: OpenAI,
    student_name: str,
    student_temperature: float,
    teacher_client: OpenAI,
    teacher_name: str,
    teacher_temperature: float,
    k_values: List[int],
    pass_k: int,
    num_samples: int,
    num_candidates: int,
    horizon: int,
    eps: float,
    use_scaffold: bool,
    is_ood: bool,
    gamma: float,
) -> Tuple[float, List[float], Dict[str, Any], Tuple[int, int, int, int]]:
    """
    Rollout for ONE candidate question (Q):

      ctx_0 = texts + [Q, A_1]
      eval_0 = pass@k(ctx_0)    # from cur_scores_s0
      then for s = 1..horizon-1:
        - propose num_candidates follow-up questions
        - teacher answers each
        - evaluate each (pass@k)
        - pick one (eps-greedy on pass@k) and append
        - eval_s = pass@k of that chosen branch
        - if eval_s <= eval_{s-1}, STOP (delta-style gating)

    Reward:
      r = max_s (gamma^s * eval_s)

    Returns:
      reward (float in [0,1]),
      evals (list of per-step evals),
      trace,
      token usage tuple
    """
    # starting context after Q/A
    ctx = list(texts) + [initial_student_reply, initial_teacher_reply]

    total_teacher_prompt_tks = total_teacher_completion_tks = 0
    total_student_prompt_tks = total_student_completion_tks = 0

    evals: List[float] = []

    # --- helper: evaluate current context and return pass@k
    def _eval_ctx(cur_ctx: List[Any]) -> Tuple[float, int, int]:
        conv = msgs_to_string(cur_ctx)
        scores_raw, _, p_tks, c_tks = evaluate(
            prev_convo=conv,
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
        score = _get_score(_to_float_dict(scores_raw), pass_k)
        return float(score), p_tks, c_tks

    # ---- s = 0: use cached scores (no extra evaluate call)
    eval_0 = _get_score(cur_scores_s0, pass_k)
    evals.append(eval_0)

    # ---- s = 1..H-1: roll forward while improving
    for s in range(1, horizon):
        prev_eval = evals[-1]

        # Student proposes follow-up questions
        student_replies, p_tks, c_tks = interact(
            True, ctx, evaluation_problem, num_candidates, data_type,
            student_client, student_name, student_temperature, use_scaffold
        )
        total_student_prompt_tks += p_tks
        total_student_completion_tks += c_tks

        if not student_replies:
            break

        # Teacher answers each suggestion
        teacher_replies = []
        for sr in student_replies:
            tr, p_tks, c_tks = interact(
                False, ctx + [sr], evaluation_problem, 1, data_type,
                teacher_client, teacher_name, teacher_temperature, use_scaffold
            )
            teacher_replies.append(tr[0])
            total_teacher_prompt_tks += p_tks
            total_teacher_completion_tks += c_tks

        # Evaluate each candidate follow-up branch
        cand_scores: List[float] = []
        for sr, tr in zip(student_replies, teacher_replies):
            score_s, p_eval, c_eval = _eval_ctx(ctx + [sr, tr])
            cand_scores.append(score_s)
            total_student_prompt_tks += p_eval
            total_student_completion_tks += c_eval

        if not cand_scores:
            break

        # Pick next step via eps-greedy on immediate score
        best_idx = int(max(range(len(cand_scores)), key=lambda i: cand_scores[i]))
        if random.random() < 1 - eps:
            idx = best_idx
        else:
            non_best = [i for i in range(len(cand_scores)) if i != best_idx]
            idx = random.choice(non_best) if non_best else best_idx

        # Append chosen follow-up Q/A to ctx
        ctx += [student_replies[idx], teacher_replies[idx]]
        eval_s = cand_scores[idx]
        evals.append(eval_s)

        # --- delta gating: stop if no improvement
        if eval_s <= prev_eval:
            break

    # --- reward: max over discounted evals
    if evals:
        discounted = [(gamma ** s) * v for s, v in enumerate(evals)]
        reward = max(discounted)
        argmax_step = int(max(range(len(discounted)), key=lambda i: discounted[i]))
    else:
        discounted = []
        reward = 0.0
        argmax_step = None

    trace = {
        "per_step_evals": evals,
        "discounted_evals": discounted,
        "final_context": ctx,
        "argmax_step": argmax_step,
    }
    tk_usage = (
        total_student_prompt_tks, total_student_completion_tks,
        total_teacher_prompt_tks, total_teacher_completion_tks
    )
    return float(reward), evals, trace, tk_usage

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
    num_candidates: int,
    assessment_turn: int,
    start_from_turn: int = 1,
    total_turns: int = 10,
    eps: float = 0.2,
    use_scaffold: bool = False,
    is_ood: bool = False,
    # OLD knobs (kept for compatibility)
    use_delta_reward: bool = False,
    dpo_margin_min: float = 0.01,
    dpo_margin_max: float = 1.0,
    dpo_max_pairs_per_turn: int = None,
    # NEW rollout knobs
    rollout_horizon: int = 1,
    rollout_gamma: float = 1.0,
    sft_gate_min: float = 0.85,
) -> Tuple[
    List[Dict[str, Any]],   # chosen_list (SFT)
    List[Dict[str, Any]],   # pairs (DPO)
    List[Dict[str, Any]],   # grpo_groups
    int, int, int, int
]:
    """
    Run a single problem end-to-end: student↔teacher interaction and inline evaluation.

    Rollout semantics:
      - For each candidate question Q_i, we compute a k-step rollout reward:
            r_i = max_s gamma^s * eval_s
        with delta-style gating (stop when eval stops improving).
      - We still save GRPO, SFT, DPO in the same schema as before:
            * GRPO candidates: text, reward, metrics_current, metrics_baseline, metrics_delta
            * SFT: prompt/response/problem/turn/propensity
            * DPO: chosen_reward / rejected_reward dicts exactly as before
        plus some extra rollout fields (reward, per_step_evals) for analysis.
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
    grpo_groups: List[Dict[str, Any]] = []

    # --- Baseline reuse cache across turns (for old delta behavior)
    last_baseline_scores: Dict[Any, float] = {}
    last_baseline_valid: bool = False

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
            logger.debug(f"[{pid}] assessment injected at turn {turn}")

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
            break

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
                use_scaffold
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
        pass_at_k_scores_current: List[Dict[Any, float]] = []
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

        # ---- Prompt messages used to elicit candidate questions
        sys_prompt = _select_sys_prompt(data_type, use_scaffold, True, evaluation_problem)
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
            "k": num_candidates,
            "pass_k_key": pass_k,  # int
            "candidates": [],
        }

        # ---- Rollout for each candidate & build rewards
        cand_rewards: List[float] = []
        cand_traces: List[Dict[str, Any]] = []

        for reply, teacher_reply, cur_scores in zip(student_replies, teacher_replies, pass_at_k_scores_current):
            q_text = _extract_text(reply)

            # metrics payloads (same format as before)
            metrics_current = {int(k) if str(k).isdigit() else k: float(v) for k, v in cur_scores.items()}
            metrics_baseline = (
                {int(k) if str(k).isdigit() else k: float(v) for k, v in base_scores.items()}
                if use_delta_reward else {}
            )
            metrics_delta = (
                {k: float(metrics_current.get(k, 0.0) - metrics_baseline.get(k, 0.0)) for k in metrics_current.keys()}
                if use_delta_reward else {}
            )

            # --- k-step rollout reward (r = max_s gamma^s * eval_s)
            R, evals, trace, (sp, sc, tp, tc) = rollout_max_reward(
                data_type=data_type,
                texts=texts,
                initial_student_reply=reply,
                initial_teacher_reply=teacher_reply,
                cur_scores_s0=cur_scores,
                evaluation_problem=evaluation_problem,
                student_client=student_client,
                student_name=student_name,
                student_temperature=student_temperature,
                teacher_client=teacher_client,
                teacher_name=teacher_name,
                teacher_temperature=teacher_temperature,
                k_values=k_values,
                pass_k=pass_k,
                num_samples=num_samples,
                num_candidates=num_candidates,
                horizon=rollout_horizon,
                eps=eps,
                use_scaffold=use_scaffold,
                is_ood=is_ood,
                gamma=rollout_gamma,
            )
            cand_rewards.append(R)
            cand_traces.append(trace)

            total_student_prompt_tks += sp
            total_student_completion_tks += sc
            total_teacher_prompt_tks += tp
            total_teacher_completion_tks += tc

            # GRPO candidate: keep same keys, just use rollout reward
            group["candidates"].append({
                "text": q_text,
                "reward": float(R),
                "metrics_current": metrics_current,
                "metrics_baseline": metrics_baseline,
                "metrics_delta": metrics_delta,
                # extra rollout info (can be ignored by old readers)
                "rollout_evals": evals,
                "rollout_gamma": rollout_gamma,
                "rollout_horizon": rollout_horizon,
            })

        grpo_groups.append(group)
        # ---------------------------------------

        # ---- Choose best by k-step rollout reward to roll convo forward
        if cand_rewards:
            best_idx = int(max(range(len(cand_rewards)), key=lambda i: cand_rewards[i]))
            best_reward = cand_rewards[best_idx]
        else:
            best_idx = 0
            best_reward = 0.0

        best_pass_at_k_score = pass_at_k_scores_current[best_idx] if pass_at_k_scores_current else {}
        chosen = student_replies[best_idx]

        # ---- Exploration vs exploitation (propensity fix)
        appended_idx = best_idx
        if random.random() < 1 - eps:
            texts += [student_replies[best_idx], teacher_replies[best_idx]]
            propensity = 1 - eps
        else:
            non_best = [i for i in range(num_candidates) if i != best_idx]
            idx = random.choice(non_best) if non_best else best_idx
            texts += [student_replies[idx], teacher_replies[idx]]
            appended_idx = idx
            # probability mass spread over (K-1) non-best choices
            propensity = eps / max(1, num_candidates)

        # ---- SFT collection gated by rollout reward (same structure as before)
        if best_reward >= sft_gate_min:
            chosen_list.append({
                "prompt": prompt_messages,
                "response": chosen,
                "problem": evaluation_problem,
                "turn": turn,
                "propensity": propensity,
                # extra rollout info
                "rollout_reward": float(best_reward),
                "rollout_gamma": rollout_gamma,
                "rollout_horizon": rollout_horizon,
            })

        # ---- DPO pair mining (margin based on rollout reward, but reward dict
        #      format remains the same as before)
        deltas_for_pairs = []
        best_val = best_reward
        for j, (Rj, reply_j, score_cur_j) in enumerate(
            zip(cand_rewards, student_replies, pass_at_k_scores_current)
        ):
            if j == best_idx:
                continue
            margin = best_val - Rj
            deltas_for_pairs.append((margin, reply_j, score_cur_j, Rj, cand_traces[j]))

        # Filter by margin band (same knobs, now on rollout margin)
        min_m = dpo_margin_min if dpo_margin_min is not None else 0.05
        max_m = dpo_margin_max if dpo_margin_max is not None else 1.0
        filtered = [
            (m, r, s, Rj, trc)
            for (m, r, s, Rj, trc) in deltas_for_pairs
            if (m >= min_m and m <= max_m)
        ]

        # If we cap pairs per turn, bias selection toward harder (smaller margin)
        if dpo_max_pairs_per_turn is not None and len(filtered) > dpo_max_pairs_per_turn:
            filtered.sort(key=lambda x: x[0])  # harder first
            filtered = filtered[:dpo_max_pairs_per_turn]

        # Save pairs
        for margin, reply_j, score_cur_j, Rj, trace_j in filtered:
            # rewards for DPO: keep format identical to old version
            if use_delta_reward:
                chosen_reward_dict = {
                    "current": best_pass_at_k_score,
                    "baseline": base_scores,
                    "delta@pass_k": _get_score(best_pass_at_k_score, pass_k) - _get_score(base_scores, pass_k)
                }
                rejected_reward_dict = {
                    "current": score_cur_j,
                    "baseline": base_scores,
                    "delta@pass_k": _get_score(score_cur_j, pass_k) - _get_score(base_scores, pass_k)
                }
            else:
                # non-delta: just pass@k dict, as before
                chosen_reward_dict = best_pass_at_k_score
                rejected_reward_dict = score_cur_j

            pairs.append({
                "prompt": prompt_messages,
                "chosen_response": chosen,
                "rejected_response": reply_j,
                "chosen_reward": chosen_reward_dict,
                "rejected_reward": rejected_reward_dict,
                "problem": evaluation_problem,
                "turn": turn,
                # extra rollout info (can be ignored by old code)
                "chosen_rollout_reward": float(best_reward),
                "rejected_rollout_reward": float(Rj),
                "rollout_gamma": rollout_gamma,
                "rollout_horizon": rollout_horizon,
            })

        logger.debug(f'pid-{evaluation_problem["id"]} turn-{turn} rollout_best={best_reward:.3f}')

        # ---- Update reusable baseline for NEXT turn (old delta path)
        if use_delta_reward and pass_at_k_scores_current:
            last_baseline_scores = pass_at_k_scores_current[appended_idx] or {}
            last_baseline_valid = True

    return (
        chosen_list,
        pairs,
        grpo_groups,
        total_teacher_prompt_tks, total_teacher_completion_tks,
        total_student_prompt_tks, total_student_completion_tks
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
    base_dir = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        str(cfg.experiment.dataset),
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
    fpath_grpo = os.path.join(
        base_dir, name,
        f"{cfg.student.model}_scaffolded_grpo_iter{cfg.gen.iteration}_{cfg.data.split}.jsonl" if cfg.experiment.scaffolded else f"{cfg.student.model}_grpo_iter{cfg.gen.iteration}_{cfg.data.split}.jsonl"
    )

    existing_pids = set()
    count_dpo, count_sft, count_grpo = 0, 0, 0

    # Resume dedupe by iteration for SFT/DPO/GRPO
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

    if os.path.exists(fpath_grpo):
        with open(fpath_grpo, "r") as f:
            for line in f:
                ex = json.loads(line)
                count_grpo += 1
                if ex.get("iteration") == cfg.gen.iteration:
                    existing_pids.add(ex["problem"]["id"])
        logger.info(f"Loaded {count_grpo} existing GRPO-{cfg.gen.iteration} examples from {fpath_grpo}")

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

    # OOD problems
    evaluation_problems = interaction_problems
    if cfg.experiment.ood:
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
        student_name, cfg.student.gpus, cfg.student.server, lora=cfg.student.lora
    )
    logger.info("Teacher=%s | Student=%s", teacher_name, student_name)

    total_teacher_prompt_tks = total_teacher_completion_tks = 0
    total_student_prompt_tks = total_student_completion_tks = 0

    # Files
    f_dpo = open(fpath_dpo, "a" if os.path.exists(fpath_dpo) else "w")
    f_sft = open(fpath_sft, "a" if os.path.exists(fpath_sft) else "w")
    f_grpo = open(fpath_grpo, "a" if os.path.exists(fpath_grpo) else "w")

    # OLD knobs from cfg (kept)
    use_delta_reward = bool(getattr(cfg.gen, "use_delta_reward", False))
    dpo_margin_min = float(getattr(cfg.gen, "dpo_margin_min", 0.01))
    dpo_margin_max = float(getattr(cfg.gen, "dpo_margin_max", 1.0))
    dpo_max_pairs_per_turn = getattr(cfg.gen, "dpo_max_pairs_per_turn", None)
    if dpo_max_pairs_per_turn is not None:
        dpo_max_pairs_per_turn = int(dpo_max_pairs_per_turn)

    # NEW rollout + gating knobs
    rollout_horizon = int(getattr(cfg.gen, "rollout_horizon", 1))
    rollout_gamma = float(getattr(cfg.gen, "rollout_gamma", 1.0))
    sft_gate_min = float(getattr(cfg.gen, "sft_gate_min", 0.85))

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
                    cfg.student.lora.name or student_name,
                    cfg.student.temperature,
                    interaction_problem,
                    evaluation_problem,
                    cfg.gen.k_values,
                    cfg.gen.pass_k,
                    cfg.gen.num_samples,
                    cfg.gen.num_candidates,
                    cfg.experiment.assessment_turn,
                    cfg.experiment.start_from_turn,
                    cfg.experiment.total_turns,
                    cfg.gen.eps,
                    cfg.experiment.scaffolded,
                    cfg.experiment.ood,
                    use_delta_reward,
                    dpo_margin_min,
                    dpo_margin_max,
                    dpo_max_pairs_per_turn,
                    rollout_horizon,
                    rollout_gamma,
                    sft_gate_min,
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
                    ex["iteration"] = cfg.gen.iteration
                    f_dpo.write(json.dumps(ex, ensure_ascii=False) + "\n")
                count_dpo += len(pairs)

                # Save SFT
                for ex in chosen:
                    ex["iteration"] = cfg.gen.iteration
                    f_sft.write(json.dumps(ex, ensure_ascii=False) + "\n")
                count_sft += len(chosen)

                # Save GRPO
                for group in grpo_groups:
                    group["iteration"] = cfg.gen.iteration
                    f_grpo.write(json.dumps(group, ensure_ascii=False) + "\n")
                count_grpo += len(grpo_groups)

                # Token usage
                total_teacher_prompt_tks += teacher_prompt_tks
                total_teacher_completion_tks += teacher_completion_tks
                total_student_prompt_tks += student_prompt_tks
                total_student_completion_tks += student_completion_tks
                pbar.update(1)
                logger.info(f"No. of examples collected DPO: {count_dpo} | SFT: {count_sft} | GRPO: {count_grpo}")
                logger.info(
                    "%s",
                    pbar.format_meter(
                        n=pbar.n,
                        total=pbar.total,
                        elapsed=pbar.format_dict["elapsed"],
                        prefix=pbar.desc + ": ",
                    )
                )

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
