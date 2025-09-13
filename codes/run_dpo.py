import os
import json
import logging
from typing import Any, List, Dict

import hydra
from omegaconf import DictConfig

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from codes.data.utils import format_messages
from codes.prompts.utils import _select_sys_prompt


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)


def load_transcript(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)  # list of {"role": ..., "turn": i, "content": ...}


def build_prompt_for_student_turn(
    problem: Dict[str, Any],
    data_type: str,
    use_cot: bool,
    texts_before_turn: List[str],
    tokenizer: Any
) -> str:
    """Recreate the exact chat-formatted prompt the student saw at a student turn."""
    sys_prompt = _select_sys_prompt(data_type, use_cot, True, problem)
    messages = [{"role": "system", "content": sys_prompt}] + \
               format_messages(texts_before_turn, starting_with_user=True)
    # Important: same chat template and add_generation_prompt=True
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt_str


def extract_pairs_for_problem(
    problem: Dict[str, Any],
    data_type: str,
    use_cot: bool,
    tokenizer,
    transcript1_fpath: str,
    transcript2_fpath: str,
    results1: Dict[str, Any],
    results2: Dict[str, Any],
    passk_key: str,
) -> List[Dict[str, Any]]:
    """
    Build DPO pairs for one problem id.
    - canonical prompt: rebuilt from the 7B transcript prefix up to each student turn
    - chosen/rejected: utility-aligned if both pass@k available, else style (32B > 7B)
    """
    if not (os.path.exists(transcript1_fpath) and os.path.exists(transcript2_fpath)):
        return []

    msgs1 = load_transcript(transcript1_fpath)
    msgs2 = load_transcript(transcript2_fpath)

    # Sanity: total turns from file (don’t trust config)
    total_turns = min(len(msgs1), len(msgs2))
    pairs = []

    # Student turns are odd indices: 1,3,5,... up to < total_turns
    for turn in range(1, total_turns, 2):
        # Get the student utterance at this turn from both transcripts
        response1  = msgs1[turn]["content"].strip() if turn < len(msgs1) else ""
        response2 = msgs2[turn]["content"].strip() if turn < len(msgs2) else ""
        if not response1 or not response2 or response1 == response2:
            continue

        # Rebuild the *canonical* prompt from the 7B history up to BEFORE this turn
        texts_before = [m["content"] for m in msgs1 if m["turn"] < turn]

        prompt = build_prompt_for_student_turn(
            problem=problem,
            data_type=data_type,
            use_cot=use_cot,
            texts_before_turn=texts_before,
            tokenizer=tokenizer,
        )

        passk_score1  = results1[str(problem["id"])]["results"][str(turn)][passk_key]
        passk_score2 = results2[str(problem["id"])]["results"][str(turn)][passk_key]

        # Preference: utility if available and non-tie, else 32B > 7B        
        if passk_score1 > passk_score2:
            chosen, rejected = response1, response2
        else:
            chosen, rejected = response2, response1

        pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "meta": {
                "problem_id": problem["id"],
                "turn_index": turn,
                "passk_score1": passk_score1,
                "passk_score2": passk_score2,
            }
        })

    return pairs


def collect_prefrence_pairs(
    tokenizer,
    problems_fpath: str,
    data_dir1: str,
    data_dir2: str,
    split: float,
    data_type: str,
    use_cot: bool,
    pass_k: str
):
    # Problems
    logger.info(f"Loading problems: {problems_fpath}")
    with open(problems_fpath, "r") as f:
        problems_blob = json.load(f)
    problems: List[Dict[str, Any]] = problems_blob["collected"]
    logger.info(f"Total problems: {len(problems)}")

    # Itemized results (per-model)
    with open(os.path.join(data_dir1, "itemized_results.json"), "r") as f:
        results1 = json.load(f)
    with open(os.path.join(data_dir2, "itemized_results.json"), "r") as f:
        results2 = json.load(f)

    # Split
    num_train = int(len(problems) * float(split))

    # Build pairs
    examples: List[Dict[str, Any]] = []
    for problem in problems[: num_train]:
        pid = problem["id"]
        transcript1_fpath = os.path.join(data_dir1, "transcripts", f"{pid}.json")
        transcript2_fpath = os.path.join(data_dir2, "transcripts", f"{pid}.json")

        pairs = extract_pairs_for_problem(
            problem=problem,
            data_type=str(data_type),
            use_cot=bool(use_cot),
            tokenizer=tokenizer,
            transcript1_fpath=transcript1_fpath,
            transcript2_fpath=transcript2_fpath,
            results1=results1,
            results2=results2,
            passk_key=str(pass_k)
        )
        examples.extend(pairs)

    return examples


def train(
    tokenizer: AutoTokenizer,
    data_fpath: str,
    model_fpath: str,
    out_dir: str,
    use_4bit: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    batch_size: int,
    grad_accum: int,
    lr: float,
    beta: float,
    epochs: int,
    gpus: List[int],
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus) 

    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import DPOTrainer, DPOConfig
    from transformers import BitsAndBytesConfig

    # ---- Tokenizer defaults for causal LM DPO ----
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    # ---- Model (QLoRA by default) ----
    bnb = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if use_4bit else None

    model = AutoModelForCausalLM.from_pretrained(
        model_fpath,
        quantization_config=bnb,
        device_map="auto",
    )

    # 1) Model config
    model.config.bos_token_id = None  # <-- your update
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # 2) Generation config
    gen = getattr(model, "generation_config", GenerationConfig())
    gen.bos_token_id = None
    gen.pad_token_id = tokenizer.pad_token_id
    gen.eos_token_id = tokenizer.eos_token_id
    model.generation_config = gen

    # Ensure vocab size matches tokenizer (needed if you added PAD)
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    # Prepare for k-bit training & ensure inputs get grads
    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def _make_inputs_require_grad(module, inputs, output):
            try:
                output.requires_grad_(True)
            except Exception:
                pass
        model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)

    # Gradient checkpointing + disable cache
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()

    # ---- LoRA config (Qwen-style modules; make configurable if you swap families) ----
    peft_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )

    # ---- Data ----
    ds = load_dataset("json", data_files=data_fpath)["train"]
    split = ds.train_test_split(test_size=0.2, seed=42)
    train_ds, val_ds = split["train"], split["test"]

    # ---- DPO config ----
    dpo_cfg = DPOConfig(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        beta=beta,
        num_train_epochs=epochs,
        gradient_checkpointing=True,
        # eval + save cadence
        eval_strategy="steps",
        eval_steps=5,
        save_strategy="steps",
        save_steps=5,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=1,
        logging_strategy="steps",
        report_to="tensorboard",
        remove_unused_columns=False,
    )

    # ---- Trainer ----
    trainer = DPOTrainer(
        model=model,
        args=dpo_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_cfg,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    model.generation_config.save_pretrained(out_dir)


@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    # Tokenizer must match the student’s chat template (Qwen-7B-Instruct)
    tokenizer = AutoTokenizer.from_pretrained(cfg.dpo.model_fpath, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        # Only fallback if truly missing
        tokenizer.pad_token = tokenizer.eos_token

    # Prefer right padding for DPO training
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    # Save DPO jsonl (or json)
    train_fpath = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        str(cfg.experiment.dataset),
        f"{cfg.data.filename}_dpo.jsonl"
    )
    if not os.path.exists(train_fpath):
        # Problems
        problems_fpath = os.path.join(
            str(cfg.paths.data),
            str(cfg.teacher.model),
            str(cfg.experiment.dataset),
            f"{cfg.data.filename}.json",
        )
        logger.info(f"Loading problems: {problems_fpath}")
        with open(problems_fpath, "r") as f:
            problems: List[Dict[str, Any]] = json.load(f)["collected"]
        logger.info(f"Total problems: {len(problems)}")

        examples = collect_prefrence_pairs(
            tokenizer,
            problems_fpath,
            cfg.dpo.data_dir1,
            cfg.dpo.data_dir2,
            cfg.dpo.split,
            cfg.data.type,
            cfg.experiment.cot,
            cfg.gen.pass_k
        )

        with open(train_fpath, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        logger.info(f"Wrote {len(examples)} preference pairs to {train_fpath}")

    ckpt_dir = os.path.join(
        str(cfg.paths.experiments),
        str(cfg.experiment.name),
        str(cfg.teacher.model),
        str(cfg.student.model),
        cfg.experiment.dataset,
        "dpo_model"
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training
    train(
        tokenizer,
        train_fpath,
        cfg.dpo.model_fpath,
        ckpt_dir,
        cfg.dpo.use_4bit,
        cfg.dpo.lora_r,
        cfg.dpo.lora_alpha,
        cfg.dpo.lora_dropout,
        cfg.dpo.batch_size,
        cfg.dpo.grad_accum,
        cfg.dpo.lr,
        cfg.dpo.beta,
        cfg.dpo.epochs,
        cfg.dpo.gpus
    )


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
