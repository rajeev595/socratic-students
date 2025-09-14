import os
import logging
from typing import List

import hydra
from omegaconf import DictConfig

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)


def preprocess_rlhf_example(
    example,
    tokenizer: AutoTokenizer
):
    prompt_str = tokenizer.apply_chat_template(
        example["prompt"], tokenize=False, add_generation_prompt=True
    )

    return {
        "prompt": prompt_str,
        "chosen": example["chosen_response"],
        "rejected": example["rejected_response"]
    }


def preprocess_sft_example(
    example,
    tokenizer: AutoTokenizer
):
    prompt_str = tokenizer.apply_chat_template(
        example["prompt"], tokenize=False, add_generation_prompt=True
    )

    return {
        "prompt": prompt_str,
        "completion": example["response"]
    }


def setup_for_training(
    tokenizer: AutoTokenizer,
    model_fpath: str,
    use_4bit: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    gpus: List[int],        
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus) 

    from peft import LoraConfig, prepare_model_for_kbit_training
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
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # 2) Generation config
    gen = getattr(model, "generation_config", GenerationConfig())
    gen.pad_token_id = tokenizer.pad_token_id
    gen.eos_token_id = tokenizer.eos_token_id
    model.generation_config = gen

    # Use this in both training functions
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

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

    return model, tokenizer, peft_cfg


def train_sft(
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
    epochs: int,
    gpus: List[int],
):
    # ---- Setup ----
    model, tokenizer, peft_cfg = setup_for_training(
        tokenizer,
        model_fpath,
        use_4bit,
        lora_r,
        lora_alpha,
        lora_dropout,
        gpus,
    )

    from trl import SFTTrainer, SFTConfig

    ds = load_dataset("json", data_files=data_fpath)["train"]
    ds = ds.map(lambda x: preprocess_sft_example(x, tokenizer=tokenizer))
    split = ds.train_test_split(test_size=0.2, seed=2025)
    train_ds, val_ds = split["train"], split["test"]

    sft_cfg = SFTConfig(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        gradient_checkpointing=True,
        # eval + save cadence
        eval_strategy="steps",
        eval_steps=5,
        save_strategy="steps",
        save_steps=5,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=1,
        logging_strategy="steps",
        report_to="tensorboard",
        remove_unused_columns=False,
        completion_only_loss=True
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_cfg,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir


def train_dpo(
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
    # ---- Setup ----
    model, tokenizer, peft_cfg = setup_for_training(
        tokenizer,
        model_fpath,
        use_4bit,
        lora_r,
        lora_alpha,
        lora_dropout,
        gpus,
    )

    from trl import DPOTrainer, DPOConfig

    # ---- Data ----
    ds = load_dataset("json", data_files=data_fpath)["train"]
    ds = ds.map(lambda x: preprocess_rlhf_example(x, tokenizer=tokenizer))
    split = ds.train_test_split(test_size=0.2, seed=2025)
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
        save_total_limit=1,
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
    tokenizer = AutoTokenizer.from_pretrained(cfg.train.model_fpath, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        # Only fallback if truly missing
        tokenizer.pad_token = tokenizer.eos_token

    # Prefer right padding for DPO training
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    # SFT data jsonl (or json)
    train_fpath_sft = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        str(cfg.experiment.dataset),
        f"{cfg.data.filename}_sft.jsonl"
    )

    # DPO data jsonl (or json)
    train_fpath_dpo = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        str(cfg.experiment.dataset),
        f"{cfg.data.filename}_rlhf.jsonl"
    )

    ckpt_dir_sft = os.path.join(
        str(cfg.paths.experiments),
        str(cfg.experiment.name),
        str(cfg.teacher.model),
        str(cfg.student.model),
        cfg.experiment.dataset,
        "sft"
    )
    os.makedirs(ckpt_dir_sft, exist_ok=True)

    ckpt_dir_dpo = os.path.join(
        str(cfg.paths.experiments),
        str(cfg.experiment.name),
        str(cfg.teacher.model),
        str(cfg.student.model),
        cfg.experiment.dataset,
        "dpo"
    )
    os.makedirs(ckpt_dir_dpo, exist_ok=True)

    # SFT
    train_sft(
        tokenizer,
        train_fpath_sft,
        cfg.train.model_fpath,
        ckpt_dir_sft,
        cfg.train.use_4bit,
        cfg.train.lora_r,
        cfg.train.lora_alpha,
        cfg.train.lora_dropout,
        cfg.train.batch_size,
        cfg.train.grad_accum,
        cfg.train.lr,
        cfg.train.epochs,
        cfg.train.gpus,
    )

    # DPO
    train_dpo(
        tokenizer,
        train_fpath_dpo,
        ckpt_dir_sft,
        ckpt_dir_dpo,
        cfg.train.use_4bit,
        cfg.train.lora_r,
        cfg.train.lora_alpha,
        cfg.train.lora_dropout,
        cfg.train.batch_size,
        cfg.train.grad_accum,
        cfg.train.lr,
        cfg.train.beta,
        cfg.train.epochs,
        cfg.train.gpus
    )


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
