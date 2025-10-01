import os
import time
import logging
from typing import Optional
from datetime import timedelta

import hydra
from omegaconf import DictConfig

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers.trainer_utils import get_last_checkpoint

from codes.utils.vars import set_envs
from codes.utils.llm.server import _MODEL_ALIASES


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)

# -----------------------------
# Preprocess helpers
# -----------------------------
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
        "rejected": example["rejected_response"],
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
        "completion": example["response"],
    }

# -----------------------------
# Model setup
# -----------------------------
def setup_for_training(
    tokenizer: AutoTokenizer,
    use_4bit: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    enable_gc: bool,  # control gradient checkpointing from caller
):
    """
    Loads the *base* model (not an adapter folder). If you want to start from an
    existing adapter (e.g., SFT), attach it later via PeftModel.from_pretrained(...)
    in the caller.
    """
    # Do NOT modify CUDA_VISIBLE_DEVICES here. Let the launcher manage visibility.
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    # Tokenizer defaults for chat LM SFT/DPO
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    # BitsAndBytes/QLoRA config (if requested)
    bnb = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if use_4bit else None

    # Load ON THE SAME DEVICE as this process will train on for 4-bit;
    # for full precision, let TRL/Accelerate place the model later.
    model = AutoModelForCausalLM.from_pretrained(
        tokenizer.name_or_path,
        quantization_config=bnb,
        device_map=({"": local_rank} if use_4bit else None),
        low_cpu_mem_usage=False,
        torch_dtype=(torch.bfloat16 if use_4bit else None),
        cache_dir=os.path.join(os.environ["HF_HOME"], "hub")
    )

    # Align token IDs
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # Generation config alignment
    gen = getattr(model, "generation_config", GenerationConfig())
    gen.pad_token_id = tokenizer.pad_token_id
    gen.eos_token_id = tokenizer.eos_token_id
    model.generation_config = gen

    # Prepare for k-bit training (adds input grad hooks etc.)
    if use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(enable_gc),
        )

    # Ensure inputs get grads (older transformers compatibility)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def _make_inputs_require_grad(module, inputs, output):
            try:
                output.requires_grad_(True)
            except Exception:
                pass
        model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)

    # Disable cache; enable GC only if requested
    model.config.use_cache = False
    if enable_gc:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    # LoRA config (for fresh adapters; if loading an existing adapter later, this can be ignored)
    from peft import LoraConfig
    peft_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    return model, tokenizer, peft_cfg

# -----------------------------
# SFT training
# -----------------------------
def train_sft(
    tokenizer: AutoTokenizer,
    train_fpath: str,
    val_fpath: str,
    out_dir: str,
    use_4bit: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    batch_size: int,
    grad_accum: int,
    lr: float,
    epochs: int,
):
    # Enable GC for SFT (usually stable and saves memory)
    model, tokenizer, peft_cfg = setup_for_training(
        tokenizer,
        use_4bit,
        lora_r,
        lora_alpha,
        lora_dropout,
        enable_gc=True,
    )

    from trl import SFTTrainer, SFTConfig

    ds = load_dataset("json", data_files=train_fpath)["train"]
    train_ds = ds.map(lambda x: preprocess_sft_example(x, tokenizer=tokenizer))
    ds = load_dataset("json", data_files=val_fpath)["train"]
    val_ds = ds.map(lambda x: preprocess_sft_example(x, tokenizer=tokenizer))

    sft_cfg = SFTConfig(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        gradient_checkpointing=True,          # keep GC for SFT
        ddp_find_unused_parameters=False,     # safer with adapters
        # eval + save cadence
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=1,
        # load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=5,
        report_to="tensorboard",
        remove_unused_columns=False,
        completion_only_loss=True,
    )

    last_checkpoint = None
    if os.path.isdir(sft_cfg.output_dir):
        last_checkpoint = get_last_checkpoint(sft_cfg.output_dir)

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_cfg,     # create & train fresh LoRA
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(out_dir)   # saves PEFT adapter (adapter_model.safetensors + adapter_config.json)
    tokenizer.save_pretrained(out_dir)

    # Free GPUs
    _free_cuda(trainer, model, peft_cfg, train_ds, val_ds)

    return out_dir

# -----------------------------
# DPO training
# -----------------------------
def train_dpo(
    tokenizer: AutoTokenizer,
    train_fpath: str,
    val_fpath: str,
    init_adapter_dir: Optional[str],   # pass your SFT adapter dir here to start from SFT
    out_dir: str,
    use_4bit: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    batch_size: int,
    grad_accum: int,
    lr: float,
    beta: float,
    epochs: int
):
    # IMPORTANT: Disable GC for DPO to avoid re-entrant backward on LoRA params
    model, tokenizer, peft_cfg = setup_for_training(
        tokenizer,
        use_4bit,
        lora_r,
        lora_alpha,
        lora_dropout,
        enable_gc=True,
    )

    # If we want to start from SFT checkpoint (adapter), attach it here
    from peft import PeftModel
    use_existing_adapter = bool(init_adapter_dir) and os.path.exists(
        os.path.join(init_adapter_dir, "adapter_config.json")
    )
    if use_existing_adapter:
        logger.info(f"Loading PEFT adapter from {init_adapter_dir}")
        model = PeftModel.from_pretrained(
            model, 
            init_adapter_dir, 
            is_trainable=True, 
            cache_dir=os.path.join(os.environ["HF_HOME"], "hub")
        )
        peft_cfg = None  # do NOT pass peft_config to trainer; model is already wrapped

    from trl import DPOTrainer, DPOConfig

    # ---- Data ----
    ds = load_dataset("json", data_files=train_fpath)["train"]
    train_ds = ds.map(lambda x: preprocess_rlhf_example(x, tokenizer=tokenizer))
    ds = load_dataset("json", data_files=val_fpath)["train"]
    val_ds = ds.map(lambda x: preprocess_rlhf_example(x, tokenizer=tokenizer))

    # ---- DPO config ----
    dpo_cfg = DPOConfig(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        beta=beta,
        num_train_epochs=epochs,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        # eval + save cadence
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=1,
        # load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=5,
        report_to="tensorboard",
        remove_unused_columns=False
    )

    last_checkpoint = None
    if os.path.isdir(dpo_cfg.output_dir):
        last_checkpoint = get_last_checkpoint(dpo_cfg.output_dir)

    # ---- Trainer ----
    trainer = DPOTrainer(
        model=model,
        args=dpo_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_cfg,  # None if we loaded an existing adapter; otherwise create fresh LoRA
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(out_dir)  # saves (updated) adapter
    tokenizer.save_pretrained(out_dir)
    model.generation_config.save_pretrained(out_dir)


def _free_cuda(*objs):
    import gc, torch
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# -----------------------------
# Hydra entry
# -----------------------------
@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    start_time = time.perf_counter()  # ---- start timer ----

    # set the OpenAI key env variable, HF HOME
    set_envs(cfg.paths.openai_api_key_fpath, cfg.paths.hf_home)

    # model names
    student_name = _MODEL_ALIASES[cfg.student.model]

    # Tokenizer must match the student’s chat template
    tokenizer = AutoTokenizer.from_pretrained(
        student_name, 
        use_fast=True, 
        cache_dir=os.path.join(os.environ["HF_HOME"], "hub")
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    # SFT/DPO paths
    train_fpath_sft = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        str(cfg.experiment.dataset),
        f"{cfg.student.model}_cot_sft_iter{cfg.gen.dagger_iter}_train.jsonl" if cfg.experiment.cot else f"{cfg.student.model}_sft_iter{cfg.gen.dagger_iter}_train.jsonl"
    )
    val_fpath_sft = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        str(cfg.experiment.dataset),
        f"{cfg.student.model}_cot_sft_iter{cfg.gen.dagger_iter}_val.jsonl" if cfg.experiment.cot else f"{cfg.student.model}_sft_iter{cfg.gen.dagger_iter}_val.jsonl"
    )

    train_fpath_dpo = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        str(cfg.experiment.dataset),
        f"{cfg.student.model}_cot_rlhf_iter{cfg.gen.dagger_iter}_train.jsonl" if cfg.experiment.cot else f"{cfg.student.model}_rlhf_iter{cfg.gen.dagger_iter}_train.jsonl"
    )
    val_fpath_dpo = os.path.join(
        str(cfg.paths.data),
        str(cfg.teacher.model),
        str(cfg.experiment.dataset),
        f"{cfg.student.model}_cot_rlhf_iter{cfg.gen.dagger_iter}_val.jsonl" if cfg.experiment.cot else f"{cfg.student.model}_rlhf_iter{cfg.gen.dagger_iter}_val.jsonl"
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
        tokenizer=tokenizer,
        train_fpath=train_fpath_sft,
        val_fpath=val_fpath_sft,
        out_dir=ckpt_dir_sft,
        use_4bit=cfg.train.sft_use_4bit,
        lora_r=cfg.train.lora_r,
        lora_alpha=cfg.train.lora_alpha,
        lora_dropout=cfg.train.lora_dropout,
        batch_size=cfg.train.sft_batch_size,
        grad_accum=cfg.train.grad_accum,
        lr=cfg.train.sft_lr,
        epochs=cfg.train.sft_epochs
    )

    # DPO (start from SFT adapter)
    train_dpo(
        tokenizer,
        train_fpath=train_fpath_dpo,
        val_fpath=val_fpath_dpo,
        init_adapter_dir=ckpt_dir_sft,
        out_dir=ckpt_dir_dpo,
        use_4bit=cfg.train.rlhf_use_4bit,
        lora_r=cfg.train.lora_r,
        lora_alpha=cfg.train.lora_alpha,
        lora_dropout=cfg.train.lora_dropout,
        batch_size=cfg.train.rlhf_batch_size,
        grad_accum=cfg.train.grad_accum,
        lr=cfg.train.rlhf_lr,
        beta=cfg.train.beta,
        epochs=cfg.train.rlhf_epochs
    )

    elapsed = time.perf_counter() - start_time
    logger.info("Total time elapsed: %s", timedelta(seconds=int(elapsed)))


if __name__ == "__main__":
    main()
