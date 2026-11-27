import os
import json
import time
import logging
from typing import List, Optional, Callable
from datetime import timedelta

import hydra
from omegaconf import DictConfig, ListConfig

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers.trainer_utils import get_last_checkpoint

from src.utils.env import set_envs, hf_cache_dir
from src.utils.llm.server import _MODEL_ALIASES


logger = logging.getLogger(__name__)


def _data_file(cfg: DictConfig, root: str, kind: str, i: int, split: str) -> str:
    """
    Path of one training-data file (kind = sft | dpo | grpo).
    Peer-ODQS files (data.source=peer) carry the peer model in their name.
    """
    source = str(cfg.data.source)
    if source not in ("self", "peer"):
        raise ValueError(f"data.source must be 'self' or 'peer', got: {source!r}")
    stem = f"{cfg.student.model}__{cfg.peer.model}__" if source == "peer" else f"{cfg.student.model}_"
    return os.path.join(root, f"{stem}{kind}_iter{i}_{split}.jsonl")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)

# -----------------------------
# Preprocess helpers
# -----------------------------
def preprocess_dpo_example(
    example,
    tokenizer: AutoTokenizer
):
    # Apply Gemma system-role workaround without mutating caller's list.
    use_messages = list(example["prompt"])
    if "gemma" in tokenizer.name_or_path.lower() and use_messages and use_messages[0].get("role") == "system":
        sys_msg = use_messages[0].get("content", "")
        if len(use_messages) >= 2 and use_messages[1].get("role") == "user":
            merged_user = f"{sys_msg}\n\n{use_messages[1].get('content', '')}"
            use_messages = [dict(use_messages[1], content=merged_user)] + use_messages[2:]
        else:
            # No user message to merge into; degrade gracefully by converting system→user.
            use_messages = [{"role": "user", "content": sys_msg}] + use_messages[1:]

    prompt_str = tokenizer.apply_chat_template(
        use_messages, tokenize=False, add_generation_prompt=True
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
def setup_for_training_8bit(
    tokenizer: AutoTokenizer,
    use_8bit: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: List[str],
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
        load_in_8bit=use_8bit,
        bnb_8bit_compute_dtype=torch.bfloat16,
    ) if use_8bit else None

    # Load ON THE SAME DEVICE as this process will train on for 4-bit;
    # for full precision, let TRL/Accelerate place the model later.
    model = AutoModelForCausalLM.from_pretrained(
        tokenizer.name_or_path,
        quantization_config=bnb,
        device_map=({"": local_rank} if use_8bit else None),
        low_cpu_mem_usage=False,
        cache_dir=hf_cache_dir()
    )

    try:
        model.config.attn_implementation = "flash_attention_2"
    except Exception:
        pass

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
    if use_8bit:
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
    peft_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules,
    )

    return model, tokenizer, peft_cfg


def setup_for_training_4bit(
    tokenizer: AutoTokenizer,
    use_4bit: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: List[str],
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

    # BitsAndBytes QLoRA 4-bit config (if requested)
    bnb = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ) if use_4bit else None

    # Load ON THE SAME DEVICE as this process will train on for 4-bit;
    # for full precision, let TRL/Accelerate place the model later.
    model = AutoModelForCausalLM.from_pretrained(
        tokenizer.name_or_path,
        quantization_config=bnb,
        device_map=({"": local_rank} if use_4bit else None),
        low_cpu_mem_usage=False,
        cache_dir=hf_cache_dir(),
        torch_dtype=torch.bfloat16 if use_4bit else None,
    )

    try:
        model.config.attn_implementation = "flash_attention_2"
    except Exception:
        pass

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
    peft_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules,
    )

    return model, tokenizer, peft_cfg


def get_best_ckpt_dir(root_dir: str) -> str:
    if root_dir is None:
        return None

    state_p = os.path.join(root_dir, "trainer_state.json")
    if os.path.exists(state_p):
        with open(state_p) as f:
            st = json.load(f)
        best = st.get("best_model_checkpoint")
        if best and os.path.exists(best):
            return best  # e.g., ".../sft/checkpoint-450"
    return None  # fallback: last-step export


# -----------------------------
# SFT training
# -----------------------------
def train_sft(
    tokenizer: AutoTokenizer,
    train_fpath: str,
    val_fpath: str,
    out_dir: str,
    use_kbit: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: List[str],
    batch_size: int,
    grad_accum: int,
    lr: float,
    epochs: int,
    eval_steps: int,
    setup_for_training: Callable
):
    # Enable GC for SFT (usually stable and saves memory)
    model, tokenizer, peft_cfg = setup_for_training(
        tokenizer,
        use_kbit,
        lora_r,
        lora_alpha,
        lora_dropout,
        lora_target_modules,
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
        gradient_checkpointing=True,        # keep GC for SFT
        ddp_find_unused_parameters=False,   # safer with adapters
        # eval + save cadence
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        # load_best_model_at_end=True,      # Causes OOM, turn it OFF
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

    # For 8-bit/4-bit models, Trainer's internal load_adapter via load_state_dict
    # can fail with "CUDA error: unspecified launch failure". Load the adapter on
    # CPU first to avoid this, then let Trainer only restore optimizer/scheduler state.
    if last_checkpoint is not None:
        from peft import PeftModel
        adapter_cfg = os.path.join(last_checkpoint, "adapter_config.json")
        if os.path.exists(adapter_cfg):
            logger.info(f"Pre-loading LoRA adapter from {last_checkpoint} onto CPU")
            model = PeftModel.from_pretrained(
                model, last_checkpoint, is_trainable=True,
                low_cpu_mem_usage=True,
            )
            peft_cfg = None  # model already wrapped; don't re-wrap

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_cfg,     # create & train fresh LoRA (None if resuming)
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
    init_adapter_dir: Optional[str],   # SFT adapter dir to start from (None = start from the base model)
    out_dir: str,
    use_kbit: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: List[str],
    batch_size: int,
    grad_accum: int,
    lr: float,
    beta: float,
    epochs: int,
    eval_steps: int,
    setup_for_training: Callable
):
    model, tokenizer, peft_cfg = setup_for_training(
        tokenizer,
        use_kbit,
        lora_r,
        lora_alpha,
        lora_dropout,
        lora_target_modules,
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
            cache_dir=hf_cache_dir()
        )
        peft_cfg = None  # do NOT pass peft_config to trainer; model is already wrapped

    from trl import DPOTrainer, DPOConfig

    # ---- Data ----
    ds = load_dataset("json", data_files=train_fpath)["train"]
    train_ds = ds.map(lambda x: preprocess_dpo_example(x, tokenizer=tokenizer))
    ds = load_dataset("json", data_files=val_fpath)["train"]
    val_ds = ds.map(lambda x: preprocess_dpo_example(x, tokenizer=tokenizer))

    # ---- DPO config ----
    dpo_cfg = DPOConfig(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        beta=beta,
        num_train_epochs=epochs,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        # eval + save cadence
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=1,
        # load_best_model_at_end=True,
        metric_for_best_model="eval_rewards/accuracies",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=5,
        report_to="tensorboard",
        remove_unused_columns=False
    )

    last_checkpoint = None
    if os.path.isdir(dpo_cfg.output_dir):
        last_checkpoint = get_last_checkpoint(dpo_cfg.output_dir)

    # Same workaround as SFT: pre-load LoRA adapter on CPU to avoid
    # "CUDA error: unspecified launch failure" from Trainer's internal load_adapter.
    if last_checkpoint is not None and peft_cfg is not None:
        from peft import PeftModel
        adapter_cfg = os.path.join(last_checkpoint, "adapter_config.json")
        if os.path.exists(adapter_cfg):
            logger.info(f"Pre-loading LoRA adapter from {last_checkpoint} onto CPU")
            model = PeftModel.from_pretrained(
                model, last_checkpoint, is_trainable=True,
                low_cpu_mem_usage=True,
            )
            peft_cfg = None  # model already wrapped; don't re-wrap

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

    # lora target modules
    lora_target_modules = cfg.train.lora_target_modules
    if isinstance(lora_target_modules, (list, ListConfig)):
        lora_target_modules = list(lora_target_modules)

    # Tokenizer must match the student’s chat template
    tokenizer = AutoTokenizer.from_pretrained(
        student_name, 
        use_fast=True, 
        cache_dir=hf_cache_dir()
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    root = os.path.join(
        str(cfg.paths.data), 
        str(cfg.teacher.model), 
        str(cfg.experiment.dataset), 
        str(cfg.data.name)
    )
    base_ckpt = os.path.join(
        str(cfg.paths.experiments), str(cfg.experiment.name),
        str(cfg.teacher.model), str(cfg.student.model),
        cfg.experiment.dataset
    )

    # SFT
    ckpt_dir_sft = None
    if cfg.train.sft_epochs > 0:
        train_fpath_sft, val_fpath_sft = [], []
        for i in range(cfg.gen.iteration + 1):
            train_fpath = _data_file(cfg, root, "sft", i, "train")
            val_fpath = _data_file(cfg, root, "sft", i, "val")
            
            train_fpath_sft.append(train_fpath)
            val_fpath_sft.append(val_fpath)

        ckpt_dir_sft = os.path.join(
            base_ckpt, 
            f"sft_{cfg.gen.iteration}"
        )
        os.makedirs(ckpt_dir_sft, exist_ok=True)
        train_sft(
            tokenizer=tokenizer,
            train_fpath=train_fpath_sft,
            val_fpath=val_fpath_sft,
            out_dir=ckpt_dir_sft,
            use_kbit=cfg.train.sft_use_8bit,
            lora_r=cfg.train.lora_r,
            lora_alpha=cfg.train.lora_alpha,
            lora_dropout=cfg.train.lora_dropout,
            lora_target_modules=lora_target_modules,
            batch_size=cfg.train.sft_batch_size,
            grad_accum=cfg.train.grad_accum,
            lr=cfg.train.sft_lr,
            epochs=cfg.train.sft_epochs,
            eval_steps=cfg.train.sft_eval_steps,
            setup_for_training=setup_for_training_8bit # if cfg.data.type == "math" else setup_for_training_4bit
        )

    # DPO
    if cfg.train.dpo_epochs > 0:
        train_fpath_dpo, val_fpath_dpo = [], []
        for i in range(cfg.gen.iteration + 1):
            train_fpath = _data_file(cfg, root, "dpo", i, "train")
            val_fpath = _data_file(cfg, root, "dpo", i, "val")

            train_fpath_dpo.append(train_fpath)
            val_fpath_dpo.append(val_fpath)

        ckpt_dir_dpo = os.path.join(
            base_ckpt,
            f"dpo_{cfg.gen.iteration}"
        )
        os.makedirs(ckpt_dir_dpo, exist_ok=True)


        train_dpo(
            tokenizer,
            train_fpath=train_fpath_dpo,
            val_fpath=val_fpath_dpo,
            init_adapter_dir=get_best_ckpt_dir(ckpt_dir_sft),
            out_dir=ckpt_dir_dpo,
            use_kbit=cfg.train.dpo_use_8bit,
            lora_r=cfg.train.lora_r,
            lora_alpha=cfg.train.lora_alpha,
            lora_dropout=cfg.train.lora_dropout,
            lora_target_modules=lora_target_modules,
            batch_size=cfg.train.dpo_batch_size,
            grad_accum=cfg.train.grad_accum,
            lr=cfg.train.dpo_lr,
            beta=cfg.train.beta,
            epochs=cfg.train.dpo_epochs,
            eval_steps=cfg.train.dpo_eval_steps,
            setup_for_training=setup_for_training_8bit
        )

    elapsed = time.perf_counter() - start_time
    logger.info("Total time elapsed: %s", timedelta(seconds=int(elapsed)))


if __name__ == "__main__":
    main()
