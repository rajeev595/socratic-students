import os
import time
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import timedelta

import hydra
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers.trainer_utils import get_last_checkpoint

from trl import SFTTrainer, SFTConfig

from codes.utils.vars import set_envs
from codes.utils.llm.server import _MODEL_ALIASES

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)

# =========================
# Preprocess (SFT)
# =========================
def preprocess_sft_example(example: Dict[str, Any], tokenizer: AutoTokenizer):
    prompt_str = tokenizer.apply_chat_template(
        example["prompt"], tokenize=False, add_generation_prompt=True
    )
    return {
        "prompt": prompt_str,
        "completion": example["response"],
    }

# =========================
# Preprocess (GRPO groups)
# =========================
def _mask_prompt(labels: List[int], prompt_len: int) -> List[int]:
    lab = list(labels)
    for i in range(min(prompt_len, len(lab))):
        lab[i] = -100
    return lab

def preprocess_grpo_group(example: Dict[str, Any], tokenizer: AutoTokenizer, max_len: int = 2048):
    """
    Input (one line per group):
      - prompt_messages: chat messages used to elicit K questions
      - candidates: list of {"text": str, "reward": float, "metrics": {...}}
      - k: int
    Output: tokenized, still grouped; collator will flatten to (B*K, T).
    """
    msgs = example["prompt_messages"]
    k = int(example.get("k", len(example["candidates"])))
    prompt_str = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_ids)

    input_ids_group, attn_group, labels_group, rewards = [], [], [], []
    for cand in example["candidates"]:
        full = prompt_str + cand["text"]
        tok = tokenizer(full, truncation=True, max_length=max_len, add_special_tokens=False)
        inp = tok["input_ids"]
        att = tok["attention_mask"]
        lab = _mask_prompt(inp.copy(), prompt_len=prompt_len)
        input_ids_group.append(inp)
        attn_group.append(att)
        labels_group.append(lab)
        rewards.append(float(cand.get("reward", 0.0)))

    return {
        "group_id": f'{example.get("ex_id","")}|turn={example.get("turn","")}',
        "k": k,
        "rewards": rewards,                 # len K
        "input_ids_group": input_ids_group, # list[K][T_i]
        "attention_group": attn_group,      # list[K][T_i]
        "labels_group": labels_group,       # list[K][T_i]
    }

# =========================
# Model setup (shared)
# =========================
def setup_for_training(
    tokenizer: AutoTokenizer,
    use_4bit: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    enable_gc: bool,
):
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    bnb = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if use_4bit else None

    model = AutoModelForCausalLM.from_pretrained(
        tokenizer.name_or_path,
        quantization_config=bnb,
        device_map=({"": local_rank} if use_4bit else None),
        low_cpu_mem_usage=False,
        torch_dtype=(torch.bfloat16 if use_4bit else None),
        cache_dir=os.path.join(os.environ["HF_HOME"], "hub")
    )

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    gen = getattr(model, "generation_config", GenerationConfig())
    gen.pad_token_id = tokenizer.pad_token_id
    gen.eos_token_id = tokenizer.eos_token_id
    model.generation_config = gen

    if use_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=bool(enable_gc),
        )

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def _make_inputs_require_grad(module, inputs, output):
            try: output.requires_grad_(True)
            except Exception: pass
        model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)

    model.config.use_cache = False
    if enable_gc:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    from peft import LoraConfig
    peft_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj",
        ],
    )
    return model, tokenizer, peft_cfg

# =========================
# SFT training
# =========================
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
    model, tokenizer, peft_cfg = setup_for_training(
        tokenizer, use_4bit, lora_r, lora_alpha, lora_dropout, enable_gc=True
    )

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
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=1,
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
        peft_config=peft_cfg,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    _free_cuda(trainer, model, peft_cfg, train_ds, val_ds)
    return out_dir

# =========================
# GRPO collator
# =========================
class DataCollatorForGRPO:
    """
    Input: list of group dicts from preprocess_grpo_group.
    Output (flattened):
      - input_ids: (B*K, T)
      - attention_mask: (B*K, T)
      - labels: (B*K, T)
      - rewards: (B, K)
    """
    def __init__(self, tokenizer: AutoTokenizer, pad_to_multiple_of: Optional[int] = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def _pad_2d(self, sequences: List[List[int]], pad_id: int) -> torch.Tensor:
        maxlen = max(len(s) for s in sequences)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            if maxlen % m != 0:
                maxlen = ((maxlen // m) + 1) * m
        out = [s + [pad_id] * (maxlen - len(s)) for s in sequences]
        return torch.tensor(out, dtype=torch.long)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        B = len(batch)
        K = batch[0]["k"]
        flat_input_ids, flat_attn, flat_labels = [], [], []
        rewards = []
        for grp in batch:
            assert grp["k"] == K, "Mixed K in GRPO batch"
            rewards.append(grp["rewards"])
            flat_input_ids.extend(grp["input_ids_group"])
            flat_attn.extend(grp["attention_group"])
            flat_labels.extend(grp["labels_group"])

        input_ids = self._pad_2d(flat_input_ids, pad_id=self.tokenizer.pad_token_id)
        attention_mask = self._pad_2d(flat_attn, pad_id=0)
        maxlen = input_ids.size(1)
        labels = torch.tensor(
            [lab + [-100]*(maxlen - len(lab)) for lab in flat_labels],
            dtype=torch.long
        )
        rewards = torch.tensor(rewards, dtype=torch.float)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "rewards": rewards,                 # (B, K)
            "K": torch.tensor([K], dtype=torch.long),
        }

# =========================
# GRPO listwise trainer
# =========================
class ListwiseTrainer(SFTTrainer):
    """
    Listwise (RRHF/GRPO-style):
      target = softmax(norm(reward)/tau) over K candidates per group
      pred   = softmax(sequence_logprob) over K candidates
      loss   = CE(target || pred) + kl_coeff * KL(pred || ref_pred) [optional]
    """
    def __init__(self, *args, tau: float = 0.7, reward_norm: str = "zscore",
                 kl_coeff: float = 0.02, ref_model: Optional[torch.nn.Module] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = float(tau)
        self.reward_norm = reward_norm
        self.kl_coeff = float(kl_coeff)
        self.ref_model = ref_model
        if self.ref_model is not None:
            for p in self.ref_model.parameters():
                p.requires_grad_(False)
            self.ref_model.eval()

    def _seq_logprob(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logp_tok = F.log_softmax(logits, dim=-1)
        tgt = labels.clone()
        mask = (tgt != -100).long()
        tgt[tgt == -100] = 0
        tok_lp = logp_tok.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * mask
        return tok_lp.sum(dim=1)  # (N,)

    @torch.no_grad()
    def _ref_pred(self, inputs) -> Optional[torch.Tensor]:
        if self.ref_model is None:
            return None
        out = self.ref_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        lp = self._seq_logprob(out.logits, inputs["labels"])
        return torch.softmax(lp.view(-1, int(inputs["K"][0].item())), dim=1)

    def compute_loss(self, model, inputs, return_outputs=False):
        K = int(inputs["K"][0].item())
        B = inputs["rewards"].size(0)

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=None,
        )
        seq_lp = self._seq_logprob(outputs.logits, inputs["labels"]).view(B, K)

        rewards = inputs["rewards"]
        if self.reward_norm == "zscore":
            mu = rewards.mean(dim=1, keepdim=True)
            sd = rewards.std(dim=1, keepdim=True).clamp_min(1e-6)
            r_tilde = (rewards - mu) / sd
        elif self.reward_norm == "minmax":
            lo = rewards.min(dim=1, keepdim=True).values
            hi = rewards.max(dim=1, keepdim=True).values
            r_tilde = (rewards - lo) / (hi - lo + 1e-6)
        else:
            r_tilde = rewards

        target = torch.softmax(r_tilde / self.tau, dim=1)  # (B, K)
        pred = torch.softmax(seq_lp, dim=1)                # (B, K)

        loss_listwise = -(target * (pred + 1e-12).log()).sum(dim=1).mean()

        loss = loss_listwise
        if self.kl_coeff > 0.0 and self.ref_model is not None:
            ref_pred = self._ref_pred(inputs)
            if ref_pred is not None:
                kl = (pred * (pred / (ref_pred + 1e-12)).log()).sum(dim=1).mean()
                loss = loss + self.kl_coeff * kl

        return (loss, outputs) if return_outputs else loss

# =========================
# GRPO training
# =========================
def train_grpo(
    tokenizer: AutoTokenizer,
    train_fpath: str,
    val_fpath: str,
    init_adapter_dir: Optional[str],
    out_dir: str,
    use_4bit: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    batch_size_groups: int,
    grad_accum: int,
    lr: float,
    epochs: int,
    tau: float = 0.7,
    reward_norm: str = "zscore",
    kl_coeff: float = 0.02,
):
    model, tokenizer, peft_cfg = setup_for_training(
        tokenizer, use_4bit, lora_r, lora_alpha, lora_dropout, enable_gc=True
    )

    # Optionally start from an existing adapter (e.g., SFT)
    from peft import PeftModel
    use_existing_adapter = bool(init_adapter_dir) and os.path.exists(
        os.path.join(init_adapter_dir, "adapter_config.json")
    )
    if use_existing_adapter:
        logger.info(f"[GRPO] Loading PEFT adapter from {init_adapter_dir}")
        model = PeftModel.from_pretrained(
            model, init_adapter_dir, is_trainable=True,
            cache_dir=os.path.join(os.environ["HF_HOME"], "hub")
        )
        peft_cfg = None

    # Optional: frozen reference for KL
    ref_model = None
    if kl_coeff > 0.0:
        logger.info("[GRPO] Initializing frozen reference model for KL")
        ref_model = AutoModelForCausalLM.from_pretrained(
            tokenizer.name_or_path,
            device_map="auto",
            torch_dtype=getattr(model, "dtype", torch.bfloat16),
            cache_dir=os.path.join(os.environ["HF_HOME"], "hub")
        )
        if use_existing_adapter:
            ref_model = PeftModel.from_pretrained(
                ref_model, init_adapter_dir, is_trainable=False,
                cache_dir=os.path.join(os.environ["HF_HOME"], "hub")
            )
        for p in ref_model.parameters():
            p.requires_grad_(False)
        ref_model.eval()

    # Data
    ds = load_dataset("json", data_files=train_fpath)["train"]
    train_ds = ds.map(lambda x: preprocess_grpo_group(x, tokenizer=tokenizer), remove_columns=ds.column_names)
    ds = load_dataset("json", data_files=val_fpath)["train"]
    val_ds = ds.map(lambda x: preprocess_grpo_group(x, tokenizer=tokenizer), remove_columns=ds.column_names)

    collator = DataCollatorForGRPO(tokenizer, pad_to_multiple_of=8)

    cfg = SFTConfig(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size_groups,  # groups per step
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=1,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=5,
        report_to="tensorboard",
        remove_unused_columns=False,
    )

    last_checkpoint = None
    if os.path.isdir(cfg.output_dir):
        last_checkpoint = get_last_checkpoint(cfg.output_dir)

    trainer = ListwiseTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        peft_config=peft_cfg,
        processing_class=tokenizer,
        tau=tau,
        reward_norm=reward_norm,
        kl_coeff=kl_coeff,
        ref_model=ref_model,
    )
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    model.generation_config.save_pretrained(out_dir)

    _free_cuda(trainer, model, peft_cfg, train_ds, val_ds, ref_model)

# =========================
# Free CUDA
# =========================
def _free_cuda(*objs):
    import gc, torch
    for o in objs:
        try: del o
        except Exception: pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# =========================
# Hydra entry (SFT -> GRPO)
# =========================
@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    start_time = time.perf_counter()

    # envs (OpenAI key may be unused here; HF_HOME is used)
    set_envs(cfg.paths.openai_api_key_fpath, cfg.paths.hf_home)

    student_name = _MODEL_ALIASES[cfg.student.model]
    tokenizer = AutoTokenizer.from_pretrained(
        student_name,
        use_fast=True,
        cache_dir=os.path.join(os.environ["HF_HOME"], "hub")
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    root = os.path.join(str(cfg.paths.data), str(cfg.teacher.model), str(cfg.experiment.dataset))

    # ---- SFT paths ----
    train_fpath_sft = os.path.join(
        root,
        f"{cfg.student.model}_cot_sft_iter{cfg.gen.dagger_iter}_train.jsonl"
        if cfg.experiment.cot else
        f"{cfg.student.model}_sft_iter{cfg.gen.dagger_iter}_train.jsonl"
    )
    val_fpath_sft = os.path.join(
        root,
        f"{cfg.student.model}_cot_sft_iter{cfg.gen.dagger_iter}_val.jsonl"
        if cfg.experiment.cot else
        f"{cfg.student.model}_sft_iter{cfg.gen.dagger_iter}_val.jsonl"
    )

    # ---- GRPO paths ----
    train_fpath_grpo = os.path.join(
        root,
        f"{cfg.student.model}_cot_grpo_iter{cfg.gen.dagger_iter}_train.jsonl"
        if cfg.experiment.cot else
        f"{cfg.student.model}_grpo_iter{cfg.gen.dagger_iter}_train.jsonl"
    )
    val_fpath_grpo = os.path.join(
        root,
        f"{cfg.student.model}_cot_grpo_iter{cfg.gen.dagger_iter}_val.jsonl"
        if cfg.experiment.cot else
        f"{cfg.student.model}_grpo_iter{cfg.gen.dagger_iter}_val.jsonl"
    )

    # ---- ckpt dirs ----
    base_ckpt = os.path.join(
        str(cfg.paths.experiments), str(cfg.experiment.name),
        str(cfg.teacher.model), str(cfg.student.model),
        cfg.experiment.dataset
    )
    ckpt_dir_sft  = os.path.join(base_ckpt, "sft");  os.makedirs(ckpt_dir_sft,  exist_ok=True)
    ckpt_dir_grpo = os.path.join(base_ckpt, "grpo"); os.makedirs(ckpt_dir_grpo, exist_ok=True)

    # ---- Run SFT ----
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

    # ---- Run GRPO (init from SFT) ----
    train_grpo(
        tokenizer=tokenizer,
        train_fpath=train_fpath_grpo,
        val_fpath=val_fpath_grpo,
        init_adapter_dir=ckpt_dir_sft,
        out_dir=ckpt_dir_grpo,
        use_4bit=cfg.train.rlhf_use_4bit,
        lora_r=cfg.train.lora_r,
        lora_alpha=cfg.train.lora_alpha,
        lora_dropout=cfg.train.lora_dropout,
        batch_size_groups=getattr(cfg.train, "grpo_batch_size_groups", 2),
        grad_accum=cfg.train.grad_accum,
        lr=getattr(cfg.train, "grpo_lr", 5e-6),
        epochs=getattr(cfg.train, "grpo_epochs", 1),
        tau=getattr(cfg.train, "grpo_tau", 0.7),
        reward_norm=getattr(cfg.train, "grpo_reward_norm", "zscore"),
        kl_coeff=getattr(cfg.train, "grpo_kl_coeff", 0.02),
    )

    elapsed = time.perf_counter() - start_time
    logger.info("Total time elapsed: %s", timedelta(seconds=int(elapsed)))

if __name__ == "__main__":
    main()
