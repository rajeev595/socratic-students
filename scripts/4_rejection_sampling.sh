#!/usr/bin/env bash
# Step 4 -- Rejection Sampling baselines: the best candidate question is chosen by an LLM judge
# (the teacher model) instead of by downstream Pass@k.
#
#   STUDENT=qwen7 DOMAIN=math bash scripts/4_rejection_sampling.sh
source "$(dirname "$0")/common.sh"

RS="$BASE data.name=rejection_sampling data.source=self"

# Data collection (judge-selected questions and preference pairs)
python -m codes.data.collect_rejection_sampling $RS experiment.name=rejection_sampling data.split=train data.data_amount=1000
python -m codes.data.collect_rejection_sampling $RS experiment.name=rejection_sampling data.split=val   data.data_amount=500

# Rejection Sampling: SFT on the judge-selected questions
accelerate launch --multi_gpu --num_processes "$NUM_GPUS" -m codes.run_sft_dpo $RS \
    experiment.name=rejection_sampling train.sft_epochs=1 train.dpo_epochs=0
CKPT=$(best_ckpt "$(ckpt_dir rejection_sampling sft)")
python -m codes.run_interact $TEST experiment.name=rejection_sampling \
    student.lora.name=${STUDENT}_rs_sft student.lora.path=$CKPT

# Rejection Sampling (DPO): DPO on judge-selected vs. rejected questions
accelerate launch --multi_gpu --num_processes "$NUM_GPUS" -m codes.run_sft_dpo $RS \
    experiment.name=rejection_sampling_dpo
CKPT=$(best_ckpt "$(ckpt_dir rejection_sampling_dpo dpo)")
python -m codes.run_interact $TEST experiment.name=rejection_sampling_dpo \
    student.lora.name=${STUDENT}_rs_dpo student.lora.path=$CKPT
