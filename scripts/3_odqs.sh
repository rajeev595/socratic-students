#!/usr/bin/env bash
# Step 3 -- Outcome-Driven Question optimization Strategy (RQ3).
#
#   SOURCE=self STUDENT=qwen7 DOMAIN=math bash scripts/3_odqs.sh     # Self-ODQS
#   SOURCE=peer STUDENT=qwen7 DOMAIN=math bash scripts/3_odqs.sh     # Peer-ODQS
#
#   (a) collect outcome-labelled question preferences on the train and val problems
#   (b) train the student with DPO on those preferences (LoRA)
#   (c) evaluate the trained student on the test problems
source "$(dirname "$0")/common.sh"

SOURCE=${SOURCE:-self}                       # self | peer
NAME=${SOURCE}_odqs                          # self_odqs | peer_odqs
ODQS="$BASE data.name=odqs data.source=$SOURCE"

# (a) Data collection. At every turn: sample candidate questions, get a teacher reply for each,
#     score each by the student's Pass@5, and record (best > other) preference pairs.
#     Output: data/<teacher>/<dataset>/odqs/<student>[__<peer>__]_{sft,dpo,grpo}_iter0_{train,val}.jsonl
python -m src.collect.${SOURCE}_odqs $ODQS experiment.name=$NAME data.split=train data.data_amount=1000
python -m src.collect.${SOURCE}_odqs $ODQS experiment.name=$NAME data.split=val   data.data_amount=500

# (b) Training. Output: experiments/<name>/<teacher>/<student>/<dataset>/dpo_0/checkpoint-*
accelerate launch --multi_gpu --num_processes "$NUM_GPUS" -m src.train_odqs $ODQS experiment.name=$NAME

# (c) Evaluation with the trained student (LoRA adapter served on top of the base model)
CKPT=$(best_ckpt "$(ckpt_dir $NAME dpo)")
LORA="student.lora.name=${STUDENT}_odqs student.lora.path=$CKPT"

# Trained student with the best assessment for the domain (the RQ3 result)
python -m src.interact $TEST $LORA experiment.name=${NAME}_${ASSESSMENT} \
    experiment.scaffolded=True experiment.assessment_turn=$ASSESSMENT_TURN

# Trained student without scaffold or assessment (used as "Full" in the generator-vs-solver ablation)
python -m src.interact $TEST $LORA experiment.name=$NAME
