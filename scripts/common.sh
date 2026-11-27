#!/usr/bin/env bash
# Shared settings for the pipeline scripts. Source this file; do not run it.
#
#   STUDENT   qwen7 | mistral7                      (default: qwen7)
#   DOMAIN    math | code                           (default: math)
#   EXP       experiments root                      (default: ./experiments)
#   NUM_GPUS  GPUs used for training                (default: 8)
#   EXTRA     extra Hydra overrides for every command, e.g.
#             EXTRA="teacher.gpus=[0,1] student.gpus=[2,3] paths.hf_home=/scratch/hf"
set -euo pipefail

STUDENT=${STUDENT:-qwen7}
DOMAIN=${DOMAIN:-math}
TEACHER=${TEACHER:-qwen72}
EXP=${EXP:-./experiments}
NUM_GPUS=${NUM_GPUS:-8}
EXTRA=${EXTRA:-}

case "$DOMAIN" in
  math) DATASET=gsm8k;         ASSESSMENT=pre_assessment; ASSESSMENT_TURN=0 ;;   # Pre-Assessment (turn 0)
  code) DATASET=humaneval_opc; ASSESSMENT=mid_assessment; ASSESSMENT_TURN=2 ;;   # Mid-Assessment (turn 2)
  *) echo "DOMAIN must be math or code" >&2; exit 1 ;;
esac

# Overrides shared by every command
BASE="teacher.model=$TEACHER student.model=$STUDENT data.type=$DOMAIN experiment.dataset=$DATASET paths.experiments=$EXP $EXTRA"
# Overrides for an evaluation run on the test split
TEST="$BASE data.split=test data.data_amount=1000"

# Best checkpoint of a training run: prints <dir>/checkpoint-N with the smallest N.
# Training keeps the best checkpoint (by validation metric) plus, at most, the latest one.
best_ckpt() {
  local ckpt
  ckpt=$(ls -d "$1"/checkpoint-* 2>/dev/null | sort -V | head -n1)
  if [ -z "$ckpt" ]; then echo "No checkpoint found under $1" >&2; exit 1; fi
  echo "$ckpt"
}

# Checkpoint directory of an ODQS / rejection-sampling training run
ckpt_dir() {  # $1 = experiment name, $2 = sft | dpo
  echo "$EXP/$1/$TEACHER/$STUDENT/$DATASET/${2}_0"
}
