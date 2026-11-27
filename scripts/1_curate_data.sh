#!/usr/bin/env bash
# Step 1 -- Problem sets: problems the teacher solves but the student does not
# (static Pass@5 < 0.5), split into train / val / test.
#
#   STUDENT=qwen7 DOMAIN=math bash scripts/1_curate_data.sh               # the paper's problem sets
#   CURATE=1 STUDENT=qwen7 DOMAIN=math bash scripts/1_curate_data.sh      # curate from scratch
#
# Output: data/<teacher>/<dataset>/splits_0.5/<student>_{train,val,test}.json
source "$(dirname "$0")/common.sh"

if [ "${CURATE:-0}" = "1" ]; then
  # Runs the student and the teacher on the source datasets; gives a new set of problems.
  python -m src.data.curate_$DOMAIN $BASE
else
  # Rebuilds the exact problem sets of the paper from problem_ids/ (CPU only).
  python -m src.data.build_splits --teacher "$TEACHER" --students "$STUDENT" --datasets "$DATASET"
fi
