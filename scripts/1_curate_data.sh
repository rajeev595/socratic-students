#!/usr/bin/env bash
# Step 1 -- Curate the problem set: problems the teacher solves but the student does not
# (static Pass@5 < 0.5), split into train / val / test.
#
#   STUDENT=qwen7 DOMAIN=math bash scripts/1_curate_data.sh
#
# Output: data/<teacher>/<dataset>/splits_0.5/<student>_{train,val,test}.json
source "$(dirname "$0")/common.sh"

python -m codes.data.$DOMAIN.sample_hard $BASE
