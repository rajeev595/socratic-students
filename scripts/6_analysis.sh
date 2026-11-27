#!/usr/bin/env bash
# Step 6 -- LLM-judge analyses of the transcripts. Requires steps 2 and 3.
#
#   STUDENT=qwen7 DOMAIN=math bash scripts/6_analysis.sh
#
# Output: experiments/<compare>/<teacher>/<student>/<dataset>/analysis/
source "$(dirname "$0")/common.sh"

SELF=self_odqs_$ASSESSMENT
PEER=peer_odqs_$ASSESSMENT
METHODS="unscaffolded scaffolded pre_assessment mid_assessment $SELF $PEER"
ANALYSIS="$TEST judge.gpus=[4,5,6,7]"

# Progress per turn (judge: Qwen2.5-72B)
for m in $METHODS; do
  python -m codes.run_analysis $ANALYSIS analysis.task=progress judge.model=qwen72 \
      experiment.name=$m analysis.compare=$m analysis.baseline=$m
done

# Question similarity between Self-ODQS and Unscaffolded questions, turn by turn (judge: Qwen2.5-72B)
python -m codes.run_analysis $ANALYSIS analysis.task=similarity_judge judge.model=qwen72 \
    experiment.name=$SELF analysis.compare=$SELF analysis.baseline=unscaffolded

# Teacher answer leakage, mean across turns (judge: Llama-3.3-70B)
for m in unscaffolded scaffolded $ASSESSMENT $SELF $PEER; do
  python -m codes.run_analysis $ANALYSIS analysis.task=leakage judge.model=llama70 \
      experiment.name=$m analysis.compare=$m analysis.baseline=$m
done

# Question types: Self-ODQS vs. the same setup without training (judge: Llama-3.3-70B)
python -m codes.run_analysis $ANALYSIS analysis.task=question_type judge.model=llama70 \
    experiment.name=$SELF analysis.compare=$SELF analysis.baseline=$ASSESSMENT
