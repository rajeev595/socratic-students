#!/usr/bin/env bash
# Step 2 -- Prompt-based questioning strategies (RQ1 and RQ2), no training.
#
#   STUDENT=qwen7 DOMAIN=math bash scripts/2_prompting_strategies.sh
#
# Output: experiments/<name>/<teacher>/<student>/<dataset>/{transcripts_0/,itemized_results_0.json,aggregate_results_0.json}
# Turn 0 of every run is the Static baseline (Pass@k before any interaction).
source "$(dirname "$0")/common.sh"

# Unscaffolded: the student asks questions with no guidance on how to ask
python -m src.interact $TEST experiment.name=unscaffolded

# Scaffolded: the student first summarises what it knows and what is missing
python -m src.interact $TEST experiment.name=scaffolded experiment.scaffolded=True

# Pre-Assessment: Scaffolded + teacher feedback on a first attempt before turn 0
python -m src.interact $TEST experiment.name=pre_assessment experiment.scaffolded=True experiment.assessment_turn=0

# Mid-Assessment: Scaffolded + teacher feedback on an attempt at turn 2
python -m src.interact $TEST experiment.name=mid_assessment experiment.scaffolded=True experiment.assessment_turn=2
