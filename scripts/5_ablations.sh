#!/usr/bin/env bash
# Step 5 -- Ablations. Requires steps 2 and 3 (Self-ODQS).
#
#   STUDENT=qwen7 DOMAIN=math bash scripts/5_ablations.sh
source "$(dirname "$0")/common.sh"

# --- Assessment position: one assessment inserted at turn t = 0..5 (unscaffolded) ---------------
for t in 0 1 2 3 4 5; do
  python -m src.interact $TEST experiment.name=assessment_turn_$t experiment.assessment_turn=$t
done

# --- Generator vs. solver: who benefits from ODQS training? --------------------------------------
# The same student model plays two roles: it asks the questions (generator) and it answers the
# problem for Pass@k (solver). Here the two roles use different weights.
CKPT=$(best_ckpt "$(ckpt_dir self_odqs dpo)")
T=$ASSESSMENT_TURN   # turn of the assessment used below (0 for math, 2 for code)
CROSS="$TEST student.gpus=[4,5] evaluator.model=$STUDENT evaluator.gpus=[6,7] experiment.assessment_turn=$ASSESSMENT_TURN"

# Qgen: ODQS-trained generator, untrained solver
python -m src.cross_eval $CROSS student.lora.name=${STUDENT}_odqs student.lora.path=$CKPT
# Solver: untrained generator, ODQS-trained solver
python -m src.cross_eval $CROSS evaluator.lora.name=${STUDENT}_odqs evaluator.lora.path=$CKPT
# Full with the same assessment (Frozen with the same assessment is assessment_turn_$T from above)
python -m src.interact $TEST experiment.name=self_odqs_assessment_turn_$T experiment.assessment_turn=$ASSESSMENT_TURN \
    student.lora.name=${STUDENT}_odqs student.lora.path=$CKPT

# --- Teacher size: the same student with teachers of different sizes -----------------------------
# The problem set is re-curated for every teacher, so that the teacher can solve every problem.
if [ "${RUN_TEACHER_SIZE:-0}" = "1" ]; then
  for t in qwen7 qwen14 qwen32; do
    python -m src.data.curate_$DOMAIN $BASE teacher.model=$t
    python -m src.interact $TEST teacher.model=$t experiment.name=unscaffolded
  done
fi
