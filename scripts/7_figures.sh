#!/usr/bin/env bash
# Step 7 -- Figures and tables from the saved results (CPU only; figures need a LaTeX installation).
#
#   bash scripts/7_figures.sh
#
# Output: experiments/plots/
EXP=${EXP:-./experiments}
set -x

python -m src.plots.results            --experiments "$EXP"   # Pass@k vs. turns for RQ1-RQ3 (k = 1, 5, 10)
python -m src.analysis.significance    --experiments "$EXP"   # paired-bootstrap significance tables
python -m src.plots.ablations progress            --experiments "$EXP"   # progress per turn
python -m src.plots.ablations assessment-position --experiments "$EXP"   # impact of assessment position
python -m src.plots.ablations teacher-size        --experiments "$EXP"   # varying the teacher
python -m src.plots.cross_eval         --experiments "$EXP"   # generator vs. solver
python -m src.plots.question_types     --experiments "$EXP"   # question-type distribution
