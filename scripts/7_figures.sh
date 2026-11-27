#!/usr/bin/env bash
# Step 7 -- Figures and tables from the saved results (CPU only; figures need a LaTeX installation).
#
#   bash scripts/7_figures.sh
#
# Output: experiments/plots/
EXP=${EXP:-./experiments}
set -x

python -m codes.plot_results    --experiments "$EXP"                      # Pass@k vs. turns for RQ1-RQ3 (k = 1, 5, 10)
python -m codes.significance    --experiments "$EXP"                      # paired-bootstrap significance tables
python -m codes.plot_ablations progress            --experiments "$EXP"   # progress per turn
python -m codes.plot_ablations assessment-position --experiments "$EXP"   # impact of assessment position
python -m codes.plot_ablations teacher-size        --experiments "$EXP"   # varying the teacher
python -m codes.plot_cross_eval --experiments "$EXP"                      # generator vs. solver
python -m codes.plot_qtype      --experiments "$EXP"                      # question-type distribution
