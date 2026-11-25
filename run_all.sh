#!/usr/bin/env bash
# Reproduce every number and figure. Order matters: E2 fixes models, calibration and
# operating points before the final test is touched.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}   # see src/__init__.py
$PY experiments/e0_cohort.py          # Member 1: audit, cohort, grouped split
$PY experiments/e2_models.py          # Members 2, 3, 5: tuning, CV, calibration, operating points
$PY experiments/e1_leakage.py         # Member 1: leakage ladder
$PY experiments/e1b_error_columns.py  # Member 1: the uncertainty-column check
$PY experiments/e3_strict_label.py    # Member 2: strict label
$PY experiments/e5_ablation.py        # Member 3: feature-group ablation
$PY experiments/e4_release_shift.py   # Member 4: DR24 <-> DR25, label flips
$PY experiments/final_test.py         # Member 5: one-time test evaluation, bootstrap CIs
$PY experiments/e6_review_budget.py   # Member 5: review-budget curve
$PY experiments/e7_error_analysis.py  # Member 6: confident mistakes
$PY experiments/explanations.py       # Member 6: permutation importance, PDP, SHAP
$PY experiments/make_figures.py       # Member 6: figures
$PY app/build_demo_model.py           # Member 6: uncalibrated boosting model for the demo's SHAP
