# Planet or Impostor?

Vetting Kepler planet candidates without the vetting answers, and across catalog releases.

A machine-learning triage study on the NASA Exoplanet Archive KOI tables. Models see only
transit and stellar measurements; the disposition score, vetting flags and every other
vetting output are removed. We measure how much those outputs would have inflated the
results, how the models hold up on unseen stars and on a different catalog release, and
where they fail.

## Headline results

| Question | Answer |
|---|---|
| RQ1 leakage | With the 4 vetting flags every model reaches PR-AUC 0.993-0.999; the flags alone (no model) get MCC 0.976. Clean features: 0.86-0.93. |
| RQ2 models (test, held-out stars) | Gradient boosting 0.929 [0.914, 0.943] = random forest 0.926 > k-NN 0.911 > tree 0.906 > logistic regression 0.860 > baseline 0.494 |
| RQ3 strict label | Same model ranking; boosting 0.974 on CONFIRMED vs FP. Candidates are the hard part. |
| RQ4 release shift | DR24 -> DR25: no loss (ROC-AUC +0.021). DR25 -> DR24: -0.066, mostly from DR24's noisier labels. |
| RQ5 review budget | Top 250 by the model: 244 planet-like; random: 124; highest-SNR-first: 32. |
| RQ6 failure modes | Confident false alarms: planet-shaped signals from neighbouring stars (centroid flag). Confident misses: binary-looking "candidates". |

## Stage 2: an AI vetter vs the Robovetter (trusted labels)

Judged on 6,557 expert-checked DR25 objects (confirmed planets vs certified false positives),
5 star-grouped folds, paired bootstrap intervals (`results/tables/s2_compare.csv`):

| Method | Mistakes | Change vs Robovetter [95% CI] |
|---|---|---|
| Robovetter verdict | 182 | baseline |
| Neural vetter (light-curve views + measurements) | 172 | -10 [-44, +25] |
| Boosting on measurements | 136 | -46 [-79, -12] |
| Neural vetter + Robovetter score (planned) | 118 | -64 [-85, -43] |

Run with `./run_stage2.sh` (streams ~45 GB of light curves, never stored; ~1.5 h + 15 min training).

The full write-up ("Planet or Impostor: Project Explanation") and the team split ("Planet or Impostor: Work Split")
are separate documents; export them to Word or PDF and keep the copies in `docs/`.

## Reproduce

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
./run_all.sh                       # every table and figure
./run_demo.sh                      # the demo (http://localhost:8501)
```

The raw tables are in `data/raw/` (downloaded from the archive's TAP service:
`cumulative`, `q1_q17_dr24_koi`, `q1_q17_dr25_koi`). To refresh them:

```bash
B="https://exoplanetarchive.ipac.caltech.edu/TAP/sync?format=csv&query="
for t in cumulative q1_q17_dr24_koi q1_q17_dr25_koi; do curl -s -o data/raw/$t.csv "${B}select+*+from+${t}"; done
```

## Layout

| Path | Owner | What |
|---|---|---|
| `src/config.py` | Member 1 | paths, seeds, labels, frozen exclusion list, feature groups |
| `src/data.py`, `src/splits.py` | Member 1 | cohort and star-grouped splits |
| `src/preprocess.py`, `src/models_classical.py` | Member 2 | log/impute/scale pipeline; baseline, logistic regression, k-NN |
| `src/models_trees.py` | Member 3 | decision tree, random forest, gradient boosting |
| `src/release.py` | Member 4 | DR24/DR25 matching, label flips |
| `src/metrics.py`, `src/cv.py`, `src/registry.py` | Member 5 | metrics, ECE, bootstrap, CV harness, calibration |
| `experiments/e0_cohort.py`, `e1_leakage.py` | Member 1 | audit, cohort, split; leakage ladder (RQ1) |
| `experiments/e2_models.py` | Members 2, 3, 5 | main model comparison, calibration, operating points (RQ2) |
| `experiments/e3_strict_label.py` | Member 2 | strict label (RQ3) |
| `experiments/e5_ablation.py` | Member 3 | feature-group ablation |
| `experiments/e4_release_shift.py` | Member 4 | DR24 <-> DR25 and label flips (RQ4) |
| `experiments/final_test.py`, `e6_review_budget.py` | Member 5 | one-time test, bootstrap intervals; review budget (RQ5) |
| `experiments/e7_error_analysis.py`, `explanations.py`, `make_figures.py`, `app/` | Member 6 | errors (RQ6), importance, PDP, SHAP, figures, demo |
| `results/tables/`, `results/figures/` | all | every number and figure in the report |
