#!/usr/bin/env bash
# Stage 2: neural vetter vs the Robovetter on trusted labels.
# Streams ~45 GB of Kepler light curves from NASA's AWS mirror (nothing large is written to disk;
# ~2 h on a 40 Mbit/s line; resumable), then trains and compares (~45 min on an Apple-silicon GPU).
# Needs data/raw/fpwg_certified_fp.csv (Kepler Certified False Positive table) and Stage 1's
# data/processed/e4_flips.csv (run ./run_all.sh first).
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
.venv/bin/python experiments/s2_build_views.py    # Member 4: light curves -> folded views
.venv/bin/python experiments/s2_vetter.py         # Members 2, 3, 5, 6: train, compare, analyse
.venv/bin/python experiments/s2_compare.py       # Member 5: paired intervals for every method vs the Robovetter
