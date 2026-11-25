#!/usr/bin/env bash
# Start the demo: ./run_demo.sh   (works from any folder; opens http://localhost:8501)
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/streamlit ]; then
  echo "First-time setup needed:  python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
exec .venv/bin/streamlit run app/app.py --browser.gatherUsageStats false "$@"
