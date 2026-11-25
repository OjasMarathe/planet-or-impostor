"""Fit the uncalibrated version of the selected model (gradient boosting) on the development
stars, for SHAP explanations in the demo. The probability shown in the demo comes from the
calibrated model saved by final_test.py; calibration is a monotone (sigmoid) map, so the
direction of every SHAP contribution is the same for both.

Owner: Member 6.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib

from src import config as C
from src.registry import load_cohort, tuned

if __name__ == "__main__":
    df = load_cohort()
    dev = df[~df.is_test]
    m = tuned("Gradient boosting").fit(dev[C.CLEAN], dev.y_main.values)
    joblib.dump(m, C.MODELS / "gradient_boosting_uncalibrated.joblib")
    print("saved", C.MODELS / "gradient_boosting_uncalibrated.joblib")
