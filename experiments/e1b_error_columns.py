"""E1b: a quieter leak? The uncertainty columns.

Owner: Member 1 (data and labels).
Permutation importance (Member 6) showed that the 'stellar temperature' signal comes from its
ERROR columns, not the temperature itself. Stars that host good candidates were more often
observed spectroscopically, which shrinks their stellar uncertainties, so small error bars can
be a trace of human follow-up rather than physics. This script measures how much the models
depend on that, by retraining without the stellar error columns, and without all error columns.
Development set, 3 x 5 grouped CV, tuned settings from E2.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from src import config as C
from src.cv import oof, summarise
from src.registry import load_cohort, tuned

MODELS = ["Logistic regression", "Random forest", "Gradient boosting"]


def main():
    df = load_cohort()
    dev = df[~df.is_test].reset_index(drop=True)
    y, g = dev.y_main.values, dev.kepid.values

    by_disp = dev.groupby("koi_disposition").agg(
        steff_err1_median=("koi_steff_err1", "median"), srad_err1_median=("koi_srad_err1", "median"),
        share_steff_err_below_100K=("koi_steff_err1", lambda s: (s < 100).mean()))
    by_disp.to_csv(C.TABLES / "e1b_stellar_errors_by_disposition.csv")
    print(by_disp.round(3).to_string())

    stellar_err = [c for c in C.STELLAR if "_err" in c]
    all_err = [c for c in C.CLEAN if "_err" in c]
    sets = {
        "all clean (30)": C.CLEAN,
        "without stellar error columns (24)": [c for c in C.CLEAN if c not in stellar_err],
        "without any error columns (12)": [c for c in C.CLEAN if c not in all_err],
        "stellar only, with errors (10)": C.STELLAR,
        "stellar only, values only (4)": [c for c in C.STELLAR if c not in stellar_err],
    }
    for label, cols in sets.items():
        n = int(label.split("(")[-1].rstrip(")"))
        assert len(cols) == n, (label, len(cols))
    rows = []
    for label, cols in sets.items():
        for name in MODELS:
            folds, _ = oof(lambda: tuned(name), dev[cols], y, g)
            rows.append({"features": label, "model": name, **summarise(folds)})
            print(f"{label:36s} {name:20s} PR-AUC={rows[-1]['pr_auc']:.4f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(C.TABLES / "e1b_error_columns.csv", index=False)
    print(res.pivot(index="features", columns="model", values="pr_auc").loc[list(sets)].round(4).to_string())


if __name__ == "__main__":
    main()
