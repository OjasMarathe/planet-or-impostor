"""Final test: the held-out 20 % of stars, used ONCE, after features, models, calibration
and operating points were fixed on the development set (E2).

Owner: Member 5 (evaluation and statistics).
Outputs: headline table with 95 % star-bootstrap intervals, paired differences against the
selected model, operating-point results, test predictions, and fitted models for Member 6.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, matthews_corrcoef, roc_auc_score

from src import config as C
from src.cv import calibrated, fit
from src.metrics import all_metrics, bootstrap_ci, ece, recall_precision_at
from src.registry import ALL, calibration_method, load_cohort, tuned


def main():
    df = load_cohort()
    dev, test = df[~df.is_test].reset_index(drop=True), df[df.is_test].reset_index(drop=True)
    ops = json.loads((C.TABLES / "operating_points.json").read_text())
    sel = ops["_selected_model"]

    preds, rows, op_rows = {}, [], []
    for name in ALL:
        est = calibrated(tuned(name), calibration_method(name), dev.y_main.values, dev.kepid.values)
        fit(est, dev[C.CLEAN], dev.y_main.values, dev.kepid.values)
        p = est.predict_proba(test[C.CLEAN])[:, 1]
        preds[name] = p
        rows.append({"model": name, "calibration": calibration_method(name), **all_metrics(test.y_main, p)})
        if name in ops:
            for label, thr in [("threshold 0.5", 0.5), ("95 % recall threshold (from dev)", ops[name]["thr_recall95"])]:
                op_rows.append({"model": name, "operating_point": label,
                                **recall_precision_at(test.y_main.values, p, thr),
                                "mcc": matthews_corrcoef(test.y_main, (p >= thr).astype(int))})
        if name in (sel, "Random forest", "Gradient boosting", "Logistic regression"):
            joblib.dump(est, C.MODELS / f"{name.lower().replace(' ', '_')}_calibrated.joblib")
        print(f"{name:22s} test PR-AUC={rows[-1]['pr_auc']:.4f}", flush=True)

    # Uncalibrated forest on the same data, for SHAP (calibration is monotone, so rankings match)
    rf = tuned("Random forest").fit(dev[C.CLEAN], dev.y_main.values)
    joblib.dump(rf, C.MODELS / "random_forest_uncalibrated.joblib")

    point = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    point.to_csv(C.TABLES / "final_test_point.csv", index=False)
    pd.DataFrame(op_rows).to_csv(C.TABLES / "final_test_operating_points.csv", index=False)

    fns = {"pr_auc": average_precision_score, "roc_auc": roc_auc_score, "brier": brier_score_loss, "ece": ece,
           "mcc@0.5": lambda y, p: matthews_corrcoef(y, (p >= 0.5).astype(int))}
    ci, diff = bootstrap_ci(test.y_main.values, preds, test.kepid.values, fns, C.N_BOOT, C.SEED, ref=sel)
    ci.to_csv(C.TABLES / "final_test_ci.csv", index=False)
    diff.to_csv(C.TABLES / "final_test_paired_diff.csv", index=False)

    out = test[["kepid", "kepoi_name", "koi_disposition", "y_main", "y_strict", "koi_parm_prov", "koi_fittype"]
               + C.CLEAN + C.FLAGS].copy()
    for name, p in preds.items():
        out["p_" + name] = p
    out.to_csv(C.PROC / "test_predictions.csv", index=False)

    wide = ci.pivot(index="model", columns="metric", values=["estimate", "ci_low", "ci_high"])
    print(point.round(4).to_string(index=False))
    print(pd.DataFrame(op_rows).round(4).to_string(index=False))
    print(ci[ci.metric == "pr_auc"].round(4).to_string(index=False))
    print(diff[diff.metric == "pr_auc"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
