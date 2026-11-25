"""E3 (RQ3): do the conclusions hold with the strict label (CONFIRMED vs FALSE POSITIVE)?

Owner: Member 2 (classical models).
  a) every model retrained and scored on the strict label (CANDIDATE rows dropped), 3 x 5 grouped CV
  b) the main-label model's out-of-fold scores, read separately for CONFIRMED / CANDIDATE / FP
  c) the strict model applied to the unconfirmed CANDIDATEs: which look most like confirmed planets
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config as C
from src.cv import calibrated, fit, oof, summarise
from src.registry import ALL, calibration_method, load_cohort, tuned


def main():
    df = load_cohort()
    dev = df[~df.is_test].reset_index(drop=True)
    strict = dev[dev.y_strict.notna()].reset_index(drop=True)
    ys, gs = strict.y_strict.astype(int).values, strict.kepid.values

    rows = []
    for name in ALL:
        m = calibration_method(name)
        folds, _ = oof(lambda: calibrated(tuned(name), m, ys, gs), strict[C.CLEAN], ys, gs)
        rows.append({"model": name, "label": "strict (CONFIRMED vs FP)", "prevalence": ys.mean(), **summarise(folds)})
        print(f"strict {name:22s} PR-AUC={rows[-1]['pr_auc']:.4f}", flush=True)
    main_res = pd.read_csv(C.TABLES / "e2_main_results.csv")
    main_res = main_res.assign(label="main (planet-like vs FP)", prevalence=dev.y_main.mean())
    both = pd.concat([main_res, pd.DataFrame(rows)], ignore_index=True)
    both["rank_by_pr_auc"] = both.groupby("label").pr_auc.rank(ascending=False).astype(int)
    both.to_csv(C.TABLES / "e3_strict_vs_main.csv", index=False)

    # b) main-label OOF scores by disposition
    oofd = pd.read_csv(C.PROC / "e2_oof_dev.csv")
    sel = __import__("json").loads((C.TABLES / "operating_points.json").read_text())["_selected_model"]
    p = oofd[sel].values
    by = pd.DataFrame({"disposition": dev.koi_disposition, "p": p}).groupby("disposition").p.describe()
    by.to_csv(C.TABLES / "e3_main_scores_by_disposition.csv")
    conf_vs_fp = dev.koi_disposition.isin(["CONFIRMED", "FALSE POSITIVE"]).values
    cand_vs_fp = dev.koi_disposition.isin(["CANDIDATE", "FALSE POSITIVE"]).values
    sub = pd.DataFrame([
        {"subset": "CONFIRMED vs FP", "pr_auc": average_precision_score(dev.y_main[conf_vs_fp], p[conf_vs_fp]),
         "roc_auc": roc_auc_score(dev.y_main[conf_vs_fp], p[conf_vs_fp])},
        {"subset": "CANDIDATE vs FP", "pr_auc": average_precision_score(dev.y_main[cand_vs_fp], p[cand_vs_fp]),
         "roc_auc": roc_auc_score(dev.y_main[cand_vs_fp], p[cand_vs_fp])},
    ])
    sub.to_csv(C.TABLES / "e3_main_model_subsets.csv", index=False)
    print(by.round(3).to_string())
    print(sub.round(4).to_string(index=False))

    # c) strict model (selected type) fitted on strict dev rows, scores the dev CANDIDATEs
    est = calibrated(tuned(sel), calibration_method(sel), ys, gs)
    fit(est, strict[C.CLEAN], ys, gs)
    cand = dev[dev.koi_disposition == "CANDIDATE"]
    pc = est.predict_proba(cand[C.CLEAN])[:, 1]
    dist = pd.Series(pc).describe(percentiles=[.1, .25, .5, .75, .9])
    dist.to_csv(C.TABLES / "e3_candidates_scored_by_strict_model.csv")
    print("strict-model probability for dev CANDIDATEs:", dist.round(3).to_dict())

    cols = ["label", "model", "pr_auc", "pr_auc_sd", "roc_auc", "mcc", "brier", "rank_by_pr_auc"]
    print(both[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
