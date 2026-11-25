"""E2: model comparison on clean features with star-grouped repeated CV (the main results table).

Owner: Member 5 runs the harness; Member 2 owns the classical models, Member 3 the trees.
Steps, all on the development set only (the test set is never touched here):
  1. tune each model by PR-AUC (grouped 5-fold, its own fold seed)
  2. out-of-fold scores over 3 x 5 grouped CV, uncalibrated
  3. the same with Platt (sigmoid) and isotonic calibration fitted inside each training fold
  4. keep the calibration with the lowest Brier score per model
  5. fix the operating point (threshold for 95 % recall) from OOF predictions, before any test use
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from src import config as C
from src.cv import calibrated, oof, summarise, tune
from src.metrics import reliability, threshold_for_recall
from src.registry import ALL, BEST_PARAMS, CALIB_CHOICE, load_cohort


def main():
    df = load_cohort()
    dev = df[~df.is_test].reset_index(drop=True)
    X, y, g = dev[C.CLEAN], dev.y_main.values, dev.kepid.values

    best, rows, choice, oof_store, rel_rows = {}, [], {}, {}, []
    for name, (factory, grid) in ALL.items():
        t0 = time.time()
        est, params = tune(factory, grid, X, y, g)
        best[name] = params
        methods = ["none"] if name == "Majority baseline" else ["none", "sigmoid", "isotonic"]
        results = {}
        for m in methods:
            folds, P = oof(lambda: calibrated(est, m, y, g), X, y, g)
            results[m] = (summarise(folds), P)
            rows.append({"model": name, "calibration": m, **results[m][0]})
        pick = min(results, key=lambda m: results[m][0]["brier"])
        choice[name] = pick
        oof_store[name] = results[pick][1][:, 0]
        rel = reliability(y, results["none"][1][:, 0]).assign(model=name, calibration="none")
        rel2 = reliability(y, results[pick][1][:, 0]).assign(model=name, calibration=pick)
        rel_rows += [rel, rel2]
        print(f"{name:22s} {time.time()-t0:6.1f}s  params={params}  calib={pick}  "
              f"PR-AUC={results[pick][0]['pr_auc']:.4f}", flush=True)

    BEST_PARAMS.write_text(json.dumps(best, indent=2, default=str))
    CALIB_CHOICE.write_text(json.dumps(choice, indent=2))
    res = pd.DataFrame(rows)
    res.to_csv(C.TABLES / "e2_cv_all_calibrations.csv", index=False)
    main_tab = res[[choice[m] == c for m, c in zip(res.model, res.calibration)]].copy()
    main_tab = main_tab.sort_values("pr_auc", ascending=False)
    main_tab.to_csv(C.TABLES / "e2_main_results.csv", index=False)
    pd.concat(rel_rows).to_csv(C.TABLES / "e2_reliability.csv", index=False)

    # Operating points, fixed here from development OOF predictions (before the final test).
    ops = {}
    for name, p in oof_store.items():
        if name == "Majority baseline":
            continue
        ops[name] = {"thr_recall95": threshold_for_recall(y, p, C.TARGET_RECALL), "thr_default": 0.5}
    best_model = main_tab[main_tab.model != "Majority baseline"].iloc[0].model
    ops["_selected_model"] = best_model
    ops["_review_budgets"] = C.REVIEW_BUDGETS
    (C.TABLES / "operating_points.json").write_text(json.dumps(ops, indent=2))
    pd.DataFrame(oof_store).assign(kepid=g, y=y).to_csv(C.PROC / "e2_oof_dev.csv", index=False)

    cols = ["model", "calibration", "pr_auc", "pr_auc_sd", "roc_auc", "mcc", "macro_f1", "brier", "ece"]
    print(main_tab[cols].round(4).to_string(index=False))
    print("selected model:", best_model)


if __name__ == "__main__":
    main()
