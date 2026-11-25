"""Explanations: permutation importance (all models), partial dependence for period and depth,
SHAP for the random forest.

Owner: Member 6 (explanations, error analysis, demo).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence, permutation_importance

from src import config as C
from src.cv import calibrated, fit
from src.registry import ALL, calibration_method, load_cohort, tuned


def base_name(c):
    return c.replace("_err1", "").replace("_err2", "")


def grouped_permutation(est, X, y, n_repeats=10, seed=C.SEED):
    """Permute a measurement together with its two error columns (they carry overlapping information)."""
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    base = average_precision_score(y, est.predict_proba(X)[:, 1])
    groups = {}
    for c in X.columns:
        groups.setdefault(base_name(c), []).append(c)
    out = []
    for gname, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].values[perm]
            drops.append(base - average_precision_score(y, est.predict_proba(Xp)[:, 1]))
        out.append({"measure": gname, "pr_auc_drop": np.mean(drops), "sd": np.std(drops)})
    return pd.DataFrame(out).sort_values("pr_auc_drop", ascending=False)


def main():
    df = load_cohort()
    dev, test = df[~df.is_test].reset_index(drop=True), df[df.is_test].reset_index(drop=True)
    Xd, yd, Xt, yt = dev[C.CLEAN], dev.y_main.values, test[C.CLEAN], test.y_main.values

    # 1) permutation importance on the test set, every model, measurement-level groups
    imp = []
    fitted = {}
    for name in ALL:
        if name == "Majority baseline":
            continue
        est = calibrated(tuned(name), calibration_method(name), yd, dev.kepid.values)
        fit(est, Xd, yd, dev.kepid.values)
        fitted[name] = est
        gi = grouped_permutation(est, Xt, yt).assign(model=name)
        imp.append(gi)
        print(name, gi.head(5)[["measure", "pr_auc_drop"]].round(4).values.tolist(), flush=True)
    imp = pd.concat(imp)
    imp.to_csv(C.TABLES / "expl_permutation_importance.csv", index=False)

    # 2) partial dependence for period and depth (log-spaced grids), forest, boosting, logistic
    pdp_rows = []
    for name in ["Random forest", "Gradient boosting", "Logistic regression"]:
        est = tuned(name).fit(Xd, yd)
        for feat in ["koi_period", "koi_depth"]:
            v = Xd[feat].dropna()
            grid = np.logspace(np.log10(max(v.quantile(0.02), 1e-3)), np.log10(v.quantile(0.98)), 40)
            pd_res = partial_dependence(est, Xd, [feat], custom_values={feat: grid}, kind="average")
            for x, yv in zip(pd_res["grid_values"][0], pd_res["average"][0]):
                pdp_rows.append({"model": name, "feature": feat, "value": x, "avg_prob": yv})
    pd.DataFrame(pdp_rows).to_csv(C.TABLES / "expl_partial_dependence.csv", index=False)

    # 3) SHAP for the random forest (uncalibrated forest fitted on all development rows)
    import shap
    rf = joblib.load(C.MODELS / "random_forest_uncalibrated.joblib")
    Xi = rf.named_steps["impute"].transform(Xt)
    sample = np.random.default_rng(C.SEED).choice(len(Xi), size=min(800, len(Xi)), replace=False)
    expl = shap.TreeExplainer(rf.named_steps["clf"])
    sv = expl.shap_values(Xi[sample])
    sv = sv[:, :, 1] if np.ndim(sv) == 3 else (sv[1] if isinstance(sv, list) else sv)
    shap_df = pd.DataFrame(sv, columns=C.CLEAN)
    shap_df.to_csv(C.TABLES / "expl_shap_values_sample.csv", index=False)
    pd.DataFrame(Xi[sample], columns=C.CLEAN).to_csv(C.TABLES / "expl_shap_inputs_sample.csv", index=False)
    mean_abs = shap_df.abs().mean().rename("mean_abs_shap")
    grouped = mean_abs.groupby(base_name).sum().sort_values(ascending=False)
    grouped.to_csv(C.TABLES / "expl_shap_mean_abs_by_measure.csv")
    print("SHAP mean |value| by measure:", grouped.round(4).head(10).to_dict())


if __name__ == "__main__":
    main()
