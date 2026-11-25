"""E6 (RQ5): if reviewers can inspect only the top k objects, how many planet-like ones do they catch?

Owner: Member 5 (evaluation and statistics). Test set, selected model vs simple orderings.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from src import config as C


def found_curve(y_sorted):
    return np.cumsum(y_sorted)


def main():
    t = pd.read_csv(C.PROC / "test_predictions.csv")
    ops = json.loads((C.TABLES / "operating_points.json").read_text())
    sel = ops["_selected_model"]
    y = t.y_main.values
    n, npos = len(y), int(y.sum())
    rng = np.random.default_rng(C.SEED)

    orders = {
        f"{sel} (model)": np.argsort(-t["p_" + sel].values, kind="stable"),
        "Logistic regression (model)": np.argsort(-t["p_Logistic regression"].values, kind="stable"),
        "Highest SNR first": np.argsort(-t.koi_model_snr.fillna(-1).values, kind="stable"),
        "Shortest period first": np.argsort(t.koi_period.values, kind="stable"),
        "Longest period first": np.argsort(-t.koi_period.values, kind="stable"),
    }
    curves = {k: found_curve(y[o]) for k, o in orders.items()}
    rand = np.array([found_curve(y[rng.permutation(n)]) for _ in range(500)])
    curves["Random order (mean of 500)"] = rand.mean(0)

    ks = np.arange(1, n + 1)
    cdf = pd.DataFrame({"k": ks, **curves, "random_p2.5": np.percentile(rand, 2.5, 0),
                        "random_p97.5": np.percentile(rand, 97.5, 0), "perfect": np.minimum(ks, npos)})
    cdf.to_csv(C.TABLES / "e6_review_curve.csv", index=False)

    budgets = sorted(set(ops["_review_budgets"] + [int(round(0.25 * n))]))
    rows = []
    for k in budgets:
        r = {"k": k, "perfect": min(k, npos)}
        for name, c in curves.items():
            r[name] = float(c[k - 1])
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(C.TABLES / "e6_review_budget.csv", index=False)

    # How deep must reviewers go to catch 50 / 80 / 90 / 95 % of planet-like objects?
    depth = []
    for frac in [0.5, 0.8, 0.9, 0.95]:
        need = int(np.ceil(frac * npos))
        d = {"recall_target": frac}
        for name, c in curves.items():
            d[name] = int(np.searchsorted(c, need) + 1)
        depth.append(d)
    pd.DataFrame(depth).to_csv(C.TABLES / "e6_depth_needed.csv", index=False)
    print(f"test KOIs={n}, planet-like={npos}")
    print(tab.round(1).to_string(index=False))
    print(pd.DataFrame(depth).to_string(index=False))


if __name__ == "__main__":
    main()
