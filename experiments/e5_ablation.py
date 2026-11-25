"""E5: feature-group ablation (transit-only, stellar-only, combined ...).

Owner: Member 3 (tree models). Development set, 3 x 5 grouped CV, tuned settings from E2.
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
    rows = []
    for group, cols in C.ABLATION.items():
        for name in MODELS:
            folds, _ = oof(lambda: tuned(name), dev[cols], y, g)
            rows.append({"features": group, "model": name, "n_features": len(cols), **summarise(folds)})
            print(f"{group:20s} {name:20s} PR-AUC={rows[-1]['pr_auc']:.4f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(C.TABLES / "e5_ablation.csv", index=False)
    print(res.pivot(index="features", columns="model", values="pr_auc").loc[list(C.ABLATION)].round(4).to_string())


if __name__ == "__main__":
    main()
