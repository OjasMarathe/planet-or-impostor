"""E1 (RQ1): how much do vetting outputs inflate performance? A leakage ladder.

Owner: Member 1 (data and labels), using the harness from Member 5.
Each tier adds columns to the clean set; models keep the settings tuned in E2 on clean
features, so only the information changes. Development set, 3 x 5 grouped CV.
Also scores two no-model 'leaky rules': koi_score used directly, and 'any flag set = FP'.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from src import config as C
from src.cv import oof, summarise
from src.metrics import all_metrics
from src.registry import load_cohort, tuned

MODELS = ["Logistic regression", "Decision tree", "Random forest", "Gradient boosting"]


def main():
    df = load_cohort()
    dev = df[~df.is_test].reset_index(drop=True)
    y, g = dev.y_main.values, dev.kepid.values

    rows = []
    for tier, cols in C.LADDER.items():
        for name in MODELS:
            folds, _ = oof(lambda: tuned(name), dev[cols], y, g)
            rows.append({"tier": tier, "model": name, "n_features": len(cols), **summarise(folds)})
            print(f"{tier:26s} {name:20s} PR-AUC={rows[-1]['pr_auc']:.4f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(C.TABLES / "e1_leakage_ladder.csv", index=False)

    # No-model rules on the same development rows
    rules = []
    s = dev.koi_score
    has = s.notna()
    rules.append({"rule": "koi_score as the probability (rows that have a score)", "rows": int(has.sum()),
                  **all_metrics(y[has], s[has].values)})
    anyflag = dev[C.FLAGS].fillna(0).sum(axis=1) > 0
    p_rule = (~anyflag).astype(float).values  # no flag -> planet-like
    rules.append({"rule": "any vetting flag set => false positive", "rows": len(dev), **all_metrics(y, p_rule)})
    pd.DataFrame(rules).to_csv(C.TABLES / "e1_leaky_rules.csv", index=False)
    print(pd.DataFrame(rules).round(4).to_string(index=False))

    # Who are the planet-like objects with a flag set? (They make the rules imperfect.)
    flagged_pl = dev[(y == 1) & anyflag.values].koi_disposition.value_counts()
    flagged_pl.to_csv(C.TABLES / "e1_flagged_planet_like.csv")
    print("planet-like objects with a flag set:", flagged_pl.to_dict())

    piv = res.pivot(index="tier", columns="model", values="pr_auc").loc[list(C.LADDER)]
    print(piv.round(4).to_string())


if __name__ == "__main__":
    main()
