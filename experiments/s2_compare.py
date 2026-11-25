"""Stage 2, step 3: every method vs the Robovetter's verdict on trusted labels, with paired
star-level bootstrap intervals for the difference in mistakes.

Owner: Member 5. Reads the out-of-fold predictions saved by s2_vetter.py (no retraining).
The neural + Robovetter average was the combination planned in advance; the boosting rows and
other averages are extra comparisons, reported in full rather than picked after the fact.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef

from src import config as C
from src.metrics import star_bootstrap

d = pd.read_csv(C.PROC / "s2_scores.csv", index_col=0)
g = d[d.y_gold.notna()].copy()
y = g.y_gold.astype(int).to_numpy()
rv = (g.koi_pdisposition == "CANDIDATE").astype(float).to_numpy()
rvs = g.koi_score.fillna(pd.Series(rv, index=g.index)).to_numpy()
nn, gb = g["oof_Neural vetter"].to_numpy(), g["oof_Boosting on measurements only"].to_numpy()
methods = {
    "Robovetter (its verdict)": rv,
    "Neural vetter": nn,
    "Boosting on measurements only": gb,
    "Neural vetter + Robovetter score (planned)": (nn + rvs) / 2,
    "Boosting + Robovetter score": (gb + rvs) / 2,
    "Neural vetter + boosting": (nn + gb) / 2,
    "All three averaged": (nn + gb + rvs) / 3,
}
err = lambda p, i: int(((p[i] >= 0.5).astype(int) != y[i]).sum())
full = np.arange(len(y))
boots = {k: [] for k in methods}
for idx in star_bootstrap(g.kepid.to_numpy(), 2000, C.SEED):
    base = err(rv, idx)
    for k, p in methods.items():
        boots[k].append(err(p, idx) - base)
rows = []
for k, p in methods.items():
    yh = (p >= 0.5).astype(int)
    b = np.array(boots[k])
    rows.append({"method": k, "mistakes": err(p, full), "false_alarms": int(((yh == 1) & (y == 0)).sum()),
                 "misses": int(((yh == 0) & (y == 1)).sum()), "mcc": matthews_corrcoef(y, yh),
                 "pr_auc": average_precision_score(y, p) if k != "Robovetter (its verdict)" else np.nan,
                 "change_vs_robovetter": err(p, full) - err(rv, full),
                 "ci_low": np.percentile(b, 2.5), "ci_high": np.percentile(b, 97.5)})
res = pd.DataFrame(rows)
res.to_csv(C.TABLES / "s2_compare.csv", index=False)
print(f"trusted objects: {len(y)} ({y.sum()} planets) on {g.kepid.nunique()} stars")
print(res.round(4).to_string(index=False))
