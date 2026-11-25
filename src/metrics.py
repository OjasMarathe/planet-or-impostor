"""Metrics, calibration error, operating points and star-level bootstrap.

Owner: Member 5 (evaluation and statistics).
The primary metric (PR-AUC for the planet-like class) was fixed in the proposal.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, matthews_corrcoef,
    precision_recall_curve, roc_auc_score,
)


def ece(y, p, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width bins: sum_b (n_b/N) |mean(y_b) - mean(p_b)|."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def all_metrics(y, p, threshold: float = 0.5) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    yhat = (p >= threshold).astype(int)
    out = {
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "mcc": matthews_corrcoef(y, yhat),
        "macro_f1": f1_score(y, yhat, average="macro"),
        "brier": brier_score_loss(y, p),
        "ece": ece(y, p),
    }
    return {k: float(v) for k, v in out.items()}


def threshold_for_recall(y, p, target: float) -> float:
    """Largest threshold whose recall on (y, p) is still >= target."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target)[0]
    return float(thr[ok[-1]]) if len(ok) else 0.0


def recall_precision_at(y, p, threshold: float) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    yhat = p >= threshold
    tp = int((yhat & (y == 1)).sum())
    return {
        "threshold": threshold,
        "recall": tp / max(int((y == 1).sum()), 1),
        "precision": tp / max(int(yhat.sum()), 1),
        "flagged_fraction": float(yhat.mean()),
    }


def reliability(y, p, n_bins: int = 10) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append({"bin": b, "mean_pred": p[m].mean(), "frac_pos": y[m].mean(), "n": int(m.sum())})
    return pd.DataFrame(rows)


def star_bootstrap(groups, n_boot: int, seed: int):
    """Yield index arrays that resample whole stars with replacement (KOIs on one star stay together)."""
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([members[g] for g in pick])


def bootstrap_ci(y, preds: dict, groups, metric_fns: dict, n_boot: int, seed: int, ref: str | None = None):
    """Paired star-level bootstrap.

    preds: {model_name: probability array}. Every model is scored on the SAME resample,
    so differences against `ref` are paired. Returns (per-model CI table, paired-difference table).
    """
    y = np.asarray(y)
    rec = {m: {k: [] for k in metric_fns} for m in preds}
    for idx in star_bootstrap(groups, n_boot, seed):
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        for m, p in preds.items():
            pb = np.asarray(p)[idx]
            for k, fn in metric_fns.items():
                rec[m][k].append(fn(yb, pb))
    rows, diffs = [], []
    for m in preds:
        for k in metric_fns:
            a = np.array(rec[m][k])
            point = metric_fns[k](y, np.asarray(preds[m]))
            rows.append({"model": m, "metric": k, "estimate": point,
                         "ci_low": np.percentile(a, 2.5), "ci_high": np.percentile(a, 97.5)})
            if ref is not None and m != ref:
                d = np.array(rec[ref][k]) - a
                diffs.append({"reference": ref, "model": m, "metric": k,
                              "diff_estimate": metric_fns[k](y, np.asarray(preds[ref])) - point,
                              "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5),
                              "p_ref_not_better": float((d <= 0).mean())})
    return pd.DataFrame(rows), pd.DataFrame(diffs)
