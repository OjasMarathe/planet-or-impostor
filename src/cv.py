"""Cross-validation harness: tuning, out-of-fold predictions, calibration wrappers.

Owner: Member 5 (evaluation and statistics); used by every experiment.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from . import config as C
from .metrics import all_metrics
from .splits import cv_splits, inner_cv

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def tune(factory, grid, X, y, groups, seed: int = C.SEED):
    """Grid search by PR-AUC with star-grouped 5-fold CV. Uses its own fold seed,
    different from the evaluation folds, to limit optimism."""
    est = factory()
    if not grid:
        return est.fit(X, y), {}
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 100)
    gs = GridSearchCV(est, grid, scoring="average_precision", cv=cv, n_jobs=-1, refit=False)
    gs.fit(X, y, groups=groups)
    best = gs.best_params_
    return factory().set_params(**best), best


def calibrated(est, method: str | None, y, groups, seed: int = C.SEED):
    """Wrap an estimator in grouped-CV calibration (fitted on training folds only)."""
    if method in (None, "none"):
        return clone(est)
    return _GroupCalibrated(est, method, seed)


class _GroupCalibrated:
    """CalibratedClassifierCV whose internal folds are grouped by star.
    Folds are built inside fit() from the groups of the rows it is given."""

    def __init__(self, est, method, seed):
        self.est, self.method, self.seed = est, method, seed

    def fit(self, X, y, groups):
        folds = inner_cv(y, groups, n_splits=3, seed=self.seed)
        self.model_ = CalibratedClassifierCV(clone(self.est), method=self.method, cv=folds)
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def fit(est, X, y, groups):
    if isinstance(est, _GroupCalibrated):
        return est.fit(X, y, groups)
    return est.fit(X, y)


def oof(est_maker, X, y, groups, n_repeats: int = C.CV_REPEATS, seed: int = C.SEED):
    """Out-of-fold predictions over repeated grouped CV.

    est_maker(): returns a fresh (possibly calibration-wrapped) estimator.
    Returns (fold metrics DataFrame, OOF probability matrix [n_rows, n_repeats]).
    """
    X = X.reset_index(drop=True) if hasattr(X, "reset_index") else X
    y = np.asarray(y)
    groups = np.asarray(groups)
    P = np.full((len(y), n_repeats), np.nan)
    rows = []
    for r, f, tr, va in cv_splits(y, groups, n_repeats=n_repeats, seed=seed):
        est = est_maker()
        fit(est, X.iloc[tr], y[tr], groups[tr])
        p = est.predict_proba(X.iloc[va])[:, 1]
        P[va, r] = p
        rows.append({"repeat": r, "fold": f, **all_metrics(y[va], p)})
    return pd.DataFrame(rows), P


def summarise(fold_df: pd.DataFrame, keys=("pr_auc", "roc_auc", "mcc", "macro_f1", "brier", "ece")) -> dict:
    out = {}
    for k in keys:
        out[k] = fold_df[k].mean()
        out[k + "_sd"] = fold_df[k].std(ddof=1)
    return out
