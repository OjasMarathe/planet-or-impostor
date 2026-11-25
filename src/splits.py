"""Grouped splits: no host star (kepid) ever appears on both sides of a split.

Owner: Member 1 (data and labels).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config as C


def make_test_split(df: pd.DataFrame, y_col: str = "y_main", seed: int = C.SEED) -> pd.Series:
    """Return a boolean Series: True for rows in the held-out test set (about 20 %).

    One fold of a stratified, star-grouped 5-fold split is kept as the test set,
    so the class balance is preserved and every star sits entirely on one side.
    """
    sgkf = StratifiedGroupKFold(n_splits=C.TEST_FRACTION_FOLDS, shuffle=True, random_state=seed)
    _, test_idx = next(sgkf.split(df, df[y_col], groups=df["kepid"]))
    is_test = pd.Series(False, index=df.index)
    is_test.iloc[test_idx] = True
    return is_test


def cv_splits(y, groups, n_splits: int = C.CV_FOLDS, n_repeats: int = C.CV_REPEATS, seed: int = C.SEED):
    """Repeated stratified grouped K-fold. Yields (repeat, fold, train_idx, val_idx)."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    for r in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + 1 + r)
        for f, (tr, va) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
            yield r, f, tr, va


def inner_cv(y, groups, n_splits: int = 3, seed: int = C.SEED):
    """Grouped folds as a list, for calibration and tuning inside a training set."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(y)), np.asarray(y), np.asarray(groups)))


def assert_grouped(train_groups, test_groups):
    overlap = set(np.asarray(train_groups)) & set(np.asarray(test_groups))
    assert not overlap, f"{len(overlap)} stars appear on both sides of a split"
