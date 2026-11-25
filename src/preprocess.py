"""Preprocessing shared by all models. Every step is inside a Pipeline, so it is
fitted on training rows only (no information from validation or test rows).
"""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def signed_log(X):
    """sign(x) * log10(1 + |x|): tames heavy tails (period, depth, SNR span 4-6 orders of magnitude)."""
    X = np.asarray(X, dtype=float)
    return np.sign(X) * np.log10(1.0 + np.abs(X))


def scaled_steps():
    """For distance- and linear models: log, median impute, standardise."""
    return [
        ("log", FunctionTransformer(signed_log, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]


def tree_steps():
    """Trees are scale-free: only median imputation."""
    return [("impute", SimpleImputer(strategy="median"))]
