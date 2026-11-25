"""Loading the KOI tables and building the labelled cohort.

Owner: Member 1 (data and labels).
"""
import numpy as np
import pandas as pd

from . import config as C


def load_raw(name: str) -> pd.DataFrame:
    """Load one of 'cumulative', 'dr24', 'dr25' exactly as downloaded."""
    return pd.read_csv(C.RAW_FILES[name], low_memory=False)


def add_labels(df: pd.DataFrame, disp_col: str = "koi_disposition") -> pd.DataFrame:
    """Add the two label columns used in the study.

    y_main   : 1 = planet-like (CANDIDATE or CONFIRMED), 0 = FALSE POSITIVE
    y_strict : 1 = CONFIRMED, 0 = FALSE POSITIVE, NaN for CANDIDATE (dropped in RQ3)
    """
    d = df[~df[disp_col].isin(C.DROP_LABELS)].copy()
    disp = d[disp_col]
    d["y_main"] = disp.isin(C.PLANET_LIKE).astype(int)
    d["y_strict"] = np.where(disp == "CONFIRMED", 1.0, np.where(disp == "FALSE POSITIVE", 0.0, np.nan))
    return d


def numeric_features(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Return the requested feature columns as floats, dropping any column that is entirely empty."""
    X = df[list(cols)].apply(pd.to_numeric, errors="coerce").astype(float)
    return X.loc[:, X.notna().any()]


def build_cohort(name: str = "cumulative") -> pd.DataFrame:
    """Labelled cohort for one table: ids, provenance, labels and every candidate column."""
    df = add_labels(load_raw(name))
    leaked = [c for c in C.CLEAN if c in C.EXCLUDED]
    assert not leaked, f"exclusion list violated: {leaked}"
    return df.reset_index(drop=True)


def stellar_type(teff: pd.Series) -> pd.Series:
    """Coarse spectral class from effective temperature (K), used only for error analysis."""
    bins = [0, 3900, 5300, 6000, 7500, np.inf]
    return pd.cut(teff, bins=bins, labels=["M", "K", "G", "F", "A+"])
