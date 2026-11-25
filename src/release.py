"""DR24 / DR25 matching and label-flip bookkeeping.

Owner: Member 4 (release shift).
In the DR24 and DR25 tables, koi_pdisposition is that release's own Kepler-data
disposition (CANDIDATE / FALSE POSITIVE). koi_disposition equals it, except that
objects confirmed as planets (in the archive today) read CONFIRMED in both tables.
"""
import pandas as pd

from . import config as C
from .data import add_labels, load_raw


def load_release(name: str) -> pd.DataFrame:
    d = add_labels(load_raw(name))
    d["y_pdisp"] = (d.koi_pdisposition == "CANDIDATE").astype(int)
    return d.set_index("kepoi_name", drop=False)


def matched_pair():
    """Both releases restricted to KOIs present in both, aligned on kepoi_name."""
    a, b = load_release("dr24"), load_release("dr25")
    shared = a.index.intersection(b.index)
    a, b = a.loc[shared], b.loc[shared]
    assert (a.kepid.values == b.kepid.values).all(), "a KOI changed host star between releases"
    return a, b


def flip_table(a: pd.DataFrame, b: pd.DataFrame, col: str) -> pd.DataFrame:
    return pd.crosstab(a[col].rename(f"DR24 {col}"), b[col].rename(f"DR25 {col}"), margins=True)


def flips(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """One row per shared KOI with both labels and the kind of change."""
    out = pd.DataFrame({
        "kepid": a.kepid.values,
        "disp24": a.koi_disposition.values, "disp25": b.koi_disposition.values,
        "pdisp24": a.koi_pdisposition.values, "pdisp25": b.koi_pdisposition.values,
        "y24": a.y_main.values, "y25": b.y_main.values,
    }, index=a.index)
    out["change"] = "stable"
    out.loc[(out.y24 == 1) & (out.y25 == 0), "change"] = "planet-like -> FP"
    out.loc[(out.y24 == 0) & (out.y25 == 1), "change"] = "FP -> planet-like"
    for f in C.FLAGS:
        out[f + "_25"] = b[f].values
        out[f + "_24"] = a[f].values
    for c in ["koi_model_snr", "koi_period", "koi_depth", "koi_prad", "koi_impact"]:
        out[c + "_24"] = pd.to_numeric(a[c], errors="coerce").values
        out[c + "_25"] = pd.to_numeric(b[c], errors="coerce").values
    return out
