"""E0 (week 1): column audit, frozen exclusion list, cohort, class balance, missingness, grouped split.

Owner: Member 1 (data and labels).
Outputs:
  data/processed/cohort.csv            labelled cohort + split column
  results/tables/column_audit.csv      role of every one of the 153 columns
  results/tables/class_balance.csv
  results/tables/missingness.csv
  results/tables/split_summary.csv
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from src import config as C
from src.data import build_cohort, load_raw
from src.splits import assert_grouped, make_test_split


def column_audit(cols):
    roles = []
    for c in cols:
        if c in C.EXCLUDED:
            role, why = "excluded", C.EXCLUDED[c]
        elif c in C.TRANSIT:
            role, why = "feature: transit", "light-curve fit measurement"
        elif c in C.DERIVED:
            role, why = "feature: derived", "planet quantity from transit fit + star"
        elif c in C.STELLAR:
            role, why = "feature: stellar", "host-star property"
        elif c in C.DIAGNOSTIC:
            role, why = "grey zone (E1 only)", "diagnostic statistic read by the Robovetter"
        else:
            role, why = "unused", "not in the proposal's feature list"
        roles.append({"column": c, "role": role, "reason": why})
    return pd.DataFrame(roles)


def main():
    raw = load_raw("cumulative")
    audit = column_audit(raw.columns)
    audit.to_csv(C.TABLES / "column_audit.csv", index=False)

    df = build_cohort("cumulative")
    df["is_test"] = make_test_split(df)
    assert_grouped(df.loc[~df.is_test, "kepid"], df.loc[df.is_test, "kepid"])
    keep = ["kepid", "kepoi_name", "koi_disposition", "koi_pdisposition", "koi_parm_prov", "koi_fittype",
            "y_main", "y_strict", "is_test"] + C.CLEAN + C.DIAGNOSTIC + C.FLAGS + C.SCORE
    df[keep].to_csv(C.PROC / "cohort.csv", index=False)

    n_dropped = len(raw) - len(df)
    bal = (df.koi_disposition.value_counts().rename("n").to_frame()
           .assign(share=lambda t: t.n / t.n.sum()))
    bal.loc["planet-like (main positive)"] = [int(df.y_main.sum()), df.y_main.mean()]
    bal.loc["NOT DISPOSITIONED dropped"] = [n_dropped, n_dropped / len(raw)]
    bal.to_csv(C.TABLES / "class_balance.csv")

    miss = pd.DataFrame({
        "missing_all": df[C.CLEAN].isna().mean(),
        "missing_planet_like": df.loc[df.y_main == 1, C.CLEAN].isna().mean(),
        "missing_false_pos": df.loc[df.y_main == 0, C.CLEAN].isna().mean(),
    })
    miss.to_csv(C.TABLES / "missingness.csv")

    rows = []
    for name, part in [("development", df[~df.is_test]), ("test", df[df.is_test]), ("all", df)]:
        rows.append({"part": name, "kois": len(part), "stars": part.kepid.nunique(),
                     "planet_like_share": part.y_main.mean(),
                     "confirmed": int((part.koi_disposition == "CONFIRMED").sum()),
                     "candidate": int((part.koi_disposition == "CANDIDATE").sum()),
                     "false_positive": int((part.koi_disposition == "FALSE POSITIVE").sum())})
    split = pd.DataFrame(rows)
    split.to_csv(C.TABLES / "split_summary.csv", index=False)

    stars = df.groupby("kepid").size()
    facts = {
        "raw_rows": len(raw), "raw_columns": raw.shape[1], "cohort_rows": len(df),
        "stars": int(df.kepid.nunique()), "multi_koi_stars": int((stars > 1).sum()),
        "kois_on_multi_stars": int(stars[stars > 1].sum()), "max_kois_per_star": int(stars.max()),
        "n_clean_features": len(C.CLEAN), "rows_without_transit_fit": int(df.koi_depth.isna().sum()),
        "audit_counts": audit.role.value_counts().to_dict(),
    }
    (C.TABLES / "cohort_facts.json").write_text(json.dumps(facts, indent=2))
    print(bal.round(3).to_string())
    print(split.round(3).to_string(index=False))
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
