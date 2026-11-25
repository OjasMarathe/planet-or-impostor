"""E7 (RQ6): which objects are confidently misclassified, and what do they share?

Owner: Member 6 (explanations, error analysis, demo).
Confident false alarm: p >= 0.9 but the object is a FALSE POSITIVE.
Confident miss:        p <= 0.1 but the object is planet-like.
Analysed on the test set (final model) and, for more examples, on development OOF predictions.
The vetting flags and comments are NOT features; here they are used only to explain mistakes.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from src import config as C
from src.data import load_raw, stellar_type

HI, LO = 0.9, 0.1
BINS = {
    "koi_period": ([0, 1, 3, 10, 30, 100, np.inf], ["<1 d", "1-3 d", "3-10 d", "10-30 d", "30-100 d", ">100 d"]),
    "koi_depth": ([0, 100, 300, 1000, 10000, np.inf], ["<100 ppm", "100-300", "300-1k", "1k-10k", ">10k ppm"]),
    "koi_model_snr": ([0, 10, 20, 50, 100, np.inf], ["<10", "10-20", "20-50", "50-100", ">100"]),
    "koi_prad": ([0, 1.25, 2, 6, 15, np.inf], ["<1.25 Re", "1.25-2", "2-6", "6-15", ">15 Re"]),
}


def label_errors(d, p):
    d = d.copy()
    d["p"] = p
    d["conf_false_alarm"] = (d.p >= HI) & (d.y_main == 0)
    d["conf_miss"] = (d.p <= LO) & (d.y_main == 1)
    d["error@0.5"] = (d.p >= 0.5).astype(int) != d.y_main
    d["stellar_type"] = stellar_type(d.koi_steff)
    d["no_transit_fit"] = d.koi_depth.isna()
    return d


def by_group(d, col, bins=None):
    key = pd.cut(d[col], bins=bins[0], labels=bins[1]) if bins else d[col]
    g = d.groupby(key, observed=True)
    return pd.DataFrame({
        "n": g.size(), "false_pos_share": g.y_main.apply(lambda s: 1 - s.mean()),
        "error_rate@0.5": g["error@0.5"].mean(),
        "conf_false_alarm_rate": g.conf_false_alarm.mean(), "conf_miss_rate": g.conf_miss.mean(),
    }).assign(variable=col).reset_index(names="bin")


def main():
    ops = json.loads((C.TABLES / "operating_points.json").read_text())
    sel = ops["_selected_model"]
    raw = load_raw("cumulative").set_index("kepoi_name")

    t = pd.read_csv(C.PROC / "test_predictions.csv")
    t = label_errors(t, t["p_" + sel].values)

    tabs = []
    for col, b in BINS.items():
        tabs.append(by_group(t, col, b))
    for col in ["stellar_type", "koi_parm_prov", "no_transit_fit", "koi_disposition"]:
        tabs.append(by_group(t, col))
    grp = pd.concat(tabs, ignore_index=True)
    grp.to_csv(C.TABLES / "e7_error_rates_by_group.csv", index=False)

    # Which vetting flag explains the confident false alarms? (compare with all FPs)
    fa, fps = t[t.conf_false_alarm], t[t.y_main == 0]
    flag_cmp = pd.DataFrame({
        "share_among_confident_false_alarms": fa[C.FLAGS].fillna(0).astype(float).mean(),
        "share_among_all_false_positives": fps[C.FLAGS].fillna(0).astype(float).mean(),
    })
    flag_cmp.loc["only centroid flag (co)"] = [
        ((fa.koi_fpflag_co == 1) & (fa[["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_ec"]].sum(1) == 0)).mean(),
        ((fps.koi_fpflag_co == 1) & (fps[["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_ec"]].sum(1) == 0)).mean()]
    flag_cmp.to_csv(C.TABLES / "e7_flags_of_false_alarms.csv")

    miss = t[t.conf_miss]
    miss_disp = miss.koi_disposition.value_counts()
    miss_disp.to_csv(C.TABLES / "e7_confident_misses_by_disposition.csv")

    # Medians: confident errors vs correctly handled objects of the same true class
    feats = ["koi_period", "koi_depth", "koi_model_snr", "koi_prad", "koi_impact", "koi_duration", "koi_steff", "koi_srad"]
    med = pd.DataFrame({
        "confident false alarms": fa[feats].median(),
        "correct false positives (p<0.5)": fps[fps.p < 0.5][feats].median(),
        "confident misses": miss[feats].median(),
        "correct planet-like (p>=0.5)": t[(t.y_main == 1) & (t.p >= 0.5)][feats].median(),
    })
    med.to_csv(C.TABLES / "e7_medians.csv")

    # Gallery for the demo: most confident mistakes each way
    cols = ["kepoi_name", "koi_disposition", "p", "koi_period", "koi_depth", "koi_model_snr", "koi_prad",
            "koi_impact"] + C.FLAGS
    gal_fa = fa.sort_values("p", ascending=False)[cols].head(12).assign(kind="confident false alarm")
    gal_m = miss.sort_values("p")[cols].head(12).assign(kind="confident miss")
    gal = pd.concat([gal_fa, gal_m])
    gal["koi_comment"] = gal.kepoi_name.map(raw.koi_comment).fillna("")
    gal.to_csv(C.TABLES / "e7_gallery.csv", index=False)

    # Same summary on development OOF predictions (more examples, robustness check)
    dev = pd.read_csv(C.PROC / "cohort.csv")
    dev = dev[~dev.is_test].reset_index(drop=True)
    oofp = pd.read_csv(C.PROC / "e2_oof_dev.csv")[sel].values
    dd = label_errors(dev, oofp)
    fa2, fps2 = dd[dd.conf_false_alarm], dd[dd.y_main == 0]
    dev_flags = pd.DataFrame({"share_among_confident_false_alarms": fa2[C.FLAGS].fillna(0).astype(float).mean(),
                              "share_among_all_false_positives": fps2[C.FLAGS].fillna(0).astype(float).mean()})
    dev_flags.to_csv(C.TABLES / "e7_flags_of_false_alarms_dev.csv")

    facts = {
        "test_n": len(t), "confident_false_alarms": int(t.conf_false_alarm.sum()),
        "confident_misses": int(t.conf_miss.sum()),
        "share_of_fps_that_are_confident_false_alarms": float(t.conf_false_alarm.sum() / (t.y_main == 0).sum()),
        "share_of_planet_like_that_are_confident_misses": float(t.conf_miss.sum() / (t.y_main == 1).sum()),
        "dev_confident_false_alarms": int(dd.conf_false_alarm.sum()), "dev_confident_misses": int(dd.conf_miss.sum()),
        "dev_confident_misses_by_disposition": dd[dd.conf_miss].koi_disposition.value_counts().to_dict(),
    }
    (C.TABLES / "e7_facts.json").write_text(json.dumps(facts, indent=2))
    print(json.dumps(facts, indent=2))
    print(flag_cmp.round(3).to_string())
    print(dev_flags.round(3).to_string())
    print(miss_disp.to_string())
    print(med.round(2).to_string())
    print(grp.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
