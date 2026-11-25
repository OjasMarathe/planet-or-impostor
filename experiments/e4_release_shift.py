"""E4 (RQ4): train on one catalog release, test on the other. Plus the label-flip study.

Owner: Member 4 (release shift).

Design A (main, star-grouped). Shared KOIs' stars are split into 5 grouped folds. For each
fold the model trains on the SOURCE release (every source row whose star is not in the test
fold) and is scored on the test-fold KOIs four ways:
    in-release      : source features, source labels      (no shift)
    feature shift   : target features, source labels
    label shift     : source features, target labels
    full shift      : target features, target labels      (the proposal's DR24 -> DR25 test)
Training is identical across the four, so the differences isolate what changed.

Design B (as literally proposed). Train on the whole source table, test on the target table's
shared KOIs. The same objects appear on both sides (in different releases), so this is an
optimistic reference, reported but not used for conclusions.

Label flips. Which shared KOIs change label between releases, what they look like, which DR25
vetting flag caused the change, and whether the model already finds them hard.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src import config as C
from src.cv import calibrated, fit
from src.metrics import all_metrics
from src.registry import calibration_method, tuned
from src.release import flip_table, flips, load_release, matched_pair


def run(name, src, tgt, src_all, label="y_main"):
    """Design A for one direction. src/tgt: shared KOIs aligned; src_all: full source table."""
    y_strat = tgt[label].values
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=C.SEED)
    P = {k: np.full(len(src), np.nan) for k in ["src", "tgt"]}
    for tr_idx, te_idx in sgkf.split(np.zeros(len(src)), y_strat, src.kepid.values):
        test_stars = set(src.kepid.values[te_idx])
        train = src_all[~src_all.kepid.isin(test_stars)]
        est = calibrated(tuned(name), calibration_method(name), train[label].values, train.kepid.values)
        fit(est, train[C.CLEAN], train[label].values, train.kepid.values)
        P["src"][te_idx] = est.predict_proba(src.iloc[te_idx][C.CLEAN])[:, 1]
        P["tgt"][te_idx] = est.predict_proba(tgt.iloc[te_idx][C.CLEAN])[:, 1]
    ys, yt = src[label].values, tgt[label].values
    res = {
        "in-release": all_metrics(ys, P["src"]),
        "feature shift": all_metrics(ys, P["tgt"]),
        "label shift": all_metrics(yt, P["src"]),
        "full shift": all_metrics(yt, P["tgt"]),
    }
    return res, P


def main():
    a, b = matched_pair()            # DR24, DR25 restricted to shared KOIs (aligned)
    dr24, dr25 = load_release("dr24"), load_release("dr25")
    facts = {"dr24_rows": len(dr24), "dr25_rows": len(dr25), "shared_kois": len(a),
             "shared_stars": int(a.kepid.nunique()), "only_dr24": len(dr24) - len(a), "only_dr25": len(dr25) - len(a)}

    # ---- label flips
    flip_table(a, b, "koi_disposition").to_csv(C.TABLES / "e4_flips_disposition.csv")
    flip_table(a, b, "koi_pdisposition").to_csv(C.TABLES / "e4_flips_pdisposition.csv")
    fl = flips(a, b)
    facts["flips_main_label"] = fl.change.value_counts().to_dict()
    facts["flips_pdisposition"] = int((fl.pdisp24 != fl.pdisp25).sum())

    # ---- shift experiments
    rows, preds = [], {}
    for name in ["Logistic regression", "Random forest", "Gradient boosting"]:
        for direction, src, tgt, src_all in [("DR24 -> DR25", a, b, dr24), ("DR25 -> DR24", b, a, dr25)]:
            res, P = run(name, src, tgt, src_all)
            for setting, m in res.items():
                rows.append({"model": name, "direction": direction, "setting": setting, **m})
            preds[(name, direction)] = P
            print(f"{name:20s} {direction}  in={res['in-release']['pr_auc']:.4f}  "
                  f"feat={res['feature shift']['pr_auc']:.4f}  lab={res['label shift']['pr_auc']:.4f}  "
                  f"full={res['full shift']['pr_auc']:.4f}", flush=True)
            # Design B (optimistic, same objects on both sides)
            est = calibrated(tuned(name), calibration_method(name), src_all.y_main.values, src_all.kepid.values)
            fit(est, src_all[C.CLEAN], src_all.y_main.values, src_all.kepid.values)
            pB = est.predict_proba(tgt[C.CLEAN])[:, 1]
            rows.append({"model": name, "direction": direction, "setting": "design B (same objects)",
                         **all_metrics(tgt.y_main.values, pB)})
    res = pd.DataFrame(rows)
    res.to_csv(C.TABLES / "e4_release_shift.csv", index=False)

    # ---- do models find flipped objects hard?
    sel = json.loads((C.TABLES / "operating_points.json").read_text())["_selected_model"]
    sel = sel if (sel, "DR24 -> DR25") in preds else "Random forest"
    P = preds[(sel, "DR24 -> DR25")]
    fl["p_dr24model_on_dr24"] = P["src"]
    fl["p_dr24model_on_dr25"] = P["tgt"]
    P2 = preds[(sel, "DR25 -> DR24")]
    fl["p_dr25model_on_dr25"] = P2["src"]
    fl["margin_dr25"] = (fl.p_dr25model_on_dr25 - 0.5).abs()
    hard = fl.groupby("change").agg(
        n=("y25", "size"),
        mean_p_dr25_model=("p_dr25model_on_dr25", "mean"),
        median_margin=("margin_dr25", "median"),
        share_uncertain_0p3_0p7=("p_dr25model_on_dr25", lambda s: s.between(0.3, 0.7).mean()),
        dr24_model_agrees_with_dr25_label=("p_dr24model_on_dr25", lambda s: np.nan),
    )
    for ch in hard.index:
        m = fl.change == ch
        hard.loc[ch, "dr24_model_agrees_with_dr25_label"] = ((fl.loc[m, "p_dr24model_on_dr25"] >= 0.5).astype(int) == fl.loc[m, "y25"]).mean()
    hard.to_csv(C.TABLES / "e4_flip_hardness.csv")

    # what flipped objects look like and which DR25 flag fired
    desc = fl.groupby("change")[["koi_model_snr_25", "koi_period_25", "koi_depth_25", "koi_prad_25", "koi_impact_25"]].median()
    desc.to_csv(C.TABLES / "e4_flip_feature_medians.csv")
    pf = fl[fl.change == "planet-like -> FP"]
    flag_share = pd.Series({f: pf[f + "_25"].fillna(0).astype(float).mean() for f in C.FLAGS})
    flag_share.to_csv(C.TABLES / "e4_flags_behind_planet_to_fp.csv")
    fp2 = fl[fl.change == "FP -> planet-like"]
    flag24 = pd.Series({f: fp2[f + "_24"].fillna(0).astype(float).mean() for f in C.FLAGS})
    flag24.to_csv(C.TABLES / "e4_dr24_flags_of_fp_to_planet.csv")
    fl.to_csv(C.PROC / "e4_flips.csv")
    (C.TABLES / "e4_facts.json").write_text(json.dumps(facts, indent=2))

    print(json.dumps(facts, indent=2))
    print(res.pivot_table(index=["model", "direction"], columns="setting", values="pr_auc").round(4).to_string())
    print(hard.round(3).to_string())
    print(desc.round(2).to_string())
    print("DR25 flags behind planet-like -> FP:", flag_share.round(3).to_dict())
    print("DR24 flags of FP -> planet-like:", flag24.round(3).to_dict())


if __name__ == "__main__":
    main()
