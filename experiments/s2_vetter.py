"""Stage 2, step 2: can an AI vetter beat the Robovetter, judged on TRUSTED labels?

Owners: Members 2 and 3 (models), Member 5 (comparison and statistics), Member 6 (disagreements).
Trusted labels ("gold"), DR25 KOIs only:
  planet   = CONFIRMED (and not certified false positive)
  impostor = CERTIFIED FP or CERTIFIED FA in the False Positive Working Group table
             (and not CONFIRMED); 7 objects on both lists are dropped.
The Robovetter's own verdict (koi_pdisposition) and score (koi_score) are compared, never used as inputs.
Every number is out-of-fold over 5 star-grouped folds; the same folds for every method.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src import config as C
from src.metrics import star_bootstrap
from src.preprocess import signed_log
from src.vetter_net import train_predict

VIEWS_L = ["local", "odd", "even", "secondary", "centroid"]
SCALARS = C.CLEAN + C.DIAGNOSTIC
assert not set(SCALARS) & set(C.EXCLUDED), "a vetting output slipped into the inputs"
NEG = {"CERTIFIED FP", "CERTIFIED FA"}
QUICK = os.environ.get("S2_QUICK") == "1"  # smoke test: 1 seed, 3 epochs, no file outputs kept
SEEDS = [0] if QUICK else [0, 1, 2]
EPOCHS = 3 if QUICK else 60
ABL_SEEDS = SEEDS[:2]  # ablations: 2 seeds; main model: 3


def load():
    d = pd.read_csv(C.RAW_FILES["dr25"], low_memory=False)
    f = pd.read_csv(C.RAW / "fpwg_certified_fp.csv", comment="#")[["kepoi_name", "fpwg_disp_status"]]
    d = d.merge(f, on="kepoi_name", how="left").set_index("kepoi_name")
    names, G, L, NQ = [], [], [], []
    for p in sorted((C.PROC / "lc_views").glob("*.npz")):
        z = np.load(p)
        for i, n in enumerate(z["names"]):
            names.append(str(n))
            G.append(z["global"][i][None])
            L.append(np.stack([z[v][i] for v in VIEWS_L]))
            NQ.append(z["n_quarters"][i])
    d = d.loc[names].copy()
    conf = d.koi_disposition == "CONFIRMED"
    neg = d.fpwg_disp_status.isin(NEG)
    d["y_gold"] = np.where(conf & ~neg, 1.0, np.where(neg & ~conf, 0.0, np.nan))
    d["rv_disp"] = (d.koi_pdisposition == "CANDIDATE").astype(float)
    d["rv_score"] = d.koi_score.fillna(d.rv_disp)
    d["n_quarters"] = NQ
    S = d[SCALARS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return d, np.array(G, np.float32), np.array(L, np.float32), S


def prep(S, tr):
    X = signed_log(S)
    med = np.nanmedian(X[tr], 0)
    med = np.where(np.isfinite(med), med, 0)
    X = np.where(np.isfinite(X), X, med)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    return ((X - mu) / sd).astype(np.float32)


def inner_split(groups, tr, seed):
    stars = np.unique(groups[tr])
    rng = np.random.default_rng(seed)
    val_stars = set(rng.choice(stars, size=max(1, len(stars) // 10), replace=False))
    m = np.array([g in val_stars for g in groups[tr]])
    return tr[~m], tr[m]


def scores(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    return {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
            "mcc": matthews_corrcoef(y, yh), "errors": int((yh != y).sum()),
            "false_alarms": int(((yh == 1) & (y == 0)).sum()), "misses": int(((yh == 0) & (y == 1)).sum())}


def main():
    t_start = time.time()
    d, G, L, S = load()
    gold = np.flatnonzero(d.y_gold.notna().to_numpy())
    y_all = d.y_gold.to_numpy()
    yg = y_all[gold].astype(int)
    groups = d.kepid.to_numpy()
    rv_y = d.rv_disp.to_numpy()
    print(f"KOIs with views: {len(d)}; gold: {len(gold)} ({yg.sum()} planets, {len(yg)-yg.sum()} impostors) "
          f"on {len(np.unique(groups[gold]))} stars", flush=True)

    variants = {
        "Neural vetter": dict(kw={}, chans=slice(None), label="gold"),
        "Neural vetter, no centroid view": dict(kw={}, chans=slice(0, 4), label="gold"),
        "Neural vetter, light curves only": dict(kw={"use_scalars": False}, chans=slice(None), label="gold"),
        "Neural vetter trained on Robovetter verdicts": dict(kw={}, chans=slice(None), label="robovetter"),
    }
    oof = {k: np.full(len(d), np.nan) for k in list(variants) + ["Boosting on measurements only"]}
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=C.SEED)
    for fold, (a, b) in enumerate(sgkf.split(gold, yg, groups[gold])):
        tr, te = gold[a], gold[b]
        Sp = prep(S, tr)
        hgb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.03, max_leaf_nodes=31,
                                             random_state=C.SEED).fit(Sp[tr], y_all[tr].astype(int))
        oof["Boosting on measurements only"][te] = hgb.predict_proba(Sp[te])[:, 1]
        for name, v in variants.items():
            lab = y_all if v["label"] == "gold" else rv_y
            ps = []
            for s in (SEEDS if name == "Neural vetter" else ABL_SEEDS):
                tr2, va = inner_split(groups, tr, 100 * fold + s)
                ps.append(train_predict(G, L[:, v["chans"]], Sp, np.nan_to_num(lab), tr2, va, te, seed=s, epochs=EPOCHS, **v["kw"]))
            oof[name][te] = np.mean(ps, 0)
        print(f"fold {fold+1}/5 done, {(time.time()-t_start)/60:.1f} min", flush=True)

    rv_score, rv_disp = d.rv_score.to_numpy(), d.rv_disp.to_numpy()
    preds = {"Robovetter (its verdict)": rv_disp, "Robovetter (its score)": rv_score, **oof,
             "Neural vetter + Robovetter score (average)": (oof["Neural vetter"] + rv_score) / 2}
    res = pd.DataFrame([{"method": k, **scores(yg, p[gold])} for k, p in preds.items()])
    res.loc[res.method == "Robovetter (its verdict)", ["pr_auc", "roc_auc"]] = np.nan  # a yes/no verdict has no ranking
    res.to_csv(C.TABLES / "s2_vs_robovetter.csv", index=False)

    # Paired star-level bootstrap: neural vetter vs Robovetter
    ai, rvd, rvs, both = oof["Neural vetter"][gold], rv_disp[gold], rv_score[gold], preds["Neural vetter + Robovetter score (average)"][gold]
    rec = {k: [] for k in ["errors_ai_minus_rv", "mcc_ai_minus_rv", "prauc_ai_minus_rvscore", "errors_combo_minus_rv"]}
    for idx in star_bootstrap(groups[gold], 1000, C.SEED):
        yb = yg[idx]
        e = lambda p: int(((p[idx] >= 0.5).astype(int) != yb).sum())
        rec["errors_ai_minus_rv"].append(e(ai) - e(rvd))
        rec["errors_combo_minus_rv"].append(e(both) - e(rvd))
        rec["mcc_ai_minus_rv"].append(matthews_corrcoef(yb, (ai[idx] >= .5).astype(int)) - matthews_corrcoef(yb, rvd[idx].astype(int)))
        rec["prauc_ai_minus_rvscore"].append(average_precision_score(yb, ai[idx]) - average_precision_score(yb, rvs[idx]))
    boot = pd.DataFrame([{"comparison": k, "mean": np.mean(v), "ci_low": np.percentile(v, 2.5),
                          "ci_high": np.percentile(v, 97.5)} for k, v in rec.items()])
    boot.to_csv(C.TABLES / "s2_bootstrap.csv", index=False)

    # Who is right when they disagree?
    aih = (ai >= 0.5).astype(int)
    dis = aih != rvd
    disagree = {"disagreements": int(dis.sum()), "ai_right": int((dis & (aih == yg)).sum()),
                "robovetter_right": int((dis & (rvd == yg)).sum())}

    # Final model on all gold -> score every other DR25 KOI with views
    others = np.flatnonzero(d.y_gold.isna().to_numpy())
    Sp = prep(S, gold)
    ps = []
    for s in SEEDS:
        tr2, va = inner_split(groups, gold, 999 + s)
        ps.append(train_predict(G, L, Sp, np.nan_to_num(y_all), tr2, va, others, seed=s, epochs=EPOCHS))
    p_all = oof["Neural vetter"].copy()
    p_all[others] = np.mean(ps, 0)
    out = d[["kepid", "koi_disposition", "koi_pdisposition", "koi_score", "fpwg_disp_status", "y_gold", "n_quarters",
             "koi_period", "koi_depth", "koi_prad", "koi_model_snr"]].copy()
    out["p_ai"] = p_all
    for k, p in oof.items():
        out["oof_" + k] = p
    out.to_csv(C.PROC / "s2_scores.csv")

    # FPWG "possible planets" the Robovetter rejected; DR24->DR25 label flips
    pp = out[out.fpwg_disp_status == "POSSIBLE PLANET"]
    pp_rej = pp[pp.koi_pdisposition == "FALSE POSITIVE"]
    fl = pd.read_csv(C.PROC / "e4_flips.csv", index_col=0)
    fl = fl[fl.change != "stable"].join(out.p_ai, how="inner")
    flips = fl.groupby("change").apply(lambda g: pd.Series({
        "n": len(g), "ai_agrees_with_dr25": ((g.p_ai >= .5).astype(int) == g.y25).mean(),
        "ai_agrees_with_dr24": ((g.p_ai >= .5).astype(int) == g.y24).mean()}), include_groups=False)
    flips.to_csv(C.TABLES / "s2_flips.csv")
    pp_rej.sort_values("p_ai", ascending=False).head(20).to_csv(C.TABLES / "s2_possible_planets_top.csv")
    facts = {"kois_with_views": len(d), "dr25_kois": 8054, "gold": len(gold), "gold_planets": int(yg.sum()),
             "gold_stars": int(len(np.unique(groups[gold]))), **disagree,
             "possible_planets": len(pp), "possible_planets_rejected_by_robovetter": len(pp_rej),
             "of_those_ai_says_planet": int((pp_rej.p_ai >= .5).sum()),
             "median_quarters": float(np.median(d.n_quarters)), "minutes": round((time.time() - t_start) / 60, 1)}
    (C.TABLES / "s2_facts.json").write_text(json.dumps(facts, indent=2))
    print(res.round(4).to_string(index=False))
    print(boot.round(4).to_string(index=False))
    print(flips.round(3).to_string())
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
