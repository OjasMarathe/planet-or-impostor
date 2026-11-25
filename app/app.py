"""Planet or Impostor - triage demo.

Owner: Member 6. Start it from anywhere with:
    ./run_demo.sh            (from the project folder)
which runs .venv/bin/streamlit run app/app.py

Tab 1: enter or load a KOI's measurements; see the calibrated probability that it is planet-like,
the decision at the two operating points fixed before the test, and what pushed the prediction.
Tab 2: confidently wrong cases (E7). Tab 3: Stage 2, the AI vetter vs the Robovetter.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src import config as C

st.set_page_config(page_title="Planet or Impostor", page_icon="🪐", layout="wide")

MAIN = [  # (column, label, unit)
    ("koi_period", "Orbital period", "days"), ("koi_duration", "Transit duration", "hours"),
    ("koi_depth", "Transit depth", "ppm"), ("koi_impact", "Impact parameter", ""),
    ("koi_model_snr", "Signal-to-noise", ""), ("koi_prad", "Planet radius", "Earth radii"),
    ("koi_teq", "Equilibrium temperature", "K"), ("koi_insol", "Insolation", "Earth flux"),
    ("koi_steff", "Stellar temperature", "K"), ("koi_slogg", "Stellar log g", "cgs"),
    ("koi_srad", "Stellar radius", "Solar radii"), ("koi_kepmag", "Kepler magnitude", "mag"),
]
NEEDED = [C.TABLES / "operating_points.json", C.PROC / "test_predictions.csv", C.TABLES / "e7_gallery.csv",
          C.PROC / "cohort.csv", C.TABLES / "e2_best_params.json", C.TABLES / "e2_calibration_choice.json"]

missing = [p.relative_to(ROOT) for p in NEEDED if not p.exists()]
if missing:
    st.error("The demo needs the project's results, which are not built yet. In a terminal, from the "
             "project folder, run `./run_all.sh` (about 40 minutes), then reload this page.\n\nMissing: "
             + ", ".join(str(m) for m in missing))
    st.stop()


@st.cache_resource(show_spinner="Loading the model (first start builds it, a few seconds)...")
def load():
    """Load the boosting model; rebuild it from the saved settings if the model files are absent
    (results/models/ is not kept in git, so a fresh copy of the project has none)."""
    from src.cv import calibrated, fit
    from src.registry import calibration_method, load_cohort, tuned
    cal_p, raw_p = C.MODELS / "gradient_boosting_calibrated.joblib", C.MODELS / "gradient_boosting_uncalibrated.joblib"
    if not (cal_p.exists() and raw_p.exists()):
        df = load_cohort()
        dev = df[~df.is_test]
        est = calibrated(tuned("Gradient boosting"), calibration_method("Gradient boosting"),
                         dev.y_main.values, dev.kepid.values)
        joblib.dump(fit(est, dev[C.CLEAN], dev.y_main.values, dev.kepid.values), cal_p)
        joblib.dump(tuned("Gradient boosting").fit(dev[C.CLEAN], dev.y_main.values), raw_p)
    model, raw = joblib.load(cal_p), joblib.load(raw_p)
    ops = json.loads((C.TABLES / "operating_points.json").read_text())
    test = pd.read_csv(C.PROC / "test_predictions.csv")
    gallery = pd.read_csv(C.TABLES / "e7_gallery.csv")
    return ops, model, shap.TreeExplainer(raw.named_steps["clf"]), test, gallery


ops, model, explainer, test, gallery = load()
thr95 = ops["Gradient boosting"]["thr_recall95"]

st.title("🪐 Planet or Impostor?")
st.caption("Triage of Kepler Objects of Interest from transit and stellar measurements only. "
           "No vetting flags, scores or dispositions are used as inputs.")

tab1, tab2, tab3 = st.tabs(["Score a candidate", "Where the model fails", "Stage 2: AI vs Robovetter"])

with tab1:
    left, right = st.columns([1, 1.3])
    with left:
        pick = st.selectbox("Start from a held-out test KOI, or edit the numbers below",
                            ["(median values)"] + test.kepoi_name.tolist())
        if pick == "(median values)":
            row, truth = test[C.CLEAN].median(), None
        else:
            r = test[test.kepoi_name == pick].iloc[0]
            row, truth = r[C.CLEAN].astype(float), r.koi_disposition
        x = row.copy()
        for col, label, unit in MAIN:
            v = float(row[col]) if pd.notna(row[col]) else float(test[col].median())
            x[col] = st.number_input(label + (f" ({unit})" if unit else ""), value=v, format="%.4g", key=f"{pick}-{col}")
    X = pd.DataFrame([x.values], columns=C.CLEAN)
    p = float(model.predict_proba(X)[0, 1])
    with right:
        a, b = st.columns(2)
        a.metric("Chance it is planet-like", f"{p:.0%}")
        b.metric("Verdict at 0.5", "planet-like" if p >= 0.5 else "likely impostor")
        if p >= thr95:
            st.success(f"Keep for review (above the high-recall threshold {thr95:.2f}, which keeps 95 % of real planets).")
        else:
            st.warning(f"Safe to deprioritise (below the high-recall threshold {thr95:.2f}).")
        if truth is not None:
            st.info(f"Catalog disposition (never shown to the model): **{truth}**")
        sv = np.asarray(explainer.shap_values(X))[0]
        contrib = (pd.Series(sv, index=C.CLEAN)
                   .groupby(lambda c: c.replace("_err1", "").replace("_err2", "").replace("koi_", "")).sum())
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)
        st.markdown("**What pushed this prediction** (right = towards planet, left = towards impostor; "
                    "SHAP, log-odds, each measurement with its error bars)")
        st.bar_chart(pd.DataFrame({"measurement": top.index, "push": top.values}), x="measurement", y="push",
                     horizontal=True, sort=False, height=300)

with tab2:
    st.markdown("Confident mistakes on held-out test stars (E7). The last columns are the vetting flags "
                "and comment the model never saw; they show why the catalog disagrees.")
    kind = st.radio("Show", ["confident false alarm", "confident miss"], horizontal=True)
    st.dataframe(gallery[gallery.kind == kind].drop(columns="kind").round(3), width="stretch", hide_index=True)

with tab3:
    cmp_p, pp_p = C.TABLES / "s2_compare.csv", C.TABLES / "s2_possible_planets_top.csv"
    if not cmp_p.exists():
        st.info("Stage 2 has not been run yet. Run `./run_stage2.sh` (about 2 hours), then reload.")
    else:
        cmp = pd.read_csv(cmp_p)
        st.markdown("Judged on **6,557 objects experts checked by hand** (confirmed planets vs certified "
                    "false positives). Fewer mistakes is better; the interval is a 95 % paired bootstrap over stars.")
        show = cmp[["method", "mistakes", "false_alarms", "misses", "change_vs_robovetter", "ci_low", "ci_high"]]
        st.dataframe(show.round(0).rename(columns={"false_alarms": "false alarms", "misses": "missed planets",
                                          "change_vs_robovetter": "vs Robovetter", "ci_low": "interval low",
                                          "ci_high": "interval high"}), width="stretch", hide_index=True)
        st.bar_chart(cmp.set_index("method")["mistakes"], horizontal=True, sort=False, height=280)
        if pp_p.exists():
            st.markdown("**Second opinions:** objects experts marked *possible planet* that the Robovetter rejected, "
                        "ranked by the AI's probability (worth a follow-up look).")
            pp = pd.read_csv(pp_p)[["kepoi_name", "p_ai", "koi_score", "koi_period", "koi_prad", "koi_model_snr"]]
            st.dataframe(pp.rename(columns={"p_ai": "AI probability", "koi_score": "Robovetter score"}).round(3).head(10),
                         width="stretch", hide_index=True)
