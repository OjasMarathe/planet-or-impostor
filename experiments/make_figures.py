"""All report figures, drawn from the CSV tables in results/tables.

Owner: Member 6 (report), with each member checking the figure for their experiment.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src import config as C

ACCENT, GREY, LIGHT = "#2563eb", "#6b7280", "#d1d5db"
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold"})
T = C.TABLES
SEL = json.loads((T / "operating_points.json").read_text())["_selected_model"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(C.FIGS / name, bbox_inches="tight")
    plt.close(fig)


def fig_leakage():
    d = pd.read_csv(T / "e1_leakage_ladder.csv")
    tiers = list(C.LADDER)
    models = list(dict.fromkeys(d.model))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        s = d[d.model == m].set_index("tier").loc[tiers]
        ax.errorbar(np.arange(len(tiers)) + (i - 1.5) * 0.08, s.pr_auc, yerr=s.pr_auc_sd, fmt="o", ms=5,
                    capsize=2, color=ACCENT if m == SEL else plt.cm.Greys(0.35 + 0.15 * i), label=m)
    ax.set_xticks(np.arange(len(tiers)), tiers, rotation=12)
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("PR-AUC (dev, 3x5 grouped CV)")
    ax.set_title("Adding vetting outputs makes the task look almost solved")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    save(fig, "e1_leakage_ladder.png")


def fig_models():
    d = pd.read_csv(T / "final_test_ci.csv")
    d = d[d.metric == "pr_auc"].sort_values("estimate")
    fig, ax = plt.subplots(figsize=(6, 3))
    y = np.arange(len(d))
    cols = [ACCENT if m == SEL else GREY for m in d.model]
    ax.errorbar(d.estimate, y, xerr=[d.estimate - d.ci_low, d.ci_high - d.estimate], fmt="none", ecolor=LIGHT, lw=3)
    ax.scatter(d.estimate, y, color=cols, zorder=3)
    for yi, (e, lo, hi) in enumerate(zip(d.estimate, d.ci_low, d.ci_high)):
        ax.text(hi + 0.004, yi, f"{e:.3f}", va="center", fontsize=8)
    ax.set_yticks(y, d.model)
    ax.set_xlabel("PR-AUC on the held-out test stars (95 % star-bootstrap interval)")
    ax.set_title("Test PR-AUC by model")
    save(fig, "e2_test_pr_auc.png")


def fig_pr_curves():
    t = pd.read_csv(C.PROC / "test_predictions.csv")
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    for name in ["Majority baseline", "Logistic regression", "k-NN", "Decision tree", "Random forest", "Gradient boosting"]:
        p, r, _ = precision_recall_curve(t.y_main, t["p_" + name])
        ax.plot(r, p, color=ACCENT if name == SEL else GREY, lw=2 if name == SEL else 1,
                alpha=1 if name == SEL else 0.6, label=name)
    ax.set_xlabel("Recall (planet-like)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall on test stars")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "e2_pr_curves.png")


def fig_reliability():
    d = pd.read_csv(T / "e2_reliability.csv")
    names = [m for m in dict.fromkeys(d.model) if m != "Majority baseline"]
    fig, axes = plt.subplots(1, len(names), figsize=(2.2 * len(names), 2.5), sharey=True)
    for ax, m in zip(axes, names):
        s = d[d.model == m]
        for cal, sub in s.groupby("calibration"):
            ax.plot(sub.mean_pred, sub.frac_pos, "o-", ms=3, color=GREY if cal == "none" else ACCENT,
                    label="uncalibrated" if cal == "none" else cal)
        ax.plot([0, 1], [0, 1], ls=":", color=LIGHT)
        ax.set_title(m, fontsize=8)
        ax.set_xlabel("predicted")
        ax.legend(frameon=False, fontsize=6)
    axes[0].set_ylabel("observed planet-like share")
    save(fig, "e2_reliability.png")


def fig_shift():
    d = pd.read_csv(T / "e4_release_shift.csv")
    order = ["in-release", "feature shift", "label shift", "full shift"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for ax, direction in zip(axes, ["DR24 -> DR25", "DR25 -> DR24"]):
        s = d[(d.direction == direction) & d.setting.isin(order)]
        models = list(dict.fromkeys(s.model))
        w = 0.8 / len(models)
        for i, m in enumerate(models):
            v = s[s.model == m].set_index("setting").loc[order].roc_auc
            ax.plot(np.arange(4), v, "o-", ms=5, color=ACCENT if m == SEL else plt.cm.Greys(0.35 + 0.2 * i), label=m)
        ax.set_xticks(np.arange(4), order, rotation=10)
        ax.set_title(direction)
    axes[0].set_ylabel("ROC-AUC (star-grouped folds)")
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, "e4_release_shift.png")


def fig_ablation():
    d = pd.read_csv(T / "e5_ablation.csv")
    order = list(C.ABLATION)
    models = list(dict.fromkeys(d.model))
    fig, ax = plt.subplots(figsize=(6.5, 3))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        s = d[d.model == m].set_index("features").loc[order]
        ax.errorbar(np.arange(len(order)) + (i - 1) * 0.1, s.pr_auc, yerr=s.pr_auc_sd, fmt="o", ms=5, capsize=2,
                    color=ACCENT if m == SEL else plt.cm.Greys(0.35 + 0.2 * i), label=m)
    ax.axhline(0.494, color=LIGHT, ls="--", lw=1)
    ax.text(len(order) - 0.6, 0.505, "chance (prevalence)", fontsize=7, color=GREY)
    ax.set_xticks(np.arange(len(order)), order)
    ax.set_ylabel("PR-AUC (dev CV)")
    ax.set_title("Transit measurements carry most of the signal")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "e5_ablation.png")


def fig_review():
    d = pd.read_csv(T / "e6_review_curve.csv")
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.fill_between(d.k, d["random_p2.5"], d["random_p97.5"], color=LIGHT, alpha=0.5, lw=0)
    ax.plot(d.k, d.perfect, color=GREY, ls=":", lw=1, label="perfect ranking")
    for c in d.columns:
        if c in ("k", "perfect", "random_p2.5", "random_p97.5"):
            continue
        model = c.startswith(SEL)
        ax.plot(d.k, d[c], lw=2 if model else 1, color=ACCENT if model else None, label=c,
                alpha=1 if model else 0.8)
    ax.set_xlabel("objects reviewed (k), best-ranked first")
    ax.set_ylabel("planet-like objects found")
    ax.set_title("Review-budget curve on test stars")
    ax.legend(frameon=False, fontsize=6.5)
    save(fig, "e6_review_budget.png")


def fig_errors():
    d = pd.read_csv(T / "e7_error_rates_by_group.csv")
    vars_ = ["koi_model_snr", "koi_period", "koi_depth", "koi_prad"]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), sharey=True)
    for ax, v in zip(axes, vars_):
        s = d[d.variable == v]
        x = np.arange(len(s))
        ax.bar(x - 0.2, s.conf_false_alarm_rate, 0.4, color=ACCENT, label="confident false alarm")
        ax.bar(x + 0.2, s.conf_miss_rate, 0.4, color=GREY, label="confident miss")
        ax.set_xticks(x, s.bin, rotation=35, fontsize=7)
        ax.set_title(v.replace("koi_", ""))
    axes[0].set_ylabel("share of objects in bin")
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, "e7_error_rates.png")


def fig_importance():
    d = pd.read_csv(T / "expl_permutation_importance.csv")
    top = d[d.model == SEL].sort_values("pr_auc_drop").tail(12)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.barh(top.measure.str.replace("koi_", ""), top.pr_auc_drop, xerr=top.sd, color=ACCENT)
    ax.set_xlabel("drop in test PR-AUC when shuffled")
    ax.set_title(f"Permutation importance ({SEL})")
    save(fig, "expl_permutation_importance.png")


def fig_pdp():
    d = pd.read_csv(T / "expl_partial_dependence.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax, feat, unit in zip(axes, ["koi_period", "koi_depth"], ["days", "ppm"]):
        for m, s in d[d.feature == feat].groupby("model"):
            ax.plot(s.value, s.avg_prob, color=ACCENT if m == SEL else GREY, lw=2 if m == SEL else 1,
                    ls="-" if m != "Logistic regression" else "--", label=m)
        ax.set_xscale("log")
        ax.set_xlabel(f"{feat.replace('koi_', '')} ({unit}, log scale)")
        ax.set_title(f"Partial dependence on {feat.replace('koi_', '')}")
    axes[0].set_ylabel("average predicted P(planet-like)")
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, "expl_partial_dependence.png")


def fig_shap():
    import shap
    sv = pd.read_csv(T / "expl_shap_values_sample.csv")
    X = pd.read_csv(T / "expl_shap_inputs_sample.csv")
    plt.figure()
    shap.summary_plot(sv.values, X, feature_names=[c.replace("koi_", "") for c in X.columns],
                      max_display=14, show=False, plot_size=(6.5, 4.5))
    plt.title("SHAP values, random forest (800 test objects)", fontsize=10)
    plt.savefig(C.FIGS / "expl_shap_summary.png", bbox_inches="tight", dpi=150)
    plt.close("all")


if __name__ == "__main__":
    for f in [fig_leakage, fig_models, fig_pr_curves, fig_reliability, fig_shift, fig_ablation, fig_review,
              fig_errors, fig_importance, fig_pdp, fig_shap]:
        try:
            f()
            print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
