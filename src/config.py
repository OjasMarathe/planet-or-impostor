"""Project-wide constants: paths, seeds, labels, the frozen exclusion list and feature groups.

Owner: Member 1 (data and labels). Everything else imports from here, so the
exclusion list is defined exactly once. It was frozen before any model was fit.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"
MODELS = ROOT / "results" / "models"
for _p in (PROC, TABLES, FIGS, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_FRACTION_FOLDS = 5      # 1 of 5 stratified-grouped folds = 20 % held-out test set
CV_FOLDS = 5
CV_REPEATS = 3
N_BOOT = 2000

RAW_FILES = {
    "cumulative": RAW / "cumulative.csv",
    "dr24": RAW / "q1_q17_dr24_koi.csv",
    "dr25": RAW / "q1_q17_dr25_koi.csv",
}

# ---------------------------------------------------------------- labels
PLANET_LIKE = {"CANDIDATE", "CONFIRMED"}
FALSE_POS = {"FALSE POSITIVE"}
DROP_LABELS = {"NOT DISPOSITIONED"}

# ---------------------------------------------------------------- exclusion list (frozen)
# Columns that must never be features. Each entry carries the reason.
EXCLUDED = {
    # the label and vetting outputs (leakage)
    "koi_disposition": "the label itself",
    "koi_pdisposition": "the pipeline's own disposition",
    "koi_score": "confidence in the disposition itself",
    "koi_fpflag_nt": "vetting flag: not transit-like",
    "koi_fpflag_ss": "vetting flag: stellar eclipse",
    "koi_fpflag_co": "vetting flag: centroid offset",
    "koi_fpflag_ec": "vetting flag: ephemeris match",
    "koi_comment": "free-text vetting notes (names the failed tests)",
    "koi_vet_stat": "vetting status",
    "koi_vet_date": "vetting date",
    "kepler_name": "filled only for confirmed planets",
    # identifiers
    "kepid": "star id; used only to group the splits",
    "kepoi_name": "object id",
    "koi_tce_plnt_num": "TCE sequence number on the star (an id)",
    "ra": "sky position, not a property of the signal",
    "dec": "sky position, not a property of the signal",
    "ra_str": "sky position string",
    "dec_str": "sky position string",
    # provenance (used only for the release-shift and provenance analyses)
    "koi_delivname": "delivery name (provenance)",
    "koi_tce_delivname": "TCE delivery name (provenance)",
    "koi_disp_prov": "disposition provenance",
    "koi_parm_prov": "parameter provenance",
    "koi_sparprov": "stellar parameter provenance",
    "koi_fittype": "fit method; 'none' marks unfitted objects (provenance)",
    "koi_datalink_dvr": "link to the vetting report",
    "koi_datalink_dvs": "link to the vetting summary",
    "koi_quarters": "observing-quarter bit string (coverage, not signal)",
    "koi_limbdark_mod": "model name string",
    "koi_trans_mod": "model name string",
}

# ---------------------------------------------------------------- feature groups
def _with_errs(cols):
    out = []
    for c in cols:
        out += [c, f"{c}_err1", f"{c}_err2"]
    return out

# Transit-fit measurements taken straight from the light curve.
TRANSIT = _with_errs(["koi_period", "koi_duration", "koi_depth", "koi_impact"]) + ["koi_model_snr"]
# Planet quantities derived from the transit fit AND the star (mixes both).
DERIVED = _with_errs(["koi_prad", "koi_insol"]) + ["koi_teq"]
# Host-star properties.
STELLAR = _with_errs(["koi_steff", "koi_slogg", "koi_srad"]) + ["koi_kepmag"]

CLEAN = TRANSIT + DERIVED + STELLAR

# Grey zone for the leakage ladder (E1): measurements that are not vetting
# outputs, but are the diagnostic statistics the Robovetter itself reads
# (odd-even depth test, centroid offsets, flux-weighted motion, single-event
# statistics, fitted stellar density) plus system multiplicity.
DIAGNOSTIC = [
    "koi_bin_oedp_sig", "koi_fwm_stat_sig", "koi_dicco_msky", "koi_dikco_msky",
    "koi_max_sngle_ev", "koi_max_mult_ev", "koi_num_transits",
    "koi_ror", "koi_dor", "koi_srho", "koi_incl", "koi_count",
]
# (koi_model_chisq would belong here but is empty in every row of the cumulative table.)
FLAGS = ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]
SCORE = ["koi_score"]

# Leakage ladder for E1 (each tier adds to the one before it, except where noted)
LADDER = {
    "T0 clean": CLEAN,
    "T1 + diagnostics": CLEAN + DIAGNOSTIC,
    "T2 clean + flags": CLEAN + FLAGS,
    "T3 clean + score": CLEAN + SCORE,
    "T4 clean + flags + score": CLEAN + FLAGS + SCORE,
}

# Ablation groups for E5
ABLATION = {
    "transit only": TRANSIT,
    "stellar only": STELLAR,
    "transit + stellar": TRANSIT + STELLAR,
    "derived only": DERIVED,
    "all clean": CLEAN,
}

PRIMARY_METRIC = "pr_auc"
TARGET_RECALL = 0.95          # operating point fixed before the final test
REVIEW_BUDGETS = [100, 250, 500]
