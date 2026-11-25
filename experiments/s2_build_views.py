"""Stage 2, step 1: build folded light-curve views for every DR25 KOI.

Owner: Member 4. Resumable: one small .npz per star in data/processed/lc_views/; finished stars
are skipped on re-run. Trusted-label (gold) stars are fetched first. Raw light curves are never
written to disk (about 45 GB in total would not fit).
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from src import config as C
from src.lightcurves import fetch_star, koi_views

OUT = C.PROC / "lc_views"
OUT.mkdir(parents=True, exist_ok=True)
VIEWS = ["global", "local", "odd", "even", "secondary", "centroid"]


def do_star(kepid, kois, fetch_pool):
    quarters = fetch_star(kepid, fetch_pool)
    rows = {v: [] for v in VIEWS}
    meta = []
    for _, k in kois.iterrows():
        v = koi_views(quarters, k.koi_period, k.koi_time0bk, k.koi_duration) if quarters else None
        if v is None:
            continue
        for name in VIEWS:
            rows[name].append(v[name])
        meta.append((k.kepoi_name, v["n_quarters"], v["n_transits"], v["depth_ppm_measured"]))
    np.savez_compressed(OUT / f"{kepid}.npz", names=np.array([m[0] for m in meta]),
                        n_quarters=np.array([m[1] for m in meta]), n_transits=np.array([m[2] for m in meta]),
                        depth_measured=np.array([m[3] for m in meta]),
                        **{v: np.array(rows[v], dtype=np.float32).reshape(len(meta), -1) for v in VIEWS})
    return kepid, len(quarters), len(meta), len(kois)


def main(limit=None):
    d = pd.read_csv(C.RAW_FILES["dr25"], low_memory=False)
    f = pd.read_csv(C.RAW / "fpwg_certified_fp.csv", comment="#")[["kepoi_name", "fpwg_disp_status"]]
    d = d.merge(f, on="kepoi_name", how="left")
    gold = (d.koi_disposition == "CONFIRMED") | d.fpwg_disp_status.isin(["CERTIFIED FP", "CERTIFIED FA"])
    order = pd.concat([d[gold].kepid, d[~gold].kepid]).drop_duplicates().tolist()
    todo = [k for k in order if not (OUT / f"{k}.npz").exists()][:limit]
    print(f"stars: {len(order)} total, {len(todo)} to do", flush=True)
    groups = {k: g for k, g in d.groupby("kepid")}
    t0, done, fails = time.time(), 0, []
    fetch_pool = ThreadPoolExecutor(96)
    with ThreadPoolExecutor(8) as star_pool:
        futs = {star_pool.submit(do_star, k, groups[k], fetch_pool): k for k in todo}
        for fu in as_completed(futs):
            try:
                _, nq, nk, nall = fu.result()
                if nk < nall:
                    fails.append(f"{futs[fu]} partial {nk}/{nall} quarters={nq}")
            except Exception as e:
                fails.append(f"{futs[fu]} error {e}")
            done += 1
            if done % 100 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"{done}/{len(todo)} stars, {el/60:.1f} min, eta {el/done*(len(todo)-done)/60:.0f} min, "
                      f"issues {len(fails)}", flush=True)
    (C.PROC / "lc_views_issues.txt").write_text("\n".join(fails))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
