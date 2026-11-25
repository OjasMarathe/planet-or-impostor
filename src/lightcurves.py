"""Stage 2: Kepler light curves -> small folded 'views' per KOI, without touching the disk.

Owner: Member 4 (light curves and release data).
Each star's 17 long-cadence quarter files are fetched into memory from NASA's public
AWS mirror of MAST, detrended, folded on each KOI's period and epoch (from the DR25 KOI
table), binned, and only the binned views are kept (a few KB per KOI).

Views per KOI (all fixed length):
  global    201 bins over the whole orbit            (shape of the full phase curve)
  local      81 bins over +-2.5 transit durations    (shape of the transit)
  odd, even  81 bins, odd- and even-numbered transits only (eclipsing binaries alternate)
  secondary  81 bins centred half an orbit later     (a secondary eclipse means two stars)
  centroid   81 bins of the star's image-centre motion, in sigma units (a shift during the
             dip means the dip comes from a neighbouring star)
Flux views are scaled so the primary transit bottoms out at -1; odd, even and secondary
use the SAME scale, so their depths stay comparable to the primary.
"""
import io
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from astropy.io import fits

MIRROR = "https://stpubdata.s3.amazonaws.com/kepler/public/lightcurves"
# Long-cadence file timestamps for Q1..Q17 (Q4 has two variants across targets).
QUARTERS = [["2009166043257"], ["2009259160929"], ["2009350155506"], ["2010078095331", "2010009091648"],
            ["2010174085026"], ["2010265121752"], ["2010355172524"], ["2011073133259"], ["2011177032512"],
            ["2011271113734"], ["2012004120508"], ["2012088054726"], ["2012179063303"], ["2012277125453"],
            ["2013011073258"], ["2013098041711"], ["2013131215648"]]
N_GLOBAL, N_LOCAL, LOCAL_HALF = 201, 81, 2.5


def _get(url, tries=8):
    """Fetch with growing waits between retries (5 s, 10 s, ... up to ~3 min in total), so a short
    network drop does not fail the star. A failed star writes no file and is retried on the next run."""
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except Exception:
            pass
        time.sleep(5 * (i + 1))
    raise IOError(f"failed after {tries} tries: {url}")


def _quarter(kepid, stamps):
    k = f"{int(kepid):09d}"
    for s in stamps:
        b = _get(f"{MIRROR}/{k[:4]}/{k}/kplr{k}-{s}_llc.fits")
        if b is not None:
            d = fits.open(io.BytesIO(b), memmap=False)[1].data
            ok = (d["SAP_QUALITY"] == 0) & np.isfinite(d["TIME"]) & np.isfinite(d["PDCSAP_FLUX"])
            t, f = d["TIME"][ok].astype(float), d["PDCSAP_FLUX"][ok].astype(float)
            c1, c2 = d["MOM_CENTR1"][ok].astype(float), d["MOM_CENTR2"][ok].astype(float)
            if len(t) < 100:
                return None
            return t, f / np.nanmedian(f), c1, c2
    return None


def fetch_star(kepid, pool):
    """Return the list of usable quarters for one star: (time, norm_flux, centr_col, centr_row)."""
    return [q for q in pool.map(lambda s: _quarter(kepid, s), QUARTERS) if q is not None]


def _phase(t, period, t0):
    return (t - t0 + 0.5 * period) % period - 0.5 * period


def _trend(t, y, oot, chunk):
    """Out-of-transit binned-median trend, interpolated to every cadence."""
    edges = np.arange(t.min(), t.max() + chunk, chunk)
    idx = np.digitize(t, edges)
    xs, ys = [], []
    for b in np.unique(idx):
        m = (idx == b) & oot & np.isfinite(y)
        if m.sum() >= 5:
            xs.append(np.median(t[m]))
            ys.append(np.median(y[m]))
    if len(xs) < 2:
        return np.full_like(y, np.nanmedian(y[oot]) if oot.any() else np.nanmedian(y))
    return np.interp(t, xs, ys)


def _bin(x, y, lo, hi, n):
    edges = np.linspace(lo, hi, n + 1)
    idx = np.digitize(x, edges) - 1
    ok = (idx >= 0) & (idx < n) & np.isfinite(y)
    cnt = np.bincount(idx[ok], minlength=n).astype(float)
    tot = np.bincount(idx[ok], weights=y[ok], minlength=n)
    out = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    good = np.isfinite(out)
    if good.sum() == 0:
        return np.zeros(n), cnt
    if (~good).any():
        out[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), out[good])
    return out, cnt


def koi_views(quarters, period, t0, dur_hours):
    """Detrend and fold one KOI. Returns a dict of views plus coverage counts, or None."""
    dur = max(dur_hours / 24.0, 0.02)
    chunk = max(0.5, 3 * dur)
    T, F, R = [], [], []
    for t, f, c1, c2 in quarters:
        oot = np.abs(_phase(t, period, t0)) > 1.5 * dur
        T.append(t)
        F.append(f / _trend(t, f, oot, chunk))
        d1 = c1 - _trend(t, c1, oot, chunk)
        d2 = c2 - _trend(t, c2, oot, chunk)
        R.append(np.hypot(d1, d2))
    t, f, r = np.concatenate(T), np.concatenate(F), np.concatenate(R)
    # drop upward outliers only (flares, cosmic rays); dips are the signal
    mad = 1.4826 * np.nanmedian(np.abs(f - np.nanmedian(f)))
    keep = f < 1 + 5 * max(mad, 1e-6)
    t, f, r = t[keep], f[keep] - 1.0, r[keep]
    ph = _phase(t, period, t0)
    ntr = np.round((t - t0) / period).astype(int)
    half = LOCAL_HALF * dur
    if (np.abs(ph) < 0.5 * dur).sum() < 3:
        return None  # transit not covered by the data we have
    loc, cnt_loc = _bin(ph, f, -half, half, N_LOCAL)
    glob, _ = _bin(ph, f, -0.5 * period, 0.5 * period, N_GLOBAL)
    odd, _ = _bin(ph[ntr % 2 == 1], f[ntr % 2 == 1], -half, half, N_LOCAL)
    even, _ = _bin(ph[ntr % 2 == 0], f[ntr % 2 == 0], -half, half, N_LOCAL)
    ph2 = _phase(t, period, t0 + 0.5 * period)
    sec, _ = _bin(ph2, f, -half, half, N_LOCAL)
    oot = np.abs(ph) > 1.5 * dur
    sig = 1.4826 * np.nanmedian(np.abs(f[oot] - np.nanmedian(f[oot]))) if oot.sum() > 10 else np.nanstd(f)
    scale = max(-loc.min(), 3 * sig / np.sqrt(max(cnt_loc.mean(), 1)), 1e-6)
    # Centroid shift divided by fractional depth ~ distance (pixels) of the dip's source from the
    # light centre. Real planets on the target give small values; a neighbour's eclipse gives large ones.
    r0 = np.nanmean(r[oot])
    cen, _ = _bin(ph, r - r0, -half, half, N_LOCAL)
    clip = lambda v, a: np.clip(v, -a, a).astype(np.float32)
    return {
        "global": clip(glob / scale, 5), "local": clip(loc / scale, 5), "odd": clip(odd / scale, 5),
        "even": clip(even / scale, 5), "secondary": clip(sec / scale, 5), "centroid": clip(cen / scale, 10),
        "n_quarters": len(quarters), "n_transits": int(len(np.unique(ntr[np.abs(ph) < 0.5 * dur]))),
        "depth_ppm_measured": float(scale * 1e6),
    }
