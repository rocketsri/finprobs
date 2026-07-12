"""
Brandi & Di Matteo (2026)-style shuffle-surrogate test, applied directly to our
own real single-stock and S&P 500 data (something no prior work -- including
our own earlier AIC-based covariance test -- has done for real market series).

Method: compute the multifractal spectrum zeta(q) on the raw log-vol series and
on a randomly shuffled surrogate (destroys temporal order, preserves the exact
empirical marginal distribution). Curvature (deviation from linear zeta(q)=qH)
that SURVIVES in the raw series but VANISHES in the shuffle is temporal;
curvature present in the shuffle too is distributional (marginal-driven).
"""
import json, glob, os
import numpy as np

DATA_DIR = "/home/user/finprobs/research/scripts/data"
rng = np.random.default_rng(2026)

def load_series(path):
    d = json.load(open(path))
    res = d["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    high = np.array(q["high"], dtype=float)
    low = np.array(q["low"], dtype=float)
    mask = ~(np.isnan(high) | np.isnan(low))
    return high[mask], low[mask]

def parkinson_logvol(high, low):
    hl = np.log(high/low)
    hl = np.clip(hl, 1e-8, None)
    var_t = (hl**2)/(4*np.log(2))
    return np.log(var_t)

def zeta_q(logvol, lags, qs):
    """Return zeta(q) estimated by log-log slope of E|diff|^q vs lag, for each q."""
    zetas = []
    for q in qs:
        vals, used = [], []
        for tau in lags:
            d = logvol[tau:] - logvol[:-tau]
            m = np.mean(np.abs(d)**q)
            if m > 0:
                vals.append(m); used.append(tau)
        used = np.array(used); vals = np.array(vals)
        slope, _ = np.polyfit(np.log(used), np.log(vals), 1)
        zetas.append(slope)
    return np.array(zetas)

def curvature(qs, zetas):
    """Quadratic fit zeta(q) = a*q + b*q^2; curvature magnitude = |b|."""
    A = np.vstack([qs, qs**2]).T
    coef, *_ = np.linalg.lstsq(A, zetas, rcond=None)
    return coef[1]  # quadratic coefficient (curvature)

lags = np.unique(np.round(np.geomspace(2, 250, 16)).astype(int))
qs = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])

n_shuffles = 40
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))

print(f"{'name':8s} {'curv_raw':>10s} {'curv_shuf':>10s} {'frac_distr':>11s} {'frac_temporal':>13s}")
rows = []
for f in files:
    name = os.path.basename(f).replace(".json", "")
    high, low = load_series(f)
    logvol = parkinson_logvol(high, low)

    curv_raw = curvature(qs, zeta_q(logvol, lags, qs))

    curv_shufs = []
    for _ in range(n_shuffles):
        shuf = rng.permutation(logvol)
        curv_shufs.append(curvature(qs, zeta_q(shuf, lags, qs)))
    curv_shuf_mean = np.mean(curv_shufs)
    curv_shuf_std = np.std(curv_shufs)

    # fraction of raw curvature explained by shuffle (distributional) vs excess (temporal)
    denom = abs(curv_raw) if abs(curv_raw) > 1e-12 else 1e-12
    frac_distr = np.clip(abs(curv_shuf_mean) / denom, 0, 1.5)
    frac_temporal = 1 - frac_distr

    rows.append((name, curv_raw, curv_shuf_mean, curv_shuf_std, frac_distr, frac_temporal))
    print(f"{name:8s} {curv_raw:10.5f} {curv_shuf_mean:10.5f} {frac_distr*100:10.1f}% {frac_temporal*100:12.1f}%")

names = [r[0] for r in rows]
idx_i = names.index("GSPC")
single_rows = [r for r in rows if r[0] != "GSPC"]

print(f"\nIndex (GSPC): curvature={rows[idx_i][1]:.5f}, distributional fraction={rows[idx_i][4]*100:.1f}%, temporal fraction={rows[idx_i][5]*100:.1f}%")
print(f"Single stocks: mean distributional fraction={np.mean([r[4] for r in single_rows])*100:.1f}%, "
      f"mean temporal fraction={np.mean([r[5] for r in single_rows])*100:.1f}%")
print(f"Single-stock range of temporal fraction: {min(r[5] for r in single_rows)*100:.1f}%--{max(r[5] for r in single_rows)*100:.1f}%")
