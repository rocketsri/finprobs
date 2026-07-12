import json, glob, os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUT_DIR = SCRIPT_DIR

def load_series(path):
    d = json.load(open(path))
    res = d["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "date": pd.to_datetime(ts, unit="s"),
        "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
    }).dropna()
    df = df.sort_values("date").reset_index(drop=True)
    return df

def parkinson_logvol(df):
    # Parkinson (1980) range-based daily variance estimator: robust daily vol proxy from OHLC
    # sigma^2_t = (1/(4 ln2)) * (ln(High/Low))^2
    hl = np.log(df["high"].values / df["low"].values)
    hl = np.clip(hl, 1e-8, None)  # avoid log(0)/negative from bad ticks
    var_t = (hl ** 2) / (4 * np.log(2))
    logvol = np.log(var_t)  # this is our omega_t proxy (log-volatility)
    return df["date"].values, logvol

def structure_function_slope(logvol, lags, q=2):
    """Estimate Hurst-type exponent via slope of log E[|omega_{t+tau}-omega_t|^q] vs log(tau)."""
    vals = []
    used_lags = []
    for tau in lags:
        if tau >= len(logvol):
            continue
        diffs = logvol[tau:] - logvol[:-tau]
        m = np.mean(np.abs(diffs) ** q)
        if m > 0:
            vals.append(m)
            used_lags.append(tau)
    used_lags = np.array(used_lags)
    vals = np.array(vals)
    logtau = np.log(used_lags)
    logval = np.log(vals)
    slope, intercept = np.polyfit(logtau, logval, 1)
    H_est = slope / q  # zeta(q) = q*H  =>  H = slope/q
    return H_est, used_lags, vals, slope, intercept

def variogram(logvol, lags):
    """Var(omega_{t+tau}-omega_t) at each lag -- used to fit covariance functional form."""
    out = {}
    for tau in lags:
        if tau >= len(logvol):
            continue
        diffs = logvol[tau:] - logvol[:-tau]
        out[tau] = np.var(diffs)
    return out

results = {}
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
lags = np.unique(np.round(np.geomspace(1, 250, 18)).astype(int))

for f in files:
    name = os.path.basename(f).replace(".json", "")
    try:
        df = load_series(f)
        dates, logvol = parkinson_logvol(df)
        n = len(logvol)
        H_est, used_lags, vals, slope, intercept = structure_function_slope(logvol, lags, q=2)
        vgram = variogram(logvol, lags)
        results[name] = dict(n=n, H_est=H_est, slope=slope, intercept=intercept,
                              vgram=vgram, logvol=logvol, dates=dates)
        print(f"{name:8s}  n={n:5d}  H_est={H_est:+.4f}  slope={slope:+.4f}")
    except Exception as e:
        print(f"{name:8s}  FAILED: {e}")

# Save summary table
rows = []
for name, r in results.items():
    rows.append(dict(name=name, n=r["n"], H_est=r["H_est"], slope=r["slope"]))
summary = pd.DataFrame(rows).sort_values("name")
summary.to_csv(os.path.join(OUT_DIR, "H_estimates.csv"), index=False)
print("\n" + summary.to_string(index=False))

import pickle
with open(os.path.join(OUT_DIR, "results.pkl"), "wb") as fh:
    pickle.dump(results, fh)
