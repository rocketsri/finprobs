"""
Apply the repaired Direction-1 diagnostic (nested power-law vs log-covariance
model selection via AIC) directly to the real single-stock and index log-vol
series already estimated from Yahoo Finance daily OHLC (Parkinson proxy).

This directly tests: for each name, does its estimated autocovariance of
log-volatility look more like a power-law/self-affine process (C(tau) ~ T^2H - tau^2H,
"apparent multiscaling could be an H>0 rough process, not necessarily a genuine
cascade") or more like a genuine logarithmic cascade (C(tau) ~ log(T/tau))?
"""
import os
import pickle
import numpy as np
from scipy.optimize import curve_fit

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(f"{OUT_DIR}/results.pkl", "rb") as fh:
    results = pickle.load(fh)

def empirical_acf(path, lags):
    path = path - path.mean()
    out = []
    for tau in lags:
        c = np.mean(path[:-tau] * path[tau:])
        out.append(c)
    return np.array(out)

def fit_power(lags, C_hat):
    def f(tau, A, B, Hh):
        Hh = np.clip(Hh, 1e-4, 0.49)
        return A - B * np.abs(tau) ** (2 * Hh)
    try:
        p0 = [C_hat[0], (C_hat[0] - C_hat[-1]) / (lags[-1] ** 0.2), 0.1]
        popt, _ = curve_fit(f, lags, C_hat, p0=p0, maxfev=20000,
                             bounds=([-np.inf, -np.inf, 1e-4], [np.inf, np.inf, 0.49]))
        resid = C_hat - f(lags, *popt)
        sse = np.sum(resid ** 2)
        return sse, 3, popt
    except Exception:
        return np.inf, 3, None

def fit_log(lags, C_hat):
    X = np.vstack([np.ones_like(lags, dtype=float), np.log(lags.astype(float))]).T
    coef, *_ = np.linalg.lstsq(X, C_hat, rcond=None)
    pred = X @ coef
    resid = C_hat - pred
    sse = np.sum(resid ** 2)
    return sse, 2, coef

def aic(sse, k, n):
    if sse <= 0:
        sse = 1e-15
    return n * np.log(sse / n) + 2 * k

lags_fit = np.unique(np.round(np.geomspace(2, 250, 22)).astype(int))
n_lags = len(lags_fit)

print(f"{'name':8s} {'H_fit':>8s} {'AIC_power':>12s} {'AIC_log':>12s} {'delta(P-L)':>12s} {'favored':>10s}")
rows = []
for name in sorted(results.keys()):
    logvol = results[name]["logvol"]
    C_hat = empirical_acf(logvol, lags_fit)
    sse_p, k_p, popt_p = fit_power(lags_fit, C_hat)
    sse_l, k_l, popt_l = fit_log(lags_fit, C_hat)
    aic_p = aic(sse_p, k_p, n_lags)
    aic_l = aic(sse_l, k_l, n_lags)
    favored = "power" if aic_p < aic_l else "log"
    H_fit = popt_p[2] if popt_p is not None else float('nan')
    rows.append((name, H_fit, aic_p, aic_l, aic_p - aic_l, favored))
    print(f"{name:8s} {H_fit:8.4f} {aic_p:12.3f} {aic_l:12.3f} {aic_p-aic_l:12.3f} {favored:>10s}")

n_power = sum(1 for r in rows if r[5] == "power")
n_log = sum(1 for r in rows if r[5] == "log")
print(f"\nModel favored across {len(rows)} series: power-law(self-affine)={n_power}, log(genuine cascade)={n_log}")

idx_row = [r for r in rows if r[0] == "GSPC"][0]
single_rows = [r for r in rows if r[0] != "GSPC"]
print(f"\nIndex (GSPC): H_fit={idx_row[1]:.4f}, favored={idx_row[5]}, delta AIC={idx_row[4]:.2f}")
print(f"Single stocks: mean H_fit={np.mean([r[1] for r in single_rows]):.4f}, "
      f"favored-power count={sum(1 for r in single_rows if r[5]=='power')}/{len(single_rows)}")
