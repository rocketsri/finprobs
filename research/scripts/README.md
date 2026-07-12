# Verification scripts

Supporting code for the "Empirical verification addendum" in
`../log-sfbm-unification-2026-07-12.md`. Requires `numpy`, `scipy`, `pandas`.

Raw data: `data/*.json` is a snapshot of Yahoo Finance daily OHLC (15y,
fetched 2026-07-12) for the S&P 500 index (`GSPC.json`) and 25 large-cap
single names, used as-is (no re-fetch needed to reproduce).

Run order:

1. `python3 estimate_H_real_data.py` — builds a Parkinson range-based daily
   log-volatility proxy for each series, estimates a structure-function-slope
   Hurst exponent, and writes `H_estimates.csv` + `results.pkl` (the latter
   is regenerated locally, not committed).
2. `python3 apply_diagnostic_real_data.py` — reads `results.pkl` and applies
   the nested power-law-vs-log-covariance AIC model-selection test to each
   series.
3. `python3 test_lambda_cancellation.py` — self-contained closed-form check
   of the λ² cancellation / T-lever claim (no data dependency).
4. `python3 mc_diagnostic_validation.py` — self-contained Monte Carlo power
   check of the same nested test against simulated ground truth (no data
   dependency; ~30s runtime).
5. `python3 aggregation_simulation.py` — self-contained simulation of the
   Zarhali et al. cross-sectional aggregation mechanism (no data dependency).

Caveats these scripts do not resolve (documented in the report): the
Parkinson estimator is a daily range-based proxy, not true high-frequency
realized variance; sample length (~3,772 trading days) is below the
10⁴–10⁵ observations the report flags as needed for stable estimation near
the H=0 boundary; the AIC-based model comparison is a simplified stand-in
for the fully proper boundary-corrected (Self–Liang) likelihood-ratio test.
