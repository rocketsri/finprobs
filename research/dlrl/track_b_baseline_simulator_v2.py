"""
Track B, Step 1 (v2): Minimal multi-agent market simulator -- HAND-SPECIFIED
baseline that structurally reproduces the "plain linear superposition of
independent Hawkes scaling limits" case, which prior analytic work proved
CANNOT generate genuine multifractal / cascade volatility structure (it stays
jointly Gaussian in the scaling limit -> long-range dependence but exactly
linear structure-function scaling zeta(q)=qH, no genuine concave curvature).

This is the NULL / trust-baseline for the project: before anyone bolts an
RL-trained agent variant on top, we confirm the plain rule-based simulator
behaves as theory predicts under the shuffle-surrogate genuineness diagnostic.

DESIGN (my independent choices; a parallel attempt exists at
track_b_baseline_simulator.py -- not read, not copied):

  Two independent timescale populations, each a DISCRETE-TIME self-exciting
  (Hawkes-type) order-flow generator:

    lambda[t] = mu + kappa*(lambda[t-1] - mu) + alpha*N[t-1]
    N[t] ~ Poisson(lambda[t])                       (event count in fine bin t)

  Stationary mean m = mu*(1-kappa)/(1-kappa-alpha), stable iff kappa+alpha<1.
  This recursion is the standard discrete analogue of an exponential-kernel
  Hawkes process: N[t-1] events each raise the next intensity by alpha, and the
  residual excitation decays geometrically at rate kappa.

    - FAST population: strong, tightly-coupled self-excitation, short memory
      (large alpha, small kappa) -> bursty microstructural clustering.
    - SLOW population: diffuse, high baseline, weak per-event excitation but
      long memory (large kappa) -> slow "macro information flow" background.

  Each event is a trade of unit size with an INDEPENDENT random sign (+/-1).
  Net signed volume per fine bin -> LINEAR (Kyle-type) price impact:

    r_fine[t] = IMPACT * ( net_signed_fast[t] + net_signed_slow[t] )
    log_price = cumsum(r_fine)

  The two populations' order flows are COMBINED PURELY ADDITIVELY, and impact is
  LINEAR. That is precisely the "linear superposition" structure: each
  population's order flow is (in its scaling limit) a Gaussian long-memory
  process, the price is their linear sum, hence jointly Gaussian -> the analytic
  negative case. No cross-population coupling, no multiplicative/switching/
  herding feedback (the only known routes to genuine multifractality) is
  present by construction.

LOG-VOLATILITY PROXY: fine returns are aggregated into non-overlapping bars of
W fine bins; per bar we form realized variance RV = sum(r_fine^2), and the
log-vol proxy is logvol = log(RV). (Analogue of the real-data Parkinson
log-variance in shuffle_surrogate_test.py, which also takes log of a per-bar
variance estimate.) RV is dominated by the total event count in the bar, i.e.
by the SUM of the two independent Hawkes intensities -> a linear superposition
of two long-memory processes.

The zeta_q() and curvature() functions below are COPIED VERBATIM from
research/scripts/shuffle_surrogate_test.py (that module has no __main__ guard
and runs data-loading code at import, so we copy rather than import). The
genuineness diagnostic is therefore byte-identical to the validated one.
"""

import json
import numpy as np

RNG = np.random.default_rng(20260718)


# ---------------------------------------------------------------------------
# Diagnostic functions -- VERBATIM from research/scripts/shuffle_surrogate_test.py
# ---------------------------------------------------------------------------
def zeta_q(logvol, lags, qs):
    """Return zeta(q) estimated by log-log slope of E|diff|^q vs lag, for each q."""
    zetas = []
    for q in qs:
        vals, used = [], []
        for tau in lags:
            d = logvol[tau:] - logvol[:-tau]
            m = np.mean(np.abs(d) ** q)
            if m > 0:
                vals.append(m)
                used.append(tau)
        used = np.array(used)
        vals = np.array(vals)
        slope, _ = np.polyfit(np.log(used), np.log(vals), 1)
        zetas.append(slope)
    return np.array(zetas)


def curvature(qs, zetas):
    """Quadratic fit zeta(q) = a*q + b*q^2; curvature magnitude = |b|."""
    A = np.vstack([qs, qs ** 2]).T
    coef, *_ = np.linalg.lstsq(A, zetas, rcond=None)
    return coef[1]  # quadratic coefficient (curvature)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
def simulate_population(n_fine, mu, kappa, alpha, rng):
    """Discrete-time self-exciting (Hawkes-type) intensity + signed order flow.

    Returns (counts, net_signed) arrays of length n_fine.
    """
    lam = np.empty(n_fine)
    counts = np.empty(n_fine, dtype=np.int64)
    m = mu * (1.0 - kappa) / (1.0 - kappa - alpha)  # stationary mean intensity
    prev_lam = m
    prev_n = m
    for t in range(n_fine):
        cur_lam = mu + kappa * (prev_lam - mu) + alpha * prev_n
        cur_lam = max(cur_lam, 1e-9)
        lam[t] = cur_lam
        n = rng.poisson(cur_lam)
        counts[t] = n
        prev_lam = cur_lam
        prev_n = n
    # net signed volume: each of n events gets an independent +/-1 sign.
    # #buys ~ Binomial(n, 0.5); net = 2*buys - n. Vectorized.
    buys = rng.binomial(counts, 0.5)
    net_signed = (2 * buys - counts).astype(float)
    return counts, net_signed


def run_simulation(n_obs=20000, W=15, impact=0.01, rng=None):
    """Build the two-population additive-impact price process and log-vol proxy."""
    if rng is None:
        rng = RNG
    n_fine = n_obs * W

    # FAST: strong self-excitation, short memory. branching ~ alpha/(1-kappa)=0.875
    cf, nf = simulate_population(n_fine, mu=0.30, kappa=0.20, alpha=0.70, rng=rng)
    # SLOW: diffuse high baseline, weak excitation, long memory. branching=0.80
    cs, ns = simulate_population(n_fine, mu=0.80, kappa=0.90, alpha=0.08, rng=rng)

    # LINEAR additive price impact -> log return per fine bin
    r_fine = impact * (nf + ns)

    # aggregate into non-overlapping bars; realized-variance log-vol proxy
    r_bars = r_fine[: n_obs * W].reshape(n_obs, W)
    rv = np.sum(r_bars ** 2, axis=1)
    rv = np.clip(rv, 1e-12, None)
    logvol = np.log(rv)

    log_price = np.cumsum(r_fine)
    diag = dict(
        mean_lam_fast=cf.mean(), mean_lam_slow=cs.mean(),
        frac_fast_var=float(np.var(nf) / (np.var(nf) + np.var(ns))),
    )
    return logvol, log_price, diag


# ---------------------------------------------------------------------------
# Shuffle-surrogate genuineness test (same protocol as the validated script)
# ---------------------------------------------------------------------------
def surrogate_test(logvol, n_shuffles=40, rng=None):
    if rng is None:
        rng = RNG
    lags = np.unique(np.round(np.geomspace(2, 250, 16)).astype(int))
    qs = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])

    zr = zeta_q(logvol, lags, qs)
    curv_raw = curvature(qs, zr)

    curv_shufs = []
    for _ in range(n_shuffles):
        shuf = rng.permutation(logvol)
        curv_shufs.append(curvature(qs, zeta_q(shuf, lags, qs)))
    curv_shufs = np.array(curv_shufs)
    curv_shuf_mean = float(curv_shufs.mean())
    curv_shuf_std = float(curv_shufs.std())

    denom = abs(curv_raw) if abs(curv_raw) > 1e-12 else 1e-12
    frac_distr = float(np.clip(abs(curv_shuf_mean) / denom, 0, 1.5))
    frac_temporal = 1.0 - frac_distr
    z_excess = float((curv_raw - curv_shuf_mean) / (curv_shuf_std + 1e-12))

    # H from q=2 structure function slope of the price (sanity on the driver)
    return dict(
        qs=qs.tolist(), lags=lags.tolist(),
        zeta_raw=zr.tolist(),
        curv_raw=float(curv_raw),
        curv_shuf_mean=curv_shuf_mean,
        curv_shuf_std=curv_shuf_std,
        frac_distr=frac_distr,
        frac_temporal=frac_temporal,
        z_excess=z_excess,
        n_shuffles=n_shuffles,
    )


if __name__ == "__main__":
    N_OBS, W = 20000, 15
    logvol, log_price, diag = run_simulation(n_obs=N_OBS, W=W)

    print("=== Track B baseline simulator v2 (linear Hawkes superposition) ===")
    print(f"n_obs (log-vol length) = {len(logvol)}, fine bins per bar W = {W}, "
          f"total fine bins = {N_OBS*W}")
    print(f"mean fast intensity = {diag['mean_lam_fast']:.3f}, "
          f"mean slow intensity = {diag['mean_lam_slow']:.3f}")
    print(f"fast share of signed-flow variance = {diag['frac_fast_var']*100:.1f}%")
    print(f"log-vol: mean={logvol.mean():.3f} std={logvol.std():.3f} "
          f"skew~{((logvol-logvol.mean())**3).mean()/logvol.std()**3:.3f} "
          f"exkurt~{((logvol-logvol.mean())**4).mean()/logvol.std()**4-3:.3f}")

    res = surrogate_test(logvol, n_shuffles=40)
    print("\n--- zeta(q) on raw log-vol ---")
    for q, z in zip(res["qs"], res["zeta_raw"]):
        print(f"  q={q:>4}: zeta={z:+.4f}   (linear ref qH gives zeta/q const)")
    print("\n--- shuffle-surrogate genuineness test ---")
    print(f"  curvature raw          = {res['curv_raw']:+.5f}")
    print(f"  curvature shuffle mean = {res['curv_shuf_mean']:+.5f} "
          f"(std {res['curv_shuf_std']:.5f}, n={res['n_shuffles']})")
    print(f"  distributional fraction = {res['frac_distr']*100:.1f}%")
    print(f"  temporal fraction       = {res['frac_temporal']*100:.1f}%")
    print(f"  z(excess temporal curv) = {res['z_excess']:+.2f}")

    out = {"config": {"n_obs": N_OBS, "W": W, "impact": 0.01,
                       "fast": {"mu": 0.30, "kappa": 0.20, "alpha": 0.70},
                       "slow": {"mu": 0.80, "kappa": 0.90, "alpha": 0.08}},
           "sim_diag": {k: float(v) for k, v in diag.items()},
           "logvol_moments": {"mean": float(logvol.mean()),
                              "std": float(logvol.std())},
           "surrogate_result": res}
    with open("/home/user/finprobs/research/dlrl/track_b_baseline_v2_results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nSaved -> research/dlrl/track_b_baseline_v2_results.json")
