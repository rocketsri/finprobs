"""
Track B, Step 1 -- Minimal multi-agent market simulator (HAND-SPECIFIED baseline).

PURPOSE
-------
Build the NULL/BASELINE simulator that structurally matches the "plain linear
superposition of Hawkes processes" case already PROVEN ANALYTICALLY to be
incapable of genuine multifractality (see research/findings-summary-2026-07-12.md,
Negative findings, first bullet):

    "Plain superposition of Hawkes processes cannot produce genuine
     multifractality. Proven to remain jointly Gaussian under time-invariant
     linear mixing, giving only linear zeta(q)=qH -- capable of exotic
     long-range dependence, not genuine cascade structure."

This script is the trustworthiness check: before anyone puts an RL-trained
(adaptive) agent on top of this engine in a follow-up, we confirm the
hand-specified (rule-based, non-adaptive) engine reproduces the theoretically
predicted NEGATIVE result -- predominantly DISTRIBUTIONAL (marginal-driven)
curvature in the shuffle-surrogate diagnostic, NOT genuine temporal cascade
curvature.

WHY THIS IS A FAITHFUL PROXY FOR THE "LINEAR HAWKES SUPERPOSITION" CASE
----------------------------------------------------------------------
The analytic negative result rests on three structural properties. This
simulator reproduces all three, on purpose:

  (a) LINEAR self-excitation. Each agent is a discrete-time linear Hawkes
      proxy: intensity lambda_t = mu + alpha * M_t, where M_t is an
      exponentially-decaying sum of past event counts. This is the
      standard Kirchner/Hawkes-INAR discrete-time approximation of a
      continuous-time linear Hawkes process (exponential kernel). It is
      LINEAR in past activity -- no quadratic/Zumbach feedback (which would
      land in the wrong Heston class) and STRICTLY SUBCRITICAL branching
      ratio n = alpha/(1-phi) < 1 (no exact criticality). Those are exactly
      the two escape routes the analytic work rules out, so we deliberately
      avoid them.

  (b) TIME-INVARIANT LINEAR MIXING. Populations and agents are combined into
      price by fixed (state-independent) impact weights:
          dP_t = gamma_fast * OF_fast_t + gamma_slow * OF_slow_t + noise.
      The mixing weights never depend on the state -- this is precisely the
      "time-invariant linear mixing" hypothesis under which superposition
      was proven to stay Gaussian. (The proof's own caveat is that
      state-dependent mixing weights could escape -- that escape is exactly
      what a later RL/adaptive variant would introduce, which is why this
      fixed-weight version is the correct null.)

  (c) TWO TIMESCALE POPULATIONS (superposition). A "fast" bursty
      short-memory population (fast kernel decay) and a "slow" diffuse
      long-memory population (slow kernel decay, macro/diffuse info). Their
      order flows are summed linearly -- a genuine superposition of
      independent Hawkes scaling limits at different timescales.

Because every stage (self-excitation, agent aggregation, price impact) is
linear and time-invariant, the aggregate signed order flow is a linear
functional of the driving Poisson/sign noise; by superposition its scaling
limit is Gaussian, so its structure function must scale linearly, zeta(q)=qH,
with at most long-range-dependence (exotic covariance) but NO genuine
multifractal concavity. That is the prediction we test below.

VOLATILITY PROXY (stated explicitly)
------------------------------------
Returns r_t = dP_t. Instantaneous volatility proxy = rolling root-mean-square
of returns over a window W (a realized-vol proxy):
    vol_t   = sqrt( mean(r^2) over the trailing window of length W )
    logvol_t = log(vol_t)
The shuffle-surrogate diagnostic is applied to this logvol series, matching
how the prior real-data test (shuffle_surrogate_test.py) operated on a
log-volatility series.

DIAGNOSTIC
----------
zeta_q() and curvature() below are COPIED VERBATIM from
research/scripts/shuffle_surrogate_test.py (the already-validated diagnostic),
so this baseline is scored by the exact same instrument used on real data and
in the project's Monte-Carlo validation. Only the data-loading / real-file
plumbing of that script is omitted.
"""
import json
import os
import numpy as np

# =============================================================================
# Diagnostic -- COPIED VERBATIM from research/scripts/shuffle_surrogate_test.py
# (reused, not rewritten, so results are consistent with prior validated work)
# =============================================================================
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


# =============================================================================
# Multi-agent simulator
# =============================================================================
def simulate_population(n_steps, n_agents, mu, alpha, phi, rng):
    """
    Ensemble of `n_agents` independent discrete-time LINEAR Hawkes agents
    (exponential kernel) sharing the same parameters, returning the
    population's net signed order flow per time step.

    Per agent, per step t:
        lambda_t   = mu + alpha * M_t            (LINEAR intensity)
        events_t   ~ Poisson(lambda_t)           (order events this step)
        M_{t+1}    = phi * M_t + events_t         (exp-decay memory kernel)
        sign_t     ~ {+1,-1} equal prob           (buy/sell)
        flow_t     = sign_t * events_t            (signed order flow)

    Branching ratio n = alpha/(1-phi) must be < 1 (subcritical, stable).
    Population net flow at step t = sum over agents of flow_t.

    Faithfulness notes:
      * LINEAR in past activity  -> no quadratic/Zumbach feedback.
      * subcritical (n<1)        -> no exact criticality.
      * intensity clustering drives |flow| clustering -> long-range dependence
        in volatility WITHOUT any nonlinear cascade mechanism.
    """
    branching = alpha / (1.0 - phi)
    assert branching < 1.0, f"unstable: branching ratio {branching:.3f} >= 1"

    M = np.zeros(n_agents)                 # decaying memory of past events
    net_flow = np.empty(n_steps)
    activity = np.empty(n_steps)           # total events (for diagnostics)
    for t in range(n_steps):
        lam = mu + alpha * M               # linear intensity, per agent
        np.maximum(lam, 0.0, out=lam)
        events = rng.poisson(lam)          # events per agent this step
        signs = rng.integers(0, 2, size=n_agents) * 2 - 1   # +/-1
        net_flow[t] = np.sum(signs * events)
        activity[t] = np.sum(events)
        M = phi * M + events               # update memory kernel
    return net_flow, activity, branching


def simulate_market(n_steps, rng,
                    # FAST population: bursty, short memory
                    n_fast=40, mu_fast=0.15, alpha_fast=0.45, phi_fast=0.30,
                    # SLOW population: diffuse macro info, long memory
                    n_slow=40, mu_slow=0.06, alpha_slow=0.045, phi_slow=0.90,
                    # time-invariant linear price impact + microstructure noise
                    gamma_fast=1.0, gamma_slow=1.4, noise_sd=0.5):
    """
    Combine two timescale populations into a price via TIME-INVARIANT LINEAR
    impact. Returns dict with returns, logvol proxy, and metadata.
    """
    of_fast, act_fast, br_fast = simulate_population(
        n_steps, n_fast, mu_fast, alpha_fast, phi_fast, rng)
    of_slow, act_slow, br_slow = simulate_population(
        n_steps, n_slow, mu_slow, alpha_slow, phi_slow, rng)

    # Fixed-weight linear price impact (state-INDEPENDENT mixing) + noise.
    noise = rng.standard_normal(n_steps) * noise_sd
    returns = gamma_fast * of_fast + gamma_slow * of_slow + noise

    return {
        "returns": returns,
        "of_fast": of_fast, "of_slow": of_slow,
        "act_fast": act_fast, "act_slow": act_slow,
        "branching_fast": br_fast, "branching_slow": br_slow,
    }


def rolling_logvol(returns, window):
    """Instantaneous vol proxy = rolling RMS of returns; logvol = log(vol)."""
    r2 = returns**2
    # trailing rolling mean of r^2 via cumulative sum
    csum = np.cumsum(np.insert(r2, 0, 0.0))
    rms2 = (csum[window:] - csum[:-window]) / window
    rms2 = np.clip(rms2, 1e-12, None)
    return np.log(np.sqrt(rms2))


# =============================================================================
# Shuffle-surrogate genuineness test (same logic as shuffle_surrogate_test.py)
# =============================================================================
def genuineness_test(logvol, lags, qs, n_shuffles, rng):
    curv_raw = curvature(qs, zeta_q(logvol, lags, qs))
    curv_shufs = np.array([
        curvature(qs, zeta_q(rng.permutation(logvol), lags, qs))
        for _ in range(n_shuffles)
    ])
    curv_shuf_mean = float(np.mean(curv_shufs))
    curv_shuf_std = float(np.std(curv_shufs))

    denom = abs(curv_raw) if abs(curv_raw) > 1e-12 else 1e-12
    frac_distr = float(np.clip(abs(curv_shuf_mean) / denom, 0, 1.5))
    frac_temporal = 1.0 - frac_distr

    # significance: how far the raw curvature sits beyond the shuffle spread.
    # (If temporal curvature were genuine, raw would be MANY sigma from the
    #  shuffle distribution. Near-zero z => curvature is distributional.)
    z_excess = float((abs(curv_raw) - abs(curv_shuf_mean)) /
                     (curv_shuf_std if curv_shuf_std > 1e-12 else 1e-12))
    return {
        "curv_raw": float(curv_raw),
        "curv_shuf_mean": curv_shuf_mean,
        "curv_shuf_std": curv_shuf_std,
        "frac_distributional": frac_distr,
        "frac_temporal": frac_temporal,
        "z_excess_over_shuffle": z_excess,
    }


def gaussianity_check(returns):
    """Excess kurtosis of returns -- linear-Gaussian null should be ~0."""
    x = returns - np.mean(returns)
    m2 = np.mean(x**2)
    m4 = np.mean(x**4)
    return float(m4 / m2**2 - 3.0)


def main():
    rng = np.random.default_rng(2026)

    N = 60000          # steps simulated (well above the 5k-10k floor; cheap)
    WINDOW = 20        # rolling-vol window
    N_SHUFFLES = 40    # matches shuffle_surrogate_test.py
    # same lag/q grid family as the validated diagnostic
    lags = np.unique(np.round(np.geomspace(2, 250, 16)).astype(int))
    qs = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])

    sim = simulate_market(N, rng)
    returns = sim["returns"]
    logvol = rolling_logvol(returns, WINDOW)

    result = genuineness_test(logvol, lags, qs, N_SHUFFLES, rng)
    exkurt = gaussianity_check(returns)

    # crude vol-clustering check: autocorr of |returns| at lag 1 and 50
    ar = np.abs(returns) - np.mean(np.abs(returns))
    def acf(x, k):
        return float(np.sum(x[:-k]*x[k:]) / np.sum(x*x))
    acf_absret = {"lag1": acf(ar, 1), "lag50": acf(ar, 50), "lag200": acf(ar, 200)}

    out = {
        "config": {
            "n_steps": N, "rolling_vol_window": WINDOW,
            "n_shuffles": N_SHUFFLES,
            "lags": lags.tolist(), "qs": qs.tolist(),
            "branching_fast": sim["branching_fast"],
            "branching_slow": sim["branching_slow"],
        },
        "genuineness": result,
        "return_excess_kurtosis": exkurt,
        "absret_autocorr": acf_absret,
    }

    print("=" * 70)
    print("TRACK B BASELINE -- linear Hawkes superposition (hand-specified null)")
    print("=" * 70)
    print(f"steps={N}, vol window={WINDOW}, shuffles={N_SHUFFLES}")
    print(f"branching ratio  fast={sim['branching_fast']:.3f}  "
          f"slow={sim['branching_slow']:.3f}  (both subcritical, <1)")
    print(f"return excess kurtosis: {exkurt:+.3f}  (linear-Gaussian null ~ 0)")
    print(f"|return| autocorr: lag1={acf_absret['lag1']:.3f}  "
          f"lag50={acf_absret['lag50']:.3f}  lag200={acf_absret['lag200']:.3f}  "
          f"(persistent => long-range dependence present)")
    print("-" * 70)
    print(f"curvature raw        : {result['curv_raw']:+.5f}")
    print(f"curvature shuffle    : {result['curv_shuf_mean']:+.5f} "
          f"(std {result['curv_shuf_std']:.5f})")
    print(f"distributional frac  : {result['frac_distributional']*100:6.1f}%")
    print(f"temporal fraction    : {result['frac_temporal']*100:6.1f}%")
    print(f"z (raw beyond shuffle): {result['z_excess_over_shuffle']:+.2f} sigma")
    print("=" * 70)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "track_b_baseline_results.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"results saved to {out_path}")
    return out


if __name__ == "__main__":
    main()
