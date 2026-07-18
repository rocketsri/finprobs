"""
Track A -- Deep Hedging under the log S-fBM stochastic-volatility model.
Steps 1-2 of the DL/RL research task (foundation + correctness check).

STEP 1  price-process simulator (Cholesky log-vol + explicit leverage)
STEP 2  Buehler-style feed-forward deep hedger trained to minimise the
        entropic risk of terminal hedging P&L of a European call, with a
        small proportional transaction cost.

CORRECTNESS TEST  drive the process to the near-Black-Scholes limit
        (H -> 0.499, lambda^2 -> ~0) and confirm the learned hedge ratio
        matches the BS delta in shape and that the realised hedging-error
        magnitude is comparable to a directly-computed BS delta-hedge
        benchmark.

Reused machinery
----------------
* log S-fBM covariance C_P(tau;H,T,nu2) = (nu2/2)[T^{2H} - tau^{2H}] , tau<T
  with nu2 = lambda^2 / (H(1-2H))  -- identical to
  research/scripts/test_lambda_cancellation.py
* Cholesky factorisation of the stationary covariance matrix to draw the
  Gaussian log-vol path -- identical approach to
  research/scripts/aggregation_simulation.py

MODELLING ASSUMPTION (mine, not part of the source log S-fBM paper)
------------------------------------------------------------------
The source log S-fBM model specifies only the (scalar) log-volatility
process; it has no price/leverage dynamics.  I add a leverage-correlated
price the way rough Bergomi does:

    d log S_t = -1/2 sigma_t^2 dt + sigma_t ( rho dW^omega_t
                                              + sqrt(1-rho^2) dW^perp_t )

In the discrete Cholesky construction omega = omega_bar + L z, the vector
z ~ N(0,I) holds the *standardised innovations* that drive the log-vol
path.  I identify dW^omega at step k with z_k (the new randomness entering
node k through the lower-triangular Cholesky factor L, whose positive
diagonal makes a positive z_k raise omega_k / vol).  The price Brownian
increment at step k is  rho*z_k + sqrt(1-rho^2)*perp_k  with perp iid.
With rho<0 this reproduces the equity leverage effect.  This choice is an
explicit assumption layered on top of the source model, stated here so it
is not mistaken for part of the original specification.
"""

from __future__ import annotations
import math
import time
import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# Reproducibility / threads (CPU, 4 cores)
# --------------------------------------------------------------------------
torch.set_num_threads(4)
DEVICE = torch.device("cpu")
DT = 1.0 / 252.0          # one trading day, in years (matches lag units of T)
ANN = 1.0 / DT            # annualisation factor for realised variance


# ==========================================================================
# STEP 1 -- simulator
# ==========================================================================
def nu2_from_lambda(lam2: float, H: float) -> float:
    """nu2 = lambda^2 / (H (1-2H))  (Wu-Muzy-Bacry 2022, valid for H<1/2)."""
    return lam2 / (H * (1.0 - 2.0 * H))


def cov_power(tau: np.ndarray, H: float, T: float, nu2: float) -> np.ndarray:
    """log S-fBM power-law covariance C_P(tau;H,T,nu2), zero beyond T."""
    tau = np.asarray(tau, dtype=float)
    return np.where(tau < T, (nu2 / 2.0) * (T ** (2 * H) - tau ** (2 * H)), 0.0)


def _stable_cholesky(Sigma: np.ndarray) -> np.ndarray:
    """Cholesky with adaptive jitter; falls back to a clipped eigen-sqrt.

    Returns a lower-triangular-ish factor L with L L^T ~= Sigma.
    """
    n = Sigma.shape[0]
    c0 = float(np.max(np.diag(Sigma))) + 1e-300
    for p in range(-12, -2):                      # jitter 1e-12*c0 ... 1e-2*c0
        try:
            return np.linalg.cholesky(Sigma + (10.0 ** p) * c0 * np.eye(n))
        except np.linalg.LinAlgError:
            continue
    # eigen fallback (kernel is only approximately PSD after truncation)
    w, V = np.linalg.eigh(Sigma)
    w = np.clip(w, 0.0, None)
    return V @ np.diag(np.sqrt(w))


def simulate_paths(H: float, lam2: float, T: float, *,
                   rho: float = -0.7, sigma0: float = 0.20,
                   n_steps: int = 21, n_paths: int = 8000,
                   seed: int = 0):
    """Simulate leverage-correlated price paths driven by a log S-fBM log-vol.

    Returns
    -------
    S      : (n_paths, n_steps+1) price paths, S0 = 1
    sigma  : (n_paths, n_steps+1) instantaneous vol (annualised)
    logret : (n_paths, n_steps)   log-returns
    """
    rng = np.random.default_rng(seed)
    nu2 = nu2_from_lambda(lam2, H)
    omega_bar = 2.0 * math.log(sigma0)            # median vol == sigma0

    m = n_steps + 1
    lag = np.abs(np.arange(m)[:, None] - np.arange(m)[None, :]).astype(float) * DT
    Sigma = cov_power(lag, H, T, nu2)
    L = _stable_cholesky(Sigma)                   # (m, m)

    z = rng.standard_normal((n_paths, m))         # driving innovations
    omega = omega_bar + z @ L.T                    # (n_paths, m), row = one path
    sigma = np.exp(0.5 * omega)                    # instantaneous vol

    # leverage-correlated price Brownian increments
    perp = rng.standard_normal((n_paths, n_steps))
    z_step = z[:, :n_steps]                        # innovation entering each step
    dW = rho * z_step + math.sqrt(1.0 - rho * rho) * perp   # unit-variance
    sig_step = sigma[:, :n_steps]
    logret = -0.5 * sig_step ** 2 * DT + sig_step * math.sqrt(DT) * dW

    logS = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(logret, axis=1)], axis=1)
    S = np.exp(logS)
    return (S.astype(np.float32), sigma.astype(np.float32), logret.astype(np.float32))


def realized_vol_proxy(logret: np.ndarray, sigma0: float, window: int = 10):
    """Causal trailing annualised realised vol available at each decision step.

    proxy[:,k] uses returns r_1..r_k (known when choosing the holding over
    [t_k,t_{k+1}]); k=0 falls back to sigma0.
    """
    n_paths, n_steps = logret.shape
    proxy = np.full((n_paths, n_steps), sigma0, dtype=np.float32)
    sq = logret ** 2
    for k in range(1, n_steps):
        lo = max(0, k - window)
        proxy[:, k] = np.sqrt(ANN * sq[:, lo:k].mean(axis=1) + 1e-12)
    return proxy


# ==========================================================================
# Black-Scholes reference (r = 0)
# ==========================================================================
def _norm_cdf(x):
    if isinstance(x, torch.Tensor):
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    from math import erf
    return 0.5 * (1.0 + np.vectorize(erf)(x / math.sqrt(2.0)))


def bs_call_price(S, K, sigma, tau):
    S = np.asarray(S, float); tau = np.asarray(tau, float)
    out = np.maximum(S - K, 0.0)
    m = tau > 1e-12
    if np.any(m):
        v = sigma * np.sqrt(tau[m] if tau.ndim else tau)
        d1 = (np.log(S[m] / K) + 0.5 * sigma ** 2 * tau[m]) / v
        d2 = d1 - v
        out = np.array(out, float)
        out[m] = S[m] * _norm_cdf(d1) - K * _norm_cdf(d2)
    return out


def bs_delta(S, K, sigma, tau):
    S = np.asarray(S, float); tau = np.asarray(tau, float)
    d = (S > K).astype(float)                      # tau->0 limit
    m = tau > 1e-12
    d1 = np.where(m, (np.log(np.maximum(S, 1e-12) / K) + 0.5 * sigma ** 2 * tau)
                  / (sigma * np.sqrt(np.where(m, tau, 1.0))), 0.0)
    return np.where(m, _norm_cdf(d1), d)


# ==========================================================================
# STEP 2 -- Buehler-style deep hedger
# ==========================================================================
class HedgeNet(nn.Module):
    """Shared feed-forward hedger, applied per (path,step).

    Input features (all O(1)):
        f1 = log(S/K) / (sigma0 sqrt(T_total))     standardised moneyness
        f2 = tau / T_total                          time-to-maturity in [0,1]
        f3 = rv_proxy / sigma0                      realised-vol proxy
    Output: scalar holding (shares) in the stock -- unbounded (faithful to
    deep hedging; the [0,1] shape of a call delta is learned, not imposed).
    """
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats):                       # feats: (..., 3)
        return self.net(feats).squeeze(-1)


def build_features(S, rv, sigma0, T_total, n_steps):
    """Assemble (B, n_steps, 3) feature tensor at each decision step k."""
    K = 1.0
    tau = (T_total - np.arange(n_steps) * DT)[None, :]         # (1, n_steps)
    f1 = np.log(S[:, :n_steps] / K) / (sigma0 * math.sqrt(T_total))
    f2 = np.broadcast_to(tau / T_total, f1.shape)
    f3 = rv / sigma0
    feats = np.stack([f1, f2, f3], axis=-1).astype(np.float32)
    return feats


def hedging_pnl(net, S_t, feats_t, premium, K, kappa):
    """Terminal hedging P&L (profit; higher is better), vectorised over batch.

    P&L = premium + sum_k delta_k (S_{k+1}-S_k) - transaction cost - payoff
    Holdings are Markovian in market state; transaction costs couple steps.
    """
    B, n_steps = feats_t.shape[0], feats_t.shape[1]
    deltas = net(feats_t)                                       # (B, n_steps)
    dS = S_t[:, 1:] - S_t[:, :n_steps]                          # (B, n_steps)
    trading = (deltas * dS).sum(dim=1)

    prev = torch.zeros(B, device=S_t.device)
    cost = torch.zeros(B, device=S_t.device)
    for k in range(n_steps):
        cost = cost + kappa * S_t[:, k] * torch.abs(deltas[:, k] - prev)
        prev = deltas[:, k]
    cost = cost + kappa * S_t[:, n_steps] * torch.abs(prev)     # close-out
    payoff = torch.clamp(S_t[:, n_steps] - K, min=0.0)
    return premium + trading - cost - payoff, deltas


def entropic_risk(pnl, gamma):
    """rho(X) = (1/gamma) log E[exp(-gamma X)]  via a stable log-sum-exp."""
    n = pnl.shape[0]
    return (torch.logsumexp(-gamma * pnl, dim=0) - math.log(n)) / gamma


def bs_benchmark_pnl(S, rv_unused, sigma0, K, T_total, n_steps, premium, kappa):
    """Directly-computed BS delta-hedge P&L under the same friction model."""
    tau = T_total - np.arange(n_steps) * DT
    deltas = np.stack([bs_delta(S[:, k], K, sigma0, tau[k]) for k in range(n_steps)],
                      axis=1)                                    # (B, n_steps)
    dS = S[:, 1:n_steps + 1] - S[:, :n_steps]
    trading = (deltas * dS).sum(axis=1)
    prev = np.zeros(S.shape[0])
    cost = np.zeros(S.shape[0])
    for k in range(n_steps):
        cost += kappa * S[:, k] * np.abs(deltas[:, k] - prev)
        prev = deltas[:, k]
    cost += kappa * S[:, n_steps] * np.abs(prev)
    payoff = np.maximum(S[:, n_steps] - K, 0.0)
    return premium + trading - cost - payoff, deltas


# ==========================================================================
# Training
# ==========================================================================
def train_hedger(S_tr, rv_tr, *, sigma0, K, T_total, n_steps, premium, kappa,
                 gamma=10.0, epochs=60, batch=512, lr=1e-3, seed=0, hidden=32,
                 log=print):
    torch.manual_seed(seed)
    net = HedgeNet(hidden=hidden).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    feats = build_features(S_tr, rv_tr, sigma0, T_total, n_steps)
    feats_t = torch.from_numpy(feats)
    S_t = torch.from_numpy(S_tr)
    N = S_tr.shape[0]

    for ep in range(epochs):
        perm = torch.randperm(N)
        tot = 0.0
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            pnl, _ = hedging_pnl(net, S_t[idx], feats_t[idx], premium, K, kappa)
            loss = entropic_risk(pnl, gamma)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx)
        sched.step()
        if (ep + 1) % 15 == 0 or ep == 0:
            log(f"    epoch {ep+1:3d}/{epochs}  entropic-risk={tot/N:+.6f}")
    return net


def evaluate(net, S_te, rv_te, *, sigma0, K, T_total, n_steps, premium, kappa):
    feats = build_features(S_te, rv_te, sigma0, T_total, n_steps)
    with torch.no_grad():
        pnl, deltas = hedging_pnl(net, torch.from_numpy(S_te),
                                  torch.from_numpy(feats), premium, K, kappa)
    return pnl.numpy(), deltas.numpy()


# ==========================================================================
# Correctness check (near-Black-Scholes limit)
# ==========================================================================
def run_correctness_check(out_lines):
    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s); out_lines.append(s)

    # --- near-BS parameters -------------------------------------------------
    H, lam2, T = 0.499, 1e-6, 0.5      # near non-rough, near-zero intermittency
    sigma0, rho = 0.20, -0.7
    K = 1.0
    n_steps = 21                        # 1-month option, daily rebalancing
    T_total = n_steps * DT
    kappa = 1e-4                        # 1 bp proportional cost
    premium = float(bs_call_price(np.array([1.0]), K, sigma0, T_total)[0])

    nu2 = nu2_from_lambda(lam2, H)
    log("=" * 74)
    log("CORRECTNESS CHECK  --  near-Black-Scholes limit")
    log("=" * 74)
    log(f"  params: H={H}  lambda^2={lam2:g}  T={T}  ->  nu2={nu2:.3e}")
    log(f"  Var(omega)=C(0)={cov_power(0.0,H,T,nu2):.3e}  "
        f"=> vol wobble std ~ {0.5*math.sqrt(cov_power(0.0,H,T,nu2))*100:.3f}%")
    log(f"  sigma0={sigma0}  rho={rho}  maturity={n_steps} days "
        f"(T_total={T_total:.4f}y)  kappa={kappa}")
    log(f"  BS ATM call premium (sigma0) = {premium:.6f}")

    # --- simulate + verify realised vol matches target ---------------------
    t0 = time.time()
    S_tr, sig_tr, lr_tr = simulate_paths(H, lam2, T, rho=rho, sigma0=sigma0,
                                         n_steps=n_steps, n_paths=8000, seed=1)
    S_te, sig_te, lr_te = simulate_paths(H, lam2, T, rho=rho, sigma0=sigma0,
                                         n_steps=n_steps, n_paths=20000, seed=2)
    rv_tr = realized_vol_proxy(lr_tr, sigma0)
    rv_te = realized_vol_proxy(lr_te, sigma0)
    realized_ann = float(np.sqrt(ANN * np.mean(lr_tr ** 2)))
    log(f"  simulated 8k train + 20k test paths in {time.time()-t0:.1f}s")
    log(f"  sanity: realised annualised vol of paths = {realized_ann:.4f} "
        f"(target sigma0={sigma0})  |  vol path range "
        f"[{sig_tr.min():.4f},{sig_tr.max():.4f}]")

    # --- train -------------------------------------------------------------
    log("  training deep hedger (entropic risk, gamma=10)...")
    t0 = time.time()
    net = train_hedger(S_tr, rv_tr, sigma0=sigma0, K=K, T_total=T_total,
                       n_steps=n_steps, premium=premium, kappa=kappa,
                       gamma=10.0, epochs=60, batch=512, lr=1e-3, seed=0, log=log)
    log(f"  trained in {time.time()-t0:.1f}s")

    # --- evaluate P&L vs BS benchmark --------------------------------------
    pnl_nn, d_nn_states = evaluate(net, S_te, rv_te, sigma0=sigma0, K=K,
                                   T_total=T_total, n_steps=n_steps,
                                   premium=premium, kappa=kappa)
    pnl_bs, d_bs_states = bs_benchmark_pnl(S_te, rv_te, sigma0, K, T_total,
                                           n_steps, premium, kappa)

    log("-" * 74)
    log("(a) terminal hedging-error distribution (test, 20k paths)")
    stats = {}
    for name, p in [("Deep hedger", pnl_nn), ("BS delta-hedge", pnl_bs)]:
        cvar95 = -np.mean(np.sort(p)[:int(0.05 * len(p))])   # CVaR95 of loss
        log(f"    {name:16s}  mean={p.mean():+.5f}  std={p.std():.5f}  "
            f"CVaR95(loss)={cvar95:+.5f}  |  as %notional std={p.std()*100:.3f}%")
        stats[name] = dict(mean=float(p.mean()), std=float(p.std()),
                           cvar95_loss=float(cvar95))
    std_ratio = float(pnl_nn.std() / pnl_bs.std())
    log(f"    std ratio  NN / BS = {std_ratio:.3f}")

    # --- learned hedge ratio vs BS delta, OVER THE VISITED STATE DIST -------
    # The economically meaningful test is agreement where the hedger actually
    # operates -- the (path,step) states that generate the P&L above -- not on
    # an arbitrary grid that probes deep-OTM/ITM states never seen in training.
    d_nn_flat = d_nn_states.reshape(-1)
    d_bs_flat = d_bs_states.reshape(-1)
    corr_emp = float(np.corrcoef(d_nn_flat, d_bs_flat)[0, 1])
    absdiff = np.abs(d_nn_flat - d_bs_flat)
    mae_emp = float(absdiff.mean())
    p99_emp = float(np.percentile(absdiff, 99.0))
    max_emp = float(absdiff.max())
    log("-" * 74)
    log("(b) learned hedge ratio vs BS delta over the EMPIRICAL state dist")
    log("    (all path x step states that produced the P&L above)")
    log(f"    correlation(delta_NN, delta_BS)  = {corr_emp:.4f}")
    log(f"    mean |delta_NN - delta_BS|       = {mae_emp:.4f}")
    log(f"    99th pct |delta_NN - delta_BS|   = {p99_emp:.4f}")
    log(f"    max |delta_NN - delta_BS|        = {max_emp:.4f}")

    # --- full-grid printout (transparency; includes never-visited wings) ---
    grid = np.linspace(-0.10, 0.10, 41)             # log-moneyness
    S_grid = np.exp(grid)
    tau_eval = T_total - 1 * DT                      # an early step
    feats = np.stack([grid / (sigma0 * math.sqrt(T_total)),
                      np.full_like(grid, tau_eval / T_total),
                      np.ones_like(grid)], axis=-1).astype(np.float32)
    with torch.no_grad():
        d_nn = net(torch.from_numpy(feats)).numpy()
    d_bs = bs_delta(S_grid, K, sigma0, tau_eval)
    corr_grid = float(np.corrcoef(d_nn, d_bs)[0, 1])
    mono = bool(np.all(np.diff(d_nn) > -1e-3))       # non-decreasing in moneyness
    # deviation restricted to the region actually visited at this early step
    # (99% of paths lie within ~+/-0.033 log-moneyness one step in)
    visited = np.abs(grid) <= 0.033
    max_visited = float(np.max(np.abs(d_nn - d_bs)[visited]))
    max_grid = float(np.max(np.abs(d_nn - d_bs)))
    log("-" * 74)
    log(f"(c) grid slice at tau={tau_eval:.4f}y, rv=sigma0 (transparency)")
    log(f"    correlation over grid           = {corr_grid:.4f}")
    log(f"    monotone non-decreasing         = {mono}")
    log(f"    max|d| within visited |lm|<=.033 = {max_visited:.4f}")
    log(f"    max|d| over full grid +/-0.10    = {max_grid:.4f}  "
        f"(wings are extrapolation; unvisited one step in)")
    log("      log-moneyness   delta_NN   delta_BS   visited?")
    for i in range(0, 41, 5):
        log(f"      {grid[i]:+8.3f}      {d_nn[i]:7.4f}   {d_bs[i]:7.4f}   "
            f"{'yes' if abs(grid[i])<=0.033 else 'no (extrap)'}")

    # --- verdict -----------------------------------------------------------
    # Shape judged over the visited state distribution (correlation + typical
    # deviation), monotonicity, and P&L parity with the BS benchmark.
    pass_shape = (corr_emp > 0.98) and mono and (p99_emp < 0.12) \
        and (max_visited < 0.12)
    pass_error = (abs(float(pnl_nn.mean())) < 0.01) and (std_ratio < 1.5)
    passed = bool(pass_shape and pass_error)
    log("=" * 74)
    log(f"  shape criterion (corr_emp>0.98, monotone, p99|d|<0.12, "
        f"visited-max<0.12): {pass_shape}")
    log(f"  error criterion (|mean|<0.01, std_ratio<1.5)              "
        f": {pass_error}")
    log(f"  CORRECTNESS CHECK PASSED = {passed}")
    log("=" * 74)

    results = dict(
        params=dict(H=H, lambda2=lam2, T=T, nu2=float(nu2), sigma0=sigma0,
                    rho=rho, K=K, n_steps=n_steps, T_total=float(T_total),
                    kappa=kappa, premium=premium, gamma=10.0, epochs=60,
                    n_train=8000, n_test=20000),
        sanity=dict(realized_ann_vol=realized_ann,
                    vol_path_min=float(sig_tr.min()),
                    vol_path_max=float(sig_tr.max())),
        pnl=stats, std_ratio=std_ratio,
        shape_empirical=dict(corr=corr_emp, mae=mae_emp, p99_absdiff=p99_emp,
                             max_absdiff=max_emp),
        shape_grid=dict(corr=corr_grid, monotone=mono,
                        max_visited=max_visited, max_full_grid=max_grid),
        verdict=dict(pass_shape=bool(pass_shape), pass_error=bool(pass_error),
                     passed=passed),
    )
    return passed, results


if __name__ == "__main__":
    import os
    import json
    out_lines = []
    ok, results = run_correctness_check(out_lines)
    here = os.path.dirname(os.path.abspath(__file__))

    txt_path = os.path.join(here, "track_a_correctness_results.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(out_lines) + "\n")

    json_path = os.path.join(here, "track_a_v1_correctness_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nresults written to {txt_path}")
    print(f"results written to {json_path}")
