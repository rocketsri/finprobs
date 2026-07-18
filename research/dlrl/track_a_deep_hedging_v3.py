"""
Track A v3 -- Deep hedging under the log S-fBM volatility model.
Independent implementation (does not read v1/v2).

GOAL (this file only builds + validates the machinery; the lambda^2/T
sensitivity experiment is deliberately NOT run here):
  Buehler-style neural hedging policy, trained on simulated log S-fBM paths,
  minimizing a convex risk measure (CVaR) of terminal hedging P&L with
  transaction costs. Correctness gate: in the H->0.499, lambda^2->0 limit the
  model degenerates to (near) Black-Scholes; the learned hedge must resemble
  BS delta-hedging and achieve comparable hedging error.

MODEL / SIMULATION
------------------
Log-volatility omega_t is a mean-zero Gaussian process with the log S-fBM
stationary covariance (Wu-Muzy-Bacry 2022, same formula as the covariance-level
scripts in research/scripts/):

    C(tau) = (nu2/2)[T^(2H) - tau^(2H)]  for tau < T,   nu2 = lambda^2/(H(1-2H))

Sampled exactly on the trading grid via a Cholesky factor of the level
covariance matrix Sigma[i,j] = C(|t_i - t_j|)  (same Cholesky machinery as
research/scripts/aggregation_simulation.py). Instantaneous vol is
    sigma_t = sigma_ref * exp(omega_t - 0.5*Var(omega_t))          [median-normalized]

LEVERAGE (my explicit added assumption -- log S-fBM has no price dynamics in the
source paper). I couple price to vol at the *increment* level. Let
    g_i = standardized(omega_{i+1} - omega_i)   (cross-path z-score of the vol increment)
    dB_i = rho * g_i + sqrt(1-rho^2) * eps_i,    eps_i ~ N(0,1) iid, rho < 0
    r_i  = (mu - 0.5 sigma_i^2) dt + sigma_i sqrt(dt) dB_i,   S_{i+1} = S_i exp(r_i)
So a positive vol shock (omega up) drives returns down when rho<0: the standard
leverage/skew effect, built directly on top of the exact log S-fBM omega path.
mu = 0 (drift-free / r=0), so the BS benchmark uses r=0.

OBJECTIVE
---------
CVaR (Rockafellar-Uryasev) of the loss L=-PnL, minimized jointly over the policy
and the VaR scalar w:  CVaR_a = min_w { w + (1/(1-a)) E[(L-w)_+] }.
(I chose CVaR over the entropic risk used by Horvath/Teichmann/Zuric 2021 to
differentiate from the closest prior work; CVaR is also the original Buehler
deep-hedging objective and is directly interpretable as a tail loss.)

ARCHITECTURE
------------
Two policies are implemented and both validated in the BS limit:
  * "ff"  : feedforward per-step net fed a lightweight running realized-vol
            feature (a cheap Markovianization of the non-Markovian state).
  * "gru" : compact 1-layer GRU hedger that carries hidden state across steps.
Zhu/Wu/Diao (2023) argue recurrence matters for rough-vol memory. I agree the
state is genuinely non-Markovian at H=0.1, so I implement the GRU and keep it as
the intended policy for the downstream experiment -- but I validate BOTH here and
report numbers, because the correctness gate itself is Markovian (BS) and a
recurrent net must be shown to not *underperform* there before I trust it.

PyTorch CPU only.
"""
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
SQRT2 = math.sqrt(2.0)


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
def cov_logsfbm(tau, H, T, nu2):
    tau = np.asarray(tau, dtype=float)
    return np.where(tau < T, (nu2 / 2.0) * (T ** (2 * H) - tau ** (2 * H)), 0.0)


def cholesky_omega_factor(n_grid, dt, H, T, nu2, jitter=1e-10):
    """Cholesky factor L of the log-vol level covariance on the grid t_i=i*dt."""
    idx = np.arange(n_grid)
    lag = np.abs(idx[:, None] - idx[None, :]).astype(float) * dt  # in years
    Sigma = cov_logsfbm(lag, H, T, nu2)
    Sigma = Sigma + jitter * np.eye(n_grid)
    return np.linalg.cholesky(Sigma), Sigma[0, 0]  # L, Var(omega)


def simulate_paths(n_paths, n_steps, dt, H, T, nu2, sigma_ref, rho, mu,
                   seed=0):
    """Return S (n_paths, n_steps+1) and sigma (n_paths, n_steps+1)."""
    rng = np.random.default_rng(seed)
    n_grid = n_steps + 1
    L, var_omega = cholesky_omega_factor(n_grid, dt, H, T, nu2)

    z = rng.standard_normal((n_grid, n_paths))
    omega = (L @ z).T                                 # (n_paths, n_grid)
    sigma = sigma_ref * np.exp(omega - 0.5 * var_omega)

    # leverage: couple price shock to standardized vol increment
    domega = np.diff(omega, axis=1)                   # (n_paths, n_steps)
    sd = domega.std(axis=0, keepdims=True) + 1e-12
    g = domega / sd
    eps = rng.standard_normal((n_paths, n_steps))
    dB = rho * g + math.sqrt(max(1e-12, 1 - rho ** 2)) * eps

    S = np.empty((n_paths, n_grid))
    S[:, 0] = 1.0
    for i in range(n_steps):
        r = (mu - 0.5 * sigma[:, i] ** 2) * dt + sigma[:, i] * math.sqrt(dt) * dB[:, i]
        S[:, i + 1] = S[:, i] * np.exp(r)
    return S.astype(np.float32), sigma.astype(np.float32), var_omega


# ----------------------------------------------------------------------------
# Black-Scholes (r=0)
# ----------------------------------------------------------------------------
def _norm_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / SQRT2))


def bs_call_price_delta(S, K, sigma, tau):
    """r=0. Returns (price, delta). tau, sigma scalars or broadcastable tensors."""
    S = torch.as_tensor(S, dtype=torch.float32)
    tau = torch.clamp(torch.as_tensor(tau, dtype=torch.float32), min=1e-8)
    sig = torch.as_tensor(sigma, dtype=torch.float32)
    vsqrt = sig * torch.sqrt(tau)
    d1 = (torch.log(S / K) + 0.5 * sig ** 2 * tau) / vsqrt
    d2 = d1 - vsqrt
    price = S * _norm_cdf(d1) - K * _norm_cdf(d2)
    return price, _norm_cdf(d1)


# ----------------------------------------------------------------------------
# Policies
# ----------------------------------------------------------------------------
class FFHedger(nn.Module):
    """Per-step feedforward. Features: [tau_frac, log-moneyness(scaled), prev_hold,
    running realized-vol / sigma_ref]."""
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats):
        return torch.sigmoid(self.net(feats)).squeeze(-1)  # call delta in (0,1)


class GRUHedger(nn.Module):
    """1-layer GRU carrying hidden state across rehedge dates.
    Per-step input: [tau_frac, log-moneyness(scaled), prev_hold]."""
    def __init__(self, hidden=16):
        super().__init__()
        self.cell = nn.GRUCell(3, hidden)
        self.head = nn.Linear(hidden, 1)
        self.hidden = hidden

    def step(self, x, h):
        h = self.cell(x, h)
        return torch.sigmoid(self.head(h)).squeeze(-1), h


def run_policy(policy, kind, S, K, sigma_ref, T_opt, dt):
    """Roll the policy forward, return holdings (n_paths, n_steps)."""
    P, n_grid = S.shape
    n_steps = n_grid - 1
    device = S.device
    holds = torch.zeros(P, n_steps, device=device)
    prev = torch.zeros(P, device=device)
    h = torch.zeros(P, policy.hidden, device=device) if kind == "gru" else None
    sumsq = torch.zeros(P, device=device)
    logS = torch.log(S)
    mny_scale = sigma_ref * math.sqrt(max(T_opt, 1e-8))
    for i in range(n_steps):
        tau = T_opt - i * dt
        tau_frac = torch.full((P,), tau / T_opt, device=device)
        mny = (logS[:, i] - math.log(K)) / mny_scale
        if kind == "ff":
            rv = torch.sqrt(sumsq / max(i, 1) / dt) / sigma_ref if i > 0 else torch.zeros(P, device=device)
            feats = torch.stack([tau_frac, mny, prev, rv], dim=1)
            d = policy(feats)
        else:
            feats = torch.stack([tau_frac, mny, prev], dim=1)
            d, h = policy.step(feats, h)
        holds[:, i] = d
        prev = d
        if i + 1 < n_grid:
            sumsq = sumsq + (logS[:, i + 1] - logS[:, i]) ** 2
    return holds


def hedging_pnl(holds, S, K, premium, cost_kappa):
    """Seller PnL: premium - payoff + trading gains - costs."""
    n_steps = holds.shape[1]
    dS = S[:, 1:n_steps + 1] - S[:, :n_steps]
    gains = (holds * dS).sum(dim=1)
    payoff = torch.clamp(S[:, -1] - K, min=0.0)
    dholds = torch.diff(holds, dim=1, prepend=torch.zeros(holds.shape[0], 1, device=holds.device))
    costs = cost_kappa * (torch.abs(dholds) * S[:, :n_steps]).sum(dim=1)
    return premium - payoff + gains - costs


def cvar_loss(pnl, w, alpha):
    loss = -pnl
    return w + (1.0 / (1.0 - alpha)) * torch.clamp(loss - w, min=0.0).mean()


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train_policy(kind, S_train, K, sigma_ref, T_opt, dt, premium, cost_kappa,
                 alpha=0.95, epochs=60, batch=4096, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    policy = FFHedger() if kind == "ff" else GRUHedger()
    w = nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam(list(policy.parameters()) + [w], lr=lr)
    P = S_train.shape[0]
    S_t = torch.from_numpy(S_train)
    for ep in range(epochs):
        perm = torch.randperm(P)
        for b in range(0, P, batch):
            idx = perm[b:b + batch]
            Sb = S_t[idx]
            holds = run_policy(policy, kind, Sb, K, sigma_ref, T_opt, dt)
            pnl = hedging_pnl(holds, Sb, K, premium, cost_kappa)
            loss = cvar_loss(pnl, w, alpha)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return policy, float(w.item())


# ----------------------------------------------------------------------------
# Correctness check: BS limit
# ----------------------------------------------------------------------------
def bs_delta_holdings(S, K, sigma_ref, T_opt, dt):
    P, n_grid = S.shape
    n_steps = n_grid - 1
    holds = torch.zeros(P, n_steps)
    for i in range(n_steps):
        tau = T_opt - i * dt
        _, delta = bs_call_price_delta(S[:, i], K, sigma_ref, tau)
        holds[:, i] = delta
    return holds


def eval_pnl_stats(pnl):
    pnl = pnl.detach().numpy()
    loss = -pnl
    var95 = np.quantile(loss, 0.95)
    cvar95 = loss[loss >= var95].mean()
    return {
        "mean_pnl": float(pnl.mean()),
        "std_pnl": float(pnl.std()),
        "cvar95_loss": float(cvar95),
        "q05_pnl": float(np.quantile(pnl, 0.05)),
    }


def run_correctness():
    t0 = time.time()
    # ---- BS-limit parameters ----
    H = 0.499
    T = 0.5
    lam2 = 1e-6
    nu2 = lam2 / (H * (1 - 2 * H))          # -> tiny, omega ~ 0
    sigma_ref = 0.20
    rho = -0.7
    mu = 0.0
    T_opt = 21 / 252                          # ~1-month horizon
    n_steps = 21                              # daily rehedge
    dt = T_opt / n_steps                      # = 1/252
    K = 1.0
    cost_kappa = 0.0                          # frictionless, to match BS delta
    n_train, n_test = 30000, 30000

    S_tr, sig_tr, var_omega = simulate_paths(n_train, n_steps, dt, H, T, nu2,
                                             sigma_ref, rho, mu, seed=1)
    S_te, sig_te, _ = simulate_paths(n_test, n_steps, dt, H, T, nu2,
                                     sigma_ref, rho, mu, seed=2)
    rel_vol_disp = float(np.std(sig_tr) / np.mean(sig_tr))

    premium, _ = bs_call_price_delta(torch.tensor(1.0), K, sigma_ref, T_opt)
    premium = float(premium.item())

    S_te_t = torch.from_numpy(S_te)

    # ---- benchmark: BS delta hedging on the same test paths ----
    bs_holds = bs_delta_holdings(S_te_t, K, sigma_ref, T_opt, dt)
    bs_pnl = hedging_pnl(bs_holds, S_te_t, K, premium, cost_kappa)
    bs_stats = eval_pnl_stats(bs_pnl)

    results = {
        "params": {
            "H": H, "T": T, "lambda2": lam2, "nu2": nu2,
            "var_omega": float(var_omega), "rel_vol_dispersion": rel_vol_disp,
            "sigma_ref": sigma_ref, "rho": rho, "T_opt": T_opt,
            "n_steps": n_steps, "dt": dt, "K": K, "cost_kappa": cost_kappa,
            "n_train": n_train, "n_test": n_test,
            "cvar_alpha": 0.95, "bs_premium": premium,
        },
        "bs_delta_benchmark": bs_stats,
    }

    for kind in ["ff", "gru"]:
        policy, w = train_policy(kind, S_tr, K, sigma_ref, T_opt, dt, premium,
                                 cost_kappa, epochs=60, seed=0)
        with torch.no_grad():
            dh_holds = run_policy(policy, kind, S_te_t, K, sigma_ref, T_opt, dt)
            dh_pnl = hedging_pnl(dh_holds, S_te_t, K, premium, cost_kappa)
        dh_stats = eval_pnl_stats(dh_pnl)

        a = dh_holds.numpy().ravel()
        b = bs_holds.numpy().ravel()
        corr = float(np.corrcoef(a, b)[0, 1])
        mae = float(np.mean(np.abs(a - b)))
        std_ratio = dh_stats["std_pnl"] / bs_stats["std_pnl"]
        cvar_ratio = dh_stats["cvar95_loss"] / bs_stats["cvar95_loss"]

        passed = (corr > 0.90) and (std_ratio < 1.5) and (cvar_ratio < 1.15)
        results[f"deep_hedge_{kind}"] = {
            **dh_stats,
            "holding_corr_vs_bs_delta": corr,
            "holding_mae_vs_bs_delta": mae,
            "std_ratio_vs_bs": std_ratio,
            "cvar95_ratio_vs_bs": cvar_ratio,
            "PASS": bool(passed),
        }

    results["wall_seconds"] = round(time.time() - t0, 1)
    return results


if __name__ == "__main__":
    res = run_correctness()
    out = "/home/user/finprobs/research/dlrl/track_a_v3_correctness_results.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print("\nSaved ->", out)
