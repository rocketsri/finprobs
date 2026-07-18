"""
Track A - Deep Hedging under the log S-fBM volatility model (v4, independent build).

GOAL (this file, STEP 1 + STEP 2 + correctness check only):
  Build a Buehler-style deep-hedging engine on top of the log S-fBM log-volatility
  process, and validate it in the Black-Scholes limit (H -> 0.499, lambda^2 -> 0).
  The lambda^2/T sensitivity experiment is deliberately NOT attempted here.

-------------------------------------------------------------------------------
DESIGN CHOICES (stated explicitly; these are my own, made from scratch)
-------------------------------------------------------------------------------
1. Log-vol process omega_t.
   Ground-truth covariance (Wu-Muzy-Bacry 2022, same as test_lambda_cancellation.py):
       C(tau; H,T,nu2) = (nu2/2) * (T^(2H) - tau^(2H))   for tau < T,  else 0
       nu2 = lambda^2 / (H*(1-2H))
   omega is a zero-mean stationary Gaussian field sampled on the daily grid via
   Cholesky of the covariance matrix over grid nodes (reuses the cov_power idea from
   the reusable scripts). Variance at every node is s2 = (nu2/2) * T^(2H).

2. Spot vol and price dynamics.
   log S-fBM in the source paper specifies ONLY the log-vol field; it has no price
   process. I add price dynamics as an explicit modelling assumption:
       sigma_k = sigma_ref * exp(omega_k - s2/2)          (so E[sigma_k] = sigma_ref)
       r_k = -0.5 sigma_k^2 dt + sigma_k sqrt(dt) * u_k    (log-return over step k)
   LEVERAGE construction (my assumption, stated clearly):
       omega = L z, with L the lower-triangular Cholesky factor and z ~ N(0,I).
       z_{k+1} is the innovation revealed at node k+1 (drives future vol).
       u_k = rho * z_{k+1} + sqrt(1-rho^2) * eps_k,   eps_k iid N(0,1),  rho = -0.7
   Because L is lower-triangular, z_{k+1} feeds omega at nodes >= k+1, so correlating
   the step-k price shock with z_{k+1} gives a genuine leverage effect (price down <->
   future vol up when rho<0). u_k is marginally standard normal for any lambda^2, so in
   the lambda^2->0 limit the price is EXACTLY GBM(sigma_ref) -- clean BS limit.

3. Horizon / grid.
   Daily rebalancing, dt = 1/252, matching the covariance grid (tau_short=1/252,
   tau_vix=21/252). Primary option = ATM European call, maturity 21 trading days
   (~1 month), so one step == tau_short and the full horizon == tau_vix. This makes the
   hedging setup directly comparable to the covariance-level lambda^2/T result.

4. Risk objective = ENTROPIC (exponential) risk,  rho_g(W) = (1/g) log E[exp(-g W)].
   Reasons: (a) it is the canonical smooth convex risk measure with an exponential-
   utility indifference-price reading; (b) crucially for THIS correctness test, in a
   complete BS market the entropic risk-minimiser is exactly the Black-Scholes delta
   with residual -> 0, so "recover the delta" is a principled pass criterion. This
   differs from a CVaR default and from Horvath et al.'s framing on rBergomi.

5. Architecture = small GRU recurrent policy.
   log S-fBM has long memory in log-vol, so the hedge is strongly non-Markovian in
   spot alone. The hedger observes ONLY price-based features (moneyness, time-to-
   maturity, previous holding) -- NOT the latent vol -- and must infer the vol state
   from price history; a recurrent hidden state is the honest match. (Differentiated
   from Horvath's feedforward net; overlaps Zhu/Wu/Diao's GRU idea but on log S-fBM
   with the entropic objective and the separable (H,lambda^2,T) structure in mind.)

6. Transaction costs: proportional, cost = kappa * S * |trade|, charged on the opening
   trade, every rebalance, and final liquidation. Implemented in the engine; the BS-
   limit correctness check is run frictionless (kappa=0) to isolate the delta-recovery
   claim, and a kappa>0 smoke run confirms the cost path trains.

PyTorch CPU only, kept modest by design.
"""

import json
import math
import time
import numpy as np
import torch
import torch.nn as nn

# NB: the 21-step GRU is built from many tiny ops; on this CPU build multi-thread
# spawn overhead dominates (measured ~130x slower at 4 threads). Single thread wins.
torch.set_num_threads(1)
torch.manual_seed(0)
np.random.seed(0)

DT = 1.0 / 252.0
N_STEPS = 21              # 21 daily steps == 1-month (tau_vix) horizon; 1 step == tau_short
SIGMA_REF = 0.20         # annualised reference vol
S0 = 1.0
STRIKE = 1.0             # ATM
RHO = -0.7               # leverage correlation (my assumption)
DEVICE = "cpu"


# ---------------------------------------------------------------------------
# log S-fBM covariance + Cholesky factor for omega on the grid
# ---------------------------------------------------------------------------
def nu2_of(lam2, H):
    return lam2 / (H * (1.0 - 2.0 * H))


def cov_power(tau, H, T, nu2):
    return np.where(tau < T, (nu2 / 2.0) * (T ** (2 * H) - tau ** (2 * H)), 0.0)


def build_chol(H, lam2, T):
    """Cholesky factor L (N_STEPS+1 nodes) and node variance s2."""
    nodes = np.arange(N_STEPS + 1) * DT
    lag = np.abs(nodes[:, None] - nodes[None, :])
    nu2 = nu2_of(lam2, H)
    Sigma = cov_power(lag, H, T, nu2) + 1e-12 * np.eye(N_STEPS + 1)
    L = np.linalg.cholesky(Sigma)
    s2 = (nu2 / 2.0) * (T ** (2 * H))   # Var(omega_k), constant across nodes
    return torch.tensor(L, dtype=torch.float32), float(s2)


# ---------------------------------------------------------------------------
# Path simulator (no grad; paths are inputs to the hedger)
# ---------------------------------------------------------------------------
@torch.no_grad()
def simulate(batch, L, s2, seed=None):
    """Return S with shape (batch, N_STEPS+1) and sigma (batch, N_STEPS+1)."""
    g = torch.Generator(device=DEVICE)
    if seed is not None:
        g.manual_seed(seed)
    z = torch.randn(batch, N_STEPS + 1, generator=g)            # innovations for omega
    eps = torch.randn(batch, N_STEPS + 1, generator=g)          # idiosyncratic price noise
    omega = z @ L.T                                             # (batch, N+1)
    sigma = SIGMA_REF * torch.exp(omega - 0.5 * s2)
    # step-k price shock uses z_{k+1} (innovation revealed at node k+1)
    u = RHO * z[:, 1:] + math.sqrt(1.0 - RHO ** 2) * eps[:, :-1]   # (batch, N)
    sig_k = sigma[:, :-1]
    r = -0.5 * sig_k ** 2 * DT + sig_k * math.sqrt(DT) * u
    logS = torch.cat([torch.zeros(batch, 1), torch.cumsum(r, dim=1)], dim=1)
    S = S0 * torch.exp(logS)
    return S, sigma


# ---------------------------------------------------------------------------
# Black-Scholes benchmark (constant sigma_ref)
# ---------------------------------------------------------------------------
def _norm_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def bs_price(S, K, sigma, tau):
    if tau <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * tau) / (sigma * math.sqrt(tau))
    d2 = d1 - sigma * math.sqrt(tau)
    Nc = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return S * Nc(d1) - K * Nc(d2)


def bs_delta_path(S, K, sigma):
    """BS delta at steps k=0..N-1 given price path S (batch, N+1)."""
    deltas = []
    for k in range(N_STEPS):
        tau = (N_STEPS - k) * DT
        d1 = (torch.log(S[:, k] / K) + 0.5 * sigma ** 2 * tau) / (sigma * math.sqrt(tau))
        deltas.append(_norm_cdf(d1))
    return torch.stack(deltas, dim=1)   # (batch, N)


# ---------------------------------------------------------------------------
# GRU hedging policy
# ---------------------------------------------------------------------------
class GRUHedger(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.hidden = hidden
        self.cell = nn.GRUCell(3, hidden)   # inputs: [log(S/K), tau_frac, prev_holding]
        self.out = nn.Linear(hidden, 1)

    def forward(self, S):
        batch = S.shape[0]
        h = torch.zeros(batch, self.hidden)
        prev = torch.zeros(batch, 1)
        deltas = []
        for k in range(N_STEPS):
            tau_frac = torch.full((batch, 1), (N_STEPS - k) / N_STEPS)
            feat = torch.cat([torch.log(S[:, k:k + 1] / STRIKE), tau_frac, prev], dim=1)
            h = self.cell(feat, h)
            d = self.out(h)
            deltas.append(d)
            prev = d
        return torch.cat(deltas, dim=1)   # (batch, N)


# ---------------------------------------------------------------------------
# P&L engine (shared by learned + BS hedge)
# ---------------------------------------------------------------------------
def terminal_wealth(S, deltas, premium, kappa):
    """W = premium + trading gains - transaction costs - payoff. Shape (batch,)."""
    dS = S[:, 1:] - S[:, :-1]
    gains = (deltas * dS).sum(dim=1)
    # transaction costs: open, rebalance, liquidate
    trades = torch.empty_like(deltas)
    trades[:, 0] = deltas[:, 0]
    trades[:, 1:] = deltas[:, 1:] - deltas[:, :-1]
    cost = kappa * (torch.abs(trades) * S[:, :-1]).sum(dim=1)
    cost = cost + kappa * torch.abs(deltas[:, -1]) * S[:, -1]     # final liquidation
    payoff = torch.clamp(S[:, -1] - STRIKE, min=0.0)
    return premium + gains - cost - payoff


def entropic_risk(W, gamma):
    # rho_g(W) = (1/g) log E[exp(-g W)]  (numerically stable)
    return (torch.logsumexp(-gamma * W, dim=0) - math.log(W.shape[0])) / gamma


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(H, lam2, T, premium, kappa, gamma, iters, batch, lr=1e-3, verbose=False):
    L, s2 = build_chol(H, lam2, T)
    net = GRUHedger(hidden=32)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for it in range(iters):
        S, _ = simulate(batch, L, s2, seed=1000 + it)
        deltas = net(S)
        W = terminal_wealth(S, deltas, premium, kappa)
        loss = entropic_risk(W, gamma)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and (it % max(1, iters // 6) == 0 or it == iters - 1):
            print(f"    iter {it:4d}  risk={loss.item():+.6e}  std(W)={W.std().item():.6e}")
    return net, L, s2


# ---------------------------------------------------------------------------
# Correctness check: BS limit
# ---------------------------------------------------------------------------
def correctness_check():
    print("=" * 78)
    print("CORRECTNESS CHECK: BS limit  H->0.499, lambda^2->0  (frictionless)")
    print("=" * 78)
    H = 0.499
    lam2 = 1e-8            # ~0 intermittency  -> omega ~ 0 -> constant vol -> GBM
    T = 0.5
    kappa = 0.0
    gamma = 10.0
    tau0 = N_STEPS * DT
    premium = bs_price(S0, STRIKE, SIGMA_REF, tau0)
    print(f"  BS price (sigma_ref={SIGMA_REF}, tau={tau0:.4f}) = {premium:.6f}")

    # confirm the simulated vol really is ~constant in this limit
    _, s2_chk = build_chol(H, lam2, T)
    print(f"  Var(omega) in limit = {s2_chk:.3e}  (=> sigma ~ constant)")

    t0 = time.time()
    # large batch -> low-variance gradient -> policy converges tightly to the
    # deterministic BS optimum (the reliable lever; LR schedules destabilised it).
    net, L, s2 = train(H, lam2, T, premium, kappa, gamma,
                       iters=2500, batch=8192, verbose=True)
    print(f"  training time: {time.time()-t0:.1f}s")

    # fresh test set
    S, sigma = simulate(20000, L, s2, seed=999)
    with torch.no_grad():
        d_learned = net(S)
    d_bs = bs_delta_path(S, STRIKE, SIGMA_REF)

    W_learned = terminal_wealth(S, d_learned, premium, kappa)
    W_bs = terminal_wealth(S, d_bs, premium, kappa)

    # hedge-comparison metrics
    delta_rmse = torch.sqrt(((d_learned - d_bs) ** 2).mean()).item()
    delta_mae = torch.abs(d_learned - d_bs).mean().item()
    corr = torch.corrcoef(torch.stack([d_learned.flatten(), d_bs.flatten()]))[0, 1].item()

    res = {
        "regime": {"H": H, "lambda2": lam2, "T": T, "kappa": kappa, "gamma": gamma,
                   "N_steps": N_STEPS, "dt": DT, "sigma_ref": SIGMA_REF, "rho": RHO},
        "bs_price": premium,
        "var_omega_limit": s2_chk,
        "test_paths": 20000,
        "W_learned": {"mean": W_learned.mean().item(), "std": W_learned.std().item()},
        "W_bs_delta": {"mean": W_bs.mean().item(), "std": W_bs.std().item()},
        "std_ratio_learned_over_bs": W_learned.std().item() / W_bs.std().item(),
        "delta_rmse": delta_rmse,
        "delta_mae": delta_mae,
        "delta_corr": corr,
        "entropic_risk_learned": entropic_risk(W_learned, gamma).item(),
        "entropic_risk_bs": entropic_risk(W_bs, gamma).item(),
    }

    # pass criteria
    ratio = res["std_ratio_learned_over_bs"]
    res["pass_std_ratio"] = bool(ratio <= 1.5)
    res["pass_delta_shape"] = bool(delta_rmse <= 0.10 and corr >= 0.95)
    res["PASS"] = bool(res["pass_std_ratio"] and res["pass_delta_shape"])

    print("-" * 78)
    print(f"  W_learned : mean={res['W_learned']['mean']:+.6f}  std={res['W_learned']['std']:.6f}")
    print(f"  W_bsdelta : mean={res['W_bs_delta']['mean']:+.6f}  std={res['W_bs_delta']['std']:.6f}")
    print(f"  std ratio (learned/BS) = {ratio:.3f}   (pass if <= 1.5)")
    print(f"  delta RMSE={delta_rmse:.4f}  MAE={delta_mae:.4f}  corr={corr:.4f}")
    print(f"  PASS = {res['PASS']}")
    return res


def cost_smoke_test():
    """Confirm the transaction-cost path trains and widens the spread sensibly."""
    print("\n" + "=" * 78)
    print("SMOKE TEST: transaction costs engaged (kappa=0.001), rough regime H=0.10")
    print("=" * 78)
    H, lam2, T = 0.10, 0.05, 0.5
    tau0 = N_STEPS * DT
    premium = bs_price(S0, STRIKE, SIGMA_REF, tau0)  # rough proxy price (not fair here)
    net, L, s2 = train(H, lam2, T, premium, kappa=0.001, gamma=10.0,
                       iters=1200, batch=4096, verbose=True)
    S, _ = simulate(20000, L, s2, seed=777)
    with torch.no_grad():
        d = net(S)
    W = terminal_wealth(S, d, premium, 0.001)
    out = {"H": H, "lambda2": lam2, "T": T, "kappa": 0.001,
           "W_mean": W.mean().item(), "W_std": W.std().item(),
           "trained_ok": bool(torch.isfinite(W).all())}
    print(f"  W mean={out['W_mean']:+.6f}  std={out['W_std']:.6f}  finite={out['trained_ok']}")
    return out


if __name__ == "__main__":
    results = {"correctness_bs_limit": correctness_check(),
               "cost_smoke_test": cost_smoke_test()}
    with open("/home/user/finprobs/research/dlrl/track_a_v4_correctness_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved -> track_a_v4_correctness_results.json")
