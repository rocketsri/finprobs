"""
Simulate the Zarhali et al. nested-factor mechanism directly:
  logvol_i(t) = beta * F(t) + X_i(t),  i = 1..N_s
F: smooth common factor (self-affine, H_F=0.15)
X_i: i.i.d. rough/idiosyncratic components (log-covariance "cascade-like", H->0)
Check whether the cross-sectional average's estimated H converges toward H_F
as N_s grows, and roughly where that happens relative to N_s.
"""
import numpy as np

rng = np.random.default_rng(7)
N = 1500          # path length
T_scale = 300.0
beta = 0.6

idx = np.arange(N)
lag = np.abs(idx[:, None] - idx[None, :]).astype(float)

def cov_power(tau, H, T, nu2):
    return np.where(tau < T, (nu2/2.0)*(T**(2*H) - tau**(2*H)), 0.0)

def cov_log(tau, lam2, T):
    val = lam2*np.log(np.maximum(T/np.maximum(tau,1e-9), 1e-9))
    return np.where(tau < T, np.maximum(val, 0.0), 0.0)

H_F = 0.15
Sigma_F = cov_power(lag, H_F, T_scale, 0.5) + 1e-8*np.eye(N)
Sigma_X = cov_log(lag, 0.08, T_scale) + 1e-8*np.eye(N)

L_F = np.linalg.cholesky(Sigma_F)
L_X = np.linalg.cholesky(Sigma_X)

def structure_H(path, lags, q=2):
    vals, used = [], []
    for tau in lags:
        d = path[tau:] - path[:-tau]
        m = np.mean(np.abs(d)**q)
        if m > 0:
            vals.append(m); used.append(tau)
    used = np.array(used); vals = np.array(vals)
    slope, _ = np.polyfit(np.log(used), np.log(vals), 1)
    return slope/q

lags = np.unique(np.round(np.geomspace(2, N//3, 20)).astype(int))

# single-name H (N_s=1, beta*F + X_i)
F = (L_F @ rng.standard_normal(N))
single_Hs = []
for _ in range(30):
    Xi = L_X @ rng.standard_normal(N)
    single_Hs.append(structure_H(beta*F + Xi, lags))
print(f"Single-name H (beta*F + X_i), N_s=1: mean={np.mean(single_Hs):.4f}  (target common-factor H_F={H_F})")

print(f"\n{'N_s':>6s} {'beta^4*N_s':>12s} {'H_avg_est':>10s}")
for N_s in [1, 2, 5, 10, 30, 100, 300, 1000]:
    reps = 15
    H_ests = []
    for _ in range(reps):
        Xs = L_X @ rng.standard_normal((N, N_s))
        idio_avg = Xs.mean(axis=1)
        agg = beta*F + idio_avg
        H_ests.append(structure_H(agg, lags))
    print(f"{N_s:6d} {beta**4*N_s:12.2f} {np.mean(H_ests):10.4f}")

print(f"\n(Common factor alone F: H = {structure_H(F, lags):.4f})")
print("(Idiosyncratic alone X_i: H ~", np.mean([structure_H(L_X @ rng.standard_normal(N), lags) for _ in range(20)]), ")")
