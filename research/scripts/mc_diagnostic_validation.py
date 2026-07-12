"""
Monte Carlo validation of the "repaired" Direction-1 diagnostic:
nested boundary test of C_P(tau; H,T) = (nu2/2)[T^2H - tau^2H]  (self-affine, H>0)
                     vs C_L(tau; T)   = lambda^2 * log(T/tau)    (genuine cascade, H->0 limit)

We simulate stationary Gaussian log-vol paths under each ground truth via direct
Cholesky factorization of the covariance matrix, then check whether a simple
AIC-based nested model-selection correctly recovers the true generating model
from the *estimated* autocovariance function alone.
"""
import numpy as np
from scipy.optimize import curve_fit

rng = np.random.default_rng(12345)

N = 1500     # path length
T_scale = 300.0  # decorrelation scale in same units as lag index
dt = 1.0

def cov_power(tau, H, T, nu2):
    tau = np.asarray(tau, dtype=float)
    return np.where(tau < T, (nu2/2.0)*(T**(2*H) - tau**(2*H)), 0.0)

def cov_log(tau, lam2, T):
    tau = np.asarray(tau, dtype=float)
    val = lam2 * np.log(np.maximum(T/np.maximum(tau,1e-9), 1e-9))
    return np.where(tau < T, np.maximum(val, 0.0), 0.0)

idx = np.arange(N)
lag_matrix = np.abs(idx[:, None] - idx[None, :]).astype(float) * dt

# Ground-truth 1: rough/self-affine (H=0.10), calibrated so total variance ~ matches ground truth 2
H_true = 0.10
nu2_true = 0.30
Sigma_P = cov_power(lag_matrix, H_true, T_scale, nu2_true)
Sigma_P += 1e-8*np.eye(N)  # jitter for numerical PSD

# Ground-truth 2: genuine cascade / log-covariance (H=0 limit)
lam2_true = 0.05
Sigma_L = cov_log(lag_matrix, lam2_true, T_scale)
Sigma_L += 1e-8*np.eye(N)

print("Cholesky factorizing Sigma_P ...")
L_P = np.linalg.cholesky(Sigma_P)
print("Cholesky factorizing Sigma_L ...")
L_L = np.linalg.cholesky(Sigma_L)

M = 150  # number of Monte Carlo replicates per scenario
lags_fit = np.unique(np.round(np.geomspace(2, N//3, 25)).astype(int))

def empirical_acf(path, lags):
    path = path - path.mean()
    var0 = np.mean(path**2)
    out = []
    for tau in lags:
        c = np.mean(path[:-tau]*path[tau:])
        out.append(c)
    return var0, np.array(out)

def fit_power(lags, C_hat):
    # C(tau) = A - B*tau^(2H), fit A,B,H via nonlinear least squares
    def f(tau, A, B, Hh):
        Hh = np.clip(Hh, 1e-4, 0.49)
        return A - B*np.abs(tau)**(2*Hh)
    try:
        p0 = [C_hat[0], (C_hat[0]-C_hat[-1])/ (lags[-1]**0.2), 0.1]
        popt, _ = curve_fit(f, lags, C_hat, p0=p0, maxfev=20000,
                             bounds=([-np.inf,-np.inf,1e-4],[np.inf,np.inf,0.49]))
        resid = C_hat - f(lags, *popt)
        sse = np.sum(resid**2)
        k = 3
        return sse, k, popt
    except Exception:
        return np.inf, 3, None

def fit_log(lags, C_hat):
    # C(tau) = A - B*log(tau), fit A,B via linear least squares
    X = np.vstack([np.ones_like(lags, dtype=float), np.log(lags.astype(float))]).T
    coef, *_ = np.linalg.lstsq(X, C_hat, rcond=None)
    A, negB = coef[0], coef[1]
    pred = X @ coef
    resid = C_hat - pred
    sse = np.sum(resid**2)
    k = 2
    return sse, k, (A, -negB)

def aic(sse, k, n):
    # Gaussian AIC using RSS
    if sse <= 0:
        sse = 1e-12
    return n*np.log(sse/n) + 2*k

def run_scenario(L, name, n_lags_used):
    correct = 0
    for m in range(M):
        z = rng.standard_normal(N)
        path = L @ z
        var0, C_hat = empirical_acf(path, lags_fit)
        sse_p, k_p, popt_p = fit_power(lags_fit, C_hat)
        sse_l, k_l, popt_l = fit_log(lags_fit, C_hat)
        aic_p = aic(sse_p, k_p, n_lags_used)
        aic_l = aic(sse_l, k_l, n_lags_used)
        chosen = "power" if aic_p < aic_l else "log"
        if (name == "power_truth" and chosen == "power") or (name == "log_truth" and chosen == "log"):
            correct += 1
    return correct, M

n_lags_used = len(lags_fit)
print(f"\nRunning {M} replicates under TRUE model = power-law (H={H_true}) ...")
c1, m1 = run_scenario(L_P, "power_truth", n_lags_used)
print(f"  Correctly selected 'power' model: {c1}/{m1} = {c1/m1:.1%}")

print(f"\nRunning {M} replicates under TRUE model = log-covariance (genuine cascade) ...")
c2, m2 = run_scenario(L_L, "log_truth", n_lags_used)
print(f"  Correctly selected 'log' model: {c2}/{m2} = {c2/m2:.1%}")

print("\n=== Summary (Monte Carlo validation of repaired Direction-1 diagnostic) ===")
print(f"Power-law ground truth (H={H_true}):  correct classification rate = {c1/m1:.1%}")
print(f"Log-covariance ground truth (genuine cascade): correct classification rate = {c2/m2:.1%}")
