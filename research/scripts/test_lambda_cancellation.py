"""
Test: does lambda^2 (intermittency) cancel out of the ratio
Cov(tau_short) / Cov(tau_VIX), while T changes it?

log S-fBM covariance (Wu-Muzy-Bacry 2022, ground truth from source paper):
  C(tau; H, T, nu2) = (nu2/2) * [T^(2H) - tau^(2H)]   for tau < T
  nu2 = lambda^2 / (H*(1-2H))
So nu2 is a pure multiplicative amplitude on the whole covariance function.
"""
import numpy as np

def cov(tau, H, T, nu2):
    tau = np.asarray(tau, dtype=float)
    out = np.where(tau < T, (nu2/2.0) * (T**(2*H) - tau**(2*H)), 0.0)
    return out

tau_short = 1/252        # ~1 trading day, "short-dated SPX" lag
tau_vix = 21/252         # ~1 month, "VIX" lag
H = 0.10
T = 0.5                  # half-year decorrelation scale (illustrative)

print("=== Claim 1: lambda^2 / nu2 cancels out of the short/VIX covariance ratio ===")
for lam2 in [0.01, 0.05, 0.10, 0.50, 2.0]:
    nu2 = lam2 / (H * (1 - 2*H))
    c_short = cov(tau_short, H, T, nu2)
    c_vix = cov(tau_vix, H, T, nu2)
    ratio = c_short / c_vix
    print(f"  lambda^2={lam2:6.3f}  nu2={nu2:8.4f}  Cov(short)={c_short:.6f}  Cov(VIX)={c_vix:.6f}  ratio={ratio:.6f}")

print("\n=== Claim 2: T changes the ratio; ratio is monotone decreasing in T ===")
lam2 = 0.1
nu2 = lam2 / (H * (1 - 2*H))
for T_test in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 20.0]:
    c_short = cov(tau_short, H, T_test, nu2)
    c_vix = cov(tau_vix, H, T_test, nu2)
    ratio = c_short / c_vix if c_vix != 0 else float('nan')
    print(f"  T={T_test:6.2f}  Cov(short)={c_short:.6f}  Cov(VIX)={c_vix:.6f}  ratio={ratio:.6f}")

print("\n=== Claim 3: as T -> infinity, ratio -> 1 (pure fBM limit, lever dies) ===")
for T_test in [1, 10, 100, 1000, 100000]:
    c_short = cov(tau_short, H, T_test, nu2)
    c_vix = cov(tau_vix, H, T_test, nu2)
    ratio = c_short / c_vix
    print(f"  T={T_test:8d}  ratio={ratio:.8f}")

print("\n=== Claim 4: as T -> tau_vix from above, ratio -> infinity (max differentiating power) ===")
for T_test in [tau_vix*1.001, tau_vix*1.01, tau_vix*1.1, tau_vix*2, tau_vix*10]:
    c_short = cov(tau_short, H, T_test, nu2)
    c_vix = cov(tau_vix, H, T_test, nu2)
    ratio = c_short / c_vix if c_vix > 1e-12 else float('inf')
    print(f"  T={T_test:.6f} (T/tau_vix={T_test/tau_vix:.3f})  Cov(VIX)={c_vix:.8f}  ratio={ratio:.4f}")
