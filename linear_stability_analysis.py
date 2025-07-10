# Have changed the code such that it inculdes linear stability analysis for both the non-trivial steady state solutions of the quadratic equations as branch 0 and branch 1. It will have turing patterns if in the plot somewhere for any K>0 the re(lambda)>0.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# ===== Parameters =====
f = 0.045
kappa = 0.055
delta = 0.4
lam = 0.98

# ===== Steady state solver =====
def steady_state(f, kappa, lam):
    def eqs(v):
        if v <= 0:
            return 1e6  # avoid division by zero or negative v
        u = (f + kappa) / (lam * v)
        return u * v**2 - f * (1 - u)
    # Try to find both branches
    v_star1 = fsolve(eqs, 0.2)[0]
    v_star2 = fsolve(eqs, 0.6)[0]
    u_star1 = (f + kappa) / (lam * v_star1)
    u_star2 = (f + kappa) / (lam * v_star2)
    return (u_star1, v_star1), (u_star2, v_star2)

# ===== Dispersion relation =====
def dispersion_relation(k_vals, delta, lam, f, kappa, branch=0):
    (u1, v1), (u2, v2) = steady_state(f, kappa, lam)
    if branch == 0:
        u_star, v_star = u1, v1
    else:
        u_star, v_star = u2, v2
    # Check for physical validity
    if np.iscomplex(u_star) or np.iscomplex(v_star) or u_star <= 0 or v_star <= 0:
        print(f"Branch {branch} not physical: u*={u_star}, v*={v_star}")
        return None
    print(f"Branch {branch}: u*={u_star:.4f}, v*={v_star:.4f}")
    a11 = -v_star**2 - f
    a12 = -2 * u_star * v_star
    a21 = lam * v_star**2
    a22 = 2 * lam * u_star * v_star - (f + kappa)
    lam_max = np.zeros_like(k_vals)
    for i, k in enumerate(k_vals):
        J = np.array([[a11 - k**2, a12],
                      [a21, a22 - delta * k**2]])
        eigs = np.linalg.eigvals(J)
        lam_max[i] = max(eigs.real)
    return lam_max

# ===== Plotting =====
k_vals = np.linspace(0, 2, 400)
lam_disp_0 = dispersion_relation(k_vals, delta, lam, f, kappa, branch=0)
lam_disp_1 = dispersion_relation(k_vals, delta, lam, f, kappa, branch=1)

plt.figure(figsize=(10, 6))
if lam_disp_0 is not None:
    plt.plot(k_vals, lam_disp_0, 'b-', lw=2, label='Branch 0')
    plt.fill_between(k_vals, 0, lam_disp_0, where=(lam_disp_0>0), color='blue', alpha=0.2)
if lam_disp_1 is not None:
    plt.plot(k_vals, lam_disp_1, 'g-', lw=2, label='Branch 1')
    plt.fill_between(k_vals, 0, lam_disp_1, where=(lam_disp_1>0), color='green', alpha=0.2)
plt.axhline(0, color='k', linestyle='--', alpha=0.7)
plt.xlabel("Wavenumber $k$", fontsize=12)
plt.ylabel("Re($\lambda_{max}$)", fontsize=12)
plt.title("Turing Instability at Non-Trivial Steady States\n(f=%.3f, κ=%.3f, δ=%.1f, λ=%.2f)" % (f, kappa, delta, lam), fontsize=14)
plt.legend()
plt.grid(alpha=0.2)
plt.show()