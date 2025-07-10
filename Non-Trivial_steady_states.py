#This is the code for turiving patterns at non-trivial steady states. The steady state solver calculates the u,v by solving the quadratic equation at a fixed value of parameters which is our steady state than that steady state is initialised everywhere in the grid. As on line 35 we are taking only the branch with the minimum u out of the two branches (i.e. two solutions of quadratic equations). As seen on the linear stability analysis the branch with larger u is showing more unstability so do change on line 35 from min to max to get more patterns.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from itertools import product

# ========== Parameter Scan Setup ==========
N = 200
steps = 20000
dt = 0.1

params = {
    'delta': [0.3,0.4,0.5,0.6],
    'lam': [0.98,0.99,1,1.01],
    'f': [0.035, 0.045, 0.055, 0.065],
    'kappa': [0.055, 0.058, 0.061, 0.064]
}

# ========== Steady State Solver ==========
def steady_state(f, kappa, lam):
    def eqs(v):
        if v <= 0:
            return 1e6
        u = (f + kappa) / (lam * v)
        return u * v**2 - f * (1 - u)
    # Try both branches
    v_branches = [fsolve(eqs, guess)[0] for guess in [0.2, 0.6]]
    u_branches = [(f + kappa) / (lam * v) if v > 0 else np.nan for v in v_branches]
    # Only keep real, positive branches
    valid = [(u, v) for u, v in zip(u_branches, v_branches) if np.isreal(u) and np.isreal(v) and u > 0 and v > 0]
    if not valid:
        return None
    # Return the branch with smaller u (typical for Turing patterns)
    return min(valid, key=lambda x: x[0])

# ========== Initialization at Non-Trivial Steady State ==========
def initialize(u_star, v_star):
    u = np.full((N, N), u_star)
    v = np.full((N, N), v_star)
    u += 0.0001 * np.random.normal(size=(N, N))
    v += 0.0001* np.random.normal(size=(N, N))
    return u, v

# ========== Laplacian ==========
def laplacian(Z):
    return (np.roll(Z, 1, 0) + np.roll(Z, -1, 0) +
            np.roll(Z, 1, 1) + np.roll(Z, -1, 1) - 4 * Z)

# ========== Simulation ==========
def simulate(delta, lam, f, kappa):
    steady = steady_state(f, kappa, lam)
    if steady is None:
        print(f"No valid non-trivial steady state for f={f}, kappa={kappa}, lam={lam}")
        return None, None
    u_star, v_star = steady
    u, v = initialize(u_star, v_star)
    for _ in range(steps):
        Lu = laplacian(u)
        Lv = laplacian(v)
        uvv = u * v * v
        u += dt * (Lu - uvv + f * (1 - u))
        v += dt * (delta * Lv + lam * uvv - (f + kappa) * v)
    return u, v

# ========== Run Simulations ==========
results = []
for delta, lam, f, kappa in product(params['delta'], params['lam'], params['f'], params['kappa']):
    print(f"Simulating: delta={delta}, lam={lam}, f={f}, kappa={kappa}")
    u_final, v_final = simulate(delta, lam, f, kappa)
    results.append((delta, lam, f, kappa, u_final, v_final))

# ========== Visualization in 2D Grid ==========
# Axis parameters
f_vals = np.array(params['f'])
kappa_vals = np.array(params['kappa'])
# Loop over fixed delta and lam to generate separate grids
for delta in params['delta']:
    for lam in params['lam']:
        fig, axes = plt.subplots(len(kappa_vals), len(f_vals),
                                 figsize=(len(f_vals)*2, len(kappa_vals)*2),
                                 squeeze=False)

        for result in results:
            d, l, f, kappa, u, v = result
            if np.isclose(d, delta) and np.isclose(l, lam):
                i = np.where(np.isclose(kappa_vals, kappa))[0][0]
                j = np.where(np.isclose(f_vals, f))[0][0]
                ax = axes[i, j]
                ax.imshow(u, cmap='viridis', vmin=0, vmax=1)
                ax.set_title(f"f={f:.3f}\nκ={kappa:.3f}", fontsize=6)
                ax.axis('off')

        fig.suptitle(f"δ={delta}, λ={lam}", fontsize=10)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Leave space for suptitle
        plt.show()