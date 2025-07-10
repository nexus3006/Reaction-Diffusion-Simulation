#This is the code for trivial steady state i.e. we have initialised the complete grid with u,v = (1,0) than given a perturbation of 0.0001. We should not observe any patterns by linear stability analysis it is always stable no  matter the parameters.

import numpy as np
import matplotlib.pyplot as plt
import math

# ========== Parameter Scan Setup ==========
N = 200
steps = 20000
dt = 0.1

params = {
    'delta': [0.3,0.4, 0.5, 0.6],
    'lam': [0.98,0.99, 1, 1.01],
    'f': [0.035, 0.045, 0.055, 0.065],
    'kappa': [0.055, 0.058, 0.061, 0.064]
}

# Generate parameter combinations
grid = np.meshgrid(*params.values())
param_combinations = np.vstack([g.ravel() for g in grid]).T

# ========== Simulation Functions ==========
def initialize():
    u = np.ones((N, N))
    v = np.zeros((N, N))
    #center = N // 2
    #r = 10
    #u[center-r:center+r, center-r:center+r] = 0.0001
    #v[center-r:center+r, center-r:center+r] = 0.005
    u += 0.0001 * np.random.normal(size=(N, N))
    v += 0.0001* np.random.normal(size=(N, N))
    return u, v

def laplacian(Z):
    return (np.roll(Z, 1, 0) + np.roll(Z, -1, 0) +
            np.roll(Z, 1, 1) + np.roll(Z, -1, 1) - 4 * Z)

def simulate(delta, lam, f, kappa):
    u, v = initialize()
    for _ in range(steps):
        Lu = laplacian(u)
        Lv = laplacian(v)
        uvv = u * v * v
        u += dt * (Lu - uvv + f * (1 - u))
        v += dt * (delta * Lv + lam * uvv - (f + kappa) * v)
    return u, v
    
# ========== Run Simulations ==========
results = []
for combo in param_combinations:
    delta, lam, f, kappa = combo
    print(f"Simulating: δ={delta:.2f}, λ={lam:.2f}, f={f:.3f}, κ={kappa:.3f}")
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