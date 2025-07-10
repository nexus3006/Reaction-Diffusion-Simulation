import numpy as np
import matplotlib.pyplot as plt
import math

# ========== Parameter Scan Setup ==========
N = 200          # Grid size (balanced for speed/quality)
steps = 20000    # Sufficient for pattern formation
dt = 0.1         # Time step

# Parameter ranges (4 values per parameter for demonstration)
params = {
    'delta': [0.3,0.4,0.5,0.6],    # Dv/Du ratio
    'lam': [0.9,1,1.1,1.2],      # Reaction strength
    'f': [0.035,0.045,0.055,0.065],    # Feed rate
    'kappa': [0.055,0.058,0.061,0.064] # Kill rate
}

# Generate parameter combinations using meshgrid
grid = np.meshgrid(*params.values())
param_combinations = np.vstack([g.ravel() for g in grid]).T

# ========== Simulation Functions ==========
def initialize():
    """Initialize system with smaller perturbation"""
    u = np.ones((N, N))
    v = np.zeros((N, N))
    center = N//2
    r = 10  # Optimal perturbation size
    u[center-r:center+r, center-r:center+r] = 0.50
    v[center-r:center+r, center-r:center+r] = 0.25
    u += 0.01*np.random.normal(size=(N,N))
    v += 0.01*np.random.normal(size=(N,N))
    return u, v

def laplacian(Z):
    """Periodic boundary laplacian"""
    return (np.roll(Z, 1, 0) + np.roll(Z, -1, 0) +
            np.roll(Z, 1, 1) + np.roll(Z, -1, 1) - 4*Z)

def simulate(delta, lam, f, kappa):
    """Run simulation for given parameters"""
    u, v = initialize()
    for _ in range(steps):
        Lu = laplacian(u)
        Lv = laplacian(v)
        uvv = u*v*v
        u += dt * (Lu - uvv + f*(1 - u))
        v += dt * (delta*Lv + lam*uvv - (f + kappa)*v)
    return u, v

# ========== Run Parameter Scan ==========
results = []
for combo in param_combinations:
    delta, lam, f, kappa = combo
    print(f"Simulating: δ={delta:.2f}, λ={lam:.2f}, f={f:.3f}, κ={kappa:.3f}")
    u_final, v_final = simulate(delta, lam, f, kappa)
    results.append((delta, lam, f, kappa, u_final, v_final))

# ========== Visualization ==========
# Total number of results
num_results = len(results)  # Should be 256

# Choose number of columns (e.g., 4), calculate rows accordingly
cols = 4
rows = math.ceil(num_results / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
axes = axes.flatten()

for idx, (delta, lam, f, kappa, u, v) in enumerate(results):
    ax = axes[idx]
    im = ax.imshow(u, cmap='viridis', vmin=0, vmax=1)
    ax.set_title(f"δ={delta:.1f} λ={lam:.1f}\nf={f:.3f} κ={kappa:.3f}", fontsize=6)

# Hide unused axes if any
for ax in axes[len(results):]:
    ax.axis('off')

plt.tight_layout()
plt.show()