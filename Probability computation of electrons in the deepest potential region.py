import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

# === Constants and Setup ===
a0 = 1 # Bohr radius
r = np.linspace(0, 5, 1000) # Radial coordinate in units of a0

# Effective nuclear charges from 2a
zeff = {
        '1s': 10.7,
        '2s': 6.85,
        '2p': 6.85,
        '3s': 2.2
}

# === Function to compute wavefunctions, probability densities, and penetration ===
def compute_penetration(label, Z_eff):
    # Step 1: Scaled coordinate rho
    rho = Z_eff * r / a0
    
    # Step 2: Orbital-specific radial wavefunction
    if label == '1s':
        psi_r = 2 * np.exp(-rho) / np.sqrt(a0**3)
    elif label == '2s':
        psi_r = (1 - rho / 2) * np.exp(-rho / 2) / np.sqrt(2 * a0**3)
    elif label == '2p':
        psi_r = rho * np.exp(-rho / 2) / np.sqrt(24 * a0**3)
    elif label == '3s':
        psi_r = (1 - 2/3 * rho + 2/27 * rho**2) * np.exp(-rho / 3) * 2 / np.sqrt(27 * a0**3)
    else:
        raise ValueError("Unknown orbital")

# Step 3: Radial probability density P(r)
Pr = r**2 * np.abs(psi_r)**2

# Step 4: Normalize P(r) so total area = 1
norm = simps(Pr, r)
Pr /= norm

# Step 5: Integrate from r=0 to a0
mask = r < a0
Pr_inside = Pr[mask]
r_inside = r[mask]
penetration_prob = simps(Pr_inside, r_inside)

print(f"--- {label} Orbital ---")
print(f"Z_eff = {Z_eff}")
print(f"Normalization factor = {norm:.5f}")
print(f"Penetration probability (r < a0): {penetration_prob:.5f}\n")
    return r, Pr, penetration_prob

# === Run for all orbitals ===
results = {}
for orb in zeff:
    r_vals, Pr_vals, P_inside = compute_penetration(orb, zeff[orb])
    results[orb] = {
        "r": r_vals,
        "Pr": Pr_vals,
        "P_inside": P_inside
}

# === Optional Plotting: Probability Distributions with r < a0 shaded ===
plt.figure(figsize=(8, 6))
for orb in results:
    plt.plot(results[orb]["r"], results[orb]["Pr"], label=f"{orb} (P<1={results[orb]['P_inside']:.2f

plt.axvspan(0, a0, color='gray', alpha=0.2, label="r < a₀ region")
plt.xlabel("r / a₀")
plt.ylabel("Normalized $r^2 |\psi(r)|^2$")
plt.title("Radial Probability Distributions with Penetration Region")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
