"""
SiO Thermal Properties: Tabular Data vs OpenFOAM Polynomial Fit
================================================================
Plots thermal conductivity (kappa) and heat capacity (Cp) comparing
raw tabular data against the OpenFOAM polynomial coefficients.

OpenFOAM polynomials are evaluated as functions of T in Kelvin.
  kappa(T) = 20.50 - 7.330e-3 * T                          [W/(m.K)]
  Cp(T)    = -109.0 + 3.545*T - 2.840e-3*T^2 + 9.494e-7*T^3  [J/(kg.K)]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Tabular data (T in °C) ────────────────────────────────────────────────────
T_C_tab = np.array([24, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200])
kappa_tab = np.array([17.89, 17.73, 16.82, 15.86, 15.25, 14.18,
                      13.64, 13.02, 12.51, 12.22, 11.41, 10.85, 9.59])
Cp_tab    = np.array([666, 812, 937, 1022, 1084, 1131,
                      1167, 1198, 1223, 1245, 1263, 1279, 1294])
T_K_tab   = T_C_tab + 273.15

# ── OpenFOAM polynomial coefficients ─────────────────────────────────────────
# kappaCoeffs<8>  ( 20.50  -7.330e-3  0  0  0  0  0  0 )
kappa_coeffs = [20.50, -7.330e-3, 0, 0, 0, 0, 0, 0]

# CpCoeffs<8>  ( -109.0  3.545  -2.840e-3  9.494e-7  0  0  0  0 )
Cp_coeffs    = [-109.0, 3.545, -2.840e-3, 9.494e-7, 0, 0, 0, 0]

def eval_poly(T, coeffs):
    """Evaluate OpenFOAM-style polynomial: sum(coeffs[i] * T^i)"""
    return sum(c * T**i for i, c in enumerate(coeffs))

# ── Smooth curve for polynomial ───────────────────────────────────────────────
T_K_smooth = np.linspace(T_K_tab.min(), T_K_tab.max(), 300)
T_C_smooth = T_K_smooth - 273.15

kappa_poly = eval_poly(T_K_smooth, kappa_coeffs)
Cp_poly    = eval_poly(T_K_smooth, Cp_coeffs)

# ── Relative error at tabular points ─────────────────────────────────────────
kappa_poly_tab = eval_poly(T_K_tab, kappa_coeffs)
Cp_poly_tab    = eval_poly(T_K_tab, Cp_coeffs)

err_kappa = np.abs((kappa_poly_tab - kappa_tab) / kappa_tab) * 100
err_Cp    = np.abs((Cp_poly_tab    - Cp_tab)    / Cp_tab)    * 100

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 10))
fig.suptitle("SiO Thermal Properties: Tabular vs OpenFOAM Polynomial",
             fontsize=14, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

BLUE  = "#2271B3"
CORAL = "#C0392B"
GRAY  = "#555555"

# ── Panel 1: kappa ────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(T_C_smooth, kappa_poly, color=CORAL, linewidth=2,
         linestyle="--", label="OpenFOAM polynomial")
ax1.scatter(T_C_tab, kappa_tab, color=BLUE, s=60, zorder=5,
            label="Tabular data")
ax1.set_xlabel("Temperature (°C)")
ax1.set_ylabel("κ  (W/m·K)")
ax1.set_title("Thermal conductivity")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# ── Panel 2: Cp ───────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(T_C_smooth, Cp_poly, color=CORAL, linewidth=2,
         linestyle="--", label="OpenFOAM polynomial")
ax2.scatter(T_C_tab, Cp_tab, color=BLUE, s=60, zorder=5,
            label="Tabular data")
ax2.set_xlabel("Temperature (°C)")
ax2.set_ylabel("Cp  (J/kg·K)")
ax2.set_title("Heat capacity")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# ── Panel 3: relative error ───────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, :])
x = np.arange(len(T_C_tab))
width = 0.35
bars1 = ax3.bar(x - width/2, err_kappa, width, label="κ error",
                color=BLUE, alpha=0.85)
bars2 = ax3.bar(x + width/2, err_Cp,    width, label="Cp error",
                color=CORAL, alpha=0.85)

ax3.set_xticks(x)
ax3.set_xticklabels([f"{t}°C" for t in T_C_tab], rotation=45, ha="right")
ax3.set_ylabel("Relative error (%)")
ax3.set_title("Polynomial vs tabular: relative error at each data point")
ax3.legend(fontsize=9)
ax3.grid(True, axis="y", alpha=0.3)

# Annotate bar values
for bar in bars1:
    h = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, h + 0.05,
             f"{h:.1f}%", ha="center", va="bottom", fontsize=7, color=GRAY)
for bar in bars2:
    h = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, h + 0.05,
             f"{h:.1f}%", ha="center", va="bottom", fontsize=7, color=GRAY)

plt.savefig("sio_thermal_properties.png", dpi=150, bbox_inches="tight")
print("Saved: sio_thermal_properties.png")
plt.show()
