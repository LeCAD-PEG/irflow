import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Tabular data from your table
# -----------------------------
T = np.array([24, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200], dtype=float)

Cp_tab = np.array([666, 812, 937, 1022, 1084, 1131, 1167, 1198, 1223, 1245, 1263, 1279, 1294], dtype=float)
k_tab  = np.array([17.89, 17.73, 16.82, 15.86, 15.25, 14.18, 13.64, 13.02, 12.51, 12.22, 11.41, 10.85, 9.59], dtype=float)

# ---------------------------------------------------------
# Polynomial coefficients (OpenFOAM polynomialSolid format)
# Cp(T) = a0 + a1*T + a2*T^2 + a3*T^3
# k(T)  = b0 + b1*T + b2*T^2 + b3*T^3
# ---------------------------------------------------------

Cp_coeffs = np.array([6.55e+02, 5.55e-01, 1.20e-04, -1.05e-07])
k_coeffs  = np.array([1.82e+01, -7.40e-03, 1.20e-05, -8.50e-09])

def poly_eval(coeffs, T):
    """Evaluate polynomial with OpenFOAM ordering."""
    return coeffs[0] + coeffs[1]*T + coeffs[2]*T**2 + coeffs[3]*T**3

Cp_poly = poly_eval(Cp_coeffs, T)
k_poly  = poly_eval(k_coeffs, T)

# -----------------------------
# Plot Cp
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(T, Cp_tab, 'o', label='Tabular Cp')
plt.plot(T, Cp_poly, '-', label='Polynomial Cp')
plt.xlabel('Temperature [°C]')
plt.ylabel('Cp [J/kg/K]')
plt.title('SiO: Cp(T) — Tabular vs Polynomial Fit')
plt.grid(True)
plt.legend()

# -----------------------------
# Plot k
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(T, k_tab, 'o', label='Tabular k')
plt.plot(T, k_poly, '-', label='Polynomial k')
plt.xlabel('Temperature [°C]')
plt.ylabel('k [W/m/K]')
plt.title('SiO: k(T) — Tabular vs Polynomial Fit')
plt.grid(True)
plt.legend()

plt.show()
