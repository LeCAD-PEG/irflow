import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Tabular data ───────────────────────────────────────────────────────────
T_C = np.array([24, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200])
T_K = T_C + 273.15

kappa_tab = np.array([17.89, 17.73, 16.82, 15.86, 15.25, 14.18,
                      13.64, 13.02, 12.51, 12.22, 11.41, 10.85,  9.59])
Cp_tab    = np.array([  666,   812,   937,  1022,  1084,  1131,
                       1167,  1198,  1223,  1245,  1263,  1279,  1294])

# ── Old (original) polynomial coefficients ────────────────────────────────
def kappa_old(T):
    return 20.50 + (-7.330e-3)*T

def Cp_old(T):
    return -109.0 + 3.545*T + (-2.840e-3)*T**2 + 9.494e-7*T**3

# ── New (tighter) polynomial coefficients — numpy least-squares fit ────────
# kappa: degree-3  →  kappaCoeffs<8> (a0..a3 0 0 0 0)
kappa_c = np.array([2.17787846e+01, -1.42273123e-02,  8.27720276e-06, -2.79531841e-09])
def kappa_new(T):
    return kappa_c[0] + kappa_c[1]*T + kappa_c[2]*T**2 + kappa_c[3]*T**3

# Cp: degree-5  →  CpCoeffs<8> (a0..a5 0 0)
Cp_c = np.array([-5.62770276e+02,  6.85239175e+00, -1.20636847e-02,
                  1.14177384e-05, -5.49067161e-09,  1.05213615e-12])
def Cp_new(T):
    return (Cp_c[0] + Cp_c[1]*T + Cp_c[2]*T**2 + Cp_c[3]*T**3
            + Cp_c[4]*T**4 + Cp_c[5]*T**5)

# ── Error calculation at table points ─────────────────────────────────────
err_kappa_old = np.abs((kappa_old(T_K) - kappa_tab) / kappa_tab) * 100
err_kappa_new = np.abs((kappa_new(T_K) - kappa_tab) / kappa_tab) * 100
err_Cp_old    = np.abs((Cp_old(T_K)    - Cp_tab)    / Cp_tab)    * 100
err_Cp_new    = np.abs((Cp_new(T_K)    - Cp_tab)    / Cp_tab)    * 100

# ── Dense T for smooth curves ──────────────────────────────────────────────
T_fine_K = np.linspace(T_K.min(), T_K.max(), 400)
T_fine_C = T_fine_K - 273.15

# ── Colours ────────────────────────────────────────────────────────────────
BLUE   = '#2477C8'
CORAL  = '#C84424'
GREEN  = '#3B8C3B'
GRAY   = '#888780'

# ── Figure ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.suptitle('SiO — Tabular data vs OpenFOAM polynomials\n'
             '(old linear/cubic  vs  new degree-3/degree-5 fits)',
             fontsize=13, fontweight='bold', y=0.99)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)

ax_k   = fig.add_subplot(gs[0, 0])
ax_cp  = fig.add_subplot(gs[0, 1])
ax_ek  = fig.add_subplot(gs[1, 0])
ax_ecp = fig.add_subplot(gs[1, 1])

def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=8)

# ── κ comparison ──────────────────────────────────────────────────────────
ax_k.plot(T_C, kappa_tab,           'o',  color=BLUE,  ms=6,  zorder=4, label='Tabular')
ax_k.plot(T_fine_C, kappa_old(T_fine_K), '--', color=CORAL, lw=2, label='Old (linear, 1 term)')
ax_k.plot(T_fine_C, kappa_new(T_fine_K), '-',  color=GREEN, lw=2, label='New (degree-3)')
style(ax_k, 'Thermal conductivity  κ', 'Temperature (°C)', 'κ  [W/(m·K)]')

# ── Cp comparison ─────────────────────────────────────────────────────────
ax_cp.plot(T_C, Cp_tab,            'o',  color=BLUE,  ms=6,  zorder=4, label='Tabular')
ax_cp.plot(T_fine_C, Cp_old(T_fine_K), '--', color=CORAL, lw=2, label='Old (degree-3)')
ax_cp.plot(T_fine_C, Cp_new(T_fine_K), '-',  color=GREEN, lw=2, label='New (degree-5)')
style(ax_cp, 'Heat capacity  Cp', 'Temperature (°C)', 'Cp  [J/(kg·K)]')

# ── κ error comparison ────────────────────────────────────────────────────
x  = np.arange(len(T_C))
w  = 0.35
ax_ek.bar(x - w/2, err_kappa_old, width=w, color=CORAL, alpha=0.8, label='Old  (max {:.1f}%)'.format(err_kappa_old.max()))
ax_ek.bar(x + w/2, err_kappa_new, width=w, color=GREEN, alpha=0.8, label='New  (max {:.1f}%)'.format(err_kappa_new.max()))
ax_ek.set_xticks(x)
ax_ek.set_xticklabels([f'{t}' for t in T_C], rotation=45, fontsize=8)
style(ax_ek, 'κ  relative error (%)', 'Temperature (°C)', 'Error (%)')

# ── Cp error comparison ───────────────────────────────────────────────────
ax_ecp.bar(x - w/2, err_Cp_old, width=w, color=CORAL, alpha=0.8, label='Old  (max {:.1f}%)'.format(err_Cp_old.max()))
ax_ecp.bar(x + w/2, err_Cp_new, width=w, color=GREEN, alpha=0.8, label='New  (max {:.1f}%)'.format(err_Cp_new.max()))
ax_ecp.set_xticks(x)
ax_ecp.set_xticklabels([f'{t}' for t in T_C], rotation=45, fontsize=8)
style(ax_ecp, 'Cp  relative error (%)', 'Temperature (°C)', 'Error (%)')

plt.savefig('sio_tight_fit.png', dpi=150, bbox_inches='tight')
print("Saved: sio_tight_fit.png")
print()
print("="*60)
print("OpenFOAM physicalProperties — updated coefficients")
print("="*60)
print()
print("transport")
print("{")
print("    kappaCoeffs<8>  ({:.6e}  {:.6e}  {:.6e}  {:.6e}  0  0  0  0);".format(*kappa_c))
print("    // degree-3 fit  |  max error: {:.2f}%".format(err_kappa_new.max()))
print("}")
print()
print("thermodynamics")
print("{")
print("    Hf              0;")
print("    Sf              0;")
print("    CpCoeffs<8>     ({:.6e}  {:.6e}  {:.6e}  {:.6e}  {:.6e}  {:.6e}  0  0);".format(*Cp_c))
print("    // degree-5 fit  |  max error: {:.2f}%".format(err_Cp_new.max()))
print("}")
