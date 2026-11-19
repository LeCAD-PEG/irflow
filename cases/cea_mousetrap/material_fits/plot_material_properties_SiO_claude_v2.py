import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Original tabular data (from measurement table, T in Kelvin) ────────────
T_orig_C = np.array([24, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200])
T_orig_K = T_orig_C + 273.15

Cv_orig  = np.array([666, 812, 937, 1022, 1084, 1131, 1167, 1198, 1223, 1245, 1263, 1279, 1294])
kappa_orig = np.array([17.89, 17.73, 16.82, 15.86, 15.25, 14.18,
                       13.64, 13.02, 12.51, 12.22, 11.41, 10.85, 9.59])

# ── OpenFOAM eIcoTabulated Cv table (extended) ────────────────────────────
T_foam_K = np.array([297, 373, 473, 573, 673, 773, 873, 973, 1073,
                     1173, 1273, 1373, 1473, 2000, 3000, 4000, 5000, 6000, 7000, 8000])
Cv_foam  = np.array([666, 812, 937, 1022, 1084, 1131, 1167, 1198, 1223,
                     1245, 1263, 1279, 1294, 1373, 1523, 1673, 1823, 1973, 2073, 2173])

# ── OpenFOAM polynomial coefficients ──────────────────────────────────────
# CvCoeffs<8>: degree-3
Cv_c = np.array([2.97608574e+01, 2.83547310e+00, -2.32946722e-03, 6.73249379e-07])
def Cv_poly(T):
    return sum(Cv_c[i] * T**i for i in range(4))

# kappaCoeffs<8>: degree-3
k_c = np.array([2.177878e+01, -1.422731e-02, 8.277203e-06, -2.795318e-09])
def kappa_poly(T):
    return sum(k_c[i] * T**i for i in range(4))

# ── Dense T arrays for smooth curves ──────────────────────────────────────
T_phys_K  = np.linspace(297, 1473, 400)          # physical fit range
T_ext_K   = np.linspace(297, 8000, 1000)          # extended range

# ── Colours ────────────────────────────────────────────────────────────────
BLUE   = '#2477C8'
CORAL  = '#C84424'
GREEN  = '#2E8B2E'
PURPLE = '#7B3FA0'
GRAY   = '#888780'

# ── Figure ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 11))
fig.suptitle('SiO thermal properties — original data vs OpenFOAM representations',
             fontsize=13, fontweight='bold', y=0.99)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.30)
ax_cv_full  = fig.add_subplot(gs[0, 0])
ax_cv_zoom  = fig.add_subplot(gs[0, 1])
ax_k_full   = fig.add_subplot(gs[1, 0])
ax_k_zoom   = fig.add_subplot(gs[1, 1])

def style(ax, title, xlabel, ylabel, legend=True):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    if legend:
        ax.legend(fontsize=8, loc='best')

# ── Cv full range (297–8000 K)
# ─────────────────────────────────────────────
ax_cv_full.plot(T_foam_K, Cv_foam, 's--', color=GREEN, ms=5, lw=1.5,
                label='OpenFOAM eIcoTabulated (extended)')
ax_cv_full.plot(T_ext_K, Cv_poly(T_ext_K), '-', color=CORAL, lw=2,
                label='Polynomial degree-3 (extrapolated)', zorder=2)
ax_cv_full.plot(T_orig_K, Cv_orig, 'o', color=BLUE, ms=6, zorder=4,
                label='Original table data')
ax_cv_full.axvline(1473, color=GRAY, lw=1, linestyle=':', label='Fit range limit (1473 K)')
ax_cv_full.set_xlim(200, 8200)
style(ax_cv_full, 'Cv — full range (297–8000 K)', 'T [K]', 'Cv  [J/(kg·K)]')

# ── Cv zoomed to original data range
# ──────────────────────────────────────
ax_cv_zoom.plot(T_foam_K[T_foam_K <= 1500], Cv_foam[T_foam_K <= 1500],
                's--', color=GREEN, ms=5, lw=1.5, label='OpenFOAM eIcoTabulated')
ax_cv_zoom.plot(T_phys_K, Cv_poly(T_phys_K), '-', color=CORAL, lw=2,
                label='Polynomial degree-3  (max err 2.75%)')
ax_cv_zoom.plot(T_orig_K, Cv_orig, 'o', color=BLUE, ms=6, zorder=4,
                label='Original table data')
style(ax_cv_zoom, 'Cv — original data range (297–1473 K)', 'T [K]', 'Cv  [J/(kg·K)]')

# ── kappa full range
# ───────────────────────────────────────────────────────
ax_k_full.plot(T_ext_K, kappa_poly(T_ext_K), '-', color=CORAL, lw=2,
               label='Polynomial degree-3 (extrapolated)')
ax_k_full.plot(T_orig_K, kappa_orig, 'o', color=BLUE, ms=6, zorder=4,
               label='Original table data')
ax_k_full.axvline(1473, color=GRAY, lw=1, linestyle=':', label='Fit range limit (1473 K)')
ax_k_full.axhline(0, color='black', lw=0.5, linestyle='-')
# Mark where kappa goes negative
T_zero = np.roots(k_c[::-1])
real_pos = T_zero[np.isreal(T_zero) & (T_zero.real > 0)].real
if len(real_pos) > 0:
    T_neg = real_pos.min()
    ax_k_full.axvline(T_neg, color='red', lw=1.5, linestyle='--',
                      label=f'kappa=0 at {T_neg:.0f} K (unphysical!)')
ax_k_full.set_xlim(200, 8200)
style(ax_k_full, 'κ — full range (297–8000 K)', 'T [K]', 'κ  [W/(m·K)]')

# ── kappa zoomed
# ──────────────────────────────────────────────────────────
ax_k_zoom.plot(T_phys_K, kappa_poly(T_phys_K), '-', color=CORAL, lw=2,
               label='Polynomial degree-3  (max err 2.67%)')
ax_k_zoom.plot(T_orig_K, kappa_orig, 'o', color=BLUE, ms=6, zorder=4,
               label='Original table data')
style(ax_k_zoom, 'κ — original data range (297–1473 K)', 'T [K]', 'κ  [W/(m·K)]')

# ── annotation box
# ────────────────────────────────────────────────────────
fig.text(0.5, 0.005,
         'Note: polynomial kappa goes negative above ~2900 K — eIcoTabulated should also be used for kappa if T > 1473 K',
         ha='center', fontsize=8, color='darkred',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff0f0', edgecolor='#C84424', linewidth=0.8))

plt.savefig('sio_full_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: sio_full_comparison.png")

# ── Print summary table ───────────────────────────────────────────────────
print()
print("="*65)
print("Cv comparison at key temperatures")
print("="*65)
print(f"{'T [K]':>8}  {'Orig table':>12}  {'Foam table':>12}  {'Polynomial':>12}")
print("-"*65)
for T, Cv_t in zip(T_orig_K, Cv_orig):
    Cv_f = np.interp(T, T_foam_K, Cv_foam)
    Cv_p = Cv_poly(T)
    print(f"{T:8.1f}  {Cv_t:12.1f}  {Cv_f:12.1f}  {Cv_p:12.1f}")
print()
print("Extended range (polynomial extrapolation warning):")
for T in [2000, 3000, 5000, 8000]:
    Cv_f = np.interp(T, T_foam_K, Cv_foam)
    Cv_p = Cv_poly(T)
    print(f"{T:8.0f}  {'---':>12}  {Cv_f:12.1f}  {Cv_p:12.1f}")
