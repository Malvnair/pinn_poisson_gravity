"""
validate_figB2.py  –  Reproduce Figure B.2 from Vorobyov et al. (2024)

Tests the numerical gravity solver (both direct-sum and FFT) against the
Durand (1964) analytic formula for the constant-density annulus (B5 case).

TOP PANEL:  ε = 0, resolution sweep  →  convergence to analytic
BOTTOM PANEL: fixed resolution, ε-sweep  →  effect of softening

Reference:  Vorobyov et al. (2024) A&A, Appendix B, Eq. B.8
            Originally from Durand (1964) Électrostatique.

Usage:
    python validate_figB2.py [--outdir results/figB2]
"""

import argparse
from pathlib import Path
import numpy as np
from scipy.special import ellipk, ellipe
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physics import (
    make_polar_grid,
    cell_sizes,
    sigma_constant,
    compute_potential_direct,
    compute_potential_fft,
    compute_gravity_fd,
)


# ── Durand (1964) analytic formula for constant-density annulus ──────────
# g_r for Σ = const between r_in and r_out, ε = 0
#
# Φ(r) = -4GΣ ∫_{r_in}^{r_out} r' K(k) / (r + r') dr'
#   where k² = 4rr'/(r+r')²
#
# For constant Σ, the radial gravity g_r = -dΦ/dr has a known closed form
# in terms of complete elliptic integrals.  Vorobyov (2024) Eq. B.8 gives:
#
#   g_r(r) = 4GΣ [ (E(k_out) + K(k_out))/k_out + K(k_in) - E(k_in) ]
#
# where k_out = r/r_out  and  k_in = r_in/r  (both ≤ 1 inside the disk).
#
# HOWEVER this expression is only valid for r_in < r < r_out.
# For r < r_in or r > r_out, different branch expressions apply.
# To be safe and general, we compute g_r by numerical differentiation
# of the potential, which uses the standard elliptic-integral expression.


def phi_durand_analytic(r: np.ndarray, Sigma_0: float,
                        r_in: float, r_out: float,
                        G: float = 1.0,
                        n_quad: int = 4096) -> np.ndarray:
    """
    Potential Φ(r) for constant-Σ annulus via high-accuracy quadrature
    of the exact (ε=0) kernel using complete elliptic integrals.
    
    Φ(r) = -4GΣ ∫_{r_in}^{r_out} [r'/(r+r')] K(k) dr'
    where k² = 4rr'/(r+r')²
    """
    r = np.asarray(r, dtype=np.float64)
    # Use Gauss-Legendre quadrature for high accuracy
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    # Map from [-1,1] to [r_in, r_out]
    rp = 0.5 * (r_out - r_in) * nodes + 0.5 * (r_out + r_in)
    w = 0.5 * (r_out - r_in) * weights

    phi = np.zeros_like(r)
    for i, r_val in enumerate(r):
        if r_val <= 0:
            phi[i] = np.nan
            continue
        k_sq = 4.0 * r_val * rp / (r_val + rp)**2
        k_sq = np.clip(k_sq, 0.0, 1.0 - 1e-15)
        K_vals = ellipk(k_sq)
        integrand = rp * K_vals / (r_val + rp)
        phi[i] = -4.0 * G * Sigma_0 * np.dot(w, integrand)
    return phi


def gr_durand_analytic(r: np.ndarray, Sigma_0: float,
                       r_in: float, r_out: float,
                       G: float = 1.0,
                       n_quad: int = 4096) -> np.ndarray:
    """
    g_r = -dΦ/dr computed via Richardson-extrapolated finite differences
    of the high-accuracy Gauss-Legendre potential.
    
    Uses 5-point stencil with h = 1e-4 * r for robust differentiation.
    """
    r = np.asarray(r, dtype=np.float64)
    gr = np.zeros_like(r)
    
    for i, r_val in enumerate(r):
        if r_val <= 0:
            gr[i] = np.nan
            continue
        
        # 5-point central difference: (-f(r+2h) + 8f(r+h) - 8f(r-h) + f(r-2h)) / (12h)
        h = 1e-4 * r_val
        h = max(h, 1e-10)
        
        stencil_r = np.array([r_val - 2*h, r_val - h, r_val + h, r_val + 2*h])
        stencil_phi = phi_durand_analytic(stencil_r, Sigma_0, r_in, r_out, G, n_quad)
        
        dphi_dr = (-stencil_phi[3] + 8*stencil_phi[2] - 8*stencil_phi[1] + stencil_phi[0]) / (12*h)
        gr[i] = -dphi_dr
    
    return gr


def gr_durand_closed_form(r: np.ndarray, Sigma_0: float,
                          r_in: float, r_out: float,
                          G: float = 1.0) -> np.ndarray:
    """
    Closed-form g_r from Durand (1964) / Vorobyov (2024) Eq. B.8.
    Valid for r_in < r < r_out.
    
    g_r(r) = 4GΣ [ {E(k_out) + K(k_out)}/k_out + K(k_in) - E(k_in) ]
    
    where k_out = r/r_out, k_in = r_in/r
    
    Sign convention: g_r < 0 means inward (attractive).
    The formula above gives the magnitude; the actual sign depends on geometry.
    For a self-gravitating disk, the net radial force is generally inward
    at intermediate radii and outward near the inner edge.
    """
    r = np.asarray(r, dtype=np.float64)
    g = np.zeros_like(r)
    
    for i, rv in enumerate(r):
        if rv <= r_in or rv >= r_out:
            g[i] = np.nan  # Use numerical derivative outside disk
            continue
        
        k_out = rv / r_out   # ≤ 1
        k_in = r_in / rv     # ≤ 1
        
        # Elliptic integrals with modulus squared = k²
        K_out = ellipk(k_out**2)
        E_out = ellipe(k_out**2)
        K_in = ellipk(k_in**2)
        E_in = ellipe(k_in**2)
        
        # Eq. B.8 from Vorobyov (2024)
        g[i] = 4.0 * G * Sigma_0 * (
            (E_out + K_out) / k_out + K_in - E_in
        )
    
    return g


# ── Numerical solver wrappers ───────────────────────────────────────────

def compute_gr_numerical(N_r: int, N_phi: int,
                         r_in: float, r_out: float,
                         Sigma_0: float, G: float = 1.0,
                         epsilon: float = 0.0,
                         method: str = "direct") -> tuple:
    """
    Compute g_r(r) at φ=0 from the numerical solver.
    Returns (r_1d, g_r_azavg) where g_r is azimuthally averaged.
    """
    r_1d, phi_1d, R, PHI = make_polar_grid(r_in, r_out, N_r, N_phi)
    dr, dphi = cell_sizes(r_1d, phi_1d)
    Sigma = sigma_constant(R, PHI, Sigma_0, r_in, r_out)
    
    if method == "direct":
        Phi = compute_potential_direct(
            r_1d, phi_1d, Sigma, r_1d, phi_1d, dr, dphi,
            G=G, epsilon=epsilon, use_singular_correction=(epsilon == 0.0)
        )
    elif method == "fft":
        Phi = compute_potential_fft(
            Sigma, r_1d, phi_1d, dr, dphi,
            G=G, epsilon=epsilon, use_singular_correction=(epsilon == 0.0)
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    g_r, g_phi = compute_gravity_fd(Phi, r_1d, phi_1d)
    
    # Azimuthal average (should be ~constant for axisymmetric case)
    g_r_avg = np.mean(g_r, axis=1)
    return r_1d, g_r_avg


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Figure B.2: solver validation for constant-density annulus"
    )
    parser.add_argument("--outdir", type=str, default="results/figB2")
    parser.add_argument("--G", type=float, default=1.0, help="Gravitational constant (code units)")
    parser.add_argument("--Sigma0", type=float, default=100.0)
    parser.add_argument("--r-in", type=float, default=0.2)
    parser.add_argument("--r-out", type=float, default=100.0)
    parser.add_argument("--N-phi", type=int, default=128)
    parser.add_argument("--method", type=str, default="direct",
                        choices=["direct", "fft"],
                        help="Numerical method for potential computation")
    parser.add_argument("--skip-high-res", action="store_true",
                        help="Skip N_r=512,1024 (slow for direct method)")
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    G = args.G
    S0 = args.Sigma0
    rin = args.r_in
    rout = args.r_out
    Nphi = args.N_phi
    
    # ── Dense analytic reference ──
    print("Computing Durand analytic reference (n_quad=4096)...")
    r_dense = np.logspace(np.log10(rin * 1.01), np.log10(rout * 0.99), 500)
    gr_analytic = gr_durand_analytic(r_dense, S0, rin, rout, G=G)
    
    # Also compute closed-form for comparison (interior only)
    gr_closed = gr_durand_closed_form(r_dense, S0, rin, rout, G=G)
    
    # Verify quadrature vs closed-form agreement
    mask_interior = ~np.isnan(gr_closed) & ~np.isnan(gr_analytic)
    if np.any(mask_interior):
        rel_diff = np.abs(gr_analytic[mask_interior] - gr_closed[mask_interior]) / (
            np.abs(gr_closed[mask_interior]) + 1e-30
        )
        print(f"  Quadrature vs closed-form max relative diff: {np.max(rel_diff):.2e}")
        print(f"  Quadrature vs closed-form mean relative diff: {np.mean(rel_diff):.2e}")
    
    # ══════════════════════════════════════════════════════════════════════
    # TOP PANEL: ε = 0, resolution sweep
    # ══════════════════════════════════════════════════════════════════════
    if args.skip_high_res:
        resolutions = [64, 128, 256]
    else:
        resolutions = [64, 128, 256, 512]
    
    colors = {64: "C3", 128: "C0", 256: "C1", 512: "C2", 1024: "C4"}
    
    fig_top, (ax_gr, ax_err) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot analytic
    ax_gr.loglog(r_dense, np.abs(gr_analytic), "k-", lw=2.5, label="Durand analytic", zorder=10)
    
    for Nr in resolutions:
        print(f"  Computing ε=0, N_r={Nr}, N_φ={Nphi}, method={args.method}...")
        r_num, gr_num = compute_gr_numerical(
            Nr, Nphi, rin, rout, S0, G=G, epsilon=0.0, method=args.method
        )
        
        # Plot |g_r|
        ax_gr.loglog(r_num, np.abs(gr_num), "--", color=colors.get(Nr, "gray"),
                     lw=1.5, label=f"N_r={Nr}")
        
        # Compute relative error vs analytic (interpolate analytic to numerical grid)
        gr_an_interp = np.interp(r_num, r_dense, gr_analytic)
        mask = np.abs(gr_an_interp) > 1e-10 * np.max(np.abs(gr_an_interp))
        rel_err = np.abs(gr_num[mask] - gr_an_interp[mask]) / np.abs(gr_an_interp[mask])
        
        ax_err.loglog(r_num[mask], rel_err, "-", color=colors.get(Nr, "gray"),
                      lw=1.5, label=f"N_r={Nr}")
        
        l2 = np.sqrt(np.mean(rel_err**2))
        print(f"    L2 relative error (g_r): {l2:.4e}")
    
    ax_gr.set_xlabel("r [AU]")
    ax_gr.set_ylabel("|g_r|")
    ax_gr.set_title("Radial gravity: ε = 0, resolution sweep")
    ax_gr.legend(fontsize=9)
    ax_gr.grid(True, which="both", alpha=0.3)
    
    ax_err.set_xlabel("r [AU]")
    ax_err.set_ylabel("Relative error |g_r,num - g_r,analytic| / |g_r,analytic|")
    ax_err.set_title("Convergence to Durand (1964)")
    ax_err.legend(fontsize=9)
    ax_err.grid(True, which="both", alpha=0.3)
    ax_err.set_ylim(1e-6, 10)
    
    fig_top.suptitle(
        f"Figure B.2 (top): Constant-density annulus, Σ={S0}, "
        f"r∈[{rin}, {rout}] AU, ε=0",
        fontsize=12, y=1.02
    )
    fig_top.tight_layout()
    p_top = outdir / "figB2_top_eps0_convergence.png"
    fig_top.savefig(p_top, dpi=180, bbox_inches="tight")
    plt.close(fig_top)
    print(f"Saved: {p_top}")
    
    # ══════════════════════════════════════════════════════════════════════
    # BOTTOM PANEL: fixed resolution, ε-sweep
    # ══════════════════════════════════════════════════════════════════════
    Nr_fixed = 256
    # ε values in units of mean Δr
    r_1d_ref, _, _, _ = make_polar_grid(rin, rout, Nr_fixed, Nphi)
    dr_mean = np.mean(np.diff(r_1d_ref))
    
    eps_dr_list = [0.0, 0.25, 0.5, 1.0, 2.0]
    eps_colors = {0.0: "k", 0.25: "C0", 0.5: "C1", 1.0: "C2", 2.0: "C3"}
    
    fig_bot, ax_bot = plt.subplots(figsize=(8, 6))
    
    # Analytic (ε=0)
    ax_bot.loglog(r_dense, np.abs(gr_analytic), "k-", lw=2.5, label="Analytic (ε=0)")
    
    for eps_dr in eps_dr_list:
        eps_phys = eps_dr * dr_mean
        label = f"ε/Δr={eps_dr:.2f}" if eps_dr > 0 else "ε=0 (singular corr.)"
        print(f"  Computing N_r={Nr_fixed}, ε/Δr={eps_dr:.2f} (ε={eps_phys:.4f})...")
        
        r_num, gr_num = compute_gr_numerical(
            Nr_fixed, Nphi, rin, rout, S0, G=G,
            epsilon=eps_phys, method=args.method
        )
        
        ls = "-" if eps_dr == 0 else "--"
        ax_bot.loglog(r_num, np.abs(gr_num), ls,
                      color=eps_colors.get(eps_dr, "gray"),
                      lw=1.5, label=label)
    
    ax_bot.set_xlabel("r [AU]")
    ax_bot.set_ylabel("|g_r|")
    ax_bot.set_title(
        f"Figure B.2 (bottom): ε-sweep at N_r={Nr_fixed}, N_φ={Nphi}"
    )
    ax_bot.legend(fontsize=9)
    ax_bot.grid(True, which="both", alpha=0.3)
    
    fig_bot.tight_layout()
    p_bot = outdir / "figB2_bottom_eps_sweep.png"
    fig_bot.savefig(p_bot, dpi=180, bbox_inches="tight")
    plt.close(fig_bot)
    print(f"Saved: {p_bot}")
    
    # ══════════════════════════════════════════════════════════════════════
    # BONUS: Potential comparison (Φ vs r) at φ = 0
    # ══════════════════════════════════════════════════════════════════════
    fig_phi, ax_phi = plt.subplots(figsize=(8, 5))
    
    phi_an = phi_durand_analytic(r_dense, S0, rin, rout, G=G)
    ax_phi.semilogx(r_dense, phi_an, "k-", lw=2.5, label="Durand analytic Φ")
    
    for Nr in [128, 256]:
        print(f"  Computing Φ for N_r={Nr}...")
        r_1d, phi_1d, R, PHI = make_polar_grid(rin, rout, Nr, Nphi)
        dr, dphi = cell_sizes(r_1d, phi_1d)
        Sigma = sigma_constant(R, PHI, S0, rin, rout)
        
        Phi = compute_potential_direct(
            r_1d, phi_1d, Sigma, r_1d, phi_1d, dr, dphi,
            G=G, epsilon=0.0, use_singular_correction=True
        )
        
        # Azimuthal average
        phi_avg = np.mean(Phi, axis=1)
        ax_phi.semilogx(r_1d, phi_avg, "--", lw=1.5, label=f"Numerical N_r={Nr}")
    
    ax_phi.set_xlabel("r [AU]")
    ax_phi.set_ylabel("Φ(r)")
    ax_phi.set_title("Gravitational potential: numerical vs Durand analytic")
    ax_phi.legend(fontsize=9)
    ax_phi.grid(True, alpha=0.3)
    fig_phi.tight_layout()
    p_phi = outdir / "figB2_potential_comparison.png"
    fig_phi.savefig(p_phi, dpi=180, bbox_inches="tight")
    plt.close(fig_phi)
    print(f"Saved: {p_phi}")
    
    print("\nDone! All Figure B.2 validation plots saved to:", outdir)


if __name__ == "__main__":
    main()
