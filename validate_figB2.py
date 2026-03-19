import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from physics import (
    cell_sizes,
    compute_gravity_fd,
    compute_potential_direct,
    compute_potential_fft,
    gr_constant_disk_axisym_quadrature,
    make_polar_grid,
    phi_constant_disk_axisym_quadrature,
    sigma_constant,
)


def compute_gr_numerical(N_r: int, N_phi: int,
                         r_in: float, r_out: float,
                         Sigma_0: float, G: float = 1.0,
                         epsilon: float = 0.0,
                         method: str = "direct") -> tuple:
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

    g_r, _ = compute_gravity_fd(Phi, r_1d, phi_1d)
    g_r_avg = np.mean(g_r, axis=1)
    return r_1d, g_r_avg


def compute_phi_numerical(N_r: int, N_phi: int,
                          r_in: float, r_out: float,
                          Sigma_0: float, G: float = 1.0,
                          epsilon: float = 0.0,
                          method: str = "direct") -> tuple:
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

    return r_1d, np.mean(Phi, axis=1)


def load_pinn_overlay(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    r = np.asarray(data["r"], dtype=np.float64)
    g_r_pinn = np.mean(np.asarray(data["g_r_pinn"], dtype=np.float64), axis=1)
    phi_pinn = np.mean(np.asarray(data["Phi_pinn"], dtype=np.float64), axis=1)
    return r, g_r_pinn, phi_pinn


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
    parser.add_argument("--method", type=str, default="fft",
                        choices=["direct", "fft"],
                        help="Numerical method for potential computation")
    parser.add_argument("--skip-high-res", action="store_true",
                        help="Skip N_r=512,1024 (slow for direct method)")
    parser.add_argument(
        "--rel-mask-frac",
        type=float,
        default=0.05,
        help="Mask relative-error curves where |g_r,ref| is below this fraction of max |g_r,ref|",
    )
    parser.add_argument(
        "--pinn-data",
        type=str,
        default="results/B5_eps000_seed123_fftref/data_B5_constant_density_eps0.00.npz",
        help="Optional NPZ with saved PINN arrays for overlay",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    G = args.G
    S0 = args.Sigma0
    rin = args.r_in
    rout = args.r_out
    Nphi = args.N_phi

    print("Computing axisymmetric annulus reference...")
    r_dense = np.logspace(np.log10(rin * 1.01), np.log10(rout * 0.99), 500)
    gr_analytic = gr_constant_disk_axisym_quadrature(
        r_dense, Sigma_0=S0, r_in=rin, r_out=rout, G=G
    )
    phi_analytic = phi_constant_disk_axisym_quadrature(
        r_dense, Sigma_0=S0, r_in=rin, r_out=rout, G=G
    )

    pinn_data_path = Path(args.pinn_data) if args.pinn_data else None
    if pinn_data_path is not None and pinn_data_path.exists():
        r_pinn, g_r_pinn, phi_pinn = load_pinn_overlay(pinn_data_path)
        g_r_pinn_ref = np.interp(r_pinn, r_dense, gr_analytic)
        phi_pinn_ref = np.interp(r_pinn, r_dense, phi_analytic)
        phi_pinn_offset = np.mean(phi_pinn_ref - phi_pinn)
        phi_pinn_aligned = phi_pinn + phi_pinn_offset
        print(f"Loaded PINN overlay: {pinn_data_path}")
        print(f"  Applied Φ offset alignment: {phi_pinn_offset:.4e}")
    else:
        r_pinn = None
        g_r_pinn = None
        g_r_pinn_ref = None
        phi_pinn_aligned = None

    if args.skip_high_res:
        resolutions = [64, 128, 256]
    else:
        resolutions = [64, 128, 256, 512]

    colors = {64: "C3", 128: "C0", 256: "C1", 512: "C2", 1024: "C4"}

    fig_top, (ax_gr, ax_err) = plt.subplots(1, 2, figsize=(14, 6))

    ax_gr.loglog(
        r_dense,
        np.abs(gr_analytic),
        "-",
        color="royalblue",
        lw=2.5,
        label="Axisymmetric reference",
        zorder=10,
    )

    for Nr in resolutions:
        print(f"  Computing ε=0, N_r={Nr}, N_φ={Nphi}, method={args.method}...")
        r_num, gr_num = compute_gr_numerical(
            Nr, Nphi, rin, rout, S0, G=G, epsilon=0.0, method=args.method
        )

        ax_gr.loglog(r_num, np.abs(gr_num), "--", color=colors.get(Nr, "gray"),
                     lw=1.5, label=f"N_r={Nr}")

        gr_an_interp = np.interp(r_num, r_dense, gr_analytic)
        mask = np.abs(gr_an_interp) > float(args.rel_mask_frac) * np.max(np.abs(gr_an_interp))
        rel_err = np.abs(gr_num[mask] - gr_an_interp[mask]) / np.abs(gr_an_interp[mask])

        ax_err.loglog(r_num[mask], rel_err, "-", color=colors.get(Nr, "gray"),
                      lw=1.5, label=f"N_r={Nr}")

        l2 = np.sqrt(np.mean(rel_err**2))
        print(f"    L2 relative error (g_r): {l2:.4e}")

    if r_pinn is not None:
        ax_gr.loglog(
            r_pinn,
            np.abs(g_r_pinn),
            "-",
            color="k",
            lw=3.0,
            label="PINN",
            zorder=12,
        )
        pinn_mask = np.abs(g_r_pinn_ref) > float(args.rel_mask_frac) * np.max(np.abs(g_r_pinn_ref))
        pinn_rel = np.abs(g_r_pinn[pinn_mask] - g_r_pinn_ref[pinn_mask]) / np.abs(g_r_pinn_ref[pinn_mask])
        ax_err.loglog(
            r_pinn[pinn_mask],
            pinn_rel,
            "-",
            color="k",
            lw=3.0,
            label="PINN",
            zorder=12,
        )

    ax_gr.set_xlabel("r [AU]")
    ax_gr.set_ylabel("|g_r|")
    ax_gr.set_title("Radial gravity: ε = 0, resolution sweep")
    ax_gr.legend(fontsize=9)
    ax_gr.grid(True, which="both", alpha=0.3)

    ax_err.set_xlabel("r [AU]")
    ax_err.set_ylabel("Relative error |g_r,num - g_r,analytic| / |g_r,analytic|")
    ax_err.set_title("Convergence to axisymmetric reference")
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

    Nr_fixed = 256
    r_1d_ref, _, _, _ = make_polar_grid(rin, rout, Nr_fixed, Nphi)
    dr_mean = np.mean(np.diff(r_1d_ref))

    eps_dr_list = [0.0, 0.25, 0.5, 1.0, 2.0]
    eps_colors = {0.0: "dimgray", 0.25: "C0", 0.5: "C1", 1.0: "C2", 2.0: "C3"}

    fig_bot, ax_bot = plt.subplots(figsize=(8, 6))

    ax_bot.loglog(
        r_dense,
        np.abs(gr_analytic),
        "-",
        color="royalblue",
        lw=2.5,
        label="Analytic (ε=0)",
        zorder=10,
    )

    for eps_dr in eps_dr_list:
        eps_phys = eps_dr * dr_mean
        label = f"ε/Δr={eps_dr:.2f}" if eps_dr > 0 else "ε=0 (singular corr.)"
        print(f"  Computing N_r={Nr_fixed}, ε/Δr={eps_dr:.2f} (ε={eps_phys:.4f})...")

        r_num, gr_num = compute_gr_numerical(
            Nr_fixed, Nphi, rin, rout, S0, G=G,
            epsilon=eps_phys, method=args.method
        )

        ls = ":" if eps_dr == 0 else "--"
        ax_bot.loglog(r_num, np.abs(gr_num), ls,
                      color=eps_colors.get(eps_dr, "gray"),
                      lw=1.5, label=label)

    if r_pinn is not None:
        ax_bot.loglog(
            r_pinn,
            np.abs(g_r_pinn),
            "-",
            color="k",
            lw=3.0,
            label="PINN",
            zorder=12,
        )

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

    fig_phi, ax_phi = plt.subplots(figsize=(8, 5))

    ax_phi.semilogx(
        r_dense,
        phi_analytic,
        "-",
        color="royalblue",
        lw=2.5,
        label="Axisymmetric reference Φ",
    )

    for Nr in [128, 256]:
        print(f"  Computing Φ for N_r={Nr}...")
        r_1d, phi_avg = compute_phi_numerical(
            Nr, Nphi, rin, rout, S0, G=G, epsilon=0.0, method=args.method
        )
        ax_phi.semilogx(r_1d, phi_avg, "--", lw=1.5, label=f"Numerical N_r={Nr}")

    if r_pinn is not None:
        ax_phi.semilogx(
            r_pinn,
            phi_pinn_aligned,
            "-",
            color="k",
            lw=3.0,
            label="PINN (aligned)",
            zorder=12,
        )

    ax_phi.set_xlabel("r [AU]")
    ax_phi.set_ylabel("Φ(r)")
    ax_phi.set_title("Gravitational potential: numerical vs axisymmetric reference")
    ax_phi.legend(fontsize=9)
    ax_phi.grid(True, alpha=0.3)
    fig_phi.tight_layout()
    p_phi = outdir / "figB2_potential_comparison.png"
    fig_phi.savefig(p_phi, dpi=180, bbox_inches="tight")
    plt.close(fig_phi)
    print(f"Saved: {p_phi}")



if __name__ == "__main__":
    main()
