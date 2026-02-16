"""
plot_gr_validation.py

Generate radial gravity validation plots for a trained PINN checkpoint:
1) Relative error curve: (g_r_PINN - g_r_analytic) / g_r_analytic
2) Log-log |g_r| comparison: analytic vs PINN
3) Signed g_r overlay (linear y-axis): analytic vs PINN

The script evaluates g_r_PINN from auto-diff at phi = phi_cut.
For checkpoints trained with target normalization, it can infer the output
scale from a saved data_*.npz file produced by train.py.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import GravityPINN
from physics import (
    gr_exponential_disk_analytic,
    gr_constant_disk_analytic,
)


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> GravityPINN:
    dom = cfg["domain"]
    net = cfg["pinn_modeA"]["network"]
    return GravityPINN(
        hidden_layers=net["hidden_layers"],
        hidden_units=net["hidden_units"],
        omega_0=net["siren_omega_0"],
        omega=net["siren_omega"],
        r_min=dom["r_min"],
        r_max=dom["r_max"],
    )


def load_checkpoint(model: GravityPINN, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()


def infer_phi_affine_from_npz(
    model: GravityPINN, npz_path: Path, device: torch.device
) -> tuple[float, float]:
    data = np.load(npz_path)
    if "r" not in data or "phi" not in data or "Phi_pinn" not in data:
        raise ValueError(f"{npz_path} must contain r, phi, Phi_pinn for scale inference.")

    r = data["r"]
    phi = data["phi"]
    phi_target = data["Phi_pinn"]
    rr, pp = np.meshgrid(r, phi, indexing="ij")

    r_t = torch.tensor(rr.ravel(), dtype=torch.float32, device=device)
    p_t = torch.tensor(pp.ravel(), dtype=torch.float32, device=device)
    with torch.no_grad():
        phi_raw = model(r_t, p_t).cpu().numpy().reshape(phi_target.shape)

    # Fit affine map: Phi_target ~= a * Phi_raw + b
    x = phi_raw.ravel()
    y = phi_target.ravel()
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    var_x = float(np.mean((x - x_mean) ** 2))
    if var_x < 1e-16:
        a = 1.0
        b = y_mean - a * x_mean
    else:
        cov_xy = float(np.mean((x - x_mean) * (y - y_mean)))
        a = cov_xy / var_x
        b = y_mean - a * x_mean
    return a, b


def analytic_gr(case_name: str, case_cfg: dict, r: np.ndarray, g_const: float,
                epsilon_analytic: float = 0.0) -> np.ndarray:
    tp = case_cfg["type"]
    if tp == "exponential":
        return gr_exponential_disk_analytic(
            r, Sigma_0=case_cfg["Sigma_0"], r_0=case_cfg["R_scale"], G=g_const
        )
    if tp == "constant":
        return gr_constant_disk_analytic(
            r,
            Sigma_0=case_cfg["Sigma_0"],
            r_in=case_cfg["r_in"],
            r_out=case_cfg["r_out"],
            G=g_const,
            epsilon=epsilon_analytic,
        )
    raise ValueError(
        f"Case '{case_name}' has type '{tp}', but no analytic g_r is implemented for this type."
    )


def make_rel_error(g_pred: np.ndarray, g_ref: np.ndarray, floor_frac: float = 1e-8) -> np.ndarray:
    scale = np.max(np.abs(g_ref)) + 1e-16
    floor = floor_frac * scale
    denom = g_ref.copy()
    small = np.abs(denom) < floor
    denom[small] = np.where(denom[small] >= 0.0, floor, -floor)
    return (g_pred - g_ref) / denom


def gr_pinn_autodiff(
    model: GravityPINN,
    r_eval: np.ndarray,
    phi_cut: float,
    phi_scale: float,
    device: torch.device,
) -> np.ndarray:
    """Compute g_r from autograd along a single azimuthal cut."""
    phi_eval = np.full_like(r_eval, float(phi_cut), dtype=np.float64)
    r_t = torch.tensor(r_eval, dtype=torch.float32, device=device, requires_grad=True)
    p_t = torch.tensor(phi_eval, dtype=torch.float32, device=device, requires_grad=True)
    g_r_raw, _ = model.gravity(r_t, p_t)
    return g_r_raw.detach().cpu().numpy().astype(np.float64) * phi_scale


def gr_pinn_fd(
    model: GravityPINN,
    r_eval: np.ndarray,
    phi_grid: np.ndarray,
    phi_cut: float,
    phi_scale: float,
    phi_offset: float,
    fd_reduction: str,
    device: torch.device,
) -> np.ndarray:
    """
    Compute g_r via finite differences on predicted Phi in physical units.

    fd_reduction:
      - "avg": average Phi over phi, then differentiate in r
      - "cut": take phi closest to phi_cut, then differentiate in r
    """
    rr, pp = np.meshgrid(r_eval, phi_grid, indexing="ij")
    r_t = torch.tensor(rr.ravel(), dtype=torch.float32, device=device)
    p_t = torch.tensor(pp.ravel(), dtype=torch.float32, device=device)
    with torch.no_grad():
        phi_raw = model(r_t, p_t).cpu().numpy().reshape(rr.shape)
    phi_phys = phi_scale * phi_raw + phi_offset

    if fd_reduction == "avg":
        phi_line = np.mean(phi_phys, axis=1)
    else:
        j_cut = int(np.argmin(np.abs(phi_grid - float(phi_cut))))
        phi_line = phi_phys[:, j_cut]

    return -np.gradient(phi_line, r_eval, edge_order=2)


def main():
    parser = argparse.ArgumentParser(description="Generate radial gravity validation plots.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--case", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--npz", type=str, default=None,
                        help="Path to data_*.npz for scale inference and r-grid.")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--phi-cut", type=float, default=0.0)
    parser.add_argument("--phi-scale", type=float, default=None,
                        help="If set, use this output scale instead of inferring from npz.")
    parser.add_argument("--phi-offset", type=float, default=0.0,
                        help="Optional affine offset for Phi. Does not affect g_r.")
    parser.add_argument("--dense-nr", type=int, default=512,
                        help="Only used if --npz is not provided.")
    parser.add_argument("--rel-mask-frac", type=float, default=1e-4,
                        help="Use points with |g_r,analytic| >= frac*max|g_r,analytic| for relative-error plot/metrics.")
    parser.add_argument("--expect-type", type=str, default=None,
                        help="Optional guard: fail if benchmark type is not this value.")
    parser.add_argument("--epsilon-analytic", type=float, default=0.0,
                        help="Softening length to use in analytic constant-annulus comparator.")
    parser.add_argument("--gravity-method", type=str, default="autodiff",
                        choices=["autodiff", "fd"],
                        help="How to compute g_r from PINN.")
    parser.add_argument("--fd-reduction", type=str, default="avg",
                        choices=["avg", "cut"],
                        help="FD method: reduce phi by azimuthal average or single phi cut.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(Path(args.config))
    if args.case not in cfg["benchmarks"]:
        raise ValueError(f"Unknown case '{args.case}'. Available: {list(cfg['benchmarks'].keys())}")
    case_cfg = cfg["benchmarks"][args.case]
    if args.expect_type is not None and case_cfg["type"] != args.expect_type:
        raise ValueError(
            f"Case '{args.case}' type is '{case_cfg['type']}', expected '{args.expect_type}'."
        )
    g_const = 1.0 if cfg["physics"]["use_code_units"] else cfg["physics"]["G"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    load_checkpoint(model, Path(args.checkpoint), device)

    # Radial grid: prefer the saved run grid.
    if args.npz is not None:
        npz_path = Path(args.npz)
        data = np.load(npz_path)
        r_eval = data["r"].astype(np.float64)
        phi_grid = data["phi"].astype(np.float64)
    else:
        r_min = float(cfg["domain"]["r_min"])
        r_max = float(cfg["domain"]["r_max"])
        r_eval = np.logspace(np.log10(r_min), np.log10(r_max), args.dense_nr)
        n_phi = int(cfg["domain"]["N_phi"])
        phi_grid = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False, dtype=np.float64)

    # Determine output scale for physical units.
    if args.phi_scale is not None:
        phi_scale = float(args.phi_scale)
        phi_offset = float(args.phi_offset)
    elif args.npz is not None:
        phi_scale, phi_offset = infer_phi_affine_from_npz(model, Path(args.npz), device)
    else:
        raise ValueError("Need either --phi-scale or --npz to recover physical scaling.")

    # PINN g_r via selected method.
    if args.gravity_method == "fd":
        g_r_pinn = gr_pinn_fd(
            model=model,
            r_eval=r_eval,
            phi_grid=phi_grid,
            phi_cut=float(args.phi_cut),
            phi_scale=phi_scale,
            phi_offset=phi_offset,
            fd_reduction=args.fd_reduction,
            device=device,
        )
    else:
        g_r_pinn = gr_pinn_autodiff(
            model=model,
            r_eval=r_eval,
            phi_cut=float(args.phi_cut),
            phi_scale=phi_scale,
            device=device,
        )

    # Analytic reference.
    g_r_an = analytic_gr(args.case, case_cfg, r_eval, g_const,
                         epsilon_analytic=float(args.epsilon_analytic))

    # Metrics.
    rel_curve_all = make_rel_error(g_r_pinn, g_r_an)
    rel_threshold = float(args.rel_mask_frac) * float(np.max(np.abs(g_r_an)) + 1e-16)
    mask = np.abs(g_r_an) >= rel_threshold
    rel_curve = np.full_like(rel_curve_all, np.nan)
    rel_curve[mask] = rel_curve_all[mask]

    l2_rel = np.linalg.norm(g_r_pinn[mask] - g_r_an[mask]) / (
        np.linalg.norm(g_r_an[mask]) + 1e-12
    )
    abs_rel = np.abs(rel_curve[mask])
    r_mask = r_eval[mask]
    i_max = int(np.argmax(abs_rel))
    max_abs_rel = float(abs_rel[i_max])
    r_at_max = float(r_mask[i_max])
    sign_mismatch_frac = float(np.mean(np.sign(g_r_pinn[mask]) != np.sign(g_r_an[mask])))

    # Plot 1: Relative error.
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(r_eval, rel_curve, "-", color="tab:blue", lw=1.8)
    ax1.axhline(0.0, color="k", lw=0.8, alpha=0.6)
    ax1.set_title("Relative Error in Radial Gravitational Acceleration")
    ax1.set_xlabel("Radial distance r [AU]")
    ax1.set_ylabel("(g_r,PINN - g_r,analytic) / g_r,analytic")
    y_lim = np.nanmax(np.abs(rel_curve)) if np.any(np.isfinite(rel_curve)) else np.nan
    if np.isfinite(y_lim) and y_lim > 10.0:
        ax1.set_yscale("symlog", linthresh=1e-3)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    p1 = outdir / f"relative_error_gr_{args.case}_phi{args.phi_cut:.2f}.png"
    fig1.savefig(p1, dpi=180, bbox_inches="tight")
    plt.close(fig1)

    # Plot 2: |g_r| comparison (log-log).
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.loglog(r_eval, np.abs(g_r_an), "k-", lw=2.0, label="Analytic")
    ax2.loglog(r_eval, np.abs(g_r_pinn), "r--", lw=1.8, label="PINN")
    ax2.set_title("Radial Gravitational Acceleration: Analytic vs PINN")
    ax2.set_xlabel("Radial distance r [AU]")
    ax2.set_ylabel("|g_r|")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    p2 = outdir / f"abs_gr_compare_{args.case}_phi{args.phi_cut:.2f}.png"
    fig2.savefig(p2, dpi=180, bbox_inches="tight")
    plt.close(fig2)

    # Plot 3: signed g_r overlay to expose sign errors directly.
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(r_eval, g_r_an, "k-", lw=2.0, label="Analytic")
    ax3.plot(r_eval, g_r_pinn, "r--", lw=1.8, label="PINN")
    ax3.axhline(0.0, color="k", lw=0.8, alpha=0.6)
    ax3.set_title("Signed Radial Gravitational Acceleration")
    ax3.set_xlabel("Radial distance r [AU]")
    ax3.set_ylabel("g_r")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    fig3.tight_layout()
    p3 = outdir / f"signed_gr_compare_{args.case}_phi{args.phi_cut:.2f}.png"
    fig3.savefig(p3, dpi=180, bbox_inches="tight")
    plt.close(fig3)

    # Save metrics.
    metrics_path = outdir / f"gr_validation_metrics_{args.case}_phi{args.phi_cut:.2f}.csv"
    with metrics_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["case", args.case])
        w.writerow(["case_type", case_cfg["type"]])
        w.writerow(["phi_cut", args.phi_cut])
        w.writerow(["phi_scale", phi_scale])
        w.writerow(["phi_offset", phi_offset])
        w.writerow(["gravity_method", args.gravity_method])
        w.writerow(["fd_reduction", args.fd_reduction])
        w.writerow(["rel_mask_threshold", rel_threshold])
        w.writerow(["rel_mask_count", int(np.sum(mask))])
        w.writerow(["epsilon_analytic", float(args.epsilon_analytic)])
        w.writerow(["L2_relative_error", l2_rel])
        w.writerow(["max_abs_relative_error", max_abs_rel])
        w.writerow(["radius_at_max_error_AU", r_at_max])
        w.writerow(["sign_mismatch_fraction", sign_mismatch_frac])

    print(f"Saved: {p1}")
    print(f"Saved: {p2}")
    print(f"Saved: {p3}")
    print(f"Saved: {metrics_path}")
    print(f"L2 relative error: {l2_rel:.6e}")
    print(f"Max absolute relative error: {max_abs_rel:.6e}")
    print(f"Radius at max error [AU]: {r_at_max:.6e}")
    print(f"Sign mismatch fraction (masked): {sign_mismatch_frac:.6e}")


if __name__ == "__main__":
    main()
