import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml

from model import GravityPINN
from physics import (
    build_integral_source_cache,
    cell_sizes,
    evaluate_integral_at_points,
    make_polar_grid,
    rel_error_L2,
    sigma_exponential,
)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calibrate_potential_offset(model, phi_scale: float,
                               r_grid: np.ndarray, phi_grid: np.ndarray,
                               Sigma: np.ndarray, dr: np.ndarray, dphi: float,
                               G: float, device: torch.device,
                               source_cache: dict) -> float:
    
    
    r_eval = np.sqrt(r_grid[:-1] * r_grid[1:])
    phi_eval = np.mod(phi_grid + 0.5 * dphi, 2.0 * np.pi)
    r_mesh, phi_mesh = np.meshgrid(r_eval, phi_eval, indexing="ij")
    r_flat = r_mesh.ravel()
    phi_flat = phi_mesh.ravel()

    phi_target = evaluate_integral_at_points(
        r_flat, phi_flat, Sigma, r_grid, phi_grid, dr, dphi,
        G=G, epsilon=0.0, use_singular_correction=False,
        source_cache=source_cache,
    )

    model.eval()
    with torch.no_grad():
        r_t = torch.tensor(r_flat, dtype=torch.float32, device=device)
        phi_t = torch.tensor(phi_flat, dtype=torch.float32, device=device)
        phi_pred = phi_scale * model(r_t, phi_t).cpu().numpy()

    return float(np.mean(phi_target - phi_pred))


def train_experiment(args, cfg: dict, device: torch.device, outdir: Path):
    dom = cfg["domain"]
    net_cfg = cfg["pinn_modeA"]["network"]
    eval_cfg = cfg.get("evaluation", {})

    r_min = float(dom["r_min"])
    r_max = float(dom["r_max"])
    r_grid, phi_grid, R, PHI = make_polar_grid(r_min, r_max, args.source_nr, args.source_nphi)
    dr, dphi = cell_sizes(r_grid, phi_grid)
    Sigma = sigma_exponential(R, PHI, args.sigma0, args.r0)
    source_cache = build_integral_source_cache(Sigma, r_grid, phi_grid, dr, dphi)

    phi_ref_scale = evaluate_integral_at_points(
        np.array([args.r0]), np.array([0.0]),
        Sigma, r_grid, phi_grid, dr, dphi,
        G=args.G, epsilon=0.0,
        use_singular_correction=False,
        source_cache=source_cache,
    )[0]
    phi_scale = max(abs(float(phi_ref_scale)), 1.0e-8)

    model = GravityPINN(
        hidden_layers=4,
        hidden_units=128,
        omega_0=float(net_cfg.get("siren_omega_0", 10.0)),
        omega=float(net_cfg.get("siren_omega", 1.0)),
        r_min=r_min,
        r_max=r_max,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1.0e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1.0e-6
    )

    history_path = outdir / "train_expdisk.csv"
    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss_total", "loss_integral", "lr", "time_s"])

    best_loss = float("inf")
    best_path = outdir / "best_expdisk.pt"
    log_r_min = np.log(r_min)
    log_r_max = np.log(r_max)

    print(f"[setup] collocation batch = {args.batch_size}")
    print(f"[setup] source grid = {args.source_nr} x {args.source_nphi}")

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()

        log_r_batch = np.random.uniform(log_r_min, log_r_max, size=args.batch_size)
        r_batch = np.exp(log_r_batch)
        phi_batch = np.random.uniform(0.0, 2.0 * np.pi, size=args.batch_size)

        phi_target = evaluate_integral_at_points(
            r_batch, phi_batch, Sigma, r_grid, phi_grid, dr, dphi,
            G=args.G, epsilon=0.0, use_singular_correction=False,
            source_cache=source_cache,
        )

        r_t = torch.tensor(r_batch, dtype=torch.float32, device=device)
        phi_t = torch.tensor(phi_batch, dtype=torch.float32, device=device)
        phi_target_t = torch.tensor(phi_target / phi_scale, dtype=torch.float32, device=device)

        optimizer.zero_grad()
        phi_pred_t = model(r_t, phi_t)
        loss_integral = torch.mean((phi_pred_t - phi_target_t) ** 2)
        loss_total = loss_integral
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        if loss_total.item() < best_loss:
            best_loss = loss_total.item()
            torch.save({"model_state_dict": model.state_dict()}, best_path)

        if epoch % 100 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            with open(history_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, loss_total.item(), loss_integral.item(), lr, elapsed])
            if epoch % 1000 == 0 or epoch == args.epochs:
                print(
                    f"Epoch {epoch:5d} | L_total={loss_total.item():.4e} | "
                    f"L_int={loss_integral.item():.4e} | lr={lr:.2e} | {elapsed:.0f}s"
                )

    train_time = time.time() - t0
    print(f"[train] best loss = {best_loss:.6e}")
    print(f"[train] time = {train_time:.1f} s")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    phi_offset = calibrate_potential_offset(
        model, phi_scale, r_grid, phi_grid, Sigma, dr, dphi,
        args.G, device, source_cache,
    )
    print(f"[gauge] recovered offset = {phi_offset:.6e}")

    return {
        "model": model,
        "r_grid": r_grid,
        "phi_grid": phi_grid,
        "Sigma": Sigma,
        "dr": dr,
        "dphi": dphi,
        "phi_scale": phi_scale,
        "phi_offset": phi_offset,
        "train_time_s": train_time,
        "r_min": r_min,
        "r_max": r_max,
        "phi_scale_ref_r": args.r0,
    }


def evaluate_experiment(args, cfg: dict, device: torch.device, results: dict, outdir: Path):
    from physics import gr_exponential_disk_analytic, phi_exponential_disk_analytic

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = int(cfg.get("evaluation", {}).get("plot_dpi", 150))
    fmt = cfg.get("evaluation", {}).get("plot_format", "png")

    r_eval = np.logspace(np.log10(results["r_min"]), np.log10(results["r_max"]), args.eval_nr)
    phi_eval = np.zeros_like(r_eval)

    model = results["model"]
    model.eval()
    with torch.enable_grad():
        r_t = torch.tensor(r_eval, dtype=torch.float32, device=device, requires_grad=True)
        phi_t = torch.tensor(phi_eval, dtype=torch.float32, device=device, requires_grad=True)
        phi_pred_norm = model(r_t, phi_t)
        g_r_pred_norm, _ = model.gravity(r_t, phi_t)

    phi_pred = results["phi_scale"] * phi_pred_norm.detach().cpu().numpy() + results["phi_offset"]
    g_r_pred = results["phi_scale"] * g_r_pred_norm.detach().cpu().numpy()

    phi_analytic = phi_exponential_disk_analytic(r_eval, args.sigma0, args.r0, args.G)
    g_r_analytic = gr_exponential_disk_analytic(r_eval, args.sigma0, args.r0, args.G)

    err_phi = rel_error_L2(phi_pred, phi_analytic)
    err_gr = rel_error_L2(g_r_pred, g_r_analytic)

    trim_mask = r_eval > args.trim_r_min
    err_phi_trim = rel_error_L2(phi_pred[trim_mask], phi_analytic[trim_mask])
    err_gr_trim = rel_error_L2(g_r_pred[trim_mask], g_r_analytic[trim_mask])

    phi_floor = 1e-8 * np.max(np.abs(phi_analytic))
    gr_floor = 1e-8 * np.max(np.abs(g_r_analytic))
    phi_signed_rel = (phi_pred - phi_analytic) / np.maximum(np.abs(phi_analytic), phi_floor)
    gr_signed_rel = (g_r_pred - g_r_analytic) / np.maximum(np.abs(g_r_analytic), gr_floor)
    phi_rel = np.abs(phi_signed_rel)
    gr_rel = np.abs(gr_signed_rel)
    trim_line_style = dict(color="gray", linestyle=":", alpha=0.8, lw=1.2)

    fig, (ax_main, ax_err) = plt.subplots(
        2,
        1,
        figsize=(7.3, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    ax_main.semilogx(r_eval, phi_analytic, "k-", lw=2, label="Analytic (Eq. B.7)")
    ax_main.semilogx(r_eval, phi_pred, "r--", lw=1.8, label="PINN (solving Eq. B.1)")
    ax_main.axvline(args.trim_r_min, **trim_line_style)
    ax_main.set_ylabel("Phi")
    ax_main.grid(True, alpha=0.3)
    ax_main.legend()

    ax_err.semilogx(r_eval, 100.0 * phi_signed_rel, "r-", lw=1.7)
    ax_err.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax_err.axhline(5.0, color="k", lw=0.6, ls="--", alpha=0.3)
    ax_err.axhline(-5.0, color="k", lw=0.6, ls="--", alpha=0.3)
    ax_err.axvline(args.trim_r_min, **trim_line_style)
    ax_err.set_xlabel("r [AU]")
    ax_err.set_ylabel("Rel. error (%)")
    ax_err.set_ylim(-15.0, 15.0)
    ax_err.grid(True, alpha=0.3)

    fig.suptitle(
        "PINN solution of Eq. B.1 vs analytic reference (Eq. B.7)\n"
        "Exponential disk potential at phi=0",
        fontsize=11,
    )
    fig.tight_layout()
    phi_plot = outdir / f"phi_compare_expdisk.{fmt}"
    fig.savefig(phi_plot, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(r_eval, np.maximum(phi_rel, 1e-14), "k-", lw=2, label="Phi")
    ax.loglog(r_eval, np.maximum(gr_rel, 1e-14), "r--", lw=1.8, label="g_r")
    ax.axvline(args.trim_r_min, **trim_line_style)
    ax.set_xlabel("r [AU]")
    ax.set_ylabel("relative error")
    ax.set_title("Relative error along phi=0")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    y_min, y_max = ax.get_ylim()
    y_text = 10.0 ** (0.5 * (np.log10(y_min) + np.log10(y_max)))
    x_text = min(args.trim_r_min * 1.12, r_eval[-1] / 1.5)
    ax.text(
        x_text,
        y_text,
        f"r > {args.trim_r_min:.1f} AU\nPhi L2={err_phi_trim:.3f}\ng_r L2={err_gr_trim:.3f}",
        fontsize=8,
        color="dimgray",
        va="center",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="lightgray", alpha=0.9),
    )
    fig.tight_layout()
    err_plot = outdir / f"relative_error_expdisk.{fmt}"
    fig.savefig(err_plot, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogx(r_eval, g_r_analytic, "k-", lw=2, label="analytic")
    ax.semilogx(r_eval, g_r_pred, "r--", lw=1.8, label="PINN")
    ax.axhline(0.0, color="k", lw=0.9, alpha=0.5, linestyle="-", zorder=3)
    ax.axvspan(results["r_min"], args.trim_r_min, alpha=0.08, color="darkblue", zorder=1)
    ax.axvline(args.trim_r_min, color="darkblue", lw=1.5, linestyle="--", alpha=0.8, zorder=5)
    gr_plot_values = np.concatenate([g_r_analytic, g_r_pred])
    gr_plot_values = gr_plot_values[np.isfinite(gr_plot_values)]
    y_text = (np.max(gr_plot_values) - 0.12 * (np.max(gr_plot_values) - np.min(gr_plot_values))) if gr_plot_values.size > 0 else 0.0
    x_text = min(args.trim_r_min * 1.07, r_eval[-1] / 1.8)
    ax.text(
        x_text,
        y_text,
        f"inner boundary\nartifacts\n(r < {args.trim_r_min:.0f} AU)",
        color="darkblue",
        fontsize=8.5,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="
            edgecolor="darkblue",
            alpha=0.75,
            lw=1.2,
        ),
        zorder=6,
    )
    ax.set_xlabel("r [AU]")
    ax.set_ylabel("g_r")
    ax.set_title("Radial gravity at phi=0")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    gr_plot = outdir / f"gr_compare_expdisk.{fmt}"
    fig.savefig(gr_plot, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    poster_mask = r_eval >= args.trim_r_min
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.semilogx(r_eval[poster_mask], g_r_analytic[poster_mask], "k-", lw=2.4, label="analytic")
    ax.semilogx(r_eval[poster_mask], g_r_pred[poster_mask], "r--", lw=2.0, label="PINN")
    ax.axhline(0.0, color="k", lw=0.9, alpha=0.45, linestyle="-")
    y_vals = np.concatenate([g_r_analytic[poster_mask], g_r_pred[poster_mask]])
    y_min = np.min(y_vals)
    y_max = np.max(y_vals)
    pad = 0.08 * (y_max - y_min)
    ax.set_xlim(args.trim_r_min, r_eval[-1])
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlabel("r [AU]")
    ax.set_ylabel("g_r")
    ax.set_title(f"Radial gravity at phi=0 (r >= {args.trim_r_min:.0f} AU) | RelErr={err_gr_trim:.2e}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    gr_plot_poster = outdir / f"gr_compare_expdisk_poster.{fmt}"
    fig.savefig(gr_plot_poster, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    region_specs = [
        ("r_lt_1", r_eval < 1.0),
        ("r_1_to_10", (r_eval >= 1.0) & (r_eval < 10.0)),
        ("r_ge_10", r_eval >= 10.0),
    ]
    region_rows = []
    for name, mask in region_specs:
        if np.count_nonzero(mask) == 0:
            continue
        region_rows.append(
            (
                name,
                rel_error_L2(phi_pred[mask], phi_analytic[mask]),
                rel_error_L2(g_r_pred[mask], g_r_analytic[mask]),
            )
        )

    metrics_path = outdir / "expdisk_l2_errors.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["phi_rel_l2", err_phi])
        writer.writerow(["gr_rel_l2", err_gr])
        writer.writerow(["phi_rel_l2_trim", err_phi_trim])
        writer.writerow(["gr_rel_l2_trim", err_gr_trim])
        writer.writerow(["trim_r_min", args.trim_r_min])
        writer.writerow(["phi_offset", results["phi_offset"]])
        writer.writerow(["phi_scale", results["phi_scale"]])
        writer.writerow(["phi_scale_ref_r", results["phi_scale_ref_r"]])

    region_metrics_path = outdir / "expdisk_region_errors.csv"
    with open(region_metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["region", "phi_rel_l2", "gr_rel_l2"])
        for row in region_rows:
            writer.writerow(row)

    np.savez(
        outdir / "expdisk_radial_eval.npz",
        r_eval=r_eval,
        phi_pred=phi_pred,
        phi_analytic=phi_analytic,
        g_r_pred=g_r_pred,
        g_r_analytic=g_r_analytic,
        phi_rel=phi_rel,
        gr_rel=gr_rel,
        phi_signed_rel=phi_signed_rel,
        gr_signed_rel=gr_signed_rel,
        r_min=results["r_min"],
        trim_r_min=float(args.trim_r_min),
    )

    print(f"[eval] Phi relative L2 (full) = {err_phi:.6e}")
    print(f"[eval] g_r relative L2 (full) = {err_gr:.6e}")
    print(f"[eval] Phi relative L2 (r > {args.trim_r_min:.2f} AU) = {err_phi_trim:.6e}")
    print(f"[eval] g_r relative L2 (r > {args.trim_r_min:.2f} AU) = {err_gr_trim:.6e}")
    print(f"[save] {phi_plot}")
    print(f"[save] {err_plot}")
    print(f"[save] {gr_plot}")
    print(f"[save] {gr_plot_poster}")
    print(f"[save] {metrics_path}")
    print(f"[save] {region_metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate GravityPINN on the exponential disk integral constraint.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--outdir", type=str, default="results/validate_expdisk_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--source-nr", type=int, default=64)
    parser.add_argument("--source-nphi", type=int, default=64)
    parser.add_argument("--eval-nr", type=int, default=400)
    parser.add_argument("--sigma0", type=float, default=10.0)
    parser.add_argument("--r0", type=float, default=10.0)
    parser.add_argument("--G", type=float, default=1.0)
    parser.add_argument("--trim-r-min", type=float, default=1.0)
    parser.add_argument("--gauge-weight", type=float, default=0.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = setup_device()

    print(f"[device] {device}")
    print(f"[seed] {args.seed}")
    results = train_experiment(args, cfg, device, outdir)
    evaluate_experiment(args, cfg, device, results, outdir)


if __name__ == "__main__":
    main()
