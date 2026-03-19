from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_data(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def build_cartesian(R: np.ndarray, PHI: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return R * np.cos(PHI), R * np.sin(PHI)


def close_azimuthal_periodicity(arr: np.ndarray) -> np.ndarray:
    return np.concatenate([arr, arr[:, :1]], axis=1)


def save_polar_error_map(data: dict[str, np.ndarray], outdir: Path) -> Path:
    err = data["Phi_pinn"] - data["Phi_ref"]
    err_abs = np.abs(err)
    vmax = float(np.nanpercentile(err_abs, 99.5))
    linthresh = max(vmax * 1.0e-2, 1.0)

    fig, ax = plt.subplots(figsize=(12.5, 8.5), subplot_kw={"projection": "polar"})
    pcm = ax.pcolormesh(
        data["PHI"],
        data["R"],
        err,
        shading="auto",
        cmap="RdBu_r",
        norm=SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax),
    )
    ax.set_rscale("log")
    ax.set_title(r"PINN Potential Error Field: $\Delta \Phi = \Phi_{\rm PINN} - \Phi_{\rm ref}$", pad=24)
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(pcm, ax=ax, pad=0.12, shrink=0.95)
    cbar.set_label(r"$\Delta \Phi$")
    fig.tight_layout()
    path = outdir / "poster_phi_error_polar.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_phi_surface_plot(data: dict[str, np.ndarray], outdir: Path) -> Path:
    X, Y = build_cartesian(data["R"], data["PHI"])
    X = close_azimuthal_periodicity(X)
    Y = close_azimuthal_periodicity(Y)
    z_ref = close_azimuthal_periodicity(data["Phi_ref"] / 1.0e5)
    z_pinn = close_azimuthal_periodicity(data["Phi_pinn"] / 1.0e5)

    fig = plt.figure(figsize=(11.5, 8.5))
    ax = fig.add_subplot(111, projection="3d")

    surf_ref = ax.plot_surface(
        X, Y, z_ref,
        rstride=3, cstride=3,
        cmap="Blues",
        alpha=0.28,
        linewidth=0.0,
        antialiased=False,
        edgecolor="none",
    )
    surf_pinn = ax.plot_surface(
        X, Y, z_pinn,
        rstride=3, cstride=3,
        cmap="magma",
        alpha=0.92,
        linewidth=0.0,
        antialiased=False,
        edgecolor="none",
    )

    ax.view_init(elev=33, azim=-58)
    ax.set_xlabel("x [AU]", labelpad=10)
    ax.set_ylabel("y [AU]", labelpad=10)
    ax.set_zlabel(r"$\Phi / 10^5$", labelpad=10)
    ax.set_title(r"3D Potential Surface: PINN vs Reference", pad=18)
    ax.legend(
        handles=[
            Patch(facecolor=plt.cm.magma(0.75), edgecolor="none", label="PINN"),
            Patch(facecolor=plt.cm.Blues(0.65), edgecolor="none", alpha=0.5, label="Reference"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
    )
    cbar = fig.colorbar(surf_pinn, ax=ax, shrink=0.62, pad=0.08)
    cbar.set_label(r"$\Phi_{\rm PINN} / 10^5$")
    fig.tight_layout()
    path = outdir / "poster_phi_surface_overlay.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_vector_overlay(data: dict[str, np.ndarray], outdir: Path) -> Path:
    X, Y = build_cartesian(data["R"], data["PHI"])
    X_fill = close_azimuthal_periodicity(X)
    Y_fill = close_azimuthal_periodicity(Y)
    Phi = data["Phi_pinn"]
    Phi_fill = close_azimuthal_periodicity(Phi)
    Phi_ref_fill = close_azimuthal_periodicity(data["Phi_ref"])
    g_r = data["g_r_pinn"]
    g_phi = data["g_phi_pinn"]

    gx = g_r * np.cos(data["PHI"]) - g_phi * np.sin(data["PHI"])
    gy = g_r * np.sin(data["PHI"]) + g_phi * np.cos(data["PHI"])
    mag = np.hypot(gx, gy)
    gx_dir = gx / np.maximum(mag, 1.0e-12)
    gy_dir = gy / np.maximum(mag, 1.0e-12)

    sample_radii = np.geomspace(10.0, float(data["r"][-1]) * 0.88, 5)
    sample_phis = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    r_idx = np.unique([int(np.argmin(np.abs(data["r"] - rv))) for rv in sample_radii])
    phi_idx = np.unique([int(np.argmin(np.abs(data["phi"] - pv))) for pv in sample_phis])
    ii = np.ix_(r_idx, phi_idx)
    arrow_len = np.clip(0.085 * data["R"][ii], 5.2, 9.5)
    u = -gx_dir[ii] * arrow_len
    v = -gy_dir[ii] * arrow_len

    fig, ax = plt.subplots(figsize=(10.5, 10.5))
    contour = ax.contourf(X_fill, Y_fill, Phi_fill, levels=72, cmap="magma")
    ax.contour(
        X_fill,
        Y_fill,
        Phi_ref_fill,
        levels=12,
        colors="white",
        linewidths=0.8,
        alpha=0.6,
        linestyles="--",
    )
    ax.quiver(
        X[ii], Y[ii],
        u, v,
        color="white",
        alpha=0.78,
        pivot="mid",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0026,
        headwidth=3.2,
        headlength=4.2,
        headaxislength=3.8,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.set_title(r"PINN Potential with Reference Contours and Gravity Vectors", pad=16)
    cbar = fig.colorbar(contour, ax=ax, pad=0.02, shrink=0.88)
    cbar.set_label(r"$\Phi_{\rm PINN}$")
    fig.tight_layout()
    path = outdir / "poster_phi_quiver.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Create poster-style plots from saved annulus NPZ data.")
    parser.add_argument(
        "--data",
        type=str,
        default="results/B5_annulus_for_figB2_v2/data_B5_constant_density_eps0.00.npz",
        help="Path to saved NPZ data from a train.py run.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results/B5_annulus_for_figB2_v2/poster_plots",
        help="Directory for poster plot outputs.",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_data(data_path)
    paths = [
        save_polar_error_map(data, outdir),
        save_phi_surface_plot(data, outdir),
        save_vector_overlay(data, outdir),
    ]

    print(f"Loaded data: {data_path}")
    for path in paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
