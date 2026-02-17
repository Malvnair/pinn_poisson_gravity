

import os
import sys
import yaml
import argparse
import time
import csv
import random
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path

from physics import (
    make_polar_grid, cell_sizes, build_sigma,
    compute_potential_direct, compute_gravity_fd,
    evaluate_integral_at_points,
    phi_exponential_disk_analytic, gr_exponential_disk_analytic,
    rel_error_L2, rel_error_L2_with_floor, max_error
)
from model import GravityPINN, total_loss






def load_config(path: str) -> dict:
    """Load YAML config."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def setup_device() -> torch.device:
    """Select best available device."""
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        print(f"[device] CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        dev = torch.device('mps')
        print("[device] Apple MPS")
    else:
        dev = torch.device('cpu')
        print("[device] CPU")
    return dev


class MetricsLogger:
    """CSV logger for training metrics."""
    
    def __init__(self, path: str, fieldnames: list):
        self.path = path
        self.fieldnames = fieldnames
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    def log(self, row: dict):
        with open(self.path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def set_global_seed(seed: int):
    """Set reproducible random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)






def stage0_sanity(cfg: dict) -> bool:
    """
    Validate the reference solver against analytic solutions.
    
    Test 1: Axisymmetric exponential disk — compare with Bessel function solution.
    Test 2: Axisymmetric constant disk — check g_r symmetry.
    """
    print("\n" + "="*60)
    print("STAGE 0: Reference solver sanity checks")
    print("="*60)
    
    dom = cfg['domain']
    G = 1.0 if cfg['physics']['use_code_units'] else cfg['physics']['G']
    
    
    Nr_test, Nphi_test = 64, 64
    r_1d, phi_1d, R, PHI = make_polar_grid(dom['r_min'], dom['r_max'],
                                            Nr_test, Nphi_test)
    dr, dphi = cell_sizes(r_1d, phi_1d)
    
    
    print("\n[Test 1] Exponential disk Σ = Σ₀ exp(-r/R)")
    Sigma_0, R_scale = 10.0, 10.0
    Sigma = Sigma_0 * np.exp(-R / R_scale)
    
    
    eps_test = dr[len(dr)//2]  
    Phi_ref = compute_potential_direct(
        r_1d, phi_1d, Sigma, r_1d, phi_1d, dr, dphi,
        G=G, epsilon=eps_test, use_singular_correction=False)
    
    
    Phi_analytic = phi_exponential_disk_analytic(r_1d, Sigma_0, R_scale, G)
    Phi_ref_avg = np.mean(Phi_ref, axis=1)
    
    
    err = rel_error_L2(Phi_ref_avg, Phi_analytic)
    print(f"  ε = {eps_test:.4f} (mid-grid Δr)")
    print(f"  Relative L2 error (Φ_ref vs analytic): {err:.4e}")
    
    
    if np.all(Phi_ref_avg < 0):
        print("Potential is negative everywhere (attractive)")
    else:
        print("Potential has positive values")
    
    
    phi_std = np.std(Phi_ref, axis=1)
    max_asym = np.max(phi_std / (np.abs(Phi_ref_avg) + 1e-10))
    print(f"  Max azimuthal asymmetry: {max_asym:.4e}")
    if max_asym < 0.05:
        print("  ✓ Axisymmetric (< 5% variation)")
    else:
        print(f"  ⚠ Notable asymmetry ({max_asym:.2%}) — may be grid artifact")
    
    
    print("\n[Test 2] Potential symmetry check")
    
    Phi_phi0 = Phi_ref[:, 0]
    Phi_phipi = Phi_ref[:, Nphi_test // 2]
    sym_err = rel_error_L2(Phi_phi0, Phi_phipi)
    print(f"  |Φ(r,0) - Φ(r,π)| / |Φ|: {sym_err:.4e}")
    
    passed = err < 0.2 and max_asym < 0.1  
    status = "PASSED" if passed else "FAILED"
    print(f"\n[Stage 0] {status}")
    return passed






def stage1_train(cfg: dict, case_name: str, epsilon_dr: float,
                 device: torch.device, outdir: str) -> dict:
    print("\n" + "="*60)
    print(f"STAGE 1: Training PINN — case={case_name}, ε/Δr={epsilon_dr}")
    print("="*60)
    
    dom = cfg['domain']
    G = 1.0 if cfg['physics']['use_code_units'] else cfg['physics']['G']
    pinn_cfg = cfg['pinn_modeA']
    net_cfg = pinn_cfg['network']
    train_cfg = pinn_cfg['training']
    
    
    r_1d, phi_1d, R, PHI = make_polar_grid(
        dom['r_min'], dom['r_max'], dom['N_r'], dom['N_phi'])
    dr, dphi = cell_sizes(r_1d, phi_1d)
    
    case_cfg = cfg['benchmarks'][case_name]
    Sigma = build_sigma(R, PHI, case_cfg)
    
    
    dr_mid = dr[len(dr) // 2]
    epsilon = epsilon_dr * dr_mid
    print(f"  Physical ε = {epsilon:.4f} AU (Δr_mid = {dr_mid:.4f} AU)")
    
    
    print("  Computing reference Φ (direct sum)...")
    t0 = time.time()
    Phi_ref = compute_potential_direct(
        r_1d, phi_1d, Sigma, r_1d, phi_1d, dr, dphi,
        G=G, epsilon=epsilon,
        use_singular_correction=(epsilon == 0.0 and cfg['smoothing']['use_singular_cell_correction']))
    t_ref = time.time() - t0
    print(f"  Reference solver time: {t_ref:.1f} s")
    
    
    g_r_ref, g_phi_ref = compute_gravity_fd(Phi_ref, r_1d, phi_1d)
    
    
    Phi_ref_mean = np.mean(Phi_ref)
    Phi_ref_zero = Phi_ref - Phi_ref_mean
    phi_scale = np.std(Phi_ref_zero)
    phi_scale = max(phi_scale, 1e-8)
    print(f"  Target normalization scale σ(Φ) = {phi_scale:.4e}")
    
    
    N_coll = train_cfg['batch_size']
    
    
    model = GravityPINN(
        hidden_layers=net_cfg['hidden_layers'],
        hidden_units=net_cfg['hidden_units'],
        omega_0=net_cfg['siren_omega_0'],
        omega=net_cfg['siren_omega'],
        r_min=dom['r_min'], r_max=dom['r_max']
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Network: {net_cfg['hidden_layers']} layers × {net_cfg['hidden_units']} units")
    print(f"  Total parameters: {n_params:,}")
    
    
    optimizer = optim.Adam(model.parameters(), lr=train_cfg['lr_initial'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg['epochs'], eta_min=train_cfg['lr_min'])
    
    
    log_path = os.path.join(outdir, f'train_{case_name}_eps{epsilon_dr:.2f}.csv')
    logger = MetricsLogger(log_path, ['epoch', 'loss_total', 'loss_integral',
                                       'loss_gauge', 'loss_force',
                                       'loss_smoothness', 'lr', 'time_s'])
    
    
    
    r_flat = R.ravel()
    phi_flat = PHI.ravel()
    Phi_target_flat = (Phi_ref_zero / phi_scale).ravel()
    
    
    g_r_target_flat = (g_r_ref / phi_scale).ravel()
    g_phi_target_flat = (g_phi_ref / phi_scale).ravel()
    N_grid = len(r_flat)
    
    
    loss_weights = train_cfg['loss_weights']
    best_loss = float('inf')
    patience_counter = 0
    
    print(f"\n  Training for {train_cfg['epochs']} epochs...")
    t_train_start = time.time()
    
    for epoch in range(1, train_cfg['epochs'] + 1):
        model.train()
        
        
        idx = np.random.choice(N_grid, size=min(N_coll, N_grid), replace=False)
        r_batch = torch.tensor(r_flat[idx], dtype=torch.float32, device=device)
        phi_batch = torch.tensor(phi_flat[idx], dtype=torch.float32, device=device)
        Phi_target_batch = torch.tensor(Phi_target_flat[idx], dtype=torch.float32, device=device)
        g_r_target_batch = torch.tensor(g_r_target_flat[idx], dtype=torch.float32, device=device)
        g_phi_target_batch = torch.tensor(g_phi_target_flat[idx], dtype=torch.float32, device=device)
        
        
        optimizer.zero_grad()
        L_total, components = total_loss(model, r_batch, phi_batch,
                                          Phi_target_batch, loss_weights,
                                          g_r_target=g_r_target_batch,
                                          g_phi_target=g_phi_target_batch)
        
        
        L_total.backward()
        if train_cfg.get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['grad_clip'])
        optimizer.step()
        scheduler.step()
        
        
        if epoch % 100 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - t_train_start
            logger.log({
                'epoch': epoch,
                'loss_total': f"{components['total']:.6e}",
                'loss_integral': f"{components['integral']:.6e}",
                'loss_gauge': f"{components['gauge']:.6e}",
                'loss_force': f"{components.get('force', 0.0):.6e}",
                'loss_smoothness': f"{components.get('smoothness', 0.0):.6e}",
                'lr': f"{lr:.6e}",
                'time_s': f"{elapsed:.1f}"
            })
            if epoch % 1000 == 0:
                print(f"    Epoch {epoch:5d} | L_total={components['total']:.4e} "
                      f"| L_int={components['integral']:.4e} "
                      f"| L_force={components.get('force', 0.0):.4e} "
                      f"| lr={lr:.2e} | {elapsed:.0f}s")
        
        
        if components['total'] < best_loss:
            best_loss = components['total']
            patience_counter = 0
            
            save_checkpoint(model, optimizer, epoch, best_loss,
                           os.path.join(outdir, f'best_{case_name}_eps{epsilon_dr:.2f}.pt'))
        else:
            patience_counter += 1
        
        if patience_counter >= train_cfg.get('patience', 2000):
            print(f"    Early stopping at epoch {epoch} (patience={train_cfg['patience']})")
            break
    
    t_train = time.time() - t_train_start
    print(f"  Training time: {t_train:.1f} s")
    
    
    model.eval()
    with torch.no_grad():
        r_eval_t = torch.tensor(r_flat, dtype=torch.float32, device=device)
        phi_eval_t = torch.tensor(phi_flat, dtype=torch.float32, device=device)
        Phi_pinn_flat = model(r_eval_t, phi_eval_t).cpu().numpy()
    
    Phi_pinn = (Phi_pinn_flat * phi_scale).reshape(R.shape)
    
    
    g_r_pinn, g_phi_pinn = compute_gravity_fd(Phi_pinn, r_1d, phi_1d)
    
    
    err_Phi = rel_error_L2(Phi_pinn, Phi_ref_zero)
    err_gr = rel_error_L2(g_r_pinn, g_r_ref)
    err_gphi_raw = rel_error_L2(g_phi_pinn, g_phi_ref)
    
    gphi_ref_norm = np.linalg.norm(g_phi_ref)
    gr_ref_norm = np.linalg.norm(g_r_ref)
    gphi_floor = gr_ref_norm
    err_gphi = rel_error_L2_with_floor(g_phi_pinn, g_phi_ref, floor=gphi_floor)
    
    print(f"\n  === RESULTS ===")
    print(f"  Relative L2 error (Φ):   {err_Phi:.4e}")
    print(f"  Relative L2 error (g_r): {err_gr:.4e}")
    if gphi_ref_norm < 1e-6 * max(gr_ref_norm, 1e-12):
        print(f"  Relative L2 error (g_φ | g_r scale): {err_gphi:.4e}")
    else:
        print(f"  Relative L2 error (g_φ):             {err_gphi:.4e}")
    print(f"  Reference solver time:   {t_ref:.1f} s")
    print(f"  PINN training time:      {t_train:.1f} s")
    
    return {
        'Phi_pinn': Phi_pinn,
        'Phi_ref': Phi_ref_zero,
        'g_r_pinn': g_r_pinn, 'g_phi_pinn': g_phi_pinn,
        'g_r_ref': g_r_ref, 'g_phi_ref': g_phi_ref,
        'Sigma': Sigma, 'R': R, 'PHI': PHI,
        'r_1d': r_1d, 'phi_1d': phi_1d,
        'err_Phi': err_Phi, 'err_gr': err_gr,
        'err_gphi': err_gphi, 'err_gphi_raw': err_gphi_raw,
        't_ref': t_ref, 't_train': t_train,
        'epsilon': epsilon, 'epsilon_dr': epsilon_dr,
        'phi_scale': phi_scale,
        'model': model,
    }






def stage2_evaluate(results: dict, cfg: dict, case_name: str, outdir: str):
    """
    Generate comparison plots and save error tables.
    """
    print("\n" + "="*60)
    print(f"STAGE 2: Evaluation — {case_name}")
    print("="*60)
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import SymLogNorm
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False
        print("  matplotlib not available — skipping plots")
    
    r_1d = results['r_1d']
    phi_1d = results['phi_1d']
    R = results['R']
    PHI = results['PHI']
    eps_dr = results['epsilon_dr']
    
    
    summary_path = os.path.join(outdir, f'summary_{case_name}_eps{eps_dr:.2f}.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['case', case_name])
        writer.writerow(['epsilon_dr', eps_dr])
        writer.writerow(['epsilon_AU', results['epsilon']])
        writer.writerow(['err_Phi_L2', results['err_Phi']])
        writer.writerow(['err_gr_L2', results['err_gr']])
        writer.writerow(['err_gphi_L2', results['err_gphi']])
        writer.writerow(['err_gphi_L2_scaled', results['err_gphi']])
        writer.writerow(['err_gphi_L2_raw', results.get('err_gphi_raw', results['err_gphi'])])
        writer.writerow(['t_ref_s', results['t_ref']])
        writer.writerow(['t_train_s', results['t_train']])
    print(f"  Saved: {summary_path}")
    
    if not HAS_MPL:
        return
    
    eval_cfg = cfg.get('evaluation', {})
    dpi = eval_cfg.get('plot_dpi', 150)
    fmt = eval_cfg.get('plot_format', 'png')
    
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), subplot_kw={'projection': 'polar'})
    
    vmin = min(np.min(results['Phi_ref']), np.min(results['Phi_pinn']))
    vmax = max(np.max(results['Phi_ref']), np.max(results['Phi_pinn']))
    
    im0 = axes[0].pcolormesh(PHI, R, results['Phi_ref'], cmap='RdBu_r',
                              vmin=vmin, vmax=vmax, shading='auto')
    axes[0].set_title('Reference Φ')
    axes[0].set_rscale('log')
    plt.colorbar(im0, ax=axes[0], pad=0.1)
    
    im1 = axes[1].pcolormesh(PHI, R, results['Phi_pinn'], cmap='RdBu_r',
                              vmin=vmin, vmax=vmax, shading='auto')
    axes[1].set_title('PINN Φ')
    axes[1].set_rscale('log')
    plt.colorbar(im1, ax=axes[1], pad=0.1)
    
    err_map = results['Phi_pinn'] - results['Phi_ref']
    emax = np.max(np.abs(err_map))
    im2 = axes[2].pcolormesh(PHI, R, err_map, cmap='RdBu_r',
                              vmin=-emax, vmax=emax, shading='auto')
    axes[2].set_title('Error (PINN − Ref)')
    axes[2].set_rscale('log')
    plt.colorbar(im2, ax=axes[2], pad=0.1)
    
    fig.suptitle(f'{case_name} | ε/Δr={eps_dr:.2f} | '
                 f'RelErr(Φ)={results["err_Phi"]:.2e}', fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(outdir, f'potential_2d_{case_name}_eps{eps_dr:.2f}.{fmt}')
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")
    
    
    cut_phis = eval_cfg.get('radial_cut_phis', [0.0, np.pi/2, np.pi])
    
    fig, axes = plt.subplots(1, len(cut_phis), figsize=(5 * len(cut_phis), 4))
    if len(cut_phis) == 1:
        axes = [axes]
    
    for ax, phi_cut in zip(axes, cut_phis):
        j_cut = np.argmin(np.abs(phi_1d - phi_cut))
        ax.semilogx(r_1d, results['Phi_ref'][:, j_cut], 'k-', label='Reference', lw=2)
        ax.semilogx(r_1d, results['Phi_pinn'][:, j_cut], 'r--', label='PINN', lw=1.5)
        ax.set_xlabel('r [AU]')
        ax.set_ylabel('Φ')
        ax.set_title(f'φ = {phi_cut:.2f} rad')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{case_name} — Radial cuts', fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(outdir, f'radial_cuts_{case_name}_eps{eps_dr:.2f}.{fmt}')
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")
    
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    j0 = 0  
    axes[0].semilogx(r_1d, results['g_r_ref'][:, j0], 'k-', label='Reference', lw=2)
    axes[0].semilogx(r_1d, results['g_r_pinn'][:, j0], 'r--', label='PINN', lw=1.5)
    axes[0].set_xlabel('r [AU]')
    axes[0].set_ylabel('g_r')
    axes[0].set_title(f'Radial gravity (φ=0) | RelErr={results["err_gr"]:.2e}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].semilogx(r_1d, results['g_phi_ref'][:, j0], 'k-', label='Reference', lw=2)
    axes[1].semilogx(r_1d, results['g_phi_pinn'][:, j0], 'r--', label='PINN', lw=1.5)
    axes[1].set_xlabel('r [AU]')
    axes[1].set_ylabel('g_φ')
    axes[1].set_title(f'Azimuthal gravity (φ=0) | ScaledErr={results["err_gphi"]:.2e}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(outdir, f'gravity_{case_name}_eps{eps_dr:.2f}.{fmt}')
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")
    
    
    np.savez(os.path.join(outdir, f'data_{case_name}_eps{eps_dr:.2f}.npz'),
             r=r_1d, phi=phi_1d, R=R, PHI=PHI,
             Sigma=results['Sigma'],
             Phi_ref=results['Phi_ref'], Phi_pinn=results['Phi_pinn'],
             g_r_ref=results['g_r_ref'], g_r_pinn=results['g_r_pinn'],
             g_phi_ref=results['g_phi_ref'], g_phi_pinn=results['g_phi_pinn'])
    print(f"  Saved: data_{case_name}_eps{eps_dr:.2f}.npz")






def main():
    parser = argparse.ArgumentParser(description='Gravity-PINN Trainer')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--case', type=str, default='B1_smooth_disk',
                        help='Benchmark case name')
    parser.add_argument('--epsilon', type=float, default=None,
                        help='Smoothing length in Δr units (overrides config)')
    parser.add_argument('--skip-sanity', action='store_true',
                        help='Skip Stage 0 sanity checks')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override training epochs from config')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override collocation batch size from config')
    parser.add_argument('--force-weight', type=float, default=None,
                        help='Override force loss weight from config')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible training')
    args = parser.parse_args()
    
    
    cfg = load_config(args.config)

    
    if args.epochs is not None:
        cfg['pinn_modeA']['training']['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        cfg['pinn_modeA']['training']['batch_size'] = int(args.batch_size)
    if args.force_weight is not None:
        cfg['pinn_modeA']['training']['loss_weights']['force'] = float(args.force_weight)

    if args.seed is not None:
        set_global_seed(int(args.seed))
    
    
    os.makedirs(args.outdir, exist_ok=True)
    
    
    device = setup_device()
    
    
    eps_dr = args.epsilon if args.epsilon is not None else cfg['smoothing']['default_epsilon_dr']
    
    print("="*60)
    print("GRAVITY-PINN: Thin-Disk Poisson Integral Solver")
    print("="*60)
    print(f"  Case:    {args.case}")
    print(f"  ε/Δr:    {eps_dr}")
    print(f"  Device:  {device}")
    print(f"  Output:  {args.outdir}")
    if args.seed is not None:
        print(f"  Seed:    {args.seed}")
    
    
    if not args.skip_sanity:
        passed = stage0_sanity(cfg)
        if not passed:
            print("\nSanity checks had issues")
    
    
    results = stage1_train(cfg, args.case, eps_dr, device, args.outdir)
    
    
    stage2_evaluate(results, cfg, args.case, args.outdir)
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)


if __name__ == '__main__':
    main()
