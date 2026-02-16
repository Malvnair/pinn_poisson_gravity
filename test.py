
import os
import sys
import argparse
import time
import csv
import subprocess
import numpy as np


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from physics import (
    make_polar_grid, cell_sizes, build_sigma,
    sigma_exponential, sigma_exponential_perturbed, sigma_clump, sigma_constant,
    compute_potential_direct, compute_gravity_fd,
    evaluate_integral_at_points,
    phi_exponential_disk_analytic, gr_exponential_disk_analytic,
    gr_constant_disk_analytic, phi_constant_disk_axisym_quadrature,
    gr_constant_disk_axisym_quadrature, gr_constant_disk_axisym_quadrature_softened,
    rel_error_L2, max_error
)






class TestResults:
    """Track pass/fail status."""
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0
    
    def record(self, name: str, passed: bool, msg: str = ""):
        self.tests.append((name, passed, msg))
        if passed:
            self.passed += 1
            print(f"  ✓ {name}")
        else:
            self.failed += 1
            print(f"  ✗ {name}: {msg}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*50}")
        print(f"  {self.passed}/{total} tests passed")
        if self.failed > 0:
            print(f"  {self.failed} FAILED:")
            for name, passed, msg in self.tests:
                if not passed:
                    print(f"    - {name}: {msg}")
        print(f"{'='*50}")
        return self.failed == 0


_TORCH_RUNTIME_CHECK = None


def torch_runtime_available() -> tuple[bool, str]:
    global _TORCH_RUNTIME_CHECK
    if _TORCH_RUNTIME_CHECK is not None:
        return _TORCH_RUNTIME_CHECK

    cmd = [sys.executable, "-c", "import torch; print(torch.__version__)"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception as e:
        _TORCH_RUNTIME_CHECK = (False, str(e))
        return _TORCH_RUNTIME_CHECK

    if proc.returncode == 0:
        version = (proc.stdout or "").strip() or "unknown"
        _TORCH_RUNTIME_CHECK = (True, f"torch {version}")
    else:
        msg = (proc.stderr or "").strip()
        if not msg:
            msg = f"torch import failed with return code {proc.returncode}"
        _TORCH_RUNTIME_CHECK = (False, msg)
    return _TORCH_RUNTIME_CHECK






def test_grid(results: TestResults):
    """Test polar grid construction."""
    print("\n--- Grid construction ---")
    
    r_1d, phi_1d, R, PHI = make_polar_grid(0.2, 100.0, 64, 64)
    
    
    results.record("Grid r_1d shape", r_1d.shape == (64,), f"got {r_1d.shape}")
    results.record("Grid phi_1d shape", phi_1d.shape == (64,), f"got {phi_1d.shape}")
    results.record("Grid R shape", R.shape == (64, 64), f"got {R.shape}")
    
    
    results.record("Grid r_min", np.isclose(r_1d[0], 0.2, rtol=1e-3))
    results.record("Grid r_max", np.isclose(r_1d[-1], 100.0, rtol=1e-3))
    results.record("Grid phi_min", np.isclose(phi_1d[0], 0.0, atol=0.01))
    results.record("Grid phi periodic", phi_1d[-1] < 2*np.pi, 
                    f"last φ = {phi_1d[-1]:.4f}")
    
    
    log_ratio = np.log(r_1d[1] / r_1d[0])
    log_ratio_end = np.log(r_1d[-1] / r_1d[-2])
    results.record("Grid log-spaced", np.isclose(log_ratio, log_ratio_end, rtol=0.01),
                    f"ratio start={log_ratio:.4f}, end={log_ratio_end:.4f}")
    
    
    dr, dphi = cell_sizes(r_1d, phi_1d)
    results.record("Cell dr positive", np.all(dr > 0))
    results.record("Cell dphi positive", dphi > 0)
    results.record("Cell dphi value", np.isclose(dphi, 2*np.pi/64, rtol=1e-3))


def test_sigma_models(results: TestResults):
    """Test surface density models."""
    print("\n--- Σ models ---")
    
    r_1d, phi_1d, R, PHI = make_polar_grid(0.2, 100.0, 32, 32)
    
    
    S1 = sigma_exponential(R, PHI, 10.0, 10.0)
    results.record("Σ_exp shape", S1.shape == (32, 32))
    results.record("Σ_exp positive", np.all(S1 > 0))
    results.record("Σ_exp peak at r_min", S1[0, 0] > S1[-1, 0],
                    "should decrease with r")
    results.record("Σ_exp axisymmetric",
                    np.allclose(S1[:, 0], S1[:, 16], rtol=1e-10))
    
    
    S2 = sigma_exponential_perturbed(R, PHI, 10.0, 10.0, 0.3, 2)
    results.record("Σ_pert shape", S2.shape == (32, 32))
    results.record("Σ_pert not axisymmetric",
                    not np.allclose(S2[:, 0], S2[:, 8], rtol=1e-3))
    
    
    S3 = sigma_clump(R, PHI, 1.0, 50.0, 20.0, np.pi, 3.0, 0.3)
    results.record("Σ_clump > bg", np.max(S3) > 1.0)
    results.record("Σ_clump bg level", np.min(S3) >= 1.0 - 1e-10)
    
    
    S5 = sigma_constant(R, PHI, 100.0, 1.0, 50.0)
    in_mask = (R >= 1.0) & (R <= 50.0)
    out_mask = ~in_mask
    results.record("Σ_const inner", np.allclose(S5[in_mask], 100.0))
    results.record("Σ_const outer", np.allclose(S5[out_mask], 0.0))
    
    
    cfg = {'type': 'exponential', 'Sigma_0': 10.0, 'R_scale': 10.0}
    S_disp = build_sigma(R, PHI, cfg)
    results.record("build_sigma dispatch", np.allclose(S_disp, S1))






def test_reference_solver(results: TestResults):
    """Validate reference solver against analytic solutions."""
    print("\n--- Reference solver validation ---")
    
    
    Nr, Nphi = 48, 48
    r_1d, phi_1d, R, PHI = make_polar_grid(0.5, 50.0, Nr, Nphi)
    dr, dphi = cell_sizes(r_1d, phi_1d)
    
    G = 1.0
    Sigma_0, R_scale = 10.0, 10.0
    
    
    Sigma = sigma_exponential(R, PHI, Sigma_0, R_scale)
    
    
    eps = dr[Nr // 2]
    
    print(f"  Computing direct sum (Nr={Nr}, Nφ={Nphi}, ε={eps:.3f})...")
    t0 = time.time()
    Phi_num = compute_potential_direct(
        r_1d, phi_1d, Sigma, r_1d, phi_1d, dr, dphi,
        G=G, epsilon=eps, use_singular_correction=False)
    t1 = time.time()
    print(f"  Direct sum time: {t1-t0:.1f} s")
    
    
    Phi_analytic = phi_exponential_disk_analytic(r_1d, Sigma_0, R_scale, G)
    
    
    Phi_num_avg = np.mean(Phi_num, axis=1)
    
    
    trim = 5
    err = rel_error_L2(Phi_num_avg[trim:-trim], Phi_analytic[trim:-trim])
    results.record(f"Φ exp-disk L2 err ({err:.2e})", err < 0.3,
                    f"err={err:.4e} (threshold 0.3 for coarse grid + ε smoothing)")
    
    
    results.record("Φ negative (attractive)", np.all(Phi_num_avg < 0),
                    f"min={np.min(Phi_num_avg):.4f}")
    
    
    abs_Phi = np.abs(Phi_num_avg)
    
    trim_inner = Nr // 4  
    diffs = np.diff(abs_Phi[trim_inner:-trim])
    frac_decreasing = np.sum(diffs < 0) / len(diffs)
    results.record("Φ mostly decreasing |Φ(r)|", frac_decreasing > 0.9,
                    f"frac decreasing = {frac_decreasing:.2f}")
    
    
    phi_var = np.std(Phi_num, axis=1) / (np.abs(Phi_num_avg) + 1e-10)
    max_var = np.max(phi_var[trim:-trim])
    results.record("Φ axisymmetric (<10% var)", max_var < 0.1,
                    f"max φ-variation = {max_var:.4e}")
    
    
    g_r_num, g_phi_num = compute_gravity_fd(Phi_num, r_1d, phi_1d)
    
    
    g_phi_max = np.max(np.abs(g_phi_num[trim:-trim, :]))
    g_r_max = np.max(np.abs(g_r_num[trim:-trim, :]))
    ratio = g_phi_max / (g_r_max + 1e-10)
    results.record("g_φ ≈ 0 for axisym disk", ratio < 0.1,
                    f"|g_φ|/|g_r| = {ratio:.4e}")
    
    
    
    g_r_avg = np.mean(g_r_num, axis=1)
    trim_inner = max(trim, Nr // 6)
    inward_ok = np.all(g_r_avg[trim_inner:-trim] < 0)
    results.record("g_r < 0 away from inner edge", inward_ok,
                    f"max g_r (trimmed) = {np.max(g_r_avg[trim_inner:-trim]):.4e}")


def test_singular_cell_correction(results: TestResults):
    """Test singular cell correction vs smoothed kernel."""
    print("\n--- Singular cell correction ---")
    
    Nr, Nphi = 32, 32
    r_1d, phi_1d, R, PHI = make_polar_grid(1.0, 50.0, Nr, Nphi)
    dr, dphi = cell_sizes(r_1d, phi_1d)
    
    Sigma = sigma_exponential(R, PHI, 10.0, 10.0)
    
    
    print("  Computing ε=0 with singular cell correction...")
    t0 = time.time()
    Phi_corr = compute_potential_direct(
        r_1d, phi_1d, Sigma, r_1d, phi_1d, dr, dphi,
        G=1.0, epsilon=0.0, use_singular_correction=True)
    t1 = time.time()
    print(f"  Time: {t1-t0:.1f} s")
    
    
    eps = 0.5 * dr[Nr // 2]
    print(f"  Computing ε={eps:.4f} (smoothed)...")
    t0 = time.time()
    Phi_smooth = compute_potential_direct(
        r_1d, phi_1d, Sigma, r_1d, phi_1d, dr, dphi,
        G=1.0, epsilon=eps, use_singular_correction=False)
    t1 = time.time()
    print(f"  Time: {t1-t0:.1f} s")
    
    
    
    
    
    corr_deeper = np.mean(Phi_corr) < np.mean(Phi_smooth)
    results.record("Singular corr gives deeper Φ than smoothed", corr_deeper,
                    f"mean_corr={np.mean(Phi_corr):.2f}, mean_smooth={np.mean(Phi_smooth):.2f}")
    
    
    results.record("Φ_corr negative", np.all(np.mean(Phi_corr, axis=1) < 0))
    results.record("Φ_smooth negative", np.all(np.mean(Phi_smooth, axis=1) < 0))


def test_constant_annulus_quadrature_validator(results: TestResults):
    print("\n--- Constant annulus 1D quadrature validator ---")

    Nr = 64
    r_min, r_max = 0.2, 100.0
    Sigma_0 = 100.0
    r_1d = np.logspace(np.log10(r_min), np.log10(r_max), Nr)

    g_r_quad = gr_constant_disk_axisym_quadrature(
        r_1d, Sigma_0=Sigma_0, r_in=r_min, r_out=r_max, G=1.0
    )
    g_r_ref_fn = gr_constant_disk_analytic(
        r_1d, Sigma_0=Sigma_0, r_in=r_min, r_out=r_max, G=1.0
    )
    
    phi_quad = phi_constant_disk_axisym_quadrature(
        r_1d, Sigma_0=Sigma_0, r_in=r_min, r_out=r_max, G=1.0
    )

    results.record("B5 quadrature Φ finite", np.all(np.isfinite(phi_quad)))
    results.record("B5 quadrature g_r finite", np.all(np.isfinite(g_r_quad)))
    results.record("B5 quadrature Φ negative", np.all(phi_quad < 0.0))

    
    
    inner = (r_1d > 0.25) & (r_1d < 1.0)
    outer = (r_1d > 10.0) & (r_1d < 95.0)
    has_outward_inner = np.any(g_r_quad[inner] > 0.0)
    has_inward_outer = np.any(g_r_quad[outer] < 0.0)
    results.record("B5 quadrature sign pattern", has_outward_inner and has_inward_outer)
    results.record("B5 analytic API matches quadrature", np.allclose(g_r_ref_fn, g_r_quad))


def test_constant_annulus_matched_epsilon_reference(results: TestResults):
    print("\n--- Constant annulus matched-ε REF check ---")

    Nr, Nphi = 48, 64
    dom_rmin, dom_rmax = 0.2, 100.0
    Sigma_0 = 100.0
    r_in, r_out = 5.0, 20.0

    r_1d, phi_1d, R, PHI = make_polar_grid(dom_rmin, dom_rmax, Nr, Nphi)
    dr, dphi = cell_sizes(r_1d, phi_1d)
    Sigma = sigma_constant(R, PHI, Sigma_0=Sigma_0, r_in=r_in, r_out=r_out)

    eps = 0.5 * dr[Nr // 2]
    Phi_ref = compute_potential_direct(
        r_1d, phi_1d, Sigma, r_1d, phi_1d, dr, dphi,
        G=1.0, epsilon=eps, use_singular_correction=False
    )
    g_r_ref, _ = compute_gravity_fd(Phi_ref, r_1d, phi_1d)
    g_r_ref_avg = np.mean(g_r_ref, axis=1)

    g_r_quad = gr_constant_disk_axisym_quadrature_softened(
        r_1d, Sigma_0=Sigma_0, r_in=r_in, r_out=r_out, epsilon=eps,
        G=1.0, n_rp=1024, n_phi=513
    )

    mask = (r_1d > 1.0) & (r_1d < 50.0)
    err = rel_error_L2(g_r_ref_avg[mask], g_r_quad[mask])
    sign_mismatch = np.mean(np.sign(g_r_ref_avg[mask]) != np.sign(g_r_quad[mask]))

    results.record("Matched-ε annulus REF vs quadrature (g_r)", err < 0.30,
                   f"rel err = {err:.3e}")
    results.record("Matched-ε annulus sign consistency", sign_mismatch < 0.05,
                   f"sign mismatch frac = {sign_mismatch:.3f}")


def test_point_integral_matches_direct(results: TestResults):
    print("\n--- Pointwise integral vs direct-sum ---")

    Nr, Nphi = 24, 24
    r_1d, phi_1d, R, PHI = make_polar_grid(1.0, 20.0, Nr, Nphi)
    dr, dphi = cell_sizes(r_1d, phi_1d)
    Sigma = sigma_exponential(R, PHI, 10.0, 8.0)

    Phi_direct = compute_potential_direct(
        r_1d, phi_1d, Sigma, r_1d, phi_1d, dr, dphi,
        G=1.0, epsilon=0.0, use_singular_correction=True
    )

    sample_idx = [(3, 4), (10, 7), (18, 0)]
    r_q = np.array([r_1d[i] for i, _ in sample_idx], dtype=np.float64)
    phi_q = np.array([phi_1d[j] for _, j in sample_idx], dtype=np.float64)
    Phi_eval = evaluate_integral_at_points(
        r_q, phi_q, Sigma, r_1d, phi_1d, dr, dphi,
        G=1.0, epsilon=0.0, use_singular_correction=True
    )
    Phi_ref = np.array([Phi_direct[i, j] for i, j in sample_idx], dtype=np.float64)

    err = rel_error_L2(Phi_eval, Phi_ref)
    results.record("Point-eval matches direct (ε=0 + correction)", err < 1e-10,
                    f"rel err = {err:.3e}")






def test_pinn_smoke(results: TestResults):
    """Quick test that PINN model can be instantiated and trained for a few steps."""
    print("\n--- PINN smoke test ---")

    torch_ok, torch_msg = torch_runtime_available()
    if not torch_ok:
        results.record("PyTorch runtime unavailable (skip PINN smoke)", True, torch_msg)
        return
    
    try:
        import torch
        from model import GravityPINN, total_loss
    except ImportError as e:
        results.record("PyTorch import", False, str(e))
        return
    
    results.record("PyTorch import", True)
    
    
    model = GravityPINN(
        hidden_layers=2,
        hidden_units=16,
        omega_0=10.0,
        omega=1.0,
        r_min=0.2,
        r_max=100.0
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    results.record("Model instantiation", True, f"{n_params} params")
    
    
    r_test = torch.tensor([1.0, 10.0, 50.0], dtype=torch.float32)
    phi_test = torch.tensor([0.0, np.pi/2, np.pi], dtype=torch.float32)
    
    Phi_out = model(r_test, phi_test)
    results.record("Forward pass shape", Phi_out.shape == (3,),
                    f"got {Phi_out.shape}")
    results.record("Forward pass finite", torch.all(torch.isfinite(Phi_out)).item())
    
    
    r_grad = torch.tensor([1.0, 10.0, 50.0], dtype=torch.float32, requires_grad=True)
    phi_grad = torch.tensor([0.0, np.pi/2, np.pi], dtype=torch.float32, requires_grad=True)
    
    g_r, g_phi = model.gravity(r_grad, phi_grad)
    results.record("Gravity g_r shape", g_r.shape == (3,))
    results.record("Gravity g_φ shape", g_phi.shape == (3,))
    results.record("Gravity finite", 
                    torch.all(torch.isfinite(g_r)).item() and 
                    torch.all(torch.isfinite(g_phi)).item())
    
    
    Phi_target = torch.tensor([-100.0, -50.0, -20.0], dtype=torch.float32)
    loss_weights = {'integral': 1.0, 'gauge': 10.0, 'smoothness': 0.0}
    
    L_total, components = total_loss(model, r_test, phi_test, Phi_target, loss_weights)
    results.record("Loss computation", torch.isfinite(L_total).item(),
                    f"L_total = {L_total.item():.4e}")
    results.record("Loss components", 
                    'integral' in components and 'gauge' in components)

    
    g_r_t = torch.tensor([-1.0, -0.5, -0.2], dtype=torch.float32)
    g_phi_t = torch.tensor([0.0, 0.1, -0.1], dtype=torch.float32)
    loss_weights_force = {'integral': 1.0, 'gauge': 10.0, 'force': 0.5, 'smoothness': 0.0}
    L_force_total, comp_force = total_loss(
        model, r_test, phi_test, Phi_target, loss_weights_force,
        g_r_target=g_r_t, g_phi_target=g_phi_t
    )
    results.record("Force loss computation", torch.isfinite(L_force_total).item(),
                    f"L_total(force) = {L_force_total.item():.4e}")
    results.record("Force loss component present", 'force' in comp_force)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    loss_before = L_total.item()
    for _ in range(10):
        optimizer.zero_grad()
        L, _ = total_loss(model, r_test, phi_test, Phi_target, loss_weights)
        L.backward()
        optimizer.step()
    
    with torch.no_grad():
        L_after, _ = total_loss(model, r_test, phi_test, Phi_target, loss_weights)
    loss_after = L_after.item()
    
    results.record("Training reduces loss", loss_after < loss_before,
                    f"before={loss_before:.4e}, after={loss_after:.4e}")
    
    
    with torch.no_grad():
        r_rep = torch.tensor([10.0, 10.0], dtype=torch.float32)
        phi_rep = torch.tensor([0.0, 2*np.pi - 0.001], dtype=torch.float32)
        Phi_rep = model(r_rep, phi_rep)
    
    period_err = abs(Phi_rep[0].item() - Phi_rep[1].item())
    results.record("Periodicity (Φ at φ=0 ≈ φ=2π)", period_err < 0.1,
                    f"|ΔΦ| = {period_err:.4e}")






def test_epsilon_sweep(results: TestResults, epsilon_list=None):
    """
    Train PINN for multiple ε values and compare errors.
    This is a longer-running benchmark test.
    """
    print("\n--- ε-sweep benchmark ---")

    torch_ok, torch_msg = torch_runtime_available()
    if not torch_ok:
        results.record("PyTorch runtime unavailable (skip ε-sweep)", True, torch_msg)
        return
    
    try:
        import torch
        from model import GravityPINN, total_loss
    except ImportError as e:
        results.record("ε-sweep: PyTorch import", False, str(e))
        return
    
    if epsilon_list is None:
        epsilon_list = [0.25, 0.5, 1.0, 2.0]  
    
    
    
    Nr, Nphi = 32, 32
    r_1d, phi_1d, R, PHI = make_polar_grid(0.5, 50.0, Nr, Nphi)
    dr, dphi = cell_sizes(r_1d, phi_1d)
    dr_mid = dr[Nr // 2]
    
    G = 1.0
    Sigma_0, R_scale = 10.0, 10.0
    Sigma = sigma_exponential(R, PHI, Sigma_0, R_scale)
    
    
    Phi_analytic = phi_exponential_disk_analytic(r_1d, Sigma_0, R_scale, G)
    
    sweep_results = []
    
    for eps_dr in epsilon_list:
        epsilon = eps_dr * dr_mid
        print(f"\n  ε/Δr = {eps_dr:.2f} (ε = {epsilon:.4f})")
        
        
        Phi_ref = compute_potential_direct(
            r_1d, phi_1d, Sigma, r_1d, phi_1d, dr, dphi,
            G=G, epsilon=epsilon, use_singular_correction=False)
        Phi_ref_mean = np.mean(Phi_ref)
        Phi_ref_zero = Phi_ref - Phi_ref_mean
        
        
        Phi_ref_avg = np.mean(Phi_ref, axis=1)
        trim = 3
        ref_err = rel_error_L2(Phi_ref_avg[trim:-trim], Phi_analytic[trim:-trim])
        
        
        device = torch.device('cpu')
        model = GravityPINN(
            hidden_layers=3,
            hidden_units=64,
            omega_0=10.0,
            omega=1.0,
            r_min=0.5, r_max=50.0
        ).to(device)

        
        phi_scale = np.std(Phi_ref_zero)
        phi_scale = max(phi_scale, 1e-8)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        n_epochs = 1200
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=1e-5)

        r_flat = R.ravel()
        phi_flat = PHI.ravel()
        Phi_target_flat = (Phi_ref_zero / phi_scale).ravel()
        
        loss_weights = {'integral': 1.0, 'gauge': 10.0, 'smoothness': 0.0}
        
        t0 = time.time()
        for epoch in range(1, n_epochs + 1):
            model.train()
            
            
            r_b = torch.tensor(r_flat, dtype=torch.float32, device=device)
            phi_b = torch.tensor(phi_flat, dtype=torch.float32, device=device)
            Phi_b = torch.tensor(Phi_target_flat, dtype=torch.float32, device=device)
            
            optimizer.zero_grad()
            L, _ = total_loss(model, r_b, phi_b, Phi_b, loss_weights)
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        t_train = time.time() - t0
        
        
        model.eval()
        with torch.no_grad():
            r_t = torch.tensor(r_flat, dtype=torch.float32, device=device)
            phi_t = torch.tensor(phi_flat, dtype=torch.float32, device=device)
            Phi_pinn_flat = model(r_t, phi_t).cpu().numpy()
        
        Phi_pinn = (Phi_pinn_flat * phi_scale).reshape(R.shape)
        pinn_err = rel_error_L2(Phi_pinn, Phi_ref_zero)
        
        print(f"    Ref vs analytic: {ref_err:.4e}")
        print(f"    PINN vs ref:     {pinn_err:.4e}")
        print(f"    Train time:      {t_train:.1f} s")
        
        sweep_results.append({
            'eps_dr': eps_dr,
            'epsilon': epsilon,
            'ref_err': ref_err,
            'pinn_err': pinn_err,
            't_train': t_train,
        })
    
    
    if len(sweep_results) >= 2:
        
        ref_errs = [r['ref_err'] for r in sweep_results]
        eps_vals = [r['eps_dr'] for r in sweep_results]
        
        
        all_ref_ok = all(e < 0.5 for e in ref_errs)
        results.record("ε-sweep: all ref errors < 0.5", all_ref_ok,
                        f"ref_errs = {[f'{e:.3f}' for e in ref_errs]}")
        
        
        pinn_errs = [r['pinn_err'] for r in sweep_results]
        all_pinn_ok = all(e < 0.02 for e in pinn_errs)
        results.record("ε-sweep: all PINN errors < 0.02", all_pinn_ok,
                        f"pinn_errs = {[f'{e:.3f}' for e in pinn_errs]}")
    
    
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, 'epsilon_sweep_test.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['eps_dr', 'epsilon', 'ref_err', 'pinn_err', 't_train'])
        writer.writeheader()
        for row in sweep_results:
            writer.writerow({k: f"{v:.6e}" if isinstance(v, float) else v for k, v in row.items()})
    print(f"\n  Saved sweep results: {csv_path}")
    results.record("ε-sweep: results saved", os.path.exists(csv_path))






def test_mini_pipeline(results: TestResults):
    print("\n--- Mini pipeline integration test ---")

    torch_ok, torch_msg = torch_runtime_available()
    if not torch_ok:
        results.record("PyTorch runtime unavailable (skip mini pipeline)", True, torch_msg)
        return
    
    try:
        import torch
        from model import GravityPINN, total_loss
    except ImportError as e:
        results.record("Mini pipeline: PyTorch", False, str(e))
        return
    
    
    Nr, Nphi = 24, 24
    r_1d, phi_1d, R, PHI = make_polar_grid(1.0, 30.0, Nr, Nphi)
    dr, dphi = cell_sizes(r_1d, phi_1d)
    
    
    Sigma = sigma_exponential_perturbed(R, PHI, 10.0, 10.0, 0.3, 2)
    results.record("Pipeline: Σ non-axisym", 
                    np.std(Sigma, axis=1).max() > 0.01)
    
    
    eps = dr[Nr // 2]
    Phi_ref = compute_potential_direct(
        r_1d, phi_1d, Sigma, r_1d, phi_1d, dr, dphi,
        G=1.0, epsilon=eps, use_singular_correction=False)
    Phi_ref_zero = Phi_ref - np.mean(Phi_ref)
    
    results.record("Pipeline: Φ_ref computed", Phi_ref.shape == (Nr, Nphi))
    results.record("Pipeline: Φ_ref finite", np.all(np.isfinite(Phi_ref)))
    
    
    g_r_ref, g_phi_ref = compute_gravity_fd(Phi_ref, r_1d, phi_1d)
    results.record("Pipeline: gravity computed", g_r_ref.shape == (Nr, Nphi))
    
    
    g_phi_rms = np.sqrt(np.mean(g_phi_ref**2))
    results.record("Pipeline: g_φ ≠ 0 (perturbed)", g_phi_rms > 1e-6,
                    f"RMS(g_φ) = {g_phi_rms:.4e}")
    
    
    device = torch.device('cpu')
    model = GravityPINN(
        hidden_layers=2, hidden_units=24,
        omega_0=10.0, omega=1.0,
        r_min=1.0, r_max=30.0
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_weights = {'integral': 1.0, 'gauge': 10.0, 'smoothness': 0.0}
    
    r_flat = R.ravel()
    phi_flat = PHI.ravel()
    Phi_target_flat = Phi_ref_zero.ravel()
    
    
    for _ in range(200):
        idx = np.random.choice(len(r_flat), size=min(256, len(r_flat)), replace=False)
        r_b = torch.tensor(r_flat[idx], dtype=torch.float32)
        phi_b = torch.tensor(phi_flat[idx], dtype=torch.float32)
        Phi_b = torch.tensor(Phi_target_flat[idx], dtype=torch.float32)
        
        optimizer.zero_grad()
        L, _ = total_loss(model, r_b, phi_b, Phi_b, loss_weights)
        L.backward()
        optimizer.step()
    
    
    model.eval()
    with torch.no_grad():
        Phi_pinn = model(
            torch.tensor(r_flat, dtype=torch.float32),
            torch.tensor(phi_flat, dtype=torch.float32)
        ).cpu().numpy().reshape(R.shape)
    
    pinn_err = rel_error_L2(Phi_pinn, Phi_ref_zero)
    results.record("Pipeline: PINN trained", pinn_err < 2.0,
                    f"RelErr = {pinn_err:.4e} (loose threshold for 200 epochs)")
    
    
    g_r_pinn, g_phi_pinn = compute_gravity_fd(Phi_pinn, r_1d, phi_1d)
    results.record("Pipeline: PINN gravity shape", g_r_pinn.shape == (Nr, Nphi))
    results.record("Pipeline: PINN gravity finite", 
                    np.all(np.isfinite(g_r_pinn)) and np.all(np.isfinite(g_phi_pinn)))


def test_pinn_holdout_generalization(results: TestResults):
    print("\n--- PINN holdout generalization test ---")

    torch_ok, torch_msg = torch_runtime_available()
    if not torch_ok:
        results.record("PyTorch runtime unavailable (skip holdout test)", True, torch_msg)
        return

    try:
        import torch
        from model import GravityPINN, total_loss
    except ImportError as e:
        results.record("Holdout test: PyTorch import", False, str(e))
        return

    Nr, Nphi = 28, 28
    r_1d, phi_1d, R, PHI = make_polar_grid(0.8, 40.0, Nr, Nphi)
    dr, dphi = cell_sizes(r_1d, phi_1d)
    eps = 0.5 * dr[Nr // 2]

    
    Sigma = sigma_clump(R, PHI, 1.0, 40.0, 15.0, 1.7, 2.5, 0.35)
    Phi_ref = compute_potential_direct(
        r_1d, phi_1d, Sigma, r_1d, phi_1d, dr, dphi,
        G=1.0, epsilon=eps, use_singular_correction=False
    )
    Phi_ref_zero = Phi_ref - np.mean(Phi_ref)

    r_flat = R.ravel()
    phi_flat = PHI.ravel()
    y_flat = Phi_ref_zero.ravel()

    seed = 1234
    rng = np.random.default_rng(seed)
    n_all = len(r_flat)
    perm = rng.permutation(n_all)
    n_train = int(0.8 * n_all)
    train_idx = perm[:n_train]
    hold_idx = perm[n_train:]

    y_scale = np.std(y_flat[train_idx])
    y_scale = max(y_scale, 1e-8)
    y_norm = y_flat / y_scale

    device = torch.device('cpu')
    torch.manual_seed(seed)
    model = GravityPINN(
        hidden_layers=3, hidden_units=96,
        omega_0=10.0, omega=1.0,
        r_min=0.8, r_max=40.0
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 1200
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    loss_weights = {'integral': 1.0, 'gauge': 3.0, 'smoothness': 0.0}

    for _ in range(epochs):
        model.train()
        
        r_b = torch.tensor(r_flat[train_idx], dtype=torch.float32, device=device)
        phi_b = torch.tensor(phi_flat[train_idx], dtype=torch.float32, device=device)
        y_b = torch.tensor(y_norm[train_idx], dtype=torch.float32, device=device)
        optimizer.zero_grad()
        L, _ = total_loss(model, r_b, phi_b, y_b, loss_weights)
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        r_h = torch.tensor(r_flat[hold_idx], dtype=torch.float32, device=device)
        phi_h = torch.tensor(phi_flat[hold_idx], dtype=torch.float32, device=device)
        y_pred_h = model(r_h, phi_h).cpu().numpy() * y_scale

    hold_err = rel_error_L2(y_pred_h, y_flat[hold_idx])
    results.record("Holdout RelErr(Φ) < 0.05", hold_err < 0.05,
                   f"holdout err = {hold_err:.4e}")






def main():
    parser = argparse.ArgumentParser(description='Gravity-PINN Test Suite')
    parser.add_argument('--quick', action='store_true',
                        help='Quick smoke test only (skip long-running tests)')
    parser.add_argument('--epsilon-sweep', action='store_true',
                        help='Run full ε-sweep benchmark')
    parser.add_argument('--all', action='store_true',
                        help='Run all tests including ε-sweep')
    args = parser.parse_args()
    
    results = TestResults()
    
    print("=" * 60)
    print("GRAVITY-PINN TEST SUITE")
    print("=" * 60)
    
    t_start = time.time()
    
    
    test_grid(results)
    test_sigma_models(results)
    
    if not args.quick:
        
        test_reference_solver(results)
        test_singular_cell_correction(results)
        test_point_integral_matches_direct(results)
        test_constant_annulus_quadrature_validator(results)
        test_constant_annulus_matched_epsilon_reference(results)
    
    
    test_pinn_smoke(results)
    
    if not args.quick:
        
        test_mini_pipeline(results)
        test_pinn_holdout_generalization(results)
    
    
    if args.epsilon_sweep or args.all:
        test_epsilon_sweep(results)
    
    t_total = time.time() - t_start
    
    all_passed = results.summary()
    print(f"\n  Total time: {t_total:.1f} s")
    
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
