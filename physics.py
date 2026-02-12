"""
physics.py — Thin-disk Poisson integral kernels, surface density models,
              analytic solutions, and singularity treatment.

Governing equation (midplane gravitational potential of a razor-thin disk):

    Φ(r,φ) = -G ∫_{r_min}^{r_max} ∫_0^{2π}  Σ(r',φ') r' dφ' dr'
              / sqrt(r² + r'² - 2rr' cos(φ-φ') + ε²)

Smoothed variant uses ε > 0 to regularize the 1/R singularity at R→0.
Singular-cell correction (Binney & Tremaine 1987) analytically evaluates
the self-contribution when ε = 0.

In-plane gravity:
    g_r  = -∂Φ/∂r
    g_φ  = -(1/r) ∂Φ/∂φ

References:
    Vorobyov et al. (2024) — Eq. (4), Appendix B
    Vorobyov & Basu (2010) — Eq. (4)
    Binney & Tremaine (1987) — Section 2.8
    Durand (1964) — constant-density disk analytic solution
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any


# ============================================================
# 1. GRID CONSTRUCTION
# ============================================================

def make_polar_grid(r_min: float, r_max: float, N_r: int, N_phi: int
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a log-spaced radial × uniform azimuthal polar grid.
    
    Returns
    -------
    r_1d   : (N_r,)        radial cell centres
    phi_1d : (N_phi,)      azimuthal cell centres
    R      : (N_r, N_phi)  2-D radial mesh
    PHI    : (N_r, N_phi)  2-D azimuthal mesh
    """
    r_1d = np.logspace(np.log10(r_min), np.log10(r_max), N_r)
    phi_1d = np.linspace(0.0, 2.0 * np.pi, N_phi, endpoint=False)
    R, PHI = np.meshgrid(r_1d, phi_1d, indexing='ij')
    return r_1d, phi_1d, R, PHI


def cell_sizes(r_1d: np.ndarray, phi_1d: np.ndarray
               ) -> Tuple[np.ndarray, float]:
    """
    Compute radial cell widths (log-spaced) and azimuthal cell width.
    
    Returns
    -------
    dr : (N_r,)  radial cell widths
    dphi : float  uniform azimuthal cell width
    """
    # For log-spaced grid, approximate cell width
    log_r = np.log(r_1d)
    dlog_r = np.diff(log_r)
    # Extend to same length: use last value for final cell
    dlog_r = np.append(dlog_r, dlog_r[-1])
    dr = r_1d * dlog_r  # dr = r * d(ln r)
    dphi = 2.0 * np.pi / len(phi_1d)
    return dr, dphi


# ============================================================
# 2. SURFACE DENSITY MODELS (test cases)
# ============================================================

def sigma_exponential(R: np.ndarray, PHI: np.ndarray,
                      Sigma_0: float, R_scale: float) -> np.ndarray:
    """B1: Smooth axisymmetric exponential disk."""
    return Sigma_0 * np.exp(-R / R_scale)


def sigma_exponential_perturbed(R: np.ndarray, PHI: np.ndarray,
                                Sigma_0: float, R_scale: float,
                                A_pert: float, m_mode: int) -> np.ndarray:
    """B2: Exponential disk with azimuthal perturbation."""
    return Sigma_0 * np.exp(-R / R_scale) * (1.0 + A_pert * np.cos(m_mode * PHI))


def sigma_clump(R: np.ndarray, PHI: np.ndarray,
                Sigma_bg: float, Sigma_clump: float,
                r_c: float, phi_c: float,
                sigma_r: float, sigma_phi: float) -> np.ndarray:
    """B3: Single Gaussian clump on uniform background."""
    dphi = PHI - phi_c
    # Wrap azimuthal difference to [-π, π]
    dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
    clump = Sigma_clump * np.exp(
        -0.5 * ((R - r_c) / sigma_r)**2
        - 0.5 * (dphi / sigma_phi)**2
    )
    return Sigma_bg + clump


def sigma_multi_clump(R: np.ndarray, PHI: np.ndarray,
                      Sigma_bg: float,
                      clumps: list) -> np.ndarray:
    """B4: Multiple Gaussian clumps on uniform background."""
    Sigma = np.full_like(R, Sigma_bg)
    for cl in clumps:
        dphi = PHI - cl['phi']
        dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
        Sigma += cl['Sigma'] * np.exp(
            -0.5 * ((R - cl['r']) / cl['sigma_r'])**2
            - 0.5 * (dphi / cl['sigma_phi'])**2
        )
    return Sigma


def sigma_constant(R: np.ndarray, PHI: np.ndarray,
                   Sigma_0: float, r_in: float, r_out: float) -> np.ndarray:
    """B5: Constant surface density between r_in and r_out."""
    mask = (R >= r_in) & (R <= r_out)
    return np.where(mask, Sigma_0, 0.0)


def build_sigma(R: np.ndarray, PHI: np.ndarray, case_cfg: Dict[str, Any]
                ) -> np.ndarray:
    """Dispatch to the appropriate Σ model based on config."""
    tp = case_cfg['type']
    if tp == 'exponential':
        return sigma_exponential(R, PHI, case_cfg['Sigma_0'], case_cfg['R_scale'])
    elif tp == 'exponential_perturbed':
        return sigma_exponential_perturbed(
            R, PHI, case_cfg['Sigma_0'], case_cfg['R_scale'],
            case_cfg['A_pert'], case_cfg['m_mode'])
    elif tp == 'clump_on_background':
        return sigma_clump(
            R, PHI, case_cfg['Sigma_bg'], case_cfg['Sigma_clump'],
            case_cfg['r_clump'], case_cfg['phi_clump'],
            case_cfg['sigma_r'], case_cfg['sigma_phi'])
    elif tp == 'multi_clump':
        return sigma_multi_clump(R, PHI, case_cfg['Sigma_bg'], case_cfg['clumps'])
    elif tp == 'constant':
        return sigma_constant(R, PHI, case_cfg['Sigma_0'],
                              case_cfg['r_in'], case_cfg['r_out'])
    else:
        raise ValueError(f"Unknown Σ type: {tp}")


# ============================================================
# 3. POISSON INTEGRAL KERNEL — REFERENCE SOLVER
# ============================================================

def poisson_kernel(r: float, phi: float,
                   r_prime: np.ndarray, phi_prime: np.ndarray,
                   epsilon: float = 0.0) -> np.ndarray:
    """
    Evaluate the thin-disk Poisson kernel:
    
        K(r,φ; r',φ') = 1 / sqrt(r² + r'² - 2rr'cos(φ-φ') + ε²)
    
    Parameters
    ----------
    r, phi        : scalars, evaluation point
    r_prime       : (N_r', N_phi') or (N_r',) primed radii
    phi_prime     : (N_r', N_phi') or (N_phi',) primed azimuths
    epsilon       : smoothing length
    
    Returns
    -------
    K : same shape as r_prime (broadcast)
    """
    cos_dphi = np.cos(phi - phi_prime)
    denom_sq = r**2 + r_prime**2 - 2.0 * r * r_prime * cos_dphi + epsilon**2
    return 1.0 / np.sqrt(np.maximum(denom_sq, 1e-30))


def compute_potential_direct(r_eval: np.ndarray, phi_eval: np.ndarray,
                             Sigma: np.ndarray,
                             r_grid: np.ndarray, phi_grid: np.ndarray,
                             dr: np.ndarray, dphi: float,
                             G: float = 1.0,
                             epsilon: float = 0.0,
                             use_singular_correction: bool = True
                             ) -> np.ndarray:
    """
    Reference solver: Direct summation of the Poisson integral on a polar grid.
    
    Φ(r_i, φ_j) = -G Σ_{i',j'} Σ(r_{i'}, φ_{j'}) r_{i'} dr_{i'} dφ
                   / sqrt(r_i² + r_{i'}² - 2 r_i r_{i'} cos(φ_j - φ_{j'}) + ε²)
    
    With singular-cell correction (Binney & Tremaine 1987, Eq. B.5 in Vorobyov+ 2024):
    When (i,j) == (i',j'), replace the divergent 1/R with the analytic integral
    over the cell area.
    
    Parameters
    ----------
    r_eval, phi_eval : (N_r,), (N_phi,)  evaluation grid
    Sigma            : (N_r, N_phi)       surface density on same grid
    r_grid, phi_grid : same as r_eval, phi_eval (source grid = eval grid here)
    dr               : (N_r,)  radial cell widths
    dphi             : float   azimuthal cell width
    G                : gravitational constant
    epsilon          : smoothing length (0 = use singular correction)
    use_singular_correction : bool
    
    Returns
    -------
    Phi : (N_r, N_phi)  gravitational potential
    """
    Nr = len(r_grid)
    Nphi = len(phi_grid)
    Phi = np.zeros((Nr, Nphi), dtype=np.float64)
    
    # Precompute source quantities: Σ * r' * dr' * dφ
    R_SRC, PHI_SRC = np.meshgrid(r_grid, phi_grid, indexing='ij')
    DR_SRC = np.outer(dr, np.ones(Nphi))
    weight = Sigma * R_SRC * DR_SRC * dphi  # (Nr, Nphi)
    
    for i in range(Nr):
        for j in range(Nphi):
            r_ij = r_grid[i]
            phi_ij = phi_grid[j]
            
            # Distance squared to all source cells
            cos_dphi = np.cos(phi_ij - PHI_SRC)
            dist_sq = r_ij**2 + R_SRC**2 - 2.0 * r_ij * R_SRC * cos_dphi + epsilon**2
            
            # Avoid exact zero for epsilon=0 case
            inv_dist = 1.0 / np.sqrt(np.maximum(dist_sq, 1e-30))
            
            # Singular cell correction
            if epsilon == 0.0 and use_singular_correction:
                # At (i,j) == (i',j'), dist_sq → 0.  Replace with analytic value.
                # Using the log-coordinate result (Vorobyov+ 2024 Eq. B.5):
                # V(0,0) = -2G S [sinh⁻¹(Δφ/Δu)/Δφ + sinh⁻¹(Δu/Δφ)/Δu]
                # where Δu = d(ln r), S = r^{3/2} Σ, V = r^{1/2} Φ
                # For the (i,j) cell:
                du = dr[i] / r_ij  # Δu ≈ Δ(ln r)
                dphi_cell = dphi
                
                # Self-contribution in reduced coordinates
                # (contribution to the reduced potential V)
                S_cell = r_ij**1.5 * Sigma[i, j]
                V_self = -2.0 * G * S_cell * (
                    np.arcsinh(dphi_cell / du) / dphi_cell
                    + np.arcsinh(du / dphi_cell) / du
                )
                # Convert back: Φ_self = V_self / r^{1/2}
                Phi_self = V_self / np.sqrt(r_ij)
                
                # Zero out the self-cell in the sum and add analytic value
                inv_dist[i, j] = 0.0
                Phi[i, j] = -G * np.sum(weight * inv_dist) + Phi_self
            else:
                Phi[i, j] = -G * np.sum(weight * inv_dist)
    
    return Phi


def compute_potential_fft(Sigma: np.ndarray,
                          r_grid: np.ndarray, phi_grid: np.ndarray,
                          dr: np.ndarray, dphi: float,
                          G: float = 1.0,
                          epsilon: float = 0.0,
                          use_singular_correction: bool = True
                          ) -> np.ndarray:
    """
    FFT/convolution gravity solver following Binney & Tremaine (1987).
    
    Uses the coordinate transform u = ln(r) to convert the Poisson integral
    into a convolution in (u, φ) space, solvable via 2D FFT.
    
    The kernel in (u, φ) coordinates:
        K(u, φ) = 2^{-1/2} / sqrt(cosh(Δu) - cos(Δφ) + 0.5 ε² e^{-(u+u')})
    
    For ε ∝ r (which preserves the convolution property), ε² e^{-(u+u')} → const.
    
    Parameters
    ----------
    Sigma    : (N_r, N_phi) surface density
    r_grid   : (N_r,)  radial cell centres (log-spaced)
    phi_grid : (N_phi,) azimuthal cell centres
    dr       : (N_r,)  radial cell widths  
    dphi     : float
    G        : gravitational constant
    epsilon  : smoothing length (absolute, or 0 for singular correction)
    
    Returns
    -------
    Phi : (N_r, N_phi) gravitational potential
    """
    Nr = len(r_grid)
    Nphi = len(phi_grid)
    
    # Log-radial coordinate
    u = np.log(r_grid)
    du = np.diff(u)
    du = np.append(du, du[-1])  # pad to same length
    
    # Reduced surface density S = r^{3/2} Σ
    S = np.zeros((Nr, Nphi))
    for j in range(Nphi):
        S[:, j] = r_grid**1.5 * Sigma[:, j]
    
    # Build kernel in (Δu, Δφ) space
    # Need double-size arrays for linear (non-circular) convolution in u
    N_u_pad = 2 * Nr
    N_phi_pad = Nphi  # φ is periodic, no padding needed
    
    # Kernel array
    kernel = np.zeros((N_u_pad, N_phi_pad))
    
    for iu in range(N_u_pad):
        delta_u = (iu - Nr) * du[min(iu, Nr - 1)]  # approximate
        for jphi in range(N_phi_pad):
            delta_phi = jphi * dphi
            if jphi > Nphi // 2:
                delta_phi = delta_phi - 2.0 * np.pi
            
            denom = np.cosh(delta_u if iu < 2 * Nr else 0) - np.cos(delta_phi)
            if epsilon > 0:
                # For ε ∝ r, the smoothing term in u-coords is ~ 0.5 * (ε/r)²
                # Use mean r for approximate convolution property
                r_mean = np.exp(np.mean(u))
                denom += 0.5 * (epsilon / r_mean)**2
            
            if denom > 1e-30:
                kernel[iu, jphi] = 2.0**(-0.5) / np.sqrt(denom)
            else:
                kernel[iu, jphi] = 0.0  # will be corrected below
    
    # Singular cell correction for the (0,0) element
    if epsilon == 0.0 and use_singular_correction:
        du0 = du[0]  # representative cell size
        kernel[Nr, 0] = 2.0 * (
            np.arcsinh(dphi / du0) / dphi
            + np.arcsinh(du0 / dphi) / du0
        )
    
    # Pad S for linear convolution in u
    S_pad = np.zeros((N_u_pad, N_phi_pad))
    S_pad[:Nr, :] = S * np.outer(du, np.ones(Nphi)) * dphi
    
    # CRITICAL: shift kernel so zero-lag (iu=Nr) goes to index 0 for FFT
    kernel_shifted = np.fft.ifftshift(kernel, axes=0)
    
    # Convolve via FFT
    S_hat = np.fft.fft2(S_pad)
    K_hat = np.fft.fft2(kernel_shifted)
    V_hat = S_hat * K_hat
    V_pad = np.real(np.fft.ifft2(V_hat))
    
    # Extract the valid region (reduced potential V)
    V = V_pad[:Nr, :]
    
    # Convert back: Φ = V / r^{1/2}
    Phi = np.zeros((Nr, Nphi))
    for j in range(Nphi):
        Phi[:, j] = -G * V[:, j] / np.sqrt(r_grid)
    
    return Phi


# ============================================================
# 4. GRAVITY (FORCE) COMPUTATION
# ============================================================

def compute_gravity_fd(Phi: np.ndarray,
                       r_grid: np.ndarray, phi_grid: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute in-plane gravitational acceleration from Φ by finite differences.
    
        g_r  = -∂Φ/∂r
        g_φ  = -(1/r) ∂Φ/∂φ
    
    Uses second-order central differences (periodic in φ).
    
    Returns
    -------
    g_r   : (N_r, N_phi)
    g_phi : (N_r, N_phi)
    """
    Nr, Nphi = Phi.shape
    g_r = np.zeros_like(Phi)
    g_phi = np.zeros_like(Phi)
    
    # ∂Φ/∂r  — central differences on log-spaced (nonuniform) grid
    # Derivation: eliminate f'' from Taylor expansions about x_i:
    #   f'(x_i) = [ h⁻² f_{i+1} + (h⁺² - h⁻²) f_i - h⁺² f_{i-1} ]
    #             / [ h⁺ h⁻ (h⁺ + h⁻) ]
    # where h⁺ = r_{i+1} - r_i, h⁻ = r_i - r_{i-1}.
    for i in range(1, Nr - 1):
        dr_fwd = r_grid[i + 1] - r_grid[i]   # h⁺
        dr_bwd = r_grid[i] - r_grid[i - 1]    # h⁻
        dPhi_dr = (
            Phi[i + 1, :] * dr_bwd / (dr_fwd * (dr_fwd + dr_bwd))
            + Phi[i, :] * (dr_fwd - dr_bwd) / (dr_fwd * dr_bwd)
            - Phi[i - 1, :] * dr_fwd / (dr_bwd * (dr_fwd + dr_bwd))
        )
        g_r[i, :] = -dPhi_dr
    # One-sided at boundaries
    dr0 = r_grid[1] - r_grid[0]
    g_r[0, :] = -(Phi[1, :] - Phi[0, :]) / dr0
    dr_end = r_grid[-1] - r_grid[-2]
    g_r[-1, :] = -(Phi[-1, :] - Phi[-2, :]) / dr_end
    
    # ∂Φ/∂φ  — central differences, periodic
    dphi = phi_grid[1] - phi_grid[0]
    for j in range(Nphi):
        jp1 = (j + 1) % Nphi
        jm1 = (j - 1) % Nphi
        g_phi[:, j] = -(1.0 / r_grid) * (Phi[:, jp1] - Phi[:, jm1]) / (2.0 * dphi)
    
    return g_r, g_phi


# ============================================================
# 5. ANALYTIC SOLUTIONS (for validation)
# ============================================================

def phi_exponential_disk_analytic(r: np.ndarray, Sigma_0: float, r_0: float,
                                 G: float = 1.0) -> np.ndarray:
    """
    Analytic potential for Σ(r) = Σ₀ exp(-r/r₀) — axisymmetric.
    
    Φ(r) = -π G Σ₀ r [I₀(y) K₁(y) - I₁(y) K₀(y)]
    
    where y = r / (2 r₀), and I_n, K_n are modified Bessel functions.
    (Binney & Tremaine 1987; Vorobyov+ 2024 Eq. B.7)
    """
    from scipy.special import iv as I_v, kv as K_v
    y = r / (2.0 * r_0)
    Phi = -np.pi * G * Sigma_0 * r * (
        I_v(0, y) * K_v(1, y) - I_v(1, y) * K_v(0, y)
    )
    return Phi


def gr_exponential_disk_analytic(r: np.ndarray, Sigma_0: float, r_0: float,
                                 G: float = 1.0) -> np.ndarray:
    """
    Analytic radial gravitational acceleration for Σ(r) = Σ₀ exp(-r/r₀).
    Computed by numerical differentiation of the analytic potential.
    """
    dr = r * 1e-5  # small step for finite difference
    Phi_plus = phi_exponential_disk_analytic(r + dr, Sigma_0, r_0, G)
    Phi_minus = phi_exponential_disk_analytic(r - dr, Sigma_0, r_0, G)
    return -(Phi_plus - Phi_minus) / (2.0 * dr)


def gr_constant_disk_analytic(r: np.ndarray, Sigma_0: float,
                              r_in: float, r_out: float,
                              G: float = 1.0) -> np.ndarray:
    """
    Reference radial acceleration for a constant-Σ annulus.

    NOTE:
    The previously implemented closed-form Eq. B.8 expression was found
    inconsistent with the independent axisymmetric quadrature baseline.
    Until a validated piecewise closed-form implementation is derived,
    this function intentionally delegates to the quadrature validator.
    """
    return gr_constant_disk_axisym_quadrature(
        r=np.asarray(r, dtype=np.float64),
        Sigma_0=Sigma_0,
        r_in=r_in,
        r_out=r_out,
        G=G,
    )


def phi_constant_disk_axisym_quadrature(r: np.ndarray, Sigma_0: float,
                                        r_in: float, r_out: float,
                                        G: float = 1.0,
                                        epsabs: float = 1e-9,
                                        epsrel: float = 1e-8,
                                        limit: int = 400) -> np.ndarray:
    """
    1D axisymmetric quadrature reference for a constant-Σ annulus potential.

    Uses the azimuthally integrated kernel:
        ∫_0^{2π} dφ / sqrt(r^2 + r'^2 - 2 r r' cosφ) = 4 K(m) / (r + r')
        m = 4 r r' / (r + r')^2

    so
        Φ(r) = -4 G Σ_0 ∫_{r_in}^{r_out} [r' K(m) / (r + r')] dr' .

    This is an independent numerical validator for B5 that avoids
    relying on the closed-form Eq. B.8 implementation.
    """
    from scipy.integrate import quad
    from scipy.special import ellipk

    r = np.asarray(r, dtype=np.float64)
    phi = np.zeros_like(r)

    for i, r_val in enumerate(r):
        if r_val <= 0.0:
            phi[i] = np.nan
            continue

        def _integrand(rp: float) -> float:
            m = 4.0 * r_val * rp / (r_val + rp) ** 2
            m = np.clip(m, 0.0, 1.0 - 1e-14)
            return rp * ellipk(m) / (r_val + rp)

        # If r lies inside the annulus, split at the logarithmic singularity.
        if r_in < r_val < r_out:
            val_l, _ = quad(
                _integrand, r_in, r_val, epsabs=epsabs, epsrel=epsrel, limit=limit
            )
            val_r, _ = quad(
                _integrand, r_val, r_out, epsabs=epsabs, epsrel=epsrel, limit=limit
            )
            integral = val_l + val_r
        else:
            integral, _ = quad(
                _integrand, r_in, r_out, epsabs=epsabs, epsrel=epsrel, limit=limit
            )

        phi[i] = -4.0 * G * Sigma_0 * integral

    return phi


def gr_constant_disk_axisym_quadrature(r: np.ndarray, Sigma_0: float,
                                       r_in: float, r_out: float,
                                       G: float = 1.0,
                                       epsabs: float = 1e-9,
                                       epsrel: float = 1e-8,
                                       limit: int = 400) -> np.ndarray:
    """
    1D axisymmetric quadrature reference for g_r of a constant-Σ annulus.

    Computes Φ(r) via `phi_constant_disk_axisym_quadrature` and differentiates
    on the provided r-grid:
        g_r = -dΦ/dr
    """
    r = np.asarray(r, dtype=np.float64)
    phi = phi_constant_disk_axisym_quadrature(
        r, Sigma_0=Sigma_0, r_in=r_in, r_out=r_out, G=G,
        epsabs=epsabs, epsrel=epsrel, limit=limit
    )

    dphi_dr = np.gradient(phi, r, edge_order=2)
    return -dphi_dr


# ============================================================
# 6. NUMERICAL INTEGRAL EVALUATOR (for PINN training targets)
# ============================================================

def evaluate_integral_at_points(r_eval: np.ndarray, phi_eval: np.ndarray,
                                Sigma: np.ndarray,
                                r_grid: np.ndarray, phi_grid: np.ndarray,
                                dr: np.ndarray, dphi: float,
                                G: float = 1.0,
                                epsilon: float = 0.0,
                                use_singular_correction: bool = False
                                ) -> np.ndarray:
    """
    Evaluate Φ_int(r_i, φ_i) for a set of query points using the
    full-grid quadrature of the Poisson integral.
    
    This is used as the PINN training target in Mode A.
    
    Parameters
    ----------
    r_eval   : (N,) query radii
    phi_eval : (N,) query azimuths
    Sigma    : (N_r_src, N_phi_src) surface density on source grid
    r_grid   : (N_r_src,) source radial grid
    phi_grid : (N_phi_src,) source azimuthal grid
    dr       : (N_r_src,) source radial cell widths
    dphi     : float
    G        : gravitational constant
    epsilon  : smoothing length
    use_singular_correction : bool
        If True and epsilon=0, replace the self-cell contribution
        with the Binney & Tremaine (1987) Eq. B.5 analytic integral.
    
    Returns
    -------
    Phi_eval : (N,) potential values at query points
    """
    N = len(r_eval)
    Nr_src = len(r_grid)
    Nphi_src = len(phi_grid)
    
    # Precompute source weights: Σ * r' * dr' * dφ
    R_SRC, PHI_SRC = np.meshgrid(r_grid, phi_grid, indexing='ij')
    DR_SRC = np.outer(dr, np.ones(Nphi_src))
    weight = Sigma * R_SRC * DR_SRC * dphi  # (Nr_src, Nphi_src)
    weight_flat = weight.ravel()
    r_src_flat = R_SRC.ravel()
    phi_src_flat = PHI_SRC.ravel()
    
    Phi_eval = np.zeros(N, dtype=np.float64)
    
    for n in range(N):
        cos_dphi = np.cos(phi_eval[n] - phi_src_flat)
        dist_sq = (r_eval[n]**2 + r_src_flat**2
                   - 2.0 * r_eval[n] * r_src_flat * cos_dphi
                   + epsilon**2)
        
        if epsilon == 0.0 and use_singular_correction:
            # Identify self-cell: nearest grid point to (r_eval[n], phi_eval[n])
            i_self = np.argmin(np.abs(r_grid - r_eval[n]))
            j_self = np.argmin(np.abs(phi_grid - phi_eval[n]))
            k_self = i_self * Nphi_src + j_self
            
            # Check if query point is actually near this grid point
            r_dist = abs(r_eval[n] - r_grid[i_self])
            phi_dist = abs(phi_eval[n] - phi_grid[j_self])
            is_on_grid = (r_dist < 0.5 * dr[i_self] and phi_dist < 0.5 * dphi)
            
            if is_on_grid:
                # Skip self-cell in regular sum
                mask = np.ones(len(r_src_flat), dtype=bool)
                mask[k_self] = False
                
                inv_dist = np.zeros_like(dist_sq)
                inv_dist[mask] = 1.0 / np.sqrt(np.maximum(dist_sq[mask], 1e-30))
                Phi_eval[n] = -G * np.dot(weight_flat[mask], inv_dist[mask])
                
                # Add analytic self-cell correction (B&T Eq. B.5)
                r_s = r_grid[i_self]
                du_cell = dr[i_self] / r_s  # log-coordinate cell width
                S_self = r_s**1.5 * Sigma[i_self, j_self]
                V_self = -2.0 * G * S_self * (
                    np.arcsinh(dphi / du_cell) / dphi
                    + np.arcsinh(du_cell / dphi) / du_cell
                )
                Phi_eval[n] += V_self / np.sqrt(r_s)
            else:
                # Off-grid point: no self-cell issue, clamp for safety
                inv_dist = 1.0 / np.sqrt(np.maximum(dist_sq, 1e-30))
                Phi_eval[n] = -G * np.dot(weight_flat, inv_dist)
        else:
            inv_dist = 1.0 / np.sqrt(np.maximum(dist_sq, 1e-30))
            Phi_eval[n] = -G * np.dot(weight_flat, inv_dist)
    
    return Phi_eval


# ============================================================
# 7. ERROR METRICS
# ============================================================

def rel_error_L2(pred: np.ndarray, ref: np.ndarray, eta: float = 1e-10
                 ) -> float:
    """Relative L2 error: ||pred - ref||_2 / (||ref||_2 + η)."""
    return np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + eta)


def rel_error_L2_with_floor(pred: np.ndarray, ref: np.ndarray,
                            floor: float = 0.0, eta: float = 1e-10) -> float:
    """
    Relative L2 error with denominator floor:
        ||pred - ref||_2 / (max(||ref||_2, floor) + eta)
    Useful when ref is (near) zero by symmetry.
    """
    denom = max(np.linalg.norm(ref), floor)
    return np.linalg.norm(pred - ref) / (denom + eta)


def max_error(pred: np.ndarray, ref: np.ndarray) -> float:
    """Maximum absolute error."""
    return np.max(np.abs(pred - ref))


def rel_error_pointwise(pred: np.ndarray, ref: np.ndarray, eta: float = 1e-10
                        ) -> np.ndarray:
    """Pointwise relative error."""
    return np.abs(pred - ref) / (np.abs(ref) + eta)
