
import numpy as np
from typing import Tuple, Optional, Dict, Any






def make_polar_grid(r_min: float, r_max: float, N_r: int, N_phi: int
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r_1d = np.logspace(np.log10(r_min), np.log10(r_max), N_r)
    phi_1d = np.linspace(0.0, 2.0 * np.pi, N_phi, endpoint=False)
    R, PHI = np.meshgrid(r_1d, phi_1d, indexing='ij')
    return r_1d, phi_1d, R, PHI


def cell_sizes(r_1d: np.ndarray, phi_1d: np.ndarray
               ) -> Tuple[np.ndarray, float]:
    log_r = np.log(r_1d)
    dlog_r = np.diff(log_r)
    
    dlog_r = np.append(dlog_r, dlog_r[-1])
    dr = r_1d * dlog_r  
    dphi = 2.0 * np.pi / len(phi_1d)
    return dr, dphi






def sigma_exponential(R: np.ndarray, PHI: np.ndarray,
                      Sigma_0: float, R_scale: float) -> np.ndarray:
    return Sigma_0 * np.exp(-R / R_scale)


def sigma_exponential_perturbed(R: np.ndarray, PHI: np.ndarray,
                                Sigma_0: float, R_scale: float,
                                A_pert: float, m_mode: int) -> np.ndarray:
    return Sigma_0 * np.exp(-R / R_scale) * (1.0 + A_pert * np.cos(m_mode * PHI))


def sigma_clump(R: np.ndarray, PHI: np.ndarray,
                Sigma_bg: float, Sigma_clump: float,
                r_c: float, phi_c: float,
                sigma_r: float, sigma_phi: float) -> np.ndarray:
    dphi = PHI - phi_c
    
    dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
    clump = Sigma_clump * np.exp(
        -0.5 * ((R - r_c) / sigma_r)**2
        - 0.5 * (dphi / sigma_phi)**2
    )
    return Sigma_bg + clump


def sigma_multi_clump(R: np.ndarray, PHI: np.ndarray,
                      Sigma_bg: float,
                      clumps: list) -> np.ndarray:
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
    mask = (R >= r_in) & (R <= r_out)
    return np.where(mask, Sigma_0, 0.0)


def build_sigma(R: np.ndarray, PHI: np.ndarray, case_cfg: Dict[str, Any]
                ) -> np.ndarray:
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






def poisson_kernel(r: float, phi: float,
                   r_prime: np.ndarray, phi_prime: np.ndarray,
                   epsilon: float = 0.0) -> np.ndarray:

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
    Nr = len(r_grid)
    Nphi = len(phi_grid)
    Phi = np.zeros((Nr, Nphi), dtype=np.float64)
    
    
    R_SRC, PHI_SRC = np.meshgrid(r_grid, phi_grid, indexing='ij')
    DR_SRC = np.outer(dr, np.ones(Nphi))
    weight = Sigma * R_SRC * DR_SRC * dphi  
    
    for i in range(Nr):
        for j in range(Nphi):
            r_ij = r_grid[i]
            phi_ij = phi_grid[j]
            
            
            cos_dphi = np.cos(phi_ij - PHI_SRC)
            dist_sq = r_ij**2 + R_SRC**2 - 2.0 * r_ij * R_SRC * cos_dphi + epsilon**2
            
            
            inv_dist = 1.0 / np.sqrt(np.maximum(dist_sq, 1e-30))
            
            
            if epsilon == 0.0 and use_singular_correction:
                
                
                
                
                
                du = dr[i] / r_ij  
                dphi_cell = dphi
                
                
                
                S_cell = r_ij**1.5 * Sigma[i, j]
                V_self = -2.0 * G * S_cell * (
                    np.arcsinh(dphi_cell / du) / dphi_cell
                    + np.arcsinh(du / dphi_cell) / du
                )
                
                Phi_self = V_self / np.sqrt(r_ij)
                
                
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
    Nr = len(r_grid)
    Nphi = len(phi_grid)
    
    
    u = np.log(r_grid)
    du = np.diff(u)
    du = np.append(du, du[-1])  
    
    
    S = np.zeros((Nr, Nphi))
    for j in range(Nphi):
        S[:, j] = r_grid**1.5 * Sigma[:, j]
    
    
    
    N_u_pad = 2 * Nr
    N_phi_pad = Nphi  
    
    
    kernel = np.zeros((N_u_pad, N_phi_pad))
    
    for iu in range(N_u_pad):
        delta_u = (iu - Nr) * du[min(iu, Nr - 1)]  
        for jphi in range(N_phi_pad):
            delta_phi = jphi * dphi
            if jphi > Nphi // 2:
                delta_phi = delta_phi - 2.0 * np.pi
            
            denom = np.cosh(delta_u if iu < 2 * Nr else 0) - np.cos(delta_phi)
            if epsilon > 0:
                
                
                r_mean = np.exp(np.mean(u))
                denom += 0.5 * (epsilon / r_mean)**2
            
            if denom > 1e-30:
                kernel[iu, jphi] = 2.0**(-0.5) / np.sqrt(denom)
            else:
                kernel[iu, jphi] = 0.0  
    
    
    if epsilon == 0.0 and use_singular_correction:
        du0 = du[0]  
        kernel[Nr, 0] = 2.0 * (
            np.arcsinh(dphi / du0) / dphi
            + np.arcsinh(du0 / dphi) / du0
        )
    
    
    S_pad = np.zeros((N_u_pad, N_phi_pad))
    S_pad[:Nr, :] = S * np.outer(du, np.ones(Nphi)) * dphi
    
    
    kernel_shifted = np.fft.ifftshift(kernel, axes=0)
    
    
    S_hat = np.fft.fft2(S_pad)
    K_hat = np.fft.fft2(kernel_shifted)
    V_hat = S_hat * K_hat
    V_pad = np.real(np.fft.ifft2(V_hat))
    
    
    V = V_pad[:Nr, :]
    
    
    Phi = np.zeros((Nr, Nphi))
    for j in range(Nphi):
        Phi[:, j] = -G * V[:, j] / np.sqrt(r_grid)
    
    return Phi






def compute_gravity_fd(Phi: np.ndarray,
                       r_grid: np.ndarray, phi_grid: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
    Nr, Nphi = Phi.shape
    g_r = np.zeros_like(Phi)
    g_phi = np.zeros_like(Phi)
    
    
    
    
    
    
    for i in range(1, Nr - 1):
        dr_fwd = r_grid[i + 1] - r_grid[i]   
        dr_bwd = r_grid[i] - r_grid[i - 1]    
        dPhi_dr = (
            Phi[i + 1, :] * dr_bwd / (dr_fwd * (dr_fwd + dr_bwd))
            + Phi[i, :] * (dr_fwd - dr_bwd) / (dr_fwd * dr_bwd)
            - Phi[i - 1, :] * dr_fwd / (dr_bwd * (dr_fwd + dr_bwd))
        )
        g_r[i, :] = -dPhi_dr
    
    dr0 = r_grid[1] - r_grid[0]
    g_r[0, :] = -(Phi[1, :] - Phi[0, :]) / dr0
    dr_end = r_grid[-1] - r_grid[-2]
    g_r[-1, :] = -(Phi[-1, :] - Phi[-2, :]) / dr_end
    
    
    dphi = phi_grid[1] - phi_grid[0]
    for j in range(Nphi):
        jp1 = (j + 1) % Nphi
        jm1 = (j - 1) % Nphi
        g_phi[:, j] = -(1.0 / r_grid) * (Phi[:, jp1] - Phi[:, jm1]) / (2.0 * dphi)
    
    return g_r, g_phi






def phi_exponential_disk_analytic(r: np.ndarray, Sigma_0: float, r_0: float,
                                 G: float = 1.0) -> np.ndarray:
    from scipy.special import iv as I_v, kv as K_v
    y = r / (2.0 * r_0)
    Phi = -np.pi * G * Sigma_0 * r * (
        I_v(0, y) * K_v(1, y) - I_v(1, y) * K_v(0, y)
    )
    return Phi


def gr_exponential_disk_analytic(r: np.ndarray, Sigma_0: float, r_0: float,
                                 G: float = 1.0) -> np.ndarray:
    dr = r * 1e-5  
    Phi_plus = phi_exponential_disk_analytic(r + dr, Sigma_0, r_0, G)
    Phi_minus = phi_exponential_disk_analytic(r - dr, Sigma_0, r_0, G)
    return -(Phi_plus - Phi_minus) / (2.0 * dr)


def gr_constant_disk_analytic(r: np.ndarray, Sigma_0: float,
                              r_in: float, r_out: float,
                              G: float = 1.0,
                              epsilon: float = 0.0,
                              n_rp: int = 1024,
                              n_phi: int = 513) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    if epsilon > 0.0:
        return gr_constant_disk_axisym_quadrature_softened(
            r=r, Sigma_0=Sigma_0, r_in=r_in, r_out=r_out,
            epsilon=epsilon, G=G, n_rp=n_rp, n_phi=n_phi
        )
    return gr_constant_disk_axisym_quadrature(r=r, Sigma_0=Sigma_0, r_in=r_in, r_out=r_out, G=G)


def phi_constant_disk_axisym_quadrature(r: np.ndarray, Sigma_0: float,
                                        r_in: float, r_out: float,
                                        G: float = 1.0,
                                        epsabs: float = 1e-9,
                                        epsrel: float = 1e-8,
                                        limit: int = 400) -> np.ndarray:
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


def phi_constant_disk_axisym_quadrature_softened(
    r: np.ndarray,
    Sigma_0: float,
    r_in: float,
    r_out: float,
    epsilon: float,
    G: float = 1.0,
    n_rp: int = 1024,
    n_phi: int = 513,
) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    rp = np.linspace(r_in, r_out, int(n_rp), dtype=np.float64)
    phi = np.linspace(0.0, 2.0 * np.pi, int(n_phi), dtype=np.float64)
    cos_phi = np.cos(phi)

    out = np.zeros_like(r)
    eps2 = float(epsilon) ** 2

    for i, r_val in enumerate(r):
        denom = np.sqrt(
            np.maximum(
                r_val**2 + rp[:, None] ** 2 - 2.0 * r_val * rp[:, None] * cos_phi[None, :] + eps2,
                1e-30,
            )
        )
        inner = np.trapezoid(1.0 / denom, phi, axis=1)   
        outer = np.trapezoid(rp * inner, rp)             
        out[i] = -G * Sigma_0 * outer

    return out


def gr_constant_disk_axisym_quadrature(r: np.ndarray, Sigma_0: float,
                                       r_in: float, r_out: float,
                                       G: float = 1.0,
                                       epsabs: float = 1e-9,
                                       epsrel: float = 1e-8,
                                       limit: int = 400) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    phi = phi_constant_disk_axisym_quadrature(
        r, Sigma_0=Sigma_0, r_in=r_in, r_out=r_out, G=G,
        epsabs=epsabs, epsrel=epsrel, limit=limit
    )

    dphi_dr = np.gradient(phi, r, edge_order=2)
    return -dphi_dr


def gr_constant_disk_axisym_quadrature_softened(
    r: np.ndarray,
    Sigma_0: float,
    r_in: float,
    r_out: float,
    epsilon: float,
    G: float = 1.0,
    n_rp: int = 1024,
    n_phi: int = 513,
) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    phi = phi_constant_disk_axisym_quadrature_softened(
        r=r, Sigma_0=Sigma_0, r_in=r_in, r_out=r_out,
        epsilon=epsilon, G=G, n_rp=n_rp, n_phi=n_phi
    )
    dphi_dr = np.gradient(phi, r, edge_order=2)
    return -dphi_dr






def evaluate_integral_at_points(r_eval: np.ndarray, phi_eval: np.ndarray,
                                Sigma: np.ndarray,
                                r_grid: np.ndarray, phi_grid: np.ndarray,
                                dr: np.ndarray, dphi: float,
                                G: float = 1.0,
                                epsilon: float = 0.0,
                                use_singular_correction: bool = False
                                ) -> np.ndarray:
    N = len(r_eval)
    Nr_src = len(r_grid)
    Nphi_src = len(phi_grid)
    
    
    R_SRC, PHI_SRC = np.meshgrid(r_grid, phi_grid, indexing='ij')
    DR_SRC = np.outer(dr, np.ones(Nphi_src))
    weight = Sigma * R_SRC * DR_SRC * dphi  
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
            
            i_self = np.argmin(np.abs(r_grid - r_eval[n]))
            j_self = np.argmin(np.abs(phi_grid - phi_eval[n]))
            k_self = i_self * Nphi_src + j_self
            
            
            r_dist = abs(r_eval[n] - r_grid[i_self])
            phi_dist = abs(phi_eval[n] - phi_grid[j_self])
            is_on_grid = (r_dist < 0.5 * dr[i_self] and phi_dist < 0.5 * dphi)
            
            if is_on_grid:
                
                mask = np.ones(len(r_src_flat), dtype=bool)
                mask[k_self] = False
                
                inv_dist = np.zeros_like(dist_sq)
                inv_dist[mask] = 1.0 / np.sqrt(np.maximum(dist_sq[mask], 1e-30))
                Phi_eval[n] = -G * np.dot(weight_flat[mask], inv_dist[mask])
                
                
                r_s = r_grid[i_self]
                du_cell = dr[i_self] / r_s  
                S_self = r_s**1.5 * Sigma[i_self, j_self]
                V_self = -2.0 * G * S_self * (
                    np.arcsinh(dphi / du_cell) / dphi
                    + np.arcsinh(du_cell / dphi) / du_cell
                )
                Phi_eval[n] += V_self / np.sqrt(r_s)
            else:
                
                inv_dist = 1.0 / np.sqrt(np.maximum(dist_sq, 1e-30))
                Phi_eval[n] = -G * np.dot(weight_flat, inv_dist)
        else:
            inv_dist = 1.0 / np.sqrt(np.maximum(dist_sq, 1e-30))
            Phi_eval[n] = -G * np.dot(weight_flat, inv_dist)
    
    return Phi_eval






def rel_error_L2(pred: np.ndarray, ref: np.ndarray, eta: float = 1e-10
                 ) -> float:
    return np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + eta)


def rel_error_L2_with_floor(pred: np.ndarray, ref: np.ndarray,
                            floor: float = 0.0, eta: float = 1e-10) -> float:
    denom = max(np.linalg.norm(ref), floor)
    return np.linalg.norm(pred - ref) / (denom + eta)


def max_error(pred: np.ndarray, ref: np.ndarray) -> float:
    return np.max(np.abs(pred - ref))


def rel_error_pointwise(pred: np.ndarray, ref: np.ndarray, eta: float = 1e-10
                        ) -> np.ndarray:
    return np.abs(pred - ref) / (np.abs(ref) + eta)
