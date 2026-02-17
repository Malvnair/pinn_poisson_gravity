
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional




class SirenLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 omega: float = 1.0, is_first: bool = False):
        super().__init__()
        self.omega = omega
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()
    
    def _init_weights(self):
        n = self.linear.in_features
        if self.is_first:
            bound = 1.0 / n
        else:
            bound = np.sqrt(6.0 / n) / self.omega
        nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.uniform_(self.linear.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * self.linear(x))


class GravityPINN(nn.Module):
    def __init__(self, hidden_layers: int = 6, hidden_units: int = 128,
                 omega_0: float = 10.0, omega: float = 1.0,
                 r_min: float = 0.2, r_max: float = 100.0):
        super().__init__()
        
        self.r_min = r_min
        self.r_max = r_max
        self.log_r_min = np.log(r_min)
        self.log_r_max = np.log(r_max)
        
        in_dim = 3
        out_dim = 1
        
        layers = []
        
        layers.append(SirenLayer(in_dim, hidden_units, omega=omega_0, is_first=True))
        
        for _ in range(hidden_layers - 1):
            layers.append(SirenLayer(hidden_units, hidden_units, omega=omega))
        
        final = nn.Linear(hidden_units, out_dim)
        
        nn.init.uniform_(final.weight, -1e-3, 1e-3)
        nn.init.zeros_(final.bias)
        layers.append(final)
        
        self.net = nn.Sequential(*layers)
    
    def _encode_input(self, r: torch.Tensor, phi: torch.Tensor
                      ) -> torch.Tensor:
        """
        Map (r, φ) to network input (r̃, sin φ, cos φ).
        
        Parameters
        ----------
        r   : (N,) or (N,1)
        phi : (N,) or (N,1)
        
        Returns
        -------
        x : (N, 3)
        """
        r = r.reshape(-1, 1)
        phi = phi.reshape(-1, 1)
        
        
        log_r = torch.log(r)
        r_tilde = 2.0 * (log_r - self.log_r_min) / (self.log_r_max - self.log_r_min) - 1.0
        
        
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)
        
        return torch.cat([r_tilde, sin_phi, cos_phi], dim=1)
    
    def forward(self, r: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        Predict Φ(r, φ).
        
        Parameters
        ----------
        r   : (N,) radii
        phi : (N,) azimuths
        
        Returns
        -------
        Phi : (N,) potential values
        """
        x = self._encode_input(r, phi)
        return self.net(x).squeeze(-1)
    
    def gravity(self, r: torch.Tensor, phi: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gravitational acceleration via auto-differentiation.
        
            g_r  = -∂Φ/∂r
            g_φ  = -(1/r) ∂Φ/∂φ
        
        Parameters
        ----------
        r   : (N,) requires_grad=True
        phi : (N,) requires_grad=True
        
        Returns
        -------
        g_r   : (N,)
        g_phi : (N,)
        """
        r = r.requires_grad_(True)
        phi = phi.requires_grad_(True)
        
        Phi = self.forward(r, phi)
        
        
        grad_outputs = torch.ones_like(Phi)
        grads = torch.autograd.grad(
            Phi, [r, phi], grad_outputs=grad_outputs,
            create_graph=True, retain_graph=True
        )
        
        dPhi_dr = grads[0]
        dPhi_dphi = grads[1]
        
        g_r = -dPhi_dr
        g_phi = -(1.0 / r) * dPhi_dphi
        
        return g_r, g_phi






def loss_integral(model: GravityPINN,
                  r_coll: torch.Tensor, phi_coll: torch.Tensor,
                  Phi_target: torch.Tensor) -> torch.Tensor:
    """
    Physics loss: network output should match the integral value.
    
    L_int = (1/N) Σ_i (Φ_θ(r_i, φ_i) - Φ_int(r_i, φ_i))²
    """
    Phi_pred = model(r_coll, phi_coll)
    return torch.mean((Phi_pred - Phi_target)**2)


def loss_gauge_mean_zero(model: GravityPINN,
                         r_coll: torch.Tensor, phi_coll: torch.Tensor
                         ) -> torch.Tensor:
    """
    Gauge constraint: mean potential over collocation points should be zero.
    
    This removes the additive constant ambiguity.
    """
    Phi_pred = model(r_coll, phi_coll)
    return torch.mean(Phi_pred)**2


def loss_smoothness(model: GravityPINN,
                    r_coll: torch.Tensor, phi_coll: torch.Tensor
                    ) -> torch.Tensor:
    """
    Optional smoothness regularizer: penalize large second derivatives.
    Helps prevent oscillatory artifacts.
    """
    r = r_coll.requires_grad_(True)
    phi = phi_coll.requires_grad_(True)
    
    Phi = model(r, phi)
    
    
    grad_outputs = torch.ones_like(Phi)
    grads = torch.autograd.grad(Phi, [r, phi], grad_outputs=grad_outputs,
                                create_graph=True)
    dPhi_dr = grads[0]
    dPhi_dphi = grads[1]
    
    
    d2Phi_dr2 = torch.autograd.grad(dPhi_dr, r, grad_outputs=torch.ones_like(dPhi_dr),
                                     create_graph=True)[0]
    d2Phi_dphi2 = torch.autograd.grad(dPhi_dphi, phi,
                                       grad_outputs=torch.ones_like(dPhi_dphi),
                                       create_graph=True)[0]
    
    return torch.mean(d2Phi_dr2**2 + d2Phi_dphi2**2)


def loss_force(model: GravityPINN,
               r_coll: torch.Tensor, phi_coll: torch.Tensor,
               g_r_target: torch.Tensor, g_phi_target: torch.Tensor
               ) -> torch.Tensor:
    """
    Optional force-matching loss on in-plane gravity components.

    L_force = MSE(g_r_pred - g_r_target) + MSE(g_phi_pred - g_phi_target)
    """
    g_r_pred, g_phi_pred = model.gravity(r_coll, phi_coll)
    return torch.mean((g_r_pred - g_r_target)**2 + (g_phi_pred - g_phi_target)**2)


def total_loss(model: GravityPINN,
               r_coll: torch.Tensor, phi_coll: torch.Tensor,
               Phi_target: torch.Tensor,
               weights: dict,
               g_r_target: Optional[torch.Tensor] = None,
               g_phi_target: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, dict]:
    """
    Combined loss: integral + gauge + smoothness.
    
    Returns total loss and individual components for logging.
    """
    L_int = loss_integral(model, r_coll, phi_coll, Phi_target)
    L_gauge = loss_gauge_mean_zero(model, r_coll, phi_coll)
    
    L_total = (weights.get('integral', 1.0) * L_int
               + weights.get('gauge', 10.0) * L_gauge)
    
    components = {'integral': L_int.item(), 'gauge': L_gauge.item()}

    if (weights.get('force', 0.0) > 0
            and g_r_target is not None
            and g_phi_target is not None):
        L_force = loss_force(model, r_coll, phi_coll, g_r_target, g_phi_target)
        L_total += weights['force'] * L_force
        components['force'] = L_force.item()

    if weights.get('smoothness', 0.0) > 0:
        L_smooth = loss_smoothness(model, r_coll, phi_coll)
        L_total += weights['smoothness'] * L_smooth
        components['smoothness'] = L_smooth.item()
    
    components['total'] = L_total.item()
    return L_total, components
