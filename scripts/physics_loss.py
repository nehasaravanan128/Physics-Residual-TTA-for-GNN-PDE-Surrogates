"""
scripts/physics_loss.py
Differentiable physics-residual loss functions for PDE surrogates.

Implements the PINN-style unsupervised PDE residual for Burgers' equation:
    R(u) = ∂u/∂t + u * ∂u/∂x - ν * ∂²u/∂x² = 0

Given current state u_t and surrogate predicted state u_{t+1},
this computes the physical discrepancy without requiring ground-truth labels.
"""

import torch
import torch.nn as nn


class BurgersPhysicsLoss(nn.Module):
    """
    Computes the 1D viscous Burgers' equation residual:
        ∂u/∂t + u * ∂u/∂x - ν * ∂²u/∂x² = 0

    Args:
        nu: Kinematic viscosity coefficient (default: 0.001 for PDEBench Burgers)
        dx: Spatial grid spacing (default: 1.0 / 1024)
        dt: Temporal timestep step size (default: 0.01)
        periodic: Whether the domain has periodic boundary conditions (PDEBench uses periodic)
        scheme: Time discretization ('implicit' at u_{t+1}, or 'crank_nicolson' at average u)
    """

    def __init__(
        self,
        nu: float = 0.001,
        dx: float = 1.0 / 1024.0,
        dt: float = 0.01,
        periodic: bool = True,
        scheme: str = "implicit",
        x_coord: torch.Tensor = None,
    ):
        super().__init__()
        self.nu = float(nu)
        self.dx = float(dx)
        self.dt = float(dt)
        self.periodic = bool(periodic)
        self.scheme = scheme
        self.x_coord = x_coord

    def spatial_derivatives(self, u: torch.Tensor, x_coord: torch.Tensor = None):
        """
        Computes 1st (∂u/∂x) and 2nd (∂²u/∂x²) spatial derivatives using central differences.
        Supports both uniform grids (dx) and arbitrary non-uniform/perturbed geometric meshes (x_coord).
        Input u can be [N] or [N, 1].
        """
        u_flat = u.squeeze(-1) if u.dim() > 1 else u
        coords = x_coord if x_coord is not None else self.x_coord

        if coords is not None:
            # --- Non-uniform geometric mesh stencil ---
            coords_flat = coords.squeeze(-1) if coords.dim() > 1 else coords
            N = u_flat.shape[-1]
            device = u_flat.device
            coords_flat = coords_flat.to(device)

            if self.periodic:
                u_right = torch.roll(u_flat, shifts=-1, dims=-1)
                u_left = torch.roll(u_flat, shifts=1, dims=-1)
                
                # Compute forward and backward spacings h_plus, h_minus with periodic boundary wrapping
                L = (coords_flat[-1] - coords_flat[0]) + (coords_flat[1] - coords_flat[0]) # domain length
                x_right = torch.roll(coords_flat, shifts=-1, dims=-1)
                x_left = torch.roll(coords_flat, shifts=1, dims=-1)
                
                h_plus = x_right - coords_flat
                h_plus[-1] = (coords_flat[0] + L) - coords_flat[-1] # periodic wrap on right boundary
                
                h_minus = coords_flat - x_left
                h_minus[0] = coords_flat[0] - (coords_flat[-1] - L) # periodic wrap on left boundary
            else:
                u_right = torch.cat([u_flat[1:], u_flat[-1:]], dim=-1)
                u_left = torch.cat([u_flat[:1], u_flat[:-1]], dim=-1)
                dx_fwd = coords_flat[1:] - coords_flat[:-1]
                h_plus = torch.cat([dx_fwd, dx_fwd[-1:]], dim=-1)
                dx_bwd = coords_flat[1:] - coords_flat[:-1]
                h_minus = torch.cat([dx_bwd[:1], dx_bwd], dim=-1)

            # Non-uniform 1st derivative (central difference): ∂u/∂x = (u_{i+1} - u_{i-1}) / (h_+ + h_-)
            du_dx = (u_right - u_left) / torch.clamp(h_plus + h_minus, min=1e-7)

            # Non-uniform 2nd derivative (conservative Laplacian): 
            # ∂²u/∂x² = 2 / (h_+ + h_-) * [ (u_{i+1} - u_i)/h_+ - (u_i - u_{i-1})/h_- ]
            diff_plus = (u_right - u_flat) / torch.clamp(h_plus, min=1e-7)
            diff_minus = (u_flat - u_left) / torch.clamp(h_minus, min=1e-7)
            d2u_dx2 = 2.0 * (diff_plus - diff_minus) / torch.clamp(h_plus + h_minus, min=1e-7)

            return du_dx, d2u_dx2

        # --- Uniform grid fast-path ---
        if self.periodic:
            u_right = torch.roll(u_flat, shifts=-1, dims=-1)
            u_left = torch.roll(u_flat, shifts=1, dims=-1)
        else:
            u_right = torch.cat([u_flat[1:], u_flat[-1:]], dim=-1)
            u_left = torch.cat([u_flat[:1], u_flat[:-1]], dim=-1)

        du_dx = (u_right - u_left) / (2.0 * self.dx)
        d2u_dx2 = (u_right - 2.0 * u_flat + u_left) / (self.dx ** 2)

        return du_dx, d2u_dx2

    def forward(self, u_next: torch.Tensor, u_curr: torch.Tensor, nu: float = None, x_coord: torch.Tensor = None):
        """
        Computes the physics residual R(u) and the mean squared loss.

        Args:
            u_next: Model prediction for state at t + dt, shape [N] or [N, 1]
            u_curr: Input physical state at time t, shape [N] or [N, 1]
            nu: Optional override for kinematic viscosity (useful under distribution shift)

        Returns:
            loss: Scalar tensor representing MSE of PDE residual
            residuals: Tensor of per-node residual values
            components: Dict with individual physical terms for inspection
        """
        effective_nu = nu if nu is not None else self.nu

        u_next_flat = u_next.squeeze(-1) if u_next.dim() > 1 else u_next
        u_curr_flat = u_curr.squeeze(-1) if u_curr.dim() > 1 else u_curr

        # 1. Temporal derivative ∂u/∂t
        du_dt = (u_next_flat - u_curr_flat) / self.dt

        # 2. Spatial terms
        if self.scheme == "crank_nicolson":
            u_mid = 0.5 * (u_next_flat + u_curr_flat)
            du_dx, d2u_dx2 = self.spatial_derivatives(u_mid, x_coord=x_coord)
            u_eval = u_mid
        else:
            du_dx, d2u_dx2 = self.spatial_derivatives(u_next_flat, x_coord=x_coord)
            u_eval = u_next_flat

        # Convective (non-linear advection) term: u * ∂u/∂x
        advection = u_eval * du_dx

        # Viscous diffusion term: ν * ∂²u/∂x²
        diffusion = effective_nu * d2u_dx2

        # Full PDE residual: R = ∂u/∂t + u * ∂u/∂x - ν * ∂²u/∂x²
        residuals = du_dt + advection - diffusion

        # Mean squared residual loss
        loss = torch.mean(residuals ** 2)

        components = {
            "du_dt": du_dt,
            "advection": advection,
            "diffusion": diffusion,
            "residual_mean": torch.mean(torch.abs(residuals)).item(),
            "residual_max": torch.max(torch.abs(residuals)).item(),
        }

        return loss, residuals, components


if __name__ == "__main__":
    # Test with synthetic wave
    x = torch.linspace(0, 1, 1024)
    u0 = torch.sin(2 * torch.pi * x)
    u1 = torch.sin(2 * torch.pi * (x - 0.01)) * 0.99
    loss_fn = BurgersPhysicsLoss(nu=0.001)
    loss, res, comp = loss_fn(u1, u0)
    print(f"Sanity Check: Loss = {loss.item():.6f}, Max Residual = {comp['residual_max']:.6f}")
