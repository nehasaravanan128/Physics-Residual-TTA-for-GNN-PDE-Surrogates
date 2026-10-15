"""
tests/test_physics_loss.py
Smoke tests for the differentiable physics residual engine and TTA core logic.

Run with:
    python -m unittest discover tests/ -v
    or
    python tests/test_physics_loss.py
"""

import sys
import os
import unittest

import torch
import numpy as np

# Add project root, scripts, and files (5) to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "files (5)"))


class TestPhysicsLoss(unittest.TestCase):
    """Smoke tests for the Burgers PDE residual computation."""

    def test_import_physics_loss(self):
        """Verify that the physics_loss module imports correctly."""
        import physics_loss
        self.assertTrue(hasattr(physics_loss, "BurgersPhysicsLoss"))

    def test_zero_residual_for_stationary_field(self):
        """A constant field u(x,t) = C should have near-zero residual (du/dt=0, du/dx=0, d2u/dx2=0)."""
        from physics_loss import BurgersPhysicsLoss

        N = 128
        C = 0.5
        u_prev = torch.full((N, 1), C, dtype=torch.float32)
        u_next = torch.full((N, 1), C, dtype=torch.float32)
        dx = 1.0 / N
        dt = 0.01
        nu = 0.001

        loss_fn = BurgersPhysicsLoss(nu=nu, dx=dx, dt=dt, periodic=True)
        loss, residuals, components = loss_fn(u_next, u_prev)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertAlmostEqual(
            loss.item(), 0.0, places=5, msg="Constant field should yield near-zero PDE residual"
        )

    def test_residual_is_differentiable(self):
        """The physics loss must propagate gradients through u_next."""
        from physics_loss import BurgersPhysicsLoss

        N = 64
        u_prev = torch.randn(N, 1)
        u_next = torch.randn(N, 1, requires_grad=True)
        dx = 1.0 / N
        dt = 0.01
        nu = 0.001

        loss_fn = BurgersPhysicsLoss(nu=nu, dx=dx, dt=dt, periodic=True)
        loss, residuals, components = loss_fn(u_next, u_prev)
        loss.backward()

        self.assertIsNotNone(u_next.grad, "Gradients should flow through the physics loss")
        self.assertEqual(u_next.grad.shape, (N, 1))

    def test_residual_positive(self):
        """Physics residual loss should always be non-negative (squared terms)."""
        from physics_loss import BurgersPhysicsLoss

        N = 128
        u_prev = torch.randn(N, 1)
        u_next = torch.randn(N, 1)
        dx = 1.0 / N
        dt = 0.01
        nu = 0.001

        loss_fn = BurgersPhysicsLoss(nu=nu, dx=dx, dt=dt, periodic=True)
        loss, residuals, components = loss_fn(u_next, u_prev)
        self.assertGreaterEqual(loss.item(), 0.0)


class TestModelArchitecture(unittest.TestCase):
    """Smoke tests for the MeshGraphNet surrogate model."""

    def test_import_model(self):
        """Verify that the model module imports correctly."""
        import model
        self.assertTrue(hasattr(model, "MeshGraphNet"))

    def test_model_forward_pass(self):
        """MeshGraphNet should produce output with correct shape for a small graph."""
        from model import MeshGraphNet

        net = MeshGraphNet(
            node_in_dim=1,
            edge_in_dim=1,
            out_dim=1,
            latent_dim=32,
            hidden_dim=32,
            num_processor_blocks=2,
        )
        N = 16
        # Build a simple chain graph
        x = torch.randn(N, 1)
        edge_index = torch.stack([
            torch.arange(N - 1),
            torch.arange(1, N),
        ])
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_attr = torch.randn(edge_index.size(1), 1)

        out = net(x, edge_index, edge_attr)

        self.assertEqual(
            out.shape, (N, 1), f"Expected output shape ({N}, 1), got {out.shape}"
        )

    def test_model_is_differentiable(self):
        """Verify gradients flow through the model."""
        from model import MeshGraphNet

        net = MeshGraphNet(
            node_in_dim=1,
            edge_in_dim=1,
            out_dim=1,
            latent_dim=32,
            hidden_dim=32,
            num_processor_blocks=2,
        )
        N = 8
        x = torch.randn(N, 1)
        edge_index = torch.stack([
            torch.arange(N - 1),
            torch.arange(1, N),
        ])
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_attr = torch.randn(edge_index.size(1), 1)

        out = net(x, edge_index, edge_attr)
        loss = out.sum()
        loss.backward()

        # Check that at least some parameters have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in net.parameters()
            if p.requires_grad
        )
        self.assertTrue(has_grad, "Model parameters should receive gradients")


class TestTTAEngine(unittest.TestCase):
    """Smoke tests for the Test-Time Adaptation logic."""

    def test_import_tta(self):
        """Verify that the TTA module imports correctly."""
        import tta
        self.assertTrue(hasattr(tta, "PhysicsResidualTTA"))

    def test_anchor_loss_computation(self):
        """Anchor loss = MSE between adapted and baseline predictions."""
        N = 64
        u_baseline = torch.randn(N, 1)
        u_adapted = u_baseline.clone()  # identical -> zero anchor loss

        anchor_loss = torch.mean((u_adapted - u_baseline) ** 2)
        self.assertAlmostEqual(anchor_loss.item(), 0.0, places=7)

    def test_anchor_loss_nonzero_for_perturbed(self):
        """Anchor loss should be > 0 when adapted != baseline."""
        N = 64
        u_baseline = torch.randn(N, 1)
        u_adapted = u_baseline + 0.1 * torch.randn(N, 1)

        anchor_loss = torch.mean((u_adapted - u_baseline) ** 2)
        self.assertGreater(anchor_loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
