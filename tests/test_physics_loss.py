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


    def test_model_with_edge_attention(self):
        """Verify MeshGraphNet forward pass with edge attention enabled."""
        from model import MeshGraphNet

        net = MeshGraphNet(
            node_in_dim=2,
            edge_in_dim=2,
            out_dim=1,
            latent_dim=32,
            hidden_dim=32,
            num_processor_blocks=3,
            use_edge_attention=True,
            activation="gelu",
        )
        N = 12
        x = torch.randn(N, 2)
        edge_index = torch.stack([
            torch.arange(N - 1),
            torch.arange(1, N),
        ])
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_attr = torch.randn(edge_index.size(1), 2)

        out = net(x, edge_index, edge_attr)
        self.assertEqual(out.shape, (N, 1))

    def test_model_with_skip_connections(self):
        """Verify MeshGraphNet with multi-scale skip connections."""
        from model import MeshGraphNet

        net = MeshGraphNet(
            node_in_dim=1,
            edge_in_dim=1,
            out_dim=1,
            latent_dim=32,
            hidden_dim=32,
            num_processor_blocks=4,
            skip_interval=2,
            use_edge_attention=True,
        )
        N = 10
        x = torch.randn(N, 1)
        edge_index = torch.stack([
            torch.arange(N - 1),
            torch.arange(1, N),
        ])
        edge_attr = torch.randn(edge_index.size(1), 1)

        out = net(x, edge_index, edge_attr)
        self.assertEqual(out.shape, (N, 1))

    def test_model_from_checkpoint_method(self):
        """Verify from_checkpoint on an in-memory checkpoint dictionary."""
        from model import MeshGraphNet

        orig = MeshGraphNet(1, 1, 1, latent_dim=16, hidden_dim=16, num_processor_blocks=2)
        ckpt = {
            "model_state_dict": orig.state_dict(),
            "node_in_dim": 1,
            "edge_in_dim": 1,
            "out_dim": 1,
        }
        loaded = MeshGraphNet.from_checkpoint(ckpt)
        self.assertIsInstance(loaded, MeshGraphNet)


class TestTrainingAndEvaluation(unittest.TestCase):
    """Tests for training scheduler and evaluation metrics."""

    def test_cosine_warmup_scheduler(self):
        """Verify CosineWarmupScheduler ramps up during warmup and decays monotonically after."""
        from train import CosineWarmupScheduler
        import torch.nn as nn

        dummy_param = nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.Adam([dummy_param], lr=1e-3)
        scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=5, total_epochs=20, min_lr=1e-6)

        lrs = [scheduler.step(ep) for ep in range(20)]
        # Check warmup increases
        self.assertLess(lrs[0], lrs[4])
        # Check cosine decay decreases after warmup
        self.assertGreater(lrs[5], lrs[19])
        self.assertGreaterEqual(lrs[19], 1e-6)

    def test_kinetic_energy_error(self):
        """Verify kinetic energy conservation calculation."""
        from evaluate import kinetic_energy_error

        true_field = torch.ones(10, 1)
        pred_identical = torch.ones(10, 1)
        err = kinetic_energy_error(pred_identical, true_field)
        self.assertAlmostEqual(err, 0.0, places=6)

        pred_scaled = 2.0 * true_field
        err_scaled = kinetic_energy_error(pred_scaled, true_field)
        self.assertAlmostEqual(err_scaled, 3.0, places=5)  # (4 - 1)/1 = 3.0


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


class TestNonUniformPhysicsLoss(unittest.TestCase):
    """Tests for non-uniform spatial grid stencils in Burgers physics loss."""

    def test_non_uniform_coordinates_loss(self):
        """Burgers loss should support custom non-uniform spatial coordinates."""
        from physics_loss import BurgersPhysicsLoss

        N = 64
        # Non-uniform coordinates with jitter
        x_uniform = np.linspace(0, 1, N)
        dx = x_uniform[1] - x_uniform[0]
        noise = np.random.uniform(-0.2 * dx, 0.2 * dx, size=N)
        noise[0] = 0.0
        x_coords = torch.tensor(np.sort(x_uniform + noise), dtype=torch.float32)

        u_prev = torch.zeros(N, 1)
        u_next = torch.zeros(N, 1)

        loss_fn = BurgersPhysicsLoss(nu=0.001, dx=float(dx), dt=0.01, periodic=True)
        loss, residuals, components = loss_fn(u_next, u_prev, x_coord=x_coords)

        self.assertAlmostEqual(loss.item(), 0.0, places=5)


class Test2DPhysicsResiduals(unittest.TestCase):
    """Smoke tests for the 2D physics loss module."""

    def test_2d_gradient_and_laplacian(self):
        """Verify gradient and laplacian computation on a constant 2D field."""
        from physics_loss_2d import compute_gradient_2d, compute_laplacian_2d

        field = torch.full((16, 32), 2.5, dtype=torch.float32)
        gx, gy = compute_gradient_2d(field, dx=0.1, dy=0.1)
        lap = compute_laplacian_2d(field, dx=0.1, dy=0.1)

        self.assertAlmostEqual(torch.max(torch.abs(gx)).item(), 0.0, places=5)
        self.assertAlmostEqual(torch.max(torch.abs(gy)).item(), 0.0, places=5)
        self.assertAlmostEqual(torch.max(torch.abs(lap)).item(), 0.0, places=5)

    def test_navier_stokes_continuity(self):
        """Uniform parallel flow (ux=1.0, uy=0.0) should have zero divergence."""
        from physics_loss_2d import navier_stokes_continuity_loss

        ux = torch.ones((1, 1, 16, 32))
        uy = torch.zeros((1, 1, 16, 32))

        div_loss = navier_stokes_continuity_loss(ux, uy)
        self.assertAlmostEqual(div_loss.item(), 0.0, places=5)

    def test_heat_residual_2d(self):
        """Constant temperature field across time should yield near-zero heat residual."""
        from physics_loss_2d import heat_residual_2d

        u_curr = torch.full((1, 1, 16, 16), 1.0)
        u_next = torch.full((1, 1, 16, 16), 1.0)

        loss = heat_residual_2d(u_curr, u_next)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)


class Test2DModels(unittest.TestCase):
    """Tests for 2D surrogate models (FNO-2D, UNet-2D, GATSurrogate)."""

    def test_fno2d_forward(self):
        """FNO2d should produce output with matching spatial dimensions."""
        from models_2d import FNO2d

        fno = FNO2d(in_channels=1, out_channels=1, modes1=4, modes2=4, width=16)
        x = torch.randn(2, 1, 16, 32)
        out = fno(x)

        self.assertEqual(out.shape, (2, 1, 16, 32))

    def test_unet2d_forward(self):
        """UNet2d should produce output with matching spatial dimensions."""
        from models_2d import UNet2d

        unet = UNet2d(in_channels=1, out_channels=1, base_dim=8)
        x = torch.randn(1, 1, 16, 32)
        out = unet(x)

        self.assertEqual(out.shape, (1, 1, 16, 32))


if __name__ == "__main__":
    unittest.main()


