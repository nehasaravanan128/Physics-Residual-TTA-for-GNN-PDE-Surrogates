"""
scripts/tta.py
Physics-Residual Test-Time Adaptation (TTA) engine for MeshGraphNet PDE Surrogates.

Algorithm:
1. Input: Test sample (x_t, mesh) WITHOUT ground truth labels.
2. Freeze GNN encoder and all processor (message-passing) blocks.
3. Keep baseline surrogate prediction u_0 as an anchor.
4. Adapt only the decoder MLP parameters for K gradient steps using
   unsupervised physics-residual loss:
       min_{θ_dec}  L_physics(u_{t+1}, u_t; ν) + λ * ||u_{t+1} - u_0||²
5. Evaluate adapted prediction u_adapted.
6. Restore baseline model weights to guarantee clean inference on subsequent samples.

Usage:
    python scripts/tta.py --data_path <hdf5> --checkpoint_path <model.pt> \
                          --traj 0 --timestep 10 --tta_steps 10 --tta_lr 0.0002
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# Ensure local imports work
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "files (5)")))

from model import MeshGraphNet
from data import load_burgers
from physics_loss import BurgersPhysicsLoss


class PhysicsResidualTTA:
    """
    Test-Time Adaptation manager for GNN PDE surrogates.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cpu"),
        dx: float = 1.0 / 1024.0,
        dt: float = 0.01,
        nu: float = 0.001,
        anchor_weight: float = 0.05,
    ):
        self.model = model.to(device)
        self.device = device
        self.dx = dx
        self.dt = dt
        self.nu = nu
        self.anchor_weight = anchor_weight
        self.loss_fn = BurgersPhysicsLoss(nu=nu, dx=dx, dt=dt, periodic=True)

    def adapt_sample(
        self,
        sample,
        tta_steps: int = 10,
        tta_lr: float = 0.0002,
        nu_override: float = None,
        anchor_override: float = None,
        x_coord_override: torch.Tensor = None,
    ):
        """
        Runs Test-Time Adaptation on a single graph test sample.

        Args:
            sample: PyG Data object with x, edge_index, edge_attr (and optional y for benchmarking)
            tta_steps: Number of test-time gradient steps (default: 10)
            tta_lr: Learning rate for decoder adaptation (default: 2e-4)
            nu_override: Viscosity value (allows testing under parameter distribution shift)
            anchor_override: Custom anchor regularization weight

        Returns:
            dict containing:
                - baseline_pred: list of floats
                - adapted_pred: list of floats
                - loss_history: list of float physics loss per step
                - baseline_residual: float
                - adapted_residual: float
                - true_y: list of floats (if sample.y exists)
                - baseline_mse: float (if sample.y exists)
                - adapted_mse: float (if sample.y exists)
                - improvement_pct: float
        """
        effective_nu = nu_override if nu_override is not None else self.nu
        effective_anchor = anchor_override if anchor_override is not None else self.anchor_weight

        coords = x_coord_override if x_coord_override is not None else getattr(sample, "pos", getattr(sample, "x_coord", None))
        if coords is not None and not isinstance(coords, torch.Tensor):
            coords = torch.tensor(coords, dtype=torch.float32, device=self.device)

        # 1. Forward pass on unadapted baseline
        self.model.eval()
        with torch.no_grad():
            x = sample.x.to(self.device)
            edge_index = sample.edge_index.to(self.device)
            edge_attr = sample.edge_attr.to(self.device)
            base_pred_t = self.model(x, edge_index, edge_attr).squeeze(-1)

            # Baseline physics residual
            base_phys_loss, _, _ = self.loss_fn(base_pred_t, x.squeeze(-1), nu=effective_nu, x_coord=coords)

        # Baseline ground truth evaluation if y is provided
        has_ground_truth = hasattr(sample, "y") and sample.y is not None
        base_mse = None
        base_rel_mse = None
        true_y_t = None
        if has_ground_truth:
            true_y_t = sample.y.to(self.device).squeeze(-1)
            base_mse = torch.mean((base_pred_t - true_y_t) ** 2).item()
            scale = torch.mean(true_y_t ** 2).item() + 1e-8
            base_rel_mse = base_mse / scale

        # 2. Freeze encoder & processor; only adapt decoder
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        for p in self.model.processor_blocks.parameters():
            p.requires_grad = False
        for p in self.model.decoder.parameters():
            p.requires_grad = True

        # Cache baseline decoder weights to restore after adaptation
        cached_decoder_state = copy.deepcopy(self.model.decoder.state_dict())
        best_decoder_state = copy.deepcopy(self.model.decoder.state_dict())
        best_phys_loss = float(base_phys_loss.item())

        optimizer = torch.optim.Adam(self.model.decoder.parameters(), lr=tta_lr)

        loss_history = [base_phys_loss.item()]
        true_loss_history = [base_mse] if has_ground_truth else []

        u_curr = x.squeeze(-1)

        # 3. Adaptation Loop (Unsupervised: uses ONLY physics residual loss)
        for step in range(1, tta_steps + 1):
            optimizer.zero_grad()

            pred_t = self.model(x, edge_index, edge_attr).squeeze(-1)

            # Physics residual loss
            phys_loss, _, _ = self.loss_fn(pred_t, u_curr, nu=effective_nu, x_coord=coords)

            # Anchor regularization against surrogate baseline manifold
            anchor_loss = effective_anchor * torch.mean((pred_t - base_pred_t) ** 2)

            total_loss = phys_loss + anchor_loss
            total_loss.backward()

            # Gradient clipping for numerical stability with 2nd derivatives
            torch.nn.utils.clip_grad_norm_(self.model.decoder.parameters(), max_norm=0.5)
            optimizer.step()

            pl_val = float(phys_loss.item())
            loss_history.append(pl_val)

            # Track best unsupervised physical state
            if pl_val < best_phys_loss:
                best_phys_loss = pl_val
                best_decoder_state = copy.deepcopy(self.model.decoder.state_dict())

            if has_ground_truth:
                with torch.no_grad():
                    curr_err = torch.mean((pred_t - true_y_t) ** 2).item()
                    true_loss_history.append(curr_err)

        # 4. Load the best physical state found during adaptation
        self.model.decoder.load_state_dict(best_decoder_state)

        with torch.no_grad():
            adapted_pred_t = self.model(x, edge_index, edge_attr).squeeze(-1)
            adapted_phys_loss, _, _ = self.loss_fn(adapted_pred_t, u_curr, nu=effective_nu, x_coord=coords)

        adapted_mse = None
        adapted_rel_mse = None
        improvement_pct = 0.0
        if has_ground_truth:
            adapted_mse = torch.mean((adapted_pred_t - true_y_t) ** 2).item()
            scale = torch.mean(true_y_t ** 2).item() + 1e-8
            adapted_rel_mse = adapted_mse / scale
            if base_mse > 1e-12:
                improvement_pct = ((base_mse - adapted_mse) / base_mse) * 100.0

        # 5. Restore baseline model weights to preserve model state
        self.model.decoder.load_state_dict(cached_decoder_state)

        return {
            "baseline_predicted": [float(v) for v in base_pred_t.cpu().numpy()],
            "adapted_predicted": [float(v) for v in adapted_pred_t.cpu().numpy()],
            "true": [float(v) for v in true_y_t.cpu().numpy()] if has_ground_truth else None,
            "baseline_physics_loss": float(base_phys_loss.item()),
            "adapted_physics_loss": float(adapted_phys_loss.item()),
            "physics_loss_history": [float(v) for v in loss_history],
            "true_error_history": [float(v) for v in true_loss_history] if has_ground_truth else None,
            "baseline_mse": base_mse,
            "adapted_mse": adapted_mse,
            "baseline_relative_mse": base_rel_mse,
            "adapted_relative_mse": adapted_rel_mse,
            "improvement_pct": float(improvement_pct),
            "tta_steps": tta_steps,
            "nu": effective_nu,
        }


def main():
    parser = argparse.ArgumentParser(description="Run Physics-Residual TTA on a PDE sample")
    parser.add_argument("--data_path", required=True, help="Path to Burgers HDF5 dataset")
    parser.add_argument("--checkpoint_path", required=True, help="Path to pretrained model checkpoint")
    parser.add_argument("--traj", type=int, default=0, help="Trajectory index")
    parser.add_argument("--timestep", type=int, default=10, help="Timestep index")
    parser.add_argument("--tta_steps", type=int, default=12, help="Number of adaptation steps")
    parser.add_argument("--tta_lr", type=float, default=0.00008, help="Learning rate for adaptation")
    parser.add_argument("--nu", type=float, default=0.001, help="Kinematic viscosity")
    parser.add_argument("--anchor", type=float, default=0.1, help="Anchor regularization weight")
    parser.add_argument("--json", action="store_true", help="Output pure JSON for API consumption")

    args = parser.parse_args()

    device = torch.device("cpu")

    # Load model
    ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    model = MeshGraphNet(
        node_in_dim=ckpt["node_in_dim"],
        edge_in_dim=ckpt["edge_in_dim"],
        out_dim=ckpt["out_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Load trajectory from HDF5
    import h5py
    with h5py.File(args.data_path, "r") as f:
        tensor = f["tensor"]  # [N, T, X]
        x_coord = f["x-coordinate"][:]
        N, T, X = tensor.shape
        traj_idx = min(args.traj, N - 1)
        t_idx = min(args.timestep, T - 2)

        u_curr_np = tensor[traj_idx, t_idx]
        u_next_np = tensor[traj_idx, t_idx + 1]

    # Build 1D chain graph
    src = np.arange(X - 1)
    dst = src + 1
    edge_index_np = np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    dx = x_coord[dst] - x_coord[src]
    edge_attr_np = np.concatenate([dx, -dx])[:, None]
    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32)

    x_in = torch.tensor(u_curr_np, dtype=torch.float32).unsqueeze(-1)
    x_target = torch.tensor(u_next_np, dtype=torch.float32).unsqueeze(-1)

    from torch_geometric.data import Data
    sample = Data(x=x_in, edge_index=edge_index, edge_attr=edge_attr, y=x_target)

    dx_val = float(x_coord[1] - x_coord[0])
    tta_engine = PhysicsResidualTTA(model=model, device=device, dx=dx_val, dt=0.01, nu=args.nu, anchor_weight=args.anchor)

    result = tta_engine.adapt_sample(
        sample=sample,
        tta_steps=args.tta_steps,
        tta_lr=args.tta_lr,
        nu_override=args.nu,
    )
    result["x_coord"] = [float(x) for x in x_coord]
    result["traj"] = traj_idx
    result["timestep"] = t_idx

    if args.json:
        print(json.dumps(result))
    else:
        print("=" * 60)
        print("  PHYSICS-RESIDUAL TEST-TIME ADAPTATION (TTA) REPORT")
        print("=" * 60)
        print(f"Sample: Trajectory {traj_idx}, Timestep t={t_idx}")
        print(f"Viscosity nu: {result['nu']} | Adaptation Steps: {result['tta_steps']}")
        print(f"Baseline (Before TTA) MSE:      {result['baseline_mse']:.6e}")
        print(f"Adapted (After TTA) MSE:       {result['adapted_mse']:.6e}")
        print(f"Baseline Relative MSE:         {result['baseline_relative_mse']:.6e}")
        print(f"Adapted Relative MSE:          {result['adapted_relative_mse']:.6e}")
        print(f"Accuracy Gain / Error Drop:    {result['improvement_pct']:+.2f}%")
        print(f"Physics Residual: {result['baseline_physics_loss']:.4f} -> {result['adapted_physics_loss']:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
