"""
scripts/train_2d_gnn.py
Trains a Geometry-Conditioned 2D Neural Surrogate (UNet2D / Neural Operator)
on 2D fluid flows past geometric bluff bodies (Cylinder, Airfoil, Square, Step).

The model takes as input:
  - Channel 0: Current velocity field snapshot u(t)
  - Channel 1: Arbitrary obstacle binary mask M(x, y)
and predicts:
  - Channel 0: Next-step physical velocity field u(t+1)

Saves trained checkpoint to: models/gnn_2d_obstacle_surrogate.pt
"""

import os
import sys
import json
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Ensure local script directory is on path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "scripts"))

from models_2d import UNet2d

DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Flow2DObstacleDataset(Dataset):
    """
    Dataset of physical flow pairs: (u(t), mask) -> u(t+1)
    aggregated across diverse 2D bluff body geometries.
    """
    def __init__(self, shape_names=None):
        if shape_names is None:
            shape_names = ["cylinder", "airfoil", "square", "step", "multi_cylinder"]

        self.samples = []
        for name in shape_names:
            fname = os.path.join(DATASETS_DIR, f"{name}_flow_2d.hdf5")
            if not os.path.exists(fname):
                continue
            try:
                with h5py.File(fname, "r") as f:
                    tensor = f["tensor"][:]  # [N_traj, T, Ny, Nx]
                    mask = f["obstacle_mask"][:] if "obstacle_mask" in f else None

                    N, T, Ny, Nx = tensor.shape
                    for n in range(N):
                        obs_m = mask[n] if mask is not None else np.zeros((Ny, Nx), dtype=bool)
                        for t in range(T - 1):
                            u_curr = tensor[n, t].astype(np.float32)
                            u_next = tensor[n, t + 1].astype(np.float32)

                            # Normalize per trajectory
                            max_val = max(np.max(u_curr), 1e-4)
                            u_curr_norm = u_curr / max_val
                            u_next_norm = u_next / max_val

                            # Input: [2, Ny, Nx] = [u_curr, mask]
                            x_in = np.stack([u_curr_norm, obs_m.astype(np.float32)], axis=0)
                            # Target: [1, Ny, Nx] = [u_next]
                            y_out = np.expand_dims(u_next_norm, axis=0)

                            self.samples.append((x_in, y_out))
            except Exception as e:
                print(f"[WARN] Could not load {fname}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_in, y_out = self.samples[idx]
        return torch.from_numpy(x_in), torch.from_numpy(y_out)


def train_surrogate(epochs=10, batch_size=4, lr=1e-3):
    print("=" * 72)
    print("  TRAINING 2D GEOMETRY-CONDITIONED SURROGATE MODEL")
    print(f"  Device: {DEVICE} | Epochs: {epochs} | Batch Size: {batch_size}")
    print("=" * 72)

    dataset = Flow2DObstacleDataset()
    if len(dataset) == 0:
        print("[ERROR] No 2D datasets found in datasets/ directory.")
        return None

    print(f"[1/4] Loaded {len(dataset)} flow transition pairs across bluff bodies.")

    # Model: in_channels=2 (velocity + obstacle mask), out_channels=1
    model = UNet2d(in_channels=2, out_channels=1, base_dim=24).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[2/4] Initialized UNet2D Surrogate ({param_count:,} parameters).")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = []
    start_time = time.time()

    print("\n[3/4] Starting training loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        boundary_penalty = 0.0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            preds = model(x_batch)

            # Supervised field MSE
            mse_loss = criterion(preds, y_batch)

            # Physics boundary constraint: flow inside obstacle mask (channel 1) should be 0
            mask_batch = x_batch[:, 1:2, :, :]
            bc_loss = torch.mean(torch.abs(preds * mask_batch))

            loss = mse_loss + 0.3 * bc_loss
            loss.backward()
            optimizer.step()

            epoch_loss += mse_loss.item() * len(x_batch)
            boundary_penalty += bc_loss.item() * len(x_batch)

        scheduler.step()
        avg_loss = epoch_loss / len(dataset)
        avg_bc = boundary_penalty / len(dataset)
        history.append({
            "epoch": epoch,
            "mse_loss": round(avg_loss, 6),
            "boundary_loss": round(avg_bc, 6),
            "lr": round(scheduler.get_last_lr()[0], 6),
        })

        if epoch % 2 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | MSE: {avg_loss:.5e} | Boundary Leak: {avg_bc:.5e} | LR: {scheduler.get_last_lr()[0]:.2e}")

    total_time = time.time() - start_time
    print(f"\n[4/4] Training complete in {total_time:.1f} seconds!")

    # Save model and log
    checkpoint_path = os.path.join(MODELS_DIR, "gnn_2d_obstacle_surrogate.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "arch": "UNet2d",
        "in_channels": 2,
        "out_channels": 1,
        "base_dim": 24,
        "param_count": param_count,
        "final_loss": history[-1]["mse_loss"],
        "epochs": epochs,
    }, checkpoint_path)
    print(f"  [SAVED] Checkpoint: {checkpoint_path}")

    log_path = os.path.join(MODELS_DIR, "gnn_2d_training_log.json")
    with open(log_path, "w") as f:
        json.dump({
            "status": "success",
            "samples": len(dataset),
            "param_count": param_count,
            "total_time_sec": round(total_time, 2),
            "history": history,
        }, f, indent=2)
    print(f"  [SAVED] Training log: {log_path}")

    return checkpoint_path


if __name__ == "__main__":
    train_surrogate(epochs=10, batch_size=8, lr=1.5e-3)
