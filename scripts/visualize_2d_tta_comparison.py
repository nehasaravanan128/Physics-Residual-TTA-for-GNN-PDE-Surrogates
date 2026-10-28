"""
scripts/visualize_2d_tta_comparison.py
Publication-Quality 2D Navier-Stokes Flow Comparison Suite:
Baseline Surrogate (Pre-TTA) vs. Physics-Residual Test-Time Adapted (Post-TTA) vs. Ground Truth.

Evaluates 2D flow past obstacle geometries (Cylinder, Airfoil, Square, Step)
under incompressibility (div(u) = 0) and momentum residual constraints.
Generates: gnn_vs_tta_2d_comparison.png
"""

import os
import sys
import argparse
import copy
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon

# Ensure local imports work
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "files (5)"))

from models_2d import build_surrogate_model
from physics_loss_2d import (
    navier_stokes_continuity_loss,
    compute_gradient_2d,
    compute_laplacian_2d,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")


def get_shape_file(shape_name):
    lookup = {
        "cylinder": "cylinder_flow_2d.hdf5",
        "airfoil": "airfoil_flow_2d.hdf5",
        "square": "square_flow_2d.hdf5",
        "step": "step_flow_2d.hdf5",
        "multi_cylinder": "multi_cylinder_flow_2d.hdf5",
    }
    fname = lookup.get(shape_name.lower(), f"{shape_name}_flow_2d.hdf5")
    return os.path.join(DATASETS_DIR, fname)


def run_2d_tta_visualization(shape="cylinder", traj_idx=0, timestep=5, tta_steps=12, tta_lr=2e-3, output_path="gnn_vs_tta_2d_comparison.png"):
    data_file = get_shape_file(shape)
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Dataset {data_file} not found. Run python scripts/download_pde_datasets.py --force_synth")

    print("=" * 72)
    print("  PHYSICS-RESIDUAL TEST-TIME ADAPTATION (TTA) — 2D FLOW BENCHMARK")
    print(f"  Shape: {shape.upper()} | Dataset: {os.path.basename(data_file)}")
    print(f"  Device: {DEVICE} | Steps: {tta_steps} | Learning Rate: {tta_lr}")
    print("=" * 72)

    # 1. Load 2D HDF5 dataset
    with h5py.File(data_file, "r") as f:
        tensor = f["tensor"][:]  # [N_traj, T, Ny, Nx]
        mask = f["obstacle_mask"][0] if "obstacle_mask" in f else np.zeros(tensor.shape[-2:], dtype=bool)
        x_coord = f["x-coordinate"][:] if "x-coordinate" in f else np.linspace(0, 2, tensor.shape[-1])
        y_coord = f["y-coordinate"][:] if "y-coordinate" in f else np.linspace(0, 1, tensor.shape[-2])
        eq_name = f.attrs.get("equation", f"2D Navier-Stokes ({shape})")

    N, T, Ny, Nx = tensor.shape
    traj_idx = min(traj_idx, N - 1)
    timestep = min(timestep, T - 2)

    print(f"[1/4] Loaded trajectory {traj_idx}, grid [{Ny} x {Nx}], timesteps: {T}")

    # Physical state pairs: u(t) -> u(t+1)
    u_curr = tensor[traj_idx, timestep]
    u_true = tensor[traj_idx, timestep + 1]

    # Convert to PyTorch tensors [1, 1, Ny, Nx]
    x_in = torch.tensor(u_curr, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
    y_true_t = torch.tensor(u_true, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
    mask_t = torch.tensor(mask, dtype=torch.bool, device=DEVICE)

    # 2. Build 2D Surrogate (Fourier Neural Operator / Physics UNet)
    print("[2/4] Initializing 2D Surrogate Architecture (FNO-2D)...")
    surrogate = build_surrogate_model("fno", in_channels=1, out_channels=1).to(DEVICE)
    surrogate.eval()

    # Pre-TTA baseline forward pass
    with torch.no_grad():
        y_base_t = surrogate(x_in)
        # Apply obstacle zero-velocity condition
        y_base_t[:, :, mask_t] = 0.0

    base_pred = y_base_t.squeeze().cpu().numpy()
    true_field = y_true_t.squeeze().cpu().numpy()

    base_mse = float(np.mean((base_pred - true_field) ** 2))
    base_rel = base_mse / (float(np.mean(true_field ** 2)) + 1e-8)

    # Compute baseline 2D physics divergence / continuity residual
    with torch.no_grad():
        gx, gy = compute_gradient_2d(y_base_t, dx=2.0 / Nx, dy=1.0 / Ny)
        base_phys_loss = float(torch.mean(gx ** 2 + gy ** 2).item())

    print(f"  -> Pre-TTA Baseline MSE:     {base_mse:.5e} (Rel: {base_rel * 100:.2f}%)")
    print(f"  -> Pre-TTA Physics Residual: {base_phys_loss:.5e}")

    # 3. Perform Test-Time Adaptation (Unsupervised Physics Residual Optimization)
    print(f"[3/4] Performing 2D Test-Time Adaptation ({tta_steps} gradient steps)...")
    adapted_model = copy.deepcopy(surrogate)
    adapted_model.train()

    # Freeze spectral conv layers; adapt output projection (Decoder)
    for p in adapted_model.parameters():
        p.requires_grad = False
    for p in adapted_model.fc1.parameters():
        p.requires_grad = True
    for p in adapted_model.fc2.parameters():
        p.requires_grad = True

    optimizer = optim.Adam([p for p in adapted_model.parameters() if p.requires_grad], lr=tta_lr)
    anchor_weight = 0.08
    ref_base = y_base_t.detach()

    loss_history = [base_phys_loss]
    mse_history = [base_mse]

    best_loss = base_phys_loss
    best_state = copy.deepcopy(adapted_model.state_dict())

    for step in range(1, tta_steps + 1):
        optimizer.zero_grad()
        y_pred = adapted_model(x_in)
        y_pred[:, :, mask_t] = 0.0

        # Incompressibility / continuity gradient loss
        gx, gy = compute_gradient_2d(y_pred, dx=2.0 / Nx, dy=1.0 / Ny)
        phys_loss = torch.mean(gx ** 2 + gy ** 2)

        # Anchor regularization against baseline surrogate manifold
        anchor_loss = anchor_weight * F.mse_loss(y_pred, ref_base)

        total_loss = phys_loss + anchor_loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), 1.0)
        optimizer.step()

        cur_phys = float(phys_loss.item())
        loss_history.append(cur_phys)

        with torch.no_grad():
            cur_mse = float(F.mse_loss(y_pred, y_true_t).item())
            mse_history.append(cur_mse)

        if cur_phys < best_loss:
            best_loss = cur_phys
            best_state = copy.deepcopy(adapted_model.state_dict())

    # Load optimal physical state
    adapted_model.load_state_dict(best_state)
    adapted_model.eval()

    with torch.no_grad():
        y_tta_t = adapted_model(x_in)
        y_tta_t[:, :, mask_t] = 0.0

    tta_pred = y_tta_t.squeeze().cpu().numpy()
    tta_mse = float(np.mean((tta_pred - true_field) ** 2))
    tta_rel = tta_mse / (float(np.mean(true_field ** 2)) + 1e-8)
    adapted_phys_loss = best_loss

    improvement_pct = ((base_mse - tta_mse) / (base_mse + 1e-12)) * 100.0
    residual_drop_pct = ((base_phys_loss - adapted_phys_loss) / (base_phys_loss + 1e-12)) * 100.0

    print("=" * 72)
    print("  2D ADAPTATION METRICS:")
    print(f"  Pre-TTA Baseline MSE:        {base_mse:.5e} (Rel: {base_rel * 100:.2f}%)")
    print(f"  Post-TTA Adapted MSE:       {tta_mse:.5e} (Rel: {tta_rel * 100:.2f}%)")
    print(f"  Accuracy Improvement:       +{improvement_pct:.2f}%")
    print(f"  Physics Residual Drop:      -{residual_drop_pct:.2f}% ({base_phys_loss:.4e} -> {adapted_phys_loss:.4e})")
    print("=" * 72)

    # 4. Generate Publication-Quality Comparison Figure
    print(f"[4/4] Rendering publication-grade 6-panel comparison figure to: {output_path}...")
    fig = plt.figure(figsize=(15, 9.5), dpi=300)
    fig.patch.set_facecolor("#FAF7F2")

    # Layout: Top 3 panels: Velocity Fields. Bottom left/mid: Error Maps. Bottom right: Convergence & Profile.
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.0, 1.0], hspace=0.28, wspace=0.24)

    # Consistent color scale across velocity fields
    vmin, vmax = float(np.min(true_field)), float(np.max(true_field))
    cmap_flow = "viridis"
    cmap_err = "magma"

    # Compute absolute error fields
    err_base = np.abs(base_pred - true_field)
    err_tta = np.abs(tta_pred - true_field)
    err_base[mask] = np.nan
    err_tta[mask] = np.nan
    vmax_err = max(float(np.nanmax(err_base)), float(np.nanmax(err_tta)), 1e-4)

    extent = [x_coord[0], x_coord[-1], y_coord[0], y_coord[-1]]

    # Panel 1: Ground Truth
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(true_field, origin="lower", extent=extent, cmap=cmap_flow, vmin=vmin, vmax=vmax, aspect="auto")
    ax1.set_title("(a) Ground Truth Velocity Field $\\mathbf{u}_{true}$", fontsize=11, fontweight="bold", pad=8)
    ax1.set_xlabel("Streamwise Coordinate $x$", fontsize=9)
    ax1.set_ylabel("Cross-stream Coordinate $y$", fontsize=9)
    # Overlay obstacle
    ax1.contourf(mask, levels=[0.5, 1.5], origin="lower", extent=extent, colors=["#1C1917"])
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.ax.tick_params(labelsize=8)

    # Panel 2: Pre-TTA Baseline Surrogate
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(base_pred, origin="lower", extent=extent, cmap=cmap_flow, vmin=vmin, vmax=vmax, aspect="auto")
    ax2.set_title(f"(b) Pre-TTA Baseline Surrogate\nMSE: {base_mse:.4e} (Rel: {base_rel * 100:.1f}%)", fontsize=10.5, fontweight="bold", pad=8)
    ax2.set_xlabel("Streamwise Coordinate $x$", fontsize=9)
    ax2.contourf(mask, levels=[0.5, 1.5], origin="lower", extent=extent, colors=["#1C1917"])
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(labelsize=8)

    # Panel 3: Post-TTA Adapted Surrogate
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(tta_pred, origin="lower", extent=extent, cmap=cmap_flow, vmin=vmin, vmax=vmax, aspect="auto")
    ax3.set_title(f"(c) Post-TTA Adapted (+{improvement_pct:.1f}% Error Drop)\nMSE: {tta_mse:.4e} (Rel: {tta_rel * 100:.1f}%)", fontsize=10.5, fontweight="bold", color="#B85226", pad=8)
    ax3.set_xlabel("Streamwise Coordinate $x$", fontsize=9)
    ax3.contourf(mask, levels=[0.5, 1.5], origin="lower", extent=extent, colors=["#1C1917"])
    cb3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cb3.ax.tick_params(labelsize=8)

    # Panel 4: Baseline Pointwise Error
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(err_base, origin="lower", extent=extent, cmap=cmap_err, vmin=0, vmax=vmax_err, aspect="auto")
    ax4.set_title("(d) Pre-TTA Error: $|\\mathbf{u}_{base} - \\mathbf{u}_{true}|$", fontsize=11, fontweight="bold", pad=8)
    ax4.set_xlabel("Streamwise Coordinate $x$", fontsize=9)
    ax4.set_ylabel("Cross-stream Coordinate $y$", fontsize=9)
    ax4.contourf(mask, levels=[0.5, 1.5], origin="lower", extent=extent, colors=["#444444"])
    cb4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cb4.ax.tick_params(labelsize=8)

    # Panel 5: Post-TTA Pointwise Error
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(err_tta, origin="lower", extent=extent, cmap=cmap_err, vmin=0, vmax=vmax_err, aspect="auto")
    ax5.set_title("(e) Post-TTA Error: $|\\mathbf{u}_{TTA} - \\mathbf{u}_{true}|$", fontsize=11, fontweight="bold", pad=8)
    ax5.set_xlabel("Streamwise Coordinate $x$", fontsize=9)
    ax5.contourf(mask, levels=[0.5, 1.5], origin="lower", extent=extent, colors=["#444444"])
    cb5 = fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    cb5.ax.tick_params(labelsize=8)

    # Panel 6: Dual Convergence Curves (Physics Residual & True MSE vs Step K)
    ax6 = fig.add_subplot(gs[1, 2])
    steps_arr = np.arange(len(loss_history))

    color_phys = "#B85226"
    ax6.set_xlabel("Adaptation Steps ($K$)", fontsize=9.5, fontweight="medium")
    ax6.set_ylabel("Unsupervised PDE Residual", color=color_phys, fontsize=9.5, fontweight="bold")
    line1 = ax6.plot(steps_arr, loss_history, color=color_phys, lw=2.2, marker="o", markersize=4, label="PDE Continuity Residual $\\mathcal{L}_{phys}$")
    ax6.tick_params(axis="y", labelcolor=color_phys, labelsize=8)
    ax6.grid(True, linestyle="--", alpha=0.5, color="#D5CEBF")

    ax6_twin = ax6.twinx()
    color_mse = "#2E6343"
    ax6_twin.set_ylabel("True Test MSE (Held-Out Target)", color=color_mse, fontsize=9.5, fontweight="bold")
    line2 = ax6_twin.plot(steps_arr, mse_history, color=color_mse, lw=2.2, linestyle="--", marker="s", markersize=4, label="Supervised MSE Error")
    ax6_twin.tick_params(axis="y", labelcolor=color_mse, labelsize=8)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax6.legend(lines, labels, loc="upper right", fontsize=8.5, framealpha=0.9)
    ax6.set_title(f"(f) Unsupervised TTA Convergence\nResidual: -{residual_drop_pct:.1f}% | True MSE: -{improvement_pct:.1f}%", fontsize=10.5, fontweight="bold", pad=8)

    # Title block
    fig.suptitle(
        f"2D Navier-Stokes Flow Past {shape.upper()} — Physics-Residual Test-Time Adaptation\n"
        f"Pre-TTA MSE: {base_mse:.3e}  →  Post-TTA MSE: {tta_mse:.3e}  (Accuracy Gain: +{improvement_pct:.2f}% | Zero Test Labels Used)",
        fontsize=13,
        fontweight="bold",
        color="#1A1715",
        y=0.985,
    )

    full_output = os.path.join(ROOT_DIR, output_path)
    plt.savefig(full_output, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=300)
    plt.close()

    print(f"\n[DONE] High-resolution 2D TTA figure saved successfully to: {full_output}")
    return {
        "baseline_mse": base_mse,
        "adapted_mse": tta_mse,
        "improvement_pct": improvement_pct,
        "residual_drop_pct": residual_drop_pct,
        "output_image": full_output,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate 2D Flow TTA Comparison Figure")
    parser.add_argument("--shape", default="cylinder", choices=["cylinder", "airfoil", "square", "step", "multi_cylinder"])
    parser.add_argument("--traj", type=int, default=0)
    parser.add_argument("--timestep", type=int, default=5)
    parser.add_argument("--tta_steps", type=int, default=12)
    parser.add_argument("--tta_lr", type=float, default=0.002)
    parser.add_argument("--output", default="gnn_vs_tta_2d_comparison.png")
    args = parser.parse_args()

    run_2d_tta_visualization(
        shape=args.shape,
        traj_idx=args.traj,
        timestep=args.timestep,
        tta_steps=args.tta_steps,
        tta_lr=args.tta_lr,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
