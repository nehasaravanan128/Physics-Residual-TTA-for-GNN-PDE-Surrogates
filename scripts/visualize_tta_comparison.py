"""
scripts/visualize_tta_comparison.py
Runs Baseline GNN vs. Physics-Residual Test-Time Adapted (TTA) GNN on 1D Burgers' PDE,
computes comparative metrics, and produces a publication-quality comparison figure.
"""

import os
import sys
import copy
import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Ensure imports work from project root and files (5)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "files (5)"))
sys.path.insert(0, os.path.join(ROOT_DIR, "scripts"))

from model import MeshGraphNet
from tta import PhysicsResidualTTA
from torch_geometric.data import Data


def run_gnn_vs_tta_benchmark():
    device = torch.device("cpu")
    ckpt_path = os.path.join(ROOT_DIR, "real_burgers_model.pt")
    data_path = os.path.join(ROOT_DIR, "datasets", "burgers_1d.hdf5")

    print(f"[1/4] Loading trained GNN checkpoint: {ckpt_path}")
    model = MeshGraphNet.from_checkpoint(ckpt_path, device=device)
    model.eval()

    print(f"[2/4] Loading test dataset: {data_path}")
    with h5py.File(data_path, "r") as f:
        tensor = f["tensor"][:]        # [N, T, X]
        x_coord = f["x-coordinate"][:] # [X]

    N, T, X = tensor.shape
    dx_val = float(x_coord[1] - x_coord[0])
    nu = 0.001
    dt = 0.01

    # 1D bidirectional chain graph
    src = np.arange(X - 1)
    dst = src + 1
    edge_index_np = np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)

    dx = x_coord[dst] - x_coord[src]
    edge_attr_np = np.concatenate([dx, -dx])[:, None]
    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32, device=device)

    tta_engine = PhysicsResidualTTA(
        model=model,
        device=device,
        dx=dx_val,
        dt=dt,
        nu=nu,
        anchor_weight=0.08,
    )

    # Pick representative trajectory and shock timestep
    traj_idx = 0
    t_snapshot = 15

    u_curr = tensor[traj_idx, t_snapshot]
    u_next = tensor[traj_idx, t_snapshot + 1]

    x_in = torch.tensor(u_curr, dtype=torch.float32, device=device).unsqueeze(-1)
    x_target = torch.tensor(u_next, dtype=torch.float32, device=device).unsqueeze(-1)
    sample = Data(x=x_in, edge_index=edge_index, edge_attr=edge_attr, y=x_target)
    sample.x_coord = x_coord

    print("[3/4] Performing Test-Time Adaptation (Decoder-Only, Unsupervised Physics Residual)...")
    res = tta_engine.adapt_sample(
        sample=sample,
        tta_steps=12,
        tta_lr=1e-4,
        x_coord_override=torch.tensor(x_coord, dtype=torch.float32, device=device),
    )

    base_pred = np.array(res["baseline_predicted"])
    tta_pred = np.array(res["adapted_predicted"])
    true_y = np.array(res["true"])
    loss_history = res["physics_loss_history"]

    base_mse = res["baseline_mse"]
    tta_mse = res["adapted_mse"]
    base_rel = base_mse / (np.mean(true_y ** 2) + 1e-8)
    tta_rel = tta_mse / (np.mean(true_y ** 2) + 1e-8)
    improvement = ((base_mse - tta_mse) / base_mse) * 100.0
    phys_drop = ((res["baseline_physics_loss"] - res["adapted_physics_loss"]) / res["baseline_physics_loss"]) * 100.0

    print("=" * 60)
    print("COMPARISON RESULTS:")
    print(f"  Baseline GNN MSE:          {base_mse:.6e} (Rel: {base_rel*100:.2f}%)")
    print(f"  GNN + Physics TTA MSE:     {tta_mse:.6e} (Rel: {tta_rel*100:.2f}%)")
    print(f"  Error Reduction (MSE):     +{improvement:.2f}%")
    print(f"  Physics Residual Drop:     -{phys_drop:.2f}%")
    print("=" * 60)

    # -------------------------------------------------------------
    # Plot Generation
    # -------------------------------------------------------------
    print("[4/4] Generating high-resolution comparison figure...")

    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#334155"
    plt.rcParams["axes.linewidth"] = 1.0

    fig = plt.figure(figsize=(16, 11), facecolor="#f8fafc")
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.05, 1.0], hspace=0.38, wspace=0.25,
                           top=0.84, bottom=0.08, left=0.08, right=0.95)

    # Main header
    fig.text(0.08, 0.94, "Baseline GNN vs. Physics-Residual TTA: Comparative Performance",
             fontsize=19, fontweight="bold", color="#0f172a")
    fig.text(0.08, 0.89,
             f"MeshGraphNet Surrogate on Viscous Burgers' PDE | Unsupervised Adaptation (K=12 steps) | "
             f"MSE: {base_mse:.3e} → {tta_mse:.3e} (+{improvement:.1f}% accuracy gain) | Residual: -{phys_drop:.1f}%",
             fontsize=11.5, color="#475569")

    # -------------------------------------------------------------
    # Panel A: Spatial Solution Profile Comparison
    # -------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor("#ffffff")
    ax_a.plot(x_coord, tensor[traj_idx, 0], color="#94a3b8", linestyle=":", linewidth=1.6, label="Initial State $u(x, t=0)$")
    ax_a.plot(x_coord, true_y, color="#0f172a", linewidth=2.8, label=f"Ground Truth ($t={t_snapshot+1}$)")
    ax_a.plot(x_coord, base_pred, color="#dc2626", linestyle="--", linewidth=2.0,
              label=f"Baseline GNN (MSE: {base_mse:.2e})")
    ax_a.plot(x_coord, tta_pred, color="#059669", linestyle="-", linewidth=2.4,
              label=f"GNN + Physics TTA (MSE: {tta_mse:.2e})")

    ax_a.set_title(f"(A) Velocity Field Wavefront at Shock Regime (t={t_snapshot+1})",
                   fontsize=12, fontweight="bold", color="#1e293b", pad=10)
    ax_a.set_xlabel("Spatial Coordinate x ∈ [0, 1]", fontsize=10, fontweight="semibold")
    ax_a.set_ylabel("Velocity Field u(x, t)", fontsize=10, fontweight="semibold")
    ax_a.legend(loc="upper right", frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=9)
    ax_a.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")

    # -------------------------------------------------------------
    # Panel B: Pointwise Absolute Error Comparison
    # -------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor("#ffffff")
    err_base = np.abs(true_y - base_pred)
    err_tta = np.abs(true_y - tta_pred)

    ax_b.plot(x_coord, err_base, color="#dc2626", linewidth=1.8, label=f"Baseline GNN Error (Mean: {np.mean(err_base):.3e})")
    ax_b.fill_between(x_coord, err_base, color="#f87171", alpha=0.25)

    ax_b.plot(x_coord, err_tta, color="#059669", linewidth=2.0, label=f"GNN + TTA Error (Mean: {np.mean(err_tta):.3e})")
    ax_b.fill_between(x_coord, err_tta, color="#34d399", alpha=0.35)

    ax_b.set_title("(B) Pointwise Error Profile |u_true - u_pred| Across Mesh",
                   fontsize=12, fontweight="bold", color="#1e293b", pad=10)
    ax_b.set_xlabel("Spatial Coordinate x ∈ [0, 1]", fontsize=10, fontweight="semibold")
    ax_b.set_ylabel("Absolute Error", fontsize=10, fontweight="semibold")
    ax_b.legend(loc="upper right", frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=9)
    ax_b.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    ax_b.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    # -------------------------------------------------------------
    # Panel C: Unsupervised Adaptation Trajectory
    # -------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor("#ffffff")
    steps = np.arange(len(loss_history))
    ax_c.plot(steps, loss_history, marker="o", markersize=5, color="#2563eb", linewidth=2.2,
              label="PDE Physics Residual $\\mathcal{L}_{physics}$ (Unsupervised)")

    ax_c.set_title("(C) Test-Time Physics Residual Convergence (Decoder Steps K)",
                   fontsize=12, fontweight="bold", color="#1e293b", pad=10)
    ax_c.set_xlabel("Adaptation Gradient Step (k)", fontsize=10, fontweight="semibold")
    ax_c.set_ylabel("Discrete PDE Residual Loss", fontsize=10, fontweight="semibold", color="#1e40af")
    ax_c.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    ax_c.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    # Add text annotation on convergence
    ax_c.annotate(f"Start: {loss_history[0]:.2e}\nEnd: {loss_history[-1]:.2e}\nDrop: -{phys_drop:.1f}%",
                  xy=(steps[-1], loss_history[-1]), xytext=(steps[-1] - 3.5, (loss_history[0] + loss_history[-1]) / 2),
                  arrowprops=dict(facecolor="#2563eb", shrink=0.08, width=1.5, headwidth=6),
                  fontsize=9, fontweight="semibold", color="#1e40af",
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="#eff6ff", edgecolor="#bfdbfe"))

    # -------------------------------------------------------------
    # Panel D: Metric Comparison Cards & Bar Chart
    # -------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor("#ffffff")

    categories = ["Mean Squared\nError (MSE ×10³)", "Relative L2\nError (%)", "Max Pointwise\nError", "Physics Residual\nLoss"]
    base_vals = [base_mse * 1e3, base_rel * 100, float(np.max(err_base)), res["baseline_physics_loss"]]
    tta_vals = [tta_mse * 1e3, tta_rel * 100, float(np.max(err_tta)), res["adapted_physics_loss"]]

    x_indices = np.arange(len(categories))
    bar_width = 0.35

    bars1 = ax_d.bar(x_indices - bar_width/2, base_vals, bar_width, label="Baseline GNN", color="#ef4444", alpha=0.85)
    bars2 = ax_d.bar(x_indices + bar_width/2, tta_vals, bar_width, label="GNN + Physics TTA", color="#10b981", alpha=0.85)

    # Attach percentage reduction badges on top of bars
    reductions = [
        f"-{improvement:.1f}%",
        f"-{((base_rel - tta_rel) / base_rel) * 100:.1f}%",
        f"-{((np.max(err_base) - np.max(err_tta)) / np.max(err_base)) * 100:.1f}%",
        f"-{phys_drop:.1f}%"
    ]
    for i, (b1, b2, red) in enumerate(zip(bars1, bars2, reductions)):
        y_max = max(b1.get_height(), b2.get_height())
        ax_d.text(x_indices[i], y_max * 1.05, red, ha="center", va="bottom",
                  fontsize=9.5, fontweight="bold", color="#047857",
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="#ecfdf5", edgecolor="#a7f3d0"))

    ax_d.set_title("(D) Quantitative Improvement Metrics Summary",
                   fontsize=12, fontweight="bold", color="#1e293b", pad=10)
    ax_d.set_xticks(x_indices)
    ax_d.set_xticklabels(categories, fontsize=9.5, fontweight="semibold")
    ax_d.legend(loc="upper right", frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=9)
    ax_d.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1", axis="y")
    ax_d.set_ylim(0, max([max(base_vals), max(tta_vals)]) * 1.35)

    # Save outputs
    output_pic_local = os.path.join(ROOT_DIR, "gnn_vs_tta_comparison.png")
    artifact_dir = r"C:\Users\nehas\.gemini\antigravity-ide\brain\900abb48-f63b-4e08-aae0-51faf7646962"
    output_pic_artifact = os.path.join(artifact_dir, "gnn_vs_tta_comparison.png")

    for path in [output_pic_local, output_pic_artifact]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison picture to: {path}")

    plt.close(fig)


if __name__ == "__main__":
    run_gnn_vs_tta_benchmark()
