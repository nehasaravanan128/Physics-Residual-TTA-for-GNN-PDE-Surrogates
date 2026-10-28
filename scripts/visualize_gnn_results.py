"""
scripts/visualize_gnn_results.py
Processes ONLY the GNN (MeshGraphNet) surrogate on the 1D Burgers' PDE test dataset,
computes evaluation metrics, and generates a publication-quality results figure.
"""

import os
import sys
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
from data import load_burgers


def load_gnn_model(checkpoint_path, device):
    print(f"[1/4] Loading GNN (MeshGraphNet) checkpoint from: {checkpoint_path}")
    model = MeshGraphNet.from_checkpoint(checkpoint_path, device=device)
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"      Model loaded successfully ({num_params:,} parameters).")
    return model


def evaluate_gnn(model, data_path, device):
    print(f"[2/4] Processing GNN evaluation on dataset: {data_path}")
    with h5py.File(data_path, "r") as f:
        tensor = f["tensor"][:]        # [N, T, X]
        x_coord = f["x-coordinate"][:] # [X]

    N, T, X = tensor.shape
    print(f"      Data dimensions: {N} trajectories, {T} timesteps, {X} spatial grid nodes.")

    # Construct 1D bidirectional chain mesh graph
    src = np.arange(X - 1)
    dst = src + 1
    edge_index_np = np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)

    dx = x_coord[dst] - x_coord[src]
    edge_attr_np = np.concatenate([dx, -dx])[:, None]
    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32, device=device)

    all_preds = []
    all_targets = []
    step_errors = np.zeros(T - 1)
    counts = np.zeros(T - 1)

    # Autoregressive rollout storage for trajectory 0
    rollout_pred = np.zeros((T, X))
    rollout_pred[0] = tensor[0, 0]

    with torch.no_grad():
        # Compute 1-step evaluation across all trajectories
        for n in range(N):
            for t in range(T - 1):
                x_in = torch.tensor(tensor[n, t], dtype=torch.float32, device=device).unsqueeze(-1)
                y_true = tensor[n, t + 1]

                pred = model(x_in, edge_index, edge_attr).squeeze(-1).cpu().numpy()
                all_preds.append(pred)
                all_targets.append(y_true)

                mse_t = np.mean((pred - y_true) ** 2)
                step_errors[t] += mse_t
                counts[t] += 1

        # Compute autoregressive multi-step rollout for trajectory 0
        curr_state = torch.tensor(tensor[0, 0], dtype=torch.float32, device=device).unsqueeze(-1)
        for t in range(1, T):
            next_state = model(curr_state, edge_index, edge_attr)
            rollout_pred[t] = next_state.squeeze(-1).cpu().numpy()
            curr_state = next_state

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    overall_mse = float(np.mean((all_preds - all_targets) ** 2))
    scale = float(np.mean(all_targets ** 2) + 1e-8)
    relative_mse = overall_mse / scale
    max_err = float(np.max(np.abs(all_preds - all_targets)))
    mae = float(np.mean(np.abs(all_preds - all_targets)))
    step_errors /= counts

    print(f"      Total Node Predictions: {all_preds.size:,}")
    print(f"      Overall GNN MSE:        {overall_mse:.6e}")
    print(f"      Relative MSE:           {relative_mse:.6e} ({relative_mse * 100:.2f}%)")
    print(f"      Mean Absolute Error:    {mae:.6e}")
    print(f"      Max Absolute Error:     {max_err:.6e}")

    return {
        "x_coord": x_coord,
        "tensor": tensor,
        "rollout_pred": rollout_pred,
        "step_errors": step_errors,
        "overall_mse": overall_mse,
        "relative_mse": relative_mse,
        "mae": mae,
        "max_err": max_err,
        "total_preds": all_preds.size,
    }


def generate_plot(results, output_paths):
    print("[3/4] Generating publication-grade GNN results figure...")

    x = results["x_coord"]
    tensor = results["tensor"]
    rollout_pred = results["rollout_pred"]
    step_errors = results["step_errors"]
    traj_idx = 0
    t_snapshot = 15  # Mid-trajectory shock formation timestep

    y_true_snapshot = tensor[traj_idx, t_snapshot]
    y_pred_snapshot = rollout_pred[t_snapshot]
    abs_err_snapshot = np.abs(y_true_snapshot - y_pred_snapshot)

    # Style configuration
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#334155"
    plt.rcParams["axes.linewidth"] = 1.0

    fig = plt.figure(figsize=(16, 10.5), facecolor="#f8fafc")
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.0], hspace=0.38, wspace=0.22,
                           top=0.84, bottom=0.08, left=0.08, right=0.95)

    # Title header banner with proper vertical padding
    fig.text(0.08, 0.94, "MeshGraphNet GNN Surrogate: PDE Inference & Evaluation Results",
             fontsize=19, fontweight="bold", color="#0f172a")
    fig.text(0.08, 0.89,
             f"Equation: 1D Viscous Burgers' PDE | Tested on {results['total_preds']:,} node predictions | "
             f"Global 1-Step MSE: {results['overall_mse']:.4e} | Relative L2: {results['relative_mse'] * 100:.2f}% | MAE: {results['mae']:.4e}",
             fontsize=11.5, color="#475569")

    # -------------------------------------------------------------
    # Panel A: Spatial Wavefront Snapshot (Ground Truth vs GNN)
    # -------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor("#ffffff")
    ax_a.plot(x, tensor[traj_idx, 0], color="#94a3b8", linestyle=":", linewidth=1.8, label="Initial State $u(x, t=0)$")
    ax_a.plot(x, y_true_snapshot, color="#1e40af", linewidth=2.5, label=f"Ground Truth $u(x, t={t_snapshot})$")
    ax_a.plot(x, y_pred_snapshot, color="#dc2626", linestyle="--", linewidth=2.2, label=f"GNN Prediction $\\hat{{u}}(x, t={t_snapshot})$")
    ax_a.fill_between(x, y_true_snapshot, y_pred_snapshot, color="#f87171", alpha=0.25, label="Discrepancy Area")

    ax_a.set_title(f"(A) Wave Field Profile at Timestep t={t_snapshot} (Shock Formation)", fontsize=12, fontweight="bold", color="#1e293b", pad=10)
    ax_a.set_xlabel("Spatial Coordinate x ∈ [0, 1]", fontsize=10, fontweight="semibold")
    ax_a.set_ylabel("Velocity Field u(x, t)", fontsize=10, fontweight="semibold")
    ax_a.legend(loc="upper right", frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=9)
    ax_a.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")

    # -------------------------------------------------------------
    # Panel B: Pointwise Absolute Error Along Mesh
    # -------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor("#ffffff")
    ax_b.fill_between(x, abs_err_snapshot, color="#3b82f6", alpha=0.3)
    ax_b.plot(x, abs_err_snapshot, color="#1d4ed8", linewidth=2.0, label="|Ground Truth - GNN|")
    ax_b.axhline(np.mean(abs_err_snapshot), color="#ef4444", linestyle="--", linewidth=1.5,
                 label=f"Mean Error: {np.mean(abs_err_snapshot):.4e}")

    ax_b.set_title(f"(B) Spatial Error Distribution |u_true - u_gnn| (t={t_snapshot})", fontsize=12, fontweight="bold", color="#1e293b", pad=10)
    ax_b.set_xlabel("Spatial Coordinate x ∈ [0, 1]", fontsize=10, fontweight="semibold")
    ax_b.set_ylabel("Absolute Error", fontsize=10, fontweight="semibold")
    ax_b.legend(loc="upper right", frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=9)
    ax_b.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    ax_b.ticklabel_format(style="scientific", axis="y", scilimits=(0,0))

    # -------------------------------------------------------------
    # Panel C: Autoregressive Multi-Step Rollout Error Evolution
    # -------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor("#ffffff")
    timesteps = np.arange(1, len(step_errors) + 1)
    ax_c.plot(timesteps, step_errors, marker="o", markersize=4, color="#7c3aed", linewidth=2.0, label="1-Step GNN Test MSE")
    ax_c.axhline(results["overall_mse"], color="#059669", linestyle="--", linewidth=1.5,
                 label=f"Mean Benchmark MSE ({results['overall_mse']:.3e})")

    ax_c.set_title("(C) Temporal Error Across Prediction Horizon", fontsize=12, fontweight="bold", color="#1e293b", pad=10)
    ax_c.set_xlabel("Timestep (t → t+1)", fontsize=10, fontweight="semibold")
    ax_c.set_ylabel("Mean Squared Error (MSE)", fontsize=10, fontweight="semibold")
    ax_c.legend(loc="upper left", frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=9)
    ax_c.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    ax_c.ticklabel_format(style="scientific", axis="y", scilimits=(0,0))

    # -------------------------------------------------------------
    # Panel D: Spatiotemporal Solution Comparison (Heatmaps)
    # -------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor("#ffffff")

    true_traj = tensor[traj_idx]  # [T, X]
    pred_traj = rollout_pred       # [T, X]
    diff_traj = np.abs(true_traj - pred_traj)

    im = ax_d.imshow(diff_traj, aspect="auto", cmap="magma",
                     extent=[x[0], x[-1], len(diff_traj)-1, 0], origin="upper")
    cbar = fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)
    cbar.set_label("Absolute Error |u_true - u_gnn|", fontsize=9.5)

    ax_d.set_title("(D) Space-Time Error Heatmap (x vs. Timestep)", fontsize=12, fontweight="bold", color="#1e293b", pad=10)
    ax_d.set_xlabel("Spatial Coordinate x", fontsize=10, fontweight="semibold")
    ax_d.set_ylabel("Rollout Timestep t", fontsize=10, fontweight="semibold")

    # Save to all target paths
    for path in output_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[4/4] Saved high-resolution results picture to: {path}")

    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running GNN Processing on: {device}")

    ckpt_path = os.path.join(ROOT_DIR, "real_burgers_model.pt")
    data_path = os.path.join(ROOT_DIR, "datasets", "burgers_1d.hdf5")

    # Verify files exist
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    model = load_gnn_model(ckpt_path, device)
    results = evaluate_gnn(model, data_path, device)

    # Output picture paths: root dir and artifact directory
    artifact_dir = r"C:\Users\nehas\.gemini\antigravity-ide\brain\900abb48-f63b-4e08-aae0-51faf7646962"
    output_pic_local = os.path.join(ROOT_DIR, "gnn_results.png")
    output_pic_artifact = os.path.join(artifact_dir, "gnn_results.png")

    generate_plot(results, [output_pic_local, output_pic_artifact])

    print("\n" + "=" * 60)
    print("GNN INFERENCE & EVALUATION COMPLETE!")
    print(f"Results Image: {output_pic_local}")
    print("=" * 60)


if __name__ == "__main__":
    main()
