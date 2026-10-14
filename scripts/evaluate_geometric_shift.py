"""
scripts/evaluate_geometric_shift.py
Comprehensive Geometric Distribution Shift Benchmark for MeshGraphNets.

Evaluates Baseline GNN Surrogate vs. Physics-Residual TTA under 3 canonical
geometric mesh distribution shifts:
1. 'jitter': Irregular, non-uniform spatial grid with stochastic coordinate perturbations.
2. 'warped': Non-uniform boundary-layer coordinate clustering (dense in gradients, coarse elsewhere).
3. 'coarse': Resolution scale shift (N=512 subsampling, doubling edge lengths Delta x).

Usage:
    python scripts/evaluate_geometric_shift.py \
        --data_path "files (5)/1D_Burgers_Sols_Nu0.001.hdf5" \
        --checkpoint_path real_burgers_model.pt \
        --num_samples 10 \
        --shift_type all
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from torch_geometric.data import Data

# Ensure local imports
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "files (5)")))

from model import MeshGraphNet
from tta import PhysicsResidualTTA


def generate_shifted_mesh(x_uniform: np.ndarray, shift_type: str = "jitter", seed: int = 42):
    """
    Generates an out-of-distribution geometric mesh coordinates array.
    """
    np.random.seed(seed)
    N = len(x_uniform)
    L = 1.0

    if shift_type == "jitter":
        # Stochastic perturbation: jitter each internal node by up to ±35% of dx
        dx = x_uniform[1] - x_uniform[0]
        noise = np.random.uniform(-0.35 * dx, 0.35 * dx, size=N)
        noise[0] = 0.0  # anchor periodic origin
        x_shifted = np.sort(x_uniform + noise)
        # Ensure strict monotonicity and bounds
        x_shifted = np.clip(x_shifted, 0.0, L)
        x_shifted[0] = 0.0
        return x_shifted, "Stochastic Irregular Mesh (Jitter ±35%)"

    elif shift_type == "warped":
        # Smooth boundary layer warping: non-uniform node concentration
        # x -> x + alpha * sin(2*pi*x) creates high node density in one zone and coarse in another
        alpha = 0.085
        x_shifted = x_uniform + alpha * np.sin(2.0 * np.pi * x_uniform)
        x_shifted = np.sort(x_shifted)
        x_shifted = (x_shifted - x_shifted[0]) / (x_shifted[-1] - x_shifted[0]) * L
        return x_shifted, "Non-Uniform Warped Refinement (Boundary Layer)"

    elif shift_type == "coarse":
        # Resolution scale shift: subsample to N=512 (2x larger edge spacing)
        x_shifted = x_uniform[::2]
        return x_shifted, "Resolution Scale Shift (N=512, 2x Edge Lengths)"

    else:
        raise ValueError(f"Unknown shift_type: {shift_type}")


def build_mesh_graph(x_coords: np.ndarray, u_curr_interp: np.ndarray, u_next_interp: np.ndarray):
    """
    Constructs a PyG Data object for an arbitrary (uniform or irregular) 1D mesh.
    """
    N = len(x_coords)
    src = np.arange(N - 1)
    dst = src + 1
    edge_index_np = np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    # Edge attributes: signed distance between adjacent mesh nodes
    dx = x_coords[dst] - x_coords[src]
    edge_attr_np = np.concatenate([dx, -dx])[:, None]
    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32)

    x_in = torch.tensor(u_curr_interp, dtype=torch.float32).unsqueeze(-1)
    x_target = torch.tensor(u_next_interp, dtype=torch.float32).unsqueeze(-1)

    return Data(
        x=x_in,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=x_target,
        x_coord=torch.tensor(x_coords, dtype=torch.float32),
    )


def run_geometric_benchmark(
    data_path: str,
    checkpoint_path: str,
    num_samples: int = 10,
    tta_steps: int = 12,
    tta_lr: float = 0.0001,
    shifts_to_test: list = None,
    output_json_path: str = None,
):
    if shifts_to_test is None:
        shifts_to_test = ["jitter", "warped", "coarse"]

    device = torch.device("cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MeshGraphNet(
        node_in_dim=ckpt["node_in_dim"],
        edge_in_dim=ckpt["edge_in_dim"],
        out_dim=ckpt["out_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    import h5py
    with h5py.File(data_path, "r") as f:
        tensor = f["tensor"]
        x_orig = f["x-coordinate"][:]
        N_trajs, T_steps, X_points = tensor.shape

        step_stride = max(1, N_trajs // num_samples)
        traj_indices = [i * step_stride for i in range(num_samples)]

        overall_results = {}

        for shift_type in shifts_to_test:
            x_shifted, shift_name = generate_shifted_mesh(x_orig, shift_type=shift_type)
            
            tta_engine = PhysicsResidualTTA(
                model=model,
                device=device,
                dx=float(np.mean(np.diff(x_shifted))),
                dt=0.01,
                nu=0.001,
                anchor_weight=0.1,
            )

            base_mses = []
            adapt_mses = []
            base_rel_mses = []
            adapt_rel_mses = []
            res_drops = []
            sample_details = []

            for idx, traj in enumerate(traj_indices):
                t_step = 10 + (idx * 4) % (T_steps - 20)
                u_curr_raw = tensor[traj, t_step]
                u_next_raw = tensor[traj, t_step + 1]

                # Extend periodic boundary endpoint so x=1.0 matches x=0.0 for exact periodic spline
                x_ext = np.append(x_orig, 1.0)
                u_curr_ext = np.append(u_curr_raw, u_curr_raw[0])
                u_next_ext = np.append(u_next_raw, u_next_raw[0])

                cs_curr = CubicSpline(x_ext, u_curr_ext, bc_type="periodic")
                cs_next = CubicSpline(x_ext, u_next_ext, bc_type="periodic")

                u_curr_shifted = cs_curr(x_shifted)
                u_next_shifted = cs_next(x_shifted)

                graph_sample = build_mesh_graph(x_shifted, u_curr_shifted, u_next_shifted)

                res = tta_engine.adapt_sample(
                    sample=graph_sample,
                    tta_steps=tta_steps,
                    tta_lr=tta_lr,
                    x_coord_override=torch.tensor(x_shifted, dtype=torch.float32),
                )

                base_mses.append(res["baseline_mse"])
                adapt_mses.append(res["adapted_mse"])
                base_rel_mses.append(res["baseline_relative_mse"])
                adapt_rel_mses.append(res["adapted_relative_mse"])

                p_before = res["baseline_physics_loss"]
                p_after = res["adapted_physics_loss"]
                p_drop = ((p_before - p_after) / max(p_before, 1e-8)) * 100.0
                res_drops.append(p_drop)

                sample_details.append({
                    "trajectory": traj,
                    "timestep": t_step,
                    "baseline_mse": res["baseline_mse"],
                    "adapted_mse": res["adapted_mse"],
                    "improvement_pct": res["improvement_pct"],
                    "residual_drop_pct": p_drop,
                })

            mean_base_mse = float(np.mean(base_mses))
            mean_adapt_mse = float(np.mean(adapt_mses))
            mean_base_rel = float(np.mean(base_rel_mses))
            mean_adapt_rel = float(np.mean(adapt_rel_mses))
            mean_gain = ((mean_base_mse - mean_adapt_mse) / mean_base_mse) * 100.0

            overall_results[shift_type] = {
                "name": shift_name,
                "num_nodes": len(x_shifted),
                "mean_baseline_mse": mean_base_mse,
                "mean_adapted_mse": mean_adapt_mse,
                "mean_baseline_rel_mse": mean_base_rel,
                "mean_adapted_rel_mse": mean_adapt_rel,
                "error_reduction_pct": mean_gain,
                "mean_residual_drop_pct": float(np.mean(res_drops)),
                "samples": sample_details,
            }

    # Print Clean Console Summary Table
    print("\n" + "=" * 86)
    print("  GEOMETRIC DISTRIBUTION SHIFT BENCHMARK: MESHGRAPHNET vs. PHYSICS-RESIDUAL TTA")
    print("=" * 86)
    print(f"{'Geometric Regime':<42} | {'Base MSE':<12} | {'Adapted MSE':<12} | {'Gain (%)':<10}")
    print("-" * 86)

    for st, data in overall_results.items():
        name = data["name"]
        b_mse = f"{data['mean_baseline_mse']:.3e}"
        a_mse = f"{data['mean_adapted_mse']:.3e}"
        gain = f"+{data['error_reduction_pct']:.2f}%" if data['error_reduction_pct'] > 0 else f"{data['error_reduction_pct']:.2f}%"
        print(f"{name:<42} | {b_mse:<12} | {a_mse:<12} | {gain:<10}")

    print("=" * 86)

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as jf:
            json.dump(overall_results, jf, indent=2)
        print(f"\n[OK] Geometric benchmark metrics saved to: {output_json_path}")

    return overall_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Physics-Residual TTA under geometric distribution shifts.")
    parser.add_argument("--data_path", default="files (5)/1D_Burgers_Sols_Nu0.001.hdf5")
    parser.add_argument("--checkpoint_path", default="real_burgers_model.pt")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--tta_steps", type=int, default=12)
    parser.add_argument("--tta_lr", type=float, default=0.0001)
    parser.add_argument("--shift_type", choices=["jitter", "warped", "coarse", "all"], default="all")
    parser.add_argument("--output_json", default="scripts/geometric_benchmark_results.json")
    args = parser.parse_args()

    shifts = ["jitter", "warped", "coarse"] if args.shift_type == "all" else [args.shift_type]

    run_geometric_benchmark(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        num_samples=args.num_samples,
        tta_steps=args.tta_steps,
        tta_lr=args.tta_lr,
        shifts_to_test=shifts,
        output_json_path=args.output_json,
    )


if __name__ == "__main__":
    main()
