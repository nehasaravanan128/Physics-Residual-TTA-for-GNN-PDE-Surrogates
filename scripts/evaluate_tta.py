"""
scripts/evaluate_tta.py
Batch benchmark comparing Baseline Surrogate vs. Physics-Residual TTA
under in-distribution and distribution-shifted PDE regimes.

Outputs comparative metrics:
- Baseline MSE and Relative MSE (No adaptation)
- Adapted MSE and Relative MSE (Physics TTA)
- Average Accuracy Gain / Error Reduction (%)
- Physics Residual Drop (%)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# Ensure local imports work
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "files (5)")))

from model import MeshGraphNet
from tta import PhysicsResidualTTA
from torch_geometric.data import Data


def evaluate_batch_tta(
    data_path: str,
    checkpoint_path: str,
    num_samples: int = 10,
    tta_steps: int = 12,
    tta_lr: float = 0.00008,
    nu_shift: float = None,
    output_json: bool = False,
):
    device = torch.device("cpu")

    model = MeshGraphNet.from_checkpoint(checkpoint_path, device=device)

    import h5py
    with h5py.File(data_path, "r") as f:
        tensor = f["tensor"]
        x_coord = f["x-coordinate"][:]
        N, T, X = tensor.shape

        dx_val = float(x_coord[1] - x_coord[0])
        base_nu = 0.001
        test_nu = nu_shift if nu_shift is not None else base_nu

        # Build 1D chain graph edges
        src = np.arange(X - 1)
        dst = src + 1
        edge_index_np = np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        dx = x_coord[dst] - x_coord[src]
        edge_attr_np = np.concatenate([dx, -dx])[:, None]
        edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32)

        tta_engine = PhysicsResidualTTA(
            model=model,
            device=device,
            dx=dx_val,
            dt=0.01,
            nu=test_nu,
            anchor_weight=0.1,
        )

        sample_results = []
        baseline_mses = []
        adapted_mses = []
        baseline_rel_mses = []
        adapted_rel_mses = []
        phys_before_list = []
        phys_after_list = []

        # Pick spaced test trajectories
        step_stride = max(1, N // num_samples)
        traj_indices = [i * step_stride for i in range(num_samples)]

        for idx, traj in enumerate(traj_indices):
            t_step = 10 + (idx * 5) % (T - 20)
            u_curr = tensor[traj, t_step]
            u_next = tensor[traj, t_step + 1]

            x_in = torch.tensor(u_curr, dtype=torch.float32).unsqueeze(-1)
            x_target = torch.tensor(u_next, dtype=torch.float32).unsqueeze(-1)
            sample = Data(x=x_in, edge_index=edge_index, edge_attr=edge_attr, y=x_target)

            res = tta_engine.adapt_sample(
                sample=sample,
                tta_steps=tta_steps,
                tta_lr=tta_lr,
                nu_override=test_nu,
            )

            baseline_mses.append(res["baseline_mse"])
            adapted_mses.append(res["adapted_mse"])
            baseline_rel_mses.append(res["baseline_relative_mse"])
            adapted_rel_mses.append(res["adapted_relative_mse"])
            phys_before_list.append(res["baseline_physics_loss"])
            phys_after_list.append(res["adapted_physics_loss"])

            sample_results.append({
                "traj": traj,
                "timestep": t_step,
                "baseline_mse": res["baseline_mse"],
                "adapted_mse": res["adapted_mse"],
                "baseline_rel_mse": res["baseline_relative_mse"],
                "adapted_rel_mse": res["adapted_relative_mse"],
                "improvement_pct": res["improvement_pct"],
                "phys_before": res["baseline_physics_loss"],
                "phys_after": res["adapted_physics_loss"],
            })

    avg_base_mse = float(np.mean(baseline_mses))
    avg_adapt_mse = float(np.mean(adapted_mses))
    avg_base_rel = float(np.mean(baseline_rel_mses))
    avg_adapt_rel = float(np.mean(adapted_rel_mses))
    avg_gain = float(((avg_base_mse - avg_adapt_mse) / avg_base_mse) * 100.0)
    avg_phys_drop = float(((np.mean(phys_before_list) - np.mean(phys_after_list)) / np.mean(phys_before_list)) * 100.0)

    summary = {
        "num_test_samples": num_samples,
        "tta_steps": tta_steps,
        "tta_lr": tta_lr,
        "nu": test_nu,
        "distribution_shift": "Parameter shift nu=" + str(test_nu) if nu_shift is not None else "In-distribution",
        "avg_baseline_mse": avg_base_mse,
        "avg_adapted_mse": avg_adapt_mse,
        "avg_baseline_relative_mse": avg_base_rel,
        "avg_adapted_relative_mse": avg_adapt_rel,
        "avg_accuracy_gain_pct": avg_gain,
        "avg_physics_residual_drop_pct": avg_phys_drop,
        "samples": sample_results,
    }

    if output_json:
        print(json.dumps(summary))
    else:
        print("\n" + "=" * 70)
        print("  PHYSICS-RESIDUAL TEST-TIME ADAPTATION (TTA) BENCHMARK REPORT")
        print("=" * 70)
        print(f"Evaluated Test Cases:       {num_samples} trajectories")
        print(f"Condition:                  {summary['distribution_shift']}")
        print(f"TTA Optimization Budget:    {tta_steps} gradient steps @ lr={tta_lr}")
        print("-" * 70)
        print(f"Mean Baseline MSE:          {avg_base_mse:.6e}")
        print(f"Mean Adapted TTA MSE:       {avg_adapt_mse:.6e}")
        print(f"Mean Baseline Relative MSE: {avg_base_rel:.6e}")
        print(f"Mean Adapted Relative MSE:  {avg_adapt_rel:.6e}")
        print("-" * 70)
        print(f"AVERAGE ACCURACY GAIN:      {avg_gain:+.2f}% error reduction")
        print(f"PHYSICS RESIDUAL DROP:      {avg_phys_drop:+.2f}% reduction")
        print("=" * 70)
        print(f"{'Traj':>5} | {'t':>3} | {'Baseline MSE':>13} | {'Adapted MSE':>13} | {'Gain (%)':>9}")
        print("-" * 70)
        for s in sample_results:
            print(f"{s['traj']:5d} | {s['timestep']:3d} | {s['baseline_mse']:13.6e} | {s['adapted_mse']:13.6e} | {s['improvement_pct']:+8.2f}%")
        print("=" * 70 + "\n")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Burgers HDF5 dataset")
    parser.add_argument("--checkpoint_path", required=True, help="Trained surrogate weights")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of test trajectories")
    parser.add_argument("--tta_steps", type=int, default=12, help="Adaptation steps per test sample")
    parser.add_argument("--tta_lr", type=float, default=0.00008, help="Adaptation learning rate")
    parser.add_argument("--nu_shift", type=float, default=None, help="Shifted viscosity (e.g. 0.002)")
    parser.add_argument("--output_json", action="store_true", help="Print JSON")

    args = parser.parse_args()

    evaluate_batch_tta(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        num_samples=args.num_samples,
        tta_steps=args.tta_steps,
        tta_lr=args.tta_lr,
        nu_shift=args.nu_shift,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
