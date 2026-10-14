"""
scripts/evaluate_ablations.py
Comprehensive Ablation Studies for Physics-Residual Test-Time Adaptation (TTA).

Studies:
1. Adaptation Step Budget K in {1, 3, 5, 8, 12, 16, 20}
2. Anchor Loss Weight lambda_anchor in {0.0, 0.005, 0.02, 0.1, 0.5, 1.0}
3. Adaptation Learning Rate eta in {1e-5, 4e-5, 8e-5, 2e-4, 5e-4}
4. Adaptation Scope: Decoder Only vs Full GNN (Encoder + Processor + Decoder)

Saves summary results to scripts/ablation_benchmark_results.json
"""

import argparse
import copy
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
from torch_geometric.data import Data

# Ensure local imports work
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "files (5)")))

from model import MeshGraphNet
from physics_loss import BurgersPhysicsLoss
from tta import PhysicsResidualTTA


def run_ablation_studies(
    data_path: str,
    checkpoint_path: str,
    num_samples: int = 6,
    output_json_path: str = None,
):
    device = torch.device("cpu")
    print("=" * 70)
    print("  PHYSICS-RESIDUAL TTA: COMPREHENSIVE ABLATION BENCHMARKS")
    print(f"  Samples: {num_samples} held-out trajectories | Device: {device}")
    print("=" * 70)

    # 1. Load model & weights
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MeshGraphNet(
        node_in_dim=ckpt["node_in_dim"],
        edge_in_dim=ckpt["edge_in_dim"],
        out_dim=ckpt["out_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # 2. Load dataset
    with h5py.File(data_path, "r") as f:
        tensor = f["tensor"]
        x_coord = f["x-coordinate"][:]
        N, T, X = tensor.shape
        dx_val = float(x_coord[1] - x_coord[0])
        dt_val = 0.01
        nu_val = 0.001

        # Build 1D chain graph
        src = np.arange(X - 1)
        dst = src + 1
        edge_index_np = np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        dx = x_coord[dst] - x_coord[src]
        edge_attr_np = np.concatenate([dx, -dx])[:, None]
        edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32)

        # Collect evaluation samples
        step_stride = max(1, N // num_samples)
        sample_tuples = []
        for i in range(num_samples):
            traj_idx = i * step_stride
            t_step = 10 + (i * 7) % (T - 25)
            u_curr = tensor[traj_idx, t_step]
            u_next = tensor[traj_idx, t_step + 1]

            x_in = torch.tensor(u_curr, dtype=torch.float32).unsqueeze(-1)
            x_target = torch.tensor(u_next, dtype=torch.float32).unsqueeze(-1)
            data_obj = Data(x=x_in, edge_index=edge_index, edge_attr=edge_attr, y=x_target)
            sample_tuples.append((traj_idx, t_step, data_obj))

    tta_engine = PhysicsResidualTTA(
        model=model,
        device=device,
        dx=dx_val,
        dt=dt_val,
        nu=nu_val,
        anchor_weight=0.1,
    )

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_samples": num_samples,
        "step_ablation": [],
        "anchor_ablation": [],
        "lr_ablation": [],
        "scope_ablation": [],
    }

    # =========================================================================
    # ABLATION 1: Step Budget K
    # =========================================================================
    print("\n--- [Ablation 1/4] Adaptation Steps K Sensitivity ---")
    step_candidates = [0, 1, 3, 5, 8, 12, 16, 20]
    for k in step_candidates:
        mses = []
        residuals = []
        for _, _, s in sample_tuples:
            if k == 0:
                with torch.no_grad():
                    pred = model(s.x, s.edge_index, s.edge_attr).squeeze(-1)
                    mse = torch.mean((pred - s.y.squeeze(-1)) ** 2).item()
                    loss_fn = BurgersPhysicsLoss(nu=nu_val, dx=dx_val, dt=dt_val, periodic=True)
                    phys_res, _, _ = loss_fn(pred, s.x.squeeze(-1), nu=nu_val)
                    mses.append(mse)
                    residuals.append(phys_res.item())
            else:
                res = tta_engine.adapt_sample(
                    sample=s,
                    tta_steps=k,
                    tta_lr=0.00008,
                    anchor_override=0.1,
                )
                mses.append(res["adapted_mse"])
                residuals.append(res["adapted_physics_loss"])

        mean_mse = float(np.mean(mses))
        mean_res = float(np.mean(residuals))
        base_mse = results["step_ablation"][0]["mean_mse"] if results["step_ablation"] else mean_mse
        improvement = ((base_mse - mean_mse) / base_mse * 100.0) if base_mse > 0 else 0.0

        results["step_ablation"].append({
            "steps": k,
            "mean_mse": mean_mse,
            "mean_residual": mean_res,
            "gain_pct": improvement,
        })
        print(f"  K = {k:2d} | MSE: {mean_mse:.6e} | Residual: {mean_res:6.2f} | Gain: {improvement:+6.2f}%")

    # =========================================================================
    # ABLATION 2: Anchor Regularization Weight lambda_anchor
    # =========================================================================
    print("\n--- [Ablation 2/4] Anchor Regularization Weight lambda ---")
    anchor_candidates = [0.0, 0.005, 0.02, 0.1, 0.5, 1.0]
    for lam in anchor_candidates:
        mses = []
        residuals = []
        for _, _, s in sample_tuples:
            res = tta_engine.adapt_sample(
                sample=s,
                tta_steps=12,
                tta_lr=0.00008,
                anchor_override=lam,
            )
            mses.append(res["adapted_mse"])
            residuals.append(res["adapted_physics_loss"])

        mean_mse = float(np.mean(mses))
        mean_res = float(np.mean(residuals))
        base_mse = results["step_ablation"][0]["mean_mse"]
        improvement = ((base_mse - mean_mse) / base_mse * 100.0)

        results["anchor_ablation"].append({
            "lambda_anchor": lam,
            "mean_mse": mean_mse,
            "mean_residual": mean_res,
            "gain_pct": improvement,
        })
        print(f"  lambda = {lam:5.3f} | MSE: {mean_mse:.6e} | Residual: {mean_res:6.2f} | Gain: {improvement:+6.2f}%")

    # =========================================================================
    # ABLATION 3: Learning Rate eta
    # =========================================================================
    print("\n--- [Ablation 3/4] Test-Time Learning Rate eta Sensitivity ---")
    lr_candidates = [1e-5, 4e-5, 8e-5, 2e-4, 5e-4]
    for lr in lr_candidates:
        mses = []
        residuals = []
        for _, _, s in sample_tuples:
            res = tta_engine.adapt_sample(
                sample=s,
                tta_steps=12,
                tta_lr=lr,
                anchor_override=0.1,
            )
            mses.append(res["adapted_mse"])
            residuals.append(res["adapted_physics_loss"])

        mean_mse = float(np.mean(mses))
        mean_res = float(np.mean(residuals))
        base_mse = results["step_ablation"][0]["mean_mse"]
        improvement = ((base_mse - mean_mse) / base_mse * 100.0)

        results["lr_ablation"].append({
            "learning_rate": lr,
            "mean_mse": mean_mse,
            "mean_residual": mean_res,
            "gain_pct": improvement,
        })
        print(f"  eta = {lr:.0e} | MSE: {mean_mse:.6e} | Residual: {mean_res:6.2f} | Gain: {improvement:+6.2f}%")

    # =========================================================================
    # ABLATION 4: Parameter Adaptation Scope (Decoder vs Full Model)
    # =========================================================================
    print("\n--- [Ablation 4/4] Parameter Adaptation Scope ---")
    scopes = ["decoder_only", "full_gnn"]
    for scope in scopes:
        mses = []
        residuals = []
        latencies = []

        for _, _, s in sample_tuples:
            x = s.x.to(device)
            edge_index = s.edge_index.to(device)
            edge_attr = s.edge_attr.to(device)
            y_true = s.y.to(device).squeeze(-1)
            u_curr = x.squeeze(-1)

            # Baseline forward
            model.eval()
            with torch.no_grad():
                base_pred = model(x, edge_index, edge_attr).squeeze(-1)

            loss_fn = BurgersPhysicsLoss(nu=nu_val, dx=dx_val, dt=dt_val, periodic=True)

            # Freeze / Unfreeze
            if scope == "decoder_only":
                for p in model.encoder.parameters():
                    p.requires_grad = False
                for p in model.processor_blocks.parameters():
                    p.requires_grad = False
                for p in model.decoder.parameters():
                    p.requires_grad = True
                opt_params = model.decoder.parameters()
            else:  # full_gnn
                for p in model.parameters():
                    p.requires_grad = True
                opt_params = model.parameters()

            # Cache initial weights
            cached_state = copy.deepcopy(model.state_dict())
            optimizer = torch.optim.Adam(opt_params, lr=0.00008)

            t0 = time.time()
            for _ in range(10):
                optimizer.zero_grad()
                pred = model(x, edge_index, edge_attr).squeeze(-1)
                phys_loss, _, _ = loss_fn(pred, u_curr, nu=nu_val)
                anchor = 0.1 * torch.mean((pred - base_pred) ** 2)
                (phys_loss + anchor).backward()
                torch.nn.utils.clip_grad_norm_(opt_params, max_norm=0.5)
                optimizer.step()
            dt_step = time.time() - t0

            with torch.no_grad():
                adapted_pred = model(x, edge_index, edge_attr).squeeze(-1)
                mse = torch.mean((adapted_pred - y_true) ** 2).item()
                phys_res, _, _ = loss_fn(adapted_pred, u_curr, nu=nu_val)

            # Restore original weights
            model.load_state_dict(cached_state)

            mses.append(mse)
            residuals.append(phys_res.item())
            latencies.append(dt_step)

        mean_mse = float(np.mean(mses))
        mean_res = float(np.mean(residuals))
        mean_lat = float(np.mean(latencies)) * 1000.0  # ms
        base_mse = results["step_ablation"][0]["mean_mse"]
        improvement = ((base_mse - mean_mse) / base_mse * 100.0)

        results["scope_ablation"].append({
            "scope": scope,
            "mean_mse": mean_mse,
            "mean_residual": mean_res,
            "latency_ms": mean_lat,
            "gain_pct": improvement,
        })
        print(f"  Scope: {scope:<12} | MSE: {mean_mse:.6e} | Res: {mean_res:5.2f} | Latency: {mean_lat:6.1f}ms | Gain: {improvement:+6.2f}%")

    if output_json_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Ablation benchmark results saved to: {output_json_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation studies for Physics-Residual TTA")
    parser.add_argument("--data_path", type=str, default="files (5)/1D_Burgers_Sols_Nu0.001.hdf5")
    parser.add_argument("--checkpoint_path", type=str, default="real_burgers_model.pt")
    parser.add_argument("--num_samples", type=int, default=6)
    parser.add_argument("--output_json", type=str, default="scripts/ablation_benchmark_results.json")

    args = parser.parse_args()
    run_ablation_studies(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        num_samples=args.num_samples,
        output_json_path=args.output_json,
    )
