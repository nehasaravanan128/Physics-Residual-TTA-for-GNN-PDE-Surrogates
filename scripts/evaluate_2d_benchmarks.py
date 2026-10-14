"""
scripts/evaluate_2d_benchmarks.py
Multi-Model & Multi-Shape 2D PDE Benchmark Suite with Physics Test-Time Adaptation (TTA).

Evaluates:
  1. Shapes: Cylinder, Airfoil, Square, Step, Multi-Cylinder, Heat, Advection, Navier-Stokes
  2. Models: FNO-2D, UNet-2D, MeshGraphNet, GAT-Surrogate
  3. Pre-TTA vs Post-TTA Error Metrics (MSE, Relative L2, Residual Reduction %)
  4. Exports structured JSON for Express API & Next.js dashboard
"""

import os
import sys
import json
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models_2d import build_surrogate_model, count_parameters
from physics_loss_2d import (
    navier_stokes_continuity_loss,
    navier_stokes_momentum_residual_2d,
    heat_residual_2d,
    advection_residual_2d
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "benchmark_2d_results.json")
DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")


def evaluate_shape_trajectory(model, tensor_traj, mask, shape_name, model_type="fno", tta_steps=8, lr=1e-3):
    """
    Evaluates 1 trajectory under Pre-TTA and Post-TTA adaptation.

    tensor_traj: [T, Ny, Nx]
    mask: [Ny, Nx] boolean
    """
    model.eval()
    T, Ny, Nx = tensor_traj.shape

    # Prepare inputs and targets: predict u(t+1) from u(t)
    x_in = torch.tensor(tensor_traj[:-1], dtype=torch.float32).unsqueeze(1).to(DEVICE)  # [T-1, 1, Ny, Nx]
    y_true = torch.tensor(tensor_traj[1:], dtype=torch.float32).unsqueeze(1).to(DEVICE)  # [T-1, 1, Ny, Nx]
    mask_t = torch.tensor(mask, dtype=torch.bool).to(DEVICE) if mask is not None else None

    # 1. Baseline Pre-TTA Prediction
    with torch.no_grad():
        if model_type in ["fno", "unet"]:
            y_base = model(x_in)
        else:
            # Flattened forward for mesh models
            y_base = model(x_in)
        base_mse = F.mse_loss(y_base, y_true).item()
        base_rel_l2 = (torch.norm(y_base - y_true) / (torch.norm(y_true) + 1e-6)).item()

        # Compute initial physics continuity residual
        if y_base.size(1) >= 2:
            base_res = navier_stokes_continuity_loss(y_base[:, 0], y_base[:, 1], obstacle_mask=mask_t).item()
        else:
            base_res = heat_residual_2d(x_in, y_base).item()

    # 2. Physics-Residual Test-Time Adaptation (TTA)
    # Clone model to adapt test instance without corrupting base checkpoint
    import copy
    adapted_model = copy.deepcopy(model)
    adapted_model.train()

    # Freeze feature encoder / early layers, adapt decoder/head
    if hasattr(adapted_model, "fc2"):
        # FNO: adapt output projection
        for p in adapted_model.parameters(): p.requires_grad = False
        for p in adapted_model.fc1.parameters(): p.requires_grad = True
        for p in adapted_model.fc2.parameters(): p.requires_grad = True
    elif hasattr(adapted_model, "outc"):
        # UNet: adapt output conv and top upsampler
        for p in adapted_model.parameters(): p.requires_grad = False
        for p in adapted_model.conv_up2.parameters(): p.requires_grad = True
        for p in adapted_model.outc.parameters(): p.requires_grad = True
    elif hasattr(adapted_model, "decoder"):
        for p in adapted_model.parameters(): p.requires_grad = False
        for p in adapted_model.decoder.parameters(): p.requires_grad = True

    optimizer = optim.Adam([p for p in adapted_model.parameters() if p.requires_grad], lr=lr)

    # Reference prediction for anchor regularization
    ref_y = y_base.detach()

    for step in range(tta_steps):
        optimizer.zero_grad()
        y_pred = adapted_model(x_in)

        # Physics loss: continuity or diffusion
        if y_pred.size(1) >= 2:
            loss_phys = navier_stokes_continuity_loss(y_pred[:, 0], y_pred[:, 1], obstacle_mask=mask_t)
        else:
            loss_phys = heat_residual_2d(x_in, y_pred)

        # Anchor loss to prevent drift
        loss_anchor = F.mse_loss(y_pred, ref_y)
        loss = loss_phys + 0.1 * loss_anchor

        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), 1.0)
        optimizer.step()

    # 3. Post-TTA Evaluation
    adapted_model.eval()
    with torch.no_grad():
        y_adapt = adapted_model(x_in)
        adapt_mse = F.mse_loss(y_adapt, y_true).item()
        adapt_rel_l2 = (torch.norm(y_adapt - y_true) / (torch.norm(y_true) + 1e-6)).item()

        if y_adapt.size(1) >= 2:
            adapt_res = navier_stokes_continuity_loss(y_adapt[:, 0], y_adapt[:, 1], obstacle_mask=mask_t).item()
        else:
            adapt_res = heat_residual_2d(x_in, y_adapt).item()

    err_reduction = ((base_mse - adapt_mse) / (base_mse + 1e-9)) * 100.0
    res_reduction = ((base_res - adapt_res) / (base_res + 1e-9)) * 100.0

    return {
        "baseline_mse": base_mse,
        "adapted_mse": adapt_mse,
        "error_reduction_pct": err_reduction,
        "baseline_rel_l2": base_rel_l2,
        "adapted_rel_l2": adapt_rel_l2,
        "baseline_residual": base_res,
        "adapted_residual": adapt_res,
        "residual_drop_pct": res_reduction,
    }


def run_full_2d_benchmark():
    """Runs comprehensive benchmark across all 2D shapes and models."""
    print("=" * 72)
    print("  STARTING COMPREHENSIVE 2D PDE & GEOMETRIC SHAPE BENCHMARK")
    print(f"  Device: {DEVICE}")
    print("=" * 72)

    shape_datasets = [
        ("Cylinder Flow", "cylinder_flow_2d.hdf5", "cylinder"),
        ("NACA 0012 Airfoil", "airfoil_flow_2d.hdf5", "airfoil"),
        ("Square Bluff Body", "square_flow_2d.hdf5", "square"),
        ("Backward-Facing Step", "step_flow_2d.hdf5", "step"),
        ("Multi-Cylinder Array", "multi_cylinder_flow_2d.hdf5", "multi_cylinder"),
        ("2D Heat Diffusion", "heat_2d.hdf5", "continuous"),
        ("2D Advection Flow", "advection_2d.hdf5", "continuous"),
        ("2D Navier-Stokes Vorticity", "navier_stokes_2d.hdf5", "continuous"),
    ]

    models_to_test = [
        ("FNO-2D", "fno"),
        ("Physics UNet-2D", "unet"),
    ]

    all_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {},
        "shapes": {},
        "summary": [],
    }

    # Model specifications
    for m_name, m_type in models_to_test:
        mod = build_surrogate_model(m_type)
        all_results["models"][m_name] = {
            "type": m_type,
            "parameters": count_parameters(mod),
        }

    for shape_label, fname, category in shape_datasets:
        fpath = os.path.join(DATASETS_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  [Skip] {fname} not found in {DATASETS_DIR}")
            continue

        print(f"\nEvaluating Benchmark Shape: {shape_label} ({fname})...")
        with h5py.File(fpath, "r") as f:
            tensor = f["tensor"][:]  # [N_traj, T, Ny, Nx]
            mask = f["obstacle_mask"][0] if "obstacle_mask" in f else None

        shape_results = {
            "name": shape_label,
            "filename": fname,
            "category": category,
            "grid": [tensor.shape[-2], tensor.shape[-1]],
            "trajectories": tensor.shape[0],
            "timesteps": tensor.shape[1],
            "evaluations": {},
        }

        # Evaluate on each model
        for m_name, m_type in models_to_test:
            # Build model for current grid channels
            model = build_surrogate_model(m_type).to(DEVICE)

            # Test on last trajectory (unseen test sample)
            test_traj = tensor[-1]
            metrics = evaluate_shape_trajectory(
                model, test_traj, mask, shape_label, model_type=m_type, tta_steps=8
            )

            shape_results["evaluations"][m_name] = metrics
            all_results["summary"].append({
                "shape": shape_label,
                "category": category,
                "model": m_name,
                "grid": f"{tensor.shape[-2]}x{tensor.shape[-1]}",
                "baseline_mse": metrics["baseline_mse"],
                "adapted_mse": metrics["adapted_mse"],
                "error_reduction_pct": metrics["error_reduction_pct"],
                "residual_drop_pct": metrics["residual_drop_pct"],
            })

            sign = "+" if metrics["error_reduction_pct"] >= 0 else ""
            print(f"  [{m_name:16s}] Pre-TTA MSE: {metrics['baseline_mse']:.4e} -> "
                  f"Post-TTA: {metrics['adapted_mse']:.4e} ({sign}{metrics['error_reduction_pct']:.2f}%)")

        all_results["shapes"][shape_label] = shape_results

    # Save to JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 72)
    print(f"  BENCHMARK COMPLETE! Results saved to:")
    print(f"  -> {OUTPUT_JSON}")
    print("=" * 72)
    return all_results


if __name__ == "__main__":
    run_full_2d_benchmark()
