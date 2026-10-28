"""
evaluate.py
Complete evaluation pipeline for trained model checkpoints. Computes baseline
accuracy metrics on held-out test sets. This produces the "before adaptation"
numbers that physics-residual TTA results get compared against later.

Metrics computed:
  - MSE (Mean Squared Error)
  - Relative MSE (scale-invariant L2 error)
  - MAE (Mean Absolute Error)
  - Max Error (worst-case localized deviation)
  - Per-node variance analysis
  - Conservation metrics (kinetic energy consistency)

Usage:
    python evaluate.py --dataset burgers --data_path ./data/test_set.hdf5 \
                        --checkpoint_path baseline_model.pt
"""

import argparse
import json
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from data import load_burgers, load_cylinder_flow
from model import MeshGraphNet


def relative_mse(pred, true, eps=1e-8):
    """
    Mean squared error, scaled by the magnitude of the true values, so error
    is comparable across cases with different value ranges:
        relative_mse = mean((pred - true)^2) / (mean(true^2) + eps)
    """
    mse = torch.mean((pred - true) ** 2)
    scale = torch.mean(true**2) + eps
    return (mse / scale).item()


def kinetic_energy_error(pred, true):
    """
    Compute relative error in kinetic energy conservation:
        KE = 0.5 * sum(u^2)
    """
    ke_true = 0.5 * torch.sum(true ** 2).item()
    ke_pred = 0.5 * torch.sum(pred ** 2).item()
    if abs(ke_true) < 1e-12:
        return 0.0
    return abs(ke_pred - ke_true) / abs(ke_true)


@torch.no_grad()
def evaluate_model(model, loader, device, detailed=False):
    model.eval()
    all_preds, all_targets = [], []
    per_sample_mse = []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr)
        all_preds.append(pred.cpu())
        all_targets.append(batch.y.cpu())
        
        # Per-sample MSE for distribution analysis
        sample_mse = torch.mean((pred - batch.y) ** 2).item()
        per_sample_mse.append(sample_mse)

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    mse = torch.mean((preds - targets) ** 2).item()
    rel_mse = relative_mse(preds, targets)
    mae = torch.mean(torch.abs(preds - targets)).item()
    max_err = torch.max(torch.abs(preds - targets)).item()
    
    # L1 and L2 norms
    l1_error = torch.sum(torch.abs(preds - targets)).item()
    l2_error = torch.sqrt(torch.sum((preds - targets) ** 2)).item()
    
    # Kinetic energy conservation
    ke_err = kinetic_energy_error(preds, targets)
    
    metrics = {
        "mse": mse,
        "relative_mse": rel_mse,
        "mae": mae,
        "max_error": max_err,
        "l1_error": l1_error,
        "l2_error": l2_error,
        "kinetic_energy_error": ke_err,
        "num_predictions": preds.shape[0],
    }
    
    if detailed:
        # Statistical distribution of per-batch errors
        per_sample_arr = np.array(per_sample_mse)
        metrics["error_stats"] = {
            "mean": float(np.mean(per_sample_arr)),
            "std": float(np.std(per_sample_arr)),
            "min": float(np.min(per_sample_arr)),
            "max": float(np.max(per_sample_arr)),
            "median": float(np.median(per_sample_arr)),
            "p95": float(np.percentile(per_sample_arr, 95)),
            "p99": float(np.percentile(per_sample_arr, 99)),
        }
    
    return metrics


def load_checkpoint(checkpoint_path, device):
    return MeshGraphNet.from_checkpoint(checkpoint_path, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["burgers", "cylinder"], required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--meta_path", default=None, help="required for --dataset cylinder")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_json", default=None, help="Path to save JSON results")
    parser.add_argument("--detailed", action="store_true", help="Include detailed error statistics")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.time()

    if args.dataset == "burgers":
        dataset = load_burgers(args.data_path)
    else:
        if args.meta_path is None:
            raise ValueError("--meta_path is required for --dataset cylinder")
        dataset = load_cylinder_flow(args.data_path, args.meta_path)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = load_checkpoint(args.checkpoint_path, device)
    metrics = evaluate_model(model, loader, device, detailed=args.detailed)

    eval_time = time.time() - start_time
    metrics["evaluation_time_s"] = round(eval_time, 2)
    metrics["device"] = str(device)
    metrics["checkpoint"] = args.checkpoint_path
    metrics["dataset"] = args.dataset

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved to {args.output_json}")

    print(f"Evaluated on {metrics['num_predictions']} node-level predictions ({eval_time:.1f}s)")
    print(f"  MSE:              {metrics['mse']:.6e}")
    print(f"  Relative MSE:     {metrics['relative_mse']:.6e}")
    print(f"  MAE:              {metrics['mae']:.6e}")
    print(f"  Max Error:        {metrics['max_error']:.6e}")
    print(f"  L2 Error:         {metrics['l2_error']:.6e}")
    print(f"  KE Conservation:  {metrics['kinetic_energy_error']:.6e}")


if __name__ == "__main__":
    main()
