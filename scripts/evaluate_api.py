"""
scripts/evaluate_api.py
Evaluate a trained checkpoint and output JSON metrics.
Wraps the evaluate.py logic with JSON output support.

Usage:
    python evaluate_api.py --dataset burgers --data_path <path> \
                           --checkpoint_path <ckpt.pt> --output_json
"""

import argparse
import json
import sys
import os

import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "files (5)")))

from model import MeshGraphNet
from data import load_burgers, load_cylinder_flow


def relative_mse(pred, true, eps=1e-8):
    mse = torch.mean((pred - true) ** 2)
    scale = torch.mean(true ** 2) + eps
    return (mse / scale).item()


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr)
        all_preds.append(pred.cpu())
        all_targets.append(batch.y.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    mse = torch.mean((preds - targets) ** 2).item()
    rel_mse = relative_mse(preds, targets)
    mae = torch.mean(torch.abs(preds - targets)).item()
    max_err = torch.max(torch.abs(preds - targets)).item()

    return {
        "mse": round(mse, 8),
        "relative_mse": round(rel_mse, 8),
        "mae": round(mae, 8),
        "max_error": round(max_err, 8),
        "num_predictions": int(preds.shape[0]),
        "num_nodes": int(preds.shape[1]) if preds.dim() > 1 else 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["burgers", "cylinder"], default="burgers")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--meta_path", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_json", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "burgers":
        dataset = load_burgers(args.data_path)
    else:
        if not args.meta_path:
            print(json.dumps({"error": "--meta_path required for cylinder dataset"}))
            return
        dataset = load_cylinder_flow(args.data_path, args.meta_path)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    model = MeshGraphNet(
        node_in_dim=ckpt["node_in_dim"],
        edge_in_dim=ckpt["edge_in_dim"],
        out_dim=ckpt["out_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    metrics = evaluate_model(model, loader, device)
    metrics["checkpoint"] = os.path.basename(args.checkpoint_path)
    metrics["dataset"] = args.dataset
    metrics["device"] = str(device)

    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
