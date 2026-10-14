"""
evaluate.py
Computes baseline accuracy metrics for a trained model on a held-out test
set. This produces the "before adaptation" numbers that physics-residual TTA
results get compared against later.

Usage:
    python evaluate.py --dataset burgers --data_path ./data/test_set.hdf5 \
                        --checkpoint_path baseline_model.pt
"""

import argparse

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

    return {"mse": mse, "relative_mse": rel_mse, "num_predictions": preds.shape[0]}


def load_checkpoint(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = MeshGraphNet(
        node_in_dim=ckpt["node_in_dim"],
        edge_in_dim=ckpt["edge_in_dim"],
        out_dim=ckpt["out_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["burgers", "cylinder"], required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--meta_path", default=None, help="required for --dataset cylinder")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "burgers":
        dataset = load_burgers(args.data_path)
    else:
        if args.meta_path is None:
            raise ValueError("--meta_path is required for --dataset cylinder")
        dataset = load_cylinder_flow(args.data_path, args.meta_path)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = load_checkpoint(args.checkpoint_path, device)
    metrics = evaluate_model(model, loader, device)

    print(f"Evaluated on {metrics['num_predictions']} node-level predictions")
    print(f"  MSE:          {metrics['mse']:.6f}")
    print(f"  Relative MSE: {metrics['relative_mse']:.6f}")


if __name__ == "__main__":
    main()
