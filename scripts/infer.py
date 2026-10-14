"""
scripts/infer.py
Run inference with a trained MeshGraphNet on a slice of the Burgers dataset
and output JSON results for the API server.

Usage (called by server/index.js):
    python infer.py --dataset burgers --data_path <path> \
                    --checkpoint_path <ckpt.pt> \
                    --traj 0 --timestep 0 --num_steps 5
"""

import argparse
import json
import sys
import os

import torch
import numpy as np

# Allow importing model/data from the files (5) directory
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "files (5)")))

from model import MeshGraphNet
from data import load_burgers


def load_checkpoint(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MeshGraphNet(
        node_in_dim=ckpt["node_in_dim"],
        edge_in_dim=ckpt["edge_in_dim"],
        out_dim=ckpt["out_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["burgers", "cylinder"], default="burgers")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--traj", type=int, default=0)
    parser.add_argument("--timestep", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint_path, device)

    if args.dataset == "burgers":
        # Load a small subset: only the requested trajectory
        dataset = load_burgers(args.data_path, num_samples=args.traj + 1)

        # Collect up to num_steps timestep graphs starting at args.timestep
        import h5py
        with h5py.File(args.data_path, "r") as f:
            tensor = f["tensor"][args.traj]  # [T, X]
            x_coord = f["x-coordinate"][:]   # [X]

        T, X = tensor.shape
        results = {"x_coord": x_coord.tolist(), "steps": [], "dataset": "burgers"}

        # Grab the edge_index / edge_attr from one sample
        sample = dataset[0]
        edge_index = sample.edge_index.to(device)
        edge_attr = sample.edge_attr.to(device)

        start = min(args.timestep, T - 2)
        end = min(start + args.num_steps, T - 1)

        for t in range(start, end):
            x_in = torch.tensor(tensor[t], dtype=torch.float32).unsqueeze(-1).to(device)  # [X,1]
            x_true = tensor[t + 1].tolist()

            with torch.no_grad():
                x_pred = model(x_in, edge_index, edge_attr).squeeze(-1).cpu().numpy()

            x_input = tensor[t].tolist()
            error = (np.array(x_pred) - np.array(x_true)) ** 2

            results["steps"].append({
                "t": t,
                "input": x_input,
                "true": x_true,
                "predicted": x_pred.tolist(),
                "mse": float(np.mean(error)),
                "relative_mse": float(np.mean(error) / (np.mean(np.array(x_true) ** 2) + 1e-8)),
            })

        print(json.dumps(results))
    else:
        print(json.dumps({"error": "CylinderFlow inference not yet supported via API"}))


if __name__ == "__main__":
    main()
