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
    model = MeshGraphNet.from_checkpoint(checkpoint_path, device=device)
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
        import h5py
        with h5py.File(args.data_path, "r") as f:
            ds = f["tensor"]
            N = ds.shape[0]
            traj_idx = min(max(0, args.traj), N - 1)
            tensor = ds[traj_idx]  # [T, X]
            x_coord = f["x-coordinate"][:] if "x-coordinate" in f else np.linspace(0, 1, tensor.shape[1])

        T, X = tensor.shape
        results = {"x_coord": x_coord.tolist(), "steps": [], "dataset": "burgers", "trajectory": traj_idx}

        # Construct 1D chain graph directly (instant, zero memory overhead)
        src = np.arange(X - 1)
        dst = src + 1
        edge_index_np = np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)
        dx = x_coord[dst] - x_coord[src]
        edge_attr_np = np.concatenate([dx, -dx])[:, None]
        edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32, device=device)

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
