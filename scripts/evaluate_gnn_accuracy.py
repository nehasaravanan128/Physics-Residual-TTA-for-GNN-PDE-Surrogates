"""
scripts/evaluate_gnn_accuracy.py
Held-out accuracy benchmark for 1D Burgers MeshGraphNet checkpoints.

Evaluates on trajectories from the END of the PDEBench file (never seen by
train.py, which loads trajectories from the start), reporting:
  - one-step MSE / relative MSE / relative L2 error
  - autoregressive rollout relative L2 error after N steps

Usage:
    python scripts/evaluate_gnn_accuracy.py --checkpoints real_burgers_model.pt "files (5)/baseline_model.pt"
"""

import argparse
import os
import sys

import h5py
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "files (5)"))

from model import MeshGraphNet

DEFAULT_DATA = os.path.join(ROOT_DIR, "files (5)", "1D_Burgers_Sols_Nu0.001.hdf5")


def build_chain_graph(x_coord):
    X = len(x_coord)
    src = np.arange(X - 1)
    dst = src + 1
    edge_index = torch.tensor(np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1), dtype=torch.long)
    dx = x_coord[dst] - x_coord[src]
    edge_attr = torch.tensor(np.concatenate([dx, -dx])[:, None], dtype=torch.float32)
    return edge_index, edge_attr


@torch.no_grad()
def evaluate(model, tensor, edge_index, edge_attr, t_stride, rollout_steps):
    model.eval()
    sq_err, sq_true, mses = 0.0, 0.0, []
    for traj in tensor:
        for t in range(0, traj.shape[0] - 1, t_stride):
            x = torch.from_numpy(traj[t]).float().unsqueeze(-1)
            y = torch.from_numpy(traj[t + 1]).float().unsqueeze(-1)
            pred = model(x, edge_index, edge_attr)
            err = torch.sum((pred - y) ** 2).item()
            sq_err += err
            sq_true += torch.sum(y ** 2).item()
            mses.append(err / y.numel())

    # Autoregressive rollout from t=0
    roll_errs = []
    for traj in tensor:
        u = torch.from_numpy(traj[0]).float().unsqueeze(-1)
        for _ in range(rollout_steps):
            u = model(u, edge_index, edge_attr)
        true = torch.from_numpy(traj[rollout_steps]).float().unsqueeze(-1)
        roll_errs.append((torch.norm(u - true) / (torch.norm(true) + 1e-8)).item())

    return {
        "mse": float(np.mean(mses)),
        "rel_mse": sq_err / sq_true,
        "rel_l2": float(np.sqrt(sq_err / sq_true)),
        "rollout_rel_l2": float(np.mean(roll_errs)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--data_path", default=DEFAULT_DATA)
    parser.add_argument("--num_test_trajs", type=int, default=20)
    parser.add_argument("--t_stride", type=int, default=10)
    parser.add_argument("--rollout_steps", type=int, default=20)
    args = parser.parse_args()

    with h5py.File(args.data_path, "r") as f:
        n = f["tensor"].shape[0]
        tensor = f["tensor"][n - args.num_test_trajs:]
        x_coord = f["x-coordinate"][:]
    edge_index, edge_attr = build_chain_graph(x_coord)
    print(f"Held-out test set: trajectories {n - args.num_test_trajs}..{n - 1}\n")

    print(f"{'checkpoint':<40} {'MSE':>10} {'rel MSE':>10} {'rel L2':>8} {'accuracy':>9} {f'rollout@{args.rollout_steps}':>11}")
    for path in args.checkpoints:
        model = MeshGraphNet.from_checkpoint(path, device="cpu")
        m = evaluate(model, tensor, edge_index, edge_attr, args.t_stride, args.rollout_steps)
        print(f"{os.path.basename(path):<40} {m['mse']:>10.3e} {m['rel_mse']:>10.3e} {m['rel_l2']:>7.2%} "
              f"{1 - m['rel_l2']:>8.2%} {m['rollout_rel_l2']:>10.2%}")


if __name__ == "__main__":
    main()
