"""
scripts/dataset_info.py
Returns comprehensive metadata about any 1D or 2D PDE dataset file as JSON
for the web UI's dataset explorer.
"""

import argparse
import json
import sys
import os
import h5py
import numpy as np


def extract_dataset_info(data_path, dataset_hint="pde"):
    with h5py.File(data_path, "r") as f:
        tensor = f["tensor"]
        shape = list(tensor.shape)
        eq_name = f.attrs.get("equation", dataset_hint)
        shape_type = f.attrs.get("shape", "none")

        if len(shape) == 3:
            # 1D Dataset: [N_traj, T, X]
            N, T, X = shape
            x_coord = f["x-coordinate"][:] if "x-coordinate" in f else np.linspace(0, 1, X)
            first_traj = tensor[0]  # [T, X]

            return {
                "dataset": str(eq_name),
                "is_2d": False,
                "dimension": "1D",
                "num_trajectories": int(N),
                "num_timesteps": int(T),
                "num_grid_points": int(X),
                "grid_shape": [int(X)],
                "x_min": float(x_coord.min()),
                "x_max": float(x_coord.max()),
                "file_size_mb": round(os.path.getsize(data_path) / 1e6, 2),
                "sample_trajectory": {
                    "index": 0,
                    "frames": [
                        {"t": t, "values": [round(float(v), 5) for v in first_traj[t]]}
                        for t in range(min(T, 15))
                    ],
                },
            }

        elif len(shape) == 4:
            # 2D Dataset: [N_traj, T, Ny, Nx]
            N, T, Ny, Nx = shape
            x_coord = f["x-coordinate"][:] if "x-coordinate" in f else np.linspace(0, 1, Nx)
            y_coord = f["y-coordinate"][:] if "y-coordinate" in f else np.linspace(0, 1, Ny)
            first_traj = tensor[0]  # [T, Ny, Nx]

            # Cross-section along centerline for 1D line chart preview
            center_y = Ny // 2
            slice_1d = first_traj[:, center_y, :]  # [T, Nx]

            # Obstacle mask summary
            has_obstacle = "obstacle_mask" in f
            obs_pct = 0.0
            if has_obstacle:
                obs_mask = f["obstacle_mask"][:]
                obs_pct = round(float(np.mean(obs_mask)) * 100, 2)

            return {
                "dataset": str(eq_name),
                "is_2d": True,
                "dimension": "2D",
                "shape_type": str(shape_type),
                "has_obstacle": bool(has_obstacle),
                "obstacle_area_pct": obs_pct,
                "num_trajectories": int(N),
                "num_timesteps": int(T),
                "num_grid_points": int(Ny * Nx),
                "grid_shape": [int(Ny), int(Nx)],
                "x_min": float(x_coord.min()),
                "x_max": float(x_coord.max()),
                "y_min": float(y_coord.min()),
                "y_max": float(y_coord.max()),
                "file_size_mb": round(os.path.getsize(data_path) / 1e6, 2),
                "sample_trajectory": {
                    "index": 0,
                    "centerline_y": center_y,
                    "frames": [
                        {"t": t, "values": [round(float(v), 5) for v in slice_1d[t]]}
                        for t in range(min(T, 15))
                    ],
                },
            }

        else:
            return {"error": f"Unsupported tensor rank: {len(shape)} (shape: {shape})"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="burgers")
    parser.add_argument("--data_path", required=True)
    args = parser.parse_args()

    try:
        if not os.path.exists(args.data_path):
            print(json.dumps({"error": f"File not found: {args.data_path}"}))
            return
        info = extract_dataset_info(args.data_path, dataset_hint=args.dataset)
        print(json.dumps(info))
    except Exception as e:
        print(json.dumps({"error": str(e)}))


if __name__ == "__main__":
    main()
