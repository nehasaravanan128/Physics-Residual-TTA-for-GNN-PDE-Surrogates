"""
scripts/dataset_info.py
Returns basic metadata about a dataset file as JSON for the UI's dataset explorer.
"""

import argparse
import json
import sys
import os


def burgers_info(data_path):
    import h5py
    with h5py.File(data_path, "r") as f:
        tensor = f["tensor"]
        N, T, X = tensor.shape
        x_coord = f["x-coordinate"][:]
        first_traj = tensor[0]  # [T, X]

    return {
        "dataset": "burgers",
        "num_trajectories": int(N),
        "num_timesteps": int(T),
        "num_grid_points": int(X),
        "x_min": float(x_coord.min()),
        "x_max": float(x_coord.max()),
        "file_size_mb": round(os.path.getsize(data_path) / 1e6, 1),
        "sample_trajectory": {
            "index": 0,
            "frames": [
                {"t": t, "values": first_traj[t].tolist()}
                for t in range(min(T, 10))  # first 10 frames for preview
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["burgers", "cylinder"], default="burgers")
    parser.add_argument("--data_path", required=True)
    args = parser.parse_args()

    try:
        if args.dataset == "burgers":
            info = burgers_info(args.data_path)
        else:
            info = {"error": "CylinderFlow info not yet implemented"}
        print(json.dumps(info))
    except Exception as e:
        print(json.dumps({"error": str(e)}))


if __name__ == "__main__":
    main()
