"""
scripts/get_2d_slice.py
Extracts one 2D frame for web canvas rendering: the true next state, the trained
surrogate's prediction (baseline) and its prediction after physics-residual TTA (adapted).

All three fields are real: the surrogate is the checkpoint trained by
scripts/evaluate_2d_benchmarks.py (models/benchmark_2d/), and TTA is the same routine
the benchmark uses. Results are cached per frame in models/benchmark_2d/slice_cache/.
"""

import os
import sys
import json
import argparse
import h5py
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
CKPT_DIR = os.path.join(ROOT_DIR, "models", "benchmark_2d")  # same folder evaluate_2d_benchmarks.py writes
CACHE_DIR = os.path.join(CKPT_DIR, "slice_cache")
MODEL_LABELS = {"fno": "FNO-2D", "unet": "UNet-2D"}


def get_shape_file(shape_name):
    lookup = {
        "cylinder": "cylinder_flow_2d.hdf5",
        "airfoil": "airfoil_flow_2d.hdf5",
        "square": "square_flow_2d.hdf5",
        "step": "step_flow_2d.hdf5",
        "multi_cylinder": "multi_cylinder_flow_2d.hdf5",
        "heat": "heat_2d.hdf5",
        "advection": "advection_2d.hdf5",
        "navier_stokes": "navier_stokes_2d.hdf5",
    }
    filename = lookup.get(shape_name.lower(), f"{shape_name}.hdf5")
    return os.path.join(DATASETS_DIR, filename)


def extract_slice(shape_name="cylinder", traj_idx=-1, timestep=0, model_type="fno"):
    filepath = get_shape_file(shape_name)
    if not os.path.exists(filepath):
        return {"error": f"Simulation dataset '{os.path.basename(filepath)}' is not present on disk."}
    fname = os.path.basename(filepath)
    ckpt_path = os.path.join(CKPT_DIR, f"{os.path.splitext(fname)[0]}_{model_type}.pt")
    if not os.path.exists(ckpt_path):
        return {"error": f"No trained model at {ckpt_path}. Run: python scripts/evaluate_2d_benchmarks.py"}

    with h5py.File(filepath, "r") as f:
        N, T = f["tensor"].shape[:2]
        eq_name = str(f.attrs.get("equation", shape_name))
    traj_idx = N - 1 if traj_idx < 0 else min(traj_idx, N - 1)
    timestep = max(0, min(timestep, T - 2))  # predict timestep -> timestep + 1

    # Fast path: a cached frame needs neither torch nor the model
    cache_path = os.path.join(CACHE_DIR, f"{shape_name}_{model_type}_{traj_idx}_{timestep}.json")
    if os.path.exists(cache_path) and os.path.getmtime(cache_path) > os.path.getmtime(ckpt_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from evaluate_2d_benchmarks import (
        PHYSICS_LABELS, RESIDUALS, SHAPE_DATASETS,
        load_checkpoint, load_dataset, make_pairs, rel_l2, run_tta,
    )
    physics = next((ph for _, f, _, ph in SHAPE_DATASETS if f == fname), None)
    if physics is None:
        return {"error": f"No physics residual is defined for '{fname}'."}
    states, extra, ctxs = load_dataset(fname, physics)

    model, ckpt = load_checkpoint(ckpt_path)
    x, y = make_pairs(states, extra, [traj_idx])
    x, y = x[timestep:timestep + 1], y[timestep:timestep + 1]
    ctx = ctxs[traj_idx]
    y_base, y_adapt, base_res, adapt_res = run_tta(model, model_type, x, RESIDUALS[physics], ctx)
    with torch.no_grad():
        truth_res = RESIDUALS[physics](x[:, :y.shape[1]], y, ctx).item()

    is_flow = physics == "lbm"
    mask = ctx["mask"].cpu().numpy() if is_flow else np.zeros(states.shape[-2:], dtype=bool)

    def display_field(s):
        s = s[0].detach().cpu().numpy()
        return np.sqrt(s[0] ** 2 + s[1] ** 2) if is_flow else s[0]

    gt_raw, base_raw, adapt_raw = display_field(y), display_field(y_base), display_field(y_adapt)
    sub_ny, sub_nx = gt_raw.shape

    # One shared colour scale (from the ground truth) so the three panels are comparable
    min_v, max_v = float(gt_raw.min()), float(gt_raw.max())
    norm = lambda f: np.clip((f - min_v) / (max_v - min_v + 1e-12), 0.0, 1.0)
    gt_field, baseline_field, adapted_field = norm(gt_raw), norm(base_raw), norm(adapt_raw)

    # 1. Centerline profile (y = mid-plane)
    center_y = sub_ny // 2
    centerline = []
    for cx in range(sub_nx):
        centerline.append({
            "x": round(cx / (sub_nx - 1), 3),
            "col": cx,
            "ground_truth": round(float(gt_field[center_y, cx]), 4),
            "baseline": round(float(baseline_field[center_y, cx]), 4),
            "adapted": round(float(adapted_field[center_y, cx]), 4),
            "is_obstacle": bool(mask[center_y, cx]),
        })

    # 2. Velocity arrows (12 x 24) from the predicted ux, uy; scalar datasets have none
    q_ny, q_nx = 12, 24
    qy_step, qx_step = max(1, sub_ny // q_ny), max(1, sub_nx // q_nx)
    vel_scale = max_v + 1e-12
    adapt_np, base_np = y_adapt[0].detach().cpu().numpy(), y_base[0].detach().cpu().numpy()
    vectors = []
    for iy in range(q_ny):
        gy = min(iy * qy_step + qy_step // 2, sub_ny - 1)
        for ix in range(q_nx):
            gx = min(ix * qx_step + qx_step // 2, sub_nx - 1)
            if is_flow:
                vx, vy = adapt_np[0, gy, gx] / vel_scale, adapt_np[1, gy, gx] / vel_scale
                base_vx, base_vy = base_np[0, gy, gx] / vel_scale, base_np[1, gy, gx] / vel_scale
            else:
                vx = vy = base_vx = base_vy = 0.0
            vectors.append({
                "ix": ix,
                "iy": iy,
                "x_norm": round(gx / (sub_nx - 1), 3),
                "y_norm": round(gy / (sub_ny - 1), 3),
                "is_obs": bool(mask[gy, gx]),
                "vx": round(float(vx), 3),
                "vy": round(float(vy), 3),
                "speed": round(float(adapted_field[gy, gx]), 3),
                "base_vx": round(float(base_vx), 3),
                "base_vy": round(float(base_vy), 3),
                "base_speed": round(float(baseline_field[gy, gx]), 3),
            })

    # 3. Physics metrics, all measured on this frame
    leak = lambda f: float(np.abs(f[mask]).mean()) if mask.any() else 0.0
    base_mse = float(torch.mean((y_base - y) ** 2))
    adapt_mse = float(torch.mean((y_adapt - y) ** 2))
    physics_metrics = {
        "physics": PHYSICS_LABELS[physics],
        "model": MODEL_LABELS.get(model_type, model_type),
        "baseline_divergence": base_res,  # field name kept for the UI; holds the full PDE residual
        "adapted_divergence": adapt_res,
        "truth_residual": truth_res,
        "divergence_drop_pct": (base_res - adapt_res) / (base_res + 1e-30) * 100.0,
        "baseline_boundary_leak": leak(base_raw),
        "adapted_boundary_leak": leak(adapt_raw),
        "wake_mse_improvement_pct": (base_mse - adapt_mse) / (base_mse + 1e-30) * 100.0,
        "baseline_rel_l2": rel_l2(y_base, y),
        "adapted_rel_l2": rel_l2(y_adapt, y),
        "held_out_trajectory": traj_idx not in ckpt["train_trajectories"],
    }

    result = {
        "shape": shape_name,
        "equation": eq_name,
        "trajectory": traj_idx,
        "timestep": timestep,
        "predicted_timestep": timestep + 1,
        "total_timesteps": T,
        "grid": [sub_ny, sub_nx],
        "min_val": round(min_v, 5),
        "max_val": round(max_v, 5),
        "mask": mask.tolist(),
        "ground_truth": [[round(float(v), 4) for v in row] for row in gt_field],
        "baseline": [[round(float(v), 4) for v in row] for row in baseline_field],
        "adapted": [[round(float(v), 4) for v in row] for row in adapted_field],
        "centerline": centerline,
        "vectors": vectors,
        "physics_metrics": physics_metrics,
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default="cylinder")
    parser.add_argument("--traj", type=int, default=-1, help="-1 = the last trajectory (never trained on)")
    parser.add_argument("--timestep", type=int, default=0)
    parser.add_argument("--model", default="fno", choices=["fno", "unet"])
    parser.add_argument("--prewarm", action="store_true", help="Cache every frame of the last trajectory of every dataset")
    args = parser.parse_args()

    if args.prewarm:
        shapes = ["cylinder", "airfoil", "square", "step", "multi_cylinder", "heat", "advection", "navier_stokes"]
        for shape in shapes:
            with h5py.File(get_shape_file(shape), "r") as f:
                T = f["tensor"].shape[1]
            for t in range(T - 1):
                extract_slice(shape, -1, t, args.model)
            print(f"cached {shape}: {T - 1} frames", flush=True)
        return

    result = extract_slice(args.shape, args.traj, args.timestep, args.model)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
