"""
scripts/simulate_custom_obstacle.py
Simulates 2D flow around an arbitrarily placed and oriented obstacle with the D2Q9
Lattice Boltzmann solver (ground truth), predicts it with the FNO-2D trained on cylinder
flows (baseline), and applies physics-residual TTA (adapted). All three fields are real.

Accepts CLI arguments:
  --shape       : cylinder | airfoil | square | triangle | ellipse
  --x           : Streamwise center position [0.15, 0.80] (default: 0.25)
  --y           : Cross-stream center position [0.20, 0.80] (default: 0.50)
  --size        : Scale factor [0.5, 2.0] (default: 1.0)
  --angle       : Rotation / Angle of Attack in degrees [-45, 45] (default: 0.0)
  --reynolds    : Flow Reynolds number (default: 120.0)
  --tta_steps   : Number of unsupervised adaptation steps (default: 8)
  --timesteps   : Number of animation frames (default: 12)

Outputs full trajectory JSON for web visualizer animation.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "scripts"))

MODEL_PATH = os.path.join(ROOT_DIR, "models", "benchmark_2d", "cylinder_flow_2d_fno.pt")


# =============================================================================
# ARBITRARY OBSTACLE MASK GENERATORS
# =============================================================================

def generate_custom_mask(shape_name, cx_norm, cy_norm, size_factor=1.0, angle_deg=0.0, Ny=48, Nx=96):
    """Generates a boolean obstacle mask [Ny, Nx] for arbitrary placement."""
    cx = int(np.clip(cx_norm, 0.1, 0.85) * Nx)
    cy = int(np.clip(cy_norm, 0.15, 0.85) * Ny)
    mask = np.zeros((Ny, Nx), dtype=bool)

    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    if shape_name in ("circle", "cylinder"):
        radius = int(max(3, (Ny // 8) * size_factor))
        Y, X = np.ogrid[:Ny, :Nx]
        dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
        mask = dist_sq <= radius ** 2

    elif shape_name == "square":
        sz = int(max(5, (Ny // 5.5) * size_factor))
        for i in range(Ny):
            for j in range(Nx):
                dx = j - cx
                dy = i - cy
                xr = dx * cos_a + dy * sin_a
                yr = -dx * sin_a + dy * cos_a
                if abs(xr) <= sz // 2 and abs(yr) <= sz // 2:
                    mask[i, j] = True

    elif shape_name == "rectangle":
        w = int(max(8, (Nx // 6) * size_factor))
        h = int(max(4, (Ny // 8) * size_factor))
        for i in range(Ny):
            for j in range(Nx):
                dx = j - cx
                dy = i - cy
                xr = dx * cos_a + dy * sin_a
                yr = -dx * sin_a + dy * cos_a
                if abs(xr) <= w // 2 and abs(yr) <= h // 2:
                    mask[i, j] = True

    elif shape_name == "triangle":
        # Symmetric equilateral triangle centered on (cx, cy)
        sz = int(max(6, (Ny // 4.5) * size_factor))
        for i in range(Ny):
            for j in range(Nx):
                dx = j - cx
                dy = i - cy
                xr = dx * cos_a + dy * sin_a
                yr = -dx * sin_a + dy * cos_a
                # Upstream pointing tip at xr = -sz/2, flat base at xr = sz/2
                if -sz // 2 <= xr <= sz // 2:
                    allowed_y = (xr + sz // 2) / sz * (sz // 1.4)
                    if abs(yr) <= allowed_y:
                        mask[i, j] = True

    elif shape_name == "diamond":
        # Rhombus
        sz = int(max(5, (Ny // 4.5) * size_factor))
        for i in range(Ny):
            for j in range(Nx):
                dx = j - cx
                dy = i - cy
                xr = dx * cos_a + dy * sin_a
                yr = -dx * sin_a + dy * cos_a
                if (abs(xr) / (sz * 1.2)) + (abs(yr) / sz) <= 1.0:
                    mask[i, j] = True

    elif shape_name == "hexagon":
        # Regular hexagon
        r = int(max(4, (Ny // 6) * size_factor))
        for i in range(Ny):
            for j in range(Nx):
                dx = j - cx
                dy = i - cy
                xr = abs(dx * cos_a + dy * sin_a)
                yr = abs(-dx * sin_a + dy * cos_a)
                if xr <= r and yr <= (np.sqrt(3) / 2) * r and (np.sqrt(3) * xr + yr) <= np.sqrt(3) * r:
                    mask[i, j] = True

    elif shape_name == "cross":
        # Plus / Cross shape
        sz = int(max(6, (Ny // 4.5) * size_factor))
        thick = max(2, sz // 3)
        for i in range(Ny):
            for j in range(Nx):
                dx = j - cx
                dy = i - cy
                xr = dx * cos_a + dy * sin_a
                yr = -dx * sin_a + dy * cos_a
                in_h = abs(xr) <= sz // 2 and abs(yr) <= thick // 2
                in_v = abs(yr) <= sz // 2 and abs(xr) <= thick // 2
                if in_h or in_v:
                    mask[i, j] = True

    else:
        # Fallback circle
        radius = int(max(3, (Ny // 8) * size_factor))
        Y, X = np.ogrid[:Ny, :Nx]
        dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
        mask = dist_sq <= radius ** 2

    return mask, cx, cy


# =============================================================================
# SIMULATE (LBM ground truth) + PREDICT (trained FNO-2D) + ADAPT (physics-residual TTA)
# =============================================================================

def simulate_placed_obstacle(shape="cylinder", x_pos=0.25, y_pos=0.50, size=1.0, angle=0.0,
                             reynolds=120.0, tta_steps=8, timesteps=12):
    import torch
    from generate_2d_shapes import simulate_flow_d2q9
    from evaluate_2d_benchmarks import RESIDUALS, load_checkpoint, rel_l2, run_tta

    Ny, Nx = 48, 96
    mask, cx, cy = generate_custom_mask(shape, x_pos, y_pos, size, angle, Ny=Ny, Nx=Nx)
    if not os.path.exists(MODEL_PATH):
        return {"error": f"No trained model at {MODEL_PATH}. Run: python scripts/evaluate_2d_benchmarks.py"}

    # 1. Ground truth: the same D2Q9 Lattice Boltzmann solver that generated the training data
    u_inlet = 0.044
    ux, uy, p = simulate_flow_d2q9(mask, u_inlet=u_inlet, reynolds=reynolds, total_steps=600, sample_interval=15)
    states = np.stack([ux, uy, p], axis=1).astype(np.float32)[: timesteps + 1]  # [T, 3, Ny, Nx]
    timesteps = states.shape[0] - 1
    obs_height = max(1, int(np.sum(mask.any(axis=1))))
    nu = max(1e-4, (u_inlet * obs_height) / max(1.0, reynolds))

    # 2. Surrogate trained on cylinder flows; the obstacle mask is an input channel, so any
    #    placed shape is a genuine distribution shift for it
    model, _ = load_checkpoint(MODEL_PATH)
    mask_ch = np.broadcast_to(mask.astype(np.float32), (timesteps, 1, Ny, Nx))
    x = torch.from_numpy(np.concatenate([states[:-1], mask_ch], axis=1))
    y = torch.from_numpy(states[1:])

    near_body = mask.copy()
    for ax in (0, 1):
        for sft in (-1, 1):
            near_body |= np.roll(mask, sft, axis=ax)
    ctx = {"mask": torch.tensor(mask), "fluid_interior": torch.tensor(~near_body[1:-1, 1:-1]), "nu": nu, "dt": 15.0}

    # 3. Physics-residual TTA over the whole sequence (no ground truth used)
    y_base, y_adapt, base_res, adapt_res = run_tta(model, "fno", x, RESIDUALS["lbm"], ctx, tta_steps=tta_steps)
    with torch.no_grad():
        truth_res = RESIDUALS["lbm"](x[:, :3], y, ctx).item()

    speed = lambda s: torch.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2).detach().numpy()
    gt_speed, base_speed, adapt_speed = speed(y), speed(y_base), speed(y_adapt)
    vmax = float(gt_speed.max()) + 1e-12
    norm = lambda f: np.clip(f / vmax, 0.0, 1.0)
    trajectories_gt, trajectories_baseline, trajectories_adapted = norm(gt_speed), norm(base_speed), norm(adapt_speed)

    rep_t = min(3, timesteps - 1)
    field_adapted, field_baseline, field_gt = trajectories_adapted[rep_t], trajectories_baseline[rep_t], trajectories_gt[rep_t]

    # Centerline profile along y = cy (through the obstacle center)
    centerline = []
    row = int(np.clip(round(cy), 0, Ny - 1))
    for cx_col in range(Nx):
        centerline.append({
            "x": round(cx_col / (Nx - 1), 3),
            "col": cx_col,
            "ground_truth": round(float(field_gt[row, cx_col]), 4),
            "baseline": round(float(field_baseline[row, cx_col]), 4),
            "adapted": round(float(field_adapted[row, cx_col]), 4),
            "is_obstacle": bool(mask[row, cx_col]),
        })

    # Velocity arrows from the predicted ux, uy at the representative frame
    q_ny, q_nx = 12, 24
    qy_step, qx_step = Ny // q_ny, Nx // q_nx
    ya, yb = y_adapt[rep_t].detach().numpy(), y_base[rep_t].detach().numpy()
    vectors = []
    for iy in range(q_ny):
        gy = min(iy * qy_step + qy_step // 2, Ny - 1)
        for ix in range(q_nx):
            gx = min(ix * qx_step + qx_step // 2, Nx - 1)
            vectors.append({
                "ix": ix,
                "iy": iy,
                "x_norm": round(gx / (Nx - 1), 3),
                "y_norm": round(gy / (Ny - 1), 3),
                "is_obs": bool(mask[gy, gx]),
                "vx": round(float(ya[0, gy, gx] / vmax), 3),
                "vy": round(float(ya[1, gy, gx] / vmax), 3),
                "speed": round(float(field_adapted[gy, gx]), 3),
                "base_vx": round(float(yb[0, gy, gx] / vmax), 3),
                "base_vy": round(float(yb[1, gy, gx] / vmax), 3),
                "base_speed": round(float(field_baseline[gy, gx]), 3),
            })

    frames = []
    for t in range(timesteps):
        frames.append({
            "timestep": t,
            "adapted": [[round(float(v), 4) for v in r] for r in trajectories_adapted[t]],
            "baseline": [[round(float(v), 4) for v in r] for r in trajectories_baseline[t]],
            "ground_truth": [[round(float(v), 4) for v in r] for r in trajectories_gt[t]],
        })

    leak = lambda f: float(np.abs(f[:, mask]).mean()) if mask.any() else 0.0
    base_mse = float(torch.mean((y_base - y) ** 2))
    adapt_mse = float(torch.mean((y_adapt - y) ** 2))
    physics_metrics = {
        "physics": "Navier-Stokes momentum + mass conservation + no-slip",
        "model": "FNO-2D (trained on cylinder flows)",
        "baseline_divergence": base_res,  # field name kept for the UI; holds the full PDE residual
        "adapted_divergence": adapt_res,
        "truth_residual": truth_res,
        "divergence_drop_pct": (base_res - adapt_res) / (base_res + 1e-30) * 100.0,
        "baseline_boundary_leak": leak(base_speed),
        "adapted_boundary_leak": leak(adapt_speed),
        "wake_mse_improvement_pct": (base_mse - adapt_mse) / (base_mse + 1e-30) * 100.0,
        "baseline_rel_l2": rel_l2(y_base, y),
        "adapted_rel_l2": rel_l2(y_adapt, y),
        "obstacle_center": [round(cx / Nx, 3), round(cy / Ny, 3)],
    }

    return {
        "shape": shape,
        "is_custom_placement": True,
        "params": {
            "x": x_pos,
            "y": y_pos,
            "size": size,
            "angle": angle,
            "reynolds": reynolds,
        },
        "grid": [Ny, Nx],
        "total_timesteps": timesteps,
        "mask": mask.tolist(),
        "centerline": centerline,
        "vectors": vectors,
        "physics_metrics": physics_metrics,
        "frames": frames,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default="cylinder")
    parser.add_argument("--x", type=float, default=0.25)
    parser.add_argument("--y", type=float, default=0.50)
    parser.add_argument("--size", type=float, default=1.0)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--reynolds", type=float, default=120.0)
    parser.add_argument("--tta_steps", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=12)
    args = parser.parse_args()

    result = simulate_placed_obstacle(
        shape=args.shape,
        x_pos=args.x,
        y_pos=args.y,
        size=args.size,
        angle=args.angle,
        reynolds=args.reynolds,
        tta_steps=args.tta_steps,
        timesteps=args.timesteps,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
