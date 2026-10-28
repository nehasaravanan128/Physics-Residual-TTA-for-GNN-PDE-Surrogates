"""
scripts/evaluate_2d_benchmarks.py
Multi-Model & Multi-Shape 2D PDE Benchmark Suite with Physics Test-Time Adaptation (TTA).

Protocol (per dataset x model):
  1. Split the trajectories (see split_trajectories): every third one is held out
     for testing, the last one is held out for calibration (used only by
     evaluate_gated_tta_2d.py), and the surrogate (FNO-2D / UNet-2D) is trained on
     the rest, predicting the normalized one-step change u(t+1) - u(t).
  2. Save the checkpoint to models/benchmark_2d/ and reload it from disk.
  3. On each held-out test trajectory, measure the trained model, then run
     physics-residual TTA (output head only, anchored, best-residual state kept)
     with the residual of the equation that generated that dataset:
       - obstacle flows (LBM, ux/uy/p): Navier-Stokes momentum + mass conservation in the fluid
         + no-slip inside the obstacle (lattice units, 15 LBM steps per snapshot)
       - heat:       du/dt = kappa * lap(u)
       - advection:  du/dt + v . grad(u) = 0   (per-trajectory constant velocity)
       - vorticity:  dw/dt + u . grad(w) = nu * lap(w)   (velocity from the stream function)
  4. Report MSE / relative L2 before and after TTA, the copy-previous-step baseline,
     and the residual of the ground truth itself as a reference floor, averaged over
     the test trajectories (with the standard deviation across them).

Exports structured JSON for the Express API & Next.js dashboard.

Usage:
    python scripts/evaluate_2d_benchmarks.py            # reuses cached checkpoints
    python scripts/evaluate_2d_benchmarks.py --retrain  # retrains every model
"""

import argparse
import copy
import json
import math
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_2d import build_surrogate_model, count_parameters

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_2d_results.json")
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
CKPT_DIR = os.path.join(ROOT_DIR, "models", "benchmark_2d")

SHAPE_DATASETS = [
    # (label, filename, category, physics)
    ("Cylinder Flow", "cylinder_flow_2d.hdf5", "cylinder", "lbm"),
    ("NACA 0012 Airfoil", "airfoil_flow_2d.hdf5", "airfoil", "lbm"),
    ("Square Bluff Body", "square_flow_2d.hdf5", "square", "lbm"),
    ("Backward-Facing Step", "step_flow_2d.hdf5", "step", "lbm"),
    ("Multi-Cylinder Array", "multi_cylinder_flow_2d.hdf5", "multi_cylinder", "lbm"),
    ("2D Heat Diffusion", "heat_2d.hdf5", "continuous", "heat"),
    ("2D Advection Flow", "advection_2d.hdf5", "continuous", "advection"),
    ("2D Navier-Stokes Vorticity", "navier_stokes_2d.hdf5", "continuous", "vorticity"),
]

MODELS_TO_TEST = [
    ("FNO-2D", "fno"),
    ("Physics UNet-2D", "unet"),
]


# =============================================================================
# PHYSICS RESIDUALS (each takes current state c, predicted next state n: [B, C, H, W])
# =============================================================================

def _ddx(f, axis, h):
    """Periodic central difference."""
    return (torch.roll(f, -1, axis) - torch.roll(f, 1, axis)) / (2 * h)


def _lap(f, h):
    """Periodic 5-point Laplacian."""
    return sum(torch.roll(f, -1, a) + torch.roll(f, 1, a) - 2 * f for a in (-2, -1)) / h ** 2


def heat_residual(c, n, ctx):
    # Crank-Nicolson: diffusion evaluated at the midpoint state. Grid axes are (x, y).
    m = 0.5 * (c + n)
    r = (n - c) / ctx["dt"] - ctx["kappa"] * _lap(m, ctx["h"])
    return r.pow(2).mean()


def advection_residual(c, n, ctx):
    m = 0.5 * (c + n)
    vx, vy = ctx["vel"]
    r = (n - c) / ctx["dt"] + vx * _ddx(m, -2, ctx["h"]) + vy * _ddx(m, -1, ctx["h"])
    return r.pow(2).mean()


def vorticity_residual(c, n, ctx):
    # Spectral derivatives on the periodic [0, 2pi)^2 domain, matching the generator.
    m = 0.5 * (c + n)[:, 0]
    N = m.shape[-1]
    k = torch.fft.fftfreq(N, d=1.0 / N, device=m.device) * (2 * math.pi / ctx["L"])
    KX, KY = torch.meshgrid(k, k, indexing="ij")
    K2 = KX ** 2 + KY ** 2
    K2_safe = K2.clone()
    K2_safe[0, 0] = 1.0
    w_hat = torch.fft.fft2(m)
    psi_hat = w_hat / K2_safe
    u = torch.fft.ifft2(1j * KY * psi_hat).real
    v = torch.fft.ifft2(-1j * KX * psi_hat).real
    dwdx = torch.fft.ifft2(1j * KX * w_hat).real
    dwdy = torch.fft.ifft2(1j * KY * w_hat).real
    lap = torch.fft.ifft2(-K2 * w_hat).real
    r = (n - c)[:, 0] / ctx["dt"] + u * dwdx + v * dwdy - ctx["nu"] * lap
    return r.pow(2).mean()


def _grad_interior(f):
    """Central differences on interior cells; returns (d/dx, d/dy) for (y, x) grid axes."""
    return 0.5 * (f[..., 1:-1, 2:] - f[..., 1:-1, :-2]), 0.5 * (f[..., 2:, 1:-1] - f[..., :-2, 1:-1])


def _lap_interior(f):
    return f[..., 1:-1, 2:] + f[..., 1:-1, :-2] + f[..., 2:, 1:-1] + f[..., :-2, 1:-1] - 4 * f[..., 1:-1, 1:-1]


def lbm_residual(c, n, ctx):
    # Lattice units (dx = 1). Grid axes are (y, x); channels are (ux, uy, p) with p = rho / 3.
    fluid = ctx["fluid_interior"]
    m = 0.5 * (c + n)  # Crank-Nicolson midpoint
    ux, uy, p = m[:, 0], m[:, 1], m[:, 2]
    inner = lambda f: f[..., 1:-1, 1:-1]
    dpdx, dpdy = _grad_interior(p)
    loss = 0.0
    for k, dp in ((0, dpdx), (1, dpdy)):
        dudx, dudy = _grad_interior(m[:, k])
        r = (inner(n[:, k] - c[:, k]) / ctx["dt"] + inner(ux) * dudx + inner(uy) * dudy
             + dp / inner(3 * p) - ctx["nu"] * _lap_interior(m[:, k]))
        loss = loss + r[:, fluid].pow(2).mean()
    # Mass conservation for the weakly compressible LBM fluid: d(rho)/dt + div(rho u) = 0, rho = 3p
    rho = 3 * p
    dmx, _ = _grad_interior(rho * ux)
    _, dmy = _grad_interior(rho * uy)
    mass = inner(3 * (n[:, 2] - c[:, 2])) / ctx["dt"] + dmx + dmy
    loss = loss + mass[:, fluid].pow(2).mean()
    if ctx["mask"].any():
        loss = loss + n[:, :2, ctx["mask"]].pow(2).mean()  # no-slip: zero velocity inside the body
    return loss


RESIDUALS = {
    "lbm": lbm_residual,
    "heat": heat_residual,
    "advection": advection_residual,
    "vorticity": vorticity_residual,
}

PHYSICS_LABELS = {
    "lbm": "Navier-Stokes momentum + mass conservation + no-slip",
    "heat": "heat equation",
    "advection": "advection equation",
    "vorticity": "Navier-Stokes vorticity equation",
}


# =============================================================================
# DATA
# =============================================================================

def load_dataset(fname, physics):
    """
    Returns states [N, T, C, H, W], extra input channels [N, E, H, W] (or None),
    and a list of per-trajectory physics contexts.
    """
    with h5py.File(os.path.join(DATASETS_DIR, fname), "r") as f:
        if physics == "lbm":
            if "velocity" not in f:
                raise KeyError(f"{fname} has no 'velocity' field; regenerate with scripts/generate_2d_shapes.py")
            states = np.concatenate([f["velocity"][:], f["pressure"][:][:, :, None]], axis=2)  # [N, T, 3, Ny, Nx]
            masks = f["obstacle_mask"][:]
            nus = f["nu"][:]
            extra = masks[:, None].astype(np.float32)
            ctxs = []
            for mk, nu in zip(masks, nus):
                near_body = mk.copy()
                for ax in (0, 1):
                    for s in (-1, 1):
                        near_body |= np.roll(mk, s, axis=ax)
                ctxs.append({
                    "mask": torch.tensor(mk, device=DEVICE),
                    "fluid_interior": torch.tensor(~near_body[1:-1, 1:-1], device=DEVICE),
                    "nu": float(nu),
                    "dt": 15.0,
                })
        else:
            states = f["tensor"][:][:, :, None]  # [N, T, 1, X, Y]
            extra = None
            N = states.shape[0]
            if physics == "heat":
                ctxs = [{"dt": 0.005, "h": 1.0 / 64, "kappa": float(f.attrs.get("kappa", 0.01))}] * N
            elif physics == "advection":
                vel = f["velocity"][:]
                ctxs = [{"dt": 0.005, "h": 1.0 / 64, "vel": (float(v[0]), float(v[1]))} for v in vel]
            else:
                ctxs = [{"dt": 0.05, "L": 2 * math.pi, "nu": float(f.attrs.get("nu", 0.001))}] * N
    return states.astype(np.float32), extra, ctxs


def split_trajectories(N):
    """(train, calibration, test) trajectory ids. With 10 trajectories: train 0,1,3,4,6,7 / calibrate 9 / test 2,5,8."""
    test = [i for i in range(N - 1) if i % 3 == 2]
    calib = [N - 1]
    train = [i for i in range(N - 1) if i not in test]
    return train, calib, test


def make_pairs(states, extra, traj_ids):
    xs, ys = [], []
    for n in traj_ids:
        cur = torch.from_numpy(states[n, :-1])
        if extra is not None:
            cur = torch.cat([cur, torch.from_numpy(extra[n]).expand(cur.shape[0], -1, -1, -1)], dim=1)
        xs.append(cur)
        ys.append(torch.from_numpy(states[n, 1:]))
    return torch.cat(xs).to(DEVICE), torch.cat(ys).to(DEVICE)


# =============================================================================
# MODEL
# =============================================================================

class DeltaSurrogate(nn.Module):
    """
    Wraps an FNO/UNet core: standardizes each state channel and predicts the per-channel
    scaled one-step change. Scales are [C] lists stored in the checkpoint.
    """

    def __init__(self, core, n_state, in_mean, in_std, delta_scale):
        super().__init__()
        self.core = core
        self.n_state = n_state
        self.register_buffer("in_mean", torch.tensor(in_mean).view(1, -1, 1, 1))
        self.register_buffer("in_std", torch.tensor(in_std).view(1, -1, 1, 1))
        self.register_buffer("delta_scale", torch.tensor(delta_scale).view(1, -1, 1, 1))

    def forward(self, x):
        state = x[:, :self.n_state]
        x_in = torch.cat([(state - self.in_mean) / self.in_std, x[:, self.n_state:]], dim=1)
        return state + self.core(x_in) * self.delta_scale


def head_parameters(model, m_type):
    """The output head adapted at test time (mirrors decoder-only TTA in 1D)."""
    core = model.core
    if m_type == "fno":
        return list(core.fc1.parameters()) + list(core.fc2.parameters())
    return list(core.conv_up2.parameters()) + list(core.outc.parameters())


def build_model(m_type, n_state, n_in, in_mean, in_std, delta_scale):
    core = build_surrogate_model(m_type, in_channels=n_in, out_channels=n_state)
    return DeltaSurrogate(core, n_state, in_mean, in_std, delta_scale).to(DEVICE)


def train_and_save(m_type, states, extra, train_ids, ckpt_path, epochs, batch_size=8, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    x, y = make_pairs(states, extra, train_ids)
    n_state = y.shape[1]
    dims = (0, 2, 3)
    in_mean = x[:, :n_state].mean(dim=dims).tolist()
    in_std = x[:, :n_state].std(dim=dims).tolist()
    delta_scale = (y - x[:, :n_state]).std(dim=dims).tolist()
    model = build_model(m_type, n_state, x.shape[1], in_mean, in_std, delta_scale)
    ds = model.delta_scale

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.02)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(x.shape[0], device=DEVICE)
        total = 0.0
        for i in range(0, x.shape[0], batch_size):
            idx = perm[i:i + batch_size]
            loss = F.mse_loss(model(x[idx]) / ds, y[idx] / ds)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "m_type": m_type,
        "n_state": n_state,
        "n_in": x.shape[1],
        "in_mean": in_mean,
        "in_std": in_std,
        "delta_scale": delta_scale,
        "train_trajectories": list(map(int, train_ids)),
        "train_samples": int(x.shape[0]),
        "epochs": epochs,
        "final_train_loss": total / x.shape[0],
    }, ckpt_path)


def load_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = build_model(ckpt["m_type"], ckpt["n_state"], ckpt["n_in"], ckpt["in_mean"], ckpt["in_std"], ckpt["delta_scale"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


# =============================================================================
# EVALUATION + TTA
# =============================================================================

def rel_l2(pred, true):
    return (torch.norm(pred - true) / (torch.norm(true) + 1e-12)).item()


def evaluate_with_tta(model, m_type, x, y, residual_fn, ctx, tta_steps=10, tta_lr=2e-4, anchor=0.1):
    state = x[:, :y.shape[1]]
    y_base, y_adapt, base_res, adapt_res = run_tta(model, m_type, x, residual_fn, ctx, tta_steps, tta_lr, anchor)
    with torch.no_grad():
        truth_res = residual_fn(state, y, ctx).item()

    base_mse = F.mse_loss(y_base, y).item()
    adapt_mse = F.mse_loss(y_adapt, y).item()
    return {
        "baseline_mse": base_mse,
        "adapted_mse": adapt_mse,
        "error_reduction_pct": (base_mse - adapt_mse) / (base_mse + 1e-30) * 100.0,
        "baseline_rel_l2": rel_l2(y_base, y),
        "adapted_rel_l2": rel_l2(y_adapt, y),
        "persistence_rel_l2": rel_l2(state, y),
        "baseline_residual": base_res,
        "adapted_residual": adapt_res,
        "truth_residual": truth_res,
        "residual_drop_pct": (base_res - adapt_res) / (base_res + 1e-30) * 100.0,
    }


def run_tta(model, m_type, x, residual_fn, ctx, tta_steps=10, tta_lr=2e-4, anchor=0.1):
    """Physics-residual TTA on the output head. Returns (baseline pred, adapted pred, residuals)."""
    n_state = model.n_state
    state = x[:, :n_state]
    model.eval()
    with torch.no_grad():
        y_base = model(x)
        base_res = residual_fn(state, y_base, ctx).item()

    # Adapt a copy so the loaded checkpoint stays untouched. BatchNorm stays in eval mode.
    adapted = copy.deepcopy(model)
    adapted.eval()
    for p in adapted.parameters():
        p.requires_grad = False
    head = head_parameters(adapted, m_type)
    for p in head:
        p.requires_grad = True
    optimizer = torch.optim.Adam(head, lr=tta_lr)

    best_res = base_res
    best_state = copy.deepcopy(adapted.state_dict())
    scale = model.delta_scale
    for _ in range(tta_steps):
        pred = adapted(x)
        res = residual_fn(state, pred, ctx)
        if res.item() < best_res:
            best_res = res.item()
            best_state = copy.deepcopy(adapted.state_dict())
        # Residual normalized by its baseline value so the anchor weight means the same on every PDE
        loss = res / (base_res + 1e-30) + anchor * torch.mean(((pred - y_base) / scale) ** 2)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head, 1.0)
        optimizer.step()

    with torch.no_grad():
        final_res = residual_fn(state, adapted(x), ctx).item()
        if final_res < best_res:
            best_state = copy.deepcopy(adapted.state_dict())
        adapted.load_state_dict(best_state)
        y_adapt = adapted(x)
        adapt_res = residual_fn(state, y_adapt, ctx).item()
    return y_base, y_adapt, base_res, adapt_res


def run_full_2d_benchmark(epochs=80, retrain=False, only=None, train_only=False):
    print("=" * 96)
    print("  2D BENCHMARK: TRAINED SURROGATES + PHYSICS-RESIDUAL TTA ON HELD-OUT TRAJECTORIES")
    print(f"  Device: {DEVICE} | Epochs: {epochs} | Checkpoints: {CKPT_DIR}")
    print("=" * 96)

    all_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "Every third trajectory held out for testing and the last one for calibration; "
                    "models trained on the rest, reloaded from disk, then evaluated with and without TTA "
                    "on each test trajectory separately. Metrics are means over test trajectories (_std = spread).",
        "tta": {"steps": 10, "lr": 2e-4, "anchor": 0.1, "adapted": "output head"},
        "models": {},
        "shapes": {},
        "summary": [],
    }
    for m_name, m_type in MODELS_TO_TEST:
        all_results["models"][m_name] = {"type": m_type, "parameters": count_parameters(build_surrogate_model(m_type))}

    header = f"{'dataset':<28}{'model':<17}{'copy-prev':>10}{'trained':>10}{'after TTA':>10}{'err red. (mean±std)':>22}{'res drop':>10}"
    print("\nRelative L2 error, mean over held-out test trajectories\n" + header)

    for label, fname, category, physics in SHAPE_DATASETS:
        if only and os.path.splitext(fname)[0] not in only:
            continue
        if not os.path.exists(os.path.join(DATASETS_DIR, fname)):
            print(f"  [Skip] {fname} not found")
            continue
        states, extra, ctxs = load_dataset(fname, physics)
        N, T = states.shape[:2]
        train_ids, calib_ids, test_ids = split_trajectories(N)

        shape_results = {
            "name": label,
            "filename": fname,
            "category": category,
            "physics": PHYSICS_LABELS[physics],
            "grid": [states.shape[-2], states.shape[-1]],
            "trajectories": N,
            "timesteps": T,
            "train_trajectories": len(train_ids),
            "train_ids": train_ids,
            "calibration_ids": calib_ids,
            "test_ids": test_ids,
            "evaluations": {},
        }

        for m_name, m_type in MODELS_TO_TEST:
            ckpt_path = os.path.join(CKPT_DIR, f"{os.path.splitext(fname)[0]}_{m_type}.pt")
            if retrain or not os.path.exists(ckpt_path):
                train_and_save(m_type, states, extra, train_ids, ckpt_path, epochs)
            if train_only:
                continue
            model, ckpt = load_checkpoint(ckpt_path)

            per_traj = []
            for tid in test_ids:
                x_test, y_test = make_pairs(states, extra, [tid])
                per_traj.append(evaluate_with_tta(model, m_type, x_test, y_test, RESIDUALS[physics], ctxs[tid]))
            metrics = {k: float(np.mean([m[k] for m in per_traj])) for k in per_traj[0]}
            metrics["error_reduction_pct_std"] = float(np.std([m["error_reduction_pct"] for m in per_traj]))
            metrics["per_trajectory"] = [dict(m, trajectory=tid) for m, tid in zip(per_traj, test_ids)]
            metrics["checkpoint"] = os.path.relpath(ckpt_path, ROOT_DIR).replace("\\", "/")
            metrics["final_train_loss"] = ckpt["final_train_loss"]
            shape_results["evaluations"][m_name] = metrics
            all_results["summary"].append({
                "shape": label,
                "category": category,
                "model": m_name,
                "grid": f"{states.shape[-2]}x{states.shape[-1]}",
                "baseline_mse": metrics["baseline_mse"],
                "adapted_mse": metrics["adapted_mse"],
                "error_reduction_pct": metrics["error_reduction_pct"],
                "error_reduction_pct_std": metrics["error_reduction_pct_std"],
                "residual_drop_pct": metrics["residual_drop_pct"],
                "baseline_rel_l2": metrics["baseline_rel_l2"],
                "adapted_rel_l2": metrics["adapted_rel_l2"],
                "persistence_rel_l2": metrics["persistence_rel_l2"],
            })
            print(f"{label:<28}{m_name:<17}{metrics['persistence_rel_l2']:>10.2%}{metrics['baseline_rel_l2']:>10.2%}"
                  f"{metrics['adapted_rel_l2']:>10.2%}{metrics['error_reduction_pct']:>+12.1f}% ± {metrics['error_reduction_pct_std']:>5.1f}%"
                  f"{metrics['residual_drop_pct']:>+9.1f}%",
                  flush=True)

        all_results["shapes"][label] = shape_results

    if train_only:
        return all_results
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_JSON}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--retrain", action="store_true", help="Retrain models even if checkpoints exist")
    parser.add_argument("--only", nargs="+", default=None, help="Dataset file stems to run (e.g. heat_2d)")
    parser.add_argument("--train_only", action="store_true", help="Train/cache checkpoints without evaluating")
    args = parser.parse_args()
    run_full_2d_benchmark(epochs=args.epochs, retrain=args.retrain, only=args.only, train_only=args.train_only)
