"""
scripts/evaluate_gated_tta.py
Held-out, multi-trajectory benchmark of plain vs. residual-gated physics TTA
for 1D Burgers MeshGraphNet checkpoints.

Protocol
  - Calibration set: trajectories 9900-9919 (never used for training, never used for testing).
  - Test set:        trajectories 9980-9999 (the held-out set used by evaluate_gnn_accuracy.py).
  - 5 timesteps per trajectory (t = 10, 50, 90, 130, 170) -> 100 samples per condition.
  - Conditions: uniform mesh, jittered mesh, warped mesh, coarse (N=512) mesh.

Residual gate (label-free at test time)
  q = R(prediction) / R(copy previous state)
  R(copy) needs no labels, so q measures how much of the "physics violation" of
  doing nothing the model still leaves behind. On calibration data we measure
  the same ratio for the TRUE next state on the same mesh (q_floor = median over
  samples): that is the best a prediction can do under this discretised residual. TTA is applied
  only when q_base > kappa * q_floor, i.e. when the residual can still see an
  error above its own noise floor. kappa is chosen per mesh on the calibration
  set; kappa = inf (never adapt) is allowed, so the gate can shut TTA off where
  calibration shows it does not help.

Reported per condition: mean +/- std across the 20 test trajectories of the
per-trajectory error reduction, a two-sided paired Wilcoxon signed-rank p-value
(is the per-trajectory change different from zero?), and the fraction of
samples TTA made worse.

Usage:
    python scripts/evaluate_gated_tta.py --checkpoint real_burgers_model.pt --out scripts/gated_tta_v1.json
"""

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "files (5)"))

from model import MeshGraphNet
from tta import PhysicsResidualTTA
from evaluate_geometric_shift import generate_shifted_mesh, build_mesh_graph

DEFAULT_DATA = os.path.join(ROOT_DIR, "files (5)", "1D_Burgers_Sols_Nu0.001.hdf5")
CALIB_TRAJS = list(range(9900, 9920))
TEST_TRAJS = list(range(9980, 10000))
TIMESTEPS = [10, 50, 90, 130, 170]
CONDITIONS = ["uniform", "jitter", "warped", "coarse"]
KAPPA_GRID = [1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, float("inf")]  # inf = never adapt


def mesh_for(condition, x_orig):
    if condition == "uniform":
        return x_orig.copy()
    return generate_shifted_mesh(x_orig, shift_type=condition)[0]


def collect(model, tensor_ds, x_orig, trajs, tta_steps, tta_lr, scheme="implicit"):
    """Runs plain TTA on every (condition, traj, t) sample and records label-free and labelled stats."""
    records = []
    x_ext = np.append(x_orig, 1.0)
    for condition in CONDITIONS:
        x_mesh = mesh_for(condition, x_orig)
        coords = torch.tensor(x_mesh, dtype=torch.float32)
        engine = PhysicsResidualTTA(model, dx=float(np.mean(np.diff(x_mesh))), dt=0.01, nu=0.001, anchor_weight=0.1, scheme=scheme)
        for traj in trajs:
            traj_data = tensor_ds[traj]
            for t in TIMESTEPS:
                u_c = CubicSpline(x_ext, np.append(traj_data[t], traj_data[t][0]), bc_type="periodic")(x_mesh)
                u_n = CubicSpline(x_ext, np.append(traj_data[t + 1], traj_data[t + 1][0]), bc_type="periodic")(x_mesh)
                sample = build_mesh_graph(x_mesh, u_c, u_n)
                res = engine.adapt_sample(sample, tta_steps=tta_steps, tta_lr=tta_lr, x_coord_override=coords)
                with torch.no_grad():
                    u_curr = sample.x.squeeze(-1)
                    copy_res = engine.loss_fn(u_curr, u_curr, x_coord=coords)[0].item()
                    truth_res = engine.loss_fn(sample.y.squeeze(-1), u_curr, x_coord=coords)[0].item()
                records.append({
                    "condition": condition,
                    "traj": int(traj),
                    "t": t,
                    "base_mse": res["baseline_mse"],
                    "adapt_mse": res["adapted_mse"],
                    "true_ms": float(torch.mean(sample.y ** 2)),
                    "base_res": res["baseline_physics_loss"],
                    "adapt_res": res["adapted_physics_loss"],
                    "copy_res": copy_res,
                    "truth_res": truth_res,
                })
    return records


def gate_open(r, q_floor, kappa):
    return r["base_res"] / (r["copy_res"] + 1e-30) > kappa[r["condition"]] * q_floor[r["condition"]]


def gated_mse(r, q_floor, kappa):
    return r["adapt_mse"] if gate_open(r, q_floor, kappa) else r["base_mse"]


def wilcoxon_p(reductions):
    """Two-sided paired Wilcoxon signed-rank p-value that the per-trajectory change is zero."""
    nonzero = [r for r in reductions if abs(r) > 1e-9]
    if len(nonzero) < 2:
        return 1.0
    return float(wilcoxon(nonzero).pvalue)


def summarise(records, q_floor, kappa):
    """Per condition: mean/std over trajectories of per-trajectory error reduction (summed MSE)."""
    out = {}
    for condition in CONDITIONS:
        rows = [r for r in records if r["condition"] == condition]
        stats = {"plain": [], "gated": []}
        for traj in sorted({r["traj"] for r in rows}):
            tr = [r for r in rows if r["traj"] == traj]
            base = sum(r["base_mse"] for r in tr)
            stats["plain"].append(100 * (base - sum(r["adapt_mse"] for r in tr)) / base)
            stats["gated"].append(100 * (base - sum(gated_mse(r, q_floor, kappa) for r in tr)) / base)

        def rel_l2(key_fn):
            return float(np.mean([np.sqrt(key_fn(r) / r["true_ms"]) for r in rows]))

        base_mse = sum(r["base_mse"] for r in rows)
        out[condition] = {
            "samples": len(rows),
            "trajectories": len(stats["plain"]),
            "baseline_rel_l2": rel_l2(lambda r: r["base_mse"]),
            "plain_tta_rel_l2": rel_l2(lambda r: r["adapt_mse"]),
            "gated_tta_rel_l2": rel_l2(lambda r: gated_mse(r, q_floor, kappa)),
            "plain_reduction_pooled_pct": 100 * (base_mse - sum(r["adapt_mse"] for r in rows)) / base_mse,
            "gated_reduction_pooled_pct": 100 * (base_mse - sum(gated_mse(r, q_floor, kappa) for r in rows)) / base_mse,
            "plain_reduction_mean_pct": float(np.mean(stats["plain"])),
            "plain_reduction_std_pct": float(np.std(stats["plain"])),
            "gated_reduction_mean_pct": float(np.mean(stats["gated"])),
            "gated_reduction_std_pct": float(np.std(stats["gated"])),
            "plain_wilcoxon_p": wilcoxon_p(stats["plain"]),
            "gated_wilcoxon_p": wilcoxon_p(stats["gated"]),
            "plain_hurt_fraction": float(np.mean([r["adapt_mse"] > r["base_mse"] for r in rows])),
            "gated_hurt_fraction": float(np.mean([gated_mse(r, q_floor, kappa) > r["base_mse"] for r in rows])),
            "gate_open_fraction": float(np.mean([gate_open(r, q_floor, kappa) for r in rows])),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", default=DEFAULT_DATA)
    parser.add_argument("--tta_steps", type=int, default=12)
    parser.add_argument("--tta_lr", type=float, default=1e-4)
    parser.add_argument("--scheme", choices=["implicit", "crank_nicolson"], default="implicit",
                        help="Time discretisation of the Burgers residual")
    parser.add_argument("--out", required=True)
    parser.add_argument("--from_json", default=None, help="Re-summarise saved records instead of re-running TTA")
    args = parser.parse_args()

    if args.from_json:
        with open(args.from_json, encoding="utf-8") as f:
            prev = json.load(f)
        calib, test, start = prev["calibration_records"], prev["test_records"], time.time() - prev["runtime_s"]
    else:
        torch.manual_seed(0)
        model = MeshGraphNet.from_checkpoint(args.checkpoint, device="cpu")
        model.eval()
        start = time.time()
        with h5py.File(args.data_path, "r") as f:
            ds = f["tensor"]
            x_orig = f["x-coordinate"][:]
            calib = collect(model, ds, x_orig, CALIB_TRAJS, args.tta_steps, args.tta_lr, args.scheme)
            test = collect(model, ds, x_orig, TEST_TRAJS, args.tta_steps, args.tta_lr, args.scheme)

    # Noise floor of the residual on each mesh: q of the true next state, from calibration
    # trajectories interpolated onto that mesh (known geometry + known solutions, no test labels)
    q_floor = {c: float(np.median([r["truth_res"] / r["copy_res"] for r in calib if r["condition"] == c]))
               for c in CONDITIONS}

    # Pick kappa per mesh on calibration data only (maximise relative error reduction);
    # kappa = inf keeps the gate shut when TTA does not help on that mesh.
    def calib_score(condition, k):
        rows = [r for r in calib if r["condition"] == condition]
        kap = {condition: k}
        base = sum(r["base_mse"] / r["true_ms"] for r in rows)
        return (base - sum(gated_mse(r, q_floor, kap) / r["true_ms"] for r in rows)) / base
    kappa = {c: max(KAPPA_GRID, key=lambda k: calib_score(c, k)) for c in CONDITIONS}

    results = {
        "checkpoint": os.path.basename(args.checkpoint),
        "gate_rule": "adapt iff R(pred)/R(copy) > kappa[mesh] * q_floor[mesh]",
        "protocol": {
            "calibration_trajectories": [CALIB_TRAJS[0], CALIB_TRAJS[-1]],
            "test_trajectories": [TEST_TRAJS[0], TEST_TRAJS[-1]],
            "timesteps": TIMESTEPS,
            "tta_steps": args.tta_steps,
            "tta_lr": args.tta_lr,
            "anchor": 0.1,
            "residual_scheme": args.scheme,
        },
        "gate": {
            "q_floor": q_floor,
            "kappa": kappa,
            "calibration_score_by_kappa": {c: {str(k): calib_score(c, k) for k in KAPPA_GRID} for c in CONDITIONS},
        },
        "calibration": summarise(calib, q_floor, kappa),
        "test": summarise(test, q_floor, kappa),
        "calibration_records": calib,
        "test_records": test,
        "runtime_s": time.time() - start,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"{results['checkpoint']}: q_floor={ {c: round(v, 4) for c, v in q_floor.items()} }, kappa={kappa} (chosen per mesh on calibration set)")
    print(f"{'condition':<10}{'base relL2':>11}{'plain':>9}{'gated':>9}  {'plain red. (mean±std)':>22}  {'gated red. (mean±std)':>22}{'hurt plain/gated':>18}{'gate open':>10}{'p plain':>9}{'p gated':>9}")
    for c, m in results["test"].items():
        print(f"{c:<10}{m['baseline_rel_l2']:>10.2%}{m['plain_tta_rel_l2']:>9.2%}{m['gated_tta_rel_l2']:>9.2%}"
              f"  {m['plain_reduction_mean_pct']:>+10.1f}% ± {m['plain_reduction_std_pct']:>5.1f}%"
              f"  {m['gated_reduction_mean_pct']:>+10.1f}% ± {m['gated_reduction_std_pct']:>5.1f}%"
              f"{m['plain_hurt_fraction']:>10.0%} / {m['gated_hurt_fraction']:<5.0%}{m['gate_open_fraction']:>9.0%}{m['plain_wilcoxon_p']:>9.1e}{m['gated_wilcoxon_p']:>9.1e}")
    print(f"Saved {args.out} ({results['runtime_s']:.0f}s)")


if __name__ == "__main__":
    main()
