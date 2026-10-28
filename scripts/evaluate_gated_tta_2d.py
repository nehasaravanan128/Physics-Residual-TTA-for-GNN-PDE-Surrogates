"""
scripts/evaluate_gated_tta_2d.py
Per-frame plain vs. residual-gated physics TTA for the trained 2D surrogates
(checkpoints and trajectory split from evaluate_2d_benchmarks.py).

Gate (same rule as scripts/evaluate_gated_tta.py for 1D):
  q = R(prediction) / R(copy previous state)        -- label-free
  q_floor = median q of the TRUE next state over the calibration trajectory
  apply TTA on a frame only if q_base > kappa * q_floor

kappa is chosen per dataset x model on the calibration trajectory, which the
model was not trained on and which is not a test trajectory. No test labels
are used. kappa = inf (never adapt) is allowed. Pass --kappa to use one fixed
value everywhere instead.

Each test frame is adapted independently. Reported: mean +/- std over frames,
mean +/- std of the per-trajectory pooled reduction over the test trajectories,
the fraction of frames made worse, and a two-sided Wilcoxon signed-rank p-value
over frames (frames of one trajectory are correlated, so treat p as indicative).

Usage:
    python scripts/evaluate_gated_tta_2d.py
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon

from evaluate_2d_benchmarks import (
    CKPT_DIR, MODELS_TO_TEST, RESIDUALS, SHAPE_DATASETS, DATASETS_DIR,
    load_checkpoint, load_dataset, make_pairs, run_tta, split_trajectories,
)

KAPPA_GRID = [1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, float("inf")]
OUTPUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gated_tta_2d_results.json")


def frame_residual(residual_fn, x, y, ctx, n_state):
    with torch.no_grad():
        return residual_fn(x[:, :n_state], y, ctx).item()


def wilcoxon_p(reductions):
    nonzero = [r for r in reductions if abs(r) > 1e-9]
    if len(nonzero) < 2:
        return 1.0
    return float(wilcoxon(nonzero).pvalue)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kappa", type=float, default=None, help="Fixed kappa (default: calibrate per dataset)")
    args = parser.parse_args()
    torch.manual_seed(0)

    results = {"kappa": args.kappa if args.kappa is not None else "calibrated per dataset x model on the calibration trajectory",
               "datasets": {}}
    print(f"{'dataset':<28}{'model':<17}{'kappa':>6}{'plain (frames)':>18}{'gated (frames)':>18}"
          f"{'gated (per traj)':>19}{'hurt plain/gated':>18}{'open':>6}{'p plain':>9}{'p gated':>9}")

    for label, fname, _, physics in SHAPE_DATASETS:
        if not os.path.exists(os.path.join(DATASETS_DIR, fname)):
            continue
        states, extra, ctxs = load_dataset(fname, physics)
        _, calib_ids, test_ids = split_trajectories(states.shape[0])
        residual_fn = RESIDUALS[physics]
        results["datasets"][label] = {}

        for m_name, m_type in MODELS_TO_TEST:
            model, _ = load_checkpoint(os.path.join(CKPT_DIR, f"{os.path.splitext(fname)[0]}_{m_type}.pt"))
            n_state = model.n_state

            def run_frames(traj):
                x_all, y_all = make_pairs(states, extra, [traj])
                rows = []
                for i in range(x_all.shape[0]):
                    xi, yi = x_all[i:i + 1], y_all[i:i + 1]
                    y_base, y_adapt, base_res, _ = run_tta(model, m_type, xi, residual_fn, ctxs[traj])
                    copy_res = frame_residual(residual_fn, xi, xi[:, :n_state], ctxs[traj], n_state)
                    rows.append({
                        "trajectory": traj,
                        "base_mse": F.mse_loss(y_base, yi).item(),
                        "adapt_mse": F.mse_loss(y_adapt, yi).item(),
                        "q_base": base_res / (copy_res + 1e-30),
                        "q_truth": frame_residual(residual_fn, xi, yi, ctxs[traj], n_state) / (copy_res + 1e-30),
                    })
                return rows

            cal = [r for t in calib_ids for r in run_frames(t)]
            q_floor = float(np.median([r["q_truth"] for r in cal]))

            def gated(rows, k):
                return [r["adapt_mse"] if r["q_base"] > k * q_floor else r["base_mse"] for r in rows]

            if args.kappa is None:
                kappa = max(KAPPA_GRID, key=lambda k: -sum(g / r["base_mse"] for g, r in zip(gated(cal, k), cal)))
            else:
                kappa = args.kappa

            frames = [r for t in test_ids for r in run_frames(t)]
            for f, g in zip(frames, gated(frames, kappa)):
                f["gated_mse"] = g
                f["gate_open"] = bool(f["q_base"] > kappa * q_floor)

            def reduction(key, rows=frames):
                return [100 * (f["base_mse"] - f[key]) / (f["base_mse"] + 1e-30) for f in rows]

            def per_traj(key):
                out = []
                for t in test_ids:
                    rows = [f for f in frames if f["trajectory"] == t]
                    base = sum(f["base_mse"] for f in rows)
                    out.append(100 * (base - sum(f[key] for f in rows)) / base)
                return out

            m = {
                "q_floor": q_floor,
                "kappa": kappa,
                "calibration_trajectories": calib_ids,
                "test_trajectories": test_ids,
                "frames": len(frames),
                "plain_reduction_mean_pct": float(np.mean(reduction("adapt_mse"))),
                "plain_reduction_std_pct": float(np.std(reduction("adapt_mse"))),
                "gated_reduction_mean_pct": float(np.mean(reduction("gated_mse"))),
                "gated_reduction_std_pct": float(np.std(reduction("gated_mse"))),
                "plain_per_traj_mean_pct": float(np.mean(per_traj("adapt_mse"))),
                "plain_per_traj_std_pct": float(np.std(per_traj("adapt_mse"))),
                "gated_per_traj_mean_pct": float(np.mean(per_traj("gated_mse"))),
                "gated_per_traj_std_pct": float(np.std(per_traj("gated_mse"))),
                "plain_hurt_fraction": float(np.mean([f["adapt_mse"] > f["base_mse"] for f in frames])),
                "gated_hurt_fraction": float(np.mean([f["gated_mse"] > f["base_mse"] for f in frames])),
                "gate_open_fraction": float(np.mean([f["gate_open"] for f in frames])),
                "plain_wilcoxon_p": wilcoxon_p(reduction("adapt_mse")),
                "gated_wilcoxon_p": wilcoxon_p(reduction("gated_mse")),
                "per_frame": frames,
            }
            results["datasets"][label][m_name] = m
            print(f"{label:<28}{m_name:<17}{kappa:>6.2f}"
                  f"{m['plain_reduction_mean_pct']:>+9.1f}±{m['plain_reduction_std_pct']:<6.1f}%"
                  f"{m['gated_reduction_mean_pct']:>+11.1f}±{m['gated_reduction_std_pct']:<6.1f}%"
                  f"{m['gated_per_traj_mean_pct']:>+11.1f}±{m['gated_per_traj_std_pct']:<6.1f}%"
                  f"{m['plain_hurt_fraction']:>9.0%} / {m['gated_hurt_fraction']:<5.0%}{m['gate_open_fraction']:>6.0%}"
                  f"{m['plain_wilcoxon_p']:>9.1e}{m['gated_wilcoxon_p']:>9.1e}",
                  flush=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
