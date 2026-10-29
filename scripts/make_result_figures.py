"""
scripts/make_result_figures.py
Presentation figures for the headline results, drawn from the saved result JSONs
(no experiments are re-run). Writes PNGs to figures/.

Every figure follows the same layout: a title that states the takeaway, a
one-line subtitle saying what was measured, the plot, and the legend under the
plot. Orange is always plain TTA, blue is always gated TTA, grey is a baseline.

  fig1_original_model.png       1D, original GNN: plain vs gated TTA per mesh
  fig2_v2_model.png             1D, accurate v2 GNN: plain vs gated TTA per mesh
  fig3_per_trajectory.png       1D, original GNN: every held-out trajectory
  fig4_samples_made_worse.png   share of samples TTA made worse, grouped by setting
  fig5_2d_unet.png              2D, UNet-2D: plain vs gated TTA per dataset
  fig6_2d_fno.png               2D, FNO-2D: plain vs gated TTA per dataset
  fig7_surrogate_accuracy.png   one-step error of copy-previous, original GNN and v2 GNN

Usage:
    python scripts/make_result_figures.py
"""

import json
import os
import sys

import matplotlib
import matplotlib.transforms
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_gated_tta import CONDITIONS, gated_mse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.join(ROOT, "scripts")
OUT = os.path.join(ROOT, "figures")

# Reference palette (light mode)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
GATED = "#2a78d6"   # categorical slot 1, blue
PLAIN = "#eb6834"   # categorical slot 2, orange
NEUTRAL = "#b4b2ab"

MESHES = {"uniform": "Uniform mesh", "jitter": "Jittered mesh", "warped": "Warped mesh", "coarse": "Coarse mesh"}
DATASETS_2D = [
    ("Cylinder Flow", "Cylinder"), ("NACA 0012 Airfoil", "Airfoil"), ("Square Bluff Body", "Square"),
    ("Backward-Facing Step", "Step"), ("Multi-Cylinder Array", "Multi-\ncylinder"),
    ("2D Heat Diffusion", "Heat"), ("2D Advection Flow", "Advection"), ("2D Navier-Stokes Vorticity", "Vorticity"),
]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 12, "text.color": TEXT,
    "axes.edgecolor": MUTED, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.labelsize": 12, "ytick.labelsize": 11,
})


def load(name):
    with open(os.path.join(SCRIPTS, name), encoding="utf-8") as f:
        return json.load(f)


def new_figure(title, subtitle, size=(11, 6.2)):
    """Title and subtitle live in a reserved band above the axes, so data labels can never reach them."""
    fig = plt.figure(figsize=size)
    fig.text(0.04, 0.965, title, fontsize=17, fontweight="bold", va="top", color=TEXT)
    fig.text(0.04, 0.895, subtitle, fontsize=11.5, va="top", color=TEXT_2)
    ax = fig.add_axes([0.09, 0.2, 0.88, 0.6])
    return fig, ax


def legend_below(ax, handles=None, labels=None):
    kwargs = {} if handles is None else {"handles": handles, "labels": labels}
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=12, **kwargs)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)
    print(f"  wrote figures/{name}")


def grouped_bars(ax, categories, series, clip=None, label_fmt="{:+.0f}%", ylabel="", zero_label="0%",
                 label_size=11):
    """
    series: list of (name, color, values). Values below `clip` are drawn to the clip line, hatched,
    and labelled just below the cut. All other labels sit just outside the bar end.
    The y-range always leaves headroom for the labels.
    """
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(categories))
    all_vals = np.concatenate([np.asarray(v, dtype=float) for _, _, v in series])
    lo = min(0.0, max(all_vals.min(), clip) if clip is not None else all_vals.min())
    hi = max(0.0, all_vals.max())
    span = hi - lo if hi > lo else 1.0
    pad = 0.14 * span
    for i, (name, color, vals) in enumerate(series):
        xs = x + (i - (n - 1) / 2) * width
        vals = np.asarray(vals, dtype=float)
        shown = np.maximum(vals, clip) if clip is not None else vals
        bars = ax.bar(xs, shown, width * 0.92, color=color, label=name, edgecolor=SURFACE, linewidth=1.5)
        for b, v, s in zip(bars, vals, shown):
            if clip is not None and v < clip:
                # Hatch overlay (unlabelled, so the legend swatch stays solid) marks the cut-off bar
                ax.bar(b.get_x() + b.get_width() / 2, s, b.get_width(), facecolor="none", edgecolor="white",
                       hatch="///", linewidth=0)
                ax.text(b.get_x() + b.get_width() / 2, s - 0.02 * span, label_fmt.format(v),
                        ha="center", va="top", fontsize=label_size, color=TEXT, fontweight="bold")
            elif abs(v) < 0.05:
                ax.text(b.get_x() + b.get_width() / 2, 0.02 * span, zero_label, ha="center", va="bottom",
                        fontsize=label_size, color=TEXT, fontweight="bold")
            else:
                off = 0.02 * span
                ax.text(b.get_x() + b.get_width() / 2, s + (off if v >= 0 else -off), label_fmt.format(v),
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=label_size, color=TEXT,
                        fontweight="bold")
    ax.axhline(0, color=TEXT_2, linewidth=1.2)
    ax.set_xticks(x, categories)
    ax.set_xlim(-0.6, len(categories) - 0.4)
    ax.set_ylim(lo - (pad if lo < 0 else 0), hi + pad)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", length=0, pad=8)


def fig1_original(orig):
    t = orig["test"]
    fig, ax = new_figure(
        "Gated TTA cuts error by 18–23% on every mesh",
        "Original GNN · 1D Burgers · 20 held-out trajectories · every gated gain is significant (Wilcoxon p < 0.0001)")
    grouped_bars(ax, list(MESHES.values()), [
        ("Plain TTA", PLAIN, [t[c]["plain_reduction_mean_pct"] for c in CONDITIONS]),
        ("Gated TTA", GATED, [t[c]["gated_reduction_mean_pct"] for c in CONDITIONS]),
    ], ylabel="Error reduction vs. no TTA (%)\nhigher is better")
    legend_below(ax)
    save(fig, "fig1_original_model.png")


def fig2_v2(v2):
    t = v2["test"]
    fig, ax = new_figure(
        "On an accurate model, the gate prevents damage and keeps the gains",
        "Improved v2 GNN (2.35% error) · 1D Burgers · 20 held-out trajectories · bars below −100% are cut off")
    grouped_bars(ax, list(MESHES.values()), [
        ("Plain TTA", PLAIN, [t[c]["plain_reduction_mean_pct"] for c in CONDITIONS]),
        ("Gated TTA", GATED, [t[c]["gated_reduction_mean_pct"] for c in CONDITIONS]),
    ], clip=-100, ylabel="Error reduction vs. no TTA (%)\nhigher is better", zero_label="0%\n(gate shut)")
    legend_below(ax)
    save(fig, "fig2_v2_model.png")


def per_traj(data, key):
    q_floor, kappa = data["gate"]["q_floor"], data["gate"]["kappa"]
    out = {}
    for c in CONDITIONS:
        rows = [r for r in data["test_records"] if r["condition"] == c]
        vals = []
        for t in sorted({r["traj"] for r in rows}):
            tr = [r for r in rows if r["traj"] == t]
            base = sum(r["base_mse"] for r in tr)
            new = sum((r["adapt_mse"] if key == "plain" else gated_mse(r, q_floor, kappa)) for r in tr)
            vals.append(100 * (base - new) / base)
        out[c] = np.array(vals)
    return out


def fig3_per_traj(orig):
    plain, gated = per_traj(orig, "plain"), per_traj(orig, "gated")
    worse_p = sum(int((plain[c] < 0).sum()) for c in CONDITIONS)
    worse_g = sum(int((gated[c] < 0).sum()) for c in CONDITIONS)
    fig, ax = new_figure(
        "The gate removes the trajectories where TTA backfires",
        f"Original GNN · each dot is one of 20 held-out trajectories, black line = mean · "
        f"trajectories made worse: plain {worse_p}/80, gated {worse_g}/80")
    rng = np.random.default_rng(0)
    for i, c in enumerate(CONDITIONS):
        for off, vals, color, name in [(-0.2, plain[c], PLAIN, "Plain TTA"), (0.2, gated[c], GATED, "Gated TTA")]:
            jit = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(i + off + jit, vals, s=46, color=color, edgecolor=SURFACE, linewidth=1.2,
                       label=name if i == 0 else None, zorder=3)
            ax.plot([i + off - 0.13, i + off + 0.13], [vals.mean()] * 2, color=TEXT, linewidth=2.5, zorder=4)
    ax.axhline(0, color=TEXT_2, linewidth=1.2)
    ax.set_xticks(range(len(CONDITIONS)), list(MESHES.values()))
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_xlim(-0.6, len(CONDITIONS) - 0.4)
    ax.set_ylim(-85, 75)
    ax.set_ylabel("Error reduction per trajectory (%)\nhigher is better")
    legend_below(ax)
    save(fig, "fig3_per_trajectory.png")


def fig4_worse(orig, v2, d2):
    obstacle = [l for l, _ in DATASETS_2D[:5]]
    exact = [l for l, _ in DATASETS_2D[5:]]

    def mean_2d(labels, key):
        return 100 * np.mean([d2["datasets"][l][m][key] for l in labels for m in d2["datasets"][l]])

    groups = ["1D original GNN\n(4 meshes)", "1D v2 GNN\n(4 meshes)", "2D obstacle flows\n(5 shapes × 2 models)",
              "2D heat, advection,\nvorticity (3 × 2 models)"]
    plain = [100 * np.mean([orig["test"][c]["plain_hurt_fraction"] for c in CONDITIONS]),
             100 * np.mean([v2["test"][c]["plain_hurt_fraction"] for c in CONDITIONS]),
             mean_2d(obstacle, "plain_hurt_fraction"), mean_2d(exact, "plain_hurt_fraction")]
    gated = [100 * np.mean([orig["test"][c]["gated_hurt_fraction"] for c in CONDITIONS]),
             100 * np.mean([v2["test"][c]["gated_hurt_fraction"] for c in CONDITIONS]),
             mean_2d(obstacle, "gated_hurt_fraction"), mean_2d(exact, "gated_hurt_fraction")]
    fig, ax = new_figure(
        "Plain TTA often makes predictions worse; gated TTA almost never does",
        "Average share of held-out test samples whose error went up after TTA (lower is better)")
    grouped_bars(ax, groups, [("Plain TTA", PLAIN, plain), ("Gated TTA", GATED, gated)],
                 label_fmt="{:.0f}%", ylabel="Samples made worse (%)\nlower is better")
    ax.tick_params(axis="x", labelsize=11)
    legend_below(ax)
    save(fig, "fig4_samples_made_worse.png")


def fig_2d(d2, model, short_name, name):
    plain = [d2["datasets"][l][model]["plain_reduction_mean_pct"] for l, _ in DATASETS_2D]
    gated = [d2["datasets"][l][model]["gated_reduction_mean_pct"] for l, _ in DATASETS_2D]
    best = max(gated)
    fig, ax = new_figure(
        f"2D ({short_name}): gains up to {best:+.0f}%, and no dataset made worse",
        "3 held-out trajectories per dataset" + (" · hatched bars are cut off at −100%" if min(plain) < -100 else ""),
        size=(12, 6.2))
    grouped_bars(ax, [s for _, s in DATASETS_2D], [("Plain TTA", PLAIN, plain), ("Gated TTA", GATED, gated)],
                 clip=-100, ylabel="Error reduction vs. no TTA (%)\nhigher is better", label_size=10)
    ax.axvspan(-0.6, 4.5, color=GRID, alpha=0.45, zorder=0)
    ax.tick_params(axis="x", labelsize=11)
    # Group labels sit in the band between the subtitle and the plot, clear of every bar label
    band = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(2, 1.03, "Obstacle flows: approximate physics check → gate stays shut", transform=band,
            ha="center", va="bottom", fontsize=11, color=TEXT_2, fontweight="bold")
    ax.text(6, 1.03, "Exact physics check → gate opens", transform=band,
            ha="center", va="bottom", fontsize=11, color=TEXT_2, fontweight="bold")
    legend_below(ax)
    save(fig, name)


def fig7_accuracy():
    # Held-out trajectories 9980-9999, from scripts/evaluate_gnn_accuracy.py (see BENCHMARK_RESULTS_SUMMARY.md)
    names = ["Copy previous\ntimestep (no model)", "Original GNN", "Improved v2 GNN"]
    vals = [9.76, 12.83, 2.35]
    fig, ax = new_figure(
        "The improved GNN is 4× more accurate than copying the last step",
        "One-step prediction error on 20 held-out 1D Burgers trajectories (lower is better)", size=(10, 6.2))
    bars = ax.bar(range(3), vals, 0.55, color=[NEUTRAL, NEUTRAL, GATED], edgecolor=SURFACE)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.2f}%", ha="center", va="bottom", fontsize=13,
                color=TEXT, fontweight="bold")
    ax.set_xticks(range(3), names)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_ylim(0, 15.5)
    ax.set_ylabel("Relative L2 error (%)\nlower is better")
    save(fig, "fig7_surrogate_accuracy.png")


def main():
    orig = load("gated_tta_original_crank_nicolson.json")
    v2 = load("gated_tta_v2_crank_nicolson.json")
    d2 = load("gated_tta_2d_results.json")
    for old in os.listdir(OUT) if os.path.isdir(OUT) else []:
        if old.endswith(".png"):
            os.remove(os.path.join(OUT, old))
    fig1_original(orig)
    fig2_v2(v2)
    fig3_per_traj(orig)
    fig4_worse(orig, v2, d2)
    fig_2d(d2, "Physics UNet-2D", "UNet-2D", "fig5_2d_unet.png")
    fig_2d(d2, "FNO-2D", "FNO-2D", "fig6_2d_fno.png")
    fig7_accuracy()


if __name__ == "__main__":
    main()
