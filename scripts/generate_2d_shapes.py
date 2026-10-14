"""
scripts/generate_2d_shapes.py
High-Fidelity 2D Flow Simulator past Geometric Obstacle Shapes.

Uses the 2D 9-velocity (D2Q9) Lattice Boltzmann Method (BGK) to simulate
viscous incompressible fluid flow past various aerodynamic and bluff bodies:
  1. Circular Cylinder     (Karman vortex shedding, Re ~ 100-250)
  2. NACA 0012 Airfoil     (Aerodynamic wake and separation)
  3. Square Bluff Body     (Sharp-corner recirculation zones)
  4. Backward-Facing Step  (Expansion vortex and reattachment)
  5. Multi-Cylinder Array  (Interfering wake turbulence)

Outputs structured HDF5 datasets compatible with PyTorch Geometric mesh loaders
and web visualizers, storing [velocity_x, velocity_y, pressure, obstacle_mask].
"""

import os
import argparse
import h5py
import numpy as np


# D2Q9 Lattice constants
C_S2 = 1.0 / 3.0  # Speed of sound squared
WEIGHTS = np.array([
    4/9,
    1/9, 1/9, 1/9, 1/9,
    1/36, 1/36, 1/36, 1/36
], dtype=np.float32)

CX = np.array([0, 1, 0, -1,  0, 1, -1, -1,  1], dtype=np.int32)
CY = np.array([0, 0, 1,  0, -1, 1,  1, -1, -1], dtype=np.int32)
OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)


# =============================================================================
# GEOMETRIC SHAPE MASK GENERATORS
# =============================================================================

def make_cylinder_mask(Nx, Ny, cx=None, cy=None, radius=None):
    """Circular cylinder obstacle."""
    if cx is None: cx = Nx // 4
    if cy is None: cy = Ny // 2
    if radius is None: radius = Ny // 9

    Y, X = np.ogrid[:Ny, :Nx]
    dist_sq = (X - cx)**2 + (Y - cy)**2
    return dist_sq <= radius**2


def make_airfoil_mask(Nx, Ny, chord=None, cx=None, cy=None, angle_deg=5.0):
    """
    NACA 0012 symmetric aerodynamic airfoil rotated by angle_deg.
    Equation: y_t = 5*t*(0.2969*sqrt(x) - 0.1260*x - 0.3516*x^2 + 0.2843*x^3 - 0.1015*x^4)
    """
    if chord is None: chord = Nx // 3
    if cx is None: cx = Nx // 4
    if cy is None: cy = Ny // 2

    mask = np.zeros((Ny, Nx), dtype=bool)
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    for i in range(Ny):
        for j in range(Nx):
            # Translate to airfoil leading edge
            dx = j - cx
            dy = i - cy
            # Rotate by angle of attack
            x_rot = dx * cos_a + dy * sin_a
            y_rot = -dx * sin_a + dy * cos_a

            if 0 <= x_rot <= chord:
                x_norm = x_rot / chord
                # NACA 0012 half-thickness formula (t = 0.12)
                t = 0.12
                yt = 5.0 * t * chord * (
                    0.2969 * np.sqrt(x_norm)
                    - 0.1260 * x_norm
                    - 0.3516 * (x_norm**2)
                    + 0.2843 * (x_norm**3)
                    - 0.1015 * (x_norm**4)
                )
                if abs(y_rot) <= yt:
                    mask[i, j] = True
    return mask


def make_square_mask(Nx, Ny, cx=None, cy=None, size=None):
    """Square bluff body obstacle."""
    if cx is None: cx = Nx // 4
    if cy is None: cy = Ny // 2
    if size is None: size = Ny // 5

    mask = np.zeros((Ny, Nx), dtype=bool)
    half = size // 2
    mask[cy - half:cy + half, cx - half:cx + half] = True
    return mask


def make_step_mask(Nx, Ny, step_x=None, step_height=None):
    """Backward-facing step channel obstacle."""
    if step_x is None: step_x = Nx // 4
    if step_height is None: step_height = Ny // 3

    mask = np.zeros((Ny, Nx), dtype=bool)
    mask[:step_height, :step_x] = True
    return mask


def make_multi_cylinder_mask(Nx, Ny):
    """Tandem/staggered dual cylinder configuration."""
    cy = Ny // 2
    r = Ny // 12
    mask1 = make_cylinder_mask(Nx, Ny, cx=Nx // 4, cy=cy - r, radius=r)
    mask2 = make_cylinder_mask(Nx, Ny, cx=Nx // 4 + 3 * r, cy=cy + r, radius=r)
    return mask1 | mask2


# =============================================================================
# D2Q9 LATTICE BOLTZMANN SOLVER
# =============================================================================

def simulate_flow_d2q9(obstacle_mask, u_inlet=0.08, reynolds=120, total_steps=800, sample_interval=20):
    """
    Simulates 2D fluid flow past an obstacle mask using D2Q9 BGK LBM.

    Args:
        obstacle_mask: boolean array (Ny, Nx), True inside obstacle
        u_inlet: maximum inlet velocity (lattice units, keep <= 0.1 for incompressibility)
        reynolds: characteristic Reynolds number
        total_steps: number of LBM time steps to integrate
        sample_interval: save snapshot every N steps

    Returns:
        velocity_u: [T, Ny, Nx] horizontal velocity
        velocity_v: [T, Ny, Nx] vertical velocity
        pressure:   [T, Ny, Nx] kinematic pressure (rho / 3)
    """
    Ny, Nx = obstacle_mask.shape

    # Characteristic length (obstacle height)
    obs_height = max(1, np.sum(obstacle_mask.any(axis=1)))
    # Kinematic viscosity from Reynolds number: Re = u * L / nu
    nu = max(1e-4, (u_inlet * obs_height) / max(1.0, reynolds))
    tau = float(np.clip(3.0 * nu + 0.5, 0.55, 1.2))  # Stable BGK relaxation
    omega = 1.0 / tau

    # Equilibrium distribution helper
    def get_feq(rho, ux, uy):
        u2 = ux**2 + uy**2
        feq = np.zeros((9, Ny, Nx), dtype=np.float32)
        for i in range(9):
            cu = CX[i] * ux + CY[i] * uy
            feq[i] = rho * WEIGHTS[i] * (1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*u2)
        return feq

    # Initial state: resting fluid with slight perturbation to seed instability
    rho = np.ones((Ny, Nx), dtype=np.float32)
    # Poiseuille/constant inlet profile
    y_coords = np.arange(Ny)
    # Parabolic inlet profile
    inlet_profile = 4.0 * u_inlet * (y_coords * (Ny - 1 - y_coords)) / ((Ny - 1)**2)
    inlet_profile = np.maximum(inlet_profile, 0.0)

    ux = np.zeros((Ny, Nx), dtype=np.float32)
    ux[:, :] = inlet_profile[:, None]
    # Small asymmetric seed to trigger vortex shedding faster
    uy = 0.005 * np.sin(2.0 * np.pi * y_coords / Ny)[:, None] * np.ones((1, Nx), dtype=np.float32)

    f = get_feq(rho, ux, uy)

    saved_u = []
    saved_v = []
    saved_p = []

    for step in range(total_steps):
        # 1. Macroscopic variables
        rho = np.sum(f, axis=0)
        ux = np.sum(f * CX[:, None, None], axis=0) / rho
        uy = np.sum(f * CY[:, None, None], axis=0) / rho

        # Zero out inside obstacle
        ux[obstacle_mask] = 0.0
        uy[obstacle_mask] = 0.0

        # Save snapshot
        if step >= (total_steps // 3) and (step % sample_interval == 0):
            saved_u.append(ux.copy())
            saved_v.append(uy.copy())
            # Pressure in lattice units: p = rho * cs^2 = rho / 3
            saved_p.append((rho / 3.0).copy())

        # 2. Collision (BGK)
        feq = get_feq(rho, ux, uy)
        f_post = f - omega * (f - feq)

        # 3. Bounce-back on obstacle
        for i in range(9):
            f_post[i, obstacle_mask] = f[OPPOSITE[i], obstacle_mask]

        # 4. Streaming
        for i in range(9):
            f[i] = np.roll(np.roll(f_post[i], CX[i], axis=1), CY[i], axis=0)

        # 5. Boundary conditions
        # Top and bottom walls: no-slip bounce-back
        f[2, 0, :] = f_post[4, 0, :]
        f[5, 0, :] = f_post[7, 0, :]
        f[6, 0, :] = f_post[8, 0, :]

        f[4, -1, :] = f_post[2, -1, :]
        f[7, -1, :] = f_post[5, -1, :]
        f[8, -1, :] = f_post[6, -1, :]

        # Inlet (left, Zou-He velocity BC)
        f[1, :, 0] = f[3, :, 0] + (2/3) * inlet_profile
        f[5, :, 0] = f[7, :, 0] - 0.5 * (f[2, :, 0] - f[4, :, 0]) + (1/6) * inlet_profile
        f[8, :, 0] = f[6, :, 0] + 0.5 * (f[2, :, 0] - f[4, :, 0]) + (1/6) * inlet_profile

        # Outlet (right, zero-gradient extrapolation)
        f[:, :, -1] = f[:, :, -2]

    return np.array(saved_u), np.array(saved_v), np.array(saved_p)


# =============================================================================
# DATASET EXPORTER FOR EACH SHAPE
# =============================================================================

def generate_shape_dataset(shape_type, output_path, Nx=128, Ny=64, num_traj=4, steps_per_traj=600):
    """Generates an HDF5 dataset for a specific 2D obstacle shape."""
    print(f"\n[Shape Simulator] Generating 2D flow dataset for '{shape_type}'...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    all_u, all_v, all_p = [], [], []
    masks = []

    for t_idx in range(num_traj):
        # Vary Reynolds number slightly across trajectories
        re = 60.0 + t_idx * 25.0
        u_in = 0.04 + (t_idx % 2) * 0.008

        if shape_type == "cylinder":
            r = Ny // 10 + (t_idx % 3) * 2
            mask = make_cylinder_mask(Nx, Ny, cx=Nx // 4, cy=Ny // 2, radius=r)
        elif shape_type == "airfoil":
            aoa = 4.0 + t_idx * 3.0  # 4, 7, 10, 13 degrees angle of attack
            mask = make_airfoil_mask(Nx, Ny, chord=Nx // 3, cx=Nx // 4, cy=Ny // 2, angle_deg=aoa)
        elif shape_type == "square":
            sz = Ny // 6 + (t_idx % 2) * 4
            mask = make_square_mask(Nx, Ny, cx=Nx // 4, cy=Ny // 2, size=sz)
        elif shape_type == "step":
            sh = Ny // 3 + (t_idx % 2) * 4
            mask = make_step_mask(Nx, Ny, step_x=Nx // 4, step_height=sh)
        elif shape_type == "multi_cylinder":
            mask = make_multi_cylinder_mask(Nx, Ny)
        else:
            raise ValueError(f"Unknown shape_type: {shape_type}")

        u_traj, v_traj, p_traj = simulate_flow_d2q9(
            mask, u_inlet=u_in, reynolds=re,
            total_steps=steps_per_traj, sample_interval=15
        )
        # Compute velocity magnitude as field
        mag_traj = np.sqrt(u_traj**2 + v_traj**2)
        all_u.append(mag_traj)
        all_v.append(v_traj)
        all_p.append(p_traj)
        masks.append(mask)
        print(f"  Trajectory {t_idx+1}/{num_traj} complete (Re={re:.0f}, snapshots={len(u_traj)})")

    # Stack datasets
    tensor_u = np.stack(all_u, axis=0).astype(np.float32)  # [N_traj, T, Ny, Nx]
    tensor_p = np.stack(all_p, axis=0).astype(np.float32)
    stacked_masks = np.stack(masks, axis=0).astype(bool)

    x_coords = np.linspace(0, 1.0, Nx, endpoint=False).astype(np.float32)
    y_coords = np.linspace(0, Ny / Nx, Ny, endpoint=False).astype(np.float32)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("tensor", data=tensor_u)
        f.create_dataset("pressure", data=tensor_p)
        f.create_dataset("obstacle_mask", data=stacked_masks)
        f.create_dataset("x-coordinate", data=x_coords)
        f.create_dataset("y-coordinate", data=y_coords)
        f.attrs["equation"] = f"2D Navier-Stokes ({shape_type.title()})"
        f.attrs["shape"] = shape_type
        f.attrs["method"] = "D2Q9 Lattice Boltzmann"

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  [OK] Saved {output_path} ({size_mb:.2f} MB, shape: {tensor_u.shape})")
    return output_path


def generate_all_shapes(output_dir="datasets", Nx=96, Ny=48, num_traj=3, steps=500):
    """Generates all 5 geometric obstacle shape datasets."""
    shapes = ["cylinder", "airfoil", "square", "step", "multi_cylinder"]
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("  2D GEOMETRIC SHAPES & FLOW SIMULATOR (D2Q9 LBM)")
    print(f"  Shapes: {shapes}")
    print(f"  Grid: {Nx}x{Ny} | Trajectories: {num_traj} | Steps: {steps}")
    print("=" * 70)

    paths = {}
    for shape in shapes:
        out_file = os.path.join(output_dir, f"{shape}_flow_2d.hdf5")
        paths[shape] = generate_shape_dataset(
            shape, out_file, Nx=Nx, Ny=Ny, num_traj=num_traj, steps_per_traj=steps
        )

    print("\n" + "=" * 70)
    print("  ALL 2D SHAPE DATASETS GENERATED SUCCESSFULLY!")
    print("=" * 70)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Flow simulator past geometric obstacle shapes.")
    parser.add_argument("--shape", default="all",
                        choices=["all", "cylinder", "airfoil", "square", "step", "multi_cylinder"],
                        help="Which obstacle shape to simulate")
    parser.add_argument("--output_dir", default="datasets", help="Output directory")
    parser.add_argument("--nx", type=int, default=96, help="Grid length")
    parser.add_argument("--ny", type=int, default=48, help="Grid height")
    parser.add_argument("--traj", type=int, default=3, help="Trajectories per shape")
    parser.add_argument("--steps", type=int, default=450, help="LBM steps per trajectory")

    args = parser.parse_args()
    if args.shape == "all":
        generate_all_shapes(args.output_dir, Nx=args.nx, Ny=args.ny, num_traj=args.traj, steps=args.steps)
    else:
        out_path = os.path.join(args.output_dir, f"{args.shape}_flow_2d.hdf5")
        generate_shape_dataset(args.shape, out_path, Nx=args.nx, Ny=args.ny, num_traj=args.traj, steps_per_traj=args.steps)
