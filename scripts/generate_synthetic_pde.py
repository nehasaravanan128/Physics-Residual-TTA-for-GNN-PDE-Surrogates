"""
scripts/generate_synthetic_pde.py
Multi-Equation Synthetic PDE Dataset Generator.

Generates compact, high-precision synthetic datasets for:
  1. 1D Burgers' Equation    (pseudo-spectral Fourier solver)
  2. 2D Heat/Diffusion Eq.   (spectral solver on square domain)
  3. 2D Advection Equation   (spectral solver with rotating velocity field)
  4. 2D Burgers' Equation    (spectral solver, coupled system)
  5. 2D Navier-Stokes        (vorticity-stream function, spectral)

All outputs follow PDEBench-compatible HDF5 format.
"""

import argparse
import os
import h5py
import numpy as np


# =============================================================================
# 1D VISCOUS BURGERS' EQUATION (pseudo-spectral Fourier)
# =============================================================================

def solve_burgers_1d(u0, nu, dx, dt, steps, n_sub=20):
    """
    Solves 1D viscous Burgers' equation using pseudo-spectral Fourier method:
      du/dt + 0.5 * d(u^2)/dx = nu * d^2u/dx^2
    with 2/3 de-aliasing and substepping for exact stability.
    """
    N = len(u0)
    k = 2 * np.pi * np.fft.rfftfreq(N, d=dx)
    k_max = k[-1] * (2.0 / 3.0)
    filter_mask = (k <= k_max).astype(float)

    dt_sub = dt / n_sub
    diff_factor = np.exp(-nu * (k ** 2) * dt_sub)

    u = u0.copy()
    trajectory = [u.copy()]

    for _ in range(steps):
        for _ in range(n_sub):
            u_hat = np.fft.rfft(u)
            u_sq = u ** 2
            flux_hat = filter_mask * np.fft.rfft(u_sq)
            adv_hat = -0.5 * 1j * k * flux_hat
            u_hat_next = (u_hat + dt_sub * adv_hat) * diff_factor
            u = np.fft.irfft(u_hat_next, n=N)
        trajectory.append(u.copy())

    return np.array(trajectory)


def generate_burgers_1d(output_path, num_traj=25, num_steps=50, grid=256, nu=0.001):
    print(f"[1D Burgers] Generating {num_traj} trajectories on N={grid} grid...")
    x = np.linspace(0, 1.0, grid, endpoint=False)
    dx = x[1] - x[0]
    dt = 0.01

    all_tensors = []
    np.random.seed(42)

    for i in range(num_traj):
        num_modes = np.random.randint(1, 4)
        u0 = np.zeros_like(x)
        for m in range(1, num_modes + 1):
            phase = np.random.uniform(0, 2 * np.pi)
            amp = np.random.uniform(0.3, 1.0) / m
            u0 += amp * np.sin(2 * np.pi * m * x + phase)
        u0 = u0 / (np.max(np.abs(u0)) + 1e-6)

        traj = solve_burgers_1d(u0, nu=nu, dx=dx, dt=dt, steps=num_steps - 1)
        all_tensors.append(traj)

    tensor_data = np.stack(all_tensors, axis=0)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("tensor", data=tensor_data.astype(np.float32))
        f.create_dataset("x-coordinate", data=x.astype(np.float32))
        f.attrs["equation"] = "1D Burgers"
        f.attrs["nu"] = nu
    print(f"  [OK] Saved {output_path} ({os.path.getsize(output_path)/1024/1024:.2f} MB)")


# =============================================================================
# 2D HEAT / DIFFUSION EQUATION (spectral)
# =============================================================================

def solve_heat_2d(u0, kappa, Lx, Ly, dt, steps, n_sub=10):
    """
    Solves 2D heat equation: du/dt = kappa * (d2u/dx2 + d2u/dy2)
    with periodic BCs using spectral method.
    """
    Nx, Ny = u0.shape
    kx = 2 * np.pi / Lx * np.fft.fftfreq(Nx, d=1.0/Nx)
    ky = 2 * np.pi / Ly * np.fft.rfftfreq(Ny, d=1.0/Ny)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2

    dt_sub = dt / n_sub
    diff = np.exp(-kappa * K2 * dt_sub)

    u = u0.copy()
    trajectory = [u.copy()]

    for _ in range(steps):
        for _ in range(n_sub):
            u_hat = np.fft.rfft2(u)
            u_hat *= diff
            u = np.fft.irfft2(u_hat, s=(Nx, Ny))
        trajectory.append(u.copy())

    return np.array(trajectory)


def generate_heat_2d(output_path, num_traj=15, num_steps=40, grid=64, kappa=0.01):
    print(f"[2D Heat] Generating {num_traj} trajectories on {grid}x{grid} grid...")
    Lx, Ly = 1.0, 1.0
    x = np.linspace(0, Lx, grid, endpoint=False)
    y = np.linspace(0, Ly, grid, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    dt = 0.005

    all_tensors = []
    np.random.seed(123)

    for i in range(num_traj):
        u0 = np.zeros((grid, grid))
        n_modes = np.random.randint(2, 6)
        for _ in range(n_modes):
            mx = np.random.randint(1, 5)
            my = np.random.randint(1, 5)
            amp = np.random.uniform(0.3, 1.0) / (mx + my)
            phase = np.random.uniform(0, 2 * np.pi)
            u0 += amp * np.sin(2*np.pi*mx*X/Lx + 2*np.pi*my*Y/Ly + phase)

        traj = solve_heat_2d(u0, kappa=kappa, Lx=Lx, Ly=Ly, dt=dt, steps=num_steps - 1)
        all_tensors.append(traj)

    tensor_data = np.stack(all_tensors, axis=0)  # [N, T, Nx, Ny]
    with h5py.File(output_path, "w") as f:
        f.create_dataset("tensor", data=tensor_data.astype(np.float32))
        f.create_dataset("x-coordinate", data=x.astype(np.float32))
        f.create_dataset("y-coordinate", data=y.astype(np.float32))
        f.attrs["equation"] = "2D Heat/Diffusion"
        f.attrs["kappa"] = kappa
    print(f"  [OK] Saved {output_path} ({os.path.getsize(output_path)/1024/1024:.2f} MB)")


# =============================================================================
# 2D ADVECTION EQUATION (spectral with rotating velocity)
# =============================================================================

def solve_advection_2d(u0, vx, vy, Lx, Ly, dt, steps, n_sub=10):
    """
    Solves 2D advection: du/dt + vx*du/dx + vy*du/dy = 0
    with constant velocity field (vx, vy) and periodic BCs.
    """
    Nx, Ny = u0.shape
    kx = 2 * np.pi / Lx * np.fft.fftfreq(Nx, d=1.0/Nx)
    ky = 2 * np.pi / Ly * np.fft.rfftfreq(Ny, d=1.0/Ny)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    dt_sub = dt / n_sub
    phase_shift = np.exp(-1j * (vx * KX + vy * KY) * dt_sub)

    u = u0.copy()
    trajectory = [u.copy()]

    for _ in range(steps):
        for _ in range(n_sub):
            u_hat = np.fft.rfft2(u)
            u_hat *= phase_shift
            u = np.fft.irfft2(u_hat, s=(Nx, Ny))
        trajectory.append(u.copy())

    return np.array(trajectory)


def generate_advection_2d(output_path, num_traj=15, num_steps=40, grid=64):
    print(f"[2D Advection] Generating {num_traj} trajectories on {grid}x{grid} grid...")
    Lx, Ly = 1.0, 1.0
    x = np.linspace(0, Lx, grid, endpoint=False)
    y = np.linspace(0, Ly, grid, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    dt = 0.005

    all_tensors, all_vel = [], []
    np.random.seed(456)

    for i in range(num_traj):
        # Random smooth initial Gaussian blobs
        u0 = np.zeros((grid, grid))
        n_blobs = np.random.randint(1, 4)
        for _ in range(n_blobs):
            cx, cy = np.random.uniform(0.2, 0.8, size=2)
            sigma = np.random.uniform(0.05, 0.15)
            amp = np.random.uniform(0.5, 1.5)
            u0 += amp * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))

        # Random constant velocity field
        angle = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(0.3, 1.5)
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)

        traj = solve_advection_2d(u0, vx, vy, Lx, Ly, dt, steps=num_steps - 1)
        all_tensors.append(traj)
        all_vel.append([vx, vy])

    tensor_data = np.stack(all_tensors, axis=0)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("tensor", data=tensor_data.astype(np.float32))
        f.create_dataset("x-coordinate", data=x.astype(np.float32))
        f.create_dataset("y-coordinate", data=y.astype(np.float32))
        f.create_dataset("velocity", data=np.array(all_vel, dtype=np.float32))  # [N, 2] constant (vx, vy)
        f.attrs["equation"] = "2D Advection"
    print(f"  [OK] Saved {output_path} ({os.path.getsize(output_path)/1024/1024:.2f} MB)")


# =============================================================================
# 2D NAVIER-STOKES (vorticity-stream function, spectral)
# =============================================================================

def solve_navier_stokes_2d(w0, nu, Lx, Ly, dt, steps, n_sub=10):
    """
    Solves 2D incompressible Navier-Stokes in vorticity-stream function form:
      dw/dt + u*dw/dx + v*dw/dy = nu*(d2w/dx2 + d2w/dy2)
    where w = curl(u) is the vorticity, and velocities are recovered from
    the stream function: u = dpsi/dy, v = -dpsi/dx, Laplacian(psi) = -w.
    """
    Nx, Ny = w0.shape
    kx = 2 * np.pi / Lx * np.fft.fftfreq(Nx, d=1.0/Nx)
    ky = 2 * np.pi / Ly * np.fft.rfftfreq(Ny, d=1.0/Ny)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0  # avoid division by zero for mean mode

    # De-aliasing 2/3 rule
    k_max_x = np.max(np.abs(kx)) * (2.0 / 3.0)
    k_max_y = ky[-1] * (2.0 / 3.0)
    filter_mask = ((np.abs(KX) <= k_max_x) & (np.abs(KY) <= k_max_y)).astype(float)

    dt_sub = dt / n_sub
    diff = np.exp(-nu * K2 * dt_sub)

    w = w0.copy()
    trajectory = [w.copy()]

    for _ in range(steps):
        for _ in range(n_sub):
            w_hat = np.fft.rfft2(w)
            # Stream function psi: Laplacian(psi) = -w => psi_hat = w_hat / K2
            psi_hat = w_hat / K2
            # Velocity: u = dpsi/dy, v = -dpsi/dx
            u = np.fft.irfft2(1j * KY * psi_hat, s=(Nx, Ny))
            v = np.fft.irfft2(-1j * KX * psi_hat, s=(Nx, Ny))
            # Vorticity gradients
            dwdx = np.fft.irfft2(1j * KX * filter_mask * w_hat, s=(Nx, Ny))
            dwdy = np.fft.irfft2(1j * KY * filter_mask * w_hat, s=(Nx, Ny))
            # Nonlinear advection
            nlw = u * dwdx + v * dwdy
            nlw_hat = filter_mask * np.fft.rfft2(nlw)
            # Time step with integrating factor
            w_hat_next = (w_hat - dt_sub * nlw_hat) * diff
            w = np.fft.irfft2(w_hat_next, s=(Nx, Ny))
        trajectory.append(w.copy())

    return np.array(trajectory)


def generate_navier_stokes_2d(output_path, num_traj=10, num_steps=30, grid=64, nu=0.001):
    print(f"[2D Navier-Stokes] Generating {num_traj} trajectories on {grid}x{grid} grid...")
    Lx, Ly = 2 * np.pi, 2 * np.pi
    x = np.linspace(0, Lx, grid, endpoint=False)
    y = np.linspace(0, Ly, grid, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    dt = 0.05

    all_tensors = []
    np.random.seed(789)

    for i in range(num_traj):
        # Taylor-Green vortex with random perturbations
        w0 = np.zeros((grid, grid))
        n_modes = np.random.randint(2, 5)
        for _ in range(n_modes):
            mx = np.random.randint(1, 4)
            my = np.random.randint(1, 4)
            amp = np.random.uniform(0.5, 2.0) / (mx + my)
            phase = np.random.uniform(0, 2 * np.pi)
            w0 += amp * np.cos(mx * X + my * Y + phase)

        traj = solve_navier_stokes_2d(w0, nu=nu, Lx=Lx, Ly=Ly, dt=dt, steps=num_steps - 1)
        all_tensors.append(traj)

    tensor_data = np.stack(all_tensors, axis=0)  # [N, T, Nx, Ny]
    with h5py.File(output_path, "w") as f:
        f.create_dataset("tensor", data=tensor_data.astype(np.float32))
        f.create_dataset("x-coordinate", data=x.astype(np.float32))
        f.create_dataset("y-coordinate", data=y.astype(np.float32))
        f.attrs["equation"] = "2D Navier-Stokes (Vorticity)"
        f.attrs["nu"] = nu
    print(f"  [OK] Saved {output_path} ({os.path.getsize(output_path)/1024/1024:.2f} MB)")


# =============================================================================
# 2D BURGERS' EQUATION (coupled spectral)
# =============================================================================

def solve_burgers_2d(u0, v0, nu, Lx, Ly, dt, steps, n_sub=10):
    """
    Solves coupled 2D Burgers' system:
      du/dt + u*du/dx + v*du/dy = nu*(d2u/dx2 + d2u/dy2)
      dv/dt + u*dv/dx + v*dv/dy = nu*(d2v/dx2 + d2v/dy2)
    """
    Nx, Ny = u0.shape
    kx = 2 * np.pi / Lx * np.fft.fftfreq(Nx, d=1.0/Nx)
    ky = 2 * np.pi / Ly * np.fft.rfftfreq(Ny, d=1.0/Ny)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2

    k_max_x = np.max(np.abs(kx)) * (2.0 / 3.0)
    k_max_y = ky[-1] * (2.0 / 3.0)
    filt = ((np.abs(KX) <= k_max_x) & (np.abs(KY) <= k_max_y)).astype(float)

    dt_sub = dt / n_sub
    diff = np.exp(-nu * K2 * dt_sub)

    u, v = u0.copy(), v0.copy()
    # Store velocity magnitude as scalar field
    trajectory = [np.sqrt(u**2 + v**2).copy()]

    for _ in range(steps):
        for _ in range(n_sub):
            u_hat = np.fft.rfft2(u)
            v_hat = np.fft.rfft2(v)

            dudx = np.fft.irfft2(1j * KX * filt * u_hat, s=(Nx, Ny))
            dudy = np.fft.irfft2(1j * KY * filt * u_hat, s=(Nx, Ny))
            dvdx = np.fft.irfft2(1j * KX * filt * v_hat, s=(Nx, Ny))
            dvdy = np.fft.irfft2(1j * KY * filt * v_hat, s=(Nx, Ny))

            nl_u = u * dudx + v * dudy
            nl_v = u * dvdx + v * dvdy

            nl_u_hat = filt * np.fft.rfft2(nl_u)
            nl_v_hat = filt * np.fft.rfft2(nl_v)

            u_hat = (u_hat - dt_sub * nl_u_hat) * diff
            v_hat = (v_hat - dt_sub * nl_v_hat) * diff

            u = np.fft.irfft2(u_hat, s=(Nx, Ny))
            v = np.fft.irfft2(v_hat, s=(Nx, Ny))
        trajectory.append(np.sqrt(u**2 + v**2).copy())

    return np.array(trajectory)


def generate_burgers_2d(output_path, num_traj=12, num_steps=35, grid=64, nu=0.005):
    print(f"[2D Burgers] Generating {num_traj} trajectories on {grid}x{grid} grid...")
    Lx, Ly = 2 * np.pi, 2 * np.pi
    x = np.linspace(0, Lx, grid, endpoint=False)
    y = np.linspace(0, Ly, grid, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    dt = 0.02

    all_tensors = []
    np.random.seed(321)

    for i in range(num_traj):
        u0 = np.zeros((grid, grid))
        v0 = np.zeros((grid, grid))
        n_modes = np.random.randint(2, 5)
        for _ in range(n_modes):
            mx = np.random.randint(1, 3)
            my = np.random.randint(1, 3)
            amp = np.random.uniform(0.3, 1.5) / (mx + my)
            u0 += amp * np.sin(mx * X + np.random.uniform(0, 2*np.pi))
            v0 += amp * np.cos(my * Y + np.random.uniform(0, 2*np.pi))

        traj = solve_burgers_2d(u0, v0, nu=nu, Lx=Lx, Ly=Ly, dt=dt, steps=num_steps - 1)
        all_tensors.append(traj)

    tensor_data = np.stack(all_tensors, axis=0)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("tensor", data=tensor_data.astype(np.float32))
        f.create_dataset("x-coordinate", data=x.astype(np.float32))
        f.create_dataset("y-coordinate", data=y.astype(np.float32))
        f.attrs["equation"] = "2D Burgers"
        f.attrs["nu"] = nu
    print(f"  [OK] Saved {output_path} ({os.path.getsize(output_path)/1024/1024:.2f} MB)")


# =============================================================================
# MASTER GENERATOR
# =============================================================================

def generate_all(output_dir="datasets", grid_1d=256, grid_2d=64, quick=False):
    """Generate all 5 PDE datasets."""
    os.makedirs(output_dir, exist_ok=True)

    n1d = 10 if quick else 25
    n2d = 5 if quick else 15
    t1d = 30 if quick else 50
    t2d = 20 if quick else 40

    print("=" * 70)
    print("  MULTI-EQUATION PDE DATASET GENERATOR")
    print(f"  Output: {output_dir}/ | Quick: {quick}")
    print("=" * 70)

    generate_burgers_1d(
        os.path.join(output_dir, "burgers_1d.hdf5"),
        num_traj=n1d, num_steps=t1d, grid=grid_1d
    )
    generate_heat_2d(
        os.path.join(output_dir, "heat_2d.hdf5"),
        num_traj=n2d, num_steps=t2d, grid=grid_2d
    )
    generate_advection_2d(
        os.path.join(output_dir, "advection_2d.hdf5"),
        num_traj=n2d, num_steps=t2d, grid=grid_2d
    )
    generate_burgers_2d(
        os.path.join(output_dir, "burgers_2d.hdf5"),
        num_traj=n2d, num_steps=t2d, grid=grid_2d
    )
    generate_navier_stokes_2d(
        os.path.join(output_dir, "navier_stokes_2d.hdf5"),
        num_traj=n2d, num_steps=t2d, grid=grid_2d
    )

    print("\n" + "=" * 70)
    print("  ALL DATASETS GENERATED SUCCESSFULLY!")
    total_mb = sum(
        os.path.getsize(os.path.join(output_dir, f)) / 1024 / 1024
        for f in os.listdir(output_dir) if f.endswith('.hdf5')
    )
    print(f"  Total: {total_mb:.2f} MB across 5 equations")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-equation synthetic PDE dataset generator.")
    parser.add_argument("--output_dir", default="datasets", help="Output directory for HDF5 files")
    parser.add_argument("--equation", default="all",
                        choices=["all", "burgers_1d", "heat_2d", "advection_2d", "burgers_2d", "navier_stokes_2d"],
                        help="Which equation to generate")
    parser.add_argument("--grid_1d", type=int, default=256)
    parser.add_argument("--grid_2d", type=int, default=64)
    parser.add_argument("--quick", action="store_true", help="Generate smaller datasets for testing")
    parser.add_argument("--num_traj", type=int, default=None, help="Override the number of trajectories")
    parser.add_argument("--num_steps", type=int, default=None, help="Override the number of timesteps")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    steps_kw = {"num_steps": args.num_steps} if args.num_steps else {}

    if args.equation == "all":
        generate_all(args.output_dir, args.grid_1d, args.grid_2d, args.quick)
    elif args.equation == "burgers_1d":
        n = args.num_traj or (10 if args.quick else 25)
        generate_burgers_1d(os.path.join(args.output_dir, "burgers_1d.hdf5"), num_traj=n, **steps_kw, grid=args.grid_1d)
    elif args.equation == "heat_2d":
        n = args.num_traj or (5 if args.quick else 15)
        generate_heat_2d(os.path.join(args.output_dir, "heat_2d.hdf5"), num_traj=n, **steps_kw, grid=args.grid_2d)
    elif args.equation == "advection_2d":
        n = args.num_traj or (5 if args.quick else 15)
        generate_advection_2d(os.path.join(args.output_dir, "advection_2d.hdf5"), num_traj=n, **steps_kw, grid=args.grid_2d)
    elif args.equation == "burgers_2d":
        n = args.num_traj or (5 if args.quick else 12)
        generate_burgers_2d(os.path.join(args.output_dir, "burgers_2d.hdf5"), num_traj=n, **steps_kw, grid=args.grid_2d)
    elif args.equation == "navier_stokes_2d":
        n = args.num_traj or (5 if args.quick else 10)
        generate_navier_stokes_2d(os.path.join(args.output_dir, "navier_stokes_2d.hdf5"), num_traj=n, **steps_kw, grid=args.grid_2d)
