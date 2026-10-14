"""
scripts/data_2d.py
2D PDE Dataset Loader & Graph Constructor.

Loads synthetic 2D PDE datasets (Heat, Advection, Burgers, Navier-Stokes)
from HDF5 files and constructs PyG graph objects using triangular mesh connectivity.

The 2D grid is converted into a graph by:
1. Delaunay triangulation of the grid coordinates → triangle cells
2. Extracting unique bidirectional edges from triangle connectivity
3. Computing edge features: relative displacement + distance norm
4. Node features: [u(t), x_coord, y_coord, one_hot_node_type]
5. Target: u(t+1)
"""

import os
import sys
import json
import h5py
import numpy as np
import torch
from torch_geometric.data import Data

# Node types: 0 = interior, 1 = boundary-x, 2 = boundary-y, 3 = corner
NUM_NODE_TYPES_2D = 4


def _assign_node_types_2d(Nx, Ny):
    """
    Assigns node types on a 2D grid:
    - 0: interior nodes
    - 1: boundary nodes on x-edges (left/right)
    - 2: boundary nodes on y-edges (top/bottom)
    - 3: corner nodes
    """
    node_types = np.zeros((Nx, Ny), dtype=np.int64)
    # Boundary x (left/right columns)
    node_types[0, :] = 1
    node_types[-1, :] = 1
    # Boundary y (top/bottom rows)
    node_types[:, 0] = 2
    node_types[:, -1] = 2
    # Corners
    for i in [0, Nx-1]:
        for j in [0, Ny-1]:
            node_types[i, j] = 3
    return node_types.flatten()


def _grid_to_triangles(Nx, Ny):
    """
    Creates Delaunay-style triangles from a structured grid.
    Each square cell [i,j] → [i+1,j] → [i+1,j+1] → [i,j+1] is split
    into 2 triangles, yielding 2*(Nx-1)*(Ny-1) triangles total.
    """
    cells = []
    for i in range(Nx - 1):
        for j in range(Ny - 1):
            n00 = i * Ny + j
            n10 = (i + 1) * Ny + j
            n11 = (i + 1) * Ny + (j + 1)
            n01 = i * Ny + (j + 1)
            # Two triangles per quad cell
            cells.append([n00, n10, n11])
            cells.append([n00, n11, n01])
    return np.array(cells, dtype=np.int64)


def _cells_to_edge_index(cells):
    """
    Converts triangular cells [num_cells, 3] into bidirectional edge_index.
    """
    edges = set()
    for tri in cells:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        for a, b in [(i, j), (j, k), (k, i)]:
            edges.add((a, b))
            edges.add((b, a))
    return np.array(sorted(edges)).T  # [2, num_edges]


def load_2d_dataset(hdf5_path, max_traj=None, max_steps=None):
    """
    Loads a 2D PDE dataset from HDF5 and constructs PyG graph objects.

    Returns:
        graphs: list of PyG Data objects
        metadata: dict with equation info, grid dimensions, etc.
    """
    with h5py.File(hdf5_path, "r") as f:
        tensor = f["tensor"][:]         # [N_traj, T, Nx, Ny]
        x_coord = f["x-coordinate"][:]
        y_coord = f["y-coordinate"][:]
        equation = f.attrs.get("equation", "Unknown")
        nu = f.attrs.get("nu", 0.0)
        kappa = f.attrs.get("kappa", 0.0)

    N_traj, T, Nx, Ny = tensor.shape
    if max_traj:
        N_traj = min(N_traj, max_traj)
    if max_steps:
        T = min(T, max_steps)

    # Build mesh coordinates
    X, Y = np.meshgrid(x_coord, y_coord, indexing='ij')
    mesh_pos = np.stack([X.flatten(), Y.flatten()], axis=-1)  # [N_nodes, 2]
    num_nodes = Nx * Ny

    # Build triangle mesh and edges
    cells = _grid_to_triangles(Nx, Ny)
    edge_index_np = _cells_to_edge_index(cells)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    # Edge features: relative position + distance
    src, dst = edge_index_np
    rel_pos = mesh_pos[dst] - mesh_pos[src]
    rel_norm = np.linalg.norm(rel_pos, axis=1, keepdims=True)
    edge_attr = torch.tensor(
        np.concatenate([rel_pos, rel_norm], axis=1),
        dtype=torch.float32
    )  # [num_edges, 3]

    # Node types
    node_types = _assign_node_types_2d(Nx, Ny)
    node_type_oh = np.eye(NUM_NODE_TYPES_2D)[node_types]  # [N_nodes, 4]
    mesh_pos_t = torch.tensor(mesh_pos, dtype=torch.float32)

    # Build graph objects for each trajectory-timestep pair
    graphs = []
    for traj_idx in range(N_traj):
        for t in range(T - 1):
            u_t = tensor[traj_idx, t].flatten()      # [N_nodes]
            u_next = tensor[traj_idx, t + 1].flatten()

            # Node features: [u, x, y, node_type_one_hot]
            node_features = np.column_stack([
                u_t[:, None],
                mesh_pos,
                node_type_oh,
            ])  # [N_nodes, 1 + 2 + 4 = 7]

            graphs.append(Data(
                x=torch.tensor(node_features, dtype=torch.float32),
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor(u_next[:, None], dtype=torch.float32),
                mesh_pos=mesh_pos_t,
                grid_shape=torch.tensor([Nx, Ny], dtype=torch.long),
            ))

    metadata = {
        "equation": str(equation),
        "nu": float(nu),
        "kappa": float(kappa),
        "grid": [Nx, Ny],
        "num_trajectories": N_traj,
        "timesteps": T,
        "num_nodes": num_nodes,
        "num_edges": edge_index.shape[1],
        "node_in_dim": 7,   # u + x + y + 4-class one-hot
        "edge_in_dim": 3,   # dx + dy + dist
        "out_dim": 1,
        "num_graphs": len(graphs),
    }
    return graphs, metadata


def get_dataset_summary(dataset_dir="datasets"):
    """Returns a summary dict of all available datasets."""
    summary = {}
    if not os.path.isdir(dataset_dir):
        return summary

    for fname in sorted(os.listdir(dataset_dir)):
        if not fname.endswith('.hdf5'):
            continue
        path = os.path.join(dataset_dir, fname)
        try:
            with h5py.File(path, "r") as f:
                eq = f.attrs.get("equation", "Unknown")
                shape = f["tensor"].shape
                summary[fname] = {
                    "equation": str(eq),
                    "shape": list(shape),
                    "ndim": len(shape),
                    "size_mb": os.path.getsize(path) / 1024 / 1024,
                }
                if "y-coordinate" in f:
                    summary[fname]["type"] = "2D"
                else:
                    summary[fname]["type"] = "1D"
        except Exception as e:
            summary[fname] = {"error": str(e)}
    return summary


if __name__ == "__main__":
    # Test with a small generated dataset
    print("=" * 60)
    print("  2D PDE Dataset Loader — Sanity Check")
    print("=" * 60)

    test_dir = "datasets"
    if os.path.isdir(test_dir):
        summary = get_dataset_summary(test_dir)
        for name, info in summary.items():
            print(f"\n  {name}:")
            for k, v in info.items():
                print(f"    {k}: {v}")

            if info.get("type") == "2D":
                path = os.path.join(test_dir, name)
                graphs, meta = load_2d_dataset(path, max_traj=2, max_steps=5)
                print(f"    -> Built {len(graphs)} graphs")
                if graphs:
                    g = graphs[0]
                    print(f"    -> x: {g.x.shape}, edge_index: {g.edge_index.shape}, "
                          f"edge_attr: {g.edge_attr.shape}, y: {g.y.shape}")
    else:
        print(f"  No datasets directory found at '{test_dir}'.")
        print(f"  Run: python scripts/generate_synthetic_pde.py --quick")
