"""
data.py
Dataset loaders for the physics-residual TTA project: converts raw PDEBench
(Burgers') and DeepMind CylinderFlow data into PyTorch Geometric `Data`
objects that model.py's MeshGraphNet can consume directly.

Both loaders build one-step-ahead prediction pairs: given the field at time t,
predict the field at time t+1 (data.x = input state, data.y = target state).
"""

import json

import numpy as np
import torch
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# 1D Burgers' equation (PDEBench)
# ---------------------------------------------------------------------------


def load_burgers(h5_path, num_samples=None, t_stride=1):
    """
    Loads a PDEBench 1D Burgers' HDF5 file and converts it into a list of PyG
    Data objects, one per (trajectory, timestep) pair.

    PDEBench stores 1D Burgers' data with these HDF5 keys:
      - 'tensor':       shape [N, T, X]  (N trajectories, T timesteps, X grid points)
      - 'x-coordinate': shape [X]        (spatial grid, used to build a 1D chain graph)
    (confirmed against the PDEBench repo/paper — if your file uses different
    key names, run print(list(f.keys())) first to check.)
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        ds = f["tensor"]
        if num_samples is not None:
            tensor = ds[:num_samples]
        else:
            tensor = ds[:]
        x_coord = f["x-coordinate"][:]

    N, T, X = tensor.shape

    # 1D chain graph: each grid point connects to its left/right neighbor.
    # Same edge_index for every sample/timestep since the grid is fixed.
    src = np.arange(X - 1)
    dst = src + 1
    edge_index_np = np.concatenate(
        [np.stack([src, dst]), np.stack([dst, src])], axis=1  # bidirectional
    )
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    # Edge feature: signed spatial distance between neighboring grid points.
    dx = x_coord[dst] - x_coord[src]
    edge_attr_np = np.concatenate([dx, -dx])[:, None]  # [num_edges, 1]
    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32)

    dataset = []
    for n in range(N):
        for t in range(0, T - 1, t_stride):  # predict t -> t+1
            x_in = torch.tensor(tensor[n, t], dtype=torch.float32).unsqueeze(-1)  # [X, 1]
            x_target = torch.tensor(tensor[n, t + 1], dtype=torch.float32).unsqueeze(-1)

            dataset.append(Data(x=x_in, edge_index=edge_index, edge_attr=edge_attr, y=x_target))

    return dataset


# ---------------------------------------------------------------------------
# CylinderFlow (DeepMind MeshGraphNets dataset)
# ---------------------------------------------------------------------------

# NodeType values, confirmed against DeepMind's MeshGraphNets common.py convention.
NODE_TYPE_NORMAL = 0
NODE_TYPE_OBSTACLE = 1
NODE_TYPE_INFLOW = 4
NODE_TYPE_OUTFLOW = 5
NODE_TYPE_WALL_BOUNDARY = 6
NUM_NODE_TYPES = 9  # one-hot width DeepMind's dataset reserves (SIZE = 9)


def _one_hot_node_type(node_type, num_types=NUM_NODE_TYPES):
    node_type = node_type.astype(np.int64).reshape(-1)
    return np.eye(num_types)[node_type]


def _cells_to_edge_index(cells):
    """
    Converts triangular mesh cells [num_cells, 3] into a bidirectional
    edge_index: each triangle contributes its 3 edges (nodes sharing a cell
    are connected), matching the MeshGraphNets paper's mesh-edge definition.
    """
    edges = set()
    for tri in cells:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        for a, b in [(i, j), (j, k), (k, i)]:
            edges.add((a, b))
            edges.add((b, a))
    return np.array(sorted(edges)).T  # [2, num_edges]


def trajectory_to_graphs(mesh_pos, node_type, velocity, pressure, cells):
    """
    Converts one CylinderFlow trajectory (already parsed into numpy arrays)
    into a list of PyG Data objects, one per timestep pair.

    Expected shapes (DeepMind CylinderFlow convention):
      mesh_pos:  [num_nodes, 2]              (static across the trajectory)
      node_type: [num_nodes, 1]              (static)
      velocity:  [num_steps, num_nodes, 2]
      pressure:  [num_steps, num_nodes, 1]
      cells:     [num_cells, 3]              (static)
    """
    edge_index_np = _cells_to_edge_index(cells)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    # Edge features: relative displacement between connected nodes + its norm
    # (encodes local mesh geometry, as in the MeshGraphNets paper).
    src, dst = edge_index_np
    rel_pos = mesh_pos[dst] - mesh_pos[src]
    rel_norm = np.linalg.norm(rel_pos, axis=1, keepdims=True)
    edge_attr = torch.tensor(np.concatenate([rel_pos, rel_norm], axis=1), dtype=torch.float32)

    node_type_oh = _one_hot_node_type(node_type)  # [num_nodes, NUM_NODE_TYPES]

    num_steps = velocity.shape[0]
    graphs = []
    for t in range(num_steps - 1):
        node_features = np.concatenate([velocity[t], pressure[t], node_type_oh], axis=1)
        target = np.concatenate([velocity[t + 1], pressure[t + 1]], axis=1)

        graphs.append(
            Data(
                x=torch.tensor(node_features, dtype=torch.float32),
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor(target, dtype=torch.float32),
            )
        )

    return graphs


def load_cylinder_flow(tfrecord_path, meta_path):
    """
    Loads the DeepMind CylinderFlow dataset from its native TFRecord format.

    IMPORTANT — this function is scaffolding, not verified end-to-end: reading
    it requires either TensorFlow or the lightweight `tfrecord` PyPI package
    (`pip install tfrecord`, avoids the full TF dependency), plus the
    `meta.json` file shipped alongside the dataset (needed because TFRecord
    stores every field as a flat byte array — meta.json gives the shape/dtype
    to reshape it back). The field-parsing loop below follows the layout
    documented in DeepMind's own `parsing_utils` — double check it against
    your actual meta.json once you've downloaded the dataset, since byte-level
    TFRecord parsing can't be tested without the real file in this environment.

    Once parsed into per-trajectory numpy arrays, `trajectory_to_graphs` above
    (which IS tested — see the sanity check at the bottom of this file)
    handles all the graph-construction logic.
    """
    from tfrecord.torch.dataset import TFRecordDataset

    with open(meta_path) as f:
        meta = json.load(f)

    raw_dataset = TFRecordDataset(tfrecord_path, index_path=None)

    all_graphs = []
    for example in raw_dataset:
        fields = {}
        for name, spec in meta["features"].items():
            shape = spec["shape"]
            dtype = np.dtype(spec.get("dtype", "float32"))
            fields[name] = np.frombuffer(example[name].numpy(), dtype=dtype).reshape(shape)

        mesh_pos = fields["mesh_pos"][0]  # static field, take timestep 0
        node_type = fields["node_type"][0]
        cells = fields["cells"][0]
        velocity = fields["velocity"]  # [T, num_nodes, 2]
        pressure = fields["pressure"]  # [T, num_nodes, 1]

        all_graphs.extend(trajectory_to_graphs(mesh_pos, node_type, velocity, pressure, cells))

    return all_graphs


if __name__ == "__main__":
    # Sanity check on synthetic data shaped like a real CylinderFlow
    # trajectory, so the graph-construction logic (the part that matters most,
    # and is fully testable without the real dataset) is verified directly.
    num_nodes, num_cells, num_steps = 30, 40, 5

    mesh_pos = np.random.rand(num_nodes, 2).astype(np.float32)
    node_type = np.random.randint(0, NUM_NODE_TYPES, size=(num_nodes, 1))
    cells = np.random.randint(0, num_nodes, size=(num_cells, 3))
    velocity = np.random.randn(num_steps, num_nodes, 2).astype(np.float32)
    pressure = np.random.randn(num_steps, num_nodes, 1).astype(np.float32)

    graphs = trajectory_to_graphs(mesh_pos, node_type, velocity, pressure, cells)
    print(f"Built {len(graphs)} graphs from a {num_steps}-step synthetic trajectory")
    print(f"First graph: {graphs[0]}")
    assert graphs[0].x.shape == (num_nodes, 2 + 1 + NUM_NODE_TYPES)
    assert graphs[0].y.shape == (num_nodes, 3)
    print("Shape checks passed.")
