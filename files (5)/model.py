"""
model.py
MeshGraphNets-style GNN surrogate architecture, following Pfaff et al. (2021).

Architecture: Encoder -> N x ProcessorBlock (message passing) -> Decoder

- Encoder: embeds raw node/edge features (e.g. velocity, mesh geometry) into a
  shared latent space.
- ProcessorBlock: one round of message passing. Edges are updated first from
  their two endpoint nodes, then nodes are updated from their aggregated
  incoming edge messages. Both updates use residual connections, which the
  original paper found necessary for stable training across many layers.
- Decoder: maps the final node latent features back to physical output
  (e.g. predicted next-step velocity/pressure).

Usage:
    model = MeshGraphNet(node_in_dim=..., edge_in_dim=..., out_dim=...)
    prediction = model(data.x, data.edge_index, data.edge_attr)

node_in_dim / edge_in_dim / out_dim depend on your dataset (Burgers' vs.
cylinder wake) and should be set once data.py defines the feature layout.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


def build_mlp(in_dim, hidden_dim, out_dim, hidden_layers=2, layer_norm=True):
    """Standard MLP block used throughout MeshGraphNets: Linear -> ReLU,
    repeated, with LayerNorm on the output (stabilizes training)."""
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(hidden_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    if layer_norm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """Embeds raw node and edge features into a shared latent space."""

    def __init__(self, node_in_dim, edge_in_dim, latent_dim=128, hidden_dim=128):
        super().__init__()
        self.node_encoder = build_mlp(node_in_dim, hidden_dim, latent_dim)
        self.edge_encoder = build_mlp(edge_in_dim, hidden_dim, latent_dim)

    def forward(self, x, edge_attr):
        return self.node_encoder(x), self.edge_encoder(edge_attr)


class ProcessorBlock(nn.Module):
    """
    One round of message passing:
      1. Edge update  — each edge looks at its two endpoint nodes + its own
         features, and produces an updated edge embedding (the "message").
      2. Aggregation  — each node sums the messages from all incoming edges.
      3. Node update  — each node combines its own features with the
         aggregated messages to produce an updated node embedding.
    Both updates are residual: output = input + MLP(...).
    """

    def __init__(self, latent_dim=128, hidden_dim=128):
        super().__init__()
        self.edge_mlp = build_mlp(3 * latent_dim, hidden_dim, latent_dim)
        self.node_mlp = build_mlp(2 * latent_dim, hidden_dim, latent_dim)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]

        # 1. Edge update
        edge_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        edge_messages = self.edge_mlp(edge_input)
        edge_attr_updated = edge_attr + edge_messages  # residual

        # 2. Aggregate messages at each receiving node (sum, as in the paper)
        agg = scatter(edge_messages, dst, dim=0, dim_size=x.size(0), reduce="sum")

        # 3. Node update
        node_input = torch.cat([x, agg], dim=-1)
        x_updated = x + self.node_mlp(node_input)  # residual

        return x_updated, edge_attr_updated


class Decoder(nn.Module):
    """Maps final node latent features back to physical output space."""

    def __init__(self, latent_dim, out_dim, hidden_dim=128):
        super().__init__()
        self.decoder = build_mlp(latent_dim, hidden_dim, out_dim, layer_norm=False)

    def forward(self, x):
        return self.decoder(x)


class MeshGraphNet(nn.Module):
    """
    Full encoder-processor-decoder GNN surrogate. Predicts per-node output
    (e.g. next-step field values) from the current mesh state.

    Args:
        node_in_dim: number of input features per node
        edge_in_dim: number of input features per edge
        out_dim: number of output features per node (prediction target)
        latent_dim: hidden representation size used throughout the processor
        hidden_dim: hidden layer width inside each MLP
        num_processor_blocks: number of message-passing rounds (paper uses ~15)
    """

    def __init__(
        self,
        node_in_dim,
        edge_in_dim,
        out_dim,
        latent_dim=128,
        hidden_dim=128,
        num_processor_blocks=10,
    ):
        super().__init__()
        self.encoder = Encoder(node_in_dim, edge_in_dim, latent_dim, hidden_dim)
        self.processor_blocks = nn.ModuleList(
            [ProcessorBlock(latent_dim, hidden_dim) for _ in range(num_processor_blocks)]
        )
        self.decoder = Decoder(latent_dim, out_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        x, edge_attr = self.encoder(x, edge_attr)
        for block in self.processor_blocks:
            x, edge_attr = block(x, edge_index, edge_attr)
        return self.decoder(x)


if __name__ == "__main__":
    # Sanity check with random data, matching PyG's Data object convention:
    #   x: [num_nodes, node_in_dim]
    #   edge_index: [2, num_edges]
    #   edge_attr: [num_edges, edge_in_dim]
    num_nodes, num_edges = 50, 150
    node_in_dim, edge_in_dim, out_dim = 4, 3, 2  # placeholders — set from your dataset

    x = torch.randn(num_nodes, node_in_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, edge_in_dim)

    model = MeshGraphNet(node_in_dim, edge_in_dim, out_dim)
    out = model(x, edge_index, edge_attr)
    print(f"Output shape: {out.shape}")  # expect [num_nodes, out_dim]
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
