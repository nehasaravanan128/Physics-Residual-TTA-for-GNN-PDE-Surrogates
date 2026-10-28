"""
model.py
Enhanced MeshGraphNet-style GNN surrogate architecture, following Pfaff et al. (2021)
with additional improvements for PDE surrogate accuracy.

Enhancements over the base MeshGraphNet:
  - Edge Attention: learned attention weights in message passing for adaptive
    receptive field (important for sharp shock fronts in Burgers' equation)
  - GELU activations: smoother than ReLU, better gradient flow for physics
  - Dropout regularization: prevents overfitting on limited PDE trajectories
  - Multi-scale skip connections: skip every 2 processor blocks to preserve
    fine-grained spatial features across deep message passing
  - Configurable architecture: easily swap latent dims, depths, activations

Architecture: Encoder -> N x ProcessorBlock (message passing) -> Decoder

Usage:
    model = MeshGraphNet(node_in_dim=..., edge_in_dim=..., out_dim=...)
    prediction = model(data.x, data.edge_index, data.edge_attr)
"""

import os
import torch
import torch.nn as nn
from torch_geometric.utils import scatter


def build_mlp(in_dim, hidden_dim, out_dim, hidden_layers=2, layer_norm=True,
              activation="gelu", dropout=0.0):
    """Standard MLP block used throughout MeshGraphNets: Linear -> Activation,
    repeated, with LayerNorm on the output (stabilizes training).
    
    Enhanced with configurable activation and optional dropout for regularization.
    """
    act_fn = nn.GELU if activation == "gelu" else nn.ReLU

    layers = [nn.Linear(in_dim, hidden_dim), act_fn()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    for _ in range(hidden_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), act_fn()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, out_dim))
    if layer_norm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """Embeds raw node and edge features into a shared latent space."""

    def __init__(self, node_in_dim, edge_in_dim, latent_dim=128, hidden_dim=128,
                 activation="gelu", dropout=0.0):
        super().__init__()
        self.node_encoder = build_mlp(node_in_dim, hidden_dim, latent_dim,
                                       activation=activation, dropout=dropout)
        self.edge_encoder = build_mlp(edge_in_dim, hidden_dim, latent_dim,
                                       activation=activation, dropout=dropout)

    def forward(self, x, edge_attr):
        return self.node_encoder(x), self.edge_encoder(edge_attr)


class ProcessorBlock(nn.Module):
    """
    One round of message passing with edge attention:
      1. Edge update  — each edge looks at its two endpoint nodes + its own
         features, and produces an updated edge embedding (the "message").
      2. Edge Attention — learned scalar attention weight per edge for
         adaptive message weighting (helps focus on shock fronts).
      3. Aggregation  — each node sums the attention-weighted messages from
         all incoming edges.
      4. Node update  — each node combines its own features with the
         aggregated messages to produce an updated node embedding.
    Both updates are residual: output = input + MLP(...).
    """

    def __init__(self, latent_dim=128, hidden_dim=128, activation="gelu",
                 dropout=0.0, use_edge_attention=True):
        super().__init__()
        self.edge_mlp = build_mlp(3 * latent_dim, hidden_dim, latent_dim,
                                   activation=activation, dropout=dropout)
        self.node_mlp = build_mlp(2 * latent_dim, hidden_dim, latent_dim,
                                   activation=activation, dropout=dropout)
        
        self.use_edge_attention = use_edge_attention
        if use_edge_attention:
            # Learned attention: project edge embedding to scalar weight
            self.attention_mlp = nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 4),
                nn.GELU(),
                nn.Linear(latent_dim // 4, 1),
            )

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]

        # 1. Edge update
        edge_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        edge_messages = self.edge_mlp(edge_input)
        edge_attr_updated = edge_attr + edge_messages  # residual

        # 2. Attention-weighted aggregation
        if self.use_edge_attention:
            # Compute per-edge attention logits and normalize per destination node
            attn_logits = self.attention_mlp(edge_attr_updated).squeeze(-1)  # [E]
            # Softmax per destination node
            attn_max = scatter(attn_logits, dst, dim=0, dim_size=x.size(0), reduce="max")
            attn_shifted = attn_logits - attn_max[dst]
            attn_exp = torch.exp(attn_shifted)
            attn_sum = scatter(attn_exp, dst, dim=0, dim_size=x.size(0), reduce="sum")
            attn_weights = attn_exp / (attn_sum[dst] + 1e-8)
            
            # Weighted message aggregation
            weighted_messages = edge_messages * attn_weights.unsqueeze(-1)
            agg = scatter(weighted_messages, dst, dim=0, dim_size=x.size(0), reduce="sum")
        else:
            # Standard sum aggregation (as in original paper)
            agg = scatter(edge_messages, dst, dim=0, dim_size=x.size(0), reduce="sum")

        # 3. Node update
        node_input = torch.cat([x, agg], dim=-1)
        x_updated = x + self.node_mlp(node_input)  # residual

        return x_updated, edge_attr_updated


class Decoder(nn.Module):
    """Maps final node latent features back to physical output space.
    Enhanced with an extra hidden layer for better expressivity at the
    decoding boundary (important for TTA which adapts only decoder weights)."""

    def __init__(self, latent_dim, out_dim, hidden_dim=128, activation="gelu",
                 dropout=0.0):
        super().__init__()
        self.decoder = build_mlp(latent_dim, hidden_dim, out_dim,
                                  hidden_layers=2, layer_norm=False,
                                  activation=activation, dropout=dropout)

    def forward(self, x):
        return self.decoder(x)


class MeshGraphNet(nn.Module):
    """
    Full encoder-processor-decoder GNN surrogate. Predicts per-node output
    (e.g. next-step field values) from the current mesh state.

    Improvements over vanilla MeshGraphNet:
      - Edge attention in processor blocks for adaptive message weighting
      - GELU activations for smoother gradient landscape
      - Dropout for regularization
      - Multi-scale skip connections every `skip_interval` processor blocks

    Args:
        node_in_dim: number of input features per node
        edge_in_dim: number of input features per edge
        out_dim: number of output features per node (prediction target)
        latent_dim: hidden representation size used throughout the processor
        hidden_dim: hidden layer width inside each MLP
        num_processor_blocks: number of message-passing rounds (paper uses ~15)
        activation: activation function ('gelu' or 'relu')
        dropout: dropout probability (0.0 = no dropout)
        use_edge_attention: whether to use learned edge attention weights
        skip_interval: add skip connection every N processor blocks (0 = disabled)
    """

    def __init__(
        self,
        node_in_dim,
        edge_in_dim,
        out_dim,
        latent_dim=128,
        hidden_dim=128,
        num_processor_blocks=10,
        activation="relu",
        dropout=0.0,
        use_edge_attention=False,
        skip_interval=0,
        predict_delta=False,
        delta_scale=1.0,
        edge_scale=1.0,
        edge_state_diff=False,
        state_diff_scale=1.0,
    ):
        super().__init__()
        # predict_delta: decoder outputs the normalized change (u_{t+1} - u_t) / delta_scale
        # instead of u_{t+1} itself; forward() still returns the absolute next state.
        # edge_scale: edge features are divided by it so they are O(1) for the encoder.
        # edge_state_diff: append (x_dst - x_src) / state_diff_scale to every edge, giving the
        # encoder the local gradient directly (neighbour differences are tiny next to |x|).
        self.predict_delta = predict_delta
        self.delta_scale = delta_scale
        self.edge_scale = edge_scale
        self.edge_state_diff = edge_state_diff
        self.state_diff_scale = state_diff_scale
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.dropout = dropout
        self.use_edge_attention = use_edge_attention
        self.skip_interval = skip_interval
        self.num_processor_blocks = num_processor_blocks

        encoder_edge_dim = edge_in_dim + (node_in_dim if edge_state_diff else 0)
        self.encoder = Encoder(node_in_dim, encoder_edge_dim, latent_dim, hidden_dim,
                                activation=activation, dropout=dropout)
        self.processor_blocks = nn.ModuleList(
            [ProcessorBlock(latent_dim, hidden_dim, activation=activation,
                           dropout=dropout, use_edge_attention=use_edge_attention)
             for _ in range(num_processor_blocks)]
        )
        self.decoder = Decoder(latent_dim, out_dim, hidden_dim,
                                activation=activation, dropout=dropout)
        
        # Skip projection layers (used when skip_interval > 0)
        if skip_interval > 0:
            num_skips = num_processor_blocks // skip_interval
            self.skip_projections = nn.ModuleList(
                [nn.Linear(latent_dim * 2, latent_dim) for _ in range(num_skips)]
            )
        else:
            self.skip_projections = nn.ModuleList()

    @classmethod
    def from_checkpoint(cls, checkpoint_or_path, device="cpu", **override_kwargs):
        """
        Auto-detects architecture configuration from checkpoint weights and initializes
        the appropriate MeshGraphNet instance (supports both baseline and enhanced checkpoints).
        """
        if isinstance(checkpoint_or_path, (str, os.PathLike)):
            ckpt = torch.load(checkpoint_or_path, map_location=device, weights_only=False)
        else:
            ckpt = checkpoint_or_path

        state_dict = ckpt.get("model_state_dict", ckpt)

        # Detect attention MLP layers in state_dict
        has_attention = any("attention_mlp" in k for k in state_dict.keys())

        # Detect skip projection layers in state_dict
        has_skips = any("skip_projections" in k for k in state_dict.keys())
        skip_keys = [k for k in state_dict.keys() if "skip_projections" in k and k.endswith(".weight")]
        num_skips = len(skip_keys)

        # Determine processor block count
        block_indices = [
            int(k.split(".")[1]) for k in state_dict.keys()
            if k.startswith("processor_blocks.") and k.split(".")[1].isdigit()
        ]
        num_blocks = (max(block_indices) + 1) if block_indices else 10
        skip_interval = (num_blocks // num_skips) if (has_skips and num_skips > 0) else 0

        # Retrieve saved dimensions or infer from state_dict tensor shapes
        node_in_dim = ckpt.get("node_in_dim")
        if node_in_dim is None and "encoder.node_encoder.0.weight" in state_dict:
            node_in_dim = state_dict["encoder.node_encoder.0.weight"].shape[1]
        elif node_in_dim is None:
            node_in_dim = 1

        edge_in_dim = ckpt.get("edge_in_dim")
        if edge_in_dim is None and "encoder.edge_encoder.0.weight" in state_dict:
            edge_in_dim = state_dict["encoder.edge_encoder.0.weight"].shape[1]
        elif edge_in_dim is None:
            edge_in_dim = 1

        out_dim = ckpt.get("out_dim")
        if out_dim is None:
            # Last weight of decoder has shape [out_dim, hidden_dim]
            dec_keys = [k for k in state_dict.keys() if k.startswith("decoder.") and k.endswith(".weight")]
            if dec_keys:
                last_dec_key = sorted(dec_keys, key=lambda k: int(k.split(".")[2]))[-1]
                out_dim = state_dict[last_dec_key].shape[0]
            else:
                out_dim = 1

        latent_dim = ckpt.get("latent_dim")
        if latent_dim is None and "encoder.node_encoder.4.weight" in state_dict:
            latent_dim = state_dict["encoder.node_encoder.4.weight"].shape[0]
        elif latent_dim is None:
            latent_dim = 128

        hidden_dim = ckpt.get("hidden_dim")
        if hidden_dim is None and "encoder.node_encoder.0.weight" in state_dict:
            hidden_dim = state_dict["encoder.node_encoder.0.weight"].shape[0]
        elif hidden_dim is None:
            hidden_dim = 128

        kwargs = {
            "node_in_dim": node_in_dim,
            "edge_in_dim": edge_in_dim,
            "out_dim": out_dim,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "num_processor_blocks": num_blocks,
            "use_edge_attention": has_attention,
            "skip_interval": skip_interval,
            "activation": ckpt.get("activation", "relu"),
            "predict_delta": ckpt.get("predict_delta", False),
            "delta_scale": ckpt.get("delta_scale", 1.0),
            "edge_scale": ckpt.get("edge_scale", 1.0),
            "edge_state_diff": ckpt.get("edge_state_diff", False),
            "state_diff_scale": ckpt.get("state_diff_scale", 1.0),
        }
        kwargs.update(override_kwargs)

        model = cls(**kwargs).to(device)
        model.load_state_dict(state_dict, strict=False)
        return model

    def encode_and_process(self, x, edge_index, edge_attr):
        """Encodes graph inputs and executes processor blocks, returning latent node features."""
        edge_attr = edge_attr / self.edge_scale
        if self.edge_state_diff:
            src, dst = edge_index[0], edge_index[1]
            edge_attr = torch.cat([edge_attr, (x[dst] - x[src]) / self.state_diff_scale], dim=-1)
        x, edge_attr = self.encoder(x, edge_attr)
        skip_cache = x  # Store initial encoded features for skip connections
        skip_idx = 0
        for i, block in enumerate(self.processor_blocks):
            x, edge_attr = block(x, edge_index, edge_attr)
            if (self.skip_interval > 0 and 
                (i + 1) % self.skip_interval == 0 and 
                skip_idx < len(self.skip_projections)):
                x = self.skip_projections[skip_idx](
                    torch.cat([x, skip_cache], dim=-1)
                )
                skip_cache = x
                skip_idx += 1
        return x

    def decode(self, latent, x):
        """Maps latent node features to the absolute next state (x is the raw model input)."""
        out = self.decoder(latent)
        if self.predict_delta:
            out = x[..., :self.out_dim] + out * self.delta_scale
        return out

    def forward(self, x, edge_index, edge_attr):
        latent = self.encode_and_process(x, edge_index, edge_attr)
        return self.decode(latent, x)


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
    
    # Test backward compatibility (no attention, no skips — original behavior)
    model_basic = MeshGraphNet(node_in_dim, edge_in_dim, out_dim,
                                use_edge_attention=False, skip_interval=0,
                                activation="relu", dropout=0.0)
    out_basic = model_basic(x, edge_index, edge_attr)
    print(f"Basic model output shape: {out_basic.shape}")
    print(f"Basic model parameters: {sum(p.numel() for p in model_basic.parameters()):,}")
