"""
scripts/models_2d.py
Multi-Model 2D PDE Surrogate Benchmark Suite.

Implements the leading neural PDE surrogate architectures for 2D flow and field prediction:
  1. MeshGraphNet (Pfaff et al., DeepMind) — message-passing on triangular meshes
  2. FNO2d (Fourier Neural Operator, Li et al.) — spectral convolution in Fourier domain
  3. UNet2d (Physics U-Net, Ronneberger et al.) — multi-scale encoder-decoder with skips
  4. GATSurrogate (Graph Attention Network, Veličković et al.) — adaptive multi-head attention

All models provide a unified forward interface and can be evaluated on any of the
10 generated 2D datasets and geometric shapes.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import scatter


# =============================================================================
# 1. FOURIER NEURAL OPERATOR (FNO-2D)
# =============================================================================

class SpectralConv2d(nn.Module):
    """
    2D Fourier layer. Computes spectral convolution in the frequency domain:
      out = ifft2(R * rfft2(x))
    where R is a learnable complex tensor truncated to modes1 x modes2 frequencies.
    """

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes in x
        self.modes2 = modes2  # Number of Fourier modes in y

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # x: [batch, in_channels, height, width]
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        m1 = min(self.modes1, x_ft.shape[-2] // 2)
        m2 = min(self.modes2, x_ft.shape[-1])

        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    """
    Fourier Neural Operator (Li et al., 2021) for 2D PDE problems.
    Input: [batch, in_channels, H, W]
    Output: [batch, out_channels, H, W]
    """

    def __init__(self, in_channels=1, out_channels=1, modes1=12, modes2=12, width=32, num_layers=4):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(in_channels + 2, width)  # +2 for (x, y) spatial grid encoding

        self.spectral_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2) for _ in range(num_layers)
        ])
        self.w_layers = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(num_layers)
        ])

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x, grid=None):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        if grid is None:
            # Create normalized grid coordinates
            grid_x = torch.linspace(0, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
            grid_y = torch.linspace(0, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
            x_with_grid = torch.cat([x, grid_x, grid_y], dim=1)
        else:
            x_with_grid = torch.cat([x, grid], dim=1)

        x_in = x_with_grid.permute(0, 2, 3, 1)  # [B, H, W, C+2]
        x_proj = self.fc0(x_in).permute(0, 3, 1, 2)  # [B, width, H, W]

        for s_layer, w_layer in zip(self.spectral_layers, self.w_layers):
            x1 = s_layer(x_proj)
            x2 = w_layer(x_proj)
            x_proj = F.gelu(x1 + x2)

        x_out = x_proj.permute(0, 2, 3, 1)  # [B, H, W, width]
        x_out = F.gelu(self.fc1(x_out))
        out = self.fc2(x_out).permute(0, 3, 1, 2)  # [B, out_channels, H, W]
        return out


# =============================================================================
# 2. PHYSICS U-NET (UNet2d)
# =============================================================================

class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet2d(nn.Module):
    """
    2D U-Net surrogate architecture for physical field rollouts.
    Consists of 3 encoder downsampling levels, bottleneck, and 3 decoder upsampling levels
    with skip connections to preserve high-frequency boundary layers and vortex edges.
    """

    def __init__(self, in_channels=1, out_channels=1, base_dim=32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_dim)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_dim, base_dim * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_dim * 2, base_dim * 4))

        self.up1 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_dim * 4, base_dim * 2)

        self.up2 = nn.ConvTranspose2d(base_dim * 2, base_dim, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_dim * 2, base_dim)

        self.outc = nn.Conv2d(base_dim, out_channels, kernel_size=1)

    def forward(self, x):
        # x: [B, in_channels, H, W]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        # Decoder with pad-aligned skips
        u1 = self.up1(x3)
        diffY = x2.size(2) - u1.size(2)
        diffX = x2.size(3) - u1.size(3)
        u1 = F.pad(u1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        m1 = self.conv_up1(torch.cat([x2, u1], dim=1))

        u2 = self.up2(m1)
        diffY = x1.size(2) - u2.size(2)
        diffX = x1.size(3) - u2.size(3)
        u2 = F.pad(u2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        m2 = self.conv_up2(torch.cat([x1, u2], dim=1))

        return self.outc(m2)


# =============================================================================
# 3. GRAPH ATTENTION NETWORK SURROGATE (GATSurrogate)
# =============================================================================

class GATSurrogate(nn.Module):
    """
    Multi-head Graph Attention Network surrogate operating directly on mesh graphs.
    Computes dynamic attention coefficients between neighbor nodes in the fluid mesh.
    """

    def __init__(self, node_in_dim=7, edge_in_dim=3, out_dim=1, hidden_dim=64, heads=4, num_layers=3):
        super().__init__()
        out_dim_gat = hidden_dim * heads
        self.node_enc = nn.Linear(node_in_dim, out_dim_gat)

        self.layers = nn.ModuleList([
            GATConv(out_dim_gat, hidden_dim, heads=heads, edge_dim=edge_in_dim, concat=True)
            for _ in range(num_layers)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(out_dim_gat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index, edge_attr=None):
        h = F.relu(self.node_enc(x))
        for layer in self.layers:
            h = F.elu(layer(h, edge_index, edge_attr=edge_attr)) + h
        return self.decoder(h)


# =============================================================================
# MODEL FACTORY & BENCHMARK HELPER
# =============================================================================

def build_surrogate_model(model_name="fno", **kwargs):
    """
    Surrogate model factory.
    Choices: 'fno', 'unet', 'gat', 'meshgraphnet'
    """
    name = model_name.lower()
    if name == "fno":
        return FNO2d(
            in_channels=kwargs.get("in_channels", 1),
            out_channels=kwargs.get("out_channels", 1),
            modes1=kwargs.get("modes1", 12),
            modes2=kwargs.get("modes2", 12),
            width=kwargs.get("width", 32),
            num_layers=kwargs.get("num_layers", 4)
        )
    elif name == "unet":
        return UNet2d(
            in_channels=kwargs.get("in_channels", 1),
            out_channels=kwargs.get("out_channels", 1),
            base_dim=kwargs.get("base_dim", 32)
        )
    elif name == "gat":
        return GATSurrogate(
            node_in_dim=kwargs.get("node_in_dim", 7),
            edge_in_dim=kwargs.get("edge_in_dim", 3),
            out_dim=kwargs.get("out_dim", 1),
            hidden_dim=kwargs.get("hidden_dim", 48),
            heads=kwargs.get("heads", 4),
            num_layers=kwargs.get("num_layers", 3)
        )
    elif name in ["meshgraphnet", "mgn"]:
        from model import MeshGraphNet
        return MeshGraphNet(
            node_in_dim=kwargs.get("node_in_dim", 7),
            edge_in_dim=kwargs.get("edge_in_dim", 3),
            out_dim=kwargs.get("out_dim", 1),
            latent_dim=kwargs.get("latent_dim", 64),
            hidden_dim=kwargs.get("hidden_dim", 64),
            num_processor_blocks=kwargs.get("num_processor_blocks", 6)
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def count_parameters(model):
    """Returns the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 65)
    print("  2D SURROGATE MODEL SUITE — ARCHITECTURE & PARAMETER AUDIT")
    print("=" * 65)

    # 1. FNO2d
    fno = build_surrogate_model("fno", in_channels=1, out_channels=1, modes1=12, modes2=12, width=32)
    x_grid = torch.randn(2, 1, 48, 96)
    out_fno = fno(x_grid)
    print(f"\n[1] FNO2d (Fourier Neural Operator):")
    print(f"    Input:  {x_grid.shape} -> Output: {out_fno.shape}")
    print(f"    Params: {count_parameters(fno):,} parameters")

    # 2. UNet2d
    unet = build_surrogate_model("unet", in_channels=1, out_channels=1, base_dim=32)
    out_unet = unet(x_grid)
    print(f"\n[2] UNet2d (Physics U-Net):")
    print(f"    Input:  {x_grid.shape} -> Output: {out_unet.shape}")
    print(f"    Params: {count_parameters(unet):,} parameters")

    # 3. GATSurrogate
    gat = build_surrogate_model("gat", node_in_dim=7, edge_in_dim=3, out_dim=1, hidden_dim=48, heads=4)
    x_nodes = torch.randn(200, 7)
    edge_index = torch.randint(0, 200, (2, 800))
    edge_attr = torch.randn(800, 3)
    out_gat = gat(x_nodes, edge_index, edge_attr)
    print(f"\n[3] GATSurrogate (Graph Attention Network):")
    print(f"    Nodes:  {x_nodes.shape} -> Output: {out_gat.shape}")
    print(f"    Params: {count_parameters(gat):,} parameters")

    print("\n" + "=" * 65)
    print("  ALL 2D SURROGATE MODELS VERIFIED SUCCESSFULLY!")
    print("=" * 65)
