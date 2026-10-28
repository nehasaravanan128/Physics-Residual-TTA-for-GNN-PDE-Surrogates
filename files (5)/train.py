"""
train.py
Enhanced supervised training loop for the baseline MeshGraphNet surrogate.
NO physics-residual loss, NO test-time adaptation. This trains the plain
"before adaptation" model that later TTA results get compared against.

Enhancements:
  - Cosine annealing LR scheduler with linear warmup
  - Gradient clipping for stability
  - Gradient accumulation for effective larger batch sizes
  - Per-epoch timing and ETA
  - Multi-metric tracking (MSE, relative MSE, MAE, max error)
  - Early stopping with patience
  - Model architecture configuration

Usage:
    python train.py --dataset burgers --data_path ./data/1D_Burgers_Sols_Nu1.0.hdf5
    python train.py --dataset cylinder --data_path ./data/cylinder_flow \
                     --meta_path ./data/cylinder_flow/meta.json
"""

import argparse
import math
import time

import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from data import load_burgers, load_cylinder_flow
from model import MeshGraphNet


def relative_mse(pred, true, eps=1e-8):
    """Scale-invariant relative MSE: mean((pred - true)^2) / (mean(true^2) + eps)"""
    mse = torch.mean((pred - true) ** 2)
    scale = torch.mean(true ** 2) + eps
    return (mse / scale).item()


class CosineWarmupScheduler:
    """Cosine annealing with linear warmup."""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


def train_one_epoch(model, loader, optimizer, device, grad_clip=1.0, accum_steps=1, noise_std=0.0):
    model.train()
    total_loss = 0.0
    num_samples = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)

        # Training noise: perturb the input state but keep the clean target, so the
        # model learns to correct its own drift during autoregressive rollouts.
        x_in = batch.x
        if noise_std > 0:
            x_in = x_in + noise_std * torch.randn_like(x_in)

        pred = model(x_in, batch.edge_index, batch.edge_attr)
        loss = torch.nn.functional.mse_loss(pred, batch.y)

        # Delta models optimize the error in normalized-delta units (O(1) gradients)
        scaled_loss = loss / (model.delta_scale ** 2) if model.predict_delta else loss
        scaled_loss = scaled_loss / accum_steps
        scaled_loss.backward()
        
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            # Gradient clipping for stability
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * batch.num_graphs
        num_samples += batch.num_graphs

    return total_loss / num_samples


@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()
    total_mse = 0.0
    all_preds = []
    all_targets = []
    num_samples = 0

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = torch.nn.functional.mse_loss(pred, batch.y)
        total_mse += loss.item() * batch.num_graphs
        num_samples += batch.num_graphs
        all_preds.append(pred.cpu())
        all_targets.append(batch.y.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    mse = total_mse / num_samples
    rel_mse = relative_mse(preds, targets)
    mae = torch.mean(torch.abs(preds - targets)).item()
    max_err = torch.max(torch.abs(preds - targets)).item()
    
    return {
        "mse": mse,
        "relative_mse": rel_mse,
        "mae": mae,
        "max_error": max_err,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["burgers", "cylinder"], required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--meta_path", default=None, help="required for --dataset cylinder")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_path", default="baseline_model.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Linear warmup epochs")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--activation", choices=["gelu", "relu"], default="gelu")
    parser.add_argument("--no_attention", action="store_true", help="Disable edge attention")
    parser.add_argument("--max_trajs", type=int, default=20, help="Maximum trajectories to load (default: 20 for fast training; <=0 for all)")
    parser.add_argument("--t_stride", type=int, default=1, help="Use every Nth timestep pair (more trajectories per epoch budget)")
    parser.add_argument("--predict_delta", action="store_true", help="Predict normalized u_{t+1} - u_t instead of u_{t+1}")
    parser.add_argument("--noise_std", type=float, default=0.0, help="Gaussian input noise std during training")
    parser.add_argument("--edge_state_diff", action="store_true", help="Append neighbour state differences to edge features")
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # --- Load data ---------------------------------------------------------
    max_samples = args.max_trajs if args.max_trajs > 0 else None
    if args.dataset == "burgers":
        print(f"Loading Burgers dataset from {args.data_path} (max {args.max_trajs} trajs)...", flush=True)
        dataset = load_burgers(args.data_path, num_samples=max_samples, t_stride=args.t_stride)
        # Split by trajectory (samples are trajectory-major) so validation never sees
        # neighbouring timesteps of a training trajectory.
        num_trajs = args.max_trajs if args.max_trajs > 0 else len(dataset)
        per_traj = len(dataset) // num_trajs
        val_trajs = max(1, int(num_trajs * args.val_split))
        split = (num_trajs - val_trajs) * per_traj
        train_set, val_set = dataset[:split], dataset[split:]
    else:
        if args.meta_path is None:
            raise ValueError("--meta_path is required for --dataset cylinder")
        dataset = load_cylinder_flow(args.data_path, args.meta_path)
        val_size = max(1, int(len(dataset) * args.val_split))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
        )
    print(f"Loaded {len(dataset)} graph samples -> Train: {len(train_set)} | Val: {len(val_set)}", flush=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # --- Build model ---------------------------------------------------------
    sample = dataset[0]
    # Normalization statistics from the training split only
    delta_scale, edge_scale = 1.0, 1.0
    if args.predict_delta:
        deltas = torch.cat([d.y[:, :d.x.shape[1]] - d.x[:, :d.y.shape[1]] for d in train_set[:500]])
        delta_scale = deltas.std().item()
    edge_scale = sample.edge_attr.abs().mean().item() or 1.0
    state_diff_scale = 1.0
    if args.edge_state_diff:
        diffs = torch.cat([d.x[d.edge_index[1]] - d.x[d.edge_index[0]] for d in train_set[:500]])
        state_diff_scale = diffs.std().item()
    print(f"Normalization: delta_scale={delta_scale:.4e} | edge_scale={edge_scale:.4e} | "
          f"state_diff_scale={state_diff_scale:.4e}", flush=True)

    model = MeshGraphNet(
        node_in_dim=sample.x.shape[1],
        edge_in_dim=sample.edge_attr.shape[1],
        out_dim=sample.y.shape[1],
        latent_dim=args.latent_dim,
        hidden_dim=args.latent_dim,
        num_processor_blocks=args.num_blocks,
        activation=args.activation,
        dropout=args.dropout,
        use_edge_attention=not args.no_attention,
        predict_delta=args.predict_delta,
        delta_scale=delta_scale,
        edge_scale=edge_scale,
        edge_state_diff=args.edge_state_diff,
        state_diff_scale=state_diff_scale,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"MeshGraphNet initialized ({num_params:,} parameters). Starting training...", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineWarmupScheduler(optimizer, args.warmup_epochs, args.epochs)

    # --- Train ---------------------------------------------------------------
    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Update learning rate
        current_lr = scheduler.step(epoch - 1)
        
        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                      grad_clip=args.grad_clip,
                                      accum_steps=args.accum_steps,
                                      noise_std=args.noise_std)
        val_metrics = evaluate_loss(model, val_loader, device)
        val_loss = val_metrics["mse"]
        
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        eta = (elapsed / epoch) * (args.epochs - epoch)

        print(f"Epoch {epoch:3d} | train MSE: {train_loss:.6f} | val MSE: {val_loss:.6f} | "
              f"rel_MSE: {val_metrics['relative_mse']:.6f} | MAE: {val_metrics['mae']:.6f} | "
              f"lr: {current_lr:.2e} | {epoch_time:.1f}s (ETA: {eta:.0f}s)", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "node_in_dim": sample.x.shape[1],
                    "edge_in_dim": sample.edge_attr.shape[1],
                    "out_dim": sample.y.shape[1],
                    "latent_dim": args.latent_dim,
                    "hidden_dim": args.latent_dim,
                    "activation": args.activation,
                    "predict_delta": args.predict_delta,
                    "delta_scale": delta_scale,
                    "edge_scale": edge_scale,
                    "edge_state_diff": args.edge_state_diff,
                    "state_diff_scale": state_diff_scale,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "epoch": epoch,
                    "num_params": num_params,
                },
                args.checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience {args.patience} exceeded)")
                break

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s")
    print(f"Best val MSE: {best_val_loss:.6f} — saved to {args.checkpoint_path}")


if __name__ == "__main__":
    main()
