"""
train.py
Standard supervised training loop for the baseline MeshGraphNet surrogate —
NO physics-residual loss, NO test-time adaptation. This trains the plain
"before adaptation" model that later TTA results get compared against.

Usage:
    python train.py --dataset burgers --data_path ./data/1D_Burgers_Sols_Nu1.0.hdf5
    python train.py --dataset cylinder --data_path ./data/cylinder_flow \
                     --meta_path ./data/cylinder_flow/meta.json
"""

import argparse

import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from data import load_burgers, load_cylinder_flow
from model import MeshGraphNet


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = torch.nn.functional.mse_loss(pred, batch.y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = torch.nn.functional.mse_loss(pred, batch.y)
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["burgers", "cylinder"], required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--meta_path", default=None, help="required for --dataset cylinder")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--checkpoint_path", default="baseline_model.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load data ---------------------------------------------------------
    if args.dataset == "burgers":
        dataset = load_burgers(args.data_path)
    else:
        if args.meta_path is None:
            raise ValueError("--meta_path is required for --dataset cylinder")
        dataset = load_cylinder_flow(args.data_path, args.meta_path)

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Train graphs: {len(train_set)} | Val graphs: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # --- Build model ---------------------------------------------------------
    sample = dataset[0]
    model = MeshGraphNet(
        node_in_dim=sample.x.shape[1],
        edge_in_dim=sample.edge_attr.shape[1],
        out_dim=sample.y.shape[1],
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Train ---------------------------------------------------------------
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, device)

        print(f"Epoch {epoch:3d} | train MSE: {train_loss:.6f} | val MSE: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "node_in_dim": sample.x.shape[1],
                    "edge_in_dim": sample.edge_attr.shape[1],
                    "out_dim": sample.y.shape[1],
                    "val_loss": val_loss,
                },
                args.checkpoint_path,
            )

    print(f"Best val MSE: {best_val_loss:.6f} — saved to {args.checkpoint_path}")


if __name__ == "__main__":
    main()
