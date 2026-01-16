"""Training script for P1 with loss tracking, plotting, and Ruff compliance."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from src.common import DEVICE, get_dataloaders, get_model
from torch import amp, nn, optim
from torch.utils.data import DataLoader


def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
    """Calculate average loss on the validation set."""
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: amp.GradScaler,
) -> float:
    """Run one training epoch using Automatic Mixed Precision (AMP).

    Args:
        model: The neural network model.
        loader: DataLoader for the training set.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        scaler: AMP GradScaler for stable float16 training.

    Returns:
        The average loss for the epoch.

    """
    model.train()
    running_loss = 0.0

    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()

        with amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def save_loss_plot(
    train_losses: List[float], val_losses: List[float], out_dir: Path
) -> None:
    """Generate and save the training vs validation loss plot."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / "loss_plot.png")
    plt.close()


def main() -> None:
    """Define main training entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaders = get_dataloaders(Path(args.data), args.batch)
    model = get_model(args.model, len(loaders["train"].dataset.classes))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = amp.GradScaler()

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(args.epochs):
        t_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, scaler)
        v_loss = validate(model, loaders["val"], criterion)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {t_loss:.4f} | "
            f"Val Loss: {v_loss:.4f}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_loss_plot(history["train_loss"], history["val_loss"], out_dir)
    torch.save(model.state_dict(), out_dir / "model.pth")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f)


if __name__ == "__main__":
    main()
