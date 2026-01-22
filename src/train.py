"""Training script for P1 with loss tracking, plotting, and Ruff compliance."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch import amp, nn, optim
from torch.utils.data import DataLoader

from src.common import DEVICE, get_dataloader, get_model


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


def main() -> None:
    """Define main training entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    with open(args.config) as conf_file:
        config = yaml.safe_load(conf_file)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader = get_dataloader(
        image_res=config["base"]["image_res"],
        data_path=Path(config["data"]["trainset_path"]),
        batch_size=config["train"]["batch_size"],
    )

    val_loader = get_dataloader(
        image_res=config["base"]["image_res"],
        data_path=Path(config["data"]["valset_path"]),
        batch_size=config["train"]["batch_size"],
    )

    model = get_model(args.model, len(train_loader.dataset.classes))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), config["train"]["lr"])
    scaler = amp.GradScaler()

    history = []

    for epoch in range(config["train"]["epochs"]):
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        v_loss = validate(model, val_loader, criterion)

        history.append({"epoch": epoch + 1, "train_loss": t_loss, "val_loss": v_loss})

        print(
            f"Epoch {epoch + 1}/{config['train']['epochs']} | "
            f"Train Loss: {t_loss:.4f} | "
            f"Val Loss: {v_loss:.4f}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    torch.save(model.state_dict(), out_dir / "model.pth")

    with open(out_dir / "loss.json", "w") as f:
        json.dump(history, f, indent=4)


if __name__ == "__main__":
    main()
