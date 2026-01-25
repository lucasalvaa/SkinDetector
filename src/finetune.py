"""Fine-tuning script for the best performing model using balanced data phase 2."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import amp, nn, optim

from src.common import DEVICE, get_dataloader, get_model, train_epoch, validate


def main() -> None:
    """Execute the fine-tuning pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    with open(args.config) as conf_file:
        config: dict[str, Any] = yaml.safe_load(conf_file)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_loader = get_dataloader(
        data_path=Path(config["finetuning"]["data_path"]),
        batch_size=config["finetuning"]["batch_size"],
    )

    v_loader = get_dataloader(
        data_path=Path(config["data"]["valset_path"]),
        batch_size=config["finetuning"]["batch_size"],
    )

    # Model initialization loading first stage's weights
    model = get_model(args.model, len(t_loader.dataset.classes))
    weights_path = out_dir.parent / "model.pth"
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE)).to(DEVICE)

    # Unfreeze layers
    for param in model.parameters():
        param.requires_grad = True

    # Fine-tuning setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["finetuning"]["lr"])
    scaler = amp.GradScaler()

    # Model fine-tuning
    history = []
    epochs = config["finetuning"]["epochs"]
    print(f"Fine-tuning {args.model}...")
    for epoch in range(epochs):
        t_loss = train_epoch(model, t_loader, criterion, optimizer, scaler)
        v_loss = validate(model, v_loader, criterion)

        history.append({"epoch": epoch + 1, "train_loss": t_loss, "val_loss": v_loss})

        print(
            f"Epoch {epoch + 1}/{epochs} | T-Loss: {t_loss:.4f} | V-Loss: {v_loss:.4f}"
        )
    print(f"Model {args.model} fine-tuned successfully!")

    # Saving the model
    torch.save(model.state_dict(), out_dir / "model.pth")

    # Saving training and validation loss in loss.json file
    with open(out_dir / "loss.json", "w") as f:
        json.dump(history, f, indent=4)


if __name__ == "__main__":
    main()
