"""Fine-tuning script for the best performing model using balanced data phase 2."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch import amp, nn, optim

from src.common import DEVICE, get_dataloader, get_model, train_epoch, validate
from src.early_stopping import EarlyStopping


def main() -> None:
    """Execute the fine-tuning pipeline."""
    choices = ["baseline", "pipeline1", "pipeline2", "pipeline3"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=choices, type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    # parser.add_argument("--config", type=str, required=True)
    # parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    params_path = Path(args.pipeline) / "params.yaml"
    with open(params_path) as f:
        config = yaml.safe_load(f)

    out_dir = Path(args.pipeline / args.model) / "finetuned"
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
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)

    best_model_path = out_dir / "model.pth"
    early_stopper = EarlyStopping(
        alpha=config["train"].get("alpha", 5.0), path=str(best_model_path)
    )

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

        early_stopper(v_loss, epoch + 1, model)
        if early_stopper.stop:
            print(
                f"Stopping at epoch {epoch + 1}. "
                f"Best model was at epoch {early_stopper.best_epoch}"
            )
            break

    print(f"Model {args.model} fine-tuned successfully!")

    # Saving the model
    torch.save(model.state_dict(), out_dir / "model.pth")

    # Saving training and validation loss in loss.json file
    with open(out_dir / "loss.json", "w") as f:
        json.dump(history, f, indent=4)


if __name__ == "__main__":
    main()
