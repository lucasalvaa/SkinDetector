import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import yaml
from torch.amp import GradScaler

from src.common import DEVICE, get_dataloader, get_model, train_epoch, validate


class LDAMLoss(nn.Module):
    """Implementation of the Label-Distribution-Aware Margin Loss."""

    def __init__(
        self, cls_num_list: list[int], max_m: float = 0.5, s: float = 30.0
    ) -> None:
        """Initializer."""
        super().__init__()
        m_list = 1.0 / torch.sqrt(
            torch.sqrt(torch.tensor(cls_num_list, dtype=torch.float))
        )
        l_cap = m_list * (max_m / torch.max(m_list))
        self.margins = l_cap.to(DEVICE)
        self.s = s

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Apply LDAM margin logic to the logits."""
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.float)
        batch_m = torch.matmul(self.margins[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return functional.cross_entropy(self.s * output, target)

def main() -> None:
    """Entry point for training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tsft", type=bool, default=False)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_loader = get_dataloader(
        image_res=config["base"]["image_res"],
        data_path=Path(config["data"]["trainset_path"]),
        batch_size=config["train"]["batch_size"],
    )

    val_loader = get_dataloader(
        image_res=config["base"]["image_res"],
        data_path=Path(config["data"]["valset_path"]),
        batch_size=config["train"]["batch_size"],
    )

    model = get_model(args.model, len(tr_loader.dataset.classes)).to(DEVICE)

    # Two-Stages Fine-Tuning
    if args.tsft:
        # Freeze layers
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze head
        head = getattr(model, "classifier", getattr(model, "head", None))
        if head:
            for param in head.parameters():
                param.requires_grad = True

        # The Two-Stage Fine-Tuning pipeline validates the model using LDAM Loss
        all_targets = [t for _, t in tr_loader.dataset]
        counts = [
            Counter(all_targets)[i] for i in range(len(tr_loader.dataset.classes))
        ]
        criterion = LDAMLoss(cls_num_list=counts)

    # The other pipelines validates the model using Cross-Entropy Loss
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config["train"]["lr"]
    )
    scaler = GradScaler()

    # Model training
    history = []
    epochs = config["train"]["epochs"]
    print(f"Training {args.model}...")
    for epoch in range(epochs):
        t_loss = train_epoch(model, tr_loader, criterion, optimizer, scaler)
        v_loss = validate(model, val_loader, criterion)

        history.append({"epoch": epoch + 1, "train_loss": t_loss, "val_loss": v_loss})

        print(
            f"Epoch {epoch + 1}/{epochs} | T-Loss: {t_loss:.4f} | V-Loss: {v_loss:.4f}"
        )
    print(f"Model {args.model} trained successfully!")

    # Saving the model
    torch.save(model.state_dict(), out_dir / "model.pth")

    # Saving training and validation loss in loss.json file
    if not args.tsft:
        with open(out_dir / "loss.json", "w") as f:
            json.dump(history, f, indent=4)


if __name__ == "__main__":
    main()
