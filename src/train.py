import json
from collections import Counter
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from omegaconf import DictConfig
from torch.amp import GradScaler

from src.common import DEVICE, get_dataloader, get_model, train_epoch, validate
from src.early_stopping import EarlyStopping


class LDAMLoss(nn.Module):
    """Implementation of the Label-Distribution-Aware Margin Loss."""

    def __init__(
        self, cls_num_list: list[int], max_m: float = 0.5, s: float = 30.0
    ) -> None:
        """Initialize LDAMLoss."""
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


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Entry point for training."""
    root = Path(hydra.utils.get_original_cwd())  # Project root
    out_dir = root / cfg.pipeline.out_dir / cfg.model.name
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_loader = get_dataloader(
        image_res=cfg.base.image_res,
        data_path=root / cfg.pipeline.trainset_path,
        batch_size=cfg.pipeline.train.batch_size,
    )

    val_loader = get_dataloader(
        image_res=cfg.base.image_res,
        data_path=root / cfg.data.valset_path,
        batch_size=cfg.pipeline.train.batch_size,
    )

    model = get_model(
        cfg.model.fullname,
        cfg.model.weights,
        cfg.model.layer,
        len(tr_loader.dataset.classes),
    ).to(DEVICE)

    best_model_path = out_dir / "model.pth"
    early_stopper = EarlyStopping(
        alpha=cfg.train.get("alpha", 5.0), path=str(best_model_path)
    )

    is_tsft = cfg.pipeline.get("tsft", False)

    # Two-Stages Fine-Tuning
    if is_tsft:
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
        filter(lambda p: p.requires_grad, model.parameters()),
        cfg.pipeline.train.get("lr", 0.001),
    )
    scaler = GradScaler()

    # Model training
    history = []
    epochs = cfg.pipeline.train.get("epochs", 4)
    print(f"Training {cfg.model.name}...")
    for epoch in range(epochs):
        t_loss = train_epoch(model, tr_loader, criterion, optimizer, scaler)
        v_loss = validate(model, val_loader, criterion)

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

    print(f"Model {cfg.model.name} trained successfully!")

    # Saving the model
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
    else:
        torch.save(model.state_dict(), out_dir / "model.pth")

    # Saving training and validation loss in loss.json file
    if not is_tsft:
        with open(out_dir / "loss.json", "w") as f:
            json.dump(history, f, indent=4)


if __name__ == "__main__":
    main()
