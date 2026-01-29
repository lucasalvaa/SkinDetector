"""Fine-tuning script for the best performing model using balanced data phase 2."""

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch import amp, nn, optim

from src.common import DEVICE, get_dataloader, get_model, train_epoch, validate
from src.early_stopping import EarlyStopping


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Execute the fine-tuning pipeline."""
    root = Path(hydra.utils.get_original_cwd())  # Project root
    out_dir = root / cfg.pipeline.out_dir / cfg.model.name / "finetuned"
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

    # Model initialization loading first stage's weights
    model = get_model(
        cfg.model.fullname,
        cfg.model.weights,
        cfg.model.layer,
        len(tr_loader.dataset.classes),
    ).to(DEVICE)

    weights_path = out_dir.parent / "model.pth"
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)

    best_model_path = out_dir / "model.pth"
    early_stopper = EarlyStopping(
        alpha=cfg.train.get("alpha", 5.0), path=str(best_model_path)
    )

    # Unfreeze layers
    for param in model.parameters():
        param.requires_grad = True

    # Fine-tuning setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        cfg.pipeline.finetuning.get("lr", 0.0001),
    )
    scaler = amp.GradScaler()

    # Model fine-tuning
    history = []
    epochs = cfg.pipeline.finetuning.get("epochs", 5)
    print(f"Fine-tuning {cfg.model.name}...")
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
            history.pop()
            break

    print(f"Model {cfg.model.name} fine-tuned successfully!")

    # Saving the model
    torch.save(model.state_dict(), out_dir / "model.pth")

    # Saving training and validation loss in loss.json file
    with open(out_dir / "loss.json", "w") as f:
        json.dump(history, f, indent=4)


if __name__ == "__main__":
    main()
