"""Fine-tuning script for the best performing model using balanced data phase 2."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import amp, nn, optim
from torch.utils.data import DataLoader

# Assumendo che la struttura dei package sia corretta rispetto alla root
from src.common import DEVICE, get_dataloader, get_model


def find_best_model(models_list: list[str], base_path: Path = Path(".")) -> str:
    """Identify the model with the highest top1 metric.

    Args:
        models_list: List of model names to check.
        base_path: Directory containing model folders.

    Returns:
        The name of the best model.

    """
    best_top1 = -1.0
    best_model_name = ""

    for model_name in models_list:
        metrics_path = base_path / model_name / "metrics.json"
        if metrics_path.exists():
            with metrics_path.open("r") as f:
                data = json.load(f)
                if data["top1"] > best_top1:
                    best_top1 = data["top1"]
                    best_model_name = model_name

    if not best_model_name:
        raise FileNotFoundError(
            "No valid metrics.json found to determine the best model."
        )

    return best_model_name


def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
    """Calculate average loss on the validation set.

    Args:
        model: The neural network model.
        loader: DataLoader for validation.
        criterion: Loss function.

    Returns:
        Average validation loss.

    """
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
    """Run one fine-tuning epoch using AMP.

    Args:
        model: The neural network model.
        loader: DataLoader for the training set.
        criterion: Loss function.
        optimizer: Optimizer.
        scaler: GradScaler for AMP.

    Returns:
        Average training loss.

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
    """Execute the fine-tuning pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    with open(args.config) as conf_file:
        config: dict[str, Any] = yaml.safe_load(conf_file)

    # 2. Setup directory di output (sovrascriviamo o creiamo una cartella fine_tuned)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Caricamento dati (Phase 2 per fine-tuning)
    # Nota: Assumiamo che phase2 sia la directory dei dati di training bilanciati
    train_loader = get_dataloader(
        data_path=Path(config["finetuning"]["data_path"]),
        batch_size=config["finetuning"]["batch_size"],
    )

    # Usiamo il set di validazione originale per il monitoraggio
    val_loader = get_dataloader(
        data_path=Path(config["data"]["valset_path"]),
        batch_size=config["finetuning"]["batch_size"],
    )

    # 4. Inizializzazione modello e caricamento pesi precedenti
    model = get_model(args.model, len(train_loader.dataset.classes))
    weights_path = out_dir.parent / "model.pth"
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)

    # 5. Configurazione training
    # Per il fine-tuning si usa solitamente un Learning Rate più basso (es. 1e-5 o 1e-4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["finetuning"]["lr"])
    scaler = amp.GradScaler()

    history = []

    # 6. Loop di fine-tuning
    epochs = config["finetuning"]["epochs"]
    for epoch in range(epochs):
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        v_loss = validate(model, val_loader, criterion)

        history.append({"epoch": epoch + 1, "train_loss": t_loss, "val_loss": v_loss})

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"FT Train Loss: {t_loss:.4f} | "
            f"FT Val Loss: {v_loss:.4f}"
        )

    # 7. Salvataggio artefatti
    torch.save(model.state_dict(), out_dir / "model.pth")
    with open(out_dir / "loss.json", "w") as f:
        json.dump(history, f, indent=4)


if __name__ == "__main__":
    main()
