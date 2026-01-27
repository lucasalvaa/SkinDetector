"""Evaluation script for P1: calculates metrics and generates plots."""

import json
from pathlib import Path
from typing import List, Tuple

import hydra
import torch
from omegaconf import DictConfig
from sklearn.metrics import precision_score
from torch.utils.data import DataLoader

from src.common import DEVICE, get_dataloader, get_model


def evaluate(
    model: torch.nn.Module, loader: DataLoader
) -> Tuple[float, float, float, List[int], List[int]]:
    """Evaluate model and return metrics and predictions.

    Args:
        model: The trained neural network.
        loader: DataLoader for the test set.

    Returns:
        A tuple with (top1_acc, top3_acc, precision, true_labels, predictions).

    """
    model.eval()
    all_preds, all_labels = [], []
    top1, top3 = 0.0, 0.0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, pred = outputs.topk(3, 1, True, True)

            all_preds.extend(pred[:, 0].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct = pred.eq(labels.view(-1, 1).expand_as(pred))
            top1 += correct[:, 0].sum().item()
            top3 += correct[:, :3].any(dim=1).sum().item()

    size = len(loader.dataset)

    # average='macro' è standard per il multiclasse (media non pesata delle classi)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)

    return top1 / size, top3 / size, precision, all_labels, all_preds


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run test evaluation and save artifacts."""
    root = Path(hydra.utils.get_original_cwd())  # Project Root
    out_dir = root / cfg.pipeline.out_dir / cfg.model.name
    if cfg.pipeline.tsft:
        out_dir = out_dir / "finetuned"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_loader = get_dataloader(
        image_res=cfg.base.image_res,
        data_path=root / cfg.data.testset_path,
        batch_size=cfg.evaluate.batch_size,
    )

    classes = test_loader.dataset.classes

    model = get_model(
        cfg.model.fullname,
        cfg.model.weights,
        cfg.model.layer,
        len(test_loader.dataset.classes),
    ).to(DEVICE)

    weights_path = out_dir / "model.pth"

    # weights_path = Path(args.model_path) if args.model_path else out_dir / "model.pth"
    print(f"[*] Loading weights from: {weights_path}")

    model.load_state_dict(
        torch.load(weights_path, map_location=DEVICE, weights_only=True)
    )

    t1, t3, prec, labels, preds = evaluate(model, test_loader)

    # Save Metrics
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(
            {"top1": t1 * 100, "top3": t3 * 100, "precision": prec * 100}, f, indent=4
        )

    import csv

    cm_data = [
        {"actual": classes[label], "predicted": classes[pred]}
        for label, pred in zip(labels, preds, strict=True)
    ]

    output_path = out_dir / "cm_data.csv"

    # CSV file for confusion matrix
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["actual", "predicted"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(cm_data)


if __name__ == "__main__":
    main()
