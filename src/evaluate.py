"""Evaluation script for P1: calculates metrics and generates plots."""

import argparse
import json
import yaml
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader

from src.common import DEVICE, get_dataloader, get_model


def evaluate(
    model: torch.nn.Module, loader: DataLoader
) -> Tuple[float, float, List[int], List[int]]:
    """Evaluate model and return metrics and predictions.

    Args:
        model: The trained neural network.
        loader: DataLoader for the test set.

    Returns:
        A tuple with (top1_acc, top3_acc, true_labels, predictions).

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
    return top1 / size, top3 / size, all_labels, all_preds


def main() -> None:
    """Run test evaluation and save artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as conf_file:
        config = yaml.safe_load(conf_file)

    out_dir = Path(args.model)
    test_loader = get_dataloader(image_res=config["base"]["image_res"],
                   data_path=Path(config["data"]["testset_path"]),
                   batch_size=config["evaluate"]["batch_size"])
    classes = test_loader.dataset.classes

    model = get_model(args.model, len(classes))
    model.load_state_dict(torch.load(out_dir / "model.pth", weights_only=True))

    t1, t3, labels, preds = evaluate(model, test_loader)

    # Save Metrics
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"top1": t1 * 100, "top3": t3 * 100}, f, indent=4)

    cm_data = [
        {"actual": classes[l], "predicted": classes[p]}
        for l, p in zip(labels, preds)
    ]

    with open(out_dir / "cm_data.json", "w") as f:
        json.dump(cm_data, f, indent=4)

'''
    # Plot Confusion Matrix
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    _fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(xticks_rotation="vertical", ax=ax)
    plt.savefig(out_dir / "test_confusion_matrix.png")
'''


if __name__ == "__main__":
    main()
