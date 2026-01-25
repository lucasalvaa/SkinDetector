"""Common utilities."""

from pathlib import Path

import torch
import torch.optim as optim
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloader(
    data_path: Path, batch_size: int, image_res: int = 224
) -> DataLoader:
    """Create a single data loader with a configurable batch size.

    Args:
        image_res: Resolution to which the image should be resized
        data_path: Path to the data split.
        batch_size: Number of images per batch.

    Returns:
        DataLoader to the split.

    """
    tfs = transforms.Compose(
        [
            transforms.Resize((image_res, image_res)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    dataset = datasets.ImageFolder(data_path, tfs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=("train" in str(data_path)),
        num_workers=4,  # Accelera il caricamento dati
        pin_memory=True,  # Accelera il trasferimento dati alla GPU
    )


def get_model(model_name: str, num_classes: int) -> nn.Module:
    """Initialize the chosen architecture with pre-trained weights.

    Args:
        model_name: Identifier for the architecture.
        num_classes: Number of output neurons.

    Returns:
        The model moved to the appropriate device.

    """
    if "effnet" in model_name:
        weights = {
            "effnet_s": models.EfficientNet_V2_S_Weights.DEFAULT,
            "effnet_m": models.EfficientNet_V2_M_Weights.DEFAULT,
        }[model_name]
        model = getattr(models, f"efficientnet_v2_{model_name[-1]}")(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:  # convnext
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    return model.to(DEVICE)


def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
    """Calculate average loss on the validation set.

    Args:
        model: The neural network model.
        loader: DataLoader for the validation set.
        criterion: Loss function (e.g., CrossEntropyLoss or LDAMLoss).

    Returns:
        The average loss over the entire dataset.

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


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
) -> float:
    """Run one training epoch with AMP.

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

        with autocast(device_type=DEVICE.type):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)
