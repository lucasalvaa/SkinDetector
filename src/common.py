"""Common utilities for the P1 pipeline, including data loading and model setup."""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
def get_dataloaders(data_path: Path, batch_size: int) -> Dict[str, DataLoader]:
    """Create data loaders with a configurable batch size.

    Args:
        data_path: Path to the training data.
        batch_size: Number of images per batch.

    Returns:
        Dictionary containing DataLoaders for each split.

    """
    root = data_path.parent
    tfs = transforms.Compose(
        [
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return {
        x: DataLoader(
            datasets.ImageFolder(root / x, tfs),
            batch_size=batch_size,
            shuffle=(x == "train"),
        )
        for x in ["train", "val", "test"]
    }
'''


def get_dataloader(image_res: int, data_path: Path, batch_size: int) -> DataLoader:
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
        num_workers=4,  # Consiglio: accelera il caricamento dati
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
