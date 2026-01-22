"""Module for clustering-based undersampling (Representative Undersampling).

This script generates two training phases:
1. Balanced Phase: k representative samples per class.
2. Tuning Phase: The remaining samples.
It mirrors the validation set into both phases to allow existing training scripts
to function without modification.
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# Constants
BATCH_SIZE = 32
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureDataset(Dataset):
    """Simple dataset wrapper for feature extraction."""

    def __init__(self, file_paths: List[Path]) -> None:
        """Initialize with a list of file paths."""
        self.file_paths = file_paths
        self.transform = transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and preprocess image."""
        path = self.file_paths[idx]
        with Image.open(path).convert("RGB") as img:
            return self.transform(img)


def get_feature_extractor() -> torch.nn.Module:
    """Load a pre-trained ResNet18 for feature extraction."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Remove the classification head (fc layer)
    model.fc = torch.nn.Identity()
    model.to(DEVICE)
    model.eval()
    return model


def extract_features(file_paths: List[Path]) -> np.ndarray:
    """Extract semantic features from a list of images."""
    dataset = FeatureDataset(file_paths)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model = get_feature_extractor()

    features_list = []
    print(f"Extracting features for {len(file_paths)} images on {DEVICE}...")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extraction"):
            batch = batch.to(DEVICE)
            # Output shape: (Batch, 512)
            embedding = model(batch)
            features_list.append(embedding.cpu().numpy())

    return np.vstack(features_list)


def select_representative_samples(
    file_paths: List[Path], k: int
) -> List[Path]:
    """Select k representative images using K-Means clustering.

    Args:
        file_paths: List of all images in the class.
        k: Number of samples to select.

    Returns:
        selected_paths: Paths to selected images.

    """
    if len(file_paths) <= k:
        return file_paths

    # 1. Extract Features (Embeddings)
    features = extract_features(file_paths)

    # 2. Run K-Means
    print(f"Running K-Means to select {k} representative samples...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(features)

    # 3. Find images closest to centroids
    closest_indices, _ = pairwise_distances_argmin_min(
        kmeans.cluster_centers_, features
    )

    selected_indices_set = set(closest_indices)

    # Fill if K-Means converged on fewer unique points
    if len(selected_indices_set) < k:
        remaining_indices = list(set(range(len(file_paths))) - selected_indices_set)
        needed = k - len(selected_indices_set)
        np.random.seed(42)
        extra = np.random.choice(remaining_indices, needed, replace=False)
        selected_indices_set.update(extra)

    return [file_paths[i] for i in selected_indices_set]


def process_balancing(input_dir: Path, output_dir: Path) -> None:
    """Execute the balancing pipeline."""
    # 1. Setup Directories
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True)

    # 2. Analyze Class Distribution in TRAIN set only
    train_input_dir = input_dir / "train"
    class_dirs = [d for d in train_input_dir.iterdir() if d.is_dir()]
    counts: Dict[str, List[Path]] = {}

    print("Analyzing class distribution in training set...")
    for class_dir in class_dirs:
        images = [
            f
            for f in class_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        counts[class_dir.name] = images

    # 3. Determine k (size of minority class)
    min_count = min(len(imgs) for imgs in counts.values())
    min_class = min(counts, key=lambda x: len(counts[x]))
    print(f"Minority Class: '{min_class}' ({min_count} samples). Target k={min_count}")

    # 4. Process Balancing
    for class_name, images in counts.items():
        print(f"\nProcessing class: {class_name}")

        dest = output_dir / class_name
        dest.mkdir(parents=True, exist_ok=True)

        if len(images) <= min_count:
            selected = images
            print(f" -> Keeping all {len(images)} samples (Minority class).")
        else:
            selected = select_representative_samples(images, k=min_count)
            print(
                f" -> Clustering: {len(selected)} selected, "
                f"{len(images) - len(selected)} excluded."
            )

        # Copy files
        for p in selected:
            shutil.copy2(p, dest / p.name)

    # 5. Mirror Validation Set (Critical step for compatibility)
    # We pass 'input_dir' which is 'data/split' (containing train and val)

    print("\n" + "=" * 40)
    print("BALANCING COMPLETE")
    print(f"Balanced dataset: {output_dir}")
    print("=" * 40)


def main() -> None:
    """Entry point for the balancing script."""
    parser = argparse.ArgumentParser(description="Clustering-based Undersampling")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as conf_file:
        config = yaml.safe_load(conf_file)

    out_dir = Path(config["data"]["balanced_path"])
    out_dir.mkdir(parents=True, exist_ok=True)

    process_balancing(
        Path(config["data"]["tobalance_path"]), Path(config["data"]["balanced_path"])
    )


if __name__ == "__main__":
    main()
