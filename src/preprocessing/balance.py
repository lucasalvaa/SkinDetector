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
) -> Tuple[List[Path], List[Path]]:
    """Select k representative images using K-Means clustering.

    Args:
        file_paths: List of all images in the class.
        k: Number of samples to select.

    Returns:
        Tuple (selected_paths, remaining_paths).

    """
    if len(file_paths) <= k:
        return file_paths, []

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

    selected_paths = [file_paths[i] for i in selected_indices_set]
    remaining_paths = [
        file_paths[i] for i in range(len(file_paths)) if i not in selected_indices_set
    ]

    return selected_paths, remaining_paths


def setup_phase_directories(output_dir: Path) -> Tuple[Path, Path]:
    """Create directory structure for Phase 1 and Phase 2."""
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Structure: data/balanced/phase1/train
    phase1_train = output_dir / "phase1" / "train"
    phase2_train = output_dir / "phase2" / "train"

    phase1_train.mkdir(parents=True)
    phase2_train.mkdir(parents=True)

    return phase1_train, phase2_train


def copy_validation_set(input_dir: Path, output_dir: Path) -> None:
    """Copy the validation set to both phases to satisfy train.py requirements."""
    src_val = input_dir / "val"

    if not src_val.exists():
        print("Warning: Validation set not found. Training script might fail.")
        return

    print("Mirroring validation set to Phase 1 and Phase 2...")
    shutil.copytree(src_val, output_dir / "phase1" / "val")
    shutil.copytree(src_val, output_dir / "phase2" / "val")


def process_balancing(input_dir: Path, output_dir: Path) -> None:
    """Execute the balancing pipeline."""
    # 1. Setup Directories
    p1_train_dir, p2_train_dir = setup_phase_directories(output_dir)

    # 2. Analyze Class Distribution in TRAIN set only
    train_input_dir = input_dir / "train"
    class_dirs = [d for d in train_input_dir.iterdir() if d.is_dir()]
    counts: Dict[str, List[Path]] = {}

    print("Analyzing class distribution in training set...")
    for class_dir in class_dirs:
        images = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        counts[class_dir.name] = images

    # 3. Determine k (size of minority class)
    min_count = min(len(imgs) for imgs in counts.values())
    min_class = min(counts, key=lambda x: len(counts[x]))
    print(f"Minority Class: '{min_class}' ({min_count} samples). Target k={min_count}")

    # 4. Process Balancing
    for class_name, images in counts.items():
        print(f"\nProcessing class: {class_name}")

        dest_t1 = p1_train_dir / class_name
        dest_t2 = p2_train_dir / class_name
        dest_t1.mkdir(parents=True, exist_ok=True)
        dest_t2.mkdir(parents=True, exist_ok=True)

        if len(images) <= min_count:
            selected, remaining = images, []
            print(f" -> Keeping all {len(images)} samples (Minority).")
        else:
            selected, remaining = select_representative_samples(images, k=min_count)
            print(
                f" -> Clustering: {len(selected)} to Phase 1, "
                f"{len(remaining)} to Phase 2."
            )

        # Copy files
        for p in selected:
            shutil.copy2(p, dest_t1 / p.name)
        for p in remaining:
            shutil.copy2(p, dest_t2 / p.name)

    # 5. Mirror Validation Set (Critical step for compatibility)
    # We pass 'input_dir' which is 'data/split' (containing train and val)
    copy_validation_set(input_dir, output_dir)

    print("\n" + "=" * 40)
    print("BALANCING COMPLETE")
    print(f"Phase 1 Dataset: {output_dir / 'phase1'}")
    print(f"Phase 2 Dataset: {output_dir / 'phase2'}")
    print("=" * 40)


def main() -> None:
    """Entry point for the balancing script."""
    parser = argparse.ArgumentParser(description="Clustering-based Undersampling")
    parser.add_argument("--input", type=str, required=True, help="Path to data/split")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    args = parser.parse_args()

    process_balancing(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
