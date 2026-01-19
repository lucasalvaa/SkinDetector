"""Module for offline data augmentation with randomized strategies.

Apply exactly 2 out of 4 possible transformations to training images.
"""

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm


def apply_gaussian_noise(image: Image.Image) -> Image.Image:
    """Add Gaussian noise to a PIL image."""
    mean = 0.0
    std = 25.0
    img_array = np.array(image).astype(np.float32)
    noise = np.random.normal(mean, std, img_array.shape)
    img_noised = img_array + noise
    img_noised = np.clip(img_noised, 0, 255).astype(np.uint8)
    return Image.fromarray(img_noised)


def apply_saturation(image: Image.Image) -> Image.Image:
    """Apply random saturation change."""
    enhancer = ImageEnhance.Color(image)
    # Factor: 0.5 (desaturated) to 1.5 (supersaturated)
    factor = np.random.uniform(0.5, 1.5)
    return enhancer.enhance(factor)


def apply_horizontal_flip(image: Image.Image) -> Image.Image:
    """Flip image horizontally."""
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def apply_vertical_flip(image: Image.Image) -> Image.Image:
    """Flip image vertically."""
    return image.transpose(Image.FLIP_TOP_BOTTOM)


def apply_random_strategy(image_path: Path, output_path: Path) -> None:
    """Select 2 out of 4 augmentations and save the image."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")

            # Define the pool of available transformations
            transformations = [
                apply_gaussian_noise,
                apply_saturation,
                apply_horizontal_flip,
                apply_vertical_flip,
            ]

            # Select exactly 2 transformations randomly
            selected_transforms = random.sample(transformations, 2)

            # Apply them sequentially
            for transform_func in selected_transforms:
                img = transform_func(img)

            img.save(output_path, quality=95)
    except (OSError, ValueError) as e:
        print(f"Error processing {image_path}: {e}")


def process_dataset(input_dir: Path, output_dir: Path) -> None:
    """Process the dataset: augment train, copy val/test."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    splits = ["train", "val", "test"]

    for split in splits:
        src_split = input_dir / split
        dst_split = output_dir / split

        if not src_split.exists():
            print(f"Warning: Split {split} not found in {input_dir}")
            continue

        print(f"Processing split: {split}...")

        for class_dir in src_split.iterdir():
            if not class_dir.is_dir():
                continue

            dst_class_dir = dst_split / class_dir.name
            dst_class_dir.mkdir(parents=True, exist_ok=True)

            images = list(class_dir.glob("*"))

            for img_path in tqdm(images, desc=f"{split}/{class_dir.name}", leave=False):
                dst_path = dst_class_dir / img_path.name

                if split == "train":
                    apply_random_strategy(img_path, dst_path)
                else:
                    shutil.copy2(img_path, dst_path)


def main() -> None:
    """Entry point for augmentation."""
    parser = argparse.ArgumentParser(description="Randomized Data Augmentation Tool")
    parser.add_argument("--input", type=str, required=True, help="Input split path")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    args = parser.parse_args()

    process_dataset(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
