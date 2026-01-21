"""Module for offline data augmentation with randomized strategies.

Apply exactly 2 out of 4 possible transformations to training images.
"""

import argparse
import random
import shutil
import yaml
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


def apply_random_strategy(img: Image.Image) -> Image.Image:
    """Applica esattamente 2 trasformazioni random a un oggetto immagine PIL."""
    transformations = [
        apply_gaussian_noise,
        apply_saturation,
        apply_horizontal_flip,
        apply_vertical_flip,
    ]
    selected_transforms = random.sample(transformations, 2)
    for transform_func in selected_transforms:
        img = transform_func(img)
    return img


def process_dataset(input_dir: Path, output_dir: Path) -> None:
    """Processa il dataset: triplica il train, copia val/test."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    splits = ["train", "val", "test"]

    for split in splits:
        src_split = input_dir / split
        dst_split = output_dir / split

        if not src_split.exists():
            continue

        print(f"Processing split: {split}...")

        for class_dir in src_split.iterdir():
            if not class_dir.is_dir():
                continue

            dst_class_dir = dst_split / class_dir.name
            dst_class_dir.mkdir(parents=True, exist_ok=True)
            images = list(class_dir.glob("*"))

            for img_path in tqdm(images, desc=f"{split}/{class_dir.name}", leave=False):
                # 1. Copia l'immagine originale (Sempre)
                shutil.copy2(img_path, dst_class_dir / img_path.name)

                # 2. Genera versioni aumentate solo per il TRAIN
                if split == "train":
                    try:
                        with Image.open(img_path) as img:
                            img = img.convert("RGB")

                            # Crea 2 varianti diverse
                            for i in range(1, 3):
                                augmented_img = apply_random_strategy(img.copy())
                                # Cambia il nome per non sovrascrivere l'originale
                                # Esempio: foto1.jpg -> aug1_foto1.jpg
                                new_name = f"aug{i}_{img_path.name}"
                                augmented_img.save(dst_class_dir / new_name, quality=95)
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as conf_file:
        config = yaml.safe_load(conf_file)

    process_dataset(
        Path(config["data"]["inputset_path"]),
        Path(config["data"]["augmentedset_path"])
    )


if __name__ == "__main__":
    main()
