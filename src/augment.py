"""Module for offline data augmentation with randomized strategies.

Apply exactly 2 out of 4 possible transformations to training images.
"""

import random
import shutil
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
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


def apply_saturation(
    image: Image.Image,
) -> Image.Image:
    """Apply random saturation change."""
    lower_bound = 0.5
    upper_bound = 1.5
    enhancer = ImageEnhance.Color(image)
    # Factor: 0.5 (desaturated) to 1.5 (supersaturated)
    factor = np.random.uniform(lower_bound, upper_bound)
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


def augment_dataset(src_dir: Path, dst_dir: Path) -> None:
    """Processa il dataset: triplica il train."""
    for class_dir in src_dir.iterdir():
        if not class_dir.is_dir():
            continue

        dst_class_dir = dst_dir / class_dir.name
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        images = list(class_dir.glob("*"))
        for img_path in tqdm(images, desc=f"train/{class_dir.name}", leave=False):
            # Copy the original image
            shutil.copy2(img_path, dst_class_dir / img_path.name)

            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")

                    # Create two variants randomly
                    for i in range(1, 3):
                        augmented_img = apply_random_strategy(img.copy())
                        # Change the name so as not to overwrite the original
                        new_name = f"aug{i}_{img_path.name}"
                        augmented_img.save(dst_class_dir / new_name, quality=95)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run data augmentation."""
    root = Path(hydra.utils.get_original_cwd())  # Project root directory
    src_dir = root / cfg.pipeline.augment.src_dir / "train"  # Source directory
    dst_dir = root / cfg.pipeline.augment.dst_dir / "train"  # Destination directory

    if dst_dir.exists() and not cfg.pipeline.augment.force:
        return

    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    augment_dataset(src_dir, dst_dir)


if __name__ == "__main__":
    main()
