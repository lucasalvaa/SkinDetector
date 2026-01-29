import shutil
from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import train_test_split

# Constants
INPUT_DIR = Path("data/dedup")
OUTPUT_DIR = Path("data/split")
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def get_image_files(class_dir: Path) -> List[Path]:
    """Retrieve all image files from a specific class directory.

    Args:
        class_dir: The directory path for a specific class.

    Returns:
        A list of Path objects pointing to the images.

    """
    valid_extensions = {".jpg", ".jpeg", ".png"}
    return [
        f
        for f in class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]


def split_data(
    files: List[Path],
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split a list of files into train, validation, and test sets.

    Logic:
        1. Split files into Train (70%) and Temp (30%).
        2. Split Temp into Validation (50% of Temp) and Test (50% of Temp).
        Result: 70% Train, 15% Val, 15% Test.

    Args:
        files: List of file paths to split.

    Returns:
        A tuple containing three lists: (train_files, val_files, test_files).

    """
    # First split: Separate Training set
    train_files, temp_files = train_test_split(
        files,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=SEED,
        shuffle=True,
    )

    # Second split: Separate Validation and Test sets from the remaining data
    # We split 50/50 because 0.15 is half of 0.30
    val_files, test_files = train_test_split(
        temp_files,
        test_size=0.5,
        random_state=SEED,
        shuffle=True,
    )

    return train_files, val_files, test_files


def copy_files(files: List[Path], destination_dir: Path) -> None:
    """Copy a list of files to the destination directory.

    Args:
        files: List of source file paths.
        destination_dir: The target directory where files will be copied.

    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    for file_path in files:
        shutil.copy2(file_path, destination_dir / file_path.name)


def process_class(class_path: Path) -> None:
    """Process a single class directory: split and copy files.

    Args:
        class_path: Path to the specific class directory in raw data.

    """
    class_name = class_path.name
    images = get_image_files(class_path)

    if not images:
        print(f"Warning: No images found in {class_name}")
        return

    train, val, test = split_data(images)

    # Define sub-directories
    splits = {
        "train": train,
        "val": val,
        "test": test,
    }

    print(f"Processing '{class_name}': {len(images)} images")
    print(f" -> Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    for split_name, split_files in splits.items():
        dest_path = OUTPUT_DIR / split_name / class_name
        copy_files(split_files, dest_path)


def main() -> None:
    """Execute the dataset splitting pipeline."""
    if not INPUT_DIR.exists():
        print(f"Error: Directory {INPUT_DIR} does not exist.")
        return

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for item in INPUT_DIR.iterdir():
        if item.is_dir():
            process_class(item)

    print(f"\nDataset split completed successfully in '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
