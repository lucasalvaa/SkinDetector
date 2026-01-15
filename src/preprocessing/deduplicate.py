import hashlib
import os
from collections import defaultdict


def get_image_hash(filepath: str) -> str:
    """Generate an MD5 hash for a file to identify identical content.

    Args:
        filepath: The path to the image file.

    Return:
        The hexadecimal MD5 hash of the file.

    """
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def find_duplicates(root_dir: str) -> dict[str, list[str]]:
    """Scan subdirectories and map hashes to file paths.

    Args:
        root_dir: The root directory containing disease subdirectories.

    Return:
        A dictionary mapping hashes to lists of duplicate file paths.

    """
    hashes = defaultdict(list)
    categories = [
        "demodicosis",
        "dermatitis",
        "fungal_infections",
        "healthy",
        "hypersensitivity",
        "ringworm",
    ]

    print(f"Scanning directories in {root_dir}...")

    for category in categories:
        cat_path = os.path.join(root_dir, category)
        if not os.path.exists(cat_path):
            continue

        for filename in os.listdir(cat_path):
            file_path = os.path.join(cat_path, filename)
            if os.path.isfile(file_path):
                file_hash = get_image_hash(file_path)
                hashes[file_hash].append(file_path)

    return {h: paths for h, paths in hashes.items() if len(paths) > 1}


def run_deduplication(dir_name: str) -> None:  # noqa: C901
    """Find duplicate images, save a report and prompt for deletion.

    This function identifies duplicates, tracks cross-class inconsistencies,
    and logs the number of removals per category.

    Args:
        dir_name: The name of the subdirectory within "data" to search.

    """
    root_data_dir = os.path.join("../../data", dir_name)
    report_path = "report/duplicates.txt"
    os.makedirs("report", exist_ok=True)

    duplicates = find_duplicates(root_data_dir)

    if not duplicates:
        print("No duplicate images found.")
        return

    files_to_delete: list[str] = []

    inconsistent_cross_class_count = 0
    removed_per_class = defaultdict(int)
    total_deleted_per_class = defaultdict(int)

    with open(report_path, "w") as f:
        f.write("DUPLICATE IMAGES REPORT\n" + "=" * 24 + "\n")

        for h, paths in duplicates.items():
            f.write(f"\nHash: {h}\n")

            for p in paths:
                f.write(f" - {p}\n")

            classes_involved = {os.path.basename(os.path.dirname(p)) for p in paths}

            if len(classes_involved) > 1:
                # Cross-class duplicates
                inconsistent_cross_class_count += len(paths)
                f.write(" [!] CROSS-CLASS ERROR: Marking ALL for deletion.\n")
                files_to_delete.extend(paths)
                for p in paths:
                    total_deleted_per_class[os.path.basename(os.path.dirname(p))] += 1
            else:
                # Same-class duplicates
                current_class = next(iter(classes_involved))
                duplicates_in_folder = paths[1:]

                f.write(f" [i] Same-class duplicates: Keeping {paths[0]}\n")

                removed_per_class[current_class] += len(duplicates_in_folder)
                total_deleted_per_class[current_class] += len(duplicates_in_folder)
                files_to_delete.extend(duplicates_in_folder)

        # Add deduplication stats in the report
        f.write("\n" + "=" * 30 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 30 + "\n")
        f.write(
            f"Inconsistent cross-class images deleted: "
            f"{inconsistent_cross_class_count}\n\n"
        )

        f.write("Redundant duplicates removed per class (same directory):\n")
        for category, count in removed_per_class.items():
            f.write(f" - {category}: {count}\n")

        f.write("\nTotal images to be deleted per class (inconsistent + redundant):\n")
        for category, count in total_deleted_per_class.items():
            f.write(f" - {category}: {count}\n")
        f.write(f"\nTOTAL FILES TO BE DELETED: {len(files_to_delete)}\n")

    print(f"\nFound {len(duplicates)} groups of duplicate hashes.")
    print(f"Total files marked for deletion: {len(files_to_delete)}")
    print(f"Summary saved to: {report_path}")

    if not files_to_delete:
        return

    confirm = input(
        f"Proceed with deletion of {len(files_to_delete)} files? [y/N]: "
    ).lower()

    if confirm == "n":
        print("\nOperation cancelled.")
        return

    for path in files_to_delete:
        try:
            os.remove(path)
        except OSError as e:
            print(f"Error deleting {path}: {e}")
    print("\nClean-up complete.")


if __name__ == "__main__":
    run_deduplication("raw")
