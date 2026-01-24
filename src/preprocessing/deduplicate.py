"""Module for dataset deduplication based on MD5 hashing."""

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml


def get_image_hash(filepath: Path) -> str:
    """Generate an MD5 hash for a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def scan_dataset(root_dir: Path) -> dict[str, list[Path]]:
    """Scan subdirectories and map ALL hashes to file paths."""
    hashes = defaultdict(list)

    if not root_dir.exists():
        print(f"[-] Error: Input directory '{root_dir}' does not exist.")
        sys.exit(1)

    categories = [d for d in root_dir.iterdir() if d.is_dir()]
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    print(f"[*] Scanning directories in '{root_dir}'...")

    for cat_path in categories:
        for file_path in cat_path.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                file_hash = get_image_hash(file_path)
                hashes[file_hash].append(file_path)

    return hashes


def generate_csv_report(
        duplicates: dict[str, list[Path]], report_path: Path
) -> None:
    """Generate a CSV report listing all duplicates found."""
    print(f"[*] Generating detailed CSV report at '{report_path}'...")

    header = ["Hash", "Issue_Type", "Class", "File_Path"]

    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for file_hash, paths in duplicates.items():
            classes_involved = {p.parent.name for p in paths}

            if len(classes_involved) > 1:
                issue_type = "CROSS_CLASS_CONFLICT"
            else:
                issue_type = "SAME_CLASS_REDUNDANCY"

            for p in paths:
                writer.writerow([file_hash, issue_type, p.parent.name, str(p)])


def create_clean_dataset(
        all_hashes: dict[str, list[Path]], src_root: Path, dest_root: Path
) -> dict:
    """Copy valid files to a new directory."""
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True)

    stats = {
        "unique_files_copied": 0,
        "redundant_files_skipped": 0,
        "cross_class_conflict_skipped": 0,
        "removed_per_class": defaultdict(int),
    }

    print(f"[*] Creating clean dataset in '{dest_root}'...")

    for _, paths in all_hashes.items():
        classes_involved = {p.parent.name for p in paths}

        # Case 1: Cross-class duplicates -> SKIP ALL
        if len(classes_involved) > 1:
            stats["cross_class_conflict_skipped"] += len(paths)
            for p in paths:
                stats["removed_per_class"][p.parent.name] += 1
            continue

        # Case 2: Same-class duplicates -> COPY ONE
        if len(paths) > 1:
            num_redundant = len(paths) - 1
            stats["redundant_files_skipped"] += num_redundant
            stats["removed_per_class"][paths[0].parent.name] += num_redundant
            paths_to_copy = [paths[0]]
        else:
            # Case 3: Unique -> COPY
            stats["unique_files_copied"] += 1
            paths_to_copy = paths

        # Copy
        for src_path in paths_to_copy:
            rel_path = src_path.relative_to(src_root)
            dest_path = dest_root / rel_path

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dest_path)

    return stats


def generate_summary_report(stats: dict, report_path: Path) -> None:
    """Write the final summary statistics to a JSON file."""
    print(f"[*] Saving summary report to '{report_path}'...")

    total_removed = (
        stats['cross_class_conflict_skipped'] + stats['redundant_files_skipped']
    )

    summary_data = {
        "valid_images_preserved": stats['unique_files_copied'],
        "images_removed_cross_class": stats['cross_class_conflict_skipped'],
        "images_removed_redundancy": stats['redundant_files_skipped'],
        "total_removed": total_removed,
        "removals_per_class": dict(stats["removed_per_class"]),
    }

    with open(report_path, "w") as f:
        json.dump(summary_data, f, indent=4)


def main() -> None:
    """Entry point supporting both Config file and CLI arguments."""
    parser = argparse.ArgumentParser(description="Dataset Deduplication")

    parser.add_argument(
        "--config", type=str, default=None, help="Path to params.yaml"
    )

    parser.add_argument(
        "--input", type=str, default=None, help="Input directory (raw data)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (deduplicated data)"
    )

    args = parser.parse_args()

    if args.config:
        print(f"[*] Loading configuration from {args.config}")
        with open(args.config) as f:
            config = yaml.safe_load(f)

        try:
            input_dir = Path(config["data"]["raw_path"])
            output_dir = Path(config["data"]["dedup_path"])
        except KeyError as e:
            print(f"[-] Error: Key {e} not found in {args.config}")
            sys.exit(1)

    elif args.input and args.output:
        print("[*] Using command line arguments")
        input_dir = Path(args.input)
        output_dir = Path(args.output)

    else:
        parser.error("You must provide either --config OR both --input and --output.")

    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)

    csv_report = report_dir / "duplicates_log.csv"
    summary_report = report_dir / "deduplication_summary.json"

    all_hashes = scan_dataset(input_dir)

    duplicates_only = {h: p for h, p in all_hashes.items() if len(p) > 1}
    generate_csv_report(duplicates_only, csv_report)

    stats = create_clean_dataset(all_hashes, input_dir, output_dir)
    generate_summary_report(stats, summary_report)

    print("\n[+] Deduplication complete.")


if __name__ == "__main__":
    main()
