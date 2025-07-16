"""
data_curation.py: Data Curation Script for Face Mask Detection

This script scans new detection images in data/collected/ and allows for manual or automated approval before adding them to the curated set. Optionally copies label files. Supports both interactive and non-interactive (auto) modes.
"""

from pathlib import Path
import shutil
import argparse

COLLECTED_DIR = Path("data/collected/")
CURATED_DIR = Path("data/curated/")
CURATED_DIR.mkdir(parents=True, exist_ok=True)


def curate_images(auto_approve=False, skip_all=False):
    print("=== Data Curation ===")
    images = list(COLLECTED_DIR.glob("**/*.*"))
    images = [f for f in images if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    approved, skipped = 0, 0
    for img_file in images:
        label_file = img_file.with_suffix(".txt")
        if skip_all:
            print(f"Skipped: {img_file}")
            skipped += 1
            continue
        approve = 'y' if auto_approve else input(f"Approve {img_file}? (y/n): ").strip().lower()
        if approve == 'y':
            shutil.copy2(img_file, CURATED_DIR / img_file.name)
            if label_file.exists():
                shutil.copy2(label_file, CURATED_DIR / label_file.name)
            print(f"  -> Approved and copied to {CURATED_DIR}")
            approved += 1
        else:
            print("  -> Skipped.")
            skipped += 1
    print(f"Curation complete. Approved: {approved}, Skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser(description="Curate new detection images for training.")
    parser.add_argument('--auto', action='store_true', help='Automatically approve all images')
    parser.add_argument('--skip-all', action='store_true', help='Skip all images (dry run)')
    args = parser.parse_args()
    curate_images(auto_approve=args.auto, skip_all=args.skip_all)

if __name__ == "__main__":
    main()