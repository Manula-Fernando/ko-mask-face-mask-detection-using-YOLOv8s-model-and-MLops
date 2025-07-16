import argparse
from pathlib import Path

def count_samples(data_dir: str) -> None:
    data_path = Path(data_dir)
    if not data_path.exists():
        print(0)
        return
    # Count all .jpg and .png files recursively
    count = sum(1 for _ in data_path.rglob("*.jpg")) + sum(1 for _ in data_path.rglob("*.png"))
    print(count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count training samples in a directory.")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to data directory")
    args = parser.parse_args()
    count_samples(args.data_dir)
