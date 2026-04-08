from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mse433_project.data import build_master_dataset


if __name__ == "__main__":
    dataset = build_master_dataset(write_outputs=True)
    print(f"Wrote master dataset with {dataset.shape[0]:,} hourly rows.")

