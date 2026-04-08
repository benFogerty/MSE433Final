from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mse433_project.reporting import render_report


if __name__ == "__main__":
    path = render_report()
    print(f"Wrote report to {path}")

