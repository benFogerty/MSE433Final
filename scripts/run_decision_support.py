from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mse433_project.decision_support import build_generator_decision_support


if __name__ == "__main__":
    outputs = build_generator_decision_support()
    print("Decision-support outputs complete.")
    print(outputs["generator_summary"].head(15).to_string(index=False))
