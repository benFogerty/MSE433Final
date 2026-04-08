from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mse433_project.data import load_master_dataset
from mse433_project.modeling import run_all_forecasts


if __name__ == "__main__":
    master = load_master_dataset()
    outputs = run_all_forecasts(master)
    print("Forecasting complete.")
    print(outputs["overall_metrics"][["fuel", "split", "method", "rmse_mw"]].head().to_string(index=False))

