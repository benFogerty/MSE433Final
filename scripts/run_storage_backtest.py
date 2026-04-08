from pathlib import Path
import json
import os
import subprocess
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mse433_project.config import RESULTS_DIR
from mse433_project.data import load_master_dataset
from mse433_project.optimization import run_storage_backtest


if __name__ == "__main__":
    master = load_master_dataset()
    predictions = pd.read_csv(RESULTS_DIR / "all_fuel_predictions.csv", parse_dates=["timestamp"])
    with (RESULTS_DIR / "forecast_model_selection.json").open("r", encoding="utf-8") as handle:
        model_selection = json.load(handle)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mse433_project.generator_uc",
            "--horizon-hours",
            "168",
            "--policies",
            "forecast_informed_uc",
            "--output-subdir",
            "full_run_168h",
        ],
        check=True,
        env=env,
    )
    outputs = run_storage_backtest(master, predictions, model_selection)
    print("Storage backtest complete.")
    print(outputs["policy_summary"].to_string(index=False))
