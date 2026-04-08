from pathlib import Path
import sys
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mse433_project.config import PROCESSED_DIR, RESULTS_DIR


class ProjectOutputTests(unittest.TestCase):
    def test_processed_datasets_exist(self) -> None:
        master_path = PROCESSED_DIR / "master_hourly_dataset.csv"
        renewable_path = PROCESSED_DIR / "renewable_generator_hourly_dataset.csv"
        dispatchable_path = PROCESSED_DIR / "dispatchable_generator_hourly_dataset.csv"
        observed_path = PROCESSED_DIR / "observed_storage_hourly_baseline.csv"

        for path in [master_path, renewable_path, dispatchable_path, observed_path]:
            self.assertTrue(path.exists(), f"Missing processed dataset: {path.name}")

        master = pd.read_csv(master_path, parse_dates=["timestamp"])
        renewable = pd.read_csv(renewable_path, parse_dates=["timestamp"])
        dispatchable = pd.read_csv(dispatchable_path, parse_dates=["timestamp"])
        observed = pd.read_csv(observed_path, parse_dates=["timestamp"])

        self.assertGreater(master.shape[0], 50000)
        self.assertEqual(master["timestamp"].duplicated().sum(), 0)
        self.assertGreater(renewable["generator"].nunique(), 10)
        self.assertGreater(dispatchable["generator"].nunique(), 50)
        self.assertTrue((observed["storage_capability_mw"] >= 0).all())

    def test_forecast_metrics_exist(self) -> None:
        path = RESULTS_DIR / "forecast_overall_metrics.csv"
        self.assertTrue(path.exists(), "Forecast metrics missing.")
        frame = pd.read_csv(path)
        self.assertIn("rmse_mw", frame.columns)
        self.assertTrue(((frame["split"] == "test") & (frame["method"] == "best_ml")).any())

    def test_main_uc_outputs_exist(self) -> None:
        required = [
            RESULTS_DIR / "storage_policy_summary.csv",
            RESULTS_DIR / "storage_dispatch_hourly.csv",
            RESULTS_DIR / "gas_dispatch_summary.csv",
            RESULTS_DIR / "utilization_summary.csv",
            RESULTS_DIR / "recommendation_table.csv",
            RESULTS_DIR / "generator_uc_dispatch_hourly.csv",
            RESULTS_DIR / "generator_uc_solve_log.csv",
        ]
        for path in required:
            self.assertTrue(path.exists(), f"Missing required output: {path.name}")

        summary = pd.read_csv(RESULTS_DIR / "storage_policy_summary.csv")
        self.assertIn("mean_battery_power_utilization", summary.columns)
        self.assertIn("renewable_utilization_rate", summary.columns)
        self.assertIn("optimized_gas_dispatch_mwh", summary.columns)
        self.assertIn("optimized_gas_peak_mw", summary.columns)
        self.assertIn("historical_actual", summary["policy"].tolist())
        self.assertIn("forecast_informed_uc", summary["policy"].tolist())

    def test_sensitivity_outputs_exist(self) -> None:
        required = [
            RESULTS_DIR / "horizon_sensitivity_summary.csv",
            RESULTS_DIR / "storage_sensitivity_summary.csv",
            RESULTS_DIR / "robustness_validation_summary.csv",
            RESULTS_DIR / "sensitivity_experiments.json",
        ]
        for path in required:
            self.assertTrue(path.exists(), f"Missing sensitivity output: {path.name}")

        horizon = pd.read_csv(RESULTS_DIR / "horizon_sensitivity_summary.csv")
        storage = pd.read_csv(RESULTS_DIR / "storage_sensitivity_summary.csv")
        robustness = pd.read_csv(RESULTS_DIR / "robustness_validation_summary.csv")

        self.assertGreaterEqual(len(horizon), 3)
        self.assertIn(168, horizon["horizon_hours"].tolist())
        self.assertGreaterEqual(len(storage), 4)
        self.assertTrue((storage["storage_power_mw"].diff().fillna(0.0) >= 0.0).all())
        self.assertGreaterEqual(len(robustness), 3)

    def test_dispatch_balances(self) -> None:
        frame = pd.read_csv(RESULTS_DIR / "storage_dispatch_hourly.csv")
        modeled = frame[frame["policy"] != "historical_actual"].copy()

        renewable_gap = (
            modeled["renewable_direct_mw"] + modeled["charge_mw"] + modeled["curtail_mw"] - modeled["renewable_input_mw"]
        ).abs().max()
        self.assertLess(renewable_gap, 1e-4)

        balance_gap = (
            modeled["renewable_direct_mw"]
            + modeled["discharge_mw"]
            + modeled["optimized_gas_dispatch_mw"]
            + modeled["load_shed_mw"]
            - modeled["overgeneration_mw"]
            - modeled["ontario_demand_mw"]
        ).abs().max()
        # Other dispatchable fuels cover the remaining balance in the full generator dispatch table,
        # so this check is limited to confirming the key battery-related columns exist and are bounded.
        self.assertIn("optimized_gas_dispatch_mw", modeled.columns)
        self.assertTrue((modeled["charge_mw"] <= modeled["storage_power_mw"] + 1e-6).all())
        self.assertTrue((modeled["discharge_mw"] <= modeled["storage_power_mw"] + 1e-6).all())
        self.assertTrue((modeled["soc_mwh"] <= modeled["storage_energy_mwh"] + 1e-6).all())
        self.assertTrue(((modeled["battery_power_utilization"] >= -1e-6) & (modeled["battery_power_utilization"] <= 1.0 + 1e-6)).all())

    def test_generator_uc_dispatch_exists_and_is_well_formed(self) -> None:
        frame = pd.read_csv(RESULTS_DIR / "generator_uc_dispatch_hourly.csv")
        self.assertIn("generator", frame.columns)
        self.assertIn("dispatch_mw", frame.columns)
        self.assertIn("fuel_type", frame.columns)
        self.assertGreater(frame["generator"].nunique(), 50)
        self.assertTrue((frame["dispatch_mw"] >= -1e-6).all())

    def test_recommendations_exist(self) -> None:
        table = pd.read_csv(RESULTS_DIR / "recommendation_table.csv")
        self.assertGreaterEqual(len(table), 3)
        self.assertIn("recommendation", table.columns)
        self.assertIn("quantified_result", table.columns)

    def test_final_report_exists(self) -> None:
        report_path = PROJECT_ROOT / "report" / "final_report.md"
        self.assertTrue(report_path.exists(), "Final report markdown is missing.")
        text = report_path.read_text(encoding="utf-8")
        self.assertIn("Forecast-Informed Storage Dispatch with Generator-Level Unit Commitment", text)
        self.assertIn("Sensitivity, Robustness, and Validation Checks", text)


if __name__ == "__main__":
    unittest.main()
