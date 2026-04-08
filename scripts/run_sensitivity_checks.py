from __future__ import annotations

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
from mse433_project.optimization import DEFAULT_MAIN_RUN_SUBDIR, load_generator_uc_results


HORIZON_WINDOWS = [24, 72, 168]
STORAGE_CASES = [
    {"label": "Current observed fleet", "power_mw": 1129.0, "energy_mwh": 4516.0},
    {"label": "1500 MW / 4h", "power_mw": 1500.0, "energy_mwh": 6000.0},
    {"label": "2000 MW / 4h", "power_mw": 2000.0, "energy_mwh": 8000.0},
    {"label": "3000 MW / 4h", "power_mw": 3000.0, "energy_mwh": 12000.0},
]


def _runner_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _run_uc(output_subdir: str, extra_args: list[str]) -> None:
    output_dir = RESULTS_DIR / "generator_uc" / output_subdir
    if (output_dir / "storage_uc_policy_summary.csv").exists() and (output_dir / "experiment_summary.json").exists():
        return
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mse433_project.generator_uc",
            "--policies",
            "forecast_informed_uc",
            "--output-subdir",
            output_subdir,
            *extra_args,
        ],
        check=True,
        env=_runner_env(),
    )


def _policy_row(source_subdir: str, policy: str = "forecast_informed_uc") -> tuple[pd.Series, dict[str, object]]:
    outputs = load_generator_uc_results(source_subdir)
    policy_summary = outputs["policy_summary"]
    row = policy_summary.loc[policy_summary["policy"] == policy]
    if row.empty:
        raise RuntimeError(f"Policy {policy} not found in {source_subdir}.")
    return row.iloc[0], outputs


def _build_horizon_summary() -> pd.DataFrame:
    baseline_row, _ = _policy_row(DEFAULT_MAIN_RUN_SUBDIR)
    historical_row, _ = _policy_row(DEFAULT_MAIN_RUN_SUBDIR, policy="historical_actual")
    records: list[dict[str, object]] = []
    for horizon in HORIZON_WINDOWS:
        subdir = DEFAULT_MAIN_RUN_SUBDIR if horizon == 168 else f"sensitivity_horizon_{horizon}h"
        if horizon != 168:
            extra_args = ["--horizon-hours", str(horizon)]
            if horizon >= 336:
                extra_args.extend(["--time-limit-per-day", "300"])
            _run_uc(subdir, extra_args)
        row, outputs = _policy_row(subdir)
        solve_log = pd.read_csv(RESULTS_DIR / "generator_uc" / subdir / "solve_log.csv")
        records.append(
            {
                "experiment": subdir,
                "horizon_hours": int(horizon),
                "storage_power_mw": float(row["storage_power_mw"]),
                "storage_energy_mwh": float(row["storage_energy_mwh"]),
                "battery_utilization_pct": 100.0 * float(row["mean_battery_power_utilization"]),
                "mean_battery_output_mw": float(row["mean_battery_output_mw"]),
                "total_discharge_mwh": float(row["total_discharge_mwh"]),
                "gas_generation_saved_vs_historical_mwh": float(row["gas_generation_saved_vs_historical_mwh"]),
                "gas_peak_reduction_vs_historical_mw": float(row["gas_peak_reduction_vs_historical_mw"]),
                "peak_nonrenewable_reduction_vs_historical_mw": float(
                    row["peak_residual_nonrenewable_reduction_vs_historical_mw"]
                ),
                "modeled_operating_cost_usd": float(row["modeled_operating_cost_usd"]),
                "operating_cost_saved_vs_historical_usd": float(row["operating_cost_saved_vs_historical_usd"]),
                "delta_vs_168h_battery_utilization_pp": 100.0
                * (float(row["mean_battery_power_utilization"]) - float(baseline_row["mean_battery_power_utilization"])),
                "delta_vs_168h_mean_battery_output_mw": float(row["mean_battery_output_mw"] - baseline_row["mean_battery_output_mw"]),
                "delta_vs_168h_gas_saved_mwh": float(
                    row["gas_generation_saved_vs_historical_mwh"] - baseline_row["gas_generation_saved_vs_historical_mwh"]
                ),
                "delta_vs_168h_operating_cost_saved_usd": float(
                    row["operating_cost_saved_vs_historical_usd"] - baseline_row["operating_cost_saved_vs_historical_usd"]
                ),
                "solve_blocks": int(outputs["experiment_summary"]["blocks_solved"]),
                "mean_block_solve_seconds": float(solve_log["elapsed_seconds"].mean()),
                "max_block_solve_seconds": float(solve_log["elapsed_seconds"].max()),
                "mean_mip_gap_pct": 100.0 * float(solve_log["mip_gap"].fillna(0.0).mean()),
                "historical_battery_utilization_pct": 100.0 * float(historical_row["mean_battery_power_utilization"]),
            }
        )
    return pd.DataFrame.from_records(records).sort_values("horizon_hours").reset_index(drop=True)


def _build_storage_summary() -> pd.DataFrame:
    baseline_row, _ = _policy_row(DEFAULT_MAIN_RUN_SUBDIR)
    records: list[dict[str, object]] = []
    for case in STORAGE_CASES:
        power_mw = float(case["power_mw"])
        energy_mwh = float(case["energy_mwh"])
        if abs(power_mw - 1129.0) < 1e-9 and abs(energy_mwh - 4516.0) < 1e-9:
            subdir = DEFAULT_MAIN_RUN_SUBDIR
        elif abs(power_mw - 3000.0) < 1e-9 and abs(energy_mwh - 12000.0) < 1e-9:
            subdir = "case_study_3000mw"
        else:
            subdir = f"sensitivity_storage_{int(round(power_mw))}mw"
            _run_uc(
                subdir,
                [
                    "--horizon-hours",
                    "168",
                    "--storage-power-mw",
                    f"{power_mw:g}",
                    "--storage-energy-mwh",
                    f"{energy_mwh:g}",
                ],
            )
        row, _ = _policy_row(subdir)
        records.append(
            {
                "experiment": subdir,
                "case_label": str(case["label"]),
                "storage_power_mw": power_mw,
                "storage_energy_mwh": energy_mwh,
                "battery_utilization_pct": 100.0 * float(row["mean_battery_power_utilization"]),
                "mean_battery_output_mw": float(row["mean_battery_output_mw"]),
                "total_discharge_mwh": float(row["total_discharge_mwh"]),
                "gas_generation_saved_vs_historical_mwh": float(row["gas_generation_saved_vs_historical_mwh"]),
                "gas_peak_reduction_vs_historical_mw": float(row["gas_peak_reduction_vs_historical_mw"]),
                "peak_nonrenewable_reduction_vs_historical_mw": float(
                    row["peak_residual_nonrenewable_reduction_vs_historical_mw"]
                ),
                "modeled_operating_cost_usd": float(row["modeled_operating_cost_usd"]),
                "operating_cost_saved_vs_historical_usd": float(row["operating_cost_saved_vs_historical_usd"]),
                "delta_vs_current_battery_utilization_pp": 100.0
                * (float(row["mean_battery_power_utilization"]) - float(baseline_row["mean_battery_power_utilization"])),
                "delta_vs_current_mean_battery_output_mw": float(row["mean_battery_output_mw"] - baseline_row["mean_battery_output_mw"]),
                "delta_vs_current_gas_saved_mwh": float(
                    row["gas_generation_saved_vs_historical_mwh"] - baseline_row["gas_generation_saved_vs_historical_mwh"]
                ),
                "delta_vs_current_operating_cost_saved_usd": float(
                    row["operating_cost_saved_vs_historical_usd"] - baseline_row["operating_cost_saved_vs_historical_usd"]
                ),
            }
        )
    return pd.DataFrame.from_records(records).sort_values("storage_power_mw").reset_index(drop=True)


def _build_validation_summary(horizon_summary: pd.DataFrame, storage_summary: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "check": "Horizon robustness: battery utilization spread",
                "value": (
                    f"{horizon_summary['battery_utilization_pct'].min():.2f}% to "
                    f"{horizon_summary['battery_utilization_pct'].max():.2f}%"
                ),
            },
            {
                "check": "Horizon robustness: gas saved spread",
                "value": (
                    f"{horizon_summary['gas_generation_saved_vs_historical_mwh'].min():,.0f} to "
                    f"{horizon_summary['gas_generation_saved_vs_historical_mwh'].max():,.0f} MWh"
                ),
            },
            {
                "check": "Storage sensitivity: best operating-cost savings case",
                "value": (
                    f"{storage_summary.loc[storage_summary['operating_cost_saved_vs_historical_usd'].idxmax(), 'case_label']} "
                    f"(${storage_summary['operating_cost_saved_vs_historical_usd'].max():,.0f} saved)"
                ),
            },
            {
                "check": "Storage sensitivity: best gas-savings case",
                "value": (
                    f"{storage_summary.loc[storage_summary['gas_generation_saved_vs_historical_mwh'].idxmax(), 'case_label']} "
                    f"({storage_summary['gas_generation_saved_vs_historical_mwh'].max():,.0f} MWh)"
                ),
            },
        ]
    )


if __name__ == "__main__":
    horizon_summary = _build_horizon_summary()
    storage_summary = _build_storage_summary()
    validation_summary = _build_validation_summary(horizon_summary, storage_summary)

    horizon_summary.to_csv(RESULTS_DIR / "horizon_sensitivity_summary.csv", index=False)
    storage_summary.to_csv(RESULTS_DIR / "storage_sensitivity_summary.csv", index=False)
    validation_summary.to_csv(RESULTS_DIR / "robustness_validation_summary.csv", index=False)

    metadata = {
        "main_reference_run": DEFAULT_MAIN_RUN_SUBDIR,
        "horizon_hours_tested": HORIZON_WINDOWS,
        "storage_cases_tested": STORAGE_CASES,
    }
    (RESULTS_DIR / "sensitivity_experiments.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Sensitivity checks complete.")
    print(horizon_summary.to_string(index=False))
    print(storage_summary.to_string(index=False))
