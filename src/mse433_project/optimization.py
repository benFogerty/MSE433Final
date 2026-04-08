from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, RESULTS_DIR


DEFAULT_MAIN_RUN_SUBDIR = "full_run_168h"


def _source_dir(source_subdir: str = DEFAULT_MAIN_RUN_SUBDIR) -> Path:
    return RESULTS_DIR / "generator_uc" / source_subdir


def _build_utilization_summary(policy_summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario",
        "scenario_label",
        "policy",
        "mean_battery_power_utilization",
        "max_battery_power_utilization",
        "mean_battery_energy_utilization",
        "battery_throughput_utilization",
        "renewable_utilization_rate",
        "curtailment_rate",
        "renewable_backed_average_mw",
        "mean_battery_output_mw",
        "battery_utilization_improvement_vs_historical_pp",
        "peak_residual_nonrenewable_reduction_vs_historical_mw",
        "high_gas_hour_battery_discharge_mwh",
        "high_gas_hour_average_discharge_mw",
        "high_gas_hour_coverage_ratio",
        "optimized_gas_dispatch_mwh",
        "optimized_gas_peak_mw",
        "gas_generation_saved_vs_historical_mwh",
        "average_gas_generation_saved_mw",
        "gas_peak_reduction_vs_historical_mw",
        "modeled_operating_cost_usd",
        "operating_cost_saved_vs_historical_usd",
    ]
    available = [column for column in columns if column in policy_summary.columns]
    return policy_summary[available].copy()


def _build_recommendation_table(policy_summary: pd.DataFrame) -> pd.DataFrame:
    historical = policy_summary[policy_summary["policy"] == "historical_actual"].iloc[0]
    best = policy_summary[policy_summary["policy"] == "forecast_informed_uc"].iloc[0]
    return pd.DataFrame.from_records(
        [
            {
                "recommendation": "Dispatch storage with the 168-hour forecast-informed generator UC policy",
                "quantified_result": (
                    f"Battery utilization rises from {100 * historical['mean_battery_power_utilization']:.2f}% "
                    f"to {100 * best['mean_battery_power_utilization']:.2f}%."
                ),
            },
            {
                "recommendation": "Use storage as renewable-backed peak support rather than keeping it underutilized",
                "quantified_result": (
                    f"Mean battery output increases from {historical['mean_battery_output_mw']:.2f} MW "
                    f"to {best['mean_battery_output_mw']:.2f} MW."
                ),
            },
            {
                "recommendation": "Expect the main benefit to come from storage utilization, not extra renewable capture",
                "quantified_result": (
                    f"Renewable utilization stays at {100 * best['renewable_utilization_rate']:.0f}% in this window, "
                    f"while peak non-renewable requirement still falls by {best['peak_residual_nonrenewable_reduction_vs_historical_mw']:.2f} MW."
                ),
            },
            {
                "recommendation": "Track gas displacement as the clearest flexible-generation impact metric",
                "quantified_result": (
                    f"Modeled gas generation falls by {best['gas_generation_saved_vs_historical_mwh']:,.0f} MWh "
                    f"relative to the historical baseline, or {best['average_gas_generation_saved_mw']:.2f} MW on an average hour."
                ),
            },
            {
                "recommendation": "Use modeled operating cost as the UC-consistent economic metric",
                "quantified_result": (
                    f"Variable, startup, and shutdown cost falls by ${best['operating_cost_saved_vs_historical_usd']:,.0f} "
                    f"relative to the historical baseline."
                ),
            },
        ]
    )


def _compute_historical_operating_cost(policy_summary: pd.DataFrame) -> float:
    dispatchable = pd.read_csv(PROCESSED_DIR / "dispatchable_generator_hourly_dataset.csv", parse_dates=["timestamp"])
    historical_row = policy_summary.loc[policy_summary["policy"] == "historical_actual"].iloc[0]
    overlap_hours = int(historical_row["overlap_hours"])
    if overlap_hours <= 0:
        return 0.0
    dispatch_hours = pd.read_csv(RESULTS_DIR / "storage_dispatch_hourly.csv", parse_dates=["timestamp"])
    hist_dispatch = dispatch_hours[dispatch_hours["policy"] == "historical_actual"].copy()
    timestamps = hist_dispatch["timestamp"].drop_duplicates().sort_values()
    if timestamps.empty:
        return 0.0
    start = timestamps.min()
    end = timestamps.max()

    actual = dispatchable[(dispatchable["timestamp"] >= start) & (dispatchable["timestamp"] <= end)].copy()
    actual["fuel_type"] = actual["fuel_type"].str.upper()
    actual["status"] = (actual["actual_output_mw"] > 1e-6).astype(int)
    actual = actual.sort_values(["generator", "timestamp"]).reset_index(drop=True)
    actual["prev_status"] = actual.groupby("generator")["status"].shift(1)

    prior = dispatchable[dispatchable["timestamp"] < start].copy()
    prior = prior.sort_values(["generator", "timestamp"]).groupby("generator", as_index=False).tail(1)
    prior_status = dict(zip(prior["generator"], (prior["actual_output_mw"] > 1e-6).astype(int)))

    first_rows = actual.groupby("generator").head(1).index
    actual.loc[first_rows, "prev_status"] = actual.loc[first_rows, "generator"].map(prior_status).fillna(0).astype(int)
    actual["startup_flag"] = ((actual["status"] == 1) & (actual["prev_status"] == 0)).astype(int)
    actual["shutdown_flag"] = ((actual["status"] == 0) & (actual["prev_status"] == 1)).astype(int)

    variable_cost = float((actual["actual_output_mw"] * actual["variable_cost_per_mwh"]).sum())
    startup_cost = float((actual["startup_flag"] * actual["startup_cost"]).sum())
    shutdown_cost = float((actual["shutdown_flag"] * actual["shutdown_cost"]).sum())
    return variable_cost + startup_cost + shutdown_cost


def _compute_optimized_operating_cost_by_policy(generator_dispatch: pd.DataFrame) -> pd.DataFrame:
    dispatchable = pd.read_csv(PROCESSED_DIR / "dispatchable_generator_hourly_dataset.csv", parse_dates=["timestamp"])
    params = (
        dispatchable[
            [
                "timestamp",
                "generator",
                "variable_cost_per_mwh",
                "startup_cost",
                "shutdown_cost",
            ]
        ]
        .drop_duplicates(subset=["timestamp", "generator"])
        .copy()
    )
    merged = generator_dispatch.merge(params, on=["timestamp", "generator"], how="left")
    merged["variable_cost_usd"] = merged["dispatch_mw"] * merged["variable_cost_per_mwh"].fillna(0.0)
    merged["startup_cost_usd"] = merged["startup"].fillna(0.0) * merged["startup_cost"].fillna(0.0)
    merged["shutdown_cost_usd"] = merged["shutdown"].fillna(0.0) * merged["shutdown_cost"].fillna(0.0)
    summary = (
        merged.groupby("policy", as_index=False)[["variable_cost_usd", "startup_cost_usd", "shutdown_cost_usd"]]
        .sum()
    )
    summary["modeled_operating_cost_usd"] = (
        summary["variable_cost_usd"] + summary["startup_cost_usd"] + summary["shutdown_cost_usd"]
    )
    return summary[["policy", "modeled_operating_cost_usd"]]


def _build_operational_guidelines(dispatch: pd.DataFrame) -> pd.DataFrame:
    optimized = dispatch[dispatch["policy"] == "forecast_informed_uc"].copy()
    if optimized.empty:
        return pd.DataFrame(columns=["hour", "mean_charge_mw", "mean_discharge_mw", "mean_peak_support_mw", "guideline_type", "rank"])
    hourly = optimized.groupby("hour", as_index=False).agg(
        mean_charge_mw=("charge_mw", "mean"),
        mean_discharge_mw=("discharge_mw", "mean"),
        mean_peak_support_mw=("storage_dispatch_during_high_gas_hours_mw", "mean"),
    )
    top_charge = hourly.nlargest(4, "mean_charge_mw").copy()
    top_charge["guideline_type"] = "charge_priority_hour"
    top_charge["rank"] = range(1, len(top_charge) + 1)
    top_discharge = hourly.nlargest(4, "mean_discharge_mw").copy()
    top_discharge["guideline_type"] = "discharge_priority_hour"
    top_discharge["rank"] = range(1, len(top_discharge) + 1)
    top_peak = hourly.nlargest(4, "mean_peak_support_mw").copy()
    top_peak["guideline_type"] = "peak_support_hour"
    top_peak["rank"] = range(1, len(top_peak) + 1)
    return pd.concat([top_charge, top_discharge, top_peak], ignore_index=True)


def load_generator_uc_results(source_subdir: str = DEFAULT_MAIN_RUN_SUBDIR) -> dict[str, object]:
    source = _source_dir(source_subdir)
    if not source.exists():
        raise FileNotFoundError(f"Generator UC result directory not found: {source}")

    policy_summary = pd.read_csv(source / "storage_uc_policy_summary.csv")
    dispatch = pd.read_csv(source / "storage_uc_dispatch_hourly.csv", parse_dates=["timestamp"])
    gas_dispatch = pd.read_csv(source / "gas_dispatch_summary.csv", parse_dates=["timestamp"])
    observed_storage = pd.read_csv(source / "observed_storage_hourly_baseline.csv", parse_dates=["timestamp"])
    generator_dispatch = (
        pd.read_csv(source / "generator_uc_dispatch_hourly.csv", parse_dates=["timestamp"])
        if (source / "generator_uc_dispatch_hourly.csv").exists()
        else pd.DataFrame()
    )
    experiment_summary = json.loads((source / "experiment_summary.json").read_text(encoding="utf-8"))

    historical = policy_summary[policy_summary["policy"] == "historical_actual"].iloc[0]
    policy_summary["battery_utilization_improvement_vs_historical_pp"] = (
        policy_summary["mean_battery_power_utilization"] - historical["mean_battery_power_utilization"]
    ) * 100.0
    policy_summary["renewable_utilization_improvement_vs_historical_pp"] = (
        policy_summary["renewable_utilization_rate"] - historical["renewable_utilization_rate"]
    ) * 100.0
    policy_summary["renewable_backed_average_mw_delta_vs_historical"] = (
        policy_summary["renewable_backed_average_mw"] - historical["renewable_backed_average_mw"]
    )
    policy_summary["peak_residual_nonrenewable_reduction_vs_historical_mw"] = (
        historical["peak_residual_nonrenewable_mw"] - policy_summary["peak_residual_nonrenewable_mw"]
    )
    historical_gas_mwh = float(historical["historical_gas_context_mwh"])
    historical_gas_peak_mw = float(gas_dispatch["historical_gas_dispatch_mw"].max())
    policy_summary["gas_generation_saved_vs_historical_mwh"] = np.where(
        policy_summary["policy"] == "historical_actual",
        0.0,
        historical_gas_mwh - policy_summary["optimized_gas_dispatch_mwh"],
    )
    policy_summary["average_gas_generation_saved_mw"] = (
        policy_summary["gas_generation_saved_vs_historical_mwh"] / policy_summary["overlap_hours"].replace(0.0, np.nan)
    ).fillna(0.0)
    policy_summary["gas_peak_reduction_vs_historical_mw"] = np.where(
        policy_summary["policy"] == "historical_actual",
        0.0,
        historical_gas_peak_mw - policy_summary["optimized_gas_peak_mw"],
    )
    historical_operating_cost = _compute_historical_operating_cost(policy_summary)
    policy_summary["modeled_operating_cost_usd"] = np.nan
    policy_summary.loc[policy_summary["policy"] == "historical_actual", "modeled_operating_cost_usd"] = historical_operating_cost
    if not generator_dispatch.empty:
        optimized_costs = _compute_optimized_operating_cost_by_policy(generator_dispatch)
        policy_summary = policy_summary.merge(optimized_costs, on="policy", how="left", suffixes=("", "_optimized"))
        policy_summary["modeled_operating_cost_usd"] = policy_summary["modeled_operating_cost_usd"].fillna(
            policy_summary["modeled_operating_cost_usd_optimized"]
        )
        policy_summary = policy_summary.drop(columns=["modeled_operating_cost_usd_optimized"])
    policy_summary["operating_cost_saved_vs_historical_usd"] = np.where(
        policy_summary["policy"] == "historical_actual",
        0.0,
        historical_operating_cost - policy_summary["modeled_operating_cost_usd"],
    )

    utilization_summary = _build_utilization_summary(policy_summary)
    recommendation_table = _build_recommendation_table(policy_summary)
    guidelines = _build_operational_guidelines(dispatch)

    overlap = {
        "overlap_start": experiment_summary["overlap_start"],
        "overlap_end": experiment_summary["overlap_end"],
        "hours": int(policy_summary["overlap_hours"].max()),
    }

    scenario = experiment_summary.get("scenario", experiment_summary.get("optimization_scenario"))
    if scenario is None:
        raise KeyError("experiment_summary must include either 'scenario' or 'optimization_scenario'.")
    assumptions = {
        str(scenario["name"]): {
            "label": scenario["label"],
            "power_mw": scenario["power_mw"],
            "energy_mwh": scenario["energy_mwh"],
            "source_note": (
                f"Main optimization uses the generator-level UC run published from results/generator_uc/{source_subdir} "
                f"with a {experiment_summary['horizon_hours']}-hour rolling horizon."
            ),
        }
    }

    return {
        "policy_summary": policy_summary,
        "dispatch": dispatch,
        "gas_dispatch": gas_dispatch,
        "observed_storage": observed_storage,
        "generator_dispatch": generator_dispatch,
        "utilization_summary": utilization_summary,
        "recommendation_table": recommendation_table,
        "guidelines": guidelines,
        "experiment_summary": experiment_summary,
        "overlap": overlap,
        "assumptions": assumptions,
    }


def publish_generator_uc_results(source_subdir: str = DEFAULT_MAIN_RUN_SUBDIR) -> dict[str, pd.DataFrame]:
    outputs = load_generator_uc_results(source_subdir)
    source = _source_dir(source_subdir)

    shutil.copy2(source / "storage_uc_dispatch_hourly.csv", RESULTS_DIR / "storage_dispatch_hourly.csv")
    shutil.copy2(source / "gas_dispatch_summary.csv", RESULTS_DIR / "gas_dispatch_summary.csv")
    shutil.copy2(source / "observed_storage_hourly_baseline.csv", RESULTS_DIR / "observed_storage_hourly_baseline.csv")
    if (source / "generator_uc_dispatch_hourly.csv").exists():
        shutil.copy2(source / "generator_uc_dispatch_hourly.csv", RESULTS_DIR / "generator_uc_dispatch_hourly.csv")
    if (source / "solve_log.csv").exists():
        shutil.copy2(source / "solve_log.csv", RESULTS_DIR / "generator_uc_solve_log.csv")

    outputs["policy_summary"].to_csv(RESULTS_DIR / "storage_policy_summary.csv", index=False)
    outputs["utilization_summary"].to_csv(RESULTS_DIR / "utilization_summary.csv", index=False)
    outputs["recommendation_table"].to_csv(RESULTS_DIR / "recommendation_table.csv", index=False)
    outputs["guidelines"].to_csv(RESULTS_DIR / "current_storage_operational_guidelines.csv", index=False)
    (RESULTS_DIR / "storage_overlap_window.json").write_text(json.dumps(outputs["overlap"], indent=2), encoding="utf-8")
    (RESULTS_DIR / "storage_scenario_assumptions.json").write_text(
        json.dumps(outputs["assumptions"], indent=2), encoding="utf-8"
    )
    return {
        "policy_summary": outputs["policy_summary"],
        "dispatch": outputs["dispatch"],
        "utilization_summary": outputs["utilization_summary"],
        "recommendation_table": outputs["recommendation_table"],
    }


def run_storage_backtest(master: pd.DataFrame, predictions: pd.DataFrame, model_selection: dict[str, object]) -> dict[str, pd.DataFrame]:
    del master, predictions, model_selection
    return publish_generator_uc_results()
