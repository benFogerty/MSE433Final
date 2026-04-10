from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import RESULTS_DIR
from .optimization import DEFAULT_MAIN_RUN_SUBDIR, load_generator_uc_results


DECISION_SUPPORT_SUMMARY_PATH = RESULTS_DIR / "generator_decision_support_summary.csv"
DECISION_SUPPORT_HOURLY_PATH = RESULTS_DIR / "generator_hourly_recommendations.csv"
SUPPLY_MIX_PATH = RESULTS_DIR / "generator_supply_mix_hourly.csv"


def _classify_action(frame: pd.DataFrame) -> pd.Series:
    startup = frame["startup"].fillna(0.0).gt(0.5)
    shutdown = frame["shutdown"].fillna(0.0).gt(0.5)
    on = frame["recommended_on"]
    ratio = frame["dispatch_ratio_to_generator_peak"].fillna(0.0)

    actions = np.select(
        [
            startup,
            shutdown,
            ~on,
            ratio >= 0.80,
            ratio >= 0.20,
        ],
        [
            "Start now",
            "Shut down",
            "Keep off",
            "Run high",
            "Run",
        ],
        default="Keep available",
    )
    return pd.Series(actions, index=frame.index)


def _assign_role(row: pd.Series) -> str:
    fuel = str(row["fuel_type"]).upper()
    if fuel == "NUCLEAR":
        return "Baseload"
    if fuel == "HYDRO":
        return "Flexible renewable support"
    if fuel == "BIOFUEL":
        return "Dispatchable support"
    if fuel == "GAS" and row["high_gas_hour_on_rate"] >= 0.60:
        return "Peak support"
    if fuel == "GAS" and row["on_rate"] >= 0.40:
        return "Mid-merit thermal"
    if fuel == "GAS":
        return "Standby peaker"
    return "Other"


def build_generator_decision_support(source_subdir: str = DEFAULT_MAIN_RUN_SUBDIR) -> dict[str, pd.DataFrame]:
    outputs = load_generator_uc_results(source_subdir)
    generator_dispatch = outputs["generator_dispatch"]
    dispatch = outputs["dispatch"]

    if generator_dispatch.empty:
        raise RuntimeError("Generator dispatch output is missing; cannot build decision-support layer.")

    generator_dispatch = generator_dispatch[generator_dispatch["policy"] == "forecast_informed_uc"].copy()
    dispatch = dispatch[dispatch["policy"] == "forecast_informed_uc"].copy()

    system_context = dispatch[
        [
            "timestamp",
            "season",
            "hour",
            "ontario_demand_mw",
            "renewable_direct_mw",
            "discharge_mw",
            "optimized_gas_dispatch_mw",
            "high_gas_hour",
        ]
    ].copy()

    generator_dispatch["timestamp"] = pd.to_datetime(generator_dispatch["timestamp"])
    generator_dispatch["recommended_on"] = np.where(
        generator_dispatch["commitment"].notna(),
        generator_dispatch["commitment"].fillna(0.0).gt(0.5),
        generator_dispatch["dispatch_mw"].gt(1e-6),
    )
    generator_dispatch = generator_dispatch.merge(system_context, on="timestamp", how="left")

    generator_peaks = (
        generator_dispatch.groupby("generator", as_index=False)["dispatch_mw"]
        .max()
        .rename(columns={"dispatch_mw": "generator_peak_dispatch_mw"})
    )
    generator_dispatch = generator_dispatch.merge(generator_peaks, on="generator", how="left")
    generator_dispatch["dispatch_ratio_to_generator_peak"] = (
        generator_dispatch["dispatch_mw"]
        / generator_dispatch["generator_peak_dispatch_mw"].replace(0.0, np.nan)
    ).fillna(0.0)
    generator_dispatch["recommended_action"] = _classify_action(generator_dispatch)
    generator_dispatch["recommended_status"] = np.where(generator_dispatch["recommended_on"], "On", "Off")

    hourly_recommendations = generator_dispatch[
        [
            "timestamp",
            "season",
            "hour",
            "generator",
            "fuel_type",
            "recommended_status",
            "recommended_action",
            "recommended_on",
            "dispatch_mw",
            "dispatch_ratio_to_generator_peak",
            "startup",
            "shutdown",
            "high_gas_hour",
            "ontario_demand_mw",
        ]
    ].copy()
    hourly_recommendations = hourly_recommendations.sort_values(
        ["timestamp", "recommended_on", "fuel_type", "dispatch_mw"],
        ascending=[True, False, True, False],
    ).reset_index(drop=True)

    on_hours = generator_dispatch["recommended_on"].astype(int)
    high_gas_mask = generator_dispatch["high_gas_hour"].fillna(0).astype(int).eq(1)
    summary = (
        generator_dispatch.assign(
            on_indicator=on_hours,
            startup_indicator=generator_dispatch["startup"].fillna(0.0).gt(0.5).astype(int),
            shutdown_indicator=generator_dispatch["shutdown"].fillna(0.0).gt(0.5).astype(int),
            dispatch_when_on=generator_dispatch["dispatch_mw"].where(generator_dispatch["recommended_on"], 0.0),
            high_gas_dispatch_mw=generator_dispatch["dispatch_mw"].where(high_gas_mask, 0.0),
            high_gas_on=generator_dispatch["recommended_on"].where(high_gas_mask, False).astype(int),
        )
        .groupby(["generator", "fuel_type"], as_index=False)
        .agg(
            on_hours=("on_indicator", "sum"),
            on_rate=("on_indicator", "mean"),
            avg_dispatch_mw=("dispatch_mw", "mean"),
            avg_dispatch_when_on_mw=("dispatch_when_on", lambda s: float(s[s > 0].mean()) if (s > 0).any() else 0.0),
            max_dispatch_mw=("dispatch_mw", "max"),
            startup_count=("startup_indicator", "sum"),
            shutdown_count=("shutdown_indicator", "sum"),
            high_gas_hour_dispatch_mwh=("high_gas_dispatch_mw", "sum"),
            high_gas_hour_on_rate=("high_gas_on", "mean"),
        )
    )
    summary["recommended_role"] = summary.apply(_assign_role, axis=1)
    summary = summary.sort_values(["fuel_type", "avg_dispatch_when_on_mw"], ascending=[True, False]).reset_index(drop=True)

    fuel_mix = (
        generator_dispatch.groupby(["timestamp", "fuel_type"], as_index=False)["dispatch_mw"]
        .sum()
        .pivot(index="timestamp", columns="fuel_type", values="dispatch_mw")
        .fillna(0.0)
        .reset_index()
    )
    fuel_mix.columns.name = None
    fuel_mix = fuel_mix.merge(
        dispatch[
            [
                "timestamp",
                "ontario_demand_mw",
                "renewable_direct_mw",
                "discharge_mw",
            ]
        ],
        on="timestamp",
        how="left",
    )
    fuel_mix = fuel_mix.rename(
        columns={
            "renewable_direct_mw": "renewables_mw",
            "discharge_mw": "battery_mw",
            "BIOFUEL": "biofuel_mw",
            "GAS": "gas_mw",
            "HYDRO": "hydro_mw",
            "NUCLEAR": "nuclear_mw",
        }
    )
    for column in ["biofuel_mw", "gas_mw", "hydro_mw", "nuclear_mw", "renewables_mw", "battery_mw"]:
        if column not in fuel_mix.columns:
            fuel_mix[column] = 0.0
    supply_mix = fuel_mix[
        [
            "timestamp",
            "ontario_demand_mw",
            "renewables_mw",
            "battery_mw",
            "nuclear_mw",
            "hydro_mw",
            "biofuel_mw",
            "gas_mw",
        ]
    ].sort_values("timestamp").reset_index(drop=True)

    hourly_recommendations.to_csv(DECISION_SUPPORT_HOURLY_PATH, index=False)
    summary.to_csv(DECISION_SUPPORT_SUMMARY_PATH, index=False)
    supply_mix.to_csv(SUPPLY_MIX_PATH, index=False)

    return {
        "hourly_recommendations": hourly_recommendations,
        "generator_summary": summary,
        "supply_mix": supply_mix,
    }
