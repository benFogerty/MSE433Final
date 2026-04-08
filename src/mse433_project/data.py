from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .config import (
    DEMAND_DIR,
    DISPATCHABLE_GENERATOR_DATASET_PATH,
    GENERATOR_PARAMETERS_PATH,
    MASTER_DATASET_PATH,
    OBSERVED_STORAGE_BASELINE_PATH,
    OUTPUT_CAPABILITY_DIR,
    PROCESSED_DIR,
    RENEWABLE_GENERATOR_DATASET_PATH,
    VARIABLE_FORECAST_DIR,
    VARIABLE_FORECAST_SUMMARY_PATH,
)


DEMAND_HEADER = ["Date", "Hour", "Market Demand", "Ontario Demand"]
OUTPUT_HEADER_PREFIX = ["Delivery Date", "Generator", "Fuel Type", "Measurement"]
HOUR_COLUMNS = [f"Hour {hour}" for hour in range(1, 25)]
MEASURE_RENAME = {
    "Output": "output",
    "Forecast": "forecast",
    "Available Capacity": "available_capacity",
    "Capability": "capability",
}
RENEWABLE_FUELS = ("WIND", "SOLAR")
DISPATCHABLE_FUELS = ("BIOFUEL", "GAS", "HYDRO", "NUCLEAR")
UC_BINARY_FUELS = ("BIOFUEL", "GAS")
STORAGE_KEYWORDS = ("BESS", "BATTERY", "STORAGE")
SEASON_MAP = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Fall",
    10: "Fall",
    11: "Fall",
}


def _find_header_row(path: Path, expected_header: list[str]) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, line in enumerate(handle):
            header = [part.strip() for part in line.rstrip("\n").split(",")[: len(expected_header)]]
            if header == expected_header:
                return index
    raise ValueError(f"Could not find expected header in {path}")


def _add_calendar_columns(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["hour"] = enriched["timestamp"].dt.hour
    enriched["day_of_week"] = enriched["timestamp"].dt.dayofweek
    enriched["month"] = enriched["timestamp"].dt.month
    enriched["year"] = enriched["timestamp"].dt.year
    enriched["season"] = enriched["month"].map(SEASON_MAP)
    enriched["is_weekend"] = enriched["day_of_week"].isin([5, 6]).astype(int)
    return enriched


def load_demand_data(start_year: int = 2019, end_year: int = 2025) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(DEMAND_DIR.glob("PUB_Demand_*.csv")):
        year_match = re.search(r"PUB_Demand_(\d{4})", path.name)
        if not year_match:
            continue
        year = int(year_match.group(1))
        if year < start_year or year > end_year:
            continue
        header_row = _find_header_row(path, DEMAND_HEADER)
        frame = pd.read_csv(path, skiprows=header_row)
        frame = frame[DEMAND_HEADER].copy()
        frame.columns = ["date", "hour_ending", "market_demand_mw", "ontario_demand_mw"]
        frame["hour_ending"] = pd.to_numeric(frame["hour_ending"], errors="coerce")
        frame["market_demand_mw"] = pd.to_numeric(frame["market_demand_mw"], errors="coerce")
        frame["ontario_demand_mw"] = pd.to_numeric(frame["ontario_demand_mw"], errors="coerce")
        frame["timestamp"] = (
            pd.to_datetime(frame["date"], format="%Y-%m-%d")
            + pd.to_timedelta(frame["hour_ending"] - 1, unit="h")
        )
        frames.append(frame)

    demand = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    demand = demand.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return demand[demand["timestamp"] >= pd.Timestamp("2019-05-01 00:00:00")].reset_index(drop=True)


def select_output_capability_files(start_month: str = "201905", end_month: str = "202512") -> list[Path]:
    selected: dict[str, tuple[int, Path]] = {}
    pattern = re.compile(r"PUB_GenOutputCapabilityMonth_(\d{6})_v(\d+)", re.IGNORECASE)
    for path in OUTPUT_CAPABILITY_DIR.glob("PUB_GenOutputCapabilityMonth_*.csv"):
        match = pattern.search(path.name)
        if not match:
            continue
        month_key, version = match.group(1), int(match.group(2))
        if month_key < start_month or month_key > end_month:
            continue
        if month_key not in selected or version > selected[month_key][0]:
            selected[month_key] = (version, path)
    return [selected[month][1] for month in sorted(selected)]


def _read_output_capability_file(
    path: Path,
    fuel_filter: tuple[str, ...] | None = None,
    measurement_filter: tuple[str, ...] | None = None,
    generator_filter: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    header_row = _find_header_row(path, OUTPUT_HEADER_PREFIX)
    columns = OUTPUT_HEADER_PREFIX + HOUR_COLUMNS + ["extra_trailing_blank"]
    frame = pd.read_csv(path, skiprows=header_row + 1, header=None, names=columns)
    frame = frame.iloc[:, : len(OUTPUT_HEADER_PREFIX) + len(HOUR_COLUMNS)]
    frame = frame[OUTPUT_HEADER_PREFIX + HOUR_COLUMNS].copy()

    if fuel_filter is not None:
        frame = frame[frame["Fuel Type"].isin(fuel_filter)].copy()
    if measurement_filter is not None:
        frame = frame[frame["Measurement"].isin(measurement_filter)].copy()
    if generator_filter is not None:
        frame = frame[frame["Generator"].isin(generator_filter)].copy()

    long_frame = frame.melt(
        id_vars=OUTPUT_HEADER_PREFIX,
        value_vars=HOUR_COLUMNS,
        var_name="hour_label",
        value_name="mw",
    )
    long_frame["hour_ending"] = long_frame["hour_label"].str.extract(r"(\d+)").astype(int)
    long_frame["mw"] = pd.to_numeric(long_frame["mw"], errors="coerce").fillna(0.0)
    long_frame["timestamp"] = (
        pd.to_datetime(long_frame["Delivery Date"], format="%Y-%m-%d")
        + pd.to_timedelta(long_frame["hour_ending"] - 1, unit="h")
    )
    long_frame["measure_key"] = long_frame["Measurement"].map(MEASURE_RENAME)
    return long_frame[
        ["timestamp", "Delivery Date", "hour_ending", "Generator", "Fuel Type", "measure_key", "mw"]
    ]


def load_generator_parameters() -> pd.DataFrame:
    frame = pd.read_csv(GENERATOR_PARAMETERS_PATH).rename(
        columns={
            "Generator": "generator",
            "FuelType": "fuel_type",
            "P_max": "p_max_mw",
            "P_min": "p_min_mw",
            "T_up (hrs)": "min_up_hours",
            "T_down (hrs)": "min_down_hours",
            "Ramp Up (MW/hr)": "ramp_up_mw_per_hr",
            "Ramp Down (MW/hr)": "ramp_down_mw_per_hr",
            "T_startup (h)": "startup_hours",
            "T_shutdown (h)": "shutdown_hours",
            "E_startup (MWh)": "startup_energy_mwh",
            "E_shutdown (MWh)": "shutdown_energy_mwh",
            "Cost_startup ($)": "startup_cost",
            "Cost_shutdown ($)": "shutdown_cost",
            "Cost_variable ($/MWh)": "variable_cost_per_mwh",
        }
    )
    numeric_columns = [
        "p_max_mw",
        "p_min_mw",
        "min_up_hours",
        "min_down_hours",
        "ramp_up_mw_per_hr",
        "ramp_down_mw_per_hr",
        "startup_hours",
        "shutdown_hours",
        "startup_energy_mwh",
        "shutdown_energy_mwh",
        "startup_cost",
        "shutdown_cost",
        "variable_cost_per_mwh",
        "commission_year",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["fuel_type"] = frame["fuel_type"].str.upper()
    frame["variable_cost_per_mwh"] = frame["variable_cost_per_mwh"].fillna(0.0)
    frame["startup_cost"] = frame["startup_cost"].fillna(0.0)
    frame["shutdown_cost"] = frame["shutdown_cost"].fillna(0.0)
    frame["p_min_mw"] = frame["p_min_mw"].fillna(0.0)
    frame["p_max_mw"] = frame["p_max_mw"].fillna(0.0)
    frame["min_up_hours"] = frame["min_up_hours"].fillna(1.0)
    frame["min_down_hours"] = frame["min_down_hours"].fillna(1.0)
    frame["ramp_up_mw_per_hr"] = frame["ramp_up_mw_per_hr"].fillna(frame["p_max_mw"].replace(0.0, 1.0))
    frame["ramp_down_mw_per_hr"] = frame["ramp_down_mw_per_hr"].fillna(frame["p_max_mw"].replace(0.0, 1.0))
    frame["is_uc_binary"] = frame["fuel_type"].isin(UC_BINARY_FUELS).astype(int)
    return frame


def load_output_capability_data() -> pd.DataFrame:
    frames = [
        _read_output_capability_file(
            path,
            fuel_filter=RENEWABLE_FUELS,
            measurement_filter=("Output", "Forecast", "Available Capacity"),
        )
        for path in select_output_capability_files()
    ]
    long_frame = pd.concat(frames, ignore_index=True)
    aggregated = (
        long_frame.groupby(["timestamp", "Fuel Type", "measure_key"], as_index=False)["mw"]
        .sum()
        .pivot(index="timestamp", columns=["Fuel Type", "measure_key"], values="mw")
        .fillna(0.0)
        .sort_index()
    )
    aggregated.columns = [f"{fuel.lower()}_{measure}" for fuel, measure in aggregated.columns]
    return aggregated.reset_index()


def load_conventional_cost_context() -> pd.DataFrame:
    params = load_generator_parameters()[["generator", "variable_cost_per_mwh"]].copy()
    frames: list[pd.DataFrame] = []
    for path in select_output_capability_files():
        long_frame = _read_output_capability_file(
            path,
            fuel_filter=DISPATCHABLE_FUELS,
            measurement_filter=("Capability",),
        )
        frames.append(
            long_frame[["timestamp", "Generator", "mw"]].rename(
                columns={"Generator": "generator", "mw": "capability_mw"}
            )
        )

    capability = pd.concat(frames, ignore_index=True)
    capability = capability.merge(params, on="generator", how="left")
    capability["variable_cost_per_mwh"] = capability["variable_cost_per_mwh"].fillna(0.0)
    capability = capability[capability["capability_mw"] > 0].copy()

    grouped = (
        capability.groupby(["timestamp", "variable_cost_per_mwh"], as_index=False)["capability_mw"]
        .sum()
        .sort_values(["timestamp", "variable_cost_per_mwh"])
    )
    grouped["cost_key"] = grouped["variable_cost_per_mwh"].map(lambda value: f"conventional_cap_at_cost_{value:.4f}")
    pivoted = grouped.pivot(index="timestamp", columns="cost_key", values="capability_mw").fillna(0.0).sort_index()
    pivoted = pivoted.reset_index()
    capability_columns = [column for column in pivoted.columns if column != "timestamp"]
    pivoted["total_conventional_capability"] = pivoted[capability_columns].sum(axis=1)
    return pivoted


def load_renewable_generator_hourly_data() -> pd.DataFrame:
    frames = [
        _read_output_capability_file(
            path,
            fuel_filter=RENEWABLE_FUELS,
            measurement_filter=("Output", "Forecast", "Available Capacity"),
        )
        for path in select_output_capability_files()
    ]
    long_frame = pd.concat(frames, ignore_index=True)
    pivoted = (
        long_frame.pivot_table(
            index=["timestamp", "Delivery Date", "hour_ending", "Generator", "Fuel Type"],
            columns="measure_key",
            values="mw",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
        .rename(
            columns={
                "Delivery Date": "delivery_date",
                "Generator": "generator",
                "Fuel Type": "fuel_type",
            }
        )
    )

    for column in ["output", "forecast", "available_capacity"]:
        if column not in pivoted.columns:
            pivoted[column] = 0.0

    pivoted["fuel_type"] = pivoted["fuel_type"].str.lower()
    params = load_generator_parameters()[["generator", "p_max_mw", "commission_year"]]
    pivoted = pivoted.merge(params, on="generator", how="left")
    pivoted = _add_calendar_columns(pivoted)
    return pivoted.sort_values(["timestamp", "fuel_type", "generator"]).reset_index(drop=True)


def _detect_observed_storage_generators() -> tuple[str, ...]:
    generators: set[str] = set()
    for path in select_output_capability_files():
        long_frame = _read_output_capability_file(
            path,
            measurement_filter=("Capability", "Output"),
        )
        other = long_frame[long_frame["Fuel Type"].eq("OTHER")].copy()
        if other.empty:
            continue
        totals = other.groupby("Generator", as_index=False)["mw"].sum()
        for row in totals.itertuples(index=False):
            name = str(row.Generator).upper()
            if row.mw <= 0:
                continue
            if any(keyword in name for keyword in STORAGE_KEYWORDS):
                generators.add(row.Generator)
    return tuple(sorted(generators))


def load_observed_storage_baseline() -> pd.DataFrame:
    storage_generators = _detect_observed_storage_generators()
    if not storage_generators:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "delivery_date",
                "hour_ending",
                "storage_generator",
                "storage_capability_mw",
                "observed_storage_output_mw",
                "battery_power_utilization",
                "hour",
                "month",
                "season",
            ]
        )

    frames = [
        _read_output_capability_file(
            path,
            generator_filter=storage_generators,
            measurement_filter=("Capability", "Output"),
        )
        for path in select_output_capability_files()
    ]
    long_frame = pd.concat(frames, ignore_index=True)
    pivoted = (
        long_frame.pivot_table(
            index=["timestamp", "Delivery Date", "hour_ending", "Generator"],
            columns="measure_key",
            values="mw",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
        .rename(
            columns={
                "Delivery Date": "delivery_date",
                "Generator": "storage_generator",
                "capability": "storage_capability_mw",
                "output": "observed_storage_output_mw",
            }
        )
    )
    for column in ["storage_capability_mw", "observed_storage_output_mw"]:
        if column not in pivoted.columns:
            pivoted[column] = 0.0
    pivoted["battery_power_utilization"] = (
        pivoted["observed_storage_output_mw"] / pivoted["storage_capability_mw"].replace(0.0, pd.NA)
    ).fillna(0.0).clip(lower=0.0, upper=1.0)
    pivoted = _add_calendar_columns(pivoted)
    return pivoted.sort_values(["timestamp", "storage_generator"]).reset_index(drop=True)


def load_dispatchable_generator_hourly_data() -> pd.DataFrame:
    frames = [
        _read_output_capability_file(
            path,
            fuel_filter=DISPATCHABLE_FUELS,
            measurement_filter=("Capability", "Output"),
        )
        for path in select_output_capability_files(start_month="202501", end_month="202512")
    ]
    long_frame = pd.concat(frames, ignore_index=True)
    pivoted = (
        long_frame.pivot_table(
            index=["timestamp", "Delivery Date", "hour_ending", "Generator", "Fuel Type"],
            columns="measure_key",
            values="mw",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
        .rename(
            columns={
                "Delivery Date": "delivery_date",
                "Generator": "generator",
                "Fuel Type": "fuel_type",
                "capability": "capability_mw",
                "output": "actual_output_mw",
            }
        )
    )
    for column in ["capability_mw", "actual_output_mw"]:
        if column not in pivoted.columns:
            pivoted[column] = 0.0

    params = load_generator_parameters()
    pivoted["fuel_type"] = pivoted["fuel_type"].str.upper()
    dispatchable = pivoted.merge(
        params,
        on=["generator", "fuel_type"],
        how="left",
        suffixes=("", "_param"),
    )
    dispatchable["is_uc_binary"] = dispatchable["fuel_type"].isin(UC_BINARY_FUELS).astype(int)
    dispatchable["p_max_mw"] = dispatchable["p_max_mw"].fillna(dispatchable["capability_mw"])
    dispatchable["p_min_mw"] = dispatchable["p_min_mw"].fillna(0.0)
    dispatchable["variable_cost_per_mwh"] = dispatchable["variable_cost_per_mwh"].fillna(0.0)
    dispatchable["startup_cost"] = dispatchable["startup_cost"].fillna(0.0)
    dispatchable["shutdown_cost"] = dispatchable["shutdown_cost"].fillna(0.0)
    dispatchable["min_up_hours"] = dispatchable["min_up_hours"].fillna(1.0)
    dispatchable["min_down_hours"] = dispatchable["min_down_hours"].fillna(1.0)
    dispatchable["ramp_up_mw_per_hr"] = dispatchable["ramp_up_mw_per_hr"].fillna(
        dispatchable["capability_mw"].clip(lower=1.0)
    )
    dispatchable["ramp_down_mw_per_hr"] = dispatchable["ramp_down_mw_per_hr"].fillna(
        dispatchable["capability_mw"].clip(lower=1.0)
    )
    dispatchable = _add_calendar_columns(dispatchable)
    return dispatchable.sort_values(["timestamp", "fuel_type", "generator"]).reset_index(drop=True)


def summarize_variable_forecast_publications() -> dict[str, object]:
    metadata_path = VARIABLE_FORECAST_DIR / "VGForecastSummary_latest_hourly_30d_metadata.json"
    csv_path = VARIABLE_FORECAST_DIR / "VGForecastSummary_latest_hourly_30d.csv"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    forecast_frame = pd.read_csv(csv_path)
    publication_timestamp = pd.to_datetime(
        forecast_frame["publication_date"] + " " + forecast_frame["publication_hour"].astype(str).str.zfill(2) + ":00:00"
    )
    forecast_timestamp = pd.to_datetime(
        forecast_frame["forecast_date"] + " " + (forecast_frame["forecast_hour"].astype(int) - 1).astype(str).str.zfill(2) + ":00:00"
    )
    lead_hours = (forecast_timestamp - publication_timestamp).dt.total_seconds() / 3600.0
    return {
        "generated_at": metadata["generated_at"],
        "publication_window_start": metadata["publication_window_start"],
        "publication_window_end": metadata["publication_window_end"],
        "snapshots_kept": int(metadata["snapshots_kept"]),
        "rows_written": int(metadata["rows_written"]),
        "fuel_types": sorted(forecast_frame["fuel_type"].dropna().unique().tolist()),
        "zones": sorted(forecast_frame["zone_name"].dropna().unique().tolist()),
        "publication_hours_observed": sorted(
            forecast_frame["publication_hour"].dropna().astype(int).unique().tolist()
        ),
        "hours_ahead_min": int(lead_hours.min()),
        "hours_ahead_max": int(lead_hours.max()),
    }


def build_master_dataset(write_outputs: bool = True) -> pd.DataFrame:
    demand = load_demand_data()
    renewables = load_output_capability_data()
    conventional_context = load_conventional_cost_context()
    renewable_generator = load_renewable_generator_hourly_data()
    dispatchable_generator = load_dispatchable_generator_hourly_data()
    observed_storage = load_observed_storage_baseline()

    master = demand.merge(renewables, on="timestamp", how="inner")
    master = master.merge(conventional_context, on="timestamp", how="left")
    master["date"] = master["timestamp"].dt.date.astype(str)
    master["hour"] = master["timestamp"].dt.hour
    master["hour_ending"] = master["hour"] + 1
    master["day_of_week"] = master["timestamp"].dt.dayofweek
    master["month"] = master["timestamp"].dt.month
    master["year"] = master["timestamp"].dt.year
    master["is_weekend"] = master["day_of_week"].isin([5, 6]).astype(int)
    master["season"] = master["month"].map(SEASON_MAP)
    master["wind_output_share"] = master["wind_output"] / master["wind_available_capacity"].replace(0, pd.NA)
    master["solar_output_share"] = master["solar_output"] / master["solar_available_capacity"].replace(0, pd.NA)
    master["wind_output_share"] = master["wind_output_share"].fillna(0.0)
    master["solar_output_share"] = master["solar_output_share"].fillna(0.0)
    master["total_renewable_output"] = master["wind_output"] + master["solar_output"]
    master["total_renewable_forecast"] = master["wind_forecast"] + master["solar_forecast"]
    master["total_renewable_available_capacity"] = (
        master["wind_available_capacity"] + master["solar_available_capacity"]
    )
    master["net_demand_actual"] = master["ontario_demand_mw"] - master["total_renewable_output"]
    master["net_demand_raw_forecast"] = master["ontario_demand_mw"] - master["total_renewable_forecast"]

    conventional_columns = sorted(
        [column for column in master.columns if column.startswith("conventional_cap_at_cost_")]
    )
    ordered_columns = [
        "timestamp",
        "date",
        "year",
        "month",
        "season",
        "day_of_week",
        "is_weekend",
        "hour",
        "hour_ending",
        "market_demand_mw",
        "ontario_demand_mw",
        "wind_output",
        "wind_forecast",
        "wind_available_capacity",
        "solar_output",
        "solar_forecast",
        "solar_available_capacity",
        "total_renewable_output",
        "total_renewable_forecast",
        "total_renewable_available_capacity",
        "total_conventional_capability",
        "net_demand_actual",
        "net_demand_raw_forecast",
    ] + conventional_columns
    master = master[ordered_columns].sort_values("timestamp").reset_index(drop=True)

    if write_outputs:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        master.to_csv(MASTER_DATASET_PATH, index=False)
        renewable_generator.to_csv(RENEWABLE_GENERATOR_DATASET_PATH, index=False)
        dispatchable_generator.to_csv(DISPATCHABLE_GENERATOR_DATASET_PATH, index=False)
        observed_storage.to_csv(OBSERVED_STORAGE_BASELINE_PATH, index=False)
        with VARIABLE_FORECAST_SUMMARY_PATH.open("w", encoding="utf-8") as handle:
            json.dump(summarize_variable_forecast_publications(), handle, indent=2)
    return master


def load_master_dataset() -> pd.DataFrame:
    if not MASTER_DATASET_PATH.exists():
        return build_master_dataset(write_outputs=True)
    return pd.read_csv(MASTER_DATASET_PATH, parse_dates=["timestamp"])


def load_renewable_generator_dataset() -> pd.DataFrame:
    if not RENEWABLE_GENERATOR_DATASET_PATH.exists():
        build_master_dataset(write_outputs=True)
    return pd.read_csv(RENEWABLE_GENERATOR_DATASET_PATH, parse_dates=["timestamp"])


def load_dispatchable_generator_dataset() -> pd.DataFrame:
    if not DISPATCHABLE_GENERATOR_DATASET_PATH.exists():
        build_master_dataset(write_outputs=True)
    return pd.read_csv(DISPATCHABLE_GENERATOR_DATASET_PATH, parse_dates=["timestamp"])


def load_observed_storage_dataset() -> pd.DataFrame:
    if not OBSERVED_STORAGE_BASELINE_PATH.exists():
        build_master_dataset(write_outputs=True)
    return pd.read_csv(OBSERVED_STORAGE_BASELINE_PATH, parse_dates=["timestamp"])
