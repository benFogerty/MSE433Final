from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from mse433_project.config import INITIAL_SOC_FRACTION, LOAD_SHEDDING_PENALTY_PER_MWH, RESULTS_DIR, ROUND_TRIP_EFFICIENCY, STORAGE_DURATION_HOURS
from mse433_project.data import load_dispatchable_generator_dataset, load_master_dataset, load_observed_storage_dataset


OUTPUT_ROOT = RESULTS_DIR / "generator_uc"
CURRENT_SCENARIO_NAME = "current_ontario_storage"
POLICY_ORDER = ["historical_actual", "no_storage_uc", "ieso_forecast_uc", "forecast_informed_uc", "perfect_foresight_uc"]
UC_BINARY_FUELS = ("BIOFUEL", "GAS")
CONTINUOUS_FUELS = ("HYDRO", "NUCLEAR")
NUCLEAR_RAMP_RATE_PER_MINUTE = 0.01


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generator-level UC with battery dispatch.")
    parser.add_argument("--start-date", type=str, default=None, help="Inclusive YYYY-MM-DD override for the overlap start.")
    parser.add_argument("--end-date", type=str, default=None, help="Inclusive YYYY-MM-DD override for the overlap end.")
    parser.add_argument("--horizon-hours", type=int, default=168, help="Rolling UC horizon length in hours.")
    parser.add_argument("--max-blocks", type=int, default=None, help="Optional solve-block limit for smoke tests.")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["forecast_informed_uc"],
        choices=["no_storage_uc", "ieso_forecast_uc", "forecast_informed_uc", "perfect_foresight_uc"],
        help="Policies to solve. historical_actual is always added as a baseline.",
    )
    parser.add_argument("--time-limit-per-day", type=float, default=120.0, help="MILP time limit per day in seconds.")
    parser.add_argument("--mip-rel-gap", type=float, default=0.01, help="Allowed relative MIP gap for each daily solve.")
    parser.add_argument("--output-subdir", type=str, default="full_run_168h", help="Subdirectory under results/generator_uc.")
    parser.add_argument("--disable-ramping", action="store_true", help="Disable generator ramp constraints.")
    parser.add_argument("--disable-min-updown", action="store_true", help="Disable generator minimum up/down constraints.")
    parser.add_argument(
        "--continuous-ramp-multiplier",
        type=float,
        default=2.0,
        help="Multiplier applied only to HYDRO/NUCLEAR ramp rates. Binary thermal UC ramps remain unchanged.",
    )
    parser.add_argument("--diagnose-ramping", action="store_true", help="Allow penalized ramp slack variables and record which generators violate ramp limits.")
    parser.add_argument("--ramp-slack-penalty", type=float, default=100000.0, help="Penalty per MW of ramp slack when --diagnose-ramping is enabled.")
    parser.add_argument(
        "--storage-power-mw",
        type=float,
        default=None,
        help="Optional storage power override for scenario studies. Defaults to the observed fleet power.",
    )
    parser.add_argument(
        "--storage-energy-mwh",
        type=float,
        default=None,
        help="Optional storage energy override. Defaults to storage power multiplied by the configured duration.",
    )
    parser.add_argument(
        "--storage-duration-hours",
        type=float,
        default=STORAGE_DURATION_HOURS,
        help="Duration used when --storage-power-mw is supplied without an explicit --storage-energy-mwh.",
    )
    return parser.parse_args()


def _build_renewable_policy_frame(master: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    prediction_pivot = predictions.pivot(index="timestamp", columns="fuel", values=["best_ml", "target_actual"]).sort_index()
    prediction_pivot.columns = [f"{metric}_{fuel.lower()}" for metric, fuel in prediction_pivot.columns]
    prediction_pivot = prediction_pivot.reset_index()

    frame = master.merge(prediction_pivot, on="timestamp", how="left").sort_values("timestamp").reset_index(drop=True)
    frame["renewable_actual_mw"] = frame["total_renewable_output"]
    frame["renewable_available_mw"] = frame["total_renewable_available_capacity"]
    frame["renewable_ieso_raw_mw"] = frame["wind_forecast"].fillna(0.0) + frame["solar_forecast"].fillna(0.0)
    frame["renewable_best_ml_mw"] = frame["best_ml_wind"].fillna(frame["wind_forecast"]) + frame["best_ml_solar"].fillna(frame["solar_forecast"])
    frame["renewable_perfect_mw"] = frame["target_actual_wind"].fillna(frame["wind_output"]) + frame["target_actual_solar"].fillna(frame["solar_output"])
    frame["renewable_ieso_raw_mw"] = frame["renewable_ieso_raw_mw"].clip(lower=0.0, upper=frame["renewable_available_mw"])
    frame["renewable_best_ml_mw"] = frame["renewable_best_ml_mw"].clip(lower=0.0, upper=frame["renewable_available_mw"])
    frame["renewable_perfect_mw"] = frame["renewable_perfect_mw"].clip(lower=0.0, upper=frame["renewable_available_mw"])
    return frame


def _aggregate_observed_storage(observed_storage: pd.DataFrame) -> pd.DataFrame:
    hourly = (
        observed_storage.groupby("timestamp", as_index=False)
        .agg(
            storage_capability_mw=("storage_capability_mw", "sum"),
            observed_storage_output_mw=("observed_storage_output_mw", "sum"),
        )
        .sort_values("timestamp")
    )
    hourly["battery_power_utilization"] = (
        hourly["observed_storage_output_mw"] / hourly["storage_capability_mw"].replace(0.0, np.nan)
    ).fillna(0.0).clip(lower=0.0, upper=1.0)
    return hourly


def _build_storage_scenario(
    observed_storage_hourly: pd.DataFrame,
    power_override_mw: float | None = None,
    energy_override_mwh: float | None = None,
    duration_hours: float = STORAGE_DURATION_HOURS,
) -> dict[str, float | str]:
    current_power_mw = float(observed_storage_hourly["storage_capability_mw"].max())
    current_energy_mwh = current_power_mw * STORAGE_DURATION_HOURS
    if power_override_mw is None and energy_override_mwh is None:
        return {
            "name": CURRENT_SCENARIO_NAME,
            "label": "Current observed storage fleet in source files",
            "power_mw": current_power_mw,
            "energy_mwh": current_energy_mwh,
        }
    scenario_power_mw = float(power_override_mw) if power_override_mw is not None else current_power_mw
    scenario_energy_mwh = (
        float(energy_override_mwh)
        if energy_override_mwh is not None
        else float(scenario_power_mw * duration_hours)
    )
    name_stub = str(int(round(scenario_power_mw))) if abs(scenario_power_mw - round(scenario_power_mw)) < 1e-9 else f"{scenario_power_mw:g}"
    return {
        "name": f"storage_case_{name_stub}mw",
        "label": f"Storage case study: {scenario_power_mw:.0f} MW / {scenario_energy_mwh:.0f} MWh",
        "power_mw": scenario_power_mw,
        "energy_mwh": scenario_energy_mwh,
    }


def _build_historical_gas_context(dispatchable: pd.DataFrame, overlap_start: pd.Timestamp, overlap_end: pd.Timestamp) -> pd.DataFrame:
    gas = dispatchable[
        (dispatchable["fuel_type"] == "GAS")
        & (dispatchable["timestamp"] >= overlap_start)
        & (dispatchable["timestamp"] <= overlap_end)
    ].copy()
    gas_hourly = gas.groupby("timestamp", as_index=False)["actual_output_mw"].sum().rename(columns={"actual_output_mw": "historical_gas_dispatch_mw"})
    threshold = float(gas_hourly["historical_gas_dispatch_mw"].quantile(0.9))
    gas_hourly["high_gas_hour"] = (gas_hourly["historical_gas_dispatch_mw"] >= threshold).astype(int)
    gas_hourly["gas_rank"] = gas_hourly["historical_gas_dispatch_mw"].rank(pct=True, method="average")
    return gas_hourly


def _build_stress_profile(master_overlap: pd.DataFrame, gas_context: pd.DataFrame) -> pd.DataFrame:
    frame = master_overlap[["timestamp", "ontario_demand_mw", "renewable_actual_mw"]].copy()
    frame["actual_net_load_mw"] = frame["ontario_demand_mw"] - frame["renewable_actual_mw"]
    frame["net_load_rank"] = frame["actual_net_load_mw"].rank(pct=True, method="average")
    frame = frame.merge(gas_context, on="timestamp", how="left")
    frame["historical_gas_dispatch_mw"] = frame["historical_gas_dispatch_mw"].fillna(0.0)
    frame["gas_rank"] = frame["gas_rank"].fillna(0.0)
    frame["high_gas_hour"] = frame["high_gas_hour"].fillna(0).astype(int)
    frame["stress_cost_adder"] = 0.15 * (20.0 + 50.0 * frame["net_load_rank"] + 30.0 * frame["gas_rank"])
    return frame


def _compute_initial_state(dispatchable: pd.DataFrame, first_timestamp: pd.Timestamp) -> dict[str, dict[str, float | int]]:
    state: dict[str, dict[str, float | int]] = {}
    for generator, group in dispatchable.groupby("generator", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        history = group[group["timestamp"] < first_timestamp]
        if history.empty:
            reference = group.iloc[0]
            reference_status = int(reference["actual_output_mw"] > 1e-6)
            duration = 999
        else:
            reference = history.iloc[-1]
            reference_status = int(reference["actual_output_mw"] > 1e-6)
            statuses = (history["actual_output_mw"].to_numpy(dtype="float64") > 1e-6).astype(int)
            duration = 0
            for value in statuses[::-1]:
                if int(value) == reference_status:
                    duration += 1
                else:
                    break
        state[generator] = {
            "commitment": reference_status,
            "output_mw": float(reference["actual_output_mw"]),
            "time_on_hours": int(duration if reference_status == 1 else 0),
            "time_off_hours": int(duration if reference_status == 0 else 0),
        }
    return state


def _summarize_initial_state(initial_state: dict[str, dict[str, float | int]]) -> dict[str, int]:
    on_count = sum(int(state["commitment"]) for state in initial_state.values())
    off_count = len(initial_state) - on_count
    return {
        "units_on_prior_hour": int(on_count),
        "units_off_prior_hour": int(off_count),
    }


def _build_generator_fleet(
    day_dispatchable: pd.DataFrame,
    block_timestamps: pd.Series,
    initial_state: dict[str, dict[str, float | int]],
) -> list[dict[str, object]]:
    fleet: list[dict[str, object]] = []
    for generator, group in day_dispatchable.groupby("generator", sort=True):
        group = group.sort_values("timestamp").reset_index(drop=True)
        aligned = pd.DataFrame({"timestamp": block_timestamps}).merge(group, on="timestamp", how="left")
        fuel_type = str(group["fuel_type"].iloc[0]).upper()
        p_max = np.minimum(
            aligned["capability_mw"].fillna(0.0).to_numpy(dtype="float64"),
            aligned["p_max_mw"].fillna(group["p_max_mw"].iloc[0]).fillna(0.0).to_numpy(dtype="float64"),
        )
        p_min = np.minimum(aligned["p_min_mw"].fillna(group["p_min_mw"].iloc[0]).fillna(0.0).to_numpy(dtype="float64"), p_max)
        ramp_up = float(group["ramp_up_mw_per_hr"].fillna(group["capability_mw"].clip(lower=1.0)).iloc[0])
        ramp_down = float(group["ramp_down_mw_per_hr"].fillna(group["capability_mw"].clip(lower=1.0)).iloc[0])
        if fuel_type == "NUCLEAR":
            nuclear_hourly_ramp = float(group["p_max_mw"].fillna(group["capability_mw"]).iloc[0]) * NUCLEAR_RAMP_RATE_PER_MINUTE * 60.0
            ramp_up = nuclear_hourly_ramp
            ramp_down = nuclear_hourly_ramp
        elif fuel_type == "HYDRO":
            hydro_hourly_ramp = float(group["p_max_mw"].fillna(group["capability_mw"]).iloc[0])
            ramp_up = hydro_hourly_ramp
            ramp_down = hydro_hourly_ramp
        prior = initial_state.get(generator, {"commitment": 0, "output_mw": 0.0, "time_on_hours": 999, "time_off_hours": 999})
        fleet.append(
            {
                "name": generator,
                "fuel_type": fuel_type,
                "binary": fuel_type in UC_BINARY_FUELS,
                "p_max": p_max,
                "p_min": p_min,
                "variable_cost": float(group["variable_cost_per_mwh"].fillna(0.0).iloc[0]),
                "startup_cost": float(group["startup_cost"].fillna(0.0).iloc[0]),
                "shutdown_cost": float(group["shutdown_cost"].fillna(0.0).iloc[0]),
                "ramp_up": ramp_up,
                "ramp_down": ramp_down,
                "min_up_hours": max(int(round(float(group["min_up_hours"].fillna(1.0).iloc[0]))), 1),
                "min_down_hours": max(int(round(float(group["min_down_hours"].fillna(1.0).iloc[0]))), 1),
                "initial_commitment": int(prior["commitment"]),
                "initial_output_mw": float(prior["output_mw"]),
                "initial_time_on_hours": int(prior["time_on_hours"]),
                "initial_time_off_hours": int(prior["time_off_hours"]),
            }
        )
    return fleet


def _effective_ramp_limit(unit: dict[str, object], direction: str, continuous_ramp_multiplier: float) -> float:
    if direction not in {"up", "down"}:
        raise ValueError(f"Unknown ramp direction: {direction}")
    base = float(unit["ramp_up"] if direction == "up" else unit["ramp_down"])
    if not bool(unit["binary"]) and str(unit["fuel_type"]).upper() in CONTINUOUS_FUELS:
        return base * float(continuous_ramp_multiplier)
    return base


def _solve_uc_day(
    day_frame: pd.DataFrame,
    day_dispatchable: pd.DataFrame,
    storage_power_mw: float,
    storage_energy_mwh: float,
    renewable_column: str,
    battery_policy: str,
    initial_state: dict[str, dict[str, float | int]],
    time_limit_per_day: float,
    mip_rel_gap: float,
    enforce_ramping: bool,
    enforce_min_updown: bool,
    continuous_ramp_multiplier: float,
    diagnose_ramping: bool,
    ramp_slack_penalty: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float | int]], dict[str, float]]:
    horizon = day_frame.shape[0]
    demand = day_frame["ontario_demand_mw"].to_numpy(dtype="float64")
    renewable_input = day_frame[renewable_column].to_numpy(dtype="float64")
    stress_cost = day_frame["stress_cost_adder"].to_numpy(dtype="float64")
    eta_charge = ROUND_TRIP_EFFICIENCY ** 0.5
    eta_discharge = ROUND_TRIP_EFFICIENCY ** 0.5
    fleet = _build_generator_fleet(
        day_dispatchable=day_dispatchable,
        block_timestamps=day_frame["timestamp"],
        initial_state=initial_state,
    )

    index = 0

    def alloc(size: int) -> np.ndarray:
        nonlocal index
        values = np.arange(index, index + size, dtype=int)
        index += size
        return values

    direct = alloc(horizon)
    curtail = alloc(horizon)
    load_shed = alloc(horizon)
    overgeneration = alloc(horizon)
    charge = discharge = soc = mode = None
    if battery_policy == "optimized":
        charge = alloc(horizon)
        discharge = alloc(horizon)
        soc = alloc(horizon)
        mode = alloc(horizon)

    dispatch_idx: dict[str, np.ndarray] = {}
    commit_idx: dict[str, np.ndarray] = {}
    startup_idx: dict[str, np.ndarray] = {}
    shutdown_idx: dict[str, np.ndarray] = {}
    ramp_up_slack_idx: dict[str, np.ndarray] = {}
    ramp_down_slack_idx: dict[str, np.ndarray] = {}
    for unit in fleet:
        dispatch_idx[str(unit["name"])] = alloc(horizon)
        if bool(unit["binary"]):
            commit_idx[str(unit["name"])] = alloc(horizon)
            startup_idx[str(unit["name"])] = alloc(horizon)
            shutdown_idx[str(unit["name"])] = alloc(horizon)
        if enforce_ramping and diagnose_ramping:
            ramp_up_slack_idx[str(unit["name"])] = alloc(horizon)
            ramp_down_slack_idx[str(unit["name"])] = alloc(horizon)

    variable_count = index
    c = np.zeros(variable_count, dtype="float64")
    lb = np.zeros(variable_count, dtype="float64")
    ub = np.full(variable_count, np.inf, dtype="float64")
    integrality = np.zeros(variable_count, dtype=int)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    curtail_penalty = 50.0
    throughput_penalty = 1.0

    for hour in range(horizon):
        ub[int(direct[hour])] = float(renewable_input[hour])
        ub[int(curtail[hour])] = float(renewable_input[hour])
        ub[int(load_shed[hour])] = float(demand[hour])
        ub[int(overgeneration[hour])] = float(sum(float(np.asarray(unit["p_max"])[hour]) for unit in fleet) + renewable_input[hour])
        c[int(curtail[hour])] = curtail_penalty
        c[int(load_shed[hour])] = LOAD_SHEDDING_PENALTY_PER_MWH
        c[int(overgeneration[hour])] = 500.0
        if battery_policy == "optimized":
            assert charge is not None and discharge is not None and soc is not None and mode is not None
            ub[int(charge[hour])] = float(storage_power_mw)
            ub[int(discharge[hour])] = float(storage_power_mw)
            ub[int(soc[hour])] = float(storage_energy_mwh)
            ub[int(mode[hour])] = 1.0
            integrality[int(mode[hour])] = 1
            c[int(charge[hour])] = throughput_penalty
        for unit in fleet:
            pmax = float(np.asarray(unit["p_max"])[hour])
            ub[int(dispatch_idx[str(unit["name"])][hour])] = pmax
            c[int(dispatch_idx[str(unit["name"])][hour])] = float(unit["variable_cost"]) + float(stress_cost[hour])
            if bool(unit["binary"]):
                for var_idx_name, lookup in (("startup", startup_idx), ("shutdown", shutdown_idx), ("commitment", commit_idx)):
                    idx = lookup[str(unit["name"])][hour]
                    ub[int(idx)] = 1.0
                    integrality[int(idx)] = 1
                    if var_idx_name == "startup":
                        c[int(idx)] = float(unit["startup_cost"])
                    elif var_idx_name == "shutdown":
                        c[int(idx)] = float(unit["shutdown_cost"])
            if enforce_ramping and diagnose_ramping:
                up_slack = ramp_up_slack_idx[str(unit["name"])][hour]
                down_slack = ramp_down_slack_idx[str(unit["name"])][hour]
                c[int(up_slack)] = float(ramp_slack_penalty)
                c[int(down_slack)] = float(ramp_slack_penalty)

    def add_constraint(coeffs: dict[int, float], lb_value: float, ub_value: float) -> None:
        row = len(lower)
        for col, value in coeffs.items():
            if abs(value) > 1e-12:
                rows.append(row)
                cols.append(int(col))
                data.append(float(value))
        lower.append(float(lb_value))
        upper.append(float(ub_value))

    for hour in range(horizon):
        renewable_coeffs = {int(direct[hour]): 1.0, int(curtail[hour]): 1.0}
        if battery_policy == "optimized":
            renewable_coeffs[int(charge[hour])] = 1.0
        add_constraint(renewable_coeffs, float(renewable_input[hour]), float(renewable_input[hour]))

        power_balance = {int(direct[hour]): 1.0, int(load_shed[hour]): 1.0, int(overgeneration[hour]): -1.0}
        if battery_policy == "optimized":
            power_balance[int(discharge[hour])] = 1.0
        for unit in fleet:
            power_balance[int(dispatch_idx[str(unit["name"])][hour])] = 1.0
        add_constraint(power_balance, float(demand[hour]), float(demand[hour]))

        if battery_policy == "optimized":
            assert charge is not None and discharge is not None and soc is not None and mode is not None
            add_constraint(
                {int(charge[hour]): 1.0, int(mode[hour]): -float(storage_power_mw)},
                -np.inf,
                0.0,
            )
            add_constraint(
                {int(discharge[hour]): 1.0, int(mode[hour]): float(storage_power_mw)},
                -np.inf,
                float(storage_power_mw),
            )
            soc_coeffs = {
                int(soc[hour]): 1.0,
                int(charge[hour]): -eta_charge,
                int(discharge[hour]): 1.0 / eta_discharge,
            }
            rhs_soc = float(storage_energy_mwh * INITIAL_SOC_FRACTION)
            if hour > 0:
                soc_coeffs[int(soc[hour - 1])] = -1.0
                rhs_soc = 0.0
            add_constraint(soc_coeffs, rhs_soc, rhs_soc)

        for unit in fleet:
            name = str(unit["name"])
            pmax = float(np.asarray(unit["p_max"])[hour])
            pmin = float(np.asarray(unit["p_min"])[hour])
            unit_dispatch = dispatch_idx[name][hour]
            if bool(unit["binary"]):
                unit_commit = commit_idx[name][hour]
                unit_start = startup_idx[name][hour]
                unit_stop = shutdown_idx[name][hour]
                add_constraint({int(unit_dispatch): 1.0, int(unit_commit): -pmax}, -np.inf, 0.0)
                if pmin > 0:
                    add_constraint({int(unit_dispatch): 1.0, int(unit_commit): -pmin}, 0.0, np.inf)

                relation = {int(unit_commit): 1.0, int(unit_start): -1.0, int(unit_stop): 1.0}
                rhs_relation = float(unit["initial_commitment"])
                if hour > 0:
                    relation[int(commit_idx[name][hour - 1])] = -1.0
                    rhs_relation = 0.0
                add_constraint(relation, rhs_relation, rhs_relation)
                add_constraint({int(unit_start): 1.0, int(unit_stop): 1.0}, -np.inf, 1.0)

                if enforce_ramping:
                    prev_output = float(unit["initial_output_mw"]) if hour == 0 else 0.0
                    ramp_up = {int(unit_dispatch): 1.0, int(unit_start): -pmax}
                    ramp_down = {int(unit_dispatch): -1.0, int(unit_stop): -max(float(unit["initial_output_mw"]) if hour == 0 else float(np.asarray(unit["p_max"])[hour - 1]), pmax)}
                    if hour > 0:
                        ramp_up[int(dispatch_idx[name][hour - 1])] = -1.0
                        ramp_down[int(dispatch_idx[name][hour - 1])] = 1.0
                        prev_output = 0.0
                    if diagnose_ramping:
                        ramp_up[int(ramp_up_slack_idx[name][hour])] = -1.0
                        ramp_down[int(ramp_down_slack_idx[name][hour])] = -1.0
                    add_constraint(ramp_up, -np.inf, _effective_ramp_limit(unit, "up", continuous_ramp_multiplier) + prev_output)
                    add_constraint(ramp_down, -np.inf, _effective_ramp_limit(unit, "down", continuous_ramp_multiplier) - prev_output)

                if enforce_min_updown:
                    min_up = int(unit["min_up_hours"])
                    min_down = int(unit["min_down_hours"])
                    start_window_start = max(0, hour - min_up + 1)
                    up_coeffs = {int(commit_idx[name][hour]): -1.0}
                    for tau in range(start_window_start, hour + 1):
                        up_coeffs[int(startup_idx[name][tau])] = up_coeffs.get(int(startup_idx[name][tau]), 0.0) + 1.0
                    add_constraint(up_coeffs, -np.inf, 0.0)

                    stop_window_start = max(0, hour - min_down + 1)
                    down_coeffs = {int(commit_idx[name][hour]): 1.0}
                    for tau in range(stop_window_start, hour + 1):
                        down_coeffs[int(shutdown_idx[name][tau])] = down_coeffs.get(int(shutdown_idx[name][tau]), 0.0) + 1.0
                    add_constraint(down_coeffs, -np.inf, 1.0)
            else:
                if enforce_ramping:
                    prev_output = float(unit["initial_output_mw"]) if hour == 0 else 0.0
                    ramp_up = {int(unit_dispatch): 1.0}
                    ramp_down = {int(unit_dispatch): -1.0}
                    if hour > 0:
                        ramp_up[int(dispatch_idx[name][hour - 1])] = -1.0
                        ramp_down[int(dispatch_idx[name][hour - 1])] = 1.0
                        prev_output = 0.0
                    if diagnose_ramping:
                        ramp_up[int(ramp_up_slack_idx[name][hour])] = -1.0
                        ramp_down[int(ramp_down_slack_idx[name][hour])] = -1.0
                    add_constraint(ramp_up, -np.inf, _effective_ramp_limit(unit, "up", continuous_ramp_multiplier) + prev_output)
                    add_constraint(ramp_down, -np.inf, _effective_ramp_limit(unit, "down", continuous_ramp_multiplier) - prev_output)

    if battery_policy == "optimized":
        assert soc is not None
        add_constraint(
            {int(soc[horizon - 1]): 1.0},
            float(storage_energy_mwh * INITIAL_SOC_FRACTION),
            float(storage_energy_mwh * INITIAL_SOC_FRACTION),
        )

    matrix = coo_matrix((data, (rows, cols)), shape=(len(lower), variable_count)).tocsr()
    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(matrix, np.array(lower), np.array(upper)),
        options={
            "disp": False,
            "time_limit": float(time_limit_per_day),
            "mip_rel_gap": float(mip_rel_gap),
        },
    )
    if not result.success:
        raise RuntimeError(f"Generator UC failed for {day_frame['timestamp'].iloc[0].date()}: {result.message}")

    x = result.x
    dispatch_frame = day_frame[["timestamp", "season", "hour", "ontario_demand_mw", renewable_column, "historical_gas_dispatch_mw", "high_gas_hour"]].copy()
    dispatch_frame = dispatch_frame.rename(columns={renewable_column: "renewable_input_mw"})
    dispatch_frame["renewable_direct_mw"] = x[direct]
    dispatch_frame["curtail_mw"] = x[curtail]
    dispatch_frame["load_shed_mw"] = x[load_shed]
    dispatch_frame["overgeneration_mw"] = x[overgeneration]
    if battery_policy == "optimized":
        dispatch_frame["charge_mw"] = x[charge]
        dispatch_frame["discharge_mw"] = x[discharge]
        dispatch_frame["soc_mwh"] = x[soc]
    else:
        dispatch_frame["charge_mw"] = 0.0
        dispatch_frame["discharge_mw"] = 0.0
        dispatch_frame["soc_mwh"] = 0.0

    generator_frames: list[pd.DataFrame] = []
    next_state: dict[str, dict[str, float | int]] = {}
    for unit in fleet:
        name = str(unit["name"])
        frame = day_frame[["timestamp"]].copy()
        frame["generator"] = name
        frame["fuel_type"] = str(unit["fuel_type"])
        frame["dispatch_mw"] = x[dispatch_idx[name]]
        if enforce_ramping and diagnose_ramping:
            frame["ramp_up_violation_mw"] = x[ramp_up_slack_idx[name]]
            frame["ramp_down_violation_mw"] = x[ramp_down_slack_idx[name]]
        else:
            frame["ramp_up_violation_mw"] = 0.0
            frame["ramp_down_violation_mw"] = 0.0
        if bool(unit["binary"]):
            commits = np.rint(x[commit_idx[name]]).astype(int)
            startups = np.rint(x[startup_idx[name]]).astype(int)
            shutdowns = np.rint(x[shutdown_idx[name]]).astype(int)
            frame["commitment"] = commits
            frame["startup"] = startups
            frame["shutdown"] = shutdowns
            final_status = int(commits[-1])
            duration = 0
            for value in commits[::-1]:
                if int(value) == final_status:
                    duration += 1
                else:
                    break
            next_state[name] = {
                "commitment": final_status,
                "output_mw": float(frame["dispatch_mw"].iloc[-1]),
                "time_on_hours": int(duration if final_status == 1 else 0),
                "time_off_hours": int(duration if final_status == 0 else 0),
            }
        else:
            frame["commitment"] = np.nan
            frame["startup"] = np.nan
            frame["shutdown"] = np.nan
            next_state[name] = {
                "commitment": 1,
                "output_mw": float(frame["dispatch_mw"].iloc[-1]),
                "time_on_hours": 999,
                "time_off_hours": 0,
            }
        generator_frames.append(frame)

    generator_dispatch = pd.concat(generator_frames, ignore_index=True)
    gas_hourly = (
        generator_dispatch[generator_dispatch["fuel_type"] == "GAS"]
        .groupby("timestamp", as_index=False)["dispatch_mw"]
        .sum()
        .rename(columns={"dispatch_mw": "optimized_gas_dispatch_mw"})
    )
    dispatch_frame = dispatch_frame.merge(gas_hourly, on="timestamp", how="left")
    dispatch_frame["optimized_gas_dispatch_mw"] = dispatch_frame["optimized_gas_dispatch_mw"].fillna(0.0)
    dispatch_frame["residual_nonrenewable_mw"] = dispatch_frame["ontario_demand_mw"] - dispatch_frame["renewable_direct_mw"] - dispatch_frame["discharge_mw"]
    solve_meta = {
        "objective_value": float(result.fun),
        "mip_gap": float(getattr(result, "mip_gap", np.nan)),
        "status": int(getattr(result, "status", -1)),
        "total_ramp_up_violation_mw": float(generator_dispatch["ramp_up_violation_mw"].sum()),
        "total_ramp_down_violation_mw": float(generator_dispatch["ramp_down_violation_mw"].sum()),
    }
    return dispatch_frame, generator_dispatch, next_state, solve_meta


def _build_historical_dispatch_frame(
    master_overlap: pd.DataFrame,
    observed_storage_hourly: pd.DataFrame,
    scenario: dict[str, float | str],
) -> pd.DataFrame:
    observed = master_overlap.merge(
        observed_storage_hourly[["timestamp", "storage_capability_mw", "observed_storage_output_mw", "battery_power_utilization"]],
        on="timestamp",
        how="left",
    )
    observed["storage_capability_mw"] = observed["storage_capability_mw"].fillna(0.0)
    observed["observed_storage_output_mw"] = observed["observed_storage_output_mw"].fillna(0.0)
    observed["battery_power_utilization"] = observed["battery_power_utilization"].fillna(0.0)

    frame = observed[
        ["timestamp", "season", "hour", "ontario_demand_mw", "renewable_actual_mw", "historical_gas_dispatch_mw", "high_gas_hour"]
    ].copy()
    frame["scenario"] = CURRENT_SCENARIO_NAME
    frame["scenario_label"] = scenario["label"]
    frame["policy"] = "historical_actual"
    frame["storage_power_mw"] = float(scenario["power_mw"])
    frame["storage_energy_mwh"] = float(scenario["energy_mwh"])
    frame["renewable_input_mw"] = frame["renewable_actual_mw"]
    frame["renewable_direct_mw"] = frame["renewable_actual_mw"]
    frame["charge_mw"] = 0.0
    frame["discharge_mw"] = observed["observed_storage_output_mw"].to_numpy()
    frame["curtail_mw"] = 0.0
    frame["soc_mwh"] = 0.0
    frame["load_shed_mw"] = 0.0
    frame["overgeneration_mw"] = 0.0
    frame["battery_energy_utilization"] = 0.0
    frame["battery_power_utilization"] = observed["battery_power_utilization"].to_numpy()
    frame["renewable_utilization_rate"] = 1.0
    frame["renewable_backed_battery_discharge_mw"] = np.nan
    frame["renewable_backed_supply_mw"] = np.nan
    frame["optimized_gas_dispatch_mw"] = np.nan
    frame["residual_nonrenewable_mw"] = frame["ontario_demand_mw"] - frame["renewable_actual_mw"] - frame["discharge_mw"]
    frame["storage_dispatch_during_high_gas_hours_mw"] = 0.0
    return frame


def _summarize_policy(dispatch: pd.DataFrame) -> pd.DataFrame:
    summary = (
        dispatch.groupby(["scenario", "scenario_label", "policy", "storage_power_mw", "storage_energy_mwh"], as_index=False)
        .agg(
            overlap_hours=("timestamp", "count"),
            total_charge_mwh=("charge_mw", "sum"),
            total_discharge_mwh=("discharge_mw", "sum"),
            mean_battery_power_utilization=("battery_power_utilization", "mean"),
            max_battery_power_utilization=("battery_power_utilization", "max"),
            mean_battery_energy_utilization=("battery_energy_utilization", "mean"),
            renewable_input_mwh=("renewable_input_mw", "sum"),
            renewable_direct_mwh=("renewable_direct_mw", "sum"),
            curtailment_mwh=("curtail_mw", "sum"),
            renewable_backed_battery_discharge_mwh=("renewable_backed_battery_discharge_mw", "sum"),
            renewable_backed_supply_mwh=("renewable_backed_supply_mw", "sum"),
            peak_residual_nonrenewable_mw=("residual_nonrenewable_mw", "max"),
            mean_residual_nonrenewable_mw=("residual_nonrenewable_mw", "mean"),
            load_shed_mwh=("load_shed_mw", "sum"),
            overgeneration_mwh=("overgeneration_mw", "sum"),
            optimized_gas_dispatch_mwh=("optimized_gas_dispatch_mw", "sum"),
            optimized_gas_peak_mw=("optimized_gas_dispatch_mw", "max"),
            high_gas_hour_battery_discharge_mwh=("storage_dispatch_during_high_gas_hours_mw", "sum"),
            historical_gas_context_mwh=("historical_gas_dispatch_mw", "sum"),
            high_gas_hour_count=("high_gas_hour", "sum"),
        )
    )
    day_count = summary["overlap_hours"] / 24.0
    summary["battery_throughput_utilization"] = summary["total_discharge_mwh"] / (
        summary["storage_energy_mwh"].replace(0.0, np.nan) * day_count.replace(0.0, np.nan)
    )
    summary["battery_throughput_utilization"] = summary["battery_throughput_utilization"].fillna(0.0)
    summary["renewable_utilization_rate"] = (
        (summary["renewable_direct_mwh"] + summary["total_charge_mwh"]) / summary["renewable_input_mwh"].replace(0.0, np.nan)
    ).fillna(0.0)
    summary["curtailment_rate"] = (
        summary["curtailment_mwh"] / summary["renewable_input_mwh"].replace(0.0, np.nan)
    ).fillna(0.0)
    summary["renewable_backed_average_mw"] = (
        summary["renewable_backed_supply_mwh"] / summary["overlap_hours"].replace(0.0, np.nan)
    ).fillna(0.0)
    summary["mean_battery_output_mw"] = (
        summary["total_discharge_mwh"] / summary["overlap_hours"].replace(0.0, np.nan)
    ).fillna(0.0)
    high_gas_hours = summary["high_gas_hour_count"].replace(0.0, np.nan)
    summary["high_gas_hour_average_discharge_mw"] = (
        summary["high_gas_hour_battery_discharge_mwh"] / high_gas_hours
    ).fillna(0.0)
    historical_high_gas = (
        dispatch[dispatch["high_gas_hour"] == 1]
        .groupby(["scenario", "policy"], as_index=False)["historical_gas_dispatch_mw"]
        .sum()
        .rename(columns={"historical_gas_dispatch_mw": "historical_high_gas_mwh"})
    )
    summary = summary.merge(historical_high_gas, on=["scenario", "policy"], how="left")
    summary["historical_high_gas_mwh"] = summary["historical_high_gas_mwh"].fillna(0.0)
    summary["high_gas_hour_coverage_ratio"] = (
        summary["high_gas_hour_battery_discharge_mwh"] / summary["historical_high_gas_mwh"].replace(0.0, np.nan)
    ).fillna(0.0)
    return summary


def _summarize_ramp_violations(generator_dispatch: pd.DataFrame) -> pd.DataFrame:
    if generator_dispatch.empty or "ramp_up_violation_mw" not in generator_dispatch.columns:
        return pd.DataFrame(
            columns=[
                "policy",
                "generator",
                "fuel_type",
                "total_ramp_up_violation_mw",
                "total_ramp_down_violation_mw",
                "total_ramp_violation_mw",
                "max_hourly_ramp_up_violation_mw",
                "max_hourly_ramp_down_violation_mw",
                "hours_with_ramp_violation",
            ]
        )
    summary = (
        generator_dispatch.groupby(["policy", "generator", "fuel_type"], as_index=False)
        .agg(
            total_ramp_up_violation_mw=("ramp_up_violation_mw", "sum"),
            total_ramp_down_violation_mw=("ramp_down_violation_mw", "sum"),
            max_hourly_ramp_up_violation_mw=("ramp_up_violation_mw", "max"),
            max_hourly_ramp_down_violation_mw=("ramp_down_violation_mw", "max"),
        )
    )
    summary["total_ramp_violation_mw"] = summary["total_ramp_up_violation_mw"] + summary["total_ramp_down_violation_mw"]
    violation_hours = generator_dispatch.copy()
    violation_hours["has_violation"] = (
        (violation_hours["ramp_up_violation_mw"] > 1e-6) | (violation_hours["ramp_down_violation_mw"] > 1e-6)
    ).astype(int)
    counts = (
        violation_hours.groupby(["policy", "generator"], as_index=False)["has_violation"]
        .sum()
        .rename(columns={"has_violation": "hours_with_ramp_violation"})
    )
    summary = summary.merge(counts, on=["policy", "generator"], how="left")
    summary["hours_with_ramp_violation"] = summary["hours_with_ramp_violation"].fillna(0).astype(int)
    return summary[summary["total_ramp_violation_mw"] > 1e-6].sort_values(
        ["policy", "total_ramp_violation_mw"], ascending=[True, False]
    ).reset_index(drop=True)


def _build_solve_blocks(frame: pd.DataFrame, horizon_hours: int, max_blocks: int | None = None) -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    blocks: list[tuple[pd.Timestamp, pd.DataFrame]] = []
    for start in range(0, len(ordered), horizon_hours):
        block = ordered.iloc[start : start + horizon_hours].copy()
        if block.empty:
            continue
        blocks.append((pd.Timestamp(block["timestamp"].iloc[0]).normalize(), block.reset_index(drop=True)))
        if max_blocks is not None and len(blocks) >= max_blocks:
            break
    return blocks


def _policy_to_renewable_column(policy_name: str) -> str:
    if policy_name == "perfect_foresight_uc":
        return "renewable_perfect_mw"
    if policy_name == "ieso_forecast_uc":
        return "renewable_ieso_raw_mw"
    return "renewable_best_ml_mw"


def _policy_to_battery_mode(policy_name: str) -> str:
    return "disabled" if policy_name == "no_storage_uc" else "optimized"


def main() -> None:
    args = _parse_args()
    output_dir = OUTPUT_ROOT / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    master = load_master_dataset()
    predictions = pd.read_csv(RESULTS_DIR / "all_fuel_predictions.csv", parse_dates=["timestamp"])
    dispatchable = load_dispatchable_generator_dataset()
    observed_storage = load_observed_storage_dataset()
    observed_storage_hourly = _aggregate_observed_storage(observed_storage)

    default_start = observed_storage_hourly.loc[observed_storage_hourly["storage_capability_mw"] > 0, "timestamp"].min()
    default_end = observed_storage_hourly["timestamp"].max()
    overlap_start = pd.Timestamp(args.start_date) if args.start_date else default_start
    overlap_end = pd.Timestamp(args.end_date) + pd.Timedelta(hours=23) if args.end_date else default_end

    renewable_frame = _build_renewable_policy_frame(master, predictions)
    master_overlap = renewable_frame[
        (renewable_frame["timestamp"] >= overlap_start) & (renewable_frame["timestamp"] <= overlap_end)
    ].copy()
    dispatchable_overlap = dispatchable[
        (dispatchable["timestamp"] >= overlap_start) & (dispatchable["timestamp"] <= overlap_end)
    ].copy()

    gas_context = _build_historical_gas_context(dispatchable, overlap_start, overlap_end)
    stress_profile = _build_stress_profile(master_overlap, gas_context)
    master_overlap = master_overlap.merge(
        stress_profile[["timestamp", "historical_gas_dispatch_mw", "high_gas_hour", "stress_cost_adder"]],
        on="timestamp",
        how="left",
    )

    historical_scenario = _build_storage_scenario(
        observed_storage_hourly[
            (observed_storage_hourly["timestamp"] >= overlap_start) & (observed_storage_hourly["timestamp"] <= overlap_end)
        ]
    )
    scenario = _build_storage_scenario(
        observed_storage_hourly[
            (observed_storage_hourly["timestamp"] >= overlap_start) & (observed_storage_hourly["timestamp"] <= overlap_end)
        ],
        power_override_mw=args.storage_power_mw,
        energy_override_mwh=args.storage_energy_mwh,
        duration_hours=float(args.storage_duration_hours),
    )

    solve_blocks = _build_solve_blocks(
        master_overlap,
        horizon_hours=int(args.horizon_hours),
        max_blocks=args.max_blocks,
    )
    if not solve_blocks:
        raise RuntimeError("No solve blocks available in the requested range.")

    dispatch_frames: list[pd.DataFrame] = []
    generator_frames: list[pd.DataFrame] = []
    solve_log: list[dict[str, object]] = []

    observed_overlap = observed_storage_hourly[
        (observed_storage_hourly["timestamp"] >= solve_blocks[0][1]["timestamp"].iloc[0])
        & (observed_storage_hourly["timestamp"] <= solve_blocks[-1][1]["timestamp"].iloc[-1])
    ].copy()
    historical_dispatch = _build_historical_dispatch_frame(
        master_overlap=master_overlap[
            (master_overlap["timestamp"] >= solve_blocks[0][1]["timestamp"].iloc[0])
            & (master_overlap["timestamp"] <= solve_blocks[-1][1]["timestamp"].iloc[-1])
        ].copy(),
        observed_storage_hourly=observed_overlap,
        scenario=historical_scenario,
    )
    dispatch_frames.append(historical_dispatch)

    base_initial_state = _compute_initial_state(dispatchable, pd.Timestamp(solve_blocks[0][1]["timestamp"].iloc[0]))
    initial_state_summary = _summarize_initial_state(base_initial_state)
    print(
        f"Initial UC state from prior hour: {initial_state_summary['units_on_prior_hour']} units on, "
        f"{initial_state_summary['units_off_prior_hour']} units off.",
        flush=True,
    )
    policy_specs = [
        (policy, _policy_to_renewable_column(policy), _policy_to_battery_mode(policy))
        for policy in args.policies
    ]

    total_blocks = len(solve_blocks)
    for policy_name, renewable_column, battery_policy in policy_specs:
        print(f"Running policy {policy_name} across {total_blocks} block(s) of {args.horizon_hours} hour(s)...", flush=True)
        policy_state = {
            key: value.copy() for key, value in base_initial_state.items()
        }
        for block_index, (block_key, block_frame) in enumerate(solve_blocks, start=1):
            del block_key
            day_frame = block_frame.sort_values("timestamp").reset_index(drop=True).copy()
            block_start = pd.Timestamp(day_frame["timestamp"].iloc[0])
            block_end = pd.Timestamp(day_frame["timestamp"].iloc[-1])
            day_dispatchable = dispatchable_overlap[
                (dispatchable_overlap["timestamp"] >= block_start)
                & (dispatchable_overlap["timestamp"] <= block_end)
            ].copy()
            start = time.time()
            dispatch_day, generators_day, policy_state, solve_meta = _solve_uc_day(
                day_frame=day_frame,
                day_dispatchable=day_dispatchable,
                storage_power_mw=float(scenario["power_mw"]),
                storage_energy_mwh=float(scenario["energy_mwh"]),
                renewable_column=renewable_column,
                battery_policy=battery_policy,
                initial_state=policy_state,
                time_limit_per_day=float(args.time_limit_per_day),
                mip_rel_gap=float(args.mip_rel_gap),
                enforce_ramping=not args.disable_ramping,
                enforce_min_updown=not args.disable_min_updown,
                continuous_ramp_multiplier=float(args.continuous_ramp_multiplier),
                diagnose_ramping=bool(args.diagnose_ramping),
                ramp_slack_penalty=float(args.ramp_slack_penalty),
            )
            elapsed = time.time() - start
            print(
                f"[{policy_name}] block {block_index}/{total_blocks} "
                f"{block_start.strftime('%Y-%m-%d %H:%M')} to {block_end.strftime('%Y-%m-%d %H:%M')} solved in {elapsed:.1f}s; "
                f"peak battery discharge={dispatch_day['discharge_mw'].max():.1f} MW; "
                f"optimized gas peak={dispatch_day['optimized_gas_dispatch_mw'].max():.1f} MW; "
                f"ramp slack={solve_meta['total_ramp_up_violation_mw'] + solve_meta['total_ramp_down_violation_mw']:.1f} MW",
                flush=True,
            )
            dispatch_day["scenario"] = CURRENT_SCENARIO_NAME
            dispatch_day["scenario_label"] = scenario["label"]
            dispatch_day["policy"] = policy_name
            dispatch_day["storage_power_mw"] = float(scenario["power_mw"])
            dispatch_day["storage_energy_mwh"] = float(scenario["energy_mwh"])
            dispatch_day["battery_power_utilization"] = np.where(
                float(scenario["power_mw"]) > 0,
                dispatch_day["discharge_mw"] / float(scenario["power_mw"]),
                0.0,
            ).clip(0.0, 1.0)
            dispatch_day["battery_energy_utilization"] = np.where(
                float(scenario["energy_mwh"]) > 0,
                dispatch_day["soc_mwh"] / float(scenario["energy_mwh"]),
                0.0,
            ).clip(0.0, 1.0)
            dispatch_day["renewable_utilization_rate"] = (
                (dispatch_day["renewable_direct_mw"] + dispatch_day["charge_mw"])
                / dispatch_day["renewable_input_mw"].replace(0.0, np.nan)
            ).fillna(0.0)
            dispatch_day["renewable_backed_battery_discharge_mw"] = dispatch_day["discharge_mw"]
            dispatch_day["renewable_backed_supply_mw"] = dispatch_day["renewable_direct_mw"] + dispatch_day["discharge_mw"]
            dispatch_day["storage_dispatch_during_high_gas_hours_mw"] = np.where(
                dispatch_day["high_gas_hour"] == 1,
                dispatch_day["discharge_mw"],
                0.0,
            )
            dispatch_frames.append(dispatch_day)

            generators_day["scenario"] = CURRENT_SCENARIO_NAME
            generators_day["policy"] = policy_name
            generator_frames.append(generators_day)
            solve_log.append(
                {
                    "policy": policy_name,
                    "block_start": str(block_start),
                    "block_end": str(block_end),
                    "elapsed_seconds": elapsed,
                    **solve_meta,
                }
            )

    dispatch = pd.concat(dispatch_frames, ignore_index=True)
    generator_dispatch = pd.concat(generator_frames, ignore_index=True) if generator_frames else pd.DataFrame()
    ramp_violation_summary = _summarize_ramp_violations(generator_dispatch)
    policy_summary = _summarize_policy(dispatch)
    historical = policy_summary[policy_summary["policy"] == "historical_actual"].iloc[0]
    if "no_storage_uc" in policy_summary["policy"].values:
        no_storage = policy_summary[policy_summary["policy"] == "no_storage_uc"].iloc[0]
        policy_summary["optimized_gas_dispatch_reduction_vs_no_storage_mwh"] = no_storage["optimized_gas_dispatch_mwh"] - policy_summary["optimized_gas_dispatch_mwh"]
        policy_summary["optimized_gas_peak_reduction_vs_no_storage_mw"] = no_storage["optimized_gas_peak_mw"] - policy_summary["optimized_gas_peak_mw"]
    else:
        policy_summary["optimized_gas_dispatch_reduction_vs_no_storage_mwh"] = np.nan
        policy_summary["optimized_gas_peak_reduction_vs_no_storage_mw"] = np.nan
    policy_summary["battery_utilization_improvement_vs_historical_pp"] = (
        policy_summary["mean_battery_power_utilization"] - historical["mean_battery_power_utilization"]
    ) * 100.0
    policy_summary["peak_residual_nonrenewable_reduction_vs_historical_mw"] = (
        historical["peak_residual_nonrenewable_mw"] - policy_summary["peak_residual_nonrenewable_mw"]
    )
    order = {name: idx for idx, name in enumerate(POLICY_ORDER)}
    policy_summary = policy_summary.sort_values("policy", key=lambda s: s.map(order).fillna(999)).reset_index(drop=True)

    gas_dispatch_summary = dispatch[
        [
            "timestamp",
            "scenario",
            "scenario_label",
            "policy",
            "historical_gas_dispatch_mw",
            "optimized_gas_dispatch_mw",
            "high_gas_hour",
            "discharge_mw",
            "storage_dispatch_during_high_gas_hours_mw",
        ]
    ].copy()

    observed_overlap.to_csv(output_dir / "observed_storage_hourly_baseline.csv", index=False)
    dispatch.to_csv(output_dir / "storage_uc_dispatch_hourly.csv", index=False)
    if not generator_dispatch.empty:
        generator_dispatch.to_csv(output_dir / "generator_uc_dispatch_hourly.csv", index=False)
    if not ramp_violation_summary.empty:
        ramp_violation_summary.to_csv(output_dir / "ramp_violation_summary.csv", index=False)
    gas_dispatch_summary.to_csv(output_dir / "gas_dispatch_summary.csv", index=False)
    policy_summary.to_csv(output_dir / "storage_uc_policy_summary.csv", index=False)
    pd.DataFrame.from_records(solve_log).to_csv(output_dir / "solve_log.csv", index=False)

    with (output_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "overlap_start": str(solve_blocks[0][1]["timestamp"].iloc[0]),
                "overlap_end": str(solve_blocks[-1][1]["timestamp"].iloc[-1]),
                "blocks_solved": len(solve_blocks),
                "horizon_hours": int(args.horizon_hours),
                "historical_baseline_scenario": historical_scenario,
                "optimization_scenario": scenario,
                "initial_state_summary": initial_state_summary,
                "policies": ["historical_actual"] + args.policies,
                "time_limit_per_day_seconds": args.time_limit_per_day,
                "mip_rel_gap": args.mip_rel_gap,
                "continuous_ramp_multiplier": args.continuous_ramp_multiplier,
                "diagnose_ramping": bool(args.diagnose_ramping),
                "ramp_slack_penalty": float(args.ramp_slack_penalty),
                "notes": [
                    "Generator-level UC experiment solved as a rolling MILP over configurable block horizons.",
                    "Gas and biofuel units are binary UC units; hydro and nuclear are continuous dispatch units.",
                    "The model includes generator dispatch, commitment, startup/shutdown, ramping, minimum up/down, battery SOC, and hourly system power balance.",
                    "Minimum up/down carryover across the first modeled day is approximated from recent actual operating history.",
                    "historical_actual always uses the observed current storage fleet from source files.",
                    "Optimization policies can optionally override storage power and energy for case-study analysis.",
                    f"Ramping enabled: {not args.disable_ramping}. Minimum up/down enabled: {not args.disable_min_updown}.",
                    f"Continuous HYDRO/NUCLEAR ramp multiplier: {args.continuous_ramp_multiplier}.",
                ],
            },
            handle,
            indent=2,
        )

    print("Generator-level UC run complete.", flush=True)
    print(policy_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
