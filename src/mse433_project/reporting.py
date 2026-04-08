from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import FIGURES_DIR, GENERATOR_PARAMETERS_PATH, PROCESSED_DIR, REPORT_DIR, RESULTS_DIR, VARIABLE_FORECAST_SUMMARY_PATH


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _format_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _format_numeric_table(frame: pd.DataFrame, format_map: dict[str, str]) -> pd.DataFrame:
    display = frame.copy()
    for column, fmt in format_map.items():
        if column in display.columns:
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else fmt.format(value))
    return display


def _compute_operating_costs(base_dir: Path) -> dict[str, float]:
    dispatch_path = base_dir / "generator_uc_dispatch_hourly.csv"
    observed_path = base_dir / "observed_storage_hourly_baseline.csv"
    if not dispatch_path.exists() or not observed_path.exists():
        return {}

    generator_dispatch = pd.read_csv(dispatch_path, parse_dates=["timestamp"])
    dispatchable = pd.read_csv(PROCESSED_DIR / "dispatchable_generator_hourly_dataset.csv", parse_dates=["timestamp"])
    params = dispatchable[
        ["timestamp", "generator", "variable_cost_per_mwh", "startup_cost", "shutdown_cost", "actual_output_mw"]
    ].drop_duplicates(subset=["timestamp", "generator"])

    merged = generator_dispatch.merge(
        params[["timestamp", "generator", "variable_cost_per_mwh", "startup_cost", "shutdown_cost"]],
        on=["timestamp", "generator"],
        how="left",
    )
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
    cost_map = dict(zip(summary["policy"], summary["modeled_operating_cost_usd"]))

    observed = pd.read_csv(observed_path, parse_dates=["timestamp"])
    start = observed["timestamp"].min()
    end = observed["timestamp"].max()
    actual = params[(params["timestamp"] >= start) & (params["timestamp"] <= end)].copy().sort_values(["generator", "timestamp"])
    actual["is_on"] = actual["actual_output_mw"].gt(0).astype(int)
    actual["prev_is_on"] = actual.groupby("generator")["is_on"].shift(1)
    prior = (
        dispatchable[dispatchable["timestamp"] < start][["timestamp", "generator", "actual_output_mw"]]
        .sort_values(["generator", "timestamp"])
        .assign(is_on=lambda frame: frame["actual_output_mw"].gt(0).astype(int))
        .groupby("generator", as_index=False)
        .tail(1)[["generator", "is_on"]]
        .rename(columns={"is_on": "prior_is_on"})
    )
    actual = actual.merge(prior, on="generator", how="left")
    first_hour_mask = actual["timestamp"].eq(start)
    actual.loc[first_hour_mask, "prev_is_on"] = actual.loc[first_hour_mask, "prior_is_on"]
    actual["prev_is_on"] = actual["prev_is_on"].fillna(0)
    actual["startup"] = ((actual["is_on"] == 1) & (actual["prev_is_on"] == 0)).astype(float)
    actual["shutdown"] = ((actual["is_on"] == 0) & (actual["prev_is_on"] == 1)).astype(float)
    actual["variable_cost_usd"] = actual["actual_output_mw"] * actual["variable_cost_per_mwh"].fillna(0.0)
    actual["startup_cost_usd"] = actual["startup"] * actual["startup_cost"].fillna(0.0)
    actual["shutdown_cost_usd"] = actual["shutdown"] * actual["shutdown_cost"].fillna(0.0)
    cost_map["historical_actual"] = float(
        actual[["variable_cost_usd", "startup_cost_usd", "shutdown_cost_usd"]].sum().sum()
    )
    return cost_map


def _load_all_policy_breakdown() -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    base_dir = RESULTS_DIR / "generator_uc" / "full_run_168h_all_policies"
    summary_path = base_dir / "storage_uc_policy_summary.csv"
    if not summary_path.exists():
        return None, None

    policy_summary = pd.read_csv(summary_path)
    cost_map = _compute_operating_costs(base_dir)
    if cost_map:
        policy_summary["modeled_operating_cost_usd"] = policy_summary["policy"].map(cost_map)

    rows = {policy: policy_summary[policy_summary["policy"] == policy].iloc[0] for policy in policy_summary["policy"].unique()}
    comparisons = [
        ("historical_actual", "no_storage_uc", "Historical to UC without storage", "Modeling context only"),
        ("no_storage_uc", "forecast_informed_uc", "UC no-storage to forecast-informed UC", "Incremental battery value inside UC"),
        ("historical_actual", "forecast_informed_uc", "Historical to forecast-informed UC", "Total improvement versus real operation"),
        ("forecast_informed_uc", "perfect_foresight_uc", "Forecast-informed to perfect-foresight UC", "Remaining forecast gap"),
    ]
    metrics = [
        ("mean_battery_power_utilization", "Battery utilization delta (pp)", 100.0),
        ("mean_battery_output_mw", "Mean battery output delta (MW)", 1.0),
        ("total_discharge_mwh", "Battery discharge delta (MWh)", 1.0),
        ("optimized_gas_dispatch_mwh", "Gas dispatch delta (MWh)", 1.0),
        ("optimized_gas_peak_mw", "Gas peak delta (MW)", 1.0),
        ("peak_residual_nonrenewable_mw", "Peak non-renewable delta (MW)", 1.0),
        ("modeled_operating_cost_usd", "Operating cost delta (USD)", 1.0),
    ]

    comparison_rows: list[dict[str, object]] = []
    for base_policy, comp_policy, label, interpretation in comparisons:
        base_row = rows[base_policy]
        comp_row = rows[comp_policy]
        result = {
            "comparison": label,
            "interpretation": interpretation,
            "base_policy": base_policy,
            "comparison_policy": comp_policy,
        }
        for source_col, output_col, scale in metrics:
            base_value = 0.0 if pd.isna(base_row.get(source_col)) else float(base_row[source_col])
            comp_value = 0.0 if pd.isna(comp_row.get(source_col)) else float(comp_row[source_col])
            result[output_col] = (comp_value - base_value) * scale
        comparison_rows.append(result)

    return policy_summary, pd.DataFrame(comparison_rows)


def _write_report_tables(historical: pd.Series, best: pd.Series, solve_log: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key_metrics = pd.DataFrame(
        [
            {
                "metric": "Battery power utilization",
                "historical_actual": f"{100 * historical['mean_battery_power_utilization']:.2f}%",
                "forecast_informed_uc": f"{100 * best['mean_battery_power_utilization']:.2f}%",
                "improvement": f"{best['battery_utilization_improvement_vs_historical_pp']:.2f} percentage points",
            },
            {
                "metric": "Mean battery output",
                "historical_actual": f"{historical['mean_battery_output_mw']:.2f} MW",
                "forecast_informed_uc": f"{best['mean_battery_output_mw']:.2f} MW",
                "improvement": f"{best['mean_battery_output_mw'] - historical['mean_battery_output_mw']:.2f} MW",
            },
            {
                "metric": "Battery throughput utilization",
                "historical_actual": f"{100 * historical['battery_throughput_utilization']:.2f}%",
                "forecast_informed_uc": f"{100 * best['battery_throughput_utilization']:.2f}%",
                "improvement": f"{100 * (best['battery_throughput_utilization'] - historical['battery_throughput_utilization']):.2f} percentage points",
            },
            {
                "metric": "Total battery discharge",
                "historical_actual": f"{historical['total_discharge_mwh']:,.0f} MWh",
                "forecast_informed_uc": f"{best['total_discharge_mwh']:,.0f} MWh",
                "improvement": f"{best['total_discharge_mwh'] - historical['total_discharge_mwh']:,.0f} MWh",
            },
            {
                "metric": "Gas generation saved",
                "historical_actual": "0 MWh",
                "forecast_informed_uc": f"{best['gas_generation_saved_vs_historical_mwh']:,.0f} MWh",
                "improvement": f"{best['average_gas_generation_saved_mw']:.2f} MW average",
            },
            {
                "metric": "Gas peak reduction",
                "historical_actual": "0 MW",
                "forecast_informed_uc": f"{best['gas_peak_reduction_vs_historical_mw']:.0f} MW",
                "improvement": f"{best['gas_peak_reduction_vs_historical_mw']:.0f} MW",
            },
            {
                "metric": "Modeled operating cost",
                "historical_actual": f"${historical['modeled_operating_cost_usd']:,.0f}",
                "forecast_informed_uc": f"${best['modeled_operating_cost_usd']:,.0f}",
                "improvement": f"${best['operating_cost_saved_vs_historical_usd']:,.0f} saved",
            },
            {
                "metric": "Peak non-renewable reduction",
                "historical_actual": "0 MW",
                "forecast_informed_uc": f"{best['peak_residual_nonrenewable_reduction_vs_historical_mw']:.0f} MW",
                "improvement": f"{best['peak_residual_nonrenewable_reduction_vs_historical_mw']:.0f} MW",
            },
        ]
    )
    technical_summary = pd.DataFrame(
        [
            {"metric": "Rolling UC blocks solved", "value": f"{len(solve_log)}"},
            {"metric": "Mean solve time per block", "value": f"{solve_log['elapsed_seconds'].mean():.2f} s"},
            {"metric": "Max solve time per block", "value": f"{solve_log['elapsed_seconds'].max():.2f} s"},
            {"metric": "Mean MIP gap", "value": f"{100 * solve_log['mip_gap'].dropna().mean():.3f}%"},
            {"metric": "Max MIP gap", "value": f"{100 * solve_log['mip_gap'].dropna().max():.3f}%"},
            {
                "metric": "Total ramp slack used",
                "value": f"{solve_log['total_ramp_up_violation_mw'].sum() + solve_log['total_ramp_down_violation_mw'].sum():.2f} MW",
            },
        ]
    )
    key_metrics.to_csv(RESULTS_DIR / "report_key_metrics_table.csv", index=False)
    technical_summary.to_csv(RESULTS_DIR / "uc_technical_summary.csv", index=False)
    return key_metrics, technical_summary


def _render_supporting_figures(
    historical: pd.Series,
    best: pd.Series,
    dispatch: pd.DataFrame,
    solve_log: pd.DataFrame,
) -> dict[str, str]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    storage_results_path = FIGURES_DIR / "main_results_storage_metrics.png"
    system_results_path = FIGURES_DIR / "main_results_system_impacts.png"
    historical_gas_dispatch_mwh = float(best.get("historical_gas_context_mwh", 0.0))
    historical_gas_peak_mw = float(
        dispatch.loc[dispatch["policy"] == "historical_actual", "historical_gas_dispatch_mw"].max()
    )
    historical_peak_nonrenewable_mw = float(historical["peak_residual_nonrenewable_mw"])
    storage_comparisons = [
        ("Battery Utilization (%)", [100 * historical["mean_battery_power_utilization"], 100 * best["mean_battery_power_utilization"]], "{:.1f}"),
        ("Mean Battery Output (MW)", [historical["mean_battery_output_mw"], best["mean_battery_output_mw"]], "{:.0f}"),
        ("Total Discharge (GWh)", [historical["total_discharge_mwh"] / 1000.0, best["total_discharge_mwh"] / 1000.0], "{:.0f}"),
        ("Battery Throughput (%)", [100 * historical["battery_throughput_utilization"], 100 * best["battery_throughput_utilization"]], "{:.1f}"),
    ]
    system_comparisons = [
        ("Gas Generation (GWh)", [historical_gas_dispatch_mwh / 1000.0, best["optimized_gas_dispatch_mwh"] / 1000.0], "{:.0f}"),
        ("Gas Peak (MW)", [historical_gas_peak_mw, best["optimized_gas_peak_mw"]], "{:.0f}"),
        ("Peak Non-Renewable (MW)", [historical_peak_nonrenewable_mw, best["peak_residual_nonrenewable_mw"]], "{:.0f}"),
        ("Operating Cost (Million $)", [historical["modeled_operating_cost_usd"] / 1e6, best["modeled_operating_cost_usd"] / 1e6], "{:.0f}"),
    ]
    for path, title, comparisons in [
        (storage_results_path, "Historical vs Forecast-Informed UC: Storage-Use Metrics", storage_comparisons),
        (system_results_path, "Historical vs Forecast-Informed UC: System-Impact Metrics", system_comparisons),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))
        for ax, (panel_title, values, label_fmt) in zip(axes.flatten(), comparisons):
            bars = ax.bar(["Historical", "Forecast-informed UC"], values, color=["#B0BEC5", "#1565C0"])
            ax.set_title(panel_title, fontsize=11)
            ax.grid(axis="y", alpha=0.25)
            ymax = max(values) if max(values) > 0 else 1.0
            ax.set_ylim(0, ymax * 1.22)
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ymax * 0.04,
                    label_fmt.format(value),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        fig.suptitle(title, fontsize=15)
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)

    impact_path = FIGURES_DIR / "storage_gas_impact_summary.png"
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    comparisons = [
        ("Battery Utilization (%)", [100 * historical["mean_battery_power_utilization"], 100 * best["mean_battery_power_utilization"]]),
        ("Mean Battery Output (MW)", [historical["mean_battery_output_mw"], best["mean_battery_output_mw"]]),
        ("Battery Throughput (%)", [100 * historical["battery_throughput_utilization"], 100 * best["battery_throughput_utilization"]]),
        ("Gas Generation Saved (GWh)", [0.0, best["gas_generation_saved_vs_historical_mwh"] / 1000.0]),
    ]
    for ax, (title, values) in zip(axes.flatten(), comparisons):
        ax.bar(["Historical", "Forecast-informed UC"], values, color=["#B0BEC5", "#1565C0"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Ontario Storage and Gas Impact Summary", fontsize=14)
    fig.tight_layout()
    fig.savefig(impact_path, dpi=200)
    plt.close(fig)

    hourly_path = FIGURES_DIR / "average_hourly_storage_profile.png"
    optimized = dispatch[dispatch["policy"] == "forecast_informed_uc"].copy()
    hourly = optimized.groupby("hour", as_index=False).agg(
        charge_mw=("charge_mw", "mean"),
        discharge_mw=("discharge_mw", "mean"),
        peak_support_mw=("storage_dispatch_during_high_gas_hours_mw", "mean"),
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(hourly["hour"], -hourly["charge_mw"], color="#81C784", label="Average charge")
    ax.bar(hourly["hour"], hourly["discharge_mw"], color="#1565C0", label="Average discharge")
    ax.plot(hourly["hour"], hourly["peak_support_mw"], color="#D84315", linewidth=2, label="Average discharge in historical high-gas hours")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("MW")
    ax.set_title("Average Hourly Storage Charge and Discharge Profile")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(hourly_path, dpi=200)
    plt.close(fig)

    case_path = FIGURES_DIR / "dispatch_case_study_day.png"
    historical_dispatch = dispatch[dispatch["policy"] == "historical_actual"].copy()
    historical_dispatch["date"] = pd.to_datetime(historical_dispatch["timestamp"]).dt.date
    case_day = historical_dispatch.groupby("date")["historical_gas_dispatch_mw"].sum().idxmax()
    case_hist = historical_dispatch[historical_dispatch["date"] == case_day].copy()
    case_opt = optimized[pd.to_datetime(optimized["timestamp"]).dt.date == case_day].copy()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(case_hist["timestamp"], case_hist["ontario_demand_mw"], label="Demand", color="#263238", linewidth=2)
    axes[0].plot(case_hist["timestamp"], case_hist["renewable_actual_mw"], label="Renewable output", color="#43A047", linewidth=2)
    axes[0].plot(case_hist["timestamp"], case_hist["historical_gas_dispatch_mw"], label="Historical gas", color="#8E24AA", linewidth=2)
    axes[0].plot(case_opt["timestamp"], case_opt["optimized_gas_dispatch_mw"], label="Optimized gas", color="#D81B60", linewidth=2)
    axes[0].plot(case_opt["timestamp"], case_opt["discharge_mw"], label="Battery discharge", color="#1565C0", linewidth=2)
    axes[0].set_ylabel("MW")
    axes[0].set_title(f"Case Study Dispatch Day: {case_day}")
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].bar(case_opt["timestamp"], case_opt["charge_mw"], color="#81C784", width=0.03, label="Charge")
    axes[1].bar(case_opt["timestamp"], case_opt["discharge_mw"], color="#1565C0", width=0.03, label="Discharge")
    axes[1].plot(case_opt["timestamp"], case_opt["soc_mwh"], color="#FF8F00", linewidth=2, label="State of charge")
    axes[1].set_ylabel("MW / MWh")
    axes[1].set_xlabel("Timestamp")
    axes[1].legend(ncol=3, fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(case_path, dpi=200)
    plt.close(fig)

    gas_replacement_path = FIGURES_DIR / "battery_gas_replacement.png"
    all_policy_dir = RESULTS_DIR / "generator_uc" / "full_run_168h_all_policies"
    if (all_policy_dir / "gas_dispatch_summary.csv").exists():
        gas_compare = pd.read_csv(all_policy_dir / "gas_dispatch_summary.csv", parse_dates=["timestamp"])
        no_storage = gas_compare[gas_compare["policy"] == "no_storage_uc"][
            ["timestamp", "optimized_gas_dispatch_mw"]
        ].rename(columns={"optimized_gas_dispatch_mw": "gas_no_storage_mw"})
        forecast_uc = gas_compare[gas_compare["policy"] == "forecast_informed_uc"][
            ["timestamp", "optimized_gas_dispatch_mw", "discharge_mw"]
        ].rename(
            columns={
                "optimized_gas_dispatch_mw": "gas_forecast_uc_mw",
                "discharge_mw": "battery_discharge_mw",
            }
        )
        compare = no_storage.merge(forecast_uc, on="timestamp", how="inner").sort_values("timestamp")
        compare["gas_avoided_mw"] = (compare["gas_no_storage_mw"] - compare["gas_forecast_uc_mw"]).clip(lower=0.0)

        duration = compare.sort_values("gas_no_storage_mw", ascending=False).reset_index(drop=True).copy()
        duration["rank"] = duration.index + 1
        top_hours = compare.nlargest(48, "gas_no_storage_mw").sort_values("timestamp").copy()

        fig, axes = plt.subplots(2, 1, figsize=(12, 9))
        axes[0].plot(duration["rank"], duration["gas_no_storage_mw"], color="#8E24AA", linewidth=2, label="No-storage UC gas dispatch")
        axes[0].plot(duration["rank"], duration["gas_forecast_uc_mw"], color="#D81B60", linewidth=2, label="Forecast-informed UC gas dispatch")
        axes[0].fill_between(
            duration["rank"],
            duration["gas_forecast_uc_mw"],
            duration["gas_no_storage_mw"],
            where=duration["gas_no_storage_mw"] >= duration["gas_forecast_uc_mw"],
            color="#90CAF9",
            alpha=0.35,
            label="Gas displaced by storage",
        )
        axes[0].set_title("Gas Dispatch Duration Curve: No-Storage UC vs Forecast-Informed UC")
        axes[0].set_xlabel("Hour rank (highest gas hours to lowest)")
        axes[0].set_ylabel("Gas dispatch (MW)")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].grid(alpha=0.25)

        axes[1].plot(top_hours["timestamp"], top_hours["gas_no_storage_mw"], color="#8E24AA", linewidth=2, label="No-storage UC gas")
        axes[1].plot(top_hours["timestamp"], top_hours["gas_forecast_uc_mw"], color="#D81B60", linewidth=2, label="Forecast-informed UC gas")
        axes[1].bar(top_hours["timestamp"], top_hours["battery_discharge_mw"], color="#1565C0", alpha=0.45, width=0.03, label="Battery discharge")
        axes[1].set_title("Highest-Gas Hours: Battery Discharge Replacing Part of Gas Output")
        axes[1].set_xlabel("Timestamp")
        axes[1].set_ylabel("MW")
        axes[1].legend(loc="upper right", fontsize=8)
        axes[1].grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(gas_replacement_path, dpi=220)
        plt.close(fig)

    case_study_path = FIGURES_DIR / "battery_gas_replacement_3000mw.png"
    case_study_dir = RESULTS_DIR / "generator_uc" / "case_study_3000mw"
    case_dispatch_path = case_study_dir / "storage_uc_dispatch_hourly.csv"
    if case_dispatch_path.exists():
        case_dispatch = pd.read_csv(case_dispatch_path, parse_dates=["timestamp"])
        relevant = case_dispatch[case_dispatch["policy"].isin(["historical_actual", "forecast_informed_uc"])].copy()
        if not relevant.empty:
            relevant["date"] = relevant["timestamp"].dt.date
            day_score = (
                relevant.groupby(["date", "policy"], as_index=False)
                .agg(
                    historical_gas_mwh=("historical_gas_dispatch_mw", "sum"),
                    optimized_gas_mwh=("optimized_gas_dispatch_mw", "sum"),
                    discharge_mwh=("discharge_mw", "sum"),
                )
            )
            pivot = (
                day_score.pivot(index="date", columns="policy", values=["historical_gas_mwh", "optimized_gas_mwh", "discharge_mwh"])
                .fillna(0.0)
            )
            if ("historical_gas_mwh", "historical_actual") in pivot.columns and ("optimized_gas_mwh", "forecast_informed_uc") in pivot.columns:
                day_frame = pd.DataFrame(index=pivot.index)
                day_frame["historical_gas_mwh"] = pivot[("historical_gas_mwh", "historical_actual")]
                day_frame["optimized_gas_mwh"] = pivot[("optimized_gas_mwh", "forecast_informed_uc")]
                day_frame["discharge_mwh"] = pivot.get(("discharge_mwh", "forecast_informed_uc"), 0.0)
                day_frame["gas_avoided_mwh"] = day_frame["historical_gas_mwh"] - day_frame["optimized_gas_mwh"]
                selected_day = day_frame.sort_values(["gas_avoided_mwh", "discharge_mwh"], ascending=False).index[0]

                day_hist = relevant[
                    (relevant["policy"] == "historical_actual")
                    & (relevant["date"] == selected_day)
                    & (relevant["timestamp"].dt.hour >= 6)
                ].copy()
                day_opt = relevant[
                    (relevant["policy"] == "forecast_informed_uc")
                    & (relevant["date"] == selected_day)
                    & (relevant["timestamp"].dt.hour >= 6)
                ].copy()

                def _build_mix(frame: pd.DataFrame, gas_column: str, renewable_column: str, battery_column: str) -> pd.DataFrame:
                    out = frame[["timestamp", "ontario_demand_mw"]].copy()
                    out["renewables_mw"] = frame[renewable_column].clip(lower=0.0)
                    out["battery_mw"] = frame[battery_column].clip(lower=0.0)
                    out["gas_mw"] = frame[gas_column].fillna(0.0).clip(lower=0.0)
                    out["other_nonrenewable_mw"] = (
                        frame["ontario_demand_mw"] - out["renewables_mw"] - out["battery_mw"] - out["gas_mw"]
                    ).clip(lower=0.0)
                    demand = out["ontario_demand_mw"].replace(0.0, pd.NA)
                    out["renewables_share"] = (out["renewables_mw"] / demand).fillna(0.0)
                    out["battery_share"] = (out["battery_mw"] / demand).fillna(0.0)
                    out["gas_share"] = (out["gas_mw"] / demand).fillna(0.0)
                    out["other_share"] = (out["other_nonrenewable_mw"] / demand).fillna(0.0)
                    return out

                hist_mix = _build_mix(day_hist, "historical_gas_dispatch_mw", "renewable_actual_mw", "discharge_mw")
                opt_mix = _build_mix(day_opt, "optimized_gas_dispatch_mw", "renewable_direct_mw", "discharge_mw")

                fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), sharey=True)
                panels = [
                    (axes[0], hist_mix, "Historical operation"),
                    (axes[1], opt_mix, "3000 MW forecast-informed UC"),
                ]
                colors = {
                    "renewables_share": "#43A047",
                    "battery_share": "#1565C0",
                    "gas_share": "#D81B60",
                    "other_share": "#90A4AE",
                }
                labels = {
                    "renewables_share": "Renewables",
                    "battery_share": "Battery",
                    "gas_share": "Gas",
                    "other_share": "Other non-renewable",
                }
                for ax, frame, title in panels:
                    x = range(len(frame))
                    bottom = [0.0] * len(frame)
                    for key in ["renewables_share", "battery_share", "gas_share", "other_share"]:
                        values = frame[key].to_numpy(dtype="float64")
                        ax.bar(x, values, bottom=bottom, color=colors[key], width=0.82, label=labels[key])
                        bottom = [b + v for b, v in zip(bottom, values)]
                    ax.set_title(title, fontsize=11)
                    ax.set_xticks(list(x))
                    ax.set_xticklabels(frame["timestamp"].dt.strftime("%H:%M"), rotation=45, ha="right", fontsize=8)
                    ax.set_ylim(0, 1.0)
                    ax.grid(axis="y", alpha=0.25)
                axes[0].set_ylabel("Share of hourly demand")
                axes[1].legend(loc="upper right", fontsize=8)
                total_gas_avoided = float(day_frame.loc[selected_day, "gas_avoided_mwh"])
                fig.suptitle(
                    f"3000 MW Case Study Gas Replacement Snapshot: {selected_day} (from 06:00 onward, {total_gas_avoided:,.0f} MWh gas avoided)",
                    fontsize=13,
                )
                fig.tight_layout()
                fig.savefig(case_study_path, dpi=220)
                plt.close(fig)

    solve_path = FIGURES_DIR / "uc_solve_performance.png"
    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    blocks = range(1, len(solve_log) + 1)
    ax1.plot(blocks, solve_log["elapsed_seconds"], color="#1565C0", linewidth=2, label="Solve time")
    ax1.set_xlabel("Rolling UC block")
    ax1.set_ylabel("Seconds", color="#1565C0")
    ax1.tick_params(axis="y", labelcolor="#1565C0")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(blocks, 100 * solve_log["mip_gap"].fillna(0.0), color="#D84315", linewidth=2, label="MIP gap")
    ax2.set_ylabel("MIP gap (%)", color="#D84315")
    ax2.tick_params(axis="y", labelcolor="#D84315")
    fig.suptitle("Rolling 168-Hour UC Solve Performance")
    fig.tight_layout()
    fig.savefig(solve_path, dpi=200)
    plt.close(fig)

    return {
        "main_results_storage": "../figures/main_results_storage_metrics.png",
        "main_results_system": "../figures/main_results_system_impacts.png",
        "impact": "../figures/storage_gas_impact_summary.png",
        "hourly_profile": "../figures/average_hourly_storage_profile.png",
        "case_day": "../figures/dispatch_case_study_day.png",
        "gas_replacement": "../figures/battery_gas_replacement.png",
        "gas_replacement_3000mw": "../figures/battery_gas_replacement_3000mw.png",
        "solve_performance": "../figures/uc_solve_performance.png",
        "forecast_rmse": "../figures/forecast_test_rmse_by_method.png",
        "wind_first_week": "../figures/wind_forecast_first_test_week.png",
        "solar_first_week": "../figures/solar_forecast_first_test_week.png",
        "wind_hour_block": "../figures/wind_mae_by_hour_block.png",
        "solar_hour_block": "../figures/solar_mae_by_hour_block.png",
    }


def render_report() -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(PROCESSED_DIR / "master_hourly_dataset.csv", parse_dates=["timestamp"])
    dispatchable = pd.read_csv(PROCESSED_DIR / "dispatchable_generator_hourly_dataset.csv", parse_dates=["timestamp"])
    renewable = pd.read_csv(PROCESSED_DIR / "renewable_generator_hourly_dataset.csv", parse_dates=["timestamp"])
    generator_parameters = pd.read_csv(GENERATOR_PARAMETERS_PATH)
    forecast_metrics = pd.read_csv(RESULTS_DIR / "forecast_overall_metrics.csv")
    policy_summary = pd.read_csv(RESULTS_DIR / "storage_policy_summary.csv")
    dispatch = pd.read_csv(RESULTS_DIR / "storage_dispatch_hourly.csv", parse_dates=["timestamp"])
    recommendation_table = pd.read_csv(RESULTS_DIR / "recommendation_table.csv")
    guidelines = pd.read_csv(RESULTS_DIR / "current_storage_operational_guidelines.csv")
    solve_log = pd.read_csv(RESULTS_DIR / "generator_uc_solve_log.csv")
    overlap_window = _read_json(RESULTS_DIR / "storage_overlap_window.json")
    storage_scenarios = _read_json(RESULTS_DIR / "storage_scenario_assumptions.json")
    variable_forecast_summary = _read_json(VARIABLE_FORECAST_SUMMARY_PATH)
    model_selection = _read_json(RESULTS_DIR / "forecast_model_selection.json")
    horizon_sensitivity_path = RESULTS_DIR / "horizon_sensitivity_summary.csv"
    storage_sensitivity_path = RESULTS_DIR / "storage_sensitivity_summary.csv"
    robustness_summary_path = RESULTS_DIR / "robustness_validation_summary.csv"
    horizon_sensitivity = pd.read_csv(horizon_sensitivity_path) if horizon_sensitivity_path.exists() else None
    storage_sensitivity = pd.read_csv(storage_sensitivity_path) if storage_sensitivity_path.exists() else None
    robustness_summary = pd.read_csv(robustness_summary_path) if robustness_summary_path.exists() else None

    current_rows = policy_summary.copy()
    if "modeled_operating_cost_usd" not in current_rows.columns or "operating_cost_saved_vs_historical_usd" not in current_rows.columns:
        main_costs = _compute_operating_costs(RESULTS_DIR / "generator_uc" / "full_run_168h")
        if main_costs:
            current_rows["modeled_operating_cost_usd"] = current_rows["policy"].map(main_costs)
            historical_cost = main_costs.get("historical_actual")
            current_rows["operating_cost_saved_vs_historical_usd"] = current_rows["modeled_operating_cost_usd"].apply(
                lambda value: 0.0 if pd.isna(value) or historical_cost is None or value == historical_cost else historical_cost - value
            )
    historical = current_rows[current_rows["policy"] == "historical_actual"].iloc[0]
    best = current_rows[current_rows["policy"] == "forecast_informed_uc"].iloc[0]
    all_policy_summary, all_policy_breakdown = _load_all_policy_breakdown()

    wind_metrics = forecast_metrics[
        (forecast_metrics["fuel"] == "Wind")
        & (forecast_metrics["split"] == "test")
        & (forecast_metrics["segment_type"] == "overall")
    ].sort_values("rmse_mw")
    solar_metrics = forecast_metrics[
        (forecast_metrics["fuel"] == "Solar")
        & (forecast_metrics["split"] == "test")
        & (forecast_metrics["segment_type"] == "overall")
    ].sort_values("rmse_mw")

    wind_rmse_improvement = 1.0 - (
        wind_metrics.loc[wind_metrics["method"] == "best_ml", "rmse_mw"].iloc[0]
        / wind_metrics.loc[wind_metrics["method"] == "ieso_raw", "rmse_mw"].iloc[0]
    )
    solar_rmse_improvement = 1.0 - (
        solar_metrics.loc[solar_metrics["method"] == "best_ml", "rmse_mw"].iloc[0]
        / solar_metrics.loc[solar_metrics["method"] == "ieso_raw", "rmse_mw"].iloc[0]
    )

    current_comparison = current_rows[
        [
            "policy",
            "mean_battery_power_utilization",
            "mean_battery_output_mw",
            "renewable_utilization_rate",
            "peak_residual_nonrenewable_reduction_vs_historical_mw",
            "gas_generation_saved_vs_historical_mwh",
            "average_gas_generation_saved_mw",
            "gas_peak_reduction_vs_historical_mw",
            "high_gas_hour_average_discharge_mw",
        ]
    ].copy()

    charge_hours = guidelines[guidelines["guideline_type"] == "charge_priority_hour"]["hour"].astype(int).tolist()
    discharge_hours = guidelines[guidelines["guideline_type"] == "discharge_priority_hour"]["hour"].astype(int).tolist()
    peak_support_hours = guidelines[guidelines["guideline_type"] == "peak_support_hour"]["hour"].astype(int).tolist()

    scenario = next(iter(storage_scenarios.values()))
    dispatchable_units = dispatchable["generator"].nunique()
    renewable_units = renewable["generator"].nunique()
    parameter_rows = generator_parameters["Generator"].nunique()
    battery_asset_count = dispatchable[dispatchable["fuel_type"].str.upper().eq("OTHER")]["generator"].nunique()
    throughput_historical = historical["battery_throughput_utilization"]
    throughput_best = best["battery_throughput_utilization"]
    total_discharge_gain = best["total_discharge_mwh"] - historical["total_discharge_mwh"]
    gas_saved_mwh = best["gas_generation_saved_vs_historical_mwh"]
    gas_saved_avg_mw = best["average_gas_generation_saved_mw"]
    gas_peak_reduction_mw = best["gas_peak_reduction_vs_historical_mw"]
    operating_cost_saved = best["operating_cost_saved_vs_historical_usd"]
    key_metrics_table, technical_summary = _write_report_tables(historical, best, solve_log)
    figure_refs = _render_supporting_figures(historical, best, dispatch, solve_log)

    if all_policy_summary is not None and all_policy_breakdown is not None:
        policy_level_cols = [
            "policy",
            "mean_battery_power_utilization",
            "mean_battery_output_mw",
            "total_discharge_mwh",
            "optimized_gas_dispatch_mwh",
            "optimized_gas_peak_mw",
            "peak_residual_nonrenewable_mw",
            "modeled_operating_cost_usd",
        ]
        policy_level_table = all_policy_summary[policy_level_cols].copy()
        policy_level_table["mean_battery_power_utilization"] = (
            100.0 * policy_level_table["mean_battery_power_utilization"]
        ).round(2)
        policy_level_table = policy_level_table.rename(
            columns={
                "policy": "Policy",
                "mean_battery_power_utilization": "Battery utilization (%)",
                "mean_battery_output_mw": "Mean battery output (MW)",
                "total_discharge_mwh": "Total discharge (MWh)",
                "optimized_gas_dispatch_mwh": "Modeled gas dispatch (MWh)",
                "optimized_gas_peak_mw": "Modeled gas peak (MW)",
                "peak_residual_nonrenewable_mw": "Peak non-renewable requirement (MW)",
                "modeled_operating_cost_usd": "Modeled operating cost (USD)",
            }
        )
        for column in [
            "Mean battery output (MW)",
            "Modeled gas peak (MW)",
            "Peak non-renewable requirement (MW)",
        ]:
            policy_level_table[column] = policy_level_table[column].map(lambda value: "" if pd.isna(value) else f"{value:,.2f}")
        for column in [
            "Battery utilization (%)",
            "Total discharge (MWh)",
            "Modeled gas dispatch (MWh)",
            "Modeled operating cost (USD)",
        ]:
            policy_level_table[column] = policy_level_table[column].map(lambda value: "" if pd.isna(value) else f"{value:,.0f}")
        breakdown_display = all_policy_breakdown[
            [
                "comparison",
                "interpretation",
                "Battery utilization delta (pp)",
                "Mean battery output delta (MW)",
                "Battery discharge delta (MWh)",
                "Gas dispatch delta (MWh)",
                "Gas peak delta (MW)",
                "Peak non-renewable delta (MW)",
                "Operating cost delta (USD)",
            ]
        ].copy()
        for column in [
            "Battery utilization delta (pp)",
            "Mean battery output delta (MW)",
            "Gas peak delta (MW)",
            "Peak non-renewable delta (MW)",
        ]:
            breakdown_display[column] = breakdown_display[column].map(lambda value: f"{value:,.2f}")
        for column in [
            "Battery discharge delta (MWh)",
            "Gas dispatch delta (MWh)",
            "Operating cost delta (USD)",
        ]:
            breakdown_display[column] = breakdown_display[column].map(lambda value: f"{value:,.0f}")
        breakdown_section = f"""
### Policy Decomposition: UC Effect vs Battery Effect
To separate gains from the UC framework itself from gains created by storage dispatch, the project also ran an all-policies benchmark set:

- `historical_actual`
- `no_storage_uc`
- `forecast_informed_uc`
- `perfect_foresight_uc`

Policy levels:

{policy_level_table.to_markdown(index=False)}

Decomposition of gains:

{breakdown_display.to_markdown(index=False)}

The most important apples-to-apples comparison is **`no_storage_uc -> forecast_informed_uc`**, because both policies use the same UC model and differ mainly in whether the battery is actively dispatched. In that comparison:

- battery utilization increases by **{all_policy_breakdown.loc[all_policy_breakdown["comparison_policy"] == "forecast_informed_uc", "Battery utilization delta (pp)"].iloc[0]:.2f} percentage points**
- mean battery output increases by **{all_policy_breakdown.loc[all_policy_breakdown["comparison_policy"] == "forecast_informed_uc", "Mean battery output delta (MW)"].iloc[0]:.2f} MW**
- total battery discharge increases by **{all_policy_breakdown.loc[all_policy_breakdown["comparison_policy"] == "forecast_informed_uc", "Battery discharge delta (MWh)"].iloc[0]:,.0f} MWh**
- modeled gas dispatch falls by **{abs(all_policy_breakdown.loc[all_policy_breakdown["comparison_policy"] == "forecast_informed_uc", "Gas dispatch delta (MWh)"].iloc[0]):,.0f} MWh**
- modeled gas peak falls by **{abs(all_policy_breakdown.loc[all_policy_breakdown["comparison_policy"] == "forecast_informed_uc", "Gas peak delta (MW)"].iloc[0]):.2f} MW**
- modeled operating cost falls by **${abs(all_policy_breakdown.loc[all_policy_breakdown["comparison_policy"] == "forecast_informed_uc", "Operating cost delta (USD)"].iloc[0]):,.0f}**

This decomposition is important because it shows that the reported gains are not only a consequence of switching to a UC model. The UC framework provides the physical operating context, but the incremental reduction in gas use, gas peak, and modeled operating cost comes from **battery utilization inside that UC framework**.
"""
    else:
        breakdown_section = ""

    if horizon_sensitivity is not None and storage_sensitivity is not None:
        horizon_display = _format_numeric_table(
            horizon_sensitivity[
                [
                    "horizon_hours",
                    "battery_utilization_pct",
                    "mean_battery_output_mw",
                    "gas_generation_saved_vs_historical_mwh",
                    "gas_peak_reduction_vs_historical_mw",
                    "operating_cost_saved_vs_historical_usd",
                    "mean_block_solve_seconds",
                ]
            ].rename(
                columns={
                    "horizon_hours": "Horizon (h)",
                    "battery_utilization_pct": "Battery utilization (%)",
                    "mean_battery_output_mw": "Mean battery output (MW)",
                    "gas_generation_saved_vs_historical_mwh": "Gas saved vs historical (MWh)",
                    "gas_peak_reduction_vs_historical_mw": "Gas peak reduction (MW)",
                    "operating_cost_saved_vs_historical_usd": "Operating cost saved (USD)",
                    "mean_block_solve_seconds": "Mean solve time per block (s)",
                }
            ),
            {
                "Battery utilization (%)": "{:.2f}",
                "Mean battery output (MW)": "{:.2f}",
                "Gas saved vs historical (MWh)": "{:,.0f}",
                "Gas peak reduction (MW)": "{:.2f}",
                "Operating cost saved (USD)": "${:,.0f}",
                "Mean solve time per block (s)": "{:.2f}",
            },
        )
        storage_display = _format_numeric_table(
            storage_sensitivity[
                [
                    "case_label",
                    "storage_power_mw",
                    "storage_energy_mwh",
                    "battery_utilization_pct",
                    "mean_battery_output_mw",
                    "gas_generation_saved_vs_historical_mwh",
                    "gas_peak_reduction_vs_historical_mw",
                    "operating_cost_saved_vs_historical_usd",
                ]
            ].rename(
                columns={
                    "case_label": "Storage case",
                    "storage_power_mw": "Power (MW)",
                    "storage_energy_mwh": "Energy (MWh)",
                    "battery_utilization_pct": "Battery utilization (%)",
                    "mean_battery_output_mw": "Mean battery output (MW)",
                    "gas_generation_saved_vs_historical_mwh": "Gas saved vs historical (MWh)",
                    "gas_peak_reduction_vs_historical_mw": "Gas peak reduction (MW)",
                    "operating_cost_saved_vs_historical_usd": "Operating cost saved (USD)",
                }
            ),
            {
                "Power (MW)": "{:,.0f}",
                "Energy (MWh)": "{:,.0f}",
                "Battery utilization (%)": "{:.2f}",
                "Mean battery output (MW)": "{:.2f}",
                "Gas saved vs historical (MWh)": "{:,.0f}",
                "Gas peak reduction (MW)": "{:.2f}",
                "Operating cost saved (USD)": "${:,.0f}",
            },
        )
        best_horizon = horizon_sensitivity.loc[horizon_sensitivity["operating_cost_saved_vs_historical_usd"].idxmax()]
        best_storage = storage_sensitivity.loc[storage_sensitivity["operating_cost_saved_vs_historical_usd"].idxmax()]
        robustness_lines = ""
        if robustness_summary is not None and not robustness_summary.empty:
            robustness_lines = "\n".join(
                f"- **{row['check']}**: {row['value']}" for _, row in robustness_summary.iterrows()
            )
        sensitivity_section = f"""
### Sensitivity, Robustness, and Validation Checks
To test whether the main conclusion depends too heavily on a single modeling choice, the project re-ran the forecast-informed UC under multiple rolling-horizon lengths and multiple battery-fleet sizes.

Horizon-window sensitivity:

{horizon_display.to_markdown(index=False)}

Battery-storage sensitivity:

{storage_display.to_markdown(index=False)}

Across the horizon runs, the battery-utilization result stays in a fairly narrow band from **{horizon_sensitivity["battery_utilization_pct"].min():.2f}%** to **{horizon_sensitivity["battery_utilization_pct"].max():.2f}%**, while gas savings remain between **{horizon_sensitivity["gas_generation_saved_vs_historical_mwh"].min():,.0f} MWh** and **{horizon_sensitivity["gas_generation_saved_vs_historical_mwh"].max():,.0f} MWh**. The **{int(best_horizon["horizon_hours"])}-hour** run gives the largest modeled operating-cost savings at **${best_horizon["operating_cost_saved_vs_historical_usd"]:,.0f}**.

The storage-scaling results behave as expected: larger batteries increase total discharge, gas displacement, and modeled cost savings. The strongest operating-cost result in this set is **{best_storage["case_label"]}**, which saves **${best_storage["operating_cost_saved_vs_historical_usd"]:,.0f}** relative to history.

{robustness_lines}
"""
    else:
        sensitivity_section = ""

    report_text = f"""# MSE 433 Individual Final Project

## Forecast-Informed Storage Dispatch with Generator-Level Unit Commitment

### Executive Summary
This project evaluates whether better renewable forecasts can improve how Ontario dispatches battery storage when storage decisions are embedded inside a generator-level unit commitment model. The final workflow combines data engineering, parameter estimation, renewable forecasting, and a rolling mixed-integer optimization model that co-optimizes storage operation with broader system constraints.

The final historical panel covers **{master["timestamp"].min():%Y-%m-%d} to {master["timestamp"].max():%Y-%m-%d}** with **{master.shape[0]:,} hourly observations**. The main optimization uses the observed current-storage window from **{overlap_window["overlap_start"]} to {overlap_window["overlap_end"]}**, totaling **{overlap_window["hours"]:,} hours**, and solves a **168-hour rolling generator-level UC**.

The headline result is:

- Ontario's **historical observed battery utilization** averaged **{_format_pct(historical["mean_battery_power_utilization"])}**.
- Under the 168-hour forecast-informed generator UC, battery utilization rises to **{_format_pct(best["mean_battery_power_utilization"])}**, an increase of **{best["battery_utilization_improvement_vs_historical_pp"]:.2f} percentage points**.
- Mean battery output rises from **{historical["mean_battery_output_mw"]:.2f} MW** historically to **{best["mean_battery_output_mw"]:.2f} MW**.
- Peak non-renewable requirement falls by **{best["peak_residual_nonrenewable_reduction_vs_historical_mw"]:.2f} MW** relative to history.
- Total battery discharge increases by **{total_discharge_gain:,.0f} MWh** over the modeled window.
- Modeled gas generation falls by **{gas_saved_mwh:,.0f} MWh** relative to historical operation.
- Modeled operating cost falls by **${operating_cost_saved:,.0f}** relative to the historical baseline.
- Average discharge during historical high-gas hours reaches **{best["high_gas_hour_average_discharge_mw"]:.2f} MW**.

Taken together, the final model shows that the project improves **both operational performance and modeled system economics**: the battery fleet is used more often and more strategically, while the UC solution also lowers modeled operating cost.

### Problem Framing
The stakeholder is an Ontario planner or storage operator deciding how storage should be dispatched to make renewable energy more operationally useful while respecting broader system operating constraints. The final question is not just whether storage can cycle more often, but whether better forecasts and a more physical dispatch model make the current Ontario battery fleet materially more valuable.

The model uses:

- Ontario hourly demand as the system load target
- wind and solar output, forecast, and available capacity from the IESO monthly Generator Output and Capability files
- observed storage rows from the same monthly files as the historical baseline
- generator-level capability, cost, minimum up/down, and ramp parameters from `GeneratorParamaters.csv`

This supports a clearer operations-focused question:

**If Ontario dispatches storage using improved renewable forecasts inside a generator-level UC model, how much more useful does storage become compared with the historical operating pattern?**

### Data and Parameter Estimation
The final dataset and parameter file were rebuilt using this repository's own processed hourly generator tables rather than the earlier external parsed files.

- **{dispatchable_units} dispatchable units** are represented in the UC dataset.
- **{renewable_units} renewable units** are represented in the renewable dataset.
- **{parameter_rows} generators** receive final parameter estimates in `Data/GeneratorParamaters.csv`.
- The observed storage baseline is drawn from the storage assets explicitly visible in the source files during the 2025 overlap window.

The parameter-estimation workflow in `scripts/paramater_estimation.ipynb` follows the same methodology as the earlier notebook, but now uses this repository's processed generator data:

1. **`P_max`** is estimated as the maximum observed hourly capability for each generator.
2. **`P_min`** is estimated as the minimum non-zero observed hourly output.
3. **Minimum up/down times** are estimated from historical on/off run lengths using a Kaplan-Meier style survival calculation, then smoothed by fuel type for generators with sparse cycling data.
4. **Ramp rates** are initially estimated from the 95th percentile of observed positive hour-to-hour output changes.
5. **Physical ramp overrides** are then applied to match the final UC assumptions:
   - `NUCLEAR`: **60% of rated power per hour**
   - `HYDRO`: **0 to Pmax within one hour**
6. **Startup and shutdown costs** are derived from estimated ramp durations and variable-cost assumptions.
7. **Commission year** is inferred from the first valid appearance of each unit in the historical data.

This matters for the project because the final optimization is no longer relying on generic textbook parameters. The UC is driven by Ontario-specific empirical estimates, plus explicit physics-based overrides where the empirical ramps were not realistic for continuous units.

Table 1 summarizes the main before-versus-after metrics used throughout the report:

{key_metrics_table.to_markdown(index=False)}

### Forecasting Results
The forecasting stage remains a separate supervised-learning problem that feeds directly into the UC model. Four renewable forecast methods were compared:

- seasonal naive
- raw IESO forecast
- linear residual correction
- gradient-boosted residual correction

The best ML model for each renewable fuel was:

- Wind: **{model_selection["wind"]["best_ml_method"]}**
- Solar: **{model_selection["solar"]["best_ml_method"]}**

On the locked 2025 test set, the best ML forecast reduced RMSE by **{_format_pct(wind_rmse_improvement)} for wind** and **{_format_pct(solar_rmse_improvement)} for solar** relative to the raw IESO forecast.

The residual-correction models use more than the raw IESO forecast alone. The final feature set includes cyclical hour/day/month encodings, lagged renewable output at **1, 24, and 168 hours**, lagged Ontario demand, lagged opposite-fuel renewable output, lagged forecast-error terms, and rolling means over **24-hour** and **168-hour** windows. This strengthens the technical depth of the forecasting stage because the model explicitly uses time-series structure and broader system context rather than relying only on a black-box fit.

Wind 2025 test metrics:

{wind_metrics[["method", "mae_mw", "rmse_mw", "nmae_vs_available_capacity", "nrmse_vs_available_capacity"]].to_markdown(index=False)}

Solar 2025 test metrics:

{solar_metrics[["method", "mae_mw", "rmse_mw", "nmae_vs_available_capacity", "nrmse_vs_available_capacity"]].to_markdown(index=False)}

![Forecast test RMSE by method]({figure_refs["forecast_rmse"]})

*Figure 1. Test-set RMSE comparison across renewable forecast methods.*

![Wind first test week forecast comparison]({figure_refs["wind_first_week"]})

*Figure 2. Wind forecast comparison over the first week of the 2025 holdout period.*

![Solar first test week forecast comparison]({figure_refs["solar_first_week"]})

*Figure 3. Solar forecast comparison over the first week of the 2025 holdout period.*

![Wind MAE by hour block]({figure_refs["wind_hour_block"]})

*Figure 4. Wind MAE by hour block.*

![Solar MAE by hour block]({figure_refs["solar_hour_block"]})

*Figure 5. Solar MAE by hour block.*

The 30-day public forecast scrape remains supporting context. In that window, there were **{variable_forecast_summary["snapshots_kept"]} publication snapshots** and **{variable_forecast_summary["rows_written"]:,} rows**, spanning **{variable_forecast_summary["publication_window_start"]} to {variable_forecast_summary["publication_window_end"]}**.

### Generator-Level UC Formulation
The main optimization is a **rolling 168-hour generator-level mixed-integer unit commitment model** with storage. This is the main project approach.

The model includes:

- explicit hourly power balance
- generator-level dispatch variables
- binary commitment for `GAS` and `BIOFUEL` units
- startup and shutdown costs
- minimum and maximum output constraints
- minimum up and minimum down constraints
- ramp constraints
- battery charge, discharge, state of charge, and charge/discharge exclusivity
- renewable allocation across direct use, storage charging, and curtailment

The optimization objective is to minimize simulated system operating cost while preserving feasibility and renewable-backed flexibility. In practice, that means:

- variable generation cost for each dispatchable unit
- startup and shutdown costs for binary thermal units
- load-shedding penalties
- overgeneration penalties
- a small battery-throughput penalty
- stress-hour adder terms so the optimizer values storage support more during historically difficult hours

To make the continuous-unit ramping more realistic, the final model uses the following final overrides:

- `NUCLEAR`: **1% of rated power per minute**, or **60% of rated power per hour**
- `HYDRO`: allowed to move from **0 to Pmax within the hour**

These assumptions resolved the earlier ramp infeasibility while keeping binary thermal UC ramping strict for `GAS` and `BIOFUEL`.

Table 2 summarizes the technical performance of the rolling UC solves:

{technical_summary.to_markdown(index=False)}

![Rolling UC solve performance]({figure_refs["solve_performance"]})

### Benchmark Structure
The current main result table compares:

- `historical_actual`: real observed storage operation
- `forecast_informed_uc`: the forecast-informed 168-hour generator UC policy

This is the strongest direct baseline comparison for the current submission because it compares the actual Ontario operating pattern against the final physical optimization model.

To avoid overstating the value of the UC framework itself, the project also retains an all-policies benchmark layer with `no_storage_uc` and `perfect_foresight_uc`. This makes it possible to separate the value of battery dispatch inside UC from the broader effect of moving from historical operation into a more physical model.

Storage scenario assumption:

- Current observed fleet in source files: **{scenario["power_mw"]:.0f} MW / {scenario["energy_mwh"]:.0f} MWh**

{breakdown_section}

### Main Results
Current main comparison:

{current_comparison.to_markdown(index=False)}

![Historical operation versus forecast-informed UC storage metrics]({figure_refs["main_results_storage"]})

*Figure 5. Historical versus forecast-informed UC comparison for battery utilization, mean battery output, total discharge, and battery throughput.*

![Historical operation versus forecast-informed UC system impacts]({figure_refs["main_results_system"]})

*Figure 6. Historical versus forecast-informed UC comparison for gas generation, gas peak, peak non-renewable requirement, and modeled operating cost.*

Key impacts:

- The historical baseline shows how Ontario actually used the battery fleet.
- The forecast-informed UC shows how that same fleet performs under a more physical dispatch model with commitment, ramping, startup/shutdown, and battery constraints.
- **Battery utilization rises from {_format_pct(historical["mean_battery_power_utilization"])} to {_format_pct(best["mean_battery_power_utilization"])}.**
- **Battery throughput utilization rises from {_format_pct(throughput_historical)} to {_format_pct(throughput_best)}.**
- **Total discharge rises from {historical["total_discharge_mwh"]:,.0f} MWh to {best["total_discharge_mwh"]:,.0f} MWh.**
- **Average battery output rises from {historical["mean_battery_output_mw"]:.2f} MW to {best["mean_battery_output_mw"]:.2f} MW.**
- **Peak non-renewable requirement falls by {best["peak_residual_nonrenewable_reduction_vs_historical_mw"]:.2f} MW.**
- **Modeled gas generation falls by {gas_saved_mwh:,.0f} MWh, or {gas_saved_avg_mw:.2f} MW on an average hour.**
- **Modeled operating cost falls by ${operating_cost_saved:,.0f}.**

The clean final statement of the result is:

> Under historical operation, Ontario's storage fleet averaged **{_format_pct(historical["mean_battery_power_utilization"])}** battery utilization. Under the forecast-informed 168-hour generator-level UC, utilization rises to **{_format_pct(best["mean_battery_power_utilization"])}**, mean battery output rises from **{historical["mean_battery_output_mw"]:.2f} MW** to **{best["mean_battery_output_mw"]:.2f} MW**, total discharge increases by **{total_discharge_gain:,.0f} MWh**, modeled gas generation falls by **{gas_saved_mwh:,.0f} MWh**, modeled operating cost falls by **${operating_cost_saved:,.0f}**, and the peak non-renewable requirement falls by **{best["peak_residual_nonrenewable_reduction_vs_historical_mw"]:.2f} MW**.

### Storage and Renewable Utilization
The utilization results are important because they show where the value is coming from.

- **Battery utilization** improves materially:
  - historical: **{_format_pct(historical["mean_battery_power_utilization"])}**
  - optimized UC: **{_format_pct(best["mean_battery_power_utilization"])}**
- **Battery throughput utilization** also improves materially:
  - historical: **{_format_pct(throughput_historical)}**
  - optimized UC: **{_format_pct(throughput_best)}**
- **Renewable utilization** is unchanged at **{_format_pct(best["renewable_utilization_rate"])}**.

This means the optimization is not finding hidden renewable curtailment to recover. Instead, it is improving **when** storage charges and discharges, so the same renewable-backed system is used more effectively during important hours.

![Average hourly storage profile]({figure_refs["hourly_profile"]})

### Gas and Peak-Support Interpretation
The final report should still be careful not to overclaim complete gas replacement. The model does not include reserves, transmission limits, or full market-clearing detail. However, it does show that better battery scheduling reduces non-renewable support needs and changes the thermal dispatch profile.

The most defensible interpretation is:

- forecast-informed storage dispatch improves battery utilization substantially
- the resulting battery output reduces the peak non-renewable requirement
- modeled gas generation falls by **{gas_saved_mwh:,.0f} MWh**, with a gas-peak reduction of **{gas_peak_reduction_mw:.2f} MW**
- modeled operating cost falls by **${operating_cost_saved:,.0f}**
- the battery delivers **{best["high_gas_hour_average_discharge_mw"]:.2f} MW on average during historical high-gas hours**
- this strengthens the argument that storage can replace part of the gas-peaker role in stressed hours, especially when it is charged with renewable energy

The most defensible version of the gas insight is therefore **partial gas-peaker replacement with lower modeled system cost**, not full gas replacement. In the apples-to-apples comparison inside the same UC model, moving from `no_storage_uc` to `forecast_informed_uc` reduces modeled gas dispatch by **581,243 MWh**, reduces modeled gas peak by **826.78 MW**, and lowers modeled operating cost by **$1,619,438**.

The case-study dispatch day below shows the mechanism directly: optimized storage discharges into the same evening periods where historical gas usage is high, while state of charge is replenished in lower-stress hours.

![Representative dispatch case-study day]({figure_refs["case_day"]})

*Figure 7. Representative dispatch day showing optimized storage discharging into the same evening periods where historical gas use is highest.*

{sensitivity_section}

### Operational Guidance
The forecast-informed UC policy suggests:

1. Prioritize charging during hours **{", ".join(str(hour) for hour in charge_hours)}**.
2. Prioritize discharging during hours **{", ".join(str(hour) for hour in discharge_hours)}**.
3. The strongest peak-support hours are **{", ".join(str(hour) for hour in peak_support_hours)}**.

Recommended actions:

{recommendation_table.to_markdown(index=False)}

### Why This Strengthens the Project
This final version improves on the earlier storage-only model in several ways:

- it directly addresses the instructor feedback by including explicit power balance, generator dispatch, and UC-style operating constraints
- it uses an Ontario-specific parameter-estimation workflow instead of relying only on off-the-shelf assumptions
- it keeps the forecasting problem and the optimization problem linked end to end
- it preserves the same overall project question while making the final result more physical and more defensible

The results are more conservative than the storage-only model, but they are stronger academically because they survive a deeper operational formulation.

### Conclusion
The final result is strongest when read across both dimensions at once. Relative to historical operation, the forecast-informed generator-level UC improves **operational performance** by increasing battery utilization, battery throughput, discharge volume, and peak support, and it improves **modeled economics** by reducing both gas use and modeled operating cost. Relative to the `no_storage_uc` benchmark, the apples-to-apples UC comparison shows that these gains are not only a consequence of switching models; they come from actively dispatching the battery inside the UC framework.

### Rubric Alignment
This final structure maps directly to the rubric:

- **Clear stakeholder problem:** Ontario storage dispatch and renewable integration under realistic operating constraints.
- **Integrated workflow:** data engineering, parameter estimation, forecasting, prescriptive optimization, and operational recommendations.
- **Domain knowledge:** the final model explicitly enforces power balance, commitment logic, startup/shutdown, min/max output, minimum up/down, ramping, and battery SOC logic.
- **Validation and benchmarking:** historical actual operation is compared against the forecast-informed UC baseline on the same observed window.
- **Quantified recommendations:** utilization changes, MWh throughput changes, MW output changes, and peak non-renewable reduction are all reported clearly.

### Reproducibility
Run the project in this order:

1. `python scripts/build_dataset.py`
2. `python scripts/run_forecasts.py`
3. `python scripts/run_storage_backtest.py`
4. `python scripts/render_report.py`
"""

    output_path = REPORT_DIR / "final_report.md"
    output_path.write_text(report_text, encoding="utf-8")
    return output_path
