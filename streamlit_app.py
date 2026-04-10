from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
HOURLY_PATH = RESULTS_DIR / "generator_hourly_recommendations.csv"
SUMMARY_PATH = RESULTS_DIR / "generator_decision_support_summary.csv"
SUPPLY_PATH = RESULTS_DIR / "generator_supply_mix_hourly.csv"


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hourly = pd.read_csv(HOURLY_PATH, parse_dates=["timestamp"])
    summary = pd.read_csv(SUMMARY_PATH)
    supply = pd.read_csv(SUPPLY_PATH, parse_dates=["timestamp"])
    return hourly, summary, supply


def main() -> None:
    st.set_page_config(page_title="MSE433 Generator Decision Support", layout="wide")
    st.title("Generator Decision Support Dashboard")
    st.caption("This dashboard reads precomputed UC outputs only. It does not rerun the optimization.")

    if not (HOURLY_PATH.exists() and SUMMARY_PATH.exists() and SUPPLY_PATH.exists()):
        st.error("Decision-support outputs are missing. Run `python scripts/run_decision_support.py` first.")
        return

    hourly, summary, supply = load_data()

    timestamps = hourly["timestamp"].sort_values().drop_duplicates().tolist()
    default_start = timestamps[0]

    st.sidebar.header("Filters")
    start_ts = st.sidebar.selectbox(
        "Start hour",
        timestamps,
        index=0,
        format_func=lambda ts: pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M"),
    )
    horizon_hours = st.sidebar.slider("Hours to display", min_value=1, max_value=48, value=24)
    fuels = sorted(hourly["fuel_type"].dropna().unique().tolist())
    selected_fuels = st.sidebar.multiselect("Fuel types", fuels, default=fuels)
    show_only_on = st.sidebar.checkbox("Show only recommended-on units", value=True)

    start_ts = pd.Timestamp(start_ts)
    end_ts = start_ts + pd.Timedelta(hours=horizon_hours - 1)

    hourly_view = hourly[
        (hourly["timestamp"] >= start_ts)
        & (hourly["timestamp"] <= end_ts)
        & (hourly["fuel_type"].isin(selected_fuels))
    ].copy()
    if show_only_on:
        hourly_view = hourly_view[hourly_view["recommended_on"]]

    supply_view = supply[(supply["timestamp"] >= start_ts) & (supply["timestamp"] <= end_ts)].copy()

    st.subheader("Hourly Energy Mix")
    mix_columns = ["renewables_mw", "battery_mw", "nuclear_mw", "hydro_mw", "biofuel_mw", "gas_mw"]
    mix_display = (
        supply_view[["timestamp", *mix_columns]]
        .melt(id_vars="timestamp", var_name="source", value_name="mw")
        .rename(columns={"timestamp": "Hour", "mw": "MW"})
    )
    st.area_chart(
        mix_display,
        x="Hour",
        y="MW",
        color="source",
        stack=True,
        height=320,
    )

    st.subheader("Hour-by-Hour Generator Recommendations")
    st.dataframe(
        hourly_view[
            [
                "timestamp",
                "generator",
                "fuel_type",
                "recommended_status",
                "recommended_action",
                "dispatch_mw",
                "startup",
                "shutdown",
                "high_gas_hour",
            ]
        ].sort_values(["timestamp", "fuel_type", "dispatch_mw"], ascending=[True, True, False]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Generator Summary")
    st.dataframe(
        summary[
            [
                "generator",
                "fuel_type",
                "recommended_role",
                "on_hours",
                "on_rate",
                "avg_dispatch_when_on_mw",
                "max_dispatch_mw",
                "startup_count",
                "shutdown_count",
                "high_gas_hour_dispatch_mwh",
            ]
        ].sort_values(["fuel_type", "avg_dispatch_when_on_mw"], ascending=[True, False]),
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
