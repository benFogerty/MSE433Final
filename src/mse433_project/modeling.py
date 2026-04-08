from __future__ import annotations

import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from .config import FIGURES_DIR, RESULTS_DIR, TEST_END, TEST_START, TRAIN_END, VALID_END, VALID_START


FORECAST_METHODS = ["seasonal_naive", "ieso_raw", "linear_residual", "gradient_boosted", "best_ml"]
HOUR_BLOCKS = {
    "Overnight": (0, 5),
    "Morning": (6, 11),
    "Afternoon": (12, 17),
    "Evening": (18, 23),
}


@dataclass
class FuelForecastArtifacts:
    fuel: str
    predictions: pd.DataFrame
    overall_metrics: pd.DataFrame
    segment_metrics: pd.DataFrame
    model_selection: dict[str, object]


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    frame["dow_sin"] = np.sin(2 * np.pi * frame["day_of_week"] / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * frame["day_of_week"] / 7)
    frame["month_sin"] = np.sin(2 * np.pi * (frame["month"] - 1) / 12)
    frame["month_cos"] = np.cos(2 * np.pi * (frame["month"] - 1) / 12)
    return frame


def _hour_block(hour: int) -> str:
    for label, (start, end) in HOUR_BLOCKS.items():
        if start <= hour <= end:
            return label
    return "Unknown"


def _rmse(actual: pd.Series, predicted: pd.Series) -> float:
    return float((((actual - predicted) ** 2).mean()) ** 0.5)


def _metric_row(
    actual: pd.Series,
    predicted: pd.Series,
    available_capacity: pd.Series,
    fuel: str,
    split: str,
    method: str,
    segment_type: str,
    segment_value: str,
) -> dict[str, object]:
    mae = float((actual - predicted).abs().mean())
    rmse = _rmse(actual, predicted)
    denom = float(available_capacity.mean()) if float(available_capacity.mean()) > 0 else 1.0
    return {
        "fuel": fuel,
        "split": split,
        "method": method,
        "segment_type": segment_type,
        "segment_value": segment_value,
        "observations": int(actual.shape[0]),
        "mae_mw": mae,
        "rmse_mw": rmse,
        "nmae_vs_available_capacity": mae / denom,
        "nrmse_vs_available_capacity": rmse / denom,
    }


def _split_label(timestamp: pd.Timestamp) -> str:
    if timestamp <= pd.Timestamp(TRAIN_END):
        return "train"
    if pd.Timestamp(VALID_START) <= timestamp <= pd.Timestamp(VALID_END):
        return "validation"
    if pd.Timestamp(TEST_START) <= timestamp <= pd.Timestamp(TEST_END):
        return "test"
    return "out_of_scope"


def prepare_fuel_frame(master: pd.DataFrame, fuel: str) -> tuple[pd.DataFrame, list[str]]:
    fuel_key = fuel.lower()
    other_key = "solar" if fuel_key == "wind" else "wind"
    target_col = f"{fuel_key}_output"
    forecast_col = f"{fuel_key}_forecast"
    capacity_col = f"{fuel_key}_available_capacity"

    frame = master.copy()
    frame["target_actual"] = frame[target_col]
    frame["ieso_raw"] = frame[forecast_col]
    frame["available_capacity"] = frame[capacity_col]
    frame["other_output"] = frame[f"{other_key}_output"]
    frame["other_forecast"] = frame[f"{other_key}_forecast"]
    frame["other_available_capacity"] = frame[f"{other_key}_available_capacity"]

    frame = add_time_features(frame)

    for lag in [1, 24, 168]:
        frame[f"target_lag_{lag}"] = frame["target_actual"].shift(lag)
        frame[f"demand_lag_{lag}"] = frame["ontario_demand_mw"].shift(lag)
        frame[f"other_output_lag_{lag}"] = frame["other_output"].shift(lag)
        frame[f"forecast_error_lag_{lag}"] = (frame["target_actual"] - frame["ieso_raw"]).shift(lag)

    for window in [24, 168]:
        frame[f"target_roll_mean_{window}"] = frame["target_actual"].shift(1).rolling(window).mean()
        frame[f"demand_roll_mean_{window}"] = frame["ontario_demand_mw"].shift(1).rolling(window).mean()
        frame[f"other_output_roll_mean_{window}"] = frame["other_output"].shift(1).rolling(window).mean()

    frame["split"] = frame["timestamp"].map(_split_label)
    frame["hour_block"] = frame["hour"].map(_hour_block)
    frame["residual_target"] = frame["target_actual"] - frame["ieso_raw"]

    feature_columns = [
        "ieso_raw",
        "available_capacity",
        "other_forecast",
        "other_available_capacity",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
        "target_lag_1",
        "target_lag_24",
        "target_lag_168",
        "demand_lag_1",
        "demand_lag_24",
        "demand_lag_168",
        "other_output_lag_1",
        "other_output_lag_24",
        "other_output_lag_168",
        "forecast_error_lag_1",
        "forecast_error_lag_24",
        "forecast_error_lag_168",
        "target_roll_mean_24",
        "target_roll_mean_168",
        "demand_roll_mean_24",
        "demand_roll_mean_168",
        "other_output_roll_mean_24",
        "other_output_roll_mean_168",
    ]
    frame = frame.dropna(subset=feature_columns + ["target_actual", "available_capacity"]).reset_index(drop=True)
    return frame, feature_columns


def _fit_gradient_boosting(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    base_valid: pd.Series,
    available_capacity_valid: pd.Series,
    fuel: str,
) -> tuple[HistGradientBoostingRegressor, dict[str, object]]:
    candidates = [
        {"learning_rate": 0.05, "max_depth": 3, "max_iter": 250, "min_samples_leaf": 30},
        {"learning_rate": 0.05, "max_depth": 5, "max_iter": 350, "min_samples_leaf": 20},
        {"learning_rate": 0.03, "max_depth": 6, "max_iter": 500, "min_samples_leaf": 25},
    ]
    best_model = None
    best_meta: dict[str, object] | None = None

    for params in candidates:
        model = HistGradientBoostingRegressor(random_state=42, **params)
        model.fit(x_train, y_train)
        residual_prediction = pd.Series(model.predict(x_valid), index=x_valid.index)
        prediction = (base_valid + residual_prediction).clip(lower=0.0)
        prediction = pd.Series(
            prediction.where(prediction <= available_capacity_valid, available_capacity_valid),
            index=x_valid.index,
        )
        rmse = _rmse(y_valid + base_valid, prediction)
        if best_meta is None or rmse < best_meta["validation_rmse_mw"]:
            best_model = model
            best_meta = {
                "fuel": fuel,
                "model": "gradient_boosted",
                "validation_rmse_mw": rmse,
                "params": params,
            }

    assert best_model is not None and best_meta is not None
    return best_model, best_meta


def run_fuel_forecast(master: pd.DataFrame, fuel: str) -> FuelForecastArtifacts:
    frame, feature_columns = prepare_fuel_frame(master, fuel)

    train = frame[frame["split"] == "train"].copy()
    valid = frame[frame["split"] == "validation"].copy()
    test = frame[frame["split"] == "test"].copy()

    x_train = train[feature_columns]
    y_train_residual = train["residual_target"]

    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train_residual)

    gradient_model, gradient_meta = _fit_gradient_boosting(
        x_train=x_train,
        y_train=y_train_residual,
        x_valid=valid[feature_columns],
        y_valid=valid["residual_target"],
        base_valid=valid["ieso_raw"],
        available_capacity_valid=valid["available_capacity"],
        fuel=fuel,
    )

    predictions = frame[
        [
            "timestamp",
            "split",
            "season",
            "hour",
            "hour_block",
            "target_actual",
            "available_capacity",
            "ieso_raw",
        ]
    ].copy()
    predictions["seasonal_naive"] = frame["target_lag_24"]

    linear_prediction = frame["ieso_raw"] + linear_model.predict(frame[feature_columns])
    gradient_prediction = frame["ieso_raw"] + gradient_model.predict(frame[feature_columns])

    predictions["linear_residual"] = linear_prediction.clip(lower=0.0)
    predictions["gradient_boosted"] = gradient_prediction.clip(lower=0.0)

    for column in ["seasonal_naive", "ieso_raw", "linear_residual", "gradient_boosted"]:
        predictions[column] = predictions[column].where(
            predictions[column] <= predictions["available_capacity"],
            predictions["available_capacity"],
        )
        predictions[column] = predictions[column].clip(lower=0.0)

    model_candidates = []
    for method in ["linear_residual", "gradient_boosted"]:
        valid_slice = predictions[predictions["split"] == "validation"]
        rmse = _rmse(valid_slice["target_actual"], valid_slice[method])
        model_candidates.append({"method": method, "validation_rmse_mw": rmse})
    best_ml_method = min(model_candidates, key=lambda item: item["validation_rmse_mw"])["method"]
    predictions["best_ml"] = predictions[best_ml_method]

    overall_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []

    for split in ["train", "validation", "test"]:
        split_frame = predictions[predictions["split"] == split]
        for method in FORECAST_METHODS:
            overall_rows.append(
                _metric_row(
                    actual=split_frame["target_actual"],
                    predicted=split_frame[method],
                    available_capacity=split_frame["available_capacity"],
                    fuel=fuel,
                    split=split,
                    method=method,
                    segment_type="overall",
                    segment_value="all",
                )
            )
            for season, season_frame in split_frame.groupby("season"):
                segment_rows.append(
                    _metric_row(
                        actual=season_frame["target_actual"],
                        predicted=season_frame[method],
                        available_capacity=season_frame["available_capacity"],
                        fuel=fuel,
                        split=split,
                        method=method,
                        segment_type="season",
                        segment_value=str(season),
                    )
                )
            for hour_block, hour_frame in split_frame.groupby("hour_block"):
                segment_rows.append(
                    _metric_row(
                        actual=hour_frame["target_actual"],
                        predicted=hour_frame[method],
                        available_capacity=hour_frame["available_capacity"],
                        fuel=fuel,
                        split=split,
                        method=method,
                        segment_type="hour_block",
                        segment_value=str(hour_block),
                    )
                )

    overall_metrics = pd.DataFrame(overall_rows)
    segment_metrics = pd.DataFrame(segment_rows)
    model_selection = {
        "fuel": fuel,
        "best_ml_method": best_ml_method,
        "gradient_boosting_selection": gradient_meta,
        "candidate_validation_rmse": model_candidates,
    }
    return FuelForecastArtifacts(
        fuel=fuel,
        predictions=predictions,
        overall_metrics=overall_metrics,
        segment_metrics=segment_metrics,
        model_selection=model_selection,
    )


def plot_forecast_results(artifacts: list[FuelForecastArtifacts]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    overall = pd.concat([artifact.overall_metrics for artifact in artifacts], ignore_index=True)
    test_overall = overall[(overall["split"] == "test") & (overall["segment_type"] == "overall")]
    metric_plot = test_overall.pivot_table(index="method", columns="fuel", values="rmse_mw")
    ax = metric_plot.loc[["seasonal_naive", "ieso_raw", "linear_residual", "gradient_boosted", "best_ml"]].plot(
        kind="bar", figsize=(10, 6)
    )
    ax.set_title("2025 Test RMSE by Forecast Method and Fuel")
    ax.set_ylabel("RMSE (MW)")
    ax.set_xlabel("Forecast method")
    ax.figure.tight_layout()
    ax.figure.savefig(FIGURES_DIR / "forecast_test_rmse_by_method.png", dpi=200)
    plt.close(ax.figure)

    for artifact in artifacts:
        fuel_key = artifact.fuel.lower()
        sample = artifact.predictions[
            (artifact.predictions["split"] == "test")
            & (artifact.predictions["timestamp"] >= pd.Timestamp("2025-01-01"))
            & (artifact.predictions["timestamp"] < pd.Timestamp("2025-01-08"))
        ].copy()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(sample["timestamp"], sample["target_actual"], label="Actual", linewidth=2)
        ax.plot(sample["timestamp"], sample["ieso_raw"], label="IESO raw forecast", alpha=0.8)
        ax.plot(sample["timestamp"], sample["best_ml"], label="Best ML forecast", alpha=0.8)
        ax.set_title(f"{artifact.fuel} Forecast Comparison, First Test Week of 2025")
        ax.set_ylabel("MW")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"{fuel_key}_forecast_first_test_week.png", dpi=200)
        plt.close(fig)

        hourly = artifact.segment_metrics[
            (artifact.segment_metrics["split"] == "test")
            & (artifact.segment_metrics["segment_type"] == "hour_block")
        ]
        hourly = hourly.pivot_table(index="segment_value", columns="method", values="mae_mw")
        fig, ax = plt.subplots(figsize=(10, 5))
        hourly[["seasonal_naive", "ieso_raw", "linear_residual", "gradient_boosted", "best_ml"]].plot(
            kind="bar", ax=ax
        )
        ax.set_title(f"{artifact.fuel} 2025 Test MAE by Hour Block")
        ax.set_ylabel("MAE (MW)")
        ax.set_xlabel("Hour block")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"{fuel_key}_mae_by_hour_block.png", dpi=200)
        plt.close(fig)


def run_all_forecasts(master: pd.DataFrame) -> dict[str, object]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = [run_fuel_forecast(master, fuel) for fuel in ["Wind", "Solar"]]

    all_predictions = []
    all_overall_metrics = []
    all_segment_metrics = []
    model_selection: dict[str, object] = {}

    for artifact in artifacts:
        fuel_key = artifact.fuel.lower()
        predictions = artifact.predictions.copy()
        predictions.insert(1, "fuel", artifact.fuel)
        predictions.to_csv(RESULTS_DIR / f"{fuel_key}_predictions.csv", index=False)
        artifact.overall_metrics.to_csv(RESULTS_DIR / f"{fuel_key}_overall_metrics.csv", index=False)
        artifact.segment_metrics.to_csv(RESULTS_DIR / f"{fuel_key}_segment_metrics.csv", index=False)

        model_selection[fuel_key] = artifact.model_selection
        all_predictions.append(predictions)
        all_overall_metrics.append(artifact.overall_metrics)
        all_segment_metrics.append(artifact.segment_metrics)

    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    combined_overall = pd.concat(all_overall_metrics, ignore_index=True)
    combined_segment = pd.concat(all_segment_metrics, ignore_index=True)

    combined_predictions.to_csv(RESULTS_DIR / "all_fuel_predictions.csv", index=False)
    combined_overall.to_csv(RESULTS_DIR / "forecast_overall_metrics.csv", index=False)
    combined_segment.to_csv(RESULTS_DIR / "forecast_segment_metrics.csv", index=False)
    with (RESULTS_DIR / "forecast_model_selection.json").open("w", encoding="utf-8") as handle:
        json.dump(model_selection, handle, indent=2)

    plot_forecast_results(artifacts)
    return {
        "predictions": combined_predictions,
        "overall_metrics": combined_overall,
        "segment_metrics": combined_segment,
        "model_selection": model_selection,
    }
