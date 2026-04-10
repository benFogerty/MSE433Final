# MSE 433 Final Project

## Forecast-Informed Storage Dispatch with Generator-Level Unit Commitment

This repository now uses a single main optimization approach: a rolling **168-hour generator-level unit commitment model** with battery storage. The older storage-only approach and its legacy run folders were removed from the active workflow.

The project asks:

**Can improved wind and solar forecasts help Ontario dispatch its current storage fleet more effectively when battery decisions are embedded inside a more physical unit commitment model?**

## Main Case Study

- Jurisdiction: Ontario
- Optimization window: `2025-04-08 08:00:00` to `2025-12-31 23:00:00`
- Hours in optimization window: `6,423`
- Forecast training history: `2019-05-01` to `2025-12-31`
- Main storage fleet modeled from observed source-file assets: `1129 MW / 4516 MWh`

## Main Result

Against the historical observed storage baseline:

- Battery utilization increases from **5.99%** to **13.61%**
- Mean battery output increases from **27.70 MW** to **153.65 MW**
- Peak non-renewable requirement falls by **654 MW**
- Renewable utilization remains **100%** in this study window

Interpretation: the gain comes from **better storage dispatch**, not from recovering previously curtailed renewables. The optimized battery provides substantially more renewable-backed peak-support flexibility and could replace part of the gas-peaking role during stressed hours, without claiming full gas replacement.

## Final Workflow

1. Build the hourly Ontario modeling panel from the provided datasets.
2. Train and evaluate renewable forecasting models.
3. Solve the main rolling 168-hour generator-level UC with storage.
4. Run horizon and storage sensitivity checks.
5. Build a decision-support layer from the precomputed UC outputs.
6. Review the outputs in the Streamlit dashboard.

## Main Optimization Model

The final optimization is a generator-level mixed-integer UC that includes:

- hourly system power balance
- generator dispatch variables
- binary commitment for `GAS` and `BIOFUEL`
- startup and shutdown costs
- minimum and maximum output constraints
- minimum up and minimum down constraints
- ramp constraints
- battery charge, discharge, state of charge, and charge/discharge exclusivity
- renewable allocation across direct use, storage charging, and curtailment

Continuous-unit ramp assumptions used in the final model:

- `NUCLEAR`: `1%` of rated power per minute, or `60%` of rated power per hour
- `HYDRO`: can move from `0` to `Pmax` within the hour

Excluded on purpose because the provided data does not support them:

- transmission constraints
- voltage constraints
- reserve requirements
- nodal or bus-level network modeling

## Forecasting Results

The best model for both wind and solar is the gradient-boosted residual-correction model.

- Wind RMSE: **162.75 MW -> 91.68 MW**
- Solar RMSE: **10.93 MW -> 8.25 MW**

Those improved renewable forecasts feed directly into the UC storage-dispatch model.

## Rubric Alignment

This final structure is designed to map directly to the course rubric:

- **Clear problem and stakeholder**: Ontario storage dispatch for renewable integration and peak support.
- **Breadth and depth**: data engineering, forecasting, optimization, and operational interpretation.
- **Domain knowledge**: the final model includes commitment logic, startup/shutdown, min/max output, minimum up/down, ramping, battery SOC, and hourly power balance.
- **Validation and benchmarking**: renewable forecasts are tested on a locked 2025 holdout and storage is benchmarked against historical observed operation.
- **Quantified recommendations**: utilization improvement, MW output increase, and peak reduction are reported explicitly.
- **Reproducibility**: scripts regenerate the canonical results and report.

## Repository Structure

- `Data/`: raw datasets and descriptions
- `src/mse433_project/`: main project code, including the generator-level UC model
- `scripts/`: reproducible entry points
- `Data/processed/`: cleaned datasets
- `results/`: canonical published outputs
- `results/generator_uc/full_run_168h/`: canonical raw UC run used for the main results
- `streamlit_app.py`: lightweight dashboard for generator recommendations and energy-mix review
- `report/final_report.md`: final written report

## Main Outputs

- `Data/processed/master_hourly_dataset.csv`
- `results/forecast_overall_metrics.csv`
- `results/storage_policy_summary.csv`
- `results/storage_dispatch_hourly.csv`
- `results/gas_dispatch_summary.csv`
- `results/generator_uc_dispatch_hourly.csv`
- `results/generator_uc_solve_log.csv`
- `results/horizon_sensitivity_summary.csv`
- `results/storage_sensitivity_summary.csv`
- `results/generator_hourly_recommendations.csv`
- `results/generator_decision_support_summary.csv`
- `results/generator_supply_mix_hourly.csv`
- `results/utilization_summary.csv`
- `results/recommendation_table.csv`
- `report/final_report.md`

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the Main Workflow

Run everything from the repository root after activating `.venv`.

1. Build the processed hourly datasets:

```bash
python scripts/build_dataset.py
```

Expected outputs:

- `Data/processed/master_hourly_dataset.csv`
- `Data/processed/renewable_generator_hourly_dataset.csv`
- `Data/processed/dispatchable_generator_hourly_dataset.csv`
- `Data/processed/observed_storage_hourly_baseline.csv`

2. Train and evaluate the renewable forecast models:

```bash
python scripts/run_forecasts.py
```

Expected outputs:

- `results/forecast_overall_metrics.csv`
- `results/all_fuel_predictions.csv`
- `results/forecast_model_selection.json`

3. Run the canonical 168-hour generator-level UC and publish the main outputs:

```bash
python scripts/run_storage_backtest.py
```

Expected outputs:

- `results/generator_uc/full_run_168h/`
- `results/storage_policy_summary.csv`
- `results/storage_dispatch_hourly.csv`
- `results/gas_dispatch_summary.csv`
- `results/generator_uc_dispatch_hourly.csv`

4. Run the horizon and storage sensitivity checks:

```bash
python scripts/run_sensitivity_checks.py
```

Expected outputs:

- `results/horizon_sensitivity_summary.csv`
- `results/storage_sensitivity_summary.csv`
- `results/robustness_validation_summary.csv`

5. Build the generator decision-support layer from the precomputed UC outputs:

```bash
python scripts/run_decision_support.py
```

Expected outputs:

- `results/generator_hourly_recommendations.csv`
- `results/generator_decision_support_summary.csv`
- `results/generator_supply_mix_hourly.csv`

`scripts/run_storage_backtest.py` runs the **main 168-hour generator-level UC**, writes the canonical run under `results/generator_uc/full_run_168h/`, and publishes the main summary outputs to the root `results/` directory.

6. Validate the submission artifacts:

```bash
python -m unittest tests/test_project_outputs.py
```

The test suite checks the processed datasets, forecast outputs, UC outputs, sensitivity outputs, and final report.

## Streamlit Dashboard

After running `python scripts/run_decision_support.py`, you can launch the dashboard with:

```bash
streamlit run streamlit_app.py
```

The dashboard reads only the precomputed decision-support CSVs. It does not rerun the optimization model.

It includes:

- a stacked hourly energy-mix area chart by source type
- an hour-by-hour table of which generators should be on
- generator-level recommendation summaries, including likely role and dispatch behavior

## Important Assumptions

- Historical `Forecast` values in the monthly generator files are the renewable forecast baseline.
- Storage is optimized at the Ontario system level.
- The current storage fleet is taken from the observed storage rows in the source files.
- The optimized battery is charged from renewable energy in the modeled policy.
- Renewable utilization staying at `100%` means the value in this window comes from better timing of storage discharge, not extra renewable capture.
