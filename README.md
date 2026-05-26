# Smart Air — One-Hour-Ahead AQI Forecasting and Personalised Advisory Framework

A reproducible Python pipeline for hourly Air Quality Index (AQI) forecasting in **London** and **Manchester**, developed as an undergraduate dissertation project. Smart Air collects open environmental data, constructs EPA-standard AQI values, engineers forecasting features, trains and compares three models, and produces publication-quality evaluation figures.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Setup and Installation](#setup-and-installation)
- [Pipeline Steps](#pipeline-steps)
- [Data Sources](#data-sources)
- [Models and Methods](#models-and-methods)
- [Results Summary](#results-summary)
- [Output Figures](#output-figures)
- [Reproducibility Notes](#reproducibility-notes)
- [Limitations](#limitations)
- [Future Work](#future-work)

---

## Project Overview

Smart Air investigates whether machine learning models can accurately forecast one-hour-ahead AQI in two UK cities, and how those forecasts can support user-oriented health recommendations.

**Research question:** To what extent can machine learning models accurately forecast one-hour-ahead urban AQI in London and Manchester, and how can these forecasts be structured into a profile-based advisory framework for daily activity planning?

**Key findings:**
- Random Forest reduced MAE by **42.3%** over a persistence baseline in London and **27.9%** in Manchester
- AQI lag features dominate predictive importance in London; PM2.5 24-hour average leads in Manchester
- A rule-based advisory layer demonstrates how the same AQI value produces differentiated guidance across user profiles (healthy adult, child/elderly, respiratory condition)

---

## Repository Structure

```
smart-air/
│
├── README.md                   ← This file
├── requirements.txt            ← Python dependencies
├── .gitignore
├── run_pipeline.py             ← Master script: runs all steps in sequence
│
├── src/                        ← Individual pipeline stage scripts
│   ├── __init__.py
│   ├── 01_clean_pollutant_data.py   ← Parse and clean raw UK-AIR CSV files
│   ├── 02_download_weather.py       ← Download weather from Open-Meteo API
│   ├── 03_merge_datasets.py         ← Merge pollutant + weather by timestamp
│   ├── 04_build_aqi.py              ← Construct AQI using EPA breakpoints
│   ├── 05_feature_engineering.py    ← Lag, rolling mean, temporal features
│   ├── 06_train_models.py           ← Train models + TimeSeriesSplit CV
│   └── 07_generate_figures.py       ← Generate all 6 dissertation figures
│
├── data/
│   ├── raw/                    ← Place downloaded UK-AIR CSV files here
│   ├── processed/              ← Intermediate and final CSVs (auto-generated)
│   └── figures/                ← Output PNG figures (auto-generated)
│
├── notebooks/                  ← Optional Jupyter notebooks for exploration
│
└── docs/                       ← Additional documentation
```

---

## Quick Start

If you already have the processed data file `smart_air_2025_features.csv` (provided separately), you can skip to step 6 and run just the model training and figure generation:

```bash
git clone https://github.com/YOUR_USERNAME/smart-air.git
cd smart-air
pip install -r requirements.txt

# Copy your data file into data/processed/
cp /path/to/smart_air_2025_features.csv data/processed/

# Run model training and generate all figures
python run_pipeline.py --from 6
```

To reproduce everything from raw data, follow the full setup below.

---

## Setup and Installation

### 1. Prerequisites

- **Python 3.10 or higher** — [Download](https://www.python.org/downloads/)
- **pip** — comes bundled with Python

Check your version:
```bash
python --version
pip --version
```

### 2. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/smart-air.git
cd smart-air
```

### 3. Create a virtual environment (recommended)

```bash
# Create
python -m venv venv

# Activate — macOS / Linux
source venv/bin/activate

# Activate — Windows PowerShell
venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

The full requirements are minimal:

| Package | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 2.0 | Data loading, cleaning, merging |
| `numpy` | ≥ 1.24 | Numerical operations |
| `scikit-learn` | ≥ 1.3 | LinearRegression, RandomForest, TimeSeriesSplit |
| `matplotlib` | ≥ 3.7 | All figure generation |
| `requests` | ≥ 2.28 | Open-Meteo API download |

### 5. Download raw data files

Place the following files inside `data/raw/`:

**Pollutant data — UK-AIR (DEFRA)**

1. Go to [https://uk-air.defra.gov.uk/data/data_selector](https://uk-air.defra.gov.uk/data/data_selector)
2. Select **London Bloomsbury** site, year **2025**, all pollutants, hourly download
3. Save as `data/raw/london_bloomsbury_2025.csv`
4. Repeat for **Manchester Piccadilly** → `data/raw/manchester_piccadilly_2025.csv`

> **Note:** Weather data is downloaded automatically in Step 2 — no manual download required.

---

## Pipeline Steps

### Running the full pipeline

```bash
python run_pipeline.py
```

This runs all seven steps in sequence. Total runtime is approximately **3–8 minutes** depending on download speed and machine.

### Running individual steps

```bash
# Run from a specific step onwards
python run_pipeline.py --from 4

# Run only one step
python run_pipeline.py --only 7

# Run each script directly
python src/01_clean_pollutant_data.py
python src/07_generate_figures.py
```

---

### Step 1 — Clean Pollutant Data (`01_clean_pollutant_data.py`)

Reads the raw UK-AIR export files, which include several metadata header rows and HTML-encoded column names. Extracts PM10, PM2.5, NO₂, and O₃ columns by positional index, builds a UTC datetime field, removes invalid timestamps and rows missing all four pollutants, and writes clean CSVs.

**Input:** `data/raw/london_bloomsbury_2025.csv`, `data/raw/manchester_piccadilly_2025.csv`  
**Output:** `data/processed/london_bloomsbury_2025_clean.csv`, `data/processed/manchester_piccadilly_2025_clean.csv`

---

### Step 2 — Download Weather Data (`02_download_weather.py`)

Calls the [Open-Meteo Historical Weather API](https://open-meteo.com/) to retrieve hourly temperature (°C), relative humidity (%), and wind speed (km/h) for the coordinates of each monitoring site. No API key is needed.

```bash
# Custom date range
python src/02_download_weather.py --start 2025-01-01 --end 2025-12-31
```

**Output:** `data/processed/london_weather_2025.csv`, `data/processed/manchester_weather_2025.csv`, `data/processed/weather_data_2025_combined.csv`

---

### Step 3 — Merge Datasets (`03_merge_datasets.py`)

Joins pollutant and weather CSVs on `city + datetime_utc` using a left join, so all pollutant records are retained even if weather data is missing for a given hour. Duplicate timestamps are removed and records missing all four pollutants are dropped.

**Output:** `data/processed/london_merged_hourly.csv`, `data/processed/manchester_merged_hourly.csv`, `data/processed/smart_air_2025_merged.csv`

---

### Step 4 — Build AQI (`04_build_aqi.py`)

Constructs the forecasting target variable using the US EPA linear-interpolation method (40 CFR Appendix G):

```
I = ((I_Hi − I_Lo) / (BP_Hi − BP_Lo)) × (C − BP_Lo) + I_Lo
```

Averaging periods applied before sub-index calculation:
- **PM2.5**: 24-hour rolling mean
- **PM10**: 24-hour rolling mean  
- **O₃**: 8-hour rolling mean
- **NO₂**: 1-hour instantaneous

The final AQI for each hour is the **maximum sub-index** across the four pollutants.

**Output:** `data/processed/smart_air_2025_with_aqi.csv`

---

### Step 5 — Feature Engineering (`05_feature_engineering.py`)

Generates the full modelling-ready feature matrix:

| Feature group | Features |
|---|---|
| AQI lags | `aqi_lag1`, `aqi_lag2`, `aqi_lag3`, `aqi_lag6` |
| Pollutant lags (×4 pollutants) | `pm10_lag1..6`, `pm25_lag1..6`, `no2_lag1..6`, `o3_lag1..6` |
| Weather lags (×3 variables) | `temperature_2m_lag1..6`, etc. |
| Rolling means (3h, 6h) | `aqi_roll3`, `aqi_roll6`, `pm25_roll3`, etc. |
| Temporal | `hour`, `day`, `month`, `dayofweek` |
| Cyclical encodings | `hour_sin`, `hour_cos`, `month_sin`, `month_cos` |

Lag features are computed within each city group to prevent cross-city data leakage. The 1-hour-ahead AQI target (`aqi_h1`) is created by shifting the AQI series forward by one time step.

**Output:** `data/processed/smart_air_2025_features.csv` (60 feature columns)

---

### Step 6 — Train Models (`06_train_models.py`)

Trains and evaluates three models per city using an **80/20 chronological train–test split**:

| Model | Description |
|---|---|
| **Persistence** | Predicts next-hour AQI = current AQI (`aqi_lag1`) |
| **Linear Regression** | Interpretable statistical baseline (`sklearn.LinearRegression`) |
| **Random Forest** | Non-linear ensemble model (`n_estimators=100`, `max_features='sqrt'`, `random_state=42`) |

A **5-fold walk-forward cross-validation** (`TimeSeriesSplit(n_splits=5, gap=1)`) is also run to assess result robustness across different time windows.

Metrics reported: **MAE**, **RMSE**, **sMAPE** (all in AQI units except sMAPE which is a percentage).

**Output:** `data/processed/model_results.csv`, `data/processed/model_cv_results.csv`, `data/processed/test_predictions.csv`

---

### Step 7 — Generate Figures (`07_generate_figures.py`)

Produces six high-resolution (200 DPI) dissertation figures:

| Filename | Description |
|---|---|
| `fig_performance_comparison.png` | MAE / RMSE / sMAPE grouped bar charts, both cities |
| `fig_actual_vs_predicted.png` | 3-week test-period time-series overlay |
| `fig_scatter_rf.png` | Actual vs Predicted scatter with R² annotation |
| `fig_feature_importance.png` | RF MDI feature importance, colour-coded by type |
| `fig_residuals.png` | Residuals over time, signed and filled |
| `fig_cv_results.png` | 5-fold CV MAE with error bars (mean ± std) |

All figures use consistent colours: Persistence = steel blue, Linear Regression = amber, Random Forest = forest green.

**Output:** `data/figures/*.png`

---

## Data Sources

| Source | Data | URL |
|---|---|---|
| UK-AIR (DEFRA) | Hourly PM10, PM2.5, NO₂, O₃ | https://uk-air.defra.gov.uk/ |
| Open-Meteo | Hourly temperature, humidity, wind speed | https://open-meteo.com/ |

**Study period:** January 2025 – December 2025  
**Monitoring sites:** London Bloomsbury, Manchester Piccadilly  
**Data licence:** Both sources are freely available for non-commercial research use.

---

## Models and Methods

### AQI Construction

EPA linear interpolation (40 CFR Appendix G to Part 58). Breakpoints used:

| AQI Range | Category | PM2.5 (µg/m³, 24h) | PM10 (µg/m³, 24h) | NO₂ (µg/m³, 1h) | O₃ (µg/m³, 8h) |
|---|---|---|---|---|---|
| 0–50 | Good | 0–9.0 | 0–54 | 0–101 | 0–108 |
| 51–100 | Moderate | 9.1–35.4 | 55–154 | 102–191 | 109–137 |
| 101–150 | Unhealthy for Sensitive Groups | 35.5–55.4 | 155–254 | 192–689 | 138–167 |
| 151–200 | Unhealthy | 55.5–125.4 | 255–354 | 690–1242 | 168–207 |
| 201–300 | Very Unhealthy | 125.5–225.4 | 355–424 | 1243–2393 | 208–393 |

### Validation Strategy

```
Full dataset (sorted chronologically)
│
├─ 80% Training set  ──→ model fitting
└─ 20% Test set      ──→ single-split evaluation (MAE, RMSE, sMAPE)

5-fold TimeSeriesSplit (gap=1)
│
├─ Fold 1: train[0:n1]   test[n1+1:n2]
├─ Fold 2: train[0:n2]   test[n2+1:n3]
├─ ...
└─ Fold 5: train[0:n4]   test[n4+1:n5]
           → CV MAE mean ± std reported
```

---

## Results Summary

### Single-split performance (80/20 temporal split)

| City | Model | MAE | RMSE | sMAPE |
|---|---|---|---|---|
| London | Persistence | 1.5150 | 3.6148 | 5.13% |
| London | Linear Regression | 1.4287 | 2.9921 | 4.40% |
| London | **Random Forest** | **0.8736** | **2.2947** | **2.81%** |
| Manchester | Persistence | 1.5850 | 2.8422 | 4.67% |
| Manchester | Linear Regression | 1.1516 | 2.1534 | 3.27% |
| Manchester | **Random Forest** | **1.1421** | **1.9573** | **3.36%** |

*MAE and RMSE in AQI units. Lower is better.*

### Cross-validation (5-fold TimeSeriesSplit, MAE mean ± std)

| City | Model | CV MAE |
|---|---|---|
| London | Persistence | 2.197 ± 0.600 |
| London | Linear Regression | 2.589 ± 0.990 |
| London | Random Forest | 1.926 ± 1.332 |
| Manchester | Persistence | 1.725 ± 0.334 |
| Manchester | Linear Regression | 1.793 ± 1.020 |
| Manchester | Random Forest | 1.672 ± 0.660 |

---

## Output Figures

All six figures are generated to `data/figures/` by running Step 7. A brief description of each:

**`fig_performance_comparison.png`** — Three-panel bar chart comparing MAE, RMSE, and sMAPE across all three models for both cities side by side. Axis labels include units.

**`fig_actual_vs_predicted.png`** — Time-series overlay of actual AQI and all three model predictions for a representative 3-week window in the test period. Useful for visually assessing how each model tracks sudden AQI changes.

**`fig_scatter_rf.png`** — Actual vs predicted scatter for Random Forest. Points on or near the y = x diagonal indicate low bias. R² and MAE annotated for each city.

**`fig_feature_importance.png`** — Horizontal bar chart of the top-10 features by mean decrease in impurity (MDI), with standard deviation error bars. Bars are colour-coded: AQI features (green), pollutant features (orange), meteorological features (blue), temporal features (grey).

**`fig_residuals.png`** — Signed residuals (Actual − Predicted) plotted over the full test period. Red fill = over-prediction, blue fill = under-prediction. Largest single-hour error is annotated.

**`fig_cv_results.png`** — Bar chart of 5-fold CV MAE (mean ± std) for each model and city. Error bars show performance variability across time windows.

---

## Reproducibility Notes

- All models use `random_state=42`
- Pipeline stages 3–7 are fully deterministic given the same input data
- Stage 2 (weather download) depends on the Open-Meteo API; results may vary if the API data is revised
- Stage 1 depends on the exact UK-AIR CSV format; if DEFRA changes their export format, column indices in `01_clean_pollutant_data.py` may need updating
- No hyperparameter tuning was performed; Random Forest uses scikit-learn defaults

---

## Limitations

- Single monitoring site per city — results may not represent city-wide air quality
- No ablation experiment (pollutant-only vs full feature set) was conducted
- Random Forest hyperparameters were not tuned
- SHAP explainability analysis was planned but not implemented — MDI feature importance is used instead
- The personalisation advisory layer is a design demonstration only and has not been validated with users or clinical experts

---

## Future Work

- Extend to additional UK cities and monitoring sites
- Implement SHAP explainability and compare to MDI results
- Conduct pollutant-only vs full-feature ablation experiment (tests H2)
- Add hyperparameter tuning with `GridSearchCV` or `RandomizedSearchCV`
- Extend forecast horizons to 3h, 6h, and 12h ahead
- Validate advisory recommendations with health practitioners

---

## References

- Breiman, L. (2001) 'Random forests', *Machine Learning*, 45(1), pp. 5–32
- Huang, K. et al. (2018) 'Predicting monthly high-resolution PM2.5 concentrations with random forest', *Environmental Pollution*, 242(Part A), pp. 675–683
- Kamara, A.A. and Harrison, R.M. (2021) 'Analysis of the air pollution climate of a central urban roadside supersite: London, Marylebone Road', *Atmospheric Environment*, 258, 118479
- Manning, M.I. et al. (2018) 'Diurnal patterns in global fine particulate matter concentration', *Environmental Science & Technology Letters*, 5(11), pp. 687–691
- Pedregosa, F. et al. (2011) 'Scikit-learn: Machine learning in Python', *JMLR*, 12, pp. 2825–2830
- US EPA (2024) *Air Quality Index: A Guide to Air Quality and Your Health*. Available at: https://www.airnow.gov/aqi/aqi-basics/
- Yu, R. et al. (2016) 'RAQ — A Random Forest approach for predicting air quality in urban sensing systems', *Sensors*, 16(1), 86

---

## Licence

This project is released for academic and educational use. Data from UK-AIR (DEFRA) and Open-Meteo is subject to their respective terms of use.
