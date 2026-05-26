"""
05_feature_engineering.py
=========================
Generates lag features, rolling-mean features, and cyclical temporal
encodings from the AQI dataset.  The output is the full modelling-ready
feature matrix used for training and evaluation.

Features created
----------------
Lag features (1, 2, 3, 6 hours back):
    aqi_lag{n}, pm10_lag{n}, pm25_lag{n}, no2_lag{n}, o3_lag{n},
    temperature_2m_lag{n}, relative_humidity_2m_lag{n}, wind_speed_10m_lag{n}

Rolling means (3h and 6h windows):
    aqi_roll{w}, pm10_roll{w}, pm25_roll{w}, no2_roll{w}, o3_roll{w}

Temporal features:
    hour, day, month, dayofweek
    hour_sin, hour_cos   (cyclical encoding of hour-of-day)
    month_sin, month_cos (cyclical encoding of month)

Input  (data/processed/):
    smart_air_2025_with_aqi.csv

Output (data/processed/):
    smart_air_2025_features.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

LAG_STEPS    = [1, 2, 3, 6]
ROLL_WINDOWS = [3, 6]
LAG_COLS     = ["aqi", "pm10", "pm25", "no2", "o3",
                 "temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
ROLL_COLS    = ["aqi", "pm10", "pm25", "no2", "o3"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["city", "datetime_utc"]).reset_index(drop=True)

    # ── Temporal features ────────────────────────────────────────────────────
    dt = pd.to_datetime(df["datetime_utc"])
    df["hour"]      = dt.dt.hour
    df["day"]       = dt.dt.day
    df["month"]     = dt.dt.month
    df["dayofweek"] = dt.dt.dayofweek  # 0 = Monday

    # Cyclical encodings (avoid discontinuity at midnight / end of year)
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ── Lag features (grouped by city to avoid leakage across cities) ────────
    grp = df.groupby("city")
    for col in LAG_COLS:
        if col not in df.columns:
            continue
        for lag in LAG_STEPS:
            df[f"{col}_lag{lag}"] = grp[col].shift(lag)

    # ── Rolling-mean features ────────────────────────────────────────────────
    for col in ROLL_COLS:
        if col not in df.columns:
            continue
        for window in ROLL_WINDOWS:
            # shift(1) ensures no same-hour leakage into the rolling window
            df[f"{col}_roll{window}"] = (
                grp[col]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )

    # Drop rows with NaN in the most critical lag (aqi_lag1)
    df = df.dropna(subset=["aqi_lag1"])
    df = df.reset_index(drop=True)

    return df


def main():
    print("=== Step 5: Feature engineering ===")
    aqi_df = pd.read_csv(PROC_DIR / "smart_air_2025_with_aqi.csv")
    aqi_df["datetime_utc"] = pd.to_datetime(aqi_df["datetime_utc"])

    features = engineer_features(aqi_df)

    out_path = PROC_DIR / "smart_air_2025_features.csv"
    features.to_csv(out_path, index=False)
    print(f"  Rows: {len(features):,}  |  Columns: {len(features.columns)}")
    print(f"  Written → {out_path.name}")

    for city in features["city"].unique():
        sub = features[features["city"] == city]
        print(f"  [{city}] {len(sub):,} rows  |  "
              f"{sub['datetime_utc'].min()} → {sub['datetime_utc'].max()}")
    print("Done.\n")


if __name__ == "__main__":
    main()
