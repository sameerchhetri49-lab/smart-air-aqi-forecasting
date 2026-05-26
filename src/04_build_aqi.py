"""
04_build_aqi.py
===============
Constructs the AQI target variable from pollutant measurements using the
US EPA linear-interpolation breakpoint method (40 CFR Appendix G).

For each hourly observation, a sub-index is computed for each available
pollutant.  The overall AQI equals the maximum sub-index across pollutants,
representing the pollutant of greatest concern at that hour.

Note on averaging periods
-------------------------
EPA AQI technically uses 24-hour averages for PM2.5 and PM10, 8-hour
rolling means for O3, and 1-hour instantaneous values for NO2.  This
script computes rolling windows within the dataset to approximate these
averaging periods.  NaN sub-index values are ignored when taking the max.

Input  (data/processed/):
    smart_air_2025_merged.csv

Output (data/processed/):
    smart_air_2025_with_aqi.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# ── EPA AQI Breakpoints ────────────────────────────────────────────────────────
# Each entry: (AQI_lo, AQI_hi, conc_lo, conc_hi)
# Source: US EPA (2024) 40 CFR Appendix G to Part 58
PM25_BPS = [          # µg/m³, 24-h average
    (0,   50,  0.0,   9.0),
    (51,  100, 9.1,  35.4),
    (101, 150, 35.5, 55.4),
    (151, 200, 55.5, 125.4),
    (201, 300, 125.5, 225.4),
    (301, 500, 225.5, 325.4),
]
PM10_BPS = [          # µg/m³, 24-h average
    (0,   50,   0,  54),
    (51,  100,  55, 154),
    (101, 150, 155, 254),
    (151, 200, 255, 354),
    (201, 300, 355, 424),
    (301, 500, 425, 604),
]
NO2_BPS = [           # µg/m³, 1-h value (converted from ppb: 1 ppb ≈ 1.913 µg/m³)
    (0,   50,    0,   101),
    (51,  100,  102,  191),
    (101, 150,  192,  689),
    (151, 200,  690, 1242),
    (201, 300, 1243, 2393),
    (301, 500, 2394, 3853),
]
O3_BPS = [            # µg/m³, 8-h rolling mean (converted from ppm: 1 ppm = 1961.4 µg/m³ at 25°C)
    (0,   50,   0,   108),
    (51,  100, 109,  137),
    (101, 150, 138,  167),
    (151, 200, 168,  207),
    (201, 300, 208,  393),
]


def _interpolate(conc: float, breakpoints: list) -> float:
    """Apply EPA linear interpolation to map a concentration to an AQI sub-index."""
    if np.isnan(conc) or conc < 0:
        return np.nan
    for (i_lo, i_hi, c_lo, c_hi) in breakpoints:
        if c_lo <= conc <= c_hi:
            return round(((i_hi - i_lo) / (c_hi - c_lo)) * (conc - c_lo) + i_lo, 2)
    # Above highest breakpoint → cap at 500
    return 500.0


def compute_aqi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Rolling averages (groupby city to avoid cross-city contamination)
    df["pm25_24h"] = (
        df.groupby("city")["pm25"]
        .transform(lambda s: s.rolling(24, min_periods=6).mean())
    )
    df["pm10_24h"] = (
        df.groupby("city")["pm10"]
        .transform(lambda s: s.rolling(24, min_periods=6).mean())
    )
    df["o3_8h"] = (
        df.groupby("city")["o3"]
        .transform(lambda s: s.rolling(8, min_periods=3).mean())
    )
    # NO2 uses 1-hour instantaneous; convert µg/m³ → ppb equivalent for NO2 breakpoints
    # UK-AIR reports NO2 in µg/m³; our breakpoints are in µg/m³ already
    # (values already converted in NO2_BPS above)

    # Sub-index for each pollutant
    df["aqi_pm25"] = df["pm25_24h"].apply(lambda x: _interpolate(x, PM25_BPS))
    df["aqi_pm10"] = df["pm10_24h"].apply(lambda x: _interpolate(x, PM10_BPS))
    df["aqi_no2"]  = df["no2"].apply(lambda x: _interpolate(x, NO2_BPS))
    df["aqi_o3"]   = df["o3_8h"].apply(lambda x: _interpolate(x, O3_BPS))

    # Overall AQI = max sub-index (ignoring NaN)
    sub_cols = ["aqi_pm25", "aqi_pm10", "aqi_no2", "aqi_o3"]
    df["aqi"] = df[sub_cols].max(axis=1)

    # Drop sub-index helper columns; keep the rolling averages for transparency
    df = df.drop(columns=sub_cols)

    return df


def main():
    print("=== Step 4: Constructing AQI values ===")
    merged = pd.read_csv(PROC_DIR / "smart_air_2025_merged.csv")
    merged["datetime_utc"] = pd.to_datetime(merged["datetime_utc"])
    merged = merged.sort_values(["city", "datetime_utc"]).reset_index(drop=True)

    result = compute_aqi(merged)

    # Drop rows where AQI could not be computed (usually the first 24h per city)
    before = len(result)
    result = result.dropna(subset=["aqi"])
    after  = len(result)
    print(f"  Rows dropped (insufficient rolling window): {before - after:,}")

    out_path = PROC_DIR / "smart_air_2025_with_aqi.csv"
    result.to_csv(out_path, index=False)
    print(f"  Rows written: {len(result):,} → {out_path.name}")

    # Summary
    for city in result["city"].unique():
        sub = result[result["city"] == city]
        print(f"  [{city}] AQI  min={sub['aqi'].min():.1f}  "
              f"mean={sub['aqi'].mean():.1f}  max={sub['aqi'].max():.1f}")
    print("Done.\n")


if __name__ == "__main__":
    main()
