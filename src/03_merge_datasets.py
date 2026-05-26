"""
03_merge_datasets.py
====================
Merges the cleaned pollutant CSVs with the downloaded weather data on
city and hourly timestamp.  Produces one merged file per city and a
combined file containing both cities.

Input  (data/processed/):
    london_bloomsbury_2025_clean.csv
    manchester_piccadilly_2025_clean.csv
    london_weather_2025.csv
    manchester_weather_2025.csv

Output (data/processed/):
    london_merged_hourly.csv
    manchester_merged_hourly.csv
    smart_air_2025_merged.csv
"""

import pandas as pd
from pathlib import Path

PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

CITIES = {
    "London": {
        "pollutant": "london_bloomsbury_2025_clean.csv",
        "weather":   "london_weather_2025.csv",
        "out":       "london_merged_hourly.csv",
    },
    "Manchester": {
        "pollutant": "manchester_piccadilly_2025_clean.csv",
        "weather":   "manchester_weather_2025.csv",
        "out":       "manchester_merged_hourly.csv",
    },
}


def _load_pollutant(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    # Normalise to the hour (strip sub-hour info if any)
    df["datetime_utc"] = df["datetime_utc"].dt.floor("h")
    return df


def _load_weather(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df["datetime_utc"] = df["datetime_utc"].dt.floor("h")
    return df[["datetime_utc", "temperature_2m",
               "relative_humidity_2m", "wind_speed_10m"]]


def merge_city(city: str, cfg: dict) -> pd.DataFrame:
    poll  = _load_pollutant(PROC_DIR / cfg["pollutant"])
    weath = _load_weather(PROC_DIR / cfg["weather"])

    merged = poll.merge(weath, on="datetime_utc", how="left")

    # Drop rows missing all four pollutants
    merged = merged.dropna(subset=["pm10", "pm25", "no2", "o3"], how="all")

    # Remove duplicate timestamps (keep first)
    merged = merged.drop_duplicates("datetime_utc").sort_values("datetime_utc")
    merged = merged.reset_index(drop=True)

    out_path = PROC_DIR / cfg["out"]
    merged.to_csv(out_path, index=False)
    print(f"  [{city}] {len(merged):,} rows → {out_path.name}")

    # Report weather coverage
    weather_coverage = merged["temperature_2m"].notna().mean() * 100
    print(f"    Weather coverage: {weather_coverage:.1f}%")
    return merged


def main():
    print("=== Step 3: Merging pollutant and weather data ===")
    frames = []
    for city, cfg in CITIES.items():
        frames.append(merge_city(city, cfg))

    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["city", "datetime_utc"]
    )
    out_path = PROC_DIR / "smart_air_2025_merged.csv"
    combined.to_csv(out_path, index=False)
    print(f"  Combined: {len(combined):,} rows → {out_path.name}")
    print("Done.\n")


if __name__ == "__main__":
    main()
