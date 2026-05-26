"""
02_download_weather.py
======================
Downloads hourly meteorological data for London and Manchester from the
Open-Meteo historical weather API.  No API key is required.

Variables downloaded:
    temperature_2m          (°C)
    relative_humidity_2m    (%)
    wind_speed_10m          (km/h)

Output (written to data/processed/):
    london_weather_2025.csv
    manchester_weather_2025.csv
    weather_data_2025_combined.csv

Usage:
    python src/02_download_weather.py
    python src/02_download_weather.py --start 2025-01-01 --end 2025-12-31
"""

import argparse
import requests
import pandas as pd
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

API_URL = "https://archive-api.open-meteo.com/v1/archive"

CITIES = {
    "London": {
        "lat": 51.5208,   # London Bloomsbury coordinates
        "lon": -0.1247,
        "out": "london_weather_2025.csv",
    },
    "Manchester": {
        "lat": 53.4814,   # Manchester Piccadilly coordinates
        "lon": -2.2374,
        "out": "manchester_weather_2025.csv",
    },
}

DEFAULT_START = "2025-01-01"
DEFAULT_END   = "2025-12-31"


def download_weather(city: str, cfg: dict, start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude":  cfg["lat"],
        "longitude": cfg["lon"],
        "start_date": start,
        "end_date":   end,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "UTC",
    }
    print(f"  [{city}] Requesting {start} → {end} …", end=" ", flush=True)
    resp = requests.get(API_URL, params=params, timeout=60)
    resp.raise_for_status()
    payload = resp.json()

    hourly = payload["hourly"]
    df = pd.DataFrame({
        "datetime_utc":          hourly["time"],
        "temperature_2m":        hourly["temperature_2m"],
        "relative_humidity_2m":  hourly["relative_humidity_2m"],
        "wind_speed_10m":        hourly["wind_speed_10m"],
        "city": city,
    })
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])

    out_path = OUT_DIR / cfg["out"]
    df.to_csv(out_path, index=False)
    print(f"{len(df):,} rows → {out_path.name}")
    return df


def main(start: str = DEFAULT_START, end: str = DEFAULT_END):
    print("=== Step 2: Downloading weather data ===")
    frames = []
    for city, cfg in CITIES.items():
        frames.append(download_weather(city, cfg, start, end))

    combined = pd.concat(frames, ignore_index=True)
    combined_path = OUT_DIR / "weather_data_2025_combined.csv"
    combined.to_csv(combined_path, index=False)
    print(f"  Combined file: {combined_path.name} ({len(combined):,} rows)")
    print("Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Open-Meteo weather data")
    parser.add_argument("--start", default=DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default=DEFAULT_END,   help="End date YYYY-MM-DD")
    args = parser.parse_args()
    main(start=args.start, end=args.end)
