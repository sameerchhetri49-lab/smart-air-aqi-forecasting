"""
01_clean_pollutant_data.py
==========================
Reads the raw UK-AIR CSV files for London Bloomsbury and Manchester Piccadilly,
strips the metadata header rows, extracts PM10, PM2.5, NO2, and O3 columns,
standardises column names, and writes clean CSVs ready for merging.

Input  (place in data/raw/):
    london_bloomsbury_2025.csv
    manchester_piccadilly_2025.csv

Output (written to data/processed/):
    london_bloomsbury_2025_clean.csv
    manchester_piccadilly_2025_clean.csv
"""

import pandas as pd
from pathlib import Path

RAW_DIR  = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT_DIR  = Path(__file__).resolve().parents[1] / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SITES = {
    "London": {
        "file": "london_bloomsbury_2025.csv",
        "site": "London Bloomsbury",
        "out":  "london_bloomsbury_2025_clean.csv",
    },
    "Manchester": {
        "file": "manchester_piccadilly_2025.csv",
        "site": "Manchester Piccadilly",
        "out":  "manchester_piccadilly_2025_clean.csv",
    },
}

# UK-AIR files have 5 metadata rows before the actual header
SKIP_ROWS = 4

# Column names as they appear in the UK-AIR export (positional indices)
# Date=0, Time=1, PM10=2, NO=5, NO2=8, NOx=11, O3=14, PM25=17
COL_IDX = {"date": 0, "time": 1, "pm10": 2, "no2": 8, "o3": 14, "pm25": 17}


def clean_site(city: str, cfg: dict) -> pd.DataFrame:
    path = RAW_DIR / cfg["file"]
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file not found: {path}\n"
            f"Download from https://uk-air.defra.gov.uk/ and place in data/raw/"
        )

    # Read ignoring the metadata rows; use low_memory=False for mixed-type columns
    raw = pd.read_csv(path, skiprows=SKIP_ROWS, header=0,
                      low_memory=False, encoding="utf-8")

    # Drop the blank separator row that UK-AIR inserts after the header
    raw = raw.dropna(how="all").reset_index(drop=True)

    # Pull only the columns we need by positional index
    df = pd.DataFrame()
    df["date"]  = raw.iloc[:, COL_IDX["date"]].astype(str).str.strip()
    df["time"]  = raw.iloc[:, COL_IDX["time"]].astype(str).str.strip()
    df["pm10"]  = pd.to_numeric(raw.iloc[:, COL_IDX["pm10"]],  errors="coerce")
    df["pm25"]  = pd.to_numeric(raw.iloc[:, COL_IDX["pm25"]],  errors="coerce")
    df["no2"]   = pd.to_numeric(raw.iloc[:, COL_IDX["no2"]],   errors="coerce")
    df["o3"]    = pd.to_numeric(raw.iloc[:, COL_IDX["o3"]],    errors="coerce")

    # Build datetime column (UK-AIR format: DD-MM-YYYY HH:MM)
    df["datetime_utc"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )
    df = df.dropna(subset=["datetime_utc"])
    df = df.drop(columns=["date", "time"])

    # Drop rows where all four pollutants are missing
    df = df.dropna(subset=["pm10", "pm25", "no2", "o3"], how="all")

    # Sort chronologically and remove duplicates
    df = df.sort_values("datetime_utc").drop_duplicates("datetime_utc")
    df = df.reset_index(drop=True)

    # Add city/site labels
    df.insert(0, "city", city)
    df.insert(1, "site", cfg["site"])

    # Reorder columns
    df = df[["city", "site", "datetime_utc", "pm10", "pm25", "no2", "o3"]]

    out_path = OUT_DIR / cfg["out"]
    df.to_csv(out_path, index=False)
    print(f"  [{city}] {len(df):,} rows written → {out_path.name}")
    return df


def main():
    print("=== Step 1: Cleaning pollutant data ===")
    for city, cfg in SITES.items():
        clean_site(city, cfg)
    print("Done.\n")


if __name__ == "__main__":
    main()
