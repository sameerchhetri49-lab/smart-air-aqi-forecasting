"""
run_pipeline.py
===============
Master script that runs all seven Smart Air pipeline stages in sequence.
Run this once to reproduce all results from raw data.

Usage:
    python run_pipeline.py              # run all stages
    python run_pipeline.py --from 3     # start from step 3 (merge onwards)
    python run_pipeline.py --only 7     # run only figure generation

Prerequisites:
    1. Place raw UK-AIR CSV files in data/raw/
    2. Install dependencies:  pip install -r requirements.txt
"""

import argparse
import sys
import time
from pathlib import Path

# Add src/ to path so individual scripts can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import importlib

STEPS = [
    (1, "01_clean_pollutant_data",  "Clean raw UK-AIR pollutant CSV files"),
    (2, "02_download_weather",      "Download Open-Meteo weather data"),
    (3, "03_merge_datasets",        "Merge pollutant and weather datasets"),
    (4, "04_build_aqi",             "Construct AQI values using EPA breakpoints"),
    (5, "05_feature_engineering",   "Engineer lag, rolling, and temporal features"),
    (6, "06_train_models",          "Train models and run cross-validation"),
    (7, "07_generate_figures",      "Generate dissertation figures"),
]


def run_step(step_num: int, module_name: str, description: str):
    print(f"\n{'='*60}")
    print(f"  Step {step_num}: {description}")
    print(f"{'='*60}")
    t0 = time.time()
    mod = importlib.import_module(module_name)
    mod.main()
    elapsed = time.time() - t0
    print(f"  Step {step_num} completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Run the Smart Air forecasting pipeline"
    )
    parser.add_argument(
        "--from", dest="from_step", type=int, default=1,
        help="Start from this step number (default: 1)",
    )
    parser.add_argument(
        "--only", dest="only_step", type=int, default=None,
        help="Run only this step number",
    )
    args = parser.parse_args()

    steps_to_run = STEPS
    if args.only_step:
        steps_to_run = [s for s in STEPS if s[0] == args.only_step]
        if not steps_to_run:
            print(f"Step {args.only_step} not found. Valid steps: 1–7")
            sys.exit(1)
    elif args.from_step > 1:
        steps_to_run = [s for s in STEPS if s[0] >= args.from_step]

    print("Smart Air — Forecasting Pipeline")
    print(f"Running steps: {[s[0] for s in steps_to_run]}")

    total_start = time.time()
    for step_num, module_name, description in steps_to_run:
        run_step(step_num, module_name, description)

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {total:.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
