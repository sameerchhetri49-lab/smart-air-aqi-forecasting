"""
06_train_models.py
==================
Trains and evaluates three forecasting models per city:
    1. Persistence baseline  (ŷ = AQI at lag-1)
    2. Linear Regression
    3. Random Forest

Validation strategy
-------------------
Primary:     80 / 20 chronological train–test split
Secondary:   5-fold walk-forward TimeSeriesSplit cross-validation (gap=1)

Outputs (data/processed/):
    model_results.csv        — single-split metrics (MAE, RMSE, sMAPE)
    model_cv_results.csv     — 5-fold CV metrics (mean ± std)
    test_predictions.csv     — full test-set predictions for all models
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# Reproducibility
RANDOM_STATE = 42

# Random Forest hyper-parameters (scikit-learn defaults; no tuning performed)
RF_PARAMS = dict(
    n_estimators=100,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# Columns that are NOT predictive features
NON_FEATURE_COLS = {"city", "site", "datetime_utc", "aqi"}


# ── metric helpers ─────────────────────────────────────────────────────────────
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        100.0 * np.mean(
            2.0 * np.abs(y_true - y_pred) /
            (np.abs(y_true) + np.abs(y_pred) + 1e-8)
        )
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE":   round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE":  round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "sMAPE": round(smape(y_true, y_pred), 4),
    }


# ── main training logic ────────────────────────────────────────────────────────
def train_and_evaluate(df: pd.DataFrame) -> tuple:
    """
    Returns
    -------
    results_rows  : list of dicts (one per city × model) for the single split
    cv_rows       : list of dicts for the TimeSeriesSplit CV
    pred_frames   : list of DataFrames with test-set predictions
    """
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    lag1_idx     = feature_cols.index("aqi_lag1")   # persistence column index

    results_rows = []
    cv_rows      = []
    pred_frames  = []

    for city in sorted(df["city"].unique()):
        sub = (
            df[df["city"] == city]
            .sort_values("datetime_utc")
            .reset_index(drop=True)
        )

        # Build 1-hour-ahead target
        sub["aqi_h1"] = sub["aqi"].shift(-1)
        sub = sub.dropna(subset=["aqi_h1"]).reset_index(drop=True)

        X = sub[feature_cols].values
        y = sub["aqi_h1"].values
        n = len(X)
        split = int(n * 0.8)

        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        # ── Persistence ──────────────────────────────────────────────────────
        persist_pred = X_te[:, lag1_idx]

        # ── Linear Regression ────────────────────────────────────────────────
        lr = LinearRegression()
        lr.fit(X_tr, y_tr)
        lr_pred = lr.predict(X_te)

        # ── Random Forest ────────────────────────────────────────────────────
        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(X_tr, y_tr)
        rf_pred = rf.predict(X_te)

        # ── Single-split metrics ─────────────────────────────────────────────
        for model_name, pred in [
            ("Persistence",       persist_pred),
            ("LinearRegression",  lr_pred),
            ("RandomForest",      rf_pred),
        ]:
            row = {"city": city, "model": model_name}
            row.update(compute_metrics(y_te, pred))
            results_rows.append(row)
            m = row
            print(f"  [{city}] {model_name:<22}  "
                  f"MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  sMAPE={m['sMAPE']:.4f}%")

        # ── Save test-set predictions ────────────────────────────────────────
        pred_df = sub.iloc[split:].copy()
        pred_df["pred_persistence"] = persist_pred
        pred_df["pred_linear"]      = lr_pred
        pred_df["pred_rf"]          = rf_pred
        pred_frames.append(pred_df)

        # ── 5-fold TimeSeriesSplit CV ────────────────────────────────────────
        tscv = TimeSeriesSplit(n_splits=5, gap=1)
        fold_data = {
            "Persistence":      [],
            "LinearRegression": [],
            "RandomForest":     [],
        }
        for tr_idx, te_idx in tscv.split(X):
            X_f_tr, X_f_te = X[tr_idx], X[te_idx]
            y_f_tr, y_f_te = y[tr_idx], y[te_idx]

            fold_data["Persistence"].append(
                mean_absolute_error(y_f_te, X_f_te[:, lag1_idx])
            )
            lr_f = LinearRegression(); lr_f.fit(X_f_tr, y_f_tr)
            fold_data["LinearRegression"].append(
                mean_absolute_error(y_f_te, lr_f.predict(X_f_te))
            )
            rf_f = RandomForestRegressor(
                n_estimators=50, max_features="sqrt",
                random_state=RANDOM_STATE, n_jobs=-1
            )
            rf_f.fit(X_f_tr, y_f_tr)
            fold_data["RandomForest"].append(
                mean_absolute_error(y_f_te, rf_f.predict(X_f_te))
            )

        for model_name, maes in fold_data.items():
            cv_rows.append({
                "city":     city,
                "model":    model_name,
                "MAE_mean": round(np.mean(maes), 4),
                "MAE_std":  round(np.std(maes),  4),
            })

    return results_rows, cv_rows, pred_frames


def main():
    print("=== Step 6: Training and evaluating models ===")
    df = pd.read_csv(PROC_DIR / "smart_air_2025_features.csv")
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])

    results_rows, cv_rows, pred_frames = train_and_evaluate(df)

    # ── Write outputs ────────────────────────────────────────────────────────
    pd.DataFrame(results_rows).to_csv(
        PROC_DIR / "model_results.csv", index=False
    )
    pd.DataFrame(cv_rows).to_csv(
        PROC_DIR / "model_cv_results.csv", index=False
    )
    pd.concat(pred_frames, ignore_index=True).to_csv(
        PROC_DIR / "test_predictions.csv", index=False
    )

    print(f"\n  Saved: model_results.csv")
    print(f"  Saved: model_cv_results.csv")
    print(f"  Saved: test_predictions.csv")
    print("Done.\n")


if __name__ == "__main__":
    main()
