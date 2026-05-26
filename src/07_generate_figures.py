"""
07_generate_figures.py
======================
Generates all six dissertation-quality figures from the processed data
and model results.  Figures are saved as high-resolution PNGs.

Figures produced (saved to data/figures/):
    fig_performance_comparison.png  — MAE / RMSE / sMAPE grouped bar charts
    fig_actual_vs_predicted.png     — 3-week test-period time-series overlay
    fig_scatter_rf.png              — Actual vs Predicted scatter (RF only)
    fig_feature_importance.png      — Random Forest MDI feature importance
    fig_residuals.png               — Residuals over time (RF)
    fig_cv_results.png              — 5-fold TimeSeriesSplit CV bar chart

All figures use consistent colours:
    Persistence      → steel blue   (#7FA7C9)
    Linear Regression → amber       (#F5A623)
    Random Forest    → forest green (#2E7D32)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
FIG_DIR  = Path(__file__).resolve().parents[1] / "data" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DPI          = 200
RANDOM_STATE = 42
RF_PARAMS    = dict(n_estimators=100, max_features="sqrt",
                    random_state=RANDOM_STATE, n_jobs=-1)

PALETTE = {
    "Persistence":       "#7FA7C9",
    "Linear Regression": "#F5A623",
    "Random Forest":     "#2E7D32",
}
CITY_COLORS = {"London": "#1A3A5C", "Manchester": "#8B1A1C"}
FEAT_COLORS = {
    "AQI":         "#2E7D32",
    "Pollutant":   "#E65100",
    "Meteorology": "#1565C0",
    "Temporal":    "#616161",
}
FEAT_LABELS = {
    "aqi_lag1":                  "AQI Lag-1",
    "aqi_lag2":                  "AQI Lag-2",
    "aqi_lag3":                  "AQI Lag-3",
    "aqi_lag6":                  "AQI Lag-6",
    "aqi_roll3":                 "AQI Roll-3h",
    "aqi_roll6":                 "AQI Roll-6h",
    "pm25_24h":                  "PM2.5 (24h avg)",
    "pm10_24h":                  "PM10 (24h avg)",
    "o3_8h":                     "O\u2083 (8h avg)",
    "pm25_lag1":                 "PM2.5 Lag-1",
    "pm10_lag1":                 "PM10 Lag-1",
    "no2_lag1":                  "NO\u2082 Lag-1",
    "pm25_roll3":                "PM2.5 Roll-3h",
    "pm10_roll3":                "PM10 Roll-3h",
    "pm25_lag6":                 "PM2.5 Lag-6",
    "pm10_roll6":                "PM10 Roll-6h",
    "wind_speed_10m_lag1":       "Wind Speed Lag-1",
    "temperature_2m_lag1":       "Temperature Lag-1",
    "relative_humidity_2m_lag1": "Humidity Lag-1",
    "hour_sin":                  "Hour (sin)",
    "hour_cos":                  "Hour (cos)",
    "month_sin":                 "Month (sin)",
}

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    11,
    "figure.dpi":         DPI,
})


# ── helpers ────────────────────────────────────────────────────────────────────
def smape(y, yhat):
    return 100.0 * np.mean(2.0 * np.abs(y - yhat) / (np.abs(y) + np.abs(yhat) + 1e-8))


def _feat_category(name: str) -> str:
    if "aqi" in name:
        return "AQI"
    if any(p in name for p in ("pm", "no2", "o3")):
        return "Pollutant"
    if any(p in name for p in ("temp", "humid", "wind")):
        return "Meteorology"
    return "Temporal"


def _save(fig, name: str):
    path = FIG_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {name}")


# ── load and retrain ───────────────────────────────────────────────────────────
def load_and_train():
    df = pd.read_csv(PROC_DIR / "smart_air_2025_features.csv")
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df.sort_values(["city", "datetime_utc"]).reset_index(drop=True)

    feat_cols = [c for c in df.columns
                 if c not in ("city", "site", "datetime_utc", "aqi")]
    lag1_idx  = feat_cols.index("aqi_lag1")

    df["aqi_h1"] = df.groupby("city")["aqi"].shift(-1)
    df = df.dropna(subset=["aqi_h1"]).reset_index(drop=True)

    results, rf_models = {}, {}
    metrics_dict = {}

    for city in ("London", "Manchester"):
        sub  = df[df["city"] == city].reset_index(drop=True)
        X    = sub[feat_cols].values
        y    = sub["aqi_h1"].values
        n    = len(X)
        split = int(n * 0.8)

        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        persist = X_te[:, lag1_idx]

        lr = LinearRegression(); lr.fit(X_tr, y_tr)
        lr_pred = lr.predict(X_te)

        rf = RandomForestRegressor(**RF_PARAMS); rf.fit(X_tr, y_tr)
        rf_pred = rf.predict(X_te)

        results[city]   = dict(
            y=y_te, persist=persist, lr=lr_pred, rf=rf_pred,
            dates=pd.to_datetime(sub["datetime_utc"].values[split:]),
        )
        rf_models[city] = rf

        metrics_dict[city] = {
            "Persistence":       {"MAE": mean_absolute_error(y_te, persist),
                                  "RMSE": float(np.sqrt(mean_squared_error(y_te, persist))),
                                  "sMAPE": smape(y_te, persist)},
            "Linear Regression": {"MAE": mean_absolute_error(y_te, lr_pred),
                                  "RMSE": float(np.sqrt(mean_squared_error(y_te, lr_pred))),
                                  "sMAPE": smape(y_te, lr_pred)},
            "Random Forest":     {"MAE": mean_absolute_error(y_te, rf_pred),
                                  "RMSE": float(np.sqrt(mean_squared_error(y_te, rf_pred))),
                                  "sMAPE": smape(y_te, rf_pred)},
        }

    return results, rf_models, metrics_dict, feat_cols, df


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — MAE / RMSE / sMAPE grouped bar chart (both cities)
# ══════════════════════════════════════════════════════════════════════════════
def fig_performance_comparison(metrics_dict):
    models  = ["Persistence", "Linear Regression", "Random Forest"]
    cities  = ["London", "Manchester"]
    x       = np.arange(len(cities))
    width   = 0.25
    colors  = [PALETTE[m] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Model Forecasting Performance: London and Manchester",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax_idx, metric in enumerate(("MAE", "RMSE", "sMAPE")):
        ax = axes[ax_idx]
        for i, model in enumerate(models):
            vals = [metrics_dict[c][model][metric] for c in cities]
            bars = ax.bar(x + (i - 1) * width, vals, width,
                          label=model, color=colors[i],
                          edgecolor="white", linewidth=0.5, zorder=3)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{val:.2f}", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(cities)
        unit = "(AQI units)" if metric in ("MAE", "RMSE") else "(%)"
        ax.set_ylabel(f"{metric} {unit}")
        ax.set_title(f"{metric} Comparison")
        ax.set_ylim(0, max(metrics_dict[c][m][metric]
                           for c in cities for m in models) * 1.35)
        ax.yaxis.grid(True, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=3,
               frameon=True, fancybox=False, edgecolor="#CCCCCC")
    plt.tight_layout()
    _save(fig, "fig_performance_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Actual vs Predicted time series (3-week window)
# ══════════════════════════════════════════════════════════════════════════════
def fig_actual_vs_predicted(results):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    fig.suptitle("Actual vs Predicted AQI — Test Period Sample (3 Weeks)",
                 fontsize=14, fontweight="bold")

    for ax_idx, city in enumerate(("London", "Manchester")):
        r  = results[city]
        ax = axes[ax_idx]
        n  = min(504, len(r["y"]))

        ax.plot(r["dates"][:n], r["y"][:n],
                color="#333333", lw=1.5, label="Actual AQI", zorder=5)
        ax.plot(r["dates"][:n], r["persist"][:n],
                color=PALETTE["Persistence"], lw=1.0, alpha=0.8,
                label="Persistence", ls="--", zorder=3)
        ax.plot(r["dates"][:n], r["lr"][:n],
                color=PALETTE["Linear Regression"], lw=1.0, alpha=0.8,
                label="Linear Regression", ls="-.", zorder=3)
        ax.plot(r["dates"][:n], r["rf"][:n],
                color=PALETTE["Random Forest"], lw=1.2, alpha=0.9,
                label="Random Forest", zorder=4)

        ax.set_title(f"{city} — Actual vs Predicted AQI")
        ax.set_ylabel("AQI (units)")
        ax.yaxis.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", fontsize=10)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    _save(fig, "fig_actual_vs_predicted.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Actual vs Predicted scatter (RF, both cities)
# ══════════════════════════════════════════════════════════════════════════════
def fig_scatter_rf(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Random Forest: Actual vs Predicted AQI Scatter",
                 fontsize=14, fontweight="bold")

    for ax_idx, city in enumerate(("London", "Manchester")):
        r   = results[city]
        ax  = axes[ax_idx]
        y, yhat = r["y"], r["rf"]

        ax.scatter(y, yhat, alpha=0.25, s=8, color=CITY_COLORS[city])
        lims = [min(y.min(), yhat.min()) - 2, max(y.max(), yhat.max()) + 2]
        ax.plot(lims, lims, "k--", lw=1.5, label="Perfect fit (y = x)", zorder=5)

        r2  = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
        mae = mean_absolute_error(y, yhat)
        ax.text(0.05, 0.93, f"R\u00b2 = {r2:.3f}\nMAE = {mae:.3f}",
                transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="#CCCCCC"))

        ax.set_xlabel("Actual AQI (units)")
        ax.set_ylabel("Predicted AQI (units)")
        ax.set_title(city)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.legend(fontsize=10)
        ax.yaxis.grid(True, alpha=0.25)
        ax.set_axisbelow(True)

    plt.tight_layout()
    _save(fig, "fig_scatter_rf.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Random Forest feature importance (MDI)
# ══════════════════════════════════════════════════════════════════════════════
def fig_feature_importance(rf_models, feat_cols):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Random Forest Feature Importance (Mean Decrease in Impurity)",
                 fontsize=14, fontweight="bold")

    for ax_idx, city in enumerate(("London", "Manchester")):
        rf  = rf_models[city]
        ax  = axes[ax_idx]
        imp = rf.feature_importances_
        std = np.std([t.feature_importances_ for t in rf.estimators_], axis=0)

        top_idx   = np.argsort(imp)[::-1][:10]
        top_names = [FEAT_LABELS.get(feat_cols[i], feat_cols[i]) for i in top_idx]
        top_vals  = imp[top_idx]
        top_std   = std[top_idx]
        bar_colors = [FEAT_COLORS[_feat_category(feat_cols[i])] for i in top_idx]

        ax.barh(range(10), top_vals[::-1], xerr=top_std[::-1],
                color=bar_colors[::-1], edgecolor="white", linewidth=0.5,
                error_kw={"ecolor": "#555555", "capsize": 3, "linewidth": 1.0},
                zorder=3)
        ax.set_yticks(range(10))
        ax.set_yticklabels(top_names[::-1], fontsize=10)
        ax.set_xlabel("Mean Decrease in Impurity")
        ax.set_title(city)
        ax.xaxis.grid(True, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        for idx, (val, s) in enumerate(zip(top_vals[::-1], top_std[::-1])):
            ax.text(val + s + 0.001, idx, f"{val:.3f}", va="center", fontsize=9)

    legend_elements = [
        Patch(facecolor=FEAT_COLORS["AQI"],         label="AQI features"),
        Patch(facecolor=FEAT_COLORS["Pollutant"],   label="Pollutant features"),
        Patch(facecolor=FEAT_COLORS["Meteorology"], label="Meteorological features"),
        Patch(facecolor=FEAT_COLORS["Temporal"],    label="Temporal features"),
    ]
    fig.legend(handles=legend_elements, loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=4,
               frameon=True, fancybox=False, edgecolor="#CCCCCC")
    plt.tight_layout()
    _save(fig, "fig_feature_importance.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Residuals over time (RF)
# ══════════════════════════════════════════════════════════════════════════════
def fig_residuals(results):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Random Forest Residuals Over Time (Test Period)",
                 fontsize=14, fontweight="bold")

    for ax_idx, city in enumerate(("London", "Manchester")):
        r   = results[city]
        ax  = axes[ax_idx]
        res = r["y"] - r["rf"]

        ax.axhline(0, color="#333333", lw=1.0, ls="--", zorder=3)
        ax.fill_between(r["dates"], 0, res, where=res >= 0,
                        alpha=0.5, color="#E53935", label="Over-prediction")
        ax.fill_between(r["dates"], 0, res, where=res < 0,
                        alpha=0.5, color="#1E88E5", label="Under-prediction")

        max_idx = int(np.argmax(np.abs(res)))
        ax.annotate(f"Max |error|: {np.abs(res[max_idx]):.1f}",
                    xy=(r["dates"][max_idx], res[max_idx]),
                    xytext=(r["dates"][max_idx],
                            res[max_idx] + (5 if res[max_idx] > 0 else -5)),
                    fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0))

        ax.set_title(f"{city} — Residuals (Actual \u2212 Predicted)")
        ax.set_ylabel("Residual (AQI units)")
        ax.yaxis.grid(True, alpha=0.25, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", fontsize=10)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    _save(fig, "fig_residuals.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — 5-fold TimeSeriesSplit CV results
# ══════════════════════════════════════════════════════════════════════════════
def fig_cv_results(df, feat_cols):
    lag1_idx = feat_cols.index("aqi_lag1")
    tscv = TimeSeriesSplit(n_splits=5, gap=1)
    cv = {}

    for city in ("London", "Manchester"):
        sub = (df[df["city"] == city]
               .sort_values("datetime_utc")
               .reset_index(drop=True))
        X   = sub[feat_cols].values
        y   = sub["aqi_h1"].values
        fold_mae = {m: [] for m in
                    ("Persistence", "Linear Regression", "Random Forest")}

        for tr_idx, te_idx in tscv.split(X):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            fold_mae["Persistence"].append(
                mean_absolute_error(y_te, X_te[:, lag1_idx]))
            lr = LinearRegression(); lr.fit(X_tr, y_tr)
            fold_mae["Linear Regression"].append(
                mean_absolute_error(y_te, lr.predict(X_te)))
            rf = RandomForestRegressor(
                n_estimators=50, max_features="sqrt",
                random_state=RANDOM_STATE, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            fold_mae["Random Forest"].append(
                mean_absolute_error(y_te, rf.predict(X_te)))

        cv[city] = fold_mae

    models = ["Persistence", "Linear Regression", "Random Forest"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("5-Fold TimeSeriesSplit Cross-Validation Results",
                 fontsize=14, fontweight="bold")

    for ax_idx, city in enumerate(("London", "Manchester")):
        ax    = axes[ax_idx]
        means = [np.mean(cv[city][m]) for m in models]
        stds  = [np.std(cv[city][m])  for m in models]

        bars = ax.bar(models, means, yerr=stds,
                      color=[PALETTE[m] for m in models],
                      edgecolor="white", linewidth=0.5, capsize=5, zorder=3,
                      error_kw={"ecolor": "#333333", "linewidth": 1.5})
        for bar, mv, sv in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    mv + sv + 0.05,
                    f"{mv:.3f}\u00b1{sv:.3f}",
                    ha="center", va="bottom", fontsize=9)

        ax.set_ylabel("MAE (AQI units)")
        ax.set_title(f"{city} — CV MAE (mean \u00b1 std)")
        ax.yaxis.grid(True, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=10)
        ax.set_ylim(0, max(means) * 1.5)

    plt.tight_layout()
    _save(fig, "fig_cv_results.png")


# ── run all ────────────────────────────────────────────────────────────────────
def main():
    print("=== Step 7: Generating figures ===")
    results, rf_models, metrics_dict, feat_cols, df = load_and_train()

    fig_performance_comparison(metrics_dict)
    fig_actual_vs_predicted(results)
    fig_scatter_rf(results)
    fig_feature_importance(rf_models, feat_cols)
    fig_residuals(results)
    fig_cv_results(df, feat_cols)

    print(f"\nAll figures saved to {FIG_DIR}\n")


if __name__ == "__main__":
    main()
