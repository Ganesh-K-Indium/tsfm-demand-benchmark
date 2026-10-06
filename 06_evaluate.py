"""Combine all forecasts, evaluate overall + by segment + by horizon, and generate visualizations."""
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import HORIZON, evaluate, wape


def main():
    data_path = "./data/m5_subset.parquet"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run 01_prepare_data.py first.")
        return

    df = pd.read_parquet(data_path)
    cutoff = df["ds"].max() - pd.Timedelta(days=HORIZON)
    actuals = df[df["ds"] > cutoff][["unique_id", "ds", "y"]].copy()

    # Discover and load all existing forecast files in results/
    forecast_files = [
        ("./results/baselines_forecast.parquet", "Baselines"),
        ("./results/lightgbm_forecast.parquet", "LightGBM"),
        ("./results/chronos_forecast.parquet", "Chronos"),
        ("./results/timesfm_forecast.parquet", "TimesFM"),
    ]

    loaded_dfs = []
    for f_path, label in forecast_files:
        if os.path.exists(f_path):
            f_df = pd.read_parquet(f_path)
            loaded_dfs.append(f_df)
            print(f"Loaded {label} from {f_path}: {f_df['model'].unique().tolist()}")
        else:
            print(f"Skipping {label} (not found: {f_path})")

    if not loaded_dfs:
        print("No forecast files found in ./results/. Run models first.")
        return

    fcsts = pd.concat(loaded_dfs, ignore_index=True)

    # Train history dictionary for MASE calculation (only up to cutoff)
    y_hist = {
        uid: g.sort_values("ds")["y"].to_numpy()
        for uid, g in df[df["ds"] <= cutoff].groupby("unique_id")
    }

    agg, per_series = evaluate(fcsts, actuals, y_hist)

    # Join segment metadata
    seg = df.groupby("unique_id")["segment"].first().reset_index()
    per_series = per_series.merge(seg, on="unique_id", how="left")

    print("\n" + "=" * 50)
    print("=== OVERALL BENCHMARK RESULTS ===")
    print("=" * 50)
    print(agg.to_string(index=False))

    print("\n" + "=" * 50)
    print("=== BREAKDOWN BY DEMAND VELOCITY SEGMENT ===")
    print("=" * 50)
    seg_summary = (
        per_series.groupby(["segment", "model"])[["WAPE", "MASE"]]
        .mean()
        .round(4)
        .unstack(level=0)
    )
    print(seg_summary)

    print("\n" + "=" * 50)
    print("=== BREAKDOWN BY FORECAST HORIZON BUCKET (WAPE) ===")
    print("=" * 50)
    fc = fcsts.merge(actuals, on=["unique_id", "ds"])
    fc["horizon_day"] = (pd.to_datetime(fc["ds"]) - cutoff).dt.days

    buckets = [
        ("h 1-7d", fc[fc.horizon_day <= 7]),
        ("h 8-14d", fc[(fc.horizon_day > 7) & (fc.horizon_day <= 14)]),
        ("h 15-28d", fc[fc.horizon_day > 14]),
    ]
    for b_name, b_df in buckets:
        if len(b_df) > 0:
            w = b_df.groupby("model").apply(
                lambda g: wape(g["y"], g["y_pred"]), include_groups=False
            ).round(4)
            print(f"\n{b_name}:\n{w.to_string()}")

    # Save summary tables
    agg.to_csv("./results/benchmark_summary.csv", index=False)
    per_series.to_csv("./results/per_series_summary.csv", index=False)
    print("\nSaved summary CSVs to ./results/")

    # Generate visual comparison chart
    try:
        plot_results(agg, per_series)
    except Exception as e:
        print(f"Notice: Plot generation skipped ({e})")


def plot_results(agg: pd.DataFrame, per_series: pd.DataFrame):
    """Plot WAPE and MASE comparisons and save to file."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # WAPE Bar Chart
    models = agg["model"].tolist()
    wape_vals = agg["WAPE"].tolist()
    mase_vals = agg["MASE"].tolist()
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"][:len(models)]

    axes[0].bar(models, wape_vals, color=colors, alpha=0.85, edgecolor="black")
    axes[0].set_title("Overall WAPE (Lower is Better)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("WAPE")
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)
    axes[0].tick_params(axis="x", rotation=25)

    # MASE Bar Chart
    axes[1].bar(models, mase_vals, color=colors, alpha=0.85, edgecolor="black")
    axes[1].axhline(1.0, color="red", linestyle=":", linewidth=1.5, label="Seasonal Naive (1.0)")
    axes[1].set_title("Overall MASE (Lower is Better, < 1 beats Seasonal Naive)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("MASE")
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend()

    plt.tight_layout()
    chart_path = "./results/benchmark_comparison.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"Saved benchmark visualization to {chart_path}")


if __name__ == "__main__":
    main()
