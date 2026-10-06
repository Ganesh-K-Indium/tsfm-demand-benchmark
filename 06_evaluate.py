"""Comprehensive evaluation, visualization, hierarchical aggregation, and automated report generation."""
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CONFIG
from utils import HORIZON, evaluate, hierarchical_wape, log_experiment, wape


def run_evaluation():
    data_path = CONFIG.data_dir / "m5_subset.parquet"
    if not data_path.exists():
        print(f"Error: {data_path} not found. Run 01_prepare_data.py first.")
        return None

    df = pd.read_parquet(data_path)
    cutoff = df["ds"].max() - pd.Timedelta(days=HORIZON)
    actuals = df[df["ds"] > cutoff][["unique_id", "ds", "y"]].copy()

    forecast_files = [
        (CONFIG.results_dir / "baselines_forecast.parquet", "Baselines"),
        (CONFIG.results_dir / "lightgbm_forecast.parquet", "LightGBM"),
        (CONFIG.results_dir / "chronos_forecast.parquet", "Chronos-2"),
        (CONFIG.results_dir / "timesfm_forecast.parquet", "TimesFM-3"),
    ]

    loaded_dfs = []
    for f_path, label in forecast_files:
        if f_path.exists():
            f_df = pd.read_parquet(f_path)
            loaded_dfs.append(f_df)
            print(f"Loaded {label}: {f_df['model'].unique().tolist()}")

    if not loaded_dfs:
        print("No forecast files found in results/. Run models first.")
        return None

    fcsts = pd.concat(loaded_dfs, ignore_index=True)

    # Train history dictionary for MASE calculation
    y_hist = {
        uid: g.sort_values("ds")["y"].to_numpy()
        for uid, g in df[df["ds"] <= cutoff].groupby("unique_id")
    }

    agg, per_series = evaluate(fcsts, actuals, y_hist)

    # Join velocity segments
    seg = df.groupby("unique_id")["segment"].first().reset_index()
    per_series = per_series.merge(seg, on="unique_id", how="left")

    # Load runtime profiles if present
    runtime_df = load_runtime_profiles()
    if runtime_df is not None:
        agg = agg.merge(runtime_df, on="model", how="left")

    # Hierarchical Aggregation (Department and Store WAPE)
    hier_results = hierarchical_wape(fcsts, actuals)
    hier_df = pd.DataFrame.from_dict(hier_results, orient="index").reset_index().rename(columns={"index": "model"})
    agg = agg.merge(hier_df, on="model", how="left")

    print("\n" + "=" * 70)
    print("=== OVERALL BENCHMARK RESULTS (WITH INVENTORY & HIERARCHY) ===")
    print("=" * 70)
    print(agg.to_string(index=False))

    print("\n" + "=" * 70)
    print("=== BREAKDOWN BY DEMAND VELOCITY SEGMENT ===")
    print("=" * 70)
    seg_summary = (
        per_series.groupby(["segment", "model"])[["WAPE", "MASE", "Inventory_Loss"]]
        .mean()
        .round(4)
        .unstack(level=0)
    )
    print(seg_summary.to_string())

    # Save summary tables
    summary_path = CONFIG.results_dir / "benchmark_summary.csv"
    per_series_path = CONFIG.results_dir / "per_series_summary.csv"
    agg.to_csv(summary_path, index=False)
    per_series.to_csv(per_series_path, index=False)

    # Generate charts
    plot_macro_metrics(agg)
    plot_sample_series(df, fcsts, cutoff)

    # Log experiment run to history
    log_experiment(agg, extra_params={"n_series": df["unique_id"].nunique(), "horizon": HORIZON})

    # Generate comprehensive markdown report
    generate_markdown_report(agg, seg_summary, df, fcsts)
    print(f"\nSaved updated benchmark report to {CONFIG.results_dir / 'BENCHMARK_REPORT.md'}")
    return agg


def load_runtime_profiles() -> pd.DataFrame | None:
    profile_path = CONFIG.results_dir / "runtime_profiles.json"
    if not profile_path.exists():
        return None
    try:
        with open(profile_path, "r") as f:
            data = json.load(f)
        records = []
        for model_key, stats in data.items():
            records.append({
                "model": stats["model"],
                "elapsed_sec": stats["elapsed_seconds"],
                "throughput_s_sec": stats["throughput_series_sec"],
                "latency_ms": stats["latency_ms_per_series"],
                "peak_mem_mb": stats["peak_mem_mb"],
            })
        return pd.DataFrame(records)
    except Exception:
        return None


def plot_macro_metrics(agg: pd.DataFrame):
    """Plot WAPE, MASE, Inventory Loss, and Throughput comparisons."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    models = agg["model"].tolist()
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"][:len(models)]

    # 1. WAPE
    axes[0].bar(models, agg["WAPE"], color=colors, alpha=0.85, edgecolor="black")
    axes[0].set_title("Overall WAPE (Lower is Better)", fontweight="bold")
    axes[0].set_ylabel("WAPE")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].tick_params(axis="x", rotation=25)

    # 2. MASE
    axes[1].bar(models, agg["MASE"], color=colors, alpha=0.85, edgecolor="black")
    axes[1].axhline(1.0, color="red", linestyle=":", linewidth=1.5, label="Seasonal Naive (1.0)")
    axes[1].set_title("Overall MASE (<1 beats Baseline)", fontweight="bold")
    axes[1].set_ylabel("MASE")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend()

    # 3. Asymmetric Inventory Loss
    if "Inventory_Loss" in agg.columns:
        axes[2].bar(models, agg["Inventory_Loss"], color=colors, alpha=0.85, edgecolor="black")
        axes[2].set_title("Inventory Loss (Cu=3, Co=1)", fontweight="bold")
        axes[2].set_ylabel("Cost Units / Item")
        axes[2].grid(axis="y", linestyle="--", alpha=0.5)
        axes[2].tick_params(axis="x", rotation=25)

    # 4. Throughput
    if "throughput_s_sec" in agg.columns and agg["throughput_s_sec"].notna().any():
        axes[3].bar(models, agg["throughput_s_sec"], color=colors, alpha=0.85, edgecolor="black")
        axes[3].set_title("Throughput (Higher is Better)", fontweight="bold")
        axes[3].set_ylabel("Series / Sec")
        axes[3].grid(axis="y", linestyle="--", alpha=0.5)
        axes[3].tick_params(axis="x", rotation=25)

    plt.tight_layout()
    chart_path = CONFIG.results_dir / "benchmark_comparison.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()


def plot_sample_series(df: pd.DataFrame, fcsts: pd.DataFrame, cutoff: pd.Timestamp):
    """Plot sample series predictions with shaded intervals across segments."""
    segments = df["segment"].unique()
    n_segs = len(segments)
    fig, axes = plt.subplots(n_segs, 1, figsize=(14, 4 * n_segs), sharex=False)
    if n_segs == 1:
        axes = [axes]

    hist_start = cutoff - pd.Timedelta(days=56)

    for ax, seg_name in zip(axes, segments):
        sample_uids = df[df["segment"] == seg_name]["unique_id"].unique()
        if len(sample_uids) == 0:
            continue
        uid = sample_uids[0]

        series_df = df[(df["unique_id"] == uid) & (df["ds"] >= hist_start)]
        ax.plot(series_df["ds"], series_df["y"], label="Actual Sales", color="black", linewidth=2.0)
        ax.axvline(cutoff, color="gray", linestyle="--", alpha=0.7, label="Forecast Cutoff")

        fcst_sub = fcsts[fcsts["unique_id"] == uid]
        palette = {"SeasonalNaive": "#999999", "AutoETS": "#e6ab02", "LightGBM": "#2ca02c", "Chronos-2": "#1f77b4", "TimesFM-3": "#d62728"}

        for model_name, m_df in fcst_sub.groupby("model"):
            m_df = m_df.sort_values("ds")
            color = palette.get(model_name, None)
            ax.plot(m_df["ds"], m_df["y_pred"], label=f"{model_name}", linewidth=1.8, color=color)

            if "q10" in m_df.columns and "q90" in m_df.columns and m_df["q10"].notna().any():
                ax.fill_between(m_df["ds"], m_df["q10"], m_df["q90"], alpha=0.15, color=color)

        ax.set_title(f"Segment: {seg_name.upper()} ({uid})", fontweight="bold", fontsize=12)
        ax.set_ylabel("Units Sold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    sample_path = CONFIG.results_dir / "sample_series_forecasts.png"
    plt.savefig(sample_path, dpi=200)
    plt.close()


def generate_markdown_report(agg: pd.DataFrame, seg_summary: pd.DataFrame, df: pd.DataFrame, fcsts: pd.DataFrame):
    """Generate a clean Markdown benchmark report."""
    report_path = CONFIG.results_dir / "BENCHMARK_REPORT.md"
    n_series = df["unique_id"].nunique()
    models = fcsts["model"].unique().tolist()

    report = f"""# 📈 Time Series Demand Forecasting Benchmark Report

Generated automatically by `tsfm-demand-benchmark`.

- **Dataset:** Walmart M5 retail sales
- **Number of Series:** {n_series}
- **Forecast Horizon:** {HORIZON} days
- **Models Evaluated:** {", ".join(models)}

---

## 1. Overall Performance Leaderboard

| Model | WAPE | MASE | RMSE | Inv. Loss (Cu=3, Co=1) | Dept WAPE | Store WAPE | Throughput (s/s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for _, r in agg.iterrows():
        inv = f"{r['Inventory_Loss']:.3f}" if "Inventory_Loss" in r and pd.notna(r["Inventory_Loss"]) else "N/A"
        dept = f"{r['Dept_WAPE']:.4f}" if "Dept_WAPE" in r and pd.notna(r["Dept_WAPE"]) else "N/A"
        store = f"{r['Store_WAPE']:.4f}" if "Store_WAPE" in r and pd.notna(r["Store_WAPE"]) else "N/A"
        tp = f"{r['throughput_s_sec']:.1f}" if "throughput_s_sec" in r and pd.notna(r["throughput_s_sec"]) else "N/A"
        report += f"| **{r['model']}** | {r['WAPE']:.4f} | {r['MASE']:.4f} | {r['RMSE']:.4f} | {inv} | {dept} | {store} | {tp} |\n"

    report += """
> [!TIP]
> - **WAPE & MASE:** Measure point forecast precision.
> - **Inventory Loss:** Asymmetric Newsvendor cost penalizing stockouts 3x more heavily than overstock.
> - **Dept & Store WAPE:** Measures whether individual item forecasts sum coherently at aggregate levels.

---

## 2. Macro Performance & Efficiency Charts

![Benchmark Comparison](benchmark_comparison.png)

---

## 3. Sample Forecasts with Uncertainty Intervals

![Sample Series Forecasts](sample_series_forecasts.png)

---

## 4. Performance Breakdown by Velocity Segment

```
""" + seg_summary.to_string() + """
```
"""
    with open(report_path, "w") as f:
        f.write(report)


def main():
    run_evaluation()


if __name__ == "__main__":
    main()
