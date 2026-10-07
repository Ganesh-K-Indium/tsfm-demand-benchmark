"""Statistical baselines: seasonal naive, AutoETS, Croston (intermittent)."""
import os
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, SeasonalNaive, CrostonOptimized

from config import CONFIG
from utils import HORIZON, SEASONALITY, RuntimeTracker, get_rolling_cutoffs


def run_baselines() -> pd.DataFrame:
    os.makedirs(CONFIG.results_dir, exist_ok=True)
    df = pd.read_parquet(CONFIG.data_dir / "m5_subset.parquet")
    df = df[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])

    cutoffs = get_rolling_cutoffs(df["ds"].max(), HORIZON, CONFIG.n_windows)
    n_series = df["unique_id"].nunique()

    models = [
        SeasonalNaive(season_length=SEASONALITY),
        AutoETS(season_length=SEASONALITY),
        CrostonOptimized(),
    ]
    # The smoke preset has very few series; avoid spawning a full loky pool.
    # This also keeps local runs reliable on Python builds with limited IPC support.
    sf = StatsForecast(models=models, freq="D", n_jobs=1)

    print(f"Fitting statistical baselines on {n_series} series across {len(cutoffs)} windows...")
    frames = []
    with RuntimeTracker(model_name="Baselines", n_series=n_series * len(cutoffs)) as tracker:
        for cutoff in cutoffs:
            hist = df[df["ds"] <= cutoff]
            fcst = sf.forecast(df=hist, h=HORIZON)
            long_window = fcst.melt(id_vars=["unique_id", "ds"], var_name="model", value_name="y_pred")
            long_window["model"] = long_window["model"].replace({"CrostonOptimized": "Croston"})
            long_window["cutoff"] = cutoff
            frames.append(long_window)
    long = pd.concat(frames, ignore_index=True)
    long["y_pred"] = long["y_pred"].clip(lower=0)

    out_path = CONFIG.results_dir / "baselines_forecast.parquet"
    long.to_parquet(out_path)
    print(f"Baselines completed in {tracker.elapsed_sec:.2f}s ({tracker.throughput:.1f} series/s). Saved to {out_path}")
    return long


def main():
    run_baselines()


if __name__ == "__main__":
    main()
