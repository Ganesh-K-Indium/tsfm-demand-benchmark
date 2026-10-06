"""Statistical baselines: seasonal naive, AutoETS, Croston (intermittent)."""
import os
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, SeasonalNaive, CrostonOptimized

from utils import HORIZON, SEASONALITY


def main():
    os.makedirs("./results", exist_ok=True)
    df = pd.read_parquet("./data/m5_subset.parquet")
    df = df[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])

    # Split: everything except last 28 days is history
    cutoff = df["ds"].max() - pd.Timedelta(days=HORIZON)
    hist = df[df["ds"] <= cutoff]

    models = [
        SeasonalNaive(season_length=SEASONALITY),
        AutoETS(season_length=SEASONALITY),
        CrostonOptimized(),
    ]
    sf = StatsForecast(models=models, freq="D", n_jobs=-1)

    print("Fitting statistical baselines (SeasonalNaive, AutoETS, Croston)...")
    fcst = sf.forecast(df=hist, h=HORIZON)
    fcst = fcst.rename(columns={
        "SeasonalNaive": "SeasonalNaive",
        "AutoETS": "AutoETS",
        "CrostonOptimized": "Croston",
    })

    # Melt to long format: unique_id, ds, model, y_pred
    long = fcst.melt(
        id_vars=["unique_id", "ds"], var_name="model", value_name="y_pred"
    )
    long["y_pred"] = long["y_pred"].clip(lower=0)

    out_path = "./results/baselines_forecast.parquet"
    long.to_parquet(out_path)
    print(f"Baselines saved to {out_path}: {list(long['model'].unique())}")


if __name__ == "__main__":
    main()
