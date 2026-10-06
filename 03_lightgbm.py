"""LightGBM via MLForecast with covariates and business drivers."""
import os
import pandas as pd
import lightgbm as lgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean

from config import CONFIG
from utils import HORIZON, RuntimeTracker


def run_lightgbm() -> pd.DataFrame:
    os.makedirs(CONFIG.results_dir, exist_ok=True)
    df = pd.read_parquet(CONFIG.data_dir / "m5_subset.parquet")

    cutoff = df["ds"].max() - pd.Timedelta(days=HORIZON)
    hist = df[df["ds"] <= cutoff].copy()
    future = df[df["ds"] > cutoff].copy()
    n_series = hist["unique_id"].nunique()

    # Categoricals
    candidate_cats = ["segment", "event_name_1", "event_type_1"]
    static_cats = [c for c in candidate_cats if c in df.columns]
    for c in static_cats:
        hist[c] = hist[c].astype("category")
        future[c] = future[c].astype("category")

    # Future known covariates: calendar features + known retail drivers
    future_cov_candidates = ["dow", "month", "is_weekend", "snap_CA", "sell_price"]
    future_covs = [c for c in future_cov_candidates if c in df.columns]

    future_cols = ["unique_id", "ds"] + [c for c in (future_covs + static_cats) if c in future.columns]
    future_df = future[list(set(future_cols))].copy()

    lgb_params = {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    }

    fcst = MLForecast(
        models=lgb.LGBMRegressor(**lgb_params),
        freq="D",
        lags=[7, 14, 21, 28],
        lag_transforms={
            7: [(RollingMean, 7), (RollingMean, 28)],
            28: [(RollingMean, 28)],
        },
        date_features=["dow", "month", "is_weekend"],
    )

    print(f"Fitting LightGBM on {n_series} series with business drivers...")
    with RuntimeTracker(model_name="LightGBM", n_series=n_series) as tracker:
        fcst.fit(
            hist,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            static_features=static_cats,
        )
        preds = fcst.predict(h=HORIZON, X_df=future_df)

    preds = preds.rename(columns={"LGBMRegressor": "y_pred"})
    preds["model"] = "LightGBM"
    preds = preds[["unique_id", "ds", "model", "y_pred"]]
    preds["y_pred"] = preds["y_pred"].clip(lower=0)

    out_path = CONFIG.results_dir / "lightgbm_forecast.parquet"
    preds.to_parquet(out_path)
    print(f"LightGBM completed in {tracker.elapsed_sec:.2f}s ({tracker.throughput:.1f} series/s). Saved to {out_path}")
    return preds


def main():
    run_lightgbm()


if __name__ == "__main__":
    main()
