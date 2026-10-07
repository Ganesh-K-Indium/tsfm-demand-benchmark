"""LightGBM via MLForecast with covariates and business drivers."""
import os
import pandas as pd
import lightgbm as lgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean

from config import CONFIG
from utils import HORIZON, RuntimeTracker, get_rolling_cutoffs


def run_lightgbm() -> pd.DataFrame:
    os.makedirs(CONFIG.results_dir, exist_ok=True)
    df = pd.read_parquet(CONFIG.data_dir / "m5_subset.parquet")

    cutoffs = get_rolling_cutoffs(df["ds"].max(), HORIZON, CONFIG.n_windows)
    n_series = df["unique_id"].nunique()

    # Categoricals
    candidate_cats = ["segment", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    static_cats = [c for c in candidate_cats if c in df.columns]
    # Future known covariates: calendar features + known retail drivers
    future_cov_candidates = [
        "dow", "month", "is_weekend", "snap_CA", "sell_price",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
    ]
    future_covs = [c for c in future_cov_candidates if c in df.columns]
    # Use identical category dictionaries in every rolling window, including
    # event categories that may be absent from a particular training slice.
    for c in set(static_cats + [c for c in future_covs if c.startswith("event_")]):
        df[c] = df[c].astype("category")

    future_cols = ["unique_id", "ds"] + [c for c in (future_covs + static_cats) if c in df.columns]

    lgb_params = {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "random_state": 42,
        "verbosity": -1,
        # The benchmark sample is small; a single worker avoids unnecessary
        # process/thread-pool overhead in local and smoke-test runs.
        "n_jobs": 1,
    }

    fcst = MLForecast(
        models=lgb.LGBMRegressor(**lgb_params),
        freq="D",
        lags=[7, 14, 21, 28],
        lag_transforms={
            7: [RollingMean(window_size=7), RollingMean(window_size=28)],
            28: [RollingMean(window_size=28)],
        },
    )

    print(f"Fitting LightGBM on {n_series} series across {len(cutoffs)} windows...")
    predictions = []
    with RuntimeTracker(model_name="LightGBM", n_series=n_series * len(cutoffs)) as tracker:
        for cutoff in cutoffs:
            hist = df[df["ds"] <= cutoff].copy()
            future = df[(df["ds"] > cutoff) & (df["ds"] <= cutoff + pd.Timedelta(days=HORIZON))].copy()
            future_df = future[future_cols].copy()
            fcst.fit(hist, id_col="unique_id", time_col="ds", target_col="y", static_features=static_cats)
            pred = fcst.predict(h=HORIZON, X_df=future_df).rename(columns={"LGBMRegressor": "y_pred"})
            pred["model"] = "LightGBM"
            pred["cutoff"] = cutoff
            predictions.append(pred[["unique_id", "ds", "model", "y_pred", "cutoff"]])
    preds = pd.concat(predictions, ignore_index=True)
    preds["y_pred"] = preds["y_pred"].clip(lower=0)

    out_path = CONFIG.results_dir / "lightgbm_forecast.parquet"
    preds.to_parquet(out_path)
    print(f"LightGBM completed in {tracker.elapsed_sec:.2f}s ({tracker.throughput:.1f} series/s). Saved to {out_path}")
    return preds


def main():
    run_lightgbm()


if __name__ == "__main__":
    main()
