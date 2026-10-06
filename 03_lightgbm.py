"""LightGBM via MLForecast with covariates and business drivers."""
import os
import pandas as pd
import lightgbm as lgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean

from utils import HORIZON


def main():
    os.makedirs("./results", exist_ok=True)
    df = pd.read_parquet("./data/m5_subset.parquet")

    cutoff = df["ds"].max() - pd.Timedelta(days=HORIZON)
    hist = df[df["ds"] <= cutoff].copy()
    future = df[df["ds"] > cutoff].copy()

    # Identify potential exogenous columns available in M5
    candidate_cats = ["segment", "event_name_1", "event_type_1"]
    static_cats = [c for c in candidate_cats if c in df.columns]

    # Convert categoricals to category dtype
    for c in static_cats:
        hist[c] = hist[c].astype("category")
        future[c] = future[c].astype("category")

    # Future known covariates: calendar features + known retail drivers
    future_cov_candidates = ["dow", "month", "is_weekend", "snap_CA", "sell_price"]
    future_covs = [c for c in future_cov_candidates if c in df.columns]

    # Dynamic features for future dataframe
    future_cols = ["unique_id", "ds"] + [c for c in (future_covs + static_cats) if c in future.columns]
    future_df = future[list(set(future_cols))].copy()

    # Configure MLForecast
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

    print("Fitting LightGBM with business drivers & lag features...")
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

    out_path = "./results/lightgbm_forecast.parquet"
    preds.to_parquet(out_path)
    print(f"LightGBM saved to {out_path}")


if __name__ == "__main__":
    main()
