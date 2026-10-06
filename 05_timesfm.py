"""TimesFM 3.0 zero-shot forecasting (google/timesfm-3.0-pytorch)."""
import os
import numpy as np
import pandas as pd
from timesfm3 import TimesFM3Forecaster

from config import CONFIG
from utils import HORIZON, RuntimeTracker, get_device

MODEL_NAME = "google/timesfm-3.0-pytorch"


def run_timesfm() -> pd.DataFrame:
    os.makedirs(CONFIG.results_dir, exist_ok=True)
    df = pd.read_parquet(CONFIG.data_dir / "m5_subset.parquet")

    cutoff = df["ds"].max() - pd.Timedelta(days=HORIZON)
    hist = df[df["ds"] <= cutoff].copy()
    n_series = hist["unique_id"].nunique()

    device = get_device()
    print(f"Loading TimesFM 3.0 ({MODEL_NAME}) on device: {device}...")

    forecaster = TimesFM3Forecaster.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME,
        device=device,
    )

    series = {
        uid: g.sort_values("ds")["y"].to_numpy().astype(np.float32)
        for uid, g in hist.groupby("unique_id")
    }
    uids = sorted(series.keys())
    contexts = [series[uid] for uid in uids]

    print(f"Running TimesFM 3.0 inference on {len(uids)} series...")
    with RuntimeTracker(model_name="TimesFM-3", n_series=n_series) as tracker:
        forecast_outputs = list(
            forecaster.predict_batch(
                contexts=contexts,
                horizon=HORIZON,
                ts_ids=uids,
                return_quantiles=True,
                make_positive=True,
            )
        )

    dates = pd.date_range(hist["ds"].max() + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    rows = []

    for f_out in forecast_outputs:
        uid = f_out.ts_id
        point_fcst = f_out.forecast
        quantiles = f_out.quantiles  # shape: (horizon, 9) corresponding to [0.1 .. 0.9]

        for j, d in enumerate(dates):
            y_pred = float(point_fcst[j]) if point_fcst is not None else float(quantiles[j, 4])
            q10 = float(quantiles[j, 0]) if quantiles is not None else np.nan
            q90 = float(quantiles[j, 8]) if quantiles is not None else np.nan
            rows.append((uid, d, "TimesFM-3", y_pred, q10, q90))

    out = pd.DataFrame(rows, columns=["unique_id", "ds", "model", "y_pred", "q10", "q90"])
    out_path = CONFIG.results_dir / "timesfm_forecast.parquet"
    out.to_parquet(out_path)
    print(f"TimesFM 3.0 completed in {tracker.elapsed_sec:.2f}s ({tracker.throughput:.1f} series/s). Saved to {out_path}")
    return out


def main():
    run_timesfm()


if __name__ == "__main__":
    main()
