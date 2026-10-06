"""Chronos foundation model zero-shot forecasting (Chronos-2 / Chronos-Bolt)."""
import os
import pandas as pd
import torch
from chronos import Chronos2Pipeline, BaseChronosPipeline

from utils import HORIZON, get_device

MODEL_NAME = "amazon/chronos-2"


def main():
    os.makedirs("./results", exist_ok=True)
    df = pd.read_parquet("./data/m5_subset.parquet")

    cutoff = df["ds"].max() - pd.Timedelta(days=HORIZON)
    hist = df[df["ds"] <= cutoff].copy()
    future = df[df["ds"] > cutoff].copy()

    device = get_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Loading {MODEL_NAME} on device: {device}...")

    try:
        pipeline = Chronos2Pipeline.from_pretrained(
            MODEL_NAME,
            device_map=device,
            torch_dtype=dtype,
        )
        is_chronos2 = True
    except Exception as e:
        print(f"Notice: Chronos2 initialization had note ({e}). Trying BaseChronosPipeline...")
        pipeline = BaseChronosPipeline.from_pretrained(
            MODEL_NAME,
            device_map=device,
            torch_dtype=dtype,
        )
        is_chronos2 = False

    if is_chronos2:
        print("Forecasting with Chronos-2 predict_df (with quantiles)...")
        # Prepare dataframes
        hist_df = hist[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])
        future_covs = [c for c in ["dow", "month", "is_weekend"] if c in future.columns]
        future_df = future[["unique_id", "ds"] + future_covs].sort_values(["unique_id", "ds"])

        fcst_df = pipeline.predict_df(
            df=hist_df,
            future_df=future_df if len(future_covs) > 0 else None,
            id_column="unique_id",
            timestamp_column="ds",
            target="y",
            prediction_length=HORIZON,
            quantile_levels=[0.1, 0.5, 0.9],
        )

        out = pd.DataFrame({
            "unique_id": fcst_df["unique_id"],
            "ds": fcst_df["ds"],
            "model": "Chronos-2",
            "y_pred": fcst_df["predictions"].clip(lower=0),
            "q10": fcst_df[0.1].clip(lower=0),
            "q90": fcst_df[0.9].clip(lower=0),
        })
    else:
        print("Forecasting with Chronos base pipeline...")
        series = {
            uid: g.sort_values("ds")["y"].to_numpy()
            for uid, g in hist.groupby("unique_id")
        }
        uids = sorted(series.keys())
        contexts = [torch.tensor(series[u], dtype=torch.float32) for u in uids]

        quantiles, mean = pipeline.predict_quantiles(
            context=contexts,
            prediction_length=HORIZON,
            quantile_levels=[0.1, 0.5, 0.9],
        )

        dates = pd.date_range(hist["ds"].max() + pd.Timedelta(days=1), periods=HORIZON, freq="D")
        rows = []
        for i, uid in enumerate(uids):
            for j, d in enumerate(dates):
                rows.append((
                    uid,
                    d,
                    "Chronos",
                    max(0.0, float(mean[i, j])),
                    max(0.0, float(quantiles[i, j, 0])),
                    max(0.0, float(quantiles[i, j, 2])),
                ))
        out = pd.DataFrame(rows, columns=["unique_id", "ds", "model", "y_pred", "q10", "q90"])

    out_path = "./results/chronos_forecast.parquet"
    out.to_parquet(out_path)
    print(f"Chronos forecast saved to {out_path}")


if __name__ == "__main__":
    main()
