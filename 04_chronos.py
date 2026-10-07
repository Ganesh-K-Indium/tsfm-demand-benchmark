"""Chronos-2 zero-shot forecasting with predict_df and quantile intervals."""
import os
import pandas as pd
import torch
from chronos import Chronos2Pipeline, BaseChronosPipeline

from config import CONFIG
from utils import HORIZON, RuntimeTracker, get_device, get_rolling_cutoffs

MODEL_NAME = "amazon/chronos-2"


def run_chronos() -> pd.DataFrame:
    os.makedirs(CONFIG.results_dir, exist_ok=True)
    df = pd.read_parquet(CONFIG.data_dir / "m5_subset.parquet")

    cutoffs = get_rolling_cutoffs(df["ds"].max(), HORIZON, CONFIG.n_windows)
    n_series = df["unique_id"].nunique()

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
        print(f"Chronos2Pipeline note ({e}). Falling back to BaseChronosPipeline...")
        pipeline = BaseChronosPipeline.from_pretrained(
            MODEL_NAME,
            device_map=device,
            torch_dtype=dtype,
        )
        is_chronos2 = False

    print(f"Running Chronos inference on {n_series} series across {len(cutoffs)} windows...")
    frames = []
    with RuntimeTracker(model_name="Chronos-2", n_series=n_series * len(cutoffs)) as tracker:
      for cutoff in cutoffs:
        hist = df[df["ds"] <= cutoff].copy()
        future = df[(df["ds"] > cutoff) & (df["ds"] <= cutoff + pd.Timedelta(days=HORIZON))].copy()
        if is_chronos2:
            future_covs = [c for c in ["dow", "month", "is_weekend"] if c in future.columns]
            hist_df = hist[["unique_id", "ds", "y"] + future_covs].sort_values(["unique_id", "ds"])
            future_df = future[["unique_id", "ds"] + future_covs].sort_values(["unique_id", "ds"])

            fcst_df = pipeline.predict_df(
                df=hist_df,
                future_df=future_df if len(future_covs) > 0 else None,
                id_column="unique_id",
                timestamp_column="ds",
                target="y",
                prediction_length=HORIZON,
                quantile_levels=list(CONFIG.quantile_levels),
            )
            out = pd.DataFrame({
                "unique_id": fcst_df["unique_id"],
                "ds": fcst_df["ds"],
                "model": "Chronos-2",
                "y_pred": fcst_df["predictions"].clip(lower=0),
                "q10": (fcst_df["0.1"] if "0.1" in fcst_df.columns else fcst_df[0.1]).clip(lower=0),
                "q90": (fcst_df["0.9"] if "0.9" in fcst_df.columns else fcst_df[0.9]).clip(lower=0),
            })
        else:
            series = {
                uid: g.sort_values("ds")["y"].to_numpy()
                for uid, g in hist.groupby("unique_id")
            }
            uids = sorted(series.keys())
            contexts = [torch.tensor(series[u], dtype=torch.float32) for u in uids]
            quantiles, mean = pipeline.predict_quantiles(
                context=contexts,
                prediction_length=HORIZON,
                quantile_levels=list(CONFIG.quantile_levels),
            )
            dates = pd.date_range(hist["ds"].max() + pd.Timedelta(days=1), periods=HORIZON, freq="D")
            rows = []
            for i, uid in enumerate(uids):
                for j, d in enumerate(dates):
                    rows.append((
                        uid,
                        d,
                        "Chronos-2",
                        max(0.0, float(mean[i, j])),
                        max(0.0, float(quantiles[i, j, 0])),
                        max(0.0, float(quantiles[i, j, 2])),
                    ))
            out = pd.DataFrame(rows, columns=["unique_id", "ds", "model", "y_pred", "q10", "q90"])
        out["cutoff"] = cutoff
        frames.append(out)

    out = pd.concat(frames, ignore_index=True)

    out_path = CONFIG.results_dir / "chronos_forecast.parquet"
    out.to_parquet(out_path)
    print(f"Chronos completed in {tracker.elapsed_sec:.2f}s ({tracker.throughput:.1f} series/s). Saved to {out_path}")
    return out


def main():
    run_chronos()


if __name__ == "__main__":
    main()
