"""Load M5, build a balanced subset with segments and business covariates, save to disk."""
import argparse
import os
from pathlib import Path
import pandas as pd
from datasetsforecast.m5 import M5

from config import CONFIG
from utils import tag_segment


def prepare_data(preset: str = "small") -> Path:
    os.makedirs(CONFIG.data_dir, exist_ok=True)
    os.makedirs(CONFIG.results_dir, exist_ok=True)

    preset_cfg = CONFIG.PRESETS.get(preset, CONFIG.PRESETS["small"])
    n_stores = preset_cfg["n_stores"]
    n_items_per_dept = preset_cfg["n_items_per_dept"]

    print(f"Preparing dataset with preset '{preset}' (Stores: {n_stores}, Items/Dept: {n_items_per_dept})...")
    print("Loading raw M5 dataset via datasetsforecast...")
    Y_df, X_df, S_df = M5.load(str(CONFIG.data_dir))   # Y: unique_id, ds, y

    # Filter stores
    stores = [f"CA_{i}" for i in range(1, n_stores + 1)]
    Y_df = Y_df[Y_df["unique_id"].str.endswith(tuple(f"_{s}" for s in stores))].copy()

    # Parse item and store
    Y_df["item"] = Y_df["unique_id"].str.rsplit("_", n=2).str[0]
    Y_df["store"] = Y_df["unique_id"].str.rsplit("_", n=2).str[-1]

    # Sample balanced items per department
    sampled = (
        Y_df.groupby(["item", "store"]).size().reset_index(name="n")
        .sort_values("item")
    )
    keep = (
        sampled.groupby(sampled["item"].str.rsplit("_", n=1).str[0])
        .head(n_items_per_dept)
    )
    keys = set(zip(keep["item"], keep["store"]))
    mask = Y_df.apply(lambda r: (r["item"], r["store"]) in keys, axis=1)
    Y_df = Y_df[mask].drop(columns=["item", "store"])

    # Merge covariates (price, promotions, SNAP)
    X_df = X_df[X_df["unique_id"].isin(Y_df["unique_id"].unique())]
    df = Y_df.merge(X_df, on=["unique_id", "ds"], how="left")

    # Calendar features
    df["ds"] = pd.to_datetime(df["ds"])
    df["dow"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # Tag velocity segment strictly using historical window to eliminate lookahead bias
    cutoff = df["ds"].max() - pd.Timedelta(days=CONFIG.horizon)
    df = tag_segment(df, cutoff=cutoff)

    out_path = CONFIG.data_dir / "m5_subset.parquet"
    df.to_parquet(out_path)
    n_series = df["unique_id"].nunique()
    print(f"Successfully saved {len(df):,} rows across {n_series} series to {out_path}")
    print("Breakdown by demand velocity segment:")
    print(df.groupby("segment")["unique_id"].nunique().to_string())
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Prepare balanced M5 dataset for benchmark.")
    parser.add_argument(
        "--preset",
        choices=list(CONFIG.PRESETS.keys()),
        default="small",
        help="Data sampling scale preset (smoke, small, standard, extended)",
    )
    args = parser.parse_args()
    prepare_data(preset=args.preset)


if __name__ == "__main__":
    main()
