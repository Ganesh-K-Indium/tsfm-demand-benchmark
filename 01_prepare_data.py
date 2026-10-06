"""Load M5, build a manageable subset with segments and covariates, save to disk."""
import os
import pandas as pd
from datasetsforecast.m5 import M5

from utils import HORIZON, tag_segment

# --- Config: keep the experiment tractable -------------------------------
N_STORES = 2          # CA_1, CA_2
N_ITEMS_PER_DEPT = 4  # sampled items per department
VAL_DAYS = HORIZON    # 28-day holdout window


def main():
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./results", exist_ok=True)

    print("Loading M5 dataset...")
    Y_df, X_df, S_df = M5.load("./data")   # Y: unique_id, ds, y

    # Filter stores + sample items per department
    stores = [f"CA_{i}" for i in range(1, N_STORES + 1)]
    Y_df = Y_df[Y_df["unique_id"].str.endswith(tuple(f"_{s}" for s in stores))].copy()

    # unique_id in M5 is like "FOODS_3_090_CA_1"
    Y_df["item"] = Y_df["unique_id"].str.rsplit("_", n=2).str[0]
    Y_df["store"] = Y_df["unique_id"].str.rsplit("_", n=2).str[-1]

    sampled = (
        Y_df.groupby(["item", "store"]).size().reset_index(name="n")
        .sort_values("item")
    )
    keep = (
        sampled.groupby(sampled["item"].str.rsplit("_", n=1).str[0])
        .head(N_ITEMS_PER_DEPT)
    )
    keys = set(zip(keep["item"], keep["store"]))
    mask = Y_df.apply(lambda r: (r["item"], r["store"]) in keys, axis=1)
    Y_df = Y_df[mask].drop(columns=["item", "store"])

    # Merge covariates (price, events) — X_df has unique_id, ds, sell_price, etc.
    X_df = X_df[X_df["unique_id"].isin(Y_df["unique_id"].unique())]
    df = Y_df.merge(X_df, on=["unique_id", "ds"], how="left")

    # Calendar features from date
    df["ds"] = pd.to_datetime(df["ds"])
    df["dow"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # Calculate segment strictly using historical window to prevent data leakage
    cutoff = df["ds"].max() - pd.Timedelta(days=VAL_DAYS)
    df = tag_segment(df, cutoff=cutoff)

    output_path = "./data/m5_subset.parquet"
    df.to_parquet(output_path)
    n_series = df["unique_id"].nunique()
    print(f"Saved {len(df):,} rows, {n_series} series to {output_path}")
    print("Series breakdown by velocity segment:")
    print(df.groupby("segment")["unique_id"].nunique())


if __name__ == "__main__":
    main()
