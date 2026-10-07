"""Load M5, build a balanced subset with segments and business covariates, save to disk."""
import argparse
import hashlib
import os
from pathlib import Path
import urllib.request
import zipfile
import pandas as pd
from datasetsforecast.m5 import M5

from config import CONFIG
from utils import tag_segment


M5_ZENODO_URL = "https://zenodo.org/records/12636070/files/m5-forecasting-accuracy.zip?download=1"
M5_ZENODO_MD5 = "86f57416a314197f40a17cc6fc60cbb4"
M5_RAW_FILES = (
    "calendar.csv",
    "sell_prices.csv",
    "sales_train_evaluation.csv",
)


def _download_official_m5(raw_dir: Path) -> None:
    """Fetch and validate the public M5 archive when the Nixtla mirror is unavailable."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir.parent / "m5-forecasting-accuracy.zip"
    partial = archive.with_suffix(archive.suffix + ".part")
    try:
        if not archive.is_file():
            print(f"Downloading M5 archive from Zenodo to {archive}...")
            digest = hashlib.md5()
            with urllib.request.urlopen(M5_ZENODO_URL, timeout=60) as response, partial.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    output.write(chunk)
            if digest.hexdigest() != M5_ZENODO_MD5:
                raise ValueError("Downloaded M5 archive checksum does not match the published Zenodo checksum")
            partial.replace(archive)
        digest = hashlib.md5()
        with archive.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != M5_ZENODO_MD5:
            raise ValueError("Downloaded M5 archive checksum does not match the published Zenodo checksum")
        with zipfile.ZipFile(archive) as archive_file:
            names = archive_file.namelist()
            if any(Path(name).name != name or name.startswith("/") for name in names):
                raise ValueError("M5 archive contains unexpected paths")
            archive_file.extractall(raw_dir)
        missing = [name for name in M5_RAW_FILES if not (raw_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"M5 archive is missing required files: {missing}")
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _load_official_m5(raw_dir: Path, stores: list[str]):
    """Build Nixtla-shaped Y/X/S tables from the original public M5 CSV files."""
    sales_path = raw_dir / "sales_train_evaluation.csv"
    header = pd.read_csv(sales_path, nrows=0).columns.tolist()
    id_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [col for col in header if col.startswith("d_")]
    sales = pd.read_csv(sales_path, usecols=id_cols + day_cols)
    sales = sales[sales["store_id"].isin(stores)].copy()
    if sales.empty:
        raise ValueError(f"No M5 sales found for configured stores: {stores}")
    sales["unique_id"] = sales["item_id"].astype(str) + "_" + sales["store_id"].astype(str)
    static = sales[["unique_id"] + id_cols].drop_duplicates("unique_id")

    # Only melt the requested stores; melting all ten stores first uses much
    # more memory while producing no additional rows for this benchmark.
    long = sales.melt(
        id_vars=["unique_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        value_vars=day_cols,
        var_name="d",
        value_name="y",
    )
    calendar_cols = [
        "d", "date", "wm_yr_wk", "event_name_1", "event_type_1",
        "event_name_2", "event_type_2", "snap_CA",
    ]
    calendar = pd.read_csv(
        raw_dir / "calendar.csv",
        usecols=lambda col: col in calendar_cols,
        parse_dates=["date"],
    )
    long = long.merge(calendar, on="d", how="left", validate="many_to_one")
    prices = pd.read_csv(raw_dir / "sell_prices.csv")
    long = long.merge(
        prices,
        on=["item_id", "store_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )
    long = long.rename(columns={"date": "ds"})
    x_cols = [
        "unique_id", "ds", "sell_price", "event_name_1", "event_type_1",
        "event_name_2", "event_type_2", "snap_CA",
    ]
    x_df = long[x_cols].copy()
    event_cols = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    for col in event_cols:
        x_df[col] = x_df[col].fillna("nan")
    y_df = long[["unique_id", "ds", "y"]].copy()
    return y_df, x_df, static


def _load_m5(stores: list[str]):
    raw_dir = CONFIG.data_dir / "m5" / "datasets"
    cache = raw_dir / "m5.p"
    nixtla_files = [*M5_RAW_FILES, "sales_test_evaluation.csv"]
    if cache.is_file() or all((raw_dir / name).is_file() for name in nixtla_files):
        return M5.load(str(CONFIG.data_dir))

    if not all((raw_dir / name).is_file() for name in M5_RAW_FILES):
        _download_official_m5(raw_dir)
    return _load_official_m5(raw_dir, stores)


def prepare_data(preset: str = "small") -> Path:
    os.makedirs(CONFIG.data_dir, exist_ok=True)
    os.makedirs(CONFIG.results_dir, exist_ok=True)

    preset_cfg = CONFIG.PRESETS.get(preset, CONFIG.PRESETS["small"])
    n_stores = preset_cfg["n_stores"]
    n_items_per_dept = preset_cfg["n_items_per_dept"]

    print(f"Preparing dataset with preset '{preset}' (Stores: {n_stores}, Items/Dept: {n_items_per_dept})...")
    print("Loading M5 data (local cache or verified public archive)...")
    stores = [f"CA_{i}" for i in range(1, n_stores + 1)]
    Y_df, X_df, S_df = _load_m5(stores)   # Y: unique_id, ds, y

    # M5's static table is the authoritative source for item, department, and
    # store identifiers. Keep it separate from the daily sales/covariate tables.
    required_static = {"unique_id", "item_id", "dept_id", "store_id"}
    missing_static = required_static.difference(S_df.columns)
    if missing_static:
        raise ValueError(f"M5 static metadata is missing required columns: {sorted(missing_static)}")
    static_cols = [c for c in ["unique_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"] if c in S_df]
    static = S_df[static_cols].drop_duplicates("unique_id")

    # Filter the static series list before joining it to daily sales. This
    # avoids expanding metadata across the full national M5 panel unnecessarily.
    selected_static = static[static["store_id"].isin(stores)]
    Y_df = Y_df.merge(selected_static, on="unique_id", how="inner", validate="many_to_one")
    if Y_df.empty:
        raise ValueError(f"No M5 sales series found for configured stores: {stores}")
    if Y_df["dept_id"].isna().any():
        raise ValueError("M5 static metadata is missing department information for some series")

    # Draw a reproducible, stratified item sample within every department/store.
    sampled = (
        Y_df[["item_id", "dept_id", "store_id"]].drop_duplicates()
    )
    keep = (
        sampled.groupby(["dept_id", "store_id"], group_keys=False, sort=True)
        .sample(n=n_items_per_dept, replace=False, random_state=42)
    )
    keys = keep[["item_id", "store_id"]].drop_duplicates()
    Y_df = Y_df.merge(keys.assign(_keep=True), on=["item_id", "store_id"], how="inner", validate="many_to_one")
    Y_df = Y_df.drop(columns="_keep")

    # Merge covariates (price, promotions, SNAP)
    X_df = X_df[X_df["unique_id"].isin(Y_df["unique_id"].unique())]
    # Join daily drivers only; metadata already came from S_df above.
    df = Y_df.merge(X_df, on=["unique_id", "ds"], how="left", validate="one_to_one")

    # Calendar features
    df["ds"] = pd.to_datetime(df["ds"])
    # Keep initial zero-sales days and sort every series into daily order.
    # Dropping leading zeros creates a truncated calendar that MLForecast rejects.
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
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
