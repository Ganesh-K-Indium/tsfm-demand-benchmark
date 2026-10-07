from datetime import datetime
import json
import os
import time
import numpy as np
import pandas as pd
import psutil
import torch

from config import CONFIG

HORIZON = CONFIG.horizon
SEASONALITY = CONFIG.seasonality
FREQ = "D"


def get_device() -> str:
    """Return the best available hardware accelerator (cuda, mps for Apple Silicon, or cpu)."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_rolling_cutoffs(max_date: pd.Timestamp, horizon: int = HORIZON, n_windows: int = 1) -> list[pd.Timestamp]:
    """Generate rolling-origin cutoff dates for multi-window backtesting."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if n_windows <= 0:
        raise ValueError("n_windows must be positive")
    return [max_date - pd.Timedelta(int(horizon * (i + 1)), unit="D") for i in reversed(range(n_windows))]


class RuntimeTracker:
    """Tracks latency, memory consumption, and series throughput for models."""

    def __init__(self, model_name: str, n_series: int):
        self.model_name = model_name
        self.n_series = n_series
        self.start_time = None
        self.elapsed_sec = 0.0
        self.initial_mem_mb = 0.0
        self.peak_mem_mb = 0.0
        self.device = get_device()

    def __enter__(self):
        process = psutil.Process(os.getpid())
        self.initial_mem_mb = process.memory_info().rss / (1024 * 1024)
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_sec = time.perf_counter() - self.start_time
        process = psutil.Process(os.getpid())
        current_mem = process.memory_info().rss / (1024 * 1024)
        self.peak_mem_mb = max(self.initial_mem_mb, current_mem)
        self.save_profile()

    @property
    def throughput(self) -> float:
        """Series forecasted per second."""
        return self.n_series / self.elapsed_sec if self.elapsed_sec > 0 else 0.0

    @property
    def latency_per_series_ms(self) -> float:
        """Milliseconds elapsed per time series."""
        return (self.elapsed_sec / self.n_series) * 1000 if self.n_series > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "n_series": self.n_series,
            "elapsed_seconds": round(self.elapsed_sec, 3),
            "throughput_series_sec": round(self.throughput, 2),
            "latency_ms_per_series": round(self.latency_per_series_ms, 2),
            "peak_mem_mb": round(self.peak_mem_mb, 1),
            "device": self.device,
        }

    def save_profile(self):
        """Append runtime benchmark stats to results/runtime_profiles.json."""
        os.makedirs(CONFIG.results_dir, exist_ok=True)
        file_path = CONFIG.results_dir / "runtime_profiles.json"
        data = {}
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        data[self.model_name] = self.to_dict()
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)


def wape(y_true, y_pred):
    """Weighted Absolute Percentage Error (WAPE)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom) if denom > 0 else np.nan


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))


def mase(y_true, y_pred, y_hist, seasonality=SEASONALITY):
    """MASE w.r.t. seasonal naive on the history available at forecast time."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    y_hist = np.asarray(y_hist, float)
    if len(y_hist) <= seasonality:
        return np.nan
    d = np.abs(y_hist[seasonality:] - y_hist[:-seasonality]).mean()
    if d == 0:
        return np.nan
    return float(np.abs(y_true - y_pred).mean() / d)


def asymmetric_inventory_loss(y_true, y_pred, understock_cost=CONFIG.understock_cost, overstock_cost=CONFIG.overstock_cost):
    """
    Newsvendor Inventory Loss Function:
    Penalizes stockouts (under-forecasting) at understock_cost ($C_u$) and
    overstocking (excess inventory holding + markdowns) at overstock_cost ($C_o$).
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    understock_err = np.maximum(0.0, y_true - y_pred)
    overstock_err = np.maximum(0.0, y_pred - y_true)
    total_loss = (understock_cost * understock_err) + (overstock_cost * overstock_err)
    return float(np.mean(total_loss))


def interval_coverage(y_true, q_low, q_high):
    """Calculate the empirical coverage of a prediction interval [q_low, q_high]."""
    y_true = np.asarray(y_true, float)
    q_low = np.asarray(q_low, float)
    q_high = np.asarray(q_high, float)
    covered = (y_true >= q_low) & (y_true <= q_high)
    return float(np.mean(covered))


def winkler_score(y_true, q_low, q_high, alpha=0.2):
    """Winkler score for (1 - alpha) prediction interval (e.g. 80% interval: alpha = 0.2)."""
    y_true = np.asarray(y_true, float)
    l = np.asarray(q_low, float)
    u = np.asarray(q_high, float)
    width = u - l
    pen_low = (2.0 / alpha) * (l - y_true) * (y_true < l)
    pen_high = (2.0 / alpha) * (y_true - u) * (y_true > u)
    return float(np.mean(width + pen_low + pen_high))


def tag_segment(df: pd.DataFrame, cutoff=None) -> pd.DataFrame:
    """Segment each series: fast / slow / intermittent without lookahead data leakage."""
    source_df = df if cutoff is None else df[df["ds"] <= cutoff]
    stats = source_df.groupby("unique_id")["y"].agg(
        mean_y="mean", zero_frac=lambda x: (x == 0).mean()
    )

    def seg(row):
        if row.zero_frac > 0.5:
            return "intermittent"
        if row.mean_y < 5:
            return "slow"
        return "fast"

    stats["segment"] = stats.apply(seg, axis=1)
    base_df = df.drop(columns=["segment"], errors="ignore")
    return base_df.merge(stats[["segment"]], on="unique_id", how="left")


def hierarchical_wape(forecast: pd.DataFrame, actuals: pd.DataFrame) -> dict[str, float]:
    """Compute bottom-up aggregate WAPE at Department and Store levels."""
    df = forecast.merge(actuals, on=["unique_id", "ds"], how="inner")
    # unique_id format: "FOODS_3_090_CA_1" -> dept: "FOODS_3", store: "CA_1"
    df["dept"] = df["unique_id"].str.rsplit("_", n=2).str[0]
    df["store"] = df["unique_id"].str.rsplit("_", n=2).str[-1]

    results = {}
    for model, m_df in df.groupby("model"):
        # Department level aggregation
        dept_agg = m_df.groupby(["dept", "ds"])[["y", "y_pred"]].sum()
        dept_w = wape(dept_agg["y"], dept_agg["y_pred"])

        # Store level aggregation
        store_agg = m_df.groupby(["store", "ds"])[["y", "y_pred"]].sum()
        store_w = wape(store_agg["y"], store_agg["y_pred"])

        results[model] = {
            "Dept_WAPE": round(dept_w, 4),
            "Store_WAPE": round(store_w, 4),
        }
    return results


def evaluate(forecast: pd.DataFrame, actuals: pd.DataFrame,
             y_hist_by_id: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate forecasts; WAPE is pooled, while Macro_WAPE equally weights series."""
    join_cols = ["unique_id", "ds"]
    has_windows = "cutoff" in forecast.columns and "cutoff" in actuals.columns
    if has_windows:
        join_cols.append("cutoff")
    df = forecast.merge(actuals, on=join_cols, how="inner")
    rows = []
    has_intervals = "q10" in df.columns and "q90" in df.columns

    group_cols = (["cutoff"] if has_windows else []) + ["unique_id", "model"]
    for keys, g in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        offset = 0
        cutoff = keys[offset] if has_windows else None
        offset += int(has_windows)
        uid, model = keys[offset:offset + 2]
        hist_key = (cutoff, uid) if has_windows else uid
        y_hist = y_hist_by_id.get(hist_key, y_hist_by_id.get(uid, np.array([])))
        res = {
            "unique_id": uid,
            "model": model,
            "WAPE": wape(g["y"], g["y_pred"]),
            "MASE": mase(g["y"], g["y_pred"], y_hist),
            "RMSE": rmse(g["y"], g["y_pred"]),
            "Inventory_Loss": asymmetric_inventory_loss(g["y"], g["y_pred"]),
        }
        if has_windows:
            res["cutoff"] = cutoff
        if has_intervals and g["q10"].notna().all() and g["q90"].notna().all():
            res["Coverage_80"] = interval_coverage(g["y"], g["q10"], g["q90"])
            res["Winkler_80"] = winkler_score(g["y"], g["q10"], g["q90"], alpha=0.2)
        rows.append(res)

    per_series = pd.DataFrame(rows)
    agg_dict = {
        "RMSE": "mean",
        "Inventory_Loss": "mean",
    }
    if "Coverage_80" in per_series.columns:
        agg_dict["Coverage_80"] = "mean"
        agg_dict["Winkler_80"] = "mean"

    agg = per_series.groupby("model").agg(
        **{k: (k, fn) for k, fn in agg_dict.items()},
        Macro_WAPE=("WAPE", "mean"),
        MASE=("MASE", "mean"),
    ).reset_index()
    pooled = {model: wape(g["y"], g["y_pred"]) for model, g in df.groupby("model")}
    agg["WAPE"] = agg["model"].map(pooled)
    agg = agg.round(4)
    return agg, per_series


def log_experiment(agg_df: pd.DataFrame, extra_params: dict | None = None):
    """Log experiment execution to results/experiment_history.json."""
    os.makedirs(CONFIG.results_dir, exist_ok=True)
    history_file = CONFIG.results_dir / "experiment_history.json"
    history = []
    if history_file.exists():
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except Exception:
            history = []

    run_record = {
        "timestamp": datetime.now().isoformat(),
        "device": get_device(),
        "params": extra_params or {},
        "results": agg_df.to_dict(orient="records"),
    }
    history.append(run_record)
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
