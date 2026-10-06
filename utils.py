import numpy as np
import pandas as pd
import torch

HORIZON = 28          # forecast horizon (M5 standard)
SEASONALITY = 7       # weekly seasonality for daily data
FREQ = "D"


def get_device() -> str:
    """Return the best available hardware accelerator (cuda, mps for Apple Silicon, or cpu)."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def wape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom) if denom > 0 else np.nan


def rmse(y_true, y_pred):
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


def interval_coverage(y_true, q_low, q_high):
    """Calculate the empirical coverage of a prediction interval [q_low, q_high]."""
    y_true = np.asarray(y_true, float)
    q_low = np.asarray(q_low, float)
    q_high = np.asarray(q_high, float)
    covered = (y_true >= q_low) & (y_true <= q_high)
    return float(np.mean(covered))


def winkler_score(y_true, q_low, q_high, alpha=0.2):
    """
    Winkler score for (1 - alpha) prediction interval (e.g. 80% interval: alpha = 0.2).
    Penalizes width of interval and deviations when true value falls outside.
    """
    y_true = np.asarray(y_true, float)
    l = np.asarray(q_low, float)
    u = np.asarray(q_high, float)
    width = u - l
    pen_low = (2.0 / alpha) * (l - y_true) * (y_true < l)
    pen_high = (2.0 / alpha) * (y_true - u) * (y_true > u)
    return float(np.mean(width + pen_low + pen_high))


def tag_segment(df: pd.DataFrame, cutoff=None) -> pd.DataFrame:
    """
    Segment each series: fast / slow / intermittent.
    If cutoff is provided, segmentation metrics are calculated strictly on data prior to cutoff
    to prevent lookahead data leakage.
    """
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
    # Remove existing segment column if present before merging
    base_df = df.drop(columns=["segment"], errors="ignore")
    return base_df.merge(stats[["segment"]], on="unique_id", how="left")


def evaluate(forecast: pd.DataFrame, actuals: pd.DataFrame,
             y_hist_by_id: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    forecast : columns [unique_id, ds, model, y_pred, (optional: q10, q90)]
    actuals  : columns [unique_id, ds, y]
    y_hist_by_id : dict unique_id -> 1D array of train history (ds <= cutoff)
    """
    df = forecast.merge(actuals, on=["unique_id", "ds"], how="inner")
    rows = []
    has_intervals = "q10" in df.columns and "q90" in df.columns

    for (uid, model), g in df.groupby(["unique_id", "model"]):
        y_hist = y_hist_by_id.get(uid, np.array([]))
        res = {
            "unique_id": uid,
            "model": model,
            "WAPE": wape(g["y"], g["y_pred"]),
            "MASE": mase(g["y"], g["y_pred"], y_hist),
            "RMSE": rmse(g["y"], g["y_pred"]),
        }
        if has_intervals and g["q10"].notna().all() and g["q90"].notna().all():
            res["Coverage_80"] = interval_coverage(g["y"], g["q10"], g["q90"])
            res["Winkler_80"] = winkler_score(g["y"], g["q10"], g["q90"], alpha=0.2)
        rows.append(res)

    per_series = pd.DataFrame(rows)
    agg_dict = {"WAPE": "mean", "MASE": "mean", "RMSE": "mean"}
    if "Coverage_80" in per_series.columns:
        agg_dict["Coverage_80"] = "mean"
        agg_dict["Winkler_80"] = "mean"

    agg = (
        per_series.groupby("model")
        .agg(**{k: (k, fn) for k, fn in agg_dict.items()})
        .round(4)
        .reset_index()
    )
    return agg, per_series
