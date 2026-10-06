"""Unit tests for the TSFM demand benchmark suite."""
import numpy as np
import pandas as pd
import pytest

from utils import (
    get_device,
    interval_coverage,
    mase,
    rmse,
    tag_segment,
    wape,
    winkler_score,
    RuntimeTracker,
)


def test_device_detection():
    device = get_device()
    assert device in ["cuda", "mps", "cpu"]


def test_wape_standard():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 30.0])
    # Total error: |10-12| + |20-18| + |30-30| = 2 + 2 + 0 = 4. Denom = 60. WAPE = 4/60 = 0.0667
    assert np.isclose(wape(y_true, y_pred), 4.0 / 60.0)


def test_wape_zero_denom():
    assert np.isnan(wape([0, 0], [1, 2]))


def test_rmse():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([13.0, 16.0])
    # Diff: 3, -4. Squares: 9, 16. Mean: 12.5. Sqrt: sqrt(12.5) ~ 3.5355
    assert np.isclose(rmse(y_true, y_pred), np.sqrt(12.5))


def test_mase_seasonal():
    # Seasonal period s = 7
    y_hist = np.array([10, 12, 14, 11, 13, 20, 25, 11, 13, 15, 12, 14, 21, 26])
    y_true = np.array([10, 13])
    y_pred = np.array([11, 12])
    m = mase(y_true, y_pred, y_hist, seasonality=7)
    assert not np.isnan(m)
    assert m > 0


def test_interval_coverage():
    y_true = np.array([5, 12, 25, 40])
    q_low = np.array([4, 10, 26, 35])   # 25 is outside [26, 45]
    q_high = np.array([8, 15, 45, 50])
    # 5 in [4,8] (yes), 12 in [10,15] (yes), 25 in [26,45] (no), 40 in [35,50] (yes) => 3/4 = 0.75
    assert interval_coverage(y_true, q_low, q_high) == 0.75


def test_winkler_score():
    y_true = np.array([10.0])
    q_low = np.array([8.0])
    q_high = np.array([14.0])
    # 10 is inside [8, 14], width = 6, penalty = 0
    assert winkler_score(y_true, q_low, q_high, alpha=0.2) == 6.0


def test_tag_segment_no_leakage():
    # Construct series with low sales initially, but high burst after cutoff
    dates = pd.date_range("2024-01-01", periods=10)
    cutoff = pd.to_datetime("2024-01-05")
    df = pd.DataFrame({
        "unique_id": ["item_1"] * 10,
        "ds": dates,
        "y": [0, 0, 1, 0, 0, 100, 200, 300, 400, 500]  # burst occurs after cutoff
    })
    tagged = tag_segment(df, cutoff=cutoff)
    # Strictly prior to cutoff, mean < 5 and zero_frac > 0.5 -> intermittent
    assert tagged["segment"].iloc[0] == "intermittent"


def test_runtime_tracker():
    with RuntimeTracker(model_name="MockModel", n_series=10) as tracker:
        assert tracker.n_series == 10
    assert tracker.elapsed_sec >= 0
    assert tracker.throughput >= 0

