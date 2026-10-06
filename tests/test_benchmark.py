"""Unit tests for the enhanced TSFM demand benchmark suite."""
import numpy as np
import pandas as pd
import pytest

from utils import (
    asymmetric_inventory_loss,
    get_device,
    get_rolling_cutoffs,
    hierarchical_wape,
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
    assert np.isclose(wape(y_true, y_pred), 4.0 / 60.0)


def test_wape_zero_denom():
    assert np.isnan(wape([0, 0], [1, 2]))


def test_rmse():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([13.0, 16.0])
    assert np.isclose(rmse(y_true, y_pred), np.sqrt(12.5))


def test_mase_seasonal():
    y_hist = np.array([10, 12, 14, 11, 13, 20, 25, 11, 13, 15, 12, 14, 21, 26])
    y_true = np.array([10, 13])
    y_pred = np.array([11, 12])
    m = mase(y_true, y_pred, y_hist, seasonality=7)
    assert not np.isnan(m)
    assert m > 0


def test_asymmetric_inventory_loss():
    # True = 10, Pred = 8 -> Understock by 2. Loss = 2 * 3.0 = 6.0
    # True = 10, Pred = 12 -> Overstock by 2. Loss = 2 * 1.0 = 2.0
    y_true = np.array([10.0, 10.0])
    y_pred = np.array([8.0, 12.0])
    loss = asymmetric_inventory_loss(y_true, y_pred, understock_cost=3.0, overstock_cost=1.0)
    assert np.isclose(loss, (6.0 + 2.0) / 2.0)


def test_get_rolling_cutoffs():
    max_d = pd.to_datetime("2024-03-01")
    cutoffs = get_rolling_cutoffs(max_d, horizon=28, n_windows=3)
    assert len(cutoffs) == 3
    assert cutoffs[0] < cutoffs[1] < cutoffs[2]


def test_hierarchical_wape():
    dates = pd.date_range("2024-01-01", periods=2)
    # 2 items in FOODS_1, store CA_1
    fcst = pd.DataFrame({
        "unique_id": ["FOODS_1_001_CA_1", "FOODS_1_001_CA_1", "FOODS_1_002_CA_1", "FOODS_1_002_CA_1"],
        "ds": [dates[0], dates[1], dates[0], dates[1]],
        "model": ["M1", "M1", "M1", "M1"],
        "y_pred": [10.0, 20.0, 5.0, 10.0],
    })
    actuals = pd.DataFrame({
        "unique_id": ["FOODS_1_001_CA_1", "FOODS_1_001_CA_1", "FOODS_1_002_CA_1", "FOODS_1_002_CA_1"],
        "ds": [dates[0], dates[1], dates[0], dates[1]],
        "y": [10.0, 20.0, 5.0, 10.0],
    })
    res = hierarchical_wape(fcst, actuals)
    assert "M1" in res
    assert res["M1"]["Dept_WAPE"] == 0.0
    assert res["M1"]["Store_WAPE"] == 0.0


def test_interval_coverage():
    y_true = np.array([5, 12, 25, 40])
    q_low = np.array([4, 10, 26, 35])
    q_high = np.array([8, 15, 45, 50])
    assert interval_coverage(y_true, q_low, q_high) == 0.75


def test_winkler_score():
    y_true = np.array([10.0])
    q_low = np.array([8.0])
    q_high = np.array([14.0])
    assert winkler_score(y_true, q_low, q_high, alpha=0.2) == 6.0


def test_tag_segment_no_leakage():
    dates = pd.date_range("2024-01-01", periods=10)
    cutoff = pd.to_datetime("2024-01-05")
    df = pd.DataFrame({
        "unique_id": ["item_1"] * 10,
        "ds": dates,
        "y": [0, 0, 1, 0, 0, 100, 200, 300, 400, 500]
    })
    tagged = tag_segment(df, cutoff=cutoff)
    assert tagged["segment"].iloc[0] == "intermittent"


def test_runtime_tracker():
    with RuntimeTracker(model_name="MockModel", n_series=10) as tracker:
        assert tracker.n_series == 10
    assert tracker.elapsed_sec >= 0
    assert tracker.throughput >= 0
