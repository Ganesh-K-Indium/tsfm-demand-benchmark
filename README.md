# Time Series Foundation Model (TSFM) Demand Benchmark

A modular, reproducible benchmark evaluating state-of-the-art **Time Series Foundation Models** against **Gradient Boosted Trees (LightGBM)** and **classical statistical baselines** on the Walmart M5 retail demand dataset.

---

## 🚀 Models Benchmarked

| Model Category | Specific Implementation | Description |
| :--- | :--- | :--- |
| **Foundation Model (Google)** | **TimesFM 3.0** (`timesfm3`, `google/timesfm-3.0-pytorch`) | Latest Google 330M parameter foundation model with native covariate support, quantiles, and non-negative forecasting. |
| **Foundation Model (Amazon)** | **Chronos-2** (`chronos-forecasting`, `amazon/chronos-2`) | Latest Amazon 120M universal forecasting transformer with cross-learning and `predict_df` API. |
| **Machine Learning (GBDT)** | **LightGBM** (`mlforecast`, `lightgbm`) | Recursive gradient boosting with lag features, rolling windows, and business drivers (prices, promotions, SNAP). |
| **Statistical Baselines** | **AutoETS**, **Seasonal Naive**, **Croston** (`statsforecast`) | Classical benchmark baselines for intermittent, seasonal, and exponential smoothing demand forecasting. |

---

## 📊 Evaluation & Metrics

Standardized against retail demand forecasting best practices across a **28-day horizon**:
- **WAPE (Weighted Absolute Percentage Error):** Scale-independent aggregate volume accuracy.
- **MASE (Mean Absolute Scaled Error):** Accuracy scaled against seasonal naive history ($\text{MASE} < 1.0$ beats seasonal naive).
- **RMSE (Root Mean Squared Error):** Heavily penalizes large stockout errors.
- **Velocity Segmentation:** Breakdown by demand profile (*fast*, *slow*, and *intermittent*).
- **Horizon Bucketing:** Breakdown by horizon (*1–7d*, *8–14d*, and *15–28d*).
- **Probabilistic Intervals:** 80% prediction interval coverage and Winkler score.

---

## 🛠️ Quickstart

### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Execution Pipeline
Run the numbered scripts sequentially:

```bash
# 1. Download M5 dataset and generate balanced subset
python 01_prepare_data.py

# 2. Run classical statistical baselines (SeasonalNaive, AutoETS, Croston)
python 02_baselines.py

# 3. Train and predict with LightGBM (with lag transforms & exogenous drivers)
python 03_lightgbm.py

# 4. Zero-shot inference with Amazon Chronos-2
python 04_chronos.py

# 5. Zero-shot inference with Google TimesFM 3.0
python 05_timesfm.py

# 6. Evaluate all forecasts, generate breakdown tables & visualization
python 06_evaluate.py
```

Results and comparison charts will be exported to the `results/` directory:
- `results/benchmark_summary.csv`
- `results/per_series_summary.csv`
- `results/benchmark_comparison.png`

