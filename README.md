# Time Series Foundation Model (TSFM) Demand Benchmark

A modular, reproducible, publication-grade benchmark evaluating state-of-the-art **Time Series Foundation Models** against **Gradient Boosted Trees (LightGBM)** and **classical statistical baselines** on the Walmart M5 retail demand dataset.

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
- **Point Accuracy:** WAPE (volume error), MASE (relative to seasonal naive), RMSE (squared error penalty).
- **Asymmetric Inventory Loss (Newsvendor Cost):** Penalizes stockouts ($C_u = 3.0$) $3\times$ higher than overstocking ($C_o = 1.0$) to measure real supply chain dollar impact.
- **Hierarchical Coherence:** Aggregates item forecasts to Department and Store levels to measure multi-level error propagation.
- **Probabilistic Calibration:** 80% prediction interval coverage rate and Winkler score.
- **Efficiency & Resource Footprint:** Throughput (series/sec), P95 latency (ms/series), and peak RAM memory.
- **Velocity Segmentation:** Breakdown by demand velocity (*Fast*, *Slow*, and *Intermittent* zero-inflated series).
- **Horizon Bucketing:** Breakdown by horizon (*1–7d*, *8–14d*, and *15–28d*).

---

## 🛠️ Quickstart

### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run with Unified CLI Orchestrator
Execute the complete benchmark with a single command:

```bash
# Fast smoke test (~14 series, validates end-to-end pipeline in seconds)
python run_benchmark.py --preset smoke

# Standard benchmark (~168 series across all product departments)
python run_benchmark.py --preset standard

# Customize supply chain inventory loss penalties
python run_benchmark.py --understock-cost 4.0 --overstock-cost 1.0

# Multi-window rolling-origin backtesting (2 rolling cutoffs)
python run_benchmark.py --n-windows 2

# Benchmark specific models (e.g. LightGBM vs TimesFM 3.0)
python run_benchmark.py --models lightgbm,timesfm --skip-prep

# Re-evaluate existing results and regenerate charts
python run_benchmark.py --evaluate-only
```

### 3. Run Unit Tests
```bash
pytest
```

---

## 📈 Generated Artifacts & Reports

After execution, all metrics and visualizations are saved to `./results/`:
- **`results/BENCHMARK_REPORT.md`**: Publication-ready Markdown summary with performance and efficiency tables.
- **`results/benchmark_comparison.png`**: Macro comparison bar charts (WAPE, MASE, Inventory Loss, and Series Throughput).
- **`results/sample_series_forecasts.png`**: Time-series charts comparing actual sales with model predictions and shaded 10%–90% prediction intervals across Fast, Slow, and Intermittent demand segments.
- **`results/experiment_history.json`**: Historical record of benchmark runs, parameters, and results.
- **`results/benchmark_summary.csv`** & **`results/runtime_profiles.json`**: Raw metrics for downstream analysis.
