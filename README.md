<div align="center">

# 📊 Time Series Foundation Model (TSFM) Demand Benchmark

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Hardware-Apple%20Silicon%20(MPS)%20%7C%20NVIDIA%20(CUDA)-76B900?logo=nvidia&logoColor=white)](https://developer.apple.com/metal/pytorch/)
[![Tests](https://img.shields.io/badge/Tests-12%2F12%20Passing-brightgreen?logo=pytest&logoColor=white)](tests/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**A reproducible, publication-grade benchmark evaluating state-of-the-art Time Series Foundation Models against Gradient Boosted Trees (LightGBM) and classical statistical baselines on the Walmart M5 demand dataset.**

[Overview](#-overview) • [Pipeline](#-benchmark-pipeline) • [Models](#-models-benchmarked) • [Quickstart](#-quickstart) • [Metrics](#-evaluation-framework) • [CLI Reference](#-cli-options)

---

</div>

## 🔍 Overview

Can zero-shot **Time Series Foundation Models (TSFMs)** replace heavily tuned **Gradient Boosted Trees (LightGBM)** in real-world retail demand forecasting?

This benchmark evaluates modern foundation models (**Google TimesFM 3.0**, **Amazon Chronos-2**) against production-standard GBDT and classical statistical baselines on Walmart M5 retail sales. It tests not just raw point accuracy, but **supply chain inventory cost**, **multi-level hierarchical aggregation**, **probabilistic uncertainty**, and **compute throughput**.

---

## 🏗️ Benchmark Pipeline

```mermaid
flowchart LR
    A["Walmart M5 Data\n(Sales, Prices, Events)"] --> B["01_prepare_data.py\n(Stratified Sampling & Lags)"]
    
    B --> C["02_baselines.py\n(AutoETS, S.Naive, Croston)"]
    B --> D["03_lightgbm.py\n(GBDT + Business Drivers)"]
    B --> E["04_chronos.py\n(Amazon Chronos-2)"]
    B --> F["05_timesfm.py\n(Google TimesFM 3.0)"]
    
    C & D & E & F --> G["06_evaluate.py\n(Accuracy, Inventory Loss, Visuals)"]
    G --> H["results/{dataset}/BENCHMARK_REPORT.md\n& Comparison Charts"]
```

---

## 🚀 Models Benchmarked

| Model Category | Architecture | Package & Checkpoint | Key Features |
| :--- | :--- | :--- | :--- |
| **Google Foundation Model** | **TimesFM 3.0** | `timesfm3`<br>`google/timesfm-3.0-pytorch` | 330M decoder-only transformer, native future covariates, non-negative projection (`make_positive=True`), direct quantile generation. |
| **Amazon Foundation Model** | **Chronos-2** | `chronos-forecasting`<br>`amazon/chronos-2` | 120M universal forecasting transformer, group attention cross-learning, direct DataFrame ingestion (`predict_df`). |
| **Gradient Boosted Tree** | **LightGBM** | `mlforecast`<br>`lightgbm` | Recursive GBDT with promotional calendar (`event_name_1`), SNAP flags, price dynamics, and rolling window aggregations. |
| **Statistical Baselines** | **AutoETS / S.Naive / Croston** | `statsforecast` | Classical benchmarks for seasonality, exponential smoothing, and intermittent zero-inflated demand. |

---

## ⚡ Quickstart

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/your-username/tsfm-demand-benchmark.git
cd tsfm-demand-benchmark

# Create virtual environment & activate
python3 -m venv .venv
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### 2. Run the Benchmark with One Command
Execute the full benchmark using the unified CLI orchestrator:

```bash
# 🧪 Ultra-fast smoke test (~14 series, end-to-end pipeline validation in 10s)
python run_benchmark.py --preset smoke

# 📊 Standard benchmark (~168 series across all product categories)
python run_benchmark.py --preset standard

# 🔌 Consumer-electronics-adjacent tech-gadget dataset (44 weekly SKU series)
python run_benchmark.py --dataset tech_gadget

# 🎯 Isolate specific models (e.g. LightGBM vs TimesFM 3.0)
python run_benchmark.py --models lightgbm,timesfm --skip-prep

# 📈 Re-evaluate existing forecasts without re-running models
python run_benchmark.py --evaluate-only
```

### 3. Run Unit Tests
Ensure system integrity and mathematical correctness:
```bash
pytest
```

---

## 📊 Evaluation Framework

Evaluation uses a 28-day daily horizon for M5 and a 13-week horizon for the tech-gadget dataset:

### **1. Accuracy & Volume Metrics**
* **WAPE (Weighted Absolute Percentage Error):** Volume-weighted percentage error across all SKUs.
* **MASE (Mean Absolute Scaled Error):** Accuracy scaled relative to in-sample seasonal naive baseline ($\text{MASE} < 1.0$ indicates outperformance).
* **RMSE (Root Mean Squared Error):** Heavily penalizes large variance and severe stockout errors.

### **2. Supply Chain Financial Impact**
* **Asymmetric Inventory Loss (Newsvendor Cost):**
  $$\mathcal{L}(y, \hat{y}) = C_u \cdot \max(0, y - \hat{y}) + C_o \cdot \max(0, \hat{y} - y)$$
  Stockout / lost margin penalty ($C_u = \$3.0$) is weighted $3\times$ higher than overstock holding/markdown loss ($C_o = \$1.0$).

### **3. Hierarchical Coherence**
* **Hierarchy WAPE:** Aggregates M5 SKU forecasts to department/store levels and gadget forecasts to functionality/vendor levels.

### **4. Operational Efficiency & Compute**
* **Throughput:** Time series forecasted per second.
* **Latency:** Milliseconds elapsed per series ($P95$).
* **Memory Footprint:** Peak RAM and GPU/MPS memory consumption.

---

## 📈 Generated Artifacts

Execution automatically generates outputs in `./results/m5/` for M5 and `./results/tech_gadget/` for the gadget dataset:

```
results/
├── m5/                         # M5 forecasts, reports, charts, summaries, and runtime profiles
└── tech_gadget/                # Tech-gadget forecasts and corresponding outputs
```

---

<details>
<summary><b>🛠️ Advanced CLI Options</b></summary>

```bash
usage: run_benchmark.py [-h] [--preset {smoke,small,standard,extended}]
                        [--dataset {m5,tech_gadget}]
                        [--models MODELS] [--understock-cost UNDERSTOCK_COST]
                        [--overstock-cost OVERSTOCK_COST]
                        [--n-windows N_WINDOWS] [--skip-prep]
                        [--evaluate-only]

options:
  -h, --help            Show this help message and exit
  --preset PRESET       Dataset scale preset: smoke (~14 series), small (~56), standard (~168), extended (~525) (default: small)
  --dataset DATASET     Dataset: M5 daily retail or tech_gadget weekly retail (default: m5)
  --models MODELS       Comma-separated list of models: 'all', 'baselines', 'lightgbm', 'chronos', 'timesfm' (default: all)
  --understock-cost CU  Asymmetric inventory loss penalty for lost sales / stockouts (default: 3.0)
  --overstock-cost CO   Asymmetric inventory loss penalty for excess holding / markdown (default: 1.0)
  --n-windows N         Number of backtesting evaluation windows (1 for holdout, 2-3 for rolling origin) (default: 1)
  --skip-prep           Skip data preparation if the selected dataset parquet already exists (default: False)
  --evaluate-only       Skip forecasting and evaluate existing result files in results/ (default: False)
```

</details>

### Tech-gadget retail dataset

`--dataset tech_gadget` downloads the authors' raw CSV from the [Demand Prediction in Retail dataset page](https://demandprediction.github.io/dataset.html), validates its SKU/week panel, and saves the normalized data to `data/tech_gadget.parquet`. The dataset contains 44 tech-gadget SKUs with 100 weekly observations each (October 2016–September 2018), including sales, price, homepage-feature status, color, vendor, and functionality. The pipeline uses all 44 series, a 13-week forecast horizon, and 52-week seasonal scaling; it saves outputs under `results/tech_gadget/` so they do not overwrite M5 results.

The benchmark assumes future prices and homepage-feature flags are known for the 13-week forecast horizon. This is a useful what-if/planned-covariate setup, but it should be reported explicitly because those values come from the historical source data. The raw source's color value changes within some SKU histories, so color is retained for inspection but excluded as a static model feature. The dataset page asks users to cite Cohen, Gras, Pentecoste, and Zhang (2022), *Demand Prediction in Retail: A Practical Guide to Leverage Data and Predictive Analytics*. The page does not state a clear reuse license; verify terms before redistributing the raw data or derived files.

<details>
<summary><b>📁 Project Directory Structure</b></summary>

```
tsfm-demand-benchmark/
├── 01_prepare_data.py    # Stratified M5 sampling & business covariate extraction
├── 02_baselines.py       # Seasonal Naive, AutoETS, Croston Optimized
├── 03_lightgbm.py        # LightGBM with MLForecast & rolling features
├── 04_chronos.py         # Amazon Chronos-2 zero-shot pipeline
├── 05_timesfm.py         # Google TimesFM 3.0 zero-shot pipeline
├── 06_evaluate.py        # Multi-metric evaluation, report & chart generation
├── config.py             # Central configuration & preset definitions
├── run_benchmark.py      # Unified CLI orchestrator
├── utils.py              # Math metrics, inventory loss, profiler & device detection
├── pytest.ini            # Pytest test configuration
├── tests/
│   └── test_benchmark.py # Complete 12-test unit testing suite
└── requirements.txt      # Dependency specification
```

</details>

---

## 🤝 Contributing & License

Contributions, new model integrations (e.g. Salesforce MOMENT, IBM Granite, PatchTST), and research discussions are welcome! Please submit a PR or open an issue.

Distributed under the **Apache 2.0 License**. See `LICENSE` for more information.
