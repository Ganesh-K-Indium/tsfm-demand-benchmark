#!/usr/bin/env python3
"""Unified CLI Orchestrator for the Time Series Foundation Model Benchmark."""
import argparse
import importlib
import sys
import time

from config import CONFIG
from utils import get_device


def log(msg: str, header: bool = False):
    line = "=" * 70
    if header:
        print(f"\n\033[1;34m{line}\033[0m")
        print(f"\033[1;32m🚀 {msg}\033[0m")
        print(f"\033[1;34m{line}\033[0m\n")
    else:
        print(f"\033[1;36m>> {msg}\033[0m")


def main():
    parser = argparse.ArgumentParser(
        description="Unified TSFM Demand Forecasting Benchmark Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        choices=list(CONFIG.PRESETS.keys()),
        default="small",
        help="Dataset scale preset: smoke (~14 series), small (~56), standard (~168), extended (~525)",
    )
    parser.add_argument(
        "--dataset",
        choices=["m5", "tech_gadget"],
        default="m5",
        help="Dataset: Walmart M5 (daily) or the authors' tech-gadget retail panel (weekly)",
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated list of models to evaluate: 'all', 'baselines', 'lightgbm', 'chronos', 'timesfm'",
    )
    parser.add_argument(
        "--understock-cost",
        type=float,
        default=CONFIG.understock_cost,
        help="Asymmetric inventory loss penalty for lost sales / stockouts (Cu)",
    )
    parser.add_argument(
        "--overstock-cost",
        type=float,
        default=CONFIG.overstock_cost,
        help="Asymmetric inventory loss penalty for excess holding / markdown (Co)",
    )
    parser.add_argument(
        "--n-windows",
        type=int,
        default=CONFIG.n_windows,
        help="Number of backtesting evaluation windows (1 for single holdout, 2-3 for rolling origin)",
    )
    parser.add_argument(
        "--skip-prep",
        action="store_true",
        help="Skip preparation if the selected dataset's prepared parquet already exists",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip forecasting and run evaluation on existing results in results/",
    )
    args = parser.parse_args()

    # Update global config
    CONFIG.configure_dataset(args.dataset)
    CONFIG.understock_cost = args.understock_cost
    CONFIG.overstock_cost = args.overstock_cost
    CONFIG.n_windows = args.n_windows

    total_start = time.perf_counter()
    device = get_device()
    log(f"Starting TSFM Benchmark (Preset: {args.preset}, Device: {device.upper()})", header=True)

    if args.evaluate_only:
        log("Running evaluation only...")
        eval_mod = importlib.import_module("06_evaluate")
        eval_mod.run_evaluation()
        return

    # 1. Data Preparation
    data_file = CONFIG.prepared_data_path
    if args.skip_prep and data_file.exists():
        log(f"Found existing dataset at {data_file}, skipping preparation.")
    else:
        log(f"Step 1/6: Preparing {CONFIG.dataset} data with business covariates...")
        prep_mod = importlib.import_module("01_prepare_data")
        prep_mod.prepare_data(preset=args.preset, dataset=args.dataset)

    selected_models = [m.strip().lower() for m in args.models.split(",")]
    run_all = "all" in selected_models

    # 2. Baselines
    if run_all or "baselines" in selected_models:
        log("Step 2/6: Running Statistical Baselines (SeasonalNaive, AutoETS, Croston)...")
        baselines_mod = importlib.import_module("02_baselines")
        baselines_mod.run_baselines()

    # 3. LightGBM
    if run_all or "lightgbm" in selected_models:
        log("Step 3/6: Training & Forecasting with LightGBM (with business drivers)...")
        lgb_mod = importlib.import_module("03_lightgbm")
        lgb_mod.run_lightgbm()

    # 4. Chronos-2
    if run_all or "chronos" in selected_models:
        log("Step 4/6: Zero-shot forecasting with Amazon Chronos-2...")
        chronos_mod = importlib.import_module("04_chronos")
        chronos_mod.run_chronos()

    # 5. TimesFM 3.0
    if run_all or "timesfm" in selected_models:
        log("Step 5/6: Zero-shot forecasting with Google TimesFM 3.0...")
        timesfm_mod = importlib.import_module("05_timesfm")
        timesfm_mod.run_timesfm()

    # 6. Evaluation & Reporting
    log("Step 6/6: Evaluating all forecasts & generating report...")
    eval_mod = importlib.import_module("06_evaluate")
    eval_mod.run_evaluation()

    total_elapsed = time.perf_counter() - total_start
    log(f"TSFM Benchmark Completed Successfully in {total_elapsed:.1f}s!", header=True)


if __name__ == "__main__":
    main()
