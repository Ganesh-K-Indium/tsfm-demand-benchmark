"""Central configuration, experiment presets, and research parameters for TSFM benchmark."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkConfig:
    data_dir: Path = Path("./data")
    results_dir: Path = Path("./results")
    horizon: int = 28
    seasonality: int = 7
    quantile_levels: list[float] = (0.1, 0.5, 0.9)

    # Supply Chain / Inventory Cost Parameters
    understock_cost: float = 3.0  # Cost per unit of lost sales / stockout (lost gross margin + penalty)
    overstock_cost: float = 1.0   # Cost per unit of overstock (holding cost + salvage/markdown loss)

    # Backtesting & Rolling Windows
    n_windows: int = 1            # 1 = single holdout, 2-3 = rolling-origin cross-validation

    # Preset configurations: (n_stores, n_items_per_dept)
    PRESETS = {
        "smoke": {"n_stores": 1, "n_items_per_dept": 2},     # ~14 series, ultra-fast sanity check
        "small": {"n_stores": 2, "n_items_per_dept": 4},     # ~56 series, quick local test
        "standard": {"n_stores": 2, "n_items_per_dept": 12}, # ~168 series, robust statistical sample
        "extended": {"n_stores": 3, "n_items_per_dept": 25}, # ~525 series, comprehensive evaluation
    }


CONFIG = BenchmarkConfig()
