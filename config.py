"""Central configuration and experiment presets for TSFM demand benchmark."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkConfig:
    data_dir: Path = Path("./data")
    results_dir: Path = Path("./results")
    horizon: int = 28
    seasonality: int = 7
    quantile_levels: list[float] = (0.1, 0.5, 0.9)

    # Preset configurations: (n_stores, n_items_per_dept)
    PRESETS = {
        "smoke": {"n_stores": 1, "n_items_per_dept": 2},     # ~14 series, ultra-fast sanity check
        "small": {"n_stores": 2, "n_items_per_dept": 4},     # ~56 series, quick local test
        "standard": {"n_stores": 2, "n_items_per_dept": 12}, # ~168 series, robust statistical sample
        "extended": {"n_stores": 3, "n_items_per_dept": 25}, # ~525 series, comprehensive evaluation
    }


CONFIG = BenchmarkConfig()

