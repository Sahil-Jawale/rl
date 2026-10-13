from __future__ import annotations

import numpy as np


def bootstrap_confidence_interval(values: list[float], confidence: float = 0.95, samples: int = 1000) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(0)
    boot = rng.choice(arr, size=(samples, arr.size), replace=True)
    means = boot.mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower = np.quantile(means, alpha)
    upper = np.quantile(means, 1.0 - alpha)
    return float(lower), float(upper)
