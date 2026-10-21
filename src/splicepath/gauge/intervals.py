from collections.abc import Callable

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = lambda v: float(np.mean(v)),
    resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64).ravel()
    if data.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sample = rng.choice(data, size=data.size, replace=True)
        estimates[index] = statistic(sample)
    lo = float(np.quantile(estimates, alpha / 2.0))
    hi = float(np.quantile(estimates, 1.0 - alpha / 2.0))
    return lo, hi
