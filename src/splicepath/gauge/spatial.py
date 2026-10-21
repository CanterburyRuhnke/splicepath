import numpy as np

from splicepath.gauge.regression import spearman


def morans_i(values: np.ndarray, coords: np.ndarray, k: int = 8) -> float:
    x = np.asarray(values, dtype=np.float64).ravel()
    points = np.asarray(coords, dtype=np.float64)
    n = x.size
    if n < 3:
        return float("nan")
    deviations = x - x.mean()
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    k_eff = min(k, n - 1)
    neighbors = np.argsort(distances, axis=1)[:, :k_eff]
    weight = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        weight[i, neighbors[i]] = 1.0
    w_sum = float(weight.sum())
    if w_sum == 0.0:
        return float("nan")
    numerator = float((weight * np.outer(deviations, deviations)).sum())
    denominator = float((deviations**2).sum())
    if denominator == 0.0:
        return float("nan")
    return (n / w_sum) * (numerator / denominator)


def trajectory_concordance(predicted: np.ndarray, reference: np.ndarray) -> float:
    return spearman(predicted, reference)
