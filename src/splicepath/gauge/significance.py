import numpy as np
from scipy.stats import chi2, norm, wilcoxon


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        return float("nan")
    pooled = ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    if pooled <= 0.0:
        return float("nan")
    return float((x.mean() - y.mean()) / np.sqrt(pooled))


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if np.allclose(x, y):
        return 1.0
    result = wilcoxon(x, y)
    return float(result.pvalue)


def holm_bonferroni(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p)
    m = p.size
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * p[idx]
        running = max(running, value)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=np.float64)
    running = 1.0
    for position in range(m - 1, -1, -1):
        idx = order[position]
        value = p[idx] * m / (position + 1)
        running = min(running, value)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def mcnemar(discordant_b: int, discordant_c: int) -> float:
    total = discordant_b + discordant_c
    if total == 0:
        return 1.0
    statistic = (abs(discordant_b - discordant_c) - 1.0) ** 2 / total
    return float(chi2.sf(statistic, 1))


def wilson_ci(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = float(norm.ppf(1.0 - alpha / 2.0))
    phat = successes / total
    denom = 1.0 + z**2 / total
    centre = (phat + z**2 / (2 * total)) / denom
    margin = z * np.sqrt(phat * (1.0 - phat) / total + z**2 / (4 * total**2)) / denom
    return float(centre - margin), float(centre + margin)
