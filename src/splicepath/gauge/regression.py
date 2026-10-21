import numpy as np
from scipy.stats import rankdata


def _pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def pcc(prediction: np.ndarray, target: np.ndarray) -> float:
    x, y = _pair(prediction, target)
    if x.size < 2 or x.std() == 0.0 or y.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(prediction: np.ndarray, target: np.ndarray) -> float:
    x, y = _pair(prediction, target)
    if x.size < 2:
        return float("nan")
    return pcc(rankdata(x), rankdata(y))


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    x, y = _pair(prediction, target)
    return float(np.sqrt(np.mean((x - y) ** 2)))


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    x, y = _pair(prediction, target)
    return float(np.mean(np.abs(x - y)))


def r2_score(prediction: np.ndarray, target: np.ndarray) -> float:
    x, y = _pair(prediction, target)
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom == 0.0:
        return float("nan")
    return float(1.0 - np.sum((y - x) ** 2) / denom)


def per_event_pcc(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = np.asarray(prediction, dtype=np.float64)
    targ = np.asarray(target, dtype=np.float64)
    return np.asarray([pcc(pred[:, k], targ[:, k]) for k in range(pred.shape[1])])


def mean_pcc(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.nanmean(per_event_pcc(prediction, target)))


def median_pcc(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.nanmedian(per_event_pcc(prediction, target)))
