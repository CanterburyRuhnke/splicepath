import numpy as np
from scipy.stats import rankdata


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    positives = y == 1
    negatives = y == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)
    rank_sum = float(ranks[positives].sum())
    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def auroc_multi(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    return np.asarray([auroc(s[:, k], y[:, k]) for k in range(s.shape[1])])
