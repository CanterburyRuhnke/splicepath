from dataclasses import dataclass
from enum import Enum

import numpy as np

from splicepath.gauge.significance import mcnemar, wilson_ci


class ConcordanceLevel(Enum):
    EXACT = "exact"
    CLOSE = "close"
    APPROXIMATE = "approximate"
    WEAK = "weak"
    DISCORDANT = "discordant"


_THRESHOLDS = (
    (0.05, ConcordanceLevel.EXACT),
    (0.10, ConcordanceLevel.CLOSE),
    (0.15, ConcordanceLevel.APPROXIMATE),
    (0.20, ConcordanceLevel.WEAK),
)


def concordance_level(delta: float) -> ConcordanceLevel:
    magnitude = abs(delta)
    for bound, level in _THRESHOLDS:
        if magnitude <= bound:
            return level
    return ConcordanceLevel.DISCORDANT


_CONCORDANT = {
    ConcordanceLevel.EXACT,
    ConcordanceLevel.CLOSE,
    ConcordanceLevel.APPROXIMATE,
}


@dataclass(frozen=True)
class RetrospectiveReport:
    concordance: float
    wilson_low: float
    wilson_high: float
    counts: dict[str, int]
    mcnemar_p: float


def retrospective_report(
    predicted: np.ndarray,
    measured: np.ndarray,
    baseline_predicted: np.ndarray,
) -> RetrospectiveReport:
    pred = np.asarray(predicted, dtype=np.float64).ravel()
    meas = np.asarray(measured, dtype=np.float64).ravel()
    base = np.asarray(baseline_predicted, dtype=np.float64).ravel()
    counts = {level.value: 0 for level in ConcordanceLevel}
    model_hits = np.zeros(pred.size, dtype=bool)
    base_hits = np.zeros(pred.size, dtype=bool)
    for index in range(pred.size):
        level = concordance_level(pred[index] - meas[index])
        counts[level.value] += 1
        model_hits[index] = level in _CONCORDANT
        base_hits[index] = concordance_level(base[index] - meas[index]) in _CONCORDANT
    concordant = int(model_hits.sum())
    total = pred.size
    low, high = wilson_ci(concordant, total)
    discordant_b = int(np.sum(model_hits & ~base_hits))
    discordant_c = int(np.sum(~model_hits & base_hits))
    return RetrospectiveReport(
        concordance=concordant / total if total else float("nan"),
        wilson_low=low,
        wilson_high=high,
        counts=counts,
        mcnemar_p=mcnemar(discordant_b, discordant_c),
    )
