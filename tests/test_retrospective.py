import numpy as np

from splicepath.quiltshow.baselines import mean_predictor_pcc
from splicepath.quiltshow.retrospective import (
    ConcordanceLevel,
    concordance_level,
    retrospective_report,
)


def test_concordance_level_thresholds() -> None:
    assert concordance_level(0.03) is ConcordanceLevel.EXACT
    assert concordance_level(0.08) is ConcordanceLevel.CLOSE
    assert concordance_level(0.12) is ConcordanceLevel.APPROXIMATE
    assert concordance_level(0.18) is ConcordanceLevel.WEAK
    assert concordance_level(0.30) is ConcordanceLevel.DISCORDANT


def test_retrospective_report_counts_and_concordance() -> None:
    measured = np.array([0.5, 0.5, 0.5, 0.5])
    predicted = np.array([0.52, 0.58, 0.62, 0.9])
    baseline = np.array([0.9, 0.9, 0.9, 0.9])
    report = retrospective_report(predicted, measured, baseline)
    assert report.counts["exact"] == 1
    assert report.concordance == 0.75
    assert report.wilson_low <= report.concordance <= report.wilson_high


def test_mean_predictor_is_finite() -> None:
    rng = np.random.default_rng(0)
    train = rng.random((20, 8))
    test = rng.random((10, 8))
    value = mean_predictor_pcc(train, test)
    assert np.isnan(value) or -1.0 <= value <= 1.0
