import numpy as np
from scipy.stats import pearsonr

from splicepath.gauge.classification import auroc
from splicepath.gauge.intervals import bootstrap_ci
from splicepath.gauge.regression import pcc, r2_score, rmse, spearman
from splicepath.gauge.significance import (
    benjamini_hochberg,
    cohens_d,
    holm_bonferroni,
    mcnemar,
    wilson_ci,
)
from splicepath.gauge.spatial import morans_i


def test_pcc_matches_scipy() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    b = 0.7 * a + 0.3 * rng.normal(size=200)
    assert abs(pcc(a, b) - pearsonr(a, b)[0]) < 1e-9


def test_pcc_perfect_linear() -> None:
    assert abs(pcc(np.array([1.0, 2, 3]), np.array([2.0, 4, 6])) - 1.0) < 1e-9


def test_spearman_monotonic_invariance() -> None:
    a = np.array([1.0, 2, 3, 4, 5])
    b = np.exp(a)
    assert abs(spearman(a, b) - 1.0) < 1e-9


def test_auroc_perfect_and_chance() -> None:
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert abs(auroc(scores, labels) - 1.0) < 1e-9
    assert abs(auroc(np.array([0.5, 0.5, 0.5, 0.5]), labels) - 0.5) < 1e-9


def test_rmse_and_r2_reference() -> None:
    pred = np.array([1.0, 2, 3])
    true = np.array([1.0, 2, 3])
    assert rmse(pred, true) == 0.0
    assert abs(r2_score(pred, true) - 1.0) < 1e-9


def test_bootstrap_ci_brackets_mean() -> None:
    rng = np.random.default_rng(1)
    data = rng.normal(loc=0.34, scale=0.05, size=300)
    low, high = bootstrap_ci(data, resamples=400, seed=2)
    assert low < data.mean() < high


def test_holm_bonferroni_reference() -> None:
    adjusted = holm_bonferroni(np.array([0.01, 0.02, 0.03]))
    expected = np.array([0.03, 0.04, 0.04])
    assert np.allclose(adjusted, expected)


def test_benjamini_hochberg_monotone() -> None:
    adjusted = benjamini_hochberg(np.array([0.001, 0.01, 0.04, 0.5]))
    assert np.all(adjusted >= np.array([0.001, 0.01, 0.04, 0.5]) - 1e-9)
    assert adjusted[0] <= adjusted[-1]


def test_cohens_d_sign() -> None:
    a = np.full(20, 1.0) + 0.01 * np.arange(20)
    b = np.zeros(20) + 0.01 * np.arange(20)
    assert cohens_d(a, b) > 2.0


def test_mcnemar_and_wilson() -> None:
    assert mcnemar(0, 0) == 1.0
    assert 0.0 <= mcnemar(20, 5) <= 1.0
    low, high = wilson_ci(68, 100)
    assert low < 0.68 < high


def test_morans_i_positive_for_gradient() -> None:
    grid = 8
    coords = np.array([[i, j] for i in range(grid) for j in range(grid)], dtype=float)
    values = coords[:, 0] + coords[:, 1]
    assert morans_i(values, coords, k=4) > 0.5
