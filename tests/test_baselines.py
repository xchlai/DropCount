import numpy as np

from baselines import naive_equal_volume_estimate, volume_aware_mle_estimate


def test_naive_equal_volume_matches_formula():
    labels = np.array([1, 0, 1, 0, 0, 1], dtype=float)
    estimate = naive_equal_volume_estimate(labels)
    n = len(labels)
    z = n - labels.sum()
    expected = -n * np.log(z / n)
    assert np.isclose(estimate.estimate, expected)


def test_volume_aware_mle_matches_naive_for_monodisperse():
    labels = np.array([1, 0, 0, 1, 0, 0, 1, 0], dtype=float)
    f = np.full_like(labels, 1.0 / len(labels), dtype=float)
    naive = naive_equal_volume_estimate(labels)
    mle = volume_aware_mle_estimate(f, labels)
    assert np.isclose(mle.estimate, naive.estimate, rtol=1e-3, atol=1e-3)


def test_all_negative_returns_zero():
    labels = np.zeros(10, dtype=float)
    f = np.full(10, 0.1)
    naive = naive_equal_volume_estimate(labels)
    mle = volume_aware_mle_estimate(f, labels)
    assert naive.estimate == 0.0
    assert mle.estimate == 0.0
