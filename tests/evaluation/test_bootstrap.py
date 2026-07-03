"""Unit tests for the percentile-bootstrap CI utilities."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from modernbert_g2p.evaluation.bootstrap import bootstrap_ci, bootstrap_diff_ci


def test_bootstrap_ci_single_value_returns_lo_eq_hi_eq_mean() -> None:
    result = bootstrap_ci([1.23])
    assert result["n"] == 1
    assert result["mean"] == pytest.approx(1.23)
    assert result["lo"] == pytest.approx(1.23)
    assert result["hi"] == pytest.approx(1.23)


def test_bootstrap_ci_empty_returns_zeros() -> None:
    result = bootstrap_ci([])
    assert result["n"] == 0
    assert result["mean"] == 0.0
    assert result["lo"] == 0.0
    assert result["hi"] == 0.0


def test_bootstrap_ci_covers_true_mean_for_standard_normal() -> None:
    rng = np.random.default_rng(2026)
    scores = rng.standard_normal(size=200).tolist()
    result = bootstrap_ci(scores, seed=2026, n_resample=5000)
    true_mean = float(np.mean(scores))
    assert result["mean"] == pytest.approx(true_mean, abs=1e-9)
    assert result["lo"] < true_mean < result["hi"]
    assert abs(result["lo"] - result["hi"]) < 1.0


def test_bootstrap_ci_is_deterministic_given_seed() -> None:
    scores = [0.1, 0.9, 1.4, 0.2, 0.7, 1.1, 0.8, 0.5, 0.6, 0.3]
    a = bootstrap_ci(scores, seed=1234, n_resample=500)
    b = bootstrap_ci(scores, seed=1234, n_resample=500)
    assert a == b


def test_bootstrap_ci_different_seeds_differ() -> None:
    scores = [0.1, 0.9, 1.4, 0.2, 0.7, 1.1, 0.8, 0.5, 0.6, 0.3]
    a = bootstrap_ci(scores, seed=1, n_resample=500)
    b = bootstrap_ci(scores, seed=2, n_resample=500)
    assert a["mean"] == pytest.approx(b["mean"])
    assert (a["lo"], a["hi"]) != (b["lo"], b["hi"])


def test_bootstrap_ci_invalid_ci_raises() -> None:
    with pytest.raises(ValueError):
        bootstrap_ci([0.1, 0.2], ci=1.5)
    with pytest.raises(ValueError):
        bootstrap_ci([0.1, 0.2], ci=0.0)


def test_bootstrap_ci_small_sample_warns() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bootstrap_ci([0.1, 0.2], seed=0, n_resample=100)
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_bootstrap_diff_ci_identical_arrays_delta_zero() -> None:
    scores = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = bootstrap_diff_ci(scores, scores, seed=0, n_resample=500)
    assert result["delta_mean"] == pytest.approx(0.0)
    assert result["lo"] == pytest.approx(0.0)
    assert result["hi"] == pytest.approx(0.0)


def test_bootstrap_diff_ci_a_strictly_greater() -> None:
    a = [1.0, 1.1, 1.2, 0.9, 1.05, 1.15]
    b = [0.1, 0.2, 0.15, 0.05, 0.12, 0.18]
    result = bootstrap_diff_ci(a, b, seed=7, n_resample=1000)
    assert result["delta_mean"] > 0
    assert result["lo"] > 0
    assert result["hi"] > 0
    assert result["prob_a_worse"] == pytest.approx(1.0)


def test_bootstrap_diff_ci_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        bootstrap_diff_ci([0.1, 0.2], [0.1, 0.2, 0.3])


def test_bootstrap_diff_ci_paired_structure() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal(50)
    b = a + 0.5
    a_list = a.tolist()
    b_list = b.tolist()
    result = bootstrap_diff_ci(a_list, b_list, seed=0, n_resample=1000)
    assert result["delta_mean"] == pytest.approx(-0.5, abs=1e-9)
    assert result["lo"] < -0.5 < result["hi"] or (
        result["lo"] == pytest.approx(-0.5) or result["hi"] == pytest.approx(-0.5)
    )
    assert result["prob_a_worse"] < 0.05


def test_bootstrap_diff_ci_single_pair() -> None:
    result = bootstrap_diff_ci([1.0], [0.5], seed=0)
    assert result["delta_mean"] == pytest.approx(0.5)
    assert result["lo"] == pytest.approx(0.5)
    assert result["hi"] == pytest.approx(0.5)
    assert result["prob_a_worse"] == pytest.approx(1.0)


def test_bootstrap_diff_ci_empty() -> None:
    result = bootstrap_diff_ci([], [])
    assert result["n"] == 0
    assert result["delta_mean"] == 0.0
    assert result["prob_a_worse"] == pytest.approx(0.5)
