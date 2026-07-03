"""Bootstrap 95% CI on evaluation scores.

Pure numpy; no scipy dependency. The percentile method is numerically
equivalent to :func:`scipy.stats.bootstrap` with ``method="percentile"`` on the
same random seed, using ``np.random.default_rng`` and ``np.quantile`` with
linear interpolation (numpy default).

Return schema (for :func:`bootstrap_ci`)::

    {
        "mean": float,       # sample mean of scores
        "lo": float,         # lower CI bound
        "hi": float,         # upper CI bound
        "n": int,            # number of scores
        "n_resample": int,   # bootstrap iterations
        "ci": float,         # nominal confidence level
    }

For :func:`bootstrap_diff_ci` an additional ``prob_a_worse`` key reports the
frequency with which paired-bootstrap resample means placed ``scores_a``
above ``scores_b``. When both arrays hold PER (lower is better), this is
readable as ``P(A more errors than B)``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

_SMALL_SAMPLE_THRESHOLD = 5


def _resample_indices(rng: np.random.Generator, n_resample: int, n: int) -> np.ndarray:
    return rng.integers(low=0, high=n, size=(n_resample, n))


def bootstrap_ci(
    scores: Sequence[float],
    *,
    n_resample: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int]:
    """Percentile bootstrap CI over the mean of ``scores``.

    Args:
        scores: Per-row metric values (e.g. per-utterance PER percentages).
        n_resample: Number of bootstrap resamples.
        ci: Nominal confidence level in ``(0, 1)``.
        seed: RNG seed for :func:`numpy.random.default_rng`.

    Returns:
        Dict with keys ``mean``, ``lo``, ``hi``, ``n``, ``n_resample``, ``ci``.
        For ``n == 0`` returns zeros; for ``n == 1`` returns ``lo == hi == mean``.
        A :class:`RuntimeWarning` is emitted when ``n < 5``.
    """
    if not 0 < ci < 1:
        raise ValueError(f"ci must be in (0, 1), got {ci!r}")
    if n_resample <= 0:
        raise ValueError(f"n_resample must be > 0, got {n_resample!r}")

    arr = np.asarray(list(scores), dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {
            "mean": 0.0,
            "lo": 0.0,
            "hi": 0.0,
            "n": 0,
            "n_resample": n_resample,
            "ci": ci,
        }
    mean = float(arr.mean())
    if n < 2:
        return {
            "mean": mean,
            "lo": mean,
            "hi": mean,
            "n": n,
            "n_resample": n_resample,
            "ci": ci,
        }
    if n < _SMALL_SAMPLE_THRESHOLD:
        warnings.warn(
            f"bootstrap_ci: n={n} < {_SMALL_SAMPLE_THRESHOLD}, CI will be unstable",
            RuntimeWarning,
            stacklevel=2,
        )
    rng = np.random.default_rng(seed)
    idx = _resample_indices(rng, n_resample, n)
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return {
        "mean": mean,
        "lo": lo,
        "hi": hi,
        "n": n,
        "n_resample": n_resample,
        "ci": ci,
    }


def bootstrap_diff_ci(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_resample: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int]:
    """Paired difference bootstrap on ``delta = scores_a[i] - scores_b[i]``.

    Args:
        scores_a: Per-row scores for system A (aligned with ``scores_b``).
        scores_b: Per-row scores for system B.
        n_resample: Bootstrap iterations.
        ci: Confidence level in ``(0, 1)``.
        seed: RNG seed.

    Returns:
        Dict with keys ``delta_mean``, ``lo``, ``hi``, ``n``, ``n_resample``,
        ``ci``, ``prob_a_worse``. ``prob_a_worse`` is the fraction of bootstrap
        resamples in which the mean delta was positive (A above B, meaning
        A worse when the metric is lower-is-better like PER).

    Raises:
        ValueError: If ``len(scores_a) != len(scores_b)``.
    """
    if not 0 < ci < 1:
        raise ValueError(f"ci must be in (0, 1), got {ci!r}")
    if n_resample <= 0:
        raise ValueError(f"n_resample must be > 0, got {n_resample!r}")

    a = np.asarray(list(scores_a), dtype=np.float64)
    b = np.asarray(list(scores_b), dtype=np.float64)
    if a.size != b.size:
        raise ValueError(f"bootstrap_diff_ci: length mismatch a={a.size} b={b.size}")
    n = int(a.size)
    if n == 0:
        return {
            "delta_mean": 0.0,
            "lo": 0.0,
            "hi": 0.0,
            "n": 0,
            "n_resample": n_resample,
            "ci": ci,
            "prob_a_worse": 0.5,
        }
    delta = a - b
    delta_mean = float(delta.mean())
    if n < 2:
        if delta_mean > 0:
            prob = 1.0
        elif delta_mean < 0:
            prob = 0.0
        else:
            prob = 0.5
        return {
            "delta_mean": delta_mean,
            "lo": delta_mean,
            "hi": delta_mean,
            "n": n,
            "n_resample": n_resample,
            "ci": ci,
            "prob_a_worse": prob,
        }
    if n < _SMALL_SAMPLE_THRESHOLD:
        warnings.warn(
            f"bootstrap_diff_ci: n={n} < {_SMALL_SAMPLE_THRESHOLD}, CI will be unstable",
            RuntimeWarning,
            stacklevel=2,
        )
    rng = np.random.default_rng(seed)
    idx = _resample_indices(rng, n_resample, n)
    deltas = delta[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(deltas, alpha))
    hi = float(np.quantile(deltas, 1.0 - alpha))
    prob_a_worse = float((deltas > 0).mean())
    return {
        "delta_mean": delta_mean,
        "lo": lo,
        "hi": hi,
        "n": n,
        "n_resample": n_resample,
        "ci": ci,
        "prob_a_worse": prob_a_worse,
    }


__all__ = ["bootstrap_ci", "bootstrap_diff_ci"]
