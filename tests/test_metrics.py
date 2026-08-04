from __future__ import annotations

import numpy as np
import pytest

from fmva.metrics import (
    auprc,
    bootstrap_interval,
    comparison_state,
    expected_calibration_error,
    paired_bootstrap_difference,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from fmva.schemas import MetricInterval


def test_exact_ranking_metrics() -> None:
    y_true = np.array([1, 0, 1, 0], dtype=int)
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    assert precision_at_k(y_true, scores, 2) == pytest.approx(0.5)
    assert recall_at_k(y_true, scores, 2) == pytest.approx(0.5)
    assert reciprocal_rank(y_true, scores) == pytest.approx(1.0)
    assert auprc(y_true, scores) == pytest.approx(5.0 / 6.0)


def test_calibration_error() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=int)
    perfect = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y_true, perfect, bins=2) == pytest.approx(0.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        expected_calibration_error(y_true, np.array([0.0, 1.2, 0.5, 0.5]))


def test_bootstrap_and_paired_difference() -> None:
    y_true = np.array([1, 0, 1, 0, 0, 1], dtype=int)
    strong = np.array([0.9, 0.1, 0.8, 0.2, 0.3, 0.7])
    weak = np.array([0.2, 0.8, 0.1, 0.7, 0.6, 0.3])
    interval = bootstrap_interval(y_true, strong, auprc, 100, seed=1)
    assert interval.lower is not None
    assert interval.upper is not None
    paired = paired_bootstrap_difference(y_true, strong, weak, auprc, 100, seed=1)
    assert paired.estimate > 0
    assert paired.valid_bootstrap_replicates is not None
    assert paired.valid_bootstrap_replicates >= 95


def test_abstention_states_and_invalid_labels() -> None:
    gain = MetricInterval(estimate=0.1, lower=0.01, upper=0.2)
    assert comparison_state(gain) == "DETECTABLE_GAIN"
    loss = MetricInterval(estimate=-0.1, lower=-0.2, upper=-0.01)
    assert comparison_state(loss) == "DETECTABLE_LOSS"
    null = MetricInterval(estimate=0.0, lower=-0.1, upper=0.1)
    assert comparison_state(null) == "NO_DETECTABLE_DIFFERENCE"
    with pytest.raises(ValueError, match="positive"):
        auprc(np.zeros(4, dtype=int), np.arange(4, dtype=float))
