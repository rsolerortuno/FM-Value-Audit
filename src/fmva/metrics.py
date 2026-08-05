"""Ranking, calibration and paired bootstrap metrics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from typing import Any
from sklearn.metrics import average_precision_score

from fmva.schemas import MetricInterval

MetricFunction = Callable[[npt.NDArray[Any], npt.NDArray[Any]], float]


def _validate_binary(y_true: npt.NDArray[Any]) -> None:
    unique = set(np.unique(y_true).tolist())
    if not unique <= {0, 1}:
        raise ValueError("Labels must be binary")
    if int(y_true.sum()) == 0:
        raise ValueError("At least one positive label is required")


def _rank_order(scores: npt.NDArray[Any]) -> npt.NDArray[Any]:
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite")
    return np.argsort(-scores, kind="mergesort")


def precision_at_k(y_true: npt.NDArray[Any], scores: npt.NDArray[Any], k: int = 20) -> float:
    """Return precision among the top-k ranked genes."""
    _validate_binary(y_true)
    if k <= 0:
        raise ValueError("k must be positive")
    actual_k = min(k, len(y_true))
    order = _rank_order(scores)
    return float(y_true[order[:actual_k]].mean())


def recall_at_k(y_true: npt.NDArray[Any], scores: npt.NDArray[Any], k: int = 20) -> float:
    """Return recall among the top-k ranked genes."""
    _validate_binary(y_true)
    if k <= 0:
        raise ValueError("k must be positive")
    actual_k = min(k, len(y_true))
    order = _rank_order(scores)
    return float(y_true[order[:actual_k]].sum() / y_true.sum())


def reciprocal_rank(y_true: npt.NDArray[Any], scores: npt.NDArray[Any]) -> float:
    """Return reciprocal rank of the first positive gene."""
    _validate_binary(y_true)
    order = _rank_order(scores)
    ranked_labels = y_true[order]
    first_positive = int(np.flatnonzero(ranked_labels == 1)[0])
    return 1.0 / float(first_positive + 1)


def auprc(y_true: npt.NDArray[Any], scores: npt.NDArray[Any]) -> float:
    """Return average precision, used as AUPRC."""
    _validate_binary(y_true)
    return float(average_precision_score(y_true, scores))


def expected_calibration_error(
    y_true: npt.NDArray[Any],
    probabilities: npt.NDArray[Any],
    bins: int = 10,
) -> float:
    """Compute equal-width expected calibration error."""
    _validate_binary(y_true)
    if bins <= 0:
        raise ValueError("bins must be positive")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Probabilities must lie in [0, 1]")
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
    total = len(y_true)
    error = 0.0
    for index in range(bins):
        mask = assignments == index
        count = int(mask.sum())
        if count == 0:
            continue
        observed = float(y_true[mask].mean())
        predicted = float(probabilities[mask].mean())
        error += (count / total) * abs(observed - predicted)
    return float(error)


def metric_bundle(y_true: npt.NDArray[Any], scores: npt.NDArray[Any], k: int = 20) -> dict[str, float]:
    """Compute all raw ranking metrics."""
    return {
        f"precision_at_{k}": precision_at_k(y_true, scores, k),
        f"recall_at_{k}": recall_at_k(y_true, scores, k),
        "mrr": reciprocal_rank(y_true, scores),
        "auprc": auprc(y_true, scores),
    }


def bootstrap_interval(
    y_true: npt.NDArray[Any],
    scores: npt.NDArray[Any],
    metric: MetricFunction,
    replicates: int,
    seed: int,
) -> MetricInterval:
    """Compute a percentile gene-bootstrap interval for one metric."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    point = metric(y_true, scores)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(replicates):
        sampled = rng.integers(0, len(y_true), len(y_true))
        sampled_y = y_true[sampled]
        if sampled_y.sum() == 0:
            continue
        values.append(metric(sampled_y, scores[sampled]))
    if not values:
        raise ValueError("No valid bootstrap replicates contained a positive")
    lower, upper = np.quantile(np.asarray(values), np.array([0.025, 0.975]))
    return MetricInterval(
        estimate=point,
        lower=float(lower),
        upper=float(upper),
        valid_bootstrap_replicates=len(values),
    )


def paired_bootstrap_difference(
    y_true: npt.NDArray[Any],
    method_scores: npt.NDArray[Any],
    reference_scores: npt.NDArray[Any],
    metric: MetricFunction,
    replicates: int,
    seed: int,
) -> MetricInterval:
    """Compute a paired percentile interval for method minus reference."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    point = metric(y_true, method_scores) - metric(y_true, reference_scores)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(replicates):
        sampled = rng.integers(0, len(y_true), len(y_true))
        sampled_y = y_true[sampled]
        if sampled_y.sum() == 0:
            continue
        method_value = metric(sampled_y, method_scores[sampled])
        reference_value = metric(sampled_y, reference_scores[sampled])
        values.append(method_value - reference_value)
    if not values:
        raise ValueError("No valid paired bootstrap replicates contained a positive")
    lower, upper = np.quantile(np.asarray(values), np.array([0.025, 0.975]))
    return MetricInterval(
        estimate=float(point),
        lower=float(lower),
        upper=float(upper),
        valid_bootstrap_replicates=len(values),
    )


def comparison_state(interval: MetricInterval) -> str:
    """Convert an excess interval into an abstaining comparison state."""
    if interval.lower is None or interval.upper is None:
        return "NOT_COMPUTED"
    if interval.lower > 0.0:
        return "DETECTABLE_GAIN"
    if interval.upper < 0.0:
        return "DETECTABLE_LOSS"
    return "NO_DETECTABLE_DIFFERENCE"
