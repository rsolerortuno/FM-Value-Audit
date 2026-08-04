"""Final holdout evaluation, abstention logic and claim generation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

import pandas as pd

from fmva.baselines import ScoreOutput
from fmva.metrics import (
    MetricFunction,
    auprc,
    bootstrap_interval,
    comparison_state,
    expected_calibration_error,
    paired_bootstrap_difference,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from fmva.schemas import LeakageStatus, MethodResult, MetricInterval
from fmva.splits import TemporalSplit


class EvaluationMetadata(TypedDict):
    """Typed metadata returned after the single holdout evaluation."""

    claims: dict[str, list[dict[str, object]]]
    holdout_audit: dict[str, object]


def _metric_functions(k: int) -> dict[str, MetricFunction]:
    return {
        f"precision_at_{k}": lambda y, s: precision_at_k(y, s, k),
        f"recall_at_{k}": lambda y, s: recall_at_k(y, s, k),
        "mrr": reciprocal_rank,
        "auprc": auprc,
    }


def _compute_normalised(excess: float, output: ScoreOutput) -> dict[str, float | None]:
    effort = output.effort
    cpu_hours = effort.cpu_seconds / 3600.0
    return {
        "gain_per_cpu_hour": excess / cpu_hours if cpu_hours > 0.0 else None,
        "gain_per_gpu_hour": excess / effort.gpu_hours if effort.gpu_hours > 0.0 else None,
        "gain_per_configuration": (
            excess / len(effort.configurations_tried) if effort.configurations_tried else None
        ),
    }


def evaluate_frozen_methods(
    frame: pd.DataFrame,
    split: TemporalSplit,
    outputs: Mapping[str, ScoreOutput],
    bootstrap_replicates: int,
    seed: int,
    k: int = 20,
    leakage_statuses: Mapping[str, LeakageStatus] | None = None,
) -> tuple[dict[str, MethodResult], EvaluationMetadata]:
    """Touch the prospective holdout once and evaluate all frozen methods together."""
    if "popularity" not in outputs:
        raise ValueError("The mandatory popularity control is missing")
    evaluation_indices = split.evaluation
    y_true = split.evaluation_labels
    reference_scores = outputs["popularity"].scores[evaluation_indices]
    metric_functions = _metric_functions(k)
    results: dict[str, MethodResult] = {}

    for method_index, (method, output) in enumerate(outputs.items()):
        method_scores = output.scores[evaluation_indices]
        raw_metrics: dict[str, float | None] = {}
        intervals: dict[str, MetricInterval] = {}
        excess_intervals: dict[str, MetricInterval] = {}
        for metric_index, (metric_name, metric_function) in enumerate(metric_functions.items()):
            raw_value = float(metric_function(y_true, method_scores))
            raw_metrics[metric_name] = raw_value
            interval_seed = seed + 1000 * method_index + 37 * metric_index
            intervals[metric_name] = bootstrap_interval(
                y_true,
                method_scores,
                metric_function,
                bootstrap_replicates,
                interval_seed,
            )
            excess_intervals[metric_name] = paired_bootstrap_difference(
                y_true,
                method_scores,
                reference_scores,
                metric_function,
                bootstrap_replicates,
                interval_seed + 17,
            )

        if output.probabilities is None:
            raw_metrics["calibration_ece"] = None
        else:
            raw_metrics["calibration_ece"] = expected_calibration_error(
                y_true,
                output.probabilities[evaluation_indices],
            )

        auprc_excess = excess_intervals["auprc"]
        results[method] = MethodResult(
            method=method,
            leakage_status=(leakage_statuses or {}).get(method, LeakageStatus.RESOLVED_CLEAN),
            metrics=raw_metrics,
            metric_intervals=intervals,
            excess_over_popularity=excess_intervals,
            comparison_state=comparison_state(auprc_excess),
            compute_normalised_value=_compute_normalised(auprc_excess.estimate, output),
            effort=output.effort,
            metadata={
                **output.metadata,
                "evaluation_gene_count": len(evaluation_indices),
                "evaluation_positive_count": int(y_true.sum()),
                "evaluation_prevalence": float(y_true.mean()),
            },
        )

    claims = build_claims(results)
    audit: dict[str, object] = {
        "holdout_access_count": 1,
        "cutoff_year": split.cutoff_year,
        "evaluation_end_year": split.evaluation_end_year,
        "evaluation_gene_count": len(evaluation_indices),
        "evaluation_positive_count": int(y_true.sum()),
        "evaluation_prevalence": float(y_true.mean()),
        "bootstrap_replicates_requested": bootstrap_replicates,
    }
    return results, {"claims": claims, "holdout_audit": audit}


def build_claims(results: Mapping[str, MethodResult]) -> dict[str, list[dict[str, object]]]:
    """Derive permitted, conditional and unsupported statements from executed results."""
    permitted: list[dict[str, object]] = [
        {
            "statement": "The committed synthetic benchmark executed without pretrained weights.",
            "scope": "synthetic_harness_only",
        },
        {
            "statement": "All reported method differences use paired intervals against popularity.",
            "scope": "executed_methods",
        },
    ]
    conditional: list[dict[str, object]] = []
    for method, result in results.items():
        if method == "popularity":
            continue
        interval = result.excess_over_popularity["auprc"]
        conditional.append(
            {
                "statement": (
                    f"On this synthetic run, {method} had AUPRC excess "
                    f"{interval.estimate:.6f} versus popularity."
                ),
                "comparison_state": result.comparison_state,
                "leakage_status": result.leakage_status.value,
                "scope": "synthetic_only",
            }
        )
    unsupported: list[dict[str, object]] = [
        {
            "statement": "Any real foundation model outperforms or underperforms cheap baselines.",
            "reason": "No pretrained checkpoint was executed.",
        },
        {
            "statement": "Any ranked gene is a clinically valid therapeutic target.",
            "reason": "The committed data are synthetic and no wet-lab validation exists.",
        },
        {
            "statement": (
                "The synthetic OOD shift predicts transfer across real tissues or diseases."
            ),
            "reason": "The OOD perturbation is a harness stress test only.",
        },
    ]
    return {"permitted": permitted, "conditional": conditional, "unsupported": unsupported}
