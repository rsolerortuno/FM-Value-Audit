"""End-to-end benchmark orchestration and required ablations."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fmva.baselines import (
    SCALAR_METHOD_COLUMNS,
    ScoreOutput,
    run_random_mlp,
    run_random_mlp_ood_transfer,
    run_random_ranking,
    run_scalar_baseline,
    run_supervised_baseline,
    run_supervised_ood_transfer,
)
from fmva.effort import EffortTracker
from fmva.evaluation import evaluate_frozen_methods
from fmva.io import write_json, write_table
from fmva.metrics import auprc
from fmva.simulation import validate_synthetic
from fmva.splits import build_random_split, build_temporal_split, deterministic_subset


@dataclass(frozen=True)
class BenchmarkArtifacts:
    """Paths emitted by one benchmark run."""

    summary_path: Path
    claims_path: Path
    effort_path: Path
    scores_path: Path


def _make_logistic(c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                    solver="liblinear",
                ),
            ),
        ]
    )


def _run_random_split_ablation(frame: pd.DataFrame, seed: int) -> dict[str, Any]:
    split = build_random_split(frame, seed)
    train_indices, validation_indices = train_test_split(
        split.train,
        test_size=0.30,
        random_state=seed,
        stratify=split.labels[split.train],
    )
    columns = [
        "mean_expression_id",
        "differential_expression_id",
        "cell_type_specificity_id",
        "network_centrality",
        "druggable_class",
        "publication_count_all_time",
    ]
    matrix = frame.loc[:, columns].to_numpy(dtype=float)
    configurations = ["C=0.1", "C=1", "C=10"]
    tracker = EffortTracker(
        method="random_split_supervised_leaky",
        seed=seed,
        configurations=configurations,
    )

    def tune() -> tuple[float, float]:
        best_c = 0.1
        best_value = -np.inf
        for c_value in [0.1, 1.0, 10.0]:
            model = _make_logistic(c_value, seed)
            model.fit(matrix[train_indices], split.labels[train_indices])
            probabilities = model.predict_proba(matrix[validation_indices])[:, 1]
            value = float(average_precision_score(split.labels[validation_indices], probabilities))
            if value > best_value:
                best_c, best_value = c_value, value
        return best_c, best_value

    selected_c, validation_auprc = tracker.run_tuning(tune)
    tracker.selected_configuration = f"C={selected_c:g}"

    def final() -> np.ndarray:
        model = _make_logistic(selected_c, seed)
        model.fit(matrix[split.train], split.labels[split.train])
        return np.asarray(model.predict_proba(matrix[split.test])[:, 1], dtype=np.float64)

    probabilities = np.asarray(tracker.run_final(final), dtype=float)
    labels = split.labels[split.test]
    popularity_scores = frame["publication_count_all_time"].to_numpy(dtype=float)[split.test]
    return {
        "status": "EXECUTED_INTENTIONALLY_LEAKY",
        "warning": (
            "Uses random labels split and all-time publication counts; "
            "not a valid primary estimate."
        ),
        "supervised_auprc": auprc(labels, probabilities),
        "popularity_auprc": auprc(labels, popularity_scores),
        "positive_fraction": float(labels.mean()),
        "validation_auprc": validation_auprc,
        "effort": tracker.to_record().model_dump(mode="json"),
    }


def _apply_subset(frame: pd.DataFrame, maximum: int | None, seed: int) -> pd.DataFrame:
    if maximum is None or len(frame) <= maximum:
        return frame.reset_index(drop=True)
    labels = frame["label_post_cutoff"].to_numpy(dtype=int)
    indices = np.arange(len(frame), dtype=int)
    selected = deterministic_subset(frame, indices, labels, maximum, seed)
    subset = frame.iloc[selected].reset_index(drop=True)
    if subset["label_pre_cutoff"].sum() < 2:
        raise ValueError("Subset retained too few historical positives for tuning")
    return subset


def _freeze_outputs(
    frame: pd.DataFrame, seed: int
) -> tuple[dict[str, ScoreOutput], dict[str, Any]]:
    split = build_temporal_split(frame, seed)
    outputs: dict[str, ScoreOutput] = {}
    for method in SCALAR_METHOD_COLUMNS:
        outputs[method] = run_scalar_baseline(frame, split, method, "id", seed)
    outputs["supervised_with_popularity"] = run_supervised_baseline(
        frame,
        split,
        "id",
        seed,
        include_popularity=True,
    )
    outputs["supervised_without_popularity"] = run_supervised_baseline(
        frame,
        split,
        "id",
        seed,
        include_popularity=False,
    )
    outputs["random_mlp_small"] = run_random_mlp(frame, split, "id", seed, "small")
    outputs["random_mlp_large"] = run_random_mlp(frame, split, "id", seed, "large")
    outputs["random_ranking"] = run_random_ranking(frame, split, seed)
    outputs["supervised_unmatched_compute"] = run_supervised_baseline(
        frame,
        split,
        "id",
        seed,
        include_popularity=True,
        unmatched_compute=True,
    )
    outputs["supervised_ood_transfer"] = run_supervised_ood_transfer(frame, split, seed)
    outputs["random_mlp_small_ood_transfer"] = run_random_mlp_ood_transfer(frame, split, seed)
    return outputs, {"split": split}


def _synthetic_diagnostics(
    frame: pd.DataFrame,
    summary_methods: dict[str, dict[str, Any]],
    score_frame: pd.DataFrame,
) -> dict[str, Any]:
    evaluation = score_frame[score_frame["is_temporal_evaluation"] == 1].copy()
    top_n = max(20, round(0.05 * len(evaluation)))
    popularity_top = evaluation.nlargest(top_n, "score_popularity")
    popularity_only_fraction = float((popularity_top["planted_group"] == "popularity_only").mean())
    supervised_excess = summary_methods["supervised_with_popularity"]["excess_over_popularity"][
        "auprc"
    ]["estimate"]
    biological_auprc = summary_methods["differential_expression"]["metrics"]["auprc"]
    random_auprc = summary_methods["random_ranking"]["metrics"]["auprc"]
    diagnostics = {
        "biology_method_beats_random": bool(biological_auprc > random_auprc),
        "differential_expression_auprc": biological_auprc,
        "random_ranking_auprc": random_auprc,
        "popularity_top_fraction_popularity_only": popularity_only_fraction,
        "popularity_only_overrepresented": bool(
            popularity_only_fraction > float((frame["planted_group"] == "popularity_only").mean())
        ),
        "supervised_excess_over_popularity_positive": bool(supervised_excess > 0.0),
    }
    diagnostics["harness_validation_passed"] = bool(
        diagnostics["biology_method_beats_random"]
        and diagnostics["popularity_only_overrepresented"]
        and diagnostics["supervised_excess_over_popularity_positive"]
    )
    return diagnostics


def _prediction_checks(
    methods: dict[str, dict[str, Any]],
    random_ablation: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    popularity_auprc = float(methods["popularity"]["metrics"]["auprc"])
    prevalence = float(methods["popularity"]["metadata"]["evaluation_prevalence"])
    non_gains = sum(
        result["comparison_state"] != "DETECTABLE_GAIN"
        for method, result in methods.items()
        if method not in {"popularity", "supervised_ood_transfer", "random_mlp_small_ood_transfer"}
    )
    executable_excess = [
        abs(float(result["excess_over_popularity"]["auprc"]["estimate"]))
        for method, result in methods.items()
        if method not in {"popularity", "supervised_ood_transfer", "random_mlp_small_ood_transfer"}
    ]
    temporal_supervised = float(methods["supervised_with_popularity"]["metrics"]["auprc"])
    random_supervised = float(random_ablation["supervised_auprc"])
    return {
        "popularity_will_be_strong": {
            "status": (
                "HELD" if popularity_auprc >= 1.1 * prevalence and non_gains >= 3 else "FAILED"
            ),
            "observed": {
                "popularity_auprc": popularity_auprc,
                "positive_fraction": prevalence,
                "methods_without_detectable_gain": non_gains,
            },
        },
        "excess_over_popularity_will_be_small": {
            "status": "HELD" if max(executable_excess) < 0.20 else "FAILED",
            "operational_definition": "absolute AUPRC excess below 0.20 for executable methods",
            "observed_max_absolute_excess": max(executable_excess),
            "foundation_model_component": "NOT_TESTED",
        },
        "random_initialisation_recovers_pretrained_share": {
            "status": "NOT_TESTED",
            "reason": "No verified pretrained checkpoint was executed.",
        },
        "temporal_scores_lower_than_random": {
            "status": "HELD" if random_supervised - temporal_supervised >= 0.05 else "FAILED",
            "observed": {
                "temporal_supervised_auprc": temporal_supervised,
                "leaky_random_supervised_auprc": random_supervised,
                "gap": random_supervised - temporal_supervised,
            },
        },
    }


def run_benchmark(
    frame: pd.DataFrame,
    output_directory: Path,
    bootstrap_replicates: int = 400,
    seed: int = 20260729,
    subset: int | None = None,
) -> BenchmarkArtifacts:
    """Run all executable methods, final evaluation and required ablations."""
    validate_synthetic(frame)
    frame = _apply_subset(frame, subset, seed)
    output_directory.mkdir(parents=True, exist_ok=True)

    random_ablation = _run_random_split_ablation(frame, seed)
    outputs, frozen_metadata = _freeze_outputs(frame, seed)
    split = frozen_metadata["split"]

    results, evaluation_metadata = evaluate_frozen_methods(
        frame,
        split,
        outputs,
        bootstrap_replicates,
        seed,
    )
    serialised_methods = {
        method: result.model_dump(mode="json") for method, result in results.items()
    }
    library_versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    split_definition = {
        "kind": "prospective_landmark",
        "cutoff_year": split.cutoff_year,
        "evaluation_end_year": split.evaluation_end_year,
    }
    for _method, result in serialised_methods.items():
        result["metadata"].update(
            {
                "method_version": "fmva-0.1.0",
                "seed": seed,
                "split_definition": split_definition,
                "library_versions": library_versions,
            }
        )

    score_frame = frame[["gene_id", "planted_group", "label_post_cutoff"]].copy()
    evaluation_mask = np.zeros(len(frame), dtype=int)
    evaluation_mask[split.evaluation] = 1
    score_frame["is_temporal_evaluation"] = evaluation_mask
    for method, output in outputs.items():
        score_frame[f"score_{method}"] = output.scores
    scores_path = output_directory / "scores.csv"
    write_table(score_frame, scores_path)

    ablations = {
        "temporal_vs_random": random_ablation,
        "with_vs_without_popularity": {
            "status": "EXECUTED",
            "with_popularity_auprc": serialised_methods["supervised_with_popularity"]["metrics"][
                "auprc"
            ],
            "without_popularity_auprc": serialised_methods["supervised_without_popularity"][
                "metrics"
            ]["auprc"],
        },
        "pretrained_vs_random_initialisation": {
            "status": "NOT_COMPUTED",
            "random_control_auprc": serialised_methods["random_mlp_small"]["metrics"]["auprc"],
            "reason": "No verified pretrained checkpoint was available.",
        },
        "frozen_vs_finetuned": {
            "status": "NOT_COMPUTED",
            "reason": "No verified pretrained checkpoint or GPU run was available.",
        },
        "id_vs_ood": {
            "status": "EXECUTED_SYNTHETIC",
            "supervised_id_auprc": serialised_methods["supervised_with_popularity"]["metrics"][
                "auprc"
            ],
            "supervised_ood_auprc": serialised_methods["supervised_ood_transfer"]["metrics"][
                "auprc"
            ],
            "supervised_ood_delta": (
                serialised_methods["supervised_ood_transfer"]["metrics"]["auprc"]
                - serialised_methods["supervised_with_popularity"]["metrics"]["auprc"]
            ),
            "random_mlp_id_auprc": serialised_methods["random_mlp_small"]["metrics"]["auprc"],
            "random_mlp_ood_auprc": serialised_methods["random_mlp_small_ood_transfer"]["metrics"][
                "auprc"
            ],
        },
        "matched_vs_unmatched_compute": {
            "status": "EXECUTED",
            "matched_auprc": serialised_methods["supervised_with_popularity"]["metrics"]["auprc"],
            "unmatched_auprc": serialised_methods["supervised_unmatched_compute"]["metrics"][
                "auprc"
            ],
            "matched_configurations": len(
                serialised_methods["supervised_with_popularity"]["effort"]["configurations_tried"]
            ),
            "unmatched_configurations": len(
                serialised_methods["supervised_unmatched_compute"]["effort"]["configurations_tried"]
            ),
        },
        "model_scale": {
            "status": "EXECUTED_RANDOM_CONTROL_ONLY",
            "small_parameter_count": serialised_methods["random_mlp_small"]["metadata"][
                "parameter_count"
            ],
            "large_parameter_count": serialised_methods["random_mlp_large"]["metadata"][
                "parameter_count"
            ],
            "small_auprc": serialised_methods["random_mlp_small"]["metrics"]["auprc"],
            "large_auprc": serialised_methods["random_mlp_large"]["metrics"]["auprc"],
            "warning": "This is not evidence about pretrained model scaling.",
        },
    }

    diagnostics = _synthetic_diagnostics(frame, serialised_methods, score_frame)
    prediction_checks = _prediction_checks(serialised_methods, random_ablation)
    effort_log = [result["effort"] for result in serialised_methods.values()]
    environment = {
        **library_versions,
        "platform": platform.platform(),
        "seed": seed,
    }
    summary = {
        "schema_version": "1.0",
        "run_kind": "SYNTHETIC_EXECUTED",
        "real_foundation_model_results": "NOT_COMPUTED",
        "methods": serialised_methods,
        "ablations": ablations,
        "synthetic_diagnostics": diagnostics,
        "pre_registered_predictions": prediction_checks,
        "holdout_audit": evaluation_metadata["holdout_audit"],
        "environment": environment,
        "substitutions": [
            "Synthetic benchmark substituted for inaccessible real target and single-cell data.",
            "Local random MLP substituted for an architecture-matched random foundation model.",
            "External pretrained adapters were mock-tested and left NOT_COMPUTED.",
        ],
    }
    summary_path = output_directory / "benchmark_summary.json"
    claims_path = output_directory / "claims.json"
    effort_path = output_directory / "effort_log.json"
    write_json(summary, summary_path)
    write_json(evaluation_metadata["claims"], claims_path)
    write_json({"effort_log": effort_log}, effort_path)
    return BenchmarkArtifacts(summary_path, claims_path, effort_path, scores_path)
