"""First-class cheap baselines and random-initialisation controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fmva.adapters.random_mlp import RandomMLPAdapter
from fmva.effort import EffortTracker
from fmva.schemas import EffortRecord
from fmva.splits import TemporalSplit

SCALAR_METHOD_COLUMNS: Final[dict[str, str]] = {
    "popularity": "publication_count_pre_cutoff",
    "mean_expression": "mean_expression_{context}",
    "differential_expression": "differential_expression_{context}",
    "cell_type_specificity": "cell_type_specificity_{context}",
    "network_centrality": "network_centrality",
    "druggable_class": "druggable_class",
}


@dataclass(frozen=True)
class ScoreOutput:
    """Scores, optional probabilities, effort and method metadata."""

    scores: np.ndarray
    probabilities: np.ndarray | None
    effort: EffortRecord
    metadata: dict[str, object]


def canonical_feature_columns(context: str, include_popularity: bool = True) -> list[str]:
    """Return canonical supervised features for one context."""
    if context not in {"id", "ood"}:
        raise ValueError("context must be 'id' or 'ood'")
    columns = [
        f"mean_expression_{context}",
        f"differential_expression_{context}",
        f"cell_type_specificity_{context}",
        "network_centrality",
        "druggable_class",
    ]
    if include_popularity:
        columns.append("publication_count_pre_cutoff")
    return columns


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    denominator = max(len(values) - 1, 1)
    return ranks / denominator


def _transform(values: np.ndarray, training_indices: np.ndarray, configuration: str) -> np.ndarray:
    if configuration == "identity":
        return values.astype(float, copy=True)
    if configuration == "percentile_rank":
        return _percentile_rank(values)
    if configuration == "robust_z":
        training = values[training_indices]
        median = float(np.median(training))
        q25, q75 = np.quantile(training, np.array([0.25, 0.75]))
        scale = float(q75 - q25)
        if scale == 0.0:
            scale = 1.0
        return (values - median) / scale
    raise ValueError(f"Unknown scalar configuration: {configuration}")


def run_scalar_baseline(
    frame: pd.DataFrame,
    split: TemporalSplit,
    method: str,
    context: str,
    seed: int,
) -> ScoreOutput:
    """Tune and execute one scalar baseline under the matched budget."""
    if method not in SCALAR_METHOD_COLUMNS:
        raise KeyError(f"Unknown scalar method: {method}")
    template = SCALAR_METHOD_COLUMNS[method]
    column = template.format(context=context)
    values = frame[column].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values in {column}")

    configurations = ["identity", "percentile_rank", "robust_z"]
    tracker = EffortTracker(method=method, seed=seed, configurations=configurations)

    def tune() -> tuple[str, np.ndarray, float]:
        best_configuration = configurations[0]
        best_scores = _transform(values, split.historical_train, best_configuration)
        validation_y = split.historical_labels[split.historical_validation]
        best_value = float(
            average_precision_score(validation_y, best_scores[split.historical_validation])
        )
        for configuration in configurations[1:]:
            candidate = _transform(values, split.historical_train, configuration)
            value = float(
                average_precision_score(validation_y, candidate[split.historical_validation])
            )
            if value > best_value:
                best_configuration = configuration
                best_scores = candidate
                best_value = value
        return best_configuration, best_scores, best_value

    selected, tuned_scores, validation_auprc = tracker.run_tuning(tune)
    tracker.selected_configuration = selected

    def final() -> np.ndarray:
        return _transform(values, np.arange(len(frame), dtype=int), selected)

    final_scores = tracker.run_final(final)
    return ScoreOutput(
        scores=np.asarray(final_scores, dtype=float),
        probabilities=None,
        effort=tracker.to_record(),
        metadata={
            "feature_column": column,
            "validation_auprc": float(validation_auprc),
            "selected_configuration": selected,
            "tuned_scores_equal_final": bool(np.allclose(tuned_scores, final_scores)),
        },
    )


def _make_logistic(c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        steps=[
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


def run_supervised_baseline(
    frame: pd.DataFrame,
    split: TemporalSplit,
    context: str,
    seed: int,
    include_popularity: bool = True,
    unmatched_compute: bool = False,
) -> ScoreOutput:
    """Tune logistic regression on historical labels and rank the prospective universe."""
    columns = canonical_feature_columns(context, include_popularity=include_popularity)
    matrix = frame.loc[:, columns].to_numpy(dtype=float)
    if unmatched_compute:
        c_values = np.logspace(-4, 3, 12).tolist()
        method = "supervised_unmatched_compute"
    else:
        c_values = [0.1, 1.0, 10.0]
        method = (
            "supervised_with_popularity" if include_popularity else "supervised_without_popularity"
        )
    configurations = [f"C={value:g}" for value in c_values]
    tracker = EffortTracker(method=method, seed=seed, configurations=configurations)

    def tune() -> tuple[float, float]:
        validation_y = split.historical_labels[split.historical_validation]
        best_c = c_values[0]
        best_value = -np.inf
        for c_value in c_values:
            model = _make_logistic(float(c_value), seed)
            model.fit(
                matrix[split.historical_train],
                split.historical_labels[split.historical_train],
            )
            probabilities = model.predict_proba(matrix[split.historical_validation])[:, 1]
            value = float(average_precision_score(validation_y, probabilities))
            if value > best_value:
                best_c = float(c_value)
                best_value = value
        return best_c, best_value

    selected_c, validation_auprc = tracker.run_tuning(tune)
    tracker.selected_configuration = f"C={selected_c:g}"

    def final() -> tuple[np.ndarray, np.ndarray]:
        model = _make_logistic(selected_c, seed)
        model.fit(matrix, split.historical_labels)
        probabilities = model.predict_proba(matrix)[:, 1]
        return probabilities.copy(), probabilities

    scores, probabilities = tracker.run_final(final)
    return ScoreOutput(
        scores=np.asarray(scores, dtype=float),
        probabilities=np.asarray(probabilities, dtype=float),
        effort=tracker.to_record(),
        metadata={
            "feature_columns": columns,
            "validation_auprc": float(validation_auprc),
            "include_popularity": include_popularity,
            "unmatched_compute": unmatched_compute,
        },
    )


def run_random_mlp(
    frame: pd.DataFrame,
    split: TemporalSplit,
    context: str,
    seed: int,
    scale: str = "small",
) -> ScoreOutput:
    """Tune random seeds for fixed random embeddings plus a logistic head."""
    columns = canonical_feature_columns(context, include_popularity=True)
    if scale == "small":
        hidden_width, embedding_width = 16, 8
    elif scale == "large":
        hidden_width, embedding_width = 64, 32
    else:
        raise ValueError("scale must be 'small' or 'large'")
    method = f"random_mlp_{scale}"
    candidate_seeds = [seed, seed + 1, seed + 2]
    configurations = [f"seed={candidate}" for candidate in candidate_seeds]
    tracker = EffortTracker(method=method, seed=seed, configurations=configurations)

    def embed_for(candidate_seed: int) -> np.ndarray:
        adapter = RandomMLPAdapter(
            model_id=method,
            seed=candidate_seed,
            hidden_width=hidden_width,
            embedding_width=embedding_width,
        )
        return adapter.embed_with_reference(frame, columns, split.historical_train)

    def tune() -> tuple[int, float]:
        validation_y = split.historical_labels[split.historical_validation]
        best_seed = candidate_seeds[0]
        best_value = -np.inf
        for candidate_seed in candidate_seeds:
            embeddings = embed_for(candidate_seed)
            model = _make_logistic(1.0, seed)
            model.fit(
                embeddings[split.historical_train],
                split.historical_labels[split.historical_train],
            )
            probabilities = model.predict_proba(embeddings[split.historical_validation])[:, 1]
            value = float(average_precision_score(validation_y, probabilities))
            if value > best_value:
                best_seed = candidate_seed
                best_value = value
        return best_seed, best_value

    selected_seed, validation_auprc = tracker.run_tuning(tune)
    tracker.selected_configuration = f"seed={selected_seed}"

    def final() -> tuple[np.ndarray, int]:
        embeddings = embed_for(selected_seed)
        model = _make_logistic(1.0, seed)
        model.fit(embeddings, split.historical_labels)
        probabilities = model.predict_proba(embeddings)[:, 1]
        input_width = len(columns)
        adapter = RandomMLPAdapter(method, selected_seed, hidden_width, embedding_width)
        return probabilities, adapter.parameter_count_for_input(input_width)

    probabilities, parameter_count = tracker.run_final(final)
    probabilities_array = np.asarray(probabilities, dtype=float)
    return ScoreOutput(
        scores=probabilities_array,
        probabilities=probabilities_array,
        effort=tracker.to_record(),
        metadata={
            "feature_columns": columns,
            "validation_auprc": float(validation_auprc),
            "parameter_count": int(parameter_count),
            "architecture": {
                "hidden_width": hidden_width,
                "embedding_width": embedding_width,
            },
            "pretrained": False,
        },
    )


def run_supervised_ood_transfer(
    frame: pd.DataFrame,
    split: TemporalSplit,
    seed: int,
) -> ScoreOutput:
    """Train and tune on ID features, then score frozen on OOD features."""
    id_columns = canonical_feature_columns("id", include_popularity=True)
    ood_columns = canonical_feature_columns("ood", include_popularity=True)
    id_matrix = frame.loc[:, id_columns].to_numpy(dtype=float)
    ood_matrix = frame.loc[:, ood_columns].to_numpy(dtype=float)
    c_values = [0.1, 1.0, 10.0]
    configurations = [f"C={value:g}" for value in c_values]
    tracker = EffortTracker(
        method="supervised_ood_transfer",
        seed=seed,
        configurations=configurations,
    )

    def tune() -> tuple[float, float]:
        validation_y = split.historical_labels[split.historical_validation]
        best_c = c_values[0]
        best_value = -np.inf
        for c_value in c_values:
            model = _make_logistic(c_value, seed)
            model.fit(
                id_matrix[split.historical_train],
                split.historical_labels[split.historical_train],
            )
            probabilities = model.predict_proba(id_matrix[split.historical_validation])[:, 1]
            value = float(average_precision_score(validation_y, probabilities))
            if value > best_value:
                best_c = c_value
                best_value = value
        return best_c, best_value

    selected_c, validation_auprc = tracker.run_tuning(tune)
    tracker.selected_configuration = f"C={selected_c:g}"

    def final() -> np.ndarray:
        model = _make_logistic(selected_c, seed)
        model.fit(id_matrix, split.historical_labels)
        return np.asarray(model.predict_proba(ood_matrix)[:, 1], dtype=np.float64)

    probabilities = tracker.run_final(final)
    return ScoreOutput(
        scores=np.asarray(probabilities, dtype=float),
        probabilities=np.asarray(probabilities, dtype=float),
        effort=tracker.to_record(),
        metadata={
            "training_feature_columns": id_columns,
            "scoring_feature_columns": ood_columns,
            "validation_auprc_id": validation_auprc,
            "transfer": "ID_TO_OOD_FROZEN",
        },
    )


def run_random_mlp_ood_transfer(
    frame: pd.DataFrame,
    split: TemporalSplit,
    seed: int,
) -> ScoreOutput:
    """Train a random-MLP head on ID embeddings and apply it frozen to OOD embeddings."""
    id_columns = canonical_feature_columns("id", include_popularity=True)
    ood_columns = canonical_feature_columns("ood", include_popularity=True)
    id_matrix = frame.loc[:, id_columns].to_numpy(dtype=float)
    ood_matrix = frame.loc[:, ood_columns].to_numpy(dtype=float)
    candidate_seeds = [seed, seed + 1, seed + 2]
    configurations = [f"seed={candidate}" for candidate in candidate_seeds]
    tracker = EffortTracker(
        method="random_mlp_small_ood_transfer",
        seed=seed,
        configurations=configurations,
    )

    def embeddings_for(candidate_seed: int) -> tuple[np.ndarray, np.ndarray, RandomMLPAdapter]:
        adapter = RandomMLPAdapter(
            model_id="random_mlp_small_ood_transfer",
            seed=candidate_seed,
            hidden_width=16,
            embedding_width=8,
        )
        id_embeddings, ood_embeddings = adapter.embed_arrays(
            id_matrix,
            ood_matrix,
            split.historical_train,
        )
        return id_embeddings, ood_embeddings, adapter

    def tune() -> tuple[int, float]:
        validation_y = split.historical_labels[split.historical_validation]
        best_seed = candidate_seeds[0]
        best_value = -np.inf
        for candidate_seed in candidate_seeds:
            id_embeddings, _, _ = embeddings_for(candidate_seed)
            model = _make_logistic(1.0, seed)
            model.fit(
                id_embeddings[split.historical_train],
                split.historical_labels[split.historical_train],
            )
            probabilities = model.predict_proba(id_embeddings[split.historical_validation])[:, 1]
            value = float(average_precision_score(validation_y, probabilities))
            if value > best_value:
                best_seed = candidate_seed
                best_value = value
        return best_seed, best_value

    selected_seed, validation_auprc = tracker.run_tuning(tune)
    tracker.selected_configuration = f"seed={selected_seed}"

    def final() -> tuple[np.ndarray, int]:
        id_embeddings, ood_embeddings, adapter = embeddings_for(selected_seed)
        model = _make_logistic(1.0, seed)
        model.fit(id_embeddings, split.historical_labels)
        probabilities = model.predict_proba(ood_embeddings)[:, 1]
        return probabilities, adapter.parameter_count_for_input(len(id_columns))

    probabilities, parameter_count = tracker.run_final(final)
    probability_array = np.asarray(probabilities, dtype=float)
    return ScoreOutput(
        scores=probability_array,
        probabilities=probability_array,
        effort=tracker.to_record(),
        metadata={
            "training_feature_columns": id_columns,
            "scoring_feature_columns": ood_columns,
            "validation_auprc_id": validation_auprc,
            "parameter_count": parameter_count,
            "transfer": "ID_TO_OOD_FROZEN",
            "pretrained": False,
        },
    )


def run_random_ranking(frame: pd.DataFrame, split: TemporalSplit, seed: int) -> ScoreOutput:
    """Tune among three seeded random rankings and retain the selected seed."""
    candidate_seeds = [seed, seed + 1, seed + 2]
    configurations = [f"seed={candidate}" for candidate in candidate_seeds]
    tracker = EffortTracker(method="random_ranking", seed=seed, configurations=configurations)

    def make_scores(candidate_seed: int) -> np.ndarray:
        return np.random.default_rng(candidate_seed).uniform(0.0, 1.0, len(frame))

    def tune() -> tuple[int, float]:
        validation_y = split.historical_labels[split.historical_validation]
        best_seed = candidate_seeds[0]
        best_value = -np.inf
        for candidate_seed in candidate_seeds:
            scores = make_scores(candidate_seed)
            value = float(
                average_precision_score(validation_y, scores[split.historical_validation])
            )
            if value > best_value:
                best_seed = candidate_seed
                best_value = value
        return best_seed, best_value

    selected_seed, validation_auprc = tracker.run_tuning(tune)
    tracker.selected_configuration = f"seed={selected_seed}"
    final_scores = tracker.run_final(lambda: make_scores(selected_seed))
    return ScoreOutput(
        scores=np.asarray(final_scores, dtype=float),
        probabilities=None,
        effort=tracker.to_record(),
        metadata={"validation_auprc": float(validation_auprc)},
    )
