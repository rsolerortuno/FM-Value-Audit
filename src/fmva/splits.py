"""Temporal and deliberately leaky random benchmark splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from typing import Any
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class TemporalSplit:
    """Indices and labels for historical tuning and prospective evaluation."""

    historical_train: npt.NDArray[Any]
    historical_validation: npt.NDArray[Any]
    evaluation: npt.NDArray[Any]
    historical_labels: npt.NDArray[Any]
    evaluation_labels: npt.NDArray[Any]
    cutoff_year: int
    evaluation_end_year: int


@dataclass(frozen=True)
class RandomSplit:
    """Indices for the intentionally leaky random-split ablation."""

    train: npt.NDArray[Any]
    test: npt.NDArray[Any]
    labels: npt.NDArray[Any]


def build_temporal_split(frame: pd.DataFrame, seed: int) -> TemporalSplit:
    """Create a landmark split without consulting post-cutoff labels for tuning."""
    required = {
        "label_pre_cutoff",
        "label_post_cutoff",
        "entry_year",
        "cutoff_year",
        "evaluation_end_year",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing temporal split columns: {sorted(missing)}")

    historical_labels = frame["label_pre_cutoff"].to_numpy(dtype=int)
    indices = np.arange(len(frame), dtype=int)
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=0.30,
        random_state=seed,
        stratify=historical_labels,
    )

    entry_year = frame["entry_year"].to_numpy(dtype=float)
    historical_positive = historical_labels == 1
    evaluation_mask = (~historical_positive) & (
        np.isnan(entry_year) | (frame["label_post_cutoff"].to_numpy(dtype=int) == 1)
    )
    evaluation_indices = indices[evaluation_mask]
    evaluation_labels = frame.loc[evaluation_mask, "label_post_cutoff"].to_numpy(dtype=int)
    if evaluation_labels.sum() == 0:
        raise ValueError("Temporal evaluation contains no positives")

    return TemporalSplit(
        historical_train=np.asarray(train_indices, dtype=int),
        historical_validation=np.asarray(validation_indices, dtype=int),
        evaluation=evaluation_indices,
        historical_labels=historical_labels,
        evaluation_labels=evaluation_labels,
        cutoff_year=int(frame["cutoff_year"].iloc[0]),
        evaluation_end_year=int(frame["evaluation_end_year"].iloc[0]),
    )


def build_random_split(frame: pd.DataFrame, seed: int) -> RandomSplit:
    """Create an intentionally leaky random split over final development labels."""
    labels = (
        frame["label_pre_cutoff"].to_numpy(dtype=int)
        | frame["label_post_cutoff"].to_numpy(dtype=int)
    ).astype(int)
    indices = np.arange(len(frame), dtype=int)
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )
    return RandomSplit(
        train=np.asarray(train_indices, dtype=int),
        test=np.asarray(test_indices, dtype=int),
        labels=labels,
    )


def deterministic_subset(
    frame: pd.DataFrame,
    indices: npt.NDArray[Any],
    labels: npt.NDArray[Any],
    maximum: int | None,
    seed: int,
) -> npt.NDArray[Any]:
    """Keep all positives and deterministically sample negatives up to a maximum."""
    if maximum is None or len(indices) <= maximum:
        return indices
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    local_labels = labels if len(labels) == len(indices) else labels[indices]
    positive_indices = indices[local_labels == 1]
    if len(positive_indices) > maximum:
        raise ValueError("subset is smaller than the number of positives")
    negative_indices = indices[local_labels == 0]
    rng = np.random.default_rng(seed)
    selected_negative = rng.choice(
        negative_indices,
        size=maximum - len(positive_indices),
        replace=False,
    )
    selected = np.concatenate([positive_indices, selected_negative])
    return np.sort(selected.astype(int))
