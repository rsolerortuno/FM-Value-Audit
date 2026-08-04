from __future__ import annotations

import numpy as np
import pytest

from fmva.simulation import simulate_benchmark
from fmva.splits import (
    build_random_split,
    build_temporal_split,
    deterministic_subset,
)


def test_temporal_split_is_prospective() -> None:
    frame = simulate_benchmark(n_genes=600, seed=11)
    split = build_temporal_split(frame, seed=11)
    assert set(split.historical_train).isdisjoint(set(split.historical_validation))
    assert split.evaluation_labels.sum() == frame["label_post_cutoff"].sum()
    assert frame.iloc[split.evaluation]["label_pre_cutoff"].sum() == 0
    assert split.cutoff_year == 2019
    assert split.evaluation_end_year == 2024


def test_random_split_is_disjoint_and_stratified() -> None:
    frame = simulate_benchmark(n_genes=600, seed=11)
    split = build_random_split(frame, seed=11)
    assert set(split.train).isdisjoint(set(split.test))
    assert len(split.train) + len(split.test) == len(frame)
    assert split.labels[split.test].sum() > 0


def test_deterministic_subset_keeps_positives() -> None:
    frame = simulate_benchmark(n_genes=500, seed=11)
    indices = np.arange(len(frame), dtype=int)
    labels = frame["label_post_cutoff"].to_numpy(dtype=int)
    selected_a = deterministic_subset(frame, indices, labels, 150, seed=3)
    selected_b = deterministic_subset(frame, indices, labels, 150, seed=3)
    np.testing.assert_array_equal(selected_a, selected_b)
    assert labels[selected_a].sum() == labels.sum()
    with pytest.raises(ValueError, match="positive"):
        deterministic_subset(frame, indices, labels, 0, seed=3)
