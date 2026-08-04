from __future__ import annotations

import pandas as pd
import pytest

from fmva.simulation import simulate_benchmark, validate_synthetic


def test_simulation_is_deterministic_and_valid() -> None:
    first = simulate_benchmark(n_genes=500, seed=11)
    second = simulate_benchmark(n_genes=500, seed=11)
    pd.testing.assert_frame_equal(first, second)
    validate_synthetic(first)
    assert first["label_pre_cutoff"].sum() >= 8
    assert first["label_post_cutoff"].sum() >= 8
    popularity_only = first["planted_group"] == "popularity_only"
    assert first.loc[popularity_only, "label_post_cutoff"].sum() == 0


def test_simulation_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="at least 100"):
        simulate_benchmark(n_genes=99)
    with pytest.raises(ValueError, match="after cutoff"):
        simulate_benchmark(n_genes=100, cutoff_year=2024, evaluation_end_year=2024)


def test_validate_synthetic_rejects_bad_tables() -> None:
    frame = simulate_benchmark(n_genes=200, seed=22)
    with pytest.raises(ValueError, match="missing columns"):
        validate_synthetic(frame.drop(columns=["gene_id"]))
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        validate_synthetic(duplicated)
    overlapping = frame.copy()
    overlapping.loc[0, ["label_pre_cutoff", "label_post_cutoff"]] = 1
    with pytest.raises(ValueError, match="must not overlap"):
        validate_synthetic(overlapping)
