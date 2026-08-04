from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fmva.adapters import ExternalNpzAdapter, MockFoundationAdapter, RandomMLPAdapter
from fmva.baselines import (
    SCALAR_METHOD_COLUMNS,
    canonical_feature_columns,
    run_random_mlp,
    run_random_mlp_ood_transfer,
    run_random_ranking,
    run_scalar_baseline,
    run_supervised_baseline,
    run_supervised_ood_transfer,
)
from fmva.simulation import simulate_benchmark
from fmva.splits import build_temporal_split


def test_all_baselines_execute_with_matched_budget() -> None:
    frame = simulate_benchmark(n_genes=500, seed=11)
    split = build_temporal_split(frame, seed=11)
    outputs = [
        run_scalar_baseline(frame, split, method, "id", seed=11) for method in SCALAR_METHOD_COLUMNS
    ]
    outputs.extend(
        [
            run_supervised_baseline(frame, split, "id", seed=11),
            run_supervised_baseline(
                frame,
                split,
                "id",
                seed=11,
                include_popularity=False,
            ),
            run_random_mlp(frame, split, "id", seed=11),
            run_random_ranking(frame, split, seed=11),
            run_supervised_ood_transfer(frame, split, seed=11),
            run_random_mlp_ood_transfer(frame, split, seed=11),
        ]
    )
    for output in outputs:
        assert output.scores.shape == (len(frame),)
        assert np.isfinite(output.scores).all()
        assert len(output.effort.configurations_tried) == 3
        assert output.effort.selected_configuration != "NOT_SELECTED"


def test_unmatched_compute_has_larger_budget() -> None:
    frame = simulate_benchmark(n_genes=400, seed=11)
    split = build_temporal_split(frame, seed=11)
    output = run_supervised_baseline(
        frame,
        split,
        "id",
        seed=11,
        unmatched_compute=True,
    )
    assert len(output.effort.configurations_tried) == 12


def test_random_adapter_uses_reference_indices() -> None:
    frame = simulate_benchmark(n_genes=200, seed=11)
    columns = canonical_feature_columns("id")
    adapter = RandomMLPAdapter("random", 1, 16, 8)
    first = adapter.embed_with_reference(frame, columns, np.arange(100, dtype=int))
    second = adapter.embed_with_reference(frame, columns, np.arange(100, 200, dtype=int))
    assert first.shape == (200, 8)
    assert not np.allclose(first, second)
    assert adapter.parameter_count_for_input(len(columns)) == 248


def test_external_adapter_aligns_and_fails_honestly(tmp_path: Path) -> None:
    frame = pd.DataFrame({"gene_id": ["B", "A"]})
    path = tmp_path / "embeddings.npz"
    np.savez(path, gene_ids=np.array(["A", "B"]), embeddings=np.array([[1.0], [2.0]]))
    adapter = ExternalNpzAdapter("external", path)
    aligned = adapter.embed(frame, [])
    np.testing.assert_array_equal(aligned[:, 0], np.array([2.0, 1.0]))
    with pytest.raises(RuntimeError, match="NOT_COMPUTED"):
        MockFoundationAdapter("mock", "missing weights").embed(frame, [])
