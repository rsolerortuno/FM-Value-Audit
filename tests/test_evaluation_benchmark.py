from __future__ import annotations

import json
from pathlib import Path

from fmva.baselines import run_random_ranking, run_scalar_baseline
from fmva.benchmark import run_benchmark
from fmva.evaluation import evaluate_frozen_methods
from fmva.simulation import simulate_benchmark
from fmva.splits import build_temporal_split


def test_evaluation_requires_popularity() -> None:
    frame = simulate_benchmark(n_genes=300, seed=11)
    split = build_temporal_split(frame, seed=11)
    random = run_random_ranking(frame, split, seed=11)
    try:
        evaluate_frozen_methods(frame, split, {"random": random}, 20, 11)
    except ValueError as error:
        assert "popularity" in str(error)
    else:
        raise AssertionError("Expected missing-popularity failure")


def test_evaluation_emits_claims_and_abstention() -> None:
    frame = simulate_benchmark(n_genes=400, seed=11)
    split = build_temporal_split(frame, seed=11)
    popularity = run_scalar_baseline(frame, split, "popularity", "id", seed=11)
    random = run_random_ranking(frame, split, seed=11)
    results, metadata = evaluate_frozen_methods(
        frame,
        split,
        {"popularity": popularity, "random": random},
        30,
        11,
    )
    assert results["popularity"].excess_over_popularity["auprc"].estimate == 0.0
    assert metadata["holdout_audit"]["holdout_access_count"] == 1
    assert metadata["claims"]["unsupported"]


def test_end_to_end_benchmark(tmp_path: Path) -> None:
    frame = simulate_benchmark(n_genes=900, seed=11)
    artifacts = run_benchmark(frame, tmp_path, bootstrap_replicates=35, seed=11)
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["run_kind"] == "SYNTHETIC_EXECUTED"
    assert summary["real_foundation_model_results"] == "NOT_COMPUTED"
    assert summary["holdout_audit"]["holdout_access_count"] == 1
    assert summary["synthetic_diagnostics"]["harness_validation_passed"]
    assert summary["ablations"]["frozen_vs_finetuned"]["status"] == "NOT_COMPUTED"
    assert summary["ablations"]["id_vs_ood"]["supervised_ood_delta"] < 0
    configurations = summary["methods"]["supervised_unmatched_compute"]["effort"][
        "configurations_tried"
    ]
    assert len(configurations) == 12
    assert artifacts.claims_path.exists()
    assert artifacts.effort_path.exists()
    assert artifacts.scores_path.exists()
