from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fmva.contextual import default_contextual_protocol
from fmva.prospective import (
    GateStatus,
    evaluate_label_only_power_gate,
    freeze_future_scores,
    validate_future_score_freeze,
)


def _labels(positives: int = 20) -> pd.DataFrame:
    rows = []
    for index in range(30):
        rows.append(
            {
                "target_gene": f"G{index}",
                "indication": f"I{index % 6}",
                "gene_fold": 0 if index < 6 else (index % 4) + 1,
                "label_new_clinical_entry": int(index < positives),
            }
        )
    return pd.DataFrame(rows)


def test_label_only_gate_passes_at_thresholds() -> None:
    report = evaluate_label_only_power_gate(
        _labels(20),
        default_contextual_protocol(),
        snapshot_date="2027-02-04",
    )
    assert report.status == GateStatus.PASS
    assert report.holdout_positive_pairs == 6
    assert report.indications_with_positive == 6


def test_label_only_gate_waits_when_underpowered() -> None:
    report = evaluate_label_only_power_gate(
        _labels(4),
        default_contextual_protocol(),
        snapshot_date="2027-02-04",
    )
    assert report.status == GateStatus.WAIT
    assert report.failures


def test_label_only_gate_rejects_scores() -> None:
    labels = _labels(20)
    labels["model_score"] = 0.5
    with pytest.raises(ValueError, match="score-like"):
        evaluate_label_only_power_gate(
            labels,
            default_contextual_protocol(),
            snapshot_date="2027-02-04",
        )


def test_label_only_gate_rejects_early_snapshot() -> None:
    with pytest.raises(ValueError, match="precedes"):
        evaluate_label_only_power_gate(
            _labels(20),
            default_contextual_protocol(),
            snapshot_date="2026-11-04",
        )


def test_future_score_freeze_roundtrip(tmp_path: Path) -> None:
    protocol = default_contextual_protocol()
    ids = _labels(0).drop(columns=["label_new_clinical_entry"])
    scores = ids.copy()
    scores["cheap_historical"] = 0.1
    scores["geneformer_contextual_plus_cheap"] = 0.2
    manifest = freeze_future_scores(
        ids,
        scores,
        {"status": "FROZEN", "primary_method": "geneformer_contextual_plus_cheap"},
        protocol,
        tmp_path,
    )
    assert manifest.rows == 30
    validated = validate_future_score_freeze(tmp_path, protocol)
    assert validated.scores_sha256 == manifest.scores_sha256


def test_future_score_freeze_rejects_labels(tmp_path: Path) -> None:
    protocol = default_contextual_protocol()
    ids = _labels(0).drop(columns=["label_new_clinical_entry"])
    scores = ids.copy()
    scores["future_label"] = 0
    with pytest.raises(ValueError, match="label-like"):
        freeze_future_scores(ids, scores, {}, protocol, tmp_path)


def test_future_score_freeze_detects_tamper(tmp_path: Path) -> None:
    protocol = default_contextual_protocol()
    ids = _labels(0).drop(columns=["label_new_clinical_entry"])
    scores = ids.copy()
    scores["method"] = 0.2
    freeze_future_scores(ids, scores, {}, protocol, tmp_path)
    with (tmp_path / "future_configuration_freeze.json").open("a") as handle:
        handle.write("tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_future_score_freeze(tmp_path, protocol)
