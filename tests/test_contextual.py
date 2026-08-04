from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from fmva.cli import app
from fmva.contextual import (
    ContextualAuditStatus,
    default_contextual_protocol,
    deterministic_context_sample,
    deterministic_gene_fold,
    snapshot_dates,
    validate_contextual_artifact,
    write_contextual_artifact,
    write_default_protocol,
)


def test_default_protocol_is_frozen_and_primary_is_registered() -> None:
    protocol = default_contextual_protocol()

    assert protocol.information_cutoff == "2026-08-04"
    assert protocol.status.startswith("FROZEN_BEFORE")
    assert protocol.primary_contrast["method"] == "geneformer_contextual_plus_cheap"
    assert protocol.historical_snapshots == {
        "open_targets": "26.06",
        "chembl": "37",
        "clinicaltrials_version_endpoint": "https://clinicaltrials.gov/api/v2/version",
        "publication_cutoff": "2026-08-04",
    }
    assert protocol.sampling.sample_balanced_pooling is True


def test_deterministic_context_sample_is_order_invariant() -> None:
    frame = pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(12)],
            "sample": ["s1"] * 6 + ["s2"] * 6,
            "context": ["A"] * 12,
        }
    )
    left = deterministic_context_sample(
        frame,
        cell_id_column="cell_id",
        stratum_columns=["sample", "context"],
        maximum_per_stratum=3,
        minimum_per_stratum=2,
        seed=17,
    )
    right = deterministic_context_sample(
        frame.sample(frac=1.0, random_state=9),
        cell_id_column="cell_id",
        stratum_columns=["sample", "context"],
        maximum_per_stratum=3,
        minimum_per_stratum=2,
        seed=17,
    )

    assert left.equals(right)
    assert left.groupby("sample").size().to_dict() == {"s1": 3, "s2": 3}


def test_deterministic_context_sample_drops_underpowered_strata() -> None:
    frame = pd.DataFrame(
        {
            "cell_id": ["a", "b", "c"],
            "sample": ["s1", "s1", "s2"],
            "context": ["A", "A", "A"],
        }
    )
    selected = deterministic_context_sample(
        frame,
        cell_id_column="cell_id",
        stratum_columns=["sample", "context"],
        maximum_per_stratum=4,
        minimum_per_stratum=2,
        seed=1,
    )

    assert selected["sample"].unique().tolist() == ["s1"]


def test_gene_fold_is_stable() -> None:
    protocol = default_contextual_protocol()

    assert deterministic_gene_fold("ENSG00000141510", protocol) == deterministic_gene_fold(
        "ENSG00000141510", protocol
    )
    assert 0 <= deterministic_gene_fold("ENSG00000171862", protocol) < 5


def test_snapshot_schedule_is_quarterly_and_bounded() -> None:
    dates = list(snapshot_dates(default_contextual_protocol()))

    assert dates[0].isoformat() == "2027-02-04"
    assert dates[-1].isoformat() == "2028-08-04"
    assert len(dates) == 7


def _write_small_artifact(tmp_path: Path) -> tuple[Path, Path]:
    protocol_path = tmp_path / "protocol.json"
    write_default_protocol(protocol_path)
    index = pd.DataFrame(
        {
            "gene_id": ["G1", "G2"],
            "disease": ["melanoma", "melanoma"],
            "cell_compartment": ["CD8", "CD8"],
            "treatment_state": ["post", "post"],
            "response_state": ["responder", "responder"],
            "n_cells": [8, 8],
            "n_samples": [2, 2],
        }
    )
    root = tmp_path / "artifact"
    write_contextual_artifact(
        root,
        protocol_path=protocol_path,
        model_id="geneformer_contextual",
        control_model_id="geneformer_random_contextual",
        checkpoint_sha256="a" * 64,
        input_hashes={"counts": "b" * 64},
        index=index,
        pretrained=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        random=np.asarray([[2.0, 1.0], [4.0, 3.0]], dtype=np.float32),
        selected_cells=16,
        represented_samples=2,
        sequence_length=1024,
        sampling_seed=20260804,
        pooling="sample-balanced",
        upstream_versions={"torch": "test"},
    )
    return root, protocol_path


def test_contextual_artifact_round_trip(tmp_path: Path) -> None:
    root, protocol_path = _write_small_artifact(tmp_path)

    report = validate_contextual_artifact(root, protocol_path=protocol_path)

    assert report.status == ContextualAuditStatus.PASS
    assert report.summary["rows"] == 2
    assert report.errors == []


def test_contextual_artifact_tampering_fails(tmp_path: Path) -> None:
    root, protocol_path = _write_small_artifact(tmp_path)
    values = np.load(root / "pretrained_embeddings.npy", allow_pickle=False)
    values[0, 0] = 999.0
    np.save(root / "pretrained_embeddings.npy", values, allow_pickle=False)

    report = validate_contextual_artifact(root, protocol_path=protocol_path)

    assert report.status == ContextualAuditStatus.FAIL
    assert any("pretrained_hash" in error for error in report.errors)


def test_contextual_cli_protocol_and_validate(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    init = CliRunner().invoke(app, ["contextual-protocol-init", str(protocol_path)])
    assert init.exit_code == 0
    assert json.loads(protocol_path.read_text())["protocol_id"] in init.stdout

    root, _ = _write_small_artifact(tmp_path / "second")
    output = tmp_path / "audit.json"
    result = CliRunner().invoke(
        app,
        [
            "contextual-validate",
            str(root),
            "--protocol",
            str(tmp_path / "second" / "protocol.json"),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0
    assert result.stdout.strip() == "PASS"
    assert json.loads(output.read_text())["status"] == "PASS"
