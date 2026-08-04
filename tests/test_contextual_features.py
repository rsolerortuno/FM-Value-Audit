from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fmva.contextual import (
    default_contextual_protocol,
    write_contextual_artifact,
    write_default_protocol,
)
from fmva.contextual_features import (
    build_contextual_gene_views,
    default_context_views,
    fit_development_contextual_pca,
    load_contextual_pair,
    write_contextual_gene_representation,
)


def _artifact(tmp_path: Path) -> Path:
    root = tmp_path / "artifact"
    index = pd.DataFrame(
        {
            "gene_id": ["A", "A", "B", "B", "C", "C"],
            "disease": ["melanoma", "nsclc", "melanoma", "nsclc", "melanoma", "nsclc"],
            "cell_compartment": [
                "immune",
                "malignant",
                "immune",
                "malignant",
                "immune",
                "malignant",
            ],
            "treatment_state": ["pre", "post", "pre", "post", "pre", "post"],
            "response_state": [
                "responder",
                "nonresponder",
                "responder",
                "nonresponder",
                "responder",
                "nonresponder",
            ],
            "n_cells": [4] * 6,
            "n_samples": [1] * 6,
        }
    )
    pretrained = np.arange(24, dtype=np.float32).reshape(6, 4)
    random = pretrained + 100.0
    protocol = default_contextual_protocol()
    protocol_path = tmp_path / "protocol.json"
    write_default_protocol(protocol_path)
    write_contextual_artifact(
        root,
        index=index,
        pretrained=pretrained,
        random=random,
        protocol_path=protocol_path,
        model_id="test_contextual",
        control_model_id="test_random",
        checkpoint_sha256="a" * 64,
        input_hashes={"input": "b" * 64},
        selected_cells=24,
        represented_samples=6,
        sequence_length=8,
        sampling_seed=protocol.sampling.seed,
        pooling="test",
        upstream_versions={"test": "1"},
    )
    return root


def test_default_context_views_are_fixed() -> None:
    assert [view.name for view in default_context_views()] == [
        "all",
        "melanoma",
        "nsclc",
        "pretreatment",
        "on_or_post_treatment",
        "responder",
        "nonresponder",
        "immune",
        "malignant",
    ]


def test_build_contextual_gene_views(tmp_path: Path) -> None:
    root = _artifact(tmp_path)
    genes, pretrained, random, metadata = build_contextual_gene_views(root)
    assert genes["target_gene"].tolist() == ["A", "B", "C"]
    assert pretrained.shape == random.shape
    assert pretrained.shape[0] == 3
    assert pretrained.shape[1] == 9 * (4 + 1)
    coverage = metadata["coverage_genes_by_view"]
    assert isinstance(coverage, dict)
    assert coverage["melanoma"] == 3
    assert not np.allclose(pretrained, random)


def test_load_contextual_pair_rejects_tamper(tmp_path: Path) -> None:
    root = _artifact(tmp_path)
    manifest = json.loads((root / "CONTEXTUAL_MANIFEST.json").read_text())
    path = root / manifest["pretrained_file"]
    with path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_contextual_pair(root)


def test_fit_contextual_pca_development_only() -> None:
    rng = np.random.default_rng(3)
    genes = pd.DataFrame({"target_gene": [f"G{i}" for i in range(30)]})
    matrix = rng.normal(size=(30, 20)).astype(np.float32)
    transformed, metadata = fit_development_contextual_pca(
        genes,
        matrix,
        {f"G{i}" for i in range(25)},
        components=5,
    )
    assert transformed.shape == (30, 5)
    assert metadata["fit_genes"] == 25
    assert metadata["pca_fit_scope"] == "development_genes_only"


def test_fit_contextual_pca_rejects_too_few_genes() -> None:
    genes = pd.DataFrame({"target_gene": ["A", "B"]})
    matrix = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="Insufficient"):
        fit_development_contextual_pca(genes, matrix, {"A", "B"}, components=2)


def test_write_contextual_gene_representation(tmp_path: Path) -> None:
    genes = pd.DataFrame({"target_gene": ["A", "B", "C"]})
    pretrained = np.ones((3, 2), dtype=np.float32)
    random = np.zeros((3, 2), dtype=np.float32)
    path, metadata_path = write_contextual_gene_representation(
        tmp_path / "representation.csv.gz",
        genes,
        pretrained,
        random,
        prefix="gf_context",
        metadata={"status": "TEST"},
    )
    frame = pd.read_csv(path)
    assert "gf_context_pc01" in frame
    assert "gf_context_random_pc02" in frame
    assert json.loads(metadata_path.read_text())["rows"] == 3
