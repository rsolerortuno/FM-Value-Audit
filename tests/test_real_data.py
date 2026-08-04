from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd

from fmva.real_data import (
    choose_model_cells,
    load_split_manifest,
    process_gse120575,
    read_gse115978_metadata,
    read_gse120575_metadata,
    reassemble_split_file,
)


def test_reassemble_v2_manifest(tmp_path: Path) -> None:
    content = b"abcdefghijk"
    part_one = tmp_path / "x.part001-of-002"
    part_two = tmp_path / "x.part002-of-002.bin"
    part_one.write_bytes(content[:5])
    part_two.write_bytes(content[5:])
    manifest = {
        "format_version": 2,
        "source": {
            "name": "x",
            "size": len(content),
            "md5Checksum": hashlib.md5(content).hexdigest(),
        },
        "parts": [
            {
                "name": "x.part001-of-002",
                "size": 5,
                "sha256": hashlib.sha256(content[:5]).hexdigest(),
            },
            {
                "name": "x.part002-of-002",
                "size": 6,
                "sha256": hashlib.sha256(content[5:]).hexdigest(),
            },
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = load_split_manifest(manifest_path)
    assert loaded.original_name == "x"
    output = tmp_path / "joined"
    audit = reassemble_split_file(manifest_path, tmp_path, output)
    assert output.read_bytes() == content
    assert audit["verified"] is True


def _write_gse120_metadata(path: Path) -> None:
    rows = [
        "SERIES\tignored\n",
        "SAMPLES\n",
        (
            "Sample name\ttitle\tsource name\torganism\t"
            "characteristics: patinet ID (Pre=baseline; Post= on treatment)\t"
            "characteristics: response\tcharacteristics: therapy\n"
        ),
        "Sample 1\tcell_a\tx\tHomo sapiens\tPre_P1\tResponder\tanti-PD1\n",
        "Sample 2\tcell_b\tx\tHomo sapiens\tPre_P2\tNon-responder\tanti-PD1\n",
        "PROTOCOLS\tshould not become a sample\n",
    ]
    with gzip.open(path, "wt", encoding="latin1") as handle:
        handle.writelines(rows)


def test_gse120575_parser_and_trailing_tab(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.txt.gz"
    _write_gse120_metadata(metadata_path)
    tpm_path = tmp_path / "matrix.txt.gz"
    with gzip.open(tpm_path, "wt", encoding="utf-8") as handle:
        handle.write("\tcell_a\tcell_b\n")
        handle.write("\tPre_P1\tPre_P2\n")
        handle.write("GENE1\t4\t0\t\n")
        handle.write("GENE2\t0\t8\t\n")
    metadata = read_gse120575_metadata(metadata_path)
    assert metadata["cell_id"].tolist() == ["cell_a", "cell_b"]
    features, summary = process_gse120575(
        tpm_path, metadata_path, tmp_path / "out", allowed_genes={"GENE1", "GENE2"}
    )
    assert summary.cells == 2
    assert features["gene_symbol"].tolist() == ["GENE1", "GENE2"]
    assert features["mean_tpm"].tolist() == [2.0, 4.0]


def test_cohort_aware_samples_and_subset(tmp_path: Path) -> None:
    annotation = pd.DataFrame(
        {
            "cells": ["a", "b", "c", "d"],
            "samples": ["Mel75", "Mel75", "Mel1", "Mel2"],
            "cell.types": ["Mal", "Mal", "T.CD8", "T.CD4"],
            "treatment.group": [
                "post.treatment",
                "treatment.naive",
                "post.treatment",
                "treatment.naive",
            ],
            "Cohort": ["New", "Tirosh", "New", "New"],
            "no.of.genes": [1, 1, 1, 1],
            "no.of.reads": [1, 1, 1, 1],
        }
    )
    path = tmp_path / "annotations.csv.gz"
    annotation.to_csv(path, index=False)
    parsed = read_gse115978_metadata(path)
    assert parsed.loc[0, "sample_key"] != parsed.loc[1, "sample_key"]
    selected = choose_model_cells(parsed, maximum=3, seed=1)
    assert len(selected) == 3
    assert selected["cell_id"].is_unique
