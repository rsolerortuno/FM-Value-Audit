from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fmva.contextual_workflow import (
    TokenGeneAccumulator,
    broad_compartment,
    build_context_metadata,
    read_sample_context_partial,
    reduce_sample_context_partials,
    safe_partial_name,
    write_sample_context_partial,
)


def test_broad_compartment_mapping() -> None:
    values = pd.Series(["CD8 exhausted", "Treg", "NK", "Macrophage", "Tumor cells"])
    assert broad_compartment(values).tolist() == [
        "cd8_t",
        "cd4_t",
        "nk",
        "myeloid",
        "tumour",
    ]


def test_build_context_metadata_uses_official_lesion_response() -> None:
    obs = pd.DataFrame(
        {
            "sample": ["P1.post.1", "P1.post.3", "P2.pre"],
            "celltype": ["CD8", "CD4", "CD8"],
            "timepoint": ["post", "post", "pre"],
            "clinical_response": ["Responder", "Responder", "Responder"],
        },
        index=["c1", "c2", "c3"],
    )
    mapping = pd.DataFrame(
        {
            "dataset_sample": ["P1.post.1", "P1.post.3"],
            "lesion_response": ["Responder", "Non-responder"],
        }
    )

    result = build_context_metadata(obs, cohort="GSE179994", official_lesion_mapping=mapping)

    assert result["response_state"].tolist() == ["responder", "non_responder", "unknown"]
    assert result["cell_compartment"].tolist() == ["cd8_t", "cd4_t", "cd8_t"]
    assert result["disease"].unique().tolist() == ["nsclc"]


def test_token_gene_accumulator_means_tokens_and_counts_cells() -> None:
    accumulator = TokenGeneAccumulator.create(n_genes=3, width=2)
    genes = np.asarray([[0, 1, -1], [0, 2, -1]], dtype=np.int32)
    pretrained = np.asarray(
        [[[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]], [[3.0, 3.0], [4.0, 4.0], [0.0, 0.0]]],
        dtype=np.float32,
    )
    random = pretrained + 1.0

    accumulator.update(genes, pretrained, random)
    gene_indices, pretrained_mean, random_mean, cell_counts = accumulator.finalise()

    assert gene_indices.tolist() == [0, 1, 2]
    assert pretrained_mean[0].tolist() == [2.0, 2.0]
    assert random_mean[0].tolist() == [3.0, 3.0]
    assert cell_counts.tolist() == [2, 1, 1]


def test_partial_round_trip_and_sample_balanced_reduce(tmp_path: Path) -> None:
    paths = []
    for sample, offset in [("s1", 0.0), ("s2", 2.0)]:
        path = tmp_path / safe_partial_name(sample, "melanoma|cd8_t|post|responder")
        write_sample_context_partial(
            path,
            gene_indices=np.asarray([0, 1], dtype=np.int32),
            pretrained=np.asarray([[1.0 + offset, 2.0], [3.0, 4.0]], dtype=np.float32),
            random=np.asarray([[2.0 + offset, 3.0], [4.0, 5.0]], dtype=np.float32),
            cell_counts=np.asarray([2, 1], dtype=np.int32),
            metadata={
                "sample_id": sample,
                "disease": "melanoma",
                "cell_compartment": "cd8_t",
                "treatment_state": "post",
                "response_state": "responder",
            },
        )
        paths.append(path)

    genes, _pretrained, _random, cells, metadata = read_sample_context_partial(paths[0])
    assert genes.tolist() == [0, 1]
    assert metadata["sample_id"] == "s1"
    assert cells.tolist() == [2, 1]

    index, reduced_pretrained, reduced_random = reduce_sample_context_partials(
        paths, gene_symbols=["G1", "G2"], width=2
    )
    assert index["n_samples"].tolist() == [2, 2]
    assert index["n_cells"].tolist() == [4, 2]
    assert reduced_pretrained[0].tolist() == [2.0, 2.0]
    assert reduced_random[0].tolist() == [3.0, 3.0]
