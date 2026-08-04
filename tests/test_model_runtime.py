from __future__ import annotations

import numpy as np

from fmva.model_runtime import tokenize_geneformer, tokenize_scgpt


def test_geneformer_tokenisation_ranks_detected_genes() -> None:
    genes = ["A", "B", "C"]
    counts = np.asarray([[10.0, 1.0, 0.0]], dtype=np.float32)
    tokens, mask, coverage = tokenize_geneformer(
        genes,
        counts,
        {"A": "ENSA", "B": "ENSB"},
        {"ENSA": 5, "ENSB": 6},
        {"ENSA": 2.0, "ENSB": 1.0},
        4,
    )
    assert tokens[0, :2].tolist() == [5, 6]
    assert mask[0, :2].all()
    assert coverage == 2 / 3


def test_scgpt_tokenisation_has_cls_and_padding() -> None:
    genes = ["A", "B", "C"]
    counts = np.asarray([[0.0, 10.0, 2.0]], dtype=np.float32)
    vocab = {"<pad>": 0, "<cls>": 1, "A": 2, "B": 3, "C": 4}
    tokens, values, mask, coverage = tokenize_scgpt(genes, counts, vocab, 4, 5)
    assert tokens[0, 0] == 1
    assert tokens[0, 1] == 3
    assert mask[0, :3].all()
    assert values[0, 0] == 0.0
    assert coverage == 1.0


def test_geneformer_contextual_tokenisation_retains_gene_indices() -> None:
    from fmva.model_runtime import tokenize_geneformer_contextual

    genes = ["A", "B", "C"]
    counts = np.asarray([[1.0, 10.0, 0.0]], dtype=np.float32)
    tokens, mask, gene_indices, coverage = tokenize_geneformer_contextual(
        genes,
        counts,
        {"A": "ENSA", "B": "ENSB"},
        {"ENSA": 5, "ENSB": 6},
        {"ENSA": 1.0, "ENSB": 1.0},
        4,
    )

    assert tokens[0, :2].tolist() == [6, 5]
    assert gene_indices[0, :2].tolist() == [1, 0]
    assert mask[0, :2].all()
    assert coverage == 2 / 3


def test_scgpt_contextual_tokenisation_excludes_cls_from_gene_indices() -> None:
    from fmva.model_runtime import tokenize_scgpt_contextual

    genes = ["A", "B"]
    counts = np.asarray([[5.0, 1.0]], dtype=np.float32)
    vocab = {"<pad>": 0, "<cls>": 1, "A": 2, "B": 3}
    tokens, values, mask, gene_indices, coverage = tokenize_scgpt_contextual(
        genes, counts, vocab, 4, 5
    )

    assert tokens[0, 0] == 1
    assert gene_indices[0, 0] == -1
    assert gene_indices[0, 1:3].tolist() == [0, 1]
    assert mask[0, :3].all()
    assert values[0, 0] == 0.0
    assert coverage == 1.0
