from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from fmva.ood_proxy import (
    _fit_matched_ridge,
    _signed_top_overlap,
    build_nsclc_lesion_mapping,
    nsclc_post_lesion_matrix,
    safe_spearman,
    stable_gene_fold,
)


def _write_supplement(path: Path) -> None:
    rows = [
        ["Supplementary Table 1: Clinical Metadata for Patient Cohort"] + [None] * 8,
        [None] * 9,
        [None] * 9,
        [
            "Sample Name",
            "Patient",
            "Age",
            "Gender",
            "Tumor Type",
            "Treatment",
            "Response",
            "Treatment Hx",
            "Biospy Site",
        ],
        [
            "P001.pre.01",
            "P001",
            57,
            "Female",
            "LUAD",
            "Drug",
            "Yes",
            "Pre-treatment",
            "LN metastasis",
        ],
        ["P001.post.01", None, None, None, None, None, "Yes", "On treatment", "LN metastasis"],
        ["P001.post.02", None, None, None, None, None, "Yes", "On treatment", "LN metastasis"],
        ["P001.post.03", None, None, None, None, None, "No", "On treatment", "Right lung tumour"],
        [
            "P013.pre.01",
            "P013",
            61,
            "Male",
            "LUAD",
            "Drug",
            "Yes",
            "Pre-treatment",
            "Liver metastasis",
        ],
        ["P013.post.01", None, None, None, None, None, "Yes", "On treatment", "Liver metastasis"],
        ["P013.post.03", None, None, None, None, None, "No", "On treatment", "LN metastasis"],
    ]
    pd.DataFrame(rows).to_excel(path, sheet_name="Supplementary Table 1", header=False, index=False)


def test_stable_gene_fold_is_deterministic() -> None:
    assert stable_gene_fold("LAG3", 5) == stable_gene_fold("LAG3", 5)
    assert 0 <= stable_gene_fold("TIGIT", 5) < 5


def test_safe_spearman_handles_constant_vector() -> None:
    value = safe_spearman(np.ones(4), np.arange(4))
    assert np.isnan(value)
    assert np.isclose(safe_spearman(np.arange(5), np.arange(5)), 1.0)


def test_lesion_mapping_preserves_mixed_patients_and_reconciles_p13(tmp_path: Path) -> None:
    supplement = tmp_path / "supplement.xlsx"
    _write_supplement(supplement)
    metadata = pd.DataFrame(
        {
            "sample": [
                "P1.pre",
                "P1.post.1",
                "P1.post.2",
                "P1.post.3",
                "P13.pre",
                "P13.post.1",
                "P13.post.2",
            ],
            "patient": ["P1", "P1", "P1", "P1", "P13", "P13", "P13"],
            "timepoint": ["pre", "post", "post", "post", "pre", "post", "post"],
        }
    )
    metadata.to_csv(tmp_path / "metadata.csv", index=False)
    mapping = build_nsclc_lesion_mapping(supplement, tmp_path / "metadata.csv")

    p1 = mapping[mapping["patient"].eq("P1")]
    assert set(p1["lesion_response"]) == {"Responder", "Non-responder"}
    assert p1[p1["lesion_response"].eq("Responder")]["lesion_id"].nunique() == 1
    p13_fallback = mapping[mapping["supplement_sample"].eq("P013.post.03")].iloc[0]
    assert p13_fallback["dataset_sample"] == "P13.post.2"
    assert p13_fallback["mapping_status"] == "FALLBACK_UNIQUE_UNMATCHED_PATIENT_TIMEPOINT"


def test_nsclc_lesion_matrix_aggregates_repeated_biopsies(tmp_path: Path) -> None:
    supplement = tmp_path / "supplement.xlsx"
    _write_supplement(supplement)
    metadata = pd.DataFrame(
        {
            "sample": [
                "P1.pre",
                "P1.post.1",
                "P1.post.2",
                "P1.post.3",
                "P13.pre",
                "P13.post.1",
                "P13.post.2",
            ],
            "patient": ["P1", "P1", "P1", "P1", "P13", "P13", "P13"],
            "timepoint": ["pre", "post", "post", "post", "pre", "post", "post"],
        }
    )
    metadata.to_csv(tmp_path / "metadata.csv", index=False)
    pd.DataFrame({"gene_id": ["G1", "G2"]}).to_csv(tmp_path / "genes.csv", index=False)
    matrix = sparse.csr_matrix(
        np.asarray(
            [
                [0, 100, 300, 10, 0, 50, 5],
                [0, 0, 100, 50, 0, 20, 60],
            ],
            dtype=np.int64,
        )
    )
    sparse.save_npz(tmp_path / "pb.npz", matrix)

    genes, lesion_matrix, lesion_metadata, _ = nsclc_post_lesion_matrix(
        tmp_path / "pb.npz",
        tmp_path / "genes.csv",
        tmp_path / "metadata.csv",
        supplement,
    )
    assert genes == ["G1", "G2"]
    # P1 responsive duplicate biopsies collapse into one lesion; total is four lesions.
    assert lesion_matrix.shape == (2, 4)
    assert lesion_metadata["patient"].nunique() == 2
    assert set(lesion_metadata["lesion_response"]) == {"Responder", "Non-responder"}
    repeated = lesion_metadata[lesion_metadata["lesion_id"].eq("P1|ln metastasis")].iloc[0]
    assert repeated["n_biopsies"] == 2


def test_matched_ridge_finds_signal() -> None:
    rng = np.random.default_rng(3)
    features = rng.normal(size=(300, 5))
    labels = features[:, 0] * 2.0 - features[:, 1] + rng.normal(scale=0.05, size=300)
    genes = [f"G{i}" for i in range(300)]
    fitted = _fit_matched_ridge(features, labels, genes, (0.01, 1.0, 100.0), 5)
    assert fitted.alpha in {0.01, 1.0, 100.0}
    assert fitted.oof_spearman > 0.95


def test_signed_top_overlap_reports_direction() -> None:
    prediction = np.asarray([10.0, -9.0, 0.1, 0.0])
    truth = np.asarray([5.0, -4.0, 0.0, 0.1])
    overlap, direction = _signed_top_overlap(prediction, truth, maximum=2)
    assert overlap == 1.0
    assert direction == 1.0


def test_run_response_transfer_proxy_end_to_end(tmp_path: Path) -> None:
    import json
    import pickle

    import torch
    from safetensors.torch import save_file

    from fmva.ood_proxy import run_response_transfer_proxy

    rng = np.random.default_rng(11)
    n_genes = 180
    genes = [f"G{i}" for i in range(n_genes)]
    latent = rng.normal(size=n_genes)

    g120 = pd.DataFrame(
        {
            "gene_symbol": genes,
            "post_responder_vs_nonresponder_log2_delta": latent + rng.normal(0, 0.1, n_genes),
            "mean_tpm": np.abs(latent) + 1,
            "fraction_positive": rng.uniform(0.1, 1.0, n_genes),
        }
    )
    g115 = pd.DataFrame(
        {
            "gene_symbol": genes,
            "mean_raw_count_per_cell": np.abs(latent) + 0.5,
            "fraction_positive": rng.uniform(0.1, 1.0, n_genes),
            "post_vs_naive_sample_logcpm_delta": latent * 0.4,
            "cell_type_specificity_max_fraction": rng.uniform(0.2, 1.0, n_genes),
        }
    )
    g120.to_csv(tmp_path / "g120.csv", index=False)
    g115.to_csv(tmp_path / "g115.csv", index=False)

    supplement = tmp_path / "supplement.xlsx"
    _write_supplement(supplement)
    samples = [
        "P1.pre",
        "P1.post.1",
        "P1.post.2",
        "P1.post.3",
        "P13.pre",
        "P13.post.1",
        "P13.post.2",
    ]
    metadata = pd.DataFrame(
        {
            "sample": samples,
            "patient": ["P1", "P1", "P1", "P1", "P13", "P13", "P13"],
            "timepoint": ["pre", "post", "post", "post", "pre", "post", "post"],
        }
    )
    metadata.to_csv(tmp_path / "metadata.csv", index=False)
    pd.DataFrame({"gene_id": genes}).to_csv(tmp_path / "genes.csv", index=False)

    # Build counts with a weak lesion-response signal aligned to latent.
    base = rng.poisson(30, size=(n_genes, len(samples))).astype(np.int64)
    positive = latent > 0
    base[positive, 1] += 20
    base[positive, 2] += 20
    base[positive, 5] += 20
    base[~positive, 3] += 20
    base[~positive, 6] += 20
    sparse.save_npz(tmp_path / "pb.npz", sparse.csr_matrix(base))

    symbol_map = {gene: f"ENSG{i}" for i, gene in enumerate(genes)}
    token_dictionary = {f"ENSG{i}": i for i in range(n_genes)}
    with (tmp_path / "symbol.pkl").open("wb") as handle:
        pickle.dump(symbol_map, handle)
    with (tmp_path / "tokens.pkl").open("wb") as handle:
        pickle.dump(token_dictionary, handle)
    geneformer_weight = torch.tensor(
        np.column_stack([latent, rng.normal(size=(n_genes, 7))]), dtype=torch.float32
    )
    save_file(
        {"bert.embeddings.word_embeddings.weight": geneformer_weight},
        str(tmp_path / "geneformer.safetensors"),
    )

    vocab = {gene: i for i, gene in enumerate(genes)}
    (tmp_path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    scgpt_weight = torch.tensor(
        np.column_stack([latent * 0.2, rng.normal(size=(n_genes, 9))]), dtype=torch.float32
    )
    torch.save({"encoder.embedding.weight": scgpt_weight}, tmp_path / "scgpt.pt")

    output = tmp_path / "output"
    summary = run_response_transfer_proxy(
        gse115978_features=tmp_path / "g115.csv",
        gse120575_features=tmp_path / "g120.csv",
        gse179994_pseudobulk=tmp_path / "pb.npz",
        gse179994_gene_order=tmp_path / "genes.csv",
        gse179994_sample_metadata=tmp_path / "metadata.csv",
        gse179994_supplement=supplement,
        geneformer_model=tmp_path / "geneformer.safetensors",
        geneformer_symbol_map=tmp_path / "symbol.pkl",
        geneformer_token_dictionary=tmp_path / "tokens.pkl",
        scgpt_model=tmp_path / "scgpt.pt",
        scgpt_vocab=tmp_path / "vocab.json",
        output_directory=output,
        pca_components=4,
        n_folds=3,
        bootstrap_iterations=20,
        seed=4,
        alphas=(0.1, 1.0),
    )
    assert summary["status"] == "COMPUTED_RESPONSE_TRANSFER_PROXY"
    assert summary["nsclc_labeled_responder_lesions"] == 2
    assert summary["nsclc_labeled_nonresponder_lesions"] == 2
    assert summary["nsclc_mixed_response_patients"] == ["P1", "P13"]
    assert (output / "REPORT.md").is_file()
    assert (output / "ood_proxy_sensitivity_analyses.csv").is_file()
    strict = json.loads((output / "ood_proxy_method_results.json").read_text())
    assert len(strict) == 8
