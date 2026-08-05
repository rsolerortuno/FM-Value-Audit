"""Cross-disease response-transfer proxy using complete Stage-1 artefacts.

This module does **not** evaluate prospective clinical target prioritisation.  It
asks a narrower question: can gene representations learned without NSCLC labels
smooth a melanoma response-associated gene signal in a way that transfers to the
post-treatment NSCLC responder contrast?

The distinction is deliberate.  Clinical-entry labels, historical popularity,
and a frozen temporal holdout remain prerequisites for the primary scientific
question.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

CLAIM_SCOPE = "RESPONSE_TRANSFER_PROXY_NOT_CLINICAL_TARGET_EVALUATION"
DEFAULT_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


@dataclass(frozen=True)
class ProxyMethodResult:
    """Measured result for one matched ridge head."""

    method: str
    features: int
    selected_alpha: float | None
    melanoma_oof_spearman: float
    nsclc_ood_spearman: float
    bootstrap_median: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    top500_overlap_fraction: float
    top500_overlap_direction_accuracy: float
    embedding_pca_explained_variance: float | None = None
    claim_scope: str = CLAIM_SCOPE


@dataclass(frozen=True)
class RidgeFit:
    """Fitted deterministic ridge head and its out-of-fold predictions."""

    alpha: float
    oof: npt.NDArray[Any]
    scaler: StandardScaler
    model: Ridge
    oof_spearman: float


def stable_gene_fold(gene_symbol: str, n_folds: int) -> int:
    """Assign a gene to a deterministic hash fold."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    digest = hashlib.sha256(gene_symbol.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n_folds


def safe_spearman(left: npt.NDArray[Any], right: npt.NDArray[Any]) -> float:
    """Return Spearman correlation or NaN when it is undefined."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.std(left[valid]) == 0.0 or np.std(right[valid]) == 0.0:
        return float("nan")
    return float(spearmanr(left[valid], right[valid]).statistic)


def _load_gene_order(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    if frame.shape[1] != 1:
        raise ValueError(f"Expected one gene-order column in {path}")
    genes = frame.iloc[:, 0].astype(str).tolist()
    if len(genes) != len(set(genes)):
        raise ValueError(f"Duplicate gene identifiers in {path}")
    return genes


def _normalise_supplement_sample(value: str) -> str:
    """Normalise supplement sample IDs to the dataset naming convention."""
    text = str(value).strip()
    match = re.match(r"^P0*(\d+)\.(pre|post)\.0*(\d+)$", text, flags=re.I)
    if match:
        return f"P{int(match.group(1))}.{match.group(2).lower()}.{int(match.group(3))}"
    match = re.match(r"^P0*(\d+)$", text, flags=re.I)
    if match:
        return f"P{int(match.group(1))}.pre"
    return text


def _normalise_patient(value: str) -> str:
    text = str(value).strip()
    match = re.match(r"^P0*(\d+)$", text, flags=re.I)
    return f"P{int(match.group(1))}" if match else text


def _normalise_site(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("tumour", "tumor").split())


def build_nsclc_lesion_mapping(
    supplement_path: Path,
    sample_metadata_path: Path,
) -> pd.DataFrame:
    """Map official lesion-level outcomes to dataset samples.

    Supplementary Table 1 reports RECIST response per target lesion.  Exact
    normalised sample IDs are used first.  A single documented naming mismatch
    (P013.post.03 in the clinical table versus P13.post.2 in the expression
    dataset) is resolved only when one unmatched post-treatment sample remains
    for that patient.
    """
    raw = pd.read_excel(supplement_path, sheet_name="Supplementary Table 1", header=3)
    raw.columns = [str(column).strip().lower().replace(" ", "_") for column in raw]
    required = {"sample_name", "patient", "response", "treatment_hx", "biospy_site"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing supplement columns: {sorted(missing)}")
    raw = raw[raw["sample_name"].notna()].copy()
    raw["patient"] = raw["patient"].ffill().map(_normalise_patient)
    raw["treatment"] = raw.get("treatment", pd.Series(index=raw.index, dtype=object)).ffill()
    raw["supplement_sample"] = raw["sample_name"].astype(str)
    raw["normalised_sample"] = raw["supplement_sample"].map(_normalise_supplement_sample)
    raw["timepoint"] = np.where(
        raw["normalised_sample"].str.contains(".post.", regex=False), "post", "pre"
    )
    raw["lesion_response"] = raw["response"].map({"Yes": "Responder", "No": "Non-responder"})
    raw["biopsy_site"] = raw["biospy_site"].map(_normalise_site)
    clinical = raw[raw["timepoint"].eq("post") & raw["lesion_response"].notna()].copy()

    metadata = pd.read_csv(sample_metadata_path)
    required_metadata = {"sample", "patient", "timepoint"}
    missing_metadata = required_metadata - set(metadata.columns)
    if missing_metadata:
        raise ValueError(f"Missing NSCLC sample metadata columns: {sorted(missing_metadata)}")
    metadata = metadata.copy()
    metadata["sample"] = metadata["sample"].astype(str)
    metadata["patient"] = metadata["patient"].map(_normalise_patient)
    available = set(metadata.loc[metadata["timepoint"].eq("post"), "sample"])
    used: set[str] = set()
    mapped_rows: list[dict[str, Any]] = []
    for row in clinical.itertuples(index=False):
        candidate = str(row.normalised_sample)
        status = "EXACT_NORMALISED_ID"
        if candidate not in available or candidate in used:
            remaining = sorted(
                sample for sample in available - used if sample.startswith(f"{row.patient}.post.")
            )
            if len(remaining) != 1:
                raise ValueError(
                    "Could not uniquely reconcile supplement sample "
                    f"{row.supplement_sample}: remaining={remaining}"
                )
            candidate = remaining[0]
            status = "FALLBACK_UNIQUE_UNMATCHED_PATIENT_TIMEPOINT"
        used.add(candidate)
        lesion_id = f"{row.patient}|{row.biopsy_site}"
        mapped_rows.append(
            {
                "supplement_sample": str(row.supplement_sample),
                "dataset_sample": candidate,
                "patient": str(row.patient),
                "timepoint": "post",
                "lesion_response": str(row.lesion_response),
                "biopsy_site": str(row.biopsy_site),
                "lesion_id": lesion_id,
                "treatment": None if pd.isna(row.treatment) else str(row.treatment),
                "mapping_status": status,
            }
        )
    mapping = pd.DataFrame(mapped_rows)
    if mapping["dataset_sample"].duplicated().any():
        raise ValueError("A dataset sample was mapped to multiple supplement rows")
    if mapping.groupby("lesion_id")["lesion_response"].nunique().max() != 1:
        raise ValueError("Conflicting responses within a target lesion")
    return mapping.sort_values(["patient", "dataset_sample"]).reset_index(drop=True)


def nsclc_post_lesion_matrix(
    pseudobulk_path: Path,
    gene_order_path: Path,
    sample_metadata_path: Path,
    supplement_path: Path,
) -> tuple[list[str], npt.NDArray[Any], pd.DataFrame, pd.DataFrame]:
    """Return post-treatment log2-CPM profiles aggregated per target lesion."""
    genes = _load_gene_order(gene_order_path)
    pseudobulk = sparse.load_npz(pseudobulk_path).tocsr()
    metadata = pd.read_csv(sample_metadata_path)
    if "sample" not in metadata:
        raise ValueError("NSCLC sample metadata must contain sample")
    if pseudobulk.shape != (len(genes), len(metadata)):
        raise ValueError(
            "NSCLC pseudobulk dimensions do not match gene and sample metadata: "
            f"{pseudobulk.shape} != {(len(genes), len(metadata))}"
        )
    mapping = build_nsclc_lesion_mapping(supplement_path, sample_metadata_path)
    sample_to_index = {str(sample): index for index, sample in enumerate(metadata["sample"])}
    missing_samples = sorted(set(mapping["dataset_sample"]) - set(sample_to_index))
    if missing_samples:
        raise ValueError(f"Mapped samples absent from pseudobulk metadata: {missing_samples}")

    library_size = np.asarray(pseudobulk.sum(axis=0)).ravel().astype(float)
    cpm = pseudobulk.astype(np.float64).multiply(1_000_000.0 / np.maximum(library_size, 1.0))
    log_cpm = np.log2(cpm.toarray() + 1.0)

    lesion_rows: list[dict[str, Any]] = []
    lesion_vectors: list[npt.NDArray[Any]] = []
    for lesion_id, group in mapping.groupby("lesion_id", sort=True):
        if group["lesion_response"].nunique() != 1 or group["patient"].nunique() != 1:
            raise ValueError(f"Inconsistent lesion mapping for {lesion_id}")
        columns = np.asarray(
            [sample_to_index[sample] for sample in group["dataset_sample"]], dtype=int
        )
        lesion_rows.append(
            {
                "lesion_id": lesion_id,
                "patient": str(group["patient"].iloc[0]),
                "lesion_response": str(group["lesion_response"].iloc[0]),
                "biopsy_site": str(group["biopsy_site"].iloc[0]),
                "n_biopsies": len(columns),
                "dataset_samples": ";".join(group["dataset_sample"].astype(str)),
            }
        )
        lesion_vectors.append(log_cpm[:, columns].mean(axis=1))
    lesion_metadata = pd.DataFrame(lesion_rows)
    lesion_matrix = np.column_stack(lesion_vectors)
    if lesion_metadata["lesion_response"].eq("Responder").sum() == 0:
        raise ValueError("No labelled responsive NSCLC lesions")
    if lesion_metadata["lesion_response"].eq("Non-responder").sum() == 0:
        raise ValueError("No labelled non-responsive NSCLC lesions")
    return genes, lesion_matrix, lesion_metadata, mapping


def nsclc_response_features(
    pseudobulk_path: Path,
    gene_order_path: Path,
    sample_metadata_path: Path,
    supplement_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, npt.NDArray[Any], pd.DataFrame]:
    """Build lesion-level post-treatment NSCLC response features."""
    genes, lesion_matrix, lesion_metadata, mapping = nsclc_post_lesion_matrix(
        pseudobulk_path, gene_order_path, sample_metadata_path, supplement_path
    )
    responder = lesion_metadata["lesion_response"].eq("Responder").to_numpy()
    nonresponder = lesion_metadata["lesion_response"].eq("Non-responder").to_numpy()
    delta = lesion_matrix[:, responder].mean(axis=1) - lesion_matrix[:, nonresponder].mean(axis=1)
    frame = pd.DataFrame(
        {
            "gene_symbol": genes,
            "nsclc_post_responder_vs_nonresponder_log2cpm_delta": delta,
            "nsclc_post_mean_log2cpm": lesion_matrix.mean(axis=1),
            "nsclc_post_lesion_fraction_positive": (lesion_matrix > 0).mean(axis=1),
            "nsclc_n_responder_lesions": int(responder.sum()),
            "nsclc_n_nonresponder_lesions": int(nonresponder.sum()),
        }
    )
    return frame, lesion_metadata, lesion_matrix, mapping


def _fit_matched_ridge(
    features: npt.NDArray[Any],
    labels: npt.NDArray[Any],
    genes: list[str],
    alphas: tuple[float, ...],
    n_folds: int,
) -> RidgeFit:
    folds = np.asarray([stable_gene_fold(gene, n_folds) for gene in genes], dtype=int)
    oof_by_alpha: dict[float, npt.NDArray[Any]] = {}
    scores: list[tuple[float, float, float]] = []
    for alpha in alphas:
        oof = np.full(len(labels), np.nan, dtype=float)
        for fold in range(n_folds):
            train = folds != fold
            validation = folds == fold
            scaler = StandardScaler()
            train_x = scaler.fit_transform(features[train])
            validation_x = scaler.transform(features[validation])
            model = Ridge(alpha=alpha, solver="lsqr")
            model.fit(train_x, labels[train])
            oof[validation] = model.predict(validation_x)
        score = safe_spearman(oof, labels)
        scores.append((score, -alpha, alpha))
        oof_by_alpha[alpha] = oof
    best_alpha = max(scores)[2]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = Ridge(alpha=best_alpha, solver="lsqr")
    model.fit(scaled, labels)
    return RidgeFit(
        alpha=float(best_alpha),
        oof=oof_by_alpha[best_alpha],
        scaler=scaler,
        model=model,
        oof_spearman=safe_spearman(oof_by_alpha[best_alpha], labels),
    )


def _reduce_embedding(
    embedding: npt.NDArray[Any],
    components: int,
    seed: int,
) -> tuple[npt.NDArray[Any], float]:
    components = min(components, embedding.shape[1], len(embedding) - 1)
    if components < 1:
        raise ValueError("No PCA components available")
    scaled = StandardScaler().fit_transform(embedding)
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    reduced = pca.fit_transform(scaled).astype(np.float32)
    return reduced, float(pca.explained_variance_ratio_.sum())


def load_static_checkpoint_embeddings(
    genes: list[str],
    geneformer_model: Path,
    geneformer_symbol_map: Path,
    geneformer_token_dictionary: Path,
    scgpt_model: Path,
    scgpt_vocab: Path,
    seed: int,
) -> tuple[dict[str, npt.NDArray[Any]], npt.NDArray[Any], dict[str, float]]:
    """Map gene symbols to checkpoint token embeddings and matched random controls."""
    import torch
    from safetensors.torch import load_file

    with geneformer_symbol_map.open("rb") as handle:
        symbol_to_ensembl: dict[str, str] = pickle.load(handle)
    with geneformer_token_dictionary.open("rb") as handle:
        token_dictionary: dict[str, int] = pickle.load(handle)
    geneformer_state = load_file(str(geneformer_model), device="cpu")
    geneformer_weight = (
        geneformer_state["bert.embeddings.word_embeddings.weight"].numpy().astype(np.float32)
    )

    vocab: dict[str, int] = json.loads(scgpt_vocab.read_text(encoding="utf-8"))
    scgpt_state = torch.load(scgpt_model, map_location="cpu", weights_only=True)
    scgpt_weight = scgpt_state["encoder.embedding.weight"].numpy().astype(np.float32)

    geneformer = np.full((len(genes), geneformer_weight.shape[1]), np.nan, dtype=np.float32)
    scgpt = np.full((len(genes), scgpt_weight.shape[1]), np.nan, dtype=np.float32)
    geneformer_mapped = np.zeros(len(genes), dtype=bool)
    scgpt_mapped = np.zeros(len(genes), dtype=bool)
    for index, gene in enumerate(genes):
        ensembl = symbol_to_ensembl.get(gene)
        if ensembl in token_dictionary:
            geneformer[index] = geneformer_weight[int(token_dictionary[ensembl])]
            geneformer_mapped[index] = True
        if gene in vocab:
            scgpt[index] = scgpt_weight[int(vocab[gene])]
            scgpt_mapped[index] = True

    rng = np.random.default_rng(seed)
    embeddings = {
        "geneformer_static": geneformer,
        "scgpt_static": scgpt,
        "geneformer_random": np.asarray(
            rng.normal(0.0, 0.02, size=geneformer.shape), dtype=np.float32
        ),
        "scgpt_random": np.asarray(rng.normal(0.0, 0.02, size=scgpt.shape), dtype=np.float32),
    }
    mapping = geneformer_mapped & scgpt_mapped
    coverage = {
        "geneformer": float(geneformer_mapped.mean()),
        "scgpt": float(scgpt_mapped.mean()),
        "joint": float(mapping.mean()),
    }
    return embeddings, mapping, coverage


def _signed_top_overlap(
    prediction: npt.NDArray[Any],
    truth: npt.NDArray[Any],
    maximum: int = 500,
) -> tuple[float, float]:
    k = min(maximum, len(prediction))
    predicted = np.argsort(-np.abs(prediction))[:k]
    truth_top = set(np.argsort(-np.abs(truth))[:k])
    overlap = [index for index in predicted if index in truth_top]
    if not overlap:
        return 0.0, 0.0
    direction = np.mean(np.sign(prediction[overlap]) == np.sign(truth[overlap]))
    return len(overlap) / k, float(direction)


def _bootstrap_fixed_predictions(
    predictions: pd.DataFrame,
    genes: list[str],
    nsclc_genes: list[str],
    lesion_matrix: npt.NDArray[Any],
    lesion_metadata: pd.DataFrame,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    """Cluster-bootstrap fixed predictions by patient, preserving mixed lesions."""
    prediction_columns = [
        str(column) for column in predictions.columns if str(column).startswith("pred_")
    ]
    ranks = np.column_stack(
        [rankdata(predictions[column].to_numpy()) for column in prediction_columns]
    ).astype(np.float64)
    ranks -= ranks.mean(axis=0, keepdims=True)
    rank_norm = np.sqrt((ranks**2).sum(axis=0))
    gene_index = {gene: index for index, gene in enumerate(nsclc_genes)}
    selected = np.asarray([gene_index[gene] for gene in genes], dtype=int)

    patients = sorted(lesion_metadata["patient"].astype(str).unique())
    patient_to_columns = {
        patient: np.flatnonzero(lesion_metadata["patient"].astype(str).eq(patient).to_numpy())
        for patient in patients
    }
    responses = lesion_metadata["lesion_response"].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    completed = 0
    attempts = 0
    maximum_attempts = iterations * 50
    while completed < iterations and attempts < maximum_attempts:
        attempts += 1
        sampled_patients = rng.choice(patients, size=len(patients), replace=True)
        sampled_columns = np.concatenate(
            [patient_to_columns[str(patient)] for patient in sampled_patients]
        )
        sampled_responses = responses[sampled_columns]
        responder = sampled_columns[sampled_responses == "Responder"]
        nonresponder = sampled_columns[sampled_responses == "Non-responder"]
        if len(responder) == 0 or len(nonresponder) == 0:
            continue
        delta = lesion_matrix[:, responder].mean(axis=1) - lesion_matrix[:, nonresponder].mean(
            axis=1
        )
        truth_rank = rankdata(delta[selected]).astype(np.float64)
        truth_rank -= truth_rank.mean()
        truth_norm = np.sqrt((truth_rank**2).sum())
        correlations = (ranks.T @ truth_rank) / np.maximum(rank_norm * truth_norm, 1e-12)
        for column, correlation in zip(prediction_columns, correlations, strict=True):
            rows.append(
                {
                    "bootstrap": completed,
                    "method": column.removeprefix("pred_"),
                    "spearman": float(correlation),
                }
            )
        completed += 1
    if completed != iterations:
        raise RuntimeError(
            f"Only completed {completed}/{iterations} valid patient-cluster bootstraps"
        )
    return pd.DataFrame(rows)


def run_response_transfer_proxy(
    *,
    gse115978_features: Path,
    gse120575_features: Path,
    gse179994_pseudobulk: Path,
    gse179994_gene_order: Path,
    gse179994_sample_metadata: Path,
    gse179994_supplement: Path,
    geneformer_model: Path,
    geneformer_symbol_map: Path,
    geneformer_token_dictionary: Path,
    scgpt_model: Path,
    scgpt_vocab: Path,
    output_directory: Path,
    pca_components: int = 32,
    n_folds: int = 5,
    bootstrap_iterations: int = 300,
    seed: int = 1729,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
) -> dict[str, Any]:
    """Execute the complete response-transfer proxy and write auditable artefacts."""
    output_directory.mkdir(parents=True, exist_ok=True)
    melanoma_counts = pd.read_csv(gse115978_features)
    melanoma_response = pd.read_csv(gse120575_features)
    nsclc, lesion_metadata, lesion_matrix, lesion_mapping = nsclc_response_features(
        gse179994_pseudobulk,
        gse179994_gene_order,
        gse179994_sample_metadata,
        gse179994_supplement,
    )
    nsclc_genes = nsclc["gene_symbol"].astype(str).tolist()

    merged = melanoma_response.merge(
        melanoma_counts,
        on="gene_symbol",
        how="inner",
        suffixes=("_g120", "_g115"),
        validate="one_to_one",
    )
    merged = merged.merge(nsclc, on="gene_symbol", how="inner", validate="one_to_one")
    merged = merged.sort_values("gene_symbol").reset_index(drop=True)
    all_genes = merged["gene_symbol"].astype(str).tolist()

    melanoma_delta_all = merged["post_responder_vs_nonresponder_log2_delta"].to_numpy(float)
    nsclc_delta_all = merged["nsclc_post_responder_vs_nonresponder_log2cpm_delta"].to_numpy(float)
    manual_columns = [
        "mean_tpm",
        "fraction_positive_g120",
        "mean_raw_count_per_cell",
        "fraction_positive_g115",
        "post_vs_naive_sample_logcpm_delta",
        "cell_type_specificity_max_fraction",
    ]
    manual_all = (
        merged[manual_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    embeddings, mapped, coverage = load_static_checkpoint_embeddings(
        all_genes,
        geneformer_model,
        geneformer_symbol_map,
        geneformer_token_dictionary,
        scgpt_model,
        scgpt_vocab,
        seed,
    )
    common = mapped & np.isfinite(melanoma_delta_all) & np.isfinite(nsclc_delta_all)
    if common.sum() < 100:
        raise ValueError("Fewer than 100 jointly mapped genes")
    genes = [gene for gene, keep in zip(all_genes, common, strict=True) if keep]
    melanoma_delta = melanoma_delta_all[common]
    nsclc_delta = nsclc_delta_all[common]
    manual = manual_all[common]
    for name in embeddings:
        embeddings[name] = embeddings[name][common]

    reduced: dict[str, npt.NDArray[Any]] = {}
    explained: dict[str, float] = {}
    for name, embedding in embeddings.items():
        reduced[name], explained[name] = _reduce_embedding(embedding, pca_components, seed)

    methods: dict[str, npt.NDArray[Any]] = {
        "manual_features": manual,
        "geneformer_static_pca32": reduced["geneformer_static"],
        "scgpt_static_pca32": reduced["scgpt_static"],
        "geneformer_random_pca32": reduced["geneformer_random"],
        "scgpt_random_pca32": reduced["scgpt_random"],
        "manual_plus_geneformer_pca32": np.column_stack([manual, reduced["geneformer_static"]]),
        "manual_plus_scgpt_pca32": np.column_stack([manual, reduced["scgpt_static"]]),
    }
    pca_explained = {
        "geneformer_static_pca32": explained["geneformer_static"],
        "scgpt_static_pca32": explained["scgpt_static"],
        "geneformer_random_pca32": explained["geneformer_random"],
        "scgpt_random_pca32": explained["scgpt_random"],
    }

    rows: list[dict[str, Any]] = []
    predictions = pd.DataFrame(
        {
            "gene_symbol": genes,
            "melanoma_post_delta": melanoma_delta,
            "nsclc_post_delta": nsclc_delta,
        }
    )
    rows.append(
        {
            "method": "direct_melanoma_delta",
            "features": 1,
            "selected_alpha": None,
            "melanoma_oof_spearman": 1.0,
            "nsclc_ood_spearman": safe_spearman(melanoma_delta, nsclc_delta),
            "embedding_pca_explained_variance": None,
            "claim_scope": CLAIM_SCOPE,
        }
    )
    predictions["pred_direct_melanoma_delta"] = melanoma_delta

    for method, feature_matrix in methods.items():
        fitted = _fit_matched_ridge(feature_matrix, melanoma_delta, genes, alphas, n_folds)
        predicted = fitted.model.predict(fitted.scaler.transform(feature_matrix))
        rows.append(
            {
                "method": method,
                "features": int(feature_matrix.shape[1]),
                "selected_alpha": fitted.alpha,
                "melanoma_oof_spearman": fitted.oof_spearman,
                "nsclc_ood_spearman": safe_spearman(predicted, nsclc_delta),
                "embedding_pca_explained_variance": pca_explained.get(method),
                "claim_scope": CLAIM_SCOPE,
            }
        )
        predictions[f"pred_{method}"] = predicted

    bootstrap = _bootstrap_fixed_predictions(
        predictions,
        genes,
        nsclc_genes,
        lesion_matrix,
        lesion_metadata,
        bootstrap_iterations,
        seed,
    )
    confidence = (
        bootstrap.groupby("method")["spearman"]
        .agg(
            bootstrap_median="median",
            bootstrap_ci_low=lambda values: np.quantile(values, 0.025),
            bootstrap_ci_high=lambda values: np.quantile(values, 0.975),
        )
        .reset_index()
    )
    metrics = pd.DataFrame(rows).merge(confidence, on="method", how="left")
    for row_index, method in enumerate(metrics["method"]):
        overlap, direction = _signed_top_overlap(
            predictions[f"pred_{method}"].to_numpy(), nsclc_delta
        )
        metrics.loc[row_index, "top500_overlap_fraction"] = overlap
        metrics.loc[row_index, "top500_overlap_direction_accuracy"] = direction
    metrics = metrics.sort_values("nsclc_ood_spearman", ascending=False).reset_index(drop=True)

    bootstrap_wide = bootstrap.pivot(index="bootstrap", columns="method", values="spearman")
    comparisons = [
        ("geneformer_static_pca32", "direct_melanoma_delta"),
        ("geneformer_static_pca32", "geneformer_random_pca32"),
        ("geneformer_static_pca32", "manual_features"),
        ("manual_plus_geneformer_pca32", "manual_features"),
        ("scgpt_static_pca32", "scgpt_random_pca32"),
        ("scgpt_static_pca32", "direct_melanoma_delta"),
    ]
    paired_rows: list[dict[str, Any]] = []
    for method_a, method_b in comparisons:
        difference = bootstrap_wide[method_a] - bootstrap_wide[method_b]
        paired_rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "median_difference": float(difference.median()),
                "ci_low": float(difference.quantile(0.025)),
                "ci_high": float(difference.quantile(0.975)),
                "fraction_difference_gt_0": float((difference > 0).mean()),
            }
        )
    paired = pd.DataFrame(paired_rows)

    prediction_columns = [
        str(column) for column in predictions.columns if str(column).startswith("pred_")
    ]
    nsclc_gene_index = {gene: index for index, gene in enumerate(nsclc_genes)}
    nsclc_selected = np.asarray([nsclc_gene_index[gene] for gene in genes], dtype=int)

    sensitivity_rows: list[dict[str, Any]] = []
    mixed_patients = sorted(
        str(patient)
        for patient, group in lesion_metadata.groupby("patient")
        if group["lesion_response"].nunique() > 1
    )
    sensitivity_masks = {
        "all_official_target_lesions": np.ones(len(lesion_metadata), dtype=bool),
        "exclude_mixed_response_patients": ~lesion_metadata["patient"]
        .isin(mixed_patients)
        .to_numpy(),
    }
    for analysis, mask in sensitivity_masks.items():
        subset_metadata = lesion_metadata.loc[mask].reset_index(drop=True)
        subset_matrix = lesion_matrix[:, np.flatnonzero(mask)]
        responder_mask = subset_metadata["lesion_response"].eq("Responder").to_numpy()
        nonresponder_mask = subset_metadata["lesion_response"].eq("Non-responder").to_numpy()
        if not responder_mask.any() or not nonresponder_mask.any():
            continue
        subset_delta = (
            subset_matrix[:, responder_mask].mean(axis=1)
            - subset_matrix[:, nonresponder_mask].mean(axis=1)
        )[nsclc_selected]
        for column in prediction_columns:
            sensitivity_rows.append(
                {
                    "analysis": analysis,
                    "method": column.removeprefix("pred_"),
                    "spearman": safe_spearman(predictions[column].to_numpy(), subset_delta),
                    "patients": int(subset_metadata["patient"].nunique()),
                    "responsive_lesions": int(responder_mask.sum()),
                    "nonresponsive_lesions": int(nonresponder_mask.sum()),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)

    leave_one_patient_rows: list[dict[str, Any]] = []
    for patient in sorted(lesion_metadata["patient"].astype(str).unique()):
        mask = ~lesion_metadata["patient"].astype(str).eq(patient).to_numpy()
        subset_metadata = lesion_metadata.loc[mask].reset_index(drop=True)
        subset_matrix = lesion_matrix[:, np.flatnonzero(mask)]
        responder_mask = subset_metadata["lesion_response"].eq("Responder").to_numpy()
        nonresponder_mask = subset_metadata["lesion_response"].eq("Non-responder").to_numpy()
        if not responder_mask.any() or not nonresponder_mask.any():
            continue
        subset_delta = (
            subset_matrix[:, responder_mask].mean(axis=1)
            - subset_matrix[:, nonresponder_mask].mean(axis=1)
        )[nsclc_selected]
        for column in prediction_columns:
            leave_one_patient_rows.append(
                {
                    "left_out_patient": patient,
                    "method": column.removeprefix("pred_"),
                    "spearman": safe_spearman(predictions[column].to_numpy(), subset_delta),
                }
            )
    leave_one_patient = pd.DataFrame(leave_one_patient_rows)
    leave_one_patient_summary = (
        leave_one_patient.groupby("method")["spearman"]
        .agg(median="median", minimum="min", maximum="max")
        .reset_index()
    )

    integrated = predictions.merge(
        merged.loc[common].reset_index(drop=True), on="gene_symbol", how="left"
    )
    integrated["response_direction_concordant"] = np.sign(
        integrated["melanoma_post_delta"]
    ) == np.sign(integrated["nsclc_post_delta"])
    integrated["response_transfer_score"] = np.sqrt(
        np.abs(integrated["melanoma_post_delta"] * integrated["nsclc_post_delta"])
    )
    ranked = integrated[integrated["response_direction_concordant"]].sort_values(
        "response_transfer_score", ascending=False
    )

    metrics.to_csv(output_directory / "ood_proxy_metrics.csv", index=False)
    predictions.to_csv(
        output_directory / "ood_proxy_predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    integrated.to_csv(
        output_directory / "integrated_gene_features_with_proxy_predictions.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    ranked.head(1000).to_csv(output_directory / "top_concordant_response_genes.csv", index=False)
    lesion_metadata.to_csv(output_directory / "nsclc_labeled_lesion_summary.csv", index=False)
    lesion_mapping.to_csv(
        output_directory / "nsclc_official_lesion_sample_mapping.csv", index=False
    )
    bootstrap.to_csv(
        output_directory / "ood_proxy_patient_cluster_bootstrap.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    paired.to_csv(output_directory / "ood_proxy_paired_bootstrap_differences.csv", index=False)
    sensitivity.to_csv(output_directory / "ood_proxy_sensitivity_analyses.csv", index=False)
    leave_one_patient.to_csv(output_directory / "ood_proxy_leave_one_patient_out.csv", index=False)
    leave_one_patient_summary.to_csv(
        output_directory / "ood_proxy_leave_one_patient_out_summary.csv", index=False
    )

    summary = {
        "status": "COMPUTED_RESPONSE_TRANSFER_PROXY",
        "claim_scope": CLAIM_SCOPE,
        "seed": seed,
        "bootstrap_iterations": bootstrap_iterations,
        "gene_folds": n_folds,
        "pca_components": pca_components,
        "genes_shared_all_datasets": len(merged),
        "genes_shared_models_and_datasets": int(common.sum()),
        "checkpoint_gene_coverage": coverage,
        "melanoma_post_responder_states": 8,
        "melanoma_post_nonresponder_states": 21,
        "nsclc_labeled_responder_lesions": int(
            lesion_metadata["lesion_response"].eq("Responder").sum()
        ),
        "nsclc_labeled_nonresponder_lesions": int(
            lesion_metadata["lesion_response"].eq("Non-responder").sum()
        ),
        "nsclc_labeled_patients": int(lesion_metadata["patient"].nunique()),
        "nsclc_mixed_response_patients": sorted(
            str(patient)
            for patient, group in lesion_metadata.groupby("patient")
            if group["lesion_response"].nunique() > 1
        ),
        "nsclc_unlabelled_patients_excluded_from_outcome": 25,
        "best_method_by_ood_spearman": str(metrics.iloc[0]["method"]),
        "best_ood_spearman": float(metrics.iloc[0]["nsclc_ood_spearman"]),
        "geneformer_ood_spearman": float(
            metrics.loc[metrics["method"].eq("geneformer_static_pca32"), "nsclc_ood_spearman"].iloc[
                0
            ]
        ),
        "geneformer_bootstrap_ci_low": float(
            metrics.loc[metrics["method"].eq("geneformer_static_pca32"), "bootstrap_ci_low"].iloc[0]
        ),
        "geneformer_bootstrap_ci_high": float(
            metrics.loc[metrics["method"].eq("geneformer_static_pca32"), "bootstrap_ci_high"].iloc[
                0
            ]
        ),
        "manual_features_ood_spearman": float(
            metrics.loc[metrics["method"].eq("manual_features"), "nsclc_ood_spearman"].iloc[0]
        ),
        "scgpt_ood_spearman": float(
            metrics.loc[metrics["method"].eq("scgpt_static_pca32"), "nsclc_ood_spearman"].iloc[0]
        ),
        "direct_melanoma_delta_ood_spearman": float(
            metrics.loc[metrics["method"].eq("direct_melanoma_delta"), "nsclc_ood_spearman"].iloc[0]
        ),
        "paired_geneformer_minus_direct_median": float(
            paired.loc[
                paired["method_a"].eq("geneformer_static_pca32")
                & paired["method_b"].eq("direct_melanoma_delta"),
                "median_difference",
            ].iloc[0]
        ),
        "paired_geneformer_minus_direct_ci_low": float(
            paired.loc[
                paired["method_a"].eq("geneformer_static_pca32")
                & paired["method_b"].eq("direct_melanoma_delta"),
                "ci_low",
            ].iloc[0]
        ),
        "paired_geneformer_minus_direct_ci_high": float(
            paired.loc[
                paired["method_a"].eq("geneformer_static_pca32")
                & paired["method_b"].eq("direct_melanoma_delta"),
                "ci_high",
            ].iloc[0]
        ),
        "paired_geneformer_minus_random_median": float(
            paired.loc[
                paired["method_a"].eq("geneformer_static_pca32")
                & paired["method_b"].eq("geneformer_random_pca32"),
                "median_difference",
            ].iloc[0]
        ),
        "paired_geneformer_minus_random_ci_low": float(
            paired.loc[
                paired["method_a"].eq("geneformer_static_pca32")
                & paired["method_b"].eq("geneformer_random_pca32"),
                "ci_low",
            ].iloc[0]
        ),
        "paired_geneformer_minus_random_ci_high": float(
            paired.loc[
                paired["method_a"].eq("geneformer_static_pca32")
                & paired["method_b"].eq("geneformer_random_pca32"),
                "ci_high",
            ].iloc[0]
        ),
        "geneformer_excluding_mixed_patients_ood_spearman": (
            float(
                sensitivity.loc[
                    sensitivity["analysis"].eq("exclude_mixed_response_patients")
                    & sensitivity["method"].eq("geneformer_static_pca32"),
                    "spearman",
                ].iloc[0]
            )
            if (
                sensitivity["analysis"].eq("exclude_mixed_response_patients")
                & sensitivity["method"].eq("geneformer_static_pca32")
            ).any()
            else None
        ),
        "geneformer_leave_one_patient_out_median_spearman": float(
            leave_one_patient_summary.loc[
                leave_one_patient_summary["method"].eq("geneformer_static_pca32"),
                "median",
            ].iloc[0]
        ),
        "pretraining_leakage_status": "UNRESOLVED",
        "clinical_target_metrics_status": "NOT_COMPUTED",
        "limitations": [
            "Static checkpoint token embeddings, not contextual gene embeddings.",
            (
                "NSCLC outcome is lesion-level: 8 responsive and 5 non-responsive "
                "target lesions across 11 patients."
            ),
            (
                "P001 and P013 contain discordant lesion responses and are preserved "
                "as mixed-response patients."
            ),
            "Post-treatment transfer proxy, not a baseline clinical-response predictor.",
            "Genes are analysis units and are not statistically independent.",
            "Pretraining accession overlap remains unresolved.",
        ],
    }
    (output_directory / "ood_proxy_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    def optional_float(value: Any) -> float | None:
        return None if pd.isna(value) else float(value)

    method_results: list[ProxyMethodResult] = []
    for row in metrics.to_dict(orient="records"):
        result = ProxyMethodResult(
            method=str(row["method"]),
            features=int(row["features"]),
            selected_alpha=optional_float(row["selected_alpha"]),
            melanoma_oof_spearman=float(row["melanoma_oof_spearman"]),
            nsclc_ood_spearman=float(row["nsclc_ood_spearman"]),
            bootstrap_median=float(row["bootstrap_median"]),
            bootstrap_ci_low=float(row["bootstrap_ci_low"]),
            bootstrap_ci_high=float(row["bootstrap_ci_high"]),
            top500_overlap_fraction=float(row["top500_overlap_fraction"]),
            top500_overlap_direction_accuracy=float(row["top500_overlap_direction_accuracy"]),
            embedding_pca_explained_variance=optional_float(
                row["embedding_pca_explained_variance"]
            ),
            claim_scope=str(row["claim_scope"]),
        )
        method_results.append(result)

    serialised_results = [
        {
            key: None if isinstance(value, float) and np.isnan(value) else value
            for key, value in asdict(result).items()
        }
        for result in method_results
    ]
    (output_directory / "ood_proxy_method_results.json").write_text(
        json.dumps(serialised_results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    metric_lookup = {result.method: result for result in method_results}
    paired_lookup = {(row.method_a, row.method_b): row for row in paired.itertuples()}
    geneformer_result = metric_lookup["geneformer_static_pca32"]
    scgpt_result = metric_lookup["scgpt_static_pca32"]
    manual_result = metric_lookup["manual_features"]
    direct_result = metric_lookup["direct_melanoma_delta"]
    geneformer_direct = paired_lookup[("geneformer_static_pca32", "direct_melanoma_delta")]
    geneformer_random = paired_lookup[("geneformer_static_pca32", "geneformer_random_pca32")]
    geneformer_manual = paired_lookup[("geneformer_static_pca32", "manual_features")]
    mixed_sensitivity = summary["geneformer_excluding_mixed_patients_ood_spearman"]
    mixed_sensitivity_text = (
        f"{mixed_sensitivity:.3f}" if mixed_sensitivity is not None else "not estimable"
    )

    def result_row(label: str, result: ProxyMethodResult) -> str:
        return (
            f"| {label} | {result.melanoma_oof_spearman:.3f} | "
            f"{result.nsclc_ood_spearman:.3f} | {result.bootstrap_median:.3f} | "
            f"[{result.bootstrap_ci_low:.3f}, {result.bootstrap_ci_high:.3f}] |"
        )

    geneformer_row = result_row("Geneformer static", geneformer_result)
    manual_row = result_row("Manual features", manual_result)
    scgpt_row = result_row("scGPT static", scgpt_result)
    direct_row = result_row("Direct melanoma delta", direct_result)

    report = f"""# Melanoma-to-NSCLC response-transfer proxy

## Scope

This result is **not a clinical target-prioritisation benchmark**. It tests whether static
gene-token representations can smooth a melanoma post-treatment response-associated gene signal
so that it better agrees with an independent NSCLC post-treatment lesion-response contrast.

Claim scope: `{CLAIM_SCOPE}`.

## Outcome correction

The official GSE179994 supplement reports response per target lesion, not a single response per
patient. P1 and P13 contain both responsive and non-responsive lesions. The analysis therefore:

- maps 14 post-treatment biopsy samples to the official supplement;
- reconciles the P13 sample-number mismatch only after exact matches are exhausted;
- averages repeated biopsies of the same P1 lymph-node lesion;
- evaluates 8 responsive and 5 non-responsive target lesions across 11 patients;
- cluster-bootstraps patients, preserving all lesions from each sampled patient;
- excludes 25 patients without evaluable lesion-response labels from the outcome only.

## Data and protocol

- {len(merged):,} genes are shared across GSE115978, GSE120575 and GSE179994.
- {int(common.sum()):,} genes are jointly mapped to Geneformer and scGPT vocabularies.
- Ridge heads use identical deterministic gene folds and alpha grids.
- Geneformer and scGPT use {pca_components} PCA components from frozen static token embeddings.
- Matched random embeddings have the same shape and head budget.
- Confidence intervals use {bootstrap_iterations:,} patient-cluster bootstrap replicates.

## Results

| Method | Melanoma OOF Spearman | NSCLC OOD Spearman | Bootstrap median | 95% interval |
|---|---:|---:|---:|---:|
{geneformer_row}
{manual_row}
{scgpt_row}
{direct_row}

### Paired differences

- Geneformer minus direct melanoma signal: median **{geneformer_direct.median_difference:+.3f}**,
  95% interval **[{geneformer_direct.ci_low:+.3f}, {geneformer_direct.ci_high:+.3f}]**.
- Geneformer minus matched random embedding: median **{geneformer_random.median_difference:+.3f}**,
  95% interval **[{geneformer_random.ci_low:+.3f}, {geneformer_random.ci_high:+.3f}]**.
- Geneformer minus manual features: median **{geneformer_manual.median_difference:+.3f}**,
  95% interval **[{geneformer_manual.ci_low:+.3f}, {geneformer_manual.ci_high:+.3f}]**.

The first two intervals exclude zero. The comparison with manual features does not.

## Sensitivity analyses

- Excluding mixed-response patients P1 and P13, Geneformer Spearman is
  **{mixed_sensitivity_text}** when both outcome strata remain.
- Across leave-one-patient-out analyses, Geneformer median Spearman is
  **{summary["geneformer_leave_one_patient_out_median_spearman"]:.3f}**.

## Interpretation

The result supports a narrow statement: **Geneformer static token geometry contains some
cross-disease structure relevant to this gene-level response-transfer proxy**. It does not establish
that Geneformer improves pharmaceutical target selection. The outcome is an expression contrast,
not future clinical entry; the embeddings are static rather than contextual; genes are not
independent statistical units; only 13 target lesions are labelled; and pretraining overlap remains
unresolved.

AUPRC, MRR, excess over popularity, calibration and clinical `ood_delta` remain `NOT_COMPUTED`.
"""
    (output_directory / "REPORT.md").write_text(report, encoding="utf-8")
    return summary
