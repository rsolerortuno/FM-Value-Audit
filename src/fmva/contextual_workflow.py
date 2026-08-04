"""Execution helpers shared by the v0.5 contextual Colab notebooks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fmva.contextual import CONTEXT_INDEX_COLUMNS, deterministic_context_sample, sha256_file

CONTEXT_COLUMNS = ["disease", "cell_compartment", "treatment_state", "response_state"]


def find_unique(root: Path, patterns: Iterable[str]) -> Path:
    """Find exactly one file under ``root`` matching the first successful pattern."""
    for pattern in patterns:
        matches = sorted(path for path in root.rglob(pattern) if path.is_file())
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Pattern {pattern!r} matched multiple files: {matches[:5]}")
    raise FileNotFoundError(f"No file under {root} matched any of: {list(patterns)}")


def _first_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower: dict[str, str] = {
        str(column).lower().replace(" ", "_"): str(column) for column in frame.columns
    }
    for candidate in candidates:
        normalised = candidate.lower().replace(" ", "_")
        if normalised in lower:
            return lower[normalised]
    return None


def _clean(values: pd.Series[Any], missing: str = "unknown") -> pd.Series[str]:
    result = values.fillna(missing).astype(str).str.strip().str.lower()
    result = result.replace(
        {
            "": missing,
            "nan": missing,
            "na": missing,
            "n/a": missing,
            "none": missing,
            "treatment.naive": "pre",
            "treatment_naive": "pre",
            "pretreatment": "pre",
            "baseline": "pre",
            "post.treatment": "post",
            "post_treatment": "post",
            "non-responder": "non_responder",
            "nonresponder": "non_responder",
            "non responder": "non_responder",
            "responder": "responder",
        }
    )
    cleaned = result.str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    return cleaned


def broad_compartment(values: pd.Series[Any]) -> pd.Series[str]:
    """Map heterogeneous cell labels to conservative broad compartments."""
    cleaned = _clean(values)

    def map_one(value: str) -> str:
        if value == "unknown":
            return value
        if "cd8" in value:
            return "cd8_t"
        if "cd4" in value or "treg" in value:
            return "cd4_t"
        if "nk" in value:
            return "nk"
        if value.startswith("t_") or value == "t" or "t_cell" in value:
            return "t_other"
        if "b_cell" in value or value.startswith("b_") or value == "b":
            return "b"
        if any(token in value for token in ("mono", "macro", "myeloid", "dc", "dendritic")):
            return "myeloid"
        if any(token in value for token in ("tumor", "malignant", "melanoma", "epithelial")):
            return "tumour"
        if any(token in value for token in ("fibro", "strom", "endo", "pericyte")):
            return "stromal"
        return value

    mapped: pd.Series[str] = cleaned.map(map_one)
    return mapped


def build_context_metadata(
    obs: pd.DataFrame,
    *,
    cohort: str,
    official_lesion_mapping: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build canonical cell, sample and biological-context columns from H5AD obs."""
    frame = obs.copy()
    frame["cell_id"] = frame.index.astype(str)
    sample_column = _first_column(
        frame,
        ["sample", "sample_id", "sample_key", "orig.ident", "library", "patient_state"],
    )
    if sample_column is None:
        raise ValueError(
            f"Could not identify sample column for {cohort}; columns={list(frame.columns)}"
        )
    frame["sample_id"] = frame[sample_column].astype(str)

    cell_column = _first_column(
        frame,
        ["celltype", "cell_type", "cell.type", "broad_celltype", "cluster", "cell_annotation"],
    )
    frame["cell_compartment"] = (
        broad_compartment(frame[cell_column]) if cell_column is not None else "unknown"
    )

    treatment_column = _first_column(
        frame,
        ["timepoint", "treatment", "treatment_state", "status", "therapy_state"],
    )
    frame["treatment_state"] = (
        _clean(frame[treatment_column]) if treatment_column is not None else "unknown"
    )

    response_column = _first_column(
        frame,
        ["lesion_response", "clinical_response", "response", "response_state"],
    )
    frame["response_state"] = (
        _clean(frame[response_column]) if response_column is not None else "unknown"
    )

    cohort_upper = cohort.upper()
    if cohort_upper == "GSE115978":
        frame["disease"] = "melanoma"
    elif cohort_upper == "GSE179994":
        frame["disease"] = "nsclc"
        if official_lesion_mapping is None:
            raise ValueError("GSE179994 requires the official lesion-level sample mapping")
        required = {"dataset_sample", "lesion_response"}
        if not required.issubset(official_lesion_mapping.columns):
            raise ValueError("Official lesion mapping lacks dataset_sample/lesion_response")
        response_map = (
            official_lesion_mapping[["dataset_sample", "lesion_response"]]
            .drop_duplicates("dataset_sample")
            .set_index("dataset_sample")["lesion_response"]
        )
        mapped = frame["sample_id"].map(response_map)
        frame["response_state"] = _clean(mapped)
    else:
        raise ValueError(f"Unsupported contextual cohort: {cohort}")

    frame["context_id"] = frame[CONTEXT_COLUMNS].astype(str).agg("|".join, axis=1)
    return frame[["cell_id", "sample_id", *CONTEXT_COLUMNS, "context_id"]].reset_index(drop=True)


def select_protocol_cells(
    metadata: pd.DataFrame,
    *,
    maximum_cells_per_sample_context: int,
    minimum_cells_per_sample_context: int,
    seed: int,
) -> pd.DataFrame:
    return deterministic_context_sample(
        metadata,
        cell_id_column="cell_id",
        stratum_columns=["sample_id", *CONTEXT_COLUMNS],
        maximum_per_stratum=maximum_cells_per_sample_context,
        minimum_per_stratum=minimum_cells_per_sample_context,
        seed=seed,
    )


@dataclass
class TokenGeneAccumulator:
    """Accumulate final token hidden states into gene means for one sample-context."""

    pretrained_sum: np.ndarray
    random_sum: np.ndarray
    token_count: np.ndarray
    cell_count: np.ndarray

    @classmethod
    def create(cls, n_genes: int, width: int) -> TokenGeneAccumulator:
        return cls(
            pretrained_sum=np.zeros((n_genes, width), dtype=np.float64),
            random_sum=np.zeros((n_genes, width), dtype=np.float64),
            token_count=np.zeros(n_genes, dtype=np.int64),
            cell_count=np.zeros(n_genes, dtype=np.int32),
        )

    def update(
        self,
        gene_indices: np.ndarray,
        pretrained_hidden: np.ndarray,
        random_hidden: np.ndarray,
    ) -> None:
        if gene_indices.shape != pretrained_hidden.shape[:2]:
            raise ValueError("gene index and pretrained hidden shapes do not align")
        if pretrained_hidden.shape != random_hidden.shape:
            raise ValueError("pretrained and random hidden shapes differ")
        for cell_index in range(len(gene_indices)):
            valid = gene_indices[cell_index] >= 0
            genes = gene_indices[cell_index, valid].astype(np.int64, copy=False)
            if len(genes) == 0:
                continue
            np.add.at(self.pretrained_sum, genes, pretrained_hidden[cell_index, valid])
            np.add.at(self.random_sum, genes, random_hidden[cell_index, valid])
            np.add.at(self.token_count, genes, 1)
            np.add.at(self.cell_count, np.unique(genes), 1)

    def finalise(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        genes = np.flatnonzero(self.token_count > 0)
        denominator = self.token_count[genes, None]
        pretrained = (self.pretrained_sum[genes] / denominator).astype(np.float32)
        random = (self.random_sum[genes] / denominator).astype(np.float32)
        return genes.astype(np.int32), pretrained, random, self.cell_count[genes]


def write_sample_context_partial(
    path: Path,
    *,
    gene_indices: np.ndarray,
    pretrained: np.ndarray,
    random: np.ndarray,
    cell_counts: np.ndarray,
    metadata: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        gene_indices=gene_indices,
        pretrained=pretrained,
        random=random,
        cell_counts=cell_counts,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def read_sample_context_partial(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        return (
            payload["gene_indices"].copy(),
            payload["pretrained"].copy(),
            payload["random"].copy(),
            payload["cell_counts"].copy(),
            metadata,
        )


def reduce_sample_context_partials(
    partial_paths: list[Path],
    *,
    gene_symbols: list[str],
    width: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Average sample-level gene embeddings equally within each biological context."""
    grouped: dict[tuple[str, str, str, str], list[Path]] = {}
    for path in partial_paths:
        _, _, _, _, metadata = read_sample_context_partial(path)
        key = (
            metadata["disease"],
            metadata["cell_compartment"],
            metadata["treatment_state"],
            metadata["response_state"],
        )
        grouped.setdefault(key, []).append(path)

    index_rows: list[dict[str, Any]] = []
    pretrained_rows: list[np.ndarray] = []
    random_rows: list[np.ndarray] = []
    n_genes = len(gene_symbols)
    for context, paths in sorted(grouped.items()):
        pretrained_sum = np.zeros((n_genes, width), dtype=np.float64)
        random_sum = np.zeros((n_genes, width), dtype=np.float64)
        sample_count = np.zeros(n_genes, dtype=np.int32)
        cell_count = np.zeros(n_genes, dtype=np.int32)
        for path in sorted(paths):
            genes, pretrained, random, cells, _ = read_sample_context_partial(path)
            pretrained_sum[genes] += pretrained
            random_sum[genes] += random
            sample_count[genes] += 1
            cell_count[genes] += cells
        represented = np.flatnonzero(sample_count > 0)
        for gene_index in represented:
            samples = int(sample_count[gene_index])
            index_rows.append(
                {
                    "gene_id": str(gene_symbols[gene_index]),
                    "disease": context[0],
                    "cell_compartment": context[1],
                    "treatment_state": context[2],
                    "response_state": context[3],
                    "n_cells": int(cell_count[gene_index]),
                    "n_samples": samples,
                }
            )
            pretrained_rows.append((pretrained_sum[gene_index] / samples).astype(np.float32))
            random_rows.append((random_sum[gene_index] / samples).astype(np.float32))
    if not index_rows:
        raise ValueError("No contextual sample partials could be reduced")
    return (
        pd.DataFrame(index_rows, columns=CONTEXT_INDEX_COLUMNS),
        np.vstack(pretrained_rows),
        np.vstack(random_rows),
    )


def h5ad_input_hashes(h5ad_paths: Iterable[Path], metadata_paths: Iterable[Path]) -> dict[str, str]:
    paths = [*h5ad_paths, *metadata_paths]
    return {path.name: sha256_file(path) for path in sorted(paths)}


def safe_partial_name(sample_id: str, context_id: str) -> str:
    digest = hashlib.sha256(f"{sample_id}|{context_id}".encode()).hexdigest()[:16]
    return f"sample_context_{digest}.npz"
