"""Fixed contextual feature views for FMVA v0.5.

The routines in this module turn row-level contextual gene embeddings into a
pre-registered, gene-level representation.  They deliberately avoid any label-
dependent view selection.  Missing views are encoded as zeros plus an explicit
availability indicator, and dimensionality reduction is fit on development
genes only.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from fmva.contextual import CONTEXT_INDEX_COLUMNS, ContextualArtifactManifest, sha256_file


@dataclass(frozen=True)
class ContextView:
    """A preregistered biological view over contextual rows."""

    name: str
    predicate: Callable[[pd.DataFrame], pd.Series[bool]]


def _normalised(series: pd.Series[Any]) -> pd.Series[str]:
    values = series.fillna("unknown").astype(str).str.strip().str.lower()
    return values


def _contains(series: pd.Series[Any], terms: Iterable[str]) -> pd.Series[bool]:
    values = _normalised(series)
    mask = pd.Series(False, index=series.index, dtype=bool)
    for term in terms:
        mask = mask | values.str.contains(term, regex=False)
    return mask


def _all_rows(frame: pd.DataFrame) -> pd.Series[bool]:
    return pd.Series(True, index=frame.index, dtype=bool)


def default_context_views() -> tuple[ContextView, ...]:
    """Return the immutable contextual views used by the alpha protocol."""
    return (
        ContextView("all", _all_rows),
        ContextView("melanoma", lambda frame: _normalised(frame["disease"]).eq("melanoma")),
        ContextView("nsclc", lambda frame: _normalised(frame["disease"]).eq("nsclc")),
        ContextView(
            "pretreatment",
            lambda frame: _contains(
                frame["treatment_state"],
                ("pre", "naive", "baseline", "untreated", "t0"),
            ),
        ),
        ContextView(
            "on_or_post_treatment",
            lambda frame: _contains(
                frame["treatment_state"],
                ("on", "post", "treated", "t1", "t2"),
            ),
        ),
        ContextView(
            "responder",
            lambda frame: (
                _contains(frame["response_state"], ("responder", "response", "cr", "pr"))
                & ~_contains(frame["response_state"], ("non", "no response", "resistant"))
            ),
        ),
        ContextView(
            "nonresponder",
            lambda frame: _contains(
                frame["response_state"],
                ("non", "no response", "resistant", "progress", "pd"),
            ),
        ),
        ContextView(
            "immune",
            lambda frame: _contains(
                frame["cell_compartment"],
                ("immune", "t cell", "b cell", "myeloid", "nk", "lymph"),
            ),
        ),
        ContextView(
            "malignant",
            lambda frame: _contains(
                frame["cell_compartment"],
                ("malignant", "tumour", "tumor", "cancer"),
            ),
        ),
    )


def load_contextual_pair(
    root: Path,
) -> tuple[pd.DataFrame, npt.NDArray[Any], npt.NDArray[Any], ContextualArtifactManifest]:
    """Load one validated contextual pretrained/random pair from ``root``."""
    manifest_path = root / "CONTEXTUAL_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = ContextualArtifactManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )

    index_path = root / manifest.index_file
    pretrained_path = root / manifest.pretrained_file
    random_path = root / manifest.random_file
    for path in (index_path, pretrained_path, random_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    hashes = {
        "index": sha256_file(index_path),
        "pretrained": sha256_file(pretrained_path),
        "random": sha256_file(random_path),
    }
    expected = {
        "index": manifest.index_sha256,
        "pretrained": manifest.pretrained_sha256,
        "random": manifest.random_sha256,
    }
    if hashes != expected:
        raise ValueError(f"Contextual artifact hash mismatch: {hashes} != {expected}")

    index = pd.read_csv(index_path)
    missing = set(CONTEXT_INDEX_COLUMNS) - set(index.columns)
    if missing:
        raise ValueError(f"Context index columns missing: {sorted(missing)}")
    pretrained = np.load(pretrained_path, mmap_mode="r")
    random = np.load(random_path, mmap_mode="r")
    expected_shape = (len(index), manifest.embedding_width)
    if pretrained.shape != expected_shape or random.shape != expected_shape:
        raise ValueError(
            "Contextual matrix shape mismatch: "
            f"pretrained={pretrained.shape}, random={random.shape}, expected={expected_shape}"
        )
    return index, np.asarray(pretrained), np.asarray(random), manifest


def _aggregate_one_matrix(
    index: pd.DataFrame,
    matrix: npt.NDArray[Any],
    genes: list[str],
    views: tuple[ContextView, ...],
) -> tuple[npt.NDArray[Any], list[str], dict[str, int]]:
    gene_to_position = {gene: position for position, gene in enumerate(genes)}
    width = matrix.shape[1]
    blocks: list[npt.NDArray[Any]] = []
    columns: list[str] = []
    coverage: dict[str, int] = {}

    for view in views:
        mask = view.predicate(index).to_numpy(dtype=bool)
        output = np.zeros((len(genes), width), dtype=np.float32)
        available = np.zeros(len(genes), dtype=np.float32)
        if mask.any():
            selected = index.loc[mask, ["gene_id"]].copy()
            selected["_row"] = np.flatnonzero(mask)
            for gene, group in selected.groupby("gene_id", sort=False):
                position = gene_to_position.get(str(gene))
                if position is None:
                    continue
                rows = group["_row"].to_numpy(dtype=int)
                output[position] = matrix[rows].mean(axis=0, dtype=np.float64).astype(np.float32)
                available[position] = 1.0
        coverage[view.name] = int(available.sum())
        blocks.extend([output, available[:, None]])
        columns.extend([f"{view.name}_dim{dimension + 1:04d}" for dimension in range(width)])
        columns.append(f"{view.name}_available")

    return np.column_stack(blocks).astype(np.float32), columns, coverage


def build_contextual_gene_views(
    root: Path,
    *,
    views: tuple[ContextView, ...] | None = None,
) -> tuple[pd.DataFrame, npt.NDArray[Any], npt.NDArray[Any], dict[str, object]]:
    """Build fixed pretrained/random gene matrices from one contextual export."""
    index, pretrained, random, manifest = load_contextual_pair(root)
    selected_views = views or default_context_views()
    genes = sorted(index["gene_id"].dropna().astype(str).unique())
    pretrained_matrix, columns, pretrained_coverage = _aggregate_one_matrix(
        index, pretrained, genes, selected_views
    )
    random_matrix, random_columns, random_coverage = _aggregate_one_matrix(
        index, random, genes, selected_views
    )
    if columns != random_columns or pretrained_coverage != random_coverage:
        raise RuntimeError("Pretrained and random contextual view definitions diverged")

    gene_frame = pd.DataFrame({"target_gene": genes})
    metadata: dict[str, object] = {
        "model_id": manifest.model_id,
        "control_model_id": manifest.control_model_id,
        "protocol_id": manifest.protocol_id,
        "source_manifest_sha256": sha256_file(root / "CONTEXTUAL_MANIFEST.json"),
        "rows": len(index),
        "genes": len(genes),
        "raw_feature_count": pretrained_matrix.shape[1],
        "views": [view.name for view in selected_views],
        "coverage_genes_by_view": pretrained_coverage,
        "columns": columns,
    }
    return gene_frame, pretrained_matrix, random_matrix, metadata


def fit_development_contextual_pca(
    gene_frame: pd.DataFrame,
    matrix: npt.NDArray[Any],
    development_genes: set[str],
    *,
    components: int = 32,
    seed: int = 20260804,
) -> tuple[npt.NDArray[Any], dict[str, object]]:
    """Fit scale/PCA only on development genes and transform all genes."""
    if matrix.ndim != 2 or len(gene_frame) != matrix.shape[0]:
        raise ValueError("Gene frame and contextual matrix are not aligned")
    if components < 1:
        raise ValueError("components must be positive")

    fit_mask = gene_frame["target_gene"].astype(str).isin(development_genes).to_numpy()
    finite = np.isfinite(matrix).all(axis=1)
    nonzero = np.any(matrix != 0.0, axis=1)
    fit_mask = fit_mask & finite & nonzero
    fit_rows = int(fit_mask.sum())
    if fit_rows <= components:
        raise ValueError(
            f"Insufficient development genes for contextual PCA: {fit_rows} <= {components}"
        )

    scaler = StandardScaler()
    fit_scaled = scaler.fit_transform(matrix[fit_mask])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    pca.fit(fit_scaled)

    valid = finite & nonzero
    output = np.zeros((len(gene_frame), components), dtype=np.float32)
    output[valid] = pca.transform(scaler.transform(matrix[valid])).astype(np.float32)
    metadata: dict[str, object] = {
        "components": components,
        "fit_genes": fit_rows,
        "transformed_genes": int(valid.sum()),
        "pca_fit_scope": "development_genes_only",
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "seed": seed,
    }
    return output, metadata


def write_contextual_gene_representation(
    output_path: Path,
    gene_frame: pd.DataFrame,
    pretrained_pca: npt.NDArray[Any],
    random_pca: npt.NDArray[Any],
    *,
    prefix: str,
    metadata: dict[str, object],
) -> tuple[Path, Path]:
    """Write a compact, hashable gene representation and metadata."""
    if pretrained_pca.shape != random_pca.shape or len(gene_frame) != len(pretrained_pca):
        raise ValueError("Contextual PCA outputs are not aligned")
    output = gene_frame.copy()
    for component in range(pretrained_pca.shape[1]):
        output[f"{prefix}_pc{component + 1:02d}"] = pretrained_pca[:, component]
        output[f"{prefix}_random_pc{component + 1:02d}"] = random_pca[:, component]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.name.endswith(".csv.gz"):
        raise ValueError("Contextual gene representation must use a .csv.gz path")
    output.to_csv(
        output_path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    metadata_path = output_path.with_suffix(".metadata.json")
    payload = {
        **metadata,
        "output_file": output_path.name,
        "output_sha256": sha256_file(output_path),
        "rows": len(output),
        "components": pretrained_pca.shape[1],
    }
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path, metadata_path
