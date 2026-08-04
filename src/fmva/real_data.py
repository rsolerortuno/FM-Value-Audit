"""Reproducible ingestion of the public real-data audit fixtures.

The functions in this module deliberately stop before clinical-target evaluation.
They create traceable gene-level features and frozen cell subsets without treating
cells as independent biological replicates.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

_SAMPLE_ROW = re.compile(r"^Sample\s+\d+$")


@dataclass(frozen=True)
class SplitManifest:
    """Validated metadata for one byte-split file."""

    original_name: str
    original_size: int
    original_sha256: str | None
    original_md5: str | None
    parts: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class MatrixSummary:
    """Compact dimensions and input provenance for one expression matrix."""

    dataset_id: str
    genes: int
    cells: int
    samples: int
    patients: int | None
    sha256: str
    expression_unit: str


def file_hash(path: Path, algorithm: str = "sha256", block_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming digest without loading a large file into memory."""
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _first_present(mapping: dict[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    raise KeyError(f"None of the expected keys are present: {list(names)}")


def load_split_manifest(path: Path) -> SplitManifest:
    """Load split metadata emitted by the Drive splitting notebook."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    parts = tuple(_first_present(raw, ("parts", "chunks")))
    source = raw.get("source", {})
    if not isinstance(source, dict):
        source = {}
    original_name = (
        raw.get("original_name")
        or raw.get("filename")
        or raw.get("source_name")
        or source.get("name")
    )
    original_size = (
        raw.get("original_size") or raw.get("size") or raw.get("source_size") or source.get("size")
    )
    if original_name is None or original_size is None:
        raise KeyError("Manifest lacks source name or size")
    return SplitManifest(
        original_name=str(original_name),
        original_size=int(original_size),
        original_sha256=raw.get("original_sha256") or raw.get("sha256") or source.get("sha256"),
        original_md5=(
            raw.get("original_md5")
            or raw.get("md5")
            or source.get("md5")
            or source.get("md5Checksum")
        ),
        parts=parts,
    )


def _part_name(part: dict[str, Any]) -> str:
    return str(_first_present(part, ("name", "filename", "part_name")))


def _part_size(part: dict[str, Any]) -> int:
    return int(_first_present(part, ("size", "bytes", "part_size")))


def _part_sha256(part: dict[str, Any]) -> str | None:
    value = part.get("sha256") or part.get("part_sha256")
    return str(value) if value else None


def reassemble_split_file(
    manifest_path: Path,
    part_directory: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Concatenate validated parts and verify the complete-file digests."""
    manifest = load_split_manifest(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    observed_parts: list[dict[str, Any]] = []
    with output_path.open("wb") as destination:
        for part in manifest.parts:
            name = _part_name(part)
            candidate = part_directory / name
            if not candidate.exists() and not name.endswith(".bin"):
                bin_candidate = part_directory / f"{name}.bin"
                if bin_candidate.exists():
                    candidate = bin_candidate
            if not candidate.exists():
                raise FileNotFoundError(candidate)
            size = candidate.stat().st_size
            expected_size = _part_size(part)
            if size != expected_size:
                raise ValueError(f"Part size mismatch for {name}: {size} != {expected_size}")
            observed_sha = file_hash(candidate)
            expected_sha = _part_sha256(part)
            if expected_sha and observed_sha != expected_sha:
                raise ValueError(f"Part SHA-256 mismatch for {name}")
            with candidate.open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    destination.write(block)
            observed_parts.append({"name": name, "size": size, "sha256": observed_sha})

    final_size = output_path.stat().st_size
    if final_size != manifest.original_size:
        raise ValueError(f"Reassembled size mismatch: {final_size} != {manifest.original_size}")
    final_sha = file_hash(output_path)
    final_md5 = file_hash(output_path, "md5")
    if manifest.original_sha256 and final_sha != manifest.original_sha256:
        raise ValueError("Reassembled SHA-256 does not match the manifest")
    if manifest.original_md5 and final_md5 != manifest.original_md5:
        raise ValueError("Reassembled MD5 does not match the manifest")
    return {
        "original_name": manifest.original_name,
        "size": final_size,
        "sha256": final_sha,
        "md5": final_md5,
        "parts": observed_parts,
        "verified": True,
    }


def read_gse120575_metadata(path: Path) -> pd.DataFrame:
    """Extract only the 16,291 biological sample rows from the GEO template."""
    rows: list[list[str]] = []
    header: list[str] | None = None
    with gzip.open(path, "rt", encoding="latin1", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if row and row[0] == "Sample name":
                header = row
                continue
            if header is not None and row and _SAMPLE_ROW.fullmatch(row[0]):
                padded = row + [""] * (len(header) - len(row))
                rows.append(padded[: len(header)])
    if header is None or not rows:
        raise ValueError("Could not locate biological sample rows in GSE120575 metadata")
    frame = pd.DataFrame(rows, columns=header)
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    rename = {
        "title": "cell_id",
        "characteristics: patinet ID (Pre=baseline; Post= on treatment)": "sample_state",
        "characteristics: response": "response",
        "characteristics: therapy": "therapy",
    }
    frame = frame.rename(columns=rename)
    required = {"cell_id", "sample_state", "response", "therapy"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing GSE120575 metadata columns: {sorted(missing)}")
    extracted = frame["sample_state"].str.extract(r"^(Pre|Post)_(.+)$")
    frame["timepoint"] = extracted[0]
    frame["patient_id"] = extracted[1]
    frame["dataset_id"] = "GSE120575"
    if frame["cell_id"].duplicated().any():
        raise ValueError("GSE120575 cell identifiers are not unique")
    return frame.reset_index(drop=True)


def read_gse115978_metadata(path: Path) -> pd.DataFrame:
    """Read annotations and create a cohort-aware sample identifier."""
    frame = pd.read_csv(path)
    required = {"cells", "samples", "cell.types", "treatment.group", "Cohort"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing GSE115978 metadata columns: {sorted(missing)}")
    frame = frame.rename(
        columns={
            "cells": "cell_id",
            "samples": "sample_id",
            "cell.types": "cell_type",
            "treatment.group": "treatment",
            "Cohort": "cohort",
        }
    )
    frame["sample_key"] = frame["cohort"].astype(str) + "|" + frame["sample_id"].astype(str)
    frame["dataset_id"] = "GSE115978"
    if frame["cell_id"].duplicated().any():
        raise ValueError("GSE115978 cell identifiers are not unique")
    return frame


def _group_matrix(labels: Sequence[str]) -> tuple[list[str], np.ndarray]:
    categories = sorted(set(labels))
    lookup = {value: index for index, value in enumerate(categories)}
    matrix = np.zeros((len(labels), len(categories)), dtype=np.float64)
    matrix[np.arange(len(labels)), [lookup[value] for value in labels]] = 1.0
    return categories, matrix


def _safe_log2_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.log2(numerator + 1.0) - np.log2(denominator + 1.0)
    return np.asarray(result, dtype=np.float64)


def _group_mean_or_nan(matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if int(mask.sum()) == 0:
        return np.full(matrix.shape[0], np.nan, dtype=np.float64)
    result = matrix[:, mask].mean(axis=1)
    return np.asarray(result, dtype=np.float64)


def write_csv(frame: pd.DataFrame | pd.Series[Any], path: Path, index: bool = False) -> None:
    """Write CSV, fixing gzip mtime so regenerated artefacts are byte-stable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.endswith(".csv.gz"):
        frame.to_csv(path, index=index, compression={"method": "gzip", "mtime": 0})
    else:
        frame.to_csv(path, index=index)


def process_gse115978(
    counts_path: Path,
    metadata_path: Path,
    output_directory: Path,
    chunk_rows: int = 256,
) -> tuple[pd.DataFrame, MatrixSummary]:
    """Create sample-level features without treating cells as replicates."""
    metadata = read_gse115978_metadata(metadata_path)
    header = pd.read_csv(counts_path, nrows=0).columns.tolist()
    cell_ids = [str(value) for value in header[1:]]
    if cell_ids != metadata["cell_id"].astype(str).tolist():
        raise ValueError("GSE115978 count columns do not match annotation order")

    sample_names, sample_design = _group_matrix(metadata["sample_key"].astype(str).tolist())
    type_names, type_design = _group_matrix(metadata["cell_type"].astype(str).tolist())
    type_counts = type_design.sum(axis=0)
    gene_names: list[str] = []
    sample_sums: list[np.ndarray] = []
    type_sums: list[np.ndarray] = []
    total_sums: list[np.ndarray] = []
    positive_counts: list[np.ndarray] = []

    for chunk in pd.read_csv(counts_path, index_col=0, chunksize=chunk_rows):
        values = chunk.to_numpy(dtype=np.float64, copy=False)
        gene_names.extend(chunk.index.astype(str).tolist())
        sample_sums.append(values @ sample_design)
        type_sums.append(values @ type_design)
        total_sums.append(values.sum(axis=1))
        positive_counts.append((values > 0).sum(axis=1))

    sample_sum = np.vstack(sample_sums)
    type_sum = np.vstack(type_sums)
    total_sum = np.concatenate(total_sums)
    positive_count = np.concatenate(positive_counts)
    sample_library = sample_sum.sum(axis=0)
    sample_cpm = sample_sum / np.maximum(sample_library, 1.0)[None, :] * 1_000_000.0
    sample_log_cpm = np.log1p(sample_cpm)
    sample_meta = metadata.drop_duplicates("sample_key").set_index("sample_key").loc[sample_names]
    post_mask = sample_meta["treatment"].eq("post.treatment").to_numpy()
    naive_mask = sample_meta["treatment"].eq("treatment.naive").to_numpy()
    post_mean = sample_log_cpm[:, post_mask].mean(axis=1)
    naive_mean = sample_log_cpm[:, naive_mask].mean(axis=1)
    type_mean = type_sum / np.maximum(type_counts, 1.0)[None, :]
    type_fraction = type_mean / np.maximum(type_mean.sum(axis=1, keepdims=True), 1e-12)
    specificity = type_fraction.max(axis=1)
    dominant_type = np.asarray(type_names, dtype=object)[np.argmax(type_mean, axis=1)]

    feature_frame = pd.DataFrame(
        {
            "gene_symbol": gene_names,
            "mean_raw_count_per_cell": total_sum / len(metadata),
            "fraction_positive": positive_count / len(metadata),
            "post_vs_naive_sample_logcpm_delta": post_mean - naive_mean,
            "cell_type_specificity_max_fraction": specificity,
            "dominant_cell_type": dominant_type,
            "n_cells": len(metadata),
            "n_samples": len(sample_names),
            "n_post_samples": int(post_mask.sum()),
            "n_naive_samples": int(naive_mask.sum()),
        }
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    write_csv(feature_frame, output_directory / "GSE115978_gene_features.csv.gz")
    sparse.save_npz(
        output_directory / "GSE115978_sample_pseudobulk_cpm.npz",
        sparse.csr_matrix(sample_cpm.astype(np.float32)),
    )
    write_csv(
        pd.DataFrame(
            {
                "sample_key": sample_names,
                "sample_library_size": sample_library,
                "treatment": sample_meta["treatment"].to_numpy(),
                "cohort": sample_meta["cohort"].to_numpy(),
            }
        ),
        output_directory / "GSE115978_sample_metadata.csv",
    )
    write_csv(
        pd.Series(gene_names, name="gene_symbol"),
        output_directory / "GSE115978_gene_order.csv",
    )
    return feature_frame, MatrixSummary(
        dataset_id="GSE115978",
        genes=len(gene_names),
        cells=len(metadata),
        samples=len(sample_names),
        patients=None,
        sha256=file_hash(counts_path),
        expression_unit="raw_count",
    )


def _read_gse120575_cell_ids(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n\r").split("\t")
    return header[1:]


def process_gse120575(
    tpm_path: Path,
    metadata_path: Path,
    output_directory: Path,
    chunk_rows: int = 128,
    allowed_genes: set[str] | None = None,
) -> tuple[pd.DataFrame, MatrixSummary]:
    """Stream TPM into patient-state summaries, explicitly skipping GEO row two.

    ``chunk_rows`` is retained for CLI compatibility; the implementation reads one
    gene line at a time because NumPy's C parser is materially faster for this
    irregular file (each gene row has one extra trailing tab).
    """
    del chunk_rows
    metadata = read_gse120575_metadata(metadata_path)
    cell_ids = _read_gse120575_cell_ids(tpm_path)
    if cell_ids != metadata["cell_id"].astype(str).tolist():
        raise ValueError("GSE120575 TPM columns do not match biological metadata order")
    state_names = sorted(metadata["sample_state"].astype(str).unique())
    state_lookup = {value: index for index, value in enumerate(state_names)}
    state_index = np.asarray(
        [state_lookup[value] for value in metadata["sample_state"].astype(str)], dtype=np.int64
    )
    state_counts = np.bincount(state_index, minlength=len(state_names)).astype(np.float64)
    state_meta = metadata.drop_duplicates("sample_state").set_index("sample_state").loc[state_names]
    pre_mask = state_meta["timepoint"].eq("Pre").to_numpy()
    post_mask = state_meta["timepoint"].eq("Post").to_numpy()
    responder_mask = state_meta["response"].eq("Responder").to_numpy()
    nonresponder_mask = state_meta["response"].eq("Non-responder").to_numpy()

    gene_names: list[str] = []
    state_sums: list[np.ndarray] = []
    total_sums: list[float] = []
    positive_counts: list[int] = []
    with gzip.open(tpm_path, "rb") as handle:
        handle.readline()  # cell identifiers
        handle.readline()  # embedded patient-state annotation row
        for raw_line in handle:
            if not raw_line.strip():
                continue
            gene_raw, values_raw = raw_line.split(b"\t", 1)
            gene_name = gene_raw.decode("utf-8")
            if allowed_genes is not None and gene_name not in allowed_genes:
                continue
            values = np.fromstring(values_raw, sep="\t", dtype=np.float32)
            if len(values) != len(cell_ids):
                raise ValueError(
                    f"GSE120575 row {gene_raw!r} has {len(values)} values; expected {len(cell_ids)}"
                )
            gene_names.append(gene_name)
            state_sums.append(np.bincount(state_index, weights=values, minlength=len(state_names)))
            total_sums.append(float(values.sum(dtype=np.float64)))
            positive_counts.append(int(np.count_nonzero(values)))

    state_sum = np.vstack(state_sums)
    state_mean = state_sum / np.maximum(state_counts, 1.0)[None, :]
    total_sum = np.asarray(total_sums, dtype=np.float64)
    positive_count = np.asarray(positive_counts, dtype=np.int64)
    baseline_r = pre_mask & responder_mask
    baseline_nr = pre_mask & nonresponder_mask
    post_r = post_mask & responder_mask
    post_nr = post_mask & nonresponder_mask

    feature_frame = pd.DataFrame(
        {
            "gene_symbol": gene_names,
            "mean_tpm": total_sum / len(metadata),
            "fraction_positive": positive_count / len(metadata),
            "baseline_responder_vs_nonresponder_log2_delta": _safe_log2_ratio(
                _group_mean_or_nan(state_mean, baseline_r),
                _group_mean_or_nan(state_mean, baseline_nr),
            ),
            "post_responder_vs_nonresponder_log2_delta": _safe_log2_ratio(
                _group_mean_or_nan(state_mean, post_r),
                _group_mean_or_nan(state_mean, post_nr),
            ),
            "n_cells": len(metadata),
            "n_patient_states": len(state_names),
            "n_baseline_responder_states": int(baseline_r.sum()),
            "n_baseline_nonresponder_states": int(baseline_nr.sum()),
        }
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    write_csv(feature_frame, output_directory / "GSE120575_gene_features.csv.gz")
    sparse.save_npz(
        output_directory / "GSE120575_patient_state_mean_tpm.npz",
        sparse.csr_matrix(state_mean.astype(np.float32)),
    )
    write_csv(
        pd.DataFrame(
            {
                "sample_state": state_names,
                "cells": state_counts.astype(int),
                "patient_id": state_meta["patient_id"].to_numpy(),
                "timepoint": state_meta["timepoint"].to_numpy(),
                "response": state_meta["response"].to_numpy(),
                "therapy": state_meta["therapy"].to_numpy(),
            }
        ),
        output_directory / "GSE120575_patient_state_metadata.csv",
    )
    write_csv(
        pd.Series(gene_names, name="gene_symbol"),
        output_directory / "GSE120575_gene_order.csv",
    )
    return feature_frame, MatrixSummary(
        dataset_id="GSE120575",
        genes=len(gene_names),
        cells=len(metadata),
        samples=len(state_names),
        patients=int(metadata["patient_id"].nunique()),
        sha256=file_hash(tpm_path),
        expression_unit="TPM",
    )


def choose_model_cells(metadata: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    """Choose a deterministic, sample-aware subset for CPU representation checks."""
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    required = {"cell_id", "sample_key", "cell_type", "treatment"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing subset columns: {sorted(missing)}")
    shuffled = metadata.sample(frac=1.0, random_state=seed)
    first = shuffled.groupby(["treatment", "cell_type"], observed=True, sort=True).head(1)
    remaining = shuffled.loc[~shuffled.index.isin(first.index)]
    selected = pd.concat([first, remaining], axis=0).head(maximum)
    return selected.sort_values(["treatment", "cell_type", "sample_key", "cell_id"]).reset_index(
        drop=True
    )


def load_selected_counts(
    counts_path: Path,
    selected_cell_ids: Sequence[str],
) -> tuple[list[str], np.ndarray]:
    """Load only selected cells from a gene-by-cell CSV matrix."""
    selected = set(selected_cell_ids)
    frame = pd.read_csv(
        counts_path,
        index_col=0,
        usecols=lambda name: str(name).startswith("Unnamed:") or str(name) in selected,
    )
    missing = selected - set(frame.columns.astype(str))
    if missing:
        raise ValueError(f"Selected cells missing from count matrix: {sorted(missing)[:3]}")
    frame = frame.loc[:, list(selected_cell_ids)]
    return frame.index.astype(str).tolist(), frame.to_numpy(dtype=np.float32, copy=False).T


def combine_real_gene_features(
    gse115978: pd.DataFrame,
    gse120575: pd.DataFrame,
    symbol_to_ensembl: dict[str, str],
) -> pd.DataFrame:
    """Join the two melanoma feature layers without inventing target labels."""
    left = gse115978.rename(
        columns={
            "mean_raw_count_per_cell": "mean_expression_counts",
            "post_vs_naive_sample_logcpm_delta": "treatment_delta_logcpm",
            "cell_type_specificity_max_fraction": "cell_type_specificity",
        }
    )
    right = gse120575.rename(
        columns={
            "mean_tpm": "mean_expression_tpm",
            "baseline_responder_vs_nonresponder_log2_delta": "baseline_response_delta",
        }
    )
    keep_left = [
        "gene_symbol",
        "mean_expression_counts",
        "treatment_delta_logcpm",
        "cell_type_specificity",
        "dominant_cell_type",
    ]
    keep_right = ["gene_symbol", "mean_expression_tpm", "baseline_response_delta"]
    joined = left[keep_left].merge(right[keep_right], on="gene_symbol", how="outer")
    joined.insert(0, "gene_id", joined["gene_symbol"].map(symbol_to_ensembl))
    joined["gene_id_status"] = np.where(joined["gene_id"].notna(), "MAPPED", "UNMAPPED")
    joined["target_label_status"] = "NOT_COMPUTED"
    return joined


def write_json(data: Any, path: Path) -> None:
    """Write deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
