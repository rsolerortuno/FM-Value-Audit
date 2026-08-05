"""Prospective contextual-embedding contracts for FM Value Audit v0.5.

The module deliberately separates three concerns:

* preregistration of the prospective study before future labels exist;
* deterministic, sample-balanced selection of real single cells;
* integrity validation of contextual embedding exports.

It does not open or evaluate future holdout labels.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections.abc import Iterable
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContextualAuditStatus(StrEnum):
    PASS = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    FAIL = "FAIL"


class ContextualCheckStatus(StrEnum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


class ContextualInvariant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    status: ContextualCheckStatus
    detail: str


class ContextualValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ContextualAuditStatus
    root: str
    model_id: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    invariants: list[ContextualInvariant] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class SnapshotSchedule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    first_eligible_snapshot: str
    cadence_months: int = Field(ge=1)
    maximum_snapshot_date: str
    selection_rule: str


class PowerGate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum_positive_pairs: int = Field(ge=1)
    minimum_positive_genes: int = Field(ge=1)
    minimum_indications_with_positive: int = Field(ge=1)
    minimum_holdout_positive_pairs: int = Field(ge=1)
    minimum_holdout_positive_genes: int = Field(ge=1)


class SamplingPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit: str
    maximum_cells_per_sample_context: int = Field(ge=1)
    minimum_cells_per_sample_context: int = Field(ge=1)
    selection: str
    seed: int
    sample_balanced_pooling: bool

    @model_validator(mode="after")
    def minimum_not_above_maximum(self) -> SamplingPlan:
        if self.minimum_cells_per_sample_context > self.maximum_cells_per_sample_context:
            raise ValueError("minimum cells cannot exceed maximum cells")
        return self


class ContextDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fields: list[str]
    missing_value: str
    cohorts: dict[str, dict[str, Any]]


class MethodDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    representation: str
    sequence_length: int = Field(ge=2)
    embedding_width: int = Field(ge=1)
    pooling: str
    random_control: str
    claim_role: str


class ContextualProtocol(BaseModel):
    """Frozen protocol for the new prospective contextual benchmark."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    protocol_id: str
    release_target: str
    status: str
    frozen_at_utc: str
    information_cutoff: str
    outcome_window_start: str
    outcome_window_end_rule: str
    snapshot_schedule: SnapshotSchedule
    power_gate: PowerGate
    candidate_universe: str
    endpoint: str
    gene_split: dict[str, Any]
    sampling: SamplingPlan
    context_definition: ContextDefinition
    methods: list[MethodDefinition]
    primary_contrast: dict[str, str]
    secondary_methods: list[str]
    forbidden_actions: list[str]
    historical_snapshots: dict[str, str]
    known_limitations: list[str]

    @model_validator(mode="after")
    def unique_methods_and_primary_membership(self) -> ContextualProtocol:
        method_ids = [method.model_id for method in self.methods]
        if len(method_ids) != len(set(method_ids)):
            raise ValueError("method identifiers must be unique")
        primary_method = self.primary_contrast.get("method")
        if primary_method not in method_ids:
            raise ValueError("primary method must be present in methods")
        return self


class ContextualArtifactManifest(BaseModel):
    """Manifest written beside one pretrained/random contextual export pair."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    protocol_id: str
    protocol_sha256: str
    model_id: str
    control_model_id: str
    status: str
    created_at_utc: str
    checkpoint_sha256: str
    input_hashes: dict[str, str]
    index_file: str
    pretrained_file: str
    random_file: str
    index_sha256: str
    pretrained_sha256: str
    random_sha256: str
    rows: int = Field(ge=1)
    embedding_width: int = Field(ge=1)
    selected_cells: int = Field(ge=1)
    represented_samples: int = Field(ge=1)
    contexts: int = Field(ge=1)
    sequence_length: int = Field(ge=2)
    sampling_seed: int
    pooling: str
    claim_scope: str
    upstream_versions: dict[str, str]


CONTEXT_INDEX_COLUMNS = [
    "gene_id",
    "disease",
    "cell_compartment",
    "treatment_state",
    "response_state",
    "n_cells",
    "n_samples",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def default_contextual_protocol() -> ContextualProtocol:
    """Return the preregistered v0.5 protocol frozen on 4 August 2026."""
    return ContextualProtocol(
        schema_version="fmva-contextual-protocol-v1",
        protocol_id="fmva-contextual-prospective-2026-08-04",
        release_target="0.5.0",
        status="FROZEN_BEFORE_CONTEXTUAL_MODEL_EXECUTION_AND_FUTURE_LABEL_ACCESS",
        frozen_at_utc="2026-08-04T15:37:00Z",
        information_cutoff="2026-08-04",
        outcome_window_start="2026-08-05",
        outcome_window_end_rule=(
            "Use the first scheduled snapshot satisfying the power gate, without inspecting any "
            "method scores or performance; otherwise stop at the maximum snapshot date and abstain."
        ),
        snapshot_schedule=SnapshotSchedule(
            first_eligible_snapshot="2027-02-04",
            cadence_months=3,
            maximum_snapshot_date="2028-08-04",
            selection_rule=(
                "Evaluate snapshots in chronological order and select the first that passes the "
                "label-only power gate. Model scores remain inaccessible during gate assessment."
            ),
        ),
        power_gate=PowerGate(
            minimum_positive_pairs=20,
            minimum_positive_genes=10,
            minimum_indications_with_positive=5,
            minimum_holdout_positive_pairs=5,
            minimum_holdout_positive_genes=5,
        ),
        candidate_universe=(
            "The v0.4.0 wide solid-tumour candidate universe, updated only for identifier "
            "retirements and additions documented before the 2026-08-04 cutoff."
        ),
        endpoint=(
            "First public ClinicalTrials.gov Phase I-or-later entry after 2026-08-04 for a "
            "single-direct-human-target therapeutic intervention in a prespecified solid-tumour "
            "indication, excluding target-indication pairs with a qualifying entry on or before "
            "the cutoff."
        ),
        gene_split={
            "algorithm": "sha256_modulo",
            "modulus": 5,
            "holdout_fold": 0,
            "development_folds": [1, 2, 3, 4],
            "salt": "fmva-v0.5-prospective-gene-split-20260804",
            "gene_disjoint": True,
        },
        sampling=SamplingPlan(
            unit="real_single_cell",
            maximum_cells_per_sample_context=32,
            minimum_cells_per_sample_context=4,
            selection="lowest_sha256(cell_id + seed) within each sample-context stratum",
            seed=20260804,
            sample_balanced_pooling=True,
        ),
        context_definition=ContextDefinition(
            fields=[
                "disease",
                "cell_compartment",
                "treatment_state",
                "response_state",
            ],
            missing_value="unknown",
            cohorts={
                "GSE115978": {
                    "disease": "melanoma",
                    "input_scale": "raw_counts",
                    "role": "melanoma treatment-context cells",
                    "response_policy": "use supplied response only when explicitly present",
                },
                "GSE179994": {
                    "disease": "nsclc",
                    "input_scale": "raw_counts",
                    "role": "independent lesion-response and treatment-context cells",
                    "response_policy": "official lesion-level mapping; never patient-collapsed",
                },
                "GSE120575": {
                    "disease": "melanoma",
                    "input_scale": "TPM",
                    "role": "manual-feature baseline only",
                    "response_policy": (
                        "excluded from contextual encoder input because counts are TPM"
                    ),
                },
            },
        ),
        methods=[
            MethodDefinition(
                model_id="cheap_historical",
                representation="Open Targets 26.06 + ChEMBL 37 + pre-cutoff popularity",
                sequence_length=2,
                embedding_width=1,
                pooling="not_applicable",
                random_control="not_applicable",
                claim_role="primary_baseline",
            ),
            MethodDefinition(
                model_id="geneformer_contextual_plus_cheap",
                representation="Geneformer V1-10M final token hidden states from real cells",
                sequence_length=512,
                embedding_width=256,
                pooling="token-to-gene mean within sample-context, then equal-weight sample mean",
                random_control="geneformer_random_contextual_plus_cheap",
                claim_role="confirmatory_primary",
            ),
            MethodDefinition(
                model_id="geneformer_random_contextual_plus_cheap",
                representation="shape-matched random Geneformer token hidden states",
                sequence_length=512,
                embedding_width=256,
                pooling="identical to pretrained",
                random_control="self",
                claim_role="negative_control",
            ),
            MethodDefinition(
                model_id="scgpt_contextual_plus_cheap",
                representation="scGPT whole-human final gene-token hidden states from real cells",
                sequence_length=512,
                embedding_width=512,
                pooling="token-to-gene mean within sample-context, then equal-weight sample mean",
                random_control="scgpt_random_contextual_plus_cheap",
                claim_role="secondary",
            ),
            MethodDefinition(
                model_id="scgpt_random_contextual_plus_cheap",
                representation="shape-matched random scGPT gene-token hidden states",
                sequence_length=512,
                embedding_width=512,
                pooling="identical to pretrained",
                random_control="self",
                claim_role="negative_control",
            ),
        ],
        primary_contrast={
            "metric": "AUPRC",
            "method": "geneformer_contextual_plus_cheap",
            "reference": "cheap_historical",
            "decision_rule": (
                "SUPPORTED only when the paired gene-bootstrap 95% interval for delta AUPRC is "
                "strictly above zero; otherwise NOT_SUPPORTED or DETECTABLE_LOSS."
            ),
        },
        secondary_methods=[
            "scgpt_contextual_plus_cheap",
            "geneformer_random_contextual_plus_cheap",
            "scgpt_random_contextual_plus_cheap",
            "manual_transcriptomic_plus_cheap",
            "geneformer_static_plus_cheap",
            "scgpt_static_plus_cheap",
        ],
        forbidden_actions=[
            "Reuse v0.4.0 holdout labels as an independent v0.5 evaluation.",
            "Choose the snapshot date after inspecting model scores or metrics.",
            (
                "Change sampling, context definitions, PCA budget or alpha grid after future "
                "labels open."
            ),
            "Promote a secondary method to confirmatory status after results are known.",
            "Treat GSE120575 TPM as raw-count input to Geneformer or scGPT.",
            "Collapse discordant GSE179994 lesions to a patient-level response label.",
        ],
        historical_snapshots={
            "open_targets": "26.06",
            "chembl": "37",
            "clinicaltrials_version_endpoint": "https://clinicaltrials.gov/api/v2/version",
            "publication_cutoff": "2026-08-04",
        },
        known_limitations=[
            "Geneformer and scGPT pretraining-accession overlap remains unresolved.",
            "Only melanoma and NSCLC provide real raw-count cellular contexts.",
            (
                "The clinical endpoint is operational and sparse; absence of entry is not "
                "biological failure."
            ),
            (
                "The final v0.5 result cannot exist until a scheduled future snapshot passes "
                "the power gate."
            ),
        ],
    )


def write_default_protocol(path: Path) -> ContextualProtocol:
    protocol = default_contextual_protocol()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(protocol.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return protocol


def _stable_digest(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def deterministic_context_sample(
    frame: pd.DataFrame,
    *,
    cell_id_column: str,
    stratum_columns: list[str],
    maximum_per_stratum: int,
    minimum_per_stratum: int,
    seed: int,
) -> pd.DataFrame:
    """Select cells reproducibly without depending on input row order.

    Strata below ``minimum_per_stratum`` are excluded. For retained strata, the
    lexicographically smallest SHA-256 digests are selected up to the cap.
    """
    required = [cell_id_column, *stratum_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing sampling columns: {missing}")
    if maximum_per_stratum < minimum_per_stratum:
        raise ValueError("maximum_per_stratum cannot be below minimum_per_stratum")
    working = frame.loc[:, required].copy()
    working[cell_id_column] = working[cell_id_column].astype(str)
    if working[cell_id_column].duplicated().any():
        duplicates = working.loc[working[cell_id_column].duplicated(), cell_id_column].head(5)
        raise ValueError(f"Cell identifiers must be unique; examples: {duplicates.tolist()}")
    for column in stratum_columns:
        working[column] = working[column].fillna("unknown").astype(str)
    working["_digest"] = working[cell_id_column].map(lambda value: _stable_digest(value, seed))
    selected: list[pd.DataFrame] = []
    group_key: str | list[str] = (
        stratum_columns[0] if len(stratum_columns) == 1 else stratum_columns
    )
    for _, group in working.groupby(group_key, sort=True, dropna=False):
        if len(group) < minimum_per_stratum:
            continue
        selected.append(
            group.sort_values("_digest", kind="mergesort").head(
                min(maximum_per_stratum, len(group))
            )
        )
    if not selected:
        return working.iloc[0:0].drop(columns="_digest")
    result = pd.concat(selected, ignore_index=True).drop(columns="_digest")
    return result.sort_values([*stratum_columns, cell_id_column], kind="mergesort").reset_index(
        drop=True
    )


def deterministic_gene_fold(gene_id: str, protocol: ContextualProtocol) -> int:
    split = protocol.gene_split
    salt = str(split["salt"])
    modulus = int(split["modulus"])
    digest = hashlib.sha256(f"{salt}|{gene_id}".encode()).hexdigest()
    return int(digest, 16) % modulus


def _read_index(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return pd.read_csv(handle)
    return pd.read_csv(path)


def write_contextual_artifact(
    root: Path,
    *,
    protocol_path: Path,
    model_id: str,
    control_model_id: str,
    checkpoint_sha256: str,
    input_hashes: dict[str, str],
    index: pd.DataFrame,
    pretrained: npt.NDArray[Any],
    random: npt.NDArray[Any],
    selected_cells: int,
    represented_samples: int,
    sequence_length: int,
    sampling_seed: int,
    pooling: str,
    upstream_versions: dict[str, str],
) -> ContextualArtifactManifest:
    """Write one complete, hash-linked contextual export."""
    missing = [column for column in CONTEXT_INDEX_COLUMNS if column not in index.columns]
    if missing:
        raise ValueError(f"Context index missing columns: {missing}")
    if pretrained.ndim != 2 or random.ndim != 2:
        raise ValueError("Contextual embeddings must be two-dimensional")
    if pretrained.shape != random.shape:
        raise ValueError("Pretrained and random embeddings must have identical shape")
    if len(index) != pretrained.shape[0]:
        raise ValueError("Index rows must equal embedding rows")
    if not np.isfinite(pretrained).all() or not np.isfinite(random).all():
        raise ValueError("Contextual embeddings contain non-finite values")
    if index[CONTEXT_INDEX_COLUMNS[:5]].duplicated().any():
        raise ValueError("Context index contains duplicate gene-context rows")

    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "contextual_index.csv.gz"
    pretrained_path = root / "pretrained_embeddings.npy"
    random_path = root / "random_embeddings.npy"
    manifest_path = root / "CONTEXTUAL_MANIFEST.json"
    index.loc[:, CONTEXT_INDEX_COLUMNS].to_csv(index_path, index=False, compression="gzip")
    np.save(pretrained_path, np.asarray(pretrained, dtype=np.float32), allow_pickle=False)
    np.save(random_path, np.asarray(random, dtype=np.float32), allow_pickle=False)
    contexts = int(index[CONTEXT_INDEX_COLUMNS[1:5]].drop_duplicates().shape[0])
    manifest = ContextualArtifactManifest(
        schema_version="fmva-contextual-artifact-v1",
        protocol_id=json.loads(protocol_path.read_text(encoding="utf-8"))["protocol_id"],
        protocol_sha256=sha256_file(protocol_path),
        model_id=model_id,
        control_model_id=control_model_id,
        status="COMPLETE_PRETRAINED_AND_SHAPE_MATCHED_RANDOM",
        created_at_utc=datetime.now(UTC).isoformat(),
        checkpoint_sha256=checkpoint_sha256,
        input_hashes=dict(sorted(input_hashes.items())),
        index_file=index_path.name,
        pretrained_file=pretrained_path.name,
        random_file=random_path.name,
        index_sha256=sha256_file(index_path),
        pretrained_sha256=sha256_file(pretrained_path),
        random_sha256=sha256_file(random_path),
        rows=len(index),
        embedding_width=int(pretrained.shape[1]),
        selected_cells=selected_cells,
        represented_samples=represented_samples,
        contexts=contexts,
        sequence_length=sequence_length,
        sampling_seed=sampling_seed,
        pooling=pooling,
        claim_scope="CONTEXTUAL_DEVELOPMENT_ONLY_FUTURE_HOLDOUT_UNOPENED",
        upstream_versions=dict(sorted(upstream_versions.items())),
    )
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_contextual_artifact(
    root: Path,
    *,
    protocol_path: Path | None = None,
) -> ContextualValidationReport:
    """Validate hashes, dimensions, context uniqueness and protocol linkage."""
    root = root.resolve()
    manifest_path = root / "CONTEXTUAL_MANIFEST.json"
    if not manifest_path.exists():
        return ContextualValidationReport(
            status=ContextualAuditStatus.FAIL,
            root=str(root),
            errors=["Missing CONTEXTUAL_MANIFEST.json"],
        )
    try:
        manifest = ContextualArtifactManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except Exception as error:
        return ContextualValidationReport(
            status=ContextualAuditStatus.FAIL,
            root=str(root),
            errors=[f"Invalid contextual manifest: {error}"],
        )

    invariants: list[ContextualInvariant] = []
    warnings: list[str] = []
    errors: list[str] = []

    def add(name: str, condition: bool, detail: str, *, warning: bool = False) -> None:
        if condition:
            status = ContextualCheckStatus.PASS
        elif warning:
            status = ContextualCheckStatus.WARNING
            warnings.append(f"{name}: {detail}")
        else:
            status = ContextualCheckStatus.FAIL
            errors.append(f"{name}: {detail}")
        invariants.append(ContextualInvariant(name=name, status=status, detail=detail))

    files = {
        "index": root / manifest.index_file,
        "pretrained": root / manifest.pretrained_file,
        "random": root / manifest.random_file,
    }
    for label, path in files.items():
        add(f"{label}_exists", path.is_file(), f"{label} file exists: {path.name}")
    if errors:
        return ContextualValidationReport(
            status=ContextualAuditStatus.FAIL,
            root=str(root),
            model_id=manifest.model_id,
            invariants=invariants,
            errors=errors,
        )

    add(
        "index_hash",
        sha256_file(files["index"]) == manifest.index_sha256,
        "Index SHA-256 matches the manifest.",
    )
    add(
        "pretrained_hash",
        sha256_file(files["pretrained"]) == manifest.pretrained_sha256,
        "Pretrained embedding SHA-256 matches the manifest.",
    )
    add(
        "random_hash",
        sha256_file(files["random"]) == manifest.random_sha256,
        "Random-control embedding SHA-256 matches the manifest.",
    )
    if protocol_path is not None:
        add(
            "protocol_hash",
            protocol_path.is_file() and sha256_file(protocol_path) == manifest.protocol_sha256,
            "The artifact is linked to the exact frozen protocol bytes.",
        )

    try:
        index = _read_index(files["index"])
        pretrained = np.load(files["pretrained"], allow_pickle=False, mmap_mode="r")
        random = np.load(files["random"], allow_pickle=False, mmap_mode="r")
    except Exception as error:
        errors.append(f"Could not read contextual outputs: {error}")
        return ContextualValidationReport(
            status=ContextualAuditStatus.FAIL,
            root=str(root),
            model_id=manifest.model_id,
            invariants=invariants,
            warnings=warnings,
            errors=errors,
        )

    missing_columns = [column for column in CONTEXT_INDEX_COLUMNS if column not in index.columns]
    add(
        "index_schema",
        not missing_columns,
        f"Index contains all required columns; missing={missing_columns}.",
    )
    add(
        "matrix_shape_match",
        pretrained.shape == random.shape == (manifest.rows, manifest.embedding_width),
        (
            f"Expected {(manifest.rows, manifest.embedding_width)}, observed pretrained="
            f"{pretrained.shape}, random={random.shape}."
        ),
    )
    add(
        "finite_embeddings",
        bool(np.isfinite(pretrained).all() and np.isfinite(random).all()),
        "Both embedding matrices contain only finite values.",
    )
    if not missing_columns:
        duplicate_count = int(index[CONTEXT_INDEX_COLUMNS[:5]].duplicated().sum())
        add(
            "unique_gene_context_rows",
            duplicate_count == 0,
            f"Duplicate gene-context rows: {duplicate_count}.",
        )
        add(
            "positive_support_counts",
            bool((index["n_cells"] > 0).all() and (index["n_samples"] > 0).all()),
            "All exported rows have positive cell and sample support.",
        )
        observed_contexts = int(index[CONTEXT_INDEX_COLUMNS[1:5]].drop_duplicates().shape[0])
        add(
            "context_count",
            observed_contexts == manifest.contexts,
            f"Manifest contexts={manifest.contexts}; observed={observed_contexts}.",
        )
        low_support = int((index["n_samples"] < 2).sum())
        add(
            "multi_sample_support",
            low_support == 0,
            f"Rows supported by fewer than two samples: {low_support}.",
            warning=True,
        )

    variance_pretrained = float(np.var(np.asarray(pretrained)))
    variance_random = float(np.var(np.asarray(random)))
    add(
        "nondegenerate_pretrained",
        math.isfinite(variance_pretrained) and variance_pretrained > 0.0,
        f"Pretrained embedding variance={variance_pretrained:.6g}.",
    )
    add(
        "nondegenerate_random",
        math.isfinite(variance_random) and variance_random > 0.0,
        f"Random embedding variance={variance_random:.6g}.",
    )

    status = (
        ContextualAuditStatus.FAIL
        if errors
        else ContextualAuditStatus.PASS_WITH_WARNINGS
        if warnings
        else ContextualAuditStatus.PASS
    )
    summary = {
        "rows": manifest.rows,
        "embedding_width": manifest.embedding_width,
        "selected_cells": manifest.selected_cells,
        "represented_samples": manifest.represented_samples,
        "contexts": manifest.contexts,
        "pretrained_variance": variance_pretrained,
        "random_variance": variance_random,
    }
    return ContextualValidationReport(
        status=status,
        root=str(root),
        model_id=manifest.model_id,
        summary=summary,
        invariants=invariants,
        warnings=warnings,
        errors=errors,
    )


def snapshot_dates(protocol: ContextualProtocol) -> Iterable[date]:
    """Yield the preregistered quarterly label-only snapshot dates."""
    current = date.fromisoformat(protocol.snapshot_schedule.first_eligible_snapshot)
    maximum = date.fromisoformat(protocol.snapshot_schedule.maximum_snapshot_date)
    months = protocol.snapshot_schedule.cadence_months
    while current <= maximum:
        yield current
        month_index = current.month - 1 + months
        year = current.year + month_index // 12
        month = month_index % 12 + 1
        day = min(current.day, 28)
        current = date(year, month, day)
