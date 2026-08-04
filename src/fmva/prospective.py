"""Label-only snapshot gates and frozen-score contracts for FMVA v0.5."""

from __future__ import annotations

import json
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from fmva.contextual import ContextualProtocol, canonical_json_sha256, sha256_file


class GateStatus(StrEnum):
    PASS = "PASS"
    WAIT = "WAIT"
    FAIL = "FAIL"


class SnapshotGateReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: GateStatus
    protocol_id: str
    snapshot_date: str
    positive_pairs: int = Field(ge=0)
    positive_genes: int = Field(ge=0)
    indications_with_positive: int = Field(ge=0)
    holdout_positive_pairs: int = Field(ge=0)
    holdout_positive_genes: int = Field(ge=0)
    thresholds: dict[str, int]
    failures: list[str]
    label_file_sha256: str | None = None


class FrozenScoreManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    protocol_id: str
    protocol_sha256: str
    status: str
    candidate_id_file: str
    scores_file: str
    configuration_file: str
    candidate_id_sha256: str
    scores_sha256: str
    configuration_sha256: str
    rows: int = Field(ge=1)
    methods: list[str]
    label_columns_present: list[str]
    claim_scope: str


FORBIDDEN_SCORE_GATE_TOKENS = ("score", "probability", "prediction", "rank", "auprc", "auroc")
LABEL_COLUMN = "label_new_clinical_entry"
ID_COLUMNS = ["target_gene", "indication", "gene_fold"]


def _score_like_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        column
        for column in frame.columns
        if any(token in column.lower() for token in FORBIDDEN_SCORE_GATE_TOKENS)
    )


def evaluate_label_only_power_gate(
    labels: pd.DataFrame,
    protocol: ContextualProtocol,
    *,
    snapshot_date: str,
    label_file_sha256: str | None = None,
) -> SnapshotGateReport:
    """Evaluate the preregistered gate without accepting score-like columns."""
    score_columns = _score_like_columns(labels)
    if score_columns:
        raise ValueError(f"Label-only gate received score-like columns: {score_columns}")
    required = {*ID_COLUMNS, LABEL_COLUMN}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Snapshot labels missing columns: {sorted(missing)}")
    if labels.duplicated(["target_gene", "indication"]).any():
        raise ValueError("Duplicate target-indication pairs in snapshot labels")

    parsed_snapshot = date.fromisoformat(snapshot_date)
    first = date.fromisoformat(protocol.snapshot_schedule.first_eligible_snapshot)
    maximum = date.fromisoformat(protocol.snapshot_schedule.maximum_snapshot_date)
    if parsed_snapshot < first:
        raise ValueError("Snapshot precedes the first eligible snapshot")
    if parsed_snapshot > maximum:
        raise ValueError("Snapshot exceeds the preregistered maximum snapshot date")

    positive = labels[labels[LABEL_COLUMN].astype(bool)].copy()
    holdout_fold = int(protocol.gene_split["holdout_fold"])
    holdout = positive[positive["gene_fold"].astype(int).eq(holdout_fold)]
    values = {
        "positive_pairs": len(positive),
        "positive_genes": positive["target_gene"].astype(str).nunique(),
        "indications_with_positive": positive["indication"].astype(str).nunique(),
        "holdout_positive_pairs": len(holdout),
        "holdout_positive_genes": holdout["target_gene"].astype(str).nunique(),
    }
    thresholds = {
        "positive_pairs": protocol.power_gate.minimum_positive_pairs,
        "positive_genes": protocol.power_gate.minimum_positive_genes,
        "indications_with_positive": protocol.power_gate.minimum_indications_with_positive,
        "holdout_positive_pairs": protocol.power_gate.minimum_holdout_positive_pairs,
        "holdout_positive_genes": protocol.power_gate.minimum_holdout_positive_genes,
    }
    failures = [
        f"{name}={values[name]} below threshold={threshold}"
        for name, threshold in thresholds.items()
        if values[name] < threshold
    ]
    status = GateStatus.PASS if not failures else GateStatus.WAIT
    return SnapshotGateReport(
        status=status,
        protocol_id=protocol.protocol_id,
        snapshot_date=snapshot_date,
        **values,
        thresholds=thresholds,
        failures=failures,
        label_file_sha256=label_file_sha256,
    )


def write_label_only_gate_report(
    labels_path: Path,
    protocol_path: Path,
    output_path: Path,
    *,
    snapshot_date: str,
) -> SnapshotGateReport:
    protocol = ContextualProtocol.model_validate_json(protocol_path.read_text(encoding="utf-8"))
    labels = (
        pd.read_parquet(labels_path)
        if labels_path.suffix == ".parquet"
        else pd.read_csv(labels_path)
    )
    report = evaluate_label_only_power_gate(
        labels,
        protocol,
        snapshot_date=snapshot_date,
        label_file_sha256=sha256_file(labels_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report


def freeze_future_scores(
    candidate_ids: pd.DataFrame,
    scores: pd.DataFrame,
    configuration: dict[str, Any],
    protocol: ContextualProtocol,
    output_dir: Path,
) -> FrozenScoreManifest:
    """Write future scores while rejecting labels and non-identical row IDs."""
    label_like = sorted(
        column
        for column in scores.columns
        if "label" in column.lower()
        or "outcome" in column.lower()
        or "event_date" in column.lower()
    )
    if label_like:
        raise ValueError(f"Future score freeze contains label-like columns: {label_like}")
    missing_ids = set(ID_COLUMNS) - set(candidate_ids.columns)
    if missing_ids:
        raise ValueError(f"Candidate IDs missing columns: {sorted(missing_ids)}")
    if candidate_ids.duplicated(["target_gene", "indication"]).any():
        raise ValueError("Duplicate candidate target-indication IDs")

    methods = sorted(column for column in scores.columns if column not in ID_COLUMNS)
    if not methods:
        raise ValueError("No method scores supplied")
    if set(ID_COLUMNS) - set(scores.columns):
        raise ValueError("Scores do not contain the complete candidate ID key")

    ids_sorted = candidate_ids[ID_COLUMNS].copy().sort_values(ID_COLUMNS).reset_index(drop=True)
    scores_sorted = (
        scores[[*ID_COLUMNS, *methods]].copy().sort_values(ID_COLUMNS).reset_index(drop=True)
    )
    if not ids_sorted.equals(scores_sorted[ID_COLUMNS]):
        raise ValueError("Candidate IDs and score rows are not identical")
    for method in methods:
        if not pd.to_numeric(scores_sorted[method], errors="coerce").notna().all():
            raise ValueError(f"Non-numeric or missing frozen scores for {method}")

    output_dir.mkdir(parents=True, exist_ok=True)
    ids_path = output_dir / "future_candidate_ids.csv.gz"
    scores_path = output_dir / "future_method_scores_frozen.csv.gz"
    configuration_path = output_dir / "future_configuration_freeze.json"
    compression = {"method": "gzip", "mtime": 0}
    ids_sorted.to_csv(ids_path, index=False, compression=compression)
    scores_sorted.to_csv(scores_path, index=False, compression=compression)
    configuration_path.write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    protocol_payload = protocol.model_dump(mode="json")
    manifest = FrozenScoreManifest(
        schema_version="fmva-future-score-freeze-v1",
        protocol_id=protocol.protocol_id,
        protocol_sha256=canonical_json_sha256(protocol_payload),
        status="FROZEN_BEFORE_FUTURE_LABEL_ACCESS",
        candidate_id_file=ids_path.name,
        scores_file=scores_path.name,
        configuration_file=configuration_path.name,
        candidate_id_sha256=sha256_file(ids_path),
        scores_sha256=sha256_file(scores_path),
        configuration_sha256=sha256_file(configuration_path),
        rows=len(ids_sorted),
        methods=methods,
        label_columns_present=[],
        claim_scope="PROSPECTIVE_SCORES_ONLY_NO_FUTURE_LABELS",
    )
    (output_dir / "FUTURE_SCORE_FREEZE_MANIFEST.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "FUTURE_SCORES_FROZEN.txt").write_text(
        "Future method scores frozen before future label access.\n", encoding="utf-8"
    )
    return manifest


def validate_future_score_freeze(root: Path, protocol: ContextualProtocol) -> FrozenScoreManifest:
    manifest_path = root / "FUTURE_SCORE_FREEZE_MANIFEST.json"
    manifest = FrozenScoreManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if manifest.protocol_id != protocol.protocol_id:
        raise ValueError("Score freeze protocol ID mismatch")
    expected_protocol_hash = canonical_json_sha256(protocol.model_dump(mode="json"))
    if manifest.protocol_sha256 != expected_protocol_hash:
        raise ValueError("Score freeze protocol hash mismatch")
    files = {
        "candidate": root / manifest.candidate_id_file,
        "scores": root / manifest.scores_file,
        "configuration": root / manifest.configuration_file,
    }
    expected = {
        "candidate": manifest.candidate_id_sha256,
        "scores": manifest.scores_sha256,
        "configuration": manifest.configuration_sha256,
    }
    actual = {name: sha256_file(path) for name, path in files.items()}
    if actual != expected:
        raise ValueError(f"Future score freeze hash mismatch: {actual} != {expected}")
    scores = pd.read_csv(files["scores"])
    if any("label" in column.lower() for column in scores.columns):
        raise ValueError("Frozen score table contains a label column")
    if len(scores) != manifest.rows:
        raise ValueError("Frozen score row count mismatch")
    return manifest
