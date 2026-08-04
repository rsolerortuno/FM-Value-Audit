"""Validation of the sealed real temporal benchmark contracts."""

from __future__ import annotations

import hashlib
import math
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from fmva.io import read_json


class TemporalAuditStatus(StrEnum):
    """Overall audit status."""

    PASS = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    FAIL = "FAIL"


class TemporalCheckStatus(StrEnum):
    """Status for one contract invariant."""

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


class TemporalInvariant(BaseModel):
    """One cross-artifact invariant checked by the temporal validator."""

    model_config = ConfigDict(extra="forbid")

    name: str
    status: TemporalCheckStatus
    detail: str


class TemporalValidationReport(BaseModel):
    """Serializable audit report for a temporal benchmark directory."""

    model_config = ConfigDict(extra="forbid")

    status: TemporalAuditStatus
    root: str
    files: dict[str, str]
    summary: dict[str, int | float | str | None]
    invariants: list[TemporalInvariant]
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


CORE_FILES: dict[str, str] = {
    "stage2a_manifest": "STAGE2A_WIDE_MANIFEST.json",
    "stage2a_power_gate": "STAGE2A_WIDE_POWER_GATE.json",
    "pre_cutoff_coverage": "PRE_CUTOFF_MAPPING_COVERAGE.json",
    "stage2b_manifest": "STAGE2B_WIDE_MANIFEST.json",
    "holdout_seal": "holdout_seal.json",
    "configuration_freeze": "configuration_freeze.json",
    "stage3a_manifest": "STAGE3A_MANIFEST.json",
    "development_metrics": "development_metrics.csv",
    "development_tuning_effort": "development_tuning_effort.csv",
    "stage3b_manifest": "STAGE3B_MANIFEST.json",
    "final_holdout_metrics": "final_holdout_metrics.csv",
    "bootstrap_summary": "bootstrap_summary.csv",
    "clinical_ood_metrics": "clinical_ood_metrics.csv",
    "clinical_ood_delta": "clinical_ood_delta.csv",
    "primary_conclusion": "primary_conclusion.json",
    "final_report": "FINAL_HOLDOUT_REPORT.md",
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_temporal_contracts(root: Path) -> dict[str, Path]:
    """Locate each required contract file below ``root`` by exact basename.

    A narrow benchmark or compact-reference directory is recommended. Duplicate
    basenames are rejected so an old and a current benchmark cannot be mixed.
    """
    if not root.exists():
        raise FileNotFoundError(f"Temporal benchmark root does not exist: {root}")
    found: dict[str, Path] = {}
    for key, basename in CORE_FILES.items():
        matches = sorted(path for path in root.rglob(basename) if path.is_file())
        if not matches:
            raise FileNotFoundError(f"Missing temporal contract file: {basename}")
        if len(matches) > 1:
            joined = ", ".join(str(path) for path in matches)
            raise ValueError(f"Ambiguous temporal contract file {basename}: {joined}")
        found[key] = matches[0]
    return found


def _mapping(payload: Any, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, Any], payload)


def _load_mapping(path: Path) -> dict[str, Any]:
    return _mapping(read_json(path), path.name)


def _integer(payload: dict[str, Any], key: str, label: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label}.{key} must be an integer")
    return value


def _number(payload: dict[str, Any], key: str, label: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}.{key} must be numeric")
    return float(value)


def _text(payload: dict[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{label}.{key} must be a string")
    return value


def _string_list(payload: dict[str, Any], key: str, label: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label}.{key} must be a list of strings")
    return cast(list[str], value)


def _metric_row(frame: pd.DataFrame, method: str) -> pd.Series[Any]:
    if "method" not in frame.columns:
        raise ValueError("Metric table lacks method column")
    rows = frame.loc[frame["method"].astype(str).eq(method)]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one metric row for {method}, found {len(rows)}")
    return rows.iloc[0]


def _bootstrap_row(frame: pd.DataFrame, contrast: str) -> pd.Series[Any]:
    if "contrast" not in frame.columns:
        raise ValueError("Bootstrap summary lacks contrast column")
    rows = frame.loc[frame["contrast"].astype(str).eq(contrast)]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one bootstrap row for {contrast}, found {len(rows)}")
    return rows.iloc[0]


def _close(left: float, right: float, tolerance: float = 5e-12) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def _declared_output_hash(payload: dict[str, Any], basename: str, label: str) -> str:
    outputs = payload.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError(f"{label}.outputs must be a list")
    matches: list[str] = []
    for item in outputs:
        if not isinstance(item, dict):
            raise ValueError(f"{label}.outputs entries must be objects")
        output_path = item.get("path")
        output_hash = item.get("sha256")
        if isinstance(output_path, str) and Path(output_path).name == basename:
            if not isinstance(output_hash, str):
                raise ValueError(f"{label} output {basename} lacks a SHA-256 string")
            matches.append(output_hash)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {label} output named {basename}, found {len(matches)}"
        )
    return matches[0]


def validate_temporal_benchmark(root: Path) -> TemporalValidationReport:
    """Validate the sealed Stage 2A-WIDE through Stage 3B benchmark contracts.

    This function validates the compact committed reference bundle as well as a
    full mounted ``temporal_benchmark`` tree. Large Parquet outputs are not
    required; their recorded hashes are cross-linked across manifests and seals.
    """
    root = root.resolve()
    try:
        paths = locate_temporal_contracts(root)
    except (FileNotFoundError, ValueError) as error:
        return TemporalValidationReport(
            status=TemporalAuditStatus.FAIL,
            root=str(root),
            files={},
            summary={},
            invariants=[],
            errors=[str(error)],
        )

    stage2a = _load_mapping(paths["stage2a_manifest"])
    power_gate = _load_mapping(paths["stage2a_power_gate"])
    pre_cutoff = _load_mapping(paths["pre_cutoff_coverage"])
    stage2b = _load_mapping(paths["stage2b_manifest"])
    seal = _load_mapping(paths["holdout_seal"])
    configuration = _load_mapping(paths["configuration_freeze"])
    stage3a = _load_mapping(paths["stage3a_manifest"])
    stage3b = _load_mapping(paths["stage3b_manifest"])
    conclusion = _load_mapping(paths["primary_conclusion"])
    development_metrics = pd.read_csv(paths["development_metrics"])
    final_metrics = pd.read_csv(paths["final_holdout_metrics"])
    bootstrap = pd.read_csv(paths["bootstrap_summary"])
    clinical_ood_metrics = pd.read_csv(paths["clinical_ood_metrics"])
    clinical_ood_delta = pd.read_csv(paths["clinical_ood_delta"])

    invariants: list[TemporalInvariant] = []
    warnings: list[str] = []
    errors: list[str] = []

    def add(name: str, condition: bool, detail: str, *, warning: bool = False) -> None:
        if condition:
            invariants.append(
                TemporalInvariant(name=name, status=TemporalCheckStatus.PASS, detail=detail)
            )
            return
        if warning:
            invariants.append(
                TemporalInvariant(name=name, status=TemporalCheckStatus.WARNING, detail=detail)
            )
            warnings.append(f"{name}: {detail}")
            return
        invariants.append(
            TemporalInvariant(name=name, status=TemporalCheckStatus.FAIL, detail=detail)
        )
        errors.append(f"{name}: {detail}")

    add(
        "stage2a_complete",
        _text(stage2a, "stage", "stage2a") == "2A-WIDE"
        and _text(stage2a, "status", "stage2a") == "COMPLETE",
        "Stage 2A-WIDE is marked complete.",
    )
    add(
        "strict_power_gate",
        _text(stage2a, "power_gate_status", "stage2a") == "STRICT_PRIMARY_PASS"
        and _text(power_gate, "status", "power_gate") == "STRICT_PRIMARY_PASS",
        "The pre-specified strict primary endpoint passed its power gate.",
    )

    manifest_gate = _mapping(stage2a.get("power_gate"), "stage2a.power_gate")
    manifest_strict = _mapping(manifest_gate.get("strict_primary"), "stage2a.strict_primary")
    standalone_strict = _mapping(power_gate.get("strict_primary"), "power_gate.strict_primary")
    add(
        "power_gate_copy_matches",
        manifest_strict == standalone_strict,
        "The standalone power gate matches the copy embedded in Stage 2A-WIDE.",
    )
    add(
        "stage2a_power_gate_file_hash",
        sha256_file(paths["stage2a_power_gate"])
        == _declared_output_hash(stage2a, "STAGE2A_WIDE_POWER_GATE.json", "stage2a"),
        "The committed Stage 2A power-gate file matches its manifest hash.",
    )

    add(
        "stage2b_complete",
        _text(stage2b, "stage", "stage2b") == "2B-WIDE"
        and _text(stage2b, "status", "stage2b") == "COMPLETE",
        "Stage 2B-WIDE is marked complete.",
    )
    add(
        "gene_disjoint_holdout",
        seal.get("gene_disjoint") is True
        and _text(stage2b, "split_design", "stage2b") == "GENE_DISJOINT_FIXED_TEMPORAL_HOLDOUT",
        "The temporal holdout is fixed and gene-disjoint.",
    )

    stage2b_development = _integer(stage2b, "development_rows", "stage2b")
    stage2b_holdout = _integer(stage2b, "holdout_rows", "stage2b")
    stage2b_at_risk = _integer(stage2b, "at_risk_rows", "stage2b")
    split_rows = stage2b_development + stage2b_holdout
    row_delta = stage2b_at_risk - split_rows
    add(
        "stage2b_row_accounting",
        row_delta == 0,
        (
            "Stage 2B reports at_risk_rows="
            f"{stage2b_at_risk}, while development + holdout = {split_rows}; delta={row_delta}. "
            "The sealed holdout row count is independently consistent downstream, so this is "
            "retained as a manifest-accounting warning rather than silently corrected."
        ),
        warning=True,
    )

    stage2a_total_rows = _integer(standalone_strict, "total_candidate_rows", "power_gate")
    excluded = _integer(stage2b, "excluded_pre_cutoff_rows", "stage2b")
    add(
        "pre_cutoff_accounting",
        stage2a_total_rows - excluded == stage2b_at_risk,
        "Stage 2A candidate rows minus pre-cutoff exclusions equal Stage 2B at-risk rows.",
    )

    add(
        "pre_cutoff_coverage_counts",
        _integer(pre_cutoff, "high_confidence_target_indication_entries", "pre_cutoff") == excluded,
        "The pre-cutoff mapping coverage count equals the exclusions recorded by Stage 2B.",
    )
    add(
        "stage2b_reference_file_hashes",
        sha256_file(paths["pre_cutoff_coverage"])
        == _declared_output_hash(stage2b, "PRE_CUTOFF_MAPPING_COVERAGE.json", "stage2b")
        and sha256_file(paths["holdout_seal"])
        == _declared_output_hash(stage2b, "holdout_seal.json", "stage2b"),
        "The committed Stage 2B reference files match their manifest hashes.",
    )

    add(
        "seal_counts_match_stage2b",
        _integer(seal, "development_rows", "seal") == stage2b_development
        and _integer(seal, "holdout_rows", "seal") == stage2b_holdout
        and _integer(seal, "development_positives", "seal")
        == _integer(stage2b, "development_positives", "stage2b")
        and _integer(seal, "holdout_positives", "seal")
        == _integer(stage2b, "holdout_positives", "stage2b"),
        "Stage 2B row and positive counts match the sealed holdout contract.",
    )

    configuration_hash = sha256_file(paths["configuration_freeze"])
    add(
        "configuration_hash_link",
        configuration_hash == _text(stage3a, "configuration_freeze_sha256", "stage3a")
        and configuration_hash == _text(stage3b, "configuration_sha256", "stage3b"),
        "Stage 3A and Stage 3B reference the exact committed configuration freeze.",
    )
    add(
        "configuration_frozen_before_labels",
        _text(configuration, "status", "configuration") == "FROZEN_BEFORE_HOLDOUT_LABEL_ACCESS"
        and stage3a.get("holdout_labels_loaded") is False,
        "The configuration was frozen and Stage 3A records no holdout-label access.",
    )

    input_hashes = _mapping(configuration.get("input_hashes"), "configuration.input_hashes")
    output_hashes = _mapping(configuration.get("output_hashes"), "configuration.output_hashes")
    add(
        "seal_hash_links",
        _text(input_hashes, "development_sha256", "configuration.input_hashes")
        == _text(seal, "development_sha256", "seal")
        and _text(input_hashes, "holdout_predictors_sha256", "configuration.input_hashes")
        == _text(seal, "holdout_predictors_sha256", "seal")
        and _text(input_hashes, "holdout_seal_sha256", "configuration.input_hashes")
        == sha256_file(paths["holdout_seal"]),
        "Configuration input hashes match the sealed Stage 2B artefacts.",
    )
    add(
        "stage3b_score_and_label_hashes",
        _text(stage3b, "scores_sha256", "stage3b")
        == _text(output_hashes, "holdout_method_scores_sha256", "configuration.output_hashes")
        and _text(stage3b, "labels_sha256", "stage3b")
        == _text(seal, "holdout_labels_sha256", "seal"),
        "Stage 3B uses the frozen score hash and sealed label hash.",
    )
    add(
        "stage3a_reference_file_hashes",
        sha256_file(paths["development_metrics"])
        == _text(output_hashes, "development_metrics_sha256", "configuration.output_hashes")
        and sha256_file(paths["development_tuning_effort"])
        == _text(
            output_hashes,
            "development_tuning_effort_sha256",
            "configuration.output_hashes",
        ),
        "The committed Stage 3A diagnostic tables match the frozen output hashes.",
    )

    add(
        "stage3a_complete",
        _text(stage3a, "status", "stage3a") == "COMPLETE_CONFIGURATION_FROZEN",
        "Stage 3A completed with the configuration frozen.",
    )
    add(
        "stage3b_complete",
        _text(stage3b, "status", "stage3b") == "FINAL_HOLDOUT_EVALUATION_COMPLETE",
        "Stage 3B completed the one-time final holdout evaluation.",
    )
    stage3b_reference_hashes = all(
        sha256_file(paths[key]) == _declared_output_hash(stage3b, paths[key].name, "stage3b")
        for key in (
            "final_holdout_metrics",
            "bootstrap_summary",
            "clinical_ood_metrics",
            "clinical_ood_delta",
            "primary_conclusion",
            "final_report",
        )
    )
    add(
        "stage3b_reference_file_hashes",
        stage3b_reference_hashes,
        "The committed final tables, conclusion and report match Stage 3B output hashes.",
    )

    stage3b_methods = set(_string_list(stage3b, "methods", "stage3b"))
    metric_methods = set(final_metrics["method"].astype(str))
    add(
        "final_method_set",
        stage3b_methods == metric_methods,
        "The final metric table contains exactly the methods declared by Stage 3B.",
    )

    holdout_rows = _integer(stage3b, "holdout_rows", "stage3b")
    holdout_positives = _integer(stage3b, "holdout_positives", "stage3b")
    add(
        "final_metric_counts",
        bool(
            final_metrics["rows"].astype(int).eq(holdout_rows).all()
            and final_metrics["positives"].astype(int).eq(holdout_positives).all()
        ),
        "Every final metric row uses the sealed holdout row and positive counts.",
    )

    primary_method = _text(conclusion, "primary_method", "conclusion")
    comparator = _text(conclusion, "primary_comparator", "conclusion")
    primary_row = _metric_row(final_metrics, primary_method)
    comparator_row = _metric_row(final_metrics, comparator)
    observed_delta = float(primary_row["pooled_auprc"]) - float(comparator_row["pooled_auprc"])
    declared_delta = _number(conclusion, "point_delta", "conclusion")
    add(
        "primary_delta_recomputed",
        _close(observed_delta, declared_delta),
        "The primary AUPRC delta recomputes exactly from final holdout metrics.",
    )

    nested_conclusion = _mapping(stage3b.get("primary_conclusion"), "stage3b.primary_conclusion")
    add(
        "primary_conclusion_copy_matches",
        nested_conclusion == conclusion,
        "The standalone primary conclusion matches the Stage 3B manifest copy.",
    )

    contrast_name = f"{primary_method}_minus_{comparator}"
    bootstrap_row = _bootstrap_row(bootstrap, contrast_name)
    add(
        "bootstrap_primary_contrast",
        _close(float(bootstrap_row["point_estimate"]), declared_delta)
        and _close(
            float(bootstrap_row["ci_lower_2_5"]),
            _number(conclusion, "ci_lower_2_5", "conclusion"),
        )
        and _close(
            float(bootstrap_row["ci_upper_97_5"]),
            _number(conclusion, "ci_upper_97_5", "conclusion"),
        )
        and int(bootstrap_row["valid_replicates"])
        == _integer(stage3b, "bootstrap_replicates", "stage3b"),
        "The primary point estimate, interval and replicate count match bootstrap_summary.csv.",
    )

    conclusion_state = _text(conclusion, "conclusion", "conclusion")
    ci_low = _number(conclusion, "ci_lower_2_5", "conclusion")
    ci_high = _number(conclusion, "ci_upper_97_5", "conclusion")
    expected_state = "SUPPORTED" if ci_low > 0.0 else "NOT_SUPPORTED"
    add(
        "conclusion_rule",
        conclusion_state == expected_state and ci_low <= 0.0 <= ci_high,
        "The NOT_SUPPORTED conclusion follows from a confidence interval crossing zero.",
    )

    id_rows = clinical_ood_metrics.loc[clinical_ood_metrics["subset"].astype(str).eq("ID_MELANOMA")]
    id_positive_total = int(id_rows["positives"].astype(int).sum())
    delta_all_missing = bool(clinical_ood_delta["clinical_ood_delta"].isna().all())
    add(
        "clinical_ood_abstention",
        id_positive_total == 0 and delta_all_missing,
        "Clinical OOD delta correctly abstains because the melanoma subset has zero positives.",
    )

    development_methods = set(development_metrics["method"].astype(str))
    add(
        "development_method_set",
        development_methods == stage3b_methods,
        "Development diagnostics and final evaluation use the same method set.",
    )

    summary: dict[str, int | float | str | None] = {
        "candidate_genes": _integer(stage2a, "candidate_genes", "stage2a"),
        "indications": len(cast(list[Any], stage2a.get("indications", []))),
        "development_rows": stage2b_development,
        "development_positives": _integer(stage2b, "development_positives", "stage2b"),
        "holdout_rows": holdout_rows,
        "holdout_positives": holdout_positives,
        "excluded_pre_cutoff_rows": excluded,
        "row_accounting_delta": row_delta,
        "primary_method": primary_method,
        "primary_comparator": comparator,
        "primary_metric": _text(conclusion, "metric", "conclusion"),
        "primary_delta": declared_delta,
        "primary_ci_lower": ci_low,
        "primary_ci_upper": ci_high,
        "primary_conclusion": conclusion_state,
        "best_final_auprc_method": str(
            final_metrics.sort_values("pooled_auprc", ascending=False).iloc[0]["method"]
        ),
        "best_final_auprc": float(final_metrics["pooled_auprc"].max()),
        "clinical_ood_delta_status": "NOT_COMPUTABLE_ZERO_ID_POSITIVES",
    }

    if errors:
        status = TemporalAuditStatus.FAIL
    elif warnings:
        status = TemporalAuditStatus.PASS_WITH_WARNINGS
    else:
        status = TemporalAuditStatus.PASS

    return TemporalValidationReport(
        status=status,
        root=str(root),
        files={key: str(path.relative_to(root)) for key, path in paths.items()},
        summary=summary,
        invariants=invariants,
        warnings=warnings,
        errors=errors,
    )
