from __future__ import annotations

import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from fmva.cli import app
from fmva.temporal import TemporalAuditStatus, validate_temporal_benchmark

REFERENCE = Path("results/temporal_benchmark_v0.4")


def _copy_reference(tmp_path: Path) -> Path:
    target = tmp_path / "temporal_reference"
    shutil.copytree(REFERENCE, target)
    return target


def _rewrite_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_committed_temporal_reference_passes_with_known_warning() -> None:
    report = validate_temporal_benchmark(REFERENCE)

    assert report.status == TemporalAuditStatus.PASS_WITH_WARNINGS
    assert report.errors == []
    assert report.summary["candidate_genes"] == 17_527
    assert report.summary["holdout_rows"] == 79_542
    assert report.summary["holdout_positives"] == 14
    assert report.summary["primary_conclusion"] == "NOT_SUPPORTED"
    assert report.summary["row_accounting_delta"] == 14
    assert len(report.warnings) == 1
    assert all(item.status != "FAIL" for item in report.invariants)


def test_configuration_hash_tampering_fails(tmp_path: Path) -> None:
    root = _copy_reference(tmp_path)
    configuration_path = root / "stage3a" / "configuration_freeze.json"
    payload = json.loads(configuration_path.read_text(encoding="utf-8"))
    payload["alpha_grid"] = [0.123]
    _rewrite_json(configuration_path, payload)

    report = validate_temporal_benchmark(root)

    assert report.status == TemporalAuditStatus.FAIL
    assert any("configuration_hash_link" in error for error in report.errors)


def test_stage3a_label_access_tampering_fails(tmp_path: Path) -> None:
    root = _copy_reference(tmp_path)
    manifest_path = root / "stage3a" / "STAGE3A_MANIFEST.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["holdout_labels_loaded"] = True
    _rewrite_json(manifest_path, payload)

    report = validate_temporal_benchmark(root)

    assert report.status == TemporalAuditStatus.FAIL
    assert any("configuration_frozen_before_labels" in error for error in report.errors)


def test_primary_conclusion_tampering_fails(tmp_path: Path) -> None:
    root = _copy_reference(tmp_path)
    conclusion_path = root / "stage3b" / "primary_conclusion.json"
    payload = json.loads(conclusion_path.read_text(encoding="utf-8"))
    payload["point_delta"] = 1.0
    _rewrite_json(conclusion_path, payload)

    report = validate_temporal_benchmark(root)

    assert report.status == TemporalAuditStatus.FAIL
    names = {item.name for item in report.invariants if item.status == "FAIL"}
    assert {"primary_delta_recomputed", "primary_conclusion_copy_matches"} <= names


def test_missing_contract_fails_cleanly(tmp_path: Path) -> None:
    root = _copy_reference(tmp_path)
    (root / "stage3b" / "primary_conclusion.json").unlink()

    report = validate_temporal_benchmark(root)

    assert report.status == TemporalAuditStatus.FAIL
    assert report.invariants == []
    assert report.errors == ["Missing temporal contract file: primary_conclusion.json"]


def test_temporal_validate_cli_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    result = CliRunner().invoke(
        app,
        ["temporal-validate", str(REFERENCE), "--output", str(output)],
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "PASS_WITH_WARNINGS"
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["primary_conclusion"] == "NOT_SUPPORTED"


def test_temporal_validate_cli_can_fail_on_warning() -> None:
    result = CliRunner().invoke(
        app,
        ["temporal-validate", str(REFERENCE), "--fail-on-warnings"],
    )

    assert result.exit_code == 2
    assert result.stdout.strip() == "PASS_WITH_WARNINGS"
