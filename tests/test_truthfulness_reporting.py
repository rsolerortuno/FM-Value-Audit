from __future__ import annotations

import json
from pathlib import Path

from fmva.reporting import render_report
from fmva.truthfulness import resolve_json_path, validate_readme_claims


def test_json_path_and_truthfulness(tmp_path: Path) -> None:
    payload = {"metrics": {"auprc": 0.125}}
    (tmp_path / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "README.md").write_text(
        (
            'Value 0.125\n<!-- fmva-claim path="results.json" '
            'jsonpath="metrics.auprc" value="0.125" -->\n'
        ),
        encoding="utf-8",
    )
    assert resolve_json_path(payload, "metrics.auprc") == 0.125
    assert validate_readme_claims(tmp_path) == []
    (tmp_path / "README.md").write_text(
        '<!-- fmva-claim path="results.json" jsonpath="metrics.auprc" value="0.5" -->\n',
        encoding="utf-8",
    )
    assert "Claim drift" in validate_readme_claims(tmp_path)[0]


def test_report_is_explicitly_synthetic() -> None:
    summary = {
        "run_kind": "SYNTHETIC_EXECUTED",
        "real_foundation_model_results": "NOT_COMPUTED",
        "methods": {
            "popularity": {
                "metrics": {"auprc": 0.1},
                "excess_over_popularity": {"auprc": {"estimate": 0.0, "lower": 0.0, "upper": 0.0}},
                "comparison_state": "NO_DETECTABLE_DIFFERENCE",
                "effort": {"cpu_seconds": 0.1, "configurations_tried": ["a"]},
            }
        },
        "holdout_audit": {
            "holdout_access_count": 1,
            "evaluation_gene_count": 100,
            "evaluation_positive_count": 5,
        },
        "ablations": {"test": {"status": "EXECUTED"}},
    }
    report = render_report(summary)
    assert "SYNTHETIC_EXECUTED" in report
    assert "do not establish" in report
