"""Human-readable reports generated only from result artefacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fmva.io import read_json


def _format_number(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_report(summary: dict[str, Any]) -> str:
    """Render a compact Markdown benchmark report."""
    lines = [
        "# fm-value-audit benchmark report",
        "",
        f"Run kind: **{summary['run_kind']}**",
        f"Real foundation-model results: **{summary['real_foundation_model_results']}**",
        "",
        "## Primary synthetic results",
        "",
        "| Method | AUPRC | Excess vs popularity | 95% CI | State | CPU s | Configs |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for method, result in summary["methods"].items():
        interval = result["excess_over_popularity"]["auprc"]
        ci = f"[{_format_number(interval['lower'])}, {_format_number(interval['upper'])}]"
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    _format_number(result["metrics"]["auprc"]),
                    _format_number(interval["estimate"]),
                    ci,
                    result["comparison_state"],
                    _format_number(result["effort"]["cpu_seconds"]),
                    str(len(result["effort"]["configurations_tried"])),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Holdout audit",
            "",
            f"- Holdout accesses: {summary['holdout_audit']['holdout_access_count']}",
            f"- Evaluation genes: {summary['holdout_audit']['evaluation_gene_count']}",
            f"- Prospective positives: {summary['holdout_audit']['evaluation_positive_count']}",
            "",
            "## Ablations",
            "",
        ]
    )
    for name, payload in summary["ablations"].items():
        lines.append(f"- **{name}:** {payload['status']}")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "These are executed synthetic results. They validate the harness but do not establish "
            "the value of any real foundation model, gene target or disease-specific ranking.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(summary_path: Path, output_path: Path) -> None:
    """Render a report from a JSON result artefact."""
    summary = read_json(summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_report(summary), encoding="utf-8")
