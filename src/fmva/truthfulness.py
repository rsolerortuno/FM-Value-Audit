"""Trace README quantitative claims to committed JSON artefacts."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

CLAIM_PATTERN = re.compile(
    r'<!-- fmva-claim path="(?P<path>[^"]+)" jsonpath="(?P<jsonpath>[^"]+)" '
    r'value="(?P<value>[^"]+)" -->'
)


def resolve_json_path(payload: Any, json_path: str) -> Any:
    """Resolve a dot-separated object path."""
    current = payload
    for component in json_path.split("."):
        if isinstance(current, dict) and component in current:
            current = current[component]
        else:
            raise KeyError(f"JSON path component not found: {component}")
    return current


def _matches(declared: str, actual: Any) -> bool:
    if isinstance(actual, bool):
        return declared == str(actual)
    if isinstance(actual, (int, float)) and not isinstance(actual, bool):
        try:
            declared_float = float(declared)
        except ValueError:
            return False
        return math.isclose(declared_float, float(actual), rel_tol=0.0, abs_tol=5e-7)
    return declared == str(actual)


def validate_readme_claims(repository_root: Path) -> list[str]:
    """Return validation errors for all machine-readable README claim markers."""
    readme = repository_root / "README.md"
    text = readme.read_text(encoding="utf-8")
    matches = list(CLAIM_PATTERN.finditer(text))
    if not matches:
        return ["README contains no fmva-claim markers"]
    errors: list[str] = []
    for match in matches:
        artifact_path = repository_root / match.group("path")
        if not artifact_path.exists():
            errors.append(f"Missing artefact: {artifact_path}")
            continue
        with artifact_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        try:
            actual = resolve_json_path(payload, match.group("jsonpath"))
        except KeyError as error:
            errors.append(str(error))
            continue
        if not _matches(match.group("value"), actual):
            errors.append(
                f"Claim drift for {match.group('jsonpath')}: "
                f"README={match.group('value')} artefact={actual}"
            )
    return errors
