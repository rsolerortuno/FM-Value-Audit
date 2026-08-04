from __future__ import annotations

from pathlib import Path

from fmva.truthfulness import validate_readme_claims


def test_committed_readme_claims_match_results() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    assert validate_readme_claims(repository_root) == []
