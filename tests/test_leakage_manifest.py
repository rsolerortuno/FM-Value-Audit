from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fmva.leakage import audit_overlap
from fmva.manifest import load_registry, validate_registry
from fmva.schemas import Availability, LeakageStatus, ModelEntry, ModelRegistry


def test_registry_loads_and_defaults_are_executable() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = validate_registry(root / "registry" / "models.yaml")
    assert registry.get("random_mlp_small").enabled_by_default
    assert registry.get("geneformer_v1_10m").leakage_status == LeakageStatus.UNRESOLVED


def test_registry_rejects_unresolved_default(tmp_path: Path) -> None:
    payload = {
        "schema_version": "1.0",
        "models": [
            {
                "model_id": "bad",
                "family": "x",
                "checkpoint_identifier": "NOT_CONFIGURED",
                "version": "UNVERIFIED",
                "licence": "UNVERIFIED",
                "documented_pretraining_corpora": ["UNVERIFIED"],
                "parameter_count": None,
                "leakage_status": "UNRESOLVED",
                "availability": "MOCK_ONLY",
                "enabled_by_default": True,
                "notes": "bad",
            }
        ],
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unresolved provenance"):
        load_registry(path)


def test_leakage_audit_states() -> None:
    clean = ModelEntry(
        model_id="clean",
        family="local",
        checkpoint_identifier="local://x",
        version="1",
        licence="Apache-2.0",
        documented_pretraining_corpora=["dataset-a"],
        parameter_count=10,
        leakage_status=LeakageStatus.RESOLVED_CLEAN,
        availability=Availability.LOCAL_ONLY,
        enabled_by_default=False,
    )
    overlap = audit_overlap(clean, ["DATASET-A", "dataset-b"])
    assert overlap.status == LeakageStatus.RESOLVED_OVERLAP
    no_overlap = audit_overlap(clean, ["dataset-b"])
    assert no_overlap.status == LeakageStatus.RESOLVED_CLEAN
    unresolved = clean.model_copy(
        update={
            "model_id": "unresolved",
            "leakage_status": LeakageStatus.UNRESOLVED,
            "documented_pretraining_corpora": ["UNVERIFIED"],
        }
    )
    assert audit_overlap(unresolved, ["dataset-a"]).status == LeakageStatus.UNRESOLVED


def test_registry_rejects_duplicate_ids() -> None:
    model = ModelEntry(
        model_id="duplicate",
        family="local",
        checkpoint_identifier="local://x",
        version="1",
        licence="Apache-2.0",
        documented_pretraining_corpora=[],
        parameter_count=10,
        leakage_status=LeakageStatus.RESOLVED_CLEAN,
        availability=Availability.LOCAL_ONLY,
        enabled_by_default=False,
    )
    with pytest.raises(ValueError, match="unique"):
        ModelRegistry(schema_version="1.0", models=[model, model])
