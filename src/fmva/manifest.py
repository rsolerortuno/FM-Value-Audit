"""Model registry loading and validation."""

from __future__ import annotations

from pathlib import Path

import yaml

from fmva.schemas import ModelRegistry


def load_registry(path: Path) -> ModelRegistry:
    """Load and validate a YAML model registry."""
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return ModelRegistry.model_validate(payload)


def validate_registry(path: Path) -> ModelRegistry:
    """Validate semantic constraints beyond YAML parsing."""
    registry = load_registry(path)
    for model in registry.models:
        if (
            model.leakage_status.value == "RESOLVED_CLEAN"
            and model.documented_pretraining_corpora
            and any(item == "UNVERIFIED" for item in model.documented_pretraining_corpora)
        ):
            raise ValueError(f"Clean model {model.model_id!r} has unverified corpora")
        if model.checkpoint_identifier == "NOT_CONFIGURED" and model.enabled_by_default:
            raise ValueError(f"Unconfigured model {model.model_id!r} cannot be default")
    return registry
