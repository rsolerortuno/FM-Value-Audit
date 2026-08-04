"""Validated data structures used across the audit harness."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LeakageStatus(StrEnum):
    """Allowed pretraining/evaluation overlap states."""

    RESOLVED_CLEAN = "RESOLVED_CLEAN"
    RESOLVED_OVERLAP = "RESOLVED_OVERLAP"
    UNRESOLVED = "UNRESOLVED"


class Availability(StrEnum):
    """Execution availability for a registered representation method."""

    EXECUTED = "EXECUTED"
    EXECUTED_SUBSET = "EXECUTED_SUBSET"
    LOCAL_ONLY = "LOCAL_ONLY"
    MOCK_ONLY = "MOCK_ONLY"
    NOT_COMPUTED = "NOT_COMPUTED"


class ModelEntry(BaseModel):
    """One model registry entry."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1)
    family: str = Field(min_length=1)
    checkpoint_identifier: str = Field(min_length=1)
    version: str = Field(min_length=1)
    licence: str = Field(min_length=1)
    documented_pretraining_corpora: list[str]
    parameter_count: int | None = Field(default=None, ge=1)
    leakage_status: LeakageStatus
    availability: Availability
    enabled_by_default: bool
    notes: str = ""

    @model_validator(mode="after")
    def validate_default_provenance(self) -> ModelEntry:
        """Default models must be fully specified and locally executable."""
        unresolved_tokens = {"UNVERIFIED", "NOT_CONFIGURED", "UNKNOWN"}
        fields = {self.checkpoint_identifier, self.version, self.licence}
        corpus_tokens = set(self.documented_pretraining_corpora)
        if self.enabled_by_default:
            if fields & unresolved_tokens or corpus_tokens & unresolved_tokens:
                msg = f"Default model {self.model_id!r} has unresolved provenance"
                raise ValueError(msg)
            if self.parameter_count is None:
                msg = f"Default model {self.model_id!r} lacks a parameter count"
                raise ValueError(msg)
            if self.availability not in {Availability.EXECUTED, Availability.LOCAL_ONLY}:
                msg = f"Default model {self.model_id!r} is not executable"
                raise ValueError(msg)
        if self.availability == Availability.MOCK_ONLY and self.enabled_by_default:
            raise ValueError("Mock-only models cannot be enabled by default")
        return self


class ModelRegistry(BaseModel):
    """Top-level model registry."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    models: list[ModelEntry]

    @model_validator(mode="after")
    def validate_registry(self) -> ModelRegistry:
        """Enforce schema version and unique model identifiers."""
        if self.schema_version != "1.0":
            raise ValueError("Unsupported registry schema version")
        model_ids = [model.model_id for model in self.models]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("Model identifiers must be unique")
        return self

    def get(self, model_id: str) -> ModelEntry:
        """Return a model entry or raise a clear key error."""
        for model in self.models:
            if model.model_id == model_id:
                return model
        raise KeyError(f"Unknown model_id: {model_id}")


class EffortRecord(BaseModel):
    """Measured tuning and execution effort for one method."""

    model_config = ConfigDict(extra="forbid")

    method: str
    configurations_tried: list[str]
    tuning_wall_seconds: float = Field(ge=0.0)
    final_fit_wall_seconds: float = Field(ge=0.0)
    cpu_seconds: float = Field(ge=0.0)
    gpu_hours: float = Field(ge=0.0)
    seed: int
    selected_configuration: str


class MetricInterval(BaseModel):
    """Point estimate and optional percentile confidence interval."""

    model_config = ConfigDict(extra="forbid")

    estimate: float
    lower: float | None = None
    upper: float | None = None
    valid_bootstrap_replicates: int | None = None


class MethodResult(BaseModel):
    """Serializable evaluation result for one method."""

    model_config = ConfigDict(extra="forbid")

    method: str
    leakage_status: LeakageStatus
    metrics: dict[str, float | None]
    metric_intervals: dict[str, MetricInterval]
    excess_over_popularity: dict[str, MetricInterval]
    comparison_state: str
    compute_normalised_value: dict[str, float | None]
    effort: EffortRecord
    metadata: dict[str, Any] = Field(default_factory=dict)
