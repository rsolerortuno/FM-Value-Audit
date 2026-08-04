"""Identifier-level pretraining/evaluation overlap audits."""

from __future__ import annotations

from dataclasses import dataclass

from fmva.schemas import LeakageStatus, ModelEntry


@dataclass(frozen=True)
class LeakageAudit:
    """Exact-overlap result for one model and evaluation identifier set."""

    model_id: str
    status: LeakageStatus
    evaluation_identifier_count: int
    documented_corpus_identifier_count: int
    overlap_identifiers: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation."""
        return {
            "model_id": self.model_id,
            "status": self.status.value,
            "evaluation_identifier_count": self.evaluation_identifier_count,
            "documented_corpus_identifier_count": self.documented_corpus_identifier_count,
            "overlap_count": len(self.overlap_identifiers),
            "overlap_identifiers": list(self.overlap_identifiers),
            "reason": self.reason,
        }


def _normalise(values: list[str]) -> set[str]:
    return {value.strip().casefold() for value in values if value.strip()}


def audit_overlap(model: ModelEntry, evaluation_identifiers: list[str]) -> LeakageAudit:
    """Assign a leakage status using exact normalised identifier overlap."""
    evaluation = _normalise(evaluation_identifiers)
    documented = _normalise(model.documented_pretraining_corpora)
    unresolved_tokens = {"unverified", "unknown", "not_configured"}

    if model.leakage_status == LeakageStatus.UNRESOLVED or documented & unresolved_tokens:
        return LeakageAudit(
            model_id=model.model_id,
            status=LeakageStatus.UNRESOLVED,
            evaluation_identifier_count=len(evaluation),
            documented_corpus_identifier_count=len(documented),
            overlap_identifiers=(),
            reason="Pretraining corpus documentation is incomplete or explicitly unresolved.",
        )

    overlap = tuple(sorted(evaluation & documented))
    if overlap:
        return LeakageAudit(
            model_id=model.model_id,
            status=LeakageStatus.RESOLVED_OVERLAP,
            evaluation_identifier_count=len(evaluation),
            documented_corpus_identifier_count=len(documented),
            overlap_identifiers=overlap,
            reason="At least one exact normalised dataset identifier overlaps.",
        )

    return LeakageAudit(
        model_id=model.model_id,
        status=LeakageStatus.RESOLVED_CLEAN,
        evaluation_identifier_count=len(evaluation),
        documented_corpus_identifier_count=len(documented),
        overlap_identifiers=(),
        reason="No exact overlap was found among fully documented identifiers.",
    )
