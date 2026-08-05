"""Adapters for locally exported external embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass(frozen=True)
class ExternalNpzAdapter:
    """Load verified, user-exported gene embeddings from an NPZ file."""

    model_id: str
    embedding_path: Path

    def embed(self, frame: pd.DataFrame, feature_columns: list[str]) -> npt.NDArray[Any]:
        """Align external embeddings to the input gene order."""
        del feature_columns
        payload = np.load(self.embedding_path, allow_pickle=False)
        if "gene_ids" not in payload or "embeddings" not in payload:
            raise ValueError("NPZ must contain gene_ids and embeddings")
        gene_ids = payload["gene_ids"].astype(str)
        embeddings = payload["embeddings"].astype(float)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(gene_ids):
            raise ValueError("Embeddings must have shape n_genes x dimensions")
        if len(set(gene_ids.tolist())) != len(gene_ids):
            raise ValueError("External gene_ids must be unique")
        lookup = {gene_id: index for index, gene_id in enumerate(gene_ids.tolist())}
        requested = frame["gene_id"].astype(str).tolist()
        missing = [gene_id for gene_id in requested if gene_id not in lookup]
        if missing:
            raise ValueError(f"External embeddings miss {len(missing)} requested genes")
        aligned = np.vstack([embeddings[lookup[gene_id]] for gene_id in requested])
        if not np.isfinite(aligned).all():
            raise ValueError("External embeddings must be finite")
        return aligned


@dataclass(frozen=True)
class MockFoundationAdapter:
    """Test-only adapter proving unavailable models cannot silently execute."""

    model_id: str
    reason: str

    def embed(self, frame: pd.DataFrame, feature_columns: list[str]) -> npt.NDArray[Any]:
        """Raise an explicit non-computation error."""
        del frame, feature_columns
        raise RuntimeError(f"NOT_COMPUTED: {self.model_id}: {self.reason}")
