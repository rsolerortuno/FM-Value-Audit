"""Representation-adapter interfaces."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class RepresentationAdapter(Protocol):
    """Protocol shared by local and external representation methods."""

    @property
    def model_id(self) -> str:
        """Stable adapter identifier."""
        ...

    def embed(self, frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
        """Return one embedding vector per input gene."""
        ...
