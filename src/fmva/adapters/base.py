"""Representation-adapter interfaces."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd


class RepresentationAdapter(Protocol):
    """Protocol shared by local and external representation methods."""

    @property
    def model_id(self) -> str:
        """Stable adapter identifier."""
        ...

    def embed(self, frame: pd.DataFrame, feature_columns: list[str]) -> npt.NDArray[Any]:
        """Return one embedding vector per input gene."""
        ...
