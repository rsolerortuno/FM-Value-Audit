"""Deterministic randomly initialised MLP controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass(frozen=True)
class RandomMLPAdapter:
    """A fixed random two-layer MLP used as an executable control."""

    model_id: str
    seed: int
    hidden_width: int
    embedding_width: int

    def parameter_count_for_input(self, input_width: int) -> int:
        """Return exact affine parameter count for a supplied input width."""
        return (
            input_width * self.hidden_width
            + self.hidden_width
            + self.hidden_width * self.embedding_width
            + self.embedding_width
        )

    def _weights(
        self, input_width: int
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
        rng = np.random.default_rng(self.seed)
        first = rng.normal(
            0.0,
            1.0 / np.sqrt(input_width),
            (input_width, self.hidden_width),
        )
        first_bias = rng.normal(0.0, 0.05, self.hidden_width)
        second = rng.normal(
            0.0,
            1.0 / np.sqrt(self.hidden_width),
            (self.hidden_width, self.embedding_width),
        )
        second_bias = rng.normal(0.0, 0.05, self.embedding_width)
        return first, first_bias, second, second_bias

    @staticmethod
    def _normalisation(
        reference_matrix: npt.NDArray[Any],
        reference_indices: npt.NDArray[Any],
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        reference = reference_matrix[reference_indices]
        means = reference.mean(axis=0)
        scales = reference.std(axis=0)
        scales[scales == 0.0] = 1.0
        return means, scales

    def embed_arrays(
        self,
        reference_matrix: npt.NDArray[Any],
        target_matrix: npt.NDArray[Any],
        reference_indices: npt.NDArray[Any],
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Embed reference and target arrays using training-only normalisation."""
        if reference_matrix.ndim != 2 or target_matrix.ndim != 2:
            raise ValueError("Input matrices must be two-dimensional")
        if reference_matrix.shape[1] != target_matrix.shape[1]:
            raise ValueError("Reference and target matrices must have equal feature width")
        if not np.isfinite(reference_matrix).all() or not np.isfinite(target_matrix).all():
            raise ValueError("Random-MLP inputs must be finite")
        means, scales = self._normalisation(reference_matrix, reference_indices)
        reference_scaled = (reference_matrix - means) / scales
        target_scaled = (target_matrix - means) / scales
        first, first_bias, second, second_bias = self._weights(reference_matrix.shape[1])
        reference_hidden = np.tanh(reference_scaled @ first + first_bias)
        target_hidden = np.tanh(target_scaled @ first + first_bias)
        return (
            np.tanh(reference_hidden @ second + second_bias),
            np.tanh(target_hidden @ second + second_bias),
        )

    def embed_with_reference(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
        reference_indices: npt.NDArray[Any],
    ) -> npt.NDArray[Any]:
        """Embed a frame using normalisation fitted on supplied indices."""
        matrix = frame.loc[:, feature_columns].to_numpy(dtype=float)
        embeddings, _ = self.embed_arrays(matrix, matrix, reference_indices)
        return embeddings

    def embed(self, frame: pd.DataFrame, feature_columns: list[str]) -> npt.NDArray[Any]:
        """Embed all rows, fitting normalisation on all rows for generic adapter use."""
        if not feature_columns:
            raise ValueError("feature_columns cannot be empty")
        matrix = frame.loc[:, feature_columns].to_numpy(dtype=float)
        indices = np.arange(len(frame), dtype=int)
        embeddings, _ = self.embed_arrays(matrix, matrix, indices)
        return embeddings
