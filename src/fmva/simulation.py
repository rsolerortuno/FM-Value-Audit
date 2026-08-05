"""Synthetic benchmark with planted biological, popularity-only and noise structure."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from typing import Any
import pandas as pd

REQUIRED_SYNTHETIC_COLUMNS = {
    "gene_id",
    "planted_group",
    "entry_year",
    "label_pre_cutoff",
    "label_post_cutoff",
    "publication_count_pre_cutoff",
    "publication_count_all_time",
    "mean_expression_id",
    "differential_expression_id",
    "cell_type_specificity_id",
    "mean_expression_ood",
    "differential_expression_ood",
    "cell_type_specificity_ood",
    "network_centrality",
    "druggable_class",
    "cutoff_year",
    "evaluation_end_year",
}


def _sigmoid(values: npt.NDArray[Any]) -> npt.NDArray[Any]:
    clipped = np.clip(values, -30.0, 30.0)
    result = 1.0 / (1.0 + np.exp(-clipped))
    return np.asarray(result, dtype=np.float64)


def _make_once(
    n_genes: int,
    seed: int,
    cutoff_year: int,
    evaluation_end_year: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gene_ids = np.array([f"GENE_{index:05d}" for index in range(n_genes)])
    groups = rng.choice(
        np.array(["biology_signal", "popularity_only", "noise"]),
        size=n_genes,
        p=np.array([0.45, 0.10, 0.45]),
    )
    biology = rng.normal(0.0, 1.0, n_genes)
    popularity = rng.normal(0.0, 1.0, n_genes)

    popularity_only = groups == "popularity_only"
    biology[popularity_only] = rng.normal(-0.8, 0.35, int(popularity_only.sum()))
    popularity[popularity_only] += 1.6

    druggable_probability = _sigmoid(-0.5 + 0.9 * biology)
    druggable = rng.binomial(1, druggable_probability).astype(int)

    log_publications = 2.0 + 0.72 * popularity + 0.35 * biology
    publication_pre = rng.poisson(np.exp(np.clip(log_publications, -1.0, 6.0))).astype(int)

    development_logit = -1.40 + 2.20 * biology + 0.60 * druggable + 0.30 * popularity
    development_probability = _sigmoid(development_logit)
    development_probability[popularity_only] = 0.0
    developed = rng.binomial(1, development_probability).astype(bool)

    entry_year = np.full(n_genes, np.nan)
    developed_indices = np.flatnonzero(developed)
    if developed_indices.size:
        timing_noise = rng.normal(0.0, 0.9, developed_indices.size)
        timing = _sigmoid(-biology[developed_indices] + timing_noise)
        year_span = evaluation_end_year - 2011
        years = 2012 + np.floor(timing * year_span).astype(int)
        entry_year[developed_indices] = np.clip(years, 2012, evaluation_end_year)

    label_pre = (np.isfinite(entry_year) & (entry_year <= cutoff_year)).astype(int)
    label_post = (
        np.isfinite(entry_year) & (entry_year > cutoff_year) & (entry_year <= evaluation_end_year)
    ).astype(int)

    future_increment = developed.astype(int) * (30 + rng.poisson(25, n_genes))
    publication_all = publication_pre + future_increment

    is_biology = (groups == "biology_signal").astype(float)
    mean_id = 1.50 * biology * is_biology + 0.35 * biology + rng.normal(0.0, 0.65, n_genes)
    de_id = 1.90 * biology * is_biology + 0.25 * biology + rng.normal(0.0, 0.55, n_genes)
    specificity_id = _sigmoid(
        1.50 * biology * is_biology + 0.20 * biology + rng.normal(0.0, 0.75, n_genes)
    )

    context_shift = rng.normal(0.0, 0.65, n_genes)
    mean_ood = -0.55 * mean_id + 0.30 * context_shift + rng.normal(0.0, 0.90, n_genes)
    de_ood = -0.60 * de_id - 0.25 * context_shift + rng.normal(0.0, 0.95, n_genes)
    specificity_ood = _sigmoid(-1.10 * biology * is_biology + rng.normal(0.0, 1.00, n_genes))

    centrality = _sigmoid(0.55 * biology + 0.70 * popularity + rng.normal(0.0, 0.8, n_genes))

    return pd.DataFrame(
        {
            "gene_id": gene_ids,
            "planted_group": groups,
            "entry_year": entry_year,
            "label_pre_cutoff": label_pre,
            "label_post_cutoff": label_post,
            "publication_count_pre_cutoff": publication_pre,
            "publication_count_all_time": publication_all,
            "mean_expression_id": mean_id,
            "differential_expression_id": de_id,
            "cell_type_specificity_id": specificity_id,
            "mean_expression_ood": mean_ood,
            "differential_expression_ood": de_ood,
            "cell_type_specificity_ood": specificity_ood,
            "network_centrality": centrality,
            "druggable_class": druggable,
            "cutoff_year": np.full(n_genes, cutoff_year, dtype=int),
            "evaluation_end_year": np.full(n_genes, evaluation_end_year, dtype=int),
        }
    )


def simulate_benchmark(
    n_genes: int = 1500,
    seed: int = 20260729,
    cutoff_year: int = 2019,
    evaluation_end_year: int = 2024,
) -> pd.DataFrame:
    """Generate a deterministic synthetic target-prioritisation benchmark."""
    if n_genes < 100:
        raise ValueError("n_genes must be at least 100")
    if evaluation_end_year <= cutoff_year:
        raise ValueError("evaluation_end_year must be after cutoff_year")

    for attempt in range(20):
        frame = _make_once(n_genes, seed + attempt, cutoff_year, evaluation_end_year)
        if frame["label_pre_cutoff"].sum() >= 8 and frame["label_post_cutoff"].sum() >= 8:
            validate_synthetic(frame)
            return frame
    raise RuntimeError("Could not generate enough positives after 20 attempts")


def validate_synthetic(frame: pd.DataFrame) -> None:
    """Validate schema and planted invariants."""
    missing = REQUIRED_SYNTHETIC_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Synthetic table is missing columns: {sorted(missing)}")
    if frame["gene_id"].duplicated().any():
        raise ValueError("gene_id must be unique")
    overlap = (frame["label_pre_cutoff"] == 1) & (frame["label_post_cutoff"] == 1)
    if overlap.any():
        raise ValueError("Pre- and post-cutoff labels must not overlap")
    popularity_only = frame["planted_group"] == "popularity_only"
    if int(frame.loc[popularity_only, "label_post_cutoff"].sum()) != 0:
        raise ValueError("Popularity-only genes must be future negatives")
    if not math.isfinite(float(frame["network_centrality"].mean())):
        raise ValueError("Synthetic features must be finite")
