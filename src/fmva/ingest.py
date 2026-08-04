"""Canonical table validation and optional network-centrality construction."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

MINIMUM_COLUMNS = {
    "gene_id",
    "entry_year",
    "label_pre_cutoff",
    "label_post_cutoff",
    "publication_count_pre_cutoff",
    "mean_expression_id",
    "differential_expression_id",
    "cell_type_specificity_id",
    "druggable_class",
    "cutoff_year",
    "evaluation_end_year",
}


def add_network_centrality(frame: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Add degree centrality from a two-column undirected interaction edge list."""
    if edges.shape[1] < 2:
        raise ValueError("Network edge table must contain at least two columns")
    source_column, target_column = edges.columns[:2]
    graph = nx.Graph()
    graph.add_edges_from(
        zip(
            edges[source_column].astype(str).tolist(),
            edges[target_column].astype(str).tolist(),
            strict=True,
        )
    )
    centrality = nx.degree_centrality(graph)
    result = frame.copy()
    result["network_centrality"] = (
        result["gene_id"].astype(str).map(centrality).fillna(0.0).astype(float)
    )
    return result


def validate_canonical(frame: pd.DataFrame) -> None:
    """Validate the minimum real/synthetic benchmark table contract."""
    missing = MINIMUM_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Canonical table is missing columns: {sorted(missing)}")
    if "network_centrality" not in frame.columns:
        raise ValueError("network_centrality is required or must be built from an edge list")
    if frame["gene_id"].duplicated().any():
        raise ValueError("gene_id must be unique")
    numeric = [column for column in MINIMUM_COLUMNS if column not in {"gene_id", "entry_year"}]
    values = frame.loc[:, numeric].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Canonical numeric fields must be finite")
    labels = set(frame["label_pre_cutoff"].astype(int).unique().tolist()) | set(
        frame["label_post_cutoff"].astype(int).unique().tolist()
    )
    if not labels <= {0, 1}:
        raise ValueError("Label columns must be binary")
