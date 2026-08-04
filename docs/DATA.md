# Data contract

## Committed synthetic table

`fmva simulate` emits one row per gene with:

- `gene_id`;
- `planted_group` (`biology_signal`, `popularity_only`, `noise`);
- `entry_year`, nullable;
- `label_pre_cutoff` and `label_post_cutoff`;
- `publication_count_pre_cutoff` and `publication_count_all_time`;
- ID and OOD versions of mean expression, differential expression and specificity;
- `network_centrality` and `druggable_class`;
- `cutoff_year` and `evaluation_end_year`.

The all-time publication field is forbidden in the temporal benchmark.

## Real-data minimum contract

A real benchmark input must contain the same canonical columns after source-specific ingestion. Every
row must additionally retain source identifiers, source versions, retrieval dates and feature
as-of dates in a sidecar provenance JSON. Target entry dates must record the evidence used to choose
the first development year and any ambiguity.

No public dataset is bundled. The repository contains no patient-level data, model weights or cached
atlas objects.

## Public real-data layer added in v0.2.0

| Resource | Committed use | Raw bytes committed? | Status |
|---|---|---:|---|
| GSE115978 counts + annotations | sample-level melanoma features and model subset | No | `EXECUTED_FULL_FEATURE_LAYER` |
| GSE120575 TPM + GEO metadata | patient-state response-associated features | No | `EXECUTED_FULL_FEATURE_LAYER` |
| GSE179994 RDS + metadata | planned NSCLC OOD layer | No | integrity/metadata executed; matrix `NOT_COMPUTED_LOCAL` |
| Geneformer V1-10M | frozen CPU representation subset | No | `EXECUTED_CPU_SUBSET`, leakage `UNRESOLVED` |
| scGPT whole-human | frozen CPU representation subset | No | `EXECUTED_CPU_SUBSET`, leakage `UNRESOLVED` |

All input hashes and Drive file identifiers are in `results/real_data/input_manifest.json`. Compact
features and embeddings are committed; transport parts, matrices and model weights are excluded.


## Full-cell Stage-1 layer added in v0.3.0

The compact committed artefacts under `results/stage1_full_cells/` correspond to complete-cell
processing. The Stage-1 audit records Drive file IDs and sizes for the large H5/H5AD matrices, plus
SHA-256 hashes for every compact artefact. GSE179994 outcome metadata are superseded by the official
lesion mapping; the old patient-level response mapping is deliberately absent from v0.3.0.
