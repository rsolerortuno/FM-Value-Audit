# FMVA v0.5.0a1 — Colab execution

This bundle freezes and executes the contextual-cell representation stage. It does not open future
clinical labels and does not produce a prospective efficacy claim.

## Upload

Copy these four files into:

```text
MyDrive/projects_bioinfo_2026/fm-value-audit-real/
└── temporal_benchmark/
    └── 08_notebooks/
        ├── fm_value_audit-0.5.0a1-py3-none-any.whl
        ├── FMVA_v05_00_preflight_protocol.ipynb
        ├── FMVA_v05_01_geneformer_contextual.ipynb
        └── FMVA_v05_02_scgpt_contextual.ipynb
```

Do not rename the wheel. The notebooks also support the legacy project root
`MyDrive/fm-value-audit-real`, but the canonical path above is preferred.

## Run order

1. `FMVA_v05_00_preflight_protocol.ipynb`
   - Runtime: CPU; no accelerator.
   - Run all cells.
   - Expected marker:
     `temporal_benchmark_v0.5/00_protocol/V05_PREFLIGHT_COMPLETE.txt`.

2. `FMVA_v05_01_geneformer_contextual.ipynb`
   - Runtime: GPU.
   - Run all cells.
   - Expected marker:
     `temporal_benchmark_v0.5/04_contextual_embeddings/geneformer/CONTEXTUAL_COMPLETE.txt`.

3. Delete or disconnect the previous runtime, then run
   `FMVA_v05_02_scgpt_contextual.ipynb`.
   - Runtime: GPU.
   - Run all cells.
   - Expected marker:
     `temporal_benchmark_v0.5/04_contextual_embeddings/scgpt/CONTEXTUAL_COMPLETE.txt`.

## Resume behaviour

The GPU notebooks write one immutable `.npz` checkpoint per sample-context. Re-running a notebook
reuses completed checkpoints. Do not delete `sample_context_partials/` after an interruption.

## Frozen parameters

Do not change:

- seed `20260804`;
- maximum 32 cells per sample-context;
- minimum 4 cells per sample-context;
- sequence length 512;
- batch size 4;
- context fields or response mapping;
- pretrained/random cell selection or pooling.

## Inputs verified by notebook 00

- GSE115978 raw-count H5AD;
- GSE179994 raw-count H5AD;
- official GSE179994 lesion-level response map;
- Geneformer V1-10M checkpoint and dictionaries;
- scGPT whole-human checkpoint, vocabulary and arguments;
- the frozen prospective protocol.

GSE120575 is TPM and is deliberately excluded from contextual encoder input.

## Completion message

After all three markers exist, report exactly:

```text
v0.5 contextual completado
```

No future ClinicalTrials.gov labels should be downloaded or opened during these three notebooks.
