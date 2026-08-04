# Prospective contextual benchmark — v0.5

## Status

`v0.5.0a1` is an executable preregistration and contextual-embedding production release. It is not a
completed prospective result. No post-4-August-2026 outcome labels are included or evaluated.

The v0.4 holdout was opened on 4 August 2026. It is now development evidence and may not be reused as
an independent test of contextual methods.

## Question

Does a representation of **genes in real cellular contexts** add prospective target-prioritisation
value over a current, inexpensive historical baseline?

The confirmatory contrast is frozen as:

```text
AUPRC(geneformer_contextual_plus_cheap) - AUPRC(cheap_historical)
```

scGPT contextual, both shape-matched random encoders, manual expression and the static token methods
are secondary or negative-control analyses. A secondary winner may not replace the confirmatory
method after future labels are opened.

## Information boundary

| Item | Frozen value |
|---|---|
| Protocol cutoff | 2026-08-04 |
| Future outcome window starts | 2026-08-05 |
| First eligible snapshot | 2027-02-04 |
| Snapshot cadence | every 3 months |
| Latest permitted snapshot | 2028-08-04 |
| Snapshot choice | first chronological snapshot passing the label-only gate |
| Holdout gene fold | SHA-256 fold 0 of 5 |

The power gate may inspect only target-indication IDs, event labels and the prespecified gene fold.
It rejects score-, probability-, prediction-, rank- and metric-like columns.

## Cellular inputs

- **GSE115978:** melanoma raw-count cells, treatment-context source.
- **GSE179994:** NSCLC raw-count cells, using the official lesion-level response mapping.
- **GSE120575:** TPM; retained for the manual baseline and excluded from contextual tokenisation.

For each sample-context stratum, at most 32 cells are selected by the lowest SHA-256 value of the
cell ID and frozen seed. Strata with fewer than four eligible cells abstain. Token hidden states are
pooled gene-wise within each sample-context, then samples are given equal weight within the same
biological context. This prevents large samples from dominating the representation.

## Fixed contextual views

No view is chosen using labels. The alpha exports nine views for each gene:

1. all contexts;
2. melanoma;
3. NSCLC;
4. pretreatment;
5. on/post treatment;
6. responder;
7. nonresponder;
8. immune compartment;
9. malignant compartment.

Each view includes the mean hidden-state vector and an availability indicator. PCA is fit only on
development genes, independently for pretrained and random controls.

## Model controls

- Geneformer V1-10M pretrained contextual states;
- identical Geneformer architecture with shape-matched random states;
- scGPT whole-human pretrained contextual states;
- identical scGPT architecture with shape-matched random states.

The same cells, sequence length, gene mapping, pooling and downstream head budget are required for a
pretrained/random pair. A representation whose random control performs similarly is not evidence of
foundation-model value.

## Prospective gate

The first eligible snapshot must reach all thresholds:

- at least 20 positive target-indication pairs;
- at least 10 distinct positive genes;
- at least 5 indications with a positive event;
- at least 5 positive pairs and 5 positive genes in holdout fold 0.

Failure means `WAIT`, not threshold reduction, reseeding or manual event movement.

## Permitted alpha claims

Before the future gate passes, the release may report:

- successful execution and integrity of contextual exports;
- cell/sample/context coverage;
- development-only diagnostics;
- frozen future candidate IDs, scores, configuration and hashes;
- measured runtime and memory.

It may not report prospective AUPRC, MRR, recall, calibration or superiority.

## Leakage boundary

The public documentation available for model pretraining does not prove that every evaluation
accession was absent. Accession overlap is therefore reported as `NO_OVERLAP_FOUND`,
`POSSIBLE_OVERLAP` or `UNRESOLVED`; it is never silently converted into `NO_LEAKAGE`.
