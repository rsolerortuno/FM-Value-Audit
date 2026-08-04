# Real temporal benchmark — v0.4.0

## Scope

The v0.4.0 benchmark asks whether frozen single-cell foundation-model gene representations improve
future solid-tumour target-entry ranking beyond inexpensive information available at the historical
cutoff. The evaluated representation features are static Geneformer and scGPT token geometry. They
are not contextual embeddings calculated from disease-specific cells.

The cutoff is 9 August 2023. Development outcomes span 10 August 2023 through 31 December 2024. The
final temporal holdout starts on 1 January 2025 and ends at the ClinicalTrials.gov snapshot used by
Stage 2A-WIDE.

## Endpoint and candidate universe

The primary label is the first public ClinicalTrials.gov Phase I-or-later interventional treatment
posting after the cutoff for an attributable target-indication pair, provided that no qualifying
same-indication clinical molecule alias was identified on or before the cutoff. The primary mapping
requires one direct human target. Ambiguous mappings are not silently converted into positives.

The prespecified panel covers 23 solid-tumour indications and 17,527 candidate genes. Stage 2A-WIDE
passed its power gate before model fitting. Stage 2B-WIDE excluded 2,343 target-indication entries
that were already clinical before the cutoff.

| Partition | Rows | Positives | Gene folds |
|---|---:|---:|---|
| Development | 321,222 | 42 | 1–4 |
| Final holdout | 79,542 | 14 | 0 |

Genes are disjoint between development and holdout. The holdout method scores and configuration
were frozen before the holdout labels were loaded.

## Historical predictors

Predictors were restricted to information available before the cutoff:

- publication counts queried with the cutoff;
- Open Targets release 23.06;
- pre-cutoff clinical status reconstructed from ClinicalTrials.gov and ChEMBL molecule aliases;
- manual transcriptomic features produced by the earlier complete-cell data layer;
- static Geneformer V1-10M or scGPT whole-human gene-token vectors;
- architecture- and dimension-matched random controls.

The historical feature release and model/checkpoint hashes are recorded in the seal and frozen
configuration. Large source data and prediction Parquet files are not duplicated in the release
ZIP; their hashes remain linked through the original manifests.

## Matched model protocol

Every method used the same SGD logistic head, gene-fold cross-validation, class weighting,
calibration procedure and alpha grid. PCA for the static or random representation blocks was fitted
on development genes only. Hyperparameters were selected using development folds, then the selected
methods generated blind holdout scores.

The confirmatory method was `geneformer_static_plus_cheap`; the comparator was
`cheap_historical`. The primary metric was pooled AUPRC. Secondary metrics included AUROC, MRR,
precision and recall at fixed ranks, excess over popularity, Brier score, log loss and ECE10.

Uncertainty was assessed with 2,000 paired bootstrap replicates at the target-gene level. Methods
were compared on the same resampled genes.

## Final result

| Method | Holdout AUPRC |
|---|---:|
| scGPT static + historical | 0.002513 |
| Historical features | 0.002307 |
| scGPT random + historical | 0.002137 |
| Manual expression + historical | 0.002061 |
| Geneformer static + historical | 0.001980 |
| Geneformer random + historical | 0.001790 |
| Popularity only | 0.000594 |

The prespecified Geneformer-static contrast was -0.0003276 AUPRC, with a paired 95% interval of
[-0.009485, 0.001392]. The confirmatory conclusion is `NOT_SUPPORTED`.

The highest observed AUPRC belonged to scGPT static + historical, but this was a secondary method.
Its improvement over popularity was not established by a confidence interval excluding zero, so it
must not be promoted to a positive confirmatory finding after inspecting the holdout.

The clinical OOD contrast defined melanoma as ID and all other indications as OOD. The melanoma
holdout subset contained zero positives, so `clinical_ood_delta` is not computable. The benchmark
correctly abstains rather than imputing a value.

## One-time holdout boundary

The final holdout was opened once on 4 August 2026 after Stage 3A froze the methods, hashes and
configuration. It may be used to reproduce or audit the v0.4.0 result. It may not be reused to select
contextual models or support a new confirmatory claim.

A future contextual-embedding study requires a new temporal snapshot, new outcomes and a newly
sealed holdout.

## Compact reference bundle

`results/temporal_benchmark_v0.4/` contains the manifests, seal, frozen configuration, compact metric
tables, bootstrap summary, conclusion and final report. Run:

```bash
fmva temporal-validate results/temporal_benchmark_v0.4
```

Expected status is `PASS_WITH_WARNINGS`. The warning preserves an original Stage 2B accounting
inconsistency: `at_risk_rows` is 400,778, whereas development plus holdout rows is 400,764. The
14-row difference does not alter the sealed development or holdout tables, whose counts and hashes
are mutually consistent through Stage 3B. The release does not rewrite the original manifest.

Use `--fail-on-warnings` in automated workflows that require a completely warning-free reference.
The command returns exit code 1 for failed invariants and exit code 2 for warnings under that flag.
