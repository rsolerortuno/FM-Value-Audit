# Critical evaluation — v0.4.0

## Score: 8.3/10

The repository now answers its primary business-facing question with a real temporal endpoint,
historical controls and a sealed gene-disjoint holdout. This is a major advance over the v0.3.0
response-transfer proxy. The negative confirmatory result is preserved without post-hoc promotion of
a secondary method.

The score remains below decision-grade because the final holdout has only 14 positives, the mappings
are operational proxies for programme entry, contextual embeddings were not evaluated and
pretraining overlap remains unresolved.

## What is strong

- **Cutoff discipline:** predictors use a frozen 9 August 2023 landmark and archived historical
  sources.
- **Endpoint relevance:** the label is future first clinical target-indication entry rather than an
  expression-response proxy.
- **At-risk construction:** pre-cutoff clinical pairs are excluded before negatives are assigned.
- **Leakage control:** development and holdout are separated by gene, not by target-indication row.
- **Matched effort:** all methods use the same head, folds, alpha grid, calibration and PCA budget.
- **One-time evaluation:** scores and configuration were frozen before holdout-label access.
- **Uncertainty:** 2,000 paired gene-bootstrap replicates support the primary interval.
- **Abstention:** clinical OOD delta remains missing when the melanoma subset has zero positives.
- **Auditability:** manifests, seals, hashes, metrics and conclusions are cross-validated by the CLI.
- **Result honesty:** the prespecified Geneformer contrast is reported as `NOT_SUPPORTED`.

## What the result says

Static Geneformer token geometry did not demonstrate additional future clinical-entry ranking value
over historical features. The point estimate was slightly negative and the confidence interval
crossed zero.

scGPT static plus historical produced the highest observed AUPRC, but it was a secondary method and
its uncertainty does not justify a new confirmatory claim. The result supports continued evaluation,
not model selection.

## Remaining weaknesses

1. **Fourteen holdout positives:** estimates are noisy and individual target genes can materially
   affect the result.
2. **Operational labels:** absence of a public clinical entry is not biological failure.
3. **Mapping recall:** conservative single-target mappings omit complex, combination and ambiguous
   programmes.
4. **Static embeddings:** contextual disease- and cell-state representations remain untested.
5. **Opened holdout:** it cannot be reused for selection of a contextual extension.
6. **Zero melanoma positives:** the prespecified clinical OOD delta is unavailable.
7. **Pretraining overlap:** accession-level contamination cannot yet be excluded.
8. **Accounting warning:** Stage 2B's aggregate at-risk count differs by 14 from its sealed split
   counts, although downstream hashes and evaluated counts agree.
9. **External reconstruction:** the compact release verifies hash chains but omits large source and
   prediction files.

## Decision relevance

The current evidence argues against paying a complexity premium for static Geneformer geometry in
this target-entry task. It does not establish that foundation models are generally unhelpful, and it
does not justify selecting scGPT based on the observed secondary ranking.

The next scientifically valid release should build contextual cell-state features against a new
outcome snapshot and a new sealed holdout. Reusing the v0.4.0 labels for model selection would lower,
not raise, the evidentiary standard.
