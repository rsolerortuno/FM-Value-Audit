# Melanoma-to-NSCLC response-transfer proxy

## Scope

This result is **not a clinical target-prioritisation benchmark**. It tests whether static
gene-token representations can smooth a melanoma post-treatment response-associated gene signal
so that it better agrees with an independent NSCLC post-treatment lesion-response contrast.

Claim scope: `RESPONSE_TRANSFER_PROXY_NOT_CLINICAL_TARGET_EVALUATION`.

## Outcome correction

The official GSE179994 supplement reports response per target lesion, not a single response per
patient. P1 and P13 contain both responsive and non-responsive lesions. The analysis therefore:

- maps 14 post-treatment biopsy samples to the official supplement;
- reconciles the P13 sample-number mismatch only after exact matches are exhausted;
- averages repeated biopsies of the same P1 lymph-node lesion;
- evaluates 8 responsive and 5 non-responsive target lesions across 11 patients;
- cluster-bootstraps patients, preserving all lesions from each sampled patient;
- excludes 25 patients without evaluable lesion-response labels from the outcome only.

## Data and protocol

- 17,999 genes are shared across GSE115978, GSE120575 and GSE179994.
- 17,450 genes are jointly mapped to Geneformer and scGPT vocabularies.
- Ridge heads use identical deterministic gene folds and alpha grids.
- Geneformer and scGPT use 32 PCA components from frozen static token embeddings.
- Matched random embeddings have the same shape and head budget.
- Confidence intervals use 1,000 patient-cluster bootstrap replicates.

## Results

| Method | Melanoma OOF Spearman | NSCLC OOD Spearman | Bootstrap median | 95% interval |
|---|---:|---:|---:|---:|
| Geneformer static | 0.582 | 0.146 | 0.124 | [0.000, 0.187] |
| Manual features | 0.511 | 0.094 | 0.080 | [-0.039, 0.161] |
| scGPT static | 0.643 | 0.071 | 0.062 | [-0.054, 0.132] |
| Direct melanoma delta | 1.000 | 0.062 | 0.053 | [-0.111, 0.158] |

### Paired differences

- Geneformer minus direct melanoma signal: median **+0.071**,
  95% interval **[+0.005, +0.140]**.
- Geneformer minus matched random embedding: median **+0.123**,
  95% interval **[+0.005, +0.185]**.
- Geneformer minus manual features: median **+0.043**,
  95% interval **[-0.027, +0.109]**.

The first two intervals exclude zero. The comparison with manual features does not.

## Sensitivity analyses

- Excluding mixed-response patients P1 and P13, Geneformer Spearman is
  **0.154** when both outcome strata remain.
- Across leave-one-patient-out analyses, Geneformer median Spearman is
  **0.144**.

## Interpretation

The result supports a narrow statement: **Geneformer static token geometry contains some
cross-disease structure relevant to this gene-level response-transfer proxy**. It does not establish
that Geneformer improves pharmaceutical target selection. The outcome is an expression contrast,
not future clinical entry; the embeddings are static rather than contextual; genes are not
independent statistical units; only 13 target lesions are labelled; and pretraining overlap remains
unresolved.

AUPRC, MRR, excess over popularity, calibration and clinical `ood_delta` remain `NOT_COMPUTED`.
