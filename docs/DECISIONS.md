# Decision log

| ID | Decision | Rationale | Rejected alternative |
|---|---|---|---|
| D01 | Commit only synthetic executed results. | The sandbox cannot access target databases, atlases or model hubs; fabricated real results are prohibited. | Inventing plausible real-data outputs or silently omitting execution. |
| D02 | Treat external checkpoint provenance as UNVERIFIED and disable it by default. | Exact checkpoint, version, licence, parameter count and corpus cannot be verified in this build. | Filling fields from memory. |
| D03 | Use a prospective landmark split with censored negatives. | It matches the decision setting better than random splitting and makes temporal leakage explicit. | Random split as the primary benchmark. |
| D04 | Permit random split only as an intentionally leaky ablation using all-time popularity. | It quantifies inflation hidden by common evaluations. | Presenting it as a valid alternative estimate. |
| D05 | Use paired gene bootstrap for excess intervals. | Direct intervals on differences answer the comparison question. | Comparing overlap of separate method confidence intervals. |
| D06 | Use three configurations per executable method for matched effort. | It is simple, auditable and small enough for CPU CI. | Giving complex methods a larger implicit search. |
| D07 | Record zero GPU hours and report GPU-normalised value as not applicable. | No GPU or pretrained model is used. | Dividing by zero or inventing equivalent GPU time. |
| D08 | Implement a local random MLP rather than claim an architecture-matched published-model control. | It keeps the harness executable while clearly exposing the unresolved substitution. | Calling a generic random projection an identical model control. |
| D09 | Use AUPRC excess over popularity as the primary quantitative comparison. | It handles imbalance better than accuracy and directly enforces the required control. | Raw AUPRC leaderboard as the headline. |
| D10 | Make scalar ranker calibration not applicable. | Arbitrary rank scores are not probabilities. | Min-max scaling and reporting misleading ECE. |
| D11 | Define MRR as reciprocal rank for one gene list. | The requested metric name normally averages queries, but this benchmark has one ranking per context. | Inventing pseudo-queries to make the name literal. |
| D12 | Execute model-scale ablation only for local random controls. | Two controlled sizes are available and real; no pretrained scaling inference is allowed. | Fabricating multiple foundation-model sizes. |
| D13 | Provide generic local embedding adapters rather than checkpoint-specific download code. | Hub access and verified model metadata are absent; user-provided embeddings preserve provenance boundaries. | Hard-coding unverified model APIs and checkpoint names. |
| D14 | Freeze prediction thresholds before the committed final seed. | Qualitative predictions need auditable pass/fail rules; thresholds are defined in METHODS before the final run. | Choosing thresholds after inspecting the committed holdout. |
| D15 | Train ID-to-OOD ablations only on ID features and apply the frozen model to OOD features. | Re-training on OOD would test a second in-distribution model rather than transfer. | Tuning separately in each context. |
| D16 | Fit random-MLP input normalisation on historical training indices. | Whole-dataset scaling would use holdout feature distribution. | Transductive scaling over all genes. |
| D17 | Use generic contributor attribution and omit an unverified repository URL. | The build prompt supplied neither authorship nor a publication URL. | Guessing provenance would violate the project's truthfulness rule. |
| D18 | Commit final synthetic seed 424242 without pre-inspection. | Prediction thresholds and implementation were frozen first. | Tuning after viewing the final holdout was rejected. |
| D19 | Record unavailable local Ruff/mypy execution as a verification failure rather than a pass. | The restricted package channel exposed neither executable. | Claiming that unexecuted checks passed. |
| D20 | Use a standard `pyproject.toml` with uv-compatible installation but no fabricated lockfile. | The restricted registry could not resolve the full development dependency set needed to generate a valid `uv.lock`. | Hand-writing a lockfile or omitting the dependency-resolution failure. |

## v0.2.0 real-data decisions

### Cohort-aware sample keys for GSE115978

**Decision:** aggregate by `Cohort|sample`, not `sample` alone.  
**Reason:** `Mel75` occurs in both the New and Tirosh cohorts. A name-only key would merge unrelated
cells into one replicate.  
**Rejected alternative:** reproduce the apparent 32-sample count by silently collapsing duplicate
labels.

### Freeze the joined gene universe to GSE115978

**Decision:** scan all GSE120575 rows but calculate joint features only for genes represented in the
GSE115978 count matrix.  
**Reason:** a comparison should not change its candidate universe by method or dataset. This leaves
23,686 candidate rows, 21,957 with GSE120575 features.  
**Rejected alternative:** outer-union all 55,737 TPM rows and give many genes missing raw-count
features.

### Preserve TPM semantics

**Decision:** retain GSE120575 as TPM and use it only for patient-state expression features.  
**Reason:** foundation-model tokenisers that require integer raw counts must not be applied to TPM as
if it were counts.  
**Rejected alternative:** round TPM values or call them pseudo-counts.

### Minimal checkpoint-faithful CPU encoders

**Decision:** implement only the frozen encoder paths required by the downloaded checkpoint keys,
using standard CPU attention for the scGPT Wqkv weights.  
**Reason:** the restricted environment could not install the full upstream packages or
FlashAttention, but all checkpoint tensors and architecture dimensions were available and could be
executed without inventing weights.  
**Rejected alternative:** call static embedding lookup a foundation-model run, or mark a mocked
adapter as executed.

### Scope the model result as a sanity check

**Decision:** execute 12 deterministic cells at sequence length 64 and label the result
`REPRESENTATION_SANITY_ONLY_NOT_TARGET_EVALUATION`.  
**Reason:** this proves the real checkpoints, tokenisation and architecture-matched random controls
run end to end while staying within CPU memory.  
**Rejected alternative:** present runtime, variance or geometry correlation as evidence of target
value.

### Defer local GSE179994 parsing

**Decision:** retain a byte-exact verified RDS and use the Colab R conversion path.  
**Reason:** no R runtime or compatible S4 RDS reader was available after multiple installation
attempts.  
**Rejected alternative:** fabricate dimensions from metadata, convert only the RAW.tar as though it
were equivalent, or drop the dataset without recording the block.

## v0.3.0 full-cell and lesion-outcome decisions

### Retain every cell

**Decision:** use all 7,186 GSE115978, 16,291 GSE120575 and 150,849 GSE179994 cells.  
**Reason:** streaming and sparse export fit within the available 16 GB Colab runtime.  
**Rejected alternative:** pre-emptive cell subsampling before testing the complete sparse route.

### Correct GSE179994 to target-lesion response

**Decision:** use Supplementary Table 1 lesion response and reject the generated patient-level
mapping.  
**Reason:** P1 and P13 have responsive and non-responsive lesions; one patient label destroys the
reported biological heterogeneity.  
**Rejected alternative:** select the first response, majority response or worst response per patient.

### Reconcile P013 conservatively

**Decision:** map `P013.post.03` to `P13.post.2` only after all exact normalised mappings are consumed
and one unmatched post-treatment sample remains for P13.  
**Reason:** the supplement and expression metadata use inconsistent numbering. The fallback is
unique, deterministic and recorded.  
**Rejected alternative:** silently edit the sample name or map by row order.

### Aggregate repeat biopsies, not discordant lesions

**Decision:** average P1.post.1 and P1.post.2 because both are biopsies of the same responsive
lymph-node lesion; preserve P1.post.3 as a distinct non-responsive lung lesion.  
**Reason:** the target lesion is the outcome unit.  
**Rejected alternative:** treat repeated biopsies as independent lesions or average every biopsy per
patient.

### Scope the proxy narrowly

**Decision:** name the command `response-transfer-proxy` and stamp outputs
`RESPONSE_TRANSFER_PROXY_NOT_CLINICAL_TARGET_EVALUATION`.  
**Reason:** expression transfer is informative but cannot substitute for temporal target-development
labels and a popularity control.  
**Rejected alternative:** report AUPRC or clinical OOD value from lesion expression contrasts.

## v0.4.0 temporal benchmark decisions

### Expand clinical outcomes while preserving biological OOD

**Decision:** use first target-indication clinical entry across a prespecified solid-tumour panel as
the temporal outcome, while retaining melanoma versus non-melanoma as a secondary biological OOD
partition.  
**Reason:** melanoma and NSCLC alone yielded too few post-cutoff positives for a valid holdout.  
**Rejected alternative:** lower the power threshold, search seeds for a favourable split or move
positive events manually.

### Exclude pre-cutoff clinical pairs

**Decision:** remove target-indication pairs with high-confidence clinical entry on or before the
cutoff.  
**Reason:** a pair already in clinical development is not at risk for first future entry.  
**Rejected alternative:** treat all absent post-cutoff events as negatives regardless of prior
status.

### Freeze a gene-disjoint holdout

**Decision:** assign fold 0 by deterministic gene hash to final evaluation and folds 1–4 to
development.  
**Reason:** the same gene across indications can otherwise leak between tuning and evaluation.  
**Rejected alternative:** row-random splitting or selecting a fold after counting performance.

### Preserve the negative confirmatory result

**Decision:** report Geneformer static + historical versus historical as `NOT_SUPPORTED`.  
**Reason:** the paired 95% AUPRC-delta interval crosses zero.  
**Rejected alternative:** promote scGPT because it had the largest observed secondary AUPRC after
opening the holdout.

### Abstain on clinical OOD delta

**Decision:** leave the melanoma-versus-non-melanoma OOD delta missing.  
**Reason:** the melanoma holdout subset has zero positive events.  
**Rejected alternative:** calculate a nominal difference from an undefined ID ranking metric.

### Preserve the Stage 2B row-count discrepancy

**Decision:** expose the 14-row aggregate accounting difference as `PASS_WITH_WARNINGS` and retain
the original manifests byte-for-byte.  
**Reason:** sealed partition counts, hashes and Stage 3 evaluations agree, but provenance artefacts
must not be retroactively rewritten.  
**Rejected alternative:** silently edit `at_risk_rows` or alter evaluated tables to force equality.

### Require a new holdout for contextual models

**Decision:** prohibit using the opened v0.4.0 holdout for a future contextual confirmatory claim.  
**Reason:** model choice informed by this holdout would invalidate its confirmatory role.  
**Rejected alternative:** append contextual methods to the frozen comparison after viewing results.

## 2026-08-04 — new prospective contextual protocol

The opened v0.4 holdout is moved to development evidence. A new protocol starts future outcomes on
2026-08-05 and uses the first chronological quarterly snapshot that passes a label-only power gate.
The decision prevents reuse of an observed holdout and prevents choosing a favourable snapshot from
model performance.

## 2026-08-04 — sample-balanced contextual pooling

The contextual run selects no more than 32 cells per sample-context and averages samples equally.
This was chosen before encoder execution to control Colab cost and prevent large samples from
implicitly receiving higher target-prioritisation weight.

## 2026-08-04 — GSE120575 excluded from contextual tokenisation

GSE120575 is TPM rather than raw counts. It remains a manual-expression comparator and is not passed
to Geneformer or scGPT as though TPM were count data.
