# Methods

## 1. Audit question and null hypothesis

`fm-value-audit` asks whether a representation method adds decision-relevant value over cheap,
pre-cutoff baselines when ranking human genes for future clinical target development in a bounded
immuno-oncology or immune-mediated disease context. The null hypothesis is that cheap methods are
sufficient. The software is an audit harness, not a target-discovery system.

The committed executable demonstration is synthetic. It validates the harness and does not support
claims about any real foundation model, target or disease.

## 2. Unit of analysis and landmark design

The unit is a gene. At a cutoff year `c`, every feature used by the temporal benchmark must represent
information available no later than `c`. A gene is a historical positive when its first clinical
entry year is at most `c`. The prospective evaluation positive is a gene whose first clinical entry
year lies in `(c, e]`, where `e` is the evaluation end year. Historical positives are excluded from
the prospective ranking universe. Genes without an observed entry by `e` are evaluation negatives,
with the explicit caveat that these are censored and incomplete negatives.

The supervised method is trained only on the historical landmark labels. Genes that enter after the
cutoff are therefore unlabeled negatives during training, as they would be prospectively. This is a
positive-unlabeled limitation, not a clean binary ground truth.

The final post-cutoff holdout is passed to evaluation exactly once per benchmark run after method
configuration is frozen. The result artefact records `holdout_access_count = 1`.

## 3. Internal tuning and matched effort

Every executable method receives three candidate configurations. Configurations are evaluated on a
stratified historical validation partition created from cutoff-time labels only. The final holdout is
not used for configuration selection. The effort log records:

- configuration identifiers tried;
- tuning wall-clock seconds;
- final-fit wall-clock seconds;
- total CPU seconds;
- GPU hours, which are zero for committed runs;
- random seed and library versions.

Equal configuration counts do not imply equal compute. Both counts and measured compute are
published. An unmatched-compute ablation deliberately gives the supervised method twelve candidate
configurations and is labelled as such.

## 4. Scores and baselines

All methods return a real-valued score where larger means higher priority.

1. **Popularity control:** `log1p(publication_count_pre_cutoff)`.
2. **Mean expression:** mean disease-context expression available before the cutoff.
3. **Differential expression:** disease-minus-control effect score.
4. **Cell-type specificity:** a bounded specificity score.
5. **Network centrality:** centrality from a supplied interaction network or a precomputed column.
6. **Druggable class:** membership of a supplied druggable, secreted or surface-protein class.
7. **Supervised baseline:** logistic regression over the preceding features, with standardisation.
8. **Random initialisation:** deterministic random-MLP embeddings followed by logistic regression.
9. **Random ranking:** seeded independent uniform scores.

For scalar baselines, the three matched configurations are identity, percentile rank and robust
z-score transformations. These are monotone for non-degenerate inputs and therefore mainly verify
that the tuning path is exercised symmetrically. Constant features return tied scores and are
reported rather than jittered into artificial signal.

## 5. Synthetic benchmark

The generator creates genes from three planted groups:

- `biology_signal`: future development is predictable from pre-cutoff expression, differential
  expression, specificity and druggability;
- `popularity_only`: study intensity is high but future target status is forced negative;
- `noise`: no useful planted relation.

All-time publication count receives a post-outcome increment for developed genes. It is prohibited
from the temporal benchmark and used only in the intentionally leaky random-split ablation. The OOD
context retains the same labels but weakens and perturbs the biological signals.

Harness validation requires that a biological method beats random ranking, that popularity-only
false positives are enriched near the top of the popularity ranking, and that excess over popularity
is positive for the planted supervised signal. These are validation statements about the simulation,
not biomedical claims.

## 6. Splits and leakage audit

### Temporal split

Features are cutoff-bounded. Tuning uses historical labels. Final evaluation ranks post-cutoff
entrants against genes with no observed entry by the evaluation end.

### Random split ablation

A stratified random split over final labels uses all-time publication count. It is intentionally
leaky and exists only to quantify how much a conventional random evaluation can inflate apparent
performance. It must never be presented as the primary result.

### Leakage status

For each model:

- `RESOLVED_CLEAN`: documented corpus identifiers were compared with evaluation identifiers and no
  overlap was found, or the method has no pretraining corpus;
- `RESOLVED_OVERLAP`: at least one documented identifier overlaps;
- `UNRESOLVED`: corpus documentation or identifiers are insufficient.

The overlap audit computes exact, case-normalised set intersections and records counts and members.
An unresolved model retains the flag in every model-specific result and claim.

## 7. Metrics

Let `y_i ∈ {0,1}` be the prospective label, `s_i` the score, and `π` the ordering of genes by
non-increasing score. Let `P = Σ_i y_i` and `I(condition)` be the indicator function.

### Precision at k

`precision_at_k = (1/k) Σ_{j=1..k} y_{π_j}` where `k` is truncated to the number of genes.

### Recall at k

`recall_at_k = (1/P) Σ_{j=1..k} y_{π_j}`. It is undefined if `P = 0`; the implementation raises a
validation error rather than returning a plausible number.

### Mean reciprocal rank

For this single ranked list, `MRR = 1 / min{j : y_{π_j}=1}`. This is reciprocal rank rather than a
query-averaged information-retrieval MRR; the naming is retained for the requested interface.

### Area under the precision-recall curve

AUPRC is average precision:

`AUPRC = Σ_n (R_n - R_{n-1}) P_n`,

using score thresholds in descending order. AUPRC is primary among the raw metrics because positives
are rare, but it is not prevalence invariant and must be interpreted with the positive fraction.

### Excess over popularity

For metric `M` and method `m`:

`excess_over_popularity(m) = M(m) - M(popularity)`.

The primary headline uses AUPRC. The confidence interval is computed directly on paired bootstrap
replicates of this difference, not by subtracting separate intervals.

### Bootstrap confidence intervals

Genes are sampled with replacement, preserving paired method and popularity scores. Metrics are
recomputed on each sample. Replicates with no positive genes are discarded. The reported interval is
the percentile interval at 2.5% and 97.5%. The committed synthetic run uses 400 replicates; smoke tests
use fewer. Gene-level bootstrap assumes genes are exchangeable and does not capture pathway or family
correlation.

### Calibration

For methods emitting probabilities, expected calibration error with `B` equal-width bins is:

`ECE = Σ_b (n_b/n) |mean(y_b) - mean(p_b)|`.

Scalar rankers do not claim calibrated probabilities and report calibration as not applicable.

### OOD delta

For metric `M`:

`ood_delta = M_OOD - M_ID`.

Negative values indicate degradation. The synthetic OOD context is a stress test, not evidence of
real tissue transfer.

### Compute-normalised value

For AUPRC excess `Δ`:

- `gain_per_cpu_hour = Δ / CPU_hours` when CPU time is positive;
- `gain_per_gpu_hour = Δ / GPU_hours` when GPU hours are positive, otherwise not applicable;
- `gain_per_configuration = Δ / configurations_tried`.

Tiny CPU timings are hardware- and scheduler-sensitive. They are recorded for transparency, not as a
portable speed leaderboard.

## 8. Abstention rule

If the paired 95% bootstrap interval for excess AUPRC includes zero, the comparison state is
`NO_DETECTABLE_DIFFERENCE`. If the lower bound is above zero it is `DETECTABLE_GAIN`; if the upper
bound is below zero it is `DETECTABLE_LOSS`. No total ordering is produced from overlapping or
non-significant differences.

## 9. Required ablations

1. Temporal versus intentionally leaky random split.
2. Supervised model with versus without popularity as a covariate.
3. Pretrained versus random initialisation (`NOT_COMPUTED` until a verified checkpoint is run).
4. Frozen versus fine-tuned (`NOT_COMPUTED` until both are run).
5. In-distribution versus synthetic OOD context.
6. Matched versus unmatched configuration budget.
7. Model scale using small and large local random controls; this is not a pretrained scaling claim.

## 10. Pre-registered predictions

Written before implementation or execution:

1. The popularity control will be strong, operationalised before the final run as AUPRC at least
   1.1 times the prospective positive fraction, with at least three executable methods failing to
   show a detectable gain over it.
2. Excess over popularity will be small, operationalised as absolute AUPRC excess below 0.20 for
   every executable method; the foundation-model component remains untested until a real run.
3. Random initialisation will recover a meaningful share of pretrained performance.
4. Temporal splits will lower supervised AUPRC by at least 0.05 relative to the intentionally
   leaky random split.

Predictions involving pretrained models remain `NOT_TESTED` in the committed synthetic-only run.

## v0.2.0 real-data preregistration boundary

The real-data feature extraction and checkpoint sanity run occur before any real clinical-target
label table is constructed. No hyperparameter, target head, cutoff date or primary real-data metric
was selected after seeing a real target holdout, because no such holdout has been created.

The following predictions remain untested on real target labels:

1. popularity will be difficult to beat under a temporal split;
2. excess over popularity will be small;
3. architecture-matched random initialisation will recover a nonzero share of pretrained target
   performance;
4. real temporal performance will be lower than random-split performance;
5. NSCLC OOD performance will be lower than melanoma ID performance.

Runtime, embedding variance and pretrained/random geometry correlation are diagnostics, not
substitutes for any of those preregistered outcomes.

## v0.3.0 lesion-corrected response-transfer proxy

This secondary analysis is explicitly outside the primary prospective target-ranking estimand.
It uses genes as analysis units and asks whether a fixed melanoma response-associated gene signal
transfers to an independent NSCLC post-treatment lesion-response contrast.

1. The GSE179994 outcome is read from Supplementary Table 1 at target-lesion level.
2. Normalised sample IDs are matched exactly before any fallback is allowed.
3. The unique residual P013 post-treatment expression sample is mapped to the clinical-table row
   `P013.post.03` and marked `FALLBACK_UNIQUE_UNMATCHED_PATIENT_TIMEPOINT`.
4. Repeated biopsies from the same patient and biopsy site are averaged into one target-lesion
   vector.
5. Discordant lesions in P1 and P13 remain separate response units.
6. NSCLC response contrast is mean post-treatment lesion log2-CPM in responsive lesions minus the
   corresponding mean in non-responsive lesions.
7. Ridge heads are selected using deterministic hash folds over genes in melanoma only.
8. Static checkpoint token embeddings, manual features and shape-matched random embeddings receive
   the same alpha grid and fold assignments.
9. Fixed predictions are evaluated in NSCLC with Spearman correlation.
10. Confidence intervals use patient-cluster bootstrap: sampled patients contribute all their
    lesions, preserving within-patient response discordance.

The proxy has no publication-popularity control and no temporal clinical-entry outcome. Its metric
must not be called clinical `ood_delta`.

## v0.4.0 cutoff-correct temporal benchmark

### Historical landmark

The protocol freezes 9 August 2023 as the information cutoff. Clinical entries dated 10 August 2023
through 31 December 2024 form development outcomes. Entries from 1 January 2025 through the snapshot
form the final holdout. Open Targets 23.06, pre-cutoff publication counts and pre-cutoff clinical
molecule aliases are the allowed historical predictors.

### Outcome construction

The primary outcome is a first public Phase I-or-later ClinicalTrials.gov entry for a
single-direct-human-target therapeutic intervention in a prespecified solid-tumour indication.
Pairs with qualifying pre-cutoff clinical entries are removed from the at-risk universe. Ambiguous,
complex or unattributable target mappings remain excluded from the primary endpoint.

### Split and freeze

A deterministic SHA-256 gene fold assigns fold 0 to holdout and folds 1–4 to development. No gene
occurs in both partitions. Stage 3A performs all feature processing, fold-level tuning, calibration
and model selection using development data only. It writes frozen holdout scores and a configuration
hash before Stage 3B reads the isolated holdout labels.

### Matched heads and controls

All methods use the same balanced SGD logistic head, alpha grid, gene-fold cross-validation and
cross-fitted Platt calibration. Static representation and random-control blocks use 32 PCA
components fitted on development genes only. Comparisons therefore differ in the supplied feature
block, not in head capacity or tuning budget.

### Confirmatory analysis

The frozen primary contrast is pooled AUPRC for `geneformer_static_plus_cheap` minus pooled AUPRC
for `cheap_historical`. Uncertainty uses 2,000 paired target-gene bootstrap replicates. A positive
claim requires the full 95% interval to exceed zero. The observed interval crosses zero, yielding
`NOT_SUPPORTED`.

### Clinical OOD abstention

Melanoma is the prespecified ID subset and non-melanoma indications are OOD. The OOD delta is only
reported when both subsets contain positive outcomes. The melanoma holdout subset has zero
positives, so the delta remains missing by design.

### Contract validation

`fmva temporal-validate` checks completion states, power-gate copies, gene-disjoint sealing, row and
positive counts, configuration and output hashes, holdout-access ordering, method sets, recomputed
primary delta, bootstrap interval, conclusion rule and OOD abstention. The validator operates on the
compact committed reference bundle without requiring the large Parquet artefacts.

## v0.5 contextual representation method

The v0.5 alpha retains final token/gene hidden states rather than static embedding-table rows. Cells
are selected deterministically within sample-context strata, encoded in batches, and accumulated by
gene. The first pooling level is within a sample-context. The second gives each represented sample
equal weight within its biological context. Nine preregistered context views are concatenated with
availability flags and reduced by a development-gene-only PCA. Pretrained and shape-matched random
controls use separate PCA fits but identical dimensionality and downstream head budgets.

The prospective endpoint, gene split, power gate, snapshot order and confirmatory contrast are
frozen in `protocols/FMVA_CONTEXTUAL_PROSPECTIVE_v0.5.json` before GPU execution and future label
access.
