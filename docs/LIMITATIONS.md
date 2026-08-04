# Limitations

This document was written before benchmark execution.

1. Known target labels are a biased and incomplete ground truth. Clinical entry reflects biology,
   tractability, commercial strategy, safety, competition and historical opportunity.
2. Absence from the label set does not mean a gene is a bad target. Evaluation negatives are
   censored non-observations, not confirmed biological negatives.
3. Pretraining corpora are incompletely documented. Leakage can be bounded where identifiers exist
   but cannot generally be excluded. Incomplete evidence is labelled `UNRESOLVED`, never clean.
4. Results are specific to the task, models, adapters, versions, data snapshots and split dates
   tested. They will age as checkpoints, corpora and clinical programmes change.
5. No wet-lab validation exists. Ranking performance is not proof of target efficacy or safety.
6. The committed benchmark uses simulated genes and simulated features. It validates implementation
   behaviour only and cannot answer whether any real foundation model has value.
7. The local random MLP is not identically shaped to unavailable published models. It is an honest
   executable substitute, not the definitive random-initialisation control requested for a real run.
8. The gene-level bootstrap assumes exchangeability and ignores pathway, paralogue and network
   dependence; intervals may be optimistic.
9. Multiple methods, metrics and ablations create multiplicity. The project pre-registers AUPRC
   excess over popularity as primary, but the synthetic demonstration does not perform family-wise
   error correction.
10. Calibration is reported only for methods producing probability-like outputs. Rank scores cannot
    be interpreted as probabilities without a separately validated calibration procedure.
11. Network centrality and druggable-class quality depend on external source definitions, versions
    and coverage. The synthetic run does not validate any public network or druggability database.
12. Runtime estimates for external models are planning ranges, not measured results, until the exact
    checkpoint, hardware, input size and adapter are executed.

## v0.2.0 real-data limitations

- Real expression features do not solve the label problem. Response-associated genes and therapeutic
  targets are not interchangeable constructs.
- GSE115978 does not provide the response endpoint required for the primary target benchmark; its
  post/naive comparison is a contextual expression feature.
- GSE120575 is TPM from Smart-seq2. It cannot be used as raw-count input for Geneformer or scGPT.
- GSE179994 raw counts were reconstructed but not interpreted locally because the sandbox lacks R/S4
  support. No OOD model result is committed.
- The model runs use 12 cells and sequence length 64. They validate execution only and do not measure
  accuracy, target ranking, scale behaviour or production throughput.
- scGPT was executed with a mathematically standard CPU attention path rather than the optional
  FlashAttention kernel. A full upstream-package parity run remains necessary before publication.
- The architecture-matched random controls have identical tensor shapes and inputs, but exact
  upstream training-time initialiser parity is not claimed where the checkpoint documentation is
  incomplete.
- Geneformer and scGPT leakage remain `UNRESOLVED`. The broad corpus descriptions do not identify
  every source accession, derivative dataset or preprocessing lineage.
- The committed real feature table has no cutoff-correct publication count, network snapshot,
  tractability snapshot or clinical-entry label. The primary benchmark remains `NOT_COMPUTED`.

## v0.3.0 full-cell and OOD-proxy limitations

- All three cohorts are now processed without cell subsampling, but complete cell coverage does not
  solve endpoint validity.
- GSE179994 response is lesion-level. Any patient-level responder/non-responder collapse is invalid
  because P1 and P13 contain discordant target lesions.
- Only 13 target lesions across 11 patients have evaluable post-treatment response labels. Patient-
  cluster bootstrap reflects this outcome uncertainty but cannot create independent patients.
- Static token embeddings test checkpoint gene geometry, not contextual embeddings derived from
  melanoma or NSCLC cells.
- The response-transfer outcome is a gene-expression contrast. It is not future clinical entry,
  target efficacy, safety, tractability or commercial value.
- Genes are correlated through pathways, families and cell programmes. Patient bootstrap addresses
  lesion sampling but not gene-level dependence.
- Geneformer exceeds the direct melanoma signal and its random control in this proxy, but its paired
  interval versus manual features crosses zero. Superiority over cheap baselines is not established.
- scGPT does not show a conclusive advantage over its random control or the direct signal.
- Pretraining accessions remain incompletely documented, so all pretrained results remain
  `UNRESOLVED` for leakage.
- AUPRC, MRR, excess over popularity, calibration and clinical OOD delta remain `NOT_COMPUTED`.

## v0.4.0 temporal benchmark limitations

- The final holdout contains only 14 positives. The benchmark is valid under the frozen gate but
  confidence intervals are necessarily wide and ranking metrics are sensitive to individual genes.
- Operational negatives mean that no qualifying first clinical entry was observed in the evaluation
  window. They do not prove that a target is biologically invalid or will never enter development.
- ClinicalTrials.gov posting dates and intervention-to-target mappings are imperfect proxies for
  internal programme initiation. Conservative mappings trade recall for attribution precision.
- The primary universe is a prespecified solid-tumour panel rather than all diseases or all oncology.
- Geneformer and scGPT inputs in v0.4.0 are static token embeddings. The result does not establish the
  value, or lack of value, of contextual cell-state representations.
- The melanoma holdout subset contains zero positives. Clinical `ood_delta` therefore abstains and
  no melanoma-versus-other-indications transfer claim is available.
- The final holdout was opened on 4 August 2026. New model selection against it would be post hoc; a
  contextual extension requires a new sealed holdout.
- Pretraining-accession overlap remains unresolved. Static token geometry reduces but does not erase
  the possibility that historical corpora encoded evaluation-related biological knowledge.
- The original Stage 2B manifest reports `at_risk_rows=400778`, while its sealed development and
  holdout counts sum to 400764. The 14-row accounting difference is preserved as a warning; the
  downstream hashes and evaluated counts are internally consistent.
- The compact release omits large snapshots, Parquet predictions and bootstrap replicates. It can
  verify their hash chain but cannot independently reconstruct those files without the external
  project tree.

## v0.5 alpha limitations

- The alpha contains no future clinical outcomes and therefore no prospective performance result.
- Context labels inherit the granularity and uncertainty of public cohort annotations.
- At most 32 cells per sample-context are encoded; this is a compute-controlled representation, not
  a complete-cell census.
- Contextual views with no represented rows are zero-filled with an explicit availability flag.
- Geneformer and scGPT use different vocabularies and widths; equal downstream budgets do not make
  the encoders biologically equivalent.
- Pretraining-accession overlap remains unresolved where public model documentation is incomplete.
- Clinical entry is an operational drug-development outcome, not proof of target validity or future
  approval.
