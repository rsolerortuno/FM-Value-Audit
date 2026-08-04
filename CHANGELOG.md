# Changelog

## 0.5.0a1 — 2026-08-04

### Portfolio release additions

- English portfolio README with a simple problem-to-conclusion narrative;
- four figures generated from versioned scientific outputs;
- machine-readable portfolio summary and model card;
- gated GitHub pre-release workflow and release body;
- explicit test, coverage and alpha-debt reporting in the README.

### Added

- preregistered post-2026-08-04 contextual clinical-entry protocol;
- deterministic, sample-balanced real-cell selection and pooling;
- Geneformer and scGPT token-state contextual exports with matched random controls;
- nine fixed biological context views and development-only PCA;
- label-only snapshot power gate and future-score freeze contracts;
- three resumable Colab notebooks and contextual CLI commands;
- prospective tamper-detection and integrity tests.

### Evidence boundary

- no post-cutoff labels have been opened;
- v0.4 outcomes are development evidence only;
- the alpha makes no prospective performance claim;
- the first eligible future snapshot is 2027-02-04 and must pass every frozen threshold.

## 0.4.0 — 2026-08-04

### Added

- real cutoff-correct solid-tumour temporal benchmark contracts from Stage 2A-WIDE through Stage 3B;
- compact hash-linked reference bundle with the frozen seal, configuration, metrics, bootstrap summary and final report;
- `temporal-validate` CLI command with machine-readable output and strict warning mode;
- cross-artifact validation of power gates, split counts, gene-disjoint sealing, label-access order, hashes, method sets, primary delta, bootstrap interval and OOD abstention;
- seven temporal integration and tamper-detection tests;
- temporal benchmark methods, runbook, limitations, decisions and release notes.

### Result

- final holdout: 79,542 target-indication rows, 3,476 genes and 14 positives;
- primary Geneformer-static-plus-historical minus historical AUPRC delta: -0.00032755;
- paired 95% gene-bootstrap interval: [-0.00948483, 0.00139163];
- confirmatory conclusion: `NOT_SUPPORTED`;
- clinical OOD delta: not computable because the melanoma subset had zero positives.

### Audit boundary

- the original Stage 2B `at_risk_rows` value exceeds development plus holdout rows by 14;
- downstream sealed counts and hashes are consistent, so the discrepancy is retained as a warning rather than rewritten;
- static Geneformer/scGPT token geometry is not contextual cell representation;
- the opened v0.4.0 holdout cannot support selection of new contextual methods.

## 0.3.1 — 2026-07-30

- Installed and executed the offline Ruff 0.16.0 and mypy 2.3.0 toolchain.
- Added pandas and PyYAML stubs and closed all strict typing errors in source and tests.
- Fixed notebook lint violations and formatted the complete repository.
- Corrected adapter protocol mutability, NumPy return typing, typed evaluation metadata, and OOD result serialisation without changing scientific outputs.
- Verified 36 tests with 87.53% branch-aware coverage.

## 0.3.0 — 2026-07-30

### Added

- complete-cell Stage-1 artefacts for GSE115978, GSE120575 and GSE179994;
- full sparse NSCLC conversion and sample pseudobulk;
- official lesion-level GSE179994 response mapping;
- preservation of discordant P1 and P13 lesion outcomes;
- `response-transfer-proxy` with matched ridge heads, random controls and static checkpoint embeddings;
- patient-cluster bootstrap, mixed-patient exclusion and leave-one-patient-out sensitivities;
- strict JSON outputs, reproducible report generation and lesion-mapping tests.

### Corrected

- removed the invalid assumption that one response label can be assigned per GSE179994 patient;
- P001 and P013 are now represented by their discordant target lesions;
- repeated biopsies of the same P001 lymph-node lesion are averaged before outcome calculation;
- P013.post.03 is reconciled to the unique unmatched expression sample P13.post.2 and flagged.

### Scientific boundary

- the OOD result is a response-expression transfer proxy, not clinical target prioritisation;
- no conclusion of superiority over manual features is permitted;
- AUPRC, MRR, excess over popularity, calibration and clinical OOD delta remain `NOT_COMPUTED`;
- pretraining accession overlap remains `UNRESOLVED`.

## 0.2.0 — 2026-07-29

### Added

- manifest-validated reconstruction of Drive-split files;
- real melanoma feature extraction for GSE115978 and GSE120575;
- integrity and metadata artefacts for GSE179994;
- frozen CPU encoder paths for Geneformer V1-10M and scGPT whole-human;
- shape-matched random controls and measured effort logs;
- `real-preprocess` and `representation-smoke` CLI commands;
- real-data claims, input manifests, tests and Colab preprocessing notebook.

### Scientific boundary

- no real clinical-target label table or historical popularity control is included;
- real target ranking, excess over popularity and OOD target performance remain `NOT_COMPUTED`;
- pretrained-model leakage remains `UNRESOLVED`.
