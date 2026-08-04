# Phase gates

Each gate records the adversarial review performed before continuing.

## Phase 0 — Design

- **Passed:** Methods, exact metric formulas, prospective split, paired-difference intervals, effort protocol, prediction thresholds, assumption ledger and pre-results limitations were written.
- **Failed or unresolved:** Real label-source dates and checkpoint corpora were unavailable.
- **Changed:** None of those unavailable claims were guessed; synthetic-only execution was selected.
- **Decision:** Real evidence remains `NOT_COMPUTED`, and model leakage remains `UNRESOLVED` unless identifiers are complete.

## Phase 1 — Skeleton

- **Passed:** `src/` packaging, Typer CLI, registry schema, Apache-2.0 licence and CPU-only CI scaffold.
- **Failed or unresolved:** The sandbox package index could not satisfy isolated build/dev-tool downloads.
- **Changed:** Editable installation used the already installed build backend without build isolation. CI retains normal public-registry installation.
- **Decision:** The environment-specific fallback is reported rather than encoded as the normal user path.

## Phase 2 — Synthetic benchmark

- **Passed:** Biology-signal, popularity-only and noise genes were generated; tests show biological signal beats random and popularity produces planted false positives.
- **Failed or unresolved:** Synthetic structure cannot validate biomedical realism.
- **Changed:** The README labels all executed results synthetic.
- **Decision:** Harness validation is separated from model-value claims.

## Phase 3 — Splits and leakage

- **Passed:** Prospective landmark split, intentionally leaky random ablation, exact normalised overlap and three leakage statuses.
- **Failed or unresolved:** Exact overlap cannot detect semantic derivatives or undocumented corpora.
- **Changed:** Incomplete external entries are forced to `UNRESOLVED`.
- **Decision:** Random split is prohibited as a primary estimate.

## Phase 4 — Baselines

- **Passed:** Popularity, expression, differential expression, specificity, centrality, druggability, supervised, random MLP and random ranking all execute.
- **Failed or unresolved:** Equal configuration count does not equal equal intellectual effort.
- **Changed:** Counts and measured compute are both published; an unmatched-compute ablation is explicit.
- **Decision:** Scalar methods still traverse the same three-configuration tuning harness.

## Phase 5 — Model adapters

- **Passed:** Common adapter protocol, executable random MLPs, external NPZ adapter and unavailable-model mock.
- **Failed or unresolved:** No verified pretrained checkpoint or architecture-matched random control could be downloaded.
- **Changed:** External results are `NOT_COMPUTED`; local random MLP is labelled a substitute.
- **Decision:** No fabricated checkpoint metadata or result is allowed.

## Phase 6 — Evaluation

- **Passed:** Ranking metrics, calibration where applicable, paired bootstrap differences, abstention states, compute normalisation and `claims.json`.
- **Failed or unresolved:** Gene bootstrap can understate dependence.
- **Changed:** The limitation is explicit and the primary comparison is pre-registered.
- **Decision:** No forced total ranking is emitted.

## Phase 7 — Ablations

- **Passed:** Temporal/random, popularity covariate, synthetic ID/OOD, matched/unmatched effort and random-control scale were executed.
- **Failed or unresolved:** Pretrained/random and frozen/fine-tuned were impossible.
- **Changed:** OOD was corrected from re-training in OOD to frozen ID-to-OOD transfer; random-MLP normalisation now uses historical training data only.
- **Decision:** Impossible ablations remain visible as `NOT_COMPUTED`.

## Phase 8 — Real-run scaffold

- **Passed:** Registry validation, leakage command, generic embedding path, subset route and runbook resource ranges.
- **Failed or unresolved:** Runtime and VRAM cannot be exact without a verified checkpoint and hardware.
- **Changed:** Values are marked `ESTIMATE`; real artefacts must replace them with measurements.
- **Decision:** No external model is enabled by default.

## Phase 9 — Docs and packaging

- **Passed:** README claim markers, reports, executed artefacts, implementation report, critical evaluation and archive assembly.
- **Failed or unresolved:** Local Ruff and mypy executables were unavailable from the restricted package channel; this is recorded in the verification artefact rather than misreported as a pass.
- **Changed:** Python compilation, full tests, coverage, CLI smoke and manifest validation were run locally; CI is configured to run the missing checks with normal package access.
- **Decision:** The reproducibility grade is capped by the unexecuted local static checks and absent real-model run.

## v0.2 extension — real data and checkpoints

- **Passed:** three large files reconstructed from validated parts; two melanoma datasets processed;
  two real checkpoints and shape-matched random controls executed; compact artefacts committed.
- **Failed or unresolved:** GSE179994 RDS parsing, real temporal target labels, historical popularity,
  real target evaluation and pretraining leakage resolution.
- **Changed:** the repository status is no longer synthetic-only, but checkpoint execution is scoped
  to `REPRESENTATION_SANITY_ONLY_NOT_TARGET_EVALUATION`.
- **Adversarial finding:** a model can now be “executed” while the decision-relevant scientific
  question remains entirely unanswered. Availability and scientific result status are therefore
  separate fields.
- **Decision:** no real AUPRC, excess-over-popularity or OOD claim appears until the temporal target
  layer is frozen and touched once.
