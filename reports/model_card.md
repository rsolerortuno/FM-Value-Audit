# Model card — FM Value Audit v0.5.0a1

## Intended use

Research benchmarking of whether single-cell foundation-model representations add incremental value to preclinical immuno-oncology target prioritisation.

## Excluded use

Clinical decision support, patient treatment selection, diagnosis, prognosis, dosing or claims that a target is safe, causal or therapeutically validated.

## Evaluated representation classes

- historical and popularity baselines;
- manual expression features;
- static Geneformer token embeddings;
- static scGPT token embeddings;
- shape-matched random controls;
- prospective contextual Geneformer cell states;
- prospective contextual scGPT cell states.

## Final evaluated result

The frozen v0.4 primary contrast compared `geneformer_static_plus_cheap` with `cheap_historical` on a gene-disjoint temporal holdout.

- point AUPRC difference: -0.00032755486429918244;
- 95% paired gene-bootstrap interval: [-0.009484827468089183, 0.0013916328916157368];
- conclusion: `NOT_SUPPORTED`.

## Prospective contextual status

The v0.5 contextual protocol is frozen but not yet evaluated. Geneformer and scGPT GPU notebooks are supplied, future labels remain unopened, and the first eligible scheduled snapshot is 2027-02-04.

## Validation design

- temporal information cutoffs;
- gene-disjoint holdout;
- development-only feature fitting;
- immutable score and label hashes;
- prespecified primary method and comparator;
- paired gene bootstrap;
- random encoder controls;
- label-only snapshot power gate;
- explicit abstention when metrics are not estimable.

## Important limitations

- rare final outcome: 14 positives;
- static token geometry is not contextual representation;
- no contextual GPU result in this pre-release;
- no future post-cutoff outcome result;
- unresolved pretraining accession overlap;
- supporting response-transfer analysis is not clinical target evaluation;

## Evidence sources

- `results/temporal_benchmark_v0.4/`;
- `results/ood_proxy/`;
- `results/stage1_full_cells/`;
- `protocols/FMVA_CONTEXTUAL_PROSPECTIVE_v0.5.json`;
- `reports/portfolio_summary.json`;
- `VALIDATION.md`.
