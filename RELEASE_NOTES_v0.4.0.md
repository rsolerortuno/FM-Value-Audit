# fm-value-audit v0.4.0

## Headline

v0.4.0 closes the first real temporal target-entry benchmark. The prespecified Geneformer static
representation did not add demonstrated value over inexpensive historical features.

## Final evidence

- Historical cutoff: 9 August 2023
- Candidate genes: 17,527
- Solid-tumour indications: 23
- Development: 321,222 rows, 42 positives
- Final holdout: 79,542 rows, 3,476 genes, 14 positives
- Bootstrap: 2,000 paired target-gene replicates
- Primary AUPRC delta: -0.00032755
- 95% interval: [-0.00948483, 0.00139163]
- Conclusion: `NOT_SUPPORTED`

scGPT static plus historical had the largest observed AUPRC (0.002513), but it was secondary and is
not promoted to a confirmatory success. Clinical OOD delta abstained because melanoma had zero
holdout positives.

## Software changes

- Added `fmva temporal-validate`.
- Added compact Stage 2A-WIDE through Stage 3B reference contracts.
- Added hash, seal, freeze, conclusion and abstention validation.
- Added tamper-detection regression tests.
- Updated README, methods, limitations, decisions, runbook and critical evaluation.

## Known warning

The original Stage 2B manifest reports 400,778 at-risk rows; its development and holdout rows sum to
400,764. The 14-row difference is retained as an audit warning. Sealed partition counts, score and
label hashes, and final evaluation counts are mutually consistent.

## Scientific boundary

The tested model features are static token embeddings, not contextual cell representations. The
v0.4.0 holdout was opened once and must not be used to select new methods. A contextual v0.5.0 study
requires a new temporal snapshot and a new sealed holdout.
