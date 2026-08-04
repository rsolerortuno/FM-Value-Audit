# FM Value Audit v0.5.0a1

This GitHub pre-release turns FM Value Audit into a portfolio-ready, executable alpha for the contextual foundation-model benchmark.

It does **not** contain a new contextual efficacy claim.

## Highlights

- English README rewritten around the problem, method, results, validation, limitations, future plans and conclusion.
- Four portfolio figures generated directly from versioned results:
  - final temporal holdout AUPRC;
  - primary paired-bootstrap contrast;
  - melanoma-to-NSCLC response-transfer proxy;
  - complete-cell Stage-1 data scale.
- Machine-readable `reports/portfolio_summary.csv` and `reports/portfolio_summary.json`.
- GitHub pre-release workflow with automated validation logs, wheel, source distribution, Colab bundle and SHA-256 manifest.
- Frozen post-2026-08-04 contextual protocol.
- Resumable Geneformer and scGPT notebooks with matched random controls.
- Label-only future power gate and score-freeze contracts.

## Final evidence inherited from v0.4

| Quantity | Result |
|---|---:|
| Final holdout rows | 79,542 |
| Holdout genes | 3,476 |
| Positive entries | 14 |
| Geneformer static + historical AUPRC | 0.001980 |
| Historical baseline AUPRC | 0.002307 |
| Primary difference | -0.0003276 |
| 95% interval | [-0.009485, 0.001392] |
| Conclusion | `NOT_SUPPORTED` |

The result does not support incremental temporal value from static Geneformer token geometry.

## v0.5 prospective scope

- Contextual real-cell Geneformer and scGPT states.
- Same cells, tokenisation and pooling for pretrained and random controls.
- Nine fixed biological context views.
- Development-only PCA and model fitting.
- Scores frozen before future labels are opened.
- First eligible future snapshot: 2027-02-04.

## Software validation

- 70 tests passed.
- 86.78% branch-aware coverage; required threshold 85%.
- Ruff check passed.
- Ruff format passed.
- Python compilation passed.
- Wheel import and CLI smoke passed.
- Three notebooks validated structurally with no stored outputs.
- Strict mypy remains visible alpha debt and blocks beta promotion.

## Release classification

Publish this tag as a **pre-release** named `v0.5.0a1`.

The final `v0.5.0` should only be published after contextual outputs are integrated, scores are frozen prospectively and a future power-gated evaluation is completed.
