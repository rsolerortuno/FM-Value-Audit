# FM Value Audit v0.5.0a1

A portfolio-ready GitHub pre-release for auditing the decision value of single-cell foundation-model representations in immuno-oncology target prioritisation.

## What this pre-release contains

- the final v0.4 cutoff-correct temporal result;
- a frozen v0.5 contextual prospective protocol;
- real-cell Geneformer and scGPT Colab notebooks;
- matched random controls;
- label-only future power gate and score-freeze contracts;
- 70 passing tests and 86.78% branch-aware coverage;
- portfolio figures, model card, machine-readable summary and release manifest.

## Main scientific result

Static Geneformer token geometry did not improve the prespecified AUPRC contrast over the historical baseline:

```text
AUPRC difference: -0.0003276
95% CI: [-0.009485, 0.001392]
Conclusion: NOT_SUPPORTED
```

## Important boundary

This is an alpha pre-release. Contextual Geneformer/scGPT execution and future post-cutoff evaluation are not complete, and no contextual efficacy claim is made.
