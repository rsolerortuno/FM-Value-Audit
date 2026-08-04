# fm-value-audit benchmark report

Run kind: **SYNTHETIC_EXECUTED**
Real foundation-model results: **NOT_COMPUTED**

## Primary synthetic results

| Method | AUPRC | Excess vs popularity | 95% CI | State | CPU s | Configs |
|---|---:|---:|---:|---|---:|---:|
| cell_type_specificity | 0.0852 | 0.0227 | [-0.0181, 0.0774] | NO_DETECTABLE_DIFFERENCE | 0.0013 | 3 |
| differential_expression | 0.0801 | 0.0176 | [-0.0041, 0.0496] | NO_DETECTABLE_DIFFERENCE | 0.0012 | 3 |
| druggable_class | 0.0806 | 0.0182 | [-0.0048, 0.0639] | NO_DETECTABLE_DIFFERENCE | 0.0012 | 3 |
| mean_expression | 0.0959 | 0.0335 | [0.0039, 0.0759] | DETECTABLE_GAIN | 0.0013 | 3 |
| network_centrality | 0.0760 | 0.0136 | [-0.0045, 0.0513] | NO_DETECTABLE_DIFFERENCE | 0.0013 | 3 |
| popularity | 0.0624 | 0.0000 | [0.0000, 0.0000] | NO_DETECTABLE_DIFFERENCE | 0.0015 | 3 |
| random_mlp_large | 0.1039 | 0.0415 | [0.0161, 0.0817] | DETECTABLE_GAIN | 0.1122 | 3 |
| random_mlp_small | 0.1119 | 0.0495 | [0.0168, 0.0971] | DETECTABLE_GAIN | 0.0133 | 3 |
| random_mlp_small_ood_transfer | 0.0515 | -0.0109 | [-0.0374, 0.0132] | NO_DETECTABLE_DIFFERENCE | 0.0479 | 3 |
| random_ranking | 0.0534 | -0.0090 | [-0.0375, 0.0273] | NO_DETECTABLE_DIFFERENCE | 0.0013 | 3 |
| supervised_ood_transfer | 0.0495 | -0.0129 | [-0.0384, 0.0070] | NO_DETECTABLE_DIFFERENCE | 0.0331 | 3 |
| supervised_unmatched_compute | 0.1110 | 0.0486 | [0.0147, 0.0874] | DETECTABLE_GAIN | 0.1122 | 12 |
| supervised_with_popularity | 0.1110 | 0.0486 | [0.0186, 0.0907] | DETECTABLE_GAIN | 0.0093 | 3 |
| supervised_without_popularity | 0.1107 | 0.0483 | [0.0178, 0.0843] | DETECTABLE_GAIN | 0.0090 | 3 |

## Holdout audit

- Holdout accesses: 1
- Evaluation genes: 1065
- Prospective positives: 55

## Ablations

- **frozen_vs_finetuned:** NOT_COMPUTED
- **id_vs_ood:** EXECUTED_SYNTHETIC
- **matched_vs_unmatched_compute:** EXECUTED
- **model_scale:** EXECUTED_RANDOM_CONTROL_ONLY
- **pretrained_vs_random_initialisation:** NOT_COMPUTED
- **temporal_vs_random:** EXECUTED_INTENTIONALLY_LEAKY
- **with_vs_without_popularity:** EXECUTED

## Interpretation boundary

These are executed synthetic results. They validate the harness but do not establish the value of any real foundation model, gene target or disease-specific ranking.
