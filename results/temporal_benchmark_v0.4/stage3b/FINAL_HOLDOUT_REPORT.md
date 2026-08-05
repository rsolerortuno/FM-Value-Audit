# FMVA final temporal holdout report

- Evaluation time: 2026-08-04T10:07:32.632317+00:00
- Holdout rows: 79,542
- Holdout positives: 14
- Holdout genes: 3,476
- Indications: 23
- Primary method: `geneformer_static_plus_cheap`
- Primary comparator: `cheap_historical`
- Primary AUPRC delta: -0.00032755
- Paired target-gene bootstrap 95% CI: [-0.00948483, 0.00139163]
- Primary conclusion: **NOT_SUPPORTED**

## Final metrics

| method                       |   rows |   positives |   prevalence |   pooled_auprc |   pooled_auroc |   brier_score |   log_loss |       ece10 |   mrr_macro_indication |   recall_at_10_macro_indication |   precision_at_10_macro_indication |   recall_at_50_macro_indication |   precision_at_50_macro_indication |   recall_at_100_macro_indication |   precision_at_100_macro_indication |   excess_over_popularity |
|:-----------------------------|-------:|------------:|-------------:|---------------:|---------------:|--------------:|-----------:|------------:|-----------------------:|--------------------------------:|-----------------------------------:|--------------------------------:|-----------------------------------:|---------------------------------:|------------------------------------:|-------------------------:|
| scgpt_static_plus_cheap      |  79542 |          14 |  0.000176008 |    0.00251303  |       0.808719 |   0.000175978 | 0.00170335 | 4.54686e-05 |              0.0380863 |                       0.0909091 |                         0.00909091 |                       0.227273  |                         0.00545455 |                        0.227273  |                         0.00272727  |               0.00191917 |
| cheap_historical             |  79542 |          14 |  0.000176008 |    0.00230714  |       0.829442 |   0.000175979 | 0.00170403 | 4.54357e-05 |              0.0304527 |                       0.0909091 |                         0.00909091 |                       0.227273  |                         0.00545455 |                        0.227273  |                         0.00272727  |               0.00171329 |
| scgpt_random_plus_cheap      |  79542 |          14 |  0.000176008 |    0.00213671  |       0.799792 |   0.000175978 | 0.0017034  | 4.55352e-05 |              0.0291767 |                       0.0909091 |                         0.00909091 |                       0.227273  |                         0.00545455 |                        0.227273  |                         0.00272727  |               0.00154285 |
| manual_expression_plus_cheap |  79542 |          14 |  0.000176008 |    0.00206127  |       0.766267 |   0.000175979 | 0.00170388 | 4.55688e-05 |              0.0286042 |                       0.0909091 |                         0.00909091 |                       0.227273  |                         0.00545455 |                        0.227273  |                         0.00272727  |               0.00146742 |
| geneformer_static_plus_cheap |  79542 |          14 |  0.000176008 |    0.00197959  |       0.868375 |   0.000175979 | 0.00170397 | 4.54444e-05 |              0.024413  |                       0.0909091 |                         0.00909091 |                       0.136364  |                         0.00363636 |                        0.227273  |                         0.00272727  |               0.00138573 |
| geneformer_random_plus_cheap |  79542 |          14 |  0.000176008 |    0.00179026  |       0.818731 |   0.000175979 | 0.00170408 | 4.5787e-05  |              0.0244851 |                       0.0909091 |                         0.00909091 |                       0.227273  |                         0.00545455 |                        0.318182  |                         0.00363636  |               0.0011964  |
| popularity_only              |  79542 |          14 |  0.000176008 |    0.000593855 |       0.685076 |   0.000175975 | 0.00169227 | 5.17012e-05 |              0.0080976 |                       0         |                         0          |                       0.0909091 |                         0.00181818 |                        0.0909091 |                         0.000909091 |               0          |

## Clinical OOD delta

| method                       |   id_melanoma_auprc |   ood_non_melanoma_auprc |   clinical_ood_delta |
|:-----------------------------|--------------------:|-------------------------:|---------------------:|
| cheap_historical             |                 nan |               0.00237919 |                  nan |
| geneformer_random_plus_cheap |                 nan |               0.00184023 |                  nan |
| geneformer_static_plus_cheap |                 nan |               0.00203527 |                  nan |
| manual_expression_plus_cheap |                 nan |               0.00212432 |                  nan |
| popularity_only              |                 nan |               0.00059431 |                  nan |
| scgpt_random_plus_cheap      |                 nan |               0.00220531 |                  nan |
| scgpt_static_plus_cheap      |                 nan |               0.00261229 |                  nan |

## Claim limits

- The outcome is retrospective first clinical entry after the temporal cutoff.
- The holdout contains only 14 positives, so confidence intervals are essential.
- Static token embeddings are not contextual cell representations.
- No method, hyperparameter, calibration rule, or split was changed after label access.
