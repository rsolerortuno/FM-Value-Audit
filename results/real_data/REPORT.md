# Real-data execution report

**Run status:** feature extraction and representation sanity checks executed; real temporal target
benchmark `NOT_COMPUTED`.

## Data

- GSE115978: 23,686 genes, 7,186 cells, 33 cohort-aware sample units.
- GSE120575: 16,291 cells, 48 patient states, 37 patients; 21,957 genes overlap the frozen
  GSE115978 candidate universe.
- Combined feature table: 23,686 genes; 20,669 map to the supplied Geneformer Ensembl dictionary.
- GSE179994: 150,849 metadata rows, 36 patients and 47 samples; raw RDS reconstructed and verified,
  but matrix interpretation is `NOT_COMPUTED_LOCAL`.

## Models

The same deterministic 12-cell GSE115978 subset and sequence length 64 were used for pretrained and
shape-matched random encoders.

| Model | Status | Encoder parameters | Coverage | Pretrained wall s | Random wall s | Peak RSS MB |
|---|---|---:|---:|---:|---:|---:|
| Geneformer V1-10M | `EXECUTED_CPU_SUBSET` | 10,199,040 | 0.7933 | 0.1326 | 0.0319 | 416.3 |
| scGPT whole-human | `EXECUTED_CPU_SUBSET` | 50,278,400 | 0.8784 | 0.2564 | 0.1177 | 753.2 |

These values establish only that the real checkpoints, tokenisers and random controls execute. They
do not measure target-prioritisation value.

## Claims

Permitted and unsupported claims are machine-readable in `claims.json`. In particular, no claim is
permitted that either pretrained model beats its random control, the cheap baselines or the
popularity control.
