# Runbook

## CPU-only synthetic audit

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
fmva validate-manifest registry/models.yaml
fmva simulate --output synthetic.csv --genes 1500 --seed 20260729
fmva benchmark --data synthetic.csv --output results/local --bootstrap 400 --seed 20260729
fmva report --results results/local/benchmark_summary.json --output results/local/REPORT.md
```

Use `--subset 300` on `benchmark` for a quick path. The subset is deterministic and preserves all
available positives before sampling negatives.

## External foundation-model path

The repository deliberately does not name a usable checkpoint until its provenance is verified.
Before a real run:

1. Replace every `UNVERIFIED` and `NOT_CONFIGURED` field for the selected registry entry.
2. Confirm the checkpoint licence permits the intended use.
3. Export one embedding row per gene to an `.npz` file containing arrays `gene_ids` (strings) and
   `embeddings` (shape `n_genes × d`). Preserve the exact model and preprocessing metadata in JSON.
4. Run the leakage audit against documented pretraining dataset identifiers.
5. Execute the generic adapter:

```bash
fmva embed \
  --data canonical_real_input.csv \
  --adapter external-npz \
  --embedding-path /absolute/path/model_gene_embeddings.npz \
  --model-id VERIFIED_MODEL_ID \
  --output model_scores.csv \
  --subset 2000

fmva leakage \
  --manifest registry/models.yaml \
  --model-id VERIFIED_MODEL_ID \
  --evaluation-ids evaluation_dataset_ids.txt \
  --output leakage_audit.json

fmva evaluate \
  --data canonical_real_input.csv \
  --scores model_scores.csv \
  --model-id VERIFIED_MODEL_ID \
  --manifest registry/models.yaml \
  --effort-json model_scores.effort.json \
  --output results/real/VERIFIED_MODEL_ID \
  --bootstrap 2000
```

The `--subset` path selects all positives available in the chosen split and a deterministic sample of
negatives. It is for adapter validation only and is not the final benchmark.

## Resource planning for external runs

These are deliberately broad planning estimates, not measured claims:

| Workload | Expected wall time | VRAM planning range | Status |
|---|---:|---:|---|
| Load already-exported gene embeddings and fit the audit head, 2,000 genes | 1–10 minutes CPU | 0 GB | ESTIMATE |
| Frozen embedding export from a locally available small/medium model on a subset | 0.5–6 hours | 8–24 GB | ESTIMATE |
| Frozen embedding export on a full atlas | 4–48 hours | 16–48 GB | ESTIMATE |
| Fine-tuning | 8–72+ hours | 24–80+ GB | ESTIMATE |

Exact runtime and VRAM cannot be stated honestly before the checkpoint, tokenisation, sequence length,
cell count, precision mode and hardware are verified. The result artefact must replace estimates with
measured values.

## Clean-check commands

```bash
ruff check .
ruff format --check .
mypy --strict src
pytest --cov=fmva --cov-report=term-missing --cov-fail-under=85
fmva validate-manifest registry/models.yaml
python -m build
```

## v0.2.0 local real-data commands

Install the optional model dependencies:

```bash
python -m pip install -e '.[dev,models]'
```

The `real-preprocess` command performs no target-label construction and can run on CPU. The
`representation-smoke` command defaults to 12 cells and length 64; it records the subset and both
pretrained/random embeddings.

For GSE179994, run `notebooks/FMVA_real_data_stage1_preprocess.ipynb` in Colab. Its R cell loads the
RDS with `readRDS`, converts an S4 sparse `Matrix` to Matrix Market components and writes compact
metadata/checksums. Do not replace this step with metadata-derived matrix dimensions.

A full-length checkpoint export is not estimated from the subset timing. Sequence length changes
attention cost nonlinearly, and full atlas runtime must be measured in the actual upstream package.

## v0.3.0 lesion-corrected response-transfer command

Set BLAS libraries to one thread for predictable CPU execution:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Then run:

```bash
fmva response-transfer-proxy \
  --gse115978-features results/stage1_full_cells/GSE115978_gene_features.csv.gz \
  --gse120575-features results/stage1_full_cells/GSE120575_gene_features.csv.gz \
  --gse179994-pseudobulk results/stage1_full_cells/GSE179994_sample_pseudobulk_counts.npz \
  --gse179994-gene-order results/stage1_full_cells/GSE179994_gene_order_ALL.csv \
  --gse179994-sample-metadata results/stage1_full_cells/GSE179994_sample_metadata.csv \
  --gse179994-supplement /data/GSE179994_supplementary_tables.xlsx \
  --geneformer-model /models/Geneformer-V1-10M/model.safetensors \
  --geneformer-symbol-map /models/gene_name_id_dict_gc30M.pkl \
  --geneformer-tokens /models/token_dictionary_gc30M.pkl \
  --scgpt-model /models/scGPT-whole-human/best_model.pt \
  --scgpt-vocab /models/scGPT-whole-human/vocab.json \
  --output results/ood_proxy \
  --bootstrap 1000 \
  --seed 1729
```

The supplement is required because the generated Stage-1 patient response field is invalid for
outcome use. Verify that the output mapping contains P1 and P13 in both lesion-response strata before
accepting the run.

## v0.4.0 temporal benchmark audit

Validate the compact committed contracts:

```bash
fmva temporal-validate results/temporal_benchmark_v0.4
```

Write the machine-readable report:

```bash
fmva temporal-validate results/temporal_benchmark_v0.4 \
  --output results/temporal_benchmark_v0.4/local_audit.json
```

The released reference returns `PASS_WITH_WARNINGS` because the original Stage 2B aggregate row
count differs by 14 from the sealed split counts. Treat scientific invariant failures as fatal:

```bash
fmva temporal-validate results/temporal_benchmark_v0.4 --fail-on-warnings
```

That command returns code 2 for the known warning. A hash, seal, label-access or conclusion mismatch
returns code 1.

To audit the full Drive tree, point the same command at a local or mounted copy of
`temporal_benchmark/`. Do not include both obsolete narrow and current wide files under the same
search root, because duplicate contract basenames are rejected to prevent accidental mixing.

The v0.4.0 holdout is evaluation-only and already opened. Do not rerun model selection or introduce
new methods against its labels. Build contextual Stage 3 models only after creating a new temporal
outcome snapshot and seal.

## v0.5.0a1 contextual prospective workflow

Place the alpha wheel and notebooks in the mounted project notebook directory:

```text
projects_bioinfo_2026/fm-value-audit-real/
└── temporal_benchmark/
    └── 08_notebooks/
```

Run in this order:

1. `FMVA_v05_00_preflight_protocol.ipynb` — CPU, verifies raw matrices, lesion mapping, checkpoints,
   vocabularies and the frozen protocol.
2. `FMVA_v05_01_geneformer_contextual.ipynb` — GPU, resumable sample-context checkpoints.
3. `FMVA_v05_02_scgpt_contextual.ipynb` — GPU, resumable sample-context checkpoints.

Do not change the seed, cell cap, minimum stratum size, sequence length or context fields. Restarting
a notebook is safe: complete sample-context checkpoints are reused after hash validation.

Local protocol and artifact checks:

```bash
fmva contextual-protocol-init protocol.json
fmva contextual-validate /path/to/geneformer_contextual --protocol protocol.json
fmva contextual-validate /path/to/scgpt_contextual --protocol protocol.json
```

A future label snapshot is checked without scores:

```bash
fmva contextual-snapshot-gate future_labels.parquet \
  --protocol protocol.json \
  --snapshot-date 2027-02-04 \
  --output gate.json
```

Exit code 3 means the preregistered gate has not yet passed and the holdout must stay closed.

Validate a score freeze before any future label access:

```bash
fmva contextual-score-freeze-validate /path/to/future_score_freeze \
  --protocol protocol.json
```
