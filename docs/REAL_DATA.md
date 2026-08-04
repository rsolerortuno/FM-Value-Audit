# Real-data execution record

## Status vocabulary

- `EXECUTED_FULL_FEATURE_LAYER`: the complete supplied expression file was scanned and compact
  sample-level features were committed.
- `EXECUTED_CPU_SUBSET`: the real checkpoint executed on a deterministic reduced cell/sequence set.
- `NOT_COMPUTED_LOCAL`: bytes and metadata were validated, but the object could not be interpreted in
  this sandbox.
- `NOT_COMPUTED`: the scientific quantity was not calculated and no proxy is substituted.

## Input integrity

`results/real_data/input_manifest.json` records SHA-256, size and Drive file ID for every primary
matrix and checkpoint. Three files exceeded the connector's 100 MB transfer limit and were supplied
as byte ranges. `results/real_data/split_reassembly_audit.json` verifies every part hash, final size
and final MD5 for:

- GSE179994 raw-count RDS;
- GSE120575 TPM matrix;
- scGPT whole-human checkpoint.

The splitter changes transport only. It does not decompress, transform or reinterpret the bytes.

## GSE115978

The raw-count CSV contains 23,686 genes by 7,186 cells. Annotation order was checked against every
matrix column. `sample_key = cohort + "|" + sample` is mandatory because `Mel75` appears under two
cohorts; merging by the visible sample string alone would create a false biological replicate.

For every gene, the pipeline computes:

- mean raw count per cell;
- fraction of cells with a non-zero count;
- post-treatment minus treatment-naive mean log-CPM at the sample level;
- maximum fraction of mean expression attributable to one annotated cell type;
- the dominant cell type under that definition.

The treatment contrast uses 33 sample-cohort units, not 7,186 cells. Cell-level p-values are not
calculated.

## GSE120575

The file is processed as TPM and is never relabelled as raw counts. It contains two nonstandard
features:

1. row two is a patient-state annotation vector embedded beneath the cell-ID header;
2. every gene row contains one trailing empty field.

The parser skips row two and ignores only the verified empty trailing field. Metadata extraction
retains rows named `Sample 1` through `Sample 16291` and excludes the remaining GEO template
sections.

The complete file has 55,737 gene rows. The committed joint feature universe is frozen to the
23,686 genes in GSE115978; 21,957 of those are present in GSE120575. This decision avoids changing
the candidate universe between the two melanoma studies.

For each retained gene, the pipeline computes patient-state mean TPM and responder versus
non-responder log2 differences separately at baseline and post-treatment. There are 48 patient
states: 19 baseline and 29 post-treatment. These are expression features, not clinical target
labels.

## GSE179994

The five transport parts reconstruct to a 441,470,188-byte gzip file with SHA-256
`8275907e6a341048c89d30cb6485dd81125f8922c952d0cd648cf6a3acaa1cc0` and MD5
`a0abd966fec8109c03605c1873e3627f`. Gzip integrity passes.

The metadata contain 150,849 cells, 36 patients and 47 samples. The full R/S4 sparse matrix was
converted in a 16 GB Colab runtime without cell subsampling. The retained outputs include a
19,790 × 150,849 sparse H5AD, a byte-level CSC bundle with 205,034,916 non-zero values and a
19,790 × 47 sample pseudobulk count matrix. Large matrices remain in Drive; compact metadata,
features and hashes are committed.

Clinical response is defined by the official supplement per target lesion, not per patient. The
valid outcome contains 8 responsive and 5 non-responsive lesions across 11 patients. P1 and P13
have discordant lesions. The original generated patient mapping is marked invalid and is not used.

## Geneformer V1-10M subset

The checkpoint contains 10,290,770 tensors/parameters in total; 10,199,040 belong to the encoder path
executed here. The tokenizer uses raw counts, cell-total normalisation, Genecorpus-30M gene medians
and rank ordering. V1 has no CLS/EOS tokens, so the committed cell vector is the mean final hidden
state across non-padding tokens.

The CPU subset contains 12 deterministic GSE115978 cells and a maximum of 64 tokens. A control with
the same encoder shapes and input was initialised from seed 20260729. The checkpoint result is
`UNRESOLVED` for leakage and is not passed to a target-ranking head.

## scGPT whole-human subset

The checkpoint contains 51,330,049 parameters; 50,278,400 are used by the gene encoder, continuous
value encoder and 12 transformer layers. Counts are normalised to 10,000, log1p-transformed, binned
within cell and supplied with a leading CLS token. The CPU path uses standard scaled dot-product
attention with the checkpoint's Wqkv/out-projection weights instead of the optional FlashAttention
kernel; in evaluation mode this preserves the encoder operation while avoiding a GPU-only dependency.

The same 12 cells and length 64 are used for the pretrained and shape-matched random runs. The
checkpoint licence field records that the code is MIT while separate checkpoint terms were not
located; default execution remains disabled.

## What the real run does not establish

The real run does not contain cutoff-correct target-development labels, historical publication
counts, a frozen target head or a prospective target holdout. It therefore cannot estimate AUPRC,
excess over popularity, compute-normalised target value or OOD target transfer for either model.
Those quantities remain `NOT_COMPUTED`.


## v0.3.0 response-transfer artefacts

`results/ood_proxy/nsclc_official_lesion_sample_mapping.csv` records every supplement-to-expression
sample mapping and its status. `nsclc_labeled_lesion_summary.csv` records the 13 lesion units after
repeat-biopsy aggregation. The fixed proxy predictions, patient-cluster bootstrap, paired
differences, mixed-patient sensitivity and leave-one-patient-out results are committed under
`results/ood_proxy/`. All carry the non-clinical claim scope.
