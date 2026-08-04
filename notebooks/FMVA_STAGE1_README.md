# FMVA real-data Stage 1

Open `FMVA_real_data_stage1_preprocess.ipynb` in Google Colab and use **Runtime → Run all**.

Recommended runtime: high-RAM CPU. A GPU is not required for Stage 1.

The notebook reads the original files directly from `MyDrive/fm-value-audit-real`, writes all derived files under `outputs/stage1`, and does not execute foundation models or inspect the final temporal holdout.

Important safeguards:

- GSE120575 is retained as TPM and never relabeled as raw counts.
- GSE115978 has no inferred response label.
- GSE179994 response is obtained only from the official paper supplement.
- CELLxGENE controls use the versioned `stable` Census release and deterministic donor-capped sampling.
- Every input is SHA-256 hashed.
