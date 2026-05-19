# Open Haiku Source Pull

Raw human-authored haiku source artifacts collected on 2026-05-18.

This directory is intentionally raw. Do not train directly from every file here
without checking `source_manifest.tsv`: some sources are clean public-domain or
permission-cleared artifacts, while others are useful bulk corpora that need
provenance review before broad reuse.

Files:

- `source_manifest.tsv`: source-level provenance, license status, and counts.
- `file_manifest.tsv`: path, byte size, and SHA-256 for every downloaded file.
- `validation_summary.json`: lightweight machine-readable validation summary.
- `raw/huggingface/davanstrien_Haiku_Dataset_train.parquet`: 144,123 English
  rows mirrored from the Kaggle haiku dataset.
- `raw/huggingface/statworx_haiku_train.parquet`: 49,024 English rows;
  downloaded as a provenance-audit candidate because the local dataset card did
  not expose a clear content license.
- `raw/plos/`: Deans and Porturas arthropod haiku workbooks from the PLOS One
  supplementary files.
- `raw/haiku_foundation/basho_zenshu_public_domain.pdf`: public-domain Japanese
  Basho source PDF.
- `raw/aozora/`: Aozora Bunko Masaoka Shiki author page, all 70 public work
  cards, and 64 available ruby text zips.

Kaggle note: the original Kaggle archive for `hjhalani30/haiku-dataset` was not
pulled directly because this environment has no Kaggle CLI or credentials. The
Hugging Face mirror above contains the same dataset family and records Kaggle as
its source.
