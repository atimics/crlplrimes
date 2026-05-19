# Reference Source Archive

This directory stores local copies of the papers cited by `paper/references.bib`.

- `sources.tsv`: BibTeX key to source PDF URL.
- `manifest.tsv`: generated download/extraction manifest with PDF paths, text paths, SHA-256 hashes, byte counts, and status.
- `mined_claims.jsonl`: manually curated source claims with local text-line evidence and scope cautions.
- `pdfs/`: downloaded source PDFs, named `{bibkey}.pdf`.
- `text/`: `pdftotext -layout` extractions, named `{bibkey}.txt`.

Refresh the archive with:

```sh
sh scripts/collect_reference_sources.sh
```

The script downloads missing PDFs only, re-extracts text when a PDF is newer than its text file, and rewrites `manifest.tsv`.
