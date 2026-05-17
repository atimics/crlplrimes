#!/bin/sh
set -eu

ROOT=${1:-paper/sources}
SOURCES="$ROOT/sources.tsv"
PDF_DIR="$ROOT/pdfs"
TEXT_DIR="$ROOT/text"
MANIFEST="$ROOT/manifest.tsv"

mkdir -p "$PDF_DIR" "$TEXT_DIR"

printf 'bibkey\turl\tpdf\ttext\tsha256\tpdf_bytes\ttext_bytes\tstatus\n' > "$MANIFEST"

tail -n +2 "$SOURCES" | while IFS="$(printf '\t')" read -r key url; do
  pdf="$PDF_DIR/$key.pdf"
  text="$TEXT_DIR/$key.txt"
  tmp="$pdf.tmp"

  if [ ! -s "$pdf" ]; then
    curl -L --fail --retry 3 --retry-delay 2 --connect-timeout 20 -o "$tmp" "$url"
    mv "$tmp" "$pdf"
  fi

  if [ ! -s "$text" ] || [ "$pdf" -nt "$text" ]; then
    pdftotext -layout "$pdf" "$text"
  fi

  sha=$(shasum -a 256 "$pdf" | awk '{print $1}')
  pdf_bytes=$(wc -c < "$pdf" | tr -d ' ')
  text_bytes=$(wc -c < "$text" | tr -d ' ')
  status=ok
  if [ "$text_bytes" -eq 0 ]; then
    status=empty_text
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$key" "$url" "$pdf" "$text" "$sha" "$pdf_bytes" "$text_bytes" "$status" >> "$MANIFEST"
done
