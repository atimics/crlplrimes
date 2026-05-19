#!/usr/bin/env python3
"""Clean Hugging Face haiku dumps into the repo haiku corpus format."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


VOWELS = set("aeiouy")
TARGETS = (5, 7, 5)
MAX_LINE_CHARS = (28, 34, 28)
TEXT_COLUMNS = ("haiku", "text", "text_punc", "poem")
SUPPORTED_SUFFIXES = {".parquet", ".jsonl", ".json", ".csv", ".txt", ".tsv"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def syllable_proxy(word: str) -> int:
    count = 0
    prev_vowel = False
    for char in word:
        vowel = char in VOWELS
        if vowel and not prev_vowel:
            count += 1
        prev_vowel = vowel
    return count


def line_syllables(line: str) -> int:
    return sum(syllable_proxy(word) for word in line.split())


def normalize_text(text: str) -> str:
    text = text.replace("\\n", "\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[-–—]+", " ", text)
    text = re.sub(r"[^a-z\n/ ]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def split_candidate(text: str) -> list[str]:
    normalized = normalize_text(text)
    if " / " in normalized or normalized.count("/") >= 2:
        parts = [part.strip() for part in normalized.split("/")]
    else:
        parts = [part.strip() for part in normalized.splitlines()]
    return [re.sub(r" +", " ", part).strip() for part in parts if part.strip()]


def validate_lines(
    lines: list[str],
    word_list: set[str] | None,
    filter_word_list: bool,
) -> tuple[str | None, str | None, list[int]]:
    if len(lines) != 3:
        return None, "line_count", []
    clean_lines = []
    syllables = []
    for i, line in enumerate(lines):
        if len(line) < 3:
            return None, "too_short", []
        if len(line) > MAX_LINE_CHARS[i]:
            return None, "too_long", []
        words = line.split()
        if not words:
            return None, "empty_line", []
        if any(not re.fullmatch(r"[a-z]+", word) for word in words):
            return None, "non_alpha", []
        if filter_word_list and word_list is not None:
            if any(word not in word_list for word in words):
                return None, "not_in_word_list", []
        count = line_syllables(line)
        if count < TARGETS[i] - 1:
            return None, "syllable_low", []
        if count > TARGETS[i] + 2:
            return None, "syllable_high", []
        clean_lines.append(line)
        syllables.append(count)
    return "\n".join(clean_lines), None, syllables


def load_word_list(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    words = set()
    with path.open("r", encoding="ascii", errors="ignore") as handle:
        for line in handle:
            word = line.strip().lower()
            if re.fullmatch(r"[a-z]+", word):
                words.add(word)
    return words


def iter_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    paths = [
        path
        for path in sorted(input_path.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return paths


def import_pandas():
    try:
        import pandas as pd  # type: ignore

        return pd
    except Exception as exc:  # pragma: no cover - dependency path
        raise SystemExit(
            "clean_haiku_hf_corpus.py requires pandas and a parquet engine "
            "(pyarrow or fastparquet) for parquet inputs"
        ) from exc


def row_texts_from_record(record: dict[str, object]) -> Iterable[str]:
    for column in TEXT_COLUMNS:
        value = record.get(column)
        if isinstance(value, str) and value.strip():
            yield value


def iter_records(path: Path) -> Iterable[dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        pd = import_pandas()
        frame = pd.read_parquet(path)
        for record in frame.to_dict(orient="records"):
            yield record
        return
    if suffix == ".csv":
        pd = import_pandas()
        frame = pd.read_csv(path)
        for record in frame.to_dict(orient="records"):
            yield record
        return
    if suffix == ".tsv":
        pd = import_pandas()
        frame = pd.read_csv(path, sep="\t")
        for record in frame.to_dict(orient="records"):
            yield record
        return
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value
        return
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield item
        elif isinstance(value, dict):
            rows = value.get("rows") or value.get("data") or value.get("train")
            if isinstance(rows, list):
                for item in rows:
                    if isinstance(item, dict):
                        yield item
            else:
                yield value
        return
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
        for stanza in re.split(r"\n\s*\n", text):
            yield {"text": stanza}


def write_corpus(path: Path, poems: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        for i, poem in enumerate(poems):
            if i:
                handle.write("\n")
            handle.write(poem)
            handle.write("\n")


def write_vocab(path: Path, poems: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    words = sorted({word for poem in poems for word in poem.split()})
    with path.open("w", encoding="ascii") as handle:
        for word in words:
            handle.write(word)
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/haiku_open/raw/huggingface")
    parser.add_argument("--output", default="data/haiku_open/clean/hf_haiku_clean.txt")
    parser.add_argument("--vocab-output", default="data/haiku_open/clean/hf_vocab.txt")
    parser.add_argument("--manifest", default="data/haiku_open/clean/hf_manifest.json")
    parser.add_argument("--word-list", default="data/wordlists/enable2k.txt")
    parser.add_argument("--filter-word-list", type=int, default=0)
    parser.add_argument("--max-poems", type=int, default=8000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    vocab_path = Path(args.vocab_output)
    manifest_path = Path(args.manifest)
    word_list_path = Path(args.word_list) if args.word_list else None
    word_list = load_word_list(word_list_path)
    filter_word_list = bool(args.filter_word_list)

    paths = iter_paths(input_path)
    poems: list[str] = []
    seen = set()
    rejects = Counter()
    exact_syllables = Counter()
    source_rows = Counter()
    source_accepted = Counter()
    source_rejected = Counter()
    input_hashes = {str(path): sha256(path) for path in paths}
    max_reached = False
    source_limits: dict[str, int] = {}
    if args.max_poems > 0 and paths:
        base = args.max_poems // len(paths)
        remainder = args.max_poems % len(paths)
        for i, path in enumerate(paths):
            source_limits[str(path)] = base + (1 if i < remainder else 0)

    for path in paths:
        for record in iter_records(path):
            source_rows[str(path)] += 1
            accepted_from_record = False
            for text in row_texts_from_record(record):
                lines = split_candidate(text)
                poem, reason, syllables = validate_lines(lines, word_list, filter_word_list)
                if poem is None:
                    rejects[reason or "unknown"] += 1
                    continue
                if poem in seen:
                    rejects["duplicate"] += 1
                    continue
                seen.add(poem)
                poems.append(poem)
                exact_syllables["-".join(str(x) for x in syllables)] += 1
                source_accepted[str(path)] += 1
                accepted_from_record = True
                break
            if not accepted_from_record:
                source_rejected[str(path)] += 1
            source_limit = source_limits.get(str(path), 0)
            if source_limit > 0 and source_accepted[str(path)] >= source_limit:
                break
            if args.max_poems > 0 and len(poems) >= args.max_poems:
                max_reached = True
                break
    if args.max_poems > 0 and len(poems) >= args.max_poems:
        max_reached = True

    write_corpus(output_path, poems)
    write_vocab(vocab_path, poems)
    words = Counter(word for poem in poems for word in poem.split())
    covered_word_types = 0
    covered_word_tokens = 0
    if word_list is not None:
        covered_word_types = sum(1 for word in words if word in word_list)
        covered_word_tokens = sum(count for word, count in words.items() if word in word_list)

    manifest = {
        "schema": "crlp.haiku_corpus.huggingface_clean.v1",
        "input": str(input_path),
        "inputs": [str(path) for path in paths],
        "input_sha256": input_hashes,
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "vocab_output": str(vocab_path),
        "vocab_sha256": sha256(vocab_path),
        "word_list": str(word_list_path) if word_list_path is not None else None,
        "word_list_sha256": sha256(word_list_path) if word_list_path is not None else None,
        "filter_word_list": filter_word_list,
        "max_poems": args.max_poems,
        "max_poems_reached": max_reached,
        "accepted_poems": len(poems),
        "accepted_lines": len(poems) * 3,
        "unique_words": len(words),
        "word_tokens": sum(words.values()),
        "word_list_covered_word_types": covered_word_types,
        "word_list_covered_word_tokens": covered_word_tokens,
        "syllable_proxy_patterns": dict(sorted(exact_syllables.items())),
        "rejects": dict(sorted(rejects.items())),
        "source_rows": dict(source_rows),
        "source_accepted": dict(source_accepted),
        "source_rejected": dict(source_rejected),
        "note": "Cleaned from user-provided Hugging Face haiku dumps; license/provenance remains dataset-specific.",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")

    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "accepted_poems": len(poems),
                "output": str(output_path),
                "vocab_output": str(vocab_path),
                "manifest": str(manifest_path),
                "output_sha256": manifest["output_sha256"],
                "max_poems_reached": max_reached,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
