"""Small file-format helpers for CRLP result artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


def read_jsonl(path: Path, *, schema: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            row = json.loads(line)
            if schema is None or row.get("schema") == schema:
                rows.append(row)
    return rows


def write_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
    *,
    sort_keys: bool = True,
    compact: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    separators = (",", ":") if compact else None
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=sort_keys, separators=separators) + "\n"
            )


def ordered_fieldnames(rows: Sequence[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
    empty_text: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if empty_text:
            path.write_text("", encoding="utf-8")
        return
    fields = list(fieldnames) if fieldnames is not None else ordered_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

