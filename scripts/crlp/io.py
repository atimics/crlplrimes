"""Small file-format helpers for CRLP result artifacts."""

from __future__ import annotations

import csv
import json
import math
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


def read_jsonl_required(path: Path, *, schema: str | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return read_jsonl(path, schema=schema)


def read_jsonl_lenient(path: Path, *, schema: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
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


def safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0, *, finite: bool = False) -> float:
    if isinstance(value, bool) or value is None:
        return default
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return default
    if finite and (math.isnan(value_float) or math.isinf(value_float)):
        return default
    return value_float


def safe_finite_float(value: Any, default: float = 0.0) -> float:
    return safe_float(value, default, finite=True)


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
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
