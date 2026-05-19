"""Subprocess helpers for experiment drivers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


def shell_label(command: Sequence[str]) -> str:
    return " ".join(str(part) for part in command)


def json_rows_from_stdout(stdout: str, *, schema: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if schema is None or row.get("schema") == schema:
            rows.append(row)
    return rows


def run_json_command(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path | None = None,
    schema: str | None = None,
    echo: bool = True,
) -> list[dict[str, Any]]:
    if echo:
        print(f"running {shell_label(command)}", flush=True)
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise subprocess.CalledProcessError(proc.returncode, list(command))
    return json_rows_from_stdout(proc.stdout, schema=schema)
