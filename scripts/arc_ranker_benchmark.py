#!/usr/bin/env python3
"""Benchmark ARC visual ranker cold and warm cache behavior."""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from crlp.io import write_jsonl


SCHEMA = "crlp.arc_ranker_benchmark.v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_one_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("{"):
                row = json.loads(line)
                if isinstance(row, dict):
                    return row
    return {}


def run_ranker(
    *,
    label: str,
    runtime: Path,
    attempts_tsv: Path,
    cache: Path,
    output_dir: Path,
    epochs: int | None,
    max_per_task: int,
) -> dict[str, Any]:
    proposals = output_dir / f"{label}.proposals.jsonl"
    summary = output_dir / f"{label}.summary.jsonl"
    command = [
        str(runtime),
        "--arc-pipeline-attempts",
        str(attempts_tsv),
        "--arc-pipeline-proposals-output",
        str(proposals),
        "--arc-pipeline-summary-output",
        str(summary),
        "--arc-pipeline-cache",
        str(cache),
        "--arc-pipeline-max-per-task",
        str(max_per_task),
    ]
    if epochs is not None:
        command.extend(["--epochs", str(epochs)])
    started = time.perf_counter()
    subprocess.run(command, check=True)
    elapsed = time.perf_counter() - started
    summary_row = load_one_json(summary)
    return {
        "label": label,
        "elapsed_seconds": round(elapsed, 6),
        "epochs": summary_row.get("epochs"),
        "summary_path": str(summary),
        "proposals_path": str(proposals),
        "summary_sha256": file_sha256(summary),
        "proposals_sha256": file_sha256(proposals),
        "source_attempt_count": summary_row.get("source_attempt_count"),
        "target_attempt_count": summary_row.get("target_attempt_count"),
        "target_task_count": summary_row.get("target_task_count"),
        "accepted_target_task_count": summary_row.get("accepted_target_task_count"),
        "top1_rate": summary_row.get("top1_rate"),
        "top1_all_task_rate": summary_row.get("top1_all_task_rate"),
        "retained_rate": summary_row.get("retained_rate"),
        "vocab_size": summary_row.get("vocab_size"),
        "proposal_count": summary_row.get("proposal_count"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--attempts-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-per-task", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.runtime.is_file():
        raise SystemExit(f"missing ranker runtime: {args.runtime}")
    if not args.attempts_tsv.is_file():
        raise SystemExit(f"missing ARC ranker attempts TSV: {args.attempts_tsv}")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if args.max_per_task <= 0:
        raise SystemExit("--max-per-task must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "arc_ranker.cache.bin"
    if cache.exists():
        cache.unlink()

    cold = run_ranker(
        label=f"cold_epochs{args.epochs}",
        runtime=args.runtime,
        attempts_tsv=args.attempts_tsv,
        cache=cache,
        output_dir=args.output_dir,
        epochs=args.epochs,
        max_per_task=args.max_per_task,
    )
    cache_size = cache.stat().st_size if cache.exists() else 0
    warm = run_ranker(
        label=f"warm_epochs{args.epochs}",
        runtime=args.runtime,
        attempts_tsv=args.attempts_tsv,
        cache=cache,
        output_dir=args.output_dir,
        epochs=args.epochs,
        max_per_task=args.max_per_task,
    )
    warm_default = run_ranker(
        label="warm_default",
        runtime=args.runtime,
        attempts_tsv=args.attempts_tsv,
        cache=cache,
        output_dir=args.output_dir,
        epochs=None,
        max_per_task=args.max_per_task,
    )
    cold_warm_summary_equal = filecmp.cmp(
        cold["summary_path"], warm["summary_path"], shallow=False
    )
    cold_warm_proposals_equal = filecmp.cmp(
        cold["proposals_path"], warm["proposals_path"], shallow=False
    )
    if not cold_warm_summary_equal or not cold_warm_proposals_equal:
        raise SystemExit("cold and warm ranker outputs differ")

    row = {
        "schema": SCHEMA,
        "report_version": 1,
        "runtime": str(args.runtime),
        "attempts_tsv": str(args.attempts_tsv),
        "attempts_tsv_size_bytes": args.attempts_tsv.stat().st_size,
        "cache_path": str(cache),
        "cache_size_bytes": cache_size,
        "cache_size_mib": round(cache_size / (1024.0 * 1024.0), 3),
        "epochs": args.epochs,
        "max_per_task": args.max_per_task,
        "cold_warm_summary_equal": cold_warm_summary_equal,
        "cold_warm_proposals_equal": cold_warm_proposals_equal,
        "runs": [cold, warm, warm_default],
    }
    write_jsonl(args.report, [row], compact=True)
    print(
        "arc-ranker-benchmark "
        f"cold={cold['elapsed_seconds']:.3f}s "
        f"warm={warm['elapsed_seconds']:.3f}s "
        f"warm_default={warm_default['elapsed_seconds']:.3f}s "
        f"cache={row['cache_size_mib']:.1f}MiB "
        f"report={args.report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
