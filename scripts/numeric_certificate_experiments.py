#!/usr/bin/env python3
"""Run experiment targets with numeric step certificate emission enabled."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


SUMMARY_SCHEMA = "crlp.numeric_certificate_experiment_summary.v1"

DEFAULT_TARGETS = [
    "action-score",
    "prime-gap-trace",
    "prime-rule-search-smoke",
    "delayed-claim-eval",
    "grid-path-eval",
    "multitask-domain",
    "symbolic-domains",
    "program-synthesis",
    "program-synthesis-baselines",
    "modular-gate-search-smoke",
    "arc-agi-induction-smoke",
    "haiku-lm-smoke",
    "haiku-word-lm-smoke",
    "haiku-understanding-smoke",
    "signal-mining-smoke",
]


def safe_name(target: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in target)


def summarize_certificates(path: Path) -> dict[str, Any]:
    rows = 0
    committed = 0
    reasons: Counter[str] = Counter()
    finite_after = 0
    if not path.exists():
        return {
            "certificate_rows": 0,
            "committed_rows": 0,
            "rejected_rows": 0,
            "finite_after_rows": 0,
            "rejection_reasons": {},
        }
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("schema") != "crlp.numeric_step_certificate.v1":
                continue
            rows += 1
            reason = str(row.get("rejection_reason", "unknown"))
            reasons[reason] += 1
            if row.get("step_committed") is True:
                committed += 1
            if row.get("finite_after") is True:
                finite_after += 1
    return {
        "certificate_rows": rows,
        "committed_rows": committed,
        "rejected_rows": rows - committed,
        "finite_after_rows": finite_after,
        "rejection_reasons": dict(sorted(reasons.items())),
    }


def run_target(args: argparse.Namespace, target: str, output_dir: Path) -> dict[str, Any]:
    name = safe_name(target)
    cert_path = output_dir / f"{name}.numeric_steps.jsonl"
    stdout_path = output_dir / f"{name}.stdout.log"
    stderr_path = output_dir / f"{name}.stderr.log"
    cert_path.write_text("", encoding="utf-8")

    env = os.environ.copy()
    env["CRLP_NUMERIC_CERT_PATH"] = str(cert_path)
    env["CRLP_NUMERIC_CERT_TARGET"] = target

    command = [args.make, target, f"REAL={args.real}"]
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_fp, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_fp:
        result = subprocess.run(
            command,
            cwd=args.repo,
            env=env,
            stdout=stdout_fp,
            stderr=stderr_fp,
            check=False,
        )
    elapsed_seconds = time.perf_counter() - started

    summary = summarize_certificates(cert_path)
    row: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "target": target,
        "status": "passed" if result.returncode == 0 else "failed",
        "exit_code": result.returncode,
        "certificate_path": str(cert_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "elapsed_seconds": elapsed_seconds,
    }
    row.update(summary)
    return row


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--make", default="make")
    parser.add_argument("--real", default="float")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/float/numeric_certificates"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="Summary JSONL path; defaults to output-dir/summary.jsonl",
    )
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        help="Make target to run. Repeat for multiple targets.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = args.repo / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary or (output_dir / "summary.jsonl")
    if not summary_path.is_absolute():
        summary_path = args.repo / summary_path

    rows: list[dict[str, Any]] = []
    failed = 0
    for target in args.targets or DEFAULT_TARGETS:
        print(f"numeric-certificates: {target}", flush=True)
        row = run_target(args, target, output_dir)
        rows.append(row)
        if row["status"] != "passed":
            failed += 1
        print(
            f"  {row['status']} rows={row['certificate_rows']} "
            f"committed={row['committed_rows']} rejected={row['rejected_rows']} "
            f"elapsed={row['elapsed_seconds']:.3f}s",
            flush=True,
        )

    write_jsonl(summary_path, rows)
    print(f"wrote {summary_path}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
