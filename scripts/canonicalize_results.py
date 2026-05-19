#!/usr/bin/env python3
"""Project a comparison.jsonl onto its canonical (timing-free) subset.

Fields like `model_ms_per_sample` and `verifier_ms_per_sample` are real wall
clock measurements that depend on the runner, so they trigger spurious
"results changed" diffs. This script:

  - Reads results/latest/comparison.jsonl (default).
  - Drops timing-sensitive fields per row.
  - Writes results/latest/comparison.canonical.jsonl with stable key ordering.

Use the canonical file as the diff target for "should we rewrite the paper?"
The full comparison.jsonl is still uploaded as an artifact for inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from crlp.io import read_jsonl, write_jsonl

# Fields that vary by runner / wallclock and must be stripped before diffing.
TIMING_FIELDS = {
    "model_ms_per_sample",
    "verifier_ms_per_sample",
    "wallclock_ms",
    "wall_clock_ms",
    "elapsed_ms",
}


def canonicalize(row: dict) -> dict:
    out = {k: v for k, v in row.items() if k not in TIMING_FIELDS}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/latest/comparison.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/latest/comparison.canonical.jsonl"),
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"missing input: {args.input}", flush=True)
        return 1

    write_jsonl(
        args.output,
        (canonicalize(row) for row in read_jsonl(args.input)),
        compact=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
