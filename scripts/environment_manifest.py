#!/usr/bin/env python3
"""Capture a deterministic identity row for the machine running an experiment.

Writes a single JSON object to the path given by --output (default
results/latest/environment.json). Captures fields that are sensitive to runner
identity so future comparisons can detect drift. Pure stdlib; no network calls.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def safe_run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10)
        return out.stdout.strip() or out.stderr.strip()
    except Exception as exc:
        return f"<error: {exc}>"


def first_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def gather() -> dict:
    cc = os.environ.get("CC", "cc")
    cc_path = shutil.which(cc) or cc
    return {
        "schema": "crlp.environment.v1",
        "uname": platform.uname()._asdict(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "cc": cc,
        "cc_path": cc_path,
        "cc_version": first_line(safe_run([cc, "--version"])),
        "typst_version": first_line(safe_run(["typst", "--version"])) or None,
        "make_version": first_line(safe_run(["make", "--version"])),
        "git_commit": safe_run(["git", "rev-parse", "HEAD"]),
        "git_branch": safe_run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": safe_run(["git", "status", "--porcelain"]).strip() != "",
        "env": {
            k: v
            for k, v in os.environ.items()
            if k in {"REAL", "ARC_AGI_TASK_DIR", "ARC_AGI_MAX_TASKS", "CI", "GITHUB_RUN_ID", "GITHUB_SHA", "RUNNER_OS"}
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/latest/environment.json"),
        help="Where to write the manifest.",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    blob = gather()
    args.output.write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n")
    json.dump(blob, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
