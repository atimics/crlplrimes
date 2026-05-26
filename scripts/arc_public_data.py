#!/usr/bin/env python3
"""Download and validate the public ARC-AGI-1 JSON corpus."""

from __future__ import annotations

import argparse
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from crlp.io import write_jsonl


SUMMARY_SCHEMA = "crlp.arc_public_data.download.v1"
DEFAULT_ZIP_URL = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"


def split_dir(data_dir: Path, split: str) -> Path:
    return data_dir / split


def task_paths(data_dir: Path, split: str) -> list[Path]:
    directory = split_dir(data_dir, split)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"))


def validate_task(path: Path) -> None:
    try:
        with path.open(encoding="utf-8") as handle:
            row = json.load(handle)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid ARC JSON in {path}: {exc}") from exc
    if not isinstance(row.get("train"), list) or not isinstance(row.get("test"), list):
        raise RuntimeError(f"invalid ARC task shape in {path}: missing train/test arrays")


def validate_data_dir(
    data_dir: Path, expected_training: int, expected_evaluation: int
) -> dict[str, Any]:
    training = task_paths(data_dir, "training")
    evaluation = task_paths(data_dir, "evaluation")
    for path in training + evaluation:
        validate_task(path)
    return {
        "training_tasks": len(training),
        "evaluation_tasks": len(evaluation),
        "expected_training_tasks": expected_training,
        "expected_evaluation_tasks": expected_evaluation,
        "complete": len(training) >= expected_training
        and len(evaluation) >= expected_evaluation,
    }


def download_zip(zip_url: str, destination: Path) -> int:
    request = urllib.request.Request(
        zip_url,
        headers={"User-Agent": "crlplrimes-arc-public-data/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        content = response.read()
    destination.write_bytes(content)
    return len(content)


def extract_arc_data(zip_path: Path, data_dir: Path) -> int:
    extracted = 0
    data_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.endswith(".json"):
                continue
            parts = Path(info.filename).parts
            if "data" not in parts:
                continue
            data_index = parts.index("data")
            if len(parts) <= data_index + 2:
                continue
            split = parts[data_index + 1]
            filename = parts[-1]
            if split not in {"training", "evaluation"}:
                continue
            target = data_dir / split / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source:
                target.write_bytes(source.read())
            extracted += 1
    return extracted


def ensure_public_data(args: argparse.Namespace) -> dict[str, Any]:
    before = validate_data_dir(
        args.data_dir, args.expected_training, args.expected_evaluation
    )
    if before["complete"] and not args.force:
        return {
            "schema": SUMMARY_SCHEMA,
            "report_version": 1,
            "status": "present",
            "data_dir": str(args.data_dir),
            "zip_url": args.zip_url,
            **before,
        }

    args.data_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="arc-public-data-", dir=args.data_dir.parent) as tmp:
        zip_path = Path(tmp) / "arc_agi.zip"
        bytes_downloaded = download_zip(args.zip_url, zip_path)
        extracted_files = extract_arc_data(zip_path, args.data_dir)

    after = validate_data_dir(
        args.data_dir, args.expected_training, args.expected_evaluation
    )
    if not after["complete"]:
        raise RuntimeError(
            "downloaded ARC data is incomplete: "
            f"training={after['training_tasks']}/{args.expected_training} "
            f"evaluation={after['evaluation_tasks']}/{args.expected_evaluation}"
        )
    return {
        "schema": SUMMARY_SCHEMA,
        "report_version": 1,
        "status": "downloaded",
        "data_dir": str(args.data_dir),
        "zip_url": args.zip_url,
        "bytes_downloaded": bytes_downloaded,
        "extracted_files": extracted_files,
        **after,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--zip-url", default=DEFAULT_ZIP_URL)
    parser.add_argument("--expected-training", type=int, default=400)
    parser.add_argument("--expected-evaluation", type=int, default=400)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = ensure_public_data(args)
    write_jsonl(args.summary, [summary], compact=True)
    print(
        "arc-public-data "
        f"{summary['status']} training={summary['training_tasks']} "
        f"evaluation={summary['evaluation_tasks']} data_dir={summary['data_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
