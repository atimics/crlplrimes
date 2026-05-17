#!/usr/bin/env python3
"""Run ARC loader baseline/induction parity checks on public ARC splits."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_json_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def run_loader(arc_agi: Path, task_dir: Path, max_tasks: int, induction: bool) -> dict[str, Any]:
    command = [
        str(arc_agi),
        "--task-dir",
        str(task_dir),
        "--max-tasks",
        str(max_tasks),
        "--program-budget-grid",
        "--loader-summary-only",
    ]
    if induction:
        command.extend(["--induction", "--min-induction-evidence", "0"])
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"{' '.join(command)} failed")
    for row in parse_json_rows(result.stdout):
        if row.get("schema") == "crlp.arc_loader.summary.v1":
            return row
    raise RuntimeError(f"{' '.join(command)} emitted no ARC loader summary")


def split_dirs(task_dir: Path) -> dict[str, Path]:
    training = task_dir / "training"
    evaluation = task_dir / "evaluation"
    if training.is_dir() and evaluation.is_dir():
        return {"train": training, "eval": evaluation}
    raise RuntimeError(
        f"{task_dir} must contain public ARC split directories: training/ and evaluation/"
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arc-agi", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int, default=400)
    args = parser.parse_args(argv)

    if not args.task_dir:
        raise RuntimeError("--task-dir is required")
    splits = split_dirs(args.task_dir)
    baseline: dict[str, dict[str, Any]] = {}
    induction: dict[str, dict[str, Any]] = {}
    for split, path in splits.items():
        baseline[split] = run_loader(args.arc_agi, path, args.max_tasks, False)
        induction[split] = run_loader(args.arc_agi, path, args.max_tasks, True)

    train_holds = int(induction["train"].get("certified_tasks", 0)) >= int(
        baseline["train"].get("certified_tasks", 0)
    )
    eval_holds = int(induction["eval"].get("certified_tasks", 0)) >= int(
        baseline["eval"].get("certified_tasks", 0)
    )
    row = {
        "schema": "crlp.arc_induction.parity.v1",
        "report_version": 1,
        "task": "arc_agi",
        "task_dir": str(args.task_dir),
        "max_tasks": args.max_tasks,
        "train_baseline_loaded": baseline["train"].get("loaded_tasks", 0),
        "train_induction_loaded": induction["train"].get("loaded_tasks", 0),
        "train_baseline_certified": baseline["train"].get("certified_tasks", 0),
        "train_induction_certified": induction["train"].get("certified_tasks", 0),
        "train_baseline_cells": baseline["train"].get("test_correct_cells", 0),
        "train_induction_cells": induction["train"].get("test_correct_cells", 0),
        "eval_baseline_loaded": baseline["eval"].get("loaded_tasks", 0),
        "eval_induction_loaded": induction["eval"].get("loaded_tasks", 0),
        "eval_baseline_certified": baseline["eval"].get("certified_tasks", 0),
        "eval_induction_certified": induction["eval"].get("certified_tasks", 0),
        "eval_baseline_cells": baseline["eval"].get("test_correct_cells", 0),
        "eval_induction_cells": induction["eval"].get("test_correct_cells", 0),
        "parity_holds": train_holds and eval_holds,
    }
    print(json.dumps(row, sort_keys=True, separators=(",", ":")))
    return 0 if row["parity_holds"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
