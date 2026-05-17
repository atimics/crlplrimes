#!/usr/bin/env python3
"""Run ARC loader baseline/induction parity checks on public ARC splits."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
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


def run_loader(
    arc_agi: Path,
    task_dir: Path,
    max_tasks: int,
    induction: bool,
    program_train_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    command = [
        str(arc_agi),
        "--task-dir",
        str(task_dir),
        "--max-tasks",
        str(max_tasks),
        "--program-budget-grid",
    ]
    if induction:
        command.extend(["--induction", "--min-induction-evidence", "0"])
        if program_train_dir is not None:
            command.extend(
                [
                    "--program-train-dir",
                    str(program_train_dir),
                    "--program-train-max-tasks",
                    str(max_tasks),
                ]
            )
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"{' '.join(command)} failed")
    summary: dict[str, Any] | None = None
    tasks: dict[str, dict[str, Any]] = {}
    for row in parse_json_rows(result.stdout):
        if row.get("schema") == "crlp.arc_loader.summary.v1":
            summary = row
        elif row.get("schema") == "crlp.arc_transform.summary.v1":
            task_id = str(row.get("task_instance", ""))
            if task_id:
                tasks[task_id] = row
    if summary is not None:
        return summary, tasks
    raise RuntimeError(f"{' '.join(command)} emitted no ARC loader summary")


def split_dirs(task_dir: Path) -> dict[str, Path]:
    training = task_dir / "training"
    evaluation = task_dir / "evaluation"
    if training.is_dir() and evaluation.is_dir():
        return {"train": training, "eval": evaluation}
    raise RuntimeError(
        f"{task_dir} must contain public ARC split directories: training/ and evaluation/"
    )


def compare_tasks(
    baseline: dict[str, dict[str, Any]],
    induction: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    added: list[str] = []
    regressions: list[str] = []
    for task_id in sorted(set(baseline) | set(induction)):
        base = baseline.get(task_id, {})
        induced = induction.get(task_id, {})
        base_certified = bool(base.get("certified", False))
        induced_certified = bool(induced.get("certified", False))
        base_cells = int(base.get("test_correct_cells", 0) or 0)
        induced_cells = int(induced.get("test_correct_cells", 0) or 0)
        if induced_certified and not base_certified:
            added.append(task_id)
        if (base_certified and not induced_certified) or induced_cells < base_cells:
            regressions.append(task_id)
    return added, regressions


def write_report(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ARC Induction Phase 4",
        "",
        f"Task dir: `{row['task_dir']}`",
        f"Program train dir: `{row['program_train_dir']}`",
        f"Max tasks per split: `{row['max_tasks']}`",
        "",
        "| Split | Baseline Certified | Induction Certified | Delta | Added Tasks | Regressions |",
        "|---|---:|---:|---:|---|---|",
    ]
    for split in ("train", "eval"):
        base = int(row[f"{split}_baseline_certified"])
        induced = int(row[f"{split}_induction_certified"])
        added = ", ".join(row[f"{split}_added_tasks"]) or "-"
        regressions = ", ".join(row[f"{split}_regressions"]) or "-"
        lines.append(
            f"| {split} | {base} | {induced} | {induced - base:+d} | {added} | {regressions} |"
        )
    lines.extend(
        [
            "",
            f"Parity holds: `{str(row['parity_holds']).lower()}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arc-agi", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int, default=400)
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args(argv)

    if not args.task_dir:
        raise RuntimeError("--task-dir is required")
    splits = split_dirs(args.task_dir)
    baseline: dict[str, dict[str, Any]] = {}
    induction: dict[str, dict[str, Any]] = {}
    baseline_tasks: dict[str, dict[str, dict[str, Any]]] = {}
    induction_tasks: dict[str, dict[str, dict[str, Any]]] = {}
    for split, path in splits.items():
        baseline[split], baseline_tasks[split] = run_loader(
            args.arc_agi, path, args.max_tasks, False
        )
        induction[split], induction_tasks[split] = run_loader(
            args.arc_agi, path, args.max_tasks, True, splits["train"]
        )

    train_added, train_regressions = compare_tasks(
        baseline_tasks["train"], induction_tasks["train"]
    )
    eval_added, eval_regressions = compare_tasks(
        baseline_tasks["eval"], induction_tasks["eval"]
    )

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
        "program_train_dir": str(splits["train"]),
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
        "train_added_tasks": train_added,
        "train_regressions": train_regressions,
        "eval_added_tasks": eval_added,
        "eval_regressions": eval_regressions,
        "parity_holds": train_holds and eval_holds,
    }
    report_path = args.report_path or (
        Path("results") / f"{date.today().isoformat()}-arc-induction-phase4.md"
    )
    write_report(report_path, row)
    row["report_path"] = str(report_path)
    print(json.dumps(row, sort_keys=True, separators=(",", ":")))
    return 0 if row["parity_holds"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
