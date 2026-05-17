#!/usr/bin/env python3
"""Check ARC exact-gate and budgeted-proposer invariance across scorer contexts."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


CONTEXTS = ["shared_scorer_context", "single_task_scorer_context"]
TRANSFORM_FIELDS = [
    "grammar_version",
    "rules",
    "train_consistent_rules",
    "unique_test_prediction",
    "test_exact",
    "certified",
    "test_pairs",
    "test_cells",
    "test_correct_cells",
    "prediction_hash",
    "accepted_rules",
]
LOADER_FIELDS = [
    "grammar_version",
    "files_seen",
    "loaded_tasks",
    "parse_errors",
    "train_pairs",
    "test_pairs",
    "rules",
    "train_consistent_rules",
    "unique_prediction_tasks",
    "scored_tasks",
    "test_exact_tasks",
    "certified_tasks",
    "unsupported_tasks",
    "test_cells",
    "test_correct_cells",
    "cell_accuracy",
]
PROPOSER_FIELDS = [
    "budget",
    "proposed_programs",
    "certified_programs",
    "accepted_programs",
    "event_count",
    "covered_events",
    "false_rejections",
    "utility",
    "program_set_hash",
]


def parse_csv_set(text: str) -> list[str]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one split")
    for value in values:
        if value not in {"train", "eval"}:
            raise argparse.ArgumentTypeError("splits must be train, eval, or train,eval")
    return values


def parse_json_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        rows.append(json.loads(line))
    return rows


def split_dir(task_dir: Path, split: str) -> Path:
    names = [split]
    if split == "train":
        names.append("training")
    elif split == "eval":
        names.append("evaluation")
    for name in names:
        candidate = task_dir / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"could not find split {split} under {task_dir}")


def run_arc_loader(command: list[str], cwd: Path, log_path: Path) -> list[dict[str, Any]]:
    print(f"running {' '.join(command)}", flush=True)
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise subprocess.CalledProcessError(proc.returncode, command)
    return parse_json_rows(proc.stdout)


def normalize(row: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    return {field: row.get(field) for field in fields}


def keyed_rows(rows: list[dict[str, Any]], schema: str, key_fields: list[str]) -> dict[tuple[Any, ...], dict[str, Any]]:
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if row.get("schema") != schema:
            continue
        key = tuple(row.get(field) for field in key_fields)
        out[key] = row
    return out


def compare_maps(
    split: str,
    row_kind: str,
    left: dict[tuple[Any, ...], dict[str, Any]],
    right: dict[tuple[Any, ...], dict[str, Any]],
    fields: list[str],
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for key in sorted(set(left) | set(right)):
        left_row = left.get(key)
        right_row = right.get(key)
        if left_row is None or right_row is None:
            mismatches.append(
                {
                    "split": split,
                    "row_kind": row_kind,
                    "key": "|".join(str(part) for part in key),
                    "field": "__presence__",
                    "shared_value": left_row is not None,
                    "single_value": right_row is not None,
                }
            )
            continue
        for field in fields:
            if left_row.get(field) != right_row.get(field):
                mismatches.append(
                    {
                        "split": split,
                        "row_kind": row_kind,
                        "key": "|".join(str(part) for part in key),
                        "field": field,
                        "shared_value": left_row.get(field),
                        "single_value": right_row.get(field),
                    }
                )
    return mismatches


def summarize_split(split: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    transforms = [row for row in rows if row.get("schema") == "crlp.arc_transform.summary.v1"]
    loader = next((row for row in rows if row.get("schema") == "crlp.arc_loader.summary.v1"), {})
    proposers = [row for row in rows if row.get("schema") == "crlp.program_proposer.summary.v1"]
    return {
        "split": split,
        "transform_rows": len(transforms),
        "loader_loaded_tasks": int(loader.get("loaded_tasks", 0) or 0),
        "loader_parse_errors": int(loader.get("parse_errors", 0) or 0),
        "loader_certified_tasks": int(loader.get("certified_tasks", 0) or 0),
        "loader_test_exact_tasks": int(loader.get("test_exact_tasks", 0) or 0),
        "loader_test_correct_cells": int(loader.get("test_correct_cells", 0) or 0),
        "proposer_rows": len(proposers),
        "budgets": ",".join(str(row.get("budget")) for row in proposers[0::5]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(
    path: Path,
    task_dir: Path,
    max_tasks: int,
    split_summaries: list[dict[str, Any]],
    mismatches: list[dict[str, Any]],
) -> None:
    exact_mismatches = [row for row in mismatches if row["row_kind"] in {"transform", "loader"}]
    proposer_mismatches = [row for row in mismatches if row["row_kind"] == "proposer"]
    lines = [
        "# ARC Gate Invariance",
        "",
        f"Task dir: `{task_dir}`",
        f"Max tasks per split: `{max_tasks}`",
        "Compared contexts: `shared_scorer_context` vs `single_task_scorer_context`.",
        "",
        "Current implementation note: the public ARC transform gate and program-proposer path do not consume a multitask scorer checkpoint. The contexts here are therefore an explicit invariance check over the current scorer-independent gate/proposer code path.",
        "",
        "| Split | Tasks | Parse errors | Certified tasks | Test-exact tasks | Correct cells | Transform rows | Proposer rows | Budgets |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in split_summaries:
        lines.append(
            f"| `{row['split']}` | {row['loader_loaded_tasks']} | "
            f"{row['loader_parse_errors']} | {row['loader_certified_tasks']} | "
            f"{row['loader_test_exact_tasks']} | {row['loader_test_correct_cells']} | "
            f"{row['transform_rows']} | {row['proposer_rows']} | `{row['budgets']}` |"
        )
    lines.extend(
        [
            "",
            "## Result",
            "",
            f"- exact gate/loader mismatches: `{len(exact_mismatches)}`",
            f"- budgeted proposer aggregate mismatches: `{len(proposer_mismatches)}`",
        ]
    )
    if not mismatches:
        lines.append(
            "- R-MT.7 holds for the measured code path: accepted rule sets, prediction hashes, loader summaries, and budgeted proposer aggregates are invariant across the two scorer contexts."
        )
    else:
        lines.append("- R-MT.7 found mismatches; inspect `rmt7_arc_gate_invariance_mismatches.csv`.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arc-agi", type=Path, default=Path("build/float/arc_agi"))
    parser.add_argument("--task-dir", type=Path, default=Path("/private/tmp/ARC-AGI/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/2026-05-17-multitask-roadmap"))
    parser.add_argument("--splits", type=parse_csv_set, default=parse_csv_set("train,eval"))
    parser.add_argument("--max-tasks", type=int, default=400)
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    arc_agi = args.arc_agi if args.arc_agi.is_absolute() else repo / args.arc_agi
    task_dir = args.task_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    commands: list[str] = []
    raw_rows: list[dict[str, Any]] = []
    rows_by_split_context: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for split in args.splits:
        path = split_dir(task_dir, split)
        for context in CONTEXTS:
            command = [
                str(arc_agi),
                "--task-dir",
                str(path),
                "--max-tasks",
                str(args.max_tasks),
                "--program-budget-grid",
            ]
            commands.append(" ".join(command) + f" # {context} {split}")
            rows = run_arc_loader(command, repo, log_dir / f"rmt7_{split}_{context}.log")
            annotated: list[dict[str, Any]] = []
            for row in rows:
                row = dict(row)
                row["roadmap_item"] = "R-MT.7"
                row["split"] = split
                row["scorer_context"] = context
                row["scorer_coupling"] = "none_current_arc_loader_path"
                annotated.append(row)
                raw_rows.append(row)
            rows_by_split_context[(split, context)] = annotated

    mismatches: list[dict[str, Any]] = []
    split_summaries: list[dict[str, Any]] = []
    for split in args.splits:
        shared_rows = rows_by_split_context[(split, CONTEXTS[0])]
        single_rows = rows_by_split_context[(split, CONTEXTS[1])]
        split_summaries.append(summarize_split(split, shared_rows))
        mismatches.extend(
            compare_maps(
                split,
                "transform",
                keyed_rows(shared_rows, "crlp.arc_transform.summary.v1", ["task_instance"]),
                keyed_rows(single_rows, "crlp.arc_transform.summary.v1", ["task_instance"]),
                TRANSFORM_FIELDS,
            )
        )
        mismatches.extend(
            compare_maps(
                split,
                "loader",
                keyed_rows(shared_rows, "crlp.arc_loader.summary.v1", ["task"]),
                keyed_rows(single_rows, "crlp.arc_loader.summary.v1", ["task"]),
                LOADER_FIELDS,
            )
        )
        mismatches.extend(
            compare_maps(
                split,
                "proposer",
                keyed_rows(shared_rows, "crlp.program_proposer.summary.v1", ["strategy", "budget"]),
                keyed_rows(single_rows, "crlp.program_proposer.summary.v1", ["strategy", "budget"]),
                PROPOSER_FIELDS,
            )
        )

    raw_path = output_dir / "rmt7_arc_gate_invariance.raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    (output_dir / "rmt7_arc_gate_invariance.commands.txt").write_text(
        "\n".join(commands) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "rmt7_arc_gate_invariance_summary.csv", split_summaries)
    write_csv(output_dir / "rmt7_arc_gate_invariance_mismatches.csv", mismatches)
    write_report(
        output_dir / "rmt7_arc_gate_invariance.md",
        task_dir,
        args.max_tasks,
        split_summaries,
        mismatches,
    )
    if mismatches:
        print(f"found {len(mismatches)} mismatches")
        return 1
    print(f"wrote {output_dir / 'rmt7_arc_gate_invariance.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
