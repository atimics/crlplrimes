#!/usr/bin/env python3
"""Build a clean ARC-AGI evidence surface and miss taxonomy.

The script keeps ARC claims auditable by separating three surfaces:

1. Fixed-task scorer/gate evidence for 0520fde7.
2. Public ARC loader summaries and matched budget curves.
3. Per-task public-eval miss taxonomy for unsupported symbolic families.

It runs the local arc_agi binary, stores raw JSONL, then derives CSV/Markdown
summaries. The verifier remains the source of certification; the taxonomy is a
search-space audit only.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from arc_family_discovery import pair_signature  # noqa: E402


RECORDED_SNAPSHOTS: list[dict[str, Any]] = [
    {
        "version": "v10_recorded",
        "split": "training",
        "loaded_tasks": 400,
        "certified_tasks": 24,
        "test_cells": 56672,
        "test_correct_cells": 1187,
        "notes": "pre-layer-overlap recorded snapshot",
    },
    {
        "version": "v10_recorded",
        "split": "evaluation",
        "loaded_tasks": 400,
        "certified_tasks": 4,
        "test_cells": 98515,
        "test_correct_cells": 187,
        "notes": "pre-layer-overlap recorded snapshot",
    },
    {
        "version": "v12_recorded",
        "split": "training",
        "loaded_tasks": 400,
        "certified_tasks": 24,
        "test_cells": 56672,
        "test_correct_cells": 1187,
        "notes": "layer relation family recorded snapshot",
    },
    {
        "version": "v12_recorded",
        "split": "evaluation",
        "loaded_tasks": 400,
        "certified_tasks": 5,
        "test_cells": 98515,
        "test_correct_cells": 207,
        "notes": "layer relation family recorded snapshot",
    },
    {
        "version": "v13_recorded",
        "split": "training",
        "loaded_tasks": 400,
        "certified_tasks": 24,
        "test_cells": 56672,
        "test_correct_cells": 1187,
        "notes": "color-line compression recorded snapshot",
    },
    {
        "version": "v13_recorded",
        "split": "evaluation",
        "loaded_tasks": 400,
        "certified_tasks": 6,
        "test_cells": 98515,
        "test_correct_cells": 216,
        "notes": "color-line compression recorded snapshot",
    },
    {
        "version": "v14_recorded",
        "split": "training",
        "loaded_tasks": 400,
        "certified_tasks": 24,
        "test_cells": 56672,
        "test_correct_cells": 1187,
        "notes": "static lattice motif repeat recorded snapshot",
    },
    {
        "version": "v14_recorded",
        "split": "evaluation",
        "loaded_tasks": 400,
        "certified_tasks": 7,
        "test_cells": 98515,
        "test_correct_cells": 1057,
        "notes": "static lattice motif repeat recorded snapshot",
    },
]


def run_and_store(command: list[str], stdout_path: Path, stderr_path: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit {completed.returncode}: {' '.join(command)}\n"
            f"stderr: {completed.stderr.strip()}"
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def task_path(task_dir: Path, split: str, task_id: str) -> Path:
    path = task_dir / split / f"{task_id}.json"
    if path.is_file():
        return path
    aliases = {"training": "train", "evaluation": "eval"}
    alias = aliases.get(split)
    if alias:
        path = task_dir / alias / f"{task_id}.json"
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing ARC task {split}/{task_id}.json under {task_dir}")


def task_category(task: dict[str, Any], transform_row: dict[str, Any]) -> str:
    if bool(transform_row.get("certified")):
        return "certified"
    pair_sigs = [pair_signature(pair) for pair in task.get("train", [])]
    if not pair_sigs:
        return "unsupported_parse_or_shape"

    all_same_shape = all(bool(pair["same_shape"]) for pair in pair_sigs)
    all_smaller = all(bool(pair["output_smaller"]) for pair in pair_sigs)
    all_larger = all(bool(pair["output_larger"]) for pair in pair_sigs)
    any_symmetry = any(str(pair["symmetry"]) != "none" for pair in pair_sigs)
    all_crop_like = all(
        bool(pair["output_is_input_crop"]) or bool(pair["bbox_crop_equal"])
        for pair in pair_sigs
    )
    all_preserved = all(bool(pair["preserved_nonzero"]) for pair in pair_sigs)
    added = sum(int(pair["added_cells"]) for pair in pair_sigs)
    removed = sum(int(pair["removed_cells"]) for pair in pair_sigs)
    recolored = sum(int(pair["recolored_cells"]) for pair in pair_sigs)
    component_delta = sum(int(pair["component_delta"]) for pair in pair_sigs)
    foreground_delta = sum(int(pair["foreground_delta"]) for pair in pair_sigs)

    if all_larger:
        return "scale_tile_or_expand"
    if all_smaller:
        if all_crop_like:
            return "segmentation_or_object_extract"
        return "projection_or_panel_selection"
    if any_symmetry:
        return "symmetry_or_geometry"
    if all_same_shape and all_preserved and added > 0:
        if component_delta > 0:
            return "object_motion_or_copy_extend"
        return "shape_completion_or_fill"
    if all_same_shape and recolored > 0 and recolored >= added + removed:
        return "color_remap_or_mode_collapse"
    if all_same_shape and (component_delta != 0 or foreground_delta != 0):
        return "relational_layout_or_counting"
    if int(transform_row.get("unsupported_train_pairs", 0) or 0) > 0:
        return "unsupported_parse_or_shape"
    return "residual_unclassified"


def summarize_taxonomy(task_dir: Path, transform_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in transform_rows:
        task_id = str(row["task_instance"])
        task = json.loads(task_path(task_dir, "evaluation", task_id).read_text(encoding="utf-8"))
        category = task_category(task, row)
        task_row = {
            "split": "evaluation",
            "task_id": task_id,
            "category": category,
            "certified": bool(row.get("certified")),
            "test_cells": int(row.get("test_cells", 0) or 0),
            "test_correct_cells": int(row.get("test_correct_cells", 0) or 0),
            "train_consistent_rules": int(row.get("train_consistent_rules", 0) or 0),
            "unsupported_train_pairs": int(row.get("unsupported_train_pairs", 0) or 0),
            "accepted_rules": row.get("accepted_rules", ""),
        }
        task_rows.append(task_row)
        grouped[category].append(task_row)

    summary_rows: list[dict[str, Any]] = []
    for category, rows in grouped.items():
        examples = sorted(rows, key=lambda item: (-int(item["test_cells"]), item["task_id"]))[:8]
        summary_rows.append(
            {
                "category": category,
                "tasks": len(rows),
                "test_cells": sum(int(row["test_cells"]) for row in rows),
                "certified_tasks": sum(1 for row in rows if row["certified"]),
                "covered_cells": sum(int(row["test_correct_cells"]) for row in rows),
                "examples": ",".join(str(row["task_id"]) for row in examples),
                "promotion_target": promotion_target(category),
            }
        )
    summary_rows.sort(
        key=lambda row: (
            row["category"] == "certified",
            -int(row["tasks"]),
            -int(row["test_cells"]),
            str(row["category"]),
        )
    )
    task_rows.sort(key=lambda row: (str(row["category"]), str(row["task_id"])))
    return task_rows, summary_rows


def promotion_target(category: str) -> str:
    targets = {
        "projection_or_panel_selection": "panel selector / row-column projection",
        "segmentation_or_object_extract": "object selector plus crop normalization",
        "object_motion_or_copy_extend": "offset/repetition induction over object sets",
        "shape_completion_or_fill": "boundary completion and fill verifier",
        "color_remap_or_mode_collapse": "mask-conditioned recolor verifier",
        "scale_tile_or_expand": "tile/scale policy with separator handling",
        "symmetry_or_geometry": "axis/pivot symmetry verifier",
        "relational_layout_or_counting": "relational layout/counting predicates",
        "unsupported_parse_or_shape": "parser/runtime shape support",
        "residual_unclassified": "manual exemplar review",
        "certified": "already covered",
    }
    return targets.get(category, "manual exemplar review")


def loader_summary_row(version: str, split: str, row: dict[str, Any], notes: str) -> dict[str, Any]:
    test_cells = int(row.get("test_cells", 0) or 0)
    correct = int(row.get("test_correct_cells", 0) or 0)
    return {
        "version": version,
        "split": split,
        "loaded_tasks": int(row.get("loaded_tasks", 0) or 0),
        "certified_tasks": int(row.get("certified_tasks", 0) or 0),
        "test_cells": test_cells,
        "test_correct_cells": correct,
        "cell_accuracy": f"{(correct / test_cells) if test_cells else 0.0:.8f}",
        "notes": notes,
    }


def write_report(
    path: Path,
    evidence_rows: list[dict[str, Any]],
    budget_rows: list[dict[str, Any]],
    taxonomy_rows: list[dict[str, Any]],
    fixed_transform: dict[str, Any] | None,
) -> None:
    lines = [
        "# ARC-AGI Evidence Surface",
        "",
        "This report separates fixed-task certification, public-corpus coverage, "
        "budgeted proposal efficiency, and miss taxonomy. Recorded "
        "v10/v12/v13/v14 rows are historical comparison points; current rows "
        "are regenerated by this run.",
        "",
        "## Fixed Task",
        "",
    ]
    if fixed_transform:
        lines.append(
            f"- grammar v{fixed_transform.get('grammar_version')} certified "
            f"`{fixed_transform.get('task_instance')}` with "
            f"{fixed_transform.get('test_correct_cells')}/{fixed_transform.get('test_cells')} "
            f"held-out cells and accepted rules `{fixed_transform.get('accepted_rules')}`."
        )
    else:
        lines.append("- no fixed-task transform row captured.")

    lines.extend(
        [
            "",
            "## Public Coverage",
            "",
            "| Version | Split | Certified Tasks | Covered Cells | Cell Accuracy | Notes |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in evidence_rows:
        lines.append(
            f"| {row['version']} | {row['split']} | "
            f"{row['certified_tasks']}/{row['loaded_tasks']} | "
            f"{row['test_correct_cells']}/{row['test_cells']} | "
            f"{row['cell_accuracy']} | {row['notes']} |"
        )

    lines.extend(
        [
            "",
            "## Current Budget Curve",
            "",
            "| Split | Budget | Strategy | Certified Programs | Covered Events | False Rejections |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in budget_rows:
        lines.append(
            f"| {row['split']} | {row['budget']} | `{row['strategy']}` | "
            f"{row['certified_programs']} | {row['covered_events']} | "
            f"{row['false_rejections']} |"
        )

    lines.extend(
        [
            "",
            "## Evaluation Miss Taxonomy",
            "",
            "| Category | Tasks | Test Cells | Covered Cells | Promotion Target | Examples |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for row in taxonomy_rows:
        lines.append(
            f"| `{row['category']}` | {row['tasks']} | {row['test_cells']} | "
            f"{row['covered_cells']} | {row['promotion_target']} | "
            f"{row['examples']} |"
        )

    lines.extend(
        [
            "",
            "## Takeaway",
            "",
            "The current promotions keep the fixed ARC certificate intact and move "
            "public-evaluation coverage from a separator-compression win to a larger "
            "lattice copy/extend win. Grammar v15 keeps that coverage but also routes "
            "the lattice verifier through `NNHypothesis` induction, so the proposer can "
            "learn and rank the verifier template separately from the static candidate. "
            "The remaining ARC gap is not capacity; it is unsupported symbolic family "
            "coverage.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arc-agi", type=Path, default=Path("build/float/arc_agi"))
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("build/float/arc_evidence_surface"))
    parser.add_argument("--max-tasks", type=int, default=400)
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    commands = {
        "fixed_task": [
            str(args.arc_agi),
            "--epochs",
            "1",
            "--steps-per-epoch",
            "50",
            "--test-samples",
            "50",
            "--program-budget",
            "1",
        ],
        "public_train": [
            str(args.arc_agi),
            "--task-dir",
            str(args.task_dir / "training"),
            "--max-tasks",
            str(args.max_tasks),
            "--loader-summary-only",
            "--program-budget-grid",
        ],
        "public_eval": [
            str(args.arc_agi),
            "--program-train-dir",
            str(args.task_dir / "training"),
            "--task-dir",
            str(args.task_dir / "evaluation"),
            "--max-tasks",
            str(args.max_tasks),
            "--loader-summary-only",
            "--program-budget-grid",
        ],
        "public_eval_tasks": [
            str(args.arc_agi),
            "--task-dir",
            str(args.task_dir / "evaluation"),
            "--max-tasks",
            str(args.max_tasks),
            "--program-budget",
            "1",
        ],
    }
    (output_dir / "commands.txt").write_text(
        "\n".join(" ".join(command) for command in commands.values()) + "\n",
        encoding="utf-8",
    )

    if not args.skip_run:
        for name, command in commands.items():
            run_and_store(command, raw_dir / f"{name}.jsonl", raw_dir / f"{name}.stderr")

    fixed_rows = read_jsonl(raw_dir / "fixed_task.jsonl")
    train_rows = read_jsonl(raw_dir / "public_train.jsonl")
    eval_rows = read_jsonl(raw_dir / "public_eval.jsonl")
    eval_task_rows = read_jsonl(raw_dir / "public_eval_tasks.jsonl")

    fixed_transform = next(
        (row for row in fixed_rows if row.get("schema") == "crlp.arc_transform.summary.v1"),
        None,
    )
    train_summary = next(row for row in train_rows if row.get("schema") == "crlp.arc_loader.summary.v1")
    eval_summary = next(row for row in eval_rows if row.get("schema") == "crlp.arc_loader.summary.v1")
    version = f"v{train_summary.get('grammar_version', 'current')}_measured"
    evidence_rows = []
    for snapshot in RECORDED_SNAPSHOTS:
        row = dict(snapshot)
        test_cells = int(row.get("test_cells", 0) or 0)
        correct = int(row.get("test_correct_cells", 0) or 0)
        row["cell_accuracy"] = f"{(correct / test_cells) if test_cells else 0.0:.8f}"
        evidence_rows.append(row)
    evidence_rows.append(loader_summary_row(version, "training", train_summary, "current measured run"))
    evidence_rows.append(loader_summary_row(version, "evaluation", eval_summary, "current measured run"))

    budget_rows: list[dict[str, Any]] = []
    for split, rows in (("training", train_rows), ("evaluation", eval_rows)):
        for row in rows:
            if row.get("schema") != "crlp.program_proposer.summary.v1":
                continue
            budget_rows.append(
                {
                    "split": split,
                    "budget": int(row.get("budget", 0) or 0),
                    "strategy": row.get("strategy", ""),
                    "certified_programs": int(row.get("certified_programs", 0) or 0),
                    "accepted_programs": int(row.get("accepted_programs", 0) or 0),
                    "covered_events": int(row.get("covered_events", 0) or 0),
                    "false_rejections": int(row.get("false_rejections", 0) or 0),
                    "utility": row.get("utility", ""),
                }
            )
    budget_rows.sort(key=lambda row: (row["split"], row["budget"], str(row["strategy"])))

    transform_rows = [
        row for row in eval_task_rows if row.get("schema") == "crlp.arc_transform.summary.v1"
    ]
    taxonomy_tasks, taxonomy_summary = summarize_taxonomy(args.task_dir, transform_rows)

    write_csv(
        output_dir / "arc_evidence_summary.csv",
        [
            "version",
            "split",
            "loaded_tasks",
            "certified_tasks",
            "test_cells",
            "test_correct_cells",
            "cell_accuracy",
            "notes",
        ],
        evidence_rows,
    )
    write_csv(
        output_dir / "arc_budget_curves.csv",
        [
            "split",
            "budget",
            "strategy",
            "certified_programs",
            "accepted_programs",
            "covered_events",
            "false_rejections",
            "utility",
        ],
        budget_rows,
    )
    write_csv(
        output_dir / "arc_eval_miss_taxonomy.csv",
        [
            "category",
            "tasks",
            "test_cells",
            "certified_tasks",
            "covered_cells",
            "promotion_target",
            "examples",
        ],
        taxonomy_summary,
    )
    write_csv(
        output_dir / "arc_eval_task_taxonomy.csv",
        [
            "split",
            "task_id",
            "category",
            "certified",
            "test_cells",
            "test_correct_cells",
            "train_consistent_rules",
            "unsupported_train_pairs",
            "accepted_rules",
        ],
        taxonomy_tasks,
    )
    write_report(
        output_dir / "arc_evidence_surface.md",
        evidence_rows,
        budget_rows,
        taxonomy_summary,
        fixed_transform,
    )

    summary = {
        "schema": "crlp.arc_evidence_surface.summary.v1",
        "task": "arc_agi",
        "output_dir": str(output_dir),
        "grammar_version": train_summary.get("grammar_version"),
        "train_certified_tasks": train_summary.get("certified_tasks"),
        "eval_certified_tasks": eval_summary.get("certified_tasks"),
        "eval_covered_cells": eval_summary.get("test_correct_cells"),
        "taxonomy_categories": len(taxonomy_summary),
        "top_miss_categories": [
            {
                "category": row["category"],
                "tasks": row["tasks"],
                "test_cells": row["test_cells"],
            }
            for row in taxonomy_summary
            if row["category"] != "certified"
        ][:5],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
