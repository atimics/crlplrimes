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


TOP_TASK_DELTA_LIMIT = 20


def current_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def current_git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


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


def empty_hypothesis_summary() -> dict[str, Any]:
    return {
        "tasks": 0,
        "proposals_attempted": 0,
        "proposals_retained": 0,
        "proposals_generated": 0,
        "proposals_train_consistent_pre_cap": 0,
        "proposals_train_consistent": 0,
        "proposals_duplicate_existing": 0,
        "proposals_train_rejected": 0,
        "proposals_heap_dropped": 0,
        "proposals_heap_replaced": 0,
        "proposals_selected": 0,
        "proposals_selected_by_coverage": 0,
        "proposals_selected_by_score_fallback": 0,
        "proposal_cap_hit_tasks": 0,
        "coverage_fallback_tasks": 0,
        "coverage_events": 0,
        "templates": {},
    }


def add_hypothesis_selection(summary: dict[str, Any], row: dict[str, Any]) -> None:
    summary["tasks"] += 1
    for key in (
        "proposals_attempted",
        "proposals_retained",
        "proposals_generated",
        "proposals_train_consistent_pre_cap",
        "proposals_train_consistent",
        "proposals_duplicate_existing",
        "proposals_train_rejected",
        "proposals_heap_dropped",
        "proposals_heap_replaced",
        "proposals_selected",
        "proposals_selected_by_coverage",
        "proposals_selected_by_score_fallback",
        "coverage_events",
    ):
        summary[key] += int(row.get(key, 0) or 0)
    if row.get("proposal_cap_hit"):
        summary["proposal_cap_hit_tasks"] += 1
    if row.get("coverage_fallback"):
        summary["coverage_fallback_tasks"] += 1
    templates = summary["templates"]
    for template in row.get("templates", []) or []:
        name = str(template.get("name", "unknown"))
        bucket = templates.setdefault(
            name,
            {
                "attempted": 0,
                "generated": 0,
                "retained": 0,
                "train_consistent_pre_cap": 0,
                "train_consistent_retained": 0,
                "heap_dropped": 0,
                "selected": 0,
            },
        )
        bucket["attempted"] += int(template.get("attempted", 0) or 0)
        bucket["generated"] += int(template.get("generated", 0) or 0)
        bucket["retained"] += int(template.get("retained", template.get("generated", 0)) or 0)
        bucket["train_consistent_pre_cap"] += int(
            template.get("train_consistent_pre_cap", 0) or 0
        )
        bucket["train_consistent_retained"] += int(
            template.get("train_consistent_retained", 0) or 0
        )
        bucket["heap_dropped"] += int(template.get("heap_dropped", 0) or 0)
        bucket["selected"] += int(template.get("selected", 0) or 0)


def run_loader(
    arc_agi: Path,
    task_dir: Path,
    max_tasks: int,
    induction: bool,
    program_train_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
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
    hypothesis_summary = empty_hypothesis_summary()
    hypothesis_tasks: dict[str, dict[str, Any]] = {}
    for row in parse_json_rows(result.stdout):
        if row.get("schema") == "crlp.arc_loader.summary.v1":
            summary = row
        elif row.get("schema") == "crlp.arc_transform.summary.v1":
            task_id = str(row.get("task_instance", ""))
            if task_id:
                tasks[task_id] = row
        elif row.get("schema") == "crlp.arc_hypothesis.selection.v1":
            task_id = str(row.get("task_instance", ""))
            if task_id:
                hypothesis_tasks[task_id] = row
            add_hypothesis_selection(hypothesis_summary, row)
    if summary is not None:
        return summary, tasks, hypothesis_summary, hypothesis_tasks
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


def task_cell_count(row: dict[str, Any]) -> int:
    return int(row.get("test_cells", 0) or 0)


def task_correct_cells(row: dict[str, Any]) -> int:
    return int(row.get("test_correct_cells", 0) or 0)


def classify_induction_task(row: dict[str, Any], hypothesis: dict[str, Any] | None) -> str:
    if not row:
        return "missing_transform_row"
    if bool(row.get("certified", False)):
        return "certified"
    if task_cell_count(row) == 0:
        return "no_test_labels"

    if hypothesis is not None:
        attempted = int(hypothesis.get("proposals_attempted", 0) or 0)
        generated = int(hypothesis.get("proposals_generated", 0) or 0)
        train_consistent = int(hypothesis.get("proposals_train_consistent", 0) or 0)
        selected = int(hypothesis.get("proposals_selected", 0) or 0)
        if attempted == 0:
            return "no_induction_attempted"
        if generated == 0:
            return "no_induction_generated"
        if train_consistent == 0:
            return "no_train_consistent_induction"
        if selected == 0:
            return "train_consistent_induction_not_selected"
        if not bool(row.get("unique_test_prediction", False)):
            return "selected_prediction_conflict"
        if task_correct_cells(row) > 0:
            return "selected_partial_test_match"
        return "selected_test_mismatch"

    if int(row.get("train_consistent_rules", 0) or 0) == 0:
        return "no_train_consistent_rule"
    if not bool(row.get("unique_test_prediction", False)):
        return "train_consistent_prediction_conflict"
    if task_correct_cells(row) > 0:
        return "partial_existing_rule_match"
    return "existing_rule_test_mismatch"


def task_delta_rows(
    baseline: dict[str, dict[str, Any]],
    induction: dict[str, dict[str, Any]],
    hypothesis: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id in sorted(set(baseline) | set(induction)):
        base = baseline.get(task_id, {})
        induced = induction.get(task_id, {})
        base_cells = task_correct_cells(base)
        induced_cells = task_correct_cells(induced)
        cell_delta = induced_cells - base_cells

        bucket = classify_induction_task(induced, hypothesis.get(task_id))
        rows.append(
            {
                "task_id": task_id,
                "baseline_cells": base_cells,
                "induction_cells": induced_cells,
                "cell_delta": cell_delta,
                "test_cells": task_cell_count(induced or base),
                "bucket": bucket,
                "baseline_certified": bool(base.get("certified", False)),
                "induction_certified": bool(induced.get("certified", False)),
                "baseline_train_consistent_rules": int(
                    base.get("train_consistent_rules", 0) or 0
                ),
                "induction_train_consistent_rules": int(
                    induced.get("train_consistent_rules", 0) or 0
                ),
                "induction_proposals_attempted": int(
                    hypothesis.get(task_id, {}).get("proposals_attempted", 0) or 0
                ),
                "induction_proposals_generated": int(
                    hypothesis.get(task_id, {}).get("proposals_generated", 0) or 0
                ),
                "induction_proposals_train_consistent": int(
                    hypothesis.get(task_id, {}).get("proposals_train_consistent", 0) or 0
                ),
                "induction_proposals_selected": int(
                    hypothesis.get(task_id, {}).get("proposals_selected", 0) or 0
                ),
            }
        )
    return rows


def task_delta_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    taxonomy: dict[str, int] = {}
    top_improvements: list[dict[str, Any]] = []
    top_regressions: list[dict[str, Any]] = []
    improved_tasks = 0
    worsened_tasks = 0
    unchanged_tasks = 0
    certified_tasks = 0
    total_cell_delta = 0
    total_cells = 0

    for row in rows:
        cell_delta = int(row["cell_delta"])
        total_cell_delta += cell_delta
        total_cells += int(row["test_cells"])
        if cell_delta > 0:
            improved_tasks += 1
            top_improvements.append(row)
        elif cell_delta < 0:
            worsened_tasks += 1
            top_regressions.append(row)
        else:
            unchanged_tasks += 1

        bucket = str(row["bucket"])
        taxonomy[bucket] = taxonomy.get(bucket, 0) + 1
        if bucket == "certified":
            certified_tasks += 1

    top_improvements.sort(
        key=lambda row: (int(row["cell_delta"]), int(row["induction_cells"]), row["task_id"]),
        reverse=True,
    )
    top_regressions.sort(
        key=lambda row: (int(row["cell_delta"]), int(row["baseline_cells"]), row["task_id"])
    )
    return {
        "tasks": len(rows),
        "certified_tasks": certified_tasks,
        "improved_cell_tasks": improved_tasks,
        "worsened_cell_tasks": worsened_tasks,
        "unchanged_cell_tasks": unchanged_tasks,
        "cell_delta_total": total_cell_delta,
        "test_cells": total_cells,
        "failure_taxonomy": dict(sorted(taxonomy.items())),
        "top_cell_improvements": top_improvements[:TOP_TASK_DELTA_LIMIT],
        "top_cell_regressions": top_regressions[:TOP_TASK_DELTA_LIMIT],
    }


def format_task_delta(row: dict[str, Any]) -> str:
    marker = "+" if int(row["cell_delta"]) >= 0 else ""
    return (
        f"`{row['task_id']}` ({marker}{row['cell_delta']} cells, "
        f"{row['baseline_cells']} -> {row['induction_cells']}, {row['bucket']})"
    )


def write_task_diagnostics(path: Path, rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for split in ("train", "eval"):
            for row in rows_by_split[split]:
                output = {"split": split, **row}
                handle.write(json.dumps(output, sort_keys=True, separators=(",", ":")) + "\n")


def write_report(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ARC Induction Phase 4",
        "",
        f"Git commit: `{row['git_commit']}`",
        f"Git dirty: `{str(row['git_dirty']).lower()}`",
        f"Task dir: `{row['task_dir']}`",
        f"Program train dir: `{row['program_train_dir']}`",
        f"Task diagnostics: `{row['task_diagnostics_path']}`",
        f"Max tasks per split: `{row['max_tasks']}`",
        f"Grammar version: `{row['grammar_version']}`",
        "",
        "| Split | Loaded | Parse Errors | Baseline Certified | Induction Certified | Delta | Added Tasks | Regressions |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for split in ("train", "eval"):
        base = int(row[f"{split}_baseline_certified"])
        induced = int(row[f"{split}_induction_certified"])
        added = ", ".join(row[f"{split}_added_tasks"]) or "-"
        regressions = ", ".join(row[f"{split}_regressions"]) or "-"
        loaded = int(row[f"{split}_baseline_loaded"])
        parse_errors = int(row[f"{split}_baseline_parse_errors"]) + int(
            row[f"{split}_induction_parse_errors"]
        )
        lines.append(
            f"| {split} | {loaded} | {parse_errors} | {base} | {induced} | {induced - base:+d} | {added} | {regressions} |"
        )
    lines.extend(["", "## Hypothesis Selection", ""])
    for split in ("train", "eval"):
        summary = row[f"{split}_hypothesis_selection"]
        lines.extend(
            [
                f"### {split}",
                "",
                f"- tasks with induction telemetry: `{summary['tasks']}`",
                f"- proposals: attempted `{summary['proposals_attempted']}`, retained `{summary['proposals_retained']}`, train-consistent before cap `{summary['proposals_train_consistent_pre_cap']}`, train-consistent retained `{summary['proposals_train_consistent']}`, selected `{summary['proposals_selected']}`",
                f"- heap pressure: dropped `{summary['proposals_heap_dropped']}`, replaced `{summary['proposals_heap_replaced']}`",
                f"- selected by coverage `{summary['proposals_selected_by_coverage']}`, score fallback `{summary['proposals_selected_by_score_fallback']}`",
                f"- proposal-cap-hit tasks `{summary['proposal_cap_hit_tasks']}`, coverage-fallback tasks `{summary['coverage_fallback_tasks']}`",
                "",
                "| Template | Attempted | Retained | Train-consistent before cap | Train-consistent retained | Heap dropped | Selected |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        templates = summary.get("templates", {})
        if templates:
            for name in sorted(templates):
                bucket = templates[name]
                lines.append(
                    f"| {name} | {int(bucket.get('attempted', 0))} | {int(bucket.get('retained', bucket.get('generated', 0)))} | {int(bucket.get('train_consistent_pre_cap', 0))} | {int(bucket.get('train_consistent_retained', 0))} | {int(bucket.get('heap_dropped', 0))} | {int(bucket.get('selected', 0))} |"
                )
        else:
            lines.append("| - | 0 | 0 | 0 | 0 | 0 | 0 |")
        lines.append("")
    lines.extend(["", "## Task-Level Diagnostics", ""])
    for split in ("train", "eval"):
        summary = row[f"{split}_task_delta_summary"]
        lines.extend(
            [
                f"### {split}",
                "",
                f"- tasks: `{summary['tasks']}`",
                f"- certified tasks: `{summary['certified_tasks']}`",
                f"- cell delta: `{summary['cell_delta_total']}` over `{summary['test_cells']}` labeled test cells",
                f"- improved / unchanged / worsened tasks: `{summary['improved_cell_tasks']}` / `{summary['unchanged_cell_tasks']}` / `{summary['worsened_cell_tasks']}`",
                "",
                "| Bucket | Tasks |",
                "|---|---:|",
            ]
        )
        for bucket, count in summary["failure_taxonomy"].items():
            lines.append(f"| `{bucket}` | {int(count)} |")
        improvements = summary.get("top_cell_improvements", [])
        regressions = summary.get("top_cell_regressions", [])
        lines.extend(["", "Top cell improvements:"])
        if improvements:
            for item in improvements[:10]:
                lines.append(f"- {format_task_delta(item)}")
        else:
            lines.append("- -")
        lines.extend(["", "Top cell regressions:"])
        if regressions:
            for item in regressions[:10]:
                lines.append(f"- {format_task_delta(item)}")
        else:
            lines.append("- -")
        lines.append("")
    lines.extend(
        [
            f"Parity holds: `{str(row['parity_holds']).lower()}`",
            f"Eval gain holds: `{str(row['eval_gain_holds']).lower()}`",
            f"Load/parse clean: `{str(row['load_parse_clean']).lower()}`",
            f"Phase 4 holds: `{str(row['phase4_holds']).lower()}`",
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
    parser.add_argument("--task-diagnostics-path", type=Path)
    args = parser.parse_args(argv)

    if not args.task_dir:
        raise RuntimeError("--task-dir is required")
    splits = split_dirs(args.task_dir)
    baseline: dict[str, dict[str, Any]] = {}
    induction: dict[str, dict[str, Any]] = {}
    baseline_tasks: dict[str, dict[str, dict[str, Any]]] = {}
    induction_tasks: dict[str, dict[str, dict[str, Any]]] = {}
    induction_hypothesis: dict[str, dict[str, Any]] = {}
    induction_hypothesis_tasks: dict[str, dict[str, dict[str, Any]]] = {}
    for split, path in splits.items():
        baseline[split], baseline_tasks[split], _, _ = run_loader(
            args.arc_agi, path, args.max_tasks, False
        )
        (
            induction[split],
            induction_tasks[split],
            induction_hypothesis[split],
            induction_hypothesis_tasks[split],
        ) = run_loader(args.arc_agi, path, args.max_tasks, True, splits["train"])

    train_added, train_regressions = compare_tasks(
        baseline_tasks["train"], induction_tasks["train"]
    )
    eval_added, eval_regressions = compare_tasks(
        baseline_tasks["eval"], induction_tasks["eval"]
    )
    train_task_delta_rows = task_delta_rows(
        baseline_tasks["train"],
        induction_tasks["train"],
        induction_hypothesis_tasks["train"],
    )
    eval_task_delta_rows = task_delta_rows(
        baseline_tasks["eval"],
        induction_tasks["eval"],
        induction_hypothesis_tasks["eval"],
    )
    train_task_delta_summary = task_delta_summary(train_task_delta_rows)
    eval_task_delta_summary = task_delta_summary(eval_task_delta_rows)

    train_holds = int(induction["train"].get("certified_tasks", 0)) >= int(
        baseline["train"].get("certified_tasks", 0)
    )
    eval_holds = int(induction["eval"].get("certified_tasks", 0)) >= int(
        baseline["eval"].get("certified_tasks", 0)
    )
    train_delta = int(induction["train"].get("certified_tasks", 0)) - int(
        baseline["train"].get("certified_tasks", 0)
    )
    eval_delta = int(induction["eval"].get("certified_tasks", 0)) - int(
        baseline["eval"].get("certified_tasks", 0)
    )
    eval_gain_holds = eval_delta >= 1
    load_parse_clean = (
        int(baseline["train"].get("loaded_tasks", 0))
        == int(induction["train"].get("loaded_tasks", 0))
        and int(baseline["eval"].get("loaded_tasks", 0))
        == int(induction["eval"].get("loaded_tasks", 0))
        and int(baseline["train"].get("parse_errors", 0) or 0) == 0
        and int(induction["train"].get("parse_errors", 0) or 0) == 0
        and int(baseline["eval"].get("parse_errors", 0) or 0) == 0
        and int(induction["eval"].get("parse_errors", 0) or 0) == 0
    )
    parity_holds = train_holds and eval_holds
    phase4_holds = parity_holds and eval_gain_holds and load_parse_clean
    report_path = args.report_path or (
        Path("results") / f"{date.today().isoformat()}-arc-induction-phase4.md"
    )
    task_diagnostics_path = args.task_diagnostics_path or report_path.with_name(
        f"{report_path.stem}.tasks.jsonl"
    )
    row = {
        "schema": "crlp.arc_induction.parity.v1",
        "report_version": 1,
        "task": "arc_agi",
        "git_commit": current_git_commit(),
        "git_dirty": current_git_dirty(),
        "task_dir": str(args.task_dir),
        "program_train_dir": str(splits["train"]),
        "task_diagnostics_path": str(task_diagnostics_path),
        "max_tasks": args.max_tasks,
        "grammar_version": induction["eval"].get(
            "grammar_version", baseline["eval"].get("grammar_version", 0)
        ),
        "train_baseline_loaded": baseline["train"].get("loaded_tasks", 0),
        "train_induction_loaded": induction["train"].get("loaded_tasks", 0),
        "train_baseline_parse_errors": baseline["train"].get("parse_errors", 0),
        "train_induction_parse_errors": induction["train"].get("parse_errors", 0),
        "train_baseline_certified": baseline["train"].get("certified_tasks", 0),
        "train_induction_certified": induction["train"].get("certified_tasks", 0),
        "train_certified_delta": train_delta,
        "train_baseline_cells": baseline["train"].get("test_correct_cells", 0),
        "train_induction_cells": induction["train"].get("test_correct_cells", 0),
        "eval_baseline_loaded": baseline["eval"].get("loaded_tasks", 0),
        "eval_induction_loaded": induction["eval"].get("loaded_tasks", 0),
        "eval_baseline_parse_errors": baseline["eval"].get("parse_errors", 0),
        "eval_induction_parse_errors": induction["eval"].get("parse_errors", 0),
        "eval_baseline_certified": baseline["eval"].get("certified_tasks", 0),
        "eval_induction_certified": induction["eval"].get("certified_tasks", 0),
        "eval_certified_delta": eval_delta,
        "eval_baseline_cells": baseline["eval"].get("test_correct_cells", 0),
        "eval_induction_cells": induction["eval"].get("test_correct_cells", 0),
        "train_added_tasks": train_added,
        "train_regressions": train_regressions,
        "eval_added_tasks": eval_added,
        "eval_regressions": eval_regressions,
        "train_hypothesis_selection": induction_hypothesis["train"],
        "eval_hypothesis_selection": induction_hypothesis["eval"],
        "train_task_delta_summary": train_task_delta_summary,
        "eval_task_delta_summary": eval_task_delta_summary,
        "parity_holds": parity_holds,
        "eval_gain_holds": eval_gain_holds,
        "load_parse_clean": load_parse_clean,
        "phase4_holds": phase4_holds,
    }
    write_task_diagnostics(
        task_diagnostics_path,
        {
            "train": train_task_delta_rows,
            "eval": eval_task_delta_rows,
        },
    )
    write_report(report_path, row)
    row["report_path"] = str(report_path)
    print(json.dumps(row, sort_keys=True, separators=(",", ":")))
    return 0 if row["phase4_holds"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
