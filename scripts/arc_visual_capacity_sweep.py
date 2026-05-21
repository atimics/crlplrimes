#!/usr/bin/env python3
"""Sweep ARC visual ranker training settings and selection capacity knobs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


RANKER_SCHEMA = "crlp.neural_grammar_discovery.arc_visual_capacity_sweep.v1"
SELECTION_SCHEMA = "crlp.neural_grammar_discovery.arc_visual_selection_capacity.v1"
SUMMARY_SCHEMA = "crlp.neural_grammar_discovery.arc_visual_capacity_sweep_summary.v1"


def parse_csv_ints(raw: str) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def parse_csv_floats(raw: str) -> list[float]:
    values: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def visual_template_stats(row: dict[str, Any]) -> dict[str, Any]:
    for template in row.get("templates", []) or []:
        if isinstance(template, dict) and template.get("name") == "arc_sketch_program":
            return template
    return {}


def summarize_arc_log(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    loader = next(
        (
            row
            for row in reversed(rows)
            if row.get("schema") == "crlp.arc_loader.summary.v1"
        ),
        {},
    )
    selection_rows = [
        row for row in rows if row.get("schema") == "crlp.arc_hypothesis.selection.v1"
    ]
    visual_selected = 0
    visual_train_consistent = 0
    visual_tasks_selected = 0
    visual_tasks_train_consistent = 0
    for row in selection_rows:
        visual = visual_template_stats(row)
        selected = safe_int(visual.get("selected"))
        train_consistent = safe_int(visual.get("train_consistent_pre_cap"))
        visual_selected += selected
        visual_train_consistent += train_consistent
        if selected > 0:
            visual_tasks_selected += 1
        if train_consistent > 0:
            visual_tasks_train_consistent += 1
    task_count = len(selection_rows)
    return {
        "log_path": str(path),
        "loaded_tasks": safe_int(loader.get("loaded_tasks")),
        "certified_tasks": safe_int(loader.get("certified_tasks")),
        "test_exact_tasks": safe_int(loader.get("test_exact_tasks")),
        "cell_accuracy": safe_float(loader.get("cell_accuracy")),
        "selection_rows": task_count,
        "visual_selected": visual_selected,
        "visual_train_consistent": visual_train_consistent,
        "visual_task_selected_rate": (
            None if task_count == 0 else visual_tasks_selected / task_count
        ),
        "visual_task_train_consistent_rate": (
            None if task_count == 0 else visual_tasks_train_consistent / task_count
        ),
    }


def run_ranker_sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not args.runtime or not args.attempts_tsv:
        return rows
    if not args.runtime.is_file() or not args.attempts_tsv.is_file():
        return rows
    ranker_dir = args.output_dir / "ranker"
    ranker_dir.mkdir(parents=True, exist_ok=True)
    for epochs in parse_csv_ints(args.epochs):
        for learning_rate in parse_csv_floats(args.learning_rates):
            for max_per_task in parse_csv_ints(args.max_per_task):
                label = f"epochs{epochs}_lr{learning_rate:g}_top{max_per_task}"
                proposals = ranker_dir / f"{label}.proposals.jsonl"
                summary_path = ranker_dir / f"{label}.summary.jsonl"
                command = [
                    str(args.runtime),
                    "--arc-pipeline-attempts",
                    str(args.attempts_tsv),
                    "--arc-pipeline-proposals-output",
                    str(proposals),
                    "--arc-pipeline-summary-output",
                    str(summary_path),
                    "--arc-pipeline-max-per-task",
                    str(max_per_task),
                    "--epochs",
                    str(epochs),
                    "--learning-rate",
                    f"{learning_rate:g}",
                ]
                completed = subprocess.run(command, check=False)
                summary_rows = load_jsonl(summary_path)
                summary = summary_rows[0] if summary_rows else {}
                rows.append(
                    {
                        "schema": RANKER_SCHEMA,
                        "report_version": 1,
                        "label": label,
                        "status": completed.returncode,
                        "model_family": "sparse_token_pairwise_ranker",
                        "capacity_parameter": "epochs_learning_rate_only",
                        "epochs": epochs,
                        "learning_rate": learning_rate,
                        "max_per_task": max_per_task,
                        "summary_path": str(summary_path),
                        "proposals_path": str(proposals),
                        "target_task_count": safe_int(summary.get("target_task_count")),
                        "accepted_target_task_count": safe_int(
                            summary.get("accepted_target_task_count")
                        ),
                        "search_space_coverage_rate": safe_float(
                            summary.get("search_space_coverage_rate")
                        ),
                        "evaluated_tasks": safe_int(summary.get("evaluated_tasks")),
                        "top1_rate": safe_float(summary.get("top1_rate")),
                        "top3_rate": safe_float(summary.get("top3_rate")),
                        "retained_rate": safe_float(summary.get("retained_rate")),
                        "top1_all_task_rate": safe_float(
                            summary.get("top1_all_task_rate")
                        ),
                        "pairwise_last_epoch_violations": safe_int(
                            summary.get("pairwise_last_epoch_violations")
                        ),
                        "pairwise_training_updates": safe_int(
                            summary.get("pairwise_training_updates")
                        ),
                    }
                )
    return rows


def run_selection_sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not args.arc_agi or not args.public_arc_dir or not args.prior:
        return rows
    if not args.arc_agi.is_file() or not args.prior.is_file():
        return rows
    selection_dir = args.output_dir / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    for hidden in parse_csv_ints(args.hidden_sizes):
        for split_name, split_dir in (("public_train", "training"), ("public_eval", "evaluation")):
            label = f"{split_name}_hidden{hidden}"
            log_path = selection_dir / f"{label}.jsonl"
            command = [
                str(args.arc_agi),
                "--task-dir",
                str(args.public_arc_dir / split_dir),
                "--max-tasks",
                str(args.max_tasks),
                "--program-budget-grid",
                "--loader-summary-only",
                "--induction",
                "--min-induction-evidence",
                "0",
                "--arc-visual-only",
                "--hypothesis-library-prior",
                str(args.prior),
                "--hypothesis-prior-hidden",
                str(hidden),
            ]
            with log_path.open("w", encoding="utf-8") as fp:
                completed = subprocess.run(
                    command,
                    stdout=fp,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            summary = summarize_arc_log(log_path)
            rows.append(
                {
                    "schema": SELECTION_SCHEMA,
                    "report_version": 1,
                    "label": label,
                    "status": completed.returncode,
                    "scope": split_name,
                    "hidden": hidden,
                    "prior": str(args.prior),
                    **summary,
                }
            )
    return rows


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ranker = [row for row in rows if row.get("schema") == RANKER_SCHEMA]
    selection = [row for row in rows if row.get("schema") == SELECTION_SCHEMA]
    lines = ["# ARC Visual Capacity Sweep", ""]
    if ranker:
        lines.extend(
            [
                "## Ranker Sweep",
                "",
                "| Label | Top-1 | Retained | All-task top-1 | Coverage | Pairwise violations |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in ranker:
            lines.append(
                "| {label} | {top1:.3f} | {retained:.3f} | {all_top1:.3f} | {coverage:.3f} | {violations} |".format(
                    label=row.get("label", ""),
                    top1=safe_float(row.get("top1_rate")),
                    retained=safe_float(row.get("retained_rate")),
                    all_top1=safe_float(row.get("top1_all_task_rate")),
                    coverage=safe_float(row.get("search_space_coverage_rate")),
                    violations=safe_int(row.get("pairwise_last_epoch_violations")),
                )
            )
        lines.append("")
    if selection:
        lines.extend(
            [
                "## Visual-Only Selection Sweep",
                "",
                "| Scope | Hidden | Certified | Exact | Visual Selected | Visual Train-Consistent | Cell Accuracy |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in selection:
            lines.append(
                "| {scope} | {hidden} | {cert} | {exact} | {selected} | {tc} | {acc:.4f} |".format(
                    scope=row.get("scope", ""),
                    hidden=safe_int(row.get("hidden")),
                    cert=safe_int(row.get("certified_tasks")),
                    exact=safe_int(row.get("test_exact_tasks")),
                    selected=safe_int(row.get("visual_selected")),
                    tc=safe_int(row.get("visual_train_consistent")),
                    acc=safe_float(row.get("cell_accuracy")),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    ranker = [row for row in rows if row.get("schema") == RANKER_SCHEMA]
    selection = [row for row in rows if row.get("schema") == SELECTION_SCHEMA]
    best_ranker = max(
        ranker,
        key=lambda row: (
            safe_float(row.get("top1_all_task_rate")),
            safe_float(row.get("retained_rate")),
            -safe_int(row.get("pairwise_last_epoch_violations")),
        ),
        default={},
    )
    return {
        "schema": SUMMARY_SCHEMA,
        "report_version": 1,
        "row_count": len(rows),
        "ranker_rows": len(ranker),
        "selection_rows": len(selection),
        "best_ranker_label": best_ranker.get("label", ""),
        "best_ranker_top1_all_task_rate": safe_float(
            best_ranker.get("top1_all_task_rate")
        ),
        "best_ranker_search_space_coverage_rate": safe_float(
            best_ranker.get("search_space_coverage_rate")
        ),
        "interpretation": (
            "ranker_capacity_saturated_on_covered_tasks"
            if best_ranker and safe_float(best_ranker.get("top1_rate")) >= 1.0
            else "ranker_capacity_not_saturated"
        ),
        "output_dir": str(args.output_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--attempts-tsv", type=Path)
    parser.add_argument("--arc-agi", type=Path)
    parser.add_argument("--public-arc-dir", type=Path)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", default="16,64,128")
    parser.add_argument("--learning-rates", default="0.08")
    parser.add_argument("--max-per-task", default="8")
    parser.add_argument("--hidden-sizes", default="0,24,64")
    parser.add_argument("--max-tasks", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [*run_ranker_sweep(args), *run_selection_sweep(args)]
    rows.append(build_summary(rows, args))
    write_jsonl(args.output, rows)
    write_markdown(args.markdown, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
