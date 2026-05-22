#!/usr/bin/env python3
"""Iteratively replay verified ARC visual-program wins into visual-only coverage."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

from crlp.io import read_jsonl_lenient as load_jsonl
from crlp.io import safe_float, safe_int, write_jsonl

ITERATION_SCHEMA = "crlp.neural_grammar_discovery.arc_visual_active_curriculum_iteration.v1"
SUMMARY_SCHEMA = "crlp.neural_grammar_discovery.arc_visual_active_curriculum_summary.v1"


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
    evidence_tasks: list[str] = []
    for row in rows:
        if (
            row.get("schema") == "crlp.hypothesis.evidence.v1"
            and row.get("template_name") == "arc_sketch_program"
            and row.get("certificate_status") == "certified"
        ):
            evidence_tasks.append(str(row.get("task_instance", "")))
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
        "visual_tasks_selected": visual_tasks_selected,
        "visual_tasks_train_consistent": visual_tasks_train_consistent,
        "visual_task_selected_rate": (
            None if task_count == 0 else visual_tasks_selected / task_count
        ),
        "visual_task_train_consistent_rate": (
            None if task_count == 0 else visual_tasks_train_consistent / task_count
        ),
        "certified_visual_tasks": sorted(task for task in evidence_tasks if task),
    }


def run_to_log(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fp:
        completed = subprocess.run(
            command,
            stdout=fp,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(completed.returncode)


def arc_visual_only_command(
    arc_agi: Path,
    task_dir: Path,
    max_tasks: int,
    prior: Path,
) -> list[str]:
    return [
        str(arc_agi),
        "--task-dir",
        str(task_dir),
        "--max-tasks",
        str(max_tasks),
        "--program-budget-grid",
        "--loader-summary-only",
        "--induction",
        "--min-induction-evidence",
        "0",
        "--arc-visual-only",
        "--hypothesis-library-prior",
        str(prior),
    ]


def merge_priors(paths: list[Path], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    header = ""
    kept = 0
    with output.open("w", encoding="utf-8") as out:
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as fp:
                local_header = fp.readline()
                if not header and local_header:
                    header = local_header
                    out.write(header)
                for line in fp:
                    line = line.rstrip("\n")
                    if not line or line in seen:
                        continue
                    seen.add(line)
                    out.write(line + "\n")
                    kept += 1
    return kept


def read_curriculum_summary(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    return rows[0] if rows else {}


def run_iteration(args: argparse.Namespace, iteration: int, prior: Path) -> tuple[dict[str, Any], Path | None]:
    iter_dir = args.output_dir / f"iter_{iteration:02d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    train_log = iter_dir / "public_train_visual_only.jsonl"
    eval_log = iter_dir / "public_eval_visual_only.jsonl"
    train_status = run_to_log(
        arc_visual_only_command(
            args.arc_agi,
            args.public_arc_dir / "training",
            args.max_tasks,
            prior,
        ),
        train_log,
    )
    eval_status = run_to_log(
        arc_visual_only_command(
            args.arc_agi,
            args.public_arc_dir / "evaluation",
            args.max_tasks,
            prior,
        ),
        eval_log,
    )

    candidate_tsv = iter_dir / "arc_visual_curriculum_candidates.tsv"
    examples_jsonl = iter_dir / "arc_visual_curriculum_examples.jsonl"
    curriculum_summary = iter_dir / "arc_visual_curriculum_summary.jsonl"
    sources_jsonl = iter_dir / "arc_visual_curriculum_sources.jsonl"
    prior_tsv = iter_dir / "arc_visual_curriculum_prior.tsv"
    attempts_jsonl = iter_dir / "arc_visual_curriculum_attempts.jsonl"

    curriculum_cmd = [
        str(args.runtime),
        "--arc-visual-curriculum",
        "--visual-only-log",
        f"public_train_visual_only={train_log}",
        "--visual-only-log",
        f"public_eval_visual_only={eval_log}",
        "--target-scope",
        "public_train_visual_only",
        "--candidate-tsv",
        str(candidate_tsv),
        "--examples",
        str(examples_jsonl),
        "--summary",
        str(curriculum_summary),
    ]
    curriculum_status = subprocess.run(curriculum_cmd, check=False).returncode

    export_cmd = [
        str(args.arc_agi),
        "--arc-sketch-probe-candidates",
        str(candidate_tsv),
        "--public-arc-dir",
        str(args.public_arc_dir),
        "--arc-sketch-probe-sources",
        str(sources_jsonl),
        "--arc-sketch-probe-prior",
        str(prior_tsv),
        "--arc-sketch-probe-attempts",
        str(attempts_jsonl),
        "--arc-sketches-per-task",
        str(args.sketches_per_task),
    ]
    export_log = iter_dir / "arc_visual_curriculum_export.log"
    export_status = run_to_log(export_cmd, export_log)

    summary_cmd = [
        str(args.runtime),
        "--arc-visual-curriculum",
        "--visual-only-log",
        f"public_train_visual_only={train_log}",
        "--visual-only-log",
        f"public_eval_visual_only={eval_log}",
        "--target-scope",
        "public_train_visual_only",
        "--attempt-log",
        str(attempts_jsonl),
        "--candidate-tsv",
        str(candidate_tsv),
        "--examples",
        str(examples_jsonl),
        "--summary",
        str(curriculum_summary),
    ]
    summary_status = subprocess.run(summary_cmd, check=False).returncode
    curriculum = read_curriculum_summary(curriculum_summary)

    prior_rows = max(0, sum(1 for _line in prior_tsv.open("r", encoding="utf-8")) - 1) if prior_tsv.exists() else 0
    iteration_row = {
        "schema": ITERATION_SCHEMA,
        "report_version": 1,
        "iteration": iteration,
        "input_prior": str(prior),
        "candidate_tsv": str(candidate_tsv),
        "curriculum_prior": str(prior_tsv),
        "attempt_log": str(attempts_jsonl),
        "train_status": train_status,
        "eval_status": eval_status,
        "curriculum_status": curriculum_status,
        "export_status": export_status,
        "summary_status": summary_status,
        "curriculum_prior_rows": prior_rows,
        "train": summarize_arc_log(train_log),
        "eval": summarize_arc_log(eval_log),
        "curriculum_summary": curriculum,
    }
    return iteration_row, prior_tsv if prior_rows > 0 else None


def run_loop(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    prior = args.seed_prior
    prior_paths = [args.seed_prior]
    last_train_certified = -1
    stopped_reason = "iteration_limit"
    for iteration in range(args.iterations):
        row, new_prior = run_iteration(args, iteration, prior)
        rows.append(row)
        train_certified = safe_int(row.get("train", {}).get("certified_tasks"))
        if train_certified <= last_train_certified and iteration > 0:
            stopped_reason = "no_train_certified_improvement"
            break
        last_train_certified = train_certified
        if new_prior is None:
            stopped_reason = "no_new_curriculum_prior"
            break
        prior_paths.append(new_prior)
        next_prior = args.output_dir / f"iter_{iteration:02d}" / "combined_next_prior.tsv"
        kept = merge_priors(prior_paths, next_prior)
        if kept <= 0:
            stopped_reason = "empty_merged_prior"
            break
        prior = next_prior
    final = rows[-1] if rows else {}
    summary = {
        "schema": SUMMARY_SCHEMA,
        "report_version": 1,
        "iterations_run": len(rows),
        "stopped_reason": stopped_reason,
        "seed_prior": str(args.seed_prior),
        "final_prior": str(prior),
        "final_train_certified_tasks": safe_int(
            final.get("train", {}).get("certified_tasks")
        ),
        "final_eval_certified_tasks": safe_int(
            final.get("eval", {}).get("certified_tasks")
        ),
        "final_train_visual_selected": safe_int(
            final.get("train", {}).get("visual_selected")
        ),
        "final_eval_visual_selected": safe_int(
            final.get("eval", {}).get("visual_selected")
        ),
        "output_dir": str(args.output_dir),
    }
    rows.append(summary)
    write_jsonl(args.summary, rows, compact=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arc-agi", type=Path, required=True)
    parser.add_argument("--public-arc-dir", type=Path, required=True)
    parser.add_argument("--seed-prior", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--sketches-per-task", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    return run_loop(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
