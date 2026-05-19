#!/usr/bin/env python3
"""Sweep certified hypothesis-library priors across held-out ARC fixture splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALL_TASKS = [
    "bbox_panel_selection",
    "connect_aligned",
    "extend_aligned_sequence",
    "fill_enclosed",
    "move_to_anchor",
    "move_to_anchor_both_positive",
    "move_to_anchor_x",
    "move_to_anchor_y",
    "object_extract",
    "panel_selection",
    "recolor_by_reference",
    "translate",
]

PROMOTION_FAMILY = [
    "move_to_anchor",
    "move_to_anchor_both_positive",
    "move_to_anchor_x",
    "move_to_anchor_y",
]

STRICT_EVAL_FLAGS = [
    "--program-budget-tight-grid",
    "--loader-summary-only",
    "--induction",
    "--hypothesis-top-k",
    "1",
    "--disable-hypothesis-coverage-selection",
    "--disable-hypothesis-structural-score",
    "--disable-hypothesis-train-filter",
]


@dataclass(frozen=True)
class Mode:
    name: str
    args: tuple[str, ...]
    primary: bool


MODES = [
    Mode(
        "no_prior",
        (
            "--disable-hypothesis-slot-prior",
            "--disable-hypothesis-neural-prior",
            "--disable-hypothesis-cold-start",
        ),
        True,
    ),
    Mode(
        "slot_only",
        (
            "--hypothesis-library-prior",
            "{prior}",
            "--disable-hypothesis-neural-prior",
            "--disable-hypothesis-cold-start",
        ),
        True,
    ),
    Mode(
        "linear_prior",
        (
            "--hypothesis-library-prior",
            "{prior}",
            "--hypothesis-prior-hidden",
            "0",
            "--disable-hypothesis-cold-start",
        ),
        True,
    ),
    Mode(
        "tiny_mlp4",
        (
            "--hypothesis-library-prior",
            "{prior}",
            "--hypothesis-prior-hidden",
            "4",
            "--disable-hypothesis-cold-start",
        ),
        False,
    ),
    Mode(
        "full_mlp24",
        (
            "--hypothesis-library-prior",
            "{prior}",
            "--hypothesis-prior-hidden",
            "24",
            "--disable-hypothesis-cold-start",
        ),
        False,
    ),
]


def parse_json_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line.startswith("{"):
                rows.append(json.loads(line))
    return rows


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def task_split_hash(source_tasks: list[str], heldout_tasks: list[str]) -> str:
    return stable_hash(
        {
            "source_tasks": sorted(source_tasks),
            "heldout_tasks": sorted(heldout_tasks),
        }
    )


def run_jsonl(command: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        subprocess.run(command, stdout=fp, check=True)


def run_promotion_script(
    script: Path,
    source_raw: Path,
    split_dir: Path,
    min_support_tasks: int,
    feedback_input: Path | None = None,
) -> None:
    command = [
        sys.executable,
        str(script),
        "--input",
        str(source_raw),
        "--output-dir",
        str(split_dir),
        "--min-support-tasks",
        str(min_support_tasks),
    ]
    if feedback_input is not None:
        command.extend(["--feedback-input", str(feedback_input)])
    subprocess.run(
        command,
        check=True,
    )


def task_list_arg(tasks: list[str]) -> str:
    return ",".join(tasks)


def parse_task_csv(text: str | None, default: list[str]) -> list[str]:
    if text is None:
        return list(default)
    tasks = [part.strip() for part in text.split(",") if part.strip()]
    if not tasks:
        raise argparse.ArgumentTypeError("task list must not be empty")
    return tasks


def mode_args(mode: Mode, prior_path: Path) -> list[str]:
    return [part.format(prior=str(prior_path)) for part in mode.args]


def program_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("schema") != "crlp.program_proposer.summary.v1":
            continue
        out[(str(row.get("strategy", "")), int(row.get("budget", 0) or 0))] = row
    return out


def selected_evidence_by_task(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("schema") != "crlp.hypothesis.evidence.v1":
            continue
        task = str(row.get("task_instance", "unknown"))
        out.setdefault(task, []).append(row)
    return out


def selection_rows_by_task(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("schema") != "crlp.arc_hypothesis.selection.v1":
            continue
        out[str(row.get("task_instance", "unknown"))] = row
    return out


def prior_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as fp:
        return max(sum(1 for _ in fp) - 1, 0)


def summarize_mode(
    split: str,
    source_tasks: list[str],
    heldout_tasks: list[str],
    mode: Mode,
    prior_rows: int,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    programs = program_rows(rows)
    learned_budget1 = programs.get(("learned", 1), {})
    learned_budget8 = programs.get(("learned", 8), {})
    oracle_budget1 = programs.get(("oracle", 1), {})
    selections = selection_rows_by_task(rows)
    evidence = selected_evidence_by_task(rows)
    if not heldout_tasks:
        heldout_tasks = sorted(set(selections) | set(evidence))
    overlap = sorted(set(source_tasks) & set(heldout_tasks))
    split_hash = task_split_hash(source_tasks, heldout_tasks)
    source_task_hash = stable_hash(sorted(source_tasks))
    heldout_task_hash = stable_hash(sorted(heldout_tasks))

    task_summaries: list[dict[str, Any]] = []
    certified_tasks = 0
    rejected_tasks = 0
    for task in heldout_tasks:
        task_evidence = evidence.get(task, [])
        certified = [row for row in task_evidence if row.get("certificate_status") == "certified"]
        rejected = [row for row in task_evidence if row.get("certificate_status") != "certified"]
        selected_templates = sorted(
            {
                str(row.get("template_name", "unknown"))
                for row in task_evidence
            }
        )
        selection = selections.get(task, {})
        if certified:
            certified_tasks += 1
        if rejected:
            rejected_tasks += 1
        task_summaries.append(
            {
                "schema": "crlp.hypothesis_library.split_sweep.task.v1",
                "split": split,
                "split_hash": split_hash,
                "mode": mode.name,
                "primary_mode": mode.primary,
                "source_tasks": source_tasks,
                "source_task_hash": source_task_hash,
                "heldout_task": task,
                "heldout_task_hash": heldout_task_hash,
                "source_heldout_overlap": overlap,
                "leakage_guard_passed": not overlap,
                "selected_templates": selected_templates,
                "selected_certified": len(certified),
                "selected_rejected": len(rejected),
                "proposals_generated": int(selection.get("proposals_generated", 0) or 0),
                "proposals_selected": int(selection.get("proposals_selected", 0) or 0),
                "proposals_train_consistent_pre_cap": int(
                    selection.get("proposals_train_consistent_pre_cap", 0) or 0
                ),
                "proposals_train_rejected": int(
                    selection.get("proposals_train_rejected", 0) or 0
                ),
                "covered_events": sum(int(row.get("covered_events", 0) or 0) for row in certified),
                "event_count": sum(int(row.get("event_count", 0) or 0) for row in task_evidence),
            }
        )

    aggregate = {
        "schema": "crlp.hypothesis_library.split_sweep.summary.v1",
        "split": split,
        "mode": mode.name,
        "primary_mode": mode.primary,
        "source_tasks": source_tasks,
        "source_task_hash": source_task_hash,
        "heldout_tasks": heldout_tasks,
        "heldout_task_hash": heldout_task_hash,
        "split_hash": split_hash,
        "source_task_count": len(source_tasks),
        "source_heldout_overlap": overlap,
        "source_heldout_overlap_count": len(overlap),
        "leakage_guard_passed": not overlap,
        "prior_rows": prior_rows,
        "heldout_task_count": len(heldout_tasks),
        "selected_certified_tasks": certified_tasks,
        "selected_rejected_tasks": rejected_tasks,
        "learned_certified_budget1": int(learned_budget1.get("certified_programs", 0) or 0),
        "learned_utility_budget1": float(learned_budget1.get("utility", 0.0) or 0.0),
        "learned_certified_budget8": int(learned_budget8.get("certified_programs", 0) or 0),
        "learned_utility_budget8": float(learned_budget8.get("utility", 0.0) or 0.0),
        "oracle_certified_budget1": int(oracle_budget1.get("certified_programs", 0) or 0),
        "proposals_generated": sum(
            int(row.get("proposals_generated", 0) or 0) for row in selections.values()
        ),
        "proposals_train_consistent_pre_cap": sum(
            int(row.get("proposals_train_consistent_pre_cap", 0) or 0)
            for row in selections.values()
        ),
        "proposals_train_rejected": sum(
            int(row.get("proposals_train_rejected", 0) or 0) for row in selections.values()
        ),
    }
    return aggregate, task_summaries


def public_split_dirs(path: Path) -> tuple[Path, Path]:
    training = path / "training"
    evaluation = path / "evaluation"
    if training.is_dir() and evaluation.is_dir():
        return training, evaluation
    train = path / "train"
    eval_dir = path / "eval"
    if train.is_dir() and eval_dir.is_dir():
        return train, eval_dir
    raise RuntimeError(f"{path} must contain training/evaluation or train/eval directories")


def heldout_result_rows(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_split_mode = {(row["split"], row["mode"]): row for row in aggregates}
    out: list[dict[str, Any]] = []
    for row in aggregates:
        baseline = by_split_mode.get((row["split"], "no_prior"))
        if baseline is None:
            continue
        out.append(
            {
                "schema": "crlp.hypothesis_library.heldout_result.v1",
                "report_version": 1,
                "split": row["split"],
                "mode": row["mode"],
                "primary_mode": row["primary_mode"],
                "source_tasks": row["source_tasks"],
                "source_task_hash": row["source_task_hash"],
                "heldout_tasks": row["heldout_tasks"],
                "heldout_task_hash": row["heldout_task_hash"],
                "split_hash": row["split_hash"],
                "source_task_count": row["source_task_count"],
                "source_heldout_overlap_count": row["source_heldout_overlap_count"],
                "leakage_guard_passed": row["leakage_guard_passed"],
                "prior_rows": row["prior_rows"],
                "heldout_task_count": row["heldout_task_count"],
                "baseline_mode": "no_prior",
                "baseline_learned_certified_budget1": baseline[
                    "learned_certified_budget1"
                ],
                "learned_certified_budget1": row["learned_certified_budget1"],
                "delta_learned_certified_budget1": (
                    int(row["learned_certified_budget1"])
                    - int(baseline["learned_certified_budget1"])
                ),
                "baseline_learned_utility_budget1": baseline[
                    "learned_utility_budget1"
                ],
                "learned_utility_budget1": row["learned_utility_budget1"],
                "delta_learned_utility_budget1": (
                    float(row["learned_utility_budget1"])
                    - float(baseline["learned_utility_budget1"])
                ),
                "baseline_oracle_certified_budget1": baseline[
                    "oracle_certified_budget1"
                ],
                "oracle_certified_budget1": row["oracle_certified_budget1"],
                "selected_rejected_tasks": row["selected_rejected_tasks"],
                "proposals_generated": row["proposals_generated"],
                "proposals_train_consistent_pre_cap": row[
                    "proposals_train_consistent_pre_cap"
                ],
                "proposals_train_rejected": row["proposals_train_rejected"],
                "tight_budget_improved_or_equal": (
                    int(row["learned_certified_budget1"])
                    >= int(baseline["learned_certified_budget1"])
                ),
                "heldout_feedback_guard_passed": (
                    bool(row["leakage_guard_passed"])
                    and int(row["learned_certified_budget1"])
                    >= int(baseline["learned_certified_budget1"])
                ),
            }
        )
    return out


def status_cell(task_row: dict[str, Any]) -> str:
    templates = ",".join(task_row["selected_templates"]) if task_row["selected_templates"] else "-"
    if task_row["selected_certified"]:
        return f"C:{templates}"
    if task_row["selected_rejected"]:
        return f"R:{templates}"
    return "-"


def write_markdown(
    path: Path,
    aggregates: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    heldout_results: list[dict[str, Any]],
) -> None:
    by_split_mode = {(row["split"], row["mode"]): row for row in aggregates}
    split_names = sorted({row["split"] for row in aggregates})
    primary_modes = [mode.name for mode in MODES if mode.primary]
    all_modes = [mode.name for mode in MODES]

    lines = [
        "# Hypothesis Library Split Sweep",
        "",
        "Protocol: promote certified hypothesis slot priors from each two-task "
        "`move_to_anchor` source split, then evaluate held-out ARC hypothesis "
        "fixtures with top-k 1, coverage selection disabled, structural scoring "
        "disabled, and pre-ranking train filtering disabled.",
        "",
        "## Aggregate Transfer",
        "",
        "| Split | Source tasks | Prior rows | no-prior @1 | no-prior oracle @1 | slot-only @1 | slot oracle @1 | linear @1 | tiny MLP @1 | full MLP @1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in split_names:
        first = by_split_mode[(split, all_modes[0])]
        no_prior = by_split_mode[(split, "no_prior")]
        slot = by_split_mode[(split, "slot_only")]
        cells = []
        for mode in ("no_prior", "slot_only", "linear_prior", "tiny_mlp4", "full_mlp24"):
            row = by_split_mode[(split, mode)]
            cells.append(
                f"{row['learned_certified_budget1']}/{row['heldout_task_count']} "
                f"({row['learned_utility_budget1']:.3f})"
            )
        lines.append(
            "| {split} | `{source}` | {prior} | {no_prior} | {no_oracle}/{total} | "
            "{slot} | {slot_oracle}/{total} | {linear} | {tiny} | {full} |".format(
                split=split,
                source=",".join(first["source_tasks"]),
                prior=first["prior_rows"],
                no_prior=cells[0],
                no_oracle=no_prior["oracle_certified_budget1"],
                slot=cells[1],
                slot_oracle=slot["oracle_certified_budget1"],
                linear=cells[2],
                tiny=cells[3],
                full=cells[4],
                total=first["heldout_task_count"],
            )
        )

    lines.extend(
        [
            "",
            "## Held-Out Result Deltas",
            "",
            "These rows are the paper-facing held-out claim surface: promotion happens on "
            "the source tasks, then every prior mode is scored against the no-prior "
            "tight-budget baseline on held-out tasks.",
            "",
            "| Split | Mode | Certified @1 delta | Utility @1 delta | Train rejected | Leakage | Tight-budget guard |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in heldout_results:
        if row["mode"] == "no_prior" or not row["primary_mode"]:
            continue
        guard = "pass" if row["tight_budget_improved_or_equal"] else "fail"
        leakage = "pass" if row["leakage_guard_passed"] else "fail"
        lines.append(
            "| {split} | `{mode}` | {cert_delta:+d} | {utility_delta:+.3f} | {rejected} | {leakage} | {guard} |".format(
                split=row["split"],
                mode=row["mode"],
                cert_delta=int(row["delta_learned_certified_budget1"]),
                utility_delta=float(row["delta_learned_utility_budget1"]),
                rejected=int(row["proposals_train_rejected"]),
                leakage=leakage,
                guard=guard,
            )
        )

    lines.extend(
        [
            "",
            "## Decoy Pressure",
            "",
            "| Split | Mode | Generated | Train-consistent pre-cap | Train rejected | Selected certified @1 | Oracle @1 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregates:
        if row["mode"] not in primary_modes:
            continue
        lines.append(
            "| {split} | `{mode}` | {generated} | {consistent} | {rejected} | {cert}/{total} | {oracle}/{total} |".format(
                split=row["split"],
                mode=row["mode"],
                generated=row["proposals_generated"],
                consistent=row["proposals_train_consistent_pre_cap"],
                rejected=row["proposals_train_rejected"],
                cert=row["learned_certified_budget1"],
                oracle=row["oracle_certified_budget1"],
                total=row["heldout_task_count"],
            )
        )

    task_by_key = {(row["split"], row["mode"], row["heldout_task"]): row for row in tasks}
    lines.extend(
        [
            "",
            "## Per-Task Transfer",
            "",
            "| Split | Held-out task | no-prior | slot-only | linear |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for split in split_names:
        heldout_tasks = by_split_mode[(split, all_modes[0])]["heldout_tasks"]
        for task in heldout_tasks:
            cells = [
                status_cell(task_by_key[(split, mode, task)])
                for mode in primary_modes
            ]
            lines.append(f"| {split} | `{task}` | " + " | ".join(cells) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True) + "\n")


def write_heldout_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "split",
        "mode",
        "source_tasks",
        "source_task_hash",
        "heldout_task_hash",
        "split_hash",
        "source_task_count",
        "source_heldout_overlap_count",
        "leakage_guard_passed",
        "heldout_task_count",
        "prior_rows",
        "learned_certified_budget1",
        "delta_learned_certified_budget1",
        "learned_utility_budget1",
        "delta_learned_utility_budget1",
        "selected_rejected_tasks",
        "proposals_train_rejected",
        "tight_budget_improved_or_equal",
        "heldout_feedback_guard_passed",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{field: row.get(field) for field in fieldnames},
                    "source_tasks": ",".join(row["source_tasks"]),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arc-agi", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, default=Path("tests/fixtures/arc_hypothesis"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/float/hypothesis_library_split_sweep"))
    parser.add_argument("--min-support-tasks", type=int, default=2)
    parser.add_argument("--task-ids", help="Comma-separated task universe; defaults to fixture task list")
    parser.add_argument("--promotion-family", help="Comma-separated task family used for source-pair promotion")
    parser.add_argument("--source-task-ids", help="Comma-separated fixed source tasks; disables pair sweep")
    parser.add_argument("--heldout-task-ids", help="Comma-separated held-out tasks for fixed source mode")
    parser.add_argument("--public-arc-dir", type=Path, help="Public ARC root with training/evaluation splits")
    parser.add_argument("--public-train-max-tasks", type=int, default=400)
    parser.add_argument("--public-eval-max-tasks", type=int, default=400)
    parser.add_argument("--public-min-induction-evidence", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path.cwd()
    promotion_script = repo_root / "scripts" / "certified_hypothesis_library.py"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    aggregates: list[dict[str, Any]] = []
    task_summaries: list[dict[str, Any]] = []
    if args.public_arc_dir:
        train_dir, eval_dir = public_split_dirs(args.public_arc_dir)
        split = "public_training__public_evaluation"
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        source_raw = split_dir / "source_raw.jsonl"
        run_jsonl(
            [
                str(args.arc_agi),
                "--task-dir",
                str(train_dir),
                "--max-tasks",
                str(args.public_train_max_tasks),
                "--program-budget-grid",
                "--loader-summary-only",
                "--induction",
                "--min-induction-evidence",
                str(args.public_min_induction_evidence),
            ],
            source_raw,
        )
        run_promotion_script(
            promotion_script,
            source_raw,
            split_dir,
            args.min_support_tasks,
        )
        prior_path = split_dir / "library_prior.tsv"
        priors = prior_row_count(prior_path)
        for mode in MODES:
            output_path = split_dir / f"{mode.name}.jsonl"
            command = [
                str(args.arc_agi),
                "--task-dir",
                str(eval_dir),
                "--max-tasks",
                str(args.public_eval_max_tasks),
                "--min-induction-evidence",
                str(args.public_min_induction_evidence),
                *STRICT_EVAL_FLAGS,
                *mode_args(mode, prior_path),
            ]
            run_jsonl(command, output_path)
            rows = parse_json_rows(output_path)
            aggregate, per_task = summarize_mode(
                split,
                ["public_training"],
                [],
                mode,
                priors,
                rows,
            )
            aggregate["source_split"] = str(train_dir)
            aggregate["heldout_split"] = str(eval_dir)
            aggregates.append(aggregate)
            task_summaries.extend(per_task)
        run_promotion_script(
            promotion_script,
            source_raw,
            split_dir,
            args.min_support_tasks,
            split_dir / "slot_only.jsonl",
        )
        heldout_results = heldout_result_rows(aggregates)
        aggregate_path = args.output_dir / "split_sweep_summary.jsonl"
        task_path = args.output_dir / "split_sweep_tasks.jsonl"
        heldout_path = args.output_dir / "heldout_result.jsonl"
        heldout_csv_path = args.output_dir / "heldout_result.csv"
        write_jsonl(aggregate_path, aggregates)
        write_jsonl(task_path, task_summaries)
        write_jsonl(heldout_path, heldout_results)
        write_heldout_csv(heldout_csv_path, heldout_results)
        summary_path = args.output_dir / "summary.md"
        write_markdown(summary_path, aggregates, task_summaries, heldout_results)
        print(f"wrote {aggregate_path}")
        print(f"wrote {task_path}")
        print(f"wrote {heldout_path}")
        print(f"wrote {heldout_csv_path}")
        print(f"wrote {summary_path}")
        return 0

    all_tasks = parse_task_csv(args.task_ids, ALL_TASKS)
    promotion_family = parse_task_csv(args.promotion_family, PROMOTION_FAMILY)
    if args.source_task_ids:
        split_pairs = [tuple(parse_task_csv(args.source_task_ids, []))]
    else:
        split_pairs = list(itertools.combinations(promotion_family, 2))
    for source_pair in split_pairs:
        source_tasks = list(source_pair)
        heldout_tasks = parse_task_csv(args.heldout_task_ids, [task for task in all_tasks if task not in source_tasks])
        split = "__".join(source_tasks)
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        source_raw = split_dir / "source_raw.jsonl"
        run_jsonl(
            [
                str(args.arc_agi),
                "--task-dir",
                str(args.task_dir),
                "--task-ids",
                task_list_arg(source_tasks),
                "--max-tasks",
                "10",
                "--program-budget-grid",
                "--loader-summary-only",
                "--induction",
                "--min-induction-evidence",
                str(len(source_tasks)),
            ],
            source_raw,
        )
        run_promotion_script(
            promotion_script,
            source_raw,
            split_dir,
            args.min_support_tasks,
        )
        prior_path = split_dir / "library_prior.tsv"
        priors = prior_row_count(prior_path)

        for mode in MODES:
            output_path = split_dir / f"{mode.name}.jsonl"
            command = [
                str(args.arc_agi),
                "--task-dir",
                str(args.task_dir),
                "--task-ids",
                task_list_arg(heldout_tasks),
                "--max-tasks",
                "10",
                "--min-induction-evidence",
                str(len(heldout_tasks)),
                *STRICT_EVAL_FLAGS,
                *mode_args(mode, prior_path),
            ]
            run_jsonl(command, output_path)
            rows = parse_json_rows(output_path)
            aggregate, per_task = summarize_mode(
                split,
                source_tasks,
                heldout_tasks,
                mode,
                priors,
                rows,
            )
            aggregates.append(aggregate)
            task_summaries.extend(per_task)
        run_promotion_script(
            promotion_script,
            source_raw,
            split_dir,
            args.min_support_tasks,
            split_dir / "slot_only.jsonl",
        )

    heldout_results = heldout_result_rows(aggregates)
    aggregate_path = args.output_dir / "split_sweep_summary.jsonl"
    task_path = args.output_dir / "split_sweep_tasks.jsonl"
    heldout_path = args.output_dir / "heldout_result.jsonl"
    heldout_csv_path = args.output_dir / "heldout_result.csv"
    write_jsonl(aggregate_path, aggregates)
    write_jsonl(task_path, task_summaries)
    write_jsonl(heldout_path, heldout_results)
    write_heldout_csv(heldout_csv_path, heldout_results)
    summary_path = args.output_dir / "summary.md"
    write_markdown(summary_path, aggregates, task_summaries, heldout_results)
    print(f"wrote {aggregate_path}")
    print(f"wrote {task_path}")
    print(f"wrote {heldout_path}")
    print(f"wrote {heldout_csv_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
