#!/usr/bin/env python3
"""Run an ARC hypothesis-library feedback loop from held-out failure rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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

DEFAULT_SOURCE_TASKS = ["move_to_anchor", "move_to_anchor_x"]
DEFAULT_FAILURE_MODE = "slot_only"
DEFAULT_FEEDBACK_MODE = "slot_only"
GENERATED_SIBLING_FAMILIES = [
    "bbox_panel_selection",
    "connect_aligned",
    "extend_aligned_sequence",
    "object_extract",
    "recolor_by_reference",
]


@dataclass(frozen=True)
class SplitRun:
    name: str
    output_dir: Path
    command: list[str]
    rows: list[dict[str, Any]]
    task_rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]
    promoted_libraries: list[dict[str, Any]]


def parse_task_csv(text: str | None, default: list[str]) -> list[str]:
    if text is None:
        return list(default)
    tasks = [part.strip() for part in text.split(",") if part.strip()]
    if not tasks:
        raise argparse.ArgumentTypeError("task list must not be empty")
    return tasks


def task_list_arg(tasks: list[str]) -> str:
    return ",".join(tasks)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line.startswith("{"):
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def remap_grid_colors(grid: list[list[int]], offset: int) -> list[list[int]]:
    remapped: list[list[int]] = []
    for row in grid:
        remapped.append(
            [
                0 if value == 0 else ((int(value) + offset - 1) % 9) + 1
                for value in row
            ]
        )
    return remapped


def remap_task_colors(task: dict[str, Any], offset: int) -> dict[str, Any]:
    out: dict[str, Any] = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in task.get(split, []) or []:
            next_pair: dict[str, Any] = {
                "input": remap_grid_colors(pair["input"], offset),
            }
            if "output" in pair:
                next_pair["output"] = remap_grid_colors(pair["output"], offset)
            out[split].append(next_pair)
    return out


def copy_task_dir(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.glob("*.json")):
        shutil.copy2(path, target / path.name)


def generate_sibling_tasks(
    source_dir: Path,
    target_dir: Path,
    families: list[str],
    siblings_per_family: int,
) -> list[str]:
    generated: list[str] = []
    if siblings_per_family <= 0:
        return generated
    copy_task_dir(source_dir, target_dir)
    for family in families:
        base_path = source_dir / f"{family}.json"
        if not base_path.exists():
            raise FileNotFoundError(base_path)
        base_task = load_json(base_path)
        for index in range(1, siblings_per_family + 1):
            task_id = f"{family}_sibling_{index}"
            write_json(
                target_dir / f"{task_id}.json",
                remap_task_colors(base_task, offset=index + 1),
            )
            generated.append(task_id)
    return generated


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def is_certified(row: dict[str, Any] | None) -> bool:
    return safe_int((row or {}).get("selected_certified")) > 0


def row_utility(row: dict[str, Any] | None) -> float:
    if row is None:
        return 0.0
    covered = safe_int(row.get("covered_events"))
    events = safe_int(row.get("event_count"))
    return 0.0 if events == 0 else covered / events


def row_by_mode_task(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("schema") != "crlp.hypothesis_library.split_sweep.task.v1":
            continue
        out[(str(row.get("mode", "")), str(row.get("heldout_task", "")))] = row
    return out


def promoted_libraries(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("*/promotions.jsonl")):
        for row in parse_jsonl(path):
            if row.get("schema") != "crlp.hypothesis_library.promotion.v1":
                continue
            if not bool(row.get("promoted", False)):
                continue
            rows.append(
                {
                    "library_id": str(row.get("library_id", "")),
                    "template_id": safe_int(row.get("template_id")),
                    "template_name": str(row.get("template_name", "unknown")),
                    "support_tasks": safe_int(row.get("support_tasks")),
                    "task_instances": list(row.get("task_instances", []) or []),
                    "covered_events": safe_int(row.get("covered_events")),
                    "false_rejections": safe_int(row.get("false_rejections")),
                    "variable_slot_count": len(row.get("variable_slots", []) or []),
                    "invariant_slot_count": len(row.get("invariant_slots", []) or []),
                }
            )
    return sorted(rows, key=lambda row: (row["template_name"], row["library_id"]))


def run_split_sweep(
    *,
    repo_root: Path,
    script: Path,
    arc_agi: Path,
    task_dir: Path,
    output_dir: Path,
    source_tasks: list[str],
    heldout_tasks: list[str],
    min_support_tasks: int,
    name: str,
    stress_no_prior: bool,
) -> SplitRun:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(script),
        "--arc-agi",
        str(arc_agi),
        "--task-dir",
        str(task_dir),
        "--source-task-ids",
        task_list_arg(source_tasks),
        "--heldout-task-ids",
        task_list_arg(heldout_tasks),
        "--min-support-tasks",
        str(min_support_tasks),
        "--output-dir",
        str(output_dir),
    ]
    if stress_no_prior:
        command.append("--stress-no-prior")
    subprocess.run(command, cwd=repo_root, check=True)
    task_rows = parse_jsonl(output_dir / "split_sweep_tasks.jsonl")
    summary_rows = parse_jsonl(output_dir / "split_sweep_summary.jsonl")
    rows = task_rows + summary_rows + parse_jsonl(output_dir / "heldout_result.jsonl")
    return SplitRun(
        name=name,
        output_dir=output_dir,
        command=command,
        rows=rows,
        task_rows=task_rows,
        summary_rows=summary_rows,
        promoted_libraries=promoted_libraries(output_dir),
    )


def feedback_candidates(
    task_rows: list[dict[str, Any]],
    failure_mode: str,
    limit: int,
    max_per_family: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in task_rows:
        if row.get("mode") != failure_mode:
            continue
        if not bool(row.get("leakage_guard_passed", False)):
            continue
        if safe_int(row.get("selected_certified")) > 0:
            continue
        if safe_int(row.get("selected_rejected")) <= 0:
            continue
        candidate = dict(row)
        candidate["feedback_candidate_score"] = [
            1 if bool(row.get("proposal_cap_hit", False)) else 0,
            safe_int(row.get("proposals_train_consistent_pre_cap")),
            safe_int(row.get("proposals_generated")),
            len(row.get("novel_heldout_templates", []) or []),
        ]
        candidates.append(candidate)
    candidates.sort(
        key=lambda row: (
            row["feedback_candidate_score"][0],
            row["feedback_candidate_score"][1],
            row["feedback_candidate_score"][2],
            row["feedback_candidate_score"][3],
            str(row.get("heldout_task", "")),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    for row in candidates:
        family = str(row.get("heldout_family", row.get("heldout_task", "unknown")))
        if max_per_family > 0 and family_counts.get(family, 0) >= max_per_family:
            continue
        selected.append(row)
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected) >= limit:
            return selected
    return selected


def library_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("template_name", "")), str(row.get("library_id", ""))


def new_feedback_libraries(
    baseline: list[dict[str, Any]],
    feedback: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_keys = {library_key(row) for row in baseline}
    return [row for row in feedback if library_key(row) not in baseline_keys]


def libraries_supported_by_feedback(
    libraries: list[dict[str, Any]],
    feedback_tasks: list[str],
) -> list[dict[str, Any]]:
    feedback_set = set(feedback_tasks)
    return [
        row
        for row in libraries
        if feedback_set & set(str(task) for task in row.get("task_instances", []) or [])
    ]


def compare_tasks(
    *,
    heldout_tasks: list[str],
    baseline_rows: list[dict[str, Any]],
    feedback_rows: list[dict[str, Any]],
    feedback_mode: str,
) -> list[dict[str, Any]]:
    baseline_by_task = row_by_mode_task(baseline_rows)
    feedback_by_task = row_by_mode_task(feedback_rows)
    out: list[dict[str, Any]] = []
    for task in heldout_tasks:
        no_prior = baseline_by_task.get(("no_prior", task))
        source_prior = baseline_by_task.get((feedback_mode, task))
        feedback_prior = feedback_by_task.get((feedback_mode, task))
        out.append(
            {
                "schema": "crlp.hypothesis_library.feedback_loop.task.v1",
                "report_version": 1,
                "heldout_task": task,
                "heldout_family": str(
                    (feedback_prior or source_prior or no_prior or {}).get(
                        "heldout_family",
                        "unknown",
                    )
                ),
                "feedback_mode": feedback_mode,
                "no_prior_certified": is_certified(no_prior),
                "source_prior_certified": is_certified(source_prior),
                "feedback_prior_certified": is_certified(feedback_prior),
                "feedback_improved_vs_no_prior": (
                    not is_certified(no_prior) and is_certified(feedback_prior)
                ),
                "feedback_improved_vs_source_prior": (
                    not is_certified(source_prior) and is_certified(feedback_prior)
                ),
                "no_prior_templates": list((no_prior or {}).get("selected_templates", []) or []),
                "source_prior_templates": list(
                    (source_prior or {}).get("selected_templates", []) or []
                ),
                "feedback_prior_templates": list(
                    (feedback_prior or {}).get("selected_templates", []) or []
                ),
                "no_prior_utility": row_utility(no_prior),
                "source_prior_utility": row_utility(source_prior),
                "feedback_prior_utility": row_utility(feedback_prior),
                "source_prior_proposals_generated": safe_int(
                    (source_prior or {}).get("proposals_generated")
                ),
                "feedback_prior_proposals_generated": safe_int(
                    (feedback_prior or {}).get("proposals_generated")
                ),
            }
        )
    return out


def count_certified(rows: list[dict[str, Any]], field: str) -> int:
    return sum(1 for row in rows if bool(row.get(field, False)))


def mean_utility(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return sum(safe_float(row.get(field)) for row in rows) / len(rows)


def write_markdown(
    path: Path,
    summary: dict[str, Any],
    task_rows: list[dict[str, Any]],
    feedback_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Hypothesis Library Feedback Loop",
        "",
        "Protocol: run the split sweep, select held-out failures, add those failed "
        "tasks to the source side as feedback evidence, re-promote certified "
        "hypotheses, then re-evaluate only the still-held-out tasks.",
        "",
        "## Result",
        "",
        f"- Loop result: `{summary['loop_result']}`",
        f"- Feedback tasks: `{', '.join(summary['feedback_tasks']) or '-'}`",
        f"- Max feedback tasks per family: `{summary['max_feedback_tasks_per_family']}`",
        f"- Generated sibling tasks: `{', '.join(summary['generated_sibling_tasks']) or '-'}`",
        f"- Evaluation held-out tasks: `{', '.join(summary['evaluation_heldout_tasks']) or '-'}`",
        f"- New feedback-supported libraries: `{', '.join(summary['new_feedback_template_names']) or '-'}`",
        f"- Strict held-out improvements vs source prior: `{', '.join(summary['strict_heldout_improved_tasks']) or '-'}`",
        f"- Stress no-prior baseline: `{'yes' if summary['stress_no_prior'] else 'no'}`",
        (
            "- Certified task counts: "
            f"no-prior `{summary['no_prior_certified_tasks']}/{summary['evaluated_task_count']}`, "
            f"source prior `{summary['source_prior_certified_tasks']}/{summary['evaluated_task_count']}`, "
            f"feedback prior `{summary['feedback_prior_certified_tasks']}/{summary['evaluated_task_count']}`"
        ),
        "",
        "## Before / After",
        "",
        "| Held-out task | Family | no-prior | source prior | feedback prior | Templates after feedback |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in task_rows:
        lines.append(
            "| {task} | {family} | {no_prior} | {source_prior} | {feedback_prior} | `{templates}` |".format(
                task=row["heldout_task"],
                family=row["heldout_family"],
                no_prior="cert" if row["no_prior_certified"] else "fail",
                source_prior="cert" if row["source_prior_certified"] else "fail",
                feedback_prior="cert" if row["feedback_prior_certified"] else "fail",
                templates=",".join(row["feedback_prior_templates"]) or "-",
            )
        )
    lines.extend(
        [
            "",
            "## Selected Failures",
            "",
            "| Feedback task | Family | Selected templates | Generated | Train-consistent pre-cap |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in feedback_rows:
        lines.append(
            "| {task} | {family} | `{templates}` | {generated} | {consistent} |".format(
                task=row.get("heldout_task", "unknown"),
                family=row.get("heldout_family", "unknown"),
                templates=",".join(row.get("selected_templates", []) or []) or "-",
                generated=safe_int(row.get("proposals_generated")),
                consistent=safe_int(row.get("proposals_train_consistent_pre_cap")),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if summary["strict_heldout_improved_tasks"]:
        lines.append(
            "The strict guard passed: feedback produced certified library evidence "
            "and improved at least one still-held-out task relative to the source "
            "prior."
        )
        if summary["stress_no_prior"]:
            lines.append(
                "The no-prior arm is a labeled stress baseline with verifier-derived "
                "train-consistency ranking disabled; use it to measure search-pressure "
                "recovery, not as the default evaluator."
            )
        if summary["delta_feedback_vs_no_prior"] == 0:
            lines.append(
                "No-prior search already certifies these held-out fixtures, so this "
                "is a prior-recovery result: feedback repairs a misleading source "
                "prior rather than beating unprioritized exhaustive search."
            )
    elif summary["new_feedback_template_names"]:
        lines.append(
            "The promotion loop fired, but the strict held-out improvement guard did "
            "not pass on this fixture split. This means the failed rows are useful "
            "for discovering certified templates, but the current fixture set either "
            "lacks sibling held-out tasks for those families or the source prior was "
            "already saturated on the remaining held-out tasks."
        )
    else:
        lines.append(
            "No feedback-supported library was promoted. The failure rows remain "
            "diagnostic until the proposer produces a certifiable hypothesis for "
            "at least one selected failure."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arc-agi", type=Path, default=Path("build/float/arc_agi"))
    parser.add_argument("--task-dir", type=Path, default=Path("tests/fixtures/arc_hypothesis"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/float/hypothesis_library_feedback_loop"))
    parser.add_argument("--source-task-ids", help="Comma-separated initial source tasks")
    parser.add_argument("--heldout-task-ids", help="Comma-separated initial held-out tasks")
    parser.add_argument("--failure-mode", default=DEFAULT_FAILURE_MODE)
    parser.add_argument("--feedback-mode", default=DEFAULT_FEEDBACK_MODE)
    parser.add_argument("--max-feedback-tasks", type=int, default=1)
    parser.add_argument(
        "--max-feedback-tasks-per-family",
        type=int,
        default=1,
        help="Prefer at most this many selected feedback failures per held-out family.",
    )
    parser.add_argument("--generated-siblings-per-family", type=int, default=0)
    parser.add_argument(
        "--generated-sibling-families",
        default=",".join(GENERATED_SIBLING_FAMILIES),
        help="Comma-separated fixture families to color-remap into sibling tasks.",
    )
    parser.add_argument("--baseline-min-support-tasks", type=int, default=2)
    parser.add_argument("--feedback-min-support-tasks", type=int, default=1)
    parser.add_argument(
        "--stress-no-prior",
        action="store_true",
        help=(
            "Run the no-prior arm without verifier-derived train-consistency ranking "
            "so the loop measures recovery under proposal search pressure."
        ),
    )
    parser.add_argument(
        "--keep-feedback-heldout",
        action="store_true",
        help="Evaluate feedback tasks again instead of enforcing strict no-leak held-out evaluation.",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    script = repo_root / "scripts" / "hypothesis_library_split_sweep.py"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_sibling_tasks: list[str] = []
    task_dir = args.task_dir
    if args.generated_siblings_per_family > 0:
        sibling_families = parse_task_csv(
            args.generated_sibling_families,
            GENERATED_SIBLING_FAMILIES,
        )
        task_dir = args.output_dir / "generated_task_dir"
        generated_sibling_tasks = generate_sibling_tasks(
            args.task_dir,
            task_dir,
            sibling_families,
            args.generated_siblings_per_family,
        )
    source_tasks = parse_task_csv(args.source_task_ids, DEFAULT_SOURCE_TASKS)
    all_tasks = list(ALL_TASKS) + generated_sibling_tasks
    default_heldout = [task for task in all_tasks if task not in set(source_tasks)]
    heldout_tasks = parse_task_csv(args.heldout_task_ids, default_heldout)
    if args.max_feedback_tasks <= 0:
        raise SystemExit("--max-feedback-tasks must be positive")
    if args.max_feedback_tasks_per_family < 0:
        raise SystemExit("--max-feedback-tasks-per-family must be non-negative")

    baseline = run_split_sweep(
        repo_root=repo_root,
        script=script,
        arc_agi=args.arc_agi,
        task_dir=task_dir,
        output_dir=args.output_dir / "baseline",
        source_tasks=source_tasks,
        heldout_tasks=heldout_tasks,
        min_support_tasks=args.baseline_min_support_tasks,
        name="baseline",
        stress_no_prior=args.stress_no_prior,
    )
    candidates = feedback_candidates(
        baseline.task_rows,
        args.failure_mode,
        args.max_feedback_tasks,
        args.max_feedback_tasks_per_family,
    )
    feedback_tasks = [str(row["heldout_task"]) for row in candidates]
    evaluation_heldout_tasks = (
        heldout_tasks if args.keep_feedback_heldout else [task for task in heldout_tasks if task not in set(feedback_tasks)]
    )

    commands = [
        " ".join(baseline.command),
    ]
    if feedback_tasks and evaluation_heldout_tasks:
        feedback = run_split_sweep(
            repo_root=repo_root,
            script=script,
            arc_agi=args.arc_agi,
            task_dir=task_dir,
            output_dir=args.output_dir / "feedback",
            source_tasks=source_tasks + feedback_tasks,
            heldout_tasks=evaluation_heldout_tasks,
            min_support_tasks=args.feedback_min_support_tasks,
            name="feedback",
            stress_no_prior=args.stress_no_prior,
        )
        commands.append(" ".join(feedback.command))
        task_comparison = compare_tasks(
            heldout_tasks=evaluation_heldout_tasks,
            baseline_rows=baseline.task_rows,
            feedback_rows=feedback.task_rows,
            feedback_mode=args.feedback_mode,
        )
        new_libraries = new_feedback_libraries(
            baseline.promoted_libraries,
            feedback.promoted_libraries,
        )
        feedback_supported_libraries = libraries_supported_by_feedback(
            new_libraries,
            feedback_tasks,
        )
    else:
        feedback = None
        task_comparison = []
        new_libraries = []
        feedback_supported_libraries = []

    strict_improved = [
        str(row["heldout_task"])
        for row in task_comparison
        if bool(row.get("feedback_improved_vs_source_prior", False))
    ]
    improved_vs_no_prior = [
        str(row["heldout_task"])
        for row in task_comparison
        if bool(row.get("feedback_improved_vs_no_prior", False))
    ]
    new_template_names = sorted(
        {str(row.get("template_name", "")) for row in feedback_supported_libraries}
    )
    if strict_improved:
        loop_result = "strict_heldout_improved"
    elif feedback_supported_libraries:
        loop_result = "certified_feedback_no_strict_heldout_gain"
    elif feedback_tasks:
        loop_result = "feedback_selected_no_new_library"
    else:
        loop_result = "no_feedback_candidate"

    summary = {
        "schema": "crlp.hypothesis_library.feedback_loop.summary.v1",
        "report_version": 1,
        "source_tasks": source_tasks,
        "source_task_hash": stable_hash(sorted(source_tasks)),
        "initial_heldout_tasks": heldout_tasks,
        "initial_heldout_task_hash": stable_hash(sorted(heldout_tasks)),
        "failure_mode": args.failure_mode,
        "feedback_mode": args.feedback_mode,
        "feedback_tasks": feedback_tasks,
        "max_feedback_tasks_per_family": args.max_feedback_tasks_per_family,
        "feedback_task_hash": stable_hash(sorted(feedback_tasks)),
        "generated_sibling_tasks": generated_sibling_tasks,
        "generated_siblings_per_family": args.generated_siblings_per_family,
        "evaluation_heldout_tasks": evaluation_heldout_tasks,
        "evaluation_heldout_task_hash": stable_hash(sorted(evaluation_heldout_tasks)),
        "strict_no_leakage": not args.keep_feedback_heldout,
        "baseline_min_support_tasks": args.baseline_min_support_tasks,
        "feedback_min_support_tasks": args.feedback_min_support_tasks,
        "stress_no_prior": bool(args.stress_no_prior),
        "baseline_promoted_libraries": baseline.promoted_libraries,
        "feedback_promoted_libraries": feedback.promoted_libraries if feedback else [],
        "new_feedback_libraries": feedback_supported_libraries,
        "new_feedback_template_names": new_template_names,
        "feedback_candidate_count": len(candidates),
        "evaluated_task_count": len(task_comparison),
        "no_prior_certified_tasks": count_certified(task_comparison, "no_prior_certified"),
        "source_prior_certified_tasks": count_certified(task_comparison, "source_prior_certified"),
        "feedback_prior_certified_tasks": count_certified(task_comparison, "feedback_prior_certified"),
        "delta_feedback_vs_no_prior": (
            count_certified(task_comparison, "feedback_prior_certified")
            - count_certified(task_comparison, "no_prior_certified")
        ),
        "delta_feedback_vs_source_prior": (
            count_certified(task_comparison, "feedback_prior_certified")
            - count_certified(task_comparison, "source_prior_certified")
        ),
        "mean_no_prior_utility": mean_utility(task_comparison, "no_prior_utility"),
        "mean_source_prior_utility": mean_utility(task_comparison, "source_prior_utility"),
        "mean_feedback_prior_utility": mean_utility(task_comparison, "feedback_prior_utility"),
        "strict_heldout_improved_tasks": strict_improved,
        "feedback_improved_vs_no_prior_tasks": improved_vs_no_prior,
        "loop_result": loop_result,
    }

    write_jsonl(args.output_dir / "feedback_candidates.jsonl", candidates)
    write_jsonl(args.output_dir / "feedback_loop_tasks.jsonl", task_comparison)
    write_jsonl(args.output_dir / "feedback_loop_summary.jsonl", [summary])
    (args.output_dir / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    write_markdown(args.output_dir / "summary.md", summary, task_comparison, candidates)
    print(f"wrote {args.output_dir / 'feedback_candidates.jsonl'}")
    print(f"wrote {args.output_dir / 'feedback_loop_tasks.jsonl'}")
    print(f"wrote {args.output_dir / 'feedback_loop_summary.jsonl'}")
    print(f"wrote {args.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
