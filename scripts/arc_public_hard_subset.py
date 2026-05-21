#!/usr/bin/env python3
"""Build a non-saturated ARC subset for hypothesis-library experiments."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crlp.io import read_jsonl_required, safe_int, write_jsonl

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

NO_PRIOR_FLAGS = [
    "--disable-hypothesis-slot-prior",
    "--disable-hypothesis-neural-prior",
    "--disable-hypothesis-cold-start",
]

STRESS_NO_PRIOR_FLAGS = [
    *STRICT_EVAL_FLAGS,
    *NO_PRIOR_FLAGS,
    "--disable-hypothesis-train-consistency-prior",
]


@dataclass(frozen=True)
class SplitSpec:
    name: str
    path: Path
    max_tasks: int


@dataclass(frozen=True)
class TaskDiag:
    task_id: str
    split: str
    mode: str
    certified_evidence: int
    rejected_evidence: int
    selected_templates: tuple[str, ...]
    certified_templates: tuple[str, ...]
    rejected_templates: tuple[str, ...]
    covered_events: int
    event_count: int
    utility: float
    proposals_attempted: int
    proposals_generated: int
    proposals_selected: int
    proposals_train_consistent_pre_cap: int
    proposals_train_rejected: int
    proposal_cap_hit: bool
    train_consistent_templates: tuple[str, ...]

    @property
    def certified(self) -> bool:
        return self.certified_evidence > 0

    @property
    def has_search_pressure(self) -> bool:
        return (
            self.proposals_train_consistent_pre_cap > 0
            or self.rejected_evidence > 0
            or self.proposal_cap_hit
        )


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


def task_files(task_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(task_dir.glob("*.json")):
        files[path.stem] = path
    return files


def run_arc(
    arc_agi: Path,
    split: SplitSpec,
    output_path: Path,
    flags: list[str],
    task_ids: list[str] | None = None,
) -> None:
    command = [
        str(arc_agi),
        "--task-dir",
        str(split.path),
        "--max-tasks",
        str(split.max_tasks),
        *flags,
    ]
    if task_ids:
        command.extend(["--task-ids", ",".join(task_ids)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        subprocess.run(command, stdout=fp, check=True)


def names_from_templates(selection: dict[str, Any], key: str) -> tuple[str, ...]:
    names = {
        str(template.get("name", "unknown"))
        for template in selection.get("templates", []) or []
        if safe_int(template.get(key)) > 0
    }
    return tuple(sorted(names))


def diagnose(rows: list[dict[str, Any]], split: str, mode: str) -> dict[str, TaskDiag]:
    selections: dict[str, dict[str, Any]] = {}
    evidence: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        schema = row.get("schema")
        task_id = str(row.get("task_instance", ""))
        if not task_id:
            continue
        if schema == "crlp.arc_hypothesis.selection.v1":
            selections[task_id] = row
        elif schema == "crlp.hypothesis.evidence.v1":
            evidence.setdefault(task_id, []).append(row)

    out: dict[str, TaskDiag] = {}
    for task_id in sorted(set(selections) | set(evidence)):
        selection = selections.get(task_id, {})
        task_evidence = evidence.get(task_id, [])
        certified_rows = [
            row for row in task_evidence if row.get("certificate_status") == "certified"
        ]
        rejected_rows = [
            row for row in task_evidence if row.get("certificate_status") != "certified"
        ]
        selected_templates = names_from_templates(selection, "selected")
        train_consistent_templates = names_from_templates(
            selection, "train_consistent_pre_cap"
        )
        out[task_id] = TaskDiag(
            task_id=task_id,
            split=split,
            mode=mode,
            certified_evidence=len(certified_rows),
            rejected_evidence=len(rejected_rows),
            selected_templates=selected_templates,
            certified_templates=tuple(
                sorted({str(row.get("template_name", "unknown")) for row in certified_rows})
            ),
            rejected_templates=tuple(
                sorted({str(row.get("template_name", "unknown")) for row in rejected_rows})
            ),
            covered_events=sum(safe_int(row.get("covered_events")) for row in task_evidence),
            event_count=sum(safe_int(row.get("event_count")) for row in task_evidence),
            utility=sum(float(row.get("utility", 0.0) or 0.0) for row in task_evidence),
            proposals_attempted=safe_int(selection.get("proposals_attempted")),
            proposals_generated=safe_int(selection.get("proposals_generated")),
            proposals_selected=safe_int(selection.get("proposals_selected")),
            proposals_train_consistent_pre_cap=safe_int(
                selection.get("proposals_train_consistent_pre_cap")
            ),
            proposals_train_rejected=safe_int(selection.get("proposals_train_rejected")),
            proposal_cap_hit=bool(selection.get("proposal_cap_hit", False)),
            train_consistent_templates=train_consistent_templates,
        )
    return out


def diag_row(diag: TaskDiag) -> dict[str, Any]:
    return {
        "schema": "crlp.arc_public_hard_subset.diagnostic.v1",
        "report_version": 1,
        "task_id": diag.task_id,
        "split": diag.split,
        "mode": diag.mode,
        "certified": diag.certified,
        "certified_evidence": diag.certified_evidence,
        "rejected_evidence": diag.rejected_evidence,
        "selected_templates": list(diag.selected_templates),
        "certified_templates": list(diag.certified_templates),
        "rejected_templates": list(diag.rejected_templates),
        "covered_events": diag.covered_events,
        "event_count": diag.event_count,
        "utility": diag.utility,
        "proposals_attempted": diag.proposals_attempted,
        "proposals_generated": diag.proposals_generated,
        "proposals_selected": diag.proposals_selected,
        "proposals_train_consistent_pre_cap": diag.proposals_train_consistent_pre_cap,
        "proposals_train_rejected": diag.proposals_train_rejected,
        "proposal_cap_hit": diag.proposal_cap_hit,
        "train_consistent_templates": list(diag.train_consistent_templates),
        "has_search_pressure": diag.has_search_pressure,
    }


def empty_diag(task_id: str, split: str, mode: str) -> TaskDiag:
    return TaskDiag(
        task_id=task_id,
        split=split,
        mode=mode,
        certified_evidence=0,
        rejected_evidence=0,
        selected_templates=(),
        certified_templates=(),
        rejected_templates=(),
        covered_events=0,
        event_count=0,
        utility=0.0,
        proposals_attempted=0,
        proposals_generated=0,
        proposals_selected=0,
        proposals_train_consistent_pre_cap=0,
        proposals_train_rejected=0,
        proposal_cap_hit=False,
        train_consistent_templates=(),
    )


def select_sources(diags: dict[str, TaskDiag], limit: int) -> list[TaskDiag]:
    candidates = [diag for diag in diags.values() if diag.certified]
    candidates.sort(
        key=lambda diag: (
            -diag.certified_evidence,
            -diag.covered_events,
            -diag.proposals_train_consistent_pre_cap,
            diag.task_id,
        )
    )
    return candidates[:limit]


def select_heldout(
    default_diags: dict[str, TaskDiag],
    stress_diags: dict[str, TaskDiag],
    *,
    limit: int,
    source_task_ids: set[str],
    min_train_consistent: int,
    require_default_failure: bool,
    require_stress_failure: bool,
) -> list[TaskDiag]:
    task_ids = sorted(set(default_diags) | set(stress_diags))
    candidates: list[tuple[TaskDiag, TaskDiag]] = []
    for task_id in task_ids:
        if task_id in source_task_ids:
            continue
        default = default_diags.get(task_id, empty_diag(task_id, "heldout", "default"))
        stress = stress_diags.get(task_id, empty_diag(task_id, "heldout", "stress_no_prior"))
        if require_default_failure and default.certified:
            continue
        if require_stress_failure and stress.certified:
            continue
        if stress.proposals_train_consistent_pre_cap < min_train_consistent:
            continue
        if not stress.has_search_pressure:
            continue
        candidates.append((default, stress))

    candidates.sort(
        key=lambda pair: (
            pair[0].certified,
            pair[1].certified,
            -pair[1].proposals_train_consistent_pre_cap,
            -int(pair[1].proposal_cap_hit),
            -pair[1].proposals_generated,
            -pair[1].rejected_evidence,
            pair[1].task_id,
        )
    )
    return [stress for _, stress in candidates[:limit]]


def selection_row(
    role: str,
    diag: TaskDiag,
    default_diag: TaskDiag | None = None,
    filtered_task_id: str | None = None,
) -> dict[str, Any]:
    row = {
        "schema": "crlp.arc_public_hard_subset.selection.v1",
        "report_version": 1,
        "role": role,
        "task_id": diag.task_id,
        "filtered_task_id": filtered_task_id or diag.task_id,
        "split": diag.split,
        "mode": diag.mode,
        "certified": diag.certified,
        "certified_evidence": diag.certified_evidence,
        "rejected_evidence": diag.rejected_evidence,
        "selected_templates": list(diag.selected_templates),
        "certified_templates": list(diag.certified_templates),
        "train_consistent_templates": list(diag.train_consistent_templates),
        "proposals_generated": diag.proposals_generated,
        "proposals_selected": diag.proposals_selected,
        "proposals_train_consistent_pre_cap": diag.proposals_train_consistent_pre_cap,
        "proposal_cap_hit": diag.proposal_cap_hit,
    }
    if default_diag is not None:
        row.update(
            {
                "default_certified": default_diag.certified,
                "default_certified_evidence": default_diag.certified_evidence,
                "default_selected_templates": list(default_diag.selected_templates),
                "default_proposals_train_consistent_pre_cap": (
                    default_diag.proposals_train_consistent_pre_cap
                ),
            }
        )
    return row


def copy_selected_tasks(
    output_dir: Path,
    source_split: SplitSpec,
    heldout_split: SplitSpec,
    source_tasks: list[TaskDiag],
    heldout_tasks: list[TaskDiag],
) -> tuple[Path, dict[tuple[str, str], str]]:
    filtered_dir = output_dir / "filtered_tasks"
    filtered_dir.mkdir(parents=True, exist_ok=True)
    source_files = task_files(source_split.path)
    heldout_files = task_files(heldout_split.path)
    copied: dict[str, Path] = {}
    filtered_ids: dict[tuple[str, str], str] = {}

    def copy_one(role: str, task_id: str, files: dict[str, Path]) -> None:
        src = files.get(task_id)
        if src is None:
            raise RuntimeError(f"could not find JSON file for selected task {task_id}")
        filtered_id = task_id
        prior = copied.get(filtered_id)
        if prior is not None and prior != src:
            filtered_id = f"{role}__{task_id}"
            suffix = 2
            while filtered_id in copied and copied[filtered_id] != src:
                filtered_id = f"{role}{suffix}__{task_id}"
                suffix += 1
        shutil.copy2(src, filtered_dir / f"{filtered_id}.json")
        copied[filtered_id] = src
        filtered_ids[(role, task_id)] = filtered_id

    for diag in source_tasks:
        copy_one("source", diag.task_id, source_files)
    for diag in heldout_tasks:
        copy_one("heldout", diag.task_id, heldout_files)
    return filtered_dir, filtered_ids


def write_task_ids(
    path: Path,
    tasks: list[TaskDiag],
    role: str,
    filtered_ids: dict[tuple[str, str], str],
) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for task in tasks:
            fp.write(filtered_ids.get((role, task.task_id), task.task_id))
            fp.write("\n")


def write_summary(
    path: Path,
    source_split: SplitSpec,
    heldout_split: SplitSpec,
    source_tasks: list[TaskDiag],
    heldout_tasks: list[TaskDiag],
    source_candidates: int,
    heldout_candidates: int,
    default_diags: dict[str, TaskDiag],
    stress_diags: dict[str, TaskDiag],
    filtered_dir: Path,
    filtered_ids: dict[tuple[str, str], str],
    args: argparse.Namespace,
) -> None:
    default_certified = sum(1 for diag in default_diags.values() if diag.certified)
    stress_certified = sum(1 for diag in stress_diags.values() if diag.certified)
    selected_default_certified = sum(
        1
        for diag in heldout_tasks
        if default_diags.get(diag.task_id, empty_diag(diag.task_id, "heldout", "default")).certified
    )
    selected_stress_certified = sum(1 for diag in heldout_tasks if diag.certified)
    source_arg = ",".join(
        filtered_ids.get(("source", task.task_id), task.task_id)
        for task in source_tasks
    )
    heldout_arg = ",".join(
        filtered_ids.get(("heldout", task.task_id), task.task_id)
        for task in heldout_tasks
    )

    lines = [
        "# ARC Public Hard Subset",
        "",
        "This filter keeps the public experiment from saturating before the library",
        "feedback loop has anything useful to explain.",
        "",
        "## Inputs",
        "",
        f"- Source split: `{source_split.path}`",
        f"- Held-out split: `{heldout_split.path}`",
        f"- Source max tasks: `{source_split.max_tasks}`",
        f"- Held-out max tasks: `{heldout_split.max_tasks}`",
        "",
        "## Selection",
        "",
        f"- Source candidates with certified evidence: `{source_candidates}`",
        f"- Selected source tasks: `{len(source_tasks)}`",
        f"- Held-out hard candidates: `{heldout_candidates}`",
        f"- Selected held-out tasks: `{len(heldout_tasks)}`",
        f"- Default held-out certified before filtering: `{default_certified}/{len(default_diags)}`",
        f"- Stress no-prior certified before filtering: `{stress_certified}/{len(stress_diags)}`",
        f"- Selected held-out default certified: `{selected_default_certified}/{len(heldout_tasks)}`",
        f"- Selected held-out stress certified: `{selected_stress_certified}/{len(heldout_tasks)}`",
        f"- Require default failure: `{bool(args.heldout_require_default_failure)}`",
        f"- Default-failure fallback used: `{bool(getattr(args, 'heldout_default_failure_fallback_used', False))}`",
        f"- Require stress failure: `{not bool(args.allow_stress_certified_heldout)}`",
        "",
        "## Selected Source Tasks",
        "",
    ]
    if source_tasks:
        for task in source_tasks:
            filtered_id = filtered_ids.get(("source", task.task_id), task.task_id)
            suffix = "" if filtered_id == task.task_id else f" -> `{filtered_id}`"
            lines.append(f"- `{task.task_id}`{suffix}: {', '.join(task.certified_templates)}")
    else:
        lines.append("- none")
    lines.extend(["", "## Selected Held-out Tasks", ""])
    if heldout_tasks:
        for task in heldout_tasks:
            default = default_diags.get(task.task_id, empty_diag(task.task_id, "heldout", "default"))
            filtered_id = filtered_ids.get(("heldout", task.task_id), task.task_id)
            display_id = f"`{task.task_id}`"
            if filtered_id != task.task_id:
                display_id = f"`{task.task_id}` -> `{filtered_id}`"
            lines.append(
                "- {}: default={}, stress={}, train_consistent_pre_cap={}, templates={}".format(
                    display_id,
                    "cert" if default.certified else "fail",
                    "cert" if task.certified else "fail",
                    task.proposals_train_consistent_pre_cap,
                    ", ".join(task.train_consistent_templates) or "none",
                )
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Next Command",
            "",
            "```sh",
            "python3 scripts/hypothesis_library_split_sweep.py \\",
            "  --arc-agi ./build/float/arc_agi \\",
            f"  --task-dir {filtered_dir} \\",
            f"  --source-task-ids {source_arg} \\",
            f"  --heldout-task-ids {heldout_arg} \\",
            "  --stress-no-prior \\",
            "  --output-dir build/float/hypothesis_library_public_filtered",
            "```",
            "",
            "## Outputs",
            "",
            "- `source_raw.jsonl`",
            "- `heldout_default_raw.jsonl`",
            "- `heldout_stress_no_prior_raw.jsonl`",
            "- `diagnostics.jsonl`",
            "- `selected_tasks.jsonl`",
            "- `source_task_ids.txt`",
            "- `heldout_task_ids.txt`",
            "- `filtered_tasks/`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arc-agi", type=Path, required=True)
    parser.add_argument("--public-arc-dir", type=Path)
    parser.add_argument("--task-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("build/float/hypothesis_library_public_filter"))
    parser.add_argument("--max-tasks", type=int, default=100)
    parser.add_argument("--source-max-tasks", type=int)
    parser.add_argument("--heldout-max-tasks", type=int)
    parser.add_argument("--source-count", type=int, default=24)
    parser.add_argument("--heldout-count", type=int, default=48)
    parser.add_argument("--min-train-consistent", type=int, default=1)
    parser.add_argument(
        "--heldout-require-default-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require selected held-out tasks to fail the default strict no-library pass.",
    )
    parser.add_argument(
        "--allow-stress-certified-heldout",
        action="store_true",
        help="Allow tasks already certified by the stress no-prior baseline into held-out.",
    )
    parser.add_argument(
        "--task-ids",
        help="Comma-separated task ids for fixture/single-dir smoke runs.",
    )
    args = parser.parse_args()

    if bool(args.public_arc_dir) == bool(args.task_dir):
        parser.error("provide exactly one of --public-arc-dir or --task-dir")

    source_max = args.source_max_tasks or args.max_tasks
    heldout_max = args.heldout_max_tasks or args.max_tasks
    task_ids = [part.strip() for part in (args.task_ids or "").split(",") if part.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.public_arc_dir:
        train_dir, eval_dir = public_split_dirs(args.public_arc_dir)
        source_split = SplitSpec("public_training", train_dir, source_max)
        heldout_split = SplitSpec("public_evaluation", eval_dir, heldout_max)
        source_task_ids = None
        heldout_task_ids = None
    else:
        task_dir = args.task_dir
        if task_dir is None:
            parser.error("--task-dir is required without --public-arc-dir")
        source_split = SplitSpec("fixture_source", task_dir, source_max)
        heldout_split = SplitSpec("fixture_heldout", task_dir, heldout_max)
        source_task_ids = task_ids or None
        heldout_task_ids = task_ids or None

    source_raw = args.output_dir / "source_raw.jsonl"
    heldout_default_raw = args.output_dir / "heldout_default_raw.jsonl"
    heldout_stress_raw = args.output_dir / "heldout_stress_no_prior_raw.jsonl"

    run_arc(
        args.arc_agi,
        source_split,
        source_raw,
        ["--program-budget-grid", "--loader-summary-only", "--induction"],
        source_task_ids,
    )
    run_arc(
        args.arc_agi,
        heldout_split,
        heldout_default_raw,
        STRICT_EVAL_FLAGS,
        heldout_task_ids,
    )
    run_arc(
        args.arc_agi,
        heldout_split,
        heldout_stress_raw,
        STRESS_NO_PRIOR_FLAGS,
        heldout_task_ids,
    )

    source_diags = diagnose(read_jsonl_required(source_raw), source_split.name, "promotion")
    default_diags = diagnose(read_jsonl_required(heldout_default_raw), heldout_split.name, "default")
    stress_diags = diagnose(read_jsonl_required(heldout_stress_raw), heldout_split.name, "stress_no_prior")

    source_tasks = select_sources(source_diags, args.source_count)
    source_ids = {task.task_id for task in source_tasks}
    args.heldout_default_failure_fallback_used = False
    heldout_candidates = select_heldout(
        default_diags,
        stress_diags,
        limit=max(args.heldout_count, len(stress_diags)),
        source_task_ids=source_ids if source_split.path == heldout_split.path else set(),
        min_train_consistent=args.min_train_consistent,
        require_default_failure=args.heldout_require_default_failure,
        require_stress_failure=not args.allow_stress_certified_heldout,
    )
    if not heldout_candidates and args.heldout_require_default_failure:
        heldout_candidates = select_heldout(
            default_diags,
            stress_diags,
            limit=max(args.heldout_count, len(stress_diags)),
            source_task_ids=source_ids if source_split.path == heldout_split.path else set(),
            min_train_consistent=args.min_train_consistent,
            require_default_failure=False,
            require_stress_failure=not args.allow_stress_certified_heldout,
        )
        args.heldout_default_failure_fallback_used = bool(heldout_candidates)
    heldout_tasks = heldout_candidates[: args.heldout_count]

    if not source_tasks:
        raise RuntimeError("no source tasks produced certified hypothesis evidence")
    if not heldout_tasks:
        raise RuntimeError(
            "no hard held-out tasks selected; try --no-heldout-require-default-failure "
            "or lower --min-train-consistent"
        )

    filtered_dir, filtered_ids = copy_selected_tasks(
        args.output_dir,
        source_split,
        heldout_split,
        source_tasks,
        heldout_tasks,
    )
    write_task_ids(args.output_dir / "source_task_ids.txt", source_tasks, "source", filtered_ids)
    write_task_ids(args.output_dir / "heldout_task_ids.txt", heldout_tasks, "heldout", filtered_ids)

    diagnostic_rows = [
        diag_row(diag)
        for diag in [
            *source_diags.values(),
            *default_diags.values(),
            *stress_diags.values(),
        ]
    ]
    write_jsonl(args.output_dir / "diagnostics.jsonl", diagnostic_rows, compact=True)

    selected_rows = [
        *(
            selection_row(
                "source",
                task,
                filtered_task_id=filtered_ids.get(("source", task.task_id)),
            )
            for task in source_tasks
        ),
        *(
            selection_row(
                "heldout",
                task,
                default_diags.get(task.task_id),
                filtered_task_id=filtered_ids.get(("heldout", task.task_id)),
            )
            for task in heldout_tasks
        ),
    ]
    write_jsonl(args.output_dir / "selected_tasks.jsonl", selected_rows, compact=True)
    write_summary(
        args.output_dir / "summary.md",
        source_split,
        heldout_split,
        source_tasks,
        heldout_tasks,
        sum(1 for diag in source_diags.values() if diag.certified),
        len(heldout_candidates),
        default_diags,
        stress_diags,
        filtered_dir,
        filtered_ids,
        args,
    )

    print(f"wrote {args.output_dir / 'summary.md'}")
    print(f"wrote {args.output_dir / 'selected_tasks.jsonl'}")
    print(f"wrote {filtered_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
