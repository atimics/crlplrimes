#!/usr/bin/env python3
"""Build ARC visual-only curriculum rows from verified visual-program wins."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


EXAMPLE_SCHEMA = "crlp.neural_grammar_discovery.arc_visual_curriculum_example.v1"
SUMMARY_SCHEMA = "crlp.neural_grammar_discovery.arc_visual_curriculum_summary.v1"
CANDIDATE_HEADER = [
    "family",
    "task_id",
    "split",
    "support_tasks",
    "proposal_count",
    "utility_mean",
    "op_hints",
    "coverage_frontier_id",
    "parent_attempt_id",
    "repair_operator",
    "dominant_residual_class",
    "dominant_failure_stage",
    "repair_strategy",
]

OP_NAMES = {
    0: "crop",
    1: "tile_repeat",
    2: "macrocell_substitute",
    3: "lattice_sample",
    4: "component_color_summary",
    5: "masked_or_normalized_crop",
    6: "point_line_project",
    7: "object_mask_extract",
    8: "axis_projection_compress",
    9: "separator_panel_map",
    10: "palette_scale",
    11: "object_color_tile",
    12: "marker_count_motif_tile",
    13: "component_count_line",
    14: "recolor_crop",
    15: "marker_polyline",
    16: "separator_intersection_color",
    17: "separator_panel_relation_mask",
    18: "visual_combinator",
}

SELECTOR_NAMES = {
    0: "grid_panel",
    1: "foreground_bbox_panel",
    2: "color_bbox",
    3: "lattice",
}


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


def parse_scoped_path(raw: str, default_scope: str) -> tuple[str, Path]:
    if "=" in raw:
        scope, path = raw.split("=", 1)
        if scope and path:
            return scope, Path(path)
    return default_scope, Path(raw)


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


def slots_by_name(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    slots: dict[str, dict[str, Any]] = {}
    for slot in row.get("slots", []) or []:
        if isinstance(slot, dict) and slot.get("name"):
            slots[str(slot["name"])] = slot
    return slots


def split_for_scope(scope: str) -> str | None:
    lowered = scope.lower()
    if "train" in lowered:
        return "public_training"
    if "eval" in lowered:
        return "public_evaluation"
    return None


def visual_template_stats(row: dict[str, Any]) -> dict[str, Any]:
    for template in row.get("templates", []) or []:
        if isinstance(template, dict) and template.get("name") == "arc_sketch_program":
            return template
    return {}


def extract_positive_examples(
    scope: str,
    path: Path,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    positives: list[dict[str, Any]] = []
    split = split_for_scope(scope)
    for row in rows:
        if (
            row.get("schema") != "crlp.hypothesis.evidence.v1"
            or row.get("template_name") != "arc_sketch_program"
            or row.get("certificate_status") != "certified"
        ):
            continue
        slots = slots_by_name(row)
        op_id = safe_int(slots.get("sketch.op", {}).get("integer"), -1)
        selector_id = safe_int(slots.get("sketch.selector", {}).get("integer"), -1)
        slot_values = {
            name: safe_int(slot.get("integer"))
            for name, slot in sorted(slots.items())
            if name.startswith("sketch.")
        }
        positives.append(
            {
                "schema": EXAMPLE_SCHEMA,
                "report_version": 1,
                "kind": "verified_visual_positive",
                "source_scope": scope,
                "source_path": str(path),
                "split": split,
                "task_id": str(row.get("task_instance", "")),
                "template_name": "arc_sketch_program",
                "op_id": op_id,
                "op_name": OP_NAMES.get(op_id, f"unknown_{op_id}"),
                "selector_id": selector_id,
                "selector_name": SELECTOR_NAMES.get(selector_id, f"unknown_{selector_id}"),
                "program_canonical_hash": row.get("program_canonical_hash"),
                "support": safe_int(row.get("support")),
                "event_count": safe_int(row.get("event_count")),
                "covered_events": safe_int(row.get("covered_events")),
                "utility": safe_float(row.get("utility")),
                "slot_values": slot_values,
            }
        )
    return positives


def extract_target_rows(
    scope: str,
    path: Path,
    rows: list[dict[str, Any]],
    op_hints: list[str],
    target_scopes: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    examples: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, str]] = []
    split = split_for_scope(scope)
    is_target_scope = scope in target_scopes
    for row in rows:
        if row.get("schema") != "crlp.arc_hypothesis.selection.v1":
            continue
        task_id = str(row.get("task_instance", ""))
        visual = visual_template_stats(row)
        attempted = safe_int(visual.get("attempted"))
        train_consistent = safe_int(visual.get("train_consistent_pre_cap"))
        retained = safe_int(visual.get("train_consistent_retained"))
        selected = safe_int(visual.get("selected"))
        if selected > 0:
            kind = "visual_target_already_selected"
        elif train_consistent > 0:
            kind = "visual_target_train_consistent_not_selected"
        elif attempted > 0:
            kind = "visual_target_no_train_match"
        else:
            kind = "visual_target_no_attempts"
        examples.append(
            {
                "schema": EXAMPLE_SCHEMA,
                "report_version": 1,
                "kind": kind,
                "source_scope": scope,
                "source_path": str(path),
                "split": split,
                "task_id": task_id,
                "template_name": "arc_sketch_program",
                "attempted": attempted,
                "train_consistent_pre_cap": train_consistent,
                "train_consistent_retained": retained,
                "selected": selected,
                "proposed_op_hints": op_hints,
            }
        )
        if not is_target_scope or split is None or not op_hints:
            continue
        candidate_rows.append(
            {
                "family": "arc.visual_curriculum_transfer_family",
                "task_id": task_id,
                "split": split,
                "support_tasks": "1",
                "proposal_count": "64",
                "utility_mean": "1.00000000",
                "op_hints": ",".join(op_hints),
                "coverage_frontier_id": f"arc_visual_curriculum:{scope}",
                "parent_attempt_id": "",
                "repair_operator": "visual_only_positive_op_transfer",
                "dominant_residual_class": kind,
                "dominant_failure_stage": "coverage",
                "repair_strategy": "replay_verified_visual_ops",
            }
        )
    return examples, candidate_rows


def load_attempt_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    accepted = 0
    rejected = 0
    by_op: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    accepted_tasks: set[str] = set()
    for row in load_jsonl(path):
        if row.get("schema") != "crlp.neural_grammar_discovery.arc_visual_pipeline_attempt.v1":
            continue
        op = str(row.get("op_name", "unknown"))
        if row.get("accepted") is True:
            accepted += 1
            accepted_tasks.add(str(row.get("task_id", "")))
            by_op[op] += 1
        else:
            rejected += 1
            reason = str(row.get("failure_stage") or row.get("reason") or "rejected")
            by_reason[reason] += 1
    return {
        "attempt_log": str(path),
        "accepted_attempts": accepted,
        "rejected_attempts": rejected,
        "accepted_tasks": sorted(task for task in accepted_tasks if task),
        "accepted_by_op": dict(sorted(by_op.items())),
        "rejected_by_reason": dict(sorted(by_reason.items())),
    }


def write_candidate_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        fp.write("\t".join(CANDIDATE_HEADER) + "\n")
        for row in rows:
            fp.write("\t".join(row.get(field, "") for field in CANDIDATE_HEADER) + "\n")


def build_curriculum(args: argparse.Namespace) -> int:
    scoped_logs = [
        parse_scoped_path(raw, f"arc_visual_only_{index}")
        for index, raw in enumerate(args.visual_only_log or [], start=1)
    ]
    if not scoped_logs:
        print("at least one --visual-only-log is required", file=sys.stderr)
        return 2

    loaded_logs: list[tuple[str, Path, list[dict[str, Any]]]] = [
        (scope, path, load_jsonl(path)) for scope, path in scoped_logs
    ]
    positives: list[dict[str, Any]] = []
    for scope, path, rows in loaded_logs:
        positives.extend(extract_positive_examples(scope, path, rows))

    op_counts: Counter[str] = Counter(
        str(row.get("op_name", ""))
        for row in positives
        if row.get("op_name") and not str(row.get("op_name", "")).startswith("unknown_")
    )
    op_hints = [op for op, _count in op_counts.most_common(args.max_op_hints)]
    op_hints.sort()

    target_scopes = set(args.target_scope or [])
    examples: list[dict[str, Any]] = list(positives)
    candidate_rows: list[dict[str, str]] = []
    target_task_count = 0
    for scope, path, rows in loaded_logs:
        target_examples, target_candidates = extract_target_rows(
            scope,
            path,
            rows,
            op_hints,
            target_scopes,
        )
        examples.extend(target_examples)
        candidate_rows.extend(target_candidates)
        if scope in target_scopes:
            target_task_count += len(target_candidates)

    attempt_summary = load_attempt_summary(args.attempt_log)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "report_version": 1,
        "source_logs": [
            {"scope": scope, "path": str(path), "rows": len(rows)}
            for scope, path, rows in loaded_logs
        ],
        "target_scopes": sorted(target_scopes),
        "positive_count": len(positives),
        "positive_task_count": len({str(row.get("task_id")) for row in positives}),
        "positive_ops": dict(sorted(op_counts.items())),
        "op_hints": op_hints,
        "target_task_count": target_task_count,
        "candidate_count": len(candidate_rows),
        "candidate_tsv": str(args.candidate_tsv),
        "example_jsonl": str(args.examples),
        "attempt_summary": attempt_summary,
    }

    write_candidate_tsv(args.candidate_tsv, candidate_rows)
    write_jsonl(args.examples, examples)
    write_jsonl(args.summary, [summary])
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--visual-only-log",
        action="append",
        help="Visual-only ARC JSONL log, optionally scope=path.",
    )
    parser.add_argument(
        "--target-scope",
        action="append",
        default=[],
        help="Scope whose tasks should receive curriculum candidate rows.",
    )
    parser.add_argument("--attempt-log", type=Path)
    parser.add_argument("--candidate-tsv", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--max-op-hints", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    return build_curriculum(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
