#!/usr/bin/env python3
"""Append C-emitted ARC probe attempts to a ranker attempt TSV."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from crlp.io import write_jsonl


SCHEMA = "crlp.arc_probe_attempts_tsv_append.v1"
ATTEMPT_SCHEMA = "crlp.neural_grammar_discovery.arc_visual_pipeline_attempt.v1"

RUNTIME_FIELDS = [
    "split_name",
    "family",
    "task_id",
    "split",
    "attempt_id",
    "library_id",
    "attempt_index",
    "pipeline_kind",
    "op_name",
    "visual_program",
    "slot_contract",
    "composition_kind",
    "normalization_frame",
    "typed_stages",
    "primitive_contract",
    "slot_tokens",
    "accepted",
    "rejection_reason",
    "failure_stage",
    "failure_detail",
    "proposal_count",
    "support_tasks",
    "utility_mean",
    "selector",
    "background",
    "color",
    "output_color",
    "selector_kind",
    "include_background",
    "orientation",
    "count_mode",
    "x_align",
    "y_align",
    "x_anchor",
    "y_anchor",
    "x_offset",
    "y_offset",
    "width",
    "height",
    "period",
    "contrast_accepted_attempt_id",
    "contrast_slot_distance",
    "contrast_pair_kind",
    "contrast_rejected_sibling_count",
    "input_width",
    "input_height",
    "output_width",
    "output_height",
    "shape_scale_x",
    "shape_scale_y",
    "period_divides_input_width",
    "period_divides_input_height",
    "slot_width_matches_output",
    "slot_height_matches_output",
    "color_in_input",
    "color_in_output",
    "output_color_in_output",
    "background_in_input",
    "verifier_train_pairs",
    "verifier_checked_cells",
    "verifier_mismatches",
    "verifier_unsupported_pairs",
    "verifier_apply_failed_pairs",
    "verifier_shape_mismatch_pairs",
    "verifier_missing_output_pairs",
    "verifier_first_failing_pair",
    "verifier_mismatch_ratio",
    "verifier_score",
    "input_component_count",
    "selected_component_count",
    "selected_component_area_min",
    "selected_component_area_max",
    "selected_component_area_total",
    "selected_component_same_shape",
    "selected_component_shape_count",
    "selected_component_bbox_width",
    "selected_component_bbox_height",
    "selected_bbox_matches_output_shape",
    "selected_bbox_rotated_output_shape",
    "output_foreground_cell_count",
    "output_component_count",
    "output_non_background_color_count",
    "selected_count_matches_output_components",
    "selected_count_matches_output_foreground",
    "selected_area_matches_output_foreground",
]


def tsv_value(value: Any) -> str:
    if value is None:
        text = ""
    elif isinstance(value, bool):
        text = "1" if value else "0"
    else:
        text = str(value)
    return text.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def bool_int(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, str):
        return 1 if value.strip().lower() in {"1", "true", "yes", "y"} else 0
    return 1 if value else 0


def slot_tokens(row: dict[str, Any]) -> str:
    slots = row.get("slot_values")
    if not isinstance(slots, dict):
        return ""
    return "|".join(
        f"{tsv_value(key).replace('|', '/')}={tsv_value(slots[key]).replace('|', '/')}"
        for key in sorted(slots)
    )


def iter_attempts(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            row = json.loads(line)
            if row.get("schema") == ATTEMPT_SCHEMA:
                yield row


def attempt_key_from_parts(parts: list[str], indexes: dict[str, int]) -> tuple[str, str, str, str, str]:
    def cell(name: str) -> str:
        index = indexes[name]
        return parts[index] if index < len(parts) else ""

    return (
        cell("split_name"),
        cell("family"),
        cell("task_id"),
        cell("split"),
        cell("library_id"),
    )


def row_key(split_name: str, row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        split_name,
        str(row.get("family", "")),
        str(row.get("task_id", "")),
        str(row.get("split", "")),
        str(row.get("library_id", "")),
    )


def tsv_fields(split_name: str, row: dict[str, Any]) -> list[Any]:
    pipeline = row.get("pipeline") if isinstance(row.get("pipeline"), dict) else {}
    features = (
        row.get("task_slot_features")
        if isinstance(row.get("task_slot_features"), dict)
        else {}
    )
    verifier = (
        row.get("verifier_distance")
        if isinstance(row.get("verifier_distance"), dict)
        else {}
    )
    component = (
        row.get("component_inventory")
        if isinstance(row.get("component_inventory"), dict)
        else {}
    )
    return [
        split_name,
        row.get("family", ""),
        row.get("task_id", ""),
        row.get("split", ""),
        row.get("attempt_id", ""),
        row.get("library_id", ""),
        row.get("attempt_index", ""),
        row.get("pipeline_kind", ""),
        row.get("op_name", ""),
        row.get("visual_program", ""),
        row.get("slot_contract", ""),
        row.get("composition_kind", pipeline.get("composition_kind", "")),
        row.get("normalization_frame", pipeline.get("normalization_frame", "")),
        row.get("typed_stages", pipeline.get("typed_stages", "")),
        row.get("primitive_contract", pipeline.get("primitive_contract", "")),
        slot_tokens(row),
        bool_int(row.get("accepted")),
        row.get("rejection_reason", ""),
        row.get("failure_stage", ""),
        row.get("failure_detail", ""),
        row.get("proposal_count", ""),
        row.get("support_tasks", ""),
        row.get("utility_mean", ""),
        pipeline.get("selector", ""),
        pipeline.get("background", ""),
        pipeline.get("color", ""),
        pipeline.get("output_color", ""),
        pipeline.get("selector_kind", ""),
        bool_int(pipeline.get("include_background")),
        pipeline.get("orientation", ""),
        pipeline.get("count_mode", ""),
        pipeline.get("x_align", ""),
        pipeline.get("y_align", ""),
        pipeline.get("x_anchor", ""),
        pipeline.get("y_anchor", ""),
        pipeline.get("x_offset", ""),
        pipeline.get("y_offset", ""),
        pipeline.get("width", ""),
        pipeline.get("height", ""),
        pipeline.get("period", ""),
        row.get("contrast_accepted_attempt_id", ""),
        row.get("contrast_slot_distance", ""),
        row.get("contrast_pair_kind", ""),
        row.get("contrast_rejected_sibling_count", ""),
        features.get("input_width", ""),
        features.get("input_height", ""),
        features.get("output_width", ""),
        features.get("output_height", ""),
        features.get("shape_scale_x", ""),
        features.get("shape_scale_y", ""),
        bool_int(features.get("period_divides_input_width")),
        bool_int(features.get("period_divides_input_height")),
        bool_int(features.get("slot_width_matches_output")),
        bool_int(features.get("slot_height_matches_output")),
        bool_int(features.get("color_in_input")),
        bool_int(features.get("color_in_output")),
        bool_int(features.get("output_color_in_output")),
        bool_int(features.get("background_in_input")),
        verifier.get("train_pairs", ""),
        verifier.get("checked_cells", ""),
        verifier.get("mismatches", ""),
        verifier.get("unsupported_pairs", ""),
        verifier.get("apply_failed_pairs", ""),
        verifier.get("shape_mismatch_pairs", ""),
        verifier.get("missing_output_pairs", ""),
        verifier.get("first_failing_pair", ""),
        verifier.get("mismatch_ratio", ""),
        verifier.get("verifier_score", ""),
        component.get("input_component_count", ""),
        component.get("selected_component_count", ""),
        component.get("selected_component_area_min", ""),
        component.get("selected_component_area_max", ""),
        component.get("selected_component_area_total", ""),
        bool_int(component.get("selected_component_same_shape")),
        component.get("selected_component_shape_count", ""),
        component.get("selected_component_bbox_width", ""),
        component.get("selected_component_bbox_height", ""),
        bool_int(component.get("selected_bbox_matches_output_shape")),
        bool_int(component.get("selected_bbox_rotated_output_shape")),
        component.get("output_foreground_cell_count", ""),
        component.get("output_component_count", ""),
        component.get("output_non_background_color_count", ""),
        bool_int(component.get("selected_count_matches_output_components")),
        bool_int(component.get("selected_count_matches_output_foreground")),
        bool_int(component.get("selected_area_matches_output_foreground")),
    ]


def append(args: argparse.Namespace) -> dict[str, Any]:
    args.combined_attempts_tsv.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str, str, str, str]] = set()
    base_rows = 0
    appended = 0
    skipped_duplicate = 0
    skipped_rejected = 0
    appended_by_acceptance: Counter[str] = Counter()
    appended_by_reason: Counter[str] = Counter()
    appended_by_task: Counter[str] = Counter()

    with args.base_attempts_tsv.open(encoding="utf-8") as source, (
        args.combined_attempts_tsv.open("w", encoding="utf-8")
    ) as target:
        header_line = source.readline()
        if not header_line:
            raise ValueError(f"empty base TSV: {args.base_attempts_tsv}")
        header = header_line.rstrip("\n").split("\t")
        if header != RUNTIME_FIELDS:
            raise ValueError("base attempt TSV header does not match ARC runtime fields")
        indexes = {name: index for index, name in enumerate(header)}
        target.write(header_line)
        for line in source:
            target.write(line)
            base_rows += 1
            parts = line.rstrip("\n").split("\t")
            seen.add(attempt_key_from_parts(parts, indexes))

        for row in iter_attempts(args.probe_attempts_jsonl):
            accepted = bool_int(row.get("accepted"))
            if args.accepted_only and not accepted:
                skipped_rejected += 1
                continue
            key = row_key(args.split_name, row)
            if args.dedupe and key in seen:
                skipped_duplicate += 1
                continue
            seen.add(key)
            fields = tsv_fields(args.split_name, row)
            target.write("\t".join(tsv_value(field) for field in fields))
            target.write("\n")
            appended += 1
            appended_by_acceptance["accepted" if accepted else "rejected"] += 1
            appended_by_reason[str(row.get("rejection_reason", "") or "unknown")] += 1
            appended_by_task[str(row.get("task_id", ""))] += 1

    summary = {
        "schema": SCHEMA,
        "report_version": 1,
        "base_attempts_tsv": str(args.base_attempts_tsv),
        "probe_attempts_jsonl": str(args.probe_attempts_jsonl),
        "combined_attempts_tsv": str(args.combined_attempts_tsv),
        "split_name": args.split_name,
        "dedupe": args.dedupe,
        "accepted_only": args.accepted_only,
        "base_rows": base_rows,
        "appended_rows": appended,
        "skipped_duplicate_rows": skipped_duplicate,
        "skipped_rejected_rows": skipped_rejected,
        "appended_by_acceptance": dict(sorted(appended_by_acceptance.items())),
        "appended_by_reason": dict(sorted(appended_by_reason.items())),
        "appended_task_count": len(appended_by_task),
        "appended_rows_by_task": dict(sorted(appended_by_task.items())),
    }
    write_jsonl(args.summary, [summary], compact=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-attempts-tsv", type=Path, required=True)
    parser.add_argument("--probe-attempts-jsonl", type=Path, required=True)
    parser.add_argument("--combined-attempts-tsv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--split-name", default="target")
    parser.add_argument("--accepted-only", action="store_true")
    parser.add_argument("--no-dedupe", dest="dedupe", action="store_false")
    parser.set_defaults(dedupe=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.base_attempts_tsv.is_file():
        raise SystemExit(f"missing base TSV: {args.base_attempts_tsv}")
    if not args.probe_attempts_jsonl.is_file():
        raise SystemExit(f"missing probe attempts: {args.probe_attempts_jsonl}")
    summary = append(args)
    print(
        "arc-append-probe-attempts-tsv "
        f"base_rows={summary['base_rows']} "
        f"appended={summary['appended_rows']} "
        f"combined={args.combined_attempts_tsv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
