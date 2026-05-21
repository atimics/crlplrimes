#!/usr/bin/env python3
"""Propose ARC visual-program hole candidates from accepted/rejected attempts.

The script learns a small, auditable proposal surface from C-emitted attempt
logs. It emits candidate rows only; `arc_agi` still compiles and verifies every
candidate before any prior row is exported.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from crlp.io import read_jsonl_lenient as load_jsonl
from crlp.io import safe_finite_float as safe_float
from crlp.io import safe_int, write_jsonl

EXAMPLE_SCHEMA = "crlp.neural_grammar_discovery.arc_visual_hole_example.v1"
SUMMARY_SCHEMA = "crlp.neural_grammar_discovery.arc_visual_hole_summary.v1"

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
OP_IDS = {name: op for op, name in OP_NAMES.items()}

SLOT_TO_PIN = {
    "sketch.background": "background",
    "sketch.color": "color",
    "sketch.height": "height",
    "sketch.op": "op",
    "sketch.output_color": "output_color",
    "sketch.period": "period",
    "sketch.selector": "selector",
    "sketch.width": "width",
    "sketch.x_anchor": "x_anchor",
    "sketch.x_offset": "x_offset",
    "sketch.y_anchor": "y_anchor",
    "sketch.y_offset": "y_offset",
}

STRUCTURAL_PINS = [
    "sketch.selector",
    "sketch.x_anchor",
    "sketch.y_anchor",
    "sketch.x_offset",
    "sketch.y_offset",
    "sketch.period",
]

GEOMETRY_PINS = [
    "sketch.selector",
    "sketch.x_anchor",
    "sketch.y_anchor",
    "sketch.x_offset",
    "sketch.y_offset",
    "sketch.period",
    "sketch.width",
    "sketch.height",
]

STAGE_REPAIR_TOKENS = {
    "selector_or_transform": [
        "counterexample_selector_phase_repair",
        "selector:component_predicate",
        "normalizer:component_selector_layout",
    ],
    "renderer_shape": [
        "counterexample_shape_frame_repair",
        "normalizer:object_lattice",
        "object_lattice_frame",
    ],
    "renderer_cells": [
        "counterexample_cell_mask_cleanup_repair",
        "emitter:component_mask",
        "normalizer:component_mask",
    ],
    "transform_or_color": [
        "counterexample_transform_color_repair",
        "emitter:component_mask",
    ],
    "unsupported_shape": [
        "counterexample_shape_frame_repair",
    ],
}


def parse_scoped_path(raw: str, default_scope: str) -> tuple[str, Path]:
    if "=" in raw:
        scope, path = raw.split("=", 1)
        if scope and path:
            return scope, Path(path)
    return default_scope, Path(raw)


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


def task_kind(visual: dict[str, Any]) -> str:
    selected = safe_int(visual.get("selected"))
    train_consistent = safe_int(visual.get("train_consistent_pre_cap"))
    attempted = safe_int(visual.get("attempted"))
    if selected > 0:
        return "visual_target_already_selected"
    if train_consistent > 0:
        return "visual_target_train_consistent_not_selected"
    if attempted > 0:
        return "visual_target_no_train_match"
    return "visual_target_no_attempts"


def clean_field(value: str) -> str:
    return value.replace("\t", " ").replace("\n", " ").strip()


def compact_tokens(tokens: list[str], max_len: int = 240) -> str:
    seen: set[str] = set()
    kept: list[str] = []
    for token in tokens:
        token = clean_field(token)
        if not token or token in seen:
            continue
        trial = ",".join(kept + [token])
        if len(trial) > max_len:
            continue
        seen.add(token)
        kept.append(token)
    return ",".join(kept)


def pin_token(pin_name: str, value: Any) -> str | None:
    if value is None:
        return None
    return f"pin:{pin_name}={safe_int(value)}"


def slot_pins(slots: dict[str, Any], slot_names: list[str]) -> list[str]:
    tokens: list[str] = []
    for slot_name in slot_names:
        pin_name = SLOT_TO_PIN.get(slot_name)
        if pin_name is None or slot_name not in slots:
            continue
        token = pin_token(pin_name, slots.get(slot_name))
        if token:
            tokens.append(token)
    return tokens


def op_pin_tokens(row: dict[str, Any], *, strict: bool) -> list[str]:
    op_name = str(row.get("op_name") or "")
    slots = row.get("slot_values") if isinstance(row.get("slot_values"), dict) else {}
    op_id = safe_int(slots.get("sketch.op"), OP_IDS.get(op_name, -1))
    tokens: list[str] = []
    if op_name:
        tokens.append(op_name)
    if strict and op_id >= 0:
        tokens.append(f"pin:op={op_id}")
    return tokens


def attempt_score(row: dict[str, Any]) -> tuple[float, int, float, int]:
    verifier = row.get("verifier_distance") or {}
    score = safe_float(verifier.get("verifier_score"))
    checked = safe_int(verifier.get("checked_cells"))
    mismatch = safe_float(verifier.get("mismatch_ratio"), 1.0)
    unsupported = safe_int(verifier.get("unsupported_pairs"))
    return (score, checked, -mismatch, -unsupported)


def task_feature_distance(target: dict[str, Any], source: dict[str, Any]) -> float:
    if not target or not source:
        return 1000.0
    numeric = [
        "input_width",
        "input_height",
        "output_width",
        "output_height",
        "shape_scale_x",
        "shape_scale_y",
    ]
    distance = 0.0
    for key in numeric:
        a = safe_int(target.get(key), -1)
        b = safe_int(source.get(key), -1)
        if a >= 0 and b >= 0:
            distance += abs(a - b)
    for key in [
        "slot_width_matches_output",
        "slot_height_matches_output",
        "period_divides_input_width",
        "period_divides_input_height",
    ]:
        if key in target and key in source and bool(target[key]) != bool(source[key]):
            distance += 1.0
    return distance


def shape_hint_tokens(features: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    input_width = safe_int(features.get("input_width"))
    input_height = safe_int(features.get("input_height"))
    output_width = safe_int(features.get("output_width"))
    output_height = safe_int(features.get("output_height"))
    scale_x = safe_int(features.get("shape_scale_x"))
    scale_y = safe_int(features.get("shape_scale_y"))
    if scale_x > 1 and scale_x == scale_y:
        tokens.extend(["macrocell_substitute", "palette_scale", "lattice_sample"])
        tokens.append(f"pin:period={scale_x}")
    if input_width > 0 and input_height > 0 and output_width > 0 and output_height > 0:
        if output_width < input_width or output_height < input_height:
            tokens.extend(
                [
                    "crop",
                    "masked_or_normalized_crop",
                    "object_mask_extract",
                    "lattice_sample",
                    "axis_projection_compress",
                ]
            )
        if output_width == 1 or output_height == 1:
            tokens.extend(
                [
                    "axis_projection_compress",
                    "component_count_line",
                    "component_color_summary",
                ]
            )
    return tokens


def stage_tokens(stage: str) -> list[str]:
    return list(STAGE_REPAIR_TOKENS.get(stage, []))


def row_attempt_id(row: dict[str, Any]) -> str:
    return str(row.get("attempt_id") or row.get("library_id") or "")


def candidate_row(
    *,
    task_id: str,
    split: str,
    tokens: list[str],
    frontier: str,
    parent: str,
    operator: str,
    residual: str,
    stage: str,
    strategy: str,
    proposal_count: int,
    utility: float,
) -> dict[str, str] | None:
    op_hints = compact_tokens(tokens)
    if not task_id or not split or not op_hints:
        return None
    return {
        "family": "arc.visual_hole_proposer_family",
        "task_id": clean_field(task_id),
        "split": clean_field(split),
        "support_tasks": "1",
        "proposal_count": str(max(1, proposal_count)),
        "utility_mean": f"{utility:.8f}",
        "op_hints": op_hints,
        "coverage_frontier_id": clean_field(frontier),
        "parent_attempt_id": clean_field(parent),
        "repair_operator": clean_field(operator),
        "dominant_residual_class": clean_field(residual),
        "dominant_failure_stage": clean_field(stage),
        "repair_strategy": clean_field(strategy),
    }


def write_candidate_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        fp.write("\t".join(CANDIDATE_HEADER) + "\n")
        for row in rows:
            fp.write("\t".join(row.get(field, "") for field in CANDIDATE_HEADER) + "\n")


def load_targets(
    scoped_logs: list[tuple[str, Path, list[dict[str, Any]]]],
    target_scopes: set[str],
    include_selected: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    examples: list[dict[str, Any]] = []
    targets: dict[str, dict[str, Any]] = {}
    for scope, path, rows in scoped_logs:
        split = split_for_scope(scope)
        if scope not in target_scopes or split is None:
            continue
        for row in rows:
            if row.get("schema") != "crlp.arc_hypothesis.selection.v1":
                continue
            visual = visual_template_stats(row)
            kind = task_kind(visual)
            task_id = str(row.get("task_instance", ""))
            example = {
                "schema": EXAMPLE_SCHEMA,
                "report_version": 1,
                "kind": kind,
                "source_scope": scope,
                "source_path": str(path),
                "split": split,
                "task_id": task_id,
                "attempted": safe_int(visual.get("attempted")),
                "train_consistent_pre_cap": safe_int(
                    visual.get("train_consistent_pre_cap")
                ),
                "train_consistent_retained": safe_int(
                    visual.get("train_consistent_retained")
                ),
                "selected": safe_int(visual.get("selected")),
            }
            examples.append(example)
            if not include_selected and kind == "visual_target_already_selected":
                continue
            targets[task_id] = {
                "task_id": task_id,
                "split": split,
                "source_scope": scope,
                "kind": kind,
                "visual": visual,
            }
    return examples, targets


def load_attempt_models(
    paths: list[Path],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
]:
    attempts: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    rejected_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    features_by_task: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in load_jsonl(path):
            if (
                row.get("schema")
                != "crlp.neural_grammar_discovery.arc_visual_pipeline_attempt.v1"
            ):
                continue
            row = dict(row)
            row["_source_path"] = str(path)
            task_id = str(row.get("task_id", ""))
            features = row.get("task_slot_features")
            if task_id and isinstance(features, dict) and task_id not in features_by_task:
                features_by_task[task_id] = features
            attempts.append(row)
            if row.get("accepted") is True:
                accepted.append(row)
            elif task_id:
                rejected_by_task[task_id].append(row)
    return attempts, accepted, rejected_by_task, features_by_task


def accepted_transfer_rows(
    target: dict[str, Any],
    target_features: dict[str, Any],
    accepted: list[dict[str, Any]],
    max_rows: int,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    ranked = sorted(
        accepted,
        key=lambda row: task_feature_distance(
            target_features, row.get("task_slot_features") or {}
        ),
    )[:max_rows]
    rows: list[dict[str, str]] = []
    examples: list[dict[str, Any]] = []
    for rank, source in enumerate(ranked, start=1):
        slots = source.get("slot_values") if isinstance(source.get("slot_values"), dict) else {}
        tokens = op_pin_tokens(source, strict=True)
        tokens.extend(slot_pins(slots, STRUCTURAL_PINS))
        stage = "nearest_positive_transfer"
        row = candidate_row(
            task_id=target["task_id"],
            split=target["split"],
            tokens=tokens,
            frontier=f"arc_visual_hole:{target['source_scope']}",
            parent=row_attempt_id(source),
            operator="nearest_positive_slot_transfer",
            residual=target["kind"],
            stage=stage,
            strategy="transfer_verified_structural_slots",
            proposal_count=32,
            utility=1.0 / rank,
        )
        if row:
            rows.append(row)
            examples.append(
                {
                    "schema": EXAMPLE_SCHEMA,
                    "report_version": 1,
                    "kind": "nearest_positive_signature",
                    "target_task_id": target["task_id"],
                    "source_task_id": source.get("task_id"),
                    "source_attempt_id": row_attempt_id(source),
                    "rank": rank,
                    "distance": task_feature_distance(
                        target_features, source.get("task_slot_features") or {}
                    ),
                    "op_hints": row["op_hints"],
                }
            )
    return rows, examples


def residual_repair_rows(
    target: dict[str, Any],
    target_features: dict[str, Any],
    rejected: list[dict[str, Any]],
    max_attempt_seeds: int,
    max_ops: int,
    emit_broad_repair: bool,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    rows: list[dict[str, str]] = []
    examples: list[dict[str, Any]] = []
    stage_counts: Counter[str] = Counter(
        str(row.get("failure_stage") or row.get("rejection_reason") or "rejected")
        for row in rejected
    )
    op_counts: Counter[str] = Counter(
        str(row.get("op_name") or "unknown") for row in rejected
    )
    dominant_stage = stage_counts.most_common(1)[0][0] if stage_counts else "coverage"
    dominant_ops = [op for op, _count in op_counts.most_common(max_ops) if op in OP_IDS]
    base_tokens = dominant_ops + stage_tokens(dominant_stage) + shape_hint_tokens(target_features)
    if emit_broad_repair and base_tokens:
        row = candidate_row(
            task_id=target["task_id"],
            split=target["split"],
            tokens=base_tokens,
            frontier=f"arc_visual_hole:{target['source_scope']}",
            parent="",
            operator="dominant_residual_op_repair",
            residual=target["kind"],
            stage=dominant_stage,
            strategy="rank_ops_by_rejected_attempt_surface",
            proposal_count=64,
            utility=1.0,
        )
        if row:
            rows.append(row)
    ranked_attempts = sorted(rejected, key=attempt_score, reverse=True)[:max_attempt_seeds]
    for rank, attempt in enumerate(ranked_attempts, start=1):
        slots = (
            attempt.get("slot_values")
            if isinstance(attempt.get("slot_values"), dict)
            else {}
        )
        stage = str(
            attempt.get("failure_stage")
            or attempt.get("rejection_reason")
            or dominant_stage
        )
        features = attempt.get("task_slot_features") or {}
        strict_tokens = op_pin_tokens(attempt, strict=True)
        if stage in {"renderer_cells", "transform_or_color"}:
            strict_tokens.extend(slot_pins(slots, GEOMETRY_PINS))
        elif stage == "renderer_shape":
            strict_tokens.extend(slot_pins(slots, ["sketch.selector"]))
        elif stage == "selector_or_transform":
            strict_tokens.extend(slot_pins(slots, ["sketch.selector"]))
        if features.get("slot_width_matches_output"):
            strict_tokens.extend(slot_pins(slots, ["sketch.width"]))
        if features.get("slot_height_matches_output"):
            strict_tokens.extend(slot_pins(slots, ["sketch.height"]))
        row = candidate_row(
            task_id=target["task_id"],
            split=target["split"],
            tokens=strict_tokens,
            frontier=f"arc_visual_hole:{target['source_scope']}",
            parent=row_attempt_id(attempt),
            operator="verifier_distance_slot_repair",
            residual=target["kind"],
            stage=stage,
            strategy="pin_high_score_attempt_geometry",
            proposal_count=32,
            utility=0.75 / rank,
        )
        if row:
            rows.append(row)
        if emit_broad_repair:
            repair_tokens = op_pin_tokens(attempt, strict=False)
            repair_tokens.extend(slot_pins(slots, STRUCTURAL_PINS))
            repair_tokens.extend(stage_tokens(stage))
            repair_tokens.extend(shape_hint_tokens(target_features))
            row = candidate_row(
                task_id=target["task_id"],
                split=target["split"],
                tokens=repair_tokens,
                frontier=f"arc_visual_hole:{target['source_scope']}",
                parent=row_attempt_id(attempt),
                operator="verifier_distance_slot_repair",
                residual=target["kind"],
                stage=stage,
                strategy="repair_high_score_attempt_slots",
                proposal_count=32,
                utility=0.70 / rank,
            )
            if row:
                rows.append(row)
        examples.append(
            {
                "schema": EXAMPLE_SCHEMA,
                "report_version": 1,
                "kind": "rejected_attempt_seed",
                "target_task_id": target["task_id"],
                "attempt_id": row_attempt_id(attempt),
                "rank": rank,
                "op_name": attempt.get("op_name"),
                "failure_stage": stage,
                "attempt_score": list(attempt_score(attempt)),
                "slot_values": slots,
            }
        )
    return rows, examples


def dedupe_candidate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str, str]] = set()
    deduped: list[dict[str, str]] = []
    for row in rows:
        key = (
            row.get("task_id", ""),
            row.get("split", ""),
            row.get("op_hints", ""),
            row.get("repair_strategy", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def build_hole_proposer(args: argparse.Namespace) -> int:
    scoped_logs = [
        parse_scoped_path(raw, f"arc_visual_only_{index}")
        for index, raw in enumerate(args.visual_only_log or [], start=1)
    ]
    if not scoped_logs:
        print("at least one --visual-only-log is required", file=sys.stderr)
        return 2
    loaded_logs = [(scope, path, load_jsonl(path)) for scope, path in scoped_logs]
    target_scopes = set(args.target_scope or [])
    target_examples, targets = load_targets(
        loaded_logs, target_scopes, include_selected=args.include_selected
    )
    attempts, accepted, rejected_by_task, features_by_task = load_attempt_models(
        args.attempt_log or []
    )

    examples: list[dict[str, Any]] = list(target_examples)
    candidate_rows: list[dict[str, str]] = []
    rows_by_strategy: Counter[str] = Counter()
    rows_by_stage: Counter[str] = Counter()
    rows_by_task: Counter[str] = Counter()

    for task_id, target in sorted(targets.items()):
        target_features = features_by_task.get(task_id, {})
        shape_tokens = shape_hint_tokens(target_features)
        if shape_tokens:
            row = candidate_row(
                task_id=target["task_id"],
                split=target["split"],
                tokens=shape_tokens,
                frontier=f"arc_visual_hole:{target['source_scope']}",
                parent="",
                operator="shape_feature_frontier",
                residual=target["kind"],
                stage="shape_feature_gap",
                strategy="route_shape_scale_and_compression_ops",
                proposal_count=64,
                utility=0.95,
            )
            if row:
                candidate_rows.append(row)
                rows_by_strategy[row["repair_strategy"]] += 1
                rows_by_stage[row["dominant_failure_stage"]] += 1
                rows_by_task[task_id] += 1
        transfer_rows, transfer_examples = accepted_transfer_rows(
            target,
            target_features,
            accepted,
            max_rows=args.nearest_signatures,
        )
        repair_rows, repair_examples = residual_repair_rows(
            target,
            target_features,
            rejected_by_task.get(task_id, []),
            max_attempt_seeds=args.max_attempt_seeds,
            max_ops=args.max_ops_per_task,
            emit_broad_repair=args.emit_broad_repair,
        )
        for row in transfer_rows + repair_rows:
            candidate_rows.append(row)
            rows_by_strategy[row["repair_strategy"]] += 1
            rows_by_stage[row["dominant_failure_stage"]] += 1
            rows_by_task[task_id] += 1
        examples.extend(transfer_examples)
        examples.extend(repair_examples)

    candidate_rows = dedupe_candidate_rows(candidate_rows)
    rows_by_strategy = Counter(row["repair_strategy"] for row in candidate_rows)
    rows_by_stage = Counter(row["dominant_failure_stage"] for row in candidate_rows)
    rows_by_task = Counter(row["task_id"] for row in candidate_rows)

    accepted_ops = Counter(str(row.get("op_name") or "unknown") for row in accepted)
    rejected_stages = Counter(
        str(row.get("failure_stage") or row.get("rejection_reason") or "rejected")
        for row in attempts
        if row.get("accepted") is not True
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "report_version": 1,
        "source_logs": [
            {"scope": scope, "path": str(path), "rows": len(rows)}
            for scope, path, rows in loaded_logs
        ],
        "attempt_logs": [str(path) for path in args.attempt_log or []],
        "target_scopes": sorted(target_scopes),
        "target_task_count": len(targets),
        "candidate_count": len(candidate_rows),
        "candidate_tsv": str(args.candidate_tsv),
        "example_jsonl": str(args.examples),
        "accepted_attempts": len(accepted),
        "accepted_ops": dict(sorted(accepted_ops.items())),
        "rejected_attempts": sum(
            count for stage, count in rejected_stages.items() if stage != "accepted"
        ),
        "rejected_stages": dict(sorted(rejected_stages.items())),
        "candidate_rows_by_strategy": dict(sorted(rows_by_strategy.items())),
        "candidate_rows_by_stage": dict(sorted(rows_by_stage.items())),
        "candidate_rows_by_task": dict(sorted(rows_by_task.items())),
    }

    write_candidate_tsv(args.candidate_tsv, candidate_rows)
    write_jsonl(args.examples, examples, compact=True)
    write_jsonl(args.summary, [summary], compact=True)
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
        help="Scope whose uncovered tasks should receive hole-proposer rows.",
    )
    parser.add_argument("--attempt-log", action="append", type=Path, default=[])
    parser.add_argument("--candidate-tsv", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--nearest-signatures", type=int, default=1)
    parser.add_argument("--max-attempt-seeds", type=int, default=2)
    parser.add_argument("--max-ops-per-task", type=int, default=2)
    parser.add_argument("--emit-broad-repair", action="store_true")
    parser.add_argument("--include-selected", action="store_true")
    return parser.parse_args()


def main() -> int:
    return build_hole_proposer(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
