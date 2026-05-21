#!/usr/bin/env python3
"""Emit ARC sketch-probe candidates for ranker shape gaps.

The ranker-only synthetic shape curriculum can teach the scorer that every
shape/grid pairing is plausible, but it cannot add missing target candidates.
This script turns those shape-gap diagnoses back into C sketch-probe rows for
the original ARC family/task keys, so `arc_agi` can mine and verify actual
candidate programs before the ranker sees them.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from crlp.io import read_jsonl_lenient, safe_float, safe_int, write_jsonl


SCHEMA = "crlp.arc_synthetic_shape_probe_candidates.v1"

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
    "unsupported_shape": ["counterexample_shape_frame_repair"],
}


def clean(value: Any) -> str:
    return str(value if value is not None else "").replace("\t", " ").replace("\n", " ").strip()


def compact_tokens(tokens: list[str], max_len: int = 240) -> str:
    seen: set[str] = set()
    kept: list[str] = []
    for raw in tokens:
        token = clean(raw)
        if not token or token in seen:
            continue
        trial = ",".join(kept + [token])
        if len(trial) > max_len:
            continue
        seen.add(token)
        kept.append(token)
    return ",".join(kept)


def task_dir_for_split(split: str) -> str:
    lowered = split.lower()
    if "eval" in lowered:
        return "evaluation"
    return "training"


def load_task(public_arc_dir: Path, split: str, task_id: str) -> dict[str, Any] | None:
    path = public_arc_dir / task_dir_for_split(split) / f"{task_id}.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def grid_shape(grid: list[list[int]]) -> tuple[int, int]:
    height = len(grid)
    width = len(grid[0]) if height else 0
    return width, height


def task_shapes(task: dict[str, Any] | None) -> dict[str, Any]:
    if not task:
        return {}
    input_shapes: list[tuple[int, int]] = []
    output_shapes: list[tuple[int, int]] = []
    for pair in task.get("train", []) or []:
        if not isinstance(pair, dict) or "input" not in pair or "output" not in pair:
            continue
        input_shapes.append(grid_shape(pair.get("input") or []))
        output_shapes.append(grid_shape(pair.get("output") or []))
    if not input_shapes or not output_shapes:
        return {}
    first_input = input_shapes[0]
    first_output = output_shapes[0]
    same_output_shape = len(set(output_shapes)) == 1
    crop_like = all(
        out_w <= in_w and out_h <= in_h
        for (in_w, in_h), (out_w, out_h) in zip(input_shapes, output_shapes)
    )
    grow_like = all(
        out_w >= in_w and out_h >= in_h
        for (in_w, in_h), (out_w, out_h) in zip(input_shapes, output_shapes)
    )
    line_like = any(out_w == 1 or out_h == 1 for out_w, out_h in output_shapes)
    scale_x = first_output[0] // first_input[0] if first_input[0] and first_output[0] % first_input[0] == 0 else 0
    scale_y = first_output[1] // first_input[1] if first_input[1] and first_output[1] % first_input[1] == 0 else 0
    stable_scale = (
        scale_x > 1
        and scale_x == scale_y
        and all(
            in_w * scale_x == out_w and in_h * scale_y == out_h
            for (in_w, in_h), (out_w, out_h) in zip(input_shapes, output_shapes)
        )
    )
    return {
        "first_input_width": first_input[0],
        "first_input_height": first_input[1],
        "first_output_width": first_output[0],
        "first_output_height": first_output[1],
        "same_output_shape": same_output_shape,
        "crop_like": crop_like,
        "grow_like": grow_like,
        "line_like": line_like,
        "scale": scale_x if stable_scale else 0,
        "train_pairs": len(output_shapes),
    }


def pin_output_shape_tokens(shapes: dict[str, Any]) -> list[str]:
    if not shapes.get("same_output_shape"):
        return []
    width = safe_int(shapes.get("first_output_width"))
    height = safe_int(shapes.get("first_output_height"))
    if width <= 0 or height <= 0:
        return []
    return [f"pin:width={width}", f"pin:height={height}"]


def pin_scale_tokens(shapes: dict[str, Any]) -> list[str]:
    scale = safe_int(shapes.get("scale"))
    if scale <= 1:
        return []
    return [f"pin:period={scale}"]


def candidate_row(
    *,
    family: str,
    task_id: str,
    split: str,
    tokens: list[str],
    parent: str,
    stage: str,
    strategy: str,
    utility: float,
    proposal_count: int,
) -> dict[str, str] | None:
    op_hints = compact_tokens(tokens)
    if not family or not task_id or not split or not op_hints:
        return None
    return {
        "family": clean(family),
        "task_id": clean(task_id),
        "split": clean(split),
        "support_tasks": "1",
        "proposal_count": str(max(1, proposal_count)),
        "utility_mean": f"{utility:.8f}",
        "op_hints": op_hints,
        "coverage_frontier_id": "arc_synthetic_shape_probe",
        "parent_attempt_id": clean(parent),
        "repair_operator": "synthetic_shape_probe",
        "dominant_residual_class": "candidate_generation_gap",
        "dominant_failure_stage": clean(stage),
        "repair_strategy": clean(strategy),
    }


def strategy_rows(
    row: dict[str, Any],
    shapes: dict[str, Any],
    max_rows: int,
    profile: str,
) -> list[dict[str, str]]:
    family = clean(row.get("family"))
    task_id = clean(row.get("task_id"))
    split = clean(row.get("split"))
    top_op = clean(row.get("top_op_name"))
    stage = clean(row.get("dominant_candidate_failure_stage") or row.get("top_failure_stage"))
    reason = clean(row.get("dominant_candidate_rejection_reason") or row.get("top_rejection_reason"))
    parent = clean(row.get("top_attempt_id"))
    stage_tokens = STAGE_REPAIR_TOKENS.get(stage, []) + STAGE_REPAIR_TOKENS.get(reason, [])
    shape_pins = pin_output_shape_tokens(shapes)
    scale_pins = pin_scale_tokens(shapes)
    rows: list[dict[str, str]] = []

    strategies: list[tuple[str, float, int, list[str]]] = []
    if top_op and (profile == "broad" or top_op != "visual_combinator"):
        strategies.append(
            (
                "replay_top_failed_op_with_synthetic_shape_hints",
                1.0,
                64,
                [top_op] + (stage_tokens if profile == "broad" else []) + shape_pins,
            )
        )
    if (
        profile == "broad"
        or "crop_object_selection" in family
        or shapes.get("crop_like")
        or top_op in {"crop", "masked_or_normalized_crop", "recolor_crop", "object_mask_extract"}
    ):
        strategies.append(
            (
                "exhaust_crop_and_normalized_shape_frames",
                0.95,
                64,
                [
                    "crop",
                    "masked_or_normalized_crop",
                    "recolor_crop",
                    "object_mask_extract",
                    "counterexample_shape_frame_repair",
                ]
                + shape_pins,
            )
        )
    if (
        profile == "broad"
        or "patch_lattice" in family
        or "tiling_canvas" in family
        or shapes.get("grow_like")
        or top_op in {"macrocell_substitute", "palette_scale", "lattice_sample"}
    ):
        strategies.append(
            (
                "exhaust_macro_lattice_shape_frames",
                0.92,
                64,
                [
                    "macrocell_substitute",
                    "palette_scale",
                    "lattice_sample",
                    "point_line_project",
                    "object_lattice_frame",
                    "normalizer:object_lattice",
                ]
                + scale_pins,
            )
        )
    if profile == "broad" or (
        top_op in {"object_mask_extract"}
        and safe_int(row.get("candidate_count")) < 5000
    ):
        strategies.append(
            (
                "exhaust_component_mask_render_shapes",
                0.90,
                64,
                [
                    "visual_combinator",
                    "object_mask_extract",
                    "emitter:component_mask",
                    "normalizer:component_mask",
                    "selector:component_predicate",
                    "normalizer:component_selector_layout",
                ]
                + shape_pins,
            )
        )
    if (
        profile == "broad"
        or "projection_compression" in family
        or shapes.get("line_like")
        or top_op in {"axis_projection_compress", "component_count_line"}
    ):
        strategies.append(
            (
                "exhaust_projection_and_count_shapes",
                0.86,
                64,
                [
                    "axis_projection_compress",
                    "component_count_line",
                    "marker_count_motif_tile",
                    "component_color_summary",
                ]
                + shape_pins,
            )
        )
    if (
        profile == "broad"
        or "separator_panel" in family
        or top_op
        in {
            "separator_panel_map",
            "separator_intersection_color",
            "separator_panel_relation_mask",
        }
    ):
        strategies.append(
            (
                "exhaust_separator_panel_shape_frames",
                0.82,
                64,
                [
                    "separator_panel_map",
                    "separator_intersection_color",
                    "separator_panel_relation_mask",
                    "normalizer:intersection",
                    "separator_relation_frame",
                ],
            )
        )

    for strategy, utility, proposal_count, tokens in strategies[:max_rows]:
        built = candidate_row(
            family=family,
            task_id=task_id,
            split=split,
            tokens=tokens,
            parent=parent,
            stage=stage or reason or "candidate_generation_gap",
            strategy=strategy,
            utility=utility,
            proposal_count=proposal_count,
        )
        if built:
            rows.append(built)
    return rows


def write_candidate_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(CANDIDATE_HEADER) + "\n")
        for row in rows:
            handle.write("\t".join(row.get(field, "") for field in CANDIDATE_HEADER) + "\n")


def build(args: argparse.Namespace) -> dict[str, Any]:
    failures = [
        row
        for row in read_jsonl_lenient(args.failure_report)
        if row.get("schema") == "crlp.arc_ranker_failure_task.v1"
    ]
    selected = [
        row
        for row in failures
        if args.include_solved or row.get("failure_bucket") == "candidate_generation_gap"
    ]
    if args.max_tasks > 0:
        selected = selected[: args.max_tasks]

    candidate_rows: list[dict[str, str]] = []
    examples: list[dict[str, Any]] = []
    missing_tasks = 0
    for failure in selected:
        task_id = clean(failure.get("task_id"))
        split = clean(failure.get("split"))
        task = load_task(args.public_arc_dir, split, task_id)
        shapes = task_shapes(task)
        if task is None:
            missing_tasks += 1
        rows = strategy_rows(
            failure,
            shapes,
            max_rows=args.max_rows_per_task,
            profile=args.profile,
        )
        candidate_rows.extend(rows)
        examples.append(
            {
                "schema": SCHEMA,
                "report_version": 1,
                "task_id": task_id,
                "split": split,
                "family": clean(failure.get("family")),
                "failure_bucket": clean(failure.get("failure_bucket")),
                "top_op_name": clean(failure.get("top_op_name")),
                "dominant_candidate_rejection_reason": clean(
                    failure.get("dominant_candidate_rejection_reason")
                ),
                "dominant_candidate_failure_stage": clean(
                    failure.get("dominant_candidate_failure_stage")
                ),
                "task_shapes": shapes,
                "candidate_rows": len(rows),
            }
        )

    seen: set[tuple[str, str, str, str]] = set()
    deduped: list[dict[str, str]] = []
    for row in candidate_rows:
        key = (
            row["family"],
            row["task_id"],
            row["split"],
            row["op_hints"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    candidate_rows = deduped
    by_strategy = Counter(row["repair_strategy"] for row in candidate_rows)
    by_stage = Counter(row["dominant_failure_stage"] for row in candidate_rows)
    summary = {
        "schema": SCHEMA,
        "report_version": 1,
        "failure_report": str(args.failure_report),
        "public_arc_dir": str(args.public_arc_dir),
        "candidate_tsv": str(args.candidate_tsv),
        "examples": str(args.examples),
        "input_failure_rows": len(failures),
        "selected_task_rows": len(selected),
        "candidate_rows": len(candidate_rows),
        "missing_task_files": missing_tasks,
        "include_solved": args.include_solved,
        "profile": args.profile,
        "max_rows_per_task": args.max_rows_per_task,
        "max_tasks": args.max_tasks,
        "rows_by_strategy": dict(sorted(by_strategy.items())),
        "rows_by_stage": dict(sorted(by_stage.items())),
        "min_top_score": min((safe_float(row.get("top_score")) for row in selected), default=0.0),
        "max_candidate_count": max((safe_int(row.get("candidate_count")) for row in selected), default=0),
    }
    write_candidate_tsv(args.candidate_tsv, candidate_rows)
    write_jsonl(args.examples, examples, compact=True)
    write_jsonl(args.summary, [summary], compact=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-report", type=Path, required=True)
    parser.add_argument("--public-arc-dir", type=Path, required=True)
    parser.add_argument("--candidate-tsv", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--max-rows-per-task", type=int, default=6)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--profile", choices=("focused", "broad"), default="focused")
    parser.add_argument("--include-solved", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_rows_per_task <= 0:
        raise SystemExit("--max-rows-per-task must be positive")
    if args.max_tasks < 0:
        raise SystemExit("--max-tasks must be non-negative")
    if not args.failure_report.is_file():
        raise SystemExit(f"missing failure report: {args.failure_report}")
    if not args.public_arc_dir.is_dir():
        raise SystemExit(f"missing public ARC dir: {args.public_arc_dir}")
    summary = build(args)
    print(
        "arc-synthetic-shape-probe-candidates "
        f"tasks={summary['selected_task_rows']} "
        f"candidate_rows={summary['candidate_rows']} "
        f"tsv={args.candidate_tsv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
