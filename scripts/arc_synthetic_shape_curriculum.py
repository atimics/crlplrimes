#!/usr/bin/env python3
"""Generate synthetic ARC shape-at-grid-size source attempts."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from crlp.io import write_jsonl


SCHEMA = "crlp.arc_synthetic_shape_curriculum.v1"

DEFAULT_OPS = (
    "masked_or_normalized_crop",
    "crop",
    "recolor_crop",
    "macrocell_substitute",
    "component_color_summary",
    "visual_combinator",
    "separator_panel_map",
)


@dataclass(frozen=True)
class PipelineSpec:
    pipeline_kind: str
    visual_program: str
    slot_contract: str
    composition_kind: str
    normalization_frame: str
    typed_stages: str
    primitive_contract: str


PIPELINE_SPECS = {
    "masked_or_normalized_crop": PipelineSpec(
        pipeline_kind="masked_or_normalized_crop_pipeline",
        visual_program="select_mask -> normalize_crop -> emit_grid",
        slot_contract="selector,width,height,color,background",
        composition_kind="object_render",
        normalization_frame="normalized_bbox_frame",
        typed_stages="select_mask|normalize_crop|emit_grid",
        primitive_contract="selector:mask,normalizer:bbox,emitter:grid",
    ),
    "crop": PipelineSpec(
        pipeline_kind="crop_pipeline",
        visual_program="select_bbox -> crop -> emit_grid",
        slot_contract="selector,width,height,x_anchor,y_anchor",
        composition_kind="object_render",
        normalization_frame="bbox_frame",
        typed_stages="select_bbox|crop|emit_grid",
        primitive_contract="selector:bbox,normalizer:crop,emitter:grid",
    ),
    "recolor_crop": PipelineSpec(
        pipeline_kind="recolor_crop_pipeline",
        visual_program="select_bbox -> crop -> recolor -> emit_grid",
        slot_contract="selector,width,height,color,output_color",
        composition_kind="object_render",
        normalization_frame="bbox_color_frame",
        typed_stages="select_bbox|crop|recolor|emit_grid",
        primitive_contract="selector:bbox,transform:recolor,emitter:grid",
    ),
    "macrocell_substitute": PipelineSpec(
        pipeline_kind="macrocell_substitute_pipeline",
        visual_program="select_macrocell -> substitute_template -> emit_grid",
        slot_contract="selector,width,height,period,color",
        composition_kind="template_render",
        normalization_frame="macrocell_frame",
        typed_stages="select_macrocell|substitute_template|emit_grid",
        primitive_contract="selector:macrocell,transform:template,emitter:grid",
    ),
    "component_color_summary": PipelineSpec(
        pipeline_kind="component_color_summary_pipeline",
        visual_program="select_components -> summarize_color -> emit_grid",
        slot_contract="selector,color,output_color,count_mode",
        composition_kind="component_summary",
        normalization_frame="component_color_frame",
        typed_stages="select_components|summarize_color|emit_grid",
        primitive_contract="selector:components,normalizer:component,emitter:grid",
    ),
    "visual_combinator": PipelineSpec(
        pipeline_kind="visual_combinator_pipeline",
        visual_program="select_components -> compose_relation -> emit_grid",
        slot_contract="selector,width,height,orientation,color",
        composition_kind="combinator_render",
        normalization_frame="component_relation_frame",
        typed_stages="select_components|compose_relation|emit_grid",
        primitive_contract="selector:components,transform:relation,emitter:grid",
    ),
    "separator_panel_map": PipelineSpec(
        pipeline_kind="separator_panel_map_pipeline",
        visual_program="find_separators -> normalize_panels -> map_panels",
        slot_contract="selector,background,color,output_color,x_anchor",
        composition_kind="panel_composition",
        normalization_frame="separator_panel_frame",
        typed_stages="find_separators|normalize_panels|map_panels",
        primitive_contract="selector:separator,normalizer:panel_grid,emitter:grid",
    ),
}

STRING_DEFAULTS = {
    "split_name",
    "family",
    "task_id",
    "split",
    "attempt_id",
    "library_id",
    "pipeline_kind",
    "op_name",
    "visual_program",
    "slot_contract",
    "composition_kind",
    "normalization_frame",
    "typed_stages",
    "primitive_contract",
    "slot_tokens",
    "rejection_reason",
    "failure_stage",
    "failure_detail",
    "contrast_accepted_attempt_id",
    "contrast_pair_kind",
}


def stable_id(*parts: object) -> str:
    digest = hashlib.blake2b(digest_size=8)
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def shape_kind(grid_w: int, grid_h: int, shape_w: int, shape_h: int) -> str:
    if shape_w == grid_w and shape_h == grid_h:
        return "full_grid"
    if shape_w == 1 and shape_h == 1:
        return "point"
    if shape_h == 1:
        return "horizontal_line"
    if shape_w == 1:
        return "vertical_line"
    if shape_w == shape_h:
        return "square"
    return "rectangle"


def iter_shapes(max_grid_size: int) -> Iterable[tuple[int, int, int, int, str]]:
    for grid_w in range(1, max_grid_size + 1):
        for grid_h in range(1, max_grid_size + 1):
            for shape_w in range(1, grid_w + 1):
                for shape_h in range(1, grid_h + 1):
                    yield (
                        grid_w,
                        grid_h,
                        shape_w,
                        shape_h,
                        shape_kind(grid_w, grid_h, shape_w, shape_h),
                    )


def default_row(header: list[str]) -> dict[str, str]:
    return {name: "" if name in STRING_DEFAULTS else "0" for name in header}


def set_if_present(row: dict[str, str], name: str, value: object) -> None:
    if name in row:
        row[name] = str(value)


def row_text(header: list[str], row: dict[str, str]) -> str:
    return "\t".join(row.get(name, "") for name in header) + "\n"


def slot_tokens(
    *,
    op_index: int,
    grid_w: int,
    grid_h: int,
    shape_w: int,
    shape_h: int,
    kind: str,
    color: int,
    output_color: int,
) -> str:
    return "|".join(
        [
            f"synthetic.shape={kind}",
            f"synthetic.grid={grid_w}x{grid_h}",
            f"synthetic.bbox={shape_w}x{shape_h}",
            f"sketch.op={op_index}",
            "sketch.background=0",
            f"sketch.color={color}",
            f"sketch.output_color={output_color}",
            f"sketch.width={shape_w}",
            f"sketch.height={shape_h}",
            f"sketch.x_anchor={(grid_w - shape_w) // 2}",
            f"sketch.y_anchor={(grid_h - shape_h) // 2}",
        ]
    )


def make_attempt_row(
    *,
    header: list[str],
    attempt_index: int,
    grid_w: int,
    grid_h: int,
    shape_w: int,
    shape_h: int,
    kind: str,
    op_name: str,
    op_index: int,
    outcome: str,
    accepted_attempt_id: str,
    negative_count: int,
) -> dict[str, str]:
    row = default_row(header)
    spec = PIPELINE_SPECS[op_name]
    color = 1 + ((grid_w + shape_w + op_index) % 9)
    output_color = 1 + ((grid_h + shape_h + op_index + 3) % 9)
    area = shape_w * shape_h
    task_id = f"synth_{grid_w:02d}x{grid_h:02d}_{shape_w:02d}x{shape_h:02d}_{kind}"
    if outcome == "accepted":
        attempt_id = accepted_attempt_id
        rejection_reason = "accepted"
        failure_stage = "accepted"
        failure_detail = "exact_synthetic_shape"
        verifier_score = 1.0
        verifier_mismatches = 0
        verifier_unsupported = 0
        verifier_apply_failed = 0
        verifier_shape_mismatch = 0
        mismatch_ratio = 0.0
        contrast_id = ""
        contrast_kind = ""
        contrast_distance = 0
        accepted = 1
        utility = 1.0
        slot_width_matches_output = 1
        slot_height_matches_output = 1
    elif outcome == "unsupported_shape":
        attempt_id = f"ngd-synth:{stable_id(accepted_attempt_id, outcome)}"
        rejection_reason = "unsupported_shape"
        failure_stage = "renderer_shape"
        failure_detail = "output_shape_mismatch"
        verifier_score = 0.20
        verifier_mismatches = max(1, area // 3)
        verifier_unsupported = 1
        verifier_apply_failed = 0
        verifier_shape_mismatch = 1
        mismatch_ratio = 0.0
        contrast_id = accepted_attempt_id
        contrast_kind = "synthetic_shape_mismatch"
        contrast_distance = 1
        accepted = 0
        utility = 0.20
        slot_width_matches_output = 0
        slot_height_matches_output = 0
    elif outcome == "no_train_match":
        attempt_id = f"ngd-synth:{stable_id(accepted_attempt_id, outcome)}"
        rejection_reason = "no_train_match"
        failure_stage = "transform_or_color"
        failure_detail = "high_cell_mismatch_ratio"
        verifier_score = 0.25
        verifier_mismatches = max(1, (area * 3) // 4)
        verifier_unsupported = 0
        verifier_apply_failed = 0
        verifier_shape_mismatch = 0
        mismatch_ratio = 0.75
        contrast_id = accepted_attempt_id
        contrast_kind = "synthetic_no_train_match"
        contrast_distance = 1
        accepted = 0
        utility = 0.25
        slot_width_matches_output = 1
        slot_height_matches_output = 1
    else:
        raise ValueError(f"unknown outcome: {outcome}")

    set_if_present(row, "split_name", "source")
    set_if_present(row, "family", "arc.synthetic_shape_family")
    set_if_present(row, "task_id", task_id)
    set_if_present(row, "split", "synthetic_shape")
    set_if_present(row, "attempt_id", attempt_id)
    set_if_present(row, "library_id", f"ngd-synth:{stable_id(task_id, op_name)}")
    set_if_present(row, "attempt_index", attempt_index)
    set_if_present(row, "pipeline_kind", spec.pipeline_kind)
    set_if_present(row, "op_name", op_name)
    set_if_present(row, "visual_program", spec.visual_program)
    set_if_present(row, "slot_contract", spec.slot_contract)
    set_if_present(row, "composition_kind", spec.composition_kind)
    set_if_present(row, "normalization_frame", spec.normalization_frame)
    set_if_present(row, "typed_stages", spec.typed_stages)
    set_if_present(row, "primitive_contract", spec.primitive_contract)
    set_if_present(
        row,
        "slot_tokens",
        slot_tokens(
            op_index=op_index,
            grid_w=grid_w,
            grid_h=grid_h,
            shape_w=shape_w,
            shape_h=shape_h,
            kind=kind,
            color=color,
            output_color=output_color,
        ),
    )
    set_if_present(row, "accepted", accepted)
    set_if_present(row, "rejection_reason", rejection_reason)
    set_if_present(row, "failure_stage", failure_stage)
    set_if_present(row, "failure_detail", failure_detail)
    set_if_present(row, "proposal_count", 64)
    set_if_present(row, "support_tasks", 1)
    set_if_present(row, "utility_mean", f"{utility:.6f}")
    set_if_present(row, "selector", op_index)
    set_if_present(row, "background", 0)
    set_if_present(row, "color", color)
    set_if_present(row, "output_color", output_color)
    set_if_present(row, "selector_kind", op_index + 1)
    set_if_present(row, "include_background", 0)
    set_if_present(row, "orientation", 1 if shape_w >= shape_h else 2)
    set_if_present(row, "count_mode", 1 if "count" in spec.typed_stages else 0)
    set_if_present(row, "x_align", 1)
    set_if_present(row, "y_align", 1)
    set_if_present(row, "x_anchor", (grid_w - shape_w) // 2)
    set_if_present(row, "y_anchor", (grid_h - shape_h) // 2)
    set_if_present(row, "x_offset", max(0, grid_w - shape_w))
    set_if_present(row, "y_offset", max(0, grid_h - shape_h))
    set_if_present(row, "width", shape_w)
    set_if_present(row, "height", shape_h)
    set_if_present(row, "period", max(1, min(shape_w, shape_h)))
    set_if_present(row, "contrast_accepted_attempt_id", contrast_id)
    set_if_present(row, "contrast_slot_distance", contrast_distance)
    set_if_present(row, "contrast_pair_kind", contrast_kind)
    set_if_present(row, "contrast_rejected_sibling_count", negative_count)
    set_if_present(row, "input_width", grid_w)
    set_if_present(row, "input_height", grid_h)
    set_if_present(row, "output_width", shape_w)
    set_if_present(row, "output_height", shape_h)
    set_if_present(row, "shape_scale_x", shape_w)
    set_if_present(row, "shape_scale_y", shape_h)
    set_if_present(row, "period_divides_input_width", 1 if grid_w % max(1, shape_w) == 0 else 0)
    set_if_present(row, "period_divides_input_height", 1 if grid_h % max(1, shape_h) == 0 else 0)
    set_if_present(row, "slot_width_matches_output", slot_width_matches_output)
    set_if_present(row, "slot_height_matches_output", slot_height_matches_output)
    set_if_present(row, "color_in_input", 1)
    set_if_present(row, "color_in_output", 1)
    set_if_present(row, "output_color_in_output", 1)
    set_if_present(row, "background_in_input", 1)
    set_if_present(row, "verifier_train_pairs", 4)
    set_if_present(row, "verifier_checked_cells", max(1, area))
    set_if_present(row, "verifier_mismatches", verifier_mismatches)
    set_if_present(row, "verifier_unsupported_pairs", verifier_unsupported)
    set_if_present(row, "verifier_apply_failed_pairs", verifier_apply_failed)
    set_if_present(row, "verifier_shape_mismatch_pairs", verifier_shape_mismatch)
    set_if_present(row, "verifier_missing_output_pairs", 0)
    set_if_present(row, "verifier_first_failing_pair", -1 if accepted else 0)
    set_if_present(row, "verifier_mismatch_ratio", f"{mismatch_ratio:.6f}")
    set_if_present(row, "verifier_score", f"{verifier_score:.6f}")
    set_if_present(row, "input_component_count", 1)
    set_if_present(row, "selected_component_count", 1)
    set_if_present(row, "selected_component_area_min", area)
    set_if_present(row, "selected_component_area_max", area)
    set_if_present(row, "selected_component_area_total", area)
    set_if_present(row, "selected_component_same_shape", 1)
    set_if_present(row, "selected_component_shape_count", 1)
    set_if_present(row, "selected_component_bbox_width", shape_w)
    set_if_present(row, "selected_component_bbox_height", shape_h)
    set_if_present(row, "selected_bbox_matches_output_shape", slot_width_matches_output)
    set_if_present(row, "selected_bbox_rotated_output_shape", 1 if shape_w == shape_h else 0)
    set_if_present(row, "output_foreground_cell_count", area)
    set_if_present(row, "output_component_count", 1)
    set_if_present(row, "output_non_background_color_count", 1)
    set_if_present(row, "selected_count_matches_output_components", 1)
    set_if_present(row, "selected_count_matches_output_foreground", 1 if area == 1 else 0)
    set_if_present(row, "selected_area_matches_output_foreground", 1)
    return row


def copy_original_attempts(source: Path, target, expected_header: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    with source.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if header != expected_header:
            raise ValueError("attempt TSV header changed while copying")
        split_index = header.index("split_name")
        for line in handle:
            target.write(line)
            parts = line.rstrip("\n").split("\t")
            split_name = parts[split_index] if split_index < len(parts) else ""
            counts[f"original_{split_name or 'unknown'}_rows"] += 1
    return counts


def generate(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.attempts_tsv.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
    if not header or header[0] != "split_name":
        raise ValueError(f"unexpected attempts TSV header in {args.attempts_tsv}")

    ops = parse_csv(args.ops)
    unknown_ops = [op for op in ops if op not in PIPELINE_SPECS]
    if unknown_ops:
        raise ValueError(f"unknown synthetic ops: {', '.join(unknown_ops)}")
    negative_kinds = parse_csv(args.negative_kinds)
    allowed_negative_kinds = {"unsupported_shape", "no_train_match"}
    unknown_negative_kinds = [
        kind for kind in negative_kinds if kind not in allowed_negative_kinds
    ]
    if unknown_negative_kinds:
        raise ValueError(
            f"unknown negative kinds: {', '.join(unknown_negative_kinds)}"
        )
    if args.op_mode not in {"cycle", "all"}:
        raise ValueError("--op-mode must be cycle or all")

    synthetic_counts: Counter[str] = Counter()
    op_counts: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    attempt_index = 0
    accepted_rows = 0
    synthetic_rows = 0

    args.synthetic_attempts.parent.mkdir(parents=True, exist_ok=True)
    args.combined_attempts.parent.mkdir(parents=True, exist_ok=True)
    with args.synthetic_attempts.open("w", encoding="utf-8") as synthetic, (
        args.combined_attempts.open("w", encoding="utf-8")
    ) as combined:
        header_line = "\t".join(header) + "\n"
        synthetic.write(header_line)
        combined.write(header_line)
        original_counts = copy_original_attempts(args.attempts_tsv, combined, header)
        for shape_index, (grid_w, grid_h, shape_w, shape_h, kind) in enumerate(
            iter_shapes(args.max_grid_size)
        ):
            if args.max_accepted and accepted_rows >= args.max_accepted:
                break
            shape_ops = ops if args.op_mode == "all" else [ops[shape_index % len(ops)]]
            for op_name in shape_ops:
                if args.max_accepted and accepted_rows >= args.max_accepted:
                    break
                op_index = ops.index(op_name)
                accepted_id = "ngd-synth:{}".format(
                    stable_id(
                        "accepted",
                        grid_w,
                        grid_h,
                        shape_w,
                        shape_h,
                        kind,
                        op_name,
                    )
                )
                accepted = make_attempt_row(
                    header=header,
                    attempt_index=attempt_index,
                    grid_w=grid_w,
                    grid_h=grid_h,
                    shape_w=shape_w,
                    shape_h=shape_h,
                    kind=kind,
                    op_name=op_name,
                    op_index=op_index,
                    outcome="accepted",
                    accepted_attempt_id=accepted_id,
                    negative_count=len(negative_kinds),
                )
                text = row_text(header, accepted)
                synthetic.write(text)
                combined.write(text)
                attempt_index += 1
                accepted_rows += 1
                synthetic_rows += 1
                synthetic_counts["accepted"] += 1
                op_counts[op_name] += 1
                shape_counts[kind] += 1
                for negative_kind in negative_kinds:
                    negative = make_attempt_row(
                        header=header,
                        attempt_index=attempt_index,
                        grid_w=grid_w,
                        grid_h=grid_h,
                        shape_w=shape_w,
                        shape_h=shape_h,
                        kind=kind,
                        op_name=op_name,
                        op_index=op_index,
                        outcome=negative_kind,
                        accepted_attempt_id=accepted_id,
                        negative_count=len(negative_kinds),
                    )
                    text = row_text(header, negative)
                    synthetic.write(text)
                    combined.write(text)
                    attempt_index += 1
                    synthetic_rows += 1
                    synthetic_counts[negative_kind] += 1

    summary = {
        "schema": SCHEMA,
        "report_version": 1,
        "attempts_tsv": str(args.attempts_tsv),
        "synthetic_attempts": str(args.synthetic_attempts),
        "combined_attempts": str(args.combined_attempts),
        "max_grid_size": args.max_grid_size,
        "op_mode": args.op_mode,
        "ops": ops,
        "negative_kinds": negative_kinds,
        "max_accepted": args.max_accepted,
        "synthetic_rows": synthetic_rows,
        "synthetic_accepted_rows": synthetic_counts["accepted"],
        "synthetic_rejected_rows": synthetic_rows - synthetic_counts["accepted"],
        "synthetic_counts": dict(sorted(synthetic_counts.items())),
        "synthetic_op_counts": dict(sorted(op_counts.items())),
        "synthetic_shape_counts": dict(sorted(shape_counts.items())),
        **dict(sorted(original_counts.items())),
    }
    write_jsonl(args.summary, [summary], compact=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--synthetic-attempts", type=Path, required=True)
    parser.add_argument("--combined-attempts", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--max-grid-size", type=int, default=30)
    parser.add_argument("--ops", default=",".join(DEFAULT_OPS))
    parser.add_argument(
        "--negative-kinds",
        default="unsupported_shape,no_train_match",
        help="comma-separated synthetic rejection kinds",
    )
    parser.add_argument("--op-mode", choices=("cycle", "all"), default="cycle")
    parser.add_argument(
        "--max-accepted",
        type=int,
        default=0,
        help="cap accepted synthetic shapes for smoke runs; 0 means exhaustive",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_grid_size <= 0:
        raise SystemExit("--max-grid-size must be positive")
    if args.max_accepted < 0:
        raise SystemExit("--max-accepted must be non-negative")
    if not args.attempts_tsv.is_file():
        raise SystemExit(f"missing attempts TSV: {args.attempts_tsv}")
    summary = generate(args)
    print(
        "arc-synthetic-shape-curriculum "
        f"synthetic_rows={summary['synthetic_rows']} "
        f"accepted={summary['synthetic_accepted_rows']} "
        f"rejected={summary['synthetic_rejected_rows']} "
        f"combined={args.combined_attempts} summary={args.summary}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
