#!/usr/bin/env python3
"""Export ARC verified self-play accepted attempts as an ARC visual prior."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from crlp.io import safe_float, safe_int, write_jsonl


SUMMARY_SCHEMA = "crlp.arc_verified_self_play.prior.v1"
GENERATIVE_SUMMARY_SCHEMA = "crlp.arc_generative_prior.summary.v1"
TEMPLATE_ID = 1036
TEMPLATE_NAME = "arc_sketch_program"

PRIOR_HEADER = [
    "schema_version",
    "template_id",
    "template_name",
    "library_id",
    "slot_name",
    "slot_type",
    "slot_value",
    "slot_integer",
    "proposal_count",
    "certified_count",
    "support_tasks",
    "covered_events",
    "utility_mean",
]

GENERATIVE_PRIOR_HEADER = [
    "schema_version",
    "op_id",
    "op_name",
    "match_scope",
    "input_width",
    "input_height",
    "output_width",
    "output_height",
    "shape_scale_x",
    "shape_scale_y",
    "input_component_count",
    "output_component_count",
    "accepted_count",
    "rejected_count",
    "near_miss_rejected_count",
    "hard_rejected_count",
    "mean_verifier_score",
    "rejection_penalty",
    "score",
]

# Keep this aligned with ARC_SKETCH_OP_* in examples/arc_agi.c. Synthetic
# primitives without an executable public ARC compiler stay out of the prior.
OP_ID_BY_NAME = {
    "crop": 0,
    "tile_repeat": 1,
    "tile_scale": 1,
    "macrocell_substitute": 2,
    "component_color_summary": 4,
    "normalized_crop": 5,
    "masked_or_normalized_crop": 5,
    "point_line_project": 6,
    "object_mask_extract": 7,
    "axis_projection_compress": 8,
    "separator_panel_map": 9,
    "palette_scale": 10,
    "object_color_tile": 11,
    "marker_count_motif_tile": 12,
    "component_count_line": 13,
    "recolor_crop": 14,
    "marker_polyline": 15,
    "separator_intersection_color": 16,
    "separator_panel_relation_mask": 17,
    "visual_combinator": 18,
    "extend_aligned_sequence": 19,
    "object_translate": 20,
}

GENERATIVE_OP_IDS_BY_NAME = {
    "object_motion": [20],
    "recolor": [14],
    "tile_scale": [1, 2, 10],
    "fill_or_complete": [18],
    "symmetry": [15, 18],
    "masked_or_normalized_crop": [5, 7],
    "crop": [0, 5],
    "recolor_crop": [14],
    "macrocell_substitute": [2],
    "component_color_summary": [4, 13],
    "visual_combinator": [18],
    "separator_panel_map": [9, 17],
}

OP_NAME_BY_ID = {
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
    19: "extend_aligned_sequence",
    20: "object_translate",
}

NEAR_MISS_FAILURE_STAGES = {
    "renderer_cells",
    "transform_or_color",
    "verifier_distance",
}

HARD_REJECTION_FAILURE_STAGES = {
    "compiler",
    "normalizer_shape",
    "renderer_shape",
    "selector_or_transform",
    "unsupported_shape",
}

SLOT_SPECS = [
    ("sketch.selector", "integer", "selector"),
    ("sketch.op", "integer", None),
    ("sketch.background", "color", "background"),
    ("sketch.color", "color", "color"),
    ("sketch.output_color", "color", "output_color"),
    ("sketch.x_anchor", "integer", "x_anchor"),
    ("sketch.y_anchor", "integer", "y_anchor"),
    ("sketch.x_offset", "offset", "x_offset"),
    ("sketch.y_offset", "offset", "y_offset"),
    ("sketch.width", "integer", "width"),
    ("sketch.height", "integer", "height"),
    ("sketch.period", "integer", "period"),
]


def load_attempts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def slot_tokens(row: dict[str, str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in str(row.get("slot_tokens", "")).split("|"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            out[key] = int(float(value))
        except ValueError:
            continue
    return out


def row_int(row: dict[str, str], tokens: dict[str, int], slot_name: str, column: str | None) -> int:
    if slot_name in tokens:
        return tokens[slot_name]
    if column:
        return safe_int(row.get(column))
    return 0


def prior_key(row: dict[str, str], tokens: dict[str, int], op_id: int) -> tuple[int, ...]:
    values = [op_id]
    for slot_name, _slot_type, column in SLOT_SPECS:
        if slot_name == "sketch.op":
            continue
        values.append(row_int(row, tokens, slot_name, column))
    return tuple(values)


def prior_line(
    schema: str,
    library_id: str,
    slot_name: str,
    slot_type: str,
    slot_integer: int,
    proposal_count: int,
    certified_count: int,
    support_tasks: int,
    utility_mean: float,
) -> dict[str, Any]:
    return {
        "schema_version": schema,
        "template_id": TEMPLATE_ID,
        "template_name": TEMPLATE_NAME,
        "library_id": library_id,
        "slot_name": slot_name,
        "slot_type": slot_type,
        "slot_value": f"{float(slot_integer):.8f}",
        "slot_integer": slot_integer,
        "proposal_count": proposal_count,
        "certified_count": certified_count,
        "support_tasks": support_tasks,
        "covered_events": support_tasks,
        "utility_mean": f"{utility_mean:.8f}",
    }


def build_prior_rows(attempts: list[dict[str, str]], max_rows: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prior_rows: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    accepted_rows = 0
    mapped_rows = 0
    skipped_duplicate = 0
    skipped_unmapped = 0
    skipped_unaccepted = 0
    skipped_unverified = 0
    op_counts: Counter[str] = Counter()
    unmapped_ops: Counter[str] = Counter()

    for row in attempts:
        if safe_int(row.get("accepted")) != 1:
            skipped_unaccepted += 1
            continue
        provenance = row.get("provenance")
        state_hash = row.get("state_hash")
        if provenance is not None and str(provenance).strip() != "selfplay_forward":
            skipped_unverified += 1
            continue
        if state_hash is not None and not str(state_hash).strip():
            skipped_unverified += 1
            continue
        accepted_rows += 1
        op_name = str(row.get("op_name", "")).strip()
        op_id = OP_ID_BY_NAME.get(op_name)
        if op_id is None:
            skipped_unmapped += 1
            unmapped_ops[op_name or "unknown"] += 1
            continue
        tokens = slot_tokens(row)
        key = prior_key(row, tokens, op_id)
        if key in seen:
            skipped_duplicate += 1
            continue
        if max_rows > 0 and mapped_rows >= max_rows:
            break
        seen.add(key)
        mapped_rows += 1
        op_counts[op_name] += 1

        library_id = str(row.get("library_id") or row.get("attempt_id") or f"self_play:{mapped_rows}")
        proposal_count = max(1, safe_int(row.get("proposal_count")))
        support_tasks = max(1, safe_int(row.get("support_tasks")))
        certified_count = support_tasks
        utility_mean = safe_float(row.get("utility_mean"))

        prior_rows.append(
            prior_line(
                "crlp.hypothesis_library.template_prior.v1",
                library_id,
                "__template__",
                "template",
                0,
                proposal_count,
                certified_count,
                support_tasks,
                utility_mean,
            )
        )
        for slot_name, slot_type, column in SLOT_SPECS:
            slot_integer = op_id if slot_name == "sketch.op" else row_int(row, tokens, slot_name, column)
            prior_rows.append(
                prior_line(
                    "crlp.hypothesis_library.prior.v1",
                    library_id,
                    slot_name,
                    slot_type,
                    slot_integer,
                    proposal_count,
                    certified_count,
                    support_tasks,
                    utility_mean,
                )
            )

    summary = {
        "schema": SUMMARY_SCHEMA,
        "report_version": 1,
        "input_attempt_rows": len(attempts),
        "accepted_attempt_rows": accepted_rows,
        "mapped_accepted_attempt_rows": mapped_rows,
        "skipped_unaccepted_rows": skipped_unaccepted,
        "skipped_unverified_accepted_rows": skipped_unverified,
        "skipped_unmapped_accepted_rows": skipped_unmapped,
        "skipped_duplicate_accepted_rows": skipped_duplicate,
        "prior_rows": len(prior_rows),
        "template_prior_rows": mapped_rows,
        "slot_prior_rows": len(prior_rows) - mapped_rows,
        "mapped_ops": dict(sorted(op_counts.items())),
        "unmapped_ops": dict(sorted(unmapped_ops.items())),
        "op_id_contract": "ARC_SKETCH_OP ids from examples/arc_agi.c",
    }
    return prior_rows, summary


def generative_scope_key(row: dict[str, str], scope: str, op_id: int) -> tuple[int, str, int, ...]:
    if scope == "global":
        values = (0, 0, 0, 0, 0, 0, 0, 0)
    elif scope == "shape":
        values = (
            safe_int(row.get("input_width")),
            safe_int(row.get("input_height")),
            safe_int(row.get("output_width")),
            safe_int(row.get("output_height")),
            safe_int(row.get("shape_scale_x")),
            safe_int(row.get("shape_scale_y")),
            0,
            0,
        )
    elif scope == "scale":
        values = (
            0,
            0,
            0,
            0,
            safe_int(row.get("shape_scale_x")),
            safe_int(row.get("shape_scale_y")),
            0,
            0,
        )
    elif scope == "components":
        values = (
            0,
            0,
            0,
            0,
            0,
            0,
            safe_int(row.get("input_component_count")),
            safe_int(row.get("output_component_count")),
        )
    else:
        raise ValueError(f"unknown generative scope: {scope}")
    return (op_id, scope, *values)


def build_generative_prior_rows(
    attempts: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counts: Counter[tuple[int, str, int, ...]] = Counter()
    accepted_counts: Counter[tuple[int, str, int, ...]] = Counter()
    rejected_counts: Counter[tuple[int, str, int, ...]] = Counter()
    near_miss_rejected_counts: Counter[tuple[int, str, int, ...]] = Counter()
    hard_rejected_counts: Counter[tuple[int, str, int, ...]] = Counter()
    verifier_score_sums: Counter[tuple[int, str, int, ...]] = Counter()
    source_ops: Counter[str] = Counter()
    mapped_ops: Counter[str] = Counter()
    unmapped_ops: Counter[str] = Counter()
    rejection_stages: Counter[str] = Counter()
    accepted_rows = 0
    rejected_rows = 0
    skipped_unverified = 0
    for row in attempts:
        provenance = row.get("provenance")
        state_hash = row.get("state_hash")
        if provenance is not None and str(provenance).strip() != "selfplay_forward":
            skipped_unverified += 1
            continue
        if state_hash is not None and not str(state_hash).strip():
            skipped_unverified += 1
            continue
        op_name = str(row.get("op_name", "")).strip()
        op_ids = GENERATIVE_OP_IDS_BY_NAME.get(op_name)
        if not op_ids:
            unmapped_ops[op_name or "unknown"] += 1
            continue
        accepted = safe_int(row.get("accepted")) == 1
        failure_stage = str(row.get("failure_stage", "")).strip() or "unknown"
        verifier_score = safe_float(row.get("verifier_score"))
        accepted_rows += 1 if accepted else 0
        rejected_rows += 0 if accepted else 1
        if not accepted:
            rejection_stages[failure_stage] += 1
        source_ops[op_name] += 1
        for op_id in op_ids:
            mapped_ops[OP_NAME_BY_ID.get(op_id, str(op_id))] += 1
            for scope in ("global", "scale", "shape", "components"):
                key = generative_scope_key(row, scope, op_id)
                counts[key] += 1
                verifier_score_sums[key] += verifier_score
                if accepted:
                    accepted_counts[key] += 1
                else:
                    rejected_counts[key] += 1
                    if failure_stage in NEAR_MISS_FAILURE_STAGES or verifier_score > 0.0:
                        near_miss_rejected_counts[key] += 1
                    if failure_stage in HARD_REJECTION_FAILURE_STAGES:
                        hard_rejected_counts[key] += 1

    rows: list[dict[str, Any]] = []
    for key, total in sorted(counts.items()):
        op_id = key[0]
        scope = key[1]
        values = key[2:]
        accepted = accepted_counts[key]
        rejected = rejected_counts[key]
        near_miss_rejected = near_miss_rejected_counts[key]
        hard_rejected = hard_rejected_counts[key]
        mean_verifier_score = verifier_score_sums[key] / total if total else 0.0
        rejection_penalty = (
            1.4 * hard_rejected
            + 0.7 * max(0, rejected - near_miss_rejected - hard_rejected)
            + 0.35 * near_miss_rejected
        )
        score = (accepted + 0.25 * near_miss_rejected + 1.0) / (
            accepted + rejection_penalty + 2.0
        )
        rows.append(
            {
                "schema_version": "crlp.arc_generative_prior.v1",
                "op_id": op_id,
                "op_name": OP_NAME_BY_ID.get(op_id, str(op_id)),
                "match_scope": scope,
                "input_width": values[0],
                "input_height": values[1],
                "output_width": values[2],
                "output_height": values[3],
                "shape_scale_x": values[4],
                "shape_scale_y": values[5],
                "input_component_count": values[6],
                "output_component_count": values[7],
                "accepted_count": accepted,
                "rejected_count": rejected,
                "near_miss_rejected_count": near_miss_rejected,
                "hard_rejected_count": hard_rejected,
                "mean_verifier_score": f"{mean_verifier_score:.8f}",
                "rejection_penalty": f"{rejection_penalty:.8f}",
                "score": f"{score:.8f}",
            }
        )
    summary = {
        "schema": GENERATIVE_SUMMARY_SCHEMA,
        "report_version": 1,
        "input_attempt_rows": len(attempts),
        "accepted_attempt_rows": accepted_rows,
        "rejected_attempt_rows": rejected_rows,
        "skipped_unverified_rows": skipped_unverified,
        "generative_prior_rows": len(rows),
        "source_ops": dict(sorted(source_ops.items())),
        "rejection_failure_stages": dict(sorted(rejection_stages.items())),
        "mapped_runtime_ops": dict(sorted(mapped_ops.items())),
        "unmapped_ops": dict(sorted(unmapped_ops.items())),
        "op_id_contract": "ARC_SKETCH_OP ids from examples/arc_agi.c",
        "rejection_training": {
            "near_miss_failure_stages": sorted(NEAR_MISS_FAILURE_STAGES),
            "hard_rejection_failure_stages": sorted(HARD_REJECTION_FAILURE_STAGES),
            "score_formula": "(accepted + 0.25 * near_miss + 1) / (accepted + weighted_rejections + 2)",
        },
        "claim_scope": "rejection-aware task-conditioned runtime miner scheduling only; candidates still require train-pair verifier acceptance",
    }
    return rows, summary


def write_prior(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=PRIOR_HEADER, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_generative_prior(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=GENERATIVE_PRIOR_HEADER,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--generative-output", type=Path)
    parser.add_argument("--generative-summary", type=Path)
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    attempts = load_attempts(args.attempts)
    prior_rows, summary = build_prior_rows(attempts, args.max_rows)
    summary["attempts"] = str(args.attempts)
    summary["prior"] = str(args.output)
    write_prior(args.output, prior_rows)
    write_jsonl(args.summary, [summary], compact=True)
    print(f"wrote {args.output}")
    print(f"wrote {args.summary}")
    if args.generative_output:
        generative_rows, generative_summary = build_generative_prior_rows(attempts)
        generative_summary["attempts"] = str(args.attempts)
        generative_summary["generative_prior"] = str(args.generative_output)
        write_generative_prior(args.generative_output, generative_rows)
        if args.generative_summary:
            write_jsonl(args.generative_summary, [generative_summary], compact=True)
            print(f"wrote {args.generative_summary}")
        print(f"wrote {args.generative_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
