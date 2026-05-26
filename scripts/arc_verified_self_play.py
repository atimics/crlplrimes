#!/usr/bin/env python3
"""Summarize ARC verified self-play curriculum evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from crlp.io import read_jsonl_lenient, safe_float, safe_int, write_jsonl


SUMMARY_SCHEMA = "crlp.arc_verified_self_play.summary.v1"
DEFAULT_ZERO_COVERAGE_FAMILIES = [
    "object_motion_or_copy_extend",
    "relational_layout_or_counting",
    "projection_or_panel_selection",
    "color_remap_or_mode_collapse",
    "scale_tile_or_expand",
    "unsupported_parse_or_shape",
    "shape_completion_or_fill",
    "segmentation_or_object_extract",
    "symmetry_or_geometry",
]

OP_TARGETS = {
    "object_motion": {"object_motion_or_copy_extend"},
    "visual_combinator": {"object_motion_or_copy_extend", "relational_layout_or_counting"},
    "recolor": {"color_remap_or_mode_collapse"},
    "recolor_crop": {"color_remap_or_mode_collapse", "segmentation_or_object_extract"},
    "tile_scale": {"scale_tile_or_expand"},
    "macrocell_substitute": {"scale_tile_or_expand", "relational_layout_or_counting"},
    "component_color_summary": {"relational_layout_or_counting", "color_remap_or_mode_collapse"},
    "masked_or_normalized_crop": {"segmentation_or_object_extract"},
    "crop": {"segmentation_or_object_extract", "projection_or_panel_selection"},
    "separator_panel_map": {"projection_or_panel_selection", "scale_tile_or_expand"},
    "fill_or_complete": {"shape_completion_or_fill"},
    "symmetry": {"symmetry_or_geometry"},
}


def first_jsonl_row(path: Path) -> dict[str, Any]:
    rows = read_jsonl_lenient(path)
    return rows[0] if rows else {}


def load_zero_coverage_families(path: Path | None) -> list[str]:
    if path is None or not path.is_file():
        return list(DEFAULT_ZERO_COVERAGE_FAMILIES)
    families: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            category = str(row.get("category", "")).strip()
            if not category or category == "certified":
                continue
            if safe_int(row.get("certified_tasks")) == 0:
                families.append(category)
    return families or list(DEFAULT_ZERO_COVERAGE_FAMILIES)


def op_targets(ops: list[str]) -> list[str]:
    targets: set[str] = set()
    for op in ops:
        targets.update(OP_TARGETS.get(op, set()))
    return sorted(targets)


def load_public_eval_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    rows = read_jsonl_lenient(path, schema="crlp.arc_loader.summary.v1")
    return rows[-1] if rows else {}


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    curriculum = first_jsonl_row(args.curriculum_summary)
    ranker = first_jsonl_row(args.ranker_summary) if args.ranker_summary else {}
    public_eval = load_public_eval_summary(args.public_eval_summary)
    ops = [str(op) for op in curriculum.get("ops", []) or []]
    zero_families = load_zero_coverage_families(args.miss_taxonomy)
    targets = op_targets(ops)
    targeted_zero = [family for family in zero_families if family in targets]
    untargeted_zero = [family for family in zero_families if family not in targets]

    eval_certified = safe_int(public_eval.get("certified_tasks"), default=-1)
    eval_delta = None if eval_certified < 0 else eval_certified - args.template_baseline
    oracle_gap = None if eval_certified < 0 else args.oracle_ceiling - eval_certified

    return {
        "schema": SUMMARY_SCHEMA,
        "report_version": 1,
        "training_loop": "verified_self_play_inverse_program_induction",
        "supervision_source": "forward primitive composition with exact verifier-compatible labels",
        "authored_task_library": False,
        "authorized_knowledge": ["visual_primitives", "exact_verifier", "public_miss_taxonomy_bias"],
        "curriculum_summary": str(args.curriculum_summary),
        "ranker_summary": str(args.ranker_summary) if args.ranker_summary else "",
        "public_eval_summary": str(args.public_eval_summary) if args.public_eval_summary else "",
        "synthetic_rows": safe_int(curriculum.get("synthetic_rows")),
        "synthetic_accepted_rows": safe_int(curriculum.get("synthetic_accepted_rows")),
        "synthetic_rejected_rows": safe_int(curriculum.get("synthetic_rejected_rows")),
        "synthetic_negative_kinds": curriculum.get("negative_kinds", []),
        "primitive_ops": ops,
        "max_grid_size": safe_int(curriculum.get("max_grid_size")),
        "op_mode": str(curriculum.get("op_mode", "")),
        "inverse_ranker": {
            "training_mode": ranker.get("training_mode", ""),
            "source_attempt_count": safe_int(ranker.get("source_attempt_count")),
            "source_accepted_count": safe_int(ranker.get("source_accepted_count")),
            "target_task_count": safe_int(ranker.get("target_task_count")),
            "accepted_target_task_count": safe_int(ranker.get("accepted_target_task_count")),
            "top1_rate": safe_float(ranker.get("top1_rate")),
            "top1_all_task_rate": safe_float(ranker.get("top1_all_task_rate")),
            "pairwise_last_epoch_violations": safe_int(
                ranker.get("pairwise_last_epoch_violations")
            ),
        },
        "yardsticks": {
            "template_baseline_certified_eval_tasks": args.template_baseline,
            "oracle_ceiling_certified_eval_tasks": args.oracle_ceiling,
            "zero_coverage_family_count": len(zero_families),
            "zero_coverage_families": zero_families,
        },
        "family_bias": {
            "targeted_zero_coverage_families": targeted_zero,
            "untargeted_zero_coverage_families": untargeted_zero,
            "targeted_zero_coverage_family_count": len(targeted_zero),
            "targeted_zero_coverage_family_rate": (
                0.0 if not zero_families else len(targeted_zero) / len(zero_families)
            ),
        },
        "public_eval": {
            "available": bool(public_eval),
            "certified_tasks": None if eval_certified < 0 else eval_certified,
            "cell_accuracy": safe_float(public_eval.get("cell_accuracy")),
            "delta_vs_template_baseline": eval_delta,
            "gap_to_oracle_ceiling": oracle_gap,
        },
        "claim_scope": (
            "self-play rows provide unlimited exact supervision for inverse proposer "
            "training; ARC competence is only measured after replay through the public "
            "ARC verifier and reported against the declared yardsticks"
        ),
        "distribution_mismatch_risk": (
            "forward generated tasks may not match ARC evaluation; generation is biased "
            "toward the current zero-coverage miss taxonomy to keep pressure on known holes"
        ),
    }


def write_markdown(path: Path, row: dict[str, Any]) -> None:
    inv = row["inverse_ranker"]
    family = row["family_bias"]
    public_eval = row["public_eval"]
    lines = [
        "# ARC Verified Self-Play",
        "",
        "The generator composes visual primitives forward, labels accepted rows by construction,",
        "adds verifier-shaped negative siblings, and trains the proposer on the inverse problem.",
        "",
        "| Rows | Accepted | Rejected | Ops | Targeted zero-coverage families |",
        "| ---: | ---: | ---: | ---: | ---: |",
        "| {rows} | {accepted} | {rejected} | {ops} | {targeted}/{total} |".format(
            rows=row["synthetic_rows"],
            accepted=row["synthetic_accepted_rows"],
            rejected=row["synthetic_rejected_rows"],
            ops=len(row["primitive_ops"]),
            targeted=family["targeted_zero_coverage_family_count"],
            total=row["yardsticks"]["zero_coverage_family_count"],
        ),
        "",
        "## Inverse Proposer",
        "",
        "| Target tasks | Accepted target tasks | Top-1 | All-task top-1 | Pairwise violations |",
        "| ---: | ---: | ---: | ---: | ---: |",
        "| {target} | {accepted} | {top1:.3f} | {all_top1:.3f} | {violations} |".format(
            target=inv["target_task_count"],
            accepted=inv["accepted_target_task_count"],
            top1=inv["top1_rate"],
            all_top1=inv["top1_all_task_rate"],
            violations=inv["pairwise_last_epoch_violations"],
        ),
        "",
        "## Yardsticks",
        "",
        "- Template baseline certified eval tasks: `{}`".format(
            row["yardsticks"]["template_baseline_certified_eval_tasks"]
        ),
        "- Oracle ceiling certified eval tasks: `{}`".format(
            row["yardsticks"]["oracle_ceiling_certified_eval_tasks"]
        ),
        "- Zero-coverage families: `{}`".format(
            row["yardsticks"]["zero_coverage_family_count"]
        ),
        "- Targeted zero-coverage families: `{}`".format(
            ", ".join(family["targeted_zero_coverage_families"]) or "none"
        ),
        "- Untargeted zero-coverage families: `{}`".format(
            ", ".join(family["untargeted_zero_coverage_families"]) or "none"
        ),
        "",
        "## Public Eval",
        "",
    ]
    if public_eval["available"]:
        lines.extend(
            [
                "- Certified eval tasks: `{}`".format(public_eval["certified_tasks"]),
                "- Delta vs template baseline: `{}`".format(
                    public_eval["delta_vs_template_baseline"]
                ),
                "- Gap to oracle ceiling: `{}`".format(public_eval["gap_to_oracle_ceiling"]),
            ]
        )
    else:
        lines.append(
            "No public eval summary was provided; self-play evidence remains a training-signal diagnostic."
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum-summary", type=Path, required=True)
    parser.add_argument("--ranker-summary", type=Path)
    parser.add_argument("--public-eval-summary", type=Path)
    parser.add_argument("--miss-taxonomy", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--template-baseline", type=int, default=7)
    parser.add_argument("--oracle-ceiling", type=int, default=14)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    row = build_summary(args)
    write_jsonl(args.output, [row], compact=True)
    if args.markdown:
        write_markdown(args.markdown, row)
    print(f"wrote {args.output}")
    if args.markdown:
        print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
