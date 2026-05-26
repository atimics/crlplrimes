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


def load_summary(path: Path | None, schema: str) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    rows = read_jsonl_lenient(path, schema=schema)
    return rows[-1] if rows else {}


def load_public_eval_summary(path: Path | None) -> dict[str, Any]:
    return load_summary(path, "crlp.arc_loader.summary.v1")


def summarize_hypothesis_runtime(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "available": False,
            "task_rows": 0,
            "runtime_mined_sketches": 0,
            "runtime_generative_prior_rows": 0,
            "runtime_generative_prior_ops": 0,
            "runtime_generative_prior_mined_sketches": 0,
        }
    rows = read_jsonl_lenient(path, schema="crlp.arc_hypothesis.selection.v1")
    return {
        "available": bool(rows),
        "task_rows": len(rows),
        "runtime_mined_sketches": sum(
            safe_int(row.get("runtime_mined_sketches")) for row in rows
        ),
        "runtime_generative_prior_rows": max(
            [safe_int(row.get("runtime_generative_prior_rows")) for row in rows]
            or [0]
        ),
        "runtime_generative_prior_ops": sum(
            safe_int(row.get("runtime_generative_prior_ops")) for row in rows
        ),
        "runtime_generative_prior_mined_sketches": sum(
            safe_int(row.get("runtime_generative_prior_mined_sketches"))
            for row in rows
        ),
    }


def summarize_generative_runtime_trace(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "available": False,
            "trace_rows": 0,
            "decision_counts": {},
            "selected_ops": 0,
            "emitted_sketches": 0,
            "filtered_by_rejection_score": 0,
        }
    rows = read_jsonl_lenient(
        path, schema="crlp.arc_generative_prior.runtime_trace.v1"
    )
    decision_counts: dict[str, int] = {}
    for row in rows:
        decision = str(row.get("decision", "unknown"))
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
    return {
        "available": bool(rows),
        "trace_rows": len(rows),
        "decision_counts": dict(sorted(decision_counts.items())),
        "selected_ops": decision_counts.get("selected", 0),
        "emitted_sketches": sum(safe_int(row.get("emitted_sketches")) for row in rows),
        "filtered_by_rejection_score": decision_counts.get(
            "filtered_rejection_score", 0
        ),
        "mean_rejection_trained_score": (
            sum(safe_float(row.get("rejection_trained_score")) for row in rows)
            / len(rows)
            if rows
            else 0.0
        ),
    }


def summarize_object_translate_miner_trace(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "available": False,
            "trace_rows": 0,
            "attempted_specs": 0,
            "accepted_specs": 0,
            "unsupported_shape": 0,
            "no_train_match": 0,
            "all_color_attempts": 0,
            "all_color_accepted": 0,
            "single_object_attempts": 0,
            "single_object_accepted": 0,
            "best_rejected_verifier_score": 0.0,
        }
    rows = read_jsonl_lenient(
        path, schema="crlp.arc_object_translate_miner.trace.v1"
    )
    best_row = max(
        rows,
        key=lambda row: safe_float(row.get("best_rejected_verifier_score")),
        default={},
    )
    return {
        "available": bool(rows),
        "trace_rows": len(rows),
        "attempted_specs": sum(safe_int(row.get("attempted_specs")) for row in rows),
        "accepted_specs": sum(safe_int(row.get("accepted_specs")) for row in rows),
        "unsupported_shape": sum(safe_int(row.get("unsupported_shape")) for row in rows),
        "no_train_match": sum(safe_int(row.get("no_train_match")) for row in rows),
        "duplicate": sum(safe_int(row.get("duplicate")) for row in rows),
        "cap_hit": sum(safe_int(row.get("cap_hit")) for row in rows),
        "all_color_attempts": sum(
            safe_int(row.get("all_color_attempts")) for row in rows
        ),
        "all_color_accepted": sum(
            safe_int(row.get("all_color_accepted")) for row in rows
        ),
        "single_object_attempts": sum(
            safe_int(row.get("single_object_attempts")) for row in rows
        ),
        "single_object_accepted": sum(
            safe_int(row.get("single_object_accepted")) for row in rows
        ),
        "best_rejected_task": str(best_row.get("task_instance", "")),
        "best_rejected_reason": str(best_row.get("best_rejected_reason", "")),
        "best_rejected_verifier_score": safe_float(
            best_row.get("best_rejected_verifier_score")
        ),
        "best_rejected_mismatches": safe_int(
            best_row.get("best_rejected_mismatches")
        ),
        "best_rejected_dx": safe_int(best_row.get("best_rejected_dx")),
        "best_rejected_dy": safe_int(best_row.get("best_rejected_dy")),
        "best_rejected_render_mode": safe_int(
            best_row.get("best_rejected_render_mode")
        ),
    }


def summarize_reference_recolor_mask_miner_trace(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "available": False,
            "trace_rows": 0,
            "attempted_specs": 0,
            "accepted_specs": 0,
            "accepted_task_count": 0,
            "accepted_tasks": [],
        }
    rows = read_jsonl_lenient(
        path, schema="crlp.arc_reference_recolor_mask_miner.trace.v1"
    )
    accepted_tasks = sorted(
        {
            str(row.get("task_instance", ""))
            for row in rows
            if safe_int(row.get("accepted_specs")) > 0
            and str(row.get("task_instance", ""))
        }
    )
    return {
        "available": bool(rows),
        "trace_rows": len(rows),
        "attempted_specs": sum(safe_int(row.get("attempted_specs")) for row in rows),
        "accepted_specs": sum(safe_int(row.get("accepted_specs")) for row in rows),
        "accepted_task_count": len(accepted_tasks),
        "accepted_tasks": accepted_tasks,
        "compiler_target": "visual_combinator" if rows else "",
        "visual_program": "reference_recolor_mask" if rows else "",
    }


def summarize_arc_public(row: dict[str, Any]) -> dict[str, Any]:
    certified = safe_int(row.get("certified_tasks"), default=-1)
    certified_cells = safe_int(row.get("certified_cells"), default=-1)
    if certified_cells < 0:
        certified_cells = safe_int(row.get("test_correct_cells"))
    return {
        "available": bool(row),
        "loaded_tasks": safe_int(row.get("loaded_tasks")),
        "certified_tasks": None if certified < 0 else certified,
        "test_exact_tasks": safe_int(row.get("test_exact_tasks")),
        "certified_cells": certified_cells,
        "cell_accuracy": safe_float(row.get("cell_accuracy")),
    }


def summarize_budget_curves(path: Path | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "available": False,
        "source": str(path) if path else "",
        "fixed_budget_required": True,
        "known_budget_confound": False,
        "learned_budget_32": None,
        "learned_budget_64": None,
        "learned_delta_32_to_64": None,
        "oracle_budget_32": None,
        "oracle_budget_64": None,
    }
    if path is None or not path.is_file():
        return out
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values: dict[tuple[str, int], int] = {}
    for row in rows:
        if str(row.get("split", "")) != "evaluation":
            continue
        strategy = str(row.get("strategy", ""))
        budget = safe_int(row.get("budget"), default=-1)
        if budget < 0:
            continue
        values[(strategy, budget)] = safe_int(row.get("certified_programs"))
    learned_32 = values.get(("learned", 32))
    learned_64 = values.get(("learned", 64))
    out.update(
        {
            "available": bool(values),
            "learned_budget_32": learned_32,
            "learned_budget_64": learned_64,
            "learned_delta_32_to_64": (
                None
                if learned_32 is None or learned_64 is None
                else learned_64 - learned_32
            ),
            "oracle_budget_32": values.get(("oracle", 32)),
            "oracle_budget_64": values.get(("oracle", 64)),
        }
    )
    out["known_budget_confound"] = bool(
        out["learned_delta_32_to_64"] is not None
        and safe_int(out["learned_delta_32_to_64"]) > 0
    )
    return out


def load_state_hashes(path: Path | None) -> set[str]:
    hashes: set[str] = set()
    if path is None or not path.is_file():
        return hashes
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = str(row.get("state_hash", "")).strip()
                if value:
                    hashes.add(value)
            elif line != "state_hash":
                hashes.add(line.split(",", 1)[0].split("\t", 1)[0])
    return hashes


def summarize_leakage_audit(
    synthetic_attempts: Path | None,
    eval_state_hashes: Path | None,
    public_eval_available: bool,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "available": False,
        "synthetic_attempts": str(synthetic_attempts) if synthetic_attempts else "",
        "eval_state_hashes": str(eval_state_hashes) if eval_state_hashes else "",
        "synthetic_rows": 0,
        "accepted_rows": 0,
        "rows_with_state_hash": 0,
        "rows_with_selfplay_forward_provenance": 0,
        "unique_state_hashes": 0,
        "state_hash_contract_ready": False,
        "eval_hashes_available": False,
        "eval_state_hash_count": 0,
        "disjoint_from_public_eval": None,
        "public_eval_precondition_satisfied": False,
        "blocked_reason": "",
    }
    if synthetic_attempts is None or not synthetic_attempts.is_file():
        out["blocked_reason"] = "synthetic attempts TSV missing"
        return out
    with synthetic_attempts.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "state_hash" not in reader.fieldnames:
            out["blocked_reason"] = "synthetic attempt rows do not expose state_hash"
            return out
        if "provenance" not in reader.fieldnames:
            out["blocked_reason"] = "synthetic attempt rows do not expose provenance"
            return out
        state_hashes: set[str] = set()
        for row in reader:
            out["synthetic_rows"] += 1
            if safe_int(row.get("accepted")) == 1:
                out["accepted_rows"] += 1
            state_hash = str(row.get("state_hash", "")).strip()
            provenance = str(row.get("provenance", "")).strip()
            if state_hash:
                out["rows_with_state_hash"] += 1
                state_hashes.add(state_hash)
            if provenance == "selfplay_forward":
                out["rows_with_selfplay_forward_provenance"] += 1
    out["available"] = True
    out["unique_state_hashes"] = len(state_hashes)
    out["state_hash_contract_ready"] = (
        out["synthetic_rows"] > 0
        and out["rows_with_state_hash"] == out["synthetic_rows"]
        and out["rows_with_selfplay_forward_provenance"] == out["synthetic_rows"]
    )
    eval_hashes = load_state_hashes(eval_state_hashes)
    out["eval_hashes_available"] = bool(eval_hashes)
    out["eval_state_hash_count"] = len(eval_hashes)
    if eval_hashes:
        out["disjoint_from_public_eval"] = state_hashes.isdisjoint(eval_hashes)
    if public_eval_available:
        out["public_eval_precondition_satisfied"] = bool(
            out["state_hash_contract_ready"] and out["disjoint_from_public_eval"] is True
        )
        if not out["public_eval_precondition_satisfied"] and not out["blocked_reason"]:
            out["blocked_reason"] = "public eval replay requires eval state hashes and disjointness"
    else:
        out["public_eval_precondition_satisfied"] = False
        if not out["blocked_reason"]:
            out["blocked_reason"] = "public eval replay not supplied"
    return out


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    curriculum = first_jsonl_row(args.curriculum_summary)
    ranker = first_jsonl_row(args.ranker_summary) if args.ranker_summary else {}
    prior = load_summary(args.prior_summary, "crlp.arc_verified_self_play.prior.v1")
    generative_prior = load_summary(
        args.generative_prior_summary, "crlp.arc_generative_prior.summary.v1"
    )
    public_train = load_public_eval_summary(args.public_train_summary)
    public_train_baseline = load_public_eval_summary(args.public_train_baseline_summary)
    public_train_visual_baseline = load_public_eval_summary(
        args.public_train_visual_baseline_summary
    )
    public_eval = load_public_eval_summary(args.public_eval_summary)
    public_eval_baseline = load_public_eval_summary(args.public_eval_baseline_summary)
    public_eval_visual_baseline = load_public_eval_summary(
        args.public_eval_visual_baseline_summary
    )
    public_train_runtime = summarize_hypothesis_runtime(args.public_train_summary)
    public_eval_runtime = summarize_hypothesis_runtime(args.public_eval_summary)
    public_train_trace = summarize_generative_runtime_trace(args.public_train_summary)
    public_eval_trace = summarize_generative_runtime_trace(args.public_eval_summary)
    public_train_object_translate_trace = summarize_object_translate_miner_trace(
        args.public_train_summary
    )
    public_eval_object_translate_trace = summarize_object_translate_miner_trace(
        args.public_eval_summary
    )
    public_train_reference_recolor_trace = summarize_reference_recolor_mask_miner_trace(
        args.public_train_summary
    )
    public_eval_reference_recolor_trace = summarize_reference_recolor_mask_miner_trace(
        args.public_eval_summary
    )
    budget_confound = summarize_budget_curves(args.budget_curves)
    ops = [str(op) for op in curriculum.get("ops", []) or []]
    zero_families = load_zero_coverage_families(args.miss_taxonomy)
    targets = op_targets(ops)
    targeted_zero = [family for family in zero_families if family in targets]
    untargeted_zero = [family for family in zero_families if family not in targets]

    eval_certified = safe_int(public_eval.get("certified_tasks"), default=-1)
    eval_certified_cells = safe_int(public_eval.get("certified_cells"), default=-1)
    if eval_certified_cells < 0:
        eval_certified_cells = safe_int(public_eval.get("test_correct_cells"))
    full_baseline_certified = safe_int(
        public_eval_baseline.get("certified_tasks"), default=-1
    )
    visual_baseline_certified = safe_int(
        public_eval_visual_baseline.get("certified_tasks"), default=-1
    )
    matched_baseline_certified = (
        visual_baseline_certified
        if visual_baseline_certified >= 0
        else full_baseline_certified
    )
    matched_baseline_kind = (
        "visual_only_no_prior"
        if visual_baseline_certified >= 0
        else "full_program_no_prior"
    )
    eval_delta = None if eval_certified < 0 else eval_certified - args.template_baseline
    full_baseline_delta = (
        None
        if eval_certified < 0 or full_baseline_certified < 0
        else eval_certified - full_baseline_certified
    )
    visual_baseline_delta = (
        None
        if eval_certified < 0 or visual_baseline_certified < 0
        else eval_certified - visual_baseline_certified
    )
    matched_baseline_delta = (
        None
        if eval_certified < 0 or matched_baseline_certified < 0
        else eval_certified - matched_baseline_certified
    )
    oracle_gap = None if eval_certified < 0 else args.oracle_ceiling - eval_certified
    if eval_certified < 0:
        promotion_status = "diagnostic_only_no_public_replay"
    elif matched_baseline_delta is not None and matched_baseline_delta < 0:
        promotion_status = f"public_replay_regression_vs_{matched_baseline_kind}"
    elif eval_delta is not None and eval_delta > 0:
        promotion_status = "public_replay_exceeds_template_baseline"
    elif matched_baseline_delta is not None and matched_baseline_delta > 0:
        promotion_status = f"public_replay_improves_{matched_baseline_kind}_below_template"
    else:
        promotion_status = "public_replay_measured_no_gain"

    leakage_audit = summarize_leakage_audit(
        args.synthetic_attempts,
        args.eval_state_hashes,
        bool(public_eval),
    )
    fixed_budget_ablation_available = bool(public_eval) and matched_baseline_certified >= 0
    opened_zero_coverage_families = 0
    claim_seed_count = max(0, args.seed_count)
    claim_supported = bool(
        eval_certified >= args.template_baseline
        and opened_zero_coverage_families >= 1
        and claim_seed_count >= 3
        and fixed_budget_ablation_available
        and leakage_audit["public_eval_precondition_satisfied"]
    )
    claim_status = "supported" if claim_supported else "underpowered"
    oracle_coverage_recovered = (
        None
        if eval_certified < 0 or args.oracle_ceiling <= 0
        else eval_certified / args.oracle_ceiling
    )
    template_oracle_coverage_recovered = (
        None
        if args.oracle_ceiling <= 0
        else args.template_baseline / args.oracle_ceiling
    )

    return {
        "schema": SUMMARY_SCHEMA,
        "report_version": 6,
        "claim_id": "C5.arc_verified_self_play",
        "claim_status": claim_status,
        "claim_seed_count": claim_seed_count,
        "claim_minimum_seed_count": 3,
        "claim_public_eval_replay_available": bool(public_eval),
        "claim_fixed_budget_ablation_available": fixed_budget_ablation_available,
        "claim_leakage_gate_passed": leakage_audit["public_eval_precondition_satisfied"],
        "claim_certified_eval_tasks": eval_certified if eval_certified >= 0 else -1,
        "claim_template_baseline": args.template_baseline,
        "claim_oracle_ceiling": args.oracle_ceiling,
        "claim_opened_zero_coverage_families": opened_zero_coverage_families,
        "training_loop": "verified_self_play_inverse_program_induction",
        "supervision_source": "forward primitive composition with exact verifier-compatible labels",
        "authored_task_library": False,
        "authored_prior_scope": (
            "No authored ARC task library is read, but the forward generator's primitive "
            "pipelines and miss-taxonomy bias are authored prior knowledge."
        ),
        "authorized_knowledge": [
            "visual_primitives",
            "exact_verifier",
            "authored_forward_generator_prior",
            "public_miss_taxonomy_bias",
        ],
        "curriculum_summary": str(args.curriculum_summary),
        "ranker_summary": str(args.ranker_summary) if args.ranker_summary else "",
        "self_play_prior": str(args.self_play_prior) if args.self_play_prior else "",
        "prior_summary": str(args.prior_summary) if args.prior_summary else "",
        "generative_prior": str(args.generative_prior) if args.generative_prior else "",
        "generative_prior_summary": (
            str(args.generative_prior_summary) if args.generative_prior_summary else ""
        ),
        "public_train_summary": str(args.public_train_summary) if args.public_train_summary else "",
        "public_train_baseline_summary": (
            str(args.public_train_baseline_summary)
            if args.public_train_baseline_summary
            else ""
        ),
        "public_train_visual_baseline_summary": (
            str(args.public_train_visual_baseline_summary)
            if args.public_train_visual_baseline_summary
            else ""
        ),
        "public_eval_summary": str(args.public_eval_summary) if args.public_eval_summary else "",
        "public_eval_baseline_summary": (
            str(args.public_eval_baseline_summary)
            if args.public_eval_baseline_summary
            else ""
        ),
        "public_eval_visual_baseline_summary": (
            str(args.public_eval_visual_baseline_summary)
            if args.public_eval_visual_baseline_summary
            else ""
        ),
        "synthetic_rows": safe_int(curriculum.get("synthetic_rows")),
        "synthetic_accepted_rows": safe_int(curriculum.get("synthetic_accepted_rows")),
        "synthetic_rejected_rows": safe_int(curriculum.get("synthetic_rejected_rows")),
        "synthetic_negative_kinds": curriculum.get("negative_kinds", []),
        "primitive_ops": ops,
        "max_grid_size": safe_int(curriculum.get("max_grid_size")),
        "op_mode": str(curriculum.get("op_mode", "")),
        "synthesis_budget": {
            "max_accepted": safe_int(curriculum.get("max_accepted")),
            "max_synthetic_rows": safe_int(curriculum.get("max_synthetic_rows")),
            "max_seconds": safe_int(curriculum.get("max_seconds")),
            "elapsed_seconds": safe_float(curriculum.get("elapsed_seconds")),
            "stop_reason": str(curriculum.get("stop_reason", "")),
        },
        "inverse_ranker": {
            "training_mode": ranker.get("training_mode", ""),
            "source_attempt_count": safe_int(ranker.get("source_attempt_count")),
            "source_accepted_count": safe_int(ranker.get("source_accepted_count")),
            "target_task_count": safe_int(ranker.get("target_task_count")),
            "accepted_target_task_count": safe_int(ranker.get("accepted_target_task_count")),
            "top1_rate": safe_float(ranker.get("top1_rate")),
            "top1_all_task_rate": safe_float(ranker.get("top1_all_task_rate")),
            "metric_scope": "in_distribution_generated_curriculum",
            "public_eval_evidence": False,
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
        "oracle_recovery": {
            "metric": "certified_eval_tasks / oracle_ceiling_certified_eval_tasks",
            "template_oracle_coverage_recovered": template_oracle_coverage_recovered,
            "self_play_oracle_coverage_recovered": oracle_coverage_recovered,
            "scope": "L1 search efficiency under fixed primitive ceiling",
        },
        "budget_attribution": {
            "fixed_budget_ablation_available": fixed_budget_ablation_available,
            "matched_baseline_kind": matched_baseline_kind,
            "same_public_eval_log_pair": bool(public_eval)
            and (
                bool(public_eval_visual_baseline)
                or bool(public_eval_baseline)
            ),
            "historical_budget_confound": budget_confound,
            "claim_rule": (
                "Self-play attribution requires same proposer, same inference budget, "
                "with versus without self-play curriculum/prior."
            ),
        },
        "leakage_audit": leakage_audit,
        "claim_decision_rule": {
            "supported_if": [
                "certified eval tasks >= template baseline",
                "at least one currently-zero coverage family opened",
                "minimum 3 seeds",
                "fixed-budget with/without self-play ablation is available",
                "generated state hashes are disjoint from public eval hashes",
            ],
            "status": claim_status,
        },
        "self_play_prior_export": {
            "available": bool(prior),
            "mapped_accepted_attempt_rows": safe_int(
                prior.get("mapped_accepted_attempt_rows")
            ),
            "prior_rows": safe_int(prior.get("prior_rows")),
            "mapped_ops": prior.get("mapped_ops", {}),
            "unmapped_ops": prior.get("unmapped_ops", {}),
            "op_id_contract": prior.get("op_id_contract", ""),
        },
        "generative_prior_export": {
            "available": bool(generative_prior),
            "generative_prior_rows": safe_int(
                generative_prior.get("generative_prior_rows")
            ),
            "mapped_runtime_ops": generative_prior.get("mapped_runtime_ops", {}),
            "source_ops": generative_prior.get("source_ops", {}),
            "rejection_failure_stages": generative_prior.get(
                "rejection_failure_stages", {}
            ),
            "rejection_training": generative_prior.get("rejection_training", {}),
            "unmapped_ops": generative_prior.get("unmapped_ops", {}),
            "claim_scope": generative_prior.get("claim_scope", ""),
        },
        "public_train": summarize_arc_public(public_train),
        "public_train_baseline": summarize_arc_public(public_train_baseline),
        "public_train_visual_baseline": summarize_arc_public(public_train_visual_baseline),
        "public_train_runtime": public_train_runtime,
        "public_train_generative_prior_trace": public_train_trace,
        "public_train_object_translate_miner_trace": public_train_object_translate_trace,
        "public_train_reference_recolor_mask_miner_trace": (
            public_train_reference_recolor_trace
        ),
        "public_eval_baseline": summarize_arc_public(public_eval_baseline),
        "public_eval_visual_baseline": summarize_arc_public(public_eval_visual_baseline),
        "public_eval_runtime": public_eval_runtime,
        "public_eval_generative_prior_trace": public_eval_trace,
        "public_eval_object_translate_miner_trace": public_eval_object_translate_trace,
        "public_eval_reference_recolor_mask_miner_trace": (
            public_eval_reference_recolor_trace
        ),
        "public_eval": {
            "available": bool(public_eval),
            "certified_tasks": None if eval_certified < 0 else eval_certified,
            "loaded_tasks": safe_int(public_eval.get("loaded_tasks")),
            "test_exact_tasks": safe_int(public_eval.get("test_exact_tasks")),
            "certified_cells": eval_certified_cells,
            "cell_accuracy": safe_float(public_eval.get("cell_accuracy")),
            "delta_vs_template_baseline": eval_delta,
            "delta_vs_public_baseline": matched_baseline_delta,
            "delta_vs_full_public_baseline": full_baseline_delta,
            "delta_vs_visual_public_baseline": visual_baseline_delta,
            "gap_to_oracle_ceiling": oracle_gap,
        },
        "public_replay": {
            "available": bool(public_eval),
            "baseline_available": bool(public_eval_baseline),
            "visual_baseline_available": bool(public_eval_visual_baseline),
            "matched_baseline_kind": matched_baseline_kind,
            "eval_full_baseline_certified_tasks": (
                None if full_baseline_certified < 0 else full_baseline_certified
            ),
            "eval_visual_baseline_certified_tasks": (
                None if visual_baseline_certified < 0 else visual_baseline_certified
            ),
            "eval_baseline_certified_tasks": (
                None
                if matched_baseline_certified < 0
                else matched_baseline_certified
            ),
            "eval_self_play_certified_tasks": (
                None if eval_certified < 0 else eval_certified
            ),
            "delta_vs_full_baseline_certified_tasks": full_baseline_delta,
            "delta_vs_visual_baseline_certified_tasks": visual_baseline_delta,
            "delta_vs_baseline_certified_tasks": matched_baseline_delta,
            "delta_vs_template_baseline_certified_tasks": eval_delta,
            "gap_to_oracle_ceiling_certified_tasks": oracle_gap,
            "coverage_regressed_vs_baseline": (
                None if matched_baseline_delta is None else matched_baseline_delta < 0
            ),
            "certified_eval_moved": (
                None if matched_baseline_delta is None else matched_baseline_delta != 0
            ),
            "promotion_status": promotion_status,
            "self_play_train_runtime": public_train_runtime,
            "self_play_eval_runtime": public_eval_runtime,
            "self_play_train_generative_prior_trace": public_train_trace,
            "self_play_eval_generative_prior_trace": public_eval_trace,
            "self_play_train_object_translate_miner_trace": (
                public_train_object_translate_trace
            ),
            "self_play_eval_object_translate_miner_trace": (
                public_eval_object_translate_trace
            ),
            "self_play_train_reference_recolor_mask_miner_trace": (
                public_train_reference_recolor_trace
            ),
            "self_play_eval_reference_recolor_mask_miner_trace": (
                public_eval_reference_recolor_trace
            ),
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
    public_replay = row["public_replay"]
    public_eval_runtime = row["public_eval_runtime"]
    public_eval_trace = row["public_eval_generative_prior_trace"]
    object_translate_trace = row["public_eval_object_translate_miner_trace"]
    reference_recolor_trace = row["public_eval_reference_recolor_mask_miner_trace"]
    prior = row["self_play_prior_export"]
    generative_prior = row["generative_prior_export"]
    claim_rule = row["claim_decision_rule"]
    oracle_recovery = row["oracle_recovery"]
    budget_attr = row["budget_attribution"]
    leakage = row["leakage_audit"]
    budget = row["synthesis_budget"]
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
        "## Synthesis Budget",
        "",
        "- Stop reason: `{}`".format(budget["stop_reason"] or "unknown"),
        "- Max accepted: `{}`".format(budget["max_accepted"]),
        "- Max synthetic rows: `{}`".format(budget["max_synthetic_rows"]),
        "- Max seconds: `{}`".format(budget["max_seconds"]),
        "- Elapsed seconds: `{:.3f}`".format(budget["elapsed_seconds"]),
        "",
        "## Inverse Proposer",
        "",
        "These ranker metrics are in-distribution on the generated curriculum; they are not public ARC coverage evidence.",
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
        "## Self-Play Prior",
        "",
        "- Prior available: `{}`".format("yes" if prior["available"] else "no"),
        "- Mapped accepted attempts: `{}`".format(prior["mapped_accepted_attempt_rows"]),
        "- Prior rows: `{}`".format(prior["prior_rows"]),
        "- Mapped ops: `{}`".format(
            ", ".join(sorted(prior["mapped_ops"].keys())) or "none"
        ),
        "- Unmapped ops: `{}`".format(
            ", ".join(sorted(prior["unmapped_ops"].keys())) or "none"
        ),
        "- Generative prior rows: `{}`".format(
            generative_prior["generative_prior_rows"]
        ),
        "- Generative runtime ops: `{}`".format(
            ", ".join(sorted(generative_prior["mapped_runtime_ops"].keys())) or "none"
        ),
        "- Rejection stages: `{}`".format(
            ", ".join(
                f"{key}={value}"
                for key, value in sorted(
                    generative_prior.get("rejection_failure_stages", {}).items()
                )
            )
            or "none"
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
        "## Claim Gate",
        "",
        "- Registered claim: `{}`".format(row["claim_id"]),
        "- Claim status: `{}`".format(claim_rule["status"]),
        "- Seed count: `{}` / `{}`".format(
            row["claim_seed_count"],
            row["claim_minimum_seed_count"],
        ),
        "- Oracle coverage recovered: `{}`".format(
            "n/a"
            if oracle_recovery["self_play_oracle_coverage_recovered"] is None
            else "{:.3f}".format(oracle_recovery["self_play_oracle_coverage_recovered"])
        ),
        "- Template oracle coverage recovered: `{}`".format(
            "n/a"
            if oracle_recovery["template_oracle_coverage_recovered"] is None
            else "{:.3f}".format(oracle_recovery["template_oracle_coverage_recovered"])
        ),
        "- Fixed-budget ablation available: `{}`".format(
            "yes" if budget_attr["fixed_budget_ablation_available"] else "no"
        ),
        "- Opened zero-coverage families: `{}`".format(
            row["claim_opened_zero_coverage_families"]
        ),
        "- Leakage gate passed: `{}`".format(
            "yes" if leakage["public_eval_precondition_satisfied"] else "no"
        ),
        "- Leakage blocked reason: `{}`".format(leakage["blocked_reason"] or "none"),
        "- Synthetic rows with state hashes: `{}` / `{}`".format(
            leakage["rows_with_state_hash"],
            leakage["synthetic_rows"],
        ),
        "- Synthetic rows with selfplay provenance: `{}` / `{}`".format(
            leakage["rows_with_selfplay_forward_provenance"],
            leakage["synthetic_rows"],
        ),
        "- Historical learned budget 32 -> 64 delta: `{}`".format(
            budget_attr["historical_budget_confound"]["learned_delta_32_to_64"]
        ),
        "",
        "## Public Eval",
        "",
    ]
    if public_eval["available"]:
        lines.extend(
            [
                "- Full baseline certified eval tasks: `{}`".format(
                    public_replay["eval_full_baseline_certified_tasks"]
                ),
                "- Visual-only baseline certified eval tasks: `{}`".format(
                    public_replay["eval_visual_baseline_certified_tasks"]
                ),
                "- Certified eval tasks: `{}`".format(public_eval["certified_tasks"]),
                "- Matched baseline: `{}`".format(
                    public_replay["matched_baseline_kind"]
                ),
                "- Delta vs matched baseline: `{}`".format(
                    public_replay["delta_vs_baseline_certified_tasks"]
                ),
                "- Delta vs full baseline: `{}`".format(
                    public_replay["delta_vs_full_baseline_certified_tasks"]
                ),
                "- Delta vs visual-only baseline: `{}`".format(
                    public_replay["delta_vs_visual_baseline_certified_tasks"]
                ),
                "- Delta vs template baseline: `{}`".format(
                    public_eval["delta_vs_template_baseline"]
                ),
                "- Gap to oracle ceiling: `{}`".format(public_eval["gap_to_oracle_ceiling"]),
                "- Promotion status: `{}`".format(public_replay["promotion_status"]),
                "- Eval generative-prior op runs: `{}`".format(
                    public_eval_runtime["runtime_generative_prior_ops"]
                ),
                "- Eval generative-prior mined sketches: `{}`".format(
                    public_eval_runtime["runtime_generative_prior_mined_sketches"]
                ),
                "- Eval rejection-trace rows: `{}`".format(
                    public_eval_trace["trace_rows"]
                ),
                "- Eval rejection-score filtered ops: `{}`".format(
                    public_eval_trace["filtered_by_rejection_score"]
                ),
                "- Eval selected prior ops: `{}`".format(
                    public_eval_trace["selected_ops"]
                ),
                "- Eval object-translate miner attempts: `{}`".format(
                    object_translate_trace["attempted_specs"]
                ),
                "- Eval object-translate accepted specs: `{}`".format(
                    object_translate_trace["accepted_specs"]
                ),
                "- Eval object-translate best rejected score: `{:.3f}`".format(
                    object_translate_trace["best_rejected_verifier_score"]
                ),
                "- Eval object-translate best rejected task: `{}`".format(
                    object_translate_trace["best_rejected_task"] or "none"
                ),
                "- Eval reference-recolor compiler attempts: `{}`".format(
                    reference_recolor_trace["attempted_specs"]
                ),
                "- Eval reference-recolor accepted specs: `{}`".format(
                    reference_recolor_trace["accepted_specs"]
                ),
                "- Eval reference-recolor accepted tasks: `{}`".format(
                    ", ".join(reference_recolor_trace["accepted_tasks"]) or "none"
                ),
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
    parser.add_argument("--self-play-prior", type=Path)
    parser.add_argument("--prior-summary", type=Path)
    parser.add_argument("--generative-prior", type=Path)
    parser.add_argument("--generative-prior-summary", type=Path)
    parser.add_argument("--public-train-summary", type=Path)
    parser.add_argument("--public-train-baseline-summary", type=Path)
    parser.add_argument("--public-train-visual-baseline-summary", type=Path)
    parser.add_argument("--public-eval-summary", type=Path)
    parser.add_argument("--public-eval-baseline-summary", type=Path)
    parser.add_argument("--public-eval-visual-baseline-summary", type=Path)
    parser.add_argument("--miss-taxonomy", type=Path)
    parser.add_argument("--synthetic-attempts", type=Path)
    parser.add_argument("--budget-curves", type=Path)
    parser.add_argument("--eval-state-hashes", type=Path)
    parser.add_argument("--seed-count", type=int, default=1)
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
