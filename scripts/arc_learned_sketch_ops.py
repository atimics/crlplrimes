#!/usr/bin/env python3
"""Mine verifier-backed ARC sketch attempts into learned operator candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from crlp.io import read_jsonl_lenient, safe_float, safe_int, write_jsonl


OPERATOR_SCHEMA = "crlp.arc_learned_sketch_op.v1"
SUMMARY_SCHEMA = "crlp.arc_learned_sketch_ops.summary.v1"
PRIOR_SCHEMA = "crlp.arc_learned_sketch_op_prior.v1"

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

REFERENCE_RECOLOR_MASK_SIGNATURE = {
    "family": "reference_recolor_mask",
    "operator_name": "arc.reference_recolor_mask",
    "compiler_target": "visual_combinator",
    "composition_kind": "reference_conditioned_recolor",
    "normalization_frame": "reference_target_bbox",
    "typed_stages": "select_reference|derive_relative_mask|select_target|recolor_overlay",
    "primitive_contract": "compiler_replay",
    "visual_program": "reference_recolor_mask",
    "slot_signature": "reference_selector|target_selector|source_color|output_color|relative_mask",
    "compiler_contract": "learn relative mask/color binding from reference object, recolor target object, then replay ARC verifier",
}

PRIOR_FIELDS = [
    "schema_version",
    "operator_id",
    "operator_name",
    "family",
    "compiler_target",
    "promotion_status",
    "proposal_count",
    "accepted_count",
    "rejected_count",
    "near_miss_count",
    "hard_rejected_count",
    "support_tasks",
    "mean_verifier_score",
    "best_verifier_score",
    "composition_kind",
    "normalization_frame",
    "typed_stages",
    "primitive_contract",
    "slot_signature",
    "learned_from",
]


@dataclass
class SketchBucket:
    family: str
    operator_name: str
    compiler_target: str
    composition_kind: str
    normalization_frame: str
    typed_stages: str
    primitive_contract: str
    visual_program: str
    slot_signature: str
    learned_from: str
    verifier_gate: str
    compiler_contract: str
    accepted_count: int = 0
    rejected_count: int = 0
    near_miss_count: int = 0
    hard_rejected_count: int = 0
    verifier_score_sum: float = 0.0
    best_verifier_score: float = 0.0
    source_tasks: set[str] = field(default_factory=set)
    accepted_tasks: set[str] = field(default_factory=set)
    rejection_reasons: Counter[str] = field(default_factory=Counter)
    failure_stages: Counter[str] = field(default_factory=Counter)
    examples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        return self.accepted_count + self.rejected_count

    @property
    def support_tasks(self) -> int:
        return len(self.accepted_tasks) if self.accepted_tasks else len(self.source_tasks)

    @property
    def mean_verifier_score(self) -> float:
        total = self.total_count
        return self.verifier_score_sum / total if total else 0.0


def load_attempts(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                row["_source_path"] = str(path)
                rows.append(row)
    return rows


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def trace_operator_id(signature: dict[str, str]) -> str:
    id_signature = {
        key: value for key, value in signature.items() if key != "primitive_contract"
    }
    return stable_hash({"kind": "object_translate_trace", **id_signature})[:12]


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return cleaned or "unknown"


def slot_signature(value: str) -> str:
    names: list[str] = []
    for part in value.split("|"):
        if not part.strip():
            continue
        name = part.split("=", 1)[0].strip()
        if name and name not in names:
            names.append(name)
    return "|".join(names)


def is_near_miss(row: dict[str, str], verifier_score: float, threshold: float) -> bool:
    if safe_int(row.get("accepted")) == 1:
        return False
    failure_stage = str(row.get("failure_stage", "")).strip()
    return verifier_score >= threshold or failure_stage in NEAR_MISS_FAILURE_STAGES


def is_hard_rejection(row: dict[str, str]) -> bool:
    failure_stage = str(row.get("failure_stage", "")).strip()
    reason = str(row.get("rejection_reason", "")).strip()
    return failure_stage in HARD_REJECTION_FAILURE_STAGES or reason == "unsupported_shape"


def compiler_target_for(row: dict[str, str]) -> str:
    op_name = str(row.get("op_name", "")).strip() or "unknown"
    if op_name == "visual_combinator":
        return "visual_combinator"
    return "arc_sketch_program"


def compiler_contract_for(row: dict[str, str]) -> str:
    op_name = str(row.get("op_name", "")).strip() or "unknown"
    if op_name == "visual_combinator":
        return "compile learned visual-program stages through the generic ARC sketch VM, then replay train-pair verifier"
    return f"compile typed slots into ARC_SKETCH_OP_{op_name}, then replay train-pair verifier"


def attempt_signature(row: dict[str, str]) -> dict[str, str]:
    return {
        "op_name": str(row.get("op_name", "")).strip() or "unknown",
        "visual_program": str(row.get("visual_program", "")).strip() or "unknown",
        "slot_signature": slot_signature(str(row.get("slot_contract", ""))),
        "composition_kind": str(row.get("composition_kind", "")).strip() or "unknown",
        "normalization_frame": str(row.get("normalization_frame", "")).strip()
        or "unknown",
        "typed_stages": str(row.get("typed_stages", "")).strip() or "unknown",
        "primitive_contract": str(row.get("primitive_contract", "")).strip()
        or "unknown",
    }


def bucket_for_attempt(
    buckets: dict[str, SketchBucket], row: dict[str, str]
) -> SketchBucket:
    signature = attempt_signature(row)
    operator_id = stable_hash({"kind": "attempt", **signature})[:12]
    op_name = signature["op_name"]
    operator_name = "arc.{op}.{composition}.{frame}".format(
        op=slug(op_name),
        composition=slug(signature["composition_kind"]),
        frame=slug(signature["normalization_frame"]),
    )
    bucket = buckets.get(operator_id)
    if bucket is None:
        bucket = SketchBucket(
            family=op_name,
            operator_name=operator_name,
            compiler_target=compiler_target_for(row),
            composition_kind=signature["composition_kind"],
            normalization_frame=signature["normalization_frame"],
            typed_stages=signature["typed_stages"],
            primitive_contract=signature["primitive_contract"],
            visual_program=signature["visual_program"],
            slot_signature=signature["slot_signature"],
            learned_from="attempt_tsv",
            verifier_gate=str(row.get("primitive_contract", "")).strip()
            or "train_pair_exact",
            compiler_contract=compiler_contract_for(row),
        )
        buckets[operator_id] = bucket
    return bucket


def add_attempt_evidence(
    bucket: SketchBucket,
    row: dict[str, str],
    *,
    near_miss_threshold: float,
    top_examples: int,
) -> None:
    accepted = safe_int(row.get("accepted")) == 1
    verifier_score = safe_float(row.get("verifier_score"))
    if accepted and verifier_score <= 0.0:
        verifier_score = 1.0
    task_id = str(row.get("task_id", "")).strip()
    if task_id:
        bucket.source_tasks.add(task_id)
    if accepted:
        bucket.accepted_count += 1
        if task_id:
            bucket.accepted_tasks.add(task_id)
    else:
        bucket.rejected_count += 1
        if is_near_miss(row, verifier_score, near_miss_threshold):
            bucket.near_miss_count += 1
        if is_hard_rejection(row):
            bucket.hard_rejected_count += 1
    reason = str(row.get("rejection_reason", "")).strip()
    if reason:
        bucket.rejection_reasons[reason] += 1
    stage = str(row.get("failure_stage", "")).strip()
    if stage:
        bucket.failure_stages[stage] += 1
    bucket.verifier_score_sum += verifier_score
    bucket.best_verifier_score = max(bucket.best_verifier_score, verifier_score)
    if len(bucket.examples) < top_examples:
        bucket.examples.append(
            {
                "task_id": task_id,
                "attempt_id": str(row.get("attempt_id", "")),
                "accepted": accepted,
                "rejection_reason": reason,
                "failure_stage": stage,
                "verifier_score": round(verifier_score, 8),
                "source_path": str(row.get("_source_path", "")),
            }
        )


def mine_attempt_buckets(
    attempts: list[dict[str, str]],
    *,
    near_miss_threshold: float,
    top_examples: int,
) -> dict[str, SketchBucket]:
    buckets: dict[str, SketchBucket] = {}
    for row in attempts:
        bucket = bucket_for_attempt(buckets, row)
        add_attempt_evidence(
            bucket,
            row,
            near_miss_threshold=near_miss_threshold,
            top_examples=top_examples,
        )
    return buckets


def diagnostic_signature(row: dict[str, Any], score_threshold: float) -> dict[str, str] | None:
    score = safe_float(row.get("best_rejected_verifier_score"))
    if score < score_threshold:
        return None
    reason = str(row.get("best_rejected_reason", "")).strip()
    dx = safe_int(row.get("best_rejected_dx"))
    dy = safe_int(row.get("best_rejected_dy"))
    if reason == "no_train_match" and dx == 0 and dy == 0:
        signature = dict(REFERENCE_RECOLOR_MASK_SIGNATURE)
        signature["primitive_contract"] = "diagnostic_near_miss"
        return signature
    if reason == "no_train_match":
        return {
            "family": "object_relation_motion_refinement",
            "operator_name": "arc.object_relation_motion_refinement",
            "compiler_target": "visual_combinator",
            "composition_kind": "reference_conditioned_motion",
            "normalization_frame": "object_relation_offset",
            "typed_stages": "select_reference|bind_offset|select_target|move_or_copy",
            "primitive_contract": "diagnostic_near_miss",
            "visual_program": "object_relation_motion_refinement",
            "slot_signature": "reference_selector|target_selector|dx|dy|render_mode",
            "compiler_contract": "learn relation-conditioned offset binding before replaying object-translate verifier",
        }
    if reason == "unsupported_shape":
        return {
            "family": "shape_frame_repair",
            "operator_name": "arc.shape_frame_repair",
            "compiler_target": "arc_sketch_program",
            "composition_kind": "shape_frame_repair",
            "normalization_frame": "train_pair_bbox",
            "typed_stages": "select_object|repair_canvas_frame|render",
            "primitive_contract": "diagnostic_near_miss",
            "visual_program": "shape_frame_repair",
            "slot_signature": "selector|width|height|x_anchor|y_anchor",
            "compiler_contract": "repair unsupported shape/frame before replaying the original sketch verifier",
        }
    return None


def add_diagnostic_evidence(
    buckets: dict[str, SketchBucket],
    row: dict[str, Any],
    signature: dict[str, str],
    *,
    source_path: Path,
    top_examples: int,
) -> None:
    operator_id = trace_operator_id(signature)
    bucket = buckets.get(operator_id)
    if bucket is None:
        bucket = SketchBucket(
            family=signature["family"],
            operator_name=signature["operator_name"],
            compiler_target=signature["compiler_target"],
            composition_kind=signature["composition_kind"],
            normalization_frame=signature["normalization_frame"],
            typed_stages=signature["typed_stages"],
            primitive_contract=signature["primitive_contract"],
            visual_program=signature["visual_program"],
            slot_signature=signature["slot_signature"],
            learned_from="object_translate_trace",
            verifier_gate="diagnostic_near_miss",
            compiler_contract=signature["compiler_contract"],
        )
        buckets[operator_id] = bucket
    score = safe_float(row.get("best_rejected_verifier_score"))
    task_id = str(row.get("task_instance", "")).strip()
    if task_id:
        bucket.source_tasks.add(task_id)
    bucket.rejected_count += 1
    bucket.near_miss_count += 1
    reason = str(row.get("best_rejected_reason", "")).strip()
    if reason:
        bucket.rejection_reasons[reason] += 1
    bucket.failure_stages["diagnostic_near_miss"] += 1
    bucket.verifier_score_sum += score
    bucket.best_verifier_score = max(bucket.best_verifier_score, score)
    if len(bucket.examples) < top_examples:
        bucket.examples.append(
            {
                "task_id": task_id,
                "accepted": False,
                "rejection_reason": reason,
                "failure_stage": "diagnostic_near_miss",
                "verifier_score": round(score, 8),
                "dx": safe_int(row.get("best_rejected_dx")),
                "dy": safe_int(row.get("best_rejected_dy")),
                "selector": safe_int(row.get("best_rejected_selector")),
                "color": safe_int(row.get("best_rejected_color")),
                "output_color": safe_int(row.get("best_rejected_output_color")),
                "render_mode": safe_int(row.get("best_rejected_render_mode")),
                "source_path": str(source_path),
            }
        )


def note_learned_from(bucket: SketchBucket, source: str) -> None:
    parts = [part for part in bucket.learned_from.split("|") if part]
    if source not in parts:
        parts.append(source)
    bucket.learned_from = "|".join(parts)


def add_reference_recolor_mask_evidence(
    buckets: dict[str, SketchBucket],
    row: dict[str, Any],
    *,
    source_path: Path,
    top_examples: int,
) -> None:
    signature = dict(REFERENCE_RECOLOR_MASK_SIGNATURE)
    operator_id = trace_operator_id(signature)
    bucket = buckets.get(operator_id)
    if bucket is None:
        bucket = SketchBucket(
            family=signature["family"],
            operator_name=signature["operator_name"],
            compiler_target=signature["compiler_target"],
            composition_kind=signature["composition_kind"],
            normalization_frame=signature["normalization_frame"],
            typed_stages=signature["typed_stages"],
            primitive_contract=signature["primitive_contract"],
            visual_program=signature["visual_program"],
            slot_signature=signature["slot_signature"],
            learned_from="compiler_replay_trace",
            verifier_gate="train_pair_exact",
            compiler_contract=signature["compiler_contract"],
        )
        buckets[operator_id] = bucket
    else:
        bucket.primitive_contract = signature["primitive_contract"]
        bucket.verifier_gate = "train_pair_exact"
        note_learned_from(bucket, "compiler_replay_trace")

    attempted = safe_int(row.get("attempted_specs"))
    accepted = safe_int(row.get("accepted_specs"))
    rejected = max(0, attempted - accepted)
    task_id = str(row.get("task_instance", "")).strip()
    if task_id:
        bucket.source_tasks.add(task_id)
    if accepted > 0:
        bucket.accepted_count += accepted
        if task_id:
            bucket.accepted_tasks.add(task_id)
        bucket.verifier_score_sum += float(accepted)
        bucket.best_verifier_score = max(bucket.best_verifier_score, 1.0)
        bucket.failure_stages["accepted"] += accepted
    if rejected > 0:
        bucket.rejected_count += rejected
        bucket.rejection_reasons["compiler_replay_rejected"] += rejected
        bucket.failure_stages["compiler_replay_rejected"] += rejected
    if len(bucket.examples) < top_examples:
        bucket.examples.append(
            {
                "task_id": task_id,
                "accepted": accepted > 0,
                "rejection_reason": "" if accepted > 0 else "compiler_replay_rejected",
                "failure_stage": "accepted" if accepted > 0 else "compiler_replay_rejected",
                "verifier_score": 1.0 if accepted > 0 else 0.0,
                "attempted_specs": attempted,
                "accepted_specs": accepted,
                "source_path": str(source_path),
            }
        )


def mine_diagnostic_buckets(
    paths: Iterable[Path],
    *,
    diagnostic_score_threshold: float,
    top_examples: int,
) -> dict[str, SketchBucket]:
    buckets: dict[str, SketchBucket] = {}
    for path in paths:
        if not path.is_file():
            continue
        rows = read_jsonl_lenient(path, schema="crlp.arc_object_translate_miner.trace.v1")
        for row in rows:
            signature = diagnostic_signature(row, diagnostic_score_threshold)
            if signature is None:
                continue
            add_diagnostic_evidence(
                buckets,
                row,
                signature,
                source_path=path,
                top_examples=top_examples,
            )
        rows = read_jsonl_lenient(
            path, schema="crlp.arc_reference_recolor_mask_miner.trace.v1"
        )
        for row in rows:
            add_reference_recolor_mask_evidence(
                buckets,
                row,
                source_path=path,
                top_examples=top_examples,
            )
    return buckets


def promotion_status(
    bucket: SketchBucket, *, min_accepted: int, min_tasks: int
) -> tuple[str, str]:
    if bucket.accepted_count >= min_accepted and len(bucket.accepted_tasks) >= min_tasks:
        return "promoted", "meets verifier-accepted support threshold"
    if bucket.accepted_count > 0:
        return "candidate", "accepted evidence is below promotion threshold"
    if bucket.near_miss_count > 0:
        return "candidate", "diagnostic near-miss only; requires compiler and verifier replay"
    return "blocked", "no accepted or near-miss evidence"


def bucket_record(
    operator_id: str,
    bucket: SketchBucket,
    *,
    min_accepted: int,
    min_tasks: int,
) -> dict[str, Any]:
    status, reason = promotion_status(
        bucket, min_accepted=min_accepted, min_tasks=min_tasks
    )
    return {
        "schema": OPERATOR_SCHEMA,
        "report_version": 1,
        "date": date.today().isoformat(),
        "operator_id": f"arc_sketch_op:{operator_id}",
        "operator_name": bucket.operator_name,
        "family": bucket.family,
        "compiler_target": bucket.compiler_target,
        "promotion_status": status,
        "promotion_policy": "accepted verifier rows promote; diagnostic near-misses remain candidates",
        "reason_code": reason,
        "verifier_gate": bucket.verifier_gate,
        "accepted_count": bucket.accepted_count,
        "rejected_count": bucket.rejected_count,
        "near_miss_count": bucket.near_miss_count,
        "hard_rejected_count": bucket.hard_rejected_count,
        "support_tasks": bucket.support_tasks,
        "source_task_count": len(bucket.source_tasks),
        "accepted_task_count": len(bucket.accepted_tasks),
        "mean_verifier_score": round(bucket.mean_verifier_score, 8),
        "best_verifier_score": round(bucket.best_verifier_score, 8),
        "composition_kind": bucket.composition_kind,
        "normalization_frame": bucket.normalization_frame,
        "typed_stages": bucket.typed_stages,
        "primitive_contract": bucket.primitive_contract,
        "visual_program": bucket.visual_program,
        "slot_signature": bucket.slot_signature,
        "learned_from": bucket.learned_from,
        "compiler_contract": bucket.compiler_contract,
        "top_rejection_reasons": dict(bucket.rejection_reasons.most_common(8)),
        "top_failure_stages": dict(bucket.failure_stages.most_common(8)),
        "source_examples": bucket.examples,
    }


def operator_records(
    buckets: dict[str, SketchBucket], *, min_accepted: int, min_tasks: int
) -> list[dict[str, Any]]:
    records = [
        bucket_record(
            operator_id,
            bucket,
            min_accepted=min_accepted,
            min_tasks=min_tasks,
        )
        for operator_id, bucket in buckets.items()
    ]
    status_rank = {"promoted": 0, "candidate": 1, "blocked": 2}
    return sorted(
        records,
        key=lambda row: (
            status_rank.get(str(row["promotion_status"]), 9),
            -safe_int(row.get("support_tasks")),
            -safe_int(row.get("accepted_count")),
            -safe_float(row.get("best_verifier_score")),
            str(row.get("operator_name")),
        ),
    )


def prior_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "schema_version": PRIOR_SCHEMA,
                "operator_id": record["operator_id"],
                "operator_name": record["operator_name"],
                "family": record["family"],
                "compiler_target": record["compiler_target"],
                "promotion_status": record["promotion_status"],
                "proposal_count": safe_int(record.get("accepted_count"))
                + safe_int(record.get("near_miss_count")),
                "accepted_count": record["accepted_count"],
                "rejected_count": record["rejected_count"],
                "near_miss_count": record["near_miss_count"],
                "hard_rejected_count": record["hard_rejected_count"],
                "support_tasks": record["support_tasks"],
                "mean_verifier_score": f"{safe_float(record.get('mean_verifier_score')):.8f}",
                "best_verifier_score": f"{safe_float(record.get('best_verifier_score')):.8f}",
                "composition_kind": record["composition_kind"],
                "normalization_frame": record["normalization_frame"],
                "typed_stages": record["typed_stages"],
                "primitive_contract": record["primitive_contract"],
                "slot_signature": record["slot_signature"],
                "learned_from": record["learned_from"],
            }
        )
    return rows


def write_prior_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=PRIOR_FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summary_row(
    records: list[dict[str, Any]],
    *,
    attempt_paths: list[Path],
    trace_paths: list[Path],
    attempt_rows: int,
) -> dict[str, Any]:
    statuses = Counter(str(row.get("promotion_status", "")) for row in records)
    learned_from = Counter(str(row.get("learned_from", "")) for row in records)
    families = Counter(str(row.get("family", "")) for row in records)
    top_candidates = [
        {
            "operator_name": row["operator_name"],
            "promotion_status": row["promotion_status"],
            "family": row["family"],
            "support_tasks": row["support_tasks"],
            "accepted_count": row["accepted_count"],
            "near_miss_count": row["near_miss_count"],
            "best_verifier_score": row["best_verifier_score"],
            "compiler_target": row["compiler_target"],
        }
        for row in records[:12]
    ]
    return {
        "schema": SUMMARY_SCHEMA,
        "report_version": 1,
        "date": date.today().isoformat(),
        "attempt_paths": [str(path) for path in attempt_paths],
        "compiler_trace_paths": [str(path) for path in trace_paths],
        "attempt_rows": attempt_rows,
        "operator_candidates": len(records),
        "promoted_operators": statuses.get("promoted", 0),
        "candidate_operators": statuses.get("candidate", 0),
        "blocked_operators": statuses.get("blocked", 0),
        "operator_status_counts": dict(sorted(statuses.items())),
        "learned_from_counts": dict(sorted(learned_from.items())),
        "family_counts": dict(families.most_common(12)),
        "top_candidates": top_candidates,
        "claim_scope": "learned sketch-op priors only; all candidates still require domain compiler replay before they can act as verifier gates",
    }


def write_markdown(path: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# ARC Learned Sketch Operators",
        "",
        f"Date: {summary['date']}",
        "",
        "This pass mines typed ARC visual-pipeline attempt rows, diagnostic traces, "
        "and compiler-replay traces into learned sketch-op families. Accepted verifier "
        "rows can promote an operator prior; diagnostic near-misses remain candidates "
        "until a compiler emits executable sketches and the ARC verifier accepts them.",
        "",
        f"Attempt rows: `{summary['attempt_rows']}`",
        f"Operator candidates: `{summary['operator_candidates']}`",
        f"Promoted operators: `{summary['promoted_operators']}`",
        f"Diagnostic/candidate operators: `{summary['candidate_operators']}`",
        "",
        "| Operator | Status | Family | Support tasks | Accepted | Near misses | Best score | Compiler |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in records[:24]:
        lines.append(
            "| `{operator}` | `{status}` | `{family}` | {tasks} | {accepted} | {near} | {score:.3f} | `{compiler}` |".format(
                operator=row["operator_name"],
                status=row["promotion_status"],
                family=row["family"],
                tasks=row["support_tasks"],
                accepted=row["accepted_count"],
                near=row["near_miss_count"],
                score=safe_float(row.get("best_verifier_score")),
                compiler=row["compiler_target"],
            )
        )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "- Output schema: `crlp.arc_learned_sketch_op.v1`.",
            "- Prior schema: `crlp.arc_learned_sketch_op_prior.v1`.",
            "- Promotion requires accepted verifier support; near-miss diagnostics are proposal pressure only.",
            "- Runtime use still requires a domain compiler and ARC verifier replay.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=Path, action="append", required=True)
    parser.add_argument("--object-translate-trace", type=Path, action="append", default=[])
    parser.add_argument("--operators", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--min-accepted", type=int, default=1)
    parser.add_argument("--min-tasks", type=int, default=1)
    parser.add_argument("--near-miss-score", type=float, default=0.75)
    parser.add_argument("--diagnostic-score", type=float, default=0.90)
    parser.add_argument("--top-examples", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    attempts = load_attempts(args.attempts)
    buckets = mine_attempt_buckets(
        attempts,
        near_miss_threshold=args.near_miss_score,
        top_examples=args.top_examples,
    )
    buckets.update(
        mine_diagnostic_buckets(
            args.object_translate_trace,
            diagnostic_score_threshold=args.diagnostic_score,
            top_examples=args.top_examples,
        )
    )
    records = operator_records(
        buckets, min_accepted=args.min_accepted, min_tasks=args.min_tasks
    )
    priors = prior_rows(records)
    summary = summary_row(
        records,
        attempt_paths=args.attempts,
        trace_paths=args.object_translate_trace,
        attempt_rows=len(attempts),
    )
    write_jsonl(args.operators, records, compact=True)
    write_prior_tsv(args.prior, priors)
    write_jsonl(args.summary, [summary], compact=True)
    if args.markdown:
        write_markdown(args.markdown, records, summary)
    print(f"wrote {args.operators}")
    print(f"wrote {args.prior}")
    print(f"wrote {args.summary}")
    if args.markdown:
        print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
