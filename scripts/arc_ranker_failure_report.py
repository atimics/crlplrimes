#!/usr/bin/env python3
"""Summarize ARC visual ranker misses at task granularity."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crlp.io import read_jsonl_lenient, safe_float, safe_int, write_jsonl


TASK_SCHEMA = "crlp.arc_ranker_failure_task.v1"

TaskKey = tuple[str, str, str]


@dataclass
class AttemptStats:
    candidate_count: int = 0
    accepted_candidate_count: int = 0
    rejection_reasons: Counter[str] = field(default_factory=Counter)
    failure_stages: Counter[str] = field(default_factory=Counter)
    failure_details: Counter[str] = field(default_factory=Counter)
    accepted_ops: Counter[str] = field(default_factory=Counter)
    candidate_ops: Counter[str] = field(default_factory=Counter)
    candidate_pipelines: Counter[str] = field(default_factory=Counter)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def counter_top(counter: Counter[str]) -> tuple[str, int]:
    if not counter:
        return "", 0
    value, count = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0]
    return value, int(count)


def maybe_round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 8)


def field_indexes(header: list[str], required: list[str]) -> dict[str, int]:
    indexes = {name: index for index, name in enumerate(header)}
    missing = [name for name in required if name not in indexes]
    if missing:
        raise ValueError(f"missing TSV fields: {', '.join(missing)}")
    return indexes


def cell(parts: list[str], indexes: dict[str, int], name: str) -> str:
    index = indexes.get(name, -1)
    if index < 0 or index >= len(parts):
        return ""
    return parts[index]


def load_attempt_stats(path: Path) -> dict[TaskKey, AttemptStats]:
    stats: dict[TaskKey, AttemptStats] = defaultdict(AttemptStats)
    required = [
        "split_name",
        "family",
        "task_id",
        "split",
        "accepted",
        "rejection_reason",
        "failure_stage",
        "failure_detail",
        "pipeline_kind",
        "op_name",
    ]
    with path.open(encoding="utf-8") as handle:
        header_line = handle.readline()
        if not header_line:
            return {}
        indexes = field_indexes(header_line.rstrip("\n").split("\t"), required)
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if cell(parts, indexes, "split_name") != "target":
                continue
            key = (
                cell(parts, indexes, "family"),
                cell(parts, indexes, "task_id"),
                cell(parts, indexes, "split"),
            )
            row = stats[key]
            row.candidate_count += 1
            accepted = parse_bool(cell(parts, indexes, "accepted"))
            reason = cell(parts, indexes, "rejection_reason") or "unknown"
            stage = cell(parts, indexes, "failure_stage") or "unknown"
            detail = cell(parts, indexes, "failure_detail") or "unknown"
            op_name = cell(parts, indexes, "op_name") or "unknown"
            pipeline_kind = cell(parts, indexes, "pipeline_kind") or "unknown"
            row.rejection_reasons[reason] += 1
            row.failure_stages[stage] += 1
            row.failure_details[detail] += 1
            row.candidate_ops[op_name] += 1
            row.candidate_pipelines[pipeline_kind] += 1
            if accepted:
                row.accepted_candidate_count += 1
                row.accepted_ops[op_name] += 1
    return dict(stats)


def load_proposals(path: Path) -> dict[TaskKey, list[dict[str, Any]]]:
    grouped: dict[TaskKey, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl_lenient(path):
        if row.get("schema") != "crlp.neural_grammar_discovery.arc_visual_pipeline_proposal.v1":
            continue
        key = (
            str(row.get("family", "")),
            str(row.get("task_id", "")),
            str(row.get("split", "")),
        )
        grouped[key].append(row)
    for rows in grouped.values():
        rows.sort(
            key=lambda row: (
                safe_int(row.get("rank"), 1_000_000),
                -safe_float(row.get("score")),
                str(row.get("proposal_id", "")),
            )
        )
    return dict(grouped)


def load_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    rows = read_jsonl_lenient(path)
    return rows[0] if rows else {}


def retained_reason_counters(proposals: list[dict[str, Any]]) -> tuple[Counter[str], Counter[str], Counter[str]]:
    reasons: Counter[str] = Counter()
    stages: Counter[str] = Counter()
    details: Counter[str] = Counter()
    for proposal in proposals:
        reasons[str(proposal.get("target_rejection_reason", "unknown") or "unknown")] += 1
        stages[str(proposal.get("failure_stage", "unknown") or "unknown")] += 1
        details[str(proposal.get("failure_detail", "unknown") or "unknown")] += 1
    return reasons, stages, details


def classify_task(
    *,
    stats: AttemptStats,
    proposals: list[dict[str, Any]],
    first_accepted_rank: int | None,
) -> tuple[str, str, str]:
    if not proposals:
        return (
            "no_retained_proposals",
            "retention",
            "ranker emitted no retained proposals for this target task",
        )
    if stats.accepted_candidate_count == 0:
        return (
            "candidate_generation_gap",
            "candidate_generation",
            "no accepted candidate exists in the target attempt search space",
        )
    if first_accepted_rank is None:
        return (
            "retention_gap",
            "retention",
            "accepted candidates exist, but none survived the retained proposal cap",
        )
    if first_accepted_rank == 1:
        return (
            "solved_top1",
            "solved",
            "accepted candidate is ranked first among retained proposals",
        )
    return (
        "ranking_gap",
        "visual_feature_ranking",
        "accepted candidate was retained but ranked below a rejected proposal",
    )


def task_row(key: TaskKey, stats: AttemptStats, proposals: list[dict[str, Any]]) -> dict[str, Any]:
    family, task_id, split = key
    accepted_proposals = [
        row for row in proposals if parse_bool(row.get("target_accepted"))
    ]
    first_accepted = min(
        (safe_int(row.get("rank")) for row in accepted_proposals),
        default=None,
    )
    top = proposals[0] if proposals else {}
    second = proposals[1] if len(proposals) > 1 else {}
    top_score = safe_float(top.get("score")) if top else None
    second_score = safe_float(second.get("score")) if second else None
    best_accepted_score = (
        max(safe_float(row.get("score")) for row in accepted_proposals)
        if accepted_proposals
        else None
    )
    retained_reasons, retained_stages, retained_details = retained_reason_counters(proposals)
    candidate_reason, candidate_reason_count = counter_top(stats.rejection_reasons)
    candidate_stage, candidate_stage_count = counter_top(stats.failure_stages)
    candidate_detail, candidate_detail_count = counter_top(stats.failure_details)
    retained_reason, retained_reason_count = counter_top(retained_reasons)
    retained_stage, retained_stage_count = counter_top(retained_stages)
    retained_detail, retained_detail_count = counter_top(retained_details)
    candidate_op, candidate_op_count = counter_top(stats.candidate_ops)
    candidate_pipeline, candidate_pipeline_count = counter_top(stats.candidate_pipelines)
    bucket, diagnosis, next_step = classify_task(
        stats=stats,
        proposals=proposals,
        first_accepted_rank=first_accepted,
    )
    rank_features = top.get("rank_features", {}) if isinstance(top, dict) else {}
    return {
        "schema": TASK_SCHEMA,
        "report_version": 1,
        "family": family,
        "task_id": task_id,
        "split": split,
        "failure_bucket": bucket,
        "ranker_diagnosis": diagnosis,
        "next_step_hint": next_step,
        "candidate_count": stats.candidate_count,
        "accepted_candidate_count": stats.accepted_candidate_count,
        "rejected_candidate_count": stats.candidate_count - stats.accepted_candidate_count,
        "retained_count": len(proposals),
        "retained_accepted_count": len(accepted_proposals),
        "first_accepted_rank": first_accepted,
        "top_score": maybe_round(top_score),
        "second_score": maybe_round(second_score),
        "top_score_margin": maybe_round(
            None
            if top_score is None or second_score is None
            else top_score - second_score
        ),
        "best_accepted_score": maybe_round(best_accepted_score),
        "top_vs_best_accepted_score_delta": maybe_round(
            None
            if top_score is None or best_accepted_score is None
            else top_score - best_accepted_score
        ),
        "top_proposal_id": top.get("proposal_id", ""),
        "top_attempt_id": top.get("target_attempt_id", ""),
        "top_rank": safe_int(top.get("rank")) if top else None,
        "top_accepted": parse_bool(top.get("target_accepted")),
        "top_rejection_reason": str(top.get("target_rejection_reason", "")),
        "top_failure_stage": str(top.get("failure_stage", "")),
        "top_failure_detail": str(top.get("failure_detail", "")),
        "top_pipeline_kind": str(top.get("pipeline_kind", "")),
        "top_op_name": str(top.get("op_name", "")),
        "top_model_score": maybe_round(safe_float(rank_features.get("model_score")))
        if rank_features
        else None,
        "top_verifier_proxy_score": maybe_round(
            safe_float(rank_features.get("verifier_proxy_score"))
        )
        if rank_features
        else None,
        "top_renderer_model_score": maybe_round(
            safe_float(rank_features.get("renderer_model_score"))
        )
        if rank_features
        else None,
        "top_component_shape_color_score": maybe_round(
            safe_float(rank_features.get("component_shape_color_score"))
        )
        if rank_features
        else None,
        "top_component_selector_layout_score": maybe_round(
            safe_float(rank_features.get("component_selector_layout_score"))
        )
        if rank_features
        else None,
        "top_sequence_block_renderer_score": maybe_round(
            safe_float(rank_features.get("sequence_block_renderer_score"))
        )
        if rank_features
        else None,
        "top_verifier_mismatch_ratio": maybe_round(
            safe_float(rank_features.get("verifier_mismatch_ratio"))
        )
        if rank_features
        else None,
        "top_verifier_unsupported_pairs": safe_int(
            rank_features.get("verifier_unsupported_pairs")
        )
        if rank_features
        else None,
        "top_verifier_apply_failed_pairs": safe_int(
            rank_features.get("verifier_apply_failed_pairs")
        )
        if rank_features
        else None,
        "top_verifier_shape_mismatch_pairs": safe_int(
            rank_features.get("verifier_shape_mismatch_pairs")
        )
        if rank_features
        else None,
        "dominant_candidate_rejection_reason": candidate_reason,
        "dominant_candidate_rejection_reason_count": candidate_reason_count,
        "dominant_candidate_failure_stage": candidate_stage,
        "dominant_candidate_failure_stage_count": candidate_stage_count,
        "dominant_candidate_failure_detail": candidate_detail,
        "dominant_candidate_failure_detail_count": candidate_detail_count,
        "dominant_retained_rejection_reason": retained_reason,
        "dominant_retained_rejection_reason_count": retained_reason_count,
        "dominant_retained_failure_stage": retained_stage,
        "dominant_retained_failure_stage_count": retained_stage_count,
        "dominant_retained_failure_detail": retained_detail,
        "dominant_retained_failure_detail_count": retained_detail_count,
        "dominant_candidate_op": candidate_op,
        "dominant_candidate_op_count": candidate_op_count,
        "dominant_candidate_pipeline": candidate_pipeline,
        "dominant_candidate_pipeline_count": candidate_pipeline_count,
        "accepted_ops": dict(sorted(stats.accepted_ops.items())),
    }


def aggregate_counts(rows: list[dict[str, Any]], key: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter[str(row.get(key, "") or "")] += 1
    return counter


def fmt_float(value: Any) -> str:
    if value is None:
        return ""
    return f"{safe_float(value):.4f}"


def fmt_int(value: Any) -> str:
    if value is None:
        return ""
    return str(safe_int(value))


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bucket_counts = aggregate_counts(rows, "failure_bucket")
    diagnosis_counts = aggregate_counts(rows, "ranker_diagnosis")
    missed = [row for row in rows if row.get("failure_bucket") != "solved_top1"]
    missed_reason_counts = aggregate_counts(missed, "dominant_candidate_rejection_reason")
    lines = [
        "# ARC Ranker Failure Report",
        "",
        f"Target tasks: `{len(rows)}`",
        f"Solved top-1: `{bucket_counts.get('solved_top1', 0)}`",
        f"Missed: `{len(missed)}`",
    ]
    if summary:
        lines.extend(
            [
                f"Ranker proposals: `{summary.get('proposal_count', '')}`",
                f"Top-1 all-task rate: `{summary.get('top1_all_task_rate', '')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Buckets",
            "",
            "| Bucket | Tasks |",
            "|---|---:|",
        ]
    )
    for bucket, count in sorted(bucket_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {bucket} | {count} |")
    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            "| Diagnosis | Tasks |",
            "|---|---:|",
        ]
    )
    for diagnosis, count in sorted(diagnosis_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {diagnosis} | {count} |")
    lines.extend(
        [
            "",
            "## Dominant Missed Candidate Rejection Reasons",
            "",
            "| Reason | Tasks |",
            "|---|---:|",
        ]
    )
    for reason, count in sorted(
        missed_reason_counts.items(), key=lambda item: (-item[1], item[0])
    ):
        lines.append(f"| {reason} | {count} |")
    lines.extend(
        [
            "",
            "## Missed Tasks",
            "",
            "| Task | Bucket | Candidates | Accepted candidates | Retained | First accepted rank | Top margin | Top reason | Candidate reason | Top op |",
            "|---|---|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in sorted(
        missed,
        key=lambda item: (
            str(item.get("failure_bucket", "")),
            -safe_int(item.get("candidate_count")),
            str(item.get("task_id", "")),
        ),
    ):
        lines.append(
            "| {task} | {bucket} | {cand} | {accepted} | {retained} | {rank} | {margin} | {top_reason} | {candidate_reason} | {top_op} |".format(
                task=row.get("task_id", ""),
                bucket=row.get("failure_bucket", ""),
                cand=fmt_int(row.get("candidate_count")),
                accepted=fmt_int(row.get("accepted_candidate_count")),
                retained=fmt_int(row.get("retained_count")),
                rank=fmt_int(row.get("first_accepted_rank")),
                margin=fmt_float(row.get("top_score_margin")),
                top_reason=row.get("top_rejection_reason", ""),
                candidate_reason=row.get("dominant_candidate_rejection_reason", ""),
                top_op=row.get("top_op_name", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts-tsv", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.attempts_tsv.is_file():
        raise SystemExit(f"missing attempts TSV: {args.attempts_tsv}")
    if not args.proposals.is_file():
        raise SystemExit(f"missing ranker proposals: {args.proposals}")
    attempt_stats = load_attempt_stats(args.attempts_tsv)
    proposals = load_proposals(args.proposals)
    keys = sorted(set(attempt_stats) | set(proposals))
    rows = [
        task_row(key, attempt_stats.get(key, AttemptStats()), proposals.get(key, []))
        for key in keys
    ]
    summary = load_summary(args.summary)
    expected_tasks = safe_int(summary.get("target_task_count"))
    if expected_tasks and expected_tasks != len(rows):
        raise SystemExit(
            f"target task count mismatch: summary={expected_tasks} report={len(rows)}"
        )
    write_jsonl(args.output, rows, compact=True)
    write_markdown(args.markdown, rows, summary)
    bucket_counts = aggregate_counts(rows, "failure_bucket")
    print(
        "arc-ranker-failure-report "
        f"tasks={len(rows)} "
        f"solved_top1={bucket_counts.get('solved_top1', 0)} "
        f"candidate_generation_gap={bucket_counts.get('candidate_generation_gap', 0)} "
        f"retention_gap={bucket_counts.get('retention_gap', 0)} "
        f"ranking_gap={bucket_counts.get('ranking_gap', 0)} "
        f"output={args.output} markdown={args.markdown}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
