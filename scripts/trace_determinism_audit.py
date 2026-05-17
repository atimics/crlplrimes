#!/usr/bin/env python3
"""Audit trace replay determinism without requiring bit-identical FP scores.

The semantic trace identity intentionally excludes raw logits, probabilities,
losses, and verifier timing/cost diagnostics. Replay identity is the stable
candidate/action contract: state, ordered action ids, selected/target action ids,
verifier result, and the declared ranking policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


TRACE_SCHEMA = "crlp.trace.v1"
TRACE_VERSION = 1
ACTION_ID_CONTRACT = "stable_unique_action_ids"
CANONICAL_SCHEMA = "crlp.trace.semantic.v1"
AUDIT_SCHEMA = "crlp.trace_determinism_audit.v1"
COMPARE_SCHEMA = "crlp.trace_determinism_compare.v1"
RANKING_POLICY = "score_desc_action_id_tiebreak_epsilon"
LEGACY_RANKING_POLICY = "legacy_score_order"
DEFAULT_RANK_SCORE_EPSILON = 1e-6


class TraceAuditError(RuntimeError):
    pass


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def as_float(row: dict[str, Any], key: str, default: float | None = None) -> float:
    value = row.get(key, default)
    if not finite_number(value):
        raise TraceAuditError(f"{key} must be a finite number")
    return float(value)


def as_int(row: dict[str, Any], key: str) -> int:
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TraceAuditError(f"{key} must be an integer")
    return value


def json_dumps(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"))


def sha256_lines(lines: list[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def rank_score_is_better(
    scores: list[float],
    action_ids: list[int],
    left: int,
    right: int,
    epsilon: float,
) -> bool:
    delta = scores[left] - scores[right]
    eps = max(epsilon, 0.0)
    if delta > eps:
        return True
    if delta < -eps:
        return False
    if action_ids[left] != action_ids[right]:
        return action_ids[left] < action_ids[right]
    return left < right


def selected_by_policy(scores: list[float], action_ids: list[int], epsilon: float) -> int:
    best = 0
    for index in range(1, len(scores)):
        if rank_score_is_better(scores, action_ids, index, best, epsilon):
            best = index
    return best


def selected_score_margin(scores: list[float], selected_index: int) -> float:
    if len(scores) == 1:
        return 0.0
    best_other = max(score for index, score in enumerate(scores) if index != selected_index)
    return scores[selected_index] - best_other


def parse_number_array(row: dict[str, Any], key: str, count: int) -> list[float]:
    values = row.get(key)
    if not isinstance(values, list) or len(values) != count:
        raise TraceAuditError(f"{key} must contain {count} values")
    out: list[float] = []
    for index, value in enumerate(values):
        if not finite_number(value):
            raise TraceAuditError(f"{key}[{index}] must be finite")
        out.append(float(value))
    return out


def parse_action_ids(row: dict[str, Any], count: int) -> list[int]:
    values = row.get("action_ids")
    if not isinstance(values, list) or len(values) != count:
        raise TraceAuditError(f"action_ids must contain {count} values")
    out: list[int] = []
    for index, value in enumerate(values):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > 0xFFFFFFFF:
            raise TraceAuditError(f"action_ids[{index}] must be a uint32")
        out.append(value)
    if len(set(out)) != len(out):
        raise TraceAuditError("action_ids must be unique")
    return out


def canonicalize_row(row: dict[str, Any], row_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if row.get("schema") != TRACE_SCHEMA:
        raise TraceAuditError(f"schema must be {TRACE_SCHEMA}")
    if row.get("trace_version") != TRACE_VERSION:
        raise TraceAuditError(f"trace_version must be {TRACE_VERSION}")
    if row.get("action_id_contract") != ACTION_ID_CONTRACT:
        raise TraceAuditError(f"action_id_contract must be {ACTION_ID_CONTRACT}")
    if not isinstance(row.get("task"), str) or not row["task"]:
        raise TraceAuditError("task must be a non-empty string")
    if not isinstance(row.get("state_hash"), str) or not row["state_hash"]:
        raise TraceAuditError("state_hash must be a non-empty string")
    if not isinstance(row.get("verifier_result"), str) or not row["verifier_result"]:
        raise TraceAuditError("verifier_result must be a non-empty string")
    feature_encoder_version = as_int(row, "feature_encoder_version")
    seed = as_int(row, "seed")
    candidate_count = as_int(row, "candidate_count")
    if candidate_count <= 0:
        raise TraceAuditError("candidate_count must be positive")
    selected_index = as_int(row, "selected_index")
    target_index = as_int(row, "target_index")
    if selected_index < 0 or selected_index >= candidate_count:
        raise TraceAuditError("selected_index is out of bounds")
    if target_index < 0 or target_index >= candidate_count:
        raise TraceAuditError("target_index is out of bounds")

    action_ids = parse_action_ids(row, candidate_count)
    scores = parse_number_array(row, "scores", candidate_count)
    probabilities = parse_number_array(row, "probabilities", candidate_count)
    probability_sum = sum(probabilities)
    if any(probability < 0.0 or probability > 1.0 for probability in probabilities):
        raise TraceAuditError("probabilities must be in [0, 1]")
    if abs(probability_sum - 1.0) > 1e-3:
        raise TraceAuditError("probability mass must sum to 1 within 1e-3")

    selected_action_id = action_ids[selected_index]
    target_action_id = action_ids[target_index]
    if "selected_action_id" in row and row["selected_action_id"] != selected_action_id:
        raise TraceAuditError("selected_action_id does not match selected_index")
    if "target_action_id" in row and row["target_action_id"] != target_action_id:
        raise TraceAuditError("target_action_id does not match target_index")

    epsilon = as_float(row, "rank_score_epsilon", DEFAULT_RANK_SCORE_EPSILON)
    if epsilon < 0.0:
        raise TraceAuditError("rank_score_epsilon must be non-negative")
    margin = selected_score_margin(scores, selected_index)
    if "selected_score_margin" in row:
        logged_margin = as_float(row, "selected_score_margin")
        tolerance = abs(margin) * 1e-5 + 1e-5
        if abs(logged_margin - margin) > tolerance:
            raise TraceAuditError("selected_score_margin does not match scores")
        margin = logged_margin

    ranking_policy = str(row.get("ranking_policy", LEGACY_RANKING_POLICY))
    if ranking_policy not in {RANKING_POLICY, LEGACY_RANKING_POLICY}:
        raise TraceAuditError("ranking_policy is unknown")
    if ranking_policy == RANKING_POLICY:
        policy_selected = selected_by_policy(scores, action_ids, epsilon)
        if policy_selected != selected_index:
            raise TraceAuditError("selected_index contradicts ranking_policy")

    computed_fp_sensitive = abs(margin) <= epsilon
    logged_fp_sensitive = row.get("rank_fp_sensitive")
    fp_sensitive_mismatch = (
        isinstance(logged_fp_sensitive, bool) and logged_fp_sensitive != computed_fp_sensitive
    )

    semantic = {
        "schema": CANONICAL_SCHEMA,
        "source_schema": TRACE_SCHEMA,
        "row_index": row_index,
        "task": row["task"],
        "trace_version": TRACE_VERSION,
        "action_id_contract": ACTION_ID_CONTRACT,
        "state_hash": row["state_hash"],
        "feature_encoder_version": feature_encoder_version,
        "candidate_count": candidate_count,
        "action_ids": action_ids,
        "selected_action_id": selected_action_id,
        "target_action_id": target_action_id,
        "verifier_result": row["verifier_result"],
        "seed": seed,
        "ranking_policy": ranking_policy,
        "rank_fp_sensitive": computed_fp_sensitive,
    }
    metrics = {
        "task": row["task"],
        "ranking_policy": ranking_policy,
        "selected_score_margin": margin,
        "rank_score_epsilon": epsilon,
        "rank_fp_sensitive": computed_fp_sensitive,
        "rank_fp_sensitive_logged_mismatch": fp_sensitive_mismatch,
    }
    return semantic, metrics


def audit_trace(path: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    canonical_lines: list[str] = []
    issues: list[str] = []
    margins: list[float] = []
    absolute_margins: list[float] = []
    epsilon_values: list[float] = []
    task_counts: dict[str, int] = {}
    policy_counts: dict[str, int] = {}
    sensitive_rows = 0
    logged_sensitive_mismatches = 0
    negative_margin_rows = 0
    row_count = 0

    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise TraceAuditError(f"failed to open {path}: {exc}") from exc

    with handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
                semantic, metrics = canonicalize_row(row, row_count)
            except (json.JSONDecodeError, TraceAuditError) as exc:
                issues.append(f"{path}:{line_number}: {exc}")
                continue
            row_count += 1
            canonical_lines.append(json_dumps(semantic))
            task = str(metrics["task"])
            policy = str(metrics["ranking_policy"])
            margin = float(metrics["selected_score_margin"])
            epsilon = float(metrics["rank_score_epsilon"])
            task_counts[task] = task_counts.get(task, 0) + 1
            policy_counts[policy] = policy_counts.get(policy, 0) + 1
            margins.append(margin)
            absolute_margins.append(abs(margin))
            epsilon_values.append(epsilon)
            if margin < 0.0:
                negative_margin_rows += 1
            if bool(metrics["rank_fp_sensitive"]):
                sensitive_rows += 1
            if bool(metrics["rank_fp_sensitive_logged_mismatch"]):
                logged_sensitive_mismatches += 1

    audit = {
        "schema": AUDIT_SCHEMA,
        "trace_path": str(path),
        "row_count": row_count,
        "task_counts": task_counts,
        "ranking_policy_counts": policy_counts,
        "semantic_hash": sha256_lines(canonical_lines),
        "semantic_schema": CANONICAL_SCHEMA,
        "rank_fp_sensitive_rows": sensitive_rows,
        "rank_fp_sensitive_rate": (sensitive_rows / row_count) if row_count else 0.0,
        "rank_fp_sensitive_logged_mismatches": logged_sensitive_mismatches,
        "negative_margin_rows": negative_margin_rows,
        "score_margin_min": min(margins) if margins else None,
        "score_margin_median": percentile(margins, 0.50),
        "score_margin_p95": percentile(margins, 0.95),
        "abs_score_margin_min": min(absolute_margins) if absolute_margins else None,
        "abs_score_margin_median": percentile(absolute_margins, 0.50),
        "abs_score_margin_p95": percentile(absolute_margins, 0.95),
        "rank_score_epsilon_min": min(epsilon_values) if epsilon_values else None,
        "rank_score_epsilon_max": max(epsilon_values) if epsilon_values else None,
        "invalid_rows": len(issues),
    }
    return audit, canonical_lines, issues


def compare_canonical(paths: list[Path], canonical_by_path: dict[Path, list[str]]) -> dict[str, Any]:
    baseline = paths[0]
    baseline_lines = canonical_by_path[baseline]
    differing_paths: list[str] = []
    first_mismatch: dict[str, Any] | None = None
    for path in paths[1:]:
        lines = canonical_by_path[path]
        if lines == baseline_lines:
            continue
        differing_paths.append(str(path))
        if first_mismatch is None:
            limit = min(len(baseline_lines), len(lines))
            mismatch_index = limit
            for index in range(limit):
                if baseline_lines[index] != lines[index]:
                    mismatch_index = index
                    break
            first_mismatch = {
                "path": str(path),
                "row_index": mismatch_index,
                "baseline_row": baseline_lines[mismatch_index] if mismatch_index < len(baseline_lines) else None,
                "comparison_row": lines[mismatch_index] if mismatch_index < len(lines) else None,
            }
    hashes = {str(path): sha256_lines(canonical_by_path[path]) for path in paths}
    row_counts = {str(path): len(canonical_by_path[path]) for path in paths}
    return {
        "schema": COMPARE_SCHEMA,
        "baseline_trace_path": str(baseline),
        "trace_paths": [str(path) for path in paths],
        "semantic_hashes": hashes,
        "row_counts": row_counts,
        "all_semantic_hashes_equal": len(set(hashes.values())) == 1,
        "row_counts_equal": len(set(row_counts.values())) == 1,
        "differing_paths": differing_paths,
        "first_mismatch": first_mismatch,
    }


def collect_inputs(args: argparse.Namespace) -> list[Path]:
    paths = [Path(value) for value in args.traces]
    paths.extend(args.input or [])
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json_dumps(row) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="*", type=Path, help="trace JSONL files to audit")
    parser.add_argument("--input", action="append", type=Path, help="trace JSONL file to audit")
    parser.add_argument("--output", type=Path, help="write audit/compare JSONL rows")
    parser.add_argument(
        "--canonical-output",
        type=Path,
        help="write canonical semantic trace JSONL; valid only for one input trace",
    )
    args = parser.parse_args()

    paths = collect_inputs(args)
    if not paths:
        parser.error("at least one trace path is required")
    if args.canonical_output is not None and len(paths) != 1:
        parser.error("--canonical-output is valid only when auditing one trace")

    rows: list[dict[str, Any]] = []
    all_issues: list[str] = []
    canonical_by_path: dict[Path, list[str]] = {}
    for path in paths:
        audit, canonical_lines, issues = audit_trace(path)
        rows.append(audit)
        canonical_by_path[path] = canonical_lines
        all_issues.extend(issues)

    if len(paths) > 1:
        rows.append(compare_canonical(paths, canonical_by_path))

    if args.output is not None:
        write_jsonl(args.output, rows)
    if args.canonical_output is not None:
        args.canonical_output.parent.mkdir(parents=True, exist_ok=True)
        args.canonical_output.write_text(
            "".join(line + "\n" for line in canonical_by_path[paths[0]]),
            encoding="utf-8",
        )

    for row in rows:
        print(json_dumps(row))
    for issue in all_issues:
        print(issue, file=sys.stderr)
    return 1 if all_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
