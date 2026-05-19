#!/usr/bin/env python3
"""Summarize guarded neural-active Signal live-brain logs.

The live brain JSONL is the authority for active-control existence claims. This
script does not score the active pilot against a baseline. It only verifies that
recent rows show a non-dry-run `neural_active` pilot using neural actions, with a
live checkpoint identity, inference timing, safety-gate fields, damage/death
telemetry, and enough frame fields for later deterministic replay.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_REPLAY_FIELDS = (
    "t_ms",
    "player_id",
    "brain_mode",
    "active_source",
    "checkpoint_hash",
    "feature_set",
    "action",
    "flags",
    "x",
    "y",
    "vx",
    "vy",
    "angle",
    "hull",
    "target",
    "candidate_scores",
)


def read_live_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("schema") == "crlp.signal_live_brain.v1":
                rows.append(row)
    return rows


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def round_or_none(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def counter_dict(counter: collections.Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def top_counts(counter: collections.Counter[str], limit: int = 6) -> list[dict[str, Any]]:
    total = sum(counter.values())
    return [
        {
            "label": label,
            "count": count,
            "rate": count / total if total else 0.0,
        }
        for label, count in counter.most_common(limit)
    ]


def protocol_runtime_checked(row: dict[str, Any]) -> bool:
    return bool(row.get("protocol_runtime_received")) and row.get("protocol_status") not in (
        None,
        "",
        "missing",
        "pending",
        "skipped",
    )


def protocol_runtime_status(row: dict[str, Any]) -> str:
    received = row.get("protocol_runtime_received")
    if received is True:
        return str(row.get("protocol_status") or "missing")
    if received is False:
        return str(row.get("protocol_status") or "pending")
    return "legacy_static_only"


def protocol_runtime_compatible(row: dict[str, Any]) -> bool:
    return protocol_runtime_checked(row) and (
        row.get("protocol_compatible") is True or row.get("protocol_status") == "ok"
    )


def protocol_source_checked(row: dict[str, Any]) -> bool:
    status = row.get("protocol_source_status")
    if status is None and not row.get("protocol_runtime_received"):
        status = row.get("protocol_status")
    return status not in (None, "", "missing", "pending", "skipped")


def protocol_source_compatible(row: dict[str, Any]) -> bool:
    if row.get("protocol_source_compatible") is not None:
        return row.get("protocol_source_compatible") is True
    return protocol_source_checked(row) and (
        row.get("protocol_compatible") is True or row.get("protocol_status") == "ok"
    )


def replay_ready(row: dict[str, Any]) -> bool:
    if any(row.get(field) in (None, "") for field in REQUIRED_REPLAY_FIELDS):
        return False
    scores = row.get("candidate_scores")
    return isinstance(scores, list) and len(scores) >= 9


def damage_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    previous_by_player: dict[Any, dict[str, Any]] = {}
    min_hull: float | None = None
    max_hull: float | None = None
    hull_loss = 0.0
    damage_events = 0
    death_events = 0
    last_damage_t_ms: int | None = None

    for row in rows:
        hull = finite_number(row.get("hull"))
        if hull is None:
            continue
        min_hull = hull if min_hull is None else min(min_hull, hull)
        max_hull = hull if max_hull is None else max(max_hull, hull)
        player_id = row.get("player_id")
        previous = previous_by_player.get(player_id)
        if previous is not None:
            previous_hull = finite_number(previous.get("hull"))
            if previous_hull is not None:
                drop = previous_hull - hull
                if drop > 0.01:
                    damage_events += 1
                    hull_loss += drop
                    t_ms = row.get("t_ms")
                    last_damage_t_ms = int(t_ms) if isinstance(t_ms, int) else last_damage_t_ms
                if previous_hull > 0 and hull <= 0:
                    death_events += 1
        previous_by_player[player_id] = row

    return {
        "min_hull": round_or_none(min_hull, 3),
        "max_hull": round_or_none(max_hull, 3),
        "hull_loss": round(hull_loss, 3),
        "damage_events": damage_events,
        "death_events": death_events,
        "last_damage_t_ms": last_damage_t_ms,
    }


def max_gap_ms(rows: list[dict[str, Any]]) -> int | None:
    times = [int(row["t_ms"]) for row in rows if isinstance(row.get("t_ms"), int)]
    if len(times) < 2:
        return None
    times.sort()
    return max(times[index + 1] - times[index] for index in range(len(times) - 1))


def latest_age_at_capture_ms(log_path: Path, latest_row: dict[str, Any] | None) -> int | None:
    if latest_row is None or not isinstance(latest_row.get("t_ms"), int) or not log_path.exists():
        return None
    mtime_ms = int(log_path.stat().st_mtime * 1000)
    return max(0, mtime_ms - int(latest_row["t_ms"]))


def summarize(rows: list[dict[str, Any]], log_path: Path, window_rows: int, stale_ms: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    active_rows = [
        row
        for row in rows
        if row.get("brain_mode") == "neural_active" and row.get("active_source") == "neural"
    ]
    window = active_rows[-window_rows:] if window_rows > 0 else active_rows
    latest = window[-1] if window else (active_rows[-1] if active_rows else (rows[-1] if rows else None))

    latencies = [
        float(row["inference_ms"])
        for row in window
        if isinstance(row.get("inference_ms"), (int, float)) and math.isfinite(float(row["inference_ms"]))
    ]
    checkpoint_hashes = collections.Counter(
        str(row.get("checkpoint_hash")) for row in window if row.get("checkpoint_hash")
    )
    checkpoint_paths = collections.Counter(
        str(row.get("checkpoint_path")) for row in window if row.get("checkpoint_path")
    )
    feature_sets = collections.Counter(
        str(row.get("feature_set")) for row in window if row.get("feature_set")
    )
    actions = collections.Counter(
        str(row.get("active_action") or row.get("action") or "unknown") for row in window
    )
    safety_gates = collections.Counter(str(row.get("safety_gate") or "missing") for row in window)
    safety_reasons: collections.Counter[str] = collections.Counter()
    for row in window:
        for reason in row.get("safety_reject_reasons") or []:
            safety_reasons[str(reason)] += 1
    protocol_statuses = collections.Counter(protocol_runtime_status(row) for row in window)
    protocol_hashes = collections.Counter(
        str(row.get("protocol_runtime_hash") or row.get("protocol_hash"))
        for row in window
        if protocol_runtime_checked(row) and (row.get("protocol_runtime_hash") or row.get("protocol_hash"))
    )
    protocol_checked_rows = sum(1 for row in window if protocol_runtime_checked(row))
    protocol_compatible_rows = sum(1 for row in window if protocol_runtime_compatible(row))
    protocol_source_statuses = collections.Counter(
        str(row.get("protocol_source_status") or row.get("protocol_status") or "missing")
        for row in window
    )
    protocol_source_hashes = collections.Counter(
        str(row.get("protocol_source_hash") or row.get("protocol_hash"))
        for row in window
        if protocol_source_checked(row) and (row.get("protocol_source_hash") or row.get("protocol_hash"))
    )
    protocol_source_checked_rows = sum(1 for row in window if protocol_source_checked(row))
    protocol_source_compatible_rows = sum(1 for row in window if protocol_source_compatible(row))

    candidate_counts = [
        len(row.get("candidate_scores") or [])
        for row in window
        if isinstance(row.get("candidate_scores"), list)
    ]
    replay_rows = [row for row in window if replay_ready(row)]
    damage = damage_stats(window)
    latest_hull = finite_number(latest.get("hull")) if latest else None
    gap_ms = max_gap_ms(window)
    active_dry_run_rows = sum(1 for row in window if bool(row.get("dry_run")))
    shadow_compared = sum(1 for row in window if row.get("neural_action") not in (None, ""))
    shadow_disagreements = sum(1 for row in window if bool(row.get("shadow_disagrees")))
    station_diag_rows = sum(1 for row in window if isinstance(row.get("target_station_diag"), dict))

    info = log_path.stat() if log_path.exists() else None
    source_modified_utc = (
        datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat() if info else None
    )

    summary = {
        "schema": "crlp.signal_live_active_summary.v1",
        "report_version": 1,
        "date": date.today().isoformat(),
        "log_path": str(log_path),
        "source_size_bytes": info.st_size if info else None,
        "source_modified_utc": source_modified_utc,
        "rows_seen": len(rows),
        "active_rows": len(active_rows),
        "window_rows": len(window),
        "window_active_rows": len(window),
        "latest_t_ms": latest.get("t_ms") if latest else None,
        "latest_brain_mode": latest.get("brain_mode") if latest else None,
        "latest_active_source": latest.get("active_source") if latest else None,
        "checkpoint_hash": checkpoint_hashes.most_common(1)[0][0] if checkpoint_hashes else None,
        "checkpoint_path": checkpoint_paths.most_common(1)[0][0] if checkpoint_paths else None,
        "checkpoint_hash_count": len(checkpoint_hashes),
        "checkpoint_hashes": counter_dict(checkpoint_hashes),
        "feature_set": feature_sets.most_common(1)[0][0] if feature_sets else None,
        "feature_sets": counter_dict(feature_sets),
        "dry_run_rows": active_dry_run_rows,
        "inference_ms_p50": round_or_none(percentile(latencies, 0.50)),
        "inference_ms_p95": round_or_none(percentile(latencies, 0.95)),
        "inference_ms_p99": round_or_none(percentile(latencies, 0.99)),
        "candidate_score_rows": len(candidate_counts),
        "candidate_score_min_count": min(candidate_counts) if candidate_counts else 0,
        "candidate_score_max_count": max(candidate_counts) if candidate_counts else 0,
        "safety_gate_rows": len(window) - safety_gates.get("missing", 0),
        "safety_gates": counter_dict(safety_gates),
        "safety_reject_reasons": counter_dict(safety_reasons),
        "protocol_checked_rows": protocol_checked_rows,
        "protocol_compatible_rows": protocol_compatible_rows,
        "protocol_statuses": counter_dict(protocol_statuses),
        "protocol_hash": protocol_hashes.most_common(1)[0][0] if protocol_hashes else None,
        "protocol_hashes": counter_dict(protocol_hashes),
        "protocol_source_checked_rows": protocol_source_checked_rows,
        "protocol_source_compatible_rows": protocol_source_compatible_rows,
        "protocol_source_statuses": counter_dict(protocol_source_statuses),
        "protocol_source_hash": protocol_source_hashes.most_common(1)[0][0] if protocol_source_hashes else None,
        "protocol_source_hashes": counter_dict(protocol_source_hashes),
        "replay_ready_rows": len(replay_rows),
        "station_diag_rows": station_diag_rows,
        "latest_station_diag_frames": latest.get("station_diag_frames") if latest else None,
        "latest_station_diag_stations": latest.get("station_diag_stations") if latest else None,
        "top_actions": top_counts(actions),
        "shadow_compared_rows": shadow_compared,
        "shadow_disagreements": shadow_disagreements,
        "shadow_disagreement_rate": shadow_disagreements / shadow_compared if shadow_compared else None,
        "latest_hull": round_or_none(latest_hull, 3),
        "max_interrow_gap_ms": gap_ms,
        "stale_within_log": bool(gap_ms is not None and gap_ms > stale_ms),
        "latest_row_age_at_capture_ms": latest_age_at_capture_ms(log_path, latest),
        **damage,
    }
    return summary, replay_rows


def frame_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "crlp.signal_live_active_frame.v1",
        "report_version": 1,
        "t_ms": row.get("t_ms"),
        "player_id": row.get("player_id"),
        "brain_mode": row.get("brain_mode"),
        "active_source": row.get("active_source"),
        "active_action": row.get("active_action", row.get("action")),
        "heuristic_action": row.get("heuristic_action"),
        "neural_action": row.get("neural_action"),
        "shadow_disagrees": bool(row.get("shadow_disagrees")),
        "flags": row.get("flags"),
        "checkpoint_hash": row.get("checkpoint_hash"),
        "checkpoint_path": row.get("checkpoint_path"),
        "feature_set": row.get("feature_set"),
        "policy": row.get("policy"),
        "protocol_status": row.get("protocol_status"),
        "protocol_compatible": row.get("protocol_compatible"),
        "protocol_hash": row.get("protocol_hash"),
        "protocol_mismatch_count": row.get("protocol_mismatch_count"),
        "protocol_missing_count": row.get("protocol_missing_count"),
        "safety_gate": row.get("safety_gate"),
        "safety_reject_reasons": row.get("safety_reject_reasons") or [],
        "inference_ms": row.get("inference_ms"),
        "hull": row.get("hull"),
        "x": row.get("x"),
        "y": row.get("y"),
        "vx": row.get("vx"),
        "vy": row.get("vy"),
        "angle": row.get("angle"),
        "dist": row.get("dist"),
        "target": row.get("target"),
        "candidate_scores": row.get("candidate_scores") or [],
        "candidate_probabilities": row.get("candidate_probabilities") or [],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Signal Live Active Summary",
        "",
        f"Date: {summary['date']}",
        f"Log: `{summary['log_path']}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Active rows | {summary['active_rows']} |",
        f"| Window active rows | {summary['window_active_rows']} |",
        f"| Brain mode | `{summary['latest_brain_mode']}` |",
        f"| Active source | `{summary['latest_active_source']}` |",
        f"| Checkpoint | `{str(summary['checkpoint_hash'] or '')[:12]}` |",
        f"| Inference p95 ms | {summary['inference_ms_p95']} |",
        f"| Protocol-compatible rows | {summary['protocol_compatible_rows']} |",
        f"| Replay-ready rows | {summary['replay_ready_rows']} |",
        f"| Damage events | {summary['damage_events']} |",
        f"| Death events | {summary['death_events']} |",
        "",
        "This summary supports live neural-active execution only. It is not a "
        "comparative active-control performance result.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=Path("build/float/signal_live_brain/live.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/float/signal_live_active_summary"))
    parser.add_argument("--window-rows", type=int, default=250)
    parser.add_argument("--stale-ms", type=int, default=3000)
    args = parser.parse_args()

    rows = read_live_rows(args.log)
    summary, frames = summarize(rows, args.log, args.window_rows, args.stale_ms)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.jsonl"
    frames_path = args.output_dir / "active_frames.jsonl"
    report_path = args.output_dir / "summary.md"
    write_jsonl(summary_path, [summary])
    write_jsonl(frames_path, [frame_row(row) for row in frames])
    write_report(report_path, summary)
    print(f"wrote {summary_path}")
    print(f"wrote {frames_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
