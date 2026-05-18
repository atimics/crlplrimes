#!/usr/bin/env python3
"""Summarize Signal live shadow-neural decisions on the same logged frames.

The live brain only executes the heuristic action in `shadow_neural` mode. This
audit therefore does not pretend to know the neural counterfactual outcome. It
records same-frame scorer choices, safety-mask decisions, inference latency, and
the next observed outcome of the executed active action so a later deterministic
replay pass has a stable contract to compare against.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from datetime import date
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line.startswith("{"):
                rows.append(json.loads(line))
    return rows


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
    return values[index]


def top_margin(scores: list[dict[str, Any]] | None) -> float | None:
    if not scores or len(scores) < 2:
        return None
    ordered = sorted((float(row.get("score", 0.0) or 0.0) for row in scores), reverse=True)
    return ordered[0] - ordered[1]


def action_rank(scores: list[dict[str, Any]] | None, action: str | None) -> int | None:
    if not scores or action is None:
        return None
    ordered = sorted(
        scores,
        key=lambda row: (-float(row.get("score", 0.0) or 0.0), str(row.get("action", ""))),
    )
    for index, row in enumerate(ordered, start=1):
        if row.get("action") == action:
            return index
    return None


def round_or_none(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def frame_rows(rows: list[dict[str, Any]], horizon_ms: int) -> list[dict[str, Any]]:
    live_rows = [row for row in rows if row.get("schema") == "crlp.signal_live_brain.v1"]
    out: list[dict[str, Any]] = []
    for index, row in enumerate(live_rows):
        if row.get("neural_action") in (None, ""):
            continue
        next_row = None
        for candidate in live_rows[index + 1 :]:
            if int(candidate.get("t_ms", 0) or 0) - int(row.get("t_ms", 0) or 0) > horizon_ms:
                break
            if candidate.get("player_id") == row.get("player_id"):
                next_row = candidate
                break
        hull_delta = None
        dist_delta = None
        if next_row is not None:
            if isinstance(row.get("hull"), (int, float)) and isinstance(next_row.get("hull"), (int, float)):
                hull_delta = float(next_row["hull"]) - float(row["hull"])
            if isinstance(row.get("dist"), (int, float)) and isinstance(next_row.get("dist"), (int, float)):
                dist_delta = float(next_row["dist"]) - float(row["dist"])
        scores = row.get("candidate_scores")
        out.append(
            {
                "schema": "crlp.signal_shadow_frame.v1",
                "report_version": 1,
                "date": date.today().isoformat(),
                "t_ms": row.get("t_ms"),
                "player_id": row.get("player_id"),
                "brain_mode": row.get("brain_mode"),
                "active_source": row.get("active_source"),
                "active_action": row.get("active_action", row.get("action")),
                "heuristic_action": row.get("heuristic_action"),
                "neural_action": row.get("neural_action"),
                "shadow_disagrees": bool(row.get("shadow_disagrees")),
                "safety_gate": row.get("safety_gate"),
                "safety_reject_reasons": row.get("safety_reject_reasons") or [],
                "protocol_status": row.get("protocol_status"),
                "protocol_compatible": row.get("protocol_compatible"),
                "protocol_hash": row.get("protocol_hash"),
                "protocol_mismatch_count": row.get("protocol_mismatch_count"),
                "protocol_missing_count": row.get("protocol_missing_count"),
                "candidate_count": len(scores or []),
                "neural_top_margin": round_or_none(top_margin(scores)),
                "heuristic_rank_in_neural_scores": action_rank(scores, row.get("heuristic_action")),
                "neural_rank": action_rank(scores, row.get("neural_action")),
                "inference_ms": row.get("inference_ms"),
                "same_frame_comparison": True,
                "counterfactual_replay_available": False,
                "executed_outcome_horizon_ms": horizon_ms,
                "next_observation_available": next_row is not None,
                "executed_hull_delta": round_or_none(hull_delta, 3),
                "executed_dist_delta": round_or_none(dist_delta, 3),
            }
        )
    return out


def summary_row(frames: list[dict[str, Any]], log_path: Path, horizon_ms: int) -> dict[str, Any]:
    disagreements = sum(1 for row in frames if row["shadow_disagrees"])
    latencies = [
        float(row["inference_ms"])
        for row in frames
        if isinstance(row.get("inference_ms"), (int, float))
    ]
    gates = collections.Counter(str(row.get("safety_gate") or "none") for row in frames)
    reasons = collections.Counter()
    for row in frames:
        for reason in row.get("safety_reject_reasons", []) or []:
            reasons[str(reason)] += 1
    protocol_statuses = collections.Counter(
        str(row.get("protocol_status") or "missing") for row in frames
    )
    protocol_compatible_rows = sum(
        1 for row in frames if row.get("protocol_compatible") is True or row.get("protocol_status") == "ok"
    )
    return {
        "schema": "crlp.signal_shadow_replay_audit.v1",
        "report_version": 1,
        "date": date.today().isoformat(),
        "log_path": str(log_path),
        "frame_rows": len(frames),
        "shadow_frames": len(frames),
        "shadow_disagreements": disagreements,
        "shadow_disagreement_rate": disagreements / len(frames) if frames else 0.0,
        "same_frame_comparison": True,
        "counterfactual_replay_available": False,
        "executed_outcome_horizon_ms": horizon_ms,
        "next_observation_rows": sum(1 for row in frames if row["next_observation_available"]),
        "inference_ms_p50": percentile(latencies, 0.50),
        "inference_ms_p95": percentile(latencies, 0.95),
        "inference_ms_p99": percentile(latencies, 0.99),
        "safety_gates": dict(gates),
        "safety_reject_reasons": dict(reasons),
        "protocol_compatible_rows": protocol_compatible_rows,
        "protocol_statuses": dict(protocol_statuses),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True) + "\n")


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Signal Shadow Replay Audit",
        "",
        f"Date: {date.today().isoformat()}",
        f"Log: `{summary['log_path']}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Shadow frames | {summary['shadow_frames']} |",
        f"| Disagreements | {summary['shadow_disagreements']} |",
        f"| Disagreement rate | {100.0 * summary['shadow_disagreement_rate']:.2f}% |",
        f"| Inference p95 ms | {summary['inference_ms_p95']} |",
        f"| Protocol-compatible rows | {summary['protocol_compatible_rows']} |",
        f"| Next observations | {summary['next_observation_rows']} |",
        "",
        "The audit is a same-frame shadow comparison. Neural counterfactual outcomes are "
        "not claimed until a deterministic replay pass consumes these frame rows.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=Path("build/float/signal_live_brain/live.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/float/signal_shadow_replay_audit"))
    parser.add_argument("--horizon-ms", type=int, default=500)
    args = parser.parse_args()

    rows = read_jsonl(args.log)
    frames = frame_rows(rows, args.horizon_ms)
    summary = summary_row(frames, args.log, args.horizon_ms)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = args.output_dir / "shadow_frames.jsonl"
    summary_path = args.output_dir / "summary.jsonl"
    report_path = args.output_dir / "summary.md"
    write_jsonl(frame_path, frames)
    write_jsonl(summary_path, [summary])
    write_summary(report_path, summary)
    print(f"wrote {frame_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
