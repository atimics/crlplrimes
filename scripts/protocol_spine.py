#!/usr/bin/env python3
"""Normalize domain traces into the shared candidate/outcome protocol spine."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


CANDIDATE_SCHEMA = "crlp.protocol.candidate_row.v1"
OUTCOME_SCHEMA = "crlp.protocol.verifier_outcome.v1"
SUMMARY_SCHEMA = "crlp.protocol_spine.summary.v1"


def stable_u32(text: str) -> int:
    value = 2166136261
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value or 1


def stable_hex(text: str) -> str:
    return f"{stable_u32(text):08x}"


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compact_ref(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        parts = []
        for key in ("sketch_id", "family", "outcome", "example_id"):
            if value.get(key) is not None:
                parts.append(f"{key}={value[key]}")
        return ";".join(parts)
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def candidate_row(
    *,
    stable_id: int,
    domain: str,
    candidate_kind: str,
    feature_count: int,
    provenance: str,
    example_text: str,
    source_schema: str,
    source_artifact: str,
    state_hash: str = "",
    frame: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema": CANDIDATE_SCHEMA,
        "stable_id": stable_id,
        "domain": domain,
        "candidate_kind": candidate_kind,
        "feature_count": max(1, feature_count),
        "provenance": provenance,
        "example_text": example_text,
    }
    if source_schema:
        row["source_schema"] = source_schema
    if source_artifact:
        row["source_artifact"] = source_artifact
    if state_hash:
        row["state_hash"] = state_hash
    if frame:
        if frame.get("sequence_id"):
            row["frame_sequence_id"] = frame["sequence_id"]
        if frame.get("length") is not None:
            row["frame_index"] = as_int(frame.get("index"))
            row["frame_length"] = as_int(frame.get("length"))
        if frame.get("position"):
            row["frame_position"] = str(frame["position"])
        previous = compact_ref(frame.get("previous"))
        next_ref = compact_ref(frame.get("next"))
        composition = compact_ref(frame.get("composition_context"))
        if previous:
            row["frame_previous"] = previous
        if next_ref:
            row["frame_next"] = next_ref
        if composition:
            row["composition_context"] = composition
    return row


def outcome_row(
    *,
    stable_id: int,
    domain: str,
    status: str,
    utility: float,
    support: int,
    false_rejections: int,
    cost: float,
    reason_code: str,
    proof_kind: str,
    source_schema: str,
    source_artifact: str,
    state_hash: str = "",
    parent_hash: str = "",
    trace_root: str = "",
    authority: str = "",
    signature: str = "",
    replay_seed: str = "",
    failure_stage: str = "",
    residual_class: str = "",
    domain_payload_schema: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema": OUTCOME_SCHEMA,
        "domain": domain,
        "stable_id": stable_id,
        "status": status,
        "utility": utility,
        "support": support,
        "false_rejections": false_rejections,
        "cost": cost,
        "reason_code": reason_code,
        "proof_kind": proof_kind,
    }
    optional = {
        "source_schema": source_schema,
        "source_artifact": source_artifact,
        "state_hash": state_hash,
        "parent_hash": parent_hash,
        "trace_root": trace_root,
        "authority": authority,
        "signature": signature,
        "replay_seed": replay_seed,
        "failure_stage": failure_stage,
        "residual_class": residual_class,
        "domain_payload_schema": domain_payload_schema,
    }
    row.update({key: value for key, value in optional.items() if value})
    return row


def normalize_signal_mining(path: Path) -> Iterable[dict[str, Any]]:
    rows = [
        row
        for row in load_jsonl(path)
        if row.get("schema") == "signal.mining_counterfactual.v1"
    ]
    group_lengths = Counter(str(row.get("group_id", "")) for row in rows)
    trace_root = stable_hex(f"{path}:signal_mining")
    for row in rows:
        group_id = str(row.get("group_id", ""))
        candidate_id = as_int(row.get("candidate_id"))
        stable_id = stable_u32(f"signal_mining:{group_id}:{candidate_id}")
        state_hash = stable_hex(
            "signal_mining:{world}:{scenario}:{prefix}:{group}".format(
                world=row.get("world_seed", ""),
                scenario=row.get("scenario", ""),
                prefix=row.get("prefix_ticks", ""),
                group=group_id,
            )
        )
        frame = {
            "sequence_id": f"sigmin:{group_id}",
            "index": candidate_id,
            "length": group_lengths[group_id],
            "position": row.get("split", ""),
            "composition_context": [
                row.get("target_rule", ""),
                row.get("route_rule", ""),
                row.get("speed_rule", ""),
                row.get("interaction_rule", ""),
            ],
        }
        candidate = candidate_row(
            stable_id=stable_id,
            domain="signal_mining",
            candidate_kind="tactical_option",
            feature_count=len(row.get("features") or []),
            provenance="signal_mining_counterfactual",
            example_text=str(row.get("candidate", "")),
            source_schema=str(row.get("schema", "")),
            source_artifact=str(path),
            state_hash=state_hash,
            frame=frame,
        )
        certified = bool(row.get("certified"))
        death_events = as_int(row.get("death_events"))
        damage_events = as_int(row.get("damage_events"))
        damage_amount = as_float(row.get("damage_amount"))
        if not certified:
            status = "rejected"
            reason = "uncertified_replay"
        elif death_events > 0:
            status = "unsafe"
            reason = "death_event"
        elif damage_events > 0:
            status = "accepted"
            reason = "damage_event"
        else:
            status = "accepted"
            reason = "replay_certified"
        residual = "death" if death_events > 0 else "damage" if damage_events > 0 else "clean"
        outcome = outcome_row(
            stable_id=stable_id,
            domain="signal_mining",
            status=status,
            utility=as_float(row.get("utility")),
            support=1 if certified else 0,
            false_rejections=0,
            cost=damage_amount + 1000.0 * float(death_events),
            reason_code=reason,
            proof_kind="signal_authoritative_replay",
            source_schema=str(row.get("schema", "")),
            source_artifact=str(path),
            state_hash=state_hash,
            trace_root=trace_root,
            authority="signal_simulator",
            replay_seed=str(row.get("world_seed", "")),
            failure_stage=reason,
            residual_class=f"{row.get('scenario', '')}:{residual}",
            domain_payload_schema=str(row.get("schema", "")),
        )
        yield candidate
        yield outcome


def normalize_grammar_sketch_examples(path: Path) -> Iterable[dict[str, Any]]:
    rows = [
        row
        for row in load_jsonl(path)
        if row.get("schema") == "crlp.grammar_sketch.example.v1"
    ]
    for row in rows:
        example_id = str(row.get("example_id", ""))
        sketch_id = str(row.get("sketch_id", ""))
        stable_id = stable_u32(f"grammar_sketch:{sketch_id}:{example_id}")
        frame = row.get("neural_frame") if isinstance(row.get("neural_frame"), dict) else None
        source_ref = row.get("source_ref") if isinstance(row.get("source_ref"), dict) else {}
        source_artifacts = row.get("source_artifacts") or []
        source_artifact = str(source_artifacts[0]) if source_artifacts else str(path)
        state_hash = str(source_ref.get("state_hash", ""))
        outcome_name = str(row.get("outcome", ""))
        status = {
            "accepted": "accepted",
            "rejected": "rejected",
            "unsafe": "unsafe",
            "no_support": "no_support",
            "diagnostic_positive": "no_support",
        }.get(outcome_name, "rejected")
        yield candidate_row(
            stable_id=stable_id,
            domain=str(row.get("domain", "")),
            candidate_kind=str(row.get("family", "")),
            feature_count=len((frame or {}).get("tokens") or []),
            provenance="grammar_sketch_example",
            example_text=example_id,
            source_schema=str(row.get("source_row_schema", "")),
            source_artifact=source_artifact,
            state_hash=state_hash,
            frame=frame,
        )
        yield outcome_row(
            stable_id=stable_id,
            domain=str(row.get("domain", "")),
            status=status,
            utility=as_float(row.get("utility")),
            support=1 if status == "accepted" else 0,
            false_rejections=1 if status == "unsafe" else 0,
            cost=0.0,
            reason_code=str(row.get("verifier_status", outcome_name)),
            proof_kind="grammar_sketch_verifier",
            source_schema=str(row.get("source_row_schema", "")),
            source_artifact=source_artifact,
            state_hash=state_hash,
            failure_stage=str(row.get("verifier_status", "")),
            residual_class=outcome_name,
            domain_payload_schema=str(row.get("schema", "")),
        )


def normalize_arc_hypothesis_evidence(path: Path) -> Iterable[dict[str, Any]]:
    rows = [
        row
        for row in load_jsonl(path)
        if row.get("schema") == "crlp.hypothesis.evidence.v1"
    ]
    for row in rows:
        canonical_id = str(row.get("canonical_id", ""))
        stable_id = stable_u32(f"arc_hypothesis:{canonical_id}")
        status = "accepted" if row.get("certificate_status") == "certified" else "rejected"
        audit = row.get("feature_audit") if isinstance(row.get("feature_audit"), dict) else {}
        yield candidate_row(
            stable_id=stable_id,
            domain=str(row.get("template_domain", row.get("task", "arc_agi"))),
            candidate_kind=str(row.get("template_name", "")),
            feature_count=as_int(audit.get("total"), as_int(row.get("slot_count"), 1)),
            provenance="arc_hypothesis_evidence",
            example_text=str(row.get("task_instance", "")),
            source_schema=str(row.get("schema", "")),
            source_artifact=str(path),
            state_hash=str(row.get("canonical_hash", "")),
        )
        yield outcome_row(
            stable_id=stable_id,
            domain=str(row.get("template_domain", row.get("task", "arc_agi"))),
            status=status,
            utility=as_float(row.get("utility")),
            support=as_int(row.get("support")),
            false_rejections=as_int(row.get("false_rejections")),
            cost=float(as_int(row.get("event_count"))),
            reason_code=str(row.get("reason", "")),
            proof_kind=str(row.get("proof_kind", "")),
            source_schema=str(row.get("schema", "")),
            source_artifact=str(path),
            state_hash=str(row.get("canonical_hash", "")),
            parent_hash=str(row.get("program_canonical_hash", "")),
            failure_stage=str(row.get("certificate_status", "")),
            residual_class=str(row.get("task_instance", "")),
            domain_payload_schema=str(row.get("schema", "")),
        )


def write_summary(path: Path, rows: list[dict[str, Any]], inputs: list[str]) -> None:
    by_schema = Counter(str(row.get("schema", "")) for row in rows)
    by_domain = Counter(str(row.get("domain", "")) for row in rows)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "inputs": inputs,
        "rows": len(rows),
        "by_schema": dict(sorted(by_schema.items())),
        "by_domain": dict(sorted(by_domain.items())),
    }
    path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-mining-trace", action="append", type=Path, default=[])
    parser.add_argument("--grammar-sketch-examples", type=Path)
    parser.add_argument("--arc-hypothesis-evidence", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    output_rows: list[dict[str, Any]] = []
    inputs: list[str] = []
    for path in args.signal_mining_trace:
        if path.exists():
            inputs.append(str(path))
            output_rows.extend(normalize_signal_mining(path))
    if args.grammar_sketch_examples and args.grammar_sketch_examples.exists():
        inputs.append(str(args.grammar_sketch_examples))
        output_rows.extend(normalize_grammar_sketch_examples(args.grammar_sketch_examples))
    for path in args.arc_hypothesis_evidence:
        if path.exists():
            inputs.append(str(path))
            output_rows.extend(normalize_arc_hypothesis_evidence(path))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fp:
        for row in output_rows:
            fp.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        write_summary(args.summary, output_rows, inputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
