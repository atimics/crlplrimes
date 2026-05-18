#!/usr/bin/env python3
"""Render the Signal flight result summary from generated JSONL artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_last_jsonl(path: Path, schema: str | None = None) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            row = json.loads(line)
            if schema is None or row.get("schema") == schema:
                last = row
    return last


def pct(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{100.0 * float(value):.2f}%"
    return "n/a"


def num(value: Any, digits: int = 5) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return "n/a"


def short_hash(value: Any) -> str:
    text = str(value or "")
    return text[:12] if text else "n/a"


def render(offline: dict[str, Any] | None, shadow: dict[str, Any] | None, live: dict[str, Any] | None) -> str:
    lines = [
        "# Signal Flight Evidence Summary",
        "",
        "Generated from Signal result JSONL artifacts.",
        "",
    ]

    if offline is None:
        lines.extend(["## Offline Replay", "", "No offline replay summary row is present.", ""])
    else:
        lines.extend(
            [
                "## Offline Replay",
                "",
                f"Source: `{offline.get('source_trace', 'build/float/signal_flight_experiment/summary.jsonl')}`.",
                "",
                "| metric | neural ranker | no-safety baseline |",
                "| --- | ---: | ---: |",
                f"| top-1 | {pct(offline.get('top1'))} | {pct(offline.get('baseline_top1'))} |",
                f"| top-3 | {pct(offline.get('top3'))} | n/a |",
                f"| selected utility | {num(offline.get('selected_utility'))} | {num(offline.get('baseline_selected_utility'))} |",
                f"| regret | {num(offline.get('regret'))} | {num(offline.get('baseline_regret'))} |",
                f"| damage rate | {pct(offline.get('selected_damage_rate'))} | {pct(offline.get('baseline_damage_rate'))} |",
                f"| death rate | {pct(offline.get('selected_death_rate'))} | {pct(offline.get('baseline_death_rate'))} |",
                "",
            ]
        )

    if shadow is not None:
        lines.extend(
            [
                "## Live Shadow Evidence",
                "",
                "| metric | value |",
                "| --- | ---: |",
                f"| shadow frames | {shadow.get('shadow_frames', 'n/a')} |",
                f"| disagreement rate | {pct(shadow.get('shadow_disagreement_rate'))} |",
                f"| inference p95 ms | {num(shadow.get('inference_ms_p95'), 3)} |",
                "",
            ]
        )

    if live is None:
        lines.extend(["## Live Neural-Active Evidence", "", "No live neural-active summary row is present.", ""])
    else:
        lines.extend(
            [
                "## Live Neural-Active Evidence",
                "",
                f"Log: `{live.get('log_path', 'n/a')}`.",
                "",
                "| metric | value |",
                "| --- | ---: |",
                f"| active rows | {live.get('active_rows', 'n/a')} |",
                f"| window active rows | {live.get('window_active_rows', 'n/a')} |",
                f"| brain mode | `{live.get('latest_brain_mode', 'n/a')}` |",
                f"| active source | `{live.get('latest_active_source', 'n/a')}` |",
                f"| checkpoint | `{short_hash(live.get('checkpoint_hash'))}` |",
                f"| inference p95 ms | {num(live.get('inference_ms_p95'), 3)} |",
                f"| replay-ready rows | {live.get('replay_ready_rows', 'n/a')} |",
                f"| damage events | {live.get('damage_events', 'n/a')} |",
                f"| death events | {live.get('death_events', 'n/a')} |",
                "",
            ]
        )

    lines.extend(
        [
            "## Claim Boundary",
            "",
            "Signal now has offline replay evidence, live shadow evidence, and a guarded "
            "`neural_active` live pilot artifact when the live-active row is present. "
            "These artifacts do not yet support a comparative active-control performance "
            "claim versus the heuristic baseline.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", type=Path, default=Path("results/2026-05-17-signal-flight/summary.jsonl"))
    parser.add_argument("--shadow", type=Path, default=Path("results/2026-05-17-signal-flight/shadow_replay_audit.jsonl"))
    parser.add_argument("--live-active", type=Path, default=Path("results/2026-05-17-signal-flight/live_active_summary.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/2026-05-17-signal-flight/summary.md"))
    args = parser.parse_args()

    text = render(
        read_last_jsonl(args.offline, "crlp.signal_flight.summary.v1"),
        read_last_jsonl(args.shadow, "crlp.signal_shadow_replay_audit.v1"),
        read_last_jsonl(args.live_active, "crlp.signal_live_active_summary.v1"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
