#!/usr/bin/env python3
"""Run the canonical prime-gate headline path across multiple seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path
from statistics import mean, stdev
from typing import Any


SCHEMA = "crlp.prime_gate_delta.seed_interval.v1"


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" in part:
            start_text, end_text = part.split(":", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            seeds.extend(range(start, end + step, step))
        else:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("expected at least one seed")
    return seeds


def run_command(command: list[str], log_path: Path) -> list[dict[str, Any]]:
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {log_path}")
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def find_first(rows: list[dict[str, Any]], **filters: Any) -> dict[str, Any]:
    for row in rows:
        if all(row.get(key) == value for key, value in filters.items()):
            return row
    raise RuntimeError(f"missing row with filters {filters}")


def run_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    seed_dir = args.output_dir / f"seed-{seed}"
    trace_dir = seed_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    v1 = seed_dir / "prime_v1.nnckpt"
    v2 = seed_dir / "prime_v2.nnckpt"
    learned_v1 = seed_dir / "prime_learned_v1.nnckpt"
    bootstrap_rules = seed_dir / "prime_rules.crlprule"
    expanded_rules = seed_dir / "prime_rules_v2.crlprule"
    trace = trace_dir / "prime_gap.jsonl"
    rule_trace = trace_dir / "prime_rule_search.jsonl"

    prime_gap_cmd = [
        str(args.prime_gap),
        str(args.trace_train_limit),
        str(args.trace_test_limit),
        str(args.supervised_epochs),
        str(args.rl_epochs),
        str(args.trace_max_train_states),
        str(args.trace_max_test_states),
        f"{args.reference_weight:.2f}",
        f"{args.entropy_weight:.2f}",
        str(seed),
        "--save",
        str(v1),
        "--save-rules",
        str(bootstrap_rules),
        "--save-learned-model",
        str(learned_v1),
        "--trace",
        str(trace),
    ]
    run_command(prime_gap_cmd, seed_dir / "prime_gap.log")

    rule_search_cmd = [
        str(args.prime_rule_search),
        "--candidate-only",
        "--unconditional-only",
        "--moduli-range",
        "3:255",
        "--max-rules",
        str(args.max_rules),
        "--proposer-seed",
        str(seed),
        "--trace",
        str(rule_trace),
        "--save",
        str(expanded_rules),
    ]
    rule_rows = run_command(rule_search_cmd, seed_dir / "prime_rule_search.log")

    replay_cmd = [
        str(args.prime_replay),
        str(trace),
        "--out",
        str(v2),
        str(args.replay_epochs),
        "--rules",
        str(expanded_rules),
    ]
    run_command(replay_cmd, seed_dir / "prime_replay.log")

    compare_cmd = [
        str(args.prime_compare),
        "--rules",
        str(expanded_rules),
        str(v1),
        str(v2),
    ]
    compare_rows = run_command(compare_cmd, seed_dir / "prime_compare.log")

    compare = [
        row
        for row in compare_rows
        if row.get("schema") == "crlp.compare.v1" and row.get("task") == "prime_gap"
    ]
    raw = find_first(compare, evaluation_mode="raw", feature_set="minimal")
    hand = find_first(compare, evaluation_mode="hand_wheel_model", feature_set="minimal")
    learned = find_first(compare, evaluation_mode="expanded_rule_model", feature_set="minimal")
    expanded = find_first(compare, evaluation_mode="expanded_rules")
    rule_summary = find_first(rule_rows, schema="crlp.rule_search.summary.v1", task="prime_gap")

    raw_top1 = float(raw["top1"])
    hand_top1 = float(hand["top1"])
    learned_top1 = float(learned["top1"])
    learned_top10 = float(learned["top10"])
    false_rejections = int(expanded["false_rejections"])
    sample_count = int(learned["sample_count"])

    return {
        "schema": SCHEMA,
        "report_version": 1,
        "row_type": "seed",
        "date": date.today().isoformat(),
        "task": "prime_gap",
        "seed": seed,
        "trace_train_limit": args.trace_train_limit,
        "trace_test_limit": args.trace_test_limit,
        "trace_max_train_states": args.trace_max_train_states,
        "trace_max_test_states": args.trace_max_test_states,
        "sample_count": sample_count,
        "raw_top1": raw_top1,
        "hand_top1": hand_top1,
        "learned_top1": learned_top1,
        "learned_top10": learned_top10,
        "hand_vs_raw_delta_pp": 100.0 * (hand_top1 - raw_top1),
        "learned_vs_raw_delta_pp": 100.0 * (learned_top1 - raw_top1),
        "learned_vs_hand_delta_pp": 100.0 * (learned_top1 - hand_top1),
        "false_rejections": false_rejections,
        "accepted_rules": int(expanded.get("accepted_rules", rule_summary.get("accepted_rules", 0))),
        "rule_set_hash": expanded.get("rule_set_hash", rule_summary.get("rule_set_hash")),
        "raw_model_hash": raw.get("model_hash"),
        "logs": {
            "prime_gap": str(seed_dir / "prime_gap.log"),
            "rule_search": str(seed_dir / "prime_rule_search.log"),
            "replay": str(seed_dir / "prime_replay.log"),
            "compare": str(seed_dir / "prime_compare.log"),
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def fmt_pp(value: float) -> str:
    return f"{value:+.2f}pp"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def values(key: str) -> list[float]:
        return [float(row[key]) for row in rows]

    def aggregate(key: str) -> dict[str, float]:
        vals = values(key)
        return {
            "mean": mean(vals),
            "stdev": stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }

    raw = aggregate("raw_top1")
    hand = aggregate("hand_top1")
    learned = aggregate("learned_top1")
    delta = aggregate("learned_vs_hand_delta_pp")
    seed_values = [int(row["seed"]) for row in rows]
    false_rejections_max = max(int(row["false_rejections"]) for row in rows)
    top10_min = min(float(row["learned_top10"]) for row in rows)
    summary = (
        f"{len(rows)} seeds: learned {fmt_pct(learned['mean'])} +/- "
        f"{100.0 * learned['stdev']:.2f}pp vs hand {fmt_pct(hand['mean'])} +/- "
        f"{100.0 * hand['stdev']:.2f}pp and raw {fmt_pct(raw['mean'])} +/- "
        f"{100.0 * raw['stdev']:.2f}pp; min learned-vs-hand delta "
        f"{fmt_pp(delta['min'])}, learned top-10 min {fmt_pct(top10_min)}, "
        f"false rejections max={false_rejections_max}."
    )
    return {
        "schema": SCHEMA,
        "report_version": 1,
        "row_type": "summary",
        "date": date.today().isoformat(),
        "task": "prime_gap",
        "seed_count": len(rows),
        "seed_start": min(seed_values),
        "seed_end": max(seed_values),
        "seeds": seed_values,
        "raw_top1": raw,
        "hand_top1": hand,
        "learned_top1": learned,
        "learned_vs_hand_delta_pp": delta,
        "learned_top10_min": top10_min,
        "false_rejections_max": false_rejections_max,
        "sample_count_min": min(int(row["sample_count"]) for row in rows),
        "summary": summary,
    }


def write_summary_md(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# Prime Gate Delta Seed Audit",
        "",
        summary["summary"],
        "",
        "| Seed | Raw top-1 | Hand top-1 | Learned top-1 | Learned top-10 | Delta vs hand | False rejections |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    fmt_pct(float(row["raw_top1"])),
                    fmt_pct(float(row["hand_top1"])),
                    fmt_pct(float(row["learned_top1"])),
                    fmt_pct(float(row["learned_top10"])),
                    fmt_pp(float(row["learned_vs_hand_delta_pp"])),
                    str(row["false_rejections"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime-gap", type=Path, required=True)
    parser.add_argument("--prime-rule-search", type=Path, required=True)
    parser.add_argument("--prime-replay", type=Path, required=True)
    parser.add_argument("--prime-compare", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="20240511:20240515")
    parser.add_argument("--trace-train-limit", type=int, default=100000)
    parser.add_argument("--trace-test-limit", type=int, default=200000)
    parser.add_argument("--supervised-epochs", type=int, default=1)
    parser.add_argument("--rl-epochs", type=int, default=0)
    parser.add_argument("--trace-max-train-states", type=int, default=4000)
    parser.add_argument("--trace-max-test-states", type=int, default=1000)
    parser.add_argument("--reference-weight", type=float, default=1.20)
    parser.add_argument("--entropy-weight", type=float, default=0.12)
    parser.add_argument("--max-rules", type=int, default=64)
    parser.add_argument("--replay-epochs", type=int, default=2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_seed(args, seed) for seed in parse_seeds(args.seeds)]
    summary = summarize(rows)
    write_jsonl(args.output_dir / "seed_interval_summary.jsonl", [*rows, summary])
    write_summary_md(args.output_dir / "summary.md", rows, summary)
    print(summary["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
