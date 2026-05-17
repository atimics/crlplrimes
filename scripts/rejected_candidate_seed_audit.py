#!/usr/bin/env python3
"""Run rejected-candidate learning across seed intervals and summarize stability."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any


METRICS = (
    "accepted_rules",
    "unsafe_rules",
    "validation_false_rejections",
    "validation_utility",
)
FAMILY_METRICS = (
    "eval_rows",
    "eval_utility",
    "eval_false_rejections",
    "selected_rows",
    "selected_utility",
    "selected_false_rejections",
)
PAPER_STRATEGIES = ("model_score", "reason_prior", "trained_residual")


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            bits = [int(item) for item in part.split(":")]
            if len(bits) not in (2, 3):
                raise argparse.ArgumentTypeError(f"bad range: {part}")
            start, stop = bits[:2]
            step = bits[2] if len(bits) == 3 else 1
            if step <= 0:
                raise argparse.ArgumentTypeError(f"range step must be positive: {part}")
            values.extend(range(start, stop + 1, step))
        else:
            values.append(int(part))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line.startswith("{"):
                rows.append(json.loads(line))
    return rows


def run_seed(args: argparse.Namespace, seed: int) -> Path:
    seed_dir = args.output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(args.script),
        "--binary",
        str(args.binary),
        "--modular-binary",
        str(args.modular_binary),
        "--delayed-claim-binary",
        str(args.delayed_claim_binary),
        "--output-dir",
        str(seed_dir),
        "--proposer-seed",
        str(seed),
        "--modular-selection-seed",
        str(seed),
        "--delayed-claim-proposer-seed",
        str(seed),
        "--budgets",
        *[str(budget) for budget in args.budgets],
    ]
    if args.skip_prime:
        command.append("--skip-prime")
    if args.skip_modular:
        command.append("--skip-modular")
    if args.skip_delayed_claim:
        command.append("--skip-delayed-claim")
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    (seed_dir / "seed_audit.stdout").write_text(
        result.stdout + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"seed {seed} failed; see {seed_dir / 'seed_audit.stdout'}")
    return seed_dir / "reason_learning.jsonl"


def seed_reason_family_path(reason_learning_path: Path) -> Path:
    return reason_learning_path.parent / "reason_family_sweep.jsonl"


def interval_rows(rows: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["task"]), str(row["strategy"]), int(row["budget"]))].append(row)

    out: list[dict[str, Any]] = []
    for (task, strategy, budget), group in sorted(groups.items()):
        metric_summary: dict[str, dict[str, float]] = {}
        for metric in METRICS:
            values = [float(row.get(metric, 0.0) or 0.0) for row in group]
            metric_summary[metric] = {
                "mean": statistics.fmean(values),
                "min": min(values),
                "max": max(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        out.append(
            {
                "schema": "crlp.rejected_candidate_learning.seed_interval.v1",
                "report_version": 1,
                "date": date.today().isoformat(),
                "experiment": "rejected_candidate_reason_learning",
                "task": task,
                "strategy": strategy,
                "budget": budget,
                "seeds": seeds,
                "seed_start": min(seeds),
                "seed_end": max(seeds),
                "seed_count": len(group),
                "accepted_rules": metric_summary["accepted_rules"],
                "unsafe_rules": metric_summary["unsafe_rules"],
                "validation_false_rejections": metric_summary[
                    "validation_false_rejections"
                ],
                "validation_utility": metric_summary["validation_utility"],
            }
        )
    return out


def reason_family_interval_rows(rows: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["task"]),
                str(row["strategy"]),
                int(row["budget"]),
                str(row["reason_family"]),
                str(row["split_key_name"]),
                int(row["split_value"]),
            )
        ].append(row)

    out: list[dict[str, Any]] = []
    for (task, strategy, budget, family, split_key_name, split_value), group in sorted(groups.items()):
        metric_summary: dict[str, dict[str, float]] = {}
        for metric in FAMILY_METRICS:
            values = [float(row.get(metric, 0.0) or 0.0) for row in group]
            metric_summary[metric] = {
                "mean": statistics.fmean(values),
                "min": min(values),
                "max": max(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        out.append(
            {
                "schema": "crlp.rejected_candidate_learning.reason_family_seed_interval.v1",
                "report_version": 1,
                "date": date.today().isoformat(),
                "experiment": "rejected_candidate_reason_learning",
                "task": task,
                "strategy": strategy,
                "budget": budget,
                "reason_family": family,
                "split_key_name": split_key_name,
                "split_value": split_value,
                "seeds": seeds,
                "seed_start": min(seeds),
                "seed_end": max(seeds),
                "seed_count": len(group),
                **metric_summary,
            }
        )
    return out


def metric_cell(row: dict[str, Any]) -> str:
    return (
        f"{row['accepted_rules']['mean']:.1f}"
        f"/{row['unsafe_rules']['mean']:.1f}"
        f"/{row['validation_false_rejections']['mean']:.1f}"
        f"/{row['validation_utility']['mean']:.1f}"
        f" [{row['validation_utility']['min']:.1f},{row['validation_utility']['max']:.1f}]"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True) + "\n")


def write_paper_csv(path: Path, intervals: list[dict[str, Any]]) -> None:
    by_key = {
        (row["task"], row["strategy"], int(row["budget"])): row
        for row in intervals
        if row["strategy"] in PAPER_STRATEGIES
    }
    tasks = sorted({row["task"] for row in intervals})
    budgets = sorted({int(row["budget"]) for row in intervals})
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "task",
                "budget",
                "baseline_candidate_score",
                "reason_prior",
                "trained_residual",
            ],
        )
        writer.writeheader()
        for task in tasks:
            for budget in budgets:
                keys = [(task, strategy, budget) for strategy in PAPER_STRATEGIES]
                if not all(key in by_key for key in keys):
                    continue
                writer.writerow(
                    {
                        "task": task,
                        "budget": budget,
                        "baseline_candidate_score": metric_cell(by_key[keys[0]]),
                        "reason_prior": metric_cell(by_key[keys[1]]),
                        "trained_residual": metric_cell(by_key[keys[2]]),
                    }
                )


def write_summary(path: Path, intervals: list[dict[str, Any]], seeds: list[int]) -> None:
    by_key = {
        (row["task"], row["strategy"], int(row["budget"])): row
        for row in intervals
    }
    tasks = sorted({row["task"] for row in intervals})
    budgets = sorted({int(row["budget"]) for row in intervals})
    lines = [
        "# Rejected-Candidate Seed Audit",
        "",
        f"Date: {date.today().isoformat()}",
        f"Seeds: {', '.join(str(seed) for seed in seeds)}",
        "",
        "Cells are mean accepted / unsafe / validation false rejections / validation utility, "
        "with the utility interval in brackets.",
        "",
        "| Task | Budget | Baseline candidate score | Reason prior | Trained residual |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for task in tasks:
        for budget in budgets:
            keys = [(task, strategy, budget) for strategy in PAPER_STRATEGIES]
            if not all(key in by_key for key in keys):
                continue
            lines.append(
                "| {task} | {budget} | {baseline} | {prior} | {residual} |".format(
                    task=task,
                    budget=budget,
                    baseline=metric_cell(by_key[keys[0]]),
                    prior=metric_cell(by_key[keys[1]]),
                    residual=metric_cell(by_key[keys[2]]),
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, default=Path("scripts/rejected_candidate_reason_learning.py"))
    parser.add_argument("--binary", type=Path, default=Path("build/float/prime_rule_search"))
    parser.add_argument("--modular-binary", type=Path, default=Path("build/float/modular_gate_search"))
    parser.add_argument("--delayed-claim-binary", type=Path, default=Path("build/float/delayed_claim"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/float/rejected_candidate_seed_audit"))
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("20240511:20240513"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--skip-prime", action="store_true")
    parser.add_argument("--skip-modular", action="store_true")
    parser.add_argument("--skip-delayed-claim", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    all_family_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        path = run_seed(args, seed)
        for row in read_jsonl(path):
            row["selection_seed"] = seed
            all_rows.append(row)
        family_path = seed_reason_family_path(path)
        if family_path.exists():
            for row in read_jsonl(family_path):
                row["selection_seed"] = seed
                all_family_rows.append(row)

    raw_path = args.output_dir / "seed_rows.jsonl"
    interval_path = args.output_dir / "seed_interval_summary.jsonl"
    family_raw_path = args.output_dir / "reason_family_seed_rows.jsonl"
    family_interval_path = args.output_dir / "reason_family_seed_interval.jsonl"
    csv_path = args.output_dir / "paper_table.csv"
    summary_path = args.output_dir / "summary.md"
    intervals = interval_rows(all_rows, args.seeds)
    family_intervals = reason_family_interval_rows(all_family_rows, args.seeds)
    write_jsonl(raw_path, all_rows)
    write_jsonl(interval_path, intervals)
    write_jsonl(family_raw_path, all_family_rows)
    write_jsonl(family_interval_path, family_intervals)
    write_paper_csv(csv_path, intervals)
    write_summary(summary_path, intervals, args.seeds)
    print(f"wrote {raw_path}")
    print(f"wrote {interval_path}")
    print(f"wrote {family_raw_path}")
    print(f"wrote {family_interval_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
