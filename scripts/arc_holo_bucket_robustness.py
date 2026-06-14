#!/usr/bin/env python3
"""Run ARC Holo bucket robustness sweeps.

The sweep treats Holo bucket routing as a proposal-order intervention only. It
keeps raw run logs, then summarizes `trace_task_prior` versus
`trace_holo_bucket` at matched task order, budget grid, min-budget floor, and
review-window size.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


BASELINE_STRATEGY = "trace_task_prior"
BUCKET_STRATEGY = "trace_holo_bucket"
SUMMARY_SCHEMA = "crlp.arc_holo_bucket_robustness.run.v1"
AGG_SCHEMA = "crlp.arc_holo_bucket_robustness.aggregate.v1"


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_order(value: str) -> tuple[str, int | None]:
    if value == "original":
        return value, None
    if value.startswith("shuffle:"):
        seed_text = value.split(":", 1)[1]
        try:
            return "shuffle", int(seed_text, 10)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid shuffle seed: {value}") from exc
    raise argparse.ArgumentTypeError(
        f"unknown order {value!r}; use original or shuffle:<seed>"
    )


def review_slot_arg(value: str) -> tuple[str, int]:
    if value == "auto":
        return value, 0
    try:
        parsed = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid review slot value: {value}"
        ) from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("review slots must be non-negative")
    return value, parsed


def parse_jsonl(path: Path) -> tuple[dict[int, dict[str, dict]], dict[int, dict[str, int]]]:
    summaries: dict[int, dict[str, dict]] = {}
    diagnostics: dict[int, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            schema = row.get("schema")
            if schema == "crlp.program_proposer.summary.v1":
                strategy = row.get("strategy")
                if strategy not in {BASELINE_STRATEGY, BUCKET_STRATEGY}:
                    continue
                budget = int(row["budget"])
                summaries.setdefault(budget, {})[strategy] = row
            elif schema == "crlp.arc_holo_bucket_diagnostic.v1":
                budget = int(row["budget"])
                diag = diagnostics.setdefault(
                    budget,
                    {
                        "diagnostic_rows": 0,
                        "crossed_into_budget": 0,
                        "crossed_out_of_budget": 0,
                        "max_holo_pressure_updates": 0,
                    },
                )
                diag["diagnostic_rows"] += 1
                if row.get("crossed_into_budget"):
                    diag["crossed_into_budget"] += 1
                if row.get("crossed_out_of_budget"):
                    diag["crossed_out_of_budget"] += 1
                diag["max_holo_pressure_updates"] = max(
                    diag["max_holo_pressure_updates"],
                    int(row.get("holo_pressure_updates", 0)),
                )
    return summaries, diagnostics


def run_condition(
    args: argparse.Namespace,
    raw_dir: Path,
    grid: str,
    order_label: str,
    order_seed: int | None,
    min_budget: int,
    review_label: str,
    review_slots: int,
) -> list[dict]:
    file_stem = (
        f"{grid}_{order_label}"
        f"{'' if order_seed is None else '_' + str(order_seed)}"
        f"_min{min_budget}_review{review_label}"
    )
    raw_path = raw_dir / f"{file_stem}.jsonl"
    stderr_path = raw_dir / f"{file_stem}.stderr"

    command = [
        args.arc_agi,
        "--task-dir",
        args.task_dir,
        "--max-tasks",
        str(args.max_tasks),
        "--loader-summary-only",
        "--induction",
        "--min-induction-evidence",
        str(args.min_induction_evidence),
        "--unique-task-gate",
        "--arc-holo-residual-diagnostics",
        "--arc-holo-bucket-min-budget",
        str(min_budget),
        "--arc-holo-bucket-review-slots",
        str(review_slots),
    ]
    if grid == "tight":
        command.append("--program-budget-tight-grid")
    elif grid == "standard":
        command.append("--program-budget-grid")
    else:
        raise ValueError(f"unknown grid: {grid}")
    if order_label == "shuffle":
        command.extend(["--shuffle", "--seed", str(order_seed)])
    elif order_label == "original" and args.seed is not None:
        command.extend(["--seed", str(args.seed)])

    with raw_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"arc_agi failed for {file_stem} with exit code {completed.returncode}; "
            f"stderr: {stderr_path}"
        )

    summaries, diagnostics = parse_jsonl(raw_path)
    rows: list[dict] = []
    for budget in sorted(summaries):
        by_strategy = summaries[budget]
        if BASELINE_STRATEGY not in by_strategy or BUCKET_STRATEGY not in by_strategy:
            continue
        baseline = by_strategy[BASELINE_STRATEGY]
        bucket = by_strategy[BUCKET_STRATEGY]
        diag = diagnostics.get(
            budget,
            {
                "diagnostic_rows": 0,
                "crossed_into_budget": 0,
                "crossed_out_of_budget": 0,
                "max_holo_pressure_updates": 0,
            },
        )
        row = {
            "schema": SUMMARY_SCHEMA,
            "grid": grid,
            "order": order_label,
            "order_seed": order_seed,
            "max_tasks": args.max_tasks,
            "min_budget": min_budget,
            "review_slots": review_label,
            "budget": budget,
            "baseline_utility": float(baseline["utility"]),
            "bucket_utility": float(bucket["utility"]),
            "utility_delta": float(bucket["utility"]) - float(baseline["utility"]),
            "baseline_certified": int(baseline["certified_programs"]),
            "bucket_certified": int(bucket["certified_programs"]),
            "certified_delta": int(bucket["certified_programs"])
            - int(baseline["certified_programs"]),
            "baseline_accepted": int(baseline["accepted_programs"]),
            "bucket_accepted": int(bucket["accepted_programs"]),
            "accepted_delta": int(bucket["accepted_programs"])
            - int(baseline["accepted_programs"]),
            "baseline_program_set_hash": baseline.get("program_set_hash", ""),
            "bucket_program_set_hash": bucket.get("program_set_hash", ""),
            "diagnostic_rows": diag["diagnostic_rows"],
            "crossed_into_budget": diag["crossed_into_budget"],
            "crossed_out_of_budget": diag["crossed_out_of_budget"],
            "max_holo_pressure_updates": diag["max_holo_pressure_updates"],
            "raw_log": str(raw_path),
            "stderr_log": str(stderr_path),
        }
        rows.append(row)
    if not rows:
        raise RuntimeError(f"{raw_path} did not contain matched strategy summaries")
    return rows


def aggregate_rows(rows: list[dict]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["grid"], row["min_budget"], row["review_slots"], row["budget"])
        buckets.setdefault(key, []).append(row)

    aggregates = []
    for key, group in sorted(buckets.items()):
        grid, min_budget, review_slots, budget = key
        deltas = [float(row["utility_delta"]) for row in group]
        certified_deltas = [int(row["certified_delta"]) for row in group]
        crossed_in = sum(int(row["crossed_into_budget"]) for row in group)
        crossed_out = sum(int(row["crossed_out_of_budget"]) for row in group)
        aggregates.append(
            {
                "schema": AGG_SCHEMA,
                "grid": grid,
                "min_budget": min_budget,
                "review_slots": review_slots,
                "budget": budget,
                "run_count": len(group),
                "median_utility_delta": statistics.median(deltas),
                "min_utility_delta": min(deltas),
                "max_utility_delta": max(deltas),
                "positive_runs": sum(1 for delta in deltas if delta > 0.0),
                "negative_runs": sum(1 for delta in deltas if delta < 0.0),
                "zero_runs": sum(1 for delta in deltas if delta == 0.0),
                "median_certified_delta": statistics.median(certified_deltas),
                "crossed_into_budget": crossed_in,
                "crossed_out_of_budget": crossed_out,
                "passes_no_regression": all(delta >= 0.0 for delta in deltas)
                and crossed_out == 0,
            }
        )
    return aggregates


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arc-agi", default="build/float/arc_agi")
    parser.add_argument("--task-dir", default="build/float/arc_agi_public/data/training")
    parser.add_argument("--out-dir", default="build/float/arc_holo_bucket_robustness")
    parser.add_argument("--max-tasks", type=int, default=400)
    parser.add_argument("--min-induction-evidence", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--grids", default="tight,standard")
    parser.add_argument(
        "--orders",
        default="original,shuffle:20240516",
        help="comma list of original or shuffle:<seed>",
    )
    parser.add_argument("--min-budgets", default="2")
    parser.add_argument("--review-slots", default="auto,1,2,4")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    grids = split_csv(args.grids)
    orders = [parse_order(value) for value in split_csv(args.orders)]
    min_budgets = [int(value) for value in split_csv(args.min_budgets)]
    review_slots = [review_slot_arg(value) for value in split_csv(args.review_slots)]

    all_rows: list[dict] = []
    for grid in grids:
        if grid not in {"tight", "standard"}:
            raise SystemExit(f"unsupported grid {grid!r}")
        for order_label, order_seed in orders:
            for min_budget in min_budgets:
                if min_budget < 0:
                    raise SystemExit("min budgets must be non-negative")
                for review_label, review_count in review_slots:
                    print(
                        "arc_holo_bucket_robustness "
                        f"grid={grid} order={order_label}"
                        f"{'' if order_seed is None else ':' + str(order_seed)} "
                        f"min_budget={min_budget} review_slots={review_label}",
                        file=sys.stderr,
                        flush=True,
                    )
                    all_rows.extend(
                        run_condition(
                            args,
                            raw_dir,
                            grid,
                            order_label,
                            order_seed,
                            min_budget,
                            review_label,
                            review_count,
                        )
                    )

    aggregate = aggregate_rows(all_rows)
    write_jsonl(out_dir / "summary.jsonl", all_rows)
    write_jsonl(out_dir / "aggregate.jsonl", aggregate)
    write_csv(out_dir / "summary.csv", all_rows)
    write_csv(out_dir / "aggregate.csv", aggregate)
    print(f"wrote {len(all_rows)} summary rows and {len(aggregate)} aggregate rows to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
