#!/usr/bin/env python3
"""Derive per-target numeric certificate budgets from passing reports."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


DISTRIBUTION_SCHEMA = "crlp.numeric_certificate_distribution.v1"
COMPARE_SCHEMA = "crlp.numeric_certificate_compare.v1"
BUDGET_SCHEMA = "crlp.numeric_certificate_budget.v1"

DEFAULT_METRICS = [
    "loss_delta_abs",
    "max_abs_parameter_after",
    "max_abs_parameter_delta",
    "max_abs_logit_after",
    "max_abs_logit_delta",
]

METRIC_FLOORS = {
    "loss_delta_abs": 1e-6,
    "max_abs_parameter_after": 1e-5,
    "max_abs_parameter_delta": 1e-6,
    "max_abs_logit_after": 1e-5,
    "max_abs_logit_delta": 1e-6,
}


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def rounded(value: float) -> float:
    return float(f"{value:.12g}")


def limit_from_observed(value: float | None, multiplier: float, floor: float) -> float | None:
    if value is None:
        return None
    magnitude = abs(value)
    return rounded(max(magnitude * multiplier, magnitude + floor, floor))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def distribution_rows(path: Path) -> list[dict[str, Any]]:
    return [row for row in read_jsonl(path) if row.get("schema") == DISTRIBUTION_SCHEMA]


def compare_rows(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    return {
        (str(row.get("target", "")), str(row.get("metric", ""))): row
        for row in read_jsonl(path)
        if row.get("schema") == COMPARE_SCHEMA
    }


def metric_budget(
    metric: str,
    metric_row: dict[str, Any],
    multiplier: float,
) -> dict[str, Any]:
    floor = METRIC_FLOORS.get(metric, 1e-6)
    observed_p99 = finite_float(metric_row.get("p99"))
    observed_max = finite_float(metric_row.get("max"))
    return {
        "observed_p99": observed_p99,
        "p99_limit": limit_from_observed(observed_p99, multiplier, floor),
        "observed_max": observed_max,
        "max_limit": limit_from_observed(observed_max, multiplier, floor),
    }


def compare_budget(
    metric: str,
    row: dict[str, Any],
    multiplier: float,
    rel_floor: float,
) -> dict[str, Any]:
    abs_floor = METRIC_FLOORS.get(metric, 1e-6)
    p99_abs_delta = finite_float(row.get("p99_abs_delta"))
    p99_rel_delta = finite_float(row.get("p99_rel_delta"))
    max_abs_delta = finite_float(row.get("max_abs_delta"))
    max_rel_delta = finite_float(row.get("max_rel_delta"))
    return {
        "observed_p99_abs_delta": p99_abs_delta,
        "p99_abs_delta_limit": limit_from_observed(p99_abs_delta, multiplier, abs_floor),
        "observed_p99_rel_delta": p99_rel_delta,
        "p99_rel_delta_abs_limit": limit_from_observed(p99_rel_delta, multiplier, rel_floor),
        "observed_max_abs_delta": max_abs_delta,
        "max_abs_delta_limit": limit_from_observed(max_abs_delta, multiplier, abs_floor),
        "observed_max_rel_delta": max_rel_delta,
        "max_rel_delta_abs_limit": limit_from_observed(max_rel_delta, multiplier, rel_floor),
    }


def build_budget_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    compare_by_key = compare_rows(args.compare)
    rows: list[dict[str, Any]] = []
    for row in distribution_rows(args.report):
        target = str(row.get("target", ""))
        metrics = row.get("metrics", {})
        if not target or not isinstance(metrics, dict):
            continue

        metric_budgets: dict[str, Any] = {}
        compare_budgets: dict[str, Any] = {}
        for metric in args.metric:
            metric_row = metrics.get(metric, {})
            if isinstance(metric_row, dict) and metric_row.get("count", 0):
                metric_budgets[metric] = metric_budget(
                    metric,
                    metric_row,
                    args.metric_multiplier,
                )
            compare_row = compare_by_key.get((target, metric))
            if compare_row is not None:
                compare_budgets[metric] = compare_budget(
                    metric,
                    compare_row,
                    args.compare_multiplier,
                    args.compare_rel_floor,
                )

        source_row_count = int(row.get("row_count", 0) or 0)
        min_rows = int(math.floor(source_row_count * args.min_row_ratio))
        rows.append(
            {
                "schema": BUDGET_SCHEMA,
                "target": target,
                "source_report": str(args.report),
                "source_label": row.get("label"),
                "source_row_count": source_row_count,
                "min_rows": max(1, min_rows),
                "max_rejected": 0,
                "max_shrink_steps": 0,
                "require_all_finite_after": True,
                "metric_multiplier": args.metric_multiplier,
                "compare_multiplier": args.compare_multiplier,
                "metrics": metric_budgets,
                "compare_metrics": compare_budgets,
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", action="append", default=DEFAULT_METRICS)
    parser.add_argument("--metric-multiplier", type=float, default=1.25)
    parser.add_argument("--compare-multiplier", type=float, default=2.0)
    parser.add_argument("--compare-rel-floor", type=float, default=0.05)
    parser.add_argument("--min-row-ratio", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    rows = build_budget_rows(args)
    if not rows:
        print("numeric-certificate-budget: no distribution rows", file=sys.stderr)
        return 1
    write_jsonl(args.output, rows)
    print(f"numeric-certificate-budget: wrote {args.output} rows={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
