#!/usr/bin/env python3
"""Summarize and compare numeric training certificate JSONL artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


CERTIFICATE_SCHEMA = "crlp.numeric_step_certificate.v1"
DISTRIBUTION_SCHEMA = "crlp.numeric_certificate_distribution.v1"
COMPARE_SCHEMA = "crlp.numeric_certificate_compare.v1"
BUDGET_SCHEMA = "crlp.numeric_certificate_budget.v1"

METRICS = [
    "attempted_learning_rate",
    "applied_learning_rate",
    "optimizer_loss",
    "loss_before",
    "loss_after",
    "loss_delta_abs",
    "max_abs_parameter_after",
    "max_abs_parameter_delta",
    "max_abs_logit_after",
    "max_abs_logit_delta",
    "selected_margin_after",
]


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def percentile(sorted_values: list[float], quantile: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = quantile * (len(sorted_values) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return sorted_values[lower]
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def metric_summary(values: list[float]) -> dict[str, Any]:
    values = sorted(values)
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": values[0],
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": values[-1],
    }


def safe_target_from_path(path: Path) -> str:
    name = path.name
    suffix = ".numeric_steps.jsonl"
    return name[: -len(suffix)] if name.endswith(suffix) else path.stem


def iter_certificate_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("schema") == CERTIFICATE_SCHEMA:
                rows.append(row)
    return rows


def summarize_certificate_file(
    target: str,
    path: Path,
    label: str,
    summary_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = iter_certificate_rows(path)
    reasons: Counter[str] = Counter()
    step_kinds: Counter[str] = Counter()
    metrics: dict[str, list[float]] = {name: [] for name in METRICS}
    committed = 0
    finite_before = 0
    finite_after = 0
    max_shrink_steps = 0

    for row in rows:
        reasons[str(row.get("rejection_reason", "unknown"))] += 1
        step_kinds[str(row.get("step_kind", "unknown"))] += 1
        if row.get("step_committed") is True:
            committed += 1
        if row.get("finite_before") is True:
            finite_before += 1
        if row.get("finite_after") is True:
            finite_after += 1
        shrink_steps = row.get("shrink_steps")
        if isinstance(shrink_steps, int) and shrink_steps > max_shrink_steps:
            max_shrink_steps = shrink_steps

        for metric in METRICS:
            if metric == "loss_delta_abs":
                before = finite_float(row.get("loss_before"))
                after = finite_float(row.get("loss_after"))
                value = abs(after - before) if before is not None and after is not None else None
            else:
                value = finite_float(row.get(metric))
            if value is not None:
                metrics[metric].append(value)

    metric_rows = {name: metric_summary(values) for name, values in metrics.items()}
    row: dict[str, Any] = {
        "schema": DISTRIBUTION_SCHEMA,
        "label": label,
        "target": target,
        "certificate_path": str(path),
        "row_count": len(rows),
        "committed_rows": committed,
        "rejected_rows": len(rows) - committed,
        "finite_before_rows": finite_before,
        "finite_after_rows": finite_after,
        "max_shrink_steps": max_shrink_steps,
        "rejection_reasons": dict(sorted(reasons.items())),
        "step_kinds": dict(sorted(step_kinds.items())),
        "metrics": metric_rows,
    }
    if summary_row is not None:
        elapsed_seconds = finite_float(summary_row.get("elapsed_seconds"))
        if elapsed_seconds is not None:
            row["experiment_elapsed_seconds"] = elapsed_seconds
        for source_field, output_field in (
            ("status", "experiment_status"),
            ("exit_code", "experiment_exit_code"),
            ("stdout_path", "experiment_stdout_path"),
            ("stderr_path", "experiment_stderr_path"),
        ):
            if source_field in summary_row:
                row[output_field] = summary_row[source_field]
    return row


def load_summary_targets(summary_path: Path, repo: Path) -> list[tuple[str, Path, dict[str, Any]]]:
    targets: list[tuple[str, Path, dict[str, Any]]] = []
    with summary_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cert_path = Path(str(row.get("certificate_path", "")))
            if not cert_path.is_absolute():
                cert_path = repo / cert_path
            target = str(row.get("target") or safe_target_from_path(cert_path))
            targets.append((target, cert_path, row))
    return targets


def load_inputs(args: argparse.Namespace, summary_path: Path | None, label: str) -> list[dict[str, Any]]:
    paths: list[tuple[str, Path, dict[str, Any] | None]] = []
    if summary_path is not None:
        paths.extend(load_summary_targets(summary_path, args.repo))
    for raw_path in args.certificate or []:
        path = Path(raw_path)
        if not path.is_absolute():
            path = args.repo / path
        paths.append((safe_target_from_path(path), path, None))
    return [
        summarize_certificate_file(target, path, label, summary_row)
        for target, path, summary_row in paths
    ]


def parse_thresholds(items: list[str] | None) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"threshold must be metric=value, got {item!r}")
        metric, value = item.split("=", 1)
        thresholds[metric] = float(value)
    return thresholds


def load_budget(path: Path | None, repo: Path) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    budget_path = path if path.is_absolute() else repo / path
    budgets: dict[str, dict[str, Any]] = {}
    with budget_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("schema") != BUDGET_SCHEMA:
                continue
            target = str(row.get("target", ""))
            if target:
                budgets[target] = row
    return budgets


def int_limit(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def audit_budgeted_distribution(
    row: dict[str, Any],
    budget: dict[str, Any],
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    target = row["target"]

    min_rows = int_limit(budget.get("min_rows"))
    if min_rows is not None and row["row_count"] < min_rows:
        violations.append(
            {
                "target": target,
                "field": "row_count",
                "observed": row["row_count"],
                "limit": min_rows,
                "budget_schema": BUDGET_SCHEMA,
            }
        )

    max_rejected = int_limit(budget.get("max_rejected"))
    if max_rejected is not None and row["rejected_rows"] > max_rejected:
        violations.append(
            {
                "target": target,
                "field": "rejected_rows",
                "observed": row["rejected_rows"],
                "limit": max_rejected,
                "budget_schema": BUDGET_SCHEMA,
            }
        )

    max_shrink_steps = int_limit(budget.get("max_shrink_steps"))
    if max_shrink_steps is not None and row["max_shrink_steps"] > max_shrink_steps:
        violations.append(
            {
                "target": target,
                "field": "max_shrink_steps",
                "observed": row["max_shrink_steps"],
                "limit": max_shrink_steps,
                "budget_schema": BUDGET_SCHEMA,
            }
        )

    if budget.get("require_all_finite_after") is True and row["finite_after_rows"] != row["row_count"]:
        violations.append(
            {
                "target": target,
                "field": "finite_after_rows",
                "observed": row["finite_after_rows"],
                "limit": row["row_count"],
                "budget_schema": BUDGET_SCHEMA,
            }
        )

    metric_budgets = budget.get("metrics", {})
    if isinstance(metric_budgets, dict):
        for metric, limits in metric_budgets.items():
            if not isinstance(limits, dict):
                continue
            observed_metric = row["metrics"].get(metric, {})
            p99_limit = finite_float(limits.get("p99_limit"))
            observed_p99 = observed_metric.get("p99")
            if p99_limit is not None and observed_p99 is not None and observed_p99 > p99_limit:
                violations.append(
                    {
                        "target": target,
                        "field": f"{metric}.p99",
                        "observed": observed_p99,
                        "limit": p99_limit,
                        "budget_schema": BUDGET_SCHEMA,
                    }
                )
            max_limit = finite_float(limits.get("max_limit"))
            observed_max = observed_metric.get("max")
            if max_limit is not None and observed_max is not None and observed_max > max_limit:
                violations.append(
                    {
                        "target": target,
                        "field": f"{metric}.max",
                        "observed": observed_max,
                        "limit": max_limit,
                        "budget_schema": BUDGET_SCHEMA,
                    }
                )

    return violations


def audit_rows(
    rows: list[dict[str, Any]],
    max_rejected: int | None,
    fail_on_reject: bool,
    min_rows: int,
    max_shrink_steps: int | None,
    max_p99: dict[str, float],
    max_max: dict[str, float],
    budgets: dict[str, dict[str, Any]] | None = None,
    allow_unbudgeted_targets: bool = False,
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for row in rows:
        target = row["target"]
        if budgets is not None:
            budget = budgets.get(target)
            if budget is None:
                if not allow_unbudgeted_targets:
                    violations.append(
                        {
                            "target": target,
                            "field": "budget",
                            "observed": "missing",
                            "limit": BUDGET_SCHEMA,
                        }
                    )
            else:
                violations.extend(audit_budgeted_distribution(row, budget))
        if row["row_count"] < min_rows:
            violations.append(
                {
                    "target": target,
                    "field": "row_count",
                    "observed": row["row_count"],
                    "limit": min_rows,
                }
            )
        if fail_on_reject and row["rejected_rows"] > 0:
            violations.append(
                {
                    "target": target,
                    "field": "rejected_rows",
                    "observed": row["rejected_rows"],
                    "limit": 0,
                }
            )
        if max_rejected is not None and row["rejected_rows"] > max_rejected:
            violations.append(
                {
                    "target": target,
                    "field": "rejected_rows",
                    "observed": row["rejected_rows"],
                    "limit": max_rejected,
                }
            )
        if max_shrink_steps is not None and row["max_shrink_steps"] > max_shrink_steps:
            violations.append(
                {
                    "target": target,
                    "field": "max_shrink_steps",
                    "observed": row["max_shrink_steps"],
                    "limit": max_shrink_steps,
                }
            )
        for metric, limit in max_p99.items():
            observed = row["metrics"].get(metric, {}).get("p99")
            if observed is not None and observed > limit:
                violations.append(
                    {
                        "target": target,
                        "field": f"{metric}.p99",
                        "observed": observed,
                        "limit": limit,
                    }
                )
        for metric, limit in max_max.items():
            observed = row["metrics"].get(metric, {}).get("max")
            if observed is not None and observed > limit:
                violations.append(
                    {
                        "target": target,
                        "field": f"{metric}.max",
                        "observed": observed,
                        "limit": limit,
                    }
                )
    return violations


def compare_rows(
    baseline_rows: list[dict[str, Any]],
    compare_rows_: list[dict[str, Any]],
    baseline_label: str,
    compare_label: str,
) -> list[dict[str, Any]]:
    by_target = {row["target"]: row for row in baseline_rows}
    other_by_target = {row["target"]: row for row in compare_rows_}
    rows: list[dict[str, Any]] = []
    for target in sorted(set(by_target) & set(other_by_target)):
        left = by_target[target]
        right = other_by_target[target]
        for metric in METRICS:
            left_metric = left["metrics"].get(metric, {})
            right_metric = right["metrics"].get(metric, {})
            left_p99 = left_metric.get("p99")
            right_p99 = right_metric.get("p99")
            left_max = left_metric.get("max")
            right_max = right_metric.get("max")
            if left_p99 is None or right_p99 is None:
                continue
            denom = max(abs(float(left_p99)), 1e-12)
            max_abs_delta = None
            max_rel_delta = None
            if left_max is not None and right_max is not None:
                max_abs_delta = right_max - left_max
                max_rel_delta = max_abs_delta / max(abs(float(left_max)), 1e-12)
            rows.append(
                {
                    "schema": COMPARE_SCHEMA,
                    "target": target,
                    "metric": metric,
                    "baseline_label": baseline_label,
                    "compare_label": compare_label,
                    "baseline_p99": left_p99,
                    "compare_p99": right_p99,
                    "p99_abs_delta": right_p99 - left_p99,
                    "p99_rel_delta": (right_p99 - left_p99) / denom,
                    "baseline_max": left_max,
                    "compare_max": right_max,
                    "max_abs_delta": max_abs_delta,
                    "max_rel_delta": max_rel_delta,
                    "baseline_rows": left["row_count"],
                    "compare_rows": right["row_count"],
                    "baseline_rejected": left["rejected_rows"],
                    "compare_rejected": right["rejected_rows"],
                }
            )
    return rows


def audit_budgeted_compare(
    rows: list[dict[str, Any]],
    budgets: dict[str, dict[str, Any]] | None,
    allow_unbudgeted_targets: bool,
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    if budgets is None:
        return violations
    for row in rows:
        target = str(row.get("target", ""))
        metric = str(row.get("metric", ""))
        budget = budgets.get(target)
        if budget is None:
            if not allow_unbudgeted_targets:
                violations.append(
                    {
                        "target": target,
                        "metric": metric,
                        "field": "compare_budget",
                        "observed": "missing",
                        "limit": BUDGET_SCHEMA,
                    }
                )
            continue
        compare_metrics = budget.get("compare_metrics", {})
        if not isinstance(compare_metrics, dict):
            continue
        limits = compare_metrics.get(metric)
        if not isinstance(limits, dict):
            continue

        checks = [
            ("p99_abs_delta", "p99_abs_delta_limit", True),
            ("p99_rel_delta", "p99_rel_delta_abs_limit", True),
            ("max_abs_delta", "max_abs_delta_limit", True),
            ("max_rel_delta", "max_rel_delta_abs_limit", True),
        ]
        for observed_field, limit_field, use_abs in checks:
            limit = finite_float(limits.get(limit_field))
            observed = finite_float(row.get(observed_field))
            if limit is None or observed is None:
                continue
            comparable = abs(observed) if use_abs else observed
            if comparable > limit:
                violations.append(
                    {
                        "target": target,
                        "metric": metric,
                        "field": observed_field,
                        "observed": observed,
                        "limit": limit,
                        "budget_schema": BUDGET_SCHEMA,
                    }
                )
    return violations


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--label", default="current")
    parser.add_argument("--certificate", action="append")
    parser.add_argument("--compare-summary", type=Path)
    parser.add_argument("--compare-label", default="compare")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=Path, help="Per-target numeric certificate budget JSONL")
    parser.add_argument("--allow-unbudgeted-targets", action="store_true")
    parser.add_argument("--fail-on-reject", action="store_true")
    parser.add_argument("--max-rejected", type=int)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--max-shrink-steps", type=int)
    parser.add_argument("--max-p99", action="append", default=[])
    parser.add_argument("--max-max", action="append", default=[])
    return parser.parse_args(argv)


def resolve(path: Path | None, repo: Path) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else repo / path


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.repo = args.repo.resolve()
    summary_path = resolve(args.summary, args.repo)
    compare_summary_path = resolve(args.compare_summary, args.repo)
    output_path = resolve(args.output, args.repo)
    if output_path is None:
        raise AssertionError("argparse requires output")

    rows = load_inputs(args, summary_path, args.label)
    if not rows:
        print("numeric-certificate-report: no certificate inputs", file=sys.stderr)
        return 1

    max_p99 = parse_thresholds(args.max_p99)
    max_max = parse_thresholds(args.max_max)
    budgets = load_budget(resolve(args.budget, args.repo), args.repo) if args.budget else None
    violations = audit_rows(
        rows,
        args.max_rejected,
        args.fail_on_reject,
        args.min_rows,
        args.max_shrink_steps,
        max_p99,
        max_max,
        budgets,
        args.allow_unbudgeted_targets,
    )

    output_rows: list[dict[str, Any]] = rows
    if compare_summary_path is not None:
        compare_inputs = load_inputs(args, compare_summary_path, args.compare_label)
        violations.extend(
            audit_rows(
                compare_inputs,
                args.max_rejected,
                args.fail_on_reject,
                args.min_rows,
                args.max_shrink_steps,
                max_p99,
                max_max,
                budgets,
                args.allow_unbudgeted_targets,
            )
        )
        comparison_rows = compare_rows(rows, compare_inputs, args.label, args.compare_label)
        violations.extend(
            audit_budgeted_compare(comparison_rows, budgets, args.allow_unbudgeted_targets)
        )
        output_rows = output_rows + compare_inputs + comparison_rows
    if violations:
        output_rows.append(
            {
                "schema": "crlp.numeric_certificate_audit_violations.v1",
                "label": args.label,
                "violation_count": len(violations),
                "violations": violations,
            }
        )

    write_jsonl(output_path, output_rows)
    print(
        f"numeric-certificate-report: wrote {output_path} "
        f"rows={len(output_rows)} violations={len(violations)}",
        flush=True,
    )
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
