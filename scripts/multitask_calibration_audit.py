#!/usr/bin/env python3
"""Summarize multitask calibration rows from crlp.multitask.summary.v1 JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def weighted_mean(pairs: list[tuple[float, float]]) -> float:
    weight = sum(w for _v, w in pairs)
    if weight <= 0.0:
        return float("nan")
    return sum(v * w for v, w in pairs) / weight


def sample_stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def fmt_pct(value: float) -> str:
    return "-" if math.isnan(value) else f"{100.0 * value:.2f}%"


def fmt_float(value: float) -> str:
    return "-" if math.isnan(value) else f"{value:.4f}"


def row_eval_name(row: dict[str, Any]) -> str:
    if row.get("roadmap_item") == "R-MT.2" and row.get("arc_eval_task") == "all":
        return "arc_agi_ood_same_envelope"
    if row.get("roadmap_item") == "R-MT.2" and row.get("arc_ood_breakdown"):
        return f"arc_agi_ood_{row.get('arc_eval_task')}"
    return str(row.get("domain", "unknown"))


def row_scope(row: dict[str, Any]) -> str:
    if row.get("roadmap_item") == "R-MT.1":
        return "fixed_task"
    if row.get("roadmap_item") == "R-MT.2" and row.get("arc_eval_task") == "all":
        return "arc_ood_aggregate"
    if row.get("roadmap_item") == "R-MT.2" and row.get("arc_ood_breakdown"):
        return "arc_ood_task"
    return str(row.get("roadmap_item", "unknown"))


def calibration_groups(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("schema") != "crlp.multitask.summary.v1":
            continue
        if row.get("model_kind") not in {"shared", "single_task"}:
            continue
        key = (row_scope(row), row_eval_name(row), str(row["model_kind"]))
        grouped.setdefault(key, []).append(row)
    return grouped


def summarize_group(scope: str, eval_name: str, model_kind: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [float(row.get("sample_count", 0) or 0) for row in rows]
    correct_counts = [float(row.get("top1", 0.0)) * sample for row, sample in zip(rows, samples)]
    incorrect_counts = [max(0.0, sample - correct) for sample, correct in zip(samples, correct_counts)]
    summary = {
        "scope": scope,
        "eval_name": eval_name,
        "model_kind": model_kind,
        "seeds": len({int(row["seed"]) for row in rows}),
        "rows": len(rows),
        "samples": int(sum(samples)),
        "top1_mean": mean([float(row["top1"]) for row in rows]),
        "top1_stdev": sample_stdev([float(row["top1"]) for row in rows]),
        "nll_mean": mean([float(row["nll"]) for row in rows]),
        "nll_stdev": sample_stdev([float(row["nll"]) for row in rows]),
        "ece_mean": mean([float(row["ece"]) for row in rows]),
        "ece_stdev": sample_stdev([float(row["ece"]) for row in rows]),
        "confidence_mean": weighted_mean(
            [(float(row.get("confidence_mean", 0.0)), sample) for row, sample in zip(rows, samples)]
        ),
        "correct_confidence_mean": weighted_mean(
            [
                (float(row.get("correct_confidence_mean", 0.0)), correct)
                for row, correct in zip(rows, correct_counts)
            ]
        ),
        "incorrect_confidence_mean": weighted_mean(
            [
                (float(row.get("incorrect_confidence_mean", 0.0)), incorrect)
                for row, incorrect in zip(rows, incorrect_counts)
            ]
        ),
    }
    return summary


def pooled_bins(
    scope: str,
    eval_name: str,
    model_kind: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in rows:
        for bin_row in row.get("reliability_bins", []):
            key = (float(bin_row["low"]), float(bin_row["high"]))
            grouped.setdefault(key, []).append(bin_row)
    total_count = sum(int(bin_row["count"]) for bins in grouped.values() for bin_row in bins)
    out: list[dict[str, Any]] = []
    for (low, high), bins in sorted(grouped.items()):
        count = sum(int(bin_row["count"]) for bin_row in bins)
        confidence = weighted_mean(
            [(float(bin_row["confidence"]), float(bin_row["count"])) for bin_row in bins]
        )
        accuracy = weighted_mean(
            [(float(bin_row["accuracy"]), float(bin_row["count"])) for bin_row in bins]
        )
        gap = confidence - accuracy
        out.append(
            {
                "scope": scope,
                "eval_name": eval_name,
                "model_kind": model_kind,
                "low": low,
                "high": high,
                "count": count,
                "fraction": count / total_count if total_count else 0.0,
                "confidence": confidence,
                "accuracy": accuracy,
                "calibration_gap": gap,
                "ece_contribution": (count / total_count) * abs(gap) if total_count else 0.0,
            }
        )
    return out


def attach_pooled_ece(summary: list[dict[str, Any]], bins: list[dict[str, Any]]) -> None:
    bins_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in bins:
        key = (str(row["scope"]), str(row["eval_name"]), str(row["model_kind"]))
        bins_by_key.setdefault(key, []).append(row)
    for row in summary:
        key = (str(row["scope"]), str(row["eval_name"]), str(row["model_kind"]))
        row["pooled_ece"] = sum(float(bin_row["ece_contribution"]) for bin_row in bins_by_key.get(key, []))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def row_index(rows: list[dict[str, Any]], scope: str) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["eval_name"]), str(row["model_kind"])): row
        for row in rows
        if row["scope"] == scope
    }


def add_delta_table(lines: list[str], rows: list[dict[str, Any]], scope: str, title: str) -> None:
    index = row_index(rows, scope)
    eval_names = sorted({eval_name for eval_name, _kind in index})
    lines.extend(
        [
            f"## {title}",
            "",
            "| Evaluation | Shared ECE | Single ECE | Delta | Shared NLL | Single NLL | Delta | Shared conf. | Single conf. |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for eval_name in eval_names:
        shared = index.get((eval_name, "shared"))
        single = index.get((eval_name, "single_task"))
        if shared is None or single is None:
            continue
        ece_delta = float(shared["ece_mean"]) - float(single["ece_mean"])
        nll_delta = float(shared["nll_mean"]) - float(single["nll_mean"])
        lines.append(
            f"| `{eval_name}` | {fmt_float(float(shared['ece_mean']))} | "
            f"{fmt_float(float(single['ece_mean']))} | {ece_delta:+.4f} | "
            f"{fmt_float(float(shared['nll_mean']))} | {fmt_float(float(single['nll_mean']))} | "
            f"{nll_delta:+.4f} | {fmt_float(float(shared['confidence_mean']))} | "
            f"{fmt_float(float(single['confidence_mean']))} |"
        )
    lines.append("")


def add_reliability_table(
    lines: list[str],
    bins: list[dict[str, Any]],
    scope: str,
    eval_name: str,
) -> None:
    selected = [
        row
        for row in bins
        if row["scope"] == scope and row["eval_name"] == eval_name
    ]
    if not selected:
        return
    lines.extend(
        [
            f"### Reliability Bins: `{eval_name}`",
            "",
            "| Model | Bin | Count | Confidence | Accuracy | Gap | ECE contribution |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(selected, key=lambda item: (str(item["model_kind"]), float(item["low"]))):
        lines.append(
            f"| `{row['model_kind']}` | {float(row['low']):.1f}-{float(row['high']):.1f} | "
            f"{int(row['count'])} | {fmt_float(float(row['confidence']))} | "
            f"{fmt_float(float(row['accuracy']))} | {float(row['calibration_gap']):+.4f} | "
            f"{fmt_float(float(row['ece_contribution']))} |"
        )
    lines.append("")


def write_markdown(path: Path, summary: list[dict[str, Any]], bins: list[dict[str, Any]]) -> None:
    fixed = row_index(summary, "fixed_task")
    ood = row_index(summary, "arc_ood_aggregate")
    lines = [
        "# Multitask Calibration Audit",
        "",
        "Source: `raw.jsonl` rows with schema `crlp.multitask.summary.v1`.",
        "ECE means are seed means; reliability-bin tables pool counts across seeds.",
        "",
    ]
    add_delta_table(lines, summary, "fixed_task", "R-MT.5 Fixed-Task Calibration")
    add_delta_table(lines, summary, "arc_ood_aggregate", "R-MT.5 ARC OOD Calibration")

    arc_shared = fixed.get(("arc_agi", "shared"))
    arc_single = fixed.get(("arc_agi", "single_task"))
    ood_shared = ood.get(("arc_agi_ood_same_envelope", "shared"))
    ood_single = ood.get(("arc_agi_ood_same_envelope", "single_task"))
    lines.extend(["## Interpretation", ""])
    if arc_shared and arc_single:
        lines.append(
            "On fixed `0520fde7`, shared training still hurts top-1, but it cuts ARC NLL "
            f"from `{fmt_float(float(arc_single['nll_mean']))}` to `{fmt_float(float(arc_shared['nll_mean']))}`; "
            f"ECE is worse (`{fmt_float(float(arc_shared['ece_mean']))}` vs "
            f"`{fmt_float(float(arc_single['ece_mean']))}`)."
        )
    if ood_shared and ood_single:
        lines.append(
            "On the same-envelope ARC OOD aggregate, shared training improves both ranking and calibration: "
            f"NLL `{fmt_float(float(ood_shared['nll_mean']))}` vs "
            f"`{fmt_float(float(ood_single['nll_mean']))}`, ECE "
            f"`{fmt_float(float(ood_shared['ece_mean']))}` vs "
            f"`{fmt_float(float(ood_single['ece_mean']))}`."
        )
    lines.append(
        "For `prime_gap`, `grid_path`, and `delayed_claim`, ECE deltas are small enough that "
        "R-MT.5 should be framed as an ARC clarification rather than a broad calibration win."
    )
    lines.append("")

    add_reliability_table(lines, bins, "fixed_task", "arc_agi")
    add_reliability_table(lines, bins, "arc_ood_aggregate", "arc_agi_ood_same_envelope")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/2026-05-17-multitask-roadmap/raw.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/2026-05-17-multitask-roadmap"))
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    input_path = args.input if args.input.is_absolute() else repo / args.input
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_path)
    grouped = calibration_groups(rows)
    summary: list[dict[str, Any]] = []
    bins: list[dict[str, Any]] = []
    for (scope, eval_name, model_kind), group in sorted(grouped.items()):
        summary.append(summarize_group(scope, eval_name, model_kind, group))
        bins.extend(pooled_bins(scope, eval_name, model_kind, group))
    attach_pooled_ece(summary, bins)

    write_csv(output_dir / "rmt5_calibration_summary.csv", summary)
    write_csv(output_dir / "rmt5_reliability_bins.csv", bins)
    write_markdown(output_dir / "rmt5_calibration.md", summary, bins)
    print(f"wrote {output_dir / 'rmt5_calibration.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
