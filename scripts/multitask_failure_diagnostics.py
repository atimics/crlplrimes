#!/usr/bin/env python3
"""Run focused diagnostics for ARC multitask scorer failure modes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_SEEDS = "20240516,20240517,20240518"
ORDER_SWEEP = ["uniform_random", "round_robin", "arc_heavy", "prime_light"]
LR_SWEEP = ["0.0015", "0.003", "0.006", "0.012"]
EPOCH_SWEEP = [1, 2, 4, 8, 16]


def parse_seeds(text: str) -> list[int]:
    seeds = [int(part.strip(), 10) for part in text.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("expected at least one seed")
    return seeds


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def sample_stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def fmt_pct(value: float) -> str:
    return "-" if math.isnan(value) else f"{100.0 * value:.2f}%"


def fmt_float(value: float) -> str:
    return "-" if math.isnan(value) else f"{value:.4f}"


def run_command(command: list[str], cwd: Path, log_path: Path) -> list[dict[str, Any]]:
    print(f"running {' '.join(command)}", flush=True)
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise subprocess.CalledProcessError(proc.returncode, command)
    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        if not line.startswith("{"):
            continue
        rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
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


def color_count(row: dict[str, Any], key: str, color: int) -> float:
    values = row.get(key)
    if not isinstance(values, list) or color >= len(values):
        return float("nan")
    return float(values[color])


def dominant_color_count(row: dict[str, Any]) -> float:
    values = row.get("arc_fixed_predicted_color_count")
    if not isinstance(values, list) or not values:
        return float("nan")
    return float(max(values))


def summarize_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("schema") != "crlp.multitask.summary.v1" or row.get("domain") != "arc_agi":
            continue
        grouped.setdefault((str(row["diagnostic_group"]), str(row["diagnostic_name"])), []).append(row)

    output: list[dict[str, Any]] = []
    for (group, name), items in sorted(grouped.items()):
        top1 = [float(row["top1"]) for row in items]
        cell_acc = [float(row.get("arc_fixed_cell_accuracy", float("nan"))) for row in items]
        nll = [float(row["nll"]) for row in items]
        fixed_nll = [float(row.get("arc_fixed_nll", float("nan"))) for row in items]
        ece = [float(row["ece"]) for row in items]
        updates = [float(row["train_updates"]) for row in items]
        pred_zero = [color_count(row, "arc_fixed_predicted_color_count", 0) for row in items]
        pred_two = [color_count(row, "arc_fixed_predicted_color_count", 2) for row in items]
        correct_zero = [color_count(row, "arc_fixed_correct_target_color_count", 0) for row in items]
        correct_two = [color_count(row, "arc_fixed_correct_target_color_count", 2) for row in items]
        dominant = [dominant_color_count(row) for row in items]
        grid_exact = [1.0 if row.get("arc_fixed_grid_exact") else 0.0 for row in items]
        hashes = {str(row.get("arc_fixed_prediction_hash", "")) for row in items}
        first = items[0]
        output.append(
            {
                "diagnostic_group": group,
                "diagnostic_name": name,
                "rows": len(items),
                "seeds": len({int(row["seed"]) for row in items}),
                "hidden": first.get("hidden", ""),
                "learning_rate": first.get("learning_rate", ""),
                "training_order": first.get("training_order", ""),
                "epochs": first.get("epochs", ""),
                "top1_mean": mean(top1),
                "top1_stdev": sample_stdev(top1),
                "arc_fixed_cell_accuracy_mean": mean(cell_acc),
                "arc_fixed_cell_accuracy_stdev": sample_stdev(cell_acc),
                "grid_exact_rate": mean(grid_exact),
                "nll_mean": mean(nll),
                "arc_fixed_nll_mean": mean(fixed_nll),
                "ece_mean": mean(ece),
                "train_updates_mean": mean(updates),
                "predicted_zero_cells_mean": mean(pred_zero),
                "predicted_two_cells_mean": mean(pred_two),
                "correct_zero_cells_mean": mean(correct_zero),
                "correct_two_cells_mean": mean(correct_two),
                "dominant_predicted_color_cells_mean": mean(dominant),
                "prediction_hashes": len(hashes),
            }
        )
    return output


def summarize_epoch_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("schema") != "crlp.multitask.epoch.v1":
            continue
        key = (
            str(row["diagnostic_group"]),
            str(row["diagnostic_name"]),
            str(row["domain"]),
        )
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for (group, name, domain), items in sorted(grouped.items()):
        first_epoch = [float(row["loss_mean"]) for row in items if int(row["epoch"]) == 1]
        final_epoch = max(int(row["epoch"]) for row in items)
        final_losses = [float(row["loss_mean"]) for row in items if int(row["epoch"]) == final_epoch]
        final_updates = [float(row["train_updates"]) for row in items if int(row["epoch"]) == final_epoch]
        output.append(
            {
                "diagnostic_group": group,
                "diagnostic_name": name,
                "domain": domain,
                "final_epoch": final_epoch,
                "first_loss_mean": mean(first_epoch),
                "final_loss_mean": mean(final_losses),
                "loss_delta_final_minus_first": mean(final_losses) - mean(first_epoch),
                "final_train_updates_mean": mean(final_updates),
            }
        )
    return output


def write_report(
    path: Path,
    summary: list[dict[str, Any]],
    epoch_summary: list[dict[str, Any]],
) -> None:
    by_group = {(row["diagnostic_group"], row["diagnostic_name"]): row for row in summary}
    lines = [
        "# Multitask ARC Failure Diagnostics",
        "",
        "All rows evaluate fixed `0520fde7` ARC with `--eval-tasks arc --no-task-baselines`.",
        "",
        "## Order Sweep",
        "",
        "| Order | Top-1 | Fixed cells | NLL | Updates | Pred 0 cells | Pred 2 cells | Dominant color cells |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for order in ORDER_SWEEP:
        row = by_group.get(("order", order))
        if row is None:
            continue
        lines.append(
            f"| `{order}` | {fmt_pct(float(row['top1_mean']))} | "
            f"{fmt_pct(float(row['arc_fixed_cell_accuracy_mean']))} | "
            f"{fmt_float(float(row['nll_mean']))} | "
            f"{float(row['train_updates_mean']):.0f} | "
            f"{float(row['predicted_zero_cells_mean']):.2f} | "
            f"{float(row['predicted_two_cells_mean']):.2f} | "
            f"{float(row['dominant_predicted_color_cells_mean']):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Learning-Rate Sweep",
            "",
            "| LR | Top-1 | Fixed cells | NLL | Fixed NLL | Pred 0 cells | Pred 2 cells | Dominant color cells |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for lr in LR_SWEEP:
        row = by_group.get(("learning_rate", lr))
        if row is None:
            continue
        lines.append(
            f"| `{lr}` | {fmt_pct(float(row['top1_mean']))} | "
            f"{fmt_pct(float(row['arc_fixed_cell_accuracy_mean']))} | "
            f"{fmt_float(float(row['nll_mean']))} | "
            f"{fmt_float(float(row['arc_fixed_nll_mean']))} | "
            f"{float(row['predicted_zero_cells_mean']):.2f} | "
            f"{float(row['predicted_two_cells_mean']):.2f} | "
            f"{float(row['dominant_predicted_color_cells_mean']):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Epoch Prefix Sweep",
            "",
            "| Epochs | Top-1 | Fixed cells | NLL | Fixed NLL | Pred 0 cells | Pred 2 cells |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for epochs in EPOCH_SWEEP:
        row = by_group.get(("epoch_prefix", str(epochs)))
        if row is None:
            continue
        lines.append(
            f"| `{epochs}` | {fmt_pct(float(row['top1_mean']))} | "
            f"{fmt_pct(float(row['arc_fixed_cell_accuracy_mean']))} | "
            f"{fmt_float(float(row['nll_mean']))} | "
            f"{fmt_float(float(row['arc_fixed_nll_mean']))} | "
            f"{float(row['predicted_zero_cells_mean']):.2f} | "
            f"{float(row['predicted_two_cells_mean']):.2f} |"
        )

    arc_epochs = [
        row for row in epoch_summary
        if row["diagnostic_group"] == "epoch_prefix" and row["domain"] == "arc_agi"
    ]
    lines.extend(
        [
            "",
            "## ARC Training Loss",
            "",
            "| Scenario | First loss | Final loss | Delta | Final updates |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(arc_epochs, key=lambda item: int(item["diagnostic_name"])):
        lines.append(
            f"| `{row['diagnostic_name']}` | "
            f"{fmt_float(float(row['first_loss_mean']))} | "
            f"{fmt_float(float(row['final_loss_mean']))} | "
            f"{fmt_float(float(row['loss_delta_final_minus_first']))} | "
            f"{float(row['final_train_updates_mean']):.0f} |"
        )

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build/float/multitask_domain"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/2026-05-17-multitask-roadmap"))
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds(DEFAULT_SEEDS))
    parser.add_argument("--steps-per-epoch", type=int, default=12000)
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--hidden", default="64x32")
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    binary = args.binary if args.binary.is_absolute() else repo / args.binary

    scenarios: list[dict[str, str | int]] = []
    for order in ORDER_SWEEP:
        scenarios.append(
            {
                "diagnostic_group": "order",
                "diagnostic_name": order,
                "epochs": 16,
                "training_order": order,
                "learning_rate": "0.006",
            }
        )
    for lr in LR_SWEEP:
        scenarios.append(
            {
                "diagnostic_group": "learning_rate",
                "diagnostic_name": lr,
                "epochs": 16,
                "training_order": "uniform_random",
                "learning_rate": lr,
            }
        )
    for epochs in EPOCH_SWEEP:
        scenarios.append(
            {
                "diagnostic_group": "epoch_prefix",
                "diagnostic_name": str(epochs),
                "epochs": epochs,
                "training_order": "uniform_random",
                "learning_rate": "0.006",
            }
        )

    rows: list[dict[str, Any]] = []
    commands: list[str] = []
    for scenario in scenarios:
        for seed in args.seeds:
            command = [
                str(binary),
                "--tasks",
                "all",
                "--eval-tasks",
                "arc",
                "--epochs",
                str(scenario["epochs"]),
                "--steps-per-epoch",
                str(args.steps_per_epoch),
                "--eval-samples",
                str(args.eval_samples),
                "--hidden",
                args.hidden,
                "--training-order",
                str(scenario["training_order"]),
                "--learning-rate",
                str(scenario["learning_rate"]),
                "--seed",
                str(seed),
                "--no-task-baselines",
            ]
            commands.append(" ".join(command))
            log_name = (
                f"failure_diag_{scenario['diagnostic_group']}_"
                f"{scenario['diagnostic_name']}_seed_{seed}.log"
            )
            for row in run_command(command, repo, log_dir / log_name):
                row["roadmap_item"] = "R-MT.diagnostic"
                row["diagnostic_group"] = scenario["diagnostic_group"]
                row["diagnostic_name"] = scenario["diagnostic_name"]
                rows.append(row)

    raw_path = output_dir / "rmt_arc_failure_diagnostics.raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    (output_dir / "rmt_arc_failure_diagnostics.commands.txt").write_text(
        "\n".join(commands) + "\n",
        encoding="utf-8",
    )
    summary = summarize_summary_rows(rows)
    epoch_summary = summarize_epoch_rows(rows)
    write_csv(output_dir / "rmt_arc_failure_diagnostics_summary.csv", summary)
    write_csv(output_dir / "rmt_arc_failure_diagnostics_epoch_summary.csv", epoch_summary)
    write_report(output_dir / "rmt_arc_failure_diagnostics.md", summary, epoch_summary)
    print(f"wrote {output_dir / 'rmt_arc_failure_diagnostics.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
