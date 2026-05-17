#!/usr/bin/env python3
"""Run the focused delayed-claim leave-one-out experiment for R-MT.3."""

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

TASK_SETS = [
    {
        "training_set": "all",
        "tasks": "all",
        "active_count": 4,
        "omitted": "none",
        "description": "prime_gap+grid_path+delayed_claim+arc_agi",
    },
    {
        "training_set": "no_prime_gap",
        "tasks": "grid,claim,arc",
        "active_count": 3,
        "omitted": "prime_gap",
        "description": "grid_path+delayed_claim+arc_agi",
    },
    {
        "training_set": "no_grid_path",
        "tasks": "prime,claim,arc",
        "active_count": 3,
        "omitted": "grid_path",
        "description": "prime_gap+delayed_claim+arc_agi",
    },
    {
        "training_set": "no_arc_agi",
        "tasks": "prime,grid,claim",
        "active_count": 3,
        "omitted": "arc_agi",
        "description": "prime_gap+grid_path+delayed_claim",
    },
    {
        "training_set": "claim_only",
        "tasks": "claim",
        "active_count": 1,
        "omitted": "all_siblings",
        "description": "delayed_claim",
    },
]


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            seeds.append(int(part, 10))
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


def fmt_delta(value: float) -> str:
    return "-" if math.isnan(value) else f"{100.0 * value:+.2f}pp"


def fmt_float(value: float) -> str:
    return "-" if math.isnan(value) else f"{value:.4f}"


def command_label(command: list[str]) -> str:
    return " ".join(command)


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
        row = json.loads(line)
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["training_set"]), []).append(row)

    output: list[dict[str, Any]] = []
    for spec in TASK_SETS:
        training_set = str(spec["training_set"])
        group = grouped.get(training_set, [])
        top1 = [float(row["top1"]) for row in group]
        nll = [float(row["nll"]) for row in group]
        ece = [float(row["ece"]) for row in group]
        updates = [float(row["train_updates"]) for row in group]
        output.append(
            {
                "training_set": training_set,
                "tasks": spec["description"],
                "omitted": spec["omitted"],
                "seeds": len({int(row["seed"]) for row in group}),
                "rows": len(group),
                "top1_mean": mean(top1),
                "top1_stdev": sample_stdev(top1),
                "nll_mean": mean(nll),
                "nll_stdev": sample_stdev(nll),
                "ece_mean": mean(ece),
                "ece_stdev": sample_stdev(ece),
                "claim_updates_mean": mean(updates),
            }
        )
    return output


def attach_deltas(summary: list[dict[str, Any]]) -> None:
    by_set = {str(row["training_set"]): row for row in summary}
    claim_only = by_set.get("claim_only")
    all_tasks = by_set.get("all")
    claim_top1 = float(claim_only["top1_mean"]) if claim_only else float("nan")
    all_top1 = float(all_tasks["top1_mean"]) if all_tasks else float("nan")
    claim_nll = float(claim_only["nll_mean"]) if claim_only else float("nan")
    all_nll = float(all_tasks["nll_mean"]) if all_tasks else float("nan")
    claim_ece = float(claim_only["ece_mean"]) if claim_only else float("nan")
    all_ece = float(all_tasks["ece_mean"]) if all_tasks else float("nan")
    for row in summary:
        row["top1_delta_vs_claim_only"] = float(row["top1_mean"]) - claim_top1
        row["top1_delta_vs_all"] = float(row["top1_mean"]) - all_top1
        row["nll_delta_vs_claim_only"] = float(row["nll_mean"]) - claim_nll
        row["nll_delta_vs_all"] = float(row["nll_mean"]) - all_nll
        row["ece_delta_vs_claim_only"] = float(row["ece_mean"]) - claim_ece
        row["ece_delta_vs_all"] = float(row["ece_mean"]) - all_ece


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


def write_report(
    path: Path,
    seeds: list[int],
    epochs: int,
    steps_per_domain: int,
    eval_samples: int,
    summary: list[dict[str, Any]],
) -> None:
    by_set = {str(row["training_set"]): row for row in summary}
    all_tasks = by_set["all"]
    claim_only = by_set["claim_only"]
    lines = [
        "# Delayed-Claim Leave-One-Out",
        "",
        f"Seeds: `{', '.join(str(seed) for seed in seeds)}`",
        f"Budget: `{epochs}` epochs x `{steps_per_domain}` steps per active domain = `{epochs * steps_per_domain}` expected updates per domain.",
        f"Evaluation samples: `{eval_samples}` delayed-claim states per row.",
        "",
        "| Training set | Omitted sibling | Claim top-1 | Δ vs claim-only | Δ vs all | NLL | ECE | Claim updates |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['training_set']}` | `{row['omitted']}` | "
            f"{fmt_pct(float(row['top1_mean']))} | "
            f"{fmt_delta(float(row['top1_delta_vs_claim_only']))} | "
            f"{fmt_delta(float(row['top1_delta_vs_all']))} | "
            f"{fmt_float(float(row['nll_mean']))} | "
            f"{fmt_float(float(row['ece_mean']))} | "
            f"{float(row['claim_updates_mean']):.0f} |"
        )

    full_delta = float(all_tasks["top1_delta_vs_claim_only"])
    best_removal = min(
        [row for row in summary if str(row["training_set"]).startswith("no_")],
        key=lambda row: float(row["top1_delta_vs_all"]),
    )
    worst_removal = max(
        [row for row in summary if str(row["training_set"]).startswith("no_")],
        key=lambda row: float(row["top1_delta_vs_all"]),
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"All-domain training is `{fmt_delta(full_delta)}` versus claim-only on delayed-claim top-1.",
            f"The largest removal gain versus all-domain is `{worst_removal['training_set']}` at `{fmt_delta(float(worst_removal['top1_delta_vs_all']))}`.",
            f"The largest removal loss versus all-domain is `{best_removal['training_set']}` at `{fmt_delta(float(best_removal['top1_delta_vs_all']))}`.",
        ]
    )
    if abs(full_delta) < 0.005:
        lines.append(
            "This does not support a stable delayed-claim positive-transfer claim at the long-run budget."
        )
    elif full_delta > 0.0:
        lines.append(
            "This keeps a delayed-claim positive-transfer claim alive, but it should be tied to the specific sibling-removal pattern above."
        )
    else:
        lines.append(
            "This is negative transfer for delayed-claim at the long-run budget."
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build/float/multitask_domain"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/2026-05-17-multitask-roadmap"))
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--steps-per-domain", type=int, default=3000)
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--hidden", default="64x32")
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    binary = args.binary if args.binary.is_absolute() else repo / args.binary

    rows: list[dict[str, Any]] = []
    commands: list[str] = []
    for seed in args.seeds:
        for spec in TASK_SETS:
            steps_per_epoch = args.steps_per_domain * int(spec["active_count"])
            command = [
                str(binary),
                "--tasks",
                str(spec["tasks"]),
                "--eval-tasks",
                "claim",
                "--epochs",
                str(args.epochs),
                "--steps-per-epoch",
                str(steps_per_epoch),
                "--eval-samples",
                str(args.eval_samples),
                "--hidden",
                args.hidden,
                "--seed",
                str(seed),
                "--no-task-baselines",
            ]
            commands.append(command_label(command))
            log_path = log_dir / f"rmt3_{spec['training_set']}_seed_{seed}.log"
            for row in run_command(command, repo, log_path):
                row["roadmap_item"] = "R-MT.3"
                row["training_set"] = spec["training_set"]
                row["training_set_description"] = spec["description"]
                row["omitted_sibling"] = spec["omitted"]
                row["steps_per_domain"] = args.steps_per_domain
                rows.append(row)

    raw_path = output_dir / "rmt3_leave_one_out.raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    (output_dir / "rmt3_leave_one_out.commands.txt").write_text(
        "\n".join(commands) + "\n",
        encoding="utf-8",
    )
    summary_rows = [
        row
        for row in rows
        if (
            row.get("schema") == "crlp.multitask.summary.v1" and
            row.get("domain") == "delayed_claim" and
            row.get("model_kind") == "shared"
        )
    ]
    summary = summarize(summary_rows)
    attach_deltas(summary)
    write_csv(output_dir / "rmt3_leave_one_out.csv", summary)
    write_report(
        output_dir / "rmt3_leave_one_out.md",
        args.seeds,
        args.epochs,
        args.steps_per_domain,
        args.eval_samples,
        summary,
    )
    print(f"wrote {output_dir / 'rmt3_leave_one_out.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
