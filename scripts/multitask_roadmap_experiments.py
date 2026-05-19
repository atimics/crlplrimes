#!/usr/bin/env python3
"""Capture the multitask roadmap experiments R-MT.1 and R-MT.2."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from crlp.cli import parse_seeds
from crlp.io import write_csv, write_jsonl
from crlp.run import run_json_command, shell_label
from crlp.stats import fmt_float, fmt_pct, mean, sample_stdev


DEFAULT_SEEDS = "20240516,20240517,20240518"


def summarize_rows(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(key_name, "")) for key_name in keys)
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        top1 = [float(row["top1"]) for row in group]
        nll = [float(row["nll"]) for row in group]
        ece = [float(row["ece"]) for row in group]
        updates = [float(row["train_updates"]) for row in group]
        out = {key_name: value for key_name, value in zip(keys, key)}
        out.update(
            {
                "seeds": len({int(row["seed"]) for row in group}),
                "rows": len(group),
                "top1_mean": mean(top1),
                "top1_stdev": sample_stdev(top1),
                "nll_mean": mean(nll),
                "nll_stdev": sample_stdev(nll),
                "ece_mean": mean(ece),
                "ece_stdev": sample_stdev(ece),
                "train_updates_mean": mean(updates),
            }
        )
        summary.append(out)
    return summary


def write_summary(
    path: Path,
    seeds: list[int],
    epochs: int,
    steps_per_epoch: int,
    eval_samples: int,
    rmt1_summary: list[dict[str, Any]],
    rmt2_aggregate: list[dict[str, Any]],
    rmt2_tasks: list[dict[str, Any]],
) -> None:
    rmt1_index = {
        (str(row["domain"]), str(row["model_kind"])): row for row in rmt1_summary
    }
    lines = [
        "# Multitask Roadmap Experiments",
        "",
        f"Seeds: `{', '.join(str(seed) for seed in seeds)}`",
        f"Budget: `{epochs}` epochs x `{steps_per_epoch}` steps = `{epochs * steps_per_epoch}` total shared updates.",
        f"Evaluation samples: `{eval_samples}` per aggregate row.",
        "",
        "## R-MT.1 Longer Shared And Single Runs",
        "",
        "| Domain | Shared top-1 | Single top-1 | Delta | Shared NLL | Single NLL | Shared ECE | Single ECE | Shared updates | Single updates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for domain in ["prime_gap", "grid_path", "delayed_claim", "arc_agi"]:
        shared = rmt1_index.get((domain, "shared"))
        single = rmt1_index.get((domain, "single_task"))
        if shared is None or single is None:
            continue
        delta = float(shared["top1_mean"]) - float(single["top1_mean"])
        lines.append(
            f"| `{domain}` | {fmt_pct(float(shared['top1_mean']))} | "
            f"{fmt_pct(float(single['top1_mean']))} | {100.0 * delta:+.2f}pp | "
            f"{fmt_float(float(shared['nll_mean']))} | {fmt_float(float(single['nll_mean']))} | "
            f"{fmt_float(float(shared['ece_mean']))} | {fmt_float(float(single['ece_mean']))} | "
            f"{float(shared['train_updates_mean']):.0f} | {float(single['train_updates_mean']):.0f} |"
        )

    rmt2_index = {str(row["model_kind"]): row for row in rmt2_aggregate}
    lines.extend(
        [
            "",
            "## R-MT.2 Same-Envelope ARC OOD",
            "",
            "The ARC scorer is trained on `0520fde7`; evaluation uses 12 held-out 7x3 -> 3x3 panel-composition tasks inside the existing 96-feature action-row envelope.",
            "",
            "| Model | Top-1 | NLL | ECE | Updates |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model_kind in ["shared", "single_task"]:
        row = rmt2_index.get(model_kind)
        if row is None:
            continue
        lines.append(
            f"| `{model_kind}` | {fmt_pct(float(row['top1_mean']))} | "
            f"{fmt_float(float(row['nll_mean']))} | {fmt_float(float(row['ece_mean']))} | "
            f"{float(row['train_updates_mean']):.0f} |"
        )
    if "shared" in rmt2_index and "single_task" in rmt2_index:
        delta = float(rmt2_index["shared"]["top1_mean"]) - float(rmt2_index["single_task"]["top1_mean"])
        lines.append("")
        lines.append(f"Aggregate shared-minus-single OOD top-1 delta: `{100.0 * delta:+.2f}pp`.")

    task_by_name: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rmt2_tasks:
        task = str(row["arc_eval_task"])
        task_by_name.setdefault(task, {})[str(row["model_kind"])] = row
    lines.extend(
        [
            "",
            "### ARC OOD Task Breakdown",
            "",
            "| Task | Family | Shared top-1 | Single top-1 | Delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for task, models in sorted(task_by_name.items()):
        shared = models.get("shared")
        single = models.get("single_task")
        if shared is None or single is None:
            continue
        delta = float(shared["top1_mean"]) - float(single["top1_mean"])
        family = str(shared.get("arc_eval_family", ""))
        lines.append(
            f"| `{task}` | `{family}` | {fmt_pct(float(shared['top1_mean']))} | "
            f"{fmt_pct(float(single['top1_mean']))} | {100.0 * delta:+.2f}pp |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build/float/multitask_domain"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/2026-05-17-multitask-roadmap"))
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--steps-per-epoch", type=int, default=12000)
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--hidden", default="64x32")
    parser.add_argument("--skip-rmt1", action="store_true")
    parser.add_argument("--skip-rmt2", action="store_true")
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    output_dir = (repo / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"

    binary = args.binary
    if not binary.is_absolute():
        binary = repo / binary

    commands: list[str] = []
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        common = [
            "--epochs",
            str(args.epochs),
            "--steps-per-epoch",
            str(args.steps_per_epoch),
            "--eval-samples",
            str(args.eval_samples),
            "--hidden",
            args.hidden,
            "--seed",
            str(seed),
        ]
        if not args.skip_rmt1:
            command = [
                str(binary),
                "--tasks",
                "all",
                "--eval-tasks",
                "all",
                *common,
            ]
            commands.append(shell_label(command))
            for row in run_json_command(
                command,
                cwd=repo,
                log_path=log_dir / f"rmt1_seed_{seed}.log",
            ):
                row["roadmap_item"] = "R-MT.1"
                row["arc_ood_breakdown"] = False
                rows.append(row)

        if not args.skip_rmt2:
            command = [
                str(binary),
                "--tasks",
                "all",
                "--eval-tasks",
                "arc",
                "--arc-eval-suite",
                "ood",
                "--arc-ood-task-breakdown",
                *common,
            ]
            commands.append(shell_label(command))
            for row in run_json_command(
                command,
                cwd=repo,
                log_path=log_dir / f"rmt2_seed_{seed}.log",
            ):
                row["roadmap_item"] = "R-MT.2"
                row["arc_ood_breakdown"] = row.get("arc_eval_task") not in (None, "all")
                rows.append(row)

    raw_path = output_dir / "raw.jsonl"
    write_jsonl(raw_path, rows, compact=True)
    (output_dir / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")

    rmt1_rows = [
        row
        for row in rows
        if (
            row.get("roadmap_item") == "R-MT.1" and
            row.get("schema") == "crlp.multitask.summary.v1"
        )
    ]
    rmt2_aggregate_rows = [
        row
        for row in rows
        if (
            row.get("roadmap_item") == "R-MT.2" and
            row.get("schema") == "crlp.multitask.summary.v1" and
            row.get("arc_eval_task") == "all"
        )
    ]
    rmt2_task_rows = [
        row
        for row in rows
        if (
            row.get("roadmap_item") == "R-MT.2" and
            row.get("schema") == "crlp.multitask.summary.v1" and
            row.get("arc_ood_breakdown")
        )
    ]

    rmt1_summary = summarize_rows(rmt1_rows, ["domain", "model_kind"])
    rmt2_aggregate = summarize_rows(rmt2_aggregate_rows, ["model_kind"])
    rmt2_tasks = summarize_rows(rmt2_task_rows, ["arc_eval_task", "arc_eval_family", "model_kind"])
    write_csv(output_dir / "rmt1_summary.csv", rmt1_summary)
    write_csv(output_dir / "rmt2_aggregate.csv", rmt2_aggregate)
    write_csv(output_dir / "rmt2_task_summary.csv", rmt2_tasks)
    write_summary(
        output_dir / "summary.md",
        args.seeds,
        args.epochs,
        args.steps_per_epoch,
        args.eval_samples,
        rmt1_summary,
        rmt2_aggregate,
        rmt2_tasks,
    )
    print(f"wrote {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
