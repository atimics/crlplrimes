#!/usr/bin/env python3
"""Run the R-MT.6 multitask trunk capacity sweep."""

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
DEFAULT_CAPACITIES = "32x16,64x32,128x64,256x128"
DOMAINS = ["prime_gap", "grid_path", "delayed_claim", "arc_agi"]


def parse_csv(text: str) -> list[str]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in parse_csv(text):
        seeds.append(int(part, 10))
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


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["hidden"]), str(row["domain"]), str(row["model_kind"]))
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for (hidden, domain, model_kind), group in sorted(grouped.items()):
        top1 = [float(row["top1"]) for row in group]
        nll = [float(row["nll"]) for row in group]
        ece = [float(row["ece"]) for row in group]
        updates = [float(row["train_updates"]) for row in group]
        params = [float(row["parameters"]) for row in group]
        flops = [float(row["forward_flops"]) for row in group]
        output.append(
            {
                "hidden": hidden,
                "domain": domain,
                "model_kind": model_kind,
                "seeds": len({int(row["seed"]) for row in group}),
                "rows": len(group),
                "parameters_mean": mean(params),
                "forward_flops_mean": mean(flops),
                "top1_mean": mean(top1),
                "top1_stdev": sample_stdev(top1),
                "nll_mean": mean(nll),
                "nll_stdev": sample_stdev(nll),
                "ece_mean": mean(ece),
                "ece_stdev": sample_stdev(ece),
                "train_updates_mean": mean(updates),
            }
        )
    return output


def attach_deltas(summary: list[dict[str, Any]], capacities: list[str]) -> list[dict[str, Any]]:
    by_key = {
        (str(row["hidden"]), str(row["domain"]), str(row["model_kind"])): row
        for row in summary
    }
    output: list[dict[str, Any]] = []
    for hidden in capacities:
        for domain in DOMAINS:
            shared = by_key.get((hidden, domain, "shared"))
            single = by_key.get((hidden, domain, "single_task"))
            if shared is None or single is None:
                continue
            output.append(
                {
                    "hidden": hidden,
                    "domain": domain,
                    "seeds": shared["seeds"],
                    "parameters_mean": shared["parameters_mean"],
                    "forward_flops_mean": shared["forward_flops_mean"],
                    "shared_top1_mean": shared["top1_mean"],
                    "single_top1_mean": single["top1_mean"],
                    "top1_delta_shared_minus_single": float(shared["top1_mean"])
                    - float(single["top1_mean"]),
                    "shared_nll_mean": shared["nll_mean"],
                    "single_nll_mean": single["nll_mean"],
                    "nll_delta_shared_minus_single": float(shared["nll_mean"])
                    - float(single["nll_mean"]),
                    "shared_ece_mean": shared["ece_mean"],
                    "single_ece_mean": single["ece_mean"],
                    "ece_delta_shared_minus_single": float(shared["ece_mean"])
                    - float(single["ece_mean"]),
                    "shared_updates_mean": shared["train_updates_mean"],
                    "single_updates_mean": single["train_updates_mean"],
                }
            )
    return output


def write_report(
    path: Path,
    seeds: list[int],
    capacities: list[str],
    epochs: int,
    steps_per_epoch: int,
    eval_samples: int,
    deltas: list[dict[str, Any]],
) -> None:
    by_hidden_domain = {
        (str(row["hidden"]), str(row["domain"])): row for row in deltas
    }
    lines = [
        "# Multitask Capacity Sweep",
        "",
        f"Seeds: `{', '.join(str(seed) for seed in seeds)}`",
        f"Capacities: `{', '.join(capacities)}`",
        f"Budget: `{epochs}` epochs x `{steps_per_epoch}` steps = `{epochs * steps_per_epoch}` total shared updates.",
        f"Evaluation samples: `{eval_samples}` per domain row.",
        "",
        "## Fixed-Task Shared Minus Single Top-1",
        "",
        "| Hidden | Params | Prime gap | Grid path | Delayed claim | ARC fixed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for hidden in capacities:
        arc_row = by_hidden_domain.get((hidden, "arc_agi"))
        params = float(arc_row["parameters_mean"]) if arc_row is not None else float("nan")
        cells = []
        for domain in DOMAINS:
            row = by_hidden_domain.get((hidden, domain))
            cells.append(fmt_delta(float(row["top1_delta_shared_minus_single"])) if row else "-")
        lines.append(
            f"| `{hidden}` | {params:.0f} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |"
        )

    lines.extend(
        [
            "",
            "## ARC Fixed-Task Detail",
            "",
            "| Hidden | Shared top-1 | Single top-1 | Delta | Shared NLL | Single NLL | Shared ECE | Single ECE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for hidden in capacities:
        row = by_hidden_domain.get((hidden, "arc_agi"))
        if row is None:
            continue
        lines.append(
            f"| `{hidden}` | {fmt_pct(float(row['shared_top1_mean']))} | "
            f"{fmt_pct(float(row['single_top1_mean']))} | "
            f"{fmt_delta(float(row['top1_delta_shared_minus_single']))} | "
            f"{fmt_float(float(row['shared_nll_mean']))} | "
            f"{fmt_float(float(row['single_nll_mean']))} | "
            f"{fmt_float(float(row['shared_ece_mean']))} | "
            f"{fmt_float(float(row['single_ece_mean']))} |"
        )

    lines.extend(
        [
            "",
            "## Delayed-Claim Guardrail",
            "",
            "| Hidden | Shared top-1 | Single top-1 | Delta | Shared NLL | Single NLL |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for hidden in capacities:
        row = by_hidden_domain.get((hidden, "delayed_claim"))
        if row is None:
            continue
        lines.append(
            f"| `{hidden}` | {fmt_pct(float(row['shared_top1_mean']))} | "
            f"{fmt_pct(float(row['single_top1_mean']))} | "
            f"{fmt_delta(float(row['top1_delta_shared_minus_single']))} | "
            f"{fmt_float(float(row['shared_nll_mean']))} | "
            f"{fmt_float(float(row['single_nll_mean']))} |"
        )

    arc_rows = [by_hidden_domain[(hidden, "arc_agi")] for hidden in capacities if (hidden, "arc_agi") in by_hidden_domain]
    best_arc = max(arc_rows, key=lambda row: float(row["shared_top1_mean"])) if arc_rows else None
    best_delta = max(arc_rows, key=lambda row: float(row["top1_delta_shared_minus_single"])) if arc_rows else None
    lines.extend(["", "## Interpretation", ""])
    if best_arc is not None and best_delta is not None:
        lines.append(
            f"- Best shared fixed-ARC top-1 is `{fmt_pct(float(best_arc['shared_top1_mean']))}` at `{best_arc['hidden']}`."
        )
        lines.append(
            f"- Least negative fixed-ARC shared-minus-single delta is `{fmt_delta(float(best_delta['top1_delta_shared_minus_single']))}` at `{best_delta['hidden']}`."
        )
    lines.append(
        "- R-MT.6 should be read as a fixed-envelope capacity diagnostic; it does not replace the same-envelope ARC OOD result from R-MT.2."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build/float/multitask_domain"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/2026-05-17-multitask-roadmap"))
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds(DEFAULT_SEEDS))
    parser.add_argument("--capacities", type=parse_csv, default=parse_csv(DEFAULT_CAPACITIES))
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--steps-per-epoch", type=int, default=12000)
    parser.add_argument("--eval-samples", type=int, default=1000)
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    binary = args.binary if args.binary.is_absolute() else repo / args.binary

    rows: list[dict[str, Any]] = []
    commands: list[str] = []
    for hidden in args.capacities:
        for seed in args.seeds:
            command = [
                str(binary),
                "--tasks",
                "all",
                "--eval-tasks",
                "all",
                "--epochs",
                str(args.epochs),
                "--steps-per-epoch",
                str(args.steps_per_epoch),
                "--eval-samples",
                str(args.eval_samples),
                "--hidden",
                hidden,
                "--seed",
                str(seed),
            ]
            commands.append(" ".join(command))
            for row in run_command(command, repo, log_dir / f"rmt6_hidden_{hidden}_seed_{seed}.log"):
                row["roadmap_item"] = "R-MT.6"
                rows.append(row)

    raw_path = output_dir / "rmt6_capacity_sweep.raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    (output_dir / "rmt6_capacity_sweep.commands.txt").write_text(
        "\n".join(commands) + "\n",
        encoding="utf-8",
    )

    summary_rows = [
        row for row in rows if row.get("schema") == "crlp.multitask.summary.v1"
    ]
    summary = summarize(summary_rows)
    deltas = attach_deltas(summary, args.capacities)
    write_csv(output_dir / "rmt6_capacity_sweep_summary.csv", summary)
    write_csv(output_dir / "rmt6_capacity_sweep_deltas.csv", deltas)
    write_report(
        output_dir / "rmt6_capacity_sweep.md",
        args.seeds,
        args.capacities,
        args.epochs,
        args.steps_per_epoch,
        args.eval_samples,
        deltas,
    )
    print(f"wrote {output_dir / 'rmt6_capacity_sweep.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
