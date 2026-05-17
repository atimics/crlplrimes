#!/usr/bin/env python3
"""Run the R-MT.4 shared-trunk plus per-task-head multitask comparison."""

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
DOMAINS = ["prime_gap", "grid_path", "delayed_claim", "arc_agi"]
MODEL_ORDER = ["shared", "shared_task_heads", "single_task"]


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
        if line.startswith("{"):
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


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("schema") != "crlp.multitask.summary.v1":
            continue
        grouped.setdefault((str(row["domain"]), str(row["model_kind"])), []).append(row)

    output: list[dict[str, Any]] = []
    for (domain, model_kind), items in sorted(grouped.items()):
        top1 = [float(row["top1"]) for row in items]
        nll = [float(row["nll"]) for row in items]
        ece = [float(row["ece"]) for row in items]
        updates = [float(row["train_updates"]) for row in items]
        utility = [float(row["decision_utility"]) for row in items]
        params = [float(row["parameters"]) for row in items]
        flops = [float(row["forward_flops"]) for row in items]
        output.append(
            {
                "domain": domain,
                "model_kind": model_kind,
                "seeds": len({int(row["seed"]) for row in items}),
                "rows": len(items),
                "top1_mean": mean(top1),
                "top1_stdev": sample_stdev(top1),
                "nll_mean": mean(nll),
                "nll_stdev": sample_stdev(nll),
                "ece_mean": mean(ece),
                "ece_stdev": sample_stdev(ece),
                "decision_utility_mean": mean(utility),
                "train_updates_mean": mean(updates),
                "parameters_mean": mean(params),
                "forward_flops_mean": mean(flops),
            }
        )
    return output


def attach_deltas(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (str(row["domain"]), str(row["model_kind"])): row for row in summary
    }
    output: list[dict[str, Any]] = []
    for domain in DOMAINS:
        headed = by_key.get((domain, "shared_task_heads"))
        shared = by_key.get((domain, "shared"))
        single = by_key.get((domain, "single_task"))
        if headed is None:
            continue
        output.append(
            {
                "domain": domain,
                "seeds": headed["seeds"],
                "headed_top1_mean": headed["top1_mean"],
                "shared_top1_mean": None if shared is None else shared["top1_mean"],
                "single_top1_mean": None if single is None else single["top1_mean"],
                "top1_delta_headed_minus_shared": None
                if shared is None
                else float(headed["top1_mean"]) - float(shared["top1_mean"]),
                "top1_delta_headed_minus_single": None
                if single is None
                else float(headed["top1_mean"]) - float(single["top1_mean"]),
                "headed_nll_mean": headed["nll_mean"],
                "shared_nll_mean": None if shared is None else shared["nll_mean"],
                "single_nll_mean": None if single is None else single["nll_mean"],
                "nll_delta_headed_minus_shared": None
                if shared is None
                else float(headed["nll_mean"]) - float(shared["nll_mean"]),
                "nll_delta_headed_minus_single": None
                if single is None
                else float(headed["nll_mean"]) - float(single["nll_mean"]),
                "headed_ece_mean": headed["ece_mean"],
                "shared_ece_mean": None if shared is None else shared["ece_mean"],
                "single_ece_mean": None if single is None else single["ece_mean"],
                "ece_delta_headed_minus_shared": None
                if shared is None
                else float(headed["ece_mean"]) - float(shared["ece_mean"]),
                "ece_delta_headed_minus_single": None
                if single is None
                else float(headed["ece_mean"]) - float(single["ece_mean"]),
                "headed_parameters_mean": headed["parameters_mean"],
                "headed_forward_flops_mean": headed["forward_flops_mean"],
            }
        )
    return output


def write_report(
    path: Path,
    seeds: list[int],
    epochs: int,
    steps_per_epoch: int,
    eval_samples: int,
    summary: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
) -> None:
    by_key = {
        (str(row["domain"]), str(row["model_kind"])): row for row in summary
    }
    delta_by_domain = {str(row["domain"]): row for row in deltas}
    lines = [
        "# Multitask Adapter-Headed Sharing",
        "",
        f"Seeds: `{', '.join(str(seed) for seed in seeds)}`",
        f"Budget: `{epochs}` epochs x `{steps_per_epoch}` steps = `{epochs * steps_per_epoch}` total shared updates.",
        f"Evaluation samples: `{eval_samples}` per domain row.",
        "",
        "This run compares the plain shared action scorer, the shared trunk with per-task scalar heads, and matched single-task baselines under the same 96-feature envelope.",
        "",
        "## Top-1",
        "",
        "| Domain | Shared | Task heads | Single | Heads - shared | Heads - single |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for domain in DOMAINS:
        shared = by_key.get((domain, "shared"))
        headed = by_key.get((domain, "shared_task_heads"))
        single = by_key.get((domain, "single_task"))
        delta = delta_by_domain.get(domain)
        lines.append(
            f"| `{domain}` | "
            f"{fmt_pct(float(shared['top1_mean'])) if shared else '-'} | "
            f"{fmt_pct(float(headed['top1_mean'])) if headed else '-'} | "
            f"{fmt_pct(float(single['top1_mean'])) if single else '-'} | "
            f"{fmt_delta(float(delta['top1_delta_headed_minus_shared'])) if delta and delta['top1_delta_headed_minus_shared'] is not None else '-'} | "
            f"{fmt_delta(float(delta['top1_delta_headed_minus_single'])) if delta and delta['top1_delta_headed_minus_single'] is not None else '-'} |"
        )

    lines.extend(
        [
            "",
            "## ARC And Claim Guardrail",
            "",
            "| Domain | Model | Top-1 | NLL | ECE | Updates | Params |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for domain in ["arc_agi", "delayed_claim"]:
        for model in MODEL_ORDER:
            row = by_key.get((domain, model))
            if row is None:
                continue
            lines.append(
                f"| `{domain}` | `{model}` | {fmt_pct(float(row['top1_mean']))} | "
                f"{fmt_float(float(row['nll_mean']))} | {fmt_float(float(row['ece_mean']))} | "
                f"{float(row['train_updates_mean']):.0f} | {float(row['parameters_mean']):.0f} |"
            )

    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "- Promote this architecture only if `shared_task_heads` recovers fixed ARC without losing the delayed-claim shared gain.",
            "- If task heads help fixed ARC but erase OOD transfer or delayed-claim transfer, keep them as a diagnostic rather than the default recipe.",
        ]
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
    parser.add_argument("--no-task-baselines", action="store_true")
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
            args.hidden,
            "--seed",
            str(seed),
            "--task-heads",
        ]
        if args.no_task_baselines:
            command.append("--no-task-baselines")
        commands.append(" ".join(command))
        for row in run_command(command, repo, log_dir / f"rmt4_task_heads_seed_{seed}.log"):
            row["roadmap_item"] = "R-MT.4"
            rows.append(row)

    raw_path = output_dir / "rmt4_adapter_heads.raw.jsonl"
    raw_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    (output_dir / "rmt4_adapter_heads.commands.txt").write_text(
        "\n".join(commands) + "\n",
        encoding="utf-8",
    )

    summary = summarize(rows)
    deltas = attach_deltas(summary)
    write_csv(output_dir / "rmt4_adapter_heads_summary.csv", summary)
    write_csv(output_dir / "rmt4_adapter_heads_deltas.csv", deltas)
    write_report(
        output_dir / "rmt4_adapter_heads.md",
        args.seeds,
        args.epochs,
        args.steps_per_epoch,
        args.eval_samples,
        summary,
        deltas,
    )
    print(f"wrote {output_dir / 'rmt4_adapter_heads.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
