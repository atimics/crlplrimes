#!/usr/bin/env python3
"""Run the shared multitask scorer across task-set and feature ablations."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path


TASK_SPECS = [
    ("prime", "prime", "prime_gap"),
    ("grid", "grid", "grid_path"),
    ("claim", "claim", "delayed_claim"),
    ("arc", "arc", "arc_agi"),
]

VARIANTS = [
    ("tagged", []),
    ("no_task_tags", ["--no-task-tags"]),
]

DOMAINS = [domain for _label, _arg, domain in TASK_SPECS]


def build_task_sets() -> list[tuple[str, str, set[str]]]:
    task_sets = []
    for size in range(1, len(TASK_SPECS) + 1):
        for combo in itertools.combinations(TASK_SPECS, size):
            labels = [label for label, _arg, _domain in combo]
            args = [arg for _label, arg, _domain in combo]
            domains = {domain for _label, _arg, domain in combo}
            if size == len(TASK_SPECS):
                task_sets.append(("all", "all", domains))
            else:
                task_sets.append(("+".join(labels), ",".join(args), domains))
    return task_sets


def parse_seeds(text: str) -> list[int]:
    seeds = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
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


def run_command(command: list[str], cwd: Path) -> list[dict[str, object]]:
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise subprocess.CalledProcessError(proc.returncode, command)

    rows = []
    for line in proc.stdout.splitlines():
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        if row.get("schema") == "crlp.multitask.summary.v1":
            rows.append(row)
    return rows


def summarize(rows: list[dict[str, object]], trained_only: bool) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        if trained_only and not row["trained_domain"]:
            continue
        if not trained_only and row["trained_domain"]:
            continue
        key = (str(row["variant"]), str(row["run_label"]), str(row["domain"]))
        grouped.setdefault(key, []).append(row)

    summary = []
    for (variant, run_label, domain), group in sorted(grouped.items()):
        top1_values = [float(row["top1"]) for row in group]
        nll_values = [float(row["nll"]) for row in group]
        update_values = [float(row["train_updates"]) for row in group]
        summary.append(
            {
                "variant": variant,
                "training_set": run_label,
                "domain": domain,
                "seeds": len(group),
                "top1_mean": mean(top1_values),
                "top1_stdev": sample_stdev(top1_values),
                "nll_mean": mean(nll_values),
                "nll_stdev": sample_stdev(nll_values),
                "train_updates_mean": mean(update_values),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "variant",
        "training_set",
        "domain",
        "seeds",
        "top1_mean",
        "top1_stdev",
        "nll_mean",
        "nll_stdev",
        "train_updates_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt_pct(value: float) -> str:
    return "-" if math.isnan(value) else f"{100.0 * value:.2f}%"


def write_summary(
    path: Path,
    trained_summary: list[dict[str, object]],
    heldout_summary: list[dict[str, object]],
) -> None:
    trained_index = {
        (str(row["variant"]), str(row["training_set"]), str(row["domain"])): row
        for row in trained_summary
    }
    heldout_by_variant: dict[tuple[str, str], list[float]] = {}
    for row in heldout_summary:
        heldout_by_variant.setdefault((str(row["variant"]), str(row["training_set"])), []).append(
            float(row["top1_mean"])
        )

    headers = [label for label, _arg, _domain in TASK_SPECS]
    lines = [
        "# Multitask Ablation Matrix",
        "",
        "Mean top-1 over the configured seeds. Trained-domain rows only count domains present in the training set; held-out top-1 averages domains evaluated but not trained.",
        "",
        "## Trained Domains",
        "",
        "| variant | training set | " + " | ".join(headers) + " | held-out avg |",
        "| --- | ---: | " + " | ".join("---:" for _header in headers) + " | ---: |",
    ]
    task_sets = build_task_sets()
    for variant, _flags in VARIANTS:
        for run_label, _tasks, _domains in task_sets:
            values = []
            for domain in DOMAINS:
                row = trained_index.get((variant, run_label, domain))
                values.append(fmt_pct(float(row["top1_mean"])) if row else "-")
            heldout_values = heldout_by_variant.get((variant, run_label), [])
            lines.append(
                f"| {variant} | {run_label} | {' | '.join(values)} | "
                f"{fmt_pct(mean(heldout_values)) if heldout_values else '-'} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build/float/multitask_domain"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/float/multitask_ablation_matrix"))
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("20240516,20240517,20240518"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--steps-per-domain", type=int, default=3000)
    parser.add_argument("--eval-samples", type=int, default=1000)
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    output_dir = (repo / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    binary = args.binary
    if not binary.is_absolute():
        binary = repo / binary

    rows: list[dict[str, object]] = []
    commands: list[str] = []
    task_sets = build_task_sets()
    for variant, variant_flags in VARIANTS:
        for run_label, task_arg, trained_domains in task_sets:
            active_count = len(trained_domains)
            steps_per_epoch = args.steps_per_domain * active_count
            for seed in args.seeds:
                command = [
                    str(binary),
                    "--tasks",
                    task_arg,
                    "--eval-tasks",
                    "all",
                    "--epochs",
                    str(args.epochs),
                    "--steps-per-epoch",
                    str(steps_per_epoch),
                    "--eval-samples",
                    str(args.eval_samples),
                    "--seed",
                    str(seed),
                    "--no-task-baselines",
                    *variant_flags,
                ]
                print(f"running {variant} {run_label} seed={seed}", flush=True)
                commands.append(" ".join(command))
                for row in run_command(command, repo):
                    row["variant"] = variant
                    row["run_label"] = run_label
                    row["trained_domain"] = row["domain"] in trained_domains
                    rows.append(row)

    raw_path = output_dir / "raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output_dir / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")

    trained_summary = summarize(rows, trained_only=True)
    heldout_summary = summarize(rows, trained_only=False)
    write_csv(output_dir / "trained_summary.csv", trained_summary)
    write_csv(output_dir / "heldout_summary.csv", heldout_summary)
    write_summary(output_dir / "summary.md", trained_summary, heldout_summary)
    print(f"wrote {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
