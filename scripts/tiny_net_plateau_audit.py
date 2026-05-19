#!/usr/bin/env python3
"""Audit whether tiny multitask trunks are saturated before increasing width."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from crlp.cli import parse_csv, parse_int_csv, parse_seeds
from crlp.io import write_csv, write_jsonl
from crlp.run import run_json_command
from crlp.stats import fmt_delta, fmt_pct, mean, sample_stdev


DEFAULT_SEEDS = "20240516,20240517,20240518"
DEFAULT_CAPACITIES = "8x4,16x8,32x16,64x32"
DEFAULT_EPOCH_GRID = "1,2,4,8,16,32"
DOMAINS = ["prime_gap", "grid_path", "delayed_claim", "arc_agi"]
MODEL_KINDS = ("shared", "shared_task_heads", "single_task")


def learning_curve_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        groups[
            (
                str(row["hidden"]),
                str(row["domain"]),
                str(row["model_kind"]),
                int(row["epochs"]),
            )
        ].append(row)

    out: list[dict[str, Any]] = []
    for (hidden, domain, model_kind, epochs), group in sorted(groups.items()):
        top1 = [float(row["top1"]) for row in group]
        nll = [float(row["nll"]) for row in group]
        ece = [float(row["ece"]) for row in group]
        params = [float(row["parameters"]) for row in group]
        flops = [float(row["forward_flops"]) for row in group]
        out.append(
            {
                "hidden": hidden,
                "domain": domain,
                "model_kind": model_kind,
                "epochs": epochs,
                "seeds": len({int(row["seed"]) for row in group}),
                "top1_mean": mean(top1),
                "top1_stdev": sample_stdev(top1),
                "top1_min": min(top1),
                "top1_max": max(top1),
                "nll_mean": mean(nll),
                "nll_stdev": sample_stdev(nll),
                "ece_mean": mean(ece),
                "ece_stdev": sample_stdev(ece),
                "parameters_mean": mean(params),
                "forward_flops_mean": mean(flops),
            }
        )
    return out


def plateau_rows(
    curves: list[dict[str, Any]],
    threshold: float,
    plateau_window: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in curves:
        grouped[(row["hidden"], row["domain"], row["model_kind"])].append(row)

    out: list[dict[str, Any]] = []
    for (hidden, domain, model_kind), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["epochs"]))
        best = max(rows, key=lambda row: float(row["top1_mean"]))
        last = rows[-1]
        prev = rows[-2] if len(rows) > 1 else last
        window = max(1, min(plateau_window, len(rows)))
        recent = rows[-window:]
        last_gain = float(last["top1_mean"]) - float(prev["top1_mean"])
        last_window_gain = float(last["top1_mean"]) - float(recent[0]["top1_mean"])
        best_gain = float(best["top1_mean"]) - float(rows[0]["top1_mean"])
        best_epoch_in_last_window = int(best["epochs"]) >= int(recent[0]["epochs"])
        out.append(
            {
                "schema": "crlp.multitask.tiny_net_plateau.v1",
                "report_version": 2,
                "date": date.today().isoformat(),
                "hidden": hidden,
                "domain": domain,
                "model_kind": model_kind,
                "best_epochs": int(best["epochs"]),
                "best_top1_mean": float(best["top1_mean"]),
                "best_top1_stdev": float(best["top1_stdev"]),
                "last_epochs": int(last["epochs"]),
                "last_top1_mean": float(last["top1_mean"]),
                "last_gain": last_gain,
                "last_window_gain": last_window_gain,
                "best_gain_from_first": best_gain,
                "plateau_threshold": threshold,
                "plateau_window": window,
                "plateau_method": "last_step_and_window_gain",
                "best_epoch_in_last_window": best_epoch_in_last_window,
                "plateau_detected": (
                    abs(last_gain) <= threshold
                    and abs(last_window_gain) <= threshold
                    and best_epoch_in_last_window
                ),
                "parameters_mean": float(last["parameters_mean"]),
                "forward_flops_mean": float(last["forward_flops_mean"]),
            }
        )
    return out


def width_rows(
    plateaus: list[dict[str, Any]],
    capacities: list[str],
    model_kinds: tuple[str, ...],
) -> list[dict[str, Any]]:
    by_key = {
        (row["hidden"], row["domain"], row["model_kind"]): row
        for row in plateaus
    }
    out: list[dict[str, Any]] = []
    for domain in DOMAINS:
        for model_kind in model_kinds:
            available = [
                by_key[(hidden, domain, model_kind)]
                for hidden in capacities
                if (hidden, domain, model_kind) in by_key
            ]
            if not available:
                continue
            best = max(available, key=lambda row: float(row["best_top1_mean"]))
            first = available[0]
            last = available[-1]
            out.append(
                {
                    "schema": "crlp.multitask.tiny_net_width_help.v1",
                    "report_version": 1,
                    "date": date.today().isoformat(),
                    "domain": domain,
                    "model_kind": model_kind,
                    "capacity_ladder": capacities,
                    "smallest_hidden": first["hidden"],
                    "largest_hidden": last["hidden"],
                    "best_hidden": best["hidden"],
                    "best_top1_mean": best["best_top1_mean"],
                    "largest_minus_smallest_top1": (
                        float(last["best_top1_mean"]) - float(first["best_top1_mean"])
                    ),
                    "larger_width_helped": best["hidden"] != first["hidden"],
                }
            )
    return out


def audit_verdict_rows(
    plateaus: list[dict[str, Any]],
    width: list[dict[str, Any]],
    capacities: list[str],
    seeds: list[int],
    epoch_grid: list[int],
    threshold: float,
    plateau_window: int,
    model_kinds: tuple[str, ...],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in plateaus:
        by_key[(str(row["domain"]), str(row["model_kind"]))].append(row)
    width_by_key = {(str(row["domain"]), str(row["model_kind"])): row for row in width}
    rows: list[dict[str, Any]] = []
    for domain in DOMAINS:
        for model_kind in model_kinds:
            group = by_key.get((domain, model_kind), [])
            hidden_seen = {str(row["hidden"]) for row in group}
            plateau_count = sum(1 for row in group if bool(row.get("plateau_detected")))
            width_row = width_by_key.get((domain, model_kind), {})
            capacity_coverage_complete = all(hidden in hidden_seen for hidden in capacities)
            plateau_complete = bool(group) and plateau_count == len(group)
            rows.append(
                {
                    "schema": "crlp.multitask.tiny_net_audit_verdict.v1",
                    "report_version": 1,
                    "date": date.today().isoformat(),
                    "domain": domain,
                    "model_kind": model_kind,
                    "capacity_ladder": capacities,
                    "capacity_count": len(group),
                    "capacity_coverage_complete": capacity_coverage_complete,
                    "restart_count": len(seeds),
                    "epoch_checkpoints": epoch_grid,
                    "epoch_checkpoint_count": len(epoch_grid),
                    "plateau_threshold": threshold,
                    "plateau_window": plateau_window,
                    "plateau_count": plateau_count,
                    "all_plateau_detected": plateau_complete,
                    "width_verdict_present": bool(width_row),
                    "best_hidden": width_row.get("best_hidden"),
                    "best_top1_mean": width_row.get("best_top1_mean"),
                    "larger_width_helped": width_row.get("larger_width_helped"),
                    "audit_ready": (
                        capacity_coverage_complete
                        and plateau_complete
                        and bool(width_row)
                    ),
                }
            )
    return rows


def write_report(
    path: Path,
    seeds: list[int],
    capacities: list[str],
    epoch_grid: list[int],
    steps_per_epoch: int,
    eval_samples: int,
    model_kinds: tuple[str, ...],
    plateaus: list[dict[str, Any]],
    width: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
) -> None:
    by_key = {
        (row["hidden"], row["domain"], row["model_kind"]): row
        for row in plateaus
    }
    width_by_key = {(row["domain"], row["model_kind"]): row for row in width}
    lines = [
        "# Fully Trained Tiny-Net Audit",
        "",
        f"Date: {date.today().isoformat()}",
        f"Seeds: `{', '.join(str(seed) for seed in seeds)}`",
        f"Capacities: `{', '.join(capacities)}`",
        f"Model variants: `{', '.join(model_kinds)}`",
        f"Epoch checkpoints: `{', '.join(str(epoch) for epoch in epoch_grid)}`",
        f"Steps per epoch: `{steps_per_epoch}`; eval samples: `{eval_samples}`",
        "",
        "## Best Checkpoints",
        "",
        "| Hidden | Model | Prime | Grid | Delayed | ARC |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for hidden in capacities:
        for model_kind in model_kinds:
            cells = []
            for domain in DOMAINS:
                row = by_key.get((hidden, domain, model_kind))
                if row is None:
                    cells.append("-")
                else:
                    cells.append(
                        f"{fmt_pct(float(row['best_top1_mean']))} @ {row['best_epochs']}"
                    )
            lines.append(f"| `{hidden}` | `{model_kind}` | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Plateau Flags",
            "",
            "| Hidden | Domain | Model | Last gain | Window gain | Plateau |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in plateaus:
        lines.append(
            "| `{hidden}` | `{domain}` | `{model}` | {gain} | {window_gain} | {plateau} |".format(
                hidden=row["hidden"],
                domain=row["domain"],
                model=row["model_kind"],
                gain=fmt_delta(float(row["last_gain"])),
                window_gain=fmt_delta(float(row["last_window_gain"])),
                plateau="yes" if row["plateau_detected"] else "no",
            )
        )

    lines.extend(
        [
            "",
            "## Width Verdict",
            "",
            "| Domain | Model | Best hidden | Largest-smallest top-1 | Larger helped |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for domain in DOMAINS:
        for model_kind in model_kinds:
            row = width_by_key.get((domain, model_kind))
            if row is None:
                continue
            lines.append(
                "| `{domain}` | `{model}` | `{best}` | {delta} | {helped} |".format(
                    domain=domain,
                    model=model_kind,
                    best=row["best_hidden"],
                    delta=fmt_delta(float(row["largest_minus_smallest_top1"])),
                    helped="yes" if row["larger_width_helped"] else "no",
                )
            )
    lines.extend(
        [
            "",
            "## Audit Verdict",
            "",
            "| Domain | Model | Coverage | Plateau rows | Best hidden | Audit ready |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in verdicts:
        lines.append(
            "| `{domain}` | `{model}` | {coverage} | {plateau}/{count} | `{best}` | {ready} |".format(
                domain=row["domain"],
                model=row["model_kind"],
                coverage="yes" if row["capacity_coverage_complete"] else "no",
                plateau=row["plateau_count"],
                count=row["capacity_count"],
                best=row.get("best_hidden") or "-",
                ready="yes" if row["audit_ready"] else "no",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/float/multitask_domain"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/float/tiny_net_plateau_audit"))
    parser.add_argument("--seeds", type=parse_int_csv, default=parse_int_csv(DEFAULT_SEEDS))
    parser.add_argument("--capacities", type=parse_csv, default=parse_csv(DEFAULT_CAPACITIES))
    parser.add_argument("--epoch-grid", type=parse_int_csv, default=parse_int_csv(DEFAULT_EPOCH_GRID))
    parser.add_argument("--steps-per-epoch", type=int, default=12000)
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--plateau-threshold", type=float, default=0.005)
    parser.add_argument("--plateau-window", type=int, default=3)
    parser.add_argument(
        "--no-task-heads",
        action="store_true",
        help="preserve the older shared/single_task audit without shared_task_heads rows",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    binary = args.binary if args.binary.is_absolute() else repo / args.binary
    model_kinds = ("shared", "single_task") if args.no_task_heads else MODEL_KINDS

    all_rows: list[dict[str, Any]] = []
    commands: list[str] = []
    for hidden in args.capacities:
        for seed in args.seeds:
            for epochs in args.epoch_grid:
                command = [
                    str(binary),
                    "--tasks",
                    "all",
                    "--eval-tasks",
                    "all",
                    "--epochs",
                    str(epochs),
                    "--steps-per-epoch",
                    str(args.steps_per_epoch),
                    "--eval-samples",
                    str(args.eval_samples),
                    "--hidden",
                    hidden,
                    "--seed",
                    str(seed),
                ]
                if not args.no_task_heads:
                    command.append("--task-heads")
                commands.append(" ".join(command))
                log_name = f"tiny_hidden_{hidden}_seed_{seed}_epochs_{epochs}.log"
                for row in run_json_command(command, cwd=repo, log_path=log_dir / log_name):
                    row["roadmap_item"] = "E-MT.9"
                    all_rows.append(row)

    raw_path = output_dir / "tiny_net_plateau.raw.jsonl"
    commands_path = output_dir / "tiny_net_plateau.commands.txt"
    curve_path = output_dir / "tiny_net_learning_curve.csv"
    plateau_path = output_dir / "tiny_net_plateau_summary.csv"
    plateau_jsonl_path = output_dir / "tiny_net_plateau_summary.jsonl"
    width_path = output_dir / "tiny_net_width_help.jsonl"
    verdict_path = output_dir / "tiny_net_audit_verdict.jsonl"
    report_path = output_dir / "tiny_net_plateau_audit.md"

    summary_rows = [
        row for row in all_rows if row.get("schema") == "crlp.multitask.summary.v1"
    ]
    curves = learning_curve_rows(summary_rows)
    plateaus = plateau_rows(curves, args.plateau_threshold, args.plateau_window)
    width = width_rows(plateaus, args.capacities, model_kinds)
    verdicts = audit_verdict_rows(
        plateaus,
        width,
        args.capacities,
        args.seeds,
        args.epoch_grid,
        args.plateau_threshold,
        args.plateau_window,
        model_kinds,
    )

    write_jsonl(raw_path, all_rows, compact=True)
    commands_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    write_csv(curve_path, curves)
    write_csv(plateau_path, plateaus)
    write_jsonl(plateau_jsonl_path, plateaus, compact=True)
    write_jsonl(width_path, width, compact=True)
    write_jsonl(verdict_path, verdicts, compact=True)
    write_report(
        report_path,
        args.seeds,
        args.capacities,
        args.epoch_grid,
        args.steps_per_epoch,
        args.eval_samples,
        model_kinds,
        plateaus,
        width,
        verdicts,
    )
    print(f"wrote {report_path}")
    print(f"wrote {verdict_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
