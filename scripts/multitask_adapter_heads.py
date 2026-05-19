#!/usr/bin/env python3
"""Run the R-MT.4 shared-trunk plus per-task-head multitask comparison."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any

from crlp.cli import parse_seeds
from crlp.io import write_csv, write_jsonl
from crlp.run import run_json_command
from crlp.stats import fmt_delta, fmt_float, fmt_pct, mean, sample_stdev


DEFAULT_SEEDS = "20240516,20240517,20240518"
DOMAINS = ["prime_gap", "grid_path", "delayed_claim", "arc_agi"]
MODEL_ORDER = ["shared", "shared_task_heads", "single_task"]
EVAL_SUITES = (("fixed", "0520fde7"), ("ood", "ood"))


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("schema") != "crlp.multitask.summary.v1":
            continue
        eval_suite = str(row.get("adapter_eval_suite", "fixed"))
        domain = str(row["domain"])
        if eval_suite != "fixed" and domain != "arc_agi":
            continue
        grouped.setdefault((eval_suite, domain, str(row["model_kind"])), []).append(row)

    output: list[dict[str, Any]] = []
    for (eval_suite, domain, model_kind), items in sorted(grouped.items()):
        top1 = [float(row["top1"]) for row in items]
        nll = [float(row["nll"]) for row in items]
        ece = [float(row["ece"]) for row in items]
        updates = [float(row["train_updates"]) for row in items]
        utility = [float(row["decision_utility"]) for row in items]
        params = [float(row["parameters"]) for row in items]
        flops = [float(row["forward_flops"]) for row in items]
        output.append(
            {
                "eval_suite": eval_suite,
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
        (str(row["eval_suite"]), str(row["domain"]), str(row["model_kind"])): row
        for row in summary
    }
    output: list[dict[str, Any]] = []
    for eval_suite, _suite_arg in EVAL_SUITES:
        domains = DOMAINS if eval_suite == "fixed" else ["arc_agi"]
        for domain in domains:
            headed = by_key.get((eval_suite, domain, "shared_task_heads"))
            shared = by_key.get((eval_suite, domain, "shared"))
            single = by_key.get((eval_suite, domain, "single_task"))
            if headed is None:
                continue
            output.append(
                {
                    "eval_suite": eval_suite,
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
                    "shared_parameters_mean": None if shared is None else shared["parameters_mean"],
                    "single_parameters_mean": None if single is None else single["parameters_mean"],
                    "headed_forward_flops_mean": headed["forward_flops_mean"],
                    "shared_forward_flops_mean": None if shared is None else shared["forward_flops_mean"],
                    "single_forward_flops_mean": None if single is None else single["forward_flops_mean"],
                    "headed_train_updates_mean": headed["train_updates_mean"],
                    "shared_train_updates_mean": None if shared is None else shared["train_updates_mean"],
                    "single_train_updates_mean": None if single is None else single["train_updates_mean"],
                }
            )
    return output


def row_float(row: dict[str, Any] | None, key: str) -> float:
    if row is None:
        return float("nan")
    value = row.get(key)
    if value in ("", None):
        return float("nan")
    return float(value)


def get_delta_row(deltas: list[dict[str, Any]], eval_suite: str, domain: str) -> dict[str, Any] | None:
    for row in deltas:
        if row.get("eval_suite") == eval_suite and row.get("domain") == domain:
            return row
    return None


def decision_rows(deltas: list[dict[str, Any]], transfer_regression_margin: float) -> list[dict[str, Any]]:
    fixed_arc = get_delta_row(deltas, "fixed", "arc_agi")
    ood_arc = get_delta_row(deltas, "ood", "arc_agi")
    delayed = get_delta_row(deltas, "fixed", "delayed_claim")
    grid = get_delta_row(deltas, "fixed", "grid_path")
    required = [fixed_arc, ood_arc, delayed, grid]
    seeds = min((int(row["seeds"]) for row in required if row is not None), default=0)
    fixed_arc_gain = row_float(fixed_arc, "top1_delta_headed_minus_shared")
    delayed_delta = row_float(delayed, "top1_delta_headed_minus_shared")
    grid_delta = row_float(grid, "top1_delta_headed_minus_shared")
    ood_arc_top1_vs_single = row_float(ood_arc, "top1_delta_headed_minus_single")
    ood_arc_nll_vs_single = row_float(ood_arc, "nll_delta_headed_minus_single")
    fixed_arc_gate = fixed_arc_gain > 0.0
    delayed_gate = delayed_delta >= -transfer_regression_margin
    grid_gate = grid_delta >= -transfer_regression_margin
    ood_gate = ood_arc_top1_vs_single >= 0.0 and ood_arc_nll_vs_single <= 0.0
    complete = all(row is not None for row in required)
    promote = complete and seeds >= 3 and fixed_arc_gate and delayed_gate and grid_gate and ood_gate
    return [
        {
            "schema": "crlp.multitask.adapter_heads_decision.v1",
            "report_version": 1,
            "date": date.today().isoformat(),
            "decision": "promote_shared_trunk_heads" if promote else "diagnostic_only",
            "promote_shared_trunk_heads": promote,
            "complete_required_rows": complete,
            "seeds": seeds,
            "transfer_regression_margin": transfer_regression_margin,
            "fixed_arc_top1_delta_heads_minus_flat_shared": fixed_arc_gain,
            "fixed_arc_improves_vs_flat_shared": fixed_arc_gate,
            "delayed_claim_top1_delta_heads_minus_flat_shared": delayed_delta,
            "delayed_claim_transfer_preserved": delayed_gate,
            "grid_path_top1_delta_heads_minus_flat_shared": grid_delta,
            "grid_path_not_materially_regressed": grid_gate,
            "ood_arc_top1_delta_heads_minus_single_task": ood_arc_top1_vs_single,
            "ood_arc_nll_delta_heads_minus_single_task": ood_arc_nll_vs_single,
            "ood_arc_preserved_vs_single_task": ood_gate,
        }
    ]


def write_report(
    path: Path,
    seeds: list[int],
    epochs: int,
    steps_per_epoch: int,
    eval_samples: int,
    summary: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> None:
    by_key = {
        (str(row["eval_suite"]), str(row["domain"]), str(row["model_kind"])): row
        for row in summary
    }
    delta_by_key = {(str(row["eval_suite"]), str(row["domain"])): row for row in deltas}
    decision = decisions[0] if decisions else {}
    lines = [
        "# Multitask Adapter-Headed Sharing",
        "",
        f"Seeds: `{', '.join(str(seed) for seed in seeds)}`",
        f"Budget: `{epochs}` epochs x `{steps_per_epoch}` steps = `{epochs * steps_per_epoch}` total shared updates.",
        f"Evaluation samples: `{eval_samples}` per domain row.",
        "",
        "This run compares the flat shared scalar scorer, a shared trunk with per-domain scalar heads, and matched single-task baselines under the same 96-feature envelope.",
        "",
        "## Fixed-Suite Top-1",
        "",
        "| Domain | Shared | Task heads | Single | Heads - shared | Heads - single |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for domain in DOMAINS:
        shared = by_key.get(("fixed", domain, "shared"))
        headed = by_key.get(("fixed", domain, "shared_task_heads"))
        single = by_key.get(("fixed", domain, "single_task"))
        delta = delta_by_key.get(("fixed", domain))
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
            "## ARC OOD Top-1",
            "",
            "| Suite | Shared | Task heads | Single | Heads - shared | Heads - single |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    shared = by_key.get(("ood", "arc_agi", "shared"))
    headed = by_key.get(("ood", "arc_agi", "shared_task_heads"))
    single = by_key.get(("ood", "arc_agi", "single_task"))
    delta = delta_by_key.get(("ood", "arc_agi"))
    lines.append(
        "| `arc_ood` | "
        f"{fmt_pct(float(shared['top1_mean'])) if shared else '-'} | "
        f"{fmt_pct(float(headed['top1_mean'])) if headed else '-'} | "
        f"{fmt_pct(float(single['top1_mean'])) if single else '-'} | "
        f"{fmt_delta(float(delta['top1_delta_headed_minus_shared'])) if delta and delta['top1_delta_headed_minus_shared'] is not None else '-'} | "
        f"{fmt_delta(float(delta['top1_delta_headed_minus_single'])) if delta and delta['top1_delta_headed_minus_single'] is not None else '-'} |"
    )

    lines.extend(
        [
            "",
            "## ARC And Transfer Guardrail",
            "",
            "| Suite | Domain | Model | Top-1 | NLL | ECE | Updates | Params | FLOPs |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for eval_suite, domain in [("fixed", "arc_agi"), ("ood", "arc_agi"), ("fixed", "delayed_claim"), ("fixed", "grid_path")]:
        for model in MODEL_ORDER:
            row = by_key.get((eval_suite, domain, model))
            if row is None:
                continue
            lines.append(
                f"| `{eval_suite}` | `{domain}` | `{model}` | {fmt_pct(float(row['top1_mean']))} | "
                f"{fmt_float(float(row['nll_mean']))} | {fmt_float(float(row['ece_mean']))} | "
                f"{float(row['train_updates_mean']):.0f} | {float(row['parameters_mean']):.0f} | "
                f"{float(row['forward_flops_mean']):.0f} |"
            )

    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            f"- Decision: `{decision.get('decision', 'missing')}`.",
            "- Promote this architecture only if task heads improve fixed ARC versus the flat shared scorer, preserve delayed-claim transfer, avoid a material grid regression, and preserve ARC OOD versus the single-task baseline.",
            "- If task heads help fixed ARC but hurt OOD or transfer, keep them as diagnostic evidence rather than the default architecture.",
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
    parser.add_argument("--arc-ood-task", default="all")
    parser.add_argument("--transfer-regression-margin", type=float, default=0.005)
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
        for eval_suite, suite_arg in EVAL_SUITES:
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
                "--arc-eval-suite",
                suite_arg,
                "--task-heads",
            ]
            if eval_suite == "ood":
                command.extend(["--arc-ood-task", args.arc_ood_task])
            if args.no_task_baselines:
                command.append("--no-task-baselines")
            commands.append(" ".join(command))
            log_name = f"rmt4_task_heads_{eval_suite}_seed_{seed}.log"
            for row in run_json_command(command, cwd=repo, log_path=log_dir / log_name):
                row["roadmap_item"] = "R-MT.4"
                row["adapter_eval_suite"] = eval_suite
                rows.append(row)

    raw_path = output_dir / "rmt4_adapter_heads.raw.jsonl"
    write_jsonl(raw_path, rows)
    (output_dir / "rmt4_adapter_heads.commands.txt").write_text(
        "\n".join(commands) + "\n",
        encoding="utf-8",
    )

    summary = summarize(rows)
    deltas = attach_deltas(summary)
    decisions = decision_rows(deltas, args.transfer_regression_margin)
    write_csv(output_dir / "rmt4_adapter_heads_summary.csv", summary, empty_text=True)
    write_csv(output_dir / "rmt4_adapter_heads_deltas.csv", deltas, empty_text=True)
    write_jsonl(output_dir / "rmt4_adapter_heads_decision.jsonl", decisions)
    write_report(
        output_dir / "rmt4_adapter_heads.md",
        args.seeds,
        args.epochs,
        args.steps_per_epoch,
        args.eval_samples,
        summary,
        deltas,
        decisions,
    )
    print(f"wrote {output_dir / 'rmt4_adapter_heads.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
