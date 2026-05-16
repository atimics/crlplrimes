#!/usr/bin/env python3
"""Run cross-task neurosymbolic tuning sweeps and normalize the results."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


JSON_SCHEMAS = {
    "crlp.experiment_schema.v1",
    "crlp.compare.v1",
    "crlp.rule_search.summary.v1",
    "crlp.multitask.summary.v1",
}


@dataclass(frozen=True)
class CommandResult:
    command_id: str
    command: list[str]
    log_path: Path
    rows: list[dict[str, object]]


def parse_csv_text(text: str) -> list[str]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def parse_csv_ints(text: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive integer values")
    return values


def parse_csv_reals(text: str) -> list[float]:
    try:
        values = [float(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not values or any(value < 0.0 for value in values):
        raise argparse.ArgumentTypeError("expected non-negative real values")
    return values


def shell_join(command: list[str]) -> str:
    return " ".join(command)


def run_command(
    repo: Path,
    command: list[str],
    log_path: Path,
    command_id: str,
    context: dict[str, object],
) -> CommandResult:
    print("$ " + shell_join(command), flush=True)
    result = subprocess.run(command, cwd=repo, text=True, capture_output=True, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"{shell_join(command)} failed; see {log_path}")

    rows: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("schema") not in JSON_SCHEMAS:
            continue
        row.update(context)
        row["command_id"] = command_id
        row["command"] = shell_join(command)
        row["log"] = str(log_path)
        rows.append(row)
    return CommandResult(command_id, command, log_path, rows)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def metric_value(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def normalize_row(row: dict[str, object]) -> dict[str, str]:
    schema = str(row.get("schema", ""))
    task = str(row.get("domain", row.get("task", "")))
    family = str(row.get("tuning_family", ""))
    objective_name = ""
    objective = ""
    larger_is_better = "yes"

    if schema == "crlp.rule_search.summary.v1":
        if task == "prime_gap" and metric_value(row, "expanded_saved_pct") is not None:
            objective_name = "expanded_saved_pct"
            objective = f"{metric_value(row, 'expanded_saved_pct'):.8f}"
        elif task == "grid_path" and metric_value(row, "test_rejects") is not None:
            objective_name = "test_rejects"
            objective = f"{metric_value(row, 'test_rejects'):.8f}"
        else:
            objective_name = "accepted_rules"
            objective = f"{metric_value(row, 'accepted_rules') or 0.0:.8f}"
    elif schema == "crlp.compare.v1":
        if metric_value(row, "decision_utility") is not None:
            objective_name = "decision_utility"
            objective = f"{metric_value(row, 'decision_utility'):.8f}"
        elif metric_value(row, "accuracy") is not None:
            objective_name = "accuracy"
            objective = f"{metric_value(row, 'accuracy'):.8f}"
        else:
            objective_name = "nll"
            objective = f"{metric_value(row, 'nll') or 0.0:.8f}"
            larger_is_better = "no"
    elif schema == "crlp.multitask.summary.v1":
        if metric_value(row, "top1") is not None:
            objective_name = "top1"
            objective = f"{metric_value(row, 'top1'):.8f}"
        elif metric_value(row, "accuracy") is not None:
            objective_name = "accuracy"
            objective = f"{metric_value(row, 'accuracy'):.8f}"

    config_value = str(row.get("tuning_config", row.get("model_kind", "")))
    if schema == "crlp.multitask.summary.v1":
        task_set = str(row.get("task_set", ""))
        model_kind = str(row.get("model_kind", ""))
        feature_envelope = str(row.get("feature_envelope", ""))
        parts = [part for part in (task_set, model_kind, feature_envelope) if part]
        config_value = ":".join(parts)

    return {
        "family": family,
        "schema": schema,
        "task": task,
        "strategy": str(row.get("selection_strategy", row.get("evaluation_mode", row.get("model_kind", "")))),
        "config": config_value,
        "budget": str(row.get("max_rules", "")),
        "seed": str(row.get("selection_seed", row.get("benchmark_seed", row.get("seed", "")))),
        "objective_name": objective_name,
        "objective": objective,
        "larger_is_better": larger_is_better,
        "false_rejections": str(row.get("false_rejections", row.get("test_false_rejections", ""))),
        "accuracy": "" if metric_value(row, "accuracy") is None else f"{metric_value(row, 'accuracy'):.8f}",
        "ece": "" if metric_value(row, "ece") is None else f"{metric_value(row, 'ece'):.8f}",
        "utility": ""
        if metric_value(row, "decision_utility") is None
        else f"{metric_value(row, 'decision_utility'):.8f}",
        "expanded_saved_pct": ""
        if metric_value(row, "expanded_saved_pct") is None
        else f"{metric_value(row, 'expanded_saved_pct'):.8f}",
        "accepted_rules": str(row.get("accepted_rules", "")),
        "command_id": str(row.get("command_id", "")),
        "log": str(row.get("log", "")),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "family",
        "schema",
        "task",
        "strategy",
        "config",
        "budget",
        "seed",
        "objective_name",
        "objective",
        "larger_is_better",
        "false_rejections",
        "accuracy",
        "ece",
        "utility",
        "expanded_saved_pct",
        "accepted_rules",
        "command_id",
        "log",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize_row(row))


def markdown_table(headers: list[str], body: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def best_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in rows:
        normalized = normalize_row(row)
        if not normalized["objective"]:
            continue
        if normalized["false_rejections"] not in {"", "0"}:
            continue
        key = (normalized["family"], normalized["task"], normalized["config"])
        value = float(normalized["objective"])
        prior = grouped.get(key)
        if prior is None:
            grouped[key] = row
            continue
        prior_norm = normalize_row(prior)
        prior_value = float(prior_norm["objective"])
        larger = normalized["larger_is_better"] == "yes"
        if (larger and value > prior_value) or (not larger and value < prior_value):
            grouped[key] = row
    return [grouped[key] for key in sorted(grouped)]


def write_summary(path: Path, rows: list[dict[str, object]], profile: str) -> None:
    normalized = [normalize_row(row) for row in rows if normalize_row(row)["objective"]]
    best = [normalize_row(row) for row in best_rows(rows)]
    lines = [
        "# Neurosymbolic Tuning",
        "",
        f"Profile: `{profile}`.",
        "",
        "This run sweeps multiple verifier-backed tasks, the shared multitask scorer, and writes raw JSONL plus a normalized CSV.",
        "",
        "## Best Zero-False-Rejection Rows",
        "",
    ]
    lines.append(
        markdown_table(
            [
                "family",
                "task",
                "strategy",
                "config",
                "budget",
                "objective",
                "value",
                "command",
            ],
            [
                [
                    row["family"],
                    row["task"],
                    row["strategy"],
                    row["config"],
                    row["budget"],
                    row["objective_name"],
                    row["objective"],
                    row["command_id"],
                ]
                for row in best
            ],
        )
        if best
        else "_No objective rows captured._"
    )
    lines.extend(["", "## All Normalized Rows", ""])
    lines.append(
        markdown_table(
            ["family", "task", "strategy", "config", "budget", "objective", "value", "false rejects"],
            [
                [
                    row["family"],
                    row["task"],
                    row["strategy"],
                    row["config"],
                    row["budget"],
                    row["objective_name"],
                    row["objective"],
                    row["false_rejections"],
                ]
                for row in normalized
            ],
        )
        if normalized
        else "_No normalized rows captured._"
    )
    lines.extend(
        [
            "",
            "Files:",
            "",
            "- `raw.jsonl`: full task-native rows",
            "- `normalized.csv`: cross-task score table",
            "- `commands.txt`: exact commands",
            "- `logs/`: stdout/stderr per run",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def profile_defaults(profile: str) -> dict[str, object]:
    if profile == "full":
        return {
            "prime_budgets": [8, 16, 32],
            "prime_max_marginal_states": 4096,
            "prime_max_train_states": 30000,
            "prime_max_test_states": 30000,
            "prime_train_limit": 1000000,
            "prime_test_limit": 3000000,
            "prime_candidate_moduli_range": "3:255",
            "grid_budgets": [1, 2, 3, 5],
            "replay_epochs": [2, 5, 8],
            "claim_soft_weights": [0.0, 0.35, 0.7],
            "trace_samples": 2000,
            "multitask_epochs": 5,
            "multitask_steps_per_epoch": 9000,
            "multitask_eval_samples": 1000,
        }
    return {
        "prime_budgets": [8, 16],
        "prime_max_marginal_states": 512,
        "prime_max_train_states": 1000,
        "prime_max_test_states": 1000,
        "prime_train_limit": 100000,
        "prime_test_limit": 200000,
        "prime_candidate_moduli_range": "3:31",
        "grid_budgets": [1, 2, 3, 5],
        "replay_epochs": [2, 5],
        "claim_soft_weights": [0.0, 0.35],
        "trace_samples": 1000,
        "multitask_epochs": 3,
        "multitask_steps_per_epoch": 3000,
        "multitask_eval_samples": 500,
    }


def add_profile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--real", default="float")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prime-proposers", type=parse_csv_text,
                        default=parse_csv_text("closed-loop,neural,grammar,structural,validation-oracle"))
    parser.add_argument("--grid-proposers", type=parse_csv_text,
                        default=parse_csv_text("neural,grammar,structural,validation-oracle,random"))
    parser.add_argument("--grid-feature-sets", type=parse_csv_text,
                        default=parse_csv_text("with-priors"),
                        help="CSV grid replay feature sets. Trace/replay currently supports with-priors.")
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--skip-build", action="store_true")


def run_schema_rows(repo: Path, output_dir: Path, binaries: dict[str, Path]) -> list[CommandResult]:
    results: list[CommandResult] = []
    for task, binary in binaries.items():
        command_id = f"schema_{task}"
        results.append(
            run_command(
                repo,
                [str(binary), "--schema"],
                output_dir / "logs" / f"{command_id}.log",
                command_id,
                {"tuning_family": "schema", "tuning_config": task},
            )
        )
    return results


def run_prime_rule_sweeps(
    repo: Path,
    output_dir: Path,
    binary: Path,
    args: argparse.Namespace,
    defaults: dict[str, object],
) -> list[CommandResult]:
    results: list[CommandResult] = []
    for grammar in ["candidate", "gap"]:
        for budget in defaults["prime_budgets"]:
            for proposer in args.prime_proposers:
                command = [
                    str(binary),
                    "--train-limit",
                    str(defaults["prime_train_limit"]),
                    "--test-limit",
                    str(defaults["prime_test_limit"]),
                    "--max-train-states",
                    str(defaults["prime_max_train_states"]),
                    "--max-test-states",
                    str(defaults["prime_max_test_states"]),
                    "--max-marginal-states",
                    str(defaults["prime_max_marginal_states"]),
                    "--max-rules",
                    str(budget),
                    "--proposer",
                    proposer,
                ]
                if grammar == "candidate":
                    command.extend([
                        "--candidate-only",
                        "--unconditional-only",
                        "--moduli-range",
                        str(defaults["prime_candidate_moduli_range"]),
                    ])
                else:
                    command.append("--gap-only")
                command_id = f"prime_{grammar}_budget-{budget}_{proposer}"
                results.append(
                    run_command(
                        repo,
                        command,
                        output_dir / "logs" / f"{command_id}.log",
                        command_id,
                        {
                            "tuning_family": "rule_search",
                            "tuning_config": f"{grammar}:budget={budget}",
                        },
                    )
                )
    return results


def run_grid_rule_sweeps(
    repo: Path,
    output_dir: Path,
    binary: Path,
    args: argparse.Namespace,
    defaults: dict[str, object],
) -> list[CommandResult]:
    results: list[CommandResult] = []
    for budget in defaults["grid_budgets"]:
        for proposer in args.grid_proposers:
            command = [
                str(binary),
                "--rule-search",
                "--max-rules",
                str(budget),
                "--proposer",
                proposer,
            ]
            command_id = f"grid_rule_budget-{budget}_{proposer}"
            results.append(
                run_command(
                    repo,
                    command,
                    output_dir / "logs" / f"{command_id}.log",
                    command_id,
                    {
                        "tuning_family": "rule_search",
                        "tuning_config": f"legality:budget={budget}",
                    },
                )
            )
    return results


def run_delayed_replay_sweeps(
    repo: Path,
    output_dir: Path,
    binary: Path,
    defaults: dict[str, object],
) -> list[CommandResult]:
    artifacts = output_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    parent = artifacts / "delayed_parent.nnckpt"
    trace = artifacts / "delayed_trace.jsonl"
    results = [
        run_command(
            repo,
            [
                str(binary),
                "--save",
                str(parent),
                "--trace",
                str(trace),
                "--trace-train",
                "--trace-samples",
                str(defaults["trace_samples"]),
            ],
            output_dir / "logs" / "delayed_parent.log",
            "delayed_parent",
            {"tuning_family": "replay", "tuning_config": "parent"},
        )
    ]
    for epochs in defaults["replay_epochs"]:
        for soft_weight in defaults["claim_soft_weights"]:
            child = artifacts / f"delayed_replay_e{epochs}_soft{soft_weight:.2f}.nnckpt"
            config = f"epochs={epochs}:hard=1.00:soft={soft_weight:.2f}"
            results.append(
                run_command(
                    repo,
                    [
                        str(binary),
                        "--replay",
                        str(trace),
                        "--out",
                        str(child),
                        "--epochs",
                        str(epochs),
                        "--hard-weight",
                        "1.0",
                        "--soft-weight",
                        f"{soft_weight:.4f}",
                    ],
                    output_dir / "logs" / f"delayed_replay_e{epochs}_soft{soft_weight:.2f}.log",
                    f"delayed_replay_e{epochs}_soft{soft_weight:.2f}",
                    {"tuning_family": "replay", "tuning_config": config},
                )
            )
            results.append(
                run_command(
                    repo,
                    [str(binary), "--compare", str(parent), str(child)],
                    output_dir / "logs" / f"delayed_compare_e{epochs}_soft{soft_weight:.2f}.log",
                    f"delayed_compare_e{epochs}_soft{soft_weight:.2f}",
                    {"tuning_family": "replay", "tuning_config": config},
                )
            )
    return results


def run_grid_replay_sweeps(
    repo: Path,
    output_dir: Path,
    binary: Path,
    args: argparse.Namespace,
    defaults: dict[str, object],
) -> list[CommandResult]:
    artifacts = output_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    results: list[CommandResult] = []
    for feature_set in args.grid_feature_sets:
        parent = artifacts / f"grid_parent_{feature_set}.nnckpt"
        trace = artifacts / f"grid_trace_{feature_set}.jsonl"
        results.append(
            run_command(
                repo,
                [
                    str(binary),
                    "--feature-set",
                    feature_set,
                    "--save",
                    str(parent),
                    "--trace",
                    str(trace),
                    "--trace-train",
                    "--trace-samples",
                    str(defaults["trace_samples"]),
                ],
                output_dir / "logs" / f"grid_parent_{feature_set}.log",
                f"grid_parent_{feature_set}",
                {"tuning_family": "replay", "tuning_config": f"grid_parent:{feature_set}"},
            )
        )
        for epochs in defaults["replay_epochs"]:
            child = artifacts / f"grid_replay_{feature_set}_e{epochs}.nnckpt"
            config = f"feature_set={feature_set}:epochs={epochs}"
            results.append(
                run_command(
                    repo,
                    [str(binary), "--replay", str(trace), "--out", str(child), "--epochs", str(epochs)],
                    output_dir / "logs" / f"grid_replay_{feature_set}_e{epochs}.log",
                    f"grid_replay_{feature_set}_e{epochs}",
                    {"tuning_family": "replay", "tuning_config": config},
                )
            )
            results.append(
                run_command(
                    repo,
                    [str(binary), "--compare", str(parent), str(child)],
                    output_dir / "logs" / f"grid_compare_{feature_set}_e{epochs}.log",
                    f"grid_compare_{feature_set}_e{epochs}",
                    {"tuning_family": "replay", "tuning_config": config},
                )
            )
    return results


def run_multitask_sweep(
    repo: Path,
    output_dir: Path,
    binary: Path,
    defaults: dict[str, object],
) -> list[CommandResult]:
    command_id = "multitask_shared_action_scorer"
    command = [
        str(binary),
        "--epochs",
        str(defaults["multitask_epochs"]),
        "--steps-per-epoch",
        str(defaults["multitask_steps_per_epoch"]),
        "--eval-samples",
        str(defaults["multitask_eval_samples"]),
    ]
    return [
        run_command(
            repo,
            command,
            output_dir / "logs" / f"{command_id}.log",
            command_id,
            {"tuning_family": "multitask"},
        )
    ]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    add_profile_arguments(parser)
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    defaults = profile_defaults(args.profile)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("build") / args.real / "neurosymbolic_tuning"
    output_dir = (repo / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        subprocess.run(["make", f"REAL={args.real}", "all"], cwd=repo, check=True)

    build_dir = repo / "build" / args.real
    binaries = {
        "prime_gap": build_dir / "prime_gap",
        "grid_path": build_dir / "grid_path",
        "delayed_claim": build_dir / "delayed_claim",
    }
    prime_rule_search = build_dir / "prime_rule_search"
    multitask_domain = build_dir / "multitask_domain"

    command_results: list[CommandResult] = []
    command_results.extend(run_schema_rows(repo, output_dir, binaries))
    command_results.extend(run_prime_rule_sweeps(repo, output_dir, prime_rule_search, args, defaults))
    command_results.extend(run_grid_rule_sweeps(repo, output_dir, binaries["grid_path"], args, defaults))
    command_results.extend(run_multitask_sweep(repo, output_dir, multitask_domain, defaults))
    if not args.skip_replay:
        command_results.extend(run_delayed_replay_sweeps(repo, output_dir, binaries["delayed_claim"], defaults))
        command_results.extend(run_grid_replay_sweeps(repo, output_dir, binaries["grid_path"], args, defaults))

    rows = [row for result in command_results for row in result.rows]
    write_jsonl(output_dir / "raw.jsonl", rows)
    write_csv(output_dir / "normalized.csv", rows)
    write_summary(output_dir / "summary.md", rows, args.profile)
    (output_dir / "commands.txt").write_text(
        "\n".join(shell_join(result.command) for result in command_results) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
