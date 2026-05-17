#!/usr/bin/env python3
"""Run canonical verifier-task evaluations and save parseable result snapshots."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


COMMANDS = (
    ("experiment_schemas", ("experiment-schemas",)),
    ("arc_agi", ("arc-agi",)),
    ("arc_agi_induction_smoke", ("arc-agi-induction-smoke",)),
    ("delayed_claim", ("delayed-claim-eval",)),
    ("grid_path", ("grid-path-eval",)),
    ("prime_rule_search", ("prime-rule-search",)),
    ("prime_rule_search_baselines", ("prime-rule-search-baselines",)),
    ("program_synthesis_baselines", ("program-synthesis-baselines",)),
    ("prime_gap", ("prime-gap-eval",)),
)


def run_make(
    repo: Path,
    real: str,
    target: str,
    log_path: Path,
    make_vars: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    command = ["make", f"REAL={real}"]
    for key, value in (make_vars or {}).items():
        command.append(f"{key}={value}")
    command.append(target)
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=repo, text=True, capture_output=True, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed; see {log_path}")

    rows: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("schema") in {
            "crlp.experiment_schema.v1",
            "crlp.compare.v1",
            "crlp.rule_search.summary.v1",
            "crlp.program_proposer.summary.v1",
            "crlp.arc_transform.summary.v1",
            "crlp.arc_loader.summary.v1",
            "crlp.hypothesis.evidence.v1",
            "crlp.arc_induction.parity.v1",
        }:
            row["source_target"] = target
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def dedupe_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped: list[dict[str, object]] = []
    seen_rule_search: set[tuple[object, ...]] = set()
    for row in rows:
        if row.get("schema") != "crlp.rule_search.summary.v1":
            deduped.append(row)
            continue
        key = (
            row.get("task"),
            row.get("rule_set_hash"),
            row.get("grammar_version"),
            row.get("max_rules"),
            row.get("selection_strategy"),
            row.get("selection_seed"),
            row.get("softmax_group_size"),
            row.get("closed_loop_hidden"),
            row.get("unified_scorer_hidden"),
            row.get("marginal_subsample_size"),
            row.get("proposed_rules"),
            row.get("verified_rules"),
            row.get("accepted_rules"),
        )
        if key in seen_rule_search:
            continue
        seen_rule_search.add(key)
        deduped.append(row)
    return deduped


def fmt_percent(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, (int, float)):
        return ""
    return f"{100.0 * float(value):.2f}%"


def fmt_number(row: dict[str, object], key: str, digits: int = 4) -> str:
    value = row.get(key)
    if not isinstance(value, (int, float)):
        return ""
    return f"{float(value):.{digits}f}"


def fmt_bool(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    if isinstance(value, bool):
        return "yes" if value else "no"
    return ""


def fmt_text(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    return "" if value is None else str(value)


def fmt_rule_filters(row: dict[str, object]) -> str:
    filters: list[str] = []
    if row.get("include_candidate_filters") is True:
        filters.append("candidate")
    if row.get("include_gap_filters") is True:
        filters.append("gap")
    return "+".join(filters)


def markdown_table(rows: list[list[str]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_summary(path: Path, rows: list[dict[str, object]], real: str) -> None:
    schemas = [row for row in rows if row.get("schema") == "crlp.experiment_schema.v1"]
    delayed = [
        row
        for row in rows
        if row.get("task") == "delayed_claim" and row.get("schema") == "crlp.compare.v1"
    ]
    arc = [
        row
        for row in rows
        if row.get("task") == "arc_agi" and row.get("schema") == "crlp.compare.v1"
    ]
    arc_transforms = [
        row
        for row in rows
        if row.get("task") == "arc_agi" and row.get("schema") == "crlp.arc_transform.summary.v1"
    ]
    arc_loaders = [
        row
        for row in rows
        if row.get("task") == "arc_agi" and row.get("schema") == "crlp.arc_loader.summary.v1"
    ]
    arc_program_proposers = [
        row
        for row in rows
        if row.get("domain") == "arc_agi"
        and row.get("schema") == "crlp.program_proposer.summary.v1"
    ]
    arc_hypotheses = [
        row
        for row in rows
        if row.get("task") == "arc_agi"
        and row.get("schema") == "crlp.hypothesis.evidence.v1"
    ]
    arc_induction_parity = [
        row
        for row in rows
        if row.get("task") == "arc_agi"
        and row.get("schema") == "crlp.arc_induction.parity.v1"
    ]
    grid = [
        row
        for row in rows
        if row.get("task") == "grid_path" and row.get("schema") == "crlp.compare.v1"
    ]
    rule_search: list[dict[str, object]] = []
    seen_rule_search: set[tuple[object, ...]] = set()
    for row in rows:
        if row.get("schema") != "crlp.rule_search.summary.v1":
            continue
        key = (
            row.get("rule_set_hash"),
            row.get("grammar_version"),
            row.get("max_rules"),
            row.get("selection_strategy"),
            row.get("softmax_group_size"),
            row.get("closed_loop_hidden"),
            row.get("unified_scorer_hidden"),
            row.get("marginal_subsample_size"),
            row.get("accepted_rules"),
            row.get("accepted_candidate_rules"),
            row.get("accepted_gap_rules"),
        )
        if key in seen_rule_search:
            continue
        seen_rule_search.add(key)
        rule_search.append(row)
    prime = [
        row
        for row in rows
        if row.get("task") == "prime_gap" and row.get("schema") == "crlp.compare.v1"
    ]

    lines = [
        "# Latest Verifier Results",
        "",
        f"Scalar type: `{real}`.",
        "",
        "Generated by `make results` via `scripts/capture_results.py`.",
        "",
        "## Experiment Schema",
        "",
    ]
    lines.append(
        markdown_table(
            [
                [
                    str(row.get("task", "")),
                    str(row.get("leakage_policy", "")),
                    str(row.get("action_space", "")),
                    str(row.get("exact_verifier", "")),
                    str(row.get("rule_grammar", "")),
                    str(row.get("certificate_checker", "")),
                    str(row.get("proposer_mode", "")),
                    (
                        str(row.get("feature_audit", {}).get("enabled_structural_prior_features", ""))
                        if isinstance(row.get("feature_audit"), dict)
                        else ""
                    ),
                ]
                for row in schemas
            ],
            [
                "task",
                "leakage",
                "actions",
                "verifier",
                "rule grammar",
                "certificate",
                "proposer mode",
                "enabled priors",
            ],
        )
        if schemas
        else "_No experiment schema rows captured._"
    )
    lines.extend(
        [
            "",
            "## ARC-AGI",
            "",
        ]
    )
    lines.append(
        markdown_table(
            [
                [
                    str(row.get("model_hash", "unknown"))[:12],
                    str(row.get("task_instance", "")),
                    str(row.get("evaluation_mode", "")),
                    str(row.get("leakage_policy", "")),
                    fmt_percent(row, "top1"),
                    fmt_percent(row, "top5"),
                    fmt_percent(row, "top10"),
                    fmt_number(row, "nll"),
                    fmt_number(row, "ece"),
                    fmt_number(row, "mean_rank", 3),
                    fmt_number(row, "proposal_steps", 3),
                    fmt_number(row, "baseline_steps", 3),
                ]
                for row in arc
            ],
            [
                "model",
                "task",
                "mode",
                "leakage",
                "top1",
                "top5",
                "top10",
                "nll",
                "ece",
                "mean rank",
                "steps",
                "baseline",
            ],
        )
        if arc
        else "_No ARC-AGI rows captured._"
    )
    if arc_transforms:
        lines.extend(["", "### ARC Transform Gate", ""])
        lines.append(
            markdown_table(
                [
                    [
                        str(row.get("task_instance", "")),
                        str(row.get("rules", "")),
                        str(row.get("train_consistent_rules", "")),
                        fmt_bool(row, "unique_test_prediction"),
                        fmt_bool(row, "test_exact"),
                        fmt_bool(row, "certified"),
                        str(row.get("test_correct_cells", "")),
                        str(row.get("test_cells", "")),
                        str(row.get("accepted_rules", "")),
                    ]
                    for row in arc_transforms
                ],
                [
                    "task",
                    "rules",
                    "train consistent",
                    "unique",
                    "test exact",
                    "certified",
                    "correct",
                    "cells",
                    "accepted rules",
                ],
            )
        )
    if arc_loaders:
        lines.extend(["", "### ARC Loader", ""])
        lines.append(
            markdown_table(
                [
                    [
                        str(row.get("task_dir", "")),
                        str(row.get("loaded_tasks", "")),
                        str(row.get("parse_errors", "")),
                        str(row.get("rules", "")),
                        str(row.get("unique_prediction_tasks", "")),
                        str(row.get("test_exact_tasks", "")),
                        str(row.get("certified_tasks", "")),
                        str(row.get("test_correct_cells", "")),
                        str(row.get("test_cells", "")),
                        fmt_percent(row, "cell_accuracy"),
                    ]
                    for row in arc_loaders
                ],
                [
                    "task dir",
                    "loaded",
                    "parse errors",
                    "rules",
                    "unique",
                    "test exact",
                    "certified",
                    "correct",
                    "cells",
                    "cell accuracy",
                ],
            )
        )
    if arc_hypotheses:
        lines.extend(["", "### ARC Hypothesis Evidence", ""])
        lines.append(
            markdown_table(
                [
                    [
                        str(row.get("task_instance", "")),
                        str(row.get("template_name", "")),
                        str(row.get("certificate_status", "")),
                        str(row.get("covered_events", "")),
                        str(row.get("event_count", "")),
                        fmt_percent(row, "utility"),
                        str(row.get("program_canonical_hash", ""))[:12],
                    ]
                    for row in arc_hypotheses
                ],
                [
                    "task",
                    "template",
                    "certificate",
                    "covered",
                    "events",
                    "utility",
                    "program hash",
                ],
            )
        )
    if arc_induction_parity:
        lines.extend(["", "### ARC Induction Parity", ""])
        lines.append(
            markdown_table(
                [
                    [
                        str(row.get("task_dir", "")),
                        str(row.get("train_baseline_certified", "")),
                        str(row.get("train_induction_certified", "")),
                        str(row.get("eval_baseline_certified", "")),
                        str(row.get("eval_induction_certified", "")),
                        fmt_bool(row, "parity_holds"),
                    ]
                    for row in arc_induction_parity
                ],
                [
                    "task dir",
                    "train baseline",
                    "train induction",
                    "eval baseline",
                    "eval induction",
                    "parity",
                ],
            )
        )
    if arc_program_proposers:
        lines.extend(["", "### ARC Program Proposers", ""])
        lines.append(
            markdown_table(
                [
                    [
                        str(row.get("task", "")),
                        str(row.get("strategy", "")),
                        str(row.get("budget", "")),
                        str(row.get("proposed_programs", "")),
                        str(row.get("certified_programs", "")),
                        str(row.get("accepted_programs", "")),
                        str(row.get("covered_events", "")),
                        str(row.get("event_count", "")),
                        fmt_percent(row, "utility"),
                    ]
                    for row in arc_program_proposers
                ],
                [
                    "task",
                    "strategy",
                    "budget",
                    "proposed",
                    "certified",
                    "accepted",
                    "covered",
                    "events",
                    "utility",
                ],
            )
        )
    lines.extend(
        [
            "",
            "## Delayed Claim",
            "",
        ]
    )
    lines.append(
        markdown_table(
            [
                [
                    str(row.get("model_hash", "unknown"))[:12],
                    str(row.get("parent_hash", "unknown"))[:12],
                    str(row.get("leakage_policy", "")),
                    fmt_percent(row, "accuracy"),
                    fmt_number(row, "nll"),
                    fmt_number(row, "ece"),
                    fmt_number(row, "decision_utility"),
                    fmt_number(row, "proposal_steps", 3),
                    fmt_number(row, "baseline_steps", 3),
                ]
                for row in delayed
            ],
            ["model", "parent", "leakage", "accuracy", "nll", "ece", "utility", "steps", "baseline"],
        )
        if delayed
        else "_No delayed-claim rows captured._"
    )
    lines.extend(["", "## Grid Path", ""])
    lines.append(
        markdown_table(
            [
                [
                    str(row.get("task", "")),
                    str(row.get("model_hash", "unknown"))[:12],
                    str(row.get("parent_hash", "unknown"))[:12],
                    str(row.get("leakage_policy", "")),
                    str(row.get("feature_set", "")),
                    str(row.get("enabled_structural_prior_features", "")),
                    str(row.get("evaluation_mode", "raw")),
                    fmt_percent(row, "accuracy"),
                    fmt_number(row, "nll"),
                    fmt_number(row, "ece"),
                    fmt_number(row, "decision_utility"),
                    fmt_number(row, "candidate_count", 2),
                    fmt_number(row, "proposal_steps", 3),
                    fmt_number(row, "verifier_nodes", 2),
                ]
                for row in grid
            ],
            [
                "task",
                "model",
                "parent",
                "leakage",
                "features",
                "priors",
                "mode",
                "accuracy",
                "nll",
                "ece",
                "utility",
                "candidates",
                "steps",
                "nodes",
            ],
        )
        if grid
        else "_No grid-path rows captured._"
    )
    lines.extend(["", "## Prime Gap", ""])
    lines.append(
        markdown_table(
            [
                [
                    str(row.get("model_hash", "unknown"))[:12],
                    str(row.get("evaluation_mode", "raw")),
                    str(row.get("feature_set", "")),
                    str(row.get("leakage_policy", "")),
                    fmt_percent(row, "top1"),
                    fmt_percent(row, "top10"),
                    fmt_number(row, "nll"),
                    fmt_number(row, "ece"),
                    fmt_number(row, "proposal_steps", 3),
                    fmt_number(row, "model_ms_per_sample", 4),
                ]
                for row in prime
                if row.get("model_hash") != "none"
            ],
            [
                "model",
                "mode",
                "features",
                "leakage",
                "top1",
                "top10",
                "nll",
                "ece",
                "steps",
                "model ms",
            ],
        )
        if prime
        else "_No prime-gap rows captured._"
    )
    lines.extend(["", "## Rule Search", ""])
    lines.append(
        markdown_table(
            [
                [
                    str(row.get("task", "")),
                    str(row.get("rule_set_hash", "unknown"))[:12],
                    str(row.get("grammar_version", "")),
                    fmt_bool(row, "include_state_conditions"),
                    fmt_rule_filters(row),
                    str(row.get("proposer_feature_set", "")),
                    str(row.get("proposer_target", "")),
                    str(row.get("selection_strategy", "")),
                    fmt_text(row, "softmax_group_size"),
                    fmt_text(row, "closed_loop_hidden"),
                    fmt_text(row, "unified_scorer_hidden"),
                    str(row.get("max_rules", "")),
                    str(row.get("proposed_rules", "")),
                    str(row.get("verified_rules", "")),
                    str(row.get("accepted_rules", "")),
                    str(row.get("false_rejections", "")),
                    str(row.get("train_marginal_samples", "")),
                    str(row.get("train_rejects", "")),
                    str(row.get("validation_rejects", "")),
                    str(row.get("test_rejects", "")),
                    fmt_number(row, "direct_exact", 3),
                    fmt_number(row, "expanded_exact", 3),
                    (f"{fmt_number(row, 'expanded_saved_pct', 2)}%"
                     if fmt_number(row, "expanded_saved_pct", 2)
                     else ""),
                ]
                for row in rule_search
            ],
            [
                "task",
                "rule hash",
                "grammar",
                "conditions",
                "filters",
                "features",
                "target",
                "strategy",
                "softmax group",
                "closed loop hidden",
                "unified hidden",
                "budget",
                "proposed",
                "verified",
                "accepted",
                "false rejects",
                "marginal samples",
                "train rejects",
                "val rejects",
                "test rejects",
                "direct",
                "gated",
                "saved",
            ],
        )
        if rule_search
        else "_No rule-search rows captured._"
    )
    lines.extend(
        [
            "",
            "Machine-readable records are in `comparison.jsonl`; command logs are under `logs/`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/latest"))
    parser.add_argument("--real", default="float")
    parser.add_argument("--arc-agi-task-dir", default="")
    parser.add_argument("--arc-agi-max-tasks", default="400")
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    output_dir = (repo / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    commands: list[str] = []
    for name, targets in COMMANDS:
        for target in targets:
            commands.append(f"make REAL={args.real} {target}")
            all_rows.extend(run_make(repo, args.real, target, output_dir / "logs" / f"{name}.log"))
    if args.arc_agi_task_dir:
        target = "arc-agi-induction-public"
        make_vars = {
            "ARC_AGI_TASK_DIR": args.arc_agi_task_dir,
            "ARC_AGI_MAX_TASKS": args.arc_agi_max_tasks,
        }
        commands.append(
            f"make REAL={args.real} ARC_AGI_TASK_DIR={args.arc_agi_task_dir} "
            f"ARC_AGI_MAX_TASKS={args.arc_agi_max_tasks} {target}"
        )
        all_rows.extend(
            run_make(
                repo,
                args.real,
                target,
                output_dir / "logs" / "arc_agi_induction_public.log",
                make_vars,
            )
        )

    all_rows = dedupe_rows(all_rows)
    write_jsonl(output_dir / "comparison.jsonl", all_rows)
    write_summary(output_dir / "summary.md", all_rows, args.real)
    (output_dir / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"wrote {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
