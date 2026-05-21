#!/usr/bin/env python3
"""Build a learned-grammar coverage regression surface from existing traces."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


COVERAGE_SCHEMA = "crlp.learned_grammar_coverage.v1"
SUMMARY_SCHEMA = "crlp.learned_grammar_coverage_summary.v1"
VIOLATION_SCHEMA = "crlp.learned_grammar_coverage_violation.v1"
BUDGET_SCHEMA = "crlp.learned_grammar_coverage_budget.v1"
ARC_GENERIC_VISUAL_TEMPLATES = {"arc_sketch_program"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                rows.append(
                    {
                        "schema": "crlp.parse_error.v1",
                        "path": str(path),
                        "line": line_no,
                    }
                )
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def limited(items: list[Any], limit: int = 24) -> list[Any]:
    return items[:limit]


def parse_input_spec(raw: str, default_scope: str) -> tuple[str, Path]:
    if "=" in raw:
        scope, path = raw.split("=", 1)
        if scope and path:
            return scope, Path(path)
    return default_scope, Path(raw)


def make_gate(
    passed: bool,
    *,
    required: bool,
    metric: Any = None,
    minimum: Any = None,
    authority: str = "verifier",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "passed": bool(passed),
        "required": bool(required),
        "authority": authority,
    }
    if metric is not None:
        row["metric"] = metric
    if minimum is not None:
        row["minimum"] = minimum
    return row


def summarize_arc(path: Path, rows: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    selection_rows = [
        row for row in rows if row.get("schema") == "crlp.arc_hypothesis.selection.v1"
    ]
    evidence_rows = [
        row for row in rows if row.get("schema") == "crlp.hypothesis.evidence.v1"
    ]
    loader_rows = [
        row for row in rows if row.get("schema") == "crlp.arc_loader.summary.v1"
    ]
    proposer_rows = [
        row
        for row in rows
        if row.get("schema") == "crlp.program_proposer.summary.v1"
        and row.get("domain") == "arc_agi"
    ]

    task_count = len(selection_rows)
    tasks_with_train_consistent = 0
    tasks_with_retained_train_consistent = 0
    tasks_with_selected = 0
    task_holes: list[dict[str, Any]] = []
    replacement_holes: list[dict[str, Any]] = []
    selected_templates: Counter[str] = Counter()
    selected_visual_templates: Counter[str] = Counter()
    selected_legacy_templates: Counter[str] = Counter()
    template_totals: dict[str, Counter[str]] = defaultdict(Counter)
    visual_template_totals: Counter[str] = Counter()
    legacy_template_totals: Counter[str] = Counter()
    tasks_with_visual_train_consistent = 0
    tasks_with_visual_retained_train_consistent = 0
    tasks_with_visual_selected = 0
    tasks_with_legacy_train_consistent = 0

    for row in selection_rows:
        task_id = str(row.get("task_instance", "unknown"))
        pre_cap = as_int(row.get("proposals_train_consistent_pre_cap"))
        retained = as_int(row.get("proposals_train_consistent"))
        selected = as_int(row.get("proposals_selected"))
        task_visual_train_consistent = 0
        task_visual_retained_train_consistent = 0
        task_visual_selected = 0
        task_legacy_train_consistent = 0
        task_visual_attempted = 0
        task_legacy_attempted = 0
        task_template_attempts: list[dict[str, Any]] = []
        if pre_cap > 0:
            tasks_with_train_consistent += 1
        if retained > 0:
            tasks_with_retained_train_consistent += 1
        if selected > 0:
            tasks_with_selected += 1

        for template in row.get("templates", []) or []:
            if not isinstance(template, dict):
                continue
            name = str(template.get("name", "unknown"))
            totals = template_totals[name]
            for key in [
                "attempted",
                "generated",
                "retained",
                "train_consistent_pre_cap",
                "train_consistent_retained",
                "selected",
            ]:
                totals[key] += as_int(template.get(key))
            attempted = as_int(template.get("attempted"))
            train_consistent = as_int(template.get("train_consistent_pre_cap"))
            retained_count = as_int(template.get("train_consistent_retained"))
            selected_count = as_int(template.get("selected"))
            if name in ARC_GENERIC_VISUAL_TEMPLATES:
                task_visual_attempted += attempted
                task_visual_train_consistent += train_consistent
                task_visual_retained_train_consistent += retained_count
                task_visual_selected += selected_count
                visual_template_totals["attempted"] += attempted
                visual_template_totals["generated"] += as_int(template.get("generated"))
                visual_template_totals["train_consistent_pre_cap"] += train_consistent
                visual_template_totals["train_consistent_retained"] += retained_count
                visual_template_totals["selected"] += selected_count
            else:
                task_legacy_attempted += attempted
                task_legacy_train_consistent += train_consistent
                legacy_template_totals["attempted"] += attempted
                legacy_template_totals["generated"] += as_int(template.get("generated"))
                legacy_template_totals["train_consistent_pre_cap"] += train_consistent
                legacy_template_totals["train_consistent_retained"] += retained_count
                legacy_template_totals["selected"] += selected_count
            if attempted > 0:
                task_template_attempts.append(
                    {
                        "template": name,
                        "attempted": attempted,
                        "train_consistent_pre_cap": train_consistent,
                    }
                )
            if as_int(template.get("selected")) > 0:
                selected_templates[name] += selected_count
                if name in ARC_GENERIC_VISUAL_TEMPLATES:
                    selected_visual_templates[name] += selected_count
                else:
                    selected_legacy_templates[name] += selected_count

        if task_visual_train_consistent > 0:
            tasks_with_visual_train_consistent += 1
        if task_visual_retained_train_consistent > 0:
            tasks_with_visual_retained_train_consistent += 1
        if task_visual_selected > 0:
            tasks_with_visual_selected += 1
        if task_legacy_train_consistent > 0:
            tasks_with_legacy_train_consistent += 1
        task_template_attempts.sort(
            key=lambda item: (-as_int(item.get("attempted")), str(item.get("template", "")))
        )
        if task_visual_train_consistent <= 0:
            task_holes.append(
                {
                    "kind": "no_train_consistent_candidate",
                    "task": task_id,
                    "attempted": as_int(row.get("proposals_attempted")),
                    "generated": as_int(row.get("proposals_generated")),
                    "attempted_visual_programs": task_visual_attempted,
                    "attempted_legacy_templates": task_legacy_attempted,
                    "legacy_train_consistent_pre_cap": task_legacy_train_consistent,
                    "top_attempted_templates": task_template_attempts[:5],
                }
            )
        elif task_visual_selected <= 0:
            task_holes.append(
                {
                    "kind": "train_consistent_not_selected",
                    "task": task_id,
                    "visual_train_consistent_pre_cap": task_visual_train_consistent,
                    "visual_train_consistent_retained": task_visual_retained_train_consistent,
                    "visual_selected": task_visual_selected,
                    "legacy_train_consistent_pre_cap": task_legacy_train_consistent,
                    "legacy_selected": selected - task_visual_selected,
                    "train_consistent_pre_cap": pre_cap,
                    "retained": retained,
                }
            )

    if selected_legacy_templates:
        replacement_holes.append(
            {
                "kind": "handcrafted_transform_dependency",
                "selected_legacy_templates": dict(sorted(selected_legacy_templates.items())),
                "selected_visual_programs": dict(sorted(selected_visual_templates.items())),
                "legacy_selected_count": sum(selected_legacy_templates.values()),
                "visual_selected_count": sum(selected_visual_templates.values()),
            }
        )

    template_holes = [
        {
            "kind": "attempted_template_no_train_consistent",
            "template": name,
            "attempted": totals["attempted"],
            "generated": totals["generated"],
        }
        for name, totals in template_totals.items()
        if totals["attempted"] > 0 and totals["train_consistent_pre_cap"] == 0
    ]
    template_holes.sort(key=lambda item: (-item["attempted"], item["template"]))

    certified_evidence_rows = [
        row for row in evidence_rows if row.get("certificate_status") == "certified"
    ]
    false_rejections = sum(as_int(row.get("false_rejections")) for row in evidence_rows)
    loader = loader_rows[-1] if loader_rows else {}
    loaded_tasks = as_int(loader.get("loaded_tasks"))
    certified_tasks = as_int(loader.get("certified_tasks"))
    test_exact_tasks = as_int(loader.get("test_exact_tasks"))
    cell_accuracy = as_float(loader.get("cell_accuracy"))

    learned_rows = [row for row in proposer_rows if row.get("strategy") == "learned"]
    best_learned = max(
        learned_rows,
        key=lambda row: (
            as_int(row.get("certified_programs")),
            as_float(row.get("utility")),
            as_int(row.get("budget")),
        ),
        default={},
    )
    learned_certified = as_int(best_learned.get("certified_programs"))
    learned_utility = as_float(best_learned.get("utility"))

    candidate_gate = task_count > 0 and tasks_with_train_consistent == task_count
    retained_gate = task_count > 0 and tasks_with_retained_train_consistent == task_count
    ranker_gate = (
        task_count > 0
        and tasks_with_selected == task_count
        and learned_certified > 0
        and learned_utility > 0.0
    )
    visual_candidate_gate = (
        task_count > 0 and tasks_with_visual_train_consistent == task_count
    )
    visual_retained_gate = (
        task_count > 0 and tasks_with_visual_retained_train_consistent == task_count
    )
    visual_ranker_gate = (
        task_count > 0
        and tasks_with_visual_selected == task_count
        and learned_certified > 0
        and learned_utility > 0.0
    )
    ownership_gate = (
        task_count > 0
        and tasks_with_visual_selected == task_count
        and legacy_template_totals["selected"] == 0
    )
    verifier_gate = certified_tasks > 0 and certified_evidence_rows and false_rejections == 0
    transfer_gate = (
        loaded_tasks > 0
        and certified_tasks == loaded_tasks
        and test_exact_tasks == loaded_tasks
        and cell_accuracy >= 1.0
    )

    strict_required = scope == "fixture_regression"
    coverage_holes = limited(replacement_holes + task_holes + template_holes)
    heldout_transfer_rate = ratio(test_exact_tasks, loaded_tasks)
    gates = {
        "candidate_space_contains_train_consistent_solution": make_gate(
            candidate_gate,
            required=strict_required,
            metric={
                "tasks": task_count,
                "tasks_with_train_consistent": tasks_with_train_consistent,
                "tasks_with_visual_train_consistent": tasks_with_visual_train_consistent,
                "tasks_with_legacy_train_consistent": tasks_with_legacy_train_consistent,
            },
        ),
        "train_consistent_solution_retained": make_gate(
            retained_gate,
            required=strict_required,
            metric={
                "tasks": task_count,
                "tasks_with_retained_train_consistent": tasks_with_retained_train_consistent,
                "tasks_with_visual_retained_train_consistent": (
                    tasks_with_visual_retained_train_consistent
                ),
            },
        ),
        "neural_ranker_places_solution_near_top": make_gate(
            ranker_gate,
            required=strict_required,
            metric={
                "tasks_with_selected_candidate": tasks_with_selected,
                "tasks_with_visual_selected_candidate": tasks_with_visual_selected,
                "learned_certified_programs": learned_certified,
                "learned_utility": learned_utility,
                "learned_budget": as_int(best_learned.get("budget")),
            },
        ),
        "learned_visual_program_contains_train_consistent_solution": make_gate(
            visual_candidate_gate,
            required=False,
            metric={
                "tasks": task_count,
                "tasks_with_visual_train_consistent": tasks_with_visual_train_consistent,
                "tasks_with_legacy_train_consistent": tasks_with_legacy_train_consistent,
            },
        ),
        "learned_visual_program_train_consistent_solution_retained": make_gate(
            visual_retained_gate,
            required=False,
            metric={
                "tasks": task_count,
                "tasks_with_visual_retained_train_consistent": (
                    tasks_with_visual_retained_train_consistent
                ),
            },
        ),
        "learned_visual_program_ranker_places_solution_near_top": make_gate(
            visual_ranker_gate,
            required=False,
            metric={
                "tasks": task_count,
                "tasks_with_visual_selected_candidate": tasks_with_visual_selected,
                "learned_certified_programs": learned_certified,
                "learned_utility": learned_utility,
                "learned_budget": as_int(best_learned.get("budget")),
            },
        ),
        "learned_visual_program_owns_selected_solution": make_gate(
            ownership_gate,
            required=False,
            metric={
                "tasks": task_count,
                "tasks_with_visual_selected_candidate": tasks_with_visual_selected,
                "legacy_selected_count": legacy_template_totals["selected"],
                "visual_selected_count": visual_template_totals["selected"],
            },
        ),
        "verifier_certifies_selected_solution": make_gate(
            verifier_gate,
            required=strict_required,
            metric={
                "certified_tasks": certified_tasks,
                "certified_evidence_rows": len(certified_evidence_rows),
                "false_rejections": false_rejections,
            },
        ),
        "heldout_transfer_survives": make_gate(
            transfer_gate,
            required=strict_required,
            metric={
                "scope": scope,
                "loaded_tasks": loaded_tasks,
                "test_exact_tasks": test_exact_tasks,
                "cell_accuracy": cell_accuracy,
                "task_exact_rate": heldout_transfer_rate,
            },
        ),
    }

    return {
        "schema": COVERAGE_SCHEMA,
        "experiment": "arc_agi",
        "domain": "arc_agi",
        "authority": "arc_train_pair_verifier",
        "scope": scope,
        "source_paths": [str(path)],
        "paper_claim_support": "scoped",
        "gates": gates,
        "passing_gates": sorted([name for name, gate in gates.items() if gate["passed"]]),
        "coverage_holes": coverage_holes,
        "certificate_rows": len(certified_evidence_rows),
        "learning_signal": {
            "selection_rows": task_count,
            "program_proposer_rows": len(proposer_rows),
            "selected_templates": dict(sorted(selected_templates.items())),
            "transform_ownership": {
                "generic_visual_templates": sorted(ARC_GENERIC_VISUAL_TEMPLATES),
                "visual_template_totals": dict(sorted(visual_template_totals.items())),
                "legacy_template_totals": dict(sorted(legacy_template_totals.items())),
                "selected_visual_templates": dict(sorted(selected_visual_templates.items())),
                "selected_legacy_templates": dict(sorted(selected_legacy_templates.items())),
            },
        },
        "verifier_delta": {
            "loaded_tasks": loaded_tasks,
            "certified_tasks": certified_tasks,
            "test_exact_tasks": test_exact_tasks,
            "cell_accuracy": cell_accuracy,
        },
        "metrics": {
            "zero_candidate_units": task_count - tasks_with_train_consistent,
            "train_consistent_candidate_rate": ratio(tasks_with_train_consistent, task_count),
            "ranker_top_k_recovery": ratio(tasks_with_selected, task_count),
            "visual_ranker_top_k_recovery": ratio(tasks_with_visual_selected, task_count),
            "verifier_certified_delta": certified_tasks,
            "heldout_transfer_rate": heldout_transfer_rate,
            "candidate_task_coverage": ratio(tasks_with_train_consistent, task_count),
            "retained_task_coverage": ratio(tasks_with_retained_train_consistent, task_count),
            "selected_task_coverage": ratio(tasks_with_selected, task_count),
            "certified_evidence_rows": len(certified_evidence_rows),
            "learned_certified_programs": learned_certified,
            "learned_utility": learned_utility,
            "visual_program_task_coverage": ratio(
                tasks_with_visual_train_consistent,
                task_count,
            ),
            "visual_program_retained_task_coverage": ratio(
                tasks_with_visual_retained_train_consistent,
                task_count,
            ),
            "visual_program_selected_task_coverage": ratio(
                tasks_with_visual_selected,
                task_count,
            ),
            "legacy_template_task_coverage": ratio(
                tasks_with_legacy_train_consistent,
                task_count,
            ),
            "visual_program_attempts": visual_template_totals["attempted"],
            "legacy_template_attempts": legacy_template_totals["attempted"],
            "visual_program_train_consistent": (
                visual_template_totals["train_consistent_pre_cap"]
            ),
            "legacy_template_train_consistent": (
                legacy_template_totals["train_consistent_pre_cap"]
            ),
            "visual_program_selected": visual_template_totals["selected"],
            "legacy_template_selected": legacy_template_totals["selected"],
        },
    }


def summarize_haiku_word(path: Path, rows: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    trace_rows = [
        row
        for row in rows
        if row.get("schema") == "crlp.haiku_word_lm.generation_trace.v1"
    ]
    total = len(trace_rows)
    candidate_positive = sum(1 for row in trace_rows if as_int(row.get("candidate_count")) > 0)
    top1 = sum(
        1
        for row in trace_rows
        if as_int(row.get("selected_index"), -1) == as_int(row.get("top_ranked_index"), -2)
    )
    strict = sum(1 for row in trace_rows if row.get("strict_adjacency") is True)
    dead_end = sum(
        as_int((row.get("rejected_candidates") or {}).get("dead_end"))
        for row in trace_rows
        if isinstance(row.get("rejected_candidates"), dict)
    )
    adjacency_rejects = sum(
        as_int((row.get("rejected_candidates") or {}).get("adjacency"))
        for row in trace_rows
        if isinstance(row.get("rejected_candidates"), dict)
    )
    gate_names = Counter()
    for row in trace_rows:
        verifier_cost = row.get("verifier_cost") or {}
        if not isinstance(verifier_cost, dict):
            continue
        for gate_name in verifier_cost.get("gates", []) or []:
            gate_names[str(gate_name)] += 1

    candidate_gate = total > 0 and candidate_positive == total and strict == total
    ranker_gate = total > 0 and top1 == total
    proxy_gate = total > 0 and strict == total and dead_end == 0
    strict_required = scope == "smoke_trace"

    holes: list[dict[str, Any]] = []
    if candidate_positive < total:
        holes.append(
            {
                "kind": "zero_candidate_step",
                "steps": total - candidate_positive,
            }
        )
    if strict < total:
        holes.append(
            {
                "kind": "strict_adjacency_failed",
                "steps": total - strict,
            }
        )
    if dead_end > 0:
        holes.append({"kind": "dead_end_rejections", "count": dead_end})

    gates = {
        "candidate_space_contains_train_consistent_solution": make_gate(
            candidate_gate,
            required=strict_required,
            metric={"steps": total, "steps_with_candidates": candidate_positive},
            authority="corpus_adjacency_proxy",
        ),
        "neural_ranker_places_solution_near_top": make_gate(
            ranker_gate,
            required=strict_required,
            metric={"top1_steps": top1, "steps": total},
            authority="neural_trace",
        ),
        "verifier_certifies_selected_solution": make_gate(
            False,
            required=False,
            metric={"proxy_verifier_passes": proxy_gate},
            authority="missing_domain_certificate",
        ),
        "heldout_transfer_survives": make_gate(
            False,
            required=False,
            metric={"scope": "smoke_trace_only"},
            authority="missing_heldout_trace",
        ),
    }

    return {
        "schema": COVERAGE_SCHEMA,
        "experiment": "haiku_word_lm",
        "domain": "haiku_word_lm",
        "authority": "corpus_adjacency_proxy",
        "scope": scope,
        "source_paths": [str(path)],
        "paper_claim_support": "diagnostic_only",
        "gates": gates,
        "passing_gates": sorted([name for name, gate in gates.items() if gate["passed"]]),
        "coverage_holes": holes,
        "certificate_rows": 0,
        "learning_signal": {
            "trace_rows": total,
            "verifier_gates": dict(sorted(gate_names.items())),
        },
        "verifier_delta": {
            "strict_adjacency_steps": strict,
            "dead_end_rejections": dead_end,
            "adjacency_rejections": adjacency_rejects,
        },
        "metrics": {
            "zero_candidate_units": total - candidate_positive,
            "train_consistent_candidate_rate": ratio(candidate_positive, total),
            "ranker_top_k_recovery": ratio(top1, total),
            "verifier_certified_delta": 0,
            "heldout_transfer_rate": 0.0,
            "candidate_step_coverage": ratio(candidate_positive, total),
            "top1_trace_rate": ratio(top1, total),
            "strict_adjacency_rate": ratio(strict, total),
        },
    }


def summarize_haiku_understanding(path: Path, rows: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    trace_rows = [
        row for row in rows if row.get("schema") == "crlp.haiku_understanding.trace.v1"
    ]
    total = len(trace_rows)
    candidate_positive = sum(1 for row in trace_rows if as_int(row.get("candidate_count")) >= 2)
    real_selected = sum(1 for row in trace_rows if row.get("selected") == "real")
    symbolic_agreement = sum(
        1 for row in trace_rows if row.get("selected") == row.get("symbolic_selected")
    )
    corruption_counts: Counter[str] = Counter()
    corruption_correct: Counter[str] = Counter()
    for row in trace_rows:
        corruption = str(row.get("corruption", "unknown"))
        corruption_counts[corruption] += 1
        if row.get("selected") == "real":
            corruption_correct[corruption] += 1

    accuracy = ratio(real_selected, total)
    candidate_gate = total > 0 and candidate_positive == total
    ranker_gate = accuracy is not None and accuracy >= 0.75
    proxy_gate = total > 0 and symbolic_agreement == total
    strict_required = scope == "smoke_trace"

    holes: list[dict[str, Any]] = []
    if candidate_positive < total:
        holes.append(
            {
                "kind": "missing_real_or_corrupt_candidate",
                "rows": total - candidate_positive,
            }
        )
    if not ranker_gate:
        holes.append(
            {
                "kind": "ranker_accuracy_below_gate",
                "accuracy": accuracy,
                "minimum": 0.75,
            }
        )
    weak_corruptions = [
        {
            "kind": "weak_corruption_family",
            "corruption": corruption,
            "accuracy": ratio(corruption_correct[corruption], count),
            "rows": count,
        }
        for corruption, count in sorted(corruption_counts.items())
        if ratio(corruption_correct[corruption], count) is not None
        and ratio(corruption_correct[corruption], count) < 0.75
    ]
    holes.extend(weak_corruptions)

    gates = {
        "candidate_space_contains_train_consistent_solution": make_gate(
            candidate_gate,
            required=strict_required,
            metric={"rows": total, "rows_with_real_and_corrupt": candidate_positive},
            authority="paired_benchmark_trace",
        ),
        "neural_ranker_places_solution_near_top": make_gate(
            ranker_gate,
            required=False,
            metric={"accuracy": accuracy, "minimum": 0.75},
            authority="benchmark_label",
        ),
        "verifier_certifies_selected_solution": make_gate(
            False,
            required=False,
            metric={"symbolic_proxy_agreement": ratio(symbolic_agreement, total)},
            authority="missing_domain_certificate",
        ),
        "heldout_transfer_survives": make_gate(
            False,
            required=False,
            metric={"scope": "smoke_trace_only"},
            authority="missing_heldout_trace",
        ),
    }

    return {
        "schema": COVERAGE_SCHEMA,
        "experiment": "haiku_understanding",
        "domain": "haiku_understanding",
        "authority": "paired_benchmark_trace",
        "scope": scope,
        "source_paths": [str(path)],
        "paper_claim_support": "diagnostic_only",
        "gates": gates,
        "passing_gates": sorted([name for name, gate in gates.items() if gate["passed"]]),
        "coverage_holes": limited(holes),
        "certificate_rows": 0,
        "learning_signal": {
            "trace_rows": total,
            "corruption_families": dict(sorted(corruption_counts.items())),
        },
        "verifier_delta": {
            "real_selected": real_selected,
            "symbolic_agreement": symbolic_agreement,
        },
        "metrics": {
            "zero_candidate_units": total - candidate_positive,
            "train_consistent_candidate_rate": ratio(candidate_positive, total),
            "ranker_top_k_recovery": accuracy,
            "verifier_certified_delta": 0,
            "heldout_transfer_rate": 0.0,
            "candidate_pair_coverage": ratio(candidate_positive, total),
            "ranker_accuracy": accuracy,
            "symbolic_proxy_agreement": ratio(symbolic_agreement, total),
            "corruption_accuracy": {
                corruption: ratio(corruption_correct[corruption], count)
                for corruption, count in sorted(corruption_counts.items())
            },
        },
    }


def load_budgets(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    budgets: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("schema") == BUDGET_SCHEMA:
                budgets.append(row)
    return budgets


def budget_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("experiment", "")), str(row.get("scope", ""))


def audit_budget_value(
    *,
    experiment: str,
    scope: str,
    metric: str,
    actual: Any,
    budget: dict[str, Any],
    maximum_field: str | None = None,
    minimum_field: str | None = None,
) -> dict[str, Any] | None:
    if maximum_field is not None and maximum_field in budget:
        maximum = as_float(budget.get(maximum_field))
        value = as_float(actual)
        if value > maximum:
            return {
                "schema": VIOLATION_SCHEMA,
                "violation_type": "budget",
                "experiment": experiment,
                "scope": scope,
                "metric": metric,
                "actual": actual,
                "maximum": maximum,
                "budget_field": maximum_field,
            }
    if minimum_field is not None and minimum_field in budget:
        minimum = as_float(budget.get(minimum_field))
        value = as_float(actual)
        if value < minimum:
            return {
                "schema": VIOLATION_SCHEMA,
                "violation_type": "budget",
                "experiment": experiment,
                "scope": scope,
                "metric": metric,
                "actual": actual,
                "minimum": minimum,
                "budget_field": minimum_field,
            }
    return None


def apply_budgets(
    rows: list[dict[str, Any]],
    budgets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    budget_by_key = {
        (str(row.get("experiment", "")), str(row.get("scope", ""))): row
        for row in budgets
    }
    coverage_by_key = {
        budget_key(row): row for row in rows if row.get("schema") == COVERAGE_SCHEMA
    }

    for (experiment, scope), budget in budget_by_key.items():
        if budget.get("required") is True and (experiment, scope) not in coverage_by_key:
            violations.append(
                {
                    "schema": VIOLATION_SCHEMA,
                    "violation_type": "missing_required_scope",
                    "experiment": experiment,
                    "scope": scope,
                    "budget_field": "required",
                }
            )

    metric_specs = [
        ("zero_candidate_units", "max_zero_candidate_tasks", None),
        (
            "train_consistent_candidate_rate",
            None,
            "min_train_consistent_candidate_rate",
        ),
        ("ranker_top_k_recovery", None, "min_ranker_top_k_recovery"),
        ("verifier_certified_delta", None, "min_verifier_certified_delta"),
        ("heldout_transfer_rate", None, "min_heldout_transfer_rate"),
    ]
    for row in rows:
        if row.get("schema") != COVERAGE_SCHEMA:
            continue
        experiment, scope = budget_key(row)
        budget = budget_by_key.get((experiment, scope))
        if budget is None:
            continue
        metrics = row.get("metrics") or {}
        row["budget"] = {
            key: value
            for key, value in budget.items()
            if key not in {"schema", "experiment", "scope"}
        }
        row_budget_violations: list[dict[str, Any]] = []
        for metric, maximum_field, minimum_field in metric_specs:
            violation = audit_budget_value(
                experiment=experiment,
                scope=scope,
                metric=metric,
                actual=metrics.get(metric),
                budget=budget,
                maximum_field=maximum_field,
                minimum_field=minimum_field,
            )
            if violation is not None:
                violations.append(violation)
                row_budget_violations.append(
                    {
                        key: value
                        for key, value in violation.items()
                        if key not in {"schema", "experiment", "scope"}
                    }
                )
        row["budget_violations"] = row_budget_violations
    return violations


def required_gate_violations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for row in rows:
        gates = row.get("gates") or {}
        if not isinstance(gates, dict):
            continue
        for name, gate in gates.items():
            if not isinstance(gate, dict):
                continue
            if gate.get("required") is True and gate.get("passed") is not True:
                violations.append(
                    {
                        "schema": VIOLATION_SCHEMA,
                        "violation_type": "required_gate",
                        "experiment": row.get("experiment"),
                        "scope": row.get("scope"),
                        "gate": name,
                        "metric": gate.get("metric"),
                        "authority": gate.get("authority"),
                    }
                )
    return violations


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            fp.write("\n")


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in args.arc_log or []:
        scope, path = parse_input_spec(raw, "fixture_regression")
        rows.append(summarize_arc(path, load_jsonl(path), scope))
    for raw in args.haiku_word_trace or []:
        scope, path = parse_input_spec(raw, "smoke_trace")
        rows.append(summarize_haiku_word(path, load_jsonl(path), scope))
    for raw in args.haiku_understanding_trace or []:
        scope, path = parse_input_spec(raw, "smoke_trace")
        rows.append(
            summarize_haiku_understanding(
                path,
                load_jsonl(path),
                scope,
            )
        )

    budget_violations = apply_budgets(rows, load_budgets(args.budget))
    violations = required_gate_violations(rows)
    violations.extend(budget_violations)
    rows.append(
        {
            "schema": SUMMARY_SCHEMA,
            "experiments": len([row for row in rows if row.get("schema") == COVERAGE_SCHEMA]),
            "scopes": len(
                {
                    (row.get("experiment"), row.get("scope"))
                    for row in rows
                    if row.get("schema") == COVERAGE_SCHEMA
                }
            ),
            "violations": len(violations),
            "experiments_with_violations": sorted(
                {
                    str(violation.get("experiment"))
                    for violation in violations
                    if violation.get("experiment")
                }
            ),
        }
    )
    rows.extend(violations)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arc-log",
        action="append",
        help="ARC JSONL log path, or scope=path.",
    )
    parser.add_argument(
        "--haiku-word-trace",
        action="append",
        help="Haiku word trace path, or scope=path.",
    )
    parser.add_argument(
        "--haiku-understanding-trace",
        action="append",
        help="Haiku understanding trace path, or scope=path.",
    )
    parser.add_argument("--budget", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Return nonzero when a required coverage gate is missing or broken.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_rows(args)
    write_jsonl(args.output, rows)
    violation_count = sum(1 for row in rows if row.get("schema") == VIOLATION_SCHEMA)
    if args.fail_on_regression and violation_count:
        print(
            f"learned grammar coverage regressions: {violation_count}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
