#!/usr/bin/env python3
"""Consolidate experiment health from coverage, verifier, numeric, and claim artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HEALTH_SCHEMA = "crlp.experiment_health.v1"
SUMMARY_SCHEMA = "crlp.experiment_health_summary.v1"
COVERAGE_SCHEMA = "crlp.learned_grammar_coverage.v1"
NUMERIC_SUMMARY_SCHEMA = "crlp.numeric_certificate_experiment_summary.v1"
NUMERIC_DISTRIBUTION_SCHEMA = "crlp.numeric_certificate_distribution.v1"
SIGNAL_FLIGHT_SUMMARY_SCHEMA = "crlp.signal_flight.summary.v1"
SIGNAL_FLIGHT_ROLLOUT_SCHEMA = "crlp.signal_flight_rollout.summary.v1"
SIGNAL_LIVE_ACTIVE_SCHEMA = "crlp.signal_live_active_summary.v1"
SIGNAL_SHADOW_REPLAY_SCHEMA = "crlp.signal_shadow_replay_audit.v1"
NEURAL_GRAMMAR_SUMMARY_SCHEMA = "crlp.neural_grammar_discovery.summary.v1"


TARGET_EXPERIMENTS = {
    "action-score": "action_score",
    "prime-gap-trace": "prime_gap",
    "prime-rule-search-smoke": "prime_gap",
    "delayed-claim-eval": "delayed_claim",
    "grid-path-eval": "grid_path",
    "multitask-domain": "multitask_domain",
    "symbolic-domains": "symbolic_domains",
    "program-synthesis": "program_synthesis",
    "program-synthesis-baselines": "program_synthesis",
    "modular-gate-search-smoke": "modular_gate_search",
    "arc-agi-induction-smoke": "arc_agi",
    "haiku-lm-smoke": "haiku_lm",
    "haiku-word-lm-smoke": "haiku_word_lm",
    "haiku-understanding-smoke": "haiku_understanding",
    "signal-flight": "signal_flight",
    "signal-mining-smoke": "signal_mining",
}

CLAIM_ID_OWNERS = {
    "C1.prime_gate_delta": ["prime_gap"],
    "C1.prime_gate_delta_ci": ["prime_gap"],
    "C2.prime_proposer_fixed_budget": ["prime_gap"],
    "C1.arc_gate_scorer_invariance": ["arc_agi"],
    "C1.arc_public_evidence_surface": ["arc_agi"],
    "C3.arc_ood_calibration_tradeoff": ["arc_agi"],
    "C4.arc_ood_shared_vs_single": ["arc_agi", "multitask_domain"],
    "C4.delayed_claim_sibling_transfer": ["delayed_claim"],
    "C4.multitask_capacity_tradeoff": ["multitask_domain"],
    "C4.adapter_heads_shared_trunk": ["multitask_domain"],
    "C5.rejected_candidate_reason_learning": ["modular_gate_search"],
    "C5.hypothesis_library_promotion": ["arc_agi"],
    "C7.tiny_net_plateau_stop_rule": ["multitask_domain"],
    "C8.trace_determinism_audit": ["prime_gap"],
    "C9.signal_flight_replay_control": ["signal_flight"],
    "C9.signal_live_neural_active": ["signal_flight"],
}

CLAIM_TEXT_FALLBACK_KEYWORDS = {
    "prime_gap": ["prime"],
    "arc_agi": ["arc"],
    "delayed_claim": ["delayed_claim"],
    "multitask_domain": ["multitask"],
    "modular_gate_search": ["modular"],
    "signal_flight": ["signal_flight"],
    "signal_mining": ["signal_mining"],
}


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            row = json.loads(line)
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


def target_to_experiment(target: str) -> str:
    if target in TARGET_EXPERIMENTS:
        return TARGET_EXPERIMENTS[target]
    return target.replace("-", "_")


def load_claim_support(path: Path | None) -> dict[str, dict[str, Any]]:
    support: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "status_counts": Counter(),
            "claim_ids": [],
            "required_claim_ids": [],
        }
    )
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    for claim in data.get("claims", []) or []:
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("id", ""))
        owners = CLAIM_ID_OWNERS.get(claim_id)
        if owners is None:
            haystack = " ".join(
                [
                    claim_id,
                    str(claim.get("title", "")),
                    str(claim.get("claim_text", "")),
                    str(claim.get("summary", "")),
                ]
            ).lower()
            owners = [
                experiment
                for experiment, keywords in CLAIM_TEXT_FALLBACK_KEYWORDS.items()
                if any(keyword in haystack for keyword in keywords)
            ]
        status = str(claim.get("status", "unknown"))
        required = bool(claim.get("required_for_submission"))
        for experiment in owners:
            support[experiment]["status_counts"][status] += 1
            support[experiment]["claim_ids"].append(claim_id)
            if required:
                support[experiment]["required_claim_ids"].append(claim_id)
    result: dict[str, dict[str, Any]] = {}
    for experiment, row in support.items():
        status_counts = dict(sorted(row["status_counts"].items()))
        result[experiment] = {
            "status_counts": status_counts,
            "claim_ids": sorted(set(row["claim_ids"])),
            "required_claim_ids": sorted(set(row["required_claim_ids"])),
            "supports_paper_claim": status_counts.get("supported", 0) > 0
            and status_counts.get("contradicted", 0) == 0,
        }
    return result


def summarize_numeric(
    summary_rows: list[dict[str, Any]],
    distribution_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    numeric: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "targets": [],
            "certificate_rows": 0,
            "committed_rows": 0,
            "rejected_rows": 0,
            "finite_after_rows": 0,
            "statuses": Counter(),
            "rejection_reasons": Counter(),
        }
    )
    for row in summary_rows:
        if row.get("schema") != NUMERIC_SUMMARY_SCHEMA:
            continue
        target = str(row.get("target", "unknown"))
        experiment = target_to_experiment(target)
        info = numeric[experiment]
        info["targets"].append(target)
        info["certificate_rows"] += as_int(row.get("certificate_rows"))
        info["committed_rows"] += as_int(row.get("committed_rows"))
        info["rejected_rows"] += as_int(row.get("rejected_rows"))
        info["finite_after_rows"] += as_int(row.get("finite_after_rows"))
        info["statuses"][str(row.get("status", "unknown"))] += 1
        reasons = row.get("rejection_reasons") or {}
        if isinstance(reasons, dict):
            for reason, count in reasons.items():
                info["rejection_reasons"][str(reason)] += as_int(count)

    distribution_by_target = {
        str(row.get("target", "")): row
        for row in distribution_rows
        if row.get("schema") == NUMERIC_DISTRIBUTION_SCHEMA
    }
    for info in numeric.values():
        metric_targets: dict[str, Any] = {}
        for target in info["targets"]:
            row = distribution_by_target.get(target)
            if row:
                metric_targets[target] = {
                    "max_shrink_steps": row.get("max_shrink_steps"),
                    "step_kinds": row.get("step_kinds"),
                }
        info["numeric_metrics"] = metric_targets
        info["targets"] = sorted(set(info["targets"]))
        info["statuses"] = dict(sorted(info["statuses"].items()))
        info["rejection_reasons"] = dict(sorted(info["rejection_reasons"].items()))
    return dict(numeric)


def latest_schema(rows: list[dict[str, Any]], schema: str) -> dict[str, Any] | None:
    matches = [row for row in rows if row.get("schema") == schema]
    return matches[-1] if matches else None


def summarize_signal_flight_evidence(args: argparse.Namespace) -> dict[str, Any] | None:
    offline_rows = load_jsonl(args.signal_flight_summary)
    rollout_rows = load_jsonl(args.signal_flight_rollout)
    live_rows = load_jsonl(args.signal_live_active)
    shadow_rows = load_jsonl(args.signal_shadow_replay)
    if not offline_rows and not rollout_rows and not live_rows and not shadow_rows:
        return None

    offline = latest_schema(offline_rows, SIGNAL_FLIGHT_SUMMARY_SCHEMA)
    live = latest_schema(live_rows, SIGNAL_LIVE_ACTIVE_SCHEMA)
    shadow = latest_schema(shadow_rows, SIGNAL_SHADOW_REPLAY_SCHEMA)
    rollout_by_controller = {
        str(row.get("controller")): row
        for row in rollout_rows
        if row.get("schema") == SIGNAL_FLIGHT_ROLLOUT_SCHEMA and row.get("controller")
    }
    guarded = rollout_by_controller.get("neural_guarded")
    rollout_baseline = rollout_by_controller.get("goal_heading_no_safety")

    top1 = as_float(offline.get("top1") if offline else None)
    baseline_top1 = as_float(offline.get("baseline_top1") if offline else None)
    selected_utility = as_float(offline.get("selected_utility") if offline else None)
    baseline_utility = as_float(
        offline.get("baseline_selected_utility") if offline else None
    )
    regret = as_float(offline.get("regret") if offline else None)
    baseline_regret = as_float(offline.get("baseline_regret") if offline else None)
    offline_passed = bool(
        offline
        and top1 > baseline_top1
        and selected_utility > baseline_utility
        and regret < baseline_regret
        and as_float(offline.get("selected_death_rate")) <= as_float(offline.get("baseline_death_rate"))
        and as_float(offline.get("selected_candidate_mask_rate")) >= 0.999
        and as_float(offline.get("target_candidate_mask_rate")) >= 0.999
    )

    rollout_passed = bool(
        guarded
        and rollout_baseline
        and as_float(guarded.get("avg_utility")) > as_float(rollout_baseline.get("avg_utility"))
        and as_float(guarded.get("death_rate")) <= as_float(rollout_baseline.get("death_rate"))
    )
    live_passed = bool(
        live
        and as_int(live.get("active_rows")) > 0
        and str(live.get("latest_brain_mode")) == "neural_active"
        and str(live.get("latest_active_source")) == "neural"
        and str(live.get("feature_set")) == "signal-flight-live-v2"
        and as_int(live.get("dry_run_rows")) == 0
        and as_int(live.get("protocol_compatible_rows")) >= as_int(live.get("active_rows"))
        and as_int(live.get("replay_ready_rows")) >= as_int(live.get("active_rows"))
    )
    shadow_passed = bool(shadow and as_int(shadow.get("shadow_frames")) > 0)

    gates = {
        "signal_flight_counterfactual_replay_surface": {
            "passed": offline_passed,
            "required": False,
        },
        "signal_flight_rollout_surface": {
            "passed": rollout_passed,
            "required": False,
        },
        "signal_live_neural_active_surface": {
            "passed": live_passed,
            "required": False,
        },
        "signal_shadow_replay_surface": {
            "passed": shadow_passed,
            "required": False,
        },
    }
    source_paths = [
        path
        for path in [
            args.signal_flight_summary,
            args.signal_flight_rollout,
            args.signal_live_active,
            args.signal_shadow_replay,
        ]
        if path and path.exists()
    ]
    metrics = {
        "offline_top1": top1 if offline else None,
        "offline_baseline_top1": baseline_top1 if offline else None,
        "offline_utility_delta": selected_utility - baseline_utility if offline else None,
        "offline_regret_delta": regret - baseline_regret if offline else None,
        "rollout_neural_guarded_utility": as_float(guarded.get("avg_utility")) if guarded else None,
        "rollout_baseline_utility": as_float(rollout_baseline.get("avg_utility"))
        if rollout_baseline
        else None,
        "live_active_rows": as_int(live.get("active_rows")) if live else 0,
        "live_protocol_compatible_rows": as_int(live.get("protocol_compatible_rows"))
        if live
        else 0,
        "shadow_frames": as_int(shadow.get("shadow_frames")) if shadow else 0,
        "shadow_disagreement_rate": as_float(shadow.get("shadow_disagreement_rate"))
        if shadow
        else None,
    }
    return {
        "kind": "signal_flight_evidence",
        "gates": gates,
        "metrics": metrics,
        "source_paths": [str(path) for path in source_paths],
    }


def summarize_neural_grammar_arc_evidence(args: argparse.Namespace) -> dict[str, Any] | None:
    rows = load_jsonl(args.neural_grammar_summary)
    summary = latest_schema(rows, NEURAL_GRAMMAR_SUMMARY_SCHEMA)
    if not summary:
        return None
    proposal_summary = summary.get("arc_visual_pipeline_proposal_summary") or {}
    if not isinstance(proposal_summary, dict):
        proposal_summary = {}
    by_proposer = proposal_summary.get("by_proposer") or {}
    neural = {}
    if isinstance(by_proposer, dict):
        neural = dict(by_proposer.get("neural_visual_pipeline_ranker") or {})
    frontier = summary.get("arc_visual_pipeline_coverage_frontier_summary") or {}
    if not isinstance(frontier, dict):
        frontier = {}

    attempt_count = as_int(summary.get("arc_visual_pipeline_attempt_count"))
    heldout_attempt_count = as_int(summary.get("arc_visual_pipeline_heldout_attempt_count"))
    neural_proposals = as_int(summary.get("arc_visual_pipeline_neural_proposal_count"))
    frontier_count = as_int(summary.get("arc_visual_pipeline_coverage_frontier_count"))
    repair_sketch_count = as_int(summary.get("arc_visual_pipeline_repair_sketch_count"))
    gates = {
        "learned_visual_program_attempt_surface": {
            "passed": attempt_count + heldout_attempt_count > 0,
            "required": False,
            "authority": "arc_agi_train_verifier",
            "metric": {
                "source_attempts": attempt_count,
                "heldout_attempts": heldout_attempt_count,
            },
        },
        "learned_visual_program_ranker_surface": {
            "passed": neural_proposals > 0,
            "required": False,
            "authority": "neural_visual_pipeline_ranker",
            "metric": {
                "neural_proposals": neural_proposals,
                "top1_all_task_rate": as_float(neural.get("top1_all_task_rate")),
                "retained_all_task_rate": as_float(neural.get("retained_all_task_rate")),
            },
        },
        "learned_visual_program_frontier_surface": {
            "passed": frontier_count > 0 or repair_sketch_count > 0,
            "required": False,
            "authority": "coverage_frontier",
            "metric": {
                "frontier_tasks": frontier_count,
                "repair_sketches": repair_sketch_count,
                "coverage_rate": as_float(frontier.get("coverage_rate")),
            },
        },
    }
    return {
        "kind": "learned_visual_program_replacement",
        "gates": gates,
        "metrics": {
            "arc_visual_pipeline_attempt_count": attempt_count,
            "arc_visual_pipeline_heldout_attempt_count": heldout_attempt_count,
            "arc_visual_pipeline_neural_proposal_count": neural_proposals,
            "arc_visual_pipeline_coverage_frontier_count": frontier_count,
            "arc_visual_pipeline_repair_sketch_count": repair_sketch_count,
            "arc_visual_pipeline_proposal_summary": proposal_summary,
        },
        "source_paths": [str(args.neural_grammar_summary)],
    }


def coverage_gate_status(coverage: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    passing: list[str] = []
    failing_required: list[str] = []
    failing_optional: list[str] = []
    gates = coverage.get("gates") or {}
    if not isinstance(gates, dict):
        return passing, failing_required, failing_optional
    for name, gate in sorted(gates.items()):
        if not isinstance(gate, dict):
            continue
        if gate.get("passed") is True:
            passing.append(name)
        elif gate.get("required") is True:
            failing_required.append(name)
        else:
            failing_optional.append(name)
    return passing, failing_required, failing_optional


def scope_name(coverage: dict[str, Any], gate: str) -> str:
    scope = str(coverage.get("scope", "unknown"))
    return f"{scope}:{gate}"


def triage_item(
    *,
    priority: str,
    kind: str,
    source: str,
    summary: str,
    action: str,
    scope: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "priority": priority,
        "kind": kind,
        "source": source,
        "summary": summary,
        "action": action,
    }
    if scope:
        row["scope"] = scope
    if details:
        row["details"] = details
    return row


def classify_hole(experiment: str, hole: dict[str, Any]) -> dict[str, Any]:
    kind = str(hole.get("kind", "unknown"))
    scope = str(hole.get("scope", ""))
    if experiment == "arc_agi" and kind == "handcrafted_transform_dependency":
        return triage_item(
            priority="P0",
            kind="handcrafted_transform_dependency",
            source="coverage_hole",
            scope=scope,
            summary="ARC selected solutions still depend on legacy handcrafted transform templates.",
            action=(
                "Re-express legacy templates as visual-algebra training examples/operators "
                "and require new coverage to flow through learned visual-program proposals."
            ),
            details={
                "selected_legacy_templates": hole.get("selected_legacy_templates"),
                "selected_visual_programs": hole.get("selected_visual_programs"),
                "legacy_selected_count": hole.get("legacy_selected_count"),
                "visual_selected_count": hole.get("visual_selected_count"),
            },
        )
    if experiment == "arc_agi" and kind == "no_train_consistent_candidate":
        return triage_item(
            priority="P0",
            kind="learned_visual_program_coverage_gap",
            source="coverage_hole",
            scope=scope,
            summary=(
                "ARC public task has generated proposals but no train-consistent "
                "learned visual program."
            ),
            action=(
                "Train and compile visual-algebra program proposals for this coverage hole; "
                "do not add another bespoke transform family as the owning abstraction."
            ),
            details={
                "task": hole.get("task"),
                "attempted": hole.get("attempted"),
                "generated": hole.get("generated"),
                "attempted_visual_programs": hole.get("attempted_visual_programs"),
                "attempted_legacy_templates": hole.get("attempted_legacy_templates"),
                "legacy_train_consistent_pre_cap": hole.get(
                    "legacy_train_consistent_pre_cap"
                ),
                "top_attempted_templates": hole.get("top_attempted_templates"),
            },
        )
    if experiment == "arc_agi" and kind == "train_consistent_not_selected":
        return triage_item(
            priority="P1",
            kind="ranker_or_retention_gap",
            source="coverage_hole",
            scope=scope,
            summary="ARC task has a train-consistent candidate that was not selected.",
            action="Improve ranker features or retention budget after candidate-space holes are lower.",
            details={
                "task": hole.get("task"),
                "train_consistent_pre_cap": hole.get("train_consistent_pre_cap"),
                "retained": hole.get("retained"),
                "visual_train_consistent_pre_cap": hole.get(
                    "visual_train_consistent_pre_cap"
                ),
                "visual_train_consistent_retained": hole.get(
                    "visual_train_consistent_retained"
                ),
                "visual_selected": hole.get("visual_selected"),
                "legacy_train_consistent_pre_cap": hole.get(
                    "legacy_train_consistent_pre_cap"
                ),
                "legacy_selected": hole.get("legacy_selected"),
            },
        )
    if experiment == "haiku_understanding" and kind == "ranker_accuracy_below_gate":
        return triage_item(
            priority="P1",
            kind="coherence_ranker_gap",
            source="coverage_hole",
            scope=scope,
            summary="Haiku understanding ranker is below the coherence gate.",
            action="Train stronger coherence features/objectives and keep topic-splice families explicit.",
            details={"accuracy": hole.get("accuracy"), "minimum": hole.get("minimum")},
        )
    if experiment == "haiku_understanding" and kind == "weak_corruption_family":
        return triage_item(
            priority="P1",
            kind="weak_corruption_family",
            source="coverage_hole",
            scope=scope,
            summary=f"Haiku corruption family `{hole.get('corruption')}` is below gate.",
            action="Use this family as a hard-negative training slice for coherence discrimination.",
            details={
                "corruption": hole.get("corruption"),
                "accuracy": hole.get("accuracy"),
                "rows": hole.get("rows"),
            },
        )
    if kind == "missing_health_evidence_surface":
        return triage_item(
            priority="P3",
            kind="instrumentation_gap",
            source="health_surface",
            summary="Experiment has paper claims but no ingested health evidence surface.",
            action="Wire the experiment's generated summaries into experiment_health.",
            details={"claim_ids": hole.get("claim_ids")},
        )
    return triage_item(
        priority="P2",
        kind=kind,
        source="coverage_hole",
        scope=scope,
        summary=f"{experiment} reported coverage hole `{kind}`.",
        action="Inspect generated coverage row and either tighten or demote the gate.",
        details=hole,
    )


def classify_optional_gate(experiment: str, gate: str) -> dict[str, Any]:
    scope, _, gate_name = gate.partition(":")
    if experiment == "arc_agi" and gate_name.startswith("learned_visual_program_"):
        return triage_item(
            priority="P0",
            kind="learned_visual_program_replacement_gate",
            source="optional_gate",
            scope=scope,
            summary=f"ARC learned visual-program replacement gate `{gate_name}` is not passing.",
            action=(
                "Move coverage ownership from legacy templates into learned visual-program "
                "proposals before treating ARC as improved."
            ),
        )
    if experiment == "arc_agi" and scope.startswith("public_"):
        return triage_item(
            priority="P0",
            kind="public_arc_optional_gate",
            source="optional_gate",
            scope=scope,
            summary=f"Public ARC optional gate `{gate_name}` is not passing.",
            action=(
                "Treat this as learned visual-program compiler coverage before ranker "
                "optimization or legacy-template expansion."
            ),
        )
    if experiment == "haiku_understanding":
        return triage_item(
            priority="P1",
            kind="haiku_understanding_optional_gate",
            source="optional_gate",
            scope=scope,
            summary=f"Haiku understanding optional gate `{gate_name}` is not passing.",
            action="Keep this as an understanding/coherence benchmark gap.",
        )
    if experiment == "haiku_word_lm":
        return triage_item(
            priority="INFO",
            kind="diagnostic_generation_gate",
            source="optional_gate",
            scope=scope,
            summary=f"Haiku word generation diagnostic gate `{gate_name}` is not a promotion gate.",
            action="Do not block health unless a required adjacency/candidate budget fails.",
        )
    return triage_item(
        priority="P2",
        kind="optional_gate",
        source="optional_gate",
        scope=scope,
        summary=f"Optional gate `{gate_name}` is not passing.",
        action="Decide whether this should be a budgeted gate or an informational diagnostic.",
    )


def classify_claims(experiment: str, claims: dict[str, Any]) -> list[dict[str, Any]]:
    triage: list[dict[str, Any]] = []
    counts = claims.get("status_counts") or {}
    if counts.get("contradicted"):
        triage.append(
            triage_item(
                priority="P2",
                kind="paper_claim_hygiene",
                source="paper_claims",
                summary=f"{experiment} has contradicted generated paper claim evidence.",
                action="Rewrite the claim as a negative result or rerun the required audit before using it.",
                details={"status_counts": counts, "claim_ids": claims.get("claim_ids", [])},
            )
        )
    if counts.get("underpowered"):
        triage.append(
            triage_item(
                priority="P2",
                kind="underpowered_claim",
                source="paper_claims",
                summary=f"{experiment} has underpowered generated paper claim evidence.",
                action="Keep it in limitations until the audit reaches its declared sample/plateau bar.",
                details={"status_counts": counts, "claim_ids": claims.get("claim_ids", [])},
            )
        )
    return triage


def warning_priority(item: dict[str, Any]) -> bool:
    return str(item.get("priority")) in {"P0", "P1", "P2"}


def build_health_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    coverage_rows = [
        row for row in load_jsonl(args.coverage) if row.get("schema") == COVERAGE_SCHEMA
    ]
    numeric = summarize_numeric(
        load_jsonl(args.numeric_summary),
        load_jsonl(args.numeric_report),
    )
    claim_support = load_claim_support(args.paper_claims)
    evidence_by_experiment: dict[str, dict[str, Any]] = {}
    signal_evidence = summarize_signal_flight_evidence(args)
    if signal_evidence is not None:
        evidence_by_experiment["signal_flight"] = signal_evidence
    neural_grammar_evidence = summarize_neural_grammar_arc_evidence(args)
    if neural_grammar_evidence is not None:
        evidence_by_experiment["arc_agi"] = neural_grammar_evidence

    experiments = sorted(
        set(numeric)
        | {str(row.get("experiment")) for row in coverage_rows if row.get("experiment")}
        | set(claim_support)
        | set(evidence_by_experiment)
    )
    coverage_by_experiment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in coverage_rows:
        if row.get("experiment"):
            coverage_by_experiment[str(row.get("experiment"))].append(row)

    health_rows: list[dict[str, Any]] = []
    for experiment in experiments:
        numeric_info = numeric.get(experiment, {})
        coverages = sorted(
            coverage_by_experiment.get(experiment, []),
            key=lambda row: str(row.get("scope", "")),
        )
        claims = claim_support.get(
            experiment,
            {
                "status_counts": {},
                "claim_ids": [],
                "required_claim_ids": [],
                "supports_paper_claim": False,
            },
        )
        passing_gates: list[str] = []
        failing_required: list[str] = []
        failing_optional: list[str] = []
        coverage_holes: list[dict[str, Any]] = []
        verifier_delta: dict[str, Any] = {}
        coverage_learning_signal: list[dict[str, Any]] = []
        coverage_sources: list[str] = []
        evidence_sources: list[str] = []
        coverage_claim_scopes: dict[str, Any] = {}
        verifier_certificate_rows = 0
        triage: list[dict[str, Any]] = []
        for coverage in coverages:
            scope = str(coverage.get("scope", "unknown"))
            passed, required, optional = coverage_gate_status(coverage)
            passing_gates.extend(scope_name(coverage, gate) for gate in passed)
            failing_required.extend(scope_name(coverage, gate) for gate in required)
            failing_optional.extend(scope_name(coverage, gate) for gate in optional)
            triage.extend(classify_optional_gate(experiment, scope_name(coverage, gate)) for gate in optional)
            for hole in coverage.get("coverage_holes") or []:
                if isinstance(hole, dict):
                    scoped_hole = {"scope": scope, **hole}
                    coverage_holes.append(scoped_hole)
                    triage.append(classify_hole(experiment, scoped_hole))
                else:
                    scoped_hole = {"scope": scope, "hole": hole}
                    coverage_holes.append(scoped_hole)
                    triage.append(classify_hole(experiment, scoped_hole))
            for violation in coverage.get("budget_violations") or []:
                if isinstance(violation, dict):
                    scoped_violation = {"scope": scope, **violation}
                    coverage_holes.append(scoped_violation)
                    triage.append(
                        triage_item(
                            priority="P0",
                            kind="coverage_budget_violation",
                            source="budget",
                            scope=scope,
                            summary="Learned grammar coverage budget was violated.",
                            action="Treat this as a regression and restore the budgeted metric.",
                            details=scoped_violation,
                        )
                    )
            verifier_delta[scope] = coverage.get("verifier_delta", {})
            coverage_learning_signal.append(
                {
                    "scope": scope,
                    "learning_signal": coverage.get("learning_signal", {}),
                    "metrics": coverage.get("metrics", {}),
                    "budget": coverage.get("budget", {}),
                    "budget_violations": coverage.get("budget_violations", []),
                }
            )
            coverage_sources.extend(str(path) for path in coverage.get("source_paths", []))
            coverage_claim_scopes[scope] = coverage.get("paper_claim_support", "none")
            verifier_certificate_rows += as_int(coverage.get("certificate_rows"))

        evidence_surface = evidence_by_experiment.get(experiment)
        if evidence_surface:
            evidence_sources.extend(evidence_surface.get("source_paths", []))
            evidence_gates = evidence_surface.get("gates") or {}
            for gate_name, gate in sorted(evidence_gates.items()):
                if not isinstance(gate, dict):
                    continue
                if gate.get("passed") is True:
                    passing_gates.append(gate_name)
                elif gate.get("required") is True:
                    failing_required.append(gate_name)
                else:
                    failing_optional.append(gate_name)
                    triage.append(
                        triage_item(
                            priority="P3",
                            kind="evidence_surface_gap",
                            source="evidence_surface",
                            summary=f"Evidence surface `{gate_name}` is present but incomplete.",
                            action="Refresh or rerun the corresponding generated signal-flight artifact.",
                        )
                    )

        numeric_rejected = as_int(numeric_info.get("rejected_rows"))
        numeric_targets = numeric_info.get("targets", [])
        numeric_passed = numeric_rejected == 0 and all(
            status != "failed" for status in (numeric_info.get("statuses") or {})
        )
        if numeric_targets:
            passing_gates.append("numeric_certificates_committed")

        claim_status_counts = claims.get("status_counts") or {}
        triage.extend(classify_claims(experiment, claims))
        has_health_evidence = bool(numeric_targets or coverages or evidence_surface)
        if claims.get("claim_ids") and not has_health_evidence:
            hole = {
                "kind": "missing_health_evidence_surface",
                "claim_ids": claims.get("claim_ids"),
            }
            coverage_holes.append(hole)
            triage.append(classify_hole(experiment, hole))
        if numeric_rejected > 0 or failing_required:
            status = "fail"
        elif any(warning_priority(item) for item in triage):
            status = "warn"
        else:
            status = "pass"
        triage_counts = Counter(str(item.get("kind", "unknown")) for item in triage)
        triage_priorities = Counter(str(item.get("priority", "unknown")) for item in triage)

        health_rows.append(
            {
                "schema": HEALTH_SCHEMA,
                "experiment": experiment,
                "status": status,
                "passing_gates": sorted(set(passing_gates)),
                "failing_required_gates": failing_required,
                "failing_optional_gates": failing_optional,
                "certificate_rows": {
                    "numeric": as_int(numeric_info.get("certificate_rows")),
                    "verifier": verifier_certificate_rows,
                },
                "learning_signal": {
                    "numeric_targets": numeric_targets,
                    "numeric_committed_rows": as_int(numeric_info.get("committed_rows")),
                    "coverage": coverage_learning_signal,
                    "evidence_surface": evidence_surface,
                },
                "coverage_holes": coverage_holes,
                "triage": triage,
                "triage_summary": {
                    "by_kind": dict(sorted(triage_counts.items())),
                    "by_priority": dict(sorted(triage_priorities.items())),
                },
                "verifier_delta": verifier_delta,
                "numeric_certificate": {
                    "passed": numeric_passed,
                    "rejected_rows": numeric_rejected,
                    "rejection_reasons": numeric_info.get("rejection_reasons", {}),
                    "statuses": numeric_info.get("statuses", {}),
                },
                "paper_claim_support": {
                    "coverage_scope": coverage_claim_scopes,
                    **claims,
                },
                "sources": {
                    "coverage": sorted(set(coverage_sources)),
                    "evidence": sorted(set(evidence_sources)),
                    "numeric_targets": numeric_targets,
                    "paper_claims": str(args.paper_claims) if args.paper_claims else None,
                },
            }
        )

    summary = {
        "schema": SUMMARY_SCHEMA,
        "experiments": len(health_rows),
        "status_counts": dict(Counter(row["status"] for row in health_rows)),
        "triage_priority_counts": dict(
            sorted(
                Counter(
                    str(item.get("priority", "unknown"))
                    for row in health_rows
                    for item in row.get("triage", [])
                ).items()
            )
        ),
        "triage_kind_counts": dict(
            sorted(
                Counter(
                    str(item.get("kind", "unknown"))
                    for row in health_rows
                    for item in row.get("triage", [])
                ).items()
            )
        ),
        "failing_experiments": sorted(
            row["experiment"] for row in health_rows if row["status"] == "fail"
        ),
        "warning_experiments": sorted(
            row["experiment"] for row in health_rows if row["status"] == "warn"
        ),
    }
    return health_rows + [summary]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            fp.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--numeric-summary", type=Path)
    parser.add_argument("--numeric-report", type=Path)
    parser.add_argument("--paper-claims", type=Path)
    parser.add_argument(
        "--signal-flight-summary",
        type=Path,
        default=Path("results/2026-05-17-signal-flight/summary.jsonl"),
    )
    parser.add_argument(
        "--signal-flight-rollout",
        type=Path,
        default=Path("results/2026-05-17-signal-flight/rollout_summary.jsonl"),
    )
    parser.add_argument(
        "--signal-live-active",
        type=Path,
        default=Path("results/2026-05-17-signal-flight/live_active_summary.jsonl"),
    )
    parser.add_argument(
        "--signal-shadow-replay",
        type=Path,
        default=Path("results/2026-05-17-signal-flight/shadow_replay_audit.jsonl"),
    )
    parser.add_argument(
        "--neural-grammar-summary",
        type=Path,
        default=Path("build/float/neural_grammar_discovery/summary.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_health_rows(args)
    write_jsonl(args.output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
