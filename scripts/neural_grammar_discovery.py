#!/usr/bin/env python3
"""Shared neural-to-symbolic grammar-delta discovery driver.

The engine owns the artifact shape and promotion ledger. Domain adapters only
translate local failure traces into candidate grammar deltas and verifier
evidence. A neural discovery row is never promoted here unless a domain verifier
has already certified an executable delta.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CANDIDATE_SCHEMA = "crlp.neural_grammar_discovery.delta_candidate.v1"
COMPILED_DELTA_SCHEMA = "crlp.neural_grammar_discovery.compiled_delta.v1"
VERIFICATION_SCHEMA = "crlp.neural_grammar_discovery.delta_verification.v1"
PROMOTED_SCHEMA = "crlp.neural_grammar_discovery.promoted_delta.v1"
SUMMARY_SCHEMA = "crlp.neural_grammar_discovery.summary.v1"
GRAMMAR_SKETCH_SCHEMA = "crlp.grammar_sketch.v1"
GRAMMAR_SKETCH_EXAMPLE_SCHEMA = "crlp.grammar_sketch.example.v1"
GRAMMAR_SKETCH_PROPOSAL_SCHEMA = "crlp.grammar_sketch.proposal.v1"
GRAMMAR_SKETCH_VERIFICATION_SCHEMA = "crlp.grammar_sketch.verification.v1"
PROMOTED_GRAMMAR_SKETCH_SCHEMA = "crlp.grammar_sketch.promoted.v1"


GRID_ACTIONS = {
    0: ("up", 0, -1),
    1: ("right", 1, 0),
    2: ("down", 0, 1),
    3: ("left", -1, 0),
}


ARC_PRIORITY_TO_DELTA = {
    "mine_same_shape_patch_lattice_templates": {
        "family": "arc.patch_lattice_template_family",
        "delta_kind": "template_family",
        "description": "Add patch/lattice same-shape transform templates.",
        "required_adapter": "arc_template_family_compiler",
    },
    "add_separator_panel_extraction_templates": {
        "family": "arc.separator_panel_extraction_family",
        "delta_kind": "template_family",
        "description": "Add panel extraction templates driven by separator structure.",
        "required_adapter": "arc_template_family_compiler",
    },
    "add_tiling_scale_or_canvas_growth_templates": {
        "family": "arc.tiling_canvas_growth_family",
        "delta_kind": "template_family",
        "description": "Add tiling, scale, and canvas-growth templates.",
        "required_adapter": "arc_template_family_compiler",
    },
    "broaden_crop_object_selection_templates": {
        "family": "arc.crop_object_selection_family",
        "delta_kind": "template_family",
        "description": "Broaden crop/object extraction templates.",
        "required_adapter": "arc_template_family_compiler",
    },
    "add_projection_compression_templates": {
        "family": "arc.projection_compression_family",
        "delta_kind": "template_family",
        "description": "Add projection/compression transform templates.",
        "required_adapter": "arc_template_family_compiler",
    },
    "inspect_for_compositional_template": {
        "family": "arc.compositional_transform_family",
        "delta_kind": "template_family",
        "description": "Add composed transform templates when no single family covers the cluster.",
        "required_adapter": "arc_template_composition_compiler",
    },
}

HAIKU_CORRUPTION_TO_DELTA = {
    "line_shuffle": {
        "family": "haiku.line_order_transition_constraint",
        "delta_kind": "constraint_family",
        "description": "Learn line-order transitions that distinguish permuted poems.",
        "required_adapter": "haiku_constraint_compiler",
    },
    "line_splice": {
        "family": "haiku.line_source_consistency_constraint",
        "delta_kind": "constraint_family",
        "description": "Learn cross-line source/coherence constraints for spliced lines.",
        "required_adapter": "haiku_constraint_compiler",
    },
    "syllable_break": {
        "family": "haiku.syllable_path_constraint",
        "delta_kind": "constraint_family",
        "description": "Learn stricter syllable-path constraints for form-valid generation.",
        "required_adapter": "haiku_constraint_compiler",
    },
    "repetition": {
        "family": "haiku.repetition_constraint",
        "delta_kind": "constraint_family",
        "description": "Learn repetition and lexical reuse constraints.",
        "required_adapter": "haiku_constraint_compiler",
    },
    "word_salad": {
        "family": "haiku.content_cooccurrence_constraint",
        "delta_kind": "constraint_family",
        "description": "Learn content-word co-occurrence constraints against word salad.",
        "required_adapter": "haiku_constraint_compiler",
    },
    "topic_splice": {
        "family": "haiku.topic_coherence_transition_constraint",
        "delta_kind": "constraint_family",
        "description": "Learn topic-transition constraints for cross-poem topic splices.",
        "required_adapter": "haiku_constraint_compiler",
    },
    "same_season_splice": {
        "family": "haiku.topic_coherence_transition_constraint",
        "delta_kind": "constraint_family",
        "description": "Learn coherence constraints beyond shared season words.",
        "required_adapter": "haiku_constraint_compiler",
    },
    "plausible_topic_splice": {
        "family": "haiku.topic_coherence_transition_constraint",
        "delta_kind": "constraint_family",
        "description": "Learn constraints for form-valid, adjacency-plausible coherence negatives.",
        "required_adapter": "haiku_constraint_compiler",
    },
}

HAIKU_GENERATION_DELTAS = {
    "strict_adjacency_bottleneck": {
        "family": "haiku.latent_adjacency_class_transition",
        "delta_kind": "constraint_family",
        "description": "Learn class-level adjacency transitions when strict word adjacency collapses the candidate set.",
        "required_adapter": "haiku_adjacency_delta_compiler",
    },
    "continuation_viability_bottleneck": {
        "family": "haiku.continuation_viability_constraint",
        "delta_kind": "constraint_family",
        "description": "Learn continuation constraints that avoid forced single-path or dead-end prefixes.",
        "required_adapter": "haiku_adjacency_delta_compiler",
    },
    "line_shape_bottleneck": {
        "family": "haiku.line_shape_transition_constraint",
        "delta_kind": "constraint_family",
        "description": "Learn line-shape transition constraints from syllable and line-length rejections.",
        "required_adapter": "haiku_shape_delta_compiler",
    },
}


HAIKU_UNDERSTANDING_FAMILY_CORRUPTIONS = {
    "haiku.line_order_transition_constraint": {"line_shuffle"},
    "haiku.line_source_consistency_constraint": {"line_splice"},
    "haiku.syllable_path_constraint": {"syllable_break"},
    "haiku.repetition_constraint": {"repetition"},
    "haiku.content_cooccurrence_constraint": {"word_salad"},
    "haiku.topic_coherence_transition_constraint": {
        "topic_splice",
        "same_season_splice",
        "plausible_topic_splice",
    },
}


@dataclass(frozen=True)
class CandidateGroup:
    domain: str
    family: str
    delta_kind: str
    description: str
    required_adapter: str
    rows: tuple[dict[str, Any], ...]
    source_artifacts: tuple[str, ...]


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def parse_json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line.startswith("{"):
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            fp.write("\n")


def stable_id(*parts: Any) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(json.dumps(part, sort_keys=True, default=str).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def pct(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / float(total)


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def top_examples(rows: tuple[dict[str, Any], ...], key: str, limit: int = 12) -> list[str]:
    values = [str(row.get(key, "")) for row in rows if row.get(key, "")]
    return sorted(dict.fromkeys(values))[:limit]


def normalized_slot(name: str, value: Any, slot_type: str = "symbol") -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": name,
        "type": slot_type,
        "value": value,
    }
    if isinstance(value, bool):
        out["integer"] = int(value)
    elif isinstance(value, int):
        out["integer"] = value
    elif isinstance(value, float) and value.is_integer():
        out["integer"] = int(value)
    return out


def sketch_signature(
    domain: str,
    family: str,
    slots: list[dict[str, Any]],
    atoms: list[str],
    composition: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "domain": domain,
        "family": family,
        "slots": sorted(slots, key=lambda row: str(row.get("name", ""))),
        "atoms": sorted(dict.fromkeys(str(atom) for atom in atoms if atom)),
        "composition": composition or [],
    }


def make_grammar_sketch(
    domain: str,
    family: str,
    slots: list[dict[str, Any]],
    atoms: list[str],
    *,
    compiler_target: str,
    source_artifact: str,
    composition: list[str] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    signature = sketch_signature(domain, family, slots, atoms, composition)
    sketch_id = "gsk:{}".format(stable_id(signature))
    return {
        "schema": GRAMMAR_SKETCH_SCHEMA,
        "report_version": 1,
        "sketch_id": sketch_id,
        "domain": domain,
        "family": family,
        "compiler_target": compiler_target,
        "slots": signature["slots"],
        "structural_atoms": signature["atoms"],
        "composition": signature["composition"],
        "numeric_params": params or {},
        "source_artifacts": [source_artifact],
        "compile_status": "existing_domain_surface",
        "promotion_status": "candidate",
    }


def merge_sketch(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    source_artifacts = sorted(
        set(existing.get("source_artifacts", [])) | set(incoming.get("source_artifacts", []))
    )
    existing = dict(existing)
    existing["source_artifacts"] = source_artifacts
    return existing


def sketch_tokens(sketch: dict[str, Any]) -> set[str]:
    tokens = {
        f"domain:{sketch.get('domain', '')}",
        f"family:{sketch.get('family', '')}",
        f"compiler:{sketch.get('compiler_target', '')}",
    }
    for atom in sketch.get("structural_atoms", []) or []:
        tokens.add(f"atom:{atom}")
    for slot in sketch.get("slots", []) or []:
        name = str(slot.get("name", ""))
        value = str(slot.get("integer", slot.get("value", "")))
        tokens.add(f"slot:{name}")
        tokens.add(f"slot:{name}={value}")
    return tokens


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def add_sketch_example(
    sketches: dict[str, dict[str, Any]],
    examples: list[dict[str, Any]],
    sketch: dict[str, Any],
    *,
    example_id: str,
    outcome: str,
    verifier_status: str,
    source_row_schema: str,
    source_artifact: str,
    source_ref: dict[str, Any],
    utility: float = 0.0,
    neural_score: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    sketch_id = str(sketch["sketch_id"])
    if sketch_id in sketches:
        sketches[sketch_id] = merge_sketch(sketches[sketch_id], sketch)
    else:
        sketches[sketch_id] = sketch
    row: dict[str, Any] = {
        "schema": GRAMMAR_SKETCH_EXAMPLE_SCHEMA,
        "report_version": 1,
        "example_id": example_id,
        "sketch_id": sketch_id,
        "domain": sketch["domain"],
        "family": sketch["family"],
        "outcome": outcome,
        "verifier_status": verifier_status,
        "source_row_schema": source_row_schema,
        "source_artifacts": [source_artifact],
        "source_ref": source_ref,
        "utility": utility,
        "metadata": metadata or {},
    }
    if neural_score is not None:
        row["neural_score"] = neural_score
    examples.append(row)


def feature_summary(domain: str, family: str, rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    if domain == "arc_agi":
        return {
            "mean_slot_probe_rows": mean(
                [safe_float(row.get("slot_probe_rows")) for row in rows]
            ),
            "mean_mutation_proposals_generated": mean(
                [safe_float(row.get("mutation_proposals_generated")) for row in rows]
            ),
            "mean_promotion_proposals_generated": mean(
                [safe_float(row.get("promotion_proposals_generated")) for row in rows]
            ),
            "current_grammar_train_consistent": sum(
                safe_int(row.get("promotion_proposals_train_consistent_pre_cap"))
                for row in rows
            ),
            "mutation_train_consistent_rows": sum(
                safe_int(row.get("mutation_train_consistent_rows")) for row in rows
            ),
        }
    if domain == "haiku":
        if family.startswith("haiku.latent_") or family.startswith("haiku.continuation_") or family.startswith("haiku.line_shape_"):
            rejected = [
                row.get("rejected_candidates", {}) for row in rows if isinstance(row, dict)
            ]
            return {
                "mean_candidate_count": mean(
                    [safe_float(row.get("candidate_count")) for row in rows]
                ),
                "one_candidate_ratio": pct(
                    sum(1 for row in rows if safe_int(row.get("candidate_count")) <= 1),
                    len(rows),
                ),
                "mean_adjacency_rejections": mean(
                    [safe_float(row.get("adjacency")) for row in rejected]
                ),
                "mean_syllable_rejections": mean(
                    [safe_float(row.get("syllable")) for row in rejected]
                ),
                "mean_dead_end_rejections": mean(
                    [safe_float(row.get("dead_end")) for row in rejected]
                ),
            }
        return {
            "neural_failure_ratio": pct(
                sum(1 for row in rows if row.get("selected") != "real"),
                len(rows),
            ),
            "symbolic_failure_ratio": pct(
                sum(1 for row in rows if row.get("symbolic_selected") != "real"),
                len(rows),
            ),
            "mean_plan_score_delta": mean(
                [safe_float(row.get("plan_score_delta")) for row in rows]
            ),
            "mean_topic_tightness_delta": mean(
                [safe_float(row.get("topic_tightness_delta")) for row in rows]
            ),
            "mean_min_line_similarity_delta": mean(
                [safe_float(row.get("min_line_similarity_delta")) for row in rows]
            ),
            "mean_corrupt_adjacency_ratio": mean(
                [safe_float(row.get("corrupt_adjacency_ratio")) for row in rows]
            ),
        }
    return {}


def build_candidate(group: CandidateGroup) -> dict[str, Any]:
    row_count = len(group.rows)
    evidence_keys: list[str]
    if group.domain == "arc_agi":
        evidence_keys = top_examples(group.rows, "task_id")
    elif group.domain == "haiku":
        evidence_keys = top_examples(group.rows, "example_id")
    else:
        evidence_keys = []
    candidate_id = "{}:{}:{}".format(
        group.domain,
        group.family.replace(".", "_"),
        stable_id(group.domain, group.family, evidence_keys, row_count),
    )
    return {
        "schema": CANDIDATE_SCHEMA,
        "report_version": 1,
        "candidate_id": candidate_id,
        "domain": group.domain,
        "family": group.family,
        "delta_kind": group.delta_kind,
        "description": group.description,
        "required_adapter": group.required_adapter,
        "support_examples": row_count,
        "evidence_examples": evidence_keys,
        "source_artifacts": list(group.source_artifacts),
        "neural_cluster": {
            "cluster_key": group.family,
            "feature_summary": feature_summary(group.domain, group.family, group.rows),
        },
        "proposed_delta": {
            "status": "candidate",
            "family": group.family,
            "compiler_contract": "domain adapter must compile this family into executable grammar before promotion",
        },
    }


def compile_delta(candidate: dict[str, Any], group: CandidateGroup) -> dict[str, Any]:
    compiler = "unsupported"
    executable = False
    program: dict[str, Any] = {}
    if group.domain == "arc_agi":
        compiler = "arc_missing_family_gap_compiler"
        executable = True
        program = {
            "runtime": "python_trace_predicate_v1",
            "verifier": "arc_failure_proof_gap_replay",
            "family": group.family,
            "required_predicates": [
                "accepted_existing_rows == 0",
                "missing_family_evidence == true",
                "promotion_replayed == true",
                "promotion_proposals_train_consistent_pre_cap == 0",
            ],
            "support_task_ids": top_examples(group.rows, "task_id", limit=64),
            "promotion_scope": "diagnostic_gap_certificate",
        }
    elif group.domain == "haiku":
        if group.family in {
            "haiku.latent_adjacency_class_transition",
            "haiku.continuation_viability_constraint",
        }:
            compiler = "haiku_generation_trace_constraint_compiler"
            executable = True
            program = {
                "runtime": "python_trace_predicate_v1",
                "verifier": "haiku_word_generation_trace_replay",
                "family": group.family,
                "adjacency_hashes": sorted(
                    {
                        str(row.get("adjacency_hash", ""))
                        for row in group.rows
                        if row.get("adjacency_hash", "")
                    }
                ),
                "required_gates": ["dataset_vocab", "dataset_adjacency", "haiku_shape_proxy"],
            }
        elif group.family == "haiku.line_shape_transition_constraint":
            compiler = "haiku_shape_trace_constraint_compiler"
            executable = True
            program = {
                "runtime": "python_trace_predicate_v1",
                "verifier": "haiku_word_generation_trace_replay",
                "family": group.family,
                "required_gates": ["haiku_shape_proxy"],
            }
        elif group.family in HAIKU_UNDERSTANDING_FAMILY_CORRUPTIONS:
            compiler = "haiku_understanding_symbolic_constraint_compiler"
            executable = True
            program = {
                "runtime": "python_trace_predicate_v1",
                "verifier": "haiku_understanding_symbolic_replay",
                "family": group.family,
                "corruptions": sorted(HAIKU_UNDERSTANDING_FAMILY_CORRUPTIONS[group.family]),
                "required_predicates": ["symbolic_selected == real"],
            }
    return {
        "schema": COMPILED_DELTA_SCHEMA,
        "report_version": 1,
        "candidate_id": candidate["candidate_id"],
        "domain": group.domain,
        "family": group.family,
        "compiler": compiler,
        "executable": executable,
        "program": program,
        "source_artifacts": list(group.source_artifacts),
    }


def verify_arc_compiled_delta(group: CandidateGroup) -> tuple[bool, str, str, dict[str, Any]]:
    counterexamples: list[dict[str, Any]] = []
    for row in group.rows:
        failures: list[str] = []
        if safe_int(row.get("accepted_existing_rows")) != 0:
            failures.append("accepted_existing_rows")
        if not bool(row.get("missing_family_evidence")):
            failures.append("missing_family_evidence")
        if not bool(row.get("promotion_replayed")):
            failures.append("promotion_replayed")
        if safe_int(row.get("promotion_proposals_train_consistent_pre_cap")) != 0:
            failures.append("promotion_proposals_train_consistent_pre_cap")
        if failures:
            counterexamples.append(
                {
                    "task_id": row.get("task_id"),
                    "split": row.get("split"),
                    "failed_predicates": failures,
                }
            )
    summary = {
        "support_examples": len(group.rows),
        "counterexamples": len(counterexamples),
        "counterexample_rows": counterexamples[:8],
        "promotion_scope": "diagnostic_gap_certificate",
        "mean_promotion_proposals_generated": mean(
            [safe_float(row.get("promotion_proposals_generated")) for row in group.rows]
        ),
        "mutation_train_consistent_rows": sum(
            safe_int(row.get("mutation_train_consistent_rows")) for row in group.rows
        ),
    }
    if counterexamples:
        return (
            False,
            "arc_gap_certificate_counterexamples",
            "compiled ARC diagnostic gap program failed replay predicates",
            summary,
        )
    return True, "arc_gap_certificate_replayed", "", summary


def haiku_generation_row_covered(family: str, row: dict[str, Any]) -> bool:
    rejects = row.get("rejected_candidates", {})
    if not isinstance(rejects, dict):
        rejects = {}
    if family == "haiku.latent_adjacency_class_transition":
        return (
            bool(row.get("strict_adjacency"))
            and safe_int(row.get("candidate_count")) <= 2
            and safe_int(rejects.get("adjacency")) > 0
        )
    if family == "haiku.continuation_viability_constraint":
        return safe_int(row.get("candidate_count")) <= 1 or safe_int(rejects.get("dead_end")) > 0
    if family == "haiku.line_shape_transition_constraint":
        return safe_int(rejects.get("syllable")) + safe_int(rejects.get("line_chars")) > 0
    return False


def verify_haiku_compiled_delta(group: CandidateGroup) -> tuple[bool, str, str, dict[str, Any]]:
    counterexamples: list[dict[str, Any]] = []
    if group.family in {
        "haiku.latent_adjacency_class_transition",
        "haiku.continuation_viability_constraint",
        "haiku.line_shape_transition_constraint",
    }:
        for row in group.rows:
            if not haiku_generation_row_covered(group.family, row):
                counterexamples.append(
                    {
                        "example_id": row.get("example_id"),
                        "sample": row.get("sample"),
                        "step": row.get("step"),
                        "candidate_count": row.get("candidate_count"),
                    }
                )
        summary = {
            "support_examples": len(group.rows),
            "counterexamples": len(counterexamples),
            "counterexample_rows": counterexamples[:8],
            "verifier": "haiku_word_generation_trace_replay",
        }
        if counterexamples:
            return (
                False,
                "haiku_generation_constraint_counterexamples",
                "compiled haiku generation constraint did not cover every support row",
                summary,
            )
        return True, "haiku_generation_constraint_replayed", "", summary

    if group.family in HAIKU_UNDERSTANDING_FAMILY_CORRUPTIONS:
        allowed_corruptions = HAIKU_UNDERSTANDING_FAMILY_CORRUPTIONS[group.family]
        for row in group.rows:
            failures: list[str] = []
            if str(row.get("corruption", "")) not in allowed_corruptions:
                failures.append("corruption_family")
            if row.get("symbolic_selected") != "real":
                failures.append("symbolic_selected")
            if failures:
                counterexamples.append(
                    {
                        "example_id": row.get("example_id"),
                        "poem_index": row.get("poem_index"),
                        "corruption": row.get("corruption"),
                        "symbolic_selected": row.get("symbolic_selected"),
                        "failed_predicates": failures,
                    }
                )
        summary = {
            "support_examples": len(group.rows),
            "counterexamples": len(counterexamples),
            "counterexample_rows": counterexamples[:8],
            "verifier": "haiku_understanding_symbolic_replay",
        }
        if counterexamples:
            return (
                False,
                "haiku_understanding_symbolic_counterexamples",
                "existing symbolic verifier does not certify every support row",
                summary,
            )
        return True, "haiku_understanding_symbolic_replayed", "", summary

    return (
        False,
        "haiku_no_compiler",
        "no executable haiku compiler for candidate family",
        {"support_examples": len(group.rows)},
    )


def build_verification(
    candidate: dict[str, Any],
    group: CandidateGroup,
    compiled: dict[str, Any],
) -> dict[str, Any]:
    summary = dict(candidate["neural_cluster"]["feature_summary"])
    if not compiled.get("executable"):
        certified = False
        promoted = False
        verifier_status = "no_executable_compiler"
        blocking_reason = "no executable compiler for proposed grammar delta"
    elif group.domain == "arc_agi":
        certified, verifier_status, blocking_reason, verification_summary = (
            verify_arc_compiled_delta(group)
        )
        promoted = certified
        summary.update(verification_summary)
    elif group.domain == "haiku":
        certified, verifier_status, blocking_reason, verification_summary = (
            verify_haiku_compiled_delta(group)
        )
        promoted = certified
        summary.update(verification_summary)
    else:
        certified = False
        promoted = False
        verifier_status = "unknown_domain"
        blocking_reason = "no domain verifier adapter"
    return {
        "schema": VERIFICATION_SCHEMA,
        "report_version": 1,
        "candidate_id": candidate["candidate_id"],
        "domain": group.domain,
        "family": group.family,
        "compiled_delta_schema": compiled.get("schema"),
        "compiler": compiled.get("compiler"),
        "verifier_status": verifier_status,
        "certified": certified,
        "promoted": promoted,
        "support_examples": len(group.rows),
        "source_artifacts": list(group.source_artifacts),
        "blocking_reason": blocking_reason,
        "verification_summary": summary,
    }


def arc_groups(arc_failure_proof: Path) -> list[CandidateGroup]:
    rows = [
        row
        for row in parse_json_rows(arc_failure_proof)
        if safe_int(row.get("accepted_existing_rows")) == 0
        and row.get("classification") != "accepted_existing_control"
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        priority = str(row.get("triage_priority", ""))
        if priority in ARC_PRIORITY_TO_DELTA:
            grouped[priority].append(row)

    groups: list[CandidateGroup] = []
    for priority, items in sorted(grouped.items()):
        spec = ARC_PRIORITY_TO_DELTA[priority]
        groups.append(
            CandidateGroup(
                domain="arc_agi",
                family=spec["family"],
                delta_kind=spec["delta_kind"],
                description=spec["description"],
                required_adapter=spec["required_adapter"],
                rows=tuple(items),
                source_artifacts=(str(arc_failure_proof),),
            )
        )
    return groups


def generation_pressure_groups(word_trace: Path) -> list[CandidateGroup]:
    rows = [
        row
        for row in parse_json_rows(word_trace)
        if row.get("schema") == "crlp.haiku_word_lm.generation_trace.v1"
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rejects = row.get("rejected_candidates", {})
        if not isinstance(rejects, dict):
            rejects = {}
        row = dict(row)
        row["example_id"] = "sample:{}:step:{}".format(
            row.get("sample", 0), row.get("step", 0)
        )
        candidate_count = safe_int(row.get("candidate_count"))
        adjacency_rejects = safe_int(rejects.get("adjacency"))
        dead_end_rejects = safe_int(rejects.get("dead_end"))
        shape_rejects = safe_int(rejects.get("syllable")) + safe_int(
            rejects.get("line_chars")
        )
        if row.get("strict_adjacency") and candidate_count <= 2 and adjacency_rejects > 0:
            groups["strict_adjacency_bottleneck"].append(row)
        if candidate_count <= 1 or dead_end_rejects > 0:
            groups["continuation_viability_bottleneck"].append(row)
        if shape_rejects > 0:
            groups["line_shape_bottleneck"].append(row)

    out: list[CandidateGroup] = []
    for key, items in sorted(groups.items()):
        spec = HAIKU_GENERATION_DELTAS[key]
        out.append(
            CandidateGroup(
                domain="haiku",
                family=spec["family"],
                delta_kind=spec["delta_kind"],
                description=spec["description"],
                required_adapter=spec["required_adapter"],
                rows=tuple(items),
                source_artifacts=(str(word_trace),),
            )
        )
    return out


def understanding_pressure_groups(understanding_trace: Path) -> list[CandidateGroup]:
    rows = [
        row
        for row in parse_json_rows(understanding_trace)
        if row.get("schema") == "crlp.haiku_understanding.trace.v1"
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        corruption = str(row.get("corruption", ""))
        if corruption not in HAIKU_CORRUPTION_TO_DELTA:
            continue
        neural_failed = row.get("selected") != "real"
        symbolic_failed = row.get("symbolic_selected") != "real"
        if not neural_failed and not symbolic_failed:
            continue
        row = dict(row)
        row["example_id"] = "poem:{}:{}".format(row.get("poem_index", 0), corruption)
        groups[corruption].append(row)

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    family_specs: dict[str, dict[str, str]] = {}
    for corruption, items in groups.items():
        spec = HAIKU_CORRUPTION_TO_DELTA[corruption]
        by_family[spec["family"]].extend(items)
        family_specs[spec["family"]] = spec

    out: list[CandidateGroup] = []
    for family, items in sorted(by_family.items()):
        spec = family_specs[family]
        out.append(
            CandidateGroup(
                domain="haiku",
                family=family,
                delta_kind=spec["delta_kind"],
                description=spec["description"],
                required_adapter=spec["required_adapter"],
                rows=tuple(items),
                source_artifacts=(str(understanding_trace),),
            )
        )
    return out


def haiku_groups(word_trace: Path, understanding_trace: Path) -> list[CandidateGroup]:
    return [
        *generation_pressure_groups(word_trace),
        *understanding_pressure_groups(understanding_trace),
    ]


def program_synthesis_sketch_examples(
    source_path: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    sketches: dict[str, dict[str, Any]] = {}
    examples: list[dict[str, Any]] = []
    source = str(source_path)
    for row in parse_json_rows(source_path):
        if (
            row.get("schema") != "crlp.hypothesis.evidence.v1"
            or row.get("task") != "program_synthesis"
        ):
            continue
        template_name = str(row.get("template_name", "unknown_template"))
        slots = [
            normalized_slot(
                str(slot.get("name", "")),
                slot.get("integer", slot.get("value")),
                str(slot.get("type", "symbol")),
            )
            for slot in row.get("slots", []) or []
            if isinstance(slot, dict)
        ]
        sketch = make_grammar_sketch(
            "program_synthesis",
            f"program_synthesis.{template_name}",
            slots,
            [
                str(row.get("proof_kind", "")),
                f"template_id:{row.get('template_id', '')}",
                "typed_list_int_dsl",
            ],
            compiler_target="program_synthesis_list_dsl",
            source_artifact=source,
            composition=[str(row.get("program_canonical_id", ""))],
        )
        status = str(row.get("certificate_status", ""))
        false_rejections = safe_int(row.get("false_rejections"))
        outcome = "accepted" if status == "certified" and false_rejections == 0 else "unsafe"
        add_sketch_example(
            sketches,
            examples,
            sketch,
            example_id="program_synthesis:{}:{}".format(
                row.get("task_instance", "unknown"), row.get("canonical_hash", "unknown")
            ),
            outcome=outcome,
            verifier_status=status or "unknown",
            source_row_schema=str(row.get("schema", "")),
            source_artifact=source,
            source_ref={
                "task_instance": row.get("task_instance"),
                "canonical_hash": row.get("canonical_hash"),
                "program_canonical_hash": row.get("program_canonical_hash"),
            },
            utility=safe_float(row.get("utility")),
            metadata={
                "support": safe_int(row.get("support")),
                "covered_events": safe_int(row.get("covered_events")),
                "false_rejections": false_rejections,
            },
        )
    return sketches, examples


def grid_action_slots(action_id: int) -> list[dict[str, Any]]:
    name, dx, dy = GRID_ACTIONS.get(action_id, (f"action_{action_id}", 0, 0))
    return [
        normalized_slot("action_id", action_id, "integer"),
        normalized_slot("direction", name, "enum"),
        normalized_slot("dx", dx, "integer"),
        normalized_slot("dy", dy, "integer"),
    ]


def grid_move_atoms(row: dict[str, Any], action_id: int) -> list[str]:
    start = row.get("start", [0, 0])
    goal = row.get("goal", [0, 0])
    try:
        sx, sy = int(start[0]), int(start[1])
        gx, gy = int(goal[0]), int(goal[1])
    except (TypeError, ValueError, IndexError):
        return ["grid_path_compass_move"]
    _, dx, dy = GRID_ACTIONS.get(action_id, ("unknown", 0, 0))
    before = abs(gx - sx) + abs(gy - sy)
    after = abs(gx - (sx + dx)) + abs(gy - (sy + dy))
    atoms = ["grid_path_compass_move"]
    if after < before:
        atoms.append("reduces_manhattan")
    elif after > before:
        atoms.append("increases_manhattan")
    else:
        atoms.append("preserves_manhattan")
    if sx + dx < 0 or sx + dx > 6 or sy + dy < 0 or sy + dy > 6:
        atoms.append("out_of_bounds")
    return atoms


def grid_path_sketch_examples(
    source_path: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    sketches: dict[str, dict[str, Any]] = {}
    examples: list[dict[str, Any]] = []
    source = str(source_path)
    for row_index, row in enumerate(parse_json_rows(source_path)):
        if row.get("schema") != "crlp.trace.v1" or row.get("task") != "grid_path":
            continue
        action_ids = [safe_int(value) for value in row.get("action_ids", []) or []]
        scores = row.get("scores", []) or []
        probabilities = row.get("probabilities", []) or []
        target_action = safe_int(row.get("target_action_id"))
        selected_action = safe_int(row.get("selected_action_id"))
        action_to_index = {action_id: index for index, action_id in enumerate(action_ids)}

        def score_for(action_id: int) -> float | None:
            index = action_to_index.get(action_id)
            if index is None:
                return None
            if index < len(probabilities):
                return safe_float(probabilities[index])
            if index < len(scores):
                return safe_float(scores[index])
            return None

        for action_id, outcome in [(target_action, "accepted")]:
            sketch = make_grammar_sketch(
                "grid_path",
                "grid_path.compass_move",
                grid_action_slots(action_id),
                grid_move_atoms(row, action_id),
                compiler_target="grid_path_action_gate",
                source_artifact=source,
            )
            add_sketch_example(
                sketches,
                examples,
                sketch,
                example_id=f"grid_path:{row.get('seed', 'unknown')}:{row_index}:target:{action_id}",
                outcome=outcome,
                verifier_status="bfs_shortest_path_successor",
                source_row_schema=str(row.get("schema", "")),
                source_artifact=source,
                source_ref={
                    "seed": row.get("seed"),
                    "row_index": row_index,
                    "state_hash": row.get("state_hash"),
                },
                utility=safe_float(row.get("baseline_steps")) - safe_float(row.get("proposal_steps")),
                neural_score=score_for(action_id),
                metadata={
                    "verifier_result": row.get("verifier_result"),
                    "candidate_count": safe_int(row.get("candidate_count")),
                },
            )

        if selected_action != target_action:
            sketch = make_grammar_sketch(
                "grid_path",
                "grid_path.compass_move",
                grid_action_slots(selected_action),
                grid_move_atoms(row, selected_action),
                compiler_target="grid_path_action_gate",
                source_artifact=source,
            )
            add_sketch_example(
                sketches,
                examples,
                sketch,
                example_id=f"grid_path:{row.get('seed', 'unknown')}:{row_index}:selected:{selected_action}",
                outcome="rejected",
                verifier_status="bfs_corrected_selected_move",
                source_row_schema=str(row.get("schema", "")),
                source_artifact=source,
                source_ref={
                    "seed": row.get("seed"),
                    "row_index": row_index,
                    "state_hash": row.get("state_hash"),
                },
                utility=safe_float(row.get("baseline_steps")) - safe_float(row.get("proposal_steps")),
                neural_score=score_for(selected_action),
                metadata={
                    "target_action_id": target_action,
                    "verifier_result": row.get("verifier_result"),
                },
            )
    return sketches, examples


def signal_mining_sketch_examples(
    source_path: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    sketches: dict[str, dict[str, Any]] = {}
    examples: list[dict[str, Any]] = []
    source = str(source_path)
    for row in parse_json_rows(source_path):
        if row.get("schema") != "signal.mining_counterfactual.v1":
            continue
        candidate = str(row.get("candidate", f"candidate_{row.get('candidate_id', '')}"))
        slots = [
            normalized_slot("candidate_id", safe_int(row.get("candidate_id")), "integer"),
            normalized_slot("candidate", candidate, "enum"),
        ]
        for name in [
            "base_option",
            "target_rule",
            "route_rule",
            "speed_rule",
            "interaction_rule",
            "opportunistic",
            "station_legal",
        ]:
            if name in row:
                slot_type = "boolean" if isinstance(row.get(name), bool) else "integer"
                slots.append(normalized_slot(name, row.get(name), slot_type))
        atoms = [
            "deterministic_signal_replay",
            f"scenario:{row.get('scenario', '')}",
        ]
        if safe_int(row.get("death_events")) > 0:
            atoms.append("death_event")
        if safe_float(row.get("damage_amount")) > 0.0:
            atoms.append("damage_event")
        if safe_int(row.get("smelts")) > 0:
            atoms.append("smelted")
        if safe_int(row.get("pickups")) > 0:
            atoms.append("picked_fragment")
        sketch = make_grammar_sketch(
            "signal_mining",
            "signal_mining.tactical_option",
            slots,
            atoms,
            compiler_target="signal_mining_tactical_option",
            source_artifact=source,
        )
        if safe_int(row.get("death_events")) > 0:
            outcome = "unsafe"
        elif bool(row.get("is_oracle")):
            outcome = "accepted"
        elif safe_float(row.get("utility")) <= 0.0:
            outcome = "no_support"
        else:
            outcome = "rejected"
        add_sketch_example(
            sketches,
            examples,
            sketch,
            example_id="signal_mining:{}:{}".format(
                row.get("group_id", "unknown"), row.get("candidate_id", "unknown")
            ),
            outcome=outcome,
            verifier_status="deterministic_replay_certified"
            if row.get("certified")
            else "not_certified",
            source_row_schema=str(row.get("schema", "")),
            source_artifact=source,
            source_ref={
                "group_id": row.get("group_id"),
                "candidate_id": row.get("candidate_id"),
                "world_seed": row.get("world_seed"),
                "split": row.get("split"),
            },
            utility=safe_float(row.get("utility")),
            metadata={
                "is_oracle": bool(row.get("is_oracle")),
                "is_baseline": bool(row.get("is_baseline")),
                "damage_events": safe_int(row.get("damage_events")),
                "death_events": safe_int(row.get("death_events")),
            },
        )
    return sketches, examples


def merge_sketch_corpus(
    sketches: dict[str, dict[str, Any]],
    examples: list[dict[str, Any]],
    next_sketches: dict[str, dict[str, Any]],
    next_examples: list[dict[str, Any]],
) -> None:
    for sketch_id, sketch in next_sketches.items():
        if sketch_id in sketches:
            sketches[sketch_id] = merge_sketch(sketches[sketch_id], sketch)
        else:
            sketches[sketch_id] = sketch
    examples.extend(next_examples)


def build_sketch_corpus(
    args: argparse.Namespace,
    domains: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sketches: dict[str, dict[str, Any]] = {}
    examples: list[dict[str, Any]] = []
    if "program_synthesis" in domains:
        merge_sketch_corpus(
            sketches,
            examples,
            *program_synthesis_sketch_examples(args.program_synthesis_trace),
        )
    if "grid_path" in domains:
        merge_sketch_corpus(
            sketches,
            examples,
            *grid_path_sketch_examples(args.grid_path_trace),
        )
    if "signal_mining" in domains:
        merge_sketch_corpus(
            sketches,
            examples,
            *signal_mining_sketch_examples(args.signal_mining_trace),
        )
    return (
        sorted(sketches.values(), key=lambda row: row["sketch_id"]),
        sorted(examples, key=lambda row: (row["domain"], row["example_id"], row["sketch_id"])),
    )


def sketch_stats(examples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "accepted": 0,
            "rejected": 0,
            "unsafe": 0,
            "no_support": 0,
            "utility_sum": 0.0,
            "neural_scores": [],
        }
    )
    for row in examples:
        sketch_id = str(row.get("sketch_id", ""))
        outcome = str(row.get("outcome", ""))
        if outcome in stats[sketch_id]:
            stats[sketch_id][outcome] += 1
        stats[sketch_id]["utility_sum"] += safe_float(row.get("utility"))
        if "neural_score" in row:
            stats[sketch_id]["neural_scores"].append(safe_float(row.get("neural_score")))
    return stats


def build_frequency_proposals(
    sketches: list[dict[str, Any]],
    stats: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sketch in sketches:
        sketch_id = str(sketch["sketch_id"])
        stat = stats.get(sketch_id, {})
        accepted = safe_int(stat.get("accepted"))
        rejected = safe_int(stat.get("rejected"))
        unsafe = safe_int(stat.get("unsafe"))
        no_support = safe_int(stat.get("no_support"))
        score = accepted - 0.75 * rejected - 2.0 * unsafe - 0.25 * no_support
        rows.append(
            {
                "schema": GRAMMAR_SKETCH_PROPOSAL_SCHEMA,
                "report_version": 1,
                "proposal_id": f"gsp:{stable_id('frequency', sketch_id)}",
                "proposer": "frequency_sketch_proposer",
                "sketch_id": sketch_id,
                "domain": sketch["domain"],
                "family": sketch["family"],
                "score": round(score, 8),
                "rank_features": {
                    "accepted": accepted,
                    "rejected": rejected,
                    "unsafe": unsafe,
                    "no_support": no_support,
                },
            }
        )
    rows.sort(key=lambda row: (-safe_float(row["score"]), row["domain"], row["sketch_id"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def build_nearest_neighbor_proposals(
    sketches: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {row["sketch_id"]: row for row in sketches}
    accepted_ids = {
        str(row.get("sketch_id", ""))
        for row in examples
        if row.get("outcome") == "accepted"
    }
    rejected_examples = [
        row for row in examples if row.get("outcome") in {"rejected", "unsafe", "no_support"}
    ]
    token_cache = {sketch_id: sketch_tokens(sketch) for sketch_id, sketch in by_id.items()}
    rows: list[dict[str, Any]] = []
    for example in rejected_examples:
        source_id = str(example.get("sketch_id", ""))
        source = by_id.get(source_id)
        if source is None:
            continue
        source_tokens = token_cache[source_id]
        best_id = ""
        best_score = -1.0
        for candidate_id in sorted(accepted_ids):
            candidate = by_id.get(candidate_id)
            if candidate is None or candidate.get("domain") != source.get("domain"):
                continue
            score = jaccard(source_tokens, token_cache[candidate_id])
            if score > best_score:
                best_id = candidate_id
                best_score = score
        if not best_id:
            continue
        target = by_id[best_id]
        rows.append(
            {
                "schema": GRAMMAR_SKETCH_PROPOSAL_SCHEMA,
                "report_version": 1,
                "proposal_id": "gsp:{}".format(
                    stable_id("nearest", example.get("example_id"), best_id)
                ),
                "proposer": "nearest_neighbor_sketch_proposer",
                "sketch_id": best_id,
                "domain": target["domain"],
                "family": target["family"],
                "score": round(best_score, 8),
                "rank_features": {
                    "failed_example_id": example.get("example_id"),
                    "failed_sketch_id": source_id,
                    "failed_outcome": example.get("outcome"),
                    "jaccard": round(best_score, 8),
                },
            }
        )
    rows.sort(key=lambda row: (-safe_float(row["score"]), row["domain"], row["proposal_id"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def build_neural_linear_proposals(
    sketches: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    *,
    epochs: int = 64,
    learning_rate: float = 0.08,
) -> list[dict[str, Any]]:
    by_id = {row["sketch_id"]: row for row in sketches}
    labels: list[tuple[str, float]] = []
    for row in examples:
        outcome = row.get("outcome")
        if outcome == "accepted":
            labels.append((str(row["sketch_id"]), 1.0))
        elif outcome in {"rejected", "unsafe", "no_support"}:
            labels.append((str(row["sketch_id"]), 0.0))
    vocab = sorted({token for sketch in sketches for token in sketch_tokens(sketch)})
    token_index = {token: index for index, token in enumerate(vocab)}
    weights = [0.0 for _ in vocab]
    bias = 0.0

    def active_indices(sketch_id: str) -> list[int]:
        return [token_index[token] for token in sketch_tokens(by_id[sketch_id]) if token in token_index]

    active_cache = {sketch_id: active_indices(sketch_id) for sketch_id in by_id}
    for _ in range(epochs):
        for sketch_id, label in labels:
            z = bias + sum(weights[index] for index in active_cache.get(sketch_id, []))
            pred = 1.0 / (1.0 + pow(2.718281828459045, -max(min(z, 30.0), -30.0)))
            error = pred - label
            bias -= learning_rate * error
            scale = learning_rate * error / max(1, len(active_cache.get(sketch_id, [])))
            for index in active_cache.get(sketch_id, []):
                weights[index] -= scale

    rows: list[dict[str, Any]] = []
    for sketch in sketches:
        sketch_id = str(sketch["sketch_id"])
        z = bias + sum(weights[index] for index in active_cache.get(sketch_id, []))
        pred = 1.0 / (1.0 + pow(2.718281828459045, -max(min(z, 30.0), -30.0)))
        rows.append(
            {
                "schema": GRAMMAR_SKETCH_PROPOSAL_SCHEMA,
                "report_version": 1,
                "proposal_id": f"gsp:{stable_id('neural-linear', sketch_id)}",
                "proposer": "neural_linear_sketch_proposer",
                "sketch_id": sketch_id,
                "domain": sketch["domain"],
                "family": sketch["family"],
                "score": round(pred, 8),
                "rank_features": {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "vocab_size": len(vocab),
                    "training_examples": len(labels),
                },
            }
        )
    rows.sort(key=lambda row: (-safe_float(row["score"]), row["domain"], row["sketch_id"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def build_sketch_proposals(
    sketches: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    stats = sketch_stats(examples)
    return [
        *build_frequency_proposals(sketches, stats),
        *build_nearest_neighbor_proposals(sketches, examples),
        *build_neural_linear_proposals(sketches, examples),
    ]


def build_sketch_verifications(
    sketches: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stats = sketch_stats(examples)
    verifications: list[dict[str, Any]] = []
    promoted: list[dict[str, Any]] = []
    for sketch in sketches:
        sketch_id = str(sketch["sketch_id"])
        stat = stats.get(sketch_id, {})
        accepted = safe_int(stat.get("accepted"))
        rejected = safe_int(stat.get("rejected"))
        unsafe = safe_int(stat.get("unsafe"))
        no_support = safe_int(stat.get("no_support"))
        if accepted > 0 and rejected == 0 and unsafe == 0 and no_support == 0:
            status = "certified_existing_verifier_surface"
            promoted_flag = True
            blocking_reason = ""
        elif unsafe > 0:
            status = "unsafe_counterexamples"
            promoted_flag = False
            blocking_reason = "existing verifier produced unsafe examples"
        elif rejected > 0:
            status = "mixed_accept_reject_examples"
            promoted_flag = False
            blocking_reason = "existing verifier produced rejected examples"
        elif no_support > 0:
            status = "no_support_examples"
            promoted_flag = False
            blocking_reason = "existing verifier found no useful support"
        else:
            status = "no_verifier_examples"
            promoted_flag = False
            blocking_reason = "no existing verifier examples for sketch"
        row = {
            "schema": GRAMMAR_SKETCH_VERIFICATION_SCHEMA,
            "report_version": 1,
            "sketch_id": sketch_id,
            "domain": sketch["domain"],
            "family": sketch["family"],
            "verifier_status": status,
            "certified": promoted_flag,
            "promoted": promoted_flag,
            "blocking_reason": blocking_reason,
            "example_counts": {
                "accepted": accepted,
                "rejected": rejected,
                "unsafe": unsafe,
                "no_support": no_support,
            },
            "promotion_rule": "accepted examples from existing verifier and zero rejected/unsafe/no-support examples",
        }
        verifications.append(row)
        if promoted_flag:
            promoted.append(
                {
                    "schema": PROMOTED_GRAMMAR_SKETCH_SCHEMA,
                    "report_version": 1,
                    "sketch_id": sketch_id,
                    "domain": sketch["domain"],
                    "family": sketch["family"],
                    "promotion_kind": "existing_verifier_bootstrap",
                    "source_sketch": sketch,
                    "verification": row,
                }
            )
    return verifications, promoted


def promoted_rows(
    candidates: list[dict[str, Any]],
    verifications: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_candidate = {row["candidate_id"]: row for row in verifications}
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        verification = by_candidate.get(candidate["candidate_id"], {})
        if not verification.get("promoted"):
            continue
        rows.append(
            {
                "schema": PROMOTED_SCHEMA,
                "report_version": 1,
                "candidate_id": candidate["candidate_id"],
                "domain": candidate["domain"],
                "family": candidate["family"],
                "delta_kind": candidate["delta_kind"],
                "source_candidate": candidate,
                "verification": verification,
            }
        )
    return rows


def markdown_summary(
    candidates: list[dict[str, Any]],
    compiled_deltas: list[dict[str, Any]],
    verifications: list[dict[str, Any]],
    promoted: list[dict[str, Any]],
    sketches: list[dict[str, Any]],
    sketch_examples: list[dict[str, Any]],
    sketch_proposals: list[dict[str, Any]],
    sketch_verifications: list[dict[str, Any]],
    promoted_sketches: list[dict[str, Any]],
) -> str:
    domain_counts = Counter(row["domain"] for row in candidates)
    compiler_counts = Counter(row["compiler"] for row in compiled_deltas)
    status_counts = Counter(row["verifier_status"] for row in verifications)
    sketch_domain_counts = Counter(row["domain"] for row in sketches)
    sketch_outcome_counts = Counter(row["outcome"] for row in sketch_examples)
    sketch_proposer_counts = Counter(row["proposer"] for row in sketch_proposals)
    sketch_status_counts = Counter(row["verifier_status"] for row in sketch_verifications)
    lines = [
        "# Neural Grammar Discovery",
        "",
        "This report applies the shared grammar-delta discovery engine to every requested domain.",
        "Rows are proposal pressure only until a domain compiler and verifier certify an executable delta.",
        "",
        "## Summary",
        "",
        f"- Delta candidates: {len(candidates)}",
        f"- Compiled deltas: {len(compiled_deltas)}",
        f"- Promoted deltas: {len(promoted)}",
        f"- Grammar sketches: {len(sketches)}",
        f"- Grammar sketch examples: {len(sketch_examples)}",
        f"- Grammar sketch proposals: {len(sketch_proposals)}",
        f"- Promoted grammar sketches: {len(promoted_sketches)}",
        "",
        "| Domain | Candidates |",
        "| --- | ---: |",
    ]
    for domain, count in sorted(domain_counts.items()):
        lines.append(f"| `{domain}` | {count} |")
    lines.extend(["", "| Compiler | Deltas |", "| --- | ---: |"])
    for compiler, count in compiler_counts.most_common():
        lines.append(f"| `{compiler}` | {count} |")
    lines.extend(["", "| Verifier status | Candidates |", "| --- | ---: |"])
    for status, count in status_counts.most_common():
        lines.append(f"| `{status}` | {count} |")

    lines.extend(["", "## Grammar Sketch Corpus", ""])
    if sketches:
        lines.extend(["| Domain | Sketches |", "| --- | ---: |"])
        for domain, count in sorted(sketch_domain_counts.items()):
            lines.append(f"| `{domain}` | {count} |")
        lines.extend(["", "| Example outcome | Rows |", "| --- | ---: |"])
        for outcome, count in sketch_outcome_counts.most_common():
            lines.append(f"| `{outcome}` | {count} |")
        lines.extend(["", "| Proposer | Proposals |", "| --- | ---: |"])
        for proposer, count in sketch_proposer_counts.most_common():
            lines.append(f"| `{proposer}` | {count} |")
        lines.extend(["", "| Sketch verifier status | Sketches |", "| --- | ---: |"])
        for status, count in sketch_status_counts.most_common():
            lines.append(f"| `{status}` | {count} |")
    else:
        lines.append("No grammar sketch examples were emitted from the requested domains.")

    lines.extend(
        [
            "",
            "## Candidates",
            "",
            "| Domain | Family | Compiler | Support | Status | Blocking reason |",
            "| --- | --- | --- | ---: | --- | --- |",
        ]
    )
    ver_by_id = {row["candidate_id"]: row for row in verifications}
    compiled_by_id = {row["candidate_id"]: row for row in compiled_deltas}
    for candidate in sorted(candidates, key=lambda row: (row["domain"], row["family"])):
        verification = ver_by_id.get(candidate["candidate_id"], {})
        compiled = compiled_by_id.get(candidate["candidate_id"], {})
        lines.append(
            "| `{}` | `{}` | `{}` | {} | `{}` | {} |".format(
                candidate["domain"],
                candidate["family"],
                compiled.get("compiler", ""),
                candidate["support_examples"],
                verification.get("verifier_status", ""),
                verification.get("blocking_reason", ""),
            )
        )
    if sketches:
        sketch_ver_by_id = {row["sketch_id"]: row for row in sketch_verifications}
        sketch_status_groups = Counter(
            (
                sketch["domain"],
                sketch["family"],
                str(sketch_ver_by_id.get(sketch["sketch_id"], {}).get("verifier_status", "")),
                str(sketch_ver_by_id.get(sketch["sketch_id"], {}).get("blocking_reason", "")),
            )
            for sketch in sketches
        )
        lines.extend(
            [
                "",
                "## Grammar Sketches",
                "",
                "| Domain | Family | Sketches | Status | Blocking reason |",
                "| --- | --- | ---: | --- | --- |",
            ]
        )
        for (domain, family, status, reason), count in sorted(sketch_status_groups.items()):
            lines.append(
                "| `{}` | `{}` | {} | `{}` | {} |".format(
                    domain,
                    family,
                    count,
                    status,
                    reason,
                )
            )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    domains = {
        domain.strip()
        for domain in str(args.domains).split(",")
        if domain.strip()
    }
    groups: list[CandidateGroup] = []
    if "arc" in domains or "arc_agi" in domains:
        groups.extend(arc_groups(args.arc_failure_proof))
    if "haiku" in domains:
        groups.extend(haiku_groups(args.haiku_word_trace, args.haiku_understanding_trace))

    candidates = [build_candidate(group) for group in groups]
    compiled_deltas = [
        compile_delta(candidate, group)
        for candidate, group in zip(candidates, groups, strict=True)
    ]
    verifications = [
        build_verification(candidate, group, compiled)
        for candidate, group, compiled in zip(
            candidates, groups, compiled_deltas, strict=True
        )
    ]
    promoted = promoted_rows(candidates, verifications)
    sketches, sketch_examples = build_sketch_corpus(args, domains)
    sketch_proposals = build_sketch_proposals(sketches, sketch_examples)
    sketch_verifications, promoted_sketches = build_sketch_verifications(sketches, sketch_examples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "grammar_delta_candidates.jsonl", candidates)
    write_jsonl(args.output_dir / "compiled_grammar_deltas.jsonl", compiled_deltas)
    write_jsonl(args.output_dir / "grammar_delta_verification.jsonl", verifications)
    write_jsonl(args.output_dir / "promoted_grammar_deltas.jsonl", promoted)
    write_jsonl(args.output_dir / "grammar_sketches.jsonl", sketches)
    write_jsonl(args.output_dir / "grammar_sketch_examples.jsonl", sketch_examples)
    write_jsonl(args.output_dir / "grammar_sketch_proposals.jsonl", sketch_proposals)
    write_jsonl(args.output_dir / "grammar_sketch_verification.jsonl", sketch_verifications)
    write_jsonl(args.output_dir / "promoted_grammar_sketches.jsonl", promoted_sketches)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "report_version": 1,
        "domains": sorted(domains),
        "candidate_count": len(candidates),
        "compiled_delta_count": len(compiled_deltas),
        "verification_count": len(verifications),
        "promoted_count": len(promoted),
        "grammar_sketch_count": len(sketches),
        "grammar_sketch_example_count": len(sketch_examples),
        "grammar_sketch_proposal_count": len(sketch_proposals),
        "grammar_sketch_verification_count": len(sketch_verifications),
        "promoted_grammar_sketch_count": len(promoted_sketches),
        "by_domain": dict(sorted(Counter(row["domain"] for row in candidates).items())),
        "compiled_delta_by_compiler": dict(
            sorted(Counter(row["compiler"] for row in compiled_deltas).items())
        ),
        "by_verifier_status": dict(
            sorted(Counter(row["verifier_status"] for row in verifications).items())
        ),
        "grammar_sketch_by_domain": dict(
            sorted(Counter(row["domain"] for row in sketches).items())
        ),
        "grammar_sketch_examples_by_outcome": dict(
            sorted(Counter(row["outcome"] for row in sketch_examples).items())
        ),
        "grammar_sketch_proposals_by_proposer": dict(
            sorted(Counter(row["proposer"] for row in sketch_proposals).items())
        ),
        "grammar_sketch_by_verifier_status": dict(
            sorted(Counter(row["verifier_status"] for row in sketch_verifications).items())
        ),
        "source_artifacts": {
            "arc_failure_proof": str(args.arc_failure_proof),
            "haiku_word_trace": str(args.haiku_word_trace),
            "haiku_understanding_trace": str(args.haiku_understanding_trace),
            "program_synthesis_trace": str(args.program_synthesis_trace),
            "grid_path_trace": str(args.grid_path_trace),
            "signal_mining_trace": str(args.signal_mining_trace),
        },
    }
    write_jsonl(args.output_dir / "summary.jsonl", [summary])
    (args.output_dir / "summary.md").write_text(
        markdown_summary(
            candidates,
            compiled_deltas,
            verifications,
            promoted,
            sketches,
            sketch_examples,
            sketch_proposals,
            sketch_verifications,
            promoted_sketches,
        ),
        encoding="utf-8",
    )
    return {
        "candidates": candidates,
        "compiled_deltas": compiled_deltas,
        "verifications": verifications,
        "promoted": promoted,
        "sketches": sketches,
        "sketch_examples": sketch_examples,
        "sketch_proposals": sketch_proposals,
        "sketch_verifications": sketch_verifications,
        "promoted_sketches": promoted_sketches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domains",
        default="arc,haiku,program_synthesis,grid_path,signal_mining",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/float/neural_grammar_discovery"),
    )
    parser.add_argument(
        "--arc-failure-proof",
        type=Path,
        default=Path(
            "build/float/hypothesis_library_neural_discovery/failure_proof/task_proof_rows.jsonl"
        ),
    )
    parser.add_argument(
        "--haiku-word-trace",
        type=Path,
        default=Path("build/float/haiku_word_lm_artifacts/smoke_trace.jsonl"),
    )
    parser.add_argument(
        "--haiku-understanding-trace",
        type=Path,
        default=Path("build/float/haiku_understanding_artifacts/smoke_trace.jsonl"),
    )
    parser.add_argument(
        "--program-synthesis-trace",
        type=Path,
        default=Path("build/float/neural_grammar_discovery/program_synthesis_source.jsonl"),
    )
    parser.add_argument(
        "--grid-path-trace",
        type=Path,
        default=Path("build/float/grid_path.jsonl"),
    )
    parser.add_argument(
        "--signal-mining-trace",
        type=Path,
        default=Path("build/float/signal_mining_smoke/trace.jsonl"),
    )
    args = parser.parse_args()

    outputs = run(args)
    print(
        "wrote {} delta candidates, {} compiled deltas, {} promoted deltas, {} grammar sketches, {} sketch examples, {} sketch proposals, {} promoted sketches to {}".format(
            len(outputs["candidates"]),
            len(outputs["compiled_deltas"]),
            len(outputs["promoted"]),
            len(outputs["sketches"]),
            len(outputs["sketch_examples"]),
            len(outputs["sketch_proposals"]),
            len(outputs["promoted_sketches"]),
            args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
