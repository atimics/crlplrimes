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
import subprocess
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
ARC_SKETCH_PROBE_SCHEMA = "crlp.neural_grammar_discovery.arc_sketch_program_probe.v1"
ARC_SKETCH_PROBE_SUMMARY_SCHEMA = (
    "crlp.neural_grammar_discovery.arc_sketch_program_probe_summary.v1"
)
ARC_SKETCH_PROBE_COMPARISON_SCHEMA = (
    "crlp.neural_grammar_discovery.arc_sketch_program_probe_comparison.v1"
)
ARC_VISUAL_PIPELINE_ATTEMPT_SCHEMA = (
    "crlp.neural_grammar_discovery.arc_visual_pipeline_attempt.v1"
)
ARC_VISUAL_PIPELINE_PROPOSAL_SCHEMA = (
    "crlp.neural_grammar_discovery.arc_visual_pipeline_proposal.v1"
)
ARC_VISUAL_PIPELINE_PROPOSAL_SUMMARY_SCHEMA = (
    "crlp.neural_grammar_discovery.arc_visual_pipeline_proposal_summary.v1"
)
ARC_VISUAL_PIPELINE_NEURAL_SUMMARY_SCHEMA = (
    "crlp.neural_grammar_discovery.arc_visual_pipeline_neural_summary.v1"
)

ARC_SKETCH_TEMPLATE_NAME = "arc_sketch_program"
ARC_SKETCH_PROBE_CANDIDATE_FIELDS = [
    "family",
    "task_id",
    "split",
    "support_tasks",
    "proposal_count",
    "utility_mean",
    "op_hints",
]
ARC_SKETCH_PRIOR_HEADER = (
    "schema_version\ttemplate_id\ttemplate_name\tlibrary_id\tslot_name\t"
    "slot_type\tslot_value\tslot_integer\tproposal_count\tcertified_count\t"
    "support_tasks\tcovered_events\tutility_mean\n"
)


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

ARC_FAMILY_STRUCTURE_OPS = {
    "arc.patch_lattice_template_family": (
        "tile_repeat",
        "palette_scale",
        "macrocell_substitute",
        "lattice_sample",
        "point_line_project",
    ),
    "arc.separator_panel_extraction_family": (
        "separator_panel_map",
        "component_color_summary",
        "crop",
        "masked_or_normalized_crop",
    ),
    "arc.tiling_canvas_growth_family": (
        "macrocell_substitute",
        "palette_scale",
        "tile_repeat",
        "lattice_sample",
    ),
    "arc.crop_object_selection_family": (
        "object_color_tile",
        "object_mask_extract",
        "crop",
        "masked_or_normalized_crop",
        "lattice_sample",
    ),
    "arc.projection_compression_family": (
        "marker_count_motif_tile",
        "component_count_line",
        "axis_projection_compress",
        "separator_panel_map",
        "lattice_sample",
    ),
    "arc.compositional_transform_family": (
        "palette_scale",
        "object_color_tile",
        "object_mask_extract",
        "axis_projection_compress",
        "separator_panel_map",
    ),
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


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


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


def tsv_value(value: Any) -> str:
    if value is None:
        text = ""
    elif isinstance(value, bool):
        text = "1" if value else "0"
    else:
        text = str(value)
    return text.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def pipe_join(values: list[Any]) -> str:
    return "|".join(tsv_value(value).replace("|", "/") for value in values if value is not None)


def write_runtime_candidate_rows(
    path: Path,
    candidates: list[dict[str, Any]],
    groups: list[CandidateGroup],
) -> None:
    header = [
        "candidate_id",
        "domain",
        "family",
        "source_artifacts",
        "feature_summary_json",
        "task_id",
        "split",
        "example_id",
        "sample",
        "step",
        "candidate_count",
        "strict_adjacency",
        "rejected_adjacency",
        "rejected_dead_end",
        "rejected_syllable",
        "rejected_line_chars",
        "accepted_existing_rows",
        "missing_family_evidence",
        "promotion_replayed",
        "promotion_proposals_train_consistent_pre_cap",
        "promotion_proposals_generated",
        "mutation_train_consistent_rows",
        "poem_index",
        "corruption",
        "symbolic_selected",
        "adjacency_hash",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        fp.write("\t".join(header) + "\n")
        for candidate, group in zip(candidates, groups, strict=True):
            source_artifacts = pipe_join(candidate.get("source_artifacts", []))
            feature_summary_json = json.dumps(
                candidate.get("neural_cluster", {}).get("feature_summary", {}),
                sort_keys=True,
                separators=(",", ":"),
            )
            for row in group.rows:
                rejects = row.get("rejected_candidates", {})
                if not isinstance(rejects, dict):
                    rejects = {}
                fields = [
                    candidate.get("candidate_id", ""),
                    group.domain,
                    group.family,
                    source_artifacts,
                    feature_summary_json,
                    row.get("task_id", ""),
                    row.get("split", ""),
                    row.get("example_id", ""),
                    row.get("sample", ""),
                    row.get("step", ""),
                    row.get("candidate_count", ""),
                    bool(row.get("strict_adjacency")),
                    rejects.get("adjacency", ""),
                    rejects.get("dead_end", ""),
                    rejects.get("syllable", ""),
                    rejects.get("line_chars", ""),
                    row.get("accepted_existing_rows", ""),
                    bool(row.get("missing_family_evidence")),
                    bool(row.get("promotion_replayed")),
                    row.get("promotion_proposals_train_consistent_pre_cap", ""),
                    row.get("promotion_proposals_generated", ""),
                    row.get("mutation_train_consistent_rows", ""),
                    row.get("poem_index", ""),
                    row.get("corruption", ""),
                    row.get("symbolic_selected", ""),
                    row.get("adjacency_hash", ""),
                ]
                fp.write("\t".join(tsv_value(field) for field in fields))
                fp.write("\n")


def runtime_slot_token(slot: dict[str, Any]) -> str:
    name = str(slot.get("name", ""))
    value = slot.get("integer", slot.get("value", ""))
    return "{}={}".format(name.replace("|", "/"), tsv_value(value).replace("|", "/"))


def write_runtime_sketch_tables(
    sketches_path: Path,
    examples_path: Path,
    sketches: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> None:
    sketches_path.parent.mkdir(parents=True, exist_ok=True)
    with sketches_path.open("w", encoding="utf-8") as fp:
        fp.write("sketch_id\tdomain\tfamily\tcompiler_target\tatoms\tslots\n")
        for sketch in sketches:
            atoms = pipe_join(list(sketch.get("structural_atoms", []) or []))
            slots = pipe_join(
                [
                    runtime_slot_token(slot)
                    for slot in sketch.get("slots", []) or []
                    if isinstance(slot, dict)
                ]
            )
            fields = [
                sketch.get("sketch_id", ""),
                sketch.get("domain", ""),
                sketch.get("family", ""),
                sketch.get("compiler_target", ""),
                atoms,
                slots,
            ]
            fp.write("\t".join(tsv_value(field) for field in fields))
            fp.write("\n")
    with examples_path.open("w", encoding="utf-8") as fp:
        fp.write("sketch_id\toutcome\n")
        for row in examples:
            fp.write(
                "{}\t{}\n".format(
                    tsv_value(row.get("sketch_id", "")),
                    tsv_value(row.get("outcome", "")),
                )
            )


def run_c_runtime(
    args: argparse.Namespace,
    candidates: list[dict[str, Any]],
    groups: list[CandidateGroup],
    sketches: list[dict[str, Any]],
    sketch_examples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    runtime = Path(args.runtime)
    if not runtime.is_file():
        raise FileNotFoundError(
            "neural grammar discovery C runtime not found: {}".format(runtime)
        )
    runtime_dir = args.output_dir / "runtime"
    candidate_rows = runtime_dir / "candidate_rows.tsv"
    sketch_rows = runtime_dir / "sketches.tsv"
    sketch_example_rows = runtime_dir / "sketch_examples.tsv"
    compiled_output = args.output_dir / "compiled_grammar_deltas.jsonl"
    verification_output = args.output_dir / "grammar_delta_verification.jsonl"
    neural_proposals_output = runtime_dir / "neural_sketch_proposals.jsonl"
    sketch_memory = Path(args.sketch_memory) if args.sketch_memory else (
        args.output_dir / "grammar_sketch_memory.tsv"
    )
    memory_input = sketch_memory if args.resume_sketch_memory and sketch_memory.is_file() else None
    write_runtime_candidate_rows(candidate_rows, candidates, groups)
    write_runtime_sketch_tables(sketch_rows, sketch_example_rows, sketches, sketch_examples)
    command = [
        str(runtime),
        "--candidate-rows",
        str(candidate_rows),
        "--compiled-output",
        str(compiled_output),
        "--verification-output",
        str(verification_output),
        "--sketches",
        str(sketch_rows),
        "--sketch-examples",
        str(sketch_example_rows),
        "--neural-proposals-output",
        str(neural_proposals_output),
        "--memory-out",
        str(sketch_memory),
    ]
    if memory_input is not None:
        command.extend(["--memory-in", str(memory_input)])
    subprocess.run(command, check=True)
    neural_proposals = parse_json_rows(neural_proposals_output)
    rank_features = (
        neural_proposals[0].get("rank_features", {})
        if neural_proposals and isinstance(neural_proposals[0].get("rank_features"), dict)
        else {}
    )
    memory_summary = {
        "schema": "crlp.grammar_sketch.memory.summary.v1",
        "report_version": 1,
        "checkpoint_path": str(sketch_memory),
        "resume_requested": bool(args.resume_sketch_memory),
        "checkpoint_loaded": bool(rank_features.get("checkpoint_loaded", False)),
        "checkpoint_epochs_seen_before": safe_int(
            rank_features.get("checkpoint_epochs_seen_before")
        ),
        "checkpoint_epochs_seen_after": safe_int(
            rank_features.get("checkpoint_epochs_seen_after")
        ),
        "checkpoint_training_examples_seen_before": safe_int(
            rank_features.get("checkpoint_training_examples_seen_before")
        ),
        "checkpoint_training_examples_seen_after": safe_int(
            rank_features.get("checkpoint_training_examples_seen_after")
        ),
        "checkpoint_restored_weights": safe_int(rank_features.get("checkpoint_restored_weights")),
        "checkpoint_ignored_weights": safe_int(rank_features.get("checkpoint_ignored_weights")),
        "vocab_size": safe_int(rank_features.get("vocab_size")),
        "training_examples": safe_int(rank_features.get("training_examples")),
    }
    return (
        parse_json_rows(compiled_output),
        parse_json_rows(verification_output),
        neural_proposals,
        memory_summary,
    )


def write_arc_probe_candidate_record_rows(
    path: Path,
    records: list[dict[str, Any]],
) -> int:
    row_count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        fp.write("\t".join(ARC_SKETCH_PROBE_CANDIDATE_FIELDS))
        fp.write("\n")
        for record in records:
            family = str(record.get("family", ""))
            task_id = str(record.get("task_id", ""))
            split = str(record.get("split", ""))
            if not family or not task_id or not split:
                continue
            fp.write(
                "{}\t{}\t{}\t{}\t{}\t{:.8f}\t{}\n".format(
                    family,
                    task_id,
                    split,
                    max(1, safe_int(record.get("support_tasks"))),
                    max(1, min(64, safe_int(record.get("proposal_count")))),
                    safe_float(record.get("utility_mean")),
                    str(record.get("op_hints", "")).replace("\t", " ").replace("\n", " "),
                )
            )
            row_count += 1
    return row_count


def write_empty_arc_probe_artifacts(
    source_path: Path,
    prior_path: Path,
    attempts_path: Path | None = None,
) -> None:
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("", encoding="utf-8")
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    prior_path.write_text(ARC_SKETCH_PRIOR_HEADER, encoding="utf-8")
    if attempts_path is not None:
        attempts_path.parent.mkdir(parents=True, exist_ok=True)
        attempts_path.write_text("", encoding="utf-8")


def arc_probe_prior_row_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as fp:
        return max(0, sum(1 for _line in fp) - 1)


def run_arc_sketch_candidate_export(
    records: list[dict[str, Any]],
    arc_agi: Path,
    public_arc_dir: Path,
    candidate_path: Path,
    source_path: Path,
    prior_path: Path,
    attempts_path: Path,
    sketches_per_task: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int, int]:
    candidate_count = write_arc_probe_candidate_record_rows(candidate_path, records)
    if (
        candidate_count == 0
        or not arc_agi.is_file()
        or str(public_arc_dir) in {"", "."}
        or not public_arc_dir.is_dir()
    ):
        write_empty_arc_probe_artifacts(source_path, prior_path, attempts_path)
        return [], [], [], 0, candidate_count
    command = [
        str(arc_agi),
        "--arc-sketch-probe-candidates",
        str(candidate_path),
        "--public-arc-dir",
        str(public_arc_dir),
        "--arc-sketch-probe-sources",
        str(source_path),
        "--arc-sketch-probe-prior",
        str(prior_path),
        "--arc-sketch-probe-attempts",
        str(attempts_path),
        "--arc-sketches-per-task",
        str(sketches_per_task),
    ]
    subprocess.run(command, check=True)
    source_rows = parse_json_rows(source_path)
    probe_rows = [
        row for row in source_rows if row.get("schema") == ARC_SKETCH_PROBE_SCHEMA
    ]
    summary_rows = [
        row
        for row in source_rows
        if row.get("schema") == ARC_SKETCH_PROBE_SUMMARY_SCHEMA
    ]
    attempt_rows = [
        row
        for row in parse_json_rows(attempts_path)
        if row.get("schema") == ARC_VISUAL_PIPELINE_ATTEMPT_SCHEMA
    ]
    return (
        probe_rows,
        summary_rows,
        attempt_rows,
        arc_probe_prior_row_count(prior_path),
        candidate_count,
    )


def run_arc_sketch_probe_export(
    groups: list[CandidateGroup],
    promoted: list[dict[str, Any]],
    arc_agi: Path,
    public_arc_dir: Path,
    output_dir: Path,
    sketches_per_task: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int, int]:
    source_path = output_dir / "arc_sketch_program_probe_sources.jsonl"
    prior_path = output_dir / "arc_sketch_program_probe_prior.tsv"
    attempts_path = output_dir / "arc_visual_pipeline_attempts.jsonl"
    candidate_path = output_dir / "runtime" / "arc_sketch_probe_candidates.tsv"
    promoted_families = {
        str(row.get("family", ""))
        for row in promoted
        if row.get("domain") == "arc_agi"
    }
    records: list[dict[str, Any]] = []
    seen_records: set[tuple[str, str, str]] = set()
    for group in groups:
        if group.domain != "arc_agi" or group.family not in promoted_families:
            continue
        support = len(group.rows)
        utility = mean(
            [safe_float(row.get("slot_probe_scored_rows")) for row in group.rows]
        )
        proposal_count = max(
            1,
            min(
                64,
                int(
                    round(
                        mean(
                            [
                                safe_float(row.get("slot_probe_seeded_proposals"))
                                for row in group.rows
                            ]
                        )
                    )
                )
                or support,
            ),
        )
        for source in group.rows:
            task_id = str(source.get("task_id", ""))
            split = str(source.get("split", ""))
            key = (group.family, task_id, split)
            if not task_id or not split or key in seen_records:
                continue
            seen_records.add(key)
            records.append(
                {
                    "family": group.family,
                    "task_id": task_id,
                    "split": split,
                    "support_tasks": support,
                    "proposal_count": proposal_count,
                    "utility_mean": utility,
                }
            )
    return run_arc_sketch_candidate_export(
        records,
        arc_agi,
        public_arc_dir,
        candidate_path,
        source_path,
        prior_path,
        attempts_path,
        sketches_per_task,
    )


def arc_task_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("task_id", "")), str(row.get("split", "")))


def arc_heldout_candidate_records(
    triage_path: Path,
    excluded_keys: set[tuple[str, str]],
    limit: int,
    per_family: int,
) -> list[dict[str, Any]]:
    if limit == 0 or per_family == 0:
        return []
    records: list[dict[str, Any]] = []
    counts_by_family: Counter[str] = Counter()
    seen: set[tuple[str, str, str]] = set()
    for row in parse_json_rows(triage_path):
        priority = str(row.get("priority") or row.get("triage_priority") or "")
        spec = ARC_PRIORITY_TO_DELTA.get(priority)
        if spec is None:
            continue
        if safe_bool(row.get("certified")):
            continue
        if safe_int(row.get("proposals_train_consistent_pre_cap")) > 0:
            continue
        task_id, split = arc_task_key(row)
        if not task_id or not split or (task_id, split) in excluded_keys:
            continue
        family = str(spec["family"])
        key = (family, task_id, split)
        if key in seen:
            continue
        if per_family >= 0 and counts_by_family[family] >= per_family:
            continue
        records.append(
            {
                "family": family,
                "task_id": task_id,
                "split": split,
                "support_tasks": max(1, safe_int(row.get("train_pairs"))),
                "proposal_count": max(1, min(64, safe_int(row.get("proposals_generated")))),
                "utility_mean": max(1.0, safe_float(row.get("rejected_evidence"))),
                "priority": priority,
                "structural_label": row.get("structural_label", ""),
                "diagnostic_bucket": row.get("diagnostic_bucket", ""),
            }
        )
        seen.add(key)
        counts_by_family[family] += 1
        if limit >= 0 and len(records) >= limit:
            break
    return records


def arc_public_task_path(public_arc_dir: Path, split: str, task_id: str) -> Path:
    if "eval" in split:
        for dirname in ("evaluation", "eval"):
            path = public_arc_dir / dirname / f"{task_id}.json"
            if path.is_file():
                return path
    for dirname in ("training", "train", "evaluation", "eval"):
        path = public_arc_dir / dirname / f"{task_id}.json"
        if path.is_file():
            return path
    return public_arc_dir / split / f"{task_id}.json"


def arc_grid_dimensions(grid: Any) -> tuple[int, int]:
    if not isinstance(grid, list) or not grid:
        return 0, 0
    height = len(grid)
    width = len(grid[0]) if isinstance(grid[0], list) else 0
    return width, height


def arc_grid_colors(grid: Any) -> set[int]:
    colors: set[int] = set()
    if not isinstance(grid, list):
        return colors
    for row in grid:
        if not isinstance(row, list):
            continue
        for value in row:
            try:
                colors.add(int(value))
            except (TypeError, ValueError):
                pass
    return colors


def arc_grid_has_separator(grid: Any) -> bool:
    if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
        return False
    width, height = arc_grid_dimensions(grid)
    if width <= 2 or height <= 2:
        return False
    colors = arc_grid_colors(grid)
    if len(colors) <= 1:
        return False
    for row in grid:
        if isinstance(row, list) and len(row) == width and len(set(row)) == 1:
            return True
    for x in range(width):
        column = [row[x] for row in grid if isinstance(row, list) and len(row) > x]
        if len(column) == height and len(set(column)) == 1:
            return True
    return False


def arc_task_shape_features(public_arc_dir: Path, split: str, task_id: str) -> dict[str, Any]:
    path = arc_public_task_path(public_arc_dir, split, task_id)
    if not path.is_file():
        return {"task_path": str(path), "found": False}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"task_path": str(path), "found": False}
    train = data.get("train", [])
    if not isinstance(train, list) or not train:
        return {"task_path": str(path), "found": False}
    features: Counter[str] = Counter()
    input_colors: set[int] = set()
    output_colors: set[int] = set()
    pair_count = 0
    first_dims: dict[str, int] = {}
    for pair in train:
        if not isinstance(pair, dict):
            continue
        input_grid = pair.get("input")
        output_grid = pair.get("output")
        input_width, input_height = arc_grid_dimensions(input_grid)
        output_width, output_height = arc_grid_dimensions(output_grid)
        if not input_width or not input_height or not output_width or not output_height:
            continue
        pair_count += 1
        if not first_dims:
            first_dims = {
                "input_width": input_width,
                "input_height": input_height,
                "output_width": output_width,
                "output_height": output_height,
            }
        input_area = input_width * input_height
        output_area = output_width * output_height
        features["same_shape"] += int(input_width == output_width and input_height == output_height)
        features["scale_up"] += int(output_area > input_area)
        features["scale_down"] += int(output_area < input_area)
        features["integer_scale_up"] += int(
            output_area > input_area
            and output_width % input_width == 0
            and output_height % input_height == 0
        )
        features["output_line"] += int(output_width == 1 or output_height == 1)
        features["output_small"] += int(output_area <= 9)
        features["input_has_separator"] += int(arc_grid_has_separator(input_grid))
        input_colors.update(arc_grid_colors(input_grid))
        output_colors.update(arc_grid_colors(output_grid))
    result: dict[str, Any] = {
        "task_path": str(path),
        "found": pair_count > 0,
        "train_pairs": pair_count,
        "input_color_count": len(input_colors),
        "output_color_count": len(output_colors),
    }
    result.update(first_dims)
    for key in (
        "same_shape",
        "scale_up",
        "scale_down",
        "integer_scale_up",
        "output_line",
        "output_small",
        "input_has_separator",
    ):
        result[key] = features[key]
    return result


def arc_structure_op_scores(
    record: dict[str, Any],
    features: dict[str, Any],
    accepted_by_family_op: Counter[tuple[str, str]],
    accepted_by_op: Counter[str],
) -> list[tuple[str, float]]:
    family = str(record.get("family", ""))
    ops = ARC_FAMILY_STRUCTURE_OPS.get(family, ())
    scores: dict[str, float] = {op: 0.05 for op in ops}
    for op in ops:
        scores[op] += 3.0 * accepted_by_family_op[(family, op)]
        scores[op] += 0.35 * accepted_by_op[op]
    if safe_int(features.get("integer_scale_up")) > 0:
        for op, bonus in (
            ("macrocell_substitute", 5.0),
            ("palette_scale", 3.0),
            ("tile_repeat", 1.5),
        ):
            if op in scores:
                scores[op] += bonus
    if safe_int(features.get("scale_up")) > 0:
        for op, bonus in (
            ("object_color_tile", 3.0),
            ("marker_count_motif_tile", 3.0),
            ("macrocell_substitute", 2.0),
            ("palette_scale", 1.5),
        ):
            if op in scores:
                scores[op] += bonus
    if safe_int(features.get("scale_down")) > 0:
        for op, bonus in (
            ("crop", 2.5),
            ("masked_or_normalized_crop", 2.5),
            ("object_mask_extract", 2.5),
            ("lattice_sample", 2.0),
            ("component_color_summary", 2.0),
            ("axis_projection_compress", 2.0),
            ("separator_panel_map", 1.5),
        ):
            if op in scores:
                scores[op] += bonus
    if safe_int(features.get("same_shape")) > 0:
        for op, bonus in (
            ("tile_repeat", 2.5),
            ("point_line_project", 2.5),
            ("palette_scale", 1.0),
            ("separator_panel_map", 0.75),
        ):
            if op in scores:
                scores[op] += bonus
    if safe_int(features.get("output_line")) > 0:
        for op, bonus in (
            ("component_count_line", 5.0),
            ("axis_projection_compress", 3.5),
        ):
            if op in scores:
                scores[op] += bonus
    if safe_int(features.get("output_small")) > 0:
        for op, bonus in (
            ("component_color_summary", 2.0),
            ("lattice_sample", 1.5),
            ("axis_projection_compress", 1.5),
            ("object_color_tile", 4.0),
            ("marker_count_motif_tile", 4.0),
        ):
            if op in scores:
                scores[op] += bonus
    if safe_int(features.get("input_has_separator")) > 0:
        for op, bonus in (
            ("separator_panel_map", 5.0),
            ("component_color_summary", 1.0),
            ("marker_count_motif_tile", 1.0),
        ):
            if op in scores:
                scores[op] += bonus
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def add_arc_structure_op_hints(
    records: list[dict[str, Any]],
    source_attempts: list[dict[str, Any]],
    public_arc_dir: Path,
    hints_per_task: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted_by_family_op: Counter[tuple[str, str]] = Counter()
    accepted_by_op: Counter[str] = Counter()
    for row in source_attempts:
        if not safe_bool(row.get("accepted")):
            continue
        family = str(row.get("family", ""))
        op_name = str(row.get("op_name", ""))
        if not family or not op_name:
            continue
        accepted_by_family_op[(family, op_name)] += 1
        accepted_by_op[op_name] += 1

    proposals: list[dict[str, Any]] = []
    hinted_records: list[dict[str, Any]] = []
    for record in records:
        family = str(record.get("family", ""))
        task_id = str(record.get("task_id", ""))
        split = str(record.get("split", ""))
        features = arc_task_shape_features(public_arc_dir, split, task_id)
        ranked = arc_structure_op_scores(
            record,
            features,
            accepted_by_family_op,
            accepted_by_op,
        )
        if hints_per_task > 0:
            selected = [op for op, _score in ranked[:hints_per_task]]
        else:
            selected = []
        if not selected and family in ARC_FAMILY_STRUCTURE_OPS:
            selected = list(ARC_FAMILY_STRUCTURE_OPS[family])
        updated = dict(record)
        updated["op_hints"] = ",".join(selected)
        hinted_records.append(updated)
        proposals.append(
            {
                "schema": "crlp.neural_grammar_discovery.arc_structure_proposal.v1",
                "report_version": 1,
                "family": family,
                "task_id": task_id,
                "split": split,
                "proposer": "shape_conditioned_structure_op_proposer",
                "proposed_ops": selected,
                "ranked_ops": [
                    {"op_name": op, "score": round(score, 8)}
                    for op, score in ranked
                ],
                "task_features": features,
                "source_family_accepts": {
                    op: accepted_by_family_op[(family, op)]
                    for op in ARC_FAMILY_STRUCTURE_OPS.get(family, ())
                },
            }
        )
    return hinted_records, proposals


def run_arc_sketch_heldout_export(
    triage_path: Path,
    excluded_keys: set[tuple[str, str]],
    source_attempts: list[dict[str, Any]],
    arc_agi: Path,
    public_arc_dir: Path,
    output_dir: Path,
    sketches_per_task: int,
    limit: int,
    per_family: int,
    hints_per_task: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    int,
    int,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    candidate_jsonl_path = output_dir / "arc_sketch_program_heldout_candidates.jsonl"
    candidate_tsv_path = output_dir / "runtime" / "arc_sketch_heldout_candidates.tsv"
    source_path = output_dir / "arc_sketch_program_heldout_sources.jsonl"
    prior_path = output_dir / "arc_sketch_program_heldout_prior.tsv"
    attempts_path = output_dir / "arc_visual_pipeline_heldout_attempts.jsonl"
    records = arc_heldout_candidate_records(
        triage_path,
        excluded_keys,
        limit,
        per_family,
    )
    records, structure_proposals = add_arc_structure_op_hints(
        records,
        source_attempts,
        public_arc_dir,
        hints_per_task,
    )
    write_jsonl(candidate_jsonl_path, records)
    (
        probe_rows,
        summary_rows,
        attempt_rows,
        prior_count,
        candidate_count,
    ) = run_arc_sketch_candidate_export(
        records,
        arc_agi,
        public_arc_dir,
        candidate_tsv_path,
        source_path,
        prior_path,
        attempts_path,
        sketches_per_task,
    )
    return (
        probe_rows,
        summary_rows,
        attempt_rows,
        prior_count,
        candidate_count,
        records,
        structure_proposals,
    )


def aggregate_arc_probe_rejections(
    summary_rows: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    for row in summary_rows:
        family = str(row.get("family", ""))
        totals[family]["attempted"] += safe_int(row.get("attempted"))
        totals[family]["emitted"] += safe_int(row.get("emitted"))
        rejects = row.get("rejection_counts", {})
        if isinstance(rejects, dict):
            for reason, count in rejects.items():
                totals[family][str(reason)] += safe_int(count)
    return {
        family: dict(sorted(counter.items()))
        for family, counter in sorted(totals.items())
    }


def arc_attempt_pipeline_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("family", "")),
        str(row.get("pipeline_kind", "")),
        str(row.get("op_name", "")),
    )


def arc_attempt_task_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("family", "")),
        str(row.get("task_id", "")),
        str(row.get("split", "")),
    )


def arc_attempt_slot_values(row: dict[str, Any]) -> dict[str, Any]:
    slots = row.get("slot_values", {})
    return slots if isinstance(slots, dict) else {}


def arc_attempt_slot_distance(lhs: dict[str, Any], rhs: dict[str, Any]) -> float:
    left = arc_attempt_slot_values(lhs)
    right = arc_attempt_slot_values(rhs)
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    distance = 0.0
    for key in keys:
        if key not in left or key not in right:
            distance += 2.0
            continue
        try:
            distance += abs(float(left[key]) - float(right[key]))
        except (TypeError, ValueError):
            distance += 0.0 if left[key] == right[key] else 1.0
    return distance / float(len(keys))


def build_arc_visual_pipeline_proposals(
    source_attempts: list[dict[str, Any]],
    target_attempts: list[dict[str, Any]],
    max_per_task: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    training_attempts = [*source_attempts, *target_attempts]
    training_accepted = [
        row for row in training_attempts if safe_bool(row.get("accepted"))
    ]
    family_stats_by_key: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    global_stats_by_key: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    family_stats_by_task_key: dict[
        tuple[tuple[str, str, str], tuple[str, str, str]], Counter[str]
    ] = defaultdict(Counter)
    global_stats_by_task_key: dict[
        tuple[tuple[str, str, str], tuple[str, str]], Counter[str]
    ] = defaultdict(Counter)
    for source in training_attempts:
        source_task_key = arc_attempt_task_key(source)
        source_family, source_pipeline, source_op = arc_attempt_pipeline_key(source)
        family_key = (source_family, source_pipeline, source_op)
        global_key = (source_pipeline, source_op)
        accepted = 1 if safe_bool(source.get("accepted")) else 0
        family_stats_by_key[family_key]["attempted"] += 1
        family_stats_by_key[family_key]["accepted"] += accepted
        global_stats_by_key[global_key]["attempted"] += 1
        global_stats_by_key[global_key]["accepted"] += accepted
        family_stats_by_task_key[(source_task_key, family_key)]["attempted"] += 1
        family_stats_by_task_key[(source_task_key, family_key)]["accepted"] += accepted
        global_stats_by_task_key[(source_task_key, global_key)]["attempted"] += 1
        global_stats_by_task_key[(source_task_key, global_key)]["accepted"] += accepted
    proposals: list[dict[str, Any]] = []
    for row in target_attempts:
        family, pipeline_kind, op_name = arc_attempt_pipeline_key(row)
        task_key = arc_attempt_task_key(row)
        family_key = (family, pipeline_kind, op_name)
        global_key = (pipeline_kind, op_name)
        family_attempted = (
            family_stats_by_key[family_key].get("attempted", 0)
            - family_stats_by_task_key[(task_key, family_key)].get("attempted", 0)
        )
        family_accepted = (
            family_stats_by_key[family_key].get("accepted", 0)
            - family_stats_by_task_key[(task_key, family_key)].get("accepted", 0)
        )
        fallback_attempted = (
            global_stats_by_key[global_key].get("attempted", 0)
            - global_stats_by_task_key[(task_key, global_key)].get("attempted", 0)
        )
        fallback_accepted = (
            global_stats_by_key[global_key].get("accepted", 0)
            - global_stats_by_task_key[(task_key, global_key)].get("accepted", 0)
        )
        attempted = family_attempted or fallback_attempted
        accepted = family_accepted or fallback_accepted
        frequency_score = accepted / attempted if attempted else 0.0
        proposals.append(
            {
                "schema": ARC_VISUAL_PIPELINE_PROPOSAL_SCHEMA,
                "report_version": 1,
                "proposal_id": "avp:{}".format(
                    stable_id("frequency", row.get("attempt_id", ""), task_key)
                ),
                "proposer": "frequency_visual_pipeline_proposer",
                "family": family,
                "task_id": task_key[1],
                "split": task_key[2],
                "target_attempt_id": row.get("attempt_id", ""),
                "pipeline_kind": pipeline_kind,
                "op_name": op_name,
                "pipeline": row.get("pipeline", {}),
                "slot_values": row.get("slot_values", {}),
                "score": frequency_score,
                "target_accepted": safe_bool(row.get("accepted")),
                "target_rejection_reason": row.get("rejection_reason", ""),
                "rank_features": {
                    "leave_one_task_out_attempted": attempted,
                    "leave_one_task_out_accepted": accepted,
                    "leave_one_task_out_acceptance_rate": frequency_score,
                },
            }
        )

        best_source: dict[str, Any] | None = None
        best_score = -1.0
        best_distance = 0.0
        for source in training_accepted:
            if arc_attempt_task_key(source) == task_key:
                continue
            source_family, source_pipeline, source_op = arc_attempt_pipeline_key(source)
            relation_score = 1.0
            if source_family == family:
                relation_score += 4.0
            if source_pipeline == pipeline_kind:
                relation_score += 4.0
            if source_op == op_name:
                relation_score += 2.0
            distance = arc_attempt_slot_distance(source, row)
            score = relation_score / (1.0 + distance)
            if score > best_score:
                best_score = score
                best_distance = distance
                best_source = source
        if best_source is not None:
            proposals.append(
                {
                    "schema": ARC_VISUAL_PIPELINE_PROPOSAL_SCHEMA,
                    "report_version": 1,
                    "proposal_id": "avp:{}".format(
                        stable_id(
                            "nearest",
                            row.get("attempt_id", ""),
                            best_source.get("attempt_id", ""),
                        )
                    ),
                    "proposer": "nearest_visual_pipeline_proposer",
                    "family": family,
                    "task_id": task_key[1],
                    "split": task_key[2],
                    "target_attempt_id": row.get("attempt_id", ""),
                    "nearest_source_attempt_id": best_source.get("attempt_id", ""),
                    "pipeline_kind": pipeline_kind,
                    "op_name": op_name,
                    "pipeline": row.get("pipeline", {}),
                    "slot_values": row.get("slot_values", {}),
                    "score": best_score,
                    "target_accepted": safe_bool(row.get("accepted")),
                    "target_rejection_reason": row.get("rejection_reason", ""),
                    "rank_features": {
                        "nearest_slot_distance": best_distance,
                        "nearest_source_family": best_source.get("family", ""),
                        "nearest_source_pipeline_kind": best_source.get(
                            "pipeline_kind", ""
                        ),
                        "nearest_source_op_name": best_source.get("op_name", ""),
                    },
                }
            )

    retained: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in proposals:
        grouped[
            (
                str(row.get("proposer", "")),
                str(row.get("family", "")),
                str(row.get("task_id", "")),
                str(row.get("split", "")),
            )
        ].append(row)
    for key in sorted(grouped):
        rows = grouped[key]
        rows.sort(
            key=lambda item: (
                -safe_float(item.get("score")),
                str(item.get("target_attempt_id", "")),
            )
        )
        retained.extend(rows[:max_per_task])

    accepted_target_tasks = {
        arc_attempt_task_key(row)
        for row in target_attempts
        if safe_bool(row.get("accepted"))
    }
    by_proposer: dict[str, Counter[str]] = defaultdict(Counter)
    grouped_retained: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in retained:
        grouped_retained[
            (
                str(row.get("proposer", "")),
                str(row.get("family", "")),
                str(row.get("task_id", "")),
                str(row.get("split", "")),
            )
        ].append(row)
    for key, rows in grouped_retained.items():
        proposer, family, task_id, split = key
        if (family, task_id, split) not in accepted_target_tasks:
            continue
        rows.sort(
            key=lambda item: (
                -safe_float(item.get("score")),
                str(item.get("target_attempt_id", "")),
            )
        )
        by_proposer[proposer]["evaluated_tasks"] += 1
        if any(safe_bool(row.get("target_accepted")) for row in rows[:1]):
            by_proposer[proposer]["top1_hits"] += 1
        if any(safe_bool(row.get("target_accepted")) for row in rows[:3]):
            by_proposer[proposer]["top3_hits"] += 1
        if any(safe_bool(row.get("target_accepted")) for row in rows[:max_per_task]):
            by_proposer[proposer]["retained_hits"] += 1

    summary = {
        "schema": ARC_VISUAL_PIPELINE_PROPOSAL_SUMMARY_SCHEMA,
        "report_version": 1,
        "source_attempt_count": len(source_attempts),
        "source_accepted_count": sum(
            1 for row in source_attempts if safe_bool(row.get("accepted"))
        ),
        "training_attempt_count": len(training_attempts),
        "training_accepted_count": len(training_accepted),
        "training_mode": "leave_one_task_out_visual_pipeline_attempts",
        "target_attempt_count": len(target_attempts),
        "target_accepted_count": sum(
            1 for row in target_attempts if safe_bool(row.get("accepted"))
        ),
        "proposal_count": len(retained),
        "accepted_target_task_count": len(accepted_target_tasks),
        "by_proposer": {
            proposer: {
                **dict(counter),
                "top1_rate": pct(counter.get("top1_hits", 0), counter.get("evaluated_tasks", 0)),
                "top3_rate": pct(counter.get("top3_hits", 0), counter.get("evaluated_tasks", 0)),
                "retained_rate": pct(
                    counter.get("retained_hits", 0),
                    counter.get("evaluated_tasks", 0),
                ),
            }
            for proposer, counter in sorted(by_proposer.items())
        },
    }
    return retained, summary


ARC_VISUAL_PIPELINE_RUNTIME_FIELDS = [
    "split_name",
    "family",
    "task_id",
    "split",
    "attempt_id",
    "library_id",
    "attempt_index",
    "pipeline_kind",
    "op_name",
    "visual_program",
    "slot_contract",
    "slot_tokens",
    "accepted",
    "rejection_reason",
    "proposal_count",
    "support_tasks",
    "utility_mean",
    "selector",
    "background",
    "color",
    "output_color",
    "selector_kind",
    "include_background",
    "orientation",
    "count_mode",
    "x_align",
    "y_align",
    "x_anchor",
    "y_anchor",
    "x_offset",
    "y_offset",
    "width",
    "height",
    "period",
    "contrast_accepted_attempt_id",
    "contrast_slot_distance",
    "contrast_pair_kind",
    "contrast_rejected_sibling_count",
    "input_width",
    "input_height",
    "output_width",
    "output_height",
    "shape_scale_x",
    "shape_scale_y",
    "period_divides_input_width",
    "period_divides_input_height",
    "slot_width_matches_output",
    "slot_height_matches_output",
    "color_in_input",
    "color_in_output",
    "output_color_in_output",
    "background_in_input",
    "verifier_train_pairs",
    "verifier_checked_cells",
    "verifier_mismatches",
    "verifier_unsupported_pairs",
    "verifier_apply_failed_pairs",
    "verifier_shape_mismatch_pairs",
    "verifier_missing_output_pairs",
    "verifier_first_failing_pair",
    "verifier_mismatch_ratio",
    "verifier_score",
]


def arc_visual_pipeline_slot_tokens(row: dict[str, Any]) -> str:
    slots = row.get("slot_values", {})
    if not isinstance(slots, dict):
        return ""
    return pipe_join(
        ["{}={}".format(str(key), slots[key]) for key in sorted(slots)]
    )


def write_arc_visual_pipeline_runtime_attempts(
    path: Path,
    source_attempts: list[dict[str, Any]],
    target_attempts: list[dict[str, Any]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with path.open("w", encoding="utf-8") as fp:
        fp.write("\t".join(ARC_VISUAL_PIPELINE_RUNTIME_FIELDS) + "\n")
        for split_name, rows in (
            ("source", source_attempts),
            ("target", target_attempts),
        ):
            for row in rows:
                pipeline = row.get("pipeline", {})
                if not isinstance(pipeline, dict):
                    pipeline = {}
                task_slot_features = row.get("task_slot_features", {})
                if not isinstance(task_slot_features, dict):
                    task_slot_features = {}
                verifier_distance = row.get("verifier_distance", {})
                if not isinstance(verifier_distance, dict):
                    verifier_distance = {}
                fields = [
                    split_name,
                    row.get("family", ""),
                    row.get("task_id", ""),
                    row.get("split", ""),
                    row.get("attempt_id", ""),
                    row.get("library_id", ""),
                    row.get("attempt_index", ""),
                    row.get("pipeline_kind", ""),
                    row.get("op_name", ""),
                    row.get("visual_program", ""),
                    row.get("slot_contract", ""),
                    arc_visual_pipeline_slot_tokens(row),
                    bool(row.get("accepted")),
                    row.get("rejection_reason", ""),
                    row.get("proposal_count", ""),
                    row.get("support_tasks", ""),
                    row.get("utility_mean", ""),
                    pipeline.get("selector", ""),
                    pipeline.get("background", ""),
                    pipeline.get("color", ""),
                    pipeline.get("output_color", ""),
                    pipeline.get("selector_kind", ""),
                    1 if safe_bool(pipeline.get("include_background")) else 0,
                    pipeline.get("orientation", ""),
                    pipeline.get("count_mode", ""),
                    pipeline.get("x_align", ""),
                    pipeline.get("y_align", ""),
                    pipeline.get("x_anchor", ""),
                    pipeline.get("y_anchor", ""),
                    pipeline.get("x_offset", ""),
                    pipeline.get("y_offset", ""),
                    pipeline.get("width", ""),
                    pipeline.get("height", ""),
                    pipeline.get("period", ""),
                    row.get("contrast_accepted_attempt_id", ""),
                    row.get("contrast_slot_distance", ""),
                    row.get("contrast_pair_kind", ""),
                    row.get("contrast_rejected_sibling_count", ""),
                    task_slot_features.get("input_width", ""),
                    task_slot_features.get("input_height", ""),
                    task_slot_features.get("output_width", ""),
                    task_slot_features.get("output_height", ""),
                    task_slot_features.get("shape_scale_x", ""),
                    task_slot_features.get("shape_scale_y", ""),
                    1
                    if safe_bool(task_slot_features.get("period_divides_input_width"))
                    else 0,
                    1
                    if safe_bool(task_slot_features.get("period_divides_input_height"))
                    else 0,
                    1
                    if safe_bool(task_slot_features.get("slot_width_matches_output"))
                    else 0,
                    1
                    if safe_bool(task_slot_features.get("slot_height_matches_output"))
                    else 0,
                    1 if safe_bool(task_slot_features.get("color_in_input")) else 0,
                    1 if safe_bool(task_slot_features.get("color_in_output")) else 0,
                    1
                    if safe_bool(task_slot_features.get("output_color_in_output"))
                    else 0,
                    1
                    if safe_bool(task_slot_features.get("background_in_input"))
                    else 0,
                    verifier_distance.get("train_pairs", ""),
                    verifier_distance.get("checked_cells", ""),
                    verifier_distance.get("mismatches", ""),
                    verifier_distance.get("unsupported_pairs", ""),
                    verifier_distance.get("apply_failed_pairs", ""),
                    verifier_distance.get("shape_mismatch_pairs", ""),
                    verifier_distance.get("missing_output_pairs", ""),
                    verifier_distance.get("first_failing_pair", ""),
                    verifier_distance.get("mismatch_ratio", ""),
                    verifier_distance.get("verifier_score", ""),
                ]
                fp.write("\t".join(tsv_value(field) for field in fields))
                fp.write("\n")
                row_count += 1
    return row_count


def empty_arc_visual_pipeline_neural_summary(
    source_attempts: list[dict[str, Any]],
    target_attempts: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    return {
        "schema": ARC_VISUAL_PIPELINE_NEURAL_SUMMARY_SCHEMA,
        "report_version": 1,
        "proposer": "neural_visual_pipeline_ranker",
        "source_attempt_count": len(source_attempts),
        "source_accepted_count": sum(
            1 for row in source_attempts if safe_bool(row.get("accepted"))
        ),
        "target_attempt_count": len(target_attempts),
        "target_accepted_count": sum(
            1 for row in target_attempts if safe_bool(row.get("accepted"))
        ),
        "proposal_count": 0,
        "accepted_target_task_count": len(
            {
                arc_attempt_task_key(row)
                for row in target_attempts
                if safe_bool(row.get("accepted"))
            }
        ),
        "evaluated_tasks": 0,
        "top1_hits": 0,
        "top3_hits": 0,
        "retained_hits": 0,
        "top1_rate": 0.0,
        "top3_rate": 0.0,
        "retained_rate": 0.0,
        "training_mode": "source_to_target_visual_pipeline_attempts_with_dense_verifier_targets",
        "source_contrast_pair_count": 0,
        "target_contrast_pair_count": 0,
        "pairwise_training_updates": 0,
        "pairwise_last_epoch_violations": 0,
        "by_proposer": {},
        "by_accepted_op": {},
        "calibration_by_rejection_reason": {},
        "blocking_reason": reason,
    }


def hydrate_arc_visual_pipeline_neural_proposals(
    proposals: list[dict[str, Any]],
    target_attempts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_attempt_id = {
        str(row.get("attempt_id", "")): row
        for row in target_attempts
        if str(row.get("attempt_id", ""))
    }
    hydrated: list[dict[str, Any]] = []
    for proposal in proposals:
        row = dict(proposal)
        source = by_attempt_id.get(str(row.get("target_attempt_id", "")))
        if source is not None:
            row["pipeline"] = source.get("pipeline", {})
            row["slot_values"] = source.get("slot_values", {})
            row["task_slot_features"] = source.get("task_slot_features", {})
            row["verifier_distance"] = source.get("verifier_distance", {})
        hydrated.append(row)
    return hydrated


def run_arc_visual_pipeline_neural_ranker(
    args: argparse.Namespace,
    source_attempts: list[dict[str, Any]],
    target_attempts: list[dict[str, Any]],
    max_per_task: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runtime = Path(args.runtime)
    if not runtime.is_file():
        return [], empty_arc_visual_pipeline_neural_summary(
            source_attempts,
            target_attempts,
            "runtime_missing",
        )
    if not source_attempts or not target_attempts:
        return [], empty_arc_visual_pipeline_neural_summary(
            source_attempts,
            target_attempts,
            "missing_attempt_surface",
        )
    runtime_dir = args.output_dir / "runtime"
    attempt_rows = runtime_dir / "arc_visual_pipeline_attempts.tsv"
    proposals_output = runtime_dir / "arc_visual_pipeline_neural_proposals.jsonl"
    summary_output = runtime_dir / "arc_visual_pipeline_neural_summary.jsonl"
    write_arc_visual_pipeline_runtime_attempts(
        attempt_rows,
        source_attempts,
        target_attempts,
    )
    command = [
        str(runtime),
        "--arc-pipeline-attempts",
        str(attempt_rows),
        "--arc-pipeline-proposals-output",
        str(proposals_output),
        "--arc-pipeline-summary-output",
        str(summary_output),
        "--arc-pipeline-max-per-task",
        str(max_per_task),
    ]
    subprocess.run(command, check=True)
    proposals = hydrate_arc_visual_pipeline_neural_proposals(
        parse_json_rows(proposals_output),
        target_attempts,
    )
    summaries = parse_json_rows(summary_output)
    summary = summaries[0] if summaries else empty_arc_visual_pipeline_neural_summary(
        source_attempts,
        target_attempts,
        "summary_missing",
    )
    return proposals, summary


def merge_arc_visual_pipeline_proposal_summaries(
    baseline_summary: dict[str, Any],
    neural_summary: dict[str, Any],
    proposal_count: int,
) -> dict[str, Any]:
    merged = dict(baseline_summary)
    by_proposer = dict(merged.get("by_proposer", {}) or {})
    neural_by_proposer = neural_summary.get("by_proposer", {})
    if isinstance(neural_by_proposer, dict):
        by_proposer.update(neural_by_proposer)
    elif neural_summary.get("proposer"):
        by_proposer[str(neural_summary.get("proposer"))] = {
            "evaluated_tasks": safe_int(neural_summary.get("evaluated_tasks")),
            "top1_hits": safe_int(neural_summary.get("top1_hits")),
            "top3_hits": safe_int(neural_summary.get("top3_hits")),
            "retained_hits": safe_int(neural_summary.get("retained_hits")),
            "top1_rate": safe_float(neural_summary.get("top1_rate")),
            "top3_rate": safe_float(neural_summary.get("top3_rate")),
            "retained_rate": safe_float(neural_summary.get("retained_rate")),
        }
    merged["proposal_count"] = proposal_count
    merged["by_proposer"] = by_proposer
    merged["neural_ranker_summary"] = neural_summary
    return merged


def run_arc_log(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fp:
        completed = subprocess.run(
            command,
            stdout=fp,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(completed.returncode)


def write_filtered_arc_prior(
    global_prior_path: Path,
    output_path: Path,
    library_ids: set[str],
) -> int:
    if not global_prior_path.is_file() or not library_ids:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(ARC_SKETCH_PRIOR_HEADER, encoding="utf-8")
        return 0
    kept = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with global_prior_path.open("r", encoding="utf-8") as source:
        header = source.readline()
        with output_path.open("w", encoding="utf-8") as target:
            target.write(header if header else ARC_SKETCH_PRIOR_HEADER)
            for line in source:
                fields = line.rstrip("\n").split("\t")
                if len(fields) >= 4 and fields[3] in library_ids:
                    target.write(line)
                    kept += 1
    return kept


def arc_log_metrics(log_path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    selection: dict[str, Any] = {}
    for row in parse_json_rows(log_path):
        if row.get("schema") == "crlp.arc_loader.summary.v1":
            summary = row
        elif row.get("schema") == "crlp.arc_hypothesis.selection.v1":
            selection = row
    sketch_template: dict[str, Any] = {}
    for template in selection.get("templates", []) or []:
        if isinstance(template, dict) and template.get("name") == ARC_SKETCH_TEMPLATE_NAME:
            sketch_template = template
            break
    return {
        "loaded_tasks": safe_int(summary.get("loaded_tasks")),
        "certified_tasks": safe_int(summary.get("certified_tasks")),
        "unique_prediction_tasks": safe_int(summary.get("unique_prediction_tasks")),
        "train_consistent_rules": safe_int(summary.get("train_consistent_rules")),
        "rules": safe_int(summary.get("rules")),
        "slot_probe_rows": safe_int(selection.get("slot_probe_rows")),
        "slot_probe_seeded_proposals": safe_int(selection.get("slot_probe_seeded_proposals")),
        "template_probe_rows": safe_int(selection.get("template_probe_rows")),
        "template_probe_seeded_proposals": safe_int(
            selection.get("template_probe_seeded_proposals")
        ),
        "proposals_generated": safe_int(selection.get("proposals_generated")),
        "proposals_retained": safe_int(selection.get("proposals_retained")),
        "proposals_train_consistent_pre_cap": safe_int(
            selection.get("proposals_train_consistent_pre_cap")
        ),
        "proposals_train_consistent": safe_int(selection.get("proposals_train_consistent")),
        "selected_template": next(
            (
                str(template.get("name", ""))
                for template in selection.get("templates", []) or []
                if isinstance(template, dict) and safe_int(template.get("selected")) > 0
            ),
            "",
        ),
        "sketch_attempted": safe_int(sketch_template.get("attempted")),
        "sketch_generated": safe_int(sketch_template.get("generated")),
        "sketch_retained": safe_int(sketch_template.get("retained")),
        "sketch_train_consistent_pre_cap": safe_int(
            sketch_template.get("train_consistent_pre_cap")
        ),
        "sketch_train_consistent_retained": safe_int(
            sketch_template.get("train_consistent_retained")
        ),
        "sketch_selected": safe_int(sketch_template.get("selected")),
        "sketch_slot_probe_score": safe_float(sketch_template.get("slot_probe_score")),
        "sketch_template_probe_score": safe_float(
            sketch_template.get("template_probe_score")
        ),
    }


def arc_probe_compare_flags() -> list[str]:
    return [
        "--program-budget-tight-grid",
        "--loader-summary-only",
        "--induction",
        "--hypothesis-top-k",
        "1",
        "--disable-hypothesis-coverage-selection",
        "--disable-hypothesis-structural-score",
        "--disable-hypothesis-train-filter",
        "--disable-hypothesis-slot-prior",
        "--disable-hypothesis-neural-prior",
        "--disable-hypothesis-cold-start",
        "--disable-hypothesis-train-consistency-prior",
    ]


def compare_arc_sketch_probes(
    arc_agi: Path,
    prior_path: Path,
    source_rows: list[dict[str, Any]],
    output_dir: Path,
    limit: int,
    log_dir_name: str = "arc_sketch_program_probe_logs",
) -> list[dict[str, Any]]:
    if not source_rows or not arc_agi.is_file() or not prior_path.is_file() or limit == 0:
        return []
    selected_rows: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    rows_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_by_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(
        source_rows,
        key=lambda item: (
            str(item.get("task_id", "")),
            str(item.get("split", "")),
            str(item.get("library_id", "")),
        ),
    ):
        rows_by_family[str(row.get("family", ""))].append(row)
        rows_by_task[(str(row.get("task_id", "")), str(row.get("split", "")))].append(row)
    family_names = sorted(rows_by_family)
    while family_names:
        progressed = False
        for family in list(family_names):
            while rows_by_family[family]:
                row = rows_by_family[family].pop(0)
                task_id = str(row.get("task_id", ""))
                if task_id in seen_tasks:
                    continue
                seen_tasks.add(task_id)
                selected_rows.append(row)
                progressed = True
                break
            if not rows_by_family[family]:
                family_names.remove(family)
            if limit > 0 and len(selected_rows) >= limit:
                family_names = []
                break
        if not progressed:
            break

    flags = arc_probe_compare_flags()
    rows: list[dict[str, Any]] = []
    log_dir = output_dir / log_dir_name
    for row in selected_rows:
        task_path = Path(str(row["task_path"]))
        task_id = str(row["task_id"])
        split = str(row.get("split", ""))
        task_dir = task_path.parent
        baseline_log = log_dir / f"{task_id}.baseline.jsonl"
        probed_log = log_dir / f"{task_id}.probed.jsonl"
        task_prior = log_dir / f"{task_id}.prior.tsv"
        task_library_ids = {
            str(task_row.get("library_id", ""))
            for task_row in rows_by_task.get((task_id, split), [])
            if task_row.get("library_id")
        }
        task_prior_rows = write_filtered_arc_prior(prior_path, task_prior, task_library_ids)
        base_command = [
            str(arc_agi),
            "--task-dir",
            str(task_dir),
            "--task-ids",
            task_id,
            "--max-tasks",
            "1",
            *flags,
        ]
        probed_command = [
            str(arc_agi),
            "--task-dir",
            str(task_dir),
            "--task-ids",
            task_id,
            "--max-tasks",
            "1",
            "--hypothesis-library-prior",
            str(task_prior),
            *flags,
        ]
        baseline_exit = run_arc_log(base_command, baseline_log)
        probed_exit = run_arc_log(probed_command, probed_log)
        baseline_metrics = arc_log_metrics(baseline_log)
        probed_metrics = arc_log_metrics(probed_log)
        baseline_certified = safe_int(baseline_metrics.get("certified_tasks")) > 0
        probed_certified = safe_int(probed_metrics.get("certified_tasks")) > 0
        rows.append(
            {
                "schema": ARC_SKETCH_PROBE_COMPARISON_SCHEMA,
                "report_version": 1,
                "task_id": task_id,
                "split": row.get("split"),
                "family": row.get("family"),
                "baseline_exit_code": baseline_exit,
                "probed_exit_code": probed_exit,
                "baseline_log": str(baseline_log),
                "probed_log": str(probed_log),
                "prior_path": str(task_prior),
                "global_prior_path": str(prior_path),
                "task_prior_row_count": task_prior_rows,
                "baseline": baseline_metrics,
                "probed": probed_metrics,
                "delta": {
                    "slot_probe_rows": safe_int(probed_metrics.get("slot_probe_rows"))
                    - safe_int(baseline_metrics.get("slot_probe_rows")),
                    "slot_probe_seeded_proposals": safe_int(
                        probed_metrics.get("slot_probe_seeded_proposals")
                    )
                    - safe_int(baseline_metrics.get("slot_probe_seeded_proposals")),
                    "sketch_train_consistent_pre_cap": safe_int(
                        probed_metrics.get("sketch_train_consistent_pre_cap")
                    )
                    - safe_int(baseline_metrics.get("sketch_train_consistent_pre_cap")),
                    "sketch_train_consistent_retained": safe_int(
                        probed_metrics.get("sketch_train_consistent_retained")
                    )
                    - safe_int(baseline_metrics.get("sketch_train_consistent_retained")),
                    "certified_tasks": safe_int(probed_metrics.get("certified_tasks"))
                    - safe_int(baseline_metrics.get("certified_tasks")),
                },
                "probe_selected_sketch": safe_int(probed_metrics.get("sketch_selected")) > 0,
                "probe_added_train_consistent_sketch": safe_int(
                    probed_metrics.get("sketch_train_consistent_pre_cap")
                )
                > safe_int(baseline_metrics.get("sketch_train_consistent_pre_cap")),
                "baseline_certified": baseline_certified,
                "probed_certified": probed_certified,
                "baseline_already_certified": baseline_certified and probed_certified,
                "newly_certified": probed_certified and not baseline_certified,
            }
        )
    return rows


def arc_sketch_probe_markdown(
    rows: list[dict[str, Any]],
    source_count: int,
    title: str = "ARC Sketch Program Probe Comparison",
) -> str:
    lines = [
        f"# {title}",
        "",
        f"- Probe source sketches: {source_count}",
        f"- Compared tasks: {len(rows)}",
        f"- Tasks with extra sketch train-consistent pre-cap proposals: {sum(1 for row in rows if row.get('probe_added_train_consistent_sketch'))}",
        f"- Tasks selecting `arc_sketch_program`: {sum(1 for row in rows if row.get('probe_selected_sketch'))}",
        f"- Certified task delta: {sum(safe_int(row.get('delta', {}).get('certified_tasks')) for row in rows)}",
        f"- Newly certified baseline-unsolved tasks: {sum(1 for row in rows if row.get('newly_certified'))}",
        f"- Baseline-already-certified tasks: {sum(1 for row in rows if row.get('baseline_already_certified'))}",
        "",
        "| Task | Family | Base TC pre-cap | Probed TC pre-cap | Base TC retained | Probed TC retained | Base Selected | Probed Selected | New Solve | Baseline Solved | Certified Delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: |",
    ]
    for row in rows:
        base = row.get("baseline", {})
        probed = row.get("probed", {})
        delta = row.get("delta", {})
        lines.append(
            "| `{}` | `{}` | {} | {} | {} | {} | `{}` | `{}` | `{}` | `{}` | {} |".format(
                row.get("task_id", ""),
                row.get("family", ""),
                safe_int(base.get("sketch_train_consistent_pre_cap")),
                safe_int(probed.get("sketch_train_consistent_pre_cap")),
                safe_int(base.get("sketch_train_consistent_retained")),
                safe_int(probed.get("sketch_train_consistent_retained")),
                base.get("selected_template", ""),
                probed.get("selected_template", ""),
                row.get("newly_certified", False),
                row.get("baseline_already_certified", False),
                safe_int(delta.get("certified_tasks")),
            )
        )
    return "\n".join(lines) + "\n"


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


def build_sketch_proposals(
    sketches: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    neural_proposals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    stats = sketch_stats(examples)
    return [
        *build_frequency_proposals(sketches, stats),
        *build_nearest_neighbor_proposals(sketches, examples),
        *neural_proposals,
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
    arc_sketch_probe_sources: list[dict[str, Any]],
    arc_sketch_probe_summaries: list[dict[str, Any]],
    arc_sketch_probe_comparison: list[dict[str, Any]],
    arc_visual_pipeline_attempts: list[dict[str, Any]],
    arc_sketch_heldout_sources: list[dict[str, Any]],
    arc_sketch_heldout_summaries: list[dict[str, Any]],
    arc_sketch_heldout_comparison: list[dict[str, Any]],
    arc_visual_pipeline_heldout_attempts: list[dict[str, Any]],
    arc_sketch_heldout_candidate_count: int,
    arc_structure_proposals: list[dict[str, Any]],
    arc_visual_pipeline_proposals: list[dict[str, Any]],
    arc_visual_pipeline_proposal_summary: dict[str, Any],
    sketch_memory_summary: dict[str, Any],
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
        f"- ARC sketch program probe sketches: {len(arc_sketch_probe_sources)}",
        f"- ARC visual pipeline source attempts: {len(arc_visual_pipeline_attempts)}",
        f"- ARC sketch program compared tasks: {len(arc_sketch_probe_comparison)}",
        f"- Held-out ARC candidate tasks: {arc_sketch_heldout_candidate_count}",
        f"- Held-out ARC structure proposals: {len(arc_structure_proposals)}",
        f"- Held-out ARC sketch program sketches: {len(arc_sketch_heldout_sources)}",
        f"- Held-out ARC visual pipeline attempts: {len(arc_visual_pipeline_heldout_attempts)}",
        f"- Held-out ARC compared tasks: {len(arc_sketch_heldout_comparison)}",
        f"- ARC visual pipeline baseline proposals: {len(arc_visual_pipeline_proposals)}",
        f"- Grammar sketches: {len(sketches)}",
        f"- Grammar sketch examples: {len(sketch_examples)}",
        f"- Grammar sketch proposals: {len(sketch_proposals)}",
        "- Grammar sketch memory: `{}` (loaded: `{}`, epochs seen: {})".format(
            sketch_memory_summary.get("checkpoint_path", ""),
            sketch_memory_summary.get("checkpoint_loaded", False),
            sketch_memory_summary.get("checkpoint_epochs_seen_after", 0),
        ),
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

    if arc_sketch_probe_sources or arc_sketch_probe_summaries:
        lines.extend(["", "## ARC Sketch Program Probes", ""])
        lines.extend(["| Family | Probe sketches |", "| --- | ---: |"])
        probe_family_counts = Counter(
            str(row.get("family", "")) for row in arc_sketch_probe_sources
        )
        promoted_arc_families = {
            str(row.get("family", ""))
            for row in promoted
            if row.get("domain") == "arc_agi"
        }
        for family in sorted(set(probe_family_counts) | promoted_arc_families):
            lines.append(f"| `{family}` | {probe_family_counts.get(family, 0)} |")
        op_counts = Counter(str(row.get("op_name", "")) for row in arc_sketch_probe_sources)
        lines.extend(["", "| Sketch op | Probe sketches |", "| --- | ---: |"])
        for op_name, count in op_counts.most_common():
            lines.append(f"| `{op_name}` | {count} |")
        if arc_sketch_probe_summaries:
            rejection_totals: dict[str, Counter[str]] = defaultdict(Counter)
            emitted_by_family: Counter[str] = Counter()
            attempted_by_family: Counter[str] = Counter()
            for row in arc_sketch_probe_summaries:
                family = str(row.get("family", ""))
                emitted_by_family[family] += safe_int(row.get("emitted"))
                attempted_by_family[family] += safe_int(row.get("attempted"))
                rejects = row.get("rejection_counts", {})
                if isinstance(rejects, dict):
                    for reason, count in rejects.items():
                        rejection_totals[family][str(reason)] += safe_int(count)
            lines.extend(
                [
                    "",
                    "| Family | Attempted | Emitted | Unsupported shape | No train match | Duplicate | Cap hit |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
                )
            for family in sorted(set(attempted_by_family) | set(emitted_by_family)):
                rejects = rejection_totals[family]
                lines.append(
                    "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                        family,
                        attempted_by_family.get(family, 0),
                        emitted_by_family.get(family, 0),
                        rejects.get("unsupported_shape", 0),
                        rejects.get("no_train_match", 0),
                        rejects.get("duplicate", 0),
                        rejects.get("cap_hit", 0),
                    )
                )
        if arc_visual_pipeline_attempts:
            accepted = sum(
                1 for row in arc_visual_pipeline_attempts if safe_bool(row.get("accepted"))
            )
            lines.extend(
                [
                    "",
                    f"- Visual pipeline attempt rows: {len(arc_visual_pipeline_attempts)}",
                    f"- Visual pipeline accepted rows: {accepted}",
                ]
            )
        if arc_sketch_probe_comparison:
            lines.extend(
                [
                    "",
                    "| Probe comparison | Tasks |",
                    "| --- | ---: |",
                    "| extra sketch train-consistent pre-cap proposals | {} |".format(
                        sum(
                            1
                            for row in arc_sketch_probe_comparison
                            if row.get("probe_added_train_consistent_sketch")
                        )
                    ),
                    "| selected `arc_sketch_program` | {} |".format(
                        sum(
                            1
                            for row in arc_sketch_probe_comparison
                            if row.get("probe_selected_sketch")
                        )
                    ),
                    "| certified task delta | {} |".format(
                        sum(
                            safe_int(row.get("delta", {}).get("certified_tasks"))
                            for row in arc_sketch_probe_comparison
                        )
                    ),
                    "| newly certified baseline-unsolved tasks | {} |".format(
                        sum(1 for row in arc_sketch_probe_comparison if row.get("newly_certified"))
                    ),
                    "| baseline-already-certified tasks | {} |".format(
                        sum(
                            1
                            for row in arc_sketch_probe_comparison
                            if row.get("baseline_already_certified")
                        )
                    ),
                ]
            )

    if (
        arc_sketch_heldout_candidate_count > 0
        or arc_sketch_heldout_sources
        or arc_sketch_heldout_summaries
    ):
        lines.extend(["", "## Held-Out ARC Sketch Program Probes", ""])
        lines.append(f"- Candidate tasks: {arc_sketch_heldout_candidate_count}")
        if arc_structure_proposals:
            hinted_ops = Counter(
                op
                for row in arc_structure_proposals
                for op in row.get("proposed_ops", [])
            )
            lines.append(
                "- Structure op hints: {}".format(
                    ", ".join(
                        f"`{op}`={count}" for op, count in hinted_ops.most_common()
                    )
                )
            )
        lines.append("")
        lines.extend(["| Family | Probe sketches |", "| --- | ---: |"])
        heldout_family_counts = Counter(
            str(row.get("family", "")) for row in arc_sketch_heldout_sources
        )
        heldout_summary_families = {
            str(row.get("family", "")) for row in arc_sketch_heldout_summaries
        }
        for family in sorted(set(heldout_family_counts) | heldout_summary_families):
            lines.append(f"| `{family}` | {heldout_family_counts.get(family, 0)} |")
        heldout_op_counts = Counter(
            str(row.get("op_name", "")) for row in arc_sketch_heldout_sources
        )
        lines.extend(["", "| Sketch op | Probe sketches |", "| --- | ---: |"])
        for op_name, count in heldout_op_counts.most_common():
            lines.append(f"| `{op_name}` | {count} |")
        if not heldout_op_counts:
            lines.append("| `(none)` | 0 |")
        if arc_sketch_heldout_summaries:
            rejection_totals: dict[str, Counter[str]] = defaultdict(Counter)
            emitted_by_family: Counter[str] = Counter()
            attempted_by_family: Counter[str] = Counter()
            for row in arc_sketch_heldout_summaries:
                family = str(row.get("family", ""))
                emitted_by_family[family] += safe_int(row.get("emitted"))
                attempted_by_family[family] += safe_int(row.get("attempted"))
                rejects = row.get("rejection_counts", {})
                if isinstance(rejects, dict):
                    for reason, count in rejects.items():
                        rejection_totals[family][str(reason)] += safe_int(count)
            lines.extend(
                [
                    "",
                    "| Family | Attempted | Emitted | Unsupported shape | No train match | Duplicate | Cap hit |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for family in sorted(set(attempted_by_family) | set(emitted_by_family)):
                rejects = rejection_totals[family]
                lines.append(
                    "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                        family,
                        attempted_by_family.get(family, 0),
                        emitted_by_family.get(family, 0),
                        rejects.get("unsupported_shape", 0),
                        rejects.get("no_train_match", 0),
                        rejects.get("duplicate", 0),
                        rejects.get("cap_hit", 0),
                    )
                )
        if arc_visual_pipeline_heldout_attempts:
            heldout_accepted = sum(
                1
                for row in arc_visual_pipeline_heldout_attempts
                if safe_bool(row.get("accepted"))
            )
            lines.extend(
                [
                    "",
                    f"- Held-out visual pipeline attempt rows: {len(arc_visual_pipeline_heldout_attempts)}",
                    f"- Held-out visual pipeline accepted rows: {heldout_accepted}",
                ]
            )
        if arc_visual_pipeline_proposals:
            lines.extend(
                [
                    "",
                    "### ARC Visual Pipeline Proposers",
                    "",
                    "| Proposer | Evaluated tasks | Top-1 | Top-3 | Retained |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            by_proposer = arc_visual_pipeline_proposal_summary.get("by_proposer", {})
            if isinstance(by_proposer, dict):
                for proposer, stats in sorted(by_proposer.items()):
                    if not isinstance(stats, dict):
                        continue
                    lines.append(
                        "| `{}` | {} | {:.1%} | {:.1%} | {:.1%} |".format(
                            proposer,
                            safe_int(stats.get("evaluated_tasks")),
                            safe_float(stats.get("top1_rate")),
                            safe_float(stats.get("top3_rate")),
                            safe_float(stats.get("retained_rate")),
                        )
                    )
            neural_summary = arc_visual_pipeline_proposal_summary.get(
                "neural_ranker_summary", {}
            )
            if isinstance(neural_summary, dict):
                lines.extend(
                    [
                        "",
                        "- Neural contrast pairs: source {} / held-out {}".format(
                            safe_int(neural_summary.get("source_contrast_pair_count")),
                            safe_int(neural_summary.get("target_contrast_pair_count")),
                        ),
                        "- Pairwise updates: {}; last-epoch pair violations: {}".format(
                            safe_int(neural_summary.get("pairwise_training_updates")),
                            safe_int(neural_summary.get("pairwise_last_epoch_violations")),
                        ),
                    ]
                )
            if isinstance(neural_summary, dict) and neural_summary.get(
                "by_accepted_op"
            ):
                lines.extend(
                    [
                        "",
                        "| Neural accepted op | Evaluated tasks | Top-1 | Top-3 | Retained |",
                        "| --- | ---: | ---: | ---: | ---: |",
                    ]
                )
                by_op = neural_summary.get("by_accepted_op", {})
                if isinstance(by_op, dict):
                    for op_name, stats in sorted(by_op.items()):
                        if not isinstance(stats, dict):
                            continue
                        lines.append(
                            "| `{}` | {} | {:.1%} | {:.1%} | {:.1%} |".format(
                                op_name,
                                safe_int(stats.get("evaluated_tasks")),
                                safe_float(stats.get("top1_rate")),
                                safe_float(stats.get("top3_rate")),
                                safe_float(stats.get("retained_rate")),
                            )
                        )
        if arc_sketch_heldout_comparison:
            lines.extend(
                [
                    "",
                    "| Held-out comparison | Tasks |",
                    "| --- | ---: |",
                    "| extra sketch train-consistent pre-cap proposals | {} |".format(
                        sum(
                            1
                            for row in arc_sketch_heldout_comparison
                            if row.get("probe_added_train_consistent_sketch")
                        )
                    ),
                    "| selected `arc_sketch_program` | {} |".format(
                        sum(
                            1
                            for row in arc_sketch_heldout_comparison
                            if row.get("probe_selected_sketch")
                        )
                    ),
                    "| certified task delta | {} |".format(
                        sum(
                            safe_int(row.get("delta", {}).get("certified_tasks"))
                            for row in arc_sketch_heldout_comparison
                        )
                    ),
                    "| newly certified baseline-unsolved tasks | {} |".format(
                        sum(
                            1
                            for row in arc_sketch_heldout_comparison
                            if row.get("newly_certified")
                        )
                    ),
                    "| baseline-already-certified tasks | {} |".format(
                        sum(
                            1
                            for row in arc_sketch_heldout_comparison
                            if row.get("baseline_already_certified")
                        )
                    ),
                ]
            )

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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [build_candidate(group) for group in groups]
    sketches, sketch_examples = build_sketch_corpus(args, domains)
    compiled_deltas, verifications, neural_sketch_proposals, sketch_memory_summary = run_c_runtime(
        args,
        candidates,
        groups,
        sketches,
        sketch_examples,
    )
    promoted = promoted_rows(candidates, verifications)
    (
        arc_sketch_probe_sources,
        arc_sketch_probe_summaries,
        arc_visual_pipeline_attempts,
        arc_sketch_probe_prior_row_count,
        arc_sketch_probe_candidate_count,
    ) = run_arc_sketch_probe_export(
        groups,
        promoted,
        args.arc_agi,
        args.public_arc_dir,
        args.output_dir,
        args.arc_probe_sketches_per_task,
    )
    excluded_arc_keys = {arc_task_key(row) for row in parse_json_rows(args.arc_failure_proof)}
    (
        arc_sketch_heldout_sources,
        arc_sketch_heldout_summaries,
        arc_visual_pipeline_heldout_attempts,
        arc_sketch_heldout_prior_row_count,
        arc_sketch_heldout_candidate_count,
        arc_sketch_heldout_candidates,
        arc_structure_proposals,
    ) = run_arc_sketch_heldout_export(
        args.arc_heldout_triage,
        excluded_arc_keys,
        arc_visual_pipeline_attempts,
        args.arc_agi,
        args.public_arc_dir,
        args.output_dir,
        args.arc_probe_sketches_per_task,
        args.arc_heldout_limit,
        args.arc_heldout_per_family,
        args.arc_heldout_op_hints_per_task,
    )
    sketch_proposals = build_sketch_proposals(
        sketches,
        sketch_examples,
        neural_sketch_proposals,
    )
    (
        arc_visual_pipeline_baseline_proposals,
        arc_visual_pipeline_baseline_summary,
    ) = build_arc_visual_pipeline_proposals(
        arc_visual_pipeline_attempts,
        arc_visual_pipeline_heldout_attempts,
    )
    (
        arc_visual_pipeline_neural_proposals,
        arc_visual_pipeline_neural_summary,
    ) = run_arc_visual_pipeline_neural_ranker(
        args,
        arc_visual_pipeline_attempts,
        arc_visual_pipeline_heldout_attempts,
    )
    arc_visual_pipeline_proposals = [
        *arc_visual_pipeline_baseline_proposals,
        *arc_visual_pipeline_neural_proposals,
    ]
    arc_visual_pipeline_proposal_summary = (
        merge_arc_visual_pipeline_proposal_summaries(
            arc_visual_pipeline_baseline_summary,
            arc_visual_pipeline_neural_summary,
            len(arc_visual_pipeline_proposals),
        )
    )
    sketch_verifications, promoted_sketches = build_sketch_verifications(sketches, sketch_examples)

    write_jsonl(args.output_dir / "grammar_delta_candidates.jsonl", candidates)
    write_jsonl(args.output_dir / "compiled_grammar_deltas.jsonl", compiled_deltas)
    write_jsonl(args.output_dir / "grammar_delta_verification.jsonl", verifications)
    write_jsonl(args.output_dir / "promoted_grammar_deltas.jsonl", promoted)
    arc_sketch_probe_comparison = compare_arc_sketch_probes(
        args.arc_agi,
        args.output_dir / "arc_sketch_program_probe_prior.tsv",
        arc_sketch_probe_sources,
        args.output_dir,
        args.arc_probe_compare_limit,
    )
    arc_sketch_heldout_comparison = compare_arc_sketch_probes(
        args.arc_agi,
        args.output_dir / "arc_sketch_program_heldout_prior.tsv",
        arc_sketch_heldout_sources,
        args.output_dir,
        args.arc_heldout_compare_limit,
        "arc_sketch_program_heldout_logs",
    )
    write_jsonl(
        args.output_dir / "arc_sketch_program_probe_comparison.jsonl",
        arc_sketch_probe_comparison,
    )
    write_jsonl(
        args.output_dir / "arc_sketch_program_heldout_comparison.jsonl",
        arc_sketch_heldout_comparison,
    )
    (args.output_dir / "arc_sketch_program_probe_comparison.md").write_text(
        arc_sketch_probe_markdown(
            arc_sketch_probe_comparison,
            len(arc_sketch_probe_sources),
        ),
        encoding="utf-8",
    )
    (args.output_dir / "arc_sketch_program_heldout_comparison.md").write_text(
        arc_sketch_probe_markdown(
            arc_sketch_heldout_comparison,
            len(arc_sketch_heldout_sources),
            "Held-Out ARC Sketch Program Probe Comparison",
        ),
        encoding="utf-8",
    )
    write_jsonl(args.output_dir / "grammar_sketches.jsonl", sketches)
    write_jsonl(args.output_dir / "grammar_sketch_examples.jsonl", sketch_examples)
    write_jsonl(args.output_dir / "grammar_sketch_proposals.jsonl", sketch_proposals)
    write_jsonl(
        args.output_dir / "arc_visual_pipeline_proposals.jsonl",
        arc_visual_pipeline_proposals,
    )
    write_jsonl(
        args.output_dir / "arc_sketch_program_heldout_structure_proposals.jsonl",
        arc_structure_proposals,
    )
    write_jsonl(
        args.output_dir / "arc_visual_pipeline_neural_proposals.jsonl",
        arc_visual_pipeline_neural_proposals,
    )
    write_jsonl(
        args.output_dir / "arc_visual_pipeline_neural_summary.jsonl",
        [arc_visual_pipeline_neural_summary],
    )
    write_jsonl(
        args.output_dir / "arc_visual_pipeline_proposal_summary.jsonl",
        [arc_visual_pipeline_proposal_summary],
    )
    write_jsonl(args.output_dir / "grammar_sketch_verification.jsonl", sketch_verifications)
    write_jsonl(args.output_dir / "promoted_grammar_sketches.jsonl", promoted_sketches)
    write_jsonl(args.output_dir / "grammar_sketch_memory_summary.jsonl", [sketch_memory_summary])
    summary = {
        "schema": SUMMARY_SCHEMA,
        "report_version": 1,
        "domains": sorted(domains),
        "candidate_count": len(candidates),
        "compiled_delta_count": len(compiled_deltas),
        "verification_count": len(verifications),
        "promoted_count": len(promoted),
        "arc_sketch_program_probe_candidate_count": arc_sketch_probe_candidate_count,
        "arc_sketch_program_probe_source_count": len(arc_sketch_probe_sources),
        "arc_visual_pipeline_attempt_count": len(arc_visual_pipeline_attempts),
        "arc_visual_pipeline_accepted_attempt_count": sum(
            1 for row in arc_visual_pipeline_attempts if safe_bool(row.get("accepted"))
        ),
        "arc_sketch_program_probe_prior_row_count": arc_sketch_probe_prior_row_count,
        "arc_sketch_program_probe_comparison_count": len(arc_sketch_probe_comparison),
        "arc_sketch_program_probe_added_train_consistent_tasks": sum(
            1
            for row in arc_sketch_probe_comparison
            if row.get("probe_added_train_consistent_sketch")
        ),
        "arc_sketch_program_probe_selected_tasks": sum(
            1 for row in arc_sketch_probe_comparison if row.get("probe_selected_sketch")
        ),
        "arc_sketch_program_probe_certified_task_delta": sum(
            safe_int(row.get("delta", {}).get("certified_tasks"))
            for row in arc_sketch_probe_comparison
        ),
        "arc_sketch_program_probe_newly_certified_tasks": sum(
            1 for row in arc_sketch_probe_comparison if row.get("newly_certified")
        ),
        "arc_sketch_program_probe_baseline_already_certified_tasks": sum(
            1
            for row in arc_sketch_probe_comparison
            if row.get("baseline_already_certified")
        ),
        "arc_sketch_program_probe_sources_by_family": {
            family: Counter(
                str(row.get("family", "")) for row in arc_sketch_probe_sources
            ).get(family, 0)
            for family in sorted(
                {
                    str(row.get("family", ""))
                    for row in promoted
                    if row.get("domain") == "arc_agi"
                }
                | {str(row.get("family", "")) for row in arc_sketch_probe_sources}
            )
        },
        "arc_sketch_program_probe_sources_by_op": dict(
            sorted(Counter(str(row.get("op_name", "")) for row in arc_sketch_probe_sources).items())
        ),
        "arc_sketch_program_probe_rejections_by_family": aggregate_arc_probe_rejections(
            arc_sketch_probe_summaries
        ),
        "arc_sketch_program_heldout_candidate_count": arc_sketch_heldout_candidate_count,
        "arc_sketch_program_heldout_structure_proposal_count": len(
            arc_structure_proposals
        ),
        "arc_sketch_program_heldout_op_hints_per_task": args.arc_heldout_op_hints_per_task,
        "arc_sketch_program_heldout_op_hints_by_op": dict(
            sorted(
                Counter(
                    op
                    for row in arc_structure_proposals
                    for op in row.get("proposed_ops", [])
                ).items()
            )
        ),
        "arc_sketch_program_heldout_source_count": len(arc_sketch_heldout_sources),
        "arc_visual_pipeline_heldout_attempt_count": len(
            arc_visual_pipeline_heldout_attempts
        ),
        "arc_visual_pipeline_heldout_accepted_attempt_count": sum(
            1
            for row in arc_visual_pipeline_heldout_attempts
            if safe_bool(row.get("accepted"))
        ),
        "arc_sketch_program_heldout_prior_row_count": arc_sketch_heldout_prior_row_count,
        "arc_sketch_program_heldout_comparison_count": len(
            arc_sketch_heldout_comparison
        ),
        "arc_sketch_program_heldout_added_train_consistent_tasks": sum(
            1
            for row in arc_sketch_heldout_comparison
            if row.get("probe_added_train_consistent_sketch")
        ),
        "arc_sketch_program_heldout_selected_tasks": sum(
            1 for row in arc_sketch_heldout_comparison if row.get("probe_selected_sketch")
        ),
        "arc_sketch_program_heldout_certified_task_delta": sum(
            safe_int(row.get("delta", {}).get("certified_tasks"))
            for row in arc_sketch_heldout_comparison
        ),
        "arc_sketch_program_heldout_newly_certified_tasks": sum(
            1 for row in arc_sketch_heldout_comparison if row.get("newly_certified")
        ),
        "arc_sketch_program_heldout_baseline_already_certified_tasks": sum(
            1
            for row in arc_sketch_heldout_comparison
            if row.get("baseline_already_certified")
        ),
        "arc_sketch_program_heldout_sources_by_family": {
            family: Counter(
                str(row.get("family", "")) for row in arc_sketch_heldout_sources
            ).get(family, 0)
            for family in sorted(
                {str(row.get("family", "")) for row in arc_sketch_heldout_candidates}
                | {str(row.get("family", "")) for row in arc_sketch_heldout_sources}
            )
        },
        "arc_sketch_program_heldout_sources_by_op": dict(
            sorted(
                Counter(
                    str(row.get("op_name", "")) for row in arc_sketch_heldout_sources
                ).items()
            )
        ),
        "arc_sketch_program_heldout_rejections_by_family": aggregate_arc_probe_rejections(
            arc_sketch_heldout_summaries
        ),
        "arc_visual_pipeline_proposal_count": len(arc_visual_pipeline_proposals),
        "arc_visual_pipeline_neural_proposal_count": len(
            arc_visual_pipeline_neural_proposals
        ),
        "arc_visual_pipeline_proposal_summary": arc_visual_pipeline_proposal_summary,
        "grammar_sketch_count": len(sketches),
        "grammar_sketch_example_count": len(sketch_examples),
        "grammar_sketch_proposal_count": len(sketch_proposals),
        "grammar_sketch_verification_count": len(sketch_verifications),
        "promoted_grammar_sketch_count": len(promoted_sketches),
        "grammar_sketch_memory": sketch_memory_summary,
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
            "arc_heldout_triage": str(args.arc_heldout_triage),
            "public_arc_dir": str(args.public_arc_dir),
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
            arc_sketch_probe_sources,
            arc_sketch_probe_summaries,
            arc_sketch_probe_comparison,
            arc_visual_pipeline_attempts,
            arc_sketch_heldout_sources,
            arc_sketch_heldout_summaries,
            arc_sketch_heldout_comparison,
            arc_visual_pipeline_heldout_attempts,
            arc_sketch_heldout_candidate_count,
            arc_structure_proposals,
            arc_visual_pipeline_proposals,
            arc_visual_pipeline_proposal_summary,
            sketch_memory_summary,
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
        "arc_sketch_probe_sources": arc_sketch_probe_sources,
        "arc_sketch_probe_prior_rows": arc_sketch_probe_prior_row_count,
        "arc_sketch_probe_summaries": arc_sketch_probe_summaries,
        "arc_sketch_probe_comparison": arc_sketch_probe_comparison,
        "arc_visual_pipeline_attempts": arc_visual_pipeline_attempts,
        "arc_sketch_heldout_sources": arc_sketch_heldout_sources,
        "arc_sketch_heldout_prior_rows": arc_sketch_heldout_prior_row_count,
        "arc_sketch_heldout_summaries": arc_sketch_heldout_summaries,
        "arc_sketch_heldout_comparison": arc_sketch_heldout_comparison,
        "arc_visual_pipeline_heldout_attempts": arc_visual_pipeline_heldout_attempts,
        "arc_visual_pipeline_proposals": arc_visual_pipeline_proposals,
        "arc_visual_pipeline_neural_proposals": arc_visual_pipeline_neural_proposals,
        "arc_visual_pipeline_proposal_summary": arc_visual_pipeline_proposal_summary,
        "sketches": sketches,
        "sketch_examples": sketch_examples,
        "sketch_proposals": sketch_proposals,
        "sketch_verifications": sketch_verifications,
        "promoted_sketches": promoted_sketches,
        "sketch_memory_summary": sketch_memory_summary,
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
        "--runtime",
        type=Path,
        default=Path("build/float/neural_grammar_discovery_runtime"),
    )
    parser.add_argument(
        "--sketch-memory",
        type=Path,
        default=None,
        help="C neural sketch proposer checkpoint path; defaults under --output-dir.",
    )
    parser.add_argument(
        "--resume-sketch-memory",
        action="store_true",
        help="Load --sketch-memory before training when the checkpoint exists.",
    )
    parser.add_argument(
        "--arc-agi",
        type=Path,
        default=Path("build/float/arc_agi"),
    )
    parser.add_argument(
        "--public-arc-dir",
        type=Path,
        default=Path(""),
    )
    parser.add_argument(
        "--arc-probe-sketches-per-task",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--arc-probe-compare-limit",
        type=int,
        default=64,
        help="Maximum promoted ARC support tasks to replay for baseline-vs-probed comparison; use -1 for no cap.",
    )
    parser.add_argument(
        "--arc-heldout-triage",
        type=Path,
        default=Path("build/float/hypothesis_library_public_triage/coverage_triage.jsonl"),
        help="Public ARC triage rows used to build held-out sketch-probe tasks.",
    )
    parser.add_argument(
        "--arc-heldout-limit",
        type=int,
        default=64,
        help="Maximum held-out ARC triage tasks to export; use -1 for no cap.",
    )
    parser.add_argument(
        "--arc-heldout-per-family",
        type=int,
        default=12,
        help="Maximum held-out ARC triage tasks per promoted family; use -1 for no cap.",
    )
    parser.add_argument(
        "--arc-heldout-compare-limit",
        type=int,
        default=64,
        help="Maximum held-out ARC tasks to replay for baseline-vs-probed comparison; use -1 for no cap.",
    )
    parser.add_argument(
        "--arc-heldout-op-hints-per-task",
        type=int,
        default=4,
        help=(
            "Number of pre-enumeration ARC sketch ops to propose per held-out "
            "candidate task; use -1 to allow all family ops."
        ),
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
        "wrote {} delta candidates, {} compiled deltas, {} promoted deltas, {} ARC sketch probes, {} ARC visual pipeline attempts, {} ARC probe comparisons, {} ARC visual pipeline proposals ({} neural), {} grammar sketches, {} sketch examples, {} sketch proposals, {} promoted sketches, sketch memory {} to {}".format(
            len(outputs["candidates"]),
            len(outputs["compiled_deltas"]),
            len(outputs["promoted"]),
            len(outputs["arc_sketch_probe_sources"]),
            len(outputs["arc_visual_pipeline_attempts"])
            + len(outputs["arc_visual_pipeline_heldout_attempts"]),
            len(outputs["arc_sketch_probe_comparison"]),
            len(outputs["arc_visual_pipeline_proposals"]),
            len(outputs["arc_visual_pipeline_neural_proposals"]),
            len(outputs["sketches"]),
            len(outputs["sketch_examples"]),
            len(outputs["sketch_proposals"]),
            len(outputs["promoted_sketches"]),
            outputs["sketch_memory_summary"].get("checkpoint_path", ""),
            args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
