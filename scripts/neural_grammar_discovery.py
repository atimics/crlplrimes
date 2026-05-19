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
ARC_SKETCH_PROBE_COMPARISON_SCHEMA = (
    "crlp.neural_grammar_discovery.arc_sketch_program_probe_comparison.v1"
)

ARC_SKETCH_TEMPLATE_ID = 1036
ARC_SKETCH_TEMPLATE_NAME = "arc_sketch_program"
ARC_SKETCH_TEMPLATE_PROBE_SCHEMA = "crlp.arc_neural_discovery.template_probe.v1"
ARC_SKETCH_SLOT_PROBE_SCHEMA = "crlp.arc_neural_discovery.slot_probe.v1"
ARC_SKETCH_PRIOR_FIELDS = [
    "schema_version",
    "template_id",
    "template_name",
    "library_id",
    "slot_name",
    "slot_type",
    "slot_value",
    "slot_integer",
    "proposal_count",
    "certified_count",
    "support_tasks",
    "covered_events",
    "utility_mean",
]
ARC_SKETCH_SLOT_SPECS = [
    ("sketch.selector", "integer"),
    ("sketch.op", "integer"),
    ("sketch.background", "color"),
    ("sketch.color", "color"),
    ("sketch.output_color", "color"),
    ("sketch.x_anchor", "integer"),
    ("sketch.y_anchor", "integer"),
    ("sketch.x_offset", "offset"),
    ("sketch.y_offset", "offset"),
    ("sketch.width", "integer"),
    ("sketch.height", "integer"),
    ("sketch.period", "integer"),
]
ARC_SKETCH_SELECTOR_GRID_PANEL = 0
ARC_SKETCH_SELECTOR_FOREGROUND_BBOX_PANEL = 1
ARC_SKETCH_SELECTOR_COLOR_BBOX = 2
ARC_SKETCH_SELECTOR_LATTICE = 3
ARC_SKETCH_OP_CROP = 0
ARC_SKETCH_OP_TILE_REPEAT = 1
ARC_PUBLIC_SPLIT_DIRS = {
    "public_training": "training",
    "training": "training",
    "train": "training",
    "public_evaluation": "evaluation",
    "evaluation": "evaluation",
    "eval": "evaluation",
}


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


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        fp.write("\t".join(fields))
        fp.write("\n")
        for row in rows:
            fp.write("\t".join(str(row.get(field, "")) for field in fields))
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


def grid_shape(grid: list[list[int]]) -> tuple[int, int]:
    if not grid:
        return 0, 0
    return len(grid[0]), len(grid)


def grid_equal(left: list[list[int]], right: list[list[int]]) -> bool:
    return left == right


def dominant_color(grid: list[list[int]]) -> int:
    counts: Counter[int] = Counter()
    for row in grid:
        for value in row:
            counts[int(value)] += 1
    if not counts:
        return 0
    return counts.most_common(1)[0][0]


def nonzero_colors(grid: list[list[int]]) -> list[int]:
    return sorted({int(value) for row in grid for value in row if int(value) != 0})


def crop_grid(
    grid: list[list[int]],
    origin_x: int,
    origin_y: int,
    width: int,
    height: int,
) -> list[list[int]] | None:
    input_width, input_height = grid_shape(grid)
    if (
        width <= 0
        or height <= 0
        or origin_x < 0
        or origin_y < 0
        or origin_x + width > input_width
        or origin_y + height > input_height
    ):
        return None
    return [row[origin_x : origin_x + width] for row in grid[origin_y : origin_y + height]]


def color_bbox(grid: list[list[int]], color: int) -> tuple[int, int, int, int] | None:
    xs: list[int] = []
    ys: list[int] = []
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            if int(value) == color:
                xs.append(x)
                ys.append(y)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def non_background_bbox(
    grid: list[list[int]],
    background: int,
) -> tuple[int, int, int, int] | None:
    xs: list[int] = []
    ys: list[int] = []
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            if int(value) != background:
                xs.append(x)
                ys.append(y)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def subgrid_origins(
    input_grid: list[list[int]],
    output_grid: list[list[int]],
) -> list[tuple[int, int]]:
    input_width, input_height = grid_shape(input_grid)
    output_width, output_height = grid_shape(output_grid)
    origins: list[tuple[int, int]] = []
    if (
        output_width <= 0
        or output_height <= 0
        or output_width > input_width
        or output_height > input_height
    ):
        return origins
    for y in range(input_height - output_height + 1):
        for x in range(input_width - output_width + 1):
            if crop_grid(input_grid, x, y, output_width, output_height) == output_grid:
                origins.append((x, y))
    return origins


def panel_origin(input_extent: int, panel_extent: int, anchor: int, offset: int) -> int | None:
    if panel_extent <= 0 or panel_extent > input_extent or anchor < 0 or anchor > 2:
        return None
    span = input_extent - panel_extent
    base = 0
    if anchor == 1:
        base = span // 2
    elif anchor == 2:
        base = span
    origin = base + offset
    if origin < 0 or origin + panel_extent > input_extent:
        return None
    return origin


def bbox_panel_origin(
    input_grid: list[list[int]],
    background: int,
    panel_width: int,
    panel_height: int,
    x_anchor: int,
    y_anchor: int,
    x_offset: int,
    y_offset: int,
) -> tuple[int, int] | None:
    bbox = non_background_bbox(input_grid, background)
    if bbox is None:
        return None
    min_x, min_y, max_x, max_y = bbox
    bbox_width = max_x - min_x + 1
    bbox_height = max_y - min_y + 1
    if panel_width > bbox_width or panel_height > bbox_height:
        return None
    x_span = bbox_width - panel_width
    y_span = bbox_height - panel_height
    x_base = min_x + (x_span // 2 if x_anchor == 1 else x_span if x_anchor == 2 else 0)
    y_base = min_y + (y_span // 2 if y_anchor == 1 else y_span if y_anchor == 2 else 0)
    origin_x = x_base + x_offset
    origin_y = y_base + y_offset
    input_width, input_height = grid_shape(input_grid)
    if (
        origin_x < 0
        or origin_y < 0
        or origin_x + panel_width > input_width
        or origin_y + panel_height > input_height
    ):
        return None
    return origin_x, origin_y


def sketch_slot_values(**values: int) -> dict[str, int]:
    defaults = {
        "sketch.selector": 0,
        "sketch.op": 0,
        "sketch.background": 0,
        "sketch.color": 0,
        "sketch.output_color": 0,
        "sketch.x_anchor": 0,
        "sketch.y_anchor": 0,
        "sketch.x_offset": 0,
        "sketch.y_offset": 0,
        "sketch.width": 0,
        "sketch.height": 0,
        "sketch.period": 0,
    }
    defaults.update(values)
    return defaults


def apply_crop_sketch(
    input_grid: list[list[int]],
    values: dict[str, int],
) -> list[list[int]] | None:
    selector = values["sketch.selector"]
    width = values["sketch.width"]
    height = values["sketch.height"]
    if selector == ARC_SKETCH_SELECTOR_GRID_PANEL:
        input_width, input_height = grid_shape(input_grid)
        origin_x = panel_origin(
            input_width, width, values["sketch.x_anchor"], values["sketch.x_offset"]
        )
        origin_y = panel_origin(
            input_height, height, values["sketch.y_anchor"], values["sketch.y_offset"]
        )
        if origin_x is None or origin_y is None:
            return None
        return crop_grid(input_grid, origin_x, origin_y, width, height)
    if selector == ARC_SKETCH_SELECTOR_FOREGROUND_BBOX_PANEL:
        origin = bbox_panel_origin(
            input_grid,
            values["sketch.background"],
            width,
            height,
            values["sketch.x_anchor"],
            values["sketch.y_anchor"],
            values["sketch.x_offset"],
            values["sketch.y_offset"],
        )
        if origin is None:
            return None
        return crop_grid(input_grid, origin[0], origin[1], width, height)
    if selector == ARC_SKETCH_SELECTOR_COLOR_BBOX:
        bbox = color_bbox(input_grid, values["sketch.color"])
        if bbox is None:
            return None
        min_x, min_y, max_x, max_y = bbox
        return crop_grid(input_grid, min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
    return None


def arc_train_pairs(task: dict[str, Any]) -> list[tuple[list[list[int]], list[list[int]]]]:
    pairs: list[tuple[list[list[int]], list[list[int]]]] = []
    for pair in task.get("train", []) or []:
        if not isinstance(pair, dict):
            continue
        input_grid = pair.get("input")
        output_grid = pair.get("output")
        if isinstance(input_grid, list) and isinstance(output_grid, list):
            pairs.append((input_grid, output_grid))
    return pairs


def arc_task_path(public_arc_dir: Path, split: str, task_id: str) -> Path:
    split_dir = ARC_PUBLIC_SPLIT_DIRS.get(split, split)
    return public_arc_dir / split_dir / f"{task_id}.json"


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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
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
    write_runtime_candidate_rows(candidate_rows, candidates, groups)
    write_runtime_sketch_tables(sketch_rows, sketch_example_rows, sketches, sketch_examples)
    subprocess.run(
        [
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
        ],
        check=True,
    )
    return (
        parse_json_rows(compiled_output),
        parse_json_rows(verification_output),
        parse_json_rows(neural_proposals_output),
    )


def arc_sketch_template_probe_row(
    library_id: str,
    proposal_count: int,
    support_tasks: int,
    utility_mean: float,
) -> dict[str, Any]:
    return {
        "schema_version": ARC_SKETCH_TEMPLATE_PROBE_SCHEMA,
        "template_id": ARC_SKETCH_TEMPLATE_ID,
        "template_name": ARC_SKETCH_TEMPLATE_NAME,
        "library_id": library_id,
        "slot_name": "__template__",
        "slot_type": "template",
        "slot_value": "0.00000000",
        "slot_integer": 0,
        "proposal_count": max(1, proposal_count),
        "certified_count": 0,
        "support_tasks": support_tasks,
        "covered_events": support_tasks,
        "utility_mean": f"{utility_mean:.8f}",
    }


def arc_sketch_slot_probe_rows(
    library_id: str,
    values: dict[str, int],
    proposal_count: int,
    support_tasks: int,
    utility_mean: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slot_name, slot_type in ARC_SKETCH_SLOT_SPECS:
        slot_integer = int(values[slot_name])
        rows.append(
            {
                "schema_version": ARC_SKETCH_SLOT_PROBE_SCHEMA,
                "template_id": ARC_SKETCH_TEMPLATE_ID,
                "template_name": ARC_SKETCH_TEMPLATE_NAME,
                "library_id": library_id,
                "slot_name": slot_name,
                "slot_type": slot_type,
                "slot_value": f"{float(slot_integer):.8f}",
                "slot_integer": slot_integer,
                "proposal_count": max(1, proposal_count),
                "certified_count": 0,
                "support_tasks": support_tasks,
                "covered_events": support_tasks,
                "utility_mean": f"{utility_mean:.8f}",
            }
        )
    return rows


def arc_sketch_values_train_consistent(
    pairs: list[tuple[list[list[int]], list[list[int]]]],
    values: dict[str, int],
) -> bool:
    if not pairs:
        return False
    if values["sketch.op"] != ARC_SKETCH_OP_CROP:
        return True
    for input_grid, output_grid in pairs:
        if not grid_equal(apply_crop_sketch(input_grid, values) or [], output_grid):
            return False
    return True


def synthesize_arc_panel_sketches(
    pairs: list[tuple[list[list[int]], list[list[int]]]],
    limit: int,
) -> list[dict[str, int]]:
    if not pairs:
        return []
    first_input, first_output = pairs[0]
    output_width, output_height = grid_shape(first_output)
    if output_width == 0 or output_height == 0:
        return []
    background = dominant_color(first_input)
    values: list[dict[str, int]] = []
    for origin_x, origin_y in subgrid_origins(first_input, first_output):
        for selector in [
            ARC_SKETCH_SELECTOR_GRID_PANEL,
            ARC_SKETCH_SELECTOR_FOREGROUND_BBOX_PANEL,
        ]:
            if selector == ARC_SKETCH_SELECTOR_GRID_PANEL:
                base_min_x = 0
                base_min_y = 0
                base_width, base_height = grid_shape(first_input)
            else:
                bbox = non_background_bbox(first_input, background)
                if bbox is None:
                    continue
                base_min_x, base_min_y, max_x, max_y = bbox
                base_width = max_x - base_min_x + 1
                base_height = max_y - base_min_y + 1
            if output_width > base_width or output_height > base_height:
                continue
            for x_anchor in range(3):
                x_span = base_width - output_width
                x_base = base_min_x + (
                    x_span // 2 if x_anchor == 1 else x_span if x_anchor == 2 else 0
                )
                x_offset = origin_x - x_base
                for y_anchor in range(3):
                    y_span = base_height - output_height
                    y_base = base_min_y + (
                        y_span // 2
                        if y_anchor == 1
                        else y_span
                        if y_anchor == 2
                        else 0
                    )
                    y_offset = origin_y - y_base
                    row = sketch_slot_values(
                        **{
                            "sketch.selector": selector,
                            "sketch.op": ARC_SKETCH_OP_CROP,
                            "sketch.background": background,
                            "sketch.x_anchor": x_anchor,
                            "sketch.y_anchor": y_anchor,
                            "sketch.x_offset": x_offset,
                            "sketch.y_offset": y_offset,
                            "sketch.width": output_width,
                            "sketch.height": output_height,
                        }
                    )
                    if arc_sketch_values_train_consistent(pairs, row):
                        values.append(row)
                        if len(values) >= limit:
                            return values
    return values


def synthesize_arc_color_bbox_sketches(
    pairs: list[tuple[list[list[int]], list[list[int]]]],
    limit: int,
) -> list[dict[str, int]]:
    if not pairs:
        return []
    first_input, _first_output = pairs[0]
    background = dominant_color(first_input)
    values: list[dict[str, int]] = []
    for color in nonzero_colors(first_input):
        row = sketch_slot_values(
            **{
                "sketch.selector": ARC_SKETCH_SELECTOR_COLOR_BBOX,
                "sketch.op": ARC_SKETCH_OP_CROP,
                "sketch.background": background,
                "sketch.color": color,
            }
        )
        if arc_sketch_values_train_consistent(pairs, row):
            values.append(row)
            if len(values) >= limit:
                return values
    return values


def synthesize_arc_lattice_sketches(
    pairs: list[tuple[list[list[int]], list[list[int]]]],
    limit: int,
) -> list[dict[str, int]]:
    if not pairs:
        return []
    backgrounds = sorted({dominant_color(input_grid) for input_grid, _output in pairs})
    if 0 not in backgrounds:
        backgrounds.insert(0, 0)
    values: list[dict[str, int]] = []
    for background in backgrounds[:3]:
        for period in [2, 3]:
            values.append(
                sketch_slot_values(
                    **{
                        "sketch.selector": ARC_SKETCH_SELECTOR_LATTICE,
                        "sketch.op": ARC_SKETCH_OP_TILE_REPEAT,
                        "sketch.background": background,
                        "sketch.period": period,
                    }
                )
            )
            if len(values) >= limit:
                return values
    return values


def synthesize_arc_sketch_values(
    family: str,
    task: dict[str, Any],
    limit: int,
) -> list[dict[str, int]]:
    pairs = arc_train_pairs(task)
    if family == "arc.crop_object_selection_family":
        return synthesize_arc_color_bbox_sketches(pairs, limit)
    if family == "arc.separator_panel_extraction_family":
        return synthesize_arc_panel_sketches(pairs, limit)
    if family in {
        "arc.patch_lattice_template_family",
        "arc.tiling_canvas_growth_family",
    }:
        return synthesize_arc_lattice_sketches(pairs, limit)
    return []


def arc_probe_family_source_rows(
    groups: list[CandidateGroup],
    promoted: list[dict[str, Any]],
    public_arc_dir: Path,
    sketches_per_task: int,
) -> list[dict[str, Any]]:
    if str(public_arc_dir) in {"", "."} or not public_arc_dir.is_dir():
        return []
    promoted_families = {
        str(row.get("family", ""))
        for row in promoted
        if row.get("domain") == "arc_agi"
    }
    source_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        if group.domain != "arc_agi" or group.family not in promoted_families:
            continue
        support = len(group.rows)
        utility = mean([safe_float(row.get("slot_probe_scored_rows")) for row in group.rows])
        proposal_count = max(
            1,
            min(64, int(round(mean([safe_float(row.get("slot_probe_seeded_proposals")) for row in group.rows]))) or support),
        )
        for row in group.rows:
            task_id = str(row.get("task_id", ""))
            split = str(row.get("split", ""))
            task_path = arc_task_path(public_arc_dir, split, task_id)
            if not task_id or not task_path.is_file():
                continue
            try:
                task = json.loads(task_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for index, values in enumerate(
                synthesize_arc_sketch_values(group.family, task, sketches_per_task),
                start=1,
            ):
                signature = stable_id(group.family, task_id, values)
                if signature in seen:
                    continue
                seen.add(signature)
                library_id = "ngd:{}:{}:{}".format(
                    group.family.replace(".", "_"),
                    task_id,
                    signature,
                )
                source_rows.append(
                    {
                        "schema": ARC_SKETCH_PROBE_SCHEMA,
                        "report_version": 1,
                        "library_id": library_id,
                        "template_id": ARC_SKETCH_TEMPLATE_ID,
                        "template_name": ARC_SKETCH_TEMPLATE_NAME,
                        "family": group.family,
                        "candidate_group_size": support,
                        "task_id": task_id,
                        "split": split,
                        "task_path": str(task_path),
                        "sketch_index": index,
                        "slot_values": values,
                        "proposal_count": proposal_count,
                        "support_tasks": support,
                        "utility_mean": utility,
                        "compiler": "arc_sketch_program_probe_compiler",
                        "promotion_path": "tsv_probe_loaded_by_arc_agi_verifier",
                    }
                )
    return source_rows


def arc_probe_prior_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    families = sorted({str(row.get("family", "")) for row in source_rows})
    if source_rows:
        rows.append(
            arc_sketch_template_probe_row(
                "ngd:arc_sketch_program:template",
                proposal_count=len(source_rows),
                support_tasks=len({str(row.get("task_id", "")) for row in source_rows}),
                utility_mean=mean([safe_float(row.get("utility_mean")) for row in source_rows]),
            )
        )
    for family in families:
        family_rows = [row for row in source_rows if row.get("family") == family]
        rows.append(
            arc_sketch_template_probe_row(
                f"ngd:{family.replace('.', '_')}:template",
                proposal_count=len(family_rows),
                support_tasks=len({str(row.get("task_id", "")) for row in family_rows}),
                utility_mean=mean([safe_float(row.get("utility_mean")) for row in family_rows]),
            )
        )
    for row in source_rows:
        rows.extend(
            arc_sketch_slot_probe_rows(
                str(row["library_id"]),
                row["slot_values"],
                safe_int(row.get("proposal_count")),
                safe_int(row.get("support_tasks")),
                safe_float(row.get("utility_mean")),
            )
        )
    return rows


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
) -> list[dict[str, Any]]:
    if not source_rows or not arc_agi.is_file() or not prior_path.is_file() or limit == 0:
        return []
    selected_rows: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    rows_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(source_rows, key=lambda item: (str(item.get("task_id", "")), str(item.get("library_id", "")))):
        rows_by_family[str(row.get("family", ""))].append(row)
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
    log_dir = output_dir / "arc_sketch_program_probe_logs"
    for row in selected_rows:
        task_path = Path(str(row["task_path"]))
        task_id = str(row["task_id"])
        task_dir = task_path.parent
        baseline_log = log_dir / f"{task_id}.baseline.jsonl"
        probed_log = log_dir / f"{task_id}.probed.jsonl"
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
            str(prior_path),
            *flags,
        ]
        baseline_exit = run_arc_log(base_command, baseline_log)
        probed_exit = run_arc_log(probed_command, probed_log)
        baseline_metrics = arc_log_metrics(baseline_log)
        probed_metrics = arc_log_metrics(probed_log)
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
                "prior_path": str(prior_path),
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
            }
        )
    return rows


def arc_sketch_probe_markdown(rows: list[dict[str, Any]], source_count: int) -> str:
    lines = [
        "# ARC Sketch Program Probe Comparison",
        "",
        f"- Probe source sketches: {source_count}",
        f"- Compared tasks: {len(rows)}",
        f"- Tasks with extra sketch train-consistent pre-cap proposals: {sum(1 for row in rows if row.get('probe_added_train_consistent_sketch'))}",
        f"- Tasks selecting `arc_sketch_program`: {sum(1 for row in rows if row.get('probe_selected_sketch'))}",
        f"- Certified task delta: {sum(safe_int(row.get('delta', {}).get('certified_tasks')) for row in rows)}",
        "",
        "| Task | Family | Base Sketch TC | Probed Sketch TC | Base Selected | Probed Selected | Certified Delta |",
        "| --- | --- | ---: | ---: | --- | --- | ---: |",
    ]
    for row in rows:
        base = row.get("baseline", {})
        probed = row.get("probed", {})
        delta = row.get("delta", {})
        lines.append(
            "| `{}` | `{}` | {} | {} | `{}` | `{}` | {} |".format(
                row.get("task_id", ""),
                row.get("family", ""),
                safe_int(base.get("sketch_train_consistent_pre_cap")),
                safe_int(probed.get("sketch_train_consistent_pre_cap")),
                base.get("selected_template", ""),
                probed.get("selected_template", ""),
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
    arc_sketch_probe_comparison: list[dict[str, Any]],
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
        f"- ARC sketch program compared tasks: {len(arc_sketch_probe_comparison)}",
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

    if arc_sketch_probe_sources:
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
    compiled_deltas, verifications, neural_sketch_proposals = run_c_runtime(
        args,
        candidates,
        groups,
        sketches,
        sketch_examples,
    )
    promoted = promoted_rows(candidates, verifications)
    arc_sketch_probe_sources = arc_probe_family_source_rows(
        groups,
        promoted,
        args.public_arc_dir,
        args.arc_probe_sketches_per_task,
    )
    arc_sketch_probe_prior_rows = arc_probe_prior_rows(arc_sketch_probe_sources)
    sketch_proposals = build_sketch_proposals(
        sketches,
        sketch_examples,
        neural_sketch_proposals,
    )
    sketch_verifications, promoted_sketches = build_sketch_verifications(sketches, sketch_examples)

    write_jsonl(args.output_dir / "grammar_delta_candidates.jsonl", candidates)
    write_jsonl(args.output_dir / "compiled_grammar_deltas.jsonl", compiled_deltas)
    write_jsonl(args.output_dir / "grammar_delta_verification.jsonl", verifications)
    write_jsonl(args.output_dir / "promoted_grammar_deltas.jsonl", promoted)
    write_jsonl(
        args.output_dir / "arc_sketch_program_probe_sources.jsonl",
        arc_sketch_probe_sources,
    )
    write_tsv(
        args.output_dir / "arc_sketch_program_probe_prior.tsv",
        arc_sketch_probe_prior_rows,
        ARC_SKETCH_PRIOR_FIELDS,
    )
    arc_sketch_probe_comparison = compare_arc_sketch_probes(
        args.arc_agi,
        args.output_dir / "arc_sketch_program_probe_prior.tsv",
        arc_sketch_probe_sources,
        args.output_dir,
        args.arc_probe_compare_limit,
    )
    write_jsonl(
        args.output_dir / "arc_sketch_program_probe_comparison.jsonl",
        arc_sketch_probe_comparison,
    )
    (args.output_dir / "arc_sketch_program_probe_comparison.md").write_text(
        arc_sketch_probe_markdown(
            arc_sketch_probe_comparison,
            len(arc_sketch_probe_sources),
        ),
        encoding="utf-8",
    )
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
        "arc_sketch_program_probe_source_count": len(arc_sketch_probe_sources),
        "arc_sketch_program_probe_prior_row_count": len(arc_sketch_probe_prior_rows),
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
            arc_sketch_probe_comparison,
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
        "arc_sketch_probe_prior_rows": arc_sketch_probe_prior_rows,
        "arc_sketch_probe_comparison": arc_sketch_probe_comparison,
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
        "--runtime",
        type=Path,
        default=Path("build/float/neural_grammar_discovery_runtime"),
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
        "wrote {} delta candidates, {} compiled deltas, {} promoted deltas, {} ARC sketch probes, {} ARC probe comparisons, {} grammar sketches, {} sketch examples, {} sketch proposals, {} promoted sketches to {}".format(
            len(outputs["candidates"]),
            len(outputs["compiled_deltas"]),
            len(outputs["promoted"]),
            len(outputs["arc_sketch_probe_sources"]),
            len(outputs["arc_sketch_probe_comparison"]),
            len(outputs["sketches"]),
            len(outputs["sketch_examples"]),
            len(outputs["sketch_proposals"]),
            len(outputs["promoted_sketches"]),
            args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
