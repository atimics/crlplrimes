#!/usr/bin/env python3
"""Raw-grid neural discovery loop for ARC hypothesis mining.

This is intentionally a discovery controller, not another handcrafted ARC
feature extractor. Model input is raw train input/output grid cells plus
mechanical validity masks. Diagnostic rows provide labels; the symbolic
verifier remains the only authority for certification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ARC_GRID_LIMIT = 30
HASH_PROBES = 4
TEMPLATE_PROBE_SCHEMA = "crlp.arc_neural_discovery.template_probe.v1"
SLOT_PROBE_SCHEMA = "crlp.arc_neural_discovery.slot_probe.v1"
PROMOTION_FLAGS = [
    "--program-budget-grid",
    "--loader-summary-only",
    "--induction",
]
STRICT_EVAL_FLAGS = [
    "--program-budget-tight-grid",
    "--loader-summary-only",
    "--induction",
    "--hypothesis-top-k",
    "1",
    "--disable-hypothesis-coverage-selection",
    "--disable-hypothesis-structural-score",
    "--disable-hypothesis-train-filter",
]
NO_PRIOR_FLAGS = [
    "--disable-hypothesis-slot-prior",
    "--disable-hypothesis-neural-prior",
    "--disable-hypothesis-cold-start",
]
STRESS_NO_PRIOR_FLAGS = [
    *STRICT_EVAL_FLAGS,
    *NO_PRIOR_FLAGS,
    "--disable-hypothesis-train-consistency-prior",
]
PRIOR_FIELDS = [
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
TEMPLATE_IDS = {
    "select_then_translate": 1024,
    "move_to_anchor": 1025,
    "recolor_by_reference": 1026,
    "connect_aligned": 1027,
    "fill_enclosed": 1028,
    "object_extract": 1029,
    "extend_aligned_sequence": 1030,
    "tile_motif_repeat": 1031,
    "projection_panel_selection": 1032,
    "projection_bbox_panel_selection": 1033,
    "masked_motif_lift": 1034,
    "same_grid_color_map": 1035,
}
SELECTOR_UNIQUE_COLOR = 1
RENDER_OVERLAY = 0
RENDER_ONLY = 1
TEMPLATE_SLOT_SPECS = {
    "same_grid_color_map": [(f"map.{color}", "color") for color in range(10)],
    "select_then_translate": [
        ("object_scene.background", "color"),
        ("select.selector_kind", "selector_kind"),
        ("select.color", "color"),
        ("select.reference_color_used", "boolean"),
        ("select.reference_color", "reference_color"),
        ("translate.dx", "offset"),
        ("translate.dy", "offset"),
        ("render.background", "color"),
        ("render.output_color", "color"),
    ],
    "connect_aligned": [
        ("object_scene.background", "color"),
        ("connect.color", "color"),
        ("connect.output_color", "color"),
        ("connect.mode", "integer"),
        ("connect.render_mode", "render_mode"),
    ],
    "fill_enclosed": [
        ("object_scene.background", "color"),
        ("fill.border_color", "color"),
        ("fill.output_color", "color"),
    ],
    "tile_motif_repeat": [
        ("lattice.background", "color"),
        ("repeat.period", "integer"),
    ],
    "masked_motif_lift": [("mask.background", "color")],
    "projection_panel_selection": [
        ("panel.background", "color"),
        ("panel.x_anchor", "integer"),
        ("panel.y_anchor", "integer"),
        ("panel.x_offset", "offset"),
        ("panel.y_offset", "offset"),
        ("panel.width", "integer"),
        ("panel.height", "integer"),
    ],
    "projection_bbox_panel_selection": [
        ("panel.background", "color"),
        ("panel.x_anchor", "integer"),
        ("panel.y_anchor", "integer"),
        ("panel.x_offset", "offset"),
        ("panel.y_offset", "offset"),
        ("panel.width", "integer"),
        ("panel.height", "integer"),
    ],
}


@dataclass(frozen=True)
class Sample:
    key: str
    task_id: str
    split: str
    mode: str
    task_path: Path
    label: str
    certified: bool
    selected_templates: tuple[str, ...]
    train_consistent_templates: tuple[str, ...]
    certified_templates: tuple[str, ...]
    proposals_generated: int
    proposals_train_consistent_pre_cap: int
    tokens: tuple[tuple[str, float], ...]


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
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
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


def write_prior_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        fp.write("\t".join(PRIOR_FIELDS) + "\n")
        for row in rows:
            fp.write("\t".join(str(row[field]) for field in PRIOR_FIELDS))
            fp.write("\n")


def clear_matching_files(path: Path, pattern: str) -> None:
    if not path.is_dir():
        return
    for child in path.glob(pattern):
        if child.is_file():
            child.unlink()


def public_split_dirs(path: Path) -> dict[str, Path]:
    training = path / "training"
    evaluation = path / "evaluation"
    if training.is_dir() and evaluation.is_dir():
        return {"public_training": training, "public_evaluation": evaluation}
    train = path / "train"
    eval_dir = path / "eval"
    if train.is_dir() and eval_dir.is_dir():
        return {"public_training": train, "public_evaluation": eval_dir}
    raise RuntimeError(f"{path} must contain training/evaluation or train/eval directories")


def as_tuple_strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def first_or_none(values: tuple[str, ...]) -> str | None:
    return values[0] if values else None


def diagnostic_label(row: dict[str, Any]) -> str:
    certified_templates = as_tuple_strings(row.get("certified_templates"))
    train_consistent_templates = as_tuple_strings(row.get("train_consistent_templates"))
    selected_templates = as_tuple_strings(row.get("selected_templates"))
    if bool(row.get("certified", False)):
        return f"certified:{first_or_none(certified_templates) or 'unknown'}"
    if train_consistent_templates:
        return f"pressure:{first_or_none(train_consistent_templates)}"
    if selected_templates:
        return f"rejected:{first_or_none(selected_templates)}"
    if safe_int(row.get("proposals_generated")) > 0:
        return "rejected:unselected"
    return "uncovered:no_proposal"


def task_train_pairs(task: dict[str, Any], max_pairs: int) -> list[tuple[list[list[int]], list[list[int]]]]:
    pairs: list[tuple[list[list[int]], list[list[int]]]] = []
    for pair in task.get("train", []) or []:
        if len(pairs) >= max_pairs:
            break
        input_grid = pair.get("input")
        output_grid = pair.get("output")
        if isinstance(input_grid, list) and isinstance(output_grid, list):
            pairs.append((input_grid, output_grid))
    return pairs


def add_grid_tokens(
    tokens: list[tuple[str, float]],
    pair_index: int,
    stream: str,
    grid: list[list[int]],
) -> None:
    height = min(len(grid), ARC_GRID_LIMIT)
    width = min(len(grid[0]) if height else 0, ARC_GRID_LIMIT)
    for y in range(height):
        row = grid[y]
        for x in range(min(len(row), width)):
            color = int(row[x])
            tokens.append((f"p{pair_index}:s{stream}:cell:{x}:{y}:c{color}", 1.0))
            tokens.append((f"p{pair_index}:s{stream}:valid:{x}:{y}", 0.25))


def task_raw_tokens(task: dict[str, Any], max_pairs: int) -> tuple[tuple[str, float], ...]:
    tokens: list[tuple[str, float]] = []
    for pair_index, (input_grid, output_grid) in enumerate(task_train_pairs(task, max_pairs)):
        add_grid_tokens(tokens, pair_index, "in", input_grid)
        add_grid_tokens(tokens, pair_index, "out", output_grid)
    return tuple(tokens)


def read_task(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def grid_shape(grid: list[list[int]]) -> tuple[int, int]:
    height = len(grid)
    width = len(grid[0]) if height else 0
    return width, height


def grid_counter(grid: list[list[int]]) -> Counter[int]:
    counter: Counter[int] = Counter()
    for row in grid:
        counter.update(int(value) for value in row)
    return counter


def most_common(counter: Counter[int], default: int = 0) -> int:
    if not counter:
        return default
    return counter.most_common(1)[0][0]


def task_background(task: dict[str, Any]) -> int:
    counter: Counter[int] = Counter()
    for input_grid, _output_grid in task_train_pairs(task, max_pairs=4):
        counter.update(grid_counter(input_grid))
    return most_common(counter, 0)


def changed_color_counters(task: dict[str, Any]) -> tuple[Counter[int], Counter[int]]:
    input_counter: Counter[int] = Counter()
    output_counter: Counter[int] = Counter()
    for input_grid, output_grid in task_train_pairs(task, max_pairs=4):
        in_w, in_h = grid_shape(input_grid)
        out_w, out_h = grid_shape(output_grid)
        if in_w != out_w or in_h != out_h:
            continue
        for y in range(in_h):
            for x in range(in_w):
                in_color = int(input_grid[y][x])
                out_color = int(output_grid[y][x])
                if in_color != out_color:
                    input_counter[in_color] += 1
                    output_counter[out_color] += 1
    return input_counter, output_counter


def non_background_counter(task: dict[str, Any], stream: str, background: int) -> Counter[int]:
    counter: Counter[int] = Counter()
    for input_grid, output_grid in task_train_pairs(task, max_pairs=4):
        grid = output_grid if stream == "out" else input_grid
        for color, count in grid_counter(grid).items():
            if color != background:
                counter[color] += count
    return counter


def without_color(counter: Counter[int], color: int) -> Counter[int]:
    out = Counter(counter)
    out.pop(color, None)
    return out


def components(grid: list[list[int]], background: int) -> list[dict[str, Any]]:
    width, height = grid_shape(grid)
    seen: set[tuple[int, int]] = set()
    out: list[dict[str, Any]] = []
    for y in range(height):
        for x in range(width):
            color = int(grid[y][x])
            if color == background or (x, y) in seen:
                continue
            stack = [(x, y)]
            seen.add((x, y))
            cells: list[tuple[int, int]] = []
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if (
                        nx < 0
                        or ny < 0
                        or nx >= width
                        or ny >= height
                        or (nx, ny) in seen
                        or int(grid[ny][nx]) != color
                    ):
                        continue
                    seen.add((nx, ny))
                    stack.append((nx, ny))
            min_x = min(cell[0] for cell in cells)
            min_y = min(cell[1] for cell in cells)
            max_x = max(cell[0] for cell in cells)
            max_y = max(cell[1] for cell in cells)
            shape = tuple(sorted((cx - min_x, cy - min_y) for cx, cy in cells))
            out.append(
                {
                    "color": color,
                    "min_x": min_x,
                    "min_y": min_y,
                    "max_x": max_x,
                    "max_y": max_y,
                    "shape": shape,
                    "area": len(cells),
                }
            )
    return out


def find_subgrid(input_grid: list[list[int]], output_grid: list[list[int]]) -> tuple[int, int] | None:
    in_w, in_h = grid_shape(input_grid)
    out_w, out_h = grid_shape(output_grid)
    if out_w == 0 or out_h == 0 or out_w > in_w or out_h > in_h:
        return None
    for y0 in range(in_h - out_h + 1):
        for x0 in range(in_w - out_w + 1):
            ok = True
            for y in range(out_h):
                if input_grid[y0 + y][x0 : x0 + out_w] != output_grid[y]:
                    ok = False
                    break
            if ok:
                return x0, y0
    return None


def slot_probe_row(
    candidate: dict[str, Any],
    library_id: str,
    slot_name: str,
    slot_type: str,
    slot_integer: int,
    proposal_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": SLOT_PROBE_SCHEMA,
        "template_id": safe_int(candidate["template_id"]),
        "template_name": str(candidate["template_name"]),
        "library_id": library_id,
        "slot_name": slot_name,
        "slot_type": slot_type,
        "slot_value": f"{float(slot_integer):.8f}",
        "slot_integer": int(slot_integer),
        "proposal_count": max(1, proposal_count),
        "certified_count": 0,
        "support_tasks": safe_int(candidate.get("support_tasks")),
        "covered_events": safe_int(candidate.get("covered_events")),
        "utility_mean": f"{safe_float(candidate.get('neural_score')):.8f}",
    }


def full_slot_probe_rows(
    candidate: dict[str, Any],
    library_suffix: str,
    values: dict[str, int],
    proposal_count: int,
) -> list[dict[str, Any]]:
    template_name = str(candidate.get("template_name", ""))
    specs = TEMPLATE_SLOT_SPECS.get(template_name, [])
    if not specs:
        return []
    if any(slot_name not in values for slot_name, _slot_type in specs):
        return []
    library_id = f"{candidate['library_id']}:{library_suffix}"
    return [
        slot_probe_row(
            candidate,
            library_id,
            slot_name,
            slot_type,
            values[slot_name],
            proposal_count,
        )
        for slot_name, slot_type in specs
    ]


def synthesize_slot_value_sets(
    template_name: str,
    task: dict[str, Any],
    limit: int = 3,
) -> list[dict[str, int]]:
    background = task_background(task)
    changed_input, changed_output = changed_color_counters(task)
    foreground_input = non_background_counter(task, "in", background)
    foreground_output = non_background_counter(task, "out", background)
    changed_input_non_background = without_color(changed_input, background)
    changed_output_non_background = without_color(changed_output, background)
    input_color = most_common(
        changed_input_non_background or foreground_input,
        1 if background == 0 else 0,
    )
    output_color = most_common(
        changed_output_non_background or foreground_output,
        input_color,
    )

    if template_name == "same_grid_color_map":
        mapping = {color: color for color in range(10)}
        for input_grid, output_grid in task_train_pairs(task, max_pairs=4):
            in_w, in_h = grid_shape(input_grid)
            out_w, out_h = grid_shape(output_grid)
            if in_w != out_w or in_h != out_h:
                return []
            for y in range(in_h):
                for x in range(in_w):
                    src = int(input_grid[y][x])
                    dst = int(output_grid[y][x])
                    if mapping.get(src, dst) != dst:
                        return []
                    mapping[src] = dst
        return [{f"map.{color}": mapping[color] for color in range(10)}]

    if template_name == "select_then_translate":
        votes: Counter[tuple[int, int, int, int]] = Counter()
        for input_grid, output_grid in task_train_pairs(task, max_pairs=4):
            for source in components(input_grid, background):
                for target in components(output_grid, background):
                    if source["shape"] != target["shape"]:
                        continue
                    dx = int(target["min_x"]) - int(source["min_x"])
                    dy = int(target["min_y"]) - int(source["min_y"])
                    if dx == 0 and dy == 0 and source["color"] == target["color"]:
                        continue
                    votes[(source["color"], target["color"], dx, dy)] += 1
        return [
            {
                "object_scene.background": background,
                "select.selector_kind": SELECTOR_UNIQUE_COLOR,
                "select.color": source_color,
                "select.reference_color_used": 0,
                "select.reference_color": 0,
                "translate.dx": dx,
                "translate.dy": dy,
                "render.background": background,
                "render.output_color": target_color,
            }
            for (source_color, target_color, dx, dy), _count in votes.most_common(limit)
        ]

    if template_name == "fill_enclosed":
        border_color = most_common(foreground_input, input_color)
        return [
            {
                "object_scene.background": background,
                "fill.border_color": border_color,
                "fill.output_color": output_color,
            }
        ]

    if template_name == "connect_aligned":
        return [
            {
                "object_scene.background": background,
                "connect.color": input_color,
                "connect.output_color": output_color,
                "connect.mode": mode,
                "connect.render_mode": render_mode,
            }
            for mode in (0, 1, 2)
            for render_mode in (RENDER_OVERLAY, RENDER_ONLY)
        ][:limit]

    if template_name == "tile_motif_repeat":
        return [
            {"lattice.background": background, "repeat.period": period}
            for period in (2, 3)
        ][:limit]

    if template_name == "masked_motif_lift":
        return [{"mask.background": background}]

    if template_name in {"projection_panel_selection", "projection_bbox_panel_selection"}:
        votes: Counter[tuple[int, int, int, int]] = Counter()
        for input_grid, output_grid in task_train_pairs(task, max_pairs=4):
            match = find_subgrid(input_grid, output_grid)
            if match is None:
                continue
            out_w, out_h = grid_shape(output_grid)
            votes[(match[0], match[1], out_w, out_h)] += 1
        return [
            {
                "panel.background": background,
                "panel.x_anchor": 0,
                "panel.y_anchor": 0,
                "panel.x_offset": x,
                "panel.y_offset": y,
                "panel.width": width,
                "panel.height": height,
            }
            for (x, y, width, height), _count in votes.most_common(limit)
        ]

    return []


def slot_probe_rows_for_candidate(candidate: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    template_name = str(candidate.get("template_name", ""))
    seen_tasks: set[str] = set()
    for task_row in candidate.get("mutation_probe_tasks", []) or []:
        if not isinstance(task_row, dict):
            continue
        task_id = str(task_row.get("task_id", ""))
        task_path = Path(str(task_row.get("task_path", "")))
        if not task_id or task_id in seen_tasks or not task_path.is_file():
            continue
        seen_tasks.add(task_id)
        try:
            task = read_task(task_path)
        except (OSError, json.JSONDecodeError):
            continue
        for sketch_index, values in enumerate(
            synthesize_slot_value_sets(template_name, task, limit=limit),
            start=1,
        ):
            rows.extend(
                full_slot_probe_rows(
                    candidate,
                    f"task:{task_id}:slot_sketch:{sketch_index}",
                    values,
                    max(1, safe_int(candidate.get("prior_proposal_count"))),
                )
            )
    return rows


def load_samples(
    public_arc_dir: Path,
    diagnostics_path: Path,
    max_pairs: int,
) -> list[Sample]:
    split_dirs = public_split_dirs(public_arc_dir)
    samples: list[Sample] = []
    for row in parse_json_rows(diagnostics_path):
        task_id = str(row.get("task_id", ""))
        split = str(row.get("split", ""))
        mode = str(row.get("mode", ""))
        if not task_id or split not in split_dirs:
            continue
        task_path = split_dirs[split] / f"{task_id}.json"
        if not task_path.is_file():
            continue
        task = read_task(task_path)
        tokens = task_raw_tokens(task, max_pairs)
        if not tokens:
            continue
        selected = as_tuple_strings(row.get("selected_templates"))
        train_consistent = as_tuple_strings(row.get("train_consistent_templates"))
        certified_templates = as_tuple_strings(row.get("certified_templates"))
        samples.append(
            Sample(
                key=f"{split}:{mode}:{task_id}",
                task_id=task_id,
                split=split,
                mode=mode,
                task_path=task_path,
                label=diagnostic_label(row),
                certified=bool(row.get("certified", False)),
                selected_templates=selected,
                train_consistent_templates=train_consistent,
                certified_templates=certified_templates,
                proposals_generated=safe_int(row.get("proposals_generated")),
                proposals_train_consistent_pre_cap=safe_int(
                    row.get("proposals_train_consistent_pre_cap")
                ),
                tokens=tokens,
            )
        )
    return samples


def hash_bytes(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def hashed_raw_grid_embedding(
    tokens: tuple[tuple[str, float], ...],
    hidden_dim: int,
    seed: int,
) -> list[float]:
    values = [0.0 for _ in range(hidden_dim)]
    if not tokens:
        return values
    for token, weight in tokens:
        digest = hash_bytes(f"{seed}:{token}")
        for probe in range(HASH_PROBES):
            offset = probe * 4
            bucket = int.from_bytes(digest[offset : offset + 2], "little") % hidden_dim
            sign = 1.0 if digest[offset + 2] & 1 else -1.0
            mag = 0.5 + (digest[offset + 3] / 255.0)
            values[bucket] += sign * mag * weight
    scale = math.sqrt(float(len(tokens)))
    return [math.tanh(value / scale) for value in values]


def softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    max_logit = max(logits)
    exps = [math.exp(value - max_logit) for value in logits]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / len(logits) for _ in logits]
    return [value / total for value in exps]


def train_softmax_head(
    embeddings: list[list[float]],
    label_ids: list[int],
    label_count: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> tuple[list[list[float]], list[float], list[float]]:
    rng = random.Random(seed)
    hidden_dim = len(embeddings[0]) if embeddings else 0
    weights = [
        [(rng.random() - 0.5) * 0.02 for _ in range(hidden_dim)]
        for _ in range(label_count)
    ]
    bias = [0.0 for _ in range(label_count)]
    losses: list[float] = []
    order = list(range(len(embeddings)))
    for epoch in range(epochs):
        rng.shuffle(order)
        loss = 0.0
        for sample_index in order:
            embedding = embeddings[sample_index]
            label_id = label_ids[sample_index]
            logits = [
                bias[label] + sum(weights[label][i] * embedding[i] for i in range(hidden_dim))
                for label in range(label_count)
            ]
            probs = softmax(logits)
            loss -= math.log(max(probs[label_id], 1e-9))
            for label in range(label_count):
                grad = probs[label] - (1.0 if label == label_id else 0.0)
                bias[label] -= learning_rate * grad
                row = weights[label]
                for i, value in enumerate(embedding):
                    row[i] -= learning_rate * grad * value
        losses.append(loss / max(1, len(embeddings)))
        learning_rate *= 0.96
    return weights, bias, losses


def predict_probs(
    embedding: list[float],
    weights: list[list[float]],
    bias: list[float],
) -> list[float]:
    if not weights:
        return []
    logits = [
        bias[label] + sum(weights[label][i] * embedding[i] for i in range(len(embedding)))
        for label in range(len(weights))
    ]
    return softmax(logits)


def learned_embedding(base: list[float], probs: list[float], probability_weight: float) -> list[float]:
    return [*base, *[probability_weight * value for value in probs]]


def squared_distance(left: list[float], right: list[float]) -> float:
    return sum((a - b) * (a - b) for a, b in zip(left, right))


def kmeans(
    vectors: list[list[float]],
    cluster_count: int,
    iterations: int,
) -> tuple[list[int], list[list[float]]]:
    if not vectors:
        return [], []
    cluster_count = max(1, min(cluster_count, len(vectors)))
    centroids = [vectors[(i * len(vectors)) // cluster_count][:] for i in range(cluster_count)]
    assignments = [0 for _ in vectors]
    for _ in range(iterations):
        changed = False
        for index, vector in enumerate(vectors):
            best = min(
                range(cluster_count),
                key=lambda cluster: squared_distance(vector, centroids[cluster]),
            )
            if assignments[index] != best:
                assignments[index] = best
                changed = True
        totals = [[0.0 for _ in vectors[0]] for _ in range(cluster_count)]
        counts = [0 for _ in range(cluster_count)]
        for assignment, vector in zip(assignments, vectors):
            counts[assignment] += 1
            for i, value in enumerate(vector):
                totals[assignment][i] += value
        for cluster in range(cluster_count):
            if counts[cluster] == 0:
                continue
            centroids[cluster] = [value / counts[cluster] for value in totals[cluster]]
        if not changed:
            break
    return assignments, centroids


def counter_dict(counter: Counter[str], limit: int = 6) -> dict[str, int]:
    return {key: count for key, count in counter.most_common(limit)}


def top_probabilities(labels: list[str], probs: list[float], limit: int = 5) -> list[dict[str, Any]]:
    ranked = sorted(range(len(labels)), key=lambda index: probs[index], reverse=True)
    return [
        {"label": labels[index], "probability": probs[index]}
        for index in ranked[:limit]
    ]


def template_from_label(label: str) -> str | None:
    for prefix in ("certified:", "pressure:", "rejected:"):
        if label.startswith(prefix):
            template = label[len(prefix) :]
            return template if template in TEMPLATE_IDS else None
    return None


def stable_id(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def template_suggestions_for_cluster(
    row: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = defaultdict(float)
    sources: dict[str, Counter[str]] = defaultdict(Counter)
    nearest_templates = row.get("nearest_certified_templates", {})
    if isinstance(nearest_templates, dict):
        for template, count in nearest_templates.items():
            template_name = str(template)
            if template_name not in TEMPLATE_IDS:
                continue
            weight = max(1.0, safe_float(count))
            scores[template_name] += 2.0 * weight
            sources[template_name]["nearest_certified_neighbor"] += int(round(weight))

    for predicted in row.get("mean_predicted_labels", []) or []:
        if not isinstance(predicted, dict):
            continue
        template_name = template_from_label(str(predicted.get("label", "")))
        if template_name is None:
            continue
        probability = safe_float(predicted.get("probability"))
        scores[template_name] += probability
        sources[template_name]["neural_label_head"] += 1

    dominant_labels = row.get("dominant_labels", {})
    if isinstance(dominant_labels, dict):
        cluster_size = max(1.0, safe_float(row.get("sample_count")))
        for label, count in dominant_labels.items():
            template_name = template_from_label(str(label))
            if template_name is None:
                continue
            weight = safe_float(count) / cluster_size
            scores[template_name] += 0.25 * weight
            sources[template_name]["verifier_label_pressure"] += safe_int(count)

    ranked = sorted(scores, key=lambda template: (-scores[template], template))
    return [
        {
            "template_name": template,
            "template_id": TEMPLATE_IDS[template],
            "neural_score": scores[template],
            "sources": dict(sources[template]),
        }
        for template in ranked[:limit]
    ]


def nearest_certified_neighbors(
    vector: list[float],
    certified_indices: list[int],
    embeddings: list[list[float]],
    samples: list[Sample],
    limit: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        certified_indices,
        key=lambda index: squared_distance(vector, embeddings[index]),
    )
    out: list[dict[str, Any]] = []
    for index in ranked[:limit]:
        sample = samples[index]
        out.append(
            {
                "task_id": sample.task_id,
                "split": sample.split,
                "mode": sample.mode,
                "distance": math.sqrt(squared_distance(vector, embeddings[index])),
                "certified_templates": list(sample.certified_templates),
            }
        )
    return out


def summarize_clusters(
    samples: list[Sample],
    embeddings: list[list[float]],
    probs_by_sample: list[list[float]],
    labels: list[str],
    cluster_count: int,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    failure_indices = [index for index, sample in enumerate(samples) if not sample.certified]
    failure_vectors = [embeddings[index] for index in failure_indices]
    assignments, centroids = kmeans(failure_vectors, cluster_count, iterations=32)
    certified_indices = [index for index, sample in enumerate(samples) if sample.certified]

    cluster_rows: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []
    by_cluster: dict[int, list[int]] = defaultdict(list)
    for local_index, assignment in enumerate(assignments):
        by_cluster[assignment].append(failure_indices[local_index])

    for cluster_id in sorted(by_cluster):
        indices = by_cluster[cluster_id]
        centroid = centroids[cluster_id]
        label_counts = Counter(samples[index].label for index in indices)
        selected_counts = Counter(
            template
            for index in indices
            for template in samples[index].selected_templates
        )
        pressure_counts = Counter(
            template
            for index in indices
            for template in samples[index].train_consistent_templates
        )
        mean_probs = [0.0 for _ in labels]
        for index in indices:
            for label_id, probability in enumerate(probs_by_sample[index]):
                mean_probs[label_id] += probability
        if indices:
            mean_probs = [value / len(indices) for value in mean_probs]
        representatives = sorted(
            indices,
            key=lambda index: squared_distance(embeddings[index], centroid),
        )[:top_k]
        neighbors = nearest_certified_neighbors(
            centroid,
            certified_indices,
            embeddings,
            samples,
            limit=5,
        )
        neighbor_templates = Counter(
            template
            for neighbor in neighbors
            for template in neighbor["certified_templates"]
        )
        sketch_kind = (
            "neural_neighbor_transfer"
            if neighbor_templates
            else "latent_unexplained_cluster"
        )
        cluster_row = {
            "schema": "crlp.arc_neural_discovery.cluster.v1",
            "report_version": 1,
            "cluster_id": cluster_id,
            "sample_count": len(indices),
            "sketch_kind": sketch_kind,
            "dominant_labels": counter_dict(label_counts),
            "selected_templates": counter_dict(selected_counts),
            "train_consistent_templates": counter_dict(pressure_counts),
            "nearest_certified_templates": counter_dict(neighbor_templates),
            "mean_predicted_labels": top_probabilities(labels, mean_probs),
            "nearest_certified_neighbors": neighbors,
            "representative_tasks": [
                {
                    "task_id": samples[index].task_id,
                    "split": samples[index].split,
                    "mode": samples[index].mode,
                    "label": samples[index].label,
                    "selected_templates": list(samples[index].selected_templates),
                    "train_consistent_templates": list(
                        samples[index].train_consistent_templates
                    ),
                    "distance_to_centroid": math.sqrt(
                        squared_distance(embeddings[index], centroid)
                    ),
                }
                for index in representatives
            ],
        }
        cluster_rows.append(cluster_row)
        for rank, index in enumerate(representatives):
            sample = samples[index]
            queue_rows.append(
                {
                    "schema": "crlp.arc_neural_discovery.verification_queue.v1",
                    "report_version": 1,
                    "cluster_id": cluster_id,
                    "rank": rank + 1,
                    "task_id": sample.task_id,
                    "split": sample.split,
                    "mode": sample.mode,
                    "task_path": str(sample.task_path),
                    "label": sample.label,
                    "sketch_kind": sketch_kind,
                    "nearest_certified_templates": counter_dict(neighbor_templates),
                }
            )
    cluster_rows.sort(key=lambda row: (-row["sample_count"], row["cluster_id"]))
    queue_rows.sort(key=lambda row: (row["cluster_id"], row["rank"]))
    return cluster_rows, queue_rows


def queue_by_cluster(queue_rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in queue_rows:
        grouped[safe_int(row.get("cluster_id"))].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: safe_int(row.get("rank")))
    return grouped


def mutation_probe_priority(row: dict[str, Any]) -> tuple[int, int, int, int]:
    label = str(row.get("label", ""))
    if label.startswith("pressure:"):
        label_priority = 0
    elif label.startswith("rejected:"):
        label_priority = 1
    else:
        label_priority = 2
    split_priority = 0 if row.get("split") == "public_evaluation" else 1
    mode_priority = 0 if row.get("mode") == "default" else 1
    return (label_priority, split_priority, mode_priority, safe_int(row.get("rank")))


def mutation_probe_rows(
    grouped_queue: dict[int, list[dict[str, Any]]],
    cluster_id: int,
    limit: int,
) -> list[dict[str, Any]]:
    rows = list(grouped_queue.get(cluster_id, []))
    rows.sort(key=mutation_probe_priority)
    return rows[:limit]


def prior_row_for_mutation(candidate: dict[str, Any]) -> dict[str, Any]:
    proposal_count = safe_int(candidate.get("prior_proposal_count"))
    proposal_count = max(1, proposal_count)
    return {
        "schema_version": TEMPLATE_PROBE_SCHEMA,
        "template_id": safe_int(candidate["template_id"]),
        "template_name": str(candidate["template_name"]),
        "library_id": str(candidate["library_id"]),
        "slot_name": "__template__",
        "slot_type": "template",
        "slot_value": "0.00000000",
        "slot_integer": 0,
        "proposal_count": proposal_count,
        "certified_count": 0,
        "support_tasks": safe_int(candidate.get("support_tasks")),
        "covered_events": safe_int(candidate.get("covered_events")),
        "utility_mean": f"{safe_float(candidate.get('neural_score')):.8f}",
    }


def synthesize_mutation_candidates(
    cluster_rows: list[dict[str, Any]],
    queue_rows: list[dict[str, Any]],
    output_dir: Path,
    mutations_per_cluster: int,
) -> list[dict[str, Any]]:
    if mutations_per_cluster <= 0:
        return []
    grouped_queue = queue_by_cluster(queue_rows)
    prior_dir = output_dir / "mutation_priors"
    clear_matching_files(prior_dir, "*.tsv")
    candidates: list[dict[str, Any]] = []
    for cluster_row in cluster_rows:
        cluster_id = safe_int(cluster_row.get("cluster_id"))
        representatives = grouped_queue.get(cluster_id, [])
        mutation_representatives = mutation_probe_rows(grouped_queue, cluster_id, 5)
        suggestions = template_suggestions_for_cluster(cluster_row, mutations_per_cluster)
        for rank, suggestion in enumerate(suggestions, start=1):
            template_name = str(suggestion["template_name"])
            candidate_id = "cluster_{}_{}_{}".format(
                cluster_id,
                rank,
                template_name,
            )
            score = safe_float(suggestion.get("neural_score"))
            prior_proposal_count = min(64, max(1, 4 + int(round(score * 4.0))))
            library_id = "neural:{}:{}".format(candidate_id, stable_id(cluster_row, suggestion))
            prior_path = prior_dir / f"{candidate_id}.tsv"
            row = {
                "schema": "crlp.arc_neural_discovery.mutation_candidate.v1",
                "report_version": 1,
                "candidate_id": candidate_id,
                "cluster_id": cluster_id,
                "rank": rank,
                "mutation_kind": "template_and_slot_probe_bias",
                "executable_form": "arc_neural_discovery_probe_tsv",
                "template_name": template_name,
                "template_id": suggestion["template_id"],
                "library_id": library_id,
                "prior_path": str(prior_path),
                "prior_proposal_count": prior_proposal_count,
                "prior_certified_count": 0,
                "support_tasks": safe_int(cluster_row.get("sample_count")),
                "covered_events": safe_int(cluster_row.get("sample_count")),
                "neural_score": score,
                "sources": suggestion.get("sources", {}),
                "nearest_certified_templates": cluster_row.get("nearest_certified_templates", {}),
                "mean_predicted_labels": cluster_row.get("mean_predicted_labels", []),
                "representative_tasks": representatives[:5],
                "mutation_probe_tasks": mutation_representatives,
            }
            slot_probe_rows = slot_probe_rows_for_candidate(row)
            row["slot_probe_row_count"] = len(slot_probe_rows)
            row["slot_probe_sketch_count"] = len(
                {
                    str(slot_row.get("library_id", ""))
                    for slot_row in slot_probe_rows
                }
            )
            write_prior_tsv(prior_path, [prior_row_for_mutation(row), *slot_probe_rows])
            candidates.append(row)
    return candidates


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "cluster_id",
        "sample_count",
        "sketch_kind",
        "dominant_labels",
        "selected_templates",
        "train_consistent_templates",
        "nearest_certified_templates",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), sort_keys=True)
                    if isinstance(row.get(key), dict)
                    else row.get(key)
                    for key in fieldnames
                }
            )


def run_verifier_probes(
    arc_agi: Path,
    queue_rows: list[dict[str, Any]],
    output_dir: Path,
    representatives_per_cluster: int,
) -> list[dict[str, Any]]:
    if representatives_per_cluster <= 0:
        return []
    selected: list[dict[str, Any]] = []
    seen_by_cluster: Counter[int] = Counter()
    for row in queue_rows:
        cluster_id = safe_int(row.get("cluster_id"))
        if seen_by_cluster[cluster_id] >= representatives_per_cluster:
            continue
        seen_by_cluster[cluster_id] += 1
        selected.append(row)

    probe_rows: list[dict[str, Any]] = []
    log_dir = output_dir / "verifier_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for row in selected:
        task_path = Path(str(row["task_path"]))
        command = verifier_command(arc_agi, task_path, str(row["task_id"]), row)
        log_path = log_dir / f"cluster_{row['cluster_id']}_{row['task_id']}.jsonl"
        exit_code = run_command_to_log(command, log_path)
        probe_rows.append(
            {
                "schema": "crlp.arc_neural_discovery.verifier_probe.v1",
                "report_version": 1,
                "cluster_id": row["cluster_id"],
                "task_id": row["task_id"],
                "split": row["split"],
                "mode": row["mode"],
                "exit_code": exit_code,
                "log_path": str(log_path),
                "command": command,
            }
        )
    return probe_rows


def parse_arc_log_metrics(log_path: Path, template_name: str | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    selection: dict[str, Any] = {}
    for row in parse_json_rows(log_path):
        if row.get("schema") == "crlp.arc_loader.summary.v1":
            summary = row
        elif row.get("schema") == "crlp.arc_hypothesis.selection.v1":
            selection = row
    template_metrics: dict[str, Any] = {}
    for template in selection.get("templates", []) or []:
        if isinstance(template, dict) and template.get("name") == template_name:
            template_metrics = template
            break
    return {
        "certified_tasks": safe_int(summary.get("certified_tasks")),
        "test_exact_tasks": safe_int(summary.get("test_exact_tasks")),
        "unique_prediction_tasks": safe_int(summary.get("unique_prediction_tasks")),
        "train_consistent_rules": safe_int(summary.get("train_consistent_rules")),
        "test_cells": safe_int(summary.get("test_cells")),
        "test_correct_cells": safe_int(summary.get("test_correct_cells")),
        "cell_accuracy": safe_float(summary.get("cell_accuracy")),
        "proposals_generated": safe_int(selection.get("proposals_generated")),
        "proposals_train_consistent_pre_cap": safe_int(
            selection.get("proposals_train_consistent_pre_cap")
        ),
        "proposals_train_consistent": safe_int(selection.get("proposals_train_consistent")),
        "proposals_selected": safe_int(selection.get("proposals_selected")),
        "proposals_train_rejected": safe_int(selection.get("proposals_train_rejected")),
        "template_attempted": safe_int(template_metrics.get("attempted")),
        "template_generated": safe_int(template_metrics.get("generated")),
        "template_retained": safe_int(template_metrics.get("retained")),
        "template_train_consistent_pre_cap": safe_int(
            template_metrics.get("train_consistent_pre_cap")
        ),
        "template_train_consistent_retained": safe_int(
            template_metrics.get("train_consistent_retained")
        ),
        "template_selected": safe_int(template_metrics.get("selected")),
        "library_prior_entries": safe_int(selection.get("library_prior_entries")),
        "library_prior_template_rows": safe_int(
            selection.get("library_prior_template_rows")
        ),
        "library_prior_seeded_proposals": safe_int(
            selection.get("library_prior_seeded_proposals")
        ),
        "template_probe_rows": safe_int(selection.get("template_probe_rows")),
        "template_probe_seeded_proposals": safe_int(
            selection.get("template_probe_seeded_proposals")
        ),
        "slot_probe_rows": safe_int(selection.get("slot_probe_rows")),
        "slot_probe_seeded_proposals": safe_int(
            selection.get("slot_probe_seeded_proposals")
        ),
        "template_probe_proposals": safe_int(
            template_metrics.get("template_probe_proposals")
        ),
        "template_probe_score": safe_float(template_metrics.get("template_probe_score")),
        "slot_probe_score": safe_float(template_metrics.get("slot_probe_score")),
    }


def evaluator_flags_for(row: dict[str, Any]) -> list[str]:
    mode = str(row.get("mode", ""))
    if mode == "promotion":
        return [*PROMOTION_FLAGS]
    if mode == "stress_no_prior":
        return [*STRESS_NO_PRIOR_FLAGS]
    return [*STRICT_EVAL_FLAGS]


def verifier_command(
    arc_agi: Path,
    task_path: Path,
    task_id: str,
    diagnostic_row: dict[str, Any],
    prior_path: Path | None = None,
) -> list[str]:
    command = [
        str(arc_agi),
        "--task-dir",
        str(task_path.parent),
        "--task-ids",
        task_id,
        *evaluator_flags_for(diagnostic_row),
        "--min-induction-evidence",
        "0",
    ]
    if prior_path is not None:
        command.extend(["--hypothesis-library-prior", str(prior_path)])
    return command


def run_command_to_log(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fp:
        completed = subprocess.run(command, stdout=fp, stderr=subprocess.STDOUT)
    return completed.returncode


def mutation_verdict(
    baseline: dict[str, Any],
    mutated: dict[str, Any],
) -> tuple[str, str]:
    if safe_int(mutated.get("certified_tasks")) > safe_int(baseline.get("certified_tasks")):
        return "accepted", "novel_certification"
    if safe_int(mutated.get("test_exact_tasks")) > safe_int(baseline.get("test_exact_tasks")):
        return "accepted", "novel_test_exact"
    if safe_int(mutated.get("certified_tasks")) > 0:
        return "accepted_existing", "already_certified_baseline"
    if safe_int(mutated.get("template_selected")) > safe_int(baseline.get("template_selected")):
        return "diagnostic", "template_selection_improved"
    if safe_int(mutated.get("proposals_selected")) > safe_int(baseline.get("proposals_selected")):
        return "diagnostic", "proposal_selection_improved"
    if safe_int(mutated.get("template_train_consistent_pre_cap")) > safe_int(
        baseline.get("template_train_consistent_pre_cap")
    ):
        return "diagnostic", "template_train_consistency_improved"
    if safe_int(mutated.get("proposals_train_consistent_pre_cap")) > safe_int(
        baseline.get("proposals_train_consistent_pre_cap")
    ):
        return "diagnostic", "train_consistency_improved"
    return "rejected", "no_verifier_gain"


def verify_mutation_candidates(
    arc_agi: Path,
    mutation_candidates: list[dict[str, Any]],
    queue_rows: list[dict[str, Any]],
    output_dir: Path,
    representatives_per_cluster: int,
) -> list[dict[str, Any]]:
    if representatives_per_cluster <= 0 or not mutation_candidates:
        return []
    grouped_queue = queue_by_cluster(queue_rows)
    log_dir = output_dir / "mutation_verifier_logs"
    clear_matching_files(log_dir, "*.jsonl")
    baseline_cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for candidate in mutation_candidates:
        cluster_id = safe_int(candidate.get("cluster_id"))
        representatives = mutation_probe_rows(
            grouped_queue,
            cluster_id,
            representatives_per_cluster,
        )
        for representative in representatives:
            task_id = str(representative["task_id"])
            task_path = Path(str(representative["task_path"]))
            mode = str(representative.get("mode", "default"))
            split = str(representative.get("split", "unknown"))
            baseline_key = f"{cluster_id}:{split}:{mode}:{task_id}"
            baseline_log = log_dir / f"baseline_cluster_{cluster_id}_{mode}_{task_id}.jsonl"
            if baseline_key not in baseline_cache:
                baseline_command = verifier_command(
                    arc_agi,
                    task_path,
                    task_id,
                    representative,
                )
                baseline_exit = run_command_to_log(baseline_command, baseline_log)
                baseline_cache[baseline_key] = {
                    "exit_code": baseline_exit,
                    "command": baseline_command,
                    "log_path": str(baseline_log),
                }
            baseline_run = baseline_cache[baseline_key]
            baseline_metrics = parse_arc_log_metrics(
                Path(str(baseline_run["log_path"])),
                str(candidate["template_name"]),
            )
            mutation_log = (
                log_dir / f"{candidate['candidate_id']}_{mode}_{task_id}.jsonl"
            )
            prior_path = Path(str(candidate["prior_path"]))
            mutation_command = verifier_command(
                arc_agi,
                task_path,
                task_id,
                representative,
                prior_path,
            )
            mutation_exit = run_command_to_log(mutation_command, mutation_log)
            mutation_metrics = parse_arc_log_metrics(
                mutation_log,
                str(candidate["template_name"]),
            )
            status, reason = mutation_verdict(
                baseline_metrics,
                mutation_metrics,
            )
            rows.append(
                {
                    "schema": "crlp.arc_neural_discovery.mutation_verification.v1",
                    "report_version": 1,
                    "candidate_id": candidate["candidate_id"],
                    "cluster_id": cluster_id,
                    "task_id": task_id,
                    "split": representative["split"],
                    "mode": representative["mode"],
                    "template_name": candidate["template_name"],
                    "template_id": candidate["template_id"],
                    "status": status,
                    "reason": reason,
                    "accepted_by_verifier": safe_int(
                        mutation_metrics.get("certified_tasks")
                    )
                    > 0,
                    "novel_certification": safe_int(
                        mutation_metrics.get("certified_tasks")
                    )
                    > safe_int(baseline_metrics.get("certified_tasks")),
                    "baseline_exit_code": baseline_run["exit_code"],
                    "mutation_exit_code": mutation_exit,
                    "baseline_metrics": baseline_metrics,
                    "mutation_metrics": mutation_metrics,
                    "baseline_log_path": baseline_run["log_path"],
                    "mutation_log_path": str(mutation_log),
                    "prior_path": str(prior_path),
                    "baseline_command": baseline_run["command"],
                    "mutation_command": mutation_command,
                }
            )
    return rows


def markdown_summary(
    sample_count: int,
    label_count: int,
    losses: list[float],
    accuracy: float,
    cluster_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    mutation_candidates: list[dict[str, Any]],
    mutation_verifications: list[dict[str, Any]],
) -> str:
    lines = [
        "# ARC Neural Discovery Loop",
        "",
        "Input policy: raw padded train input/output grid cells and validity masks only.",
        "The model does not consume palette counts, shape labels, separator labels,",
        "handwritten ARC triage labels, verifier-derived utilities as input features, or HTM state.",
        "Verifier diagnostics are labels and certification authority, not model input.",
        "",
        "## Model",
        "",
        f"- Samples: `{sample_count}`",
        f"- Labels: `{label_count}`",
        f"- Final training loss: `{losses[-1] if losses else 0.0:.4f}`",
        f"- Training accuracy: `{accuracy:.3f}`",
        "",
        "## Latent Failure Clusters",
        "",
        "| Cluster | Failures | Sketch kind | Dominant labels | Nearest certified templates | Representatives |",
        "| ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in cluster_rows:
        representatives = ",".join(
            f"`{task['task_id']}`" for task in row["representative_tasks"][:5]
        )
        lines.append(
            "| {} | {} | `{}` | `{}` | `{}` | {} |".format(
                row["cluster_id"],
                row["sample_count"],
                row["sketch_kind"],
                json.dumps(row["dominant_labels"], sort_keys=True),
                json.dumps(row["nearest_certified_templates"], sort_keys=True),
                representatives or "`none`",
            )
        )
    lines.extend(["", "## Verifier Probes", ""])
    if probe_rows:
        lines.extend(["| Cluster | Task | Exit | Log |", "| ---: | --- | ---: | --- |"])
        for row in probe_rows:
            lines.append(
                "| {} | `{}` | {} | `{}` |".format(
                    row["cluster_id"],
                    row["task_id"],
                    row["exit_code"],
                    row["log_path"],
                )
            )
    else:
        lines.append("No verifier probes requested. Set `--verify-representatives` above zero to run them.")
    lines.extend(["", "## Executable Symbolic Mutations", ""])
    if mutation_candidates:
        lines.extend(
            [
                "| Candidate | Cluster | Template | Template probes | Slot sketches | Slot rows | Sources | Prior |",
                "| --- | ---: | --- | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for row in mutation_candidates[:24]:
            lines.append(
                "| `{}` | {} | `{}` | {} | {} | {} | `{}` | `{}` |".format(
                    row["candidate_id"],
                    row["cluster_id"],
                    row["template_name"],
                    row["prior_proposal_count"],
                    row.get("slot_probe_sketch_count", 0),
                    row.get("slot_probe_row_count", 0),
                    json.dumps(row["sources"], sort_keys=True),
                    row["prior_path"],
                )
            )
        if len(mutation_candidates) > 24:
            lines.append(
                f"| ... | ... | ... | ... | ... | ... | ... | `{len(mutation_candidates) - 24} more` |"
            )
    else:
        lines.append("No mutation candidates requested. Set `--mutations-per-cluster` above zero.")

    lines.extend(["", "## Mutation Verification", ""])
    if mutation_verifications:
        status_counts = Counter(str(row["status"]) for row in mutation_verifications)
        novel = sum(1 for row in mutation_verifications if row.get("novel_certification"))
        verifier_accepts = sum(
            1 for row in mutation_verifications if row.get("accepted_by_verifier")
        )
        lines.extend(
            [
                f"- Runs: `{len(mutation_verifications)}`",
                f"- Status counts: `{json.dumps(dict(status_counts), sort_keys=True)}`",
                f"- Verifier-accepted runs: `{verifier_accepts}`",
                f"- Novel certifications over baseline: `{novel}`",
                "",
                "| Candidate | Task | Template | Status | Reason | Base cert | Mut cert | Log |",
                "| --- | --- | --- | --- | --- | ---: | ---: | --- |",
            ]
        )
        for row in mutation_verifications[:24]:
            lines.append(
                "| `{}` | `{}` | `{}` | `{}` | `{}` | {} | {} | `{}` |".format(
                    row["candidate_id"],
                    row["task_id"],
                    row["template_name"],
                    row["status"],
                    row["reason"],
                    row["baseline_metrics"]["certified_tasks"],
                    row["mutation_metrics"]["certified_tasks"],
                    row["mutation_log_path"],
                )
            )
        if len(mutation_verifications) > 24:
            lines.append(
                f"| ... | ... | ... | ... | ... | ... | ... | `{len(mutation_verifications) - 24} more` |"
            )
    else:
        lines.append("No mutation verification requested. Set `--mutation-representatives` above zero.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-arc-dir", type=Path, required=True)
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=Path("build/float/hypothesis_library_public_filter/diagnostics.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/float/hypothesis_library_neural_discovery"),
    )
    parser.add_argument("--arc-agi", type=Path)
    parser.add_argument("--max-train-pairs", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--clusters", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--probability-weight", type=float, default=1.5)
    parser.add_argument("--verify-representatives", type=int, default=0)
    parser.add_argument("--mutations-per-cluster", type=int, default=2)
    parser.add_argument("--mutation-representatives", type=int, default=0)
    args = parser.parse_args()

    samples = load_samples(args.public_arc_dir, args.diagnostics, args.max_train_pairs)
    if not samples:
        raise RuntimeError("no samples loaded; run public filter diagnostics first")

    labels = sorted({sample.label for sample in samples})
    label_to_id = {label: index for index, label in enumerate(labels)}
    base_embeddings = [
        hashed_raw_grid_embedding(sample.tokens, args.hidden_dim, args.seed)
        for sample in samples
    ]
    label_ids = [label_to_id[sample.label] for sample in samples]
    weights, bias, losses = train_softmax_head(
        base_embeddings,
        label_ids,
        len(labels),
        args.epochs,
        args.learning_rate,
        args.seed,
    )
    probs_by_sample = [
        predict_probs(embedding, weights, bias)
        for embedding in base_embeddings
    ]
    predictions = [
        max(range(len(probs)), key=lambda index: probs[index])
        for probs in probs_by_sample
    ]
    accuracy = sum(
        1 for predicted, label_id in zip(predictions, label_ids) if predicted == label_id
    ) / max(1, len(label_ids))
    neural_embeddings = [
        learned_embedding(base, probs, args.probability_weight)
        for base, probs in zip(base_embeddings, probs_by_sample)
    ]

    cluster_rows, queue_rows = summarize_clusters(
        samples,
        neural_embeddings,
        probs_by_sample,
        labels,
        args.clusters,
        args.top_k,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mutation_candidates = synthesize_mutation_candidates(
        cluster_rows,
        queue_rows,
        args.output_dir,
        args.mutations_per_cluster,
    )
    mutation_verifications: list[dict[str, Any]] = []
    if args.mutation_representatives > 0:
        if args.arc_agi is None:
            raise RuntimeError("--mutation-representatives requires --arc-agi")
        mutation_verifications = verify_mutation_candidates(
            args.arc_agi,
            mutation_candidates,
            queue_rows,
            args.output_dir,
            args.mutation_representatives,
        )

    probe_rows: list[dict[str, Any]] = []
    if args.verify_representatives > 0:
        if args.arc_agi is None:
            raise RuntimeError("--verify-representatives requires --arc-agi")
        probe_rows = run_verifier_probes(
            args.arc_agi,
            queue_rows,
            args.output_dir,
            args.verify_representatives,
        )

    sample_rows = [
        {
            "schema": "crlp.arc_neural_discovery.sample.v1",
            "report_version": 1,
            "key": sample.key,
            "task_id": sample.task_id,
            "split": sample.split,
            "mode": sample.mode,
            "label": sample.label,
            "certified": sample.certified,
            "selected_templates": list(sample.selected_templates),
            "train_consistent_templates": list(sample.train_consistent_templates),
            "certified_templates": list(sample.certified_templates),
            "token_count": len(sample.tokens),
        }
        for sample in samples
    ]
    embedding_rows = [
        {
            "schema": "crlp.arc_neural_discovery.embedding.v1",
            "report_version": 1,
            "key": sample.key,
            "task_id": sample.task_id,
            "split": sample.split,
            "mode": sample.mode,
            "label": sample.label,
            "certified": sample.certified,
            "predicted_label": labels[predictions[index]],
            "predicted_label_probability": probs_by_sample[index][predictions[index]],
            "top_predicted_labels": top_probabilities(labels, probs_by_sample[index]),
            "embedding": [round(value, 8) for value in neural_embeddings[index]],
        }
        for index, sample in enumerate(samples)
    ]
    model_row = {
        "schema": "crlp.arc_neural_discovery.model.v1",
        "report_version": 1,
        "feature_policy": "raw_grid_pair_tokens_only",
        "hand_coded_arc_features_used": False,
        "htm_used": False,
        "sample_count": len(samples),
        "label_count": len(labels),
        "labels": labels,
        "hidden_dim": args.hidden_dim,
        "max_train_pairs": args.max_train_pairs,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "mutations_per_cluster": args.mutations_per_cluster,
        "mutation_representatives": args.mutation_representatives,
        "losses": losses,
        "training_accuracy": accuracy,
    }

    write_jsonl(args.output_dir / "samples.jsonl", sample_rows)
    write_jsonl(args.output_dir / "embeddings.jsonl", embedding_rows)
    write_jsonl(args.output_dir / "clusters.jsonl", cluster_rows)
    write_jsonl(args.output_dir / "candidate_sketches.jsonl", mutation_candidates)
    write_jsonl(args.output_dir / "verification_queue.jsonl", queue_rows)
    write_jsonl(args.output_dir / "model.jsonl", [model_row])
    write_jsonl(args.output_dir / "verifier_probes.jsonl", probe_rows)
    write_jsonl(args.output_dir / "mutation_verification.jsonl", mutation_verifications)
    write_csv(args.output_dir / "clusters.csv", cluster_rows)
    (args.output_dir / "summary.md").write_text(
        markdown_summary(
            len(samples),
            len(labels),
            losses,
            accuracy,
            cluster_rows,
            probe_rows,
            mutation_candidates,
            mutation_verifications,
        ),
        encoding="utf-8",
    )
    print(f"wrote neural discovery loop artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
