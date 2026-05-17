#!/usr/bin/env python3
"""Mine ARC induction failures into recurring symbolic-family candidates.

This is the first offline loop for learned induction-family discovery:

1. Read per-task ARC induction diagnostics.
2. Load the corresponding ARC JSON tasks.
3. Extract structural train-pair signatures.
4. Cluster repeated signatures into candidate symbolic families.
5. Emit report, cluster rows, task signatures, and pseudo-labels for a future
   neural family proposer.

The verifier still owns correctness. This script only proposes where the
symbolic search space should grow next.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DISCOVERY_SCHEMA = "crlp.arc_family_discovery.summary.v1"
CLUSTER_SCHEMA = "crlp.arc_family_discovery.cluster.v1"
TASK_SCHEMA = "crlp.arc_family_discovery.task_signature.v1"
LABEL_SCHEMA = "crlp.arc_family_discovery.label.v1"

DEFAULT_BUCKETS = "no_induction_generated,no_induction_attempted"
DEFAULT_SPLITS = "eval"


Grid = list[list[int]]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_csv_set(text: str) -> set[str]:
    return {part.strip() for part in text.split(",") if part.strip()}


def grid_height(grid: Grid) -> int:
    return len(grid)


def grid_width(grid: Grid) -> int:
    return len(grid[0]) if grid else 0


def grid_area(grid: Grid) -> int:
    return grid_height(grid) * grid_width(grid)


def grid_cells(grid: Grid) -> list[int]:
    return [cell for row in grid for cell in row]


def most_common_color(grid: Grid) -> int:
    counts = Counter(grid_cells(grid))
    if not counts:
        return 0
    return min(counts.items(), key=lambda item: (-item[1], item[0]))[0]


def color_set(grid: Grid) -> set[int]:
    return set(grid_cells(grid))


def foreground_positions(grid: Grid, background: int | None = None) -> set[tuple[int, int]]:
    bg = most_common_color(grid) if background is None else background
    return {
        (y, x)
        for y, row in enumerate(grid)
        for x, value in enumerate(row)
        if value != bg
    }


def bbox(positions: set[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    if not positions:
        return None
    ys = [pos[0] for pos in positions]
    xs = [pos[1] for pos in positions]
    return min(ys), min(xs), max(ys), max(xs)


def crop(grid: Grid, box: tuple[int, int, int, int]) -> Grid:
    y0, x0, y1, x1 = box
    return [row[x0 : x1 + 1] for row in grid[y0 : y1 + 1]]


def grids_equal(a: Grid, b: Grid) -> bool:
    return a == b


def mirror_horizontal(grid: Grid) -> Grid:
    return [list(reversed(row)) for row in grid]


def mirror_vertical(grid: Grid) -> Grid:
    return list(reversed(grid))


def rotate_180(grid: Grid) -> Grid:
    return mirror_vertical(mirror_horizontal(grid))


def find_subgrid(haystack: Grid, needle: Grid) -> bool:
    nh = grid_height(needle)
    nw = grid_width(needle)
    hh = grid_height(haystack)
    hw = grid_width(haystack)
    if nh == 0 or nw == 0 or nh > hh or nw > hw:
        return False
    for y0 in range(hh - nh + 1):
        for x0 in range(hw - nw + 1):
            if all(haystack[y0 + dy][x0 : x0 + nw] == needle[dy] for dy in range(nh)):
                return True
    return False


def component_count(grid: Grid, background: int | None = None) -> int:
    bg = most_common_color(grid) if background is None else background
    height = grid_height(grid)
    width = grid_width(grid)
    seen: set[tuple[int, int]] = set()
    count = 0
    for y in range(height):
        for x in range(width):
            if (y, x) in seen or grid[y][x] == bg:
                continue
            count += 1
            color = grid[y][x]
            stack = [(y, x)]
            seen.add((y, x))
            while stack:
                cy, cx = stack.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and (ny, nx) not in seen
                        and grid[ny][nx] == color
                    ):
                        seen.add((ny, nx))
                        stack.append((ny, nx))
    return count


def diff_stats(inp: Grid, out: Grid, background: int | None = None) -> dict[str, Any]:
    if grid_height(inp) != grid_height(out) or grid_width(inp) != grid_width(out):
        return {
            "same_shape": False,
            "preserved_nonzero": False,
            "added_cells": 0,
            "removed_cells": 0,
            "recolored_cells": 0,
            "changed_cells": 0,
            "change_bbox_area": 0,
            "change_bbox_fraction": 0.0,
        }
    bg = most_common_color(inp) if background is None else background
    added = 0
    removed = 0
    recolored = 0
    changed_positions: set[tuple[int, int]] = set()
    preserved_nonzero = True
    for y in range(grid_height(inp)):
        for x in range(grid_width(inp)):
            a = inp[y][x]
            b = out[y][x]
            if a != b:
                changed_positions.add((y, x))
            if a != bg and b != a:
                preserved_nonzero = False
            if a == bg and b != bg:
                added += 1
            elif a != bg and b == bg:
                removed += 1
            elif a != bg and b != bg and a != b:
                recolored += 1
    box = bbox(changed_positions)
    box_area = 0
    if box is not None:
        y0, x0, y1, x1 = box
        box_area = (y1 - y0 + 1) * (x1 - x0 + 1)
    area = grid_area(inp) or 1
    return {
        "same_shape": True,
        "preserved_nonzero": preserved_nonzero,
        "added_cells": added,
        "removed_cells": removed,
        "recolored_cells": recolored,
        "changed_cells": len(changed_positions),
        "change_bbox_area": box_area,
        "change_bbox_fraction": box_area / area,
    }


def pair_signature(pair: dict[str, Any]) -> dict[str, Any]:
    inp: Grid = pair["input"]
    out: Grid = pair["output"]
    input_bg = most_common_color(inp)
    output_bg = most_common_color(out)
    input_fg = foreground_positions(inp, input_bg)
    output_fg = foreground_positions(out, output_bg)
    input_box = bbox(input_fg)
    output_box = bbox(output_fg)
    stats = diff_stats(inp, out, input_bg)
    output_is_input_crop = find_subgrid(inp, out)
    bbox_crop_equal = bool(input_box is not None and grids_equal(crop(inp, input_box), out))
    same_shape = grid_height(inp) == grid_height(out) and grid_width(inp) == grid_width(out)
    if same_shape and grids_equal(out, mirror_horizontal(inp)):
        symmetry = "mirror_horizontal"
    elif same_shape and grids_equal(out, mirror_vertical(inp)):
        symmetry = "mirror_vertical"
    elif same_shape and grids_equal(out, rotate_180(inp)):
        symmetry = "rotate_180"
    else:
        symmetry = "none"

    in_colors = color_set(inp)
    out_colors = color_set(out)
    input_components = component_count(inp, input_bg)
    output_components = component_count(out, output_bg)
    area = grid_area(inp) or 1
    out_area = grid_area(out) or 1
    integer_scale = (
        grid_height(out) % max(1, grid_height(inp)) == 0
        and grid_width(out) % max(1, grid_width(inp)) == 0
        and grid_area(out) > grid_area(inp)
    )
    return {
        "input_shape": [grid_height(inp), grid_width(inp)],
        "output_shape": [grid_height(out), grid_width(out)],
        "same_shape": same_shape,
        "output_smaller": out_area < area,
        "output_larger": out_area > area,
        "integer_scale": integer_scale,
        "input_color_count": len(in_colors),
        "output_color_count": len(out_colors),
        "added_color_count": len(out_colors - in_colors),
        "removed_color_count": len(in_colors - out_colors),
        "input_foreground_cells": len(input_fg),
        "output_foreground_cells": len(output_fg),
        "foreground_delta": len(output_fg) - len(input_fg),
        "input_component_count": input_components,
        "output_component_count": output_components,
        "component_delta": output_components - input_components,
        "output_is_input_crop": output_is_input_crop,
        "bbox_crop_equal": bbox_crop_equal,
        "symmetry": symmetry,
        **stats,
    }


def all_true(pairs: list[dict[str, Any]], key: str) -> bool:
    return bool(pairs) and all(bool(pair.get(key)) for pair in pairs)


def any_true(pairs: list[dict[str, Any]], key: str) -> bool:
    return any(bool(pair.get(key)) for pair in pairs)


def total(pairs: list[dict[str, Any]], key: str) -> int:
    return sum(int(pair.get(key, 0) or 0) for pair in pairs)


def sign_bucket(value: int) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def propose_family(pair_sigs: list[dict[str, Any]]) -> tuple[str, str, list[str], list[str]]:
    same_shape = all_true(pair_sigs, "same_shape")
    shape_relation = "same_shape" if same_shape else "shape_change"
    if all_true(pair_sigs, "output_smaller"):
        shape_relation = "output_smaller"
    elif all_true(pair_sigs, "output_larger"):
        shape_relation = "output_larger"

    slots: list[str] = ["background_color"]
    preconditions: list[str] = []
    family = "residual_unclassified"

    if all_true(pair_sigs, "output_is_input_crop") or all_true(pair_sigs, "bbox_crop_equal"):
        family = "crop_or_object_extract"
        slots += ["object_selector", "crop_frame", "background_policy"]
        preconditions += ["output is a train-pair crop or foreground bounding box"]
    elif any(pair.get("symmetry") != "none" for pair in pair_sigs) and same_shape:
        family = "symmetry_or_reflection"
        slots += ["axis", "pivot", "copy_policy", "color_policy"]
        preconditions += ["output shape equals input shape", "train output matches a grid symmetry"]
    elif same_shape and all_true(pair_sigs, "preserved_nonzero") and total(pair_sigs, "added_cells") > 0:
        foreground_delta = sum(int(pair["foreground_delta"]) for pair in pair_sigs)
        component_delta = sum(int(pair["component_delta"]) for pair in pair_sigs)
        if component_delta > 0:
            family = "copy_or_extend_objects"
            slots += ["source_object_selector", "offset_or_repetition_rule", "render_color_policy"]
        else:
            family = "shape_completion_or_fill"
            slots += ["boundary_selector", "interior_or_completion_rule", "fill_color_policy"]
        preconditions += ["input foreground is preserved", "output adds cells on same canvas"]
        if foreground_delta > 0:
            preconditions.append("foreground count increases")
    elif same_shape and total(pair_sigs, "recolored_cells") > 0 and total(pair_sigs, "changed_cells") > 0:
        family = "color_mapping_or_rewrite"
        slots += ["source_color_selector", "target_color_policy", "mask_selector"]
        preconditions += ["input/output shape is stable", "changed cells are primarily recolors"]
    elif all_true(pair_sigs, "integer_scale") or all_true(pair_sigs, "output_larger"):
        family = "scale_tile_or_expand"
        slots += ["scale_x", "scale_y", "tile_policy", "separator_policy"]
        preconditions += ["output is larger than input"]
    elif all_true(pair_sigs, "output_smaller"):
        family = "projection_or_panel_selection"
        slots += ["panel_selector", "mask_selector", "projection_axis"]
        preconditions += ["output is smaller than input"]

    added_colors = sign_bucket(total(pair_sigs, "added_color_count"))
    component_delta = sign_bucket(total(pair_sigs, "component_delta"))
    foreground_delta = sign_bucket(total(pair_sigs, "foreground_delta"))
    change_profile = "mixed"
    if same_shape:
        if total(pair_sigs, "added_cells") > 0 and total(pair_sigs, "removed_cells") == 0:
            change_profile = "additive"
        elif total(pair_sigs, "recolored_cells") > 0 and total(pair_sigs, "added_cells") == 0:
            change_profile = "recolor"
        elif total(pair_sigs, "removed_cells") > 0:
            change_profile = "subtractive"
        elif total(pair_sigs, "changed_cells") == 0:
            change_profile = "identity_or_test_only"
    cluster_key = "|".join(
        [
            family,
            f"shape={shape_relation}",
            f"change={change_profile}",
            f"fg_delta={foreground_delta}",
            f"component_delta={component_delta}",
            f"added_colors={added_colors}",
        ]
    )
    return family, cluster_key, slots, preconditions


def numeric_feature_summary(pair_sigs: list[dict[str, Any]]) -> dict[str, float]:
    count = max(1, len(pair_sigs))
    return {
        "train_pairs": float(len(pair_sigs)),
        "same_shape_fraction": sum(1 for pair in pair_sigs if pair["same_shape"]) / count,
        "output_smaller_fraction": sum(1 for pair in pair_sigs if pair["output_smaller"]) / count,
        "output_larger_fraction": sum(1 for pair in pair_sigs if pair["output_larger"]) / count,
        "median_input_area": median([pair["input_shape"][0] * pair["input_shape"][1] for pair in pair_sigs]),
        "median_output_area": median(
            [pair["output_shape"][0] * pair["output_shape"][1] for pair in pair_sigs]
        ),
        "median_input_colors": median([pair["input_color_count"] for pair in pair_sigs]),
        "median_output_colors": median([pair["output_color_count"] for pair in pair_sigs]),
        "median_input_components": median([pair["input_component_count"] for pair in pair_sigs]),
        "median_output_components": median([pair["output_component_count"] for pair in pair_sigs]),
        "median_change_bbox_fraction": median(
            [float(pair["change_bbox_fraction"]) for pair in pair_sigs]
        ),
        "total_added_cells": float(total(pair_sigs, "added_cells")),
        "total_removed_cells": float(total(pair_sigs, "removed_cells")),
        "total_recolored_cells": float(total(pair_sigs, "recolored_cells")),
    }


def task_file(task_dir: Path, split: str, task_id: str) -> Path:
    split_names = [split]
    if split == "train":
        split_names.append("training")
    elif split == "eval":
        split_names.append("evaluation")
    for split_name in split_names:
        split_path = task_dir / split_name / f"{task_id}.json"
        if split_path.is_file():
            return split_path
    direct_path = task_dir / f"{task_id}.json"
    if direct_path.is_file():
        return direct_path
    raise FileNotFoundError(f"could not find {task_id}.json under {task_dir}")


def cluster_id(cluster_key: str) -> str:
    digest = hashlib.sha256(cluster_key.encode("utf-8")).hexdigest()[:12]
    return f"arc_family_{digest}"


def template_stub(family: str, slots: list[str], preconditions: list[str]) -> dict[str, Any]:
    return {
        "family_name": family,
        "program_skeleton": {
            "crop_or_object_extract": "select_object_or_region -> crop -> normalize_background",
            "symmetry_or_reflection": "select_canvas_or_object -> reflect/rotate -> render",
            "copy_or_extend_objects": "select_source_objects -> infer_offsets_or_repetition -> render_new_objects",
            "shape_completion_or_fill": "select_boundary_or_partial_shape -> infer_missing_cells -> fill_or_overlay",
            "color_mapping_or_rewrite": "select_mask -> infer_color_map -> recolor",
            "scale_tile_or_expand": "select_source_grid -> scale_or_tile -> render",
            "projection_or_panel_selection": "select_panel_or_mask -> project_or_extract -> render",
            "residual_unclassified": "discover_program_shape -> canonicalize -> verify",
        }.get(family, "discover_program_shape -> canonicalize -> verify"),
        "slots": slots,
        "preconditions": preconditions,
        "verifier_contract": "candidate must match all train pairs and produce a unique test prediction",
    }


def task_signature(
    task_dir: Path,
    row: dict[str, Any],
) -> dict[str, Any]:
    split = str(row["split"])
    task_id = str(row["task_id"])
    path = task_file(task_dir, split, task_id)
    task = json.loads(path.read_text(encoding="utf-8"))
    pair_sigs = [pair_signature(pair) for pair in task.get("train", [])]
    family, key, slots, preconditions = propose_family(pair_sigs)
    features = numeric_feature_summary(pair_sigs)
    return {
        "schema": TASK_SCHEMA,
        "split": split,
        "task_id": task_id,
        "bucket": row.get("bucket"),
        "test_cells": int(row.get("test_cells", 0) or 0),
        "cell_delta": int(row.get("cell_delta", 0) or 0),
        "baseline_certified": bool(row.get("baseline_certified", False)),
        "induction_certified": bool(row.get("induction_certified", False)),
        "induction_proposals_attempted": int(row.get("induction_proposals_attempted", 0) or 0),
        "induction_proposals_generated": int(row.get("induction_proposals_generated", 0) or 0),
        "proposed_family": family,
        "cluster_key": key,
        "cluster_id": cluster_id(key),
        "pair_count": len(pair_sigs),
        "features": features,
        "slots": slots,
        "preconditions": preconditions,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def build_clusters(signatures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in signatures:
        grouped[str(row["cluster_key"])].append(row)
    clusters: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda row: (-int(row["test_cells"]), row["task_id"]))
        family = str(rows_sorted[0]["proposed_family"])
        support = len(rows_sorted)
        total_cells = sum(int(row["test_cells"]) for row in rows_sorted)
        total_attempts = sum(int(row["induction_proposals_attempted"]) for row in rows_sorted)
        generated = sum(int(row["induction_proposals_generated"]) for row in rows_sorted)
        feature_keys = rows_sorted[0]["features"].keys()
        feature_summary = {
            key_name: sum(float(row["features"][key_name]) for row in rows_sorted) / support
            for key_name in feature_keys
        }
        slots = list(rows_sorted[0]["slots"])
        preconditions = list(rows_sorted[0]["preconditions"])
        clusters.append(
            {
                "schema": CLUSTER_SCHEMA,
                "cluster_id": cluster_id(key),
                "cluster_key": key,
                "proposed_family": family,
                "support": support,
                "test_cells": total_cells,
                "induction_attempts": total_attempts,
                "induction_generated": generated,
                "examples": [row["task_id"] for row in rows_sorted[:10]],
                "splits": sorted({str(row["split"]) for row in rows_sorted}),
                "buckets": dict(Counter(str(row["bucket"]) for row in rows_sorted)),
                "feature_summary": feature_summary,
                "template_stub": template_stub(family, slots, preconditions),
            }
        )
    clusters.sort(key=lambda row: (-int(row["support"]), -int(row["test_cells"]), row["cluster_id"]))
    return clusters


def write_labels(path: Path, signatures: list[dict[str, Any]]) -> None:
    labels = []
    for row in signatures:
        labels.append(
            {
                "schema": LABEL_SCHEMA,
                "task": "arc_agi",
                "split": row["split"],
                "task_id": row["task_id"],
                "target_family": row["proposed_family"],
                "target_cluster_id": row["cluster_id"],
                "features": row["features"],
                "source_bucket": row["bucket"],
            }
        )
    write_jsonl(path, labels)


def write_report(path: Path, summary: dict[str, Any], clusters: list[dict[str, Any]]) -> None:
    lines = [
        "# ARC Family Discovery",
        "",
        f"Task dir: `{summary['task_dir']}`",
        f"Diagnostics: `{summary['diagnostics_path']}`",
        f"Splits: `{', '.join(summary['splits'])}`",
        f"Buckets: `{', '.join(summary['buckets'])}`",
        f"Tasks clustered: `{summary['tasks_clustered']}`",
        f"Clusters: `{summary['clusters']}`",
        "",
        "| Rank | Cluster | Family | Support | Test Cells | Attempts | Generated | Examples |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for index, cluster in enumerate(clusters, start=1):
        examples = ", ".join(f"`{task_id}`" for task_id in cluster["examples"][:5])
        lines.append(
            f"| {index} | `{cluster['cluster_id']}` | `{cluster['proposed_family']}` | "
            f"{cluster['support']} | {cluster['test_cells']} | "
            f"{cluster['induction_attempts']} | {cluster['induction_generated']} | {examples} |"
        )
    lines.extend(["", "## Template Stubs", ""])
    for cluster in clusters[:10]:
        stub = cluster["template_stub"]
        lines.extend(
            [
                f"### {cluster['cluster_id']} - `{cluster['proposed_family']}`",
                "",
                f"- support: `{cluster['support']}` tasks",
                f"- skeleton: `{stub['program_skeleton']}`",
                f"- slots: `{', '.join(stub['slots'])}`",
                f"- verifier: {stub['verifier_contract']}",
                "- preconditions:",
            ]
        )
        if stub["preconditions"]:
            for precondition in stub["preconditions"]:
                lines.append(f"  - {precondition}")
        else:
            lines.append("  - inferred from verified cluster traces")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("build/float/arc_family_discovery"))
    parser.add_argument("--splits", default=DEFAULT_SPLITS)
    parser.add_argument("--buckets", default=DEFAULT_BUCKETS)
    parser.add_argument("--min-cluster-size", type=int, default=2)
    args = parser.parse_args(argv)

    selected_splits = parse_csv_set(args.splits)
    selected_buckets = parse_csv_set(args.buckets)
    diagnostics = read_jsonl(args.diagnostics)
    filtered = [
        row
        for row in diagnostics
        if str(row.get("split")) in selected_splits and str(row.get("bucket")) in selected_buckets
    ]
    signatures = [task_signature(args.task_dir, row) for row in filtered]
    clusters = [
        cluster
        for cluster in build_clusters(signatures)
        if int(cluster["support"]) >= args.min_cluster_size
    ]
    kept_ids = {cluster["cluster_id"] for cluster in clusters}
    kept_signatures = [row for row in signatures if row["cluster_id"] in kept_ids]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_signature_path = args.output_dir / "task_signatures.jsonl"
    cluster_path = args.output_dir / "family_clusters.jsonl"
    label_path = args.output_dir / "family_labels.jsonl"
    report_path = args.output_dir / "summary.md"

    write_jsonl(task_signature_path, kept_signatures)
    write_jsonl(cluster_path, clusters)
    write_labels(label_path, kept_signatures)
    summary = {
        "schema": DISCOVERY_SCHEMA,
        "task": "arc_agi",
        "task_dir": str(args.task_dir),
        "diagnostics_path": str(args.diagnostics),
        "splits": sorted(selected_splits),
        "buckets": sorted(selected_buckets),
        "tasks_scanned": len(filtered),
        "tasks_clustered": len(kept_signatures),
        "clusters": len(clusters),
        "min_cluster_size": args.min_cluster_size,
        "task_signature_path": str(task_signature_path),
        "cluster_path": str(cluster_path),
        "label_path": str(label_path),
        "report_path": str(report_path),
        "top_clusters": [
            {
                "cluster_id": cluster["cluster_id"],
                "family": cluster["proposed_family"],
                "support": cluster["support"],
                "test_cells": cluster["test_cells"],
                "examples": cluster["examples"][:5],
            }
            for cluster in clusters[:10]
        ],
    }
    write_report(report_path, summary, clusters)
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
