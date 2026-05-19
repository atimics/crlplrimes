#!/usr/bin/env python3
"""Attach structural next-template labels to ARC public diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


Grid = list[list[int]]
Key = tuple[str, str, str]


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


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


def grid_dims(grid: Any) -> tuple[int, int]:
    if not isinstance(grid, list) or not grid:
        return (0, 0)
    first = grid[0]
    if not isinstance(first, list):
        return (0, 0)
    return (len(first), len(grid))


def cells(grid: Grid) -> list[int]:
    out: list[int] = []
    for row in grid:
        out.extend(int(value) for value in row)
    return out


def train_pairs(task: dict[str, Any]) -> list[tuple[Grid, Grid]]:
    pairs: list[tuple[Grid, Grid]] = []
    for pair in task.get("train", []) or []:
        input_grid = pair.get("input")
        output_grid = pair.get("output")
        if isinstance(input_grid, list) and isinstance(output_grid, list):
            pairs.append((input_grid, output_grid))
    return pairs


def palette_size(pairs: list[tuple[Grid, Grid]]) -> int:
    palette: set[int] = set()
    for input_grid, output_grid in pairs:
        palette.update(cells(input_grid))
        palette.update(cells(output_grid))
    return len(palette)


def consistent_color_map(pairs: list[tuple[Grid, Grid]]) -> tuple[bool, bool, dict[int, int]]:
    mapping: dict[int, int] = {}
    changed = False
    if not pairs:
        return (False, False, {})
    for input_grid, output_grid in pairs:
        if grid_dims(input_grid) != grid_dims(output_grid):
            return (False, False, {})
        input_cells = cells(input_grid)
        output_cells = cells(output_grid)
        if len(input_cells) != len(output_cells):
            return (False, False, {})
        for input_color, output_color in zip(input_cells, output_cells):
            previous = mapping.get(input_color)
            if previous is None:
                mapping[input_color] = output_color
            elif previous != output_color:
                return (False, False, {})
            if input_color != output_color:
                changed = True
    return (True, changed, mapping)


def full_separator_colors(grid: Grid) -> set[int]:
    width, height = grid_dims(grid)
    colors: set[int] = set()
    if width == 0 or height == 0:
        return colors
    for row in grid:
        if row and all(value == row[0] for value in row):
            colors.add(int(row[0]))
    for x in range(width):
        color = int(grid[0][x])
        if all(int(grid[y][x]) == color for y in range(height)):
            colors.add(color)
    return colors


def classify_task(task: dict[str, Any]) -> dict[str, Any]:
    pairs = train_pairs(task)
    shape_pairs = [
        {
            "input": list(grid_dims(input_grid)),
            "output": list(grid_dims(output_grid)),
        }
        for input_grid, output_grid in pairs
    ]
    same_shape = bool(pairs) and all(
        grid_dims(input_grid) == grid_dims(output_grid)
        for input_grid, output_grid in pairs
    )
    all_expand = bool(pairs) and all(
        grid_dims(output_grid)[0] >= grid_dims(input_grid)[0]
        and grid_dims(output_grid)[1] >= grid_dims(input_grid)[1]
        and grid_dims(output_grid) != grid_dims(input_grid)
        for input_grid, output_grid in pairs
    )
    all_crop = bool(pairs) and all(
        grid_dims(output_grid)[0] <= grid_dims(input_grid)[0]
        and grid_dims(output_grid)[1] <= grid_dims(input_grid)[1]
        and grid_dims(output_grid) != grid_dims(input_grid)
        for input_grid, output_grid in pairs
    )
    any_one_dim_compress = any(
        grid_dims(output_grid)[0] < grid_dims(input_grid)[0]
        or grid_dims(output_grid)[1] < grid_dims(input_grid)[1]
        for input_grid, output_grid in pairs
    )
    has_separator = any(full_separator_colors(input_grid) for input_grid, _ in pairs)
    color_map_ok, color_map_changed, mapping = consistent_color_map(pairs)

    if color_map_ok and color_map_changed:
        label = "same_grid_color_map"
        priority = "promote_color_map_hypothesis"
    elif all_expand:
        label = "shape_expansion_or_tiling"
        priority = "add_tiling_scale_or_canvas_growth_templates"
    elif all_crop and has_separator:
        label = "panel_crop_or_separator_extract"
        priority = "add_separator_panel_extraction_templates"
    elif all_crop:
        label = "crop_or_object_extract"
        priority = "broaden_crop_object_selection_templates"
    elif same_shape:
        label = "same_shape_object_or_lattice"
        priority = "mine_same_shape_patch_lattice_templates"
    elif any_one_dim_compress:
        label = "line_compression_or_projection"
        priority = "add_projection_compression_templates"
    else:
        label = "mixed_shape_transform"
        priority = "inspect_for_compositional_template"

    changed_mappings = {
        str(input_color): output_color
        for input_color, output_color in sorted(mapping.items())
        if input_color != output_color
    }
    return {
        "train_pairs": len(pairs),
        "palette_size": palette_size(pairs),
        "shape_pairs": shape_pairs,
        "same_shape_all": same_shape,
        "has_full_row_or_column_separator": has_separator,
        "consistent_color_map": color_map_ok and color_map_changed,
        "changed_color_map": changed_mappings,
        "structural_label": label,
        "priority": priority,
    }


def row_key(row: dict[str, Any]) -> Key:
    return (
        str(row.get("split", "")),
        str(row.get("mode", "")),
        str(row.get("task_id", "")),
    )


def build_rows(
    public_arc_dir: Path,
    diagnostics_path: Path,
    buckets_path: Path,
) -> list[dict[str, Any]]:
    split_dirs = public_split_dirs(public_arc_dir)
    diagnostics = {row_key(row): row for row in parse_json_rows(diagnostics_path)}
    buckets = {row_key(row): row for row in parse_json_rows(buckets_path)}
    rows: list[dict[str, Any]] = []

    for key in sorted(set(diagnostics) | set(buckets)):
        split, mode, task_id = key
        if not task_id:
            continue
        task_path = split_dirs.get(split, public_arc_dir) / f"{task_id}.json"
        task_features: dict[str, Any]
        if task_path.is_file():
            with task_path.open("r", encoding="utf-8") as fp:
                task_features = classify_task(json.load(fp))
        else:
            task_features = {
                "train_pairs": 0,
                "palette_size": 0,
                "shape_pairs": [],
                "same_shape_all": False,
                "has_full_row_or_column_separator": False,
                "consistent_color_map": False,
                "changed_color_map": {},
                "structural_label": "task_file_missing",
                "priority": "resolve_public_data_path",
            }

        diagnostic = diagnostics.get(key, {})
        bucket = buckets.get(key, {})
        selected_templates = [
            str(value)
            for value in diagnostic.get(
                "selected_templates", bucket.get("selected_templates", [])
            )
            or []
        ]
        train_consistent_templates = [
            str(value)
            for value in diagnostic.get(
                "train_consistent_templates",
                bucket.get("train_consistent_templates", []),
            )
            or []
        ]
        rows.append(
            {
                "schema": "crlp.arc_public_coverage_triage.v1",
                "report_version": 1,
                "task_id": task_id,
                "split": split,
                "mode": mode,
                "diagnostic_bucket": str(bucket.get("bucket", "")),
                "diagnostic_bucket_label": str(bucket.get("bucket_label", "")),
                "certified": bool(diagnostic.get("certified", bucket.get("certified", False))),
                "selected_templates": selected_templates,
                "train_consistent_templates": train_consistent_templates,
                "certified_evidence": safe_int(
                    diagnostic.get("certified_evidence", bucket.get("certified_evidence"))
                ),
                "rejected_evidence": safe_int(
                    diagnostic.get("rejected_evidence", bucket.get("rejected_evidence"))
                ),
                "proposals_generated": safe_int(
                    diagnostic.get("proposals_generated", bucket.get("proposals_generated"))
                ),
                "proposals_train_consistent_pre_cap": safe_int(
                    diagnostic.get(
                        "proposals_train_consistent_pre_cap",
                        bucket.get("proposals_train_consistent_pre_cap"),
                    )
                ),
                "proposal_cap_hit": bool(
                    diagnostic.get("proposal_cap_hit", bucket.get("proposal_cap_hit", False))
                ),
                **task_features,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "task_id",
        "split",
        "mode",
        "certified",
        "diagnostic_bucket",
        "structural_label",
        "priority",
        "proposals_generated",
        "proposals_train_consistent_pre_cap",
        "selected_templates",
        "train_consistent_templates",
        "consistent_color_map",
        "changed_color_map",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row.get(key) for key in fieldnames},
                    "selected_templates": ",".join(row["selected_templates"]),
                    "train_consistent_templates": ",".join(
                        row["train_consistent_templates"]
                    ),
                    "changed_color_map": json.dumps(
                        row["changed_color_map"], sort_keys=True
                    ),
                }
            )


def pct(count: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{100.0 * count / total:.1f}%"


def markdown_summary(rows: list[dict[str, Any]], top_k: int) -> str:
    by_mode: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[(row["split"], row["mode"])].append(row)

    lines = [
        "# ARC Public Coverage Triage",
        "",
        "This report joins public-filter diagnostics with structural task labels.",
        "Use it to choose the next hypothesis templates from failures where the",
        "framework currently has no meaningful proposal surface.",
        "",
        "## Structural Labels",
        "",
        "| Split | Mode | Tasks | Certified | Color map | Expansion/tiling | Crop/extract | Same-shape lattice | Compression/projection | Mixed |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in sorted(by_mode):
        group = by_mode[key]
        total = len(group)
        counts = Counter(row["structural_label"] for row in group)
        certified = sum(1 for row in group if row["certified"])
        lines.append(
            "| {} | `{}` | {} | {} ({}) | {} | {} | {} | {} | {} | {} |".format(
                key[0],
                key[1],
                total,
                certified,
                pct(certified, total),
                counts["same_grid_color_map"],
                counts["shape_expansion_or_tiling"],
                counts["crop_or_object_extract"] + counts["panel_crop_or_separator_extract"],
                counts["same_shape_object_or_lattice"],
                counts["line_compression_or_projection"],
                counts["mixed_shape_transform"],
            )
        )

    lines.extend(["", "## Failure Wedges", ""])
    failures = [row for row in rows if not row["certified"]]
    wedge_counts = Counter(row["priority"] for row in failures)
    if wedge_counts:
        lines.extend(["| Priority | Failures |", "| --- | ---: |"])
        for priority, count in wedge_counts.most_common():
            lines.append(f"| `{priority}` | {count} |")
    else:
        lines.append("No uncertified rows in diagnostics.")

    lines.extend(["", "## Top No-Proposal Examples", ""])
    no_proposal = [
        row
        for row in failures
        if row["diagnostic_bucket"] == "no_meaningful_proposals"
    ]
    no_proposal.sort(
        key=lambda row: (
            row["priority"],
            -safe_int(row["proposals_generated"]),
            row["task_id"],
        )
    )
    if no_proposal:
        lines.extend(
            [
                "| Task | Split | Mode | Structural label | Priority | Shape pairs | Color map |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in no_proposal[:top_k]:
            lines.append(
                "| `{}` | {} | `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                    row["task_id"],
                    row["split"],
                    row["mode"],
                    row["structural_label"],
                    row["priority"],
                    json.dumps(row["shape_pairs"], separators=(",", ":")),
                    json.dumps(row["changed_color_map"], sort_keys=True),
                )
            )
    else:
        lines.append("No no-meaningful-proposal rows found.")

    lines.extend(["", "## Top Color-Map Candidates", ""])
    color_map_rows = [
        row
        for row in rows
        if row["consistent_color_map"] and "same_grid_color_map" not in row["selected_templates"]
    ]
    color_map_rows.sort(key=lambda row: (row["certified"], row["split"], row["task_id"]))
    if color_map_rows:
        lines.extend(
            [
                "| Task | Split | Mode | Certified | Bucket | Map |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in color_map_rows[:top_k]:
            lines.append(
                "| `{}` | {} | `{}` | {} | `{}` | `{}` |".format(
                    row["task_id"],
                    row["split"],
                    row["mode"],
                    row["certified"],
                    row["diagnostic_bucket"],
                    json.dumps(row["changed_color_map"], sort_keys=True),
                )
            )
    else:
        lines.append("No uncovered same-grid color-map candidates found.")

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
        "--buckets",
        type=Path,
        default=Path(
            "build/float/hypothesis_library_public_filter/arc_public_diagnostics_buckets.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/float/hypothesis_library_public_triage"),
    )
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    rows = build_rows(args.public_arc_dir, args.diagnostics, args.buckets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "coverage_triage.jsonl", rows)
    write_csv(args.output_dir / "coverage_triage.csv", rows)
    (args.output_dir / "coverage_triage_summary.md").write_text(
        markdown_summary(rows, args.top_k),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} triage rows to {args.output_dir}")


if __name__ == "__main__":
    main()
