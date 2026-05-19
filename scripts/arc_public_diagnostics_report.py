#!/usr/bin/env python3
"""Summarize ARC public diagnostics into actionable failure buckets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

BUCKET_ORDER = [
    "certified",
    "train_consistent_not_selected",
    "selected_hypothesis_rejected",
    "proposal_cap_no_train_consistent",
    "no_meaningful_proposals",
]

BUCKET_LABELS = {
    "certified": "Certified",
    "train_consistent_not_selected": "Train-consistent hypothesis not selected",
    "selected_hypothesis_rejected": "Selected hypothesis rejected",
    "proposal_cap_no_train_consistent": "Proposal cap hit, no train-consistent hypothesis",
    "no_meaningful_proposals": "No meaningful proposals",
}


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def parse_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line.startswith("{"):
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            fp.write("\n")


def overlap(left: list[str], right: list[str]) -> list[str]:
    return sorted(set(left) & set(right))


def bucket_for(row: dict[str, Any]) -> str:
    if bool(row.get("certified", False)):
        return "certified"

    selected = [str(value) for value in row.get("selected_templates", []) or []]
    train_consistent = [
        str(value) for value in row.get("train_consistent_templates", []) or []
    ]
    train_consistent_count = safe_int(row.get("proposals_train_consistent_pre_cap"))
    rejected_count = safe_int(row.get("rejected_evidence"))
    cap_hit = bool(row.get("proposal_cap_hit", False))

    if train_consistent_count > 0 and not overlap(selected, train_consistent):
        return "train_consistent_not_selected"
    if rejected_count > 0:
        return "selected_hypothesis_rejected"
    if cap_hit and train_consistent_count == 0:
        return "proposal_cap_no_train_consistent"
    return "no_meaningful_proposals"


def bucket_row(row: dict[str, Any]) -> dict[str, Any]:
    bucket = bucket_for(row)
    selected = [str(value) for value in row.get("selected_templates", []) or []]
    train_consistent = [
        str(value) for value in row.get("train_consistent_templates", []) or []
    ]
    return {
        "schema": "crlp.arc_public_diagnostics_bucket.v1",
        "report_version": 1,
        "task_id": str(row.get("task_id", "")),
        "split": str(row.get("split", "")),
        "mode": str(row.get("mode", "")),
        "bucket": bucket,
        "bucket_label": BUCKET_LABELS[bucket],
        "certified": bool(row.get("certified", False)),
        "selected_templates": selected,
        "train_consistent_templates": train_consistent,
        "overlap_templates": overlap(selected, train_consistent),
        "certified_evidence": safe_int(row.get("certified_evidence")),
        "rejected_evidence": safe_int(row.get("rejected_evidence")),
        "proposals_generated": safe_int(row.get("proposals_generated")),
        "proposals_selected": safe_int(row.get("proposals_selected")),
        "proposals_train_consistent_pre_cap": safe_int(
            row.get("proposals_train_consistent_pre_cap")
        ),
        "proposal_cap_hit": bool(row.get("proposal_cap_hit", False)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "task_id",
        "split",
        "mode",
        "bucket",
        "bucket_label",
        "certified",
        "proposals_generated",
        "proposals_train_consistent_pre_cap",
        "proposal_cap_hit",
        "selected_templates",
        "train_consistent_templates",
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
                }
            )


def pct(count: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{(100.0 * count / total):.1f}%"


def top_tasks(rows: list[dict[str, Any]], bucket: str, limit: int) -> list[dict[str, Any]]:
    matches = [row for row in rows if row["bucket"] == bucket]
    matches.sort(
        key=lambda row: (
            -safe_int(row["proposals_train_consistent_pre_cap"]),
            -safe_int(row["proposals_generated"]),
            str(row["task_id"]),
        )
    )
    return matches[:limit]


def markdown_summary(rows: list[dict[str, Any]], limit: int) -> str:
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[(row["split"], row["mode"])].append(row)

    lines = [
        "# ARC Public Diagnostics Report",
        "",
        "Buckets are priority ordered: certified first, then rank/search misses where",
        "train-consistent hypotheses existed but were not selected, then selected",
        "hypotheses rejected by the verifier, then cap-hit searches with no",
        "train-consistent hypothesis, then low-signal proposal failures.",
        "",
        "## Bucket Counts",
        "",
        "| Split | Mode | Tasks | Certified | TC not selected | Selected rejected | Cap no TC | No meaningful proposals |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in sorted(by_group):
        group_rows = by_group[key]
        counts = Counter(row["bucket"] for row in group_rows)
        total = len(group_rows)
        split, mode = key
        lines.append(
            "| {} | `{}` | {} | {} ({}) | {} ({}) | {} ({}) | {} ({}) | {} ({}) |".format(
                split,
                mode,
                total,
                counts["certified"],
                pct(counts["certified"], total),
                counts["train_consistent_not_selected"],
                pct(counts["train_consistent_not_selected"], total),
                counts["selected_hypothesis_rejected"],
                pct(counts["selected_hypothesis_rejected"], total),
                counts["proposal_cap_no_train_consistent"],
                pct(counts["proposal_cap_no_train_consistent"], total),
                counts["no_meaningful_proposals"],
                pct(counts["no_meaningful_proposals"], total),
            )
        )

    lines.extend(["", "## Top Ranking/Search Misses", ""])
    ranking_rows = [
        row for row in rows if row["bucket"] == "train_consistent_not_selected"
    ]
    if ranking_rows:
        lines.extend(
            [
                "| Split | Mode | Task | Train-consistent | Selected | TC pre-cap | Generated |",
                "| --- | --- | --- | --- | --- | ---: | ---: |",
            ]
        )
        ranking_rows.sort(
            key=lambda row: (
                row["split"] != "public_evaluation",
                row["mode"] != "stress_no_prior",
                -safe_int(row["proposals_train_consistent_pre_cap"]),
                str(row["task_id"]),
            )
        )
        for row in ranking_rows[:limit]:
            lines.append(
                "| {} | `{}` | `{}` | `{}` | `{}` | {} | {} |".format(
                    row["split"],
                    row["mode"],
                    row["task_id"],
                    ",".join(row["train_consistent_templates"]) or "none",
                    ",".join(row["selected_templates"]) or "none",
                    row["proposals_train_consistent_pre_cap"],
                    row["proposals_generated"],
                )
            )
    else:
        lines.append("No train-consistent-not-selected tasks found.")

    lines.extend(["", "## Top Selected-Rejected Failures", ""])
    rejected_rows = [
        row for row in rows if row["bucket"] == "selected_hypothesis_rejected"
    ]
    if rejected_rows:
        lines.extend(
            [
                "| Split | Mode | Task | Selected | Generated | TC pre-cap |",
                "| --- | --- | --- | --- | ---: | ---: |",
            ]
        )
        rejected_rows.sort(
            key=lambda row: (
                row["split"] != "public_evaluation",
                row["mode"] != "stress_no_prior",
                -safe_int(row["proposals_generated"]),
                str(row["task_id"]),
            )
        )
        for row in rejected_rows[:limit]:
            lines.append(
                "| {} | `{}` | `{}` | `{}` | {} | {} |".format(
                    row["split"],
                    row["mode"],
                    row["task_id"],
                    ",".join(row["selected_templates"]) or "none",
                    row["proposals_generated"],
                    row["proposals_train_consistent_pre_cap"],
                )
            )
    else:
        lines.append("No selected-rejected failures found.")

    lines.extend(["", "## Top Template-Coverage Failures", ""])
    coverage_rows = [
        *top_tasks(rows, "proposal_cap_no_train_consistent", limit),
        *top_tasks(rows, "no_meaningful_proposals", limit),
    ][:limit]
    if coverage_rows:
        lines.extend(
            [
                "| Split | Mode | Task | Bucket | Generated | Cap hit | Selected |",
                "| --- | --- | --- | --- | ---: | --- | --- |",
            ]
        )
        for row in coverage_rows:
            lines.append(
                "| {} | `{}` | `{}` | {} | {} | {} | `{}` |".format(
                    row["split"],
                    row["mode"],
                    row["task_id"],
                    BUCKET_LABELS[row["bucket"]],
                    row["proposals_generated"],
                    "yes" if row["proposal_cap_hit"] else "no",
                    ",".join(row["selected_templates"]) or "none",
                )
            )
    else:
        lines.append("No template-coverage failures found.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [bucket_row(row) for row in parse_rows(args.diagnostics)]
    write_jsonl(args.output_dir / "arc_public_diagnostics_buckets.jsonl", rows)
    write_csv(args.output_dir / "arc_public_diagnostics_buckets.csv", rows)
    (args.output_dir / "arc_public_diagnostics_report.md").write_text(
        markdown_summary(rows, args.top_k),
        encoding="utf-8",
    )
    print(f"wrote {args.output_dir / 'arc_public_diagnostics_report.md'}")
    print(f"wrote {args.output_dir / 'arc_public_diagnostics_buckets.jsonl'}")
    print(f"wrote {args.output_dir / 'arc_public_diagnostics_buckets.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
