#!/usr/bin/env python3
"""Promote repeated certified hypotheses into library-entry candidates."""

from __future__ import annotations

import argparse
import collections
import json
from datetime import date
from pathlib import Path
from typing import Any

PRIOR_SCHEMA = "crlp.hypothesis_library.prior.v1"


def parse_json_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line.startswith("{"):
                continue
            rows.append(json.loads(line))
    return rows


def slot_key(slot: dict[str, Any]) -> tuple[str, str]:
    return str(slot.get("name", "")), str(slot.get("type", "unknown"))


def slot_value(slot: dict[str, Any]) -> dict[str, Any]:
    return {
        "value": slot.get("value"),
        "integer": slot.get("integer"),
        "bound": bool(slot.get("bound", False)),
    }


def distinct_slot_values(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for value in values:
        key = json.dumps(value, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def group_evidence(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        if row.get("schema") != "crlp.hypothesis.evidence.v1":
            continue
        if row.get("certificate_status") != "certified":
            continue
        key = (
            row.get("template_domain", ""),
            row.get("template_id", 0),
            row.get("template_name", ""),
            row.get("proof_kind", ""),
            row.get("reason", ""),
        )
        groups[key].append(row)
    return groups


def slot_integer(slot: dict[str, Any]) -> int:
    value = slot.get("integer", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def slot_numeric_value(slot: dict[str, Any]) -> float:
    value = slot.get("value", slot.get("integer", 0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def promotion_row(
    key: tuple[Any, ...],
    rows: list[dict[str, Any]],
    min_support_tasks: int,
    min_covered_events: int,
) -> dict[str, Any]:
    domain, template_id, template_name, proof_kind, reason = key
    task_instances = sorted({str(row.get("task_instance", "")) for row in rows})
    false_rejections = sum(int(row.get("false_rejections", 0) or 0) for row in rows)
    covered_events = sum(int(row.get("covered_events", 0) or 0) for row in rows)
    event_count = sum(int(row.get("event_count", 0) or 0) for row in rows)
    support = sum(int(row.get("support", 0) or 0) for row in rows)
    utility_sum = sum(float(row.get("utility", 0.0) or 0.0) for row in rows)

    slot_values: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        for slot in row.get("slots", []) or []:
            slot_values[slot_key(slot)].append(slot_value(slot))
    invariant_slots: list[dict[str, Any]] = []
    variable_slots: list[dict[str, Any]] = []
    for (name, slot_type), values in sorted(slot_values.items()):
        distinct = distinct_slot_values(values)
        target = invariant_slots if len(distinct) == 1 else variable_slots
        target.append(
            {
                "name": name,
                "type": slot_type,
                "distinct_count": len(distinct),
                "values": distinct[:8],
            }
        )

    promoted = (
        len(task_instances) >= min_support_tasks
        and false_rejections == 0
        and covered_events >= min_covered_events
    )
    promotion_reason = (
        "meets support and zero-false-rejection thresholds"
        if promoted
        else "below promotion threshold"
    )
    library_id = f"{domain}:{template_name}:library:{len(task_instances)}x:{covered_events}"
    return {
        "schema": "crlp.hypothesis_library.promotion.v1",
        "report_version": 1,
        "date": date.today().isoformat(),
        "template_domain": domain,
        "template_id": template_id,
        "template_name": template_name,
        "proof_kind": proof_kind,
        "certificate_reason": reason,
        "library_id": library_id,
        "promoted": promoted,
        "promotion_reason": promotion_reason,
        "certified_hypotheses": len(rows),
        "support_tasks": len(task_instances),
        "task_instances": task_instances,
        "event_count": event_count,
        "covered_events": covered_events,
        "support": support,
        "false_rejections": false_rejections,
        "utility_mean": utility_sum / len(rows) if rows else 0.0,
        "invariant_slots": invariant_slots,
        "variable_slots": variable_slots,
    }


def prior_rows(
    groups: dict[tuple[Any, ...], list[dict[str, Any]]],
    promotions_by_key: dict[tuple[Any, ...], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_slot: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, evidence_rows in groups.items():
        promotion = promotions_by_key.get(key)
        if promotion is None or not promotion["promoted"]:
            continue
        for row in evidence_rows:
            for slot in row.get("slots", []) or []:
                if not bool(slot.get("bound", False)):
                    continue
                name, slot_type = slot_key(slot)
                slot_int = slot_integer(slot)
                slot_value_float = slot_numeric_value(slot)
                prior_key = (
                    promotion["template_id"],
                    promotion["template_name"],
                    promotion["library_id"],
                    name,
                    slot_type,
                    slot_int,
                    f"{slot_value_float:.8f}",
                )
                prior = rows_by_slot.get(prior_key)
                if prior is None:
                    prior = {
                        "schema_version": PRIOR_SCHEMA,
                        "template_id": promotion["template_id"],
                        "template_name": promotion["template_name"],
                        "library_id": promotion["library_id"],
                        "slot_name": name,
                        "slot_type": slot_type,
                        "slot_value": f"{slot_value_float:.8f}",
                        "slot_integer": slot_int,
                        "proposal_count": 0,
                        "certified_count": 0,
                        "support_tasks": promotion["support_tasks"],
                        "covered_events": promotion["covered_events"],
                        "utility_mean": promotion["utility_mean"],
                    }
                    rows_by_slot[prior_key] = prior
                prior["proposal_count"] += 1
                prior["certified_count"] += 1
    return sorted(
        rows_by_slot.values(),
        key=lambda row: (
            int(row["template_id"]),
            str(row["library_id"]),
            str(row["slot_name"]),
            int(row["slot_integer"]),
        ),
    )


def selection_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    library_keys = {
        "library_prior_entries",
        "library_prior_slot_rows",
        "library_prior_seeded_proposals",
        "library_prior_seeded_certified",
    }
    max_keys = library_keys | {"hypothesis_prior_hidden"}
    out: dict[str, Any] = {
        "selection_rows": 0,
        "library_prior_entries": 0,
        "library_prior_slot_rows": 0,
        "library_prior_seeded_proposals": 0,
        "library_prior_seeded_certified": 0,
        "hypothesis_prior_hidden": 0,
        "hypothesis_top_k": 0,
        "hypothesis_neural_prior_enabled": False,
        "hypothesis_slot_prior_enabled": False,
        "hypothesis_cold_start_enabled": False,
        "hypothesis_structural_score_enabled": False,
        "hypothesis_coverage_selection_enabled": False,
        "hypothesis_train_filter_enabled": False,
        "proposals_attempted": 0,
        "proposals_generated": 0,
        "proposals_train_consistent": 0,
        "proposals_train_rejected": 0,
        "proposals_selected": 0,
    }
    for row in rows:
        if row.get("schema") != "crlp.arc_hypothesis.selection.v1":
            continue
        out["selection_rows"] += 1
        out["hypothesis_neural_prior_enabled"] = (
            out["hypothesis_neural_prior_enabled"]
            or bool(row.get("hypothesis_neural_prior_enabled", False))
        )
        out["hypothesis_slot_prior_enabled"] = (
            out["hypothesis_slot_prior_enabled"]
            or bool(row.get("hypothesis_slot_prior_enabled", False))
        )
        out["hypothesis_cold_start_enabled"] = (
            out["hypothesis_cold_start_enabled"]
            or bool(row.get("hypothesis_cold_start_enabled", False))
        )
        out["hypothesis_structural_score_enabled"] = (
            out["hypothesis_structural_score_enabled"]
            or bool(row.get("hypothesis_structural_score_enabled", False))
        )
        out["hypothesis_coverage_selection_enabled"] = (
            out["hypothesis_coverage_selection_enabled"]
            or bool(row.get("hypothesis_coverage_selection_enabled", False))
        )
        out["hypothesis_train_filter_enabled"] = (
            out["hypothesis_train_filter_enabled"]
            or bool(row.get("hypothesis_train_filter_enabled", False))
        )
        for key in (
            "library_prior_entries",
            "library_prior_slot_rows",
            "library_prior_seeded_proposals",
            "library_prior_seeded_certified",
            "hypothesis_prior_hidden",
            "top_k",
            "proposals_attempted",
            "proposals_generated",
            "proposals_train_consistent",
            "proposals_train_rejected",
            "proposals_selected",
        ):
            value = int(row.get(key, 0) or 0)
            if key in max_keys:
                out[key] = max(out[key], value)
            elif key == "top_k":
                out["hypothesis_top_k"] = max(out["hypothesis_top_k"], value)
            else:
                out[key] += value
    return out


def program_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "program_rows": 0,
        "learned_by_budget": {},
    }
    learned: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("schema") != "crlp.program_proposer.summary.v1":
            continue
        out["program_rows"] += 1
        if row.get("strategy") != "learned":
            continue
        budget = int(row.get("budget", 0) or 0)
        learned[budget] = {
            "certified_programs": int(row.get("certified_programs", 0) or 0),
            "accepted_programs": int(row.get("accepted_programs", 0) or 0),
            "covered_events": int(row.get("covered_events", 0) or 0),
            "utility": float(row.get("utility", 0.0) or 0.0),
        }
    out["learned_by_budget"] = learned
    return out


def write_prior_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
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
    with path.open("w", encoding="utf-8") as fp:
        fp.write("\t".join(fields) + "\n")
        for row in rows:
            fp.write("\t".join(str(row[field]) for field in fields) + "\n")


def append_selection_lines(lines: list[str], title: str, selection: dict[str, Any]) -> None:
    lines.extend([title, ""])
    for key in sorted(selection):
        lines.append(f"- `{key}`: {selection[key]}")
    lines.append("")


def append_feedback_table(
    lines: list[str],
    baseline: dict[str, Any],
    feedback: dict[str, Any] | None,
) -> None:
    if feedback is None:
        return
    lines.extend(
        [
            "## Proposal Feedback Check",
            "",
            "| Metric | Baseline | Library prior | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    metrics = [
        "library_prior_slot_rows",
        "library_prior_seeded_certified",
        "proposals_attempted",
        "proposals_generated",
        "proposals_train_consistent",
        "proposals_train_rejected",
        "proposals_selected",
    ]
    for metric in metrics:
        base_value = int(baseline.get(metric, 0) or 0)
        feedback_value = int(feedback.get(metric, 0) or 0)
        lines.append(
            f"| `{metric}` | {base_value} | {feedback_value} | {feedback_value - base_value} |"
        )
    lines.append("")


def parse_labeled_path(text: str) -> tuple[str, Path]:
    for separator in ("=", ":"):
        if separator in text:
            label, path = text.split(separator, 1)
            label = label.strip()
            if label and path:
                return label, Path(path)
    raise ValueError(f"expected label=path ablation input, got {text!r}")


def prior_mode(selection: dict[str, Any]) -> str:
    parts: list[str] = []
    if selection.get("hypothesis_structural_score_enabled"):
        parts.append("struct")
    if selection.get("hypothesis_coverage_selection_enabled"):
        parts.append("coverage")
    if selection.get("hypothesis_train_filter_enabled"):
        parts.append("train_filter")
    if selection.get("hypothesis_slot_prior_enabled"):
        parts.append("slot")
    if selection.get("hypothesis_neural_prior_enabled"):
        hidden = int(selection.get("hypothesis_prior_hidden", 0) or 0)
        parts.append("linear" if hidden == 0 else f"mlp{hidden}")
    if selection.get("hypothesis_cold_start_enabled"):
        parts.append("cold")
    return "+".join(parts) if parts else "none"


def learned_budget_value(program: dict[str, Any], budget: int, key: str) -> Any:
    learned = program.get("learned_by_budget", {})
    return learned.get(budget, {}).get(key, 0)


def append_ablation_table(
    lines: list[str],
    ablations: list[tuple[str, dict[str, Any], dict[str, Any]]],
) -> None:
    if not ablations:
        return
    lines.extend(
        [
            "## Capacity Ablation Check",
            "",
            "| Run | Mode | Top-k | Prior rows | Seeded cert | Selected | Learned certified @1 | @2 | @4 | @8 | Learned utility @8 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, selection, program in ablations:
        lines.append(
            "| {label} | `{mode}` | {top_k} | {rows} | {seeded} | {selected} | "
            "{cert1} | {cert2} | {cert4} | {cert8} | {utility8:.4f} |".format(
                label=label,
                mode=prior_mode(selection),
                top_k=int(selection.get("hypothesis_top_k", 0) or 0),
                rows=int(selection.get("library_prior_slot_rows", 0) or 0),
                seeded=int(selection.get("library_prior_seeded_certified", 0) or 0),
                selected=int(selection.get("proposals_selected", 0) or 0),
                cert1=int(learned_budget_value(program, 1, "certified_programs") or 0),
                cert2=int(learned_budget_value(program, 2, "certified_programs") or 0),
                cert4=int(learned_budget_value(program, 4, "certified_programs") or 0),
                cert8=int(learned_budget_value(program, 8, "certified_programs") or 0),
                utility8=float(learned_budget_value(program, 8, "utility") or 0.0),
            )
        )
    lines.append("")


def write_markdown(
    path: Path,
    promotions: list[dict[str, Any]],
    selection: dict[str, Any],
    prior_row_count: int,
    feedback_selection: dict[str, Any] | None,
    ablations: list[tuple[str, dict[str, Any], dict[str, Any]]],
) -> None:
    lines = [
        "# Certified Hypothesis Library Promotion",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "This experiment groups certified `crlp.hypothesis.evidence.v1` rows by template, "
        "proof kind, and certificate reason. Repeated zero-false-rejection groups become "
        "library-entry candidates with invariant and variable slot summaries, then exports "
        "certified slot priors that can seed ARC hypothesis proposal ranking.",
        "",
        f"Prior rows exported: `{prior_row_count}`",
        "",
    ]
    append_selection_lines(lines, "## Promotion Source Selection Summary", selection)
    append_feedback_table(lines, selection, feedback_selection)
    append_ablation_table(lines, ablations)
    lines.extend(
        [
            "## Promotion Candidates",
            "",
            "| Template | Promoted | Tasks | Covered events | Variable slots | Library id |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in promotions:
        lines.append(
            "| {template} | {promoted} | {tasks} | {covered} | {vars} | `{library}` |".format(
                template=row["template_name"],
                promoted="yes" if row["promoted"] else "no",
                tasks=row["support_tasks"],
                covered=row["covered_events"],
                vars=len(row["variable_slots"]),
                library=row["library_id"],
            )
        )
    lines.extend(
        [
            "",
            "## Trace Contract",
            "",
            "- Candidate set: certified hypothesis evidence rows.",
            "- Verifier: domain certificate checker that produced the source evidence.",
            "- Certificate/gate: zero false rejections plus proof-kind and reason agreement.",
            "- Scorer role: promoted entries seed the ARC hypothesis slot prior before proposal ranking.",
            "- Trace artifact: `crlp.hypothesis_library.promotion.v1` rows plus `crlp.hypothesis_library.prior.v1` TSV rows.",
            "- Protocol point: repeated successful slot bindings become explicit library objects.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("build/float/hypothesis_library"))
    parser.add_argument("--min-support-tasks", type=int, default=2)
    parser.add_argument("--min-covered-events", type=int, default=1)
    parser.add_argument("--feedback-input", type=Path)
    parser.add_argument(
        "--ablation-input",
        action="append",
        default=[],
        help="label=path JSONL run to include in the capacity ablation table",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = parse_json_rows(args.input)
    groups = group_evidence(rows)
    promotions: list[dict[str, Any]] = []
    promotions_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, group in groups.items():
        row = promotion_row(key, group, args.min_support_tasks, args.min_covered_events)
        promotions.append(row)
        promotions_by_key[key] = row
    promotions.sort(
        key=lambda row: (
            not row["promoted"],
            -int(row["support_tasks"]),
            -int(row["covered_events"]),
            str(row["template_name"]),
        )
    )

    jsonl_path = args.output_dir / "promotions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fp:
        for row in promotions:
            fp.write(json.dumps(row, sort_keys=True) + "\n")
    priors = prior_rows(groups, promotions_by_key)
    prior_path = args.output_dir / "library_prior.tsv"
    write_prior_tsv(prior_path, priors)
    feedback_selection = (
        selection_summary(parse_json_rows(args.feedback_input))
        if args.feedback_input is not None and args.feedback_input.exists()
        else None
    )
    ablations: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for labeled_input in args.ablation_input:
        label, path = parse_labeled_path(labeled_input)
        if not path.exists():
            raise FileNotFoundError(path)
        ablation_rows = parse_json_rows(path)
        ablations.append(
            (label, selection_summary(ablation_rows), program_summary(ablation_rows))
        )
    write_markdown(
        args.output_dir / "summary.md",
        promotions,
        selection_summary(rows),
        len(priors),
        feedback_selection,
        ablations,
    )
    print(f"wrote {jsonl_path}")
    print(f"wrote {prior_path}")
    print(f"wrote {args.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
