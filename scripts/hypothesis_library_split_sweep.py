#!/usr/bin/env python3
"""Sweep certified hypothesis-library priors across held-out ARC fixture splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALL_TASKS = [
    "bbox_panel_selection",
    "connect_aligned",
    "extend_aligned_sequence",
    "fill_enclosed",
    "move_to_anchor",
    "move_to_anchor_both_positive",
    "move_to_anchor_x",
    "move_to_anchor_y",
    "object_extract",
    "panel_selection",
    "recolor_by_reference",
    "translate",
]

PROMOTION_FAMILY = [
    "move_to_anchor",
    "move_to_anchor_both_positive",
    "move_to_anchor_x",
    "move_to_anchor_y",
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


@dataclass(frozen=True)
class Mode:
    name: str
    args: tuple[str, ...]
    primary: bool


MODES = [
    Mode(
        "no_prior",
        (
            "--disable-hypothesis-slot-prior",
            "--disable-hypothesis-neural-prior",
            "--disable-hypothesis-cold-start",
        ),
        True,
    ),
    Mode(
        "slot_only",
        (
            "--hypothesis-library-prior",
            "{prior}",
            "--disable-hypothesis-neural-prior",
            "--disable-hypothesis-cold-start",
        ),
        True,
    ),
    Mode(
        "linear_prior",
        (
            "--hypothesis-library-prior",
            "{prior}",
            "--hypothesis-prior-hidden",
            "0",
            "--disable-hypothesis-cold-start",
        ),
        True,
    ),
    Mode(
        "tiny_mlp4",
        (
            "--hypothesis-library-prior",
            "{prior}",
            "--hypothesis-prior-hidden",
            "4",
            "--disable-hypothesis-cold-start",
        ),
        False,
    ),
    Mode(
        "full_mlp24",
        (
            "--hypothesis-library-prior",
            "{prior}",
            "--hypothesis-prior-hidden",
            "24",
            "--disable-hypothesis-cold-start",
        ),
        False,
    ),
]


def parse_json_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line.startswith("{"):
                rows.append(json.loads(line))
    return rows


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def task_split_hash(source_tasks: list[str], heldout_tasks: list[str]) -> str:
    return stable_hash(
        {
            "source_tasks": sorted(source_tasks),
            "heldout_tasks": sorted(heldout_tasks),
        }
    )


def run_jsonl(command: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        subprocess.run(command, stdout=fp, check=True)


def run_promotion_script(
    script: Path,
    source_raw: Path,
    split_dir: Path,
    min_support_tasks: int,
    feedback_input: Path | None = None,
) -> None:
    command = [
        sys.executable,
        str(script),
        "--input",
        str(source_raw),
        "--output-dir",
        str(split_dir),
        "--min-support-tasks",
        str(min_support_tasks),
    ]
    if feedback_input is not None:
        command.extend(["--feedback-input", str(feedback_input)])
    subprocess.run(
        command,
        check=True,
    )


def task_list_arg(tasks: list[str]) -> str:
    return ",".join(tasks)


def parse_task_csv(text: str | None, default: list[str]) -> list[str]:
    if text is None:
        return list(default)
    tasks = [part.strip() for part in text.split(",") if part.strip()]
    if not tasks:
        raise argparse.ArgumentTypeError("task list must not be empty")
    return tasks


def mode_args(mode: Mode, prior_path: Path) -> list[str]:
    return [part.format(prior=str(prior_path)) for part in mode.args]


def stress_mode_args(mode: Mode, stress_no_prior: bool) -> list[str]:
    if stress_no_prior and mode.name == "no_prior":
        return ["--disable-hypothesis-train-consistency-prior"]
    return []


def safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def optional_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def sorted_names(values: set[str]) -> list[str]:
    return sorted(value for value in values if value)


def task_family(task: str) -> str:
    if "_sibling_" in task:
        task = task.split("_sibling_", 1)[0]
    if task.startswith("move_to_anchor"):
        return "move_to_anchor"
    if "panel_selection" in task:
        return "panel_selection"
    return task


def family_label(tasks: list[str]) -> str:
    families = sorted_names({task_family(task) for task in tasks})
    if not families:
        return "unknown"
    if len(families) == 1:
        return families[0]
    return "+".join(families)


def template_family(templates: list[str]) -> str:
    families = sorted_names({task_family(template) for template in templates})
    if not families:
        return "unknown"
    if len(families) == 1:
        return families[0]
    return "+".join(families)


def program_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("schema") != "crlp.program_proposer.summary.v1":
            continue
        out[(str(row.get("strategy", "")), int(row.get("budget", 0) or 0))] = row
    return out


def selected_evidence_by_task(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("schema") != "crlp.hypothesis.evidence.v1":
            continue
        task = str(row.get("task_instance", "unknown"))
        out.setdefault(task, []).append(row)
    return out


def selection_rows_by_task(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("schema") != "crlp.arc_hypothesis.selection.v1":
            continue
        out[str(row.get("task_instance", "unknown"))] = row
    return out


def prior_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as fp:
        return max(sum(1 for _ in fp) - 1, 0)


def evidence_template_set(
    rows: list[dict[str, Any]],
    *,
    certified_only: bool = False,
) -> list[str]:
    templates: set[str] = set()
    for row in rows:
        if row.get("schema") != "crlp.hypothesis.evidence.v1":
            continue
        if certified_only and row.get("certificate_status") != "certified":
            continue
        templates.add(str(row.get("template_name", "unknown")))
    return sorted_names(templates)


def selection_generated_template_set(selections: dict[str, dict[str, Any]]) -> list[str]:
    templates: set[str] = set()
    for selection in selections.values():
        for template in selection.get("templates", []) or []:
            if safe_int(template.get("generated")) > 0:
                templates.add(str(template.get("name", "unknown")))
    return sorted_names(templates)


def compact_promoted_libraries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("schema") != "crlp.hypothesis_library.promotion.v1":
            continue
        if not bool(row.get("promoted", False)):
            continue
        out.append(
            {
                "library_id": str(row.get("library_id", "")),
                "template_id": safe_int(row.get("template_id")),
                "template_name": str(row.get("template_name", "unknown")),
                "support_tasks": safe_int(row.get("support_tasks")),
                "task_instances": list(row.get("task_instances", []) or []),
                "covered_events": safe_int(row.get("covered_events")),
                "invariant_slot_count": len(row.get("invariant_slots", []) or []),
                "variable_slot_count": len(row.get("variable_slots", []) or []),
            }
        )
    return sorted(out, key=lambda row: (row["template_name"], row["library_id"]))


def discovery_context(
    source_tasks: list[str],
    source_rows: list[dict[str, Any]],
    promotion_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    promoted_libraries = compact_promoted_libraries(promotion_rows)
    source_template_set = evidence_template_set(source_rows)
    source_certified_template_set = evidence_template_set(source_rows, certified_only=True)
    source_promoted_template_set = sorted_names(
        {str(row["template_name"]) for row in promoted_libraries}
    )
    promotion_family = template_family(source_promoted_template_set)
    if promotion_family == "unknown":
        promotion_family = family_label(source_tasks)
    return {
        "source_family": family_label(source_tasks),
        "source_families": sorted_names({task_family(task) for task in source_tasks}),
        "promotion_family": promotion_family,
        "source_template_set": source_template_set,
        "source_certified_template_set": source_certified_template_set,
        "source_promoted_template_set": source_promoted_template_set,
        "source_template_count": len(source_template_set),
        "source_certified_template_count": len(source_certified_template_set),
        "source_promoted_template_count": len(source_promoted_template_set),
        "promoted_libraries": promoted_libraries,
    }


def empty_discovery_context(source_tasks: list[str]) -> dict[str, Any]:
    return discovery_context(source_tasks, [], [])


def template_search_budget(selection: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for template in selection.get("templates", []) or []:
        attempted = safe_int(template.get("attempted"))
        generated = safe_int(template.get("generated"))
        selected = safe_int(template.get("selected"))
        train_consistent = safe_int(template.get("train_consistent_pre_cap"))
        if attempted == 0 and generated == 0 and selected == 0 and train_consistent == 0:
            continue
        rows.append(
            {
                "template_id": safe_int(template.get("template_id")),
                "template_name": str(template.get("name", "unknown")),
                "attempted": attempted,
                "generated": generated,
                "retained": safe_int(template.get("retained")),
                "template_prior_proposals": safe_int(
                    template.get("template_prior_proposals")
                ),
                "template_prior_certified": safe_int(
                    template.get("template_prior_certified")
                ),
                "template_prior_score": float(
                    template.get("template_prior_score", 0.0) or 0.0
                ),
                "template_family_prior_source": str(
                    template.get("template_family_prior_source", "none")
                ),
                "train_consistent_pre_cap": train_consistent,
                "train_consistent_retained": safe_int(
                    template.get("train_consistent_retained")
                ),
                "heap_dropped": safe_int(template.get("heap_dropped")),
                "selected": selected,
                "train_consistency_rate_pre_cap": safe_ratio(
                    train_consistent,
                    generated,
                ),
                "selection_rate": safe_ratio(selected, generated),
            }
        )
    return rows


def template_efficiency_summary(
    selections: dict[str, dict[str, Any]],
    evidence: dict[str, list[dict[str, Any]]],
    source_template_set: set[str],
    promoted_template_set: set[str],
) -> list[dict[str, Any]]:
    by_template: dict[str, dict[str, Any]] = {}

    def entry_for(name: str, template_id: int = 0) -> dict[str, Any]:
        entry = by_template.get(name)
        if entry is None:
            entry = {
                "template_name": name,
                "template_id": template_id,
                "source_template": name in source_template_set,
                "promoted_template": name in promoted_template_set,
                "attempted": 0,
                "generated": 0,
                "retained": 0,
                "train_consistent_pre_cap": 0,
                "train_consistent_retained": 0,
                "heap_dropped": 0,
                "selected": 0,
                "selected_certified": 0,
                "selected_rejected": 0,
                "covered_events": 0,
                "event_count": 0,
            }
            by_template[name] = entry
        elif template_id and not entry["template_id"]:
            entry["template_id"] = template_id
        return entry

    for selection in selections.values():
        for template in selection.get("templates", []) or []:
            name = str(template.get("name", "unknown"))
            entry = entry_for(name, safe_int(template.get("template_id")))
            for key in (
                "attempted",
                "generated",
                "retained",
                "train_consistent_pre_cap",
                "train_consistent_retained",
                "heap_dropped",
                "selected",
            ):
                entry[key] += safe_int(template.get(key))

    for task_rows in evidence.values():
        for row in task_rows:
            name = str(row.get("template_name", "unknown"))
            entry = entry_for(name, safe_int(row.get("template_id")))
            if row.get("certificate_status") == "certified":
                entry["selected_certified"] += 1
            else:
                entry["selected_rejected"] += 1
            entry["covered_events"] += safe_int(row.get("covered_events"))
            entry["event_count"] += safe_int(row.get("event_count"))

    out: list[dict[str, Any]] = []
    for entry in by_template.values():
        if all(
            safe_int(entry[key]) == 0
            for key in (
                "attempted",
                "generated",
                "selected",
                "selected_certified",
                "selected_rejected",
                "train_consistent_pre_cap",
            )
        ):
            continue
        entry["train_consistency_rate_pre_cap"] = safe_ratio(
            entry["train_consistent_pre_cap"],
            entry["generated"],
        )
        entry["selection_rate"] = safe_ratio(entry["selected"], entry["generated"])
        entry["certified_selection_rate"] = safe_ratio(
            entry["selected_certified"],
            entry["selected"],
        )
        out.append(entry)
    return sorted(
        out,
        key=lambda row: (
            -safe_int(row["selected_certified"]),
            -safe_int(row["selected"]),
            -safe_int(row["generated"]),
            str(row["template_name"]),
        ),
    )


def heldout_family_summary(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, dict[str, Any]] = {}
    for row in tasks:
        family = str(row.get("heldout_family", "unknown"))
        entry = by_family.get(family)
        if entry is None:
            entry = {
                "family": family,
                "heldout_task_count": 0,
                "selected_certified_task_count": 0,
                "selected_rejected_task_count": 0,
                "success_reuses_source_template_count": 0,
                "success_reuses_promoted_template_count": 0,
                "success_requires_novel_template_count": 0,
                "selected_template_set": set(),
                "certified_template_set": set(),
                "novel_heldout_templates": set(),
                "novel_heldout_vs_promoted_templates": set(),
                "proposals_generated": 0,
                "proposals_selected": 0,
            }
            by_family[family] = entry
        entry["heldout_task_count"] += 1
        if safe_int(row.get("selected_certified")) > 0:
            entry["selected_certified_task_count"] += 1
        if safe_int(row.get("selected_rejected")) > 0:
            entry["selected_rejected_task_count"] += 1
        if bool(row.get("success_reuses_source_template", False)):
            entry["success_reuses_source_template_count"] += 1
        if bool(row.get("success_reuses_promoted_template", False)):
            entry["success_reuses_promoted_template_count"] += 1
        if bool(row.get("success_requires_novel_template", False)):
            entry["success_requires_novel_template_count"] += 1
        entry["selected_template_set"].update(row.get("selected_templates", []) or [])
        entry["certified_template_set"].update(row.get("certified_templates", []) or [])
        entry["novel_heldout_templates"].update(row.get("novel_heldout_templates", []) or [])
        entry["novel_heldout_vs_promoted_templates"].update(
            row.get("novel_heldout_vs_promoted_templates", []) or []
        )
        entry["proposals_generated"] += safe_int(row.get("proposals_generated"))
        entry["proposals_selected"] += safe_int(row.get("proposals_selected"))

    out: list[dict[str, Any]] = []
    for entry in by_family.values():
        out.append(
            {
                "family": entry["family"],
                "heldout_task_count": entry["heldout_task_count"],
                "selected_certified_task_count": entry[
                    "selected_certified_task_count"
                ],
                "selected_rejected_task_count": entry["selected_rejected_task_count"],
                "success_reuses_source_template_count": entry[
                    "success_reuses_source_template_count"
                ],
                "success_reuses_promoted_template_count": entry[
                    "success_reuses_promoted_template_count"
                ],
                "success_requires_novel_template_count": entry[
                    "success_requires_novel_template_count"
                ],
                "selected_template_set": sorted_names(entry["selected_template_set"]),
                "certified_template_set": sorted_names(entry["certified_template_set"]),
                "novel_heldout_templates": sorted_names(
                    entry["novel_heldout_templates"]
                ),
                "novel_heldout_vs_promoted_templates": sorted_names(
                    entry["novel_heldout_vs_promoted_templates"]
                ),
                "proposals_generated": entry["proposals_generated"],
                "proposals_selected": entry["proposals_selected"],
                "proposals_per_selected": safe_ratio(
                    entry["proposals_generated"],
                    entry["proposals_selected"],
                ),
            }
        )
    return sorted(out, key=lambda row: str(row["family"]))


def summarize_mode(
    split: str,
    source_tasks: list[str],
    heldout_tasks: list[str],
    mode: Mode,
    prior_rows: int,
    rows: list[dict[str, Any]],
    discovery: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    discovery = discovery or empty_discovery_context(source_tasks)
    programs = program_rows(rows)
    learned_budget1 = programs.get(("learned", 1), {})
    learned_budget8 = programs.get(("learned", 8), {})
    oracle_budget1 = programs.get(("oracle", 1), {})
    selections = selection_rows_by_task(rows)
    evidence = selected_evidence_by_task(rows)
    if not heldout_tasks:
        heldout_tasks = sorted(set(selections) | set(evidence))
    overlap = sorted(set(source_tasks) & set(heldout_tasks))
    split_hash = task_split_hash(source_tasks, heldout_tasks)
    source_task_hash = stable_hash(sorted(source_tasks))
    heldout_task_hash = stable_hash(sorted(heldout_tasks))
    source_template_set = set(discovery.get("source_template_set", []) or [])
    source_certified_template_set = set(
        discovery.get("source_certified_template_set", []) or []
    )
    source_promoted_template_set = set(
        discovery.get("source_promoted_template_set", []) or []
    )

    task_summaries: list[dict[str, Any]] = []
    certified_tasks = 0
    rejected_tasks = 0
    for task in heldout_tasks:
        task_evidence = evidence.get(task, [])
        certified = [row for row in task_evidence if row.get("certificate_status") == "certified"]
        rejected = [row for row in task_evidence if row.get("certificate_status") != "certified"]
        selected_templates = sorted_names(
            {str(row.get("template_name", "unknown")) for row in task_evidence}
        )
        certified_templates = sorted_names(
            {str(row.get("template_name", "unknown")) for row in certified}
        )
        generated_templates = selection_generated_template_set(
            {task: selections.get(task, {})}
        )
        selected_template_set = set(selected_templates)
        certified_template_set = set(certified_templates)
        selection = selections.get(task, {})
        proposals_generated = safe_int(selection.get("proposals_generated"))
        proposals_attempted = safe_int(selection.get("proposals_attempted"))
        proposals_selected = safe_int(selection.get("proposals_selected"))
        proposals_train_consistent_pre_cap = safe_int(
            selection.get("proposals_train_consistent_pre_cap")
        )
        proposals_train_rejected = safe_int(selection.get("proposals_train_rejected"))
        reused_source_templates = sorted_names(selected_template_set & source_template_set)
        reused_certified_source_templates = sorted_names(
            selected_template_set & source_certified_template_set
        )
        reused_promoted_templates = sorted_names(
            selected_template_set & source_promoted_template_set
        )
        novel_heldout_templates = sorted_names(selected_template_set - source_template_set)
        novel_heldout_vs_promoted_templates = sorted_names(
            selected_template_set - source_promoted_template_set
        )
        success_reuses_source_template = bool(certified_template_set & source_template_set)
        success_reuses_promoted_template = bool(
            certified_template_set & source_promoted_template_set
        )
        success_requires_novel_template = bool(certified) and not success_reuses_source_template
        if certified:
            certified_tasks += 1
        if rejected:
            rejected_tasks += 1
        task_summaries.append(
            {
                "schema": "crlp.hypothesis_library.split_sweep.task.v1",
                "split": split,
                "split_hash": split_hash,
                "mode": mode.name,
                "primary_mode": mode.primary,
                "source_family": discovery["source_family"],
                "promotion_family": discovery["promotion_family"],
                "heldout_family": task_family(task),
                "source_tasks": source_tasks,
                "source_task_hash": source_task_hash,
                "heldout_task": task,
                "heldout_task_hash": heldout_task_hash,
                "source_heldout_overlap": overlap,
                "leakage_guard_passed": not overlap,
                "source_template_set": discovery["source_template_set"],
                "source_certified_template_set": discovery[
                    "source_certified_template_set"
                ],
                "source_promoted_template_set": discovery[
                    "source_promoted_template_set"
                ],
                "heldout_generated_template_set": generated_templates,
                "selected_templates": selected_templates,
                "certified_templates": certified_templates,
                "reused_source_templates": reused_source_templates,
                "reused_certified_source_templates": reused_certified_source_templates,
                "reused_promoted_templates": reused_promoted_templates,
                "novel_heldout_templates": novel_heldout_templates,
                "novel_heldout_vs_promoted_templates": (
                    novel_heldout_vs_promoted_templates
                ),
                "success_reuses_source_template": success_reuses_source_template,
                "success_reuses_promoted_template": success_reuses_promoted_template,
                "success_requires_novel_template": success_requires_novel_template,
                "selected_certified": len(certified),
                "selected_rejected": len(rejected),
                "proposal_capacity": safe_int(selection.get("proposal_capacity")),
                "top_k": safe_int(selection.get("top_k")),
                "registered_templates": safe_int(selection.get("registered_templates")),
                "library_prior_entries": safe_int(selection.get("library_prior_entries")),
                "library_prior_template_rows": safe_int(
                    selection.get("library_prior_template_rows")
                ),
                "library_prior_slot_rows": safe_int(
                    selection.get("library_prior_slot_rows")
                ),
                "library_prior_seeded_proposals": safe_int(
                    selection.get("library_prior_seeded_proposals")
                ),
                "library_prior_seeded_certified": safe_int(
                    selection.get("library_prior_seeded_certified")
                ),
                "hypothesis_train_consistency_prior_enabled": bool(
                    selection.get("hypothesis_train_consistency_prior_enabled", True)
                ),
                "train_consistency_score_weight": safe_int(
                    (selection.get("score_weights") or {}).get("train_consistency")
                ),
                "proposals_attempted": proposals_attempted,
                "proposals_retained": safe_int(selection.get("proposals_retained")),
                "proposals_generated": proposals_generated,
                "proposals_selected": proposals_selected,
                "proposals_train_consistent_pre_cap": (
                    proposals_train_consistent_pre_cap
                ),
                "proposals_train_consistent": safe_int(
                    selection.get("proposals_train_consistent")
                ),
                "proposals_train_rejected": proposals_train_rejected,
                "proposal_cap_hit": bool(selection.get("proposal_cap_hit", False)),
                "proposals_per_selected": safe_ratio(
                    proposals_generated,
                    proposals_selected,
                ),
                "proposals_per_certified": optional_ratio(
                    proposals_generated,
                    len(certified),
                ),
                "train_consistency_rate_pre_cap": safe_ratio(
                    proposals_train_consistent_pre_cap,
                    proposals_generated,
                ),
                "selection_rate": safe_ratio(proposals_selected, proposals_generated),
                "template_search_budget": template_search_budget(selection),
                "covered_events": sum(safe_int(row.get("covered_events")) for row in certified),
                "event_count": sum(safe_int(row.get("event_count")) for row in task_evidence),
            }
        )

    heldout_template_set = sorted_names(
        {
            template
            for task_row in task_summaries
            for template in task_row["selected_templates"]
        }
    )
    heldout_generated_template_set = sorted_names(
        {
            template
            for task_row in task_summaries
            for template in task_row["heldout_generated_template_set"]
        }
    )
    heldout_certified_template_set = sorted_names(
        {
            template
            for task_row in task_summaries
            for template in task_row["certified_templates"]
        }
    )
    novel_heldout_templates = sorted_names(set(heldout_template_set) - source_template_set)
    novel_heldout_vs_promoted_templates = sorted_names(
        set(heldout_template_set) - source_promoted_template_set
    )
    family_summary = heldout_family_summary(task_summaries)
    proposals_attempted = sum(safe_int(row.get("proposals_attempted")) for row in selections.values())
    proposals_generated = sum(safe_int(row.get("proposals_generated")) for row in selections.values())
    proposals_selected = sum(safe_int(row.get("proposals_selected")) for row in selections.values())
    proposals_train_consistent_pre_cap = sum(
        safe_int(row.get("proposals_train_consistent_pre_cap"))
        for row in selections.values()
    )
    proposals_train_rejected = sum(
        safe_int(row.get("proposals_train_rejected")) for row in selections.values()
    )
    proposal_cap_hit_count = sum(
        1 for row in selections.values() if bool(row.get("proposal_cap_hit", False))
    )
    aggregate = {
        "schema": "crlp.hypothesis_library.split_sweep.summary.v1",
        "split": split,
        "mode": mode.name,
        "primary_mode": mode.primary,
        "source_family": discovery["source_family"],
        "source_families": discovery["source_families"],
        "promotion_family": discovery["promotion_family"],
        "source_tasks": source_tasks,
        "source_task_hash": source_task_hash,
        "heldout_tasks": heldout_tasks,
        "heldout_families": sorted_names({task_family(task) for task in heldout_tasks}),
        "heldout_family_summary": family_summary,
        "heldout_task_hash": heldout_task_hash,
        "split_hash": split_hash,
        "source_task_count": len(source_tasks),
        "source_heldout_overlap": overlap,
        "source_heldout_overlap_count": len(overlap),
        "leakage_guard_passed": not overlap,
        "prior_rows": prior_rows,
        "heldout_task_count": len(heldout_tasks),
        "source_template_set": discovery["source_template_set"],
        "source_certified_template_set": discovery["source_certified_template_set"],
        "source_promoted_template_set": discovery["source_promoted_template_set"],
        "source_template_count": discovery["source_template_count"],
        "source_certified_template_count": discovery["source_certified_template_count"],
        "source_promoted_template_count": discovery["source_promoted_template_count"],
        "promoted_libraries": discovery["promoted_libraries"],
        "heldout_generated_template_set": heldout_generated_template_set,
        "heldout_template_set": heldout_template_set,
        "heldout_certified_template_set": heldout_certified_template_set,
        "novel_heldout_templates": novel_heldout_templates,
        "novel_heldout_vs_promoted_templates": novel_heldout_vs_promoted_templates,
        "reused_source_templates": sorted_names(
            set(heldout_template_set) & source_template_set
        ),
        "reused_promoted_templates": sorted_names(
            set(heldout_template_set) & source_promoted_template_set
        ),
        "success_reuses_source_template_count": sum(
            1 for row in task_summaries if row["success_reuses_source_template"]
        ),
        "success_reuses_promoted_template_count": sum(
            1 for row in task_summaries if row["success_reuses_promoted_template"]
        ),
        "success_requires_novel_template_count": sum(
            1 for row in task_summaries if row["success_requires_novel_template"]
        ),
        "selected_certified_tasks": certified_tasks,
        "selected_rejected_tasks": rejected_tasks,
        "learned_certified_budget1": int(learned_budget1.get("certified_programs", 0) or 0),
        "learned_utility_budget1": float(learned_budget1.get("utility", 0.0) or 0.0),
        "learned_certified_budget8": int(learned_budget8.get("certified_programs", 0) or 0),
        "learned_utility_budget8": float(learned_budget8.get("utility", 0.0) or 0.0),
        "oracle_certified_budget1": int(oracle_budget1.get("certified_programs", 0) or 0),
        "library_prior_seeded_proposals": sum(
            safe_int(row.get("library_prior_seeded_proposals"))
            for row in selections.values()
        ),
        "library_prior_template_rows": max(
            (safe_int(row.get("library_prior_template_rows")) for row in selections.values()),
            default=0,
        ),
        "library_prior_seeded_certified": sum(
            safe_int(row.get("library_prior_seeded_certified"))
            for row in selections.values()
        ),
        "proposals_attempted": proposals_attempted,
        "proposals_generated": proposals_generated,
        "proposals_selected": proposals_selected,
        "proposals_train_consistent_pre_cap": proposals_train_consistent_pre_cap,
        "proposals_train_rejected": proposals_train_rejected,
        "proposal_cap_hit_count": proposal_cap_hit_count,
        "proposals_per_selected": safe_ratio(proposals_generated, proposals_selected),
        "proposals_per_certified_task": optional_ratio(
            proposals_generated,
            certified_tasks,
        ),
        "train_consistency_rate_pre_cap": safe_ratio(
            proposals_train_consistent_pre_cap,
            proposals_generated,
        ),
        "selection_rate": safe_ratio(proposals_selected, proposals_generated),
        "heldout_template_efficiency": template_efficiency_summary(
            selections,
            evidence,
            source_template_set,
            source_promoted_template_set,
        ),
    }
    return aggregate, task_summaries


def public_split_dirs(path: Path) -> tuple[Path, Path]:
    training = path / "training"
    evaluation = path / "evaluation"
    if training.is_dir() and evaluation.is_dir():
        return training, evaluation
    train = path / "train"
    eval_dir = path / "eval"
    if train.is_dir() and eval_dir.is_dir():
        return train, eval_dir
    raise RuntimeError(f"{path} must contain training/evaluation or train/eval directories")


def heldout_result_rows(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_split_mode = {(row["split"], row["mode"]): row for row in aggregates}
    out: list[dict[str, Any]] = []
    for row in aggregates:
        baseline = by_split_mode.get((row["split"], "no_prior"))
        if baseline is None:
            continue
        out.append(
            {
                "schema": "crlp.hypothesis_library.heldout_result.v1",
                "report_version": 1,
                "split": row["split"],
                "mode": row["mode"],
                "primary_mode": row["primary_mode"],
                "source_family": row["source_family"],
                "source_families": row["source_families"],
                "promotion_family": row["promotion_family"],
                "source_tasks": row["source_tasks"],
                "source_task_hash": row["source_task_hash"],
                "heldout_tasks": row["heldout_tasks"],
                "heldout_families": row["heldout_families"],
                "heldout_family_summary": row["heldout_family_summary"],
                "heldout_task_hash": row["heldout_task_hash"],
                "split_hash": row["split_hash"],
                "source_task_count": row["source_task_count"],
                "source_heldout_overlap_count": row["source_heldout_overlap_count"],
                "leakage_guard_passed": row["leakage_guard_passed"],
                "prior_rows": row["prior_rows"],
                "heldout_task_count": row["heldout_task_count"],
                "source_template_set": row["source_template_set"],
                "source_certified_template_set": row["source_certified_template_set"],
                "source_promoted_template_set": row["source_promoted_template_set"],
                "source_template_count": row["source_template_count"],
                "source_certified_template_count": row["source_certified_template_count"],
                "source_promoted_template_count": row["source_promoted_template_count"],
                "promoted_libraries": row["promoted_libraries"],
                "heldout_generated_template_set": row["heldout_generated_template_set"],
                "heldout_template_set": row["heldout_template_set"],
                "heldout_certified_template_set": row["heldout_certified_template_set"],
                "novel_heldout_templates": row["novel_heldout_templates"],
                "novel_heldout_vs_promoted_templates": row[
                    "novel_heldout_vs_promoted_templates"
                ],
                "reused_source_templates": row["reused_source_templates"],
                "reused_promoted_templates": row["reused_promoted_templates"],
                "success_reuses_source_template_count": row[
                    "success_reuses_source_template_count"
                ],
                "success_reuses_promoted_template_count": row[
                    "success_reuses_promoted_template_count"
                ],
                "success_requires_novel_template_count": row[
                    "success_requires_novel_template_count"
                ],
                "baseline_mode": "no_prior",
                "baseline_learned_certified_budget1": baseline[
                    "learned_certified_budget1"
                ],
                "learned_certified_budget1": row["learned_certified_budget1"],
                "delta_learned_certified_budget1": (
                    int(row["learned_certified_budget1"])
                    - int(baseline["learned_certified_budget1"])
                ),
                "baseline_learned_utility_budget1": baseline[
                    "learned_utility_budget1"
                ],
                "learned_utility_budget1": row["learned_utility_budget1"],
                "delta_learned_utility_budget1": (
                    float(row["learned_utility_budget1"])
                    - float(baseline["learned_utility_budget1"])
                ),
                "baseline_oracle_certified_budget1": baseline[
                    "oracle_certified_budget1"
                ],
                "oracle_certified_budget1": row["oracle_certified_budget1"],
                "selected_rejected_tasks": row["selected_rejected_tasks"],
                "library_prior_seeded_proposals": row["library_prior_seeded_proposals"],
                "library_prior_template_rows": row["library_prior_template_rows"],
                "library_prior_seeded_certified": row["library_prior_seeded_certified"],
                "proposals_attempted": row["proposals_attempted"],
                "proposals_generated": row["proposals_generated"],
                "proposals_selected": row["proposals_selected"],
                "proposals_train_consistent_pre_cap": row[
                    "proposals_train_consistent_pre_cap"
                ],
                "proposals_train_rejected": row["proposals_train_rejected"],
                "proposal_cap_hit_count": row["proposal_cap_hit_count"],
                "proposals_per_selected": row["proposals_per_selected"],
                "proposals_per_certified_task": row["proposals_per_certified_task"],
                "train_consistency_rate_pre_cap": row[
                    "train_consistency_rate_pre_cap"
                ],
                "selection_rate": row["selection_rate"],
                "tight_budget_improved_or_equal": (
                    int(row["learned_certified_budget1"])
                    >= int(baseline["learned_certified_budget1"])
                ),
                "heldout_feedback_guard_passed": (
                    bool(row["leakage_guard_passed"])
                    and int(row["learned_certified_budget1"])
                    >= int(baseline["learned_certified_budget1"])
                ),
            }
        )
    return out


def status_cell(task_row: dict[str, Any]) -> str:
    templates = ",".join(task_row["selected_templates"]) if task_row["selected_templates"] else "-"
    if task_row["selected_certified"]:
        return f"C:{templates}"
    if task_row["selected_rejected"]:
        return f"R:{templates}"
    return "-"


def compact_list(values: list[str], limit: int = 4) -> str:
    if not values:
        return "-"
    head = values[:limit]
    suffix = "" if len(values) <= limit else f",+{len(values) - limit}"
    return "`" + "`,`".join(head) + "`" + suffix


def optional_float_cell(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def family_counts_cell(row: dict[str, Any]) -> str:
    parts = []
    for family in row.get("heldout_family_summary", []) or []:
        parts.append(
            "{family} {success}/{total}".format(
                family=family["family"],
                success=family["selected_certified_task_count"],
                total=family["heldout_task_count"],
            )
        )
    return "; ".join(parts) if parts else "-"


def reuse_cell(row: dict[str, Any]) -> str:
    return "{promoted} promoted, {source} source, {novel} novel-required".format(
        promoted=row.get("success_reuses_promoted_template_count", 0),
        source=row.get("success_reuses_source_template_count", 0),
        novel=row.get("success_requires_novel_template_count", 0),
    )


def write_markdown(
    path: Path,
    aggregates: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    heldout_results: list[dict[str, Any]],
) -> None:
    by_split_mode = {(row["split"], row["mode"]): row for row in aggregates}
    split_names = sorted({row["split"] for row in aggregates})
    primary_modes = [mode.name for mode in MODES if mode.primary]
    all_modes = [mode.name for mode in MODES]

    lines = [
        "# Hypothesis Library Split Sweep",
        "",
        "Protocol: promote certified hypothesis slot priors from each two-task "
        "`move_to_anchor` source split, then evaluate held-out ARC hypothesis "
        "fixtures with top-k 1, coverage selection disabled, structural scoring "
        "disabled, and pre-ranking train filtering disabled.",
        "",
        "## Aggregate Transfer",
        "",
        "| Split | Source tasks | Prior rows | no-prior @1 | no-prior oracle @1 | slot-only @1 | slot oracle @1 | linear @1 | tiny MLP @1 | full MLP @1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in split_names:
        first = by_split_mode[(split, all_modes[0])]
        no_prior = by_split_mode[(split, "no_prior")]
        slot = by_split_mode[(split, "slot_only")]
        cells = []
        for mode in ("no_prior", "slot_only", "linear_prior", "tiny_mlp4", "full_mlp24"):
            row = by_split_mode[(split, mode)]
            cells.append(
                f"{row['learned_certified_budget1']}/{row['heldout_task_count']} "
                f"({row['learned_utility_budget1']:.3f})"
            )
        lines.append(
            "| {split} | `{source}` | {prior} | {no_prior} | {no_oracle}/{total} | "
            "{slot} | {slot_oracle}/{total} | {linear} | {tiny} | {full} |".format(
                split=split,
                source=",".join(first["source_tasks"]),
                prior=first["prior_rows"],
                no_prior=cells[0],
                no_oracle=no_prior["oracle_certified_budget1"],
                slot=cells[1],
                slot_oracle=slot["oracle_certified_budget1"],
                linear=cells[2],
                tiny=cells[3],
                full=cells[4],
                total=first["heldout_task_count"],
            )
        )

    lines.extend(
        [
            "",
            "## Discovery Reuse",
            "",
            "These rows compare symbolic templates discovered on the source split "
            "with the templates selected on held-out families. `novel` means the "
            "held-out selected template was not seen in source evidence; "
            "`novel vs promoted` is stricter and compares only against promoted "
            "library templates.",
            "",
            "| Split | Mode | Source family | Promoted templates | Held-out families success/total | Held-out templates | Novel | Novel vs promoted | Success reuse |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in aggregates:
        if row["mode"] not in primary_modes:
            continue
        lines.append(
            "| {split} | `{mode}` | {source_family} -> {promotion_family} | {promoted} | {families} | {heldout} | {novel} | {novel_promoted} | {reuse} |".format(
                split=row["split"],
                mode=row["mode"],
                source_family=row["source_family"],
                promotion_family=row["promotion_family"],
                promoted=compact_list(row["source_promoted_template_set"]),
                families=family_counts_cell(row),
                heldout=compact_list(row["heldout_template_set"]),
                novel=compact_list(row["novel_heldout_templates"]),
                novel_promoted=compact_list(row["novel_heldout_vs_promoted_templates"]),
                reuse=reuse_cell(row),
            )
        )

    lines.extend(
        [
            "",
            "## Held-Out Result Deltas",
            "",
            "These rows are the paper-facing held-out claim surface: promotion happens on "
            "the source tasks, then every prior mode is scored against the no-prior "
            "tight-budget baseline on held-out tasks.",
            "",
            "| Split | Mode | Certified @1 delta | Utility @1 delta | Train rejected | Leakage | Tight-budget guard |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in heldout_results:
        if row["mode"] == "no_prior" or not row["primary_mode"]:
            continue
        guard = "pass" if row["tight_budget_improved_or_equal"] else "fail"
        leakage = "pass" if row["leakage_guard_passed"] else "fail"
        lines.append(
            "| {split} | `{mode}` | {cert_delta:+d} | {utility_delta:+.3f} | {rejected} | {leakage} | {guard} |".format(
                split=row["split"],
                mode=row["mode"],
                cert_delta=int(row["delta_learned_certified_budget1"]),
                utility_delta=float(row["delta_learned_utility_budget1"]),
                rejected=int(row["proposals_train_rejected"]),
                leakage=leakage,
                guard=guard,
            )
        )

    lines.extend(
        [
            "",
            "## Decoy Pressure",
            "",
            "| Split | Mode | Attempted | Generated | Selected | Train-consistent pre-cap | Train rejected | Cap hits | Consistency rate | Gen/cert task | Seeded cert | Selected certified @1 | Oracle @1 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregates:
        if row["mode"] not in primary_modes:
            continue
        lines.append(
            "| {split} | `{mode}` | {attempted} | {generated} | {selected} | {consistent} | {rejected} | {cap_hits} | {consistency:.3f} | {gen_per_cert} | {seeded_cert} | {cert}/{total} | {oracle}/{total} |".format(
                split=row["split"],
                mode=row["mode"],
                attempted=row["proposals_attempted"],
                generated=row["proposals_generated"],
                selected=row["proposals_selected"],
                consistent=row["proposals_train_consistent_pre_cap"],
                rejected=row["proposals_train_rejected"],
                cap_hits=row["proposal_cap_hit_count"],
                consistency=row["train_consistency_rate_pre_cap"],
                gen_per_cert=optional_float_cell(row["proposals_per_certified_task"]),
                seeded_cert=row["library_prior_seeded_certified"],
                cert=row["learned_certified_budget1"],
                oracle=row["oracle_certified_budget1"],
                total=row["heldout_task_count"],
            )
        )

    task_by_key = {(row["split"], row["mode"], row["heldout_task"]): row for row in tasks}
    lines.extend(
        [
            "",
            "## Per-Task Transfer",
            "",
            "| Split | Held-out family | Held-out task | no-prior | slot-only | linear |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for split in split_names:
        heldout_tasks = by_split_mode[(split, all_modes[0])]["heldout_tasks"]
        for task in heldout_tasks:
            cells = [
                status_cell(task_by_key[(split, mode, task)])
                for mode in primary_modes
            ]
            family = task_by_key[(split, primary_modes[0], task)]["heldout_family"]
            lines.append(
                f"| {split} | {family} | `{task}` | " + " | ".join(cells) + " |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True) + "\n")


def write_heldout_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "split",
        "mode",
        "source_tasks",
        "source_task_hash",
        "heldout_task_hash",
        "split_hash",
        "source_task_count",
        "source_heldout_overlap_count",
        "leakage_guard_passed",
        "heldout_task_count",
        "prior_rows",
        "learned_certified_budget1",
        "delta_learned_certified_budget1",
        "learned_utility_budget1",
        "delta_learned_utility_budget1",
        "selected_rejected_tasks",
        "proposals_train_rejected",
        "tight_budget_improved_or_equal",
        "heldout_feedback_guard_passed",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{field: row.get(field) for field in fieldnames},
                    "source_tasks": ",".join(row["source_tasks"]),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arc-agi", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, default=Path("tests/fixtures/arc_hypothesis"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/float/hypothesis_library_split_sweep"))
    parser.add_argument("--min-support-tasks", type=int, default=2)
    parser.add_argument("--task-ids", help="Comma-separated task universe; defaults to fixture task list")
    parser.add_argument("--promotion-family", help="Comma-separated task family used for source-pair promotion")
    parser.add_argument("--source-task-ids", help="Comma-separated fixed source tasks; disables pair sweep")
    parser.add_argument("--heldout-task-ids", help="Comma-separated held-out tasks for fixed source mode")
    parser.add_argument(
        "--stress-no-prior",
        action="store_true",
        help=(
            "Disable verifier-derived train-consistency ranking only for the no-prior "
            "mode, creating a search-pressure baseline that can expose library-prior gains."
        ),
    )
    parser.add_argument("--public-arc-dir", type=Path, help="Public ARC root with training/evaluation splits")
    parser.add_argument("--public-train-max-tasks", type=int, default=400)
    parser.add_argument("--public-eval-max-tasks", type=int, default=400)
    parser.add_argument("--public-min-induction-evidence", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path.cwd()
    promotion_script = repo_root / "scripts" / "certified_hypothesis_library.py"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    aggregates: list[dict[str, Any]] = []
    task_summaries: list[dict[str, Any]] = []
    if args.public_arc_dir:
        train_dir, eval_dir = public_split_dirs(args.public_arc_dir)
        split = "public_training__public_evaluation"
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        source_raw = split_dir / "source_raw.jsonl"
        run_jsonl(
            [
                str(args.arc_agi),
                "--task-dir",
                str(train_dir),
                "--max-tasks",
                str(args.public_train_max_tasks),
                "--program-budget-grid",
                "--loader-summary-only",
                "--induction",
                "--min-induction-evidence",
                str(args.public_min_induction_evidence),
            ],
            source_raw,
        )
        run_promotion_script(
            promotion_script,
            source_raw,
            split_dir,
            args.min_support_tasks,
        )
        source_rows = parse_json_rows(source_raw)
        promotion_rows = parse_json_rows(split_dir / "promotions.jsonl")
        discovery = discovery_context(["public_training"], source_rows, promotion_rows)
        prior_path = split_dir / "library_prior.tsv"
        priors = prior_row_count(prior_path)
        for mode in MODES:
            output_path = split_dir / f"{mode.name}.jsonl"
            command = [
                str(args.arc_agi),
                "--task-dir",
                str(eval_dir),
                "--max-tasks",
                str(args.public_eval_max_tasks),
                "--min-induction-evidence",
                str(args.public_min_induction_evidence),
                *STRICT_EVAL_FLAGS,
                *mode_args(mode, prior_path),
                *stress_mode_args(mode, args.stress_no_prior),
            ]
            run_jsonl(command, output_path)
            rows = parse_json_rows(output_path)
            aggregate, per_task = summarize_mode(
                split,
                ["public_training"],
                [],
                mode,
                priors,
                rows,
                discovery,
            )
            aggregate["stress_no_prior"] = bool(args.stress_no_prior)
            for row in per_task:
                row["stress_no_prior"] = bool(args.stress_no_prior)
            aggregate["source_split"] = str(train_dir)
            aggregate["heldout_split"] = str(eval_dir)
            aggregates.append(aggregate)
            task_summaries.extend(per_task)
        run_promotion_script(
            promotion_script,
            source_raw,
            split_dir,
            args.min_support_tasks,
            split_dir / "slot_only.jsonl",
        )
        heldout_results = heldout_result_rows(aggregates)
        aggregate_path = args.output_dir / "split_sweep_summary.jsonl"
        task_path = args.output_dir / "split_sweep_tasks.jsonl"
        heldout_path = args.output_dir / "heldout_result.jsonl"
        heldout_csv_path = args.output_dir / "heldout_result.csv"
        write_jsonl(aggregate_path, aggregates)
        write_jsonl(task_path, task_summaries)
        write_jsonl(heldout_path, heldout_results)
        write_heldout_csv(heldout_csv_path, heldout_results)
        summary_path = args.output_dir / "summary.md"
        write_markdown(summary_path, aggregates, task_summaries, heldout_results)
        print(f"wrote {aggregate_path}")
        print(f"wrote {task_path}")
        print(f"wrote {heldout_path}")
        print(f"wrote {heldout_csv_path}")
        print(f"wrote {summary_path}")
        return 0

    all_tasks = parse_task_csv(args.task_ids, ALL_TASKS)
    promotion_family = parse_task_csv(args.promotion_family, PROMOTION_FAMILY)
    if args.source_task_ids:
        split_pairs = [tuple(parse_task_csv(args.source_task_ids, []))]
    else:
        split_pairs = list(itertools.combinations(promotion_family, 2))
    for source_pair in split_pairs:
        source_tasks = list(source_pair)
        heldout_tasks = parse_task_csv(args.heldout_task_ids, [task for task in all_tasks if task not in source_tasks])
        split = "__".join(source_tasks)
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        source_raw = split_dir / "source_raw.jsonl"
        run_jsonl(
            [
                str(args.arc_agi),
                "--task-dir",
                str(args.task_dir),
                "--task-ids",
                task_list_arg(source_tasks),
                "--max-tasks",
                str(max(10, len(source_tasks))),
                "--program-budget-grid",
                "--loader-summary-only",
                "--induction",
                "--min-induction-evidence",
                str(len(source_tasks)),
            ],
            source_raw,
        )
        run_promotion_script(
            promotion_script,
            source_raw,
            split_dir,
            args.min_support_tasks,
        )
        source_rows = parse_json_rows(source_raw)
        promotion_rows = parse_json_rows(split_dir / "promotions.jsonl")
        discovery = discovery_context(source_tasks, source_rows, promotion_rows)
        prior_path = split_dir / "library_prior.tsv"
        priors = prior_row_count(prior_path)

        for mode in MODES:
            output_path = split_dir / f"{mode.name}.jsonl"
            command = [
                str(args.arc_agi),
                "--task-dir",
                str(args.task_dir),
                "--task-ids",
                task_list_arg(heldout_tasks),
                "--max-tasks",
                str(max(10, len(heldout_tasks))),
                "--min-induction-evidence",
                str(len(heldout_tasks)),
                *STRICT_EVAL_FLAGS,
                *mode_args(mode, prior_path),
                *stress_mode_args(mode, args.stress_no_prior),
            ]
            run_jsonl(command, output_path)
            rows = parse_json_rows(output_path)
            aggregate, per_task = summarize_mode(
                split,
                source_tasks,
                heldout_tasks,
                mode,
                priors,
                rows,
                discovery,
            )
            aggregate["stress_no_prior"] = bool(args.stress_no_prior)
            for row in per_task:
                row["stress_no_prior"] = bool(args.stress_no_prior)
            aggregates.append(aggregate)
            task_summaries.extend(per_task)
        run_promotion_script(
            promotion_script,
            source_raw,
            split_dir,
            args.min_support_tasks,
            split_dir / "slot_only.jsonl",
        )

    heldout_results = heldout_result_rows(aggregates)
    aggregate_path = args.output_dir / "split_sweep_summary.jsonl"
    task_path = args.output_dir / "split_sweep_tasks.jsonl"
    heldout_path = args.output_dir / "heldout_result.jsonl"
    heldout_csv_path = args.output_dir / "heldout_result.csv"
    write_jsonl(aggregate_path, aggregates)
    write_jsonl(task_path, task_summaries)
    write_jsonl(heldout_path, heldout_results)
    write_heldout_csv(heldout_csv_path, heldout_results)
    summary_path = args.output_dir / "summary.md"
    write_markdown(summary_path, aggregates, task_summaries, heldout_results)
    print(f"wrote {aggregate_path}")
    print(f"wrote {task_path}")
    print(f"wrote {heldout_path}")
    print(f"wrote {heldout_csv_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
