#!/usr/bin/env python3
"""Prove or falsify current ARC neural-discovery failure hypotheses."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "crlp.arc_neural_discovery.failure_proof.v1"
PROMOTION_REPLAY_SCHEMA = "crlp.arc_neural_discovery.promotion_replay.v1"
PROMOTION_FLAGS = [
    "--program-budget-grid",
    "--loader-summary-only",
    "--induction",
    "--min-induction-evidence",
    "0",
]
MISSING_FAMILY_PRIORITIES = {
    "add_tiling_scale_or_canvas_growth_templates",
    "add_separator_panel_extraction_templates",
    "broaden_crop_object_selection_templates",
    "mine_same_shape_patch_lattice_templates",
    "add_projection_compression_templates",
    "inspect_for_compositional_template",
}


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


def task_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("split", "")), str(row.get("task_id", "")))


def triage_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        split = str(row.get("split", ""))
        mode = str(row.get("mode", ""))
        task_id = str(row.get("task_id", ""))
        if not task_id:
            continue
        key = (split, task_id)
        existing = by_key.get(key)
        if existing is None or mode == "default":
            by_key[key] = row
    return by_key


def summarize_candidate_sketches(rows: list[dict[str, Any]]) -> dict[str, Any]:
    templates = Counter(str(row.get("template_name", "")) for row in rows)
    with_slot = [row for row in rows if safe_int(row.get("slot_probe_row_count")) > 0]
    return {
        "candidate_count": len(rows),
        "candidates_with_slot_sketches": len(with_slot),
        "slot_probe_rows": sum(safe_int(row.get("slot_probe_row_count")) for row in rows),
        "slot_probe_sketches": sum(safe_int(row.get("slot_probe_sketch_count")) for row in rows),
        "templates": dict(sorted(templates.items())),
    }


def selected_templates(selection: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for template in selection.get("templates", []) or []:
        if isinstance(template, dict) and safe_int(template.get("selected")) > 0:
            name = str(template.get("name", ""))
            if name:
                names.append(name)
    return names


def train_consistent_templates(selection: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for template in selection.get("templates", []) or []:
        if not isinstance(template, dict):
            continue
        pre_cap = safe_int(template.get("train_consistent_pre_cap"))
        retained = safe_int(template.get("train_consistent_retained"))
        if pre_cap > 0 or retained > 0:
            name = str(template.get("name", ""))
            if name:
                names.append(name)
    return names


def parse_replay_log(log_path: Path) -> dict[str, dict[str, Any]]:
    by_task: dict[str, dict[str, Any]] = defaultdict(dict)
    evidence_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in parse_json_rows(log_path):
        schema = row.get("schema")
        if schema == "crlp.arc_hypothesis.selection.v1":
            task_id = str(row.get("task_instance", ""))
            if not task_id:
                continue
            by_task[task_id].update(
                {
                    "task_id": task_id,
                    "proposals_generated": safe_int(row.get("proposals_generated")),
                    "proposals_retained": safe_int(row.get("proposals_retained")),
                    "proposals_train_consistent_pre_cap": safe_int(
                        row.get("proposals_train_consistent_pre_cap")
                    ),
                    "proposals_train_consistent": safe_int(
                        row.get("proposals_train_consistent")
                    ),
                    "proposals_train_rejected": safe_int(
                        row.get("proposals_train_rejected")
                    ),
                    "proposals_selected": safe_int(row.get("proposals_selected")),
                    "proposal_cap_hit": bool(row.get("proposal_cap_hit")),
                    "selected_templates": selected_templates(row),
                    "train_consistent_templates": train_consistent_templates(row),
                }
            )
        elif schema == "crlp.hypothesis.evidence.v1":
            task_id = str(row.get("task_instance", ""))
            if task_id:
                evidence_by_task[task_id].append(row)

    for task_id, evidences in evidence_by_task.items():
        certified = [
            row for row in evidences if row.get("certificate_status") == "certified"
        ]
        rejected = [
            row for row in evidences if row.get("certificate_status") == "rejected"
        ]
        best = certified[0] if certified else (evidences[0] if evidences else {})
        by_task[task_id].update(
            {
                "evidence_rows": len(evidences),
                "certified_evidence_rows": len(certified),
                "rejected_evidence_rows": len(rejected),
                "certified": bool(certified),
                "evidence_template": str(best.get("template_name", "")),
                "evidence_status": str(best.get("certificate_status", "")),
                "evidence_reason": str(best.get("reason", "")),
                "covered_events": safe_int(best.get("covered_events")),
                "event_count": safe_int(best.get("event_count")),
            }
        )
    return dict(by_task)


def run_replay(
    arc_agi: Path,
    split: str,
    task_ids: list[str],
    task_dir: Path,
    output_dir: Path,
) -> tuple[Path, int, list[str], dict[str, dict[str, Any]]]:
    log_path = output_dir / f"promotion_replay_{split}.jsonl"
    command = [
        str(arc_agi),
        "--task-dir",
        str(task_dir),
        "--task-ids",
        ",".join(task_ids),
        *PROMOTION_FLAGS,
    ]
    with log_path.open("w", encoding="utf-8") as fp:
        result = subprocess.run(
            command,
            check=False,
            stdout=fp,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return log_path, result.returncode, command, parse_replay_log(log_path)


def aggregate_mutations(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = task_key(row)
        if key[0] and key[1]:
            grouped[key].append(row)

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, group in grouped.items():
        metrics = [row.get("mutation_metrics", {}) for row in group]
        baseline_metrics = [row.get("baseline_metrics", {}) for row in group]
        statuses = Counter(str(row.get("status", "")) for row in group)
        reasons = Counter(str(row.get("reason", "")) for row in group)
        templates = Counter(str(row.get("template_name", "")) for row in group)
        train_consistent_rows = [
            metric
            for metric in metrics
            if safe_int(metric.get("proposals_train_consistent_pre_cap")) > 0
            or safe_int(metric.get("proposals_train_consistent")) > 0
        ]
        out[key] = {
            "mutation_rows": len(group),
            "candidate_ids": sorted({str(row.get("candidate_id", "")) for row in group}),
            "candidate_templates": dict(sorted(templates.items())),
            "statuses": dict(sorted(statuses.items())),
            "reasons": dict(sorted(reasons.items())),
            "accepted_existing_rows": statuses.get("accepted_existing", 0),
            "diagnostic_rows": statuses.get("diagnostic", 0),
            "rejected_rows": statuses.get("rejected", 0),
            "novel_certification_rows": sum(
                1 for row in group if bool(row.get("novel_certification"))
            ),
            "accepted_by_verifier_rows": sum(
                1 for row in group if bool(row.get("accepted_by_verifier"))
            ),
            "slot_probe_rows": sum(safe_int(metric.get("slot_probe_rows")) for metric in metrics),
            "slot_probe_seeded_proposals": sum(
                safe_int(metric.get("slot_probe_seeded_proposals")) for metric in metrics
            ),
            "slot_probe_scored_rows": sum(
                1 for metric in metrics if safe_float(metric.get("slot_probe_score")) > 0.0
            ),
            "template_probe_rows": sum(
                safe_int(metric.get("template_probe_rows")) for metric in metrics
            ),
            "template_probe_seeded_proposals": sum(
                safe_int(metric.get("template_probe_seeded_proposals")) for metric in metrics
            ),
            "template_selected_rows": sum(
                1 for metric in metrics if safe_int(metric.get("template_selected")) > 0
            ),
            "template_selection_improved_rows": reasons.get(
                "template_selection_improved", 0
            ),
            "mutation_train_consistent_rows": len(train_consistent_rows),
            "mutation_certified_rows": sum(
                1 for metric in metrics if safe_int(metric.get("certified_tasks")) > 0
            ),
            "mutation_proposals_generated": sum(
                safe_int(metric.get("proposals_generated")) for metric in metrics
            ),
            "mutation_proposals_train_rejected": sum(
                safe_int(metric.get("proposals_train_rejected")) for metric in metrics
            ),
            "baseline_train_consistent_rows": sum(
                1
                for metric in baseline_metrics
                if safe_int(metric.get("proposals_train_consistent_pre_cap")) > 0
                or safe_int(metric.get("proposals_train_consistent")) > 0
            ),
        }
    return out


def classify(row: dict[str, Any]) -> str:
    if safe_int(row.get("accepted_existing_rows")) > 0:
        return "accepted_existing_control"
    slot_decoder = bool(row.get("slot_decoder_quality_evidence"))
    missing_family = bool(row.get("missing_family_evidence"))
    if slot_decoder and missing_family:
        return "slot_decoder_and_missing_template_family"
    if slot_decoder:
        return "slot_decoder_quality"
    if missing_family:
        return "missing_template_family"
    if safe_int(row.get("promotion_proposals_train_consistent_pre_cap")) > 0:
        return "ranking_or_selection_gap"
    return "insufficient_evidence"


def build_summary(rows: list[dict[str, Any]], sketch_summary: dict[str, Any]) -> str:
    total_rows = sum(safe_int(row.get("mutation_rows")) for row in rows)
    classifications = Counter(str(row.get("classification", "")) for row in rows)
    priorities = Counter(str(row.get("triage_priority", "")) for row in rows)
    control_rows = [
        row for row in rows if safe_int(row.get("accepted_existing_rows")) > 0
    ]
    failure_rows = [
        row for row in rows if safe_int(row.get("accepted_existing_rows")) == 0
    ]
    templates = Counter()
    for row in rows:
        for template, count in (row.get("candidate_templates") or {}).items():
            templates[str(template)] += safe_int(count)

    slot_evidence = sum(1 for row in failure_rows if row.get("slot_decoder_quality_evidence"))
    missing_evidence = sum(1 for row in failure_rows if row.get("missing_family_evidence"))
    zero_train = sum(
        1 for row in rows if safe_int(row.get("promotion_proposals_train_consistent_pre_cap")) == 0
    )
    any_train = sum(
        1 for row in rows if safe_int(row.get("promotion_proposals_train_consistent_pre_cap")) > 0
    )
    certified = sum(1 for row in rows if row.get("promotion_certified"))
    loaded_replays = sum(1 for row in rows if row.get("promotion_replayed"))

    lines = [
        "# ARC Neural Failure Proof",
        "",
        "## Verdict",
        "",
        (
            f"- Mutation verifier rows: {total_rows} across {len(rows)} unique "
            "representative tasks."
        ),
        (
            f"- Failure tasks tested: {len(failure_rows)} after excluding "
            f"{len(control_rows)} accepted-existing control task(s)."
        ),
        (
            f"- Slot-decoder evidence: {slot_evidence}/{len(failure_rows)} tasks had neural "
            "slot probes loaded/selected while producing zero train-consistent mutations."
        ),
        (
            f"- Current-template-family evidence: {missing_evidence}/{len(failure_rows)} tasks "
            "had zero promotion-mode train-consistent proposals plus a missing-family "
            "triage label."
        ),
        (
            f"- Promotion replay: {loaded_replays}/{len(rows)} tasks replayed, "
            f"{zero_train} with zero train-consistent proposals, {any_train} with at "
            f"least one, {certified} certified."
        ),
        "",
        "Interpretation: slot decoding is failing when probes can bias the selector but "
        "cannot instantiate verifier-valid slots. Missing template families are "
        "supported when an unconstrained promotion replay cannot produce any "
        "train-consistent proposal from the current grammar for the same task.",
        "",
        "## Classifications",
        "",
        "| Classification | Tasks |",
        "| --- | ---: |",
    ]
    for name, count in classifications.most_common():
        lines.append(f"| `{name}` | {count} |")

    lines.extend(["", "## Candidate Sketch Surface", ""])
    lines.extend(
        [
            f"- Candidates: {sketch_summary['candidate_count']}",
            f"- Candidates with slot sketches: {sketch_summary['candidates_with_slot_sketches']}",
            f"- Slot-probe rows emitted: {sketch_summary['slot_probe_rows']}",
            f"- Slot sketches emitted: {sketch_summary['slot_probe_sketches']}",
        ]
    )
    if sketch_summary["templates"]:
        lines.extend(["", "| Template | Candidate rows |", "| --- | ---: |"])
        for template, count in sketch_summary["templates"].items():
            lines.append(f"| `{template}` | {count} |")

    lines.extend(["", "## Triage Priorities", "", "| Priority | Tasks |", "| --- | ---: |"])
    for priority, count in priorities.most_common():
        lines.append(f"| `{priority}` | {count} |")

    lines.extend(["", "## Mutation Candidate Templates", "", "| Template | Rows |", "| --- | ---: |"])
    for template, count in templates.most_common():
        lines.append(f"| `{template}` | {count} |")

    interesting = sorted(
        rows,
        key=lambda row: (
            str(row.get("classification", "")),
            str(row.get("triage_priority", "")),
            str(row.get("task_id", "")),
        ),
    )
    lines.extend(["", "## Task Evidence", ""])
    lines.extend(
        [
            "| Task | Class | Slot rows | Mutation train-consistent | Promotion train-consistent | Triage priority | Selected templates |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in interesting:
        lines.append(
            "| `{}` | `{}` | {} | {} | {} | `{}` | `{}` |".format(
                row.get("task_id", ""),
                row.get("classification", ""),
                safe_int(row.get("slot_probe_rows")),
                safe_int(row.get("mutation_train_consistent_rows")),
                safe_int(row.get("promotion_proposals_train_consistent_pre_cap")),
                row.get("triage_priority", ""),
                ",".join(row.get("promotion_selected_templates") or []),
            )
        )

    lines.extend(["", "## Falsifiers", ""])
    lines.extend(
        [
            "- If a failed task has promotion-mode train-consistent proposals, it is not a current-template-family gap; treat it as ranking, cap, or slot-selection quality.",
            "- If a mutation row has no slot probes and no template probe selection effect, it cannot support a slot-decoder claim.",
            "- If future neural probes certify without adding template families, the missing-family share should shrink in this report.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_proof(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    mutation_rows = parse_json_rows(args.mutation_verification)
    if not mutation_rows:
        raise RuntimeError(
            "no mutation verification rows loaded from "
            f"{args.mutation_verification}; run arc_neural_discovery with "
            "--mutation-representatives greater than zero before building "
            "ARC neural failure proof"
        )
    candidate_rows = parse_json_rows(args.candidate_sketches)
    triage_rows = parse_json_rows(args.triage)
    split_dirs = public_split_dirs(args.public_arc_dir)

    mutation_by_task = aggregate_mutations(mutation_rows)
    triage_by_task = triage_index(triage_rows)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks_by_split: dict[str, list[str]] = defaultdict(list)
    for split, task_id in sorted(mutation_by_task):
        if split in split_dirs:
            tasks_by_split[split].append(task_id)

    replay_rows: list[dict[str, Any]] = []
    replay_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for split, task_ids in sorted(tasks_by_split.items()):
        if args.max_promotion_tasks > 0:
            task_ids = task_ids[: args.max_promotion_tasks]
        log_path, exit_code, command, metrics_by_task = run_replay(
            args.arc_agi,
            split,
            task_ids,
            split_dirs[split],
            output_dir,
        )
        for task_id in task_ids:
            metrics = metrics_by_task.get(task_id, {})
            row = {
                "schema": PROMOTION_REPLAY_SCHEMA,
                "report_version": 1,
                "split": split,
                "task_id": task_id,
                "exit_code": exit_code,
                "log_path": str(log_path),
                "command": command,
                "replayed": bool(metrics),
                **metrics,
            }
            replay_rows.append(row)
            replay_by_key[(split, task_id)] = row

    proof_rows: list[dict[str, Any]] = []
    for key, mutation in sorted(mutation_by_task.items()):
        split, task_id = key
        triage = triage_by_task.get(key, {})
        replay = replay_by_key.get(key, {})
        accepted_existing = safe_int(mutation.get("accepted_existing_rows")) > 0
        slot_decoder_evidence = (
            not accepted_existing
            and safe_int(mutation.get("slot_probe_rows")) > 0
            and (
                safe_int(mutation.get("template_selected_rows")) > 0
                or safe_int(mutation.get("slot_probe_seeded_proposals")) > 0
            )
            and safe_int(mutation.get("mutation_train_consistent_rows")) == 0
        )
        missing_family_evidence = (
            not accepted_existing
            and bool(replay.get("replayed"))
            and safe_int(replay.get("proposals_train_consistent_pre_cap")) == 0
            and not replay.get("certified", False)
            and str(triage.get("priority", "")) in MISSING_FAMILY_PRIORITIES
        )
        row = {
            "schema": SCHEMA,
            "report_version": 1,
            "split": split,
            "task_id": task_id,
            **mutation,
            "promotion_replayed": bool(replay.get("replayed")),
            "promotion_certified": bool(replay.get("certified", False)),
            "promotion_evidence_status": str(replay.get("evidence_status", "")),
            "promotion_evidence_template": str(replay.get("evidence_template", "")),
            "promotion_proposals_generated": safe_int(
                replay.get("proposals_generated")
            ),
            "promotion_proposals_train_consistent_pre_cap": safe_int(
                replay.get("proposals_train_consistent_pre_cap")
            ),
            "promotion_proposals_train_consistent": safe_int(
                replay.get("proposals_train_consistent")
            ),
            "promotion_proposals_train_rejected": safe_int(
                replay.get("proposals_train_rejected")
            ),
            "promotion_selected_templates": replay.get("selected_templates", []),
            "promotion_train_consistent_templates": replay.get(
                "train_consistent_templates", []
            ),
            "triage_priority": str(triage.get("priority", "")),
            "triage_structural_label": str(triage.get("structural_label", "")),
            "triage_diagnostic_bucket": str(triage.get("diagnostic_bucket", "")),
            "triage_proposals_train_consistent_pre_cap": safe_int(
                triage.get("proposals_train_consistent_pre_cap")
            ),
            "slot_decoder_quality_evidence": slot_decoder_evidence,
            "missing_family_evidence": missing_family_evidence,
        }
        row["classification"] = classify(row)
        proof_rows.append(row)

    sketch_summary = summarize_candidate_sketches(candidate_rows)
    return proof_rows, replay_rows, sketch_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arc-agi", type=Path, default=Path("build/float/arc_agi"))
    parser.add_argument("--public-arc-dir", type=Path, required=True)
    parser.add_argument(
        "--mutation-verification",
        type=Path,
        default=Path(
            "build/float/hypothesis_library_neural_discovery/mutation_verification.jsonl"
        ),
    )
    parser.add_argument(
        "--candidate-sketches",
        type=Path,
        default=Path(
            "build/float/hypothesis_library_neural_discovery/candidate_sketches.jsonl"
        ),
    )
    parser.add_argument(
        "--triage",
        type=Path,
        default=Path("build/float/hypothesis_library_public_triage/coverage_triage.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/float/hypothesis_library_neural_discovery/failure_proof"),
    )
    parser.add_argument(
        "--max-promotion-tasks",
        type=int,
        default=0,
        help="Optional per-split cap for promotion replay tasks; 0 means no cap.",
    )
    args = parser.parse_args()

    proof_rows, replay_rows, sketch_summary = build_proof(args)
    write_jsonl(args.output_dir / "task_proof_rows.jsonl", proof_rows)
    write_jsonl(args.output_dir / "promotion_replay.jsonl", replay_rows)
    (args.output_dir / "summary.md").write_text(
        build_summary(proof_rows, sketch_summary),
        encoding="utf-8",
    )
    print(
        "wrote {} proof rows and {} replay rows to {}".format(
            len(proof_rows),
            len(replay_rows),
            args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
