#!/usr/bin/env python3
"""Evaluate paper claims against machine-readable result artifacts.

The claim registry is stored as a strict JSON subset of YAML so this script can
stay dependency-free. It emits:

  - paper/generated/claims.json
  - paper/generated/claims.typ
  - paper/generated/agent_instructions.md
  - paper/generated/gap_report.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATUS_ORDER = ("supported", "contradicted", "underpowered", "missing", "stale")
FAIL_STATUSES = {"contradicted", "underpowered", "missing", "stale"}
ROADMAP_ROW_RE = re.compile(
    r"^\s*\[(R-[A-Z0-9]+(?:-[A-Z0-9]+)?(?:\.\d+[a-z]?)?)"
    r"(?: \*(?:done|prototype|audit)\*)?\], \[(.*?)\], \[(.*?)\], \[(.*?)\],"
)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"{path} must be valid JSON. It may still be valid YAML, but this "
            "script intentionally parses only the JSON subset for portability."
        ) from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if value is None:
                continue
            text = value.strip()
            if text == "":
                row[key] = ""
                continue
            try:
                if any(ch in text for ch in ".eE"):
                    row[key] = float(text)
                else:
                    row[key] = int(text)
            except ValueError:
                row[key] = value
    return rows


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_source_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip().startswith("{"))
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            row_count = sum(1 for _ in csv.reader(handle))
        return max(row_count - 1, 0)
    return None


def source_metadata(repo: Path, source: str) -> dict[str, Any]:
    path = repo / source
    if not path.exists():
        return {
            "exists": False,
            "sha256": None,
            "size_bytes": None,
            "modified_utc": None,
            "row_count": None,
        }
    stat = path.stat()
    return {
        "exists": True,
        "sha256": file_sha256(path),
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "row_count": count_source_rows(path),
    }


def pct(value: float | int | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{100.0 * float(value):.{digits}f}%"


def num(value: float | int | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def find_first(rows: list[dict[str, Any]], **filters: Any) -> dict[str, Any] | None:
    for row in rows:
        ok = True
        for key, value in filters.items():
            if row.get(key) != value:
                ok = False
                break
        if ok:
            return row
    return None


def grammar_name(row: dict[str, Any]) -> str:
    if row.get("include_candidate_filters") is True and row.get("include_gap_filters") is False:
        return "candidate"
    if row.get("include_candidate_filters") is False and row.get("include_gap_filters") is True:
        return "conditional_gap"
    if row.get("include_candidate_filters") is True and row.get("include_gap_filters") is True:
        return "mixed"
    return "unknown"


def metric_row(label: str, row: dict[str, Any] | None, keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"label": label, "present": row is not None}
    if row is None:
        return out
    for key in keys:
        if key in row:
            out[key] = row[key]
    return out


def as_float(row: dict[str, Any] | None, key: str, default: float = float("nan")) -> float:
    if row is None:
        return default
    value = row.get(key, default)
    if value in ("", None):
        return default
    return float(value)


def as_int(row: dict[str, Any] | None, key: str, default: int = 0) -> int:
    if row is None:
        return default
    value = row.get(key, default)
    if value in ("", None):
        return default
    return int(value)


def secondary_jsonl(
    claim: dict[str, Any],
    repo: Path,
    index: int,
    schema: str | None = None,
) -> list[dict[str, Any]]:
    sources = claim.get("evidence", {}).get("secondary_sources", [])
    if index >= len(sources):
        return []
    rows = read_jsonl(repo / sources[index])
    if schema is None:
        return rows
    return [row for row in rows if row.get("schema") == schema]


def evidence_source_paths(claim: dict[str, Any]) -> list[str]:
    evidence = claim.get("evidence", {})
    paths: list[str] = []
    source = evidence.get("source")
    if source:
        paths.append(str(source))
    for source in evidence.get("secondary_sources", []):
        if source:
            paths.append(str(source))
    return paths


def is_monotone_non_decreasing(values: list[float]) -> bool:
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def version_number(version: str | None) -> int:
    match = re.search(r"v(\d+)", version or "")
    return int(match.group(1)) if match else -1


def base_result(claim: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": claim["id"],
        "title": claim.get("title", ""),
        "claim_type": claim.get("claim_type", ""),
        "claim_text": claim.get("claim_text", ""),
        "required_for_submission": bool(claim.get("required_for_submission", False)),
        "paper_surfaces": list(claim.get("paper_surfaces", [])),
        "roadmap_items": list(claim.get("roadmap_items", [])),
        "experiments": list(claim.get("experiments", [])),
        "decision_rule": claim.get("decision_rule", {}),
        "derived_claims": list(claim.get("derived_claims", [])),
        "provenance": {},
        "residual": [],
        "metrics": {},
        "evidence_rows": [],
        "chart": {},
        "paper_explanation": "",
        "status": "missing",
        "summary": "",
    }


def attach_provenance(result: dict[str, Any], claim: dict[str, Any], sources: dict[str, Any]) -> None:
    evidence = claim.get("evidence", {})
    source = evidence.get("source")
    result["provenance"] = {
        "source": source,
        "secondary_sources": list(evidence.get("secondary_sources", [])),
        "source_command": evidence.get("command"),
        "source_sha256": sources.get(source, {}).get("sha256") if source else None,
        "source_row_count": sources.get(source, {}).get("row_count") if source else None,
        "schemas": list(evidence.get("schemas", [])),
        "filters": evidence.get("filters", {}),
        "required_rows": list(evidence.get("required_rows", [])),
    }


def apply_staleness(result: dict[str, Any], claim: dict[str, Any], sources: dict[str, Any]) -> None:
    staleness = claim.get("staleness", {})
    pinned = staleness.get("source_sha256")
    if not pinned or pinned == "auto":
        return
    source = claim.get("evidence", {}).get("source")
    actual = sources.get(source, {}).get("sha256") if source else None
    if actual != pinned:
        result["status"] = "stale"
        result["residual"] = claim.get("residual_when_not_supported", [])
        result["summary"] = (
            f"Evidence source hash changed: expected {pinned[:12]}, "
            f"got {(actual or 'missing')[:12]}."
        )


def evaluate_prime_gate_delta(claim: dict[str, Any], repo: Path) -> dict[str, Any]:
    result = base_result(claim)
    source = repo / claim["evidence"]["source"]
    rows = read_jsonl(source)
    source_rows = [
        row
        for row in rows
        if row.get("schema") == "crlp.compare.v1" and row.get("task") == "prime_gap"
    ]
    raw = find_first(source_rows, evaluation_mode="raw", feature_set="minimal")
    hand = find_first(source_rows, evaluation_mode="hand_wheel_model", feature_set="minimal")
    learned = find_first(source_rows, evaluation_mode="expanded_rule_model", feature_set="minimal")
    expanded = find_first(source_rows, evaluation_mode="expanded_rules")
    result["evidence_rows"] = [
        metric_row("raw", raw, ["evaluation_mode", "feature_set", "top1", "top10", "nll", "ece", "sample_count"]),
        metric_row("hand_wheel_model", hand, ["evaluation_mode", "feature_set", "top1", "top10", "nll", "ece", "sample_count"]),
        metric_row("expanded_rule_model", learned, ["evaluation_mode", "feature_set", "top1", "top10", "nll", "ece", "sample_count"]),
        metric_row("expanded_rules", expanded, ["evaluation_mode", "false_rejections", "accepted_rules", "sample_count"]),
    ]
    if raw is None or hand is None or learned is None or expanded is None:
        result["status"] = "missing"
        result["residual"] = claim.get("residual_when_not_supported", [])
        result["summary"] = "Missing one or more prime_gap gate-ablation rows."
        return result

    raw_top1 = float(raw.get("top1", float("nan")))
    hand_top1 = float(hand.get("top1", float("nan")))
    learned_top1 = float(learned.get("top1", float("nan")))
    learned_top10 = float(learned.get("top10", float("nan")))
    false_rejections = int(expanded.get("false_rejections", -1))
    sample_count = int(learned.get("sample_count", 0))

    result["metrics"] = {
        "raw_top1": raw_top1,
        "hand_top1": hand_top1,
        "learned_top1": learned_top1,
        "learned_top10": learned_top10,
        "hand_delta_pp": 100.0 * (hand_top1 - raw_top1),
        "learned_delta_pp": 100.0 * (learned_top1 - raw_top1),
        "learned_vs_hand_delta_pp": 100.0 * (learned_top1 - hand_top1),
        "false_rejections": false_rejections,
        "sample_count": sample_count,
    }

    if sample_count < 30000:
        result["status"] = "underpowered"
    elif learned_top1 > hand_top1 > raw_top1 and learned_top10 >= 0.999 and false_rejections == 0:
        result["status"] = "supported"
    else:
        result["status"] = "contradicted"
    if result["status"] != "supported":
        result["residual"] = claim.get("residual_when_not_supported", [])
    result["summary"] = (
        f"learned gate {pct(learned_top1)} vs hand-wheel {pct(hand_top1)} "
        f"vs raw {pct(raw_top1)}; false rejections={false_rejections}."
    )
    return result


def evaluate_prime_proposer_fixed_budget(claim: dict[str, Any], repo: Path) -> dict[str, Any]:
    result = base_result(claim)
    source = repo / claim["evidence"]["source"]
    rows = read_jsonl(source)
    candidates = [
        row
        for row in rows
        if row.get("schema") == "crlp.rule_search.summary.v1"
        and row.get("task") == "prime_gap"
        and int(row.get("max_rules", 0) or 0) == 16
    ]
    by_grammar: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in candidates:
        by_grammar.setdefault(grammar_name(row), {}).setdefault(str(row.get("selection_strategy")), []).append(row)

    needed = ["candidate", "conditional_gap"]
    evidence_rows: list[dict[str, Any]] = []
    grammar_results: dict[str, Any] = {}
    missing = False
    any_static_trails = False
    all_best_neural_win = True

    for grammar in needed:
        group = by_grammar.get(grammar, {})
        static_neural = (group.get("neural_structural") or [None])[0]
        grammar_order = (group.get("grammar_order") or [None])[0]
        structural = (group.get("structural_heuristic") or [None])[0]
        neural_rows = [
            row
            for strategy in ("neural_structural", "closed_loop_neural", "unified_scorer")
            for row in group.get(strategy, [])
        ]
        best_neural = max(neural_rows, key=lambda row: float(row.get("expanded_saved_pct", -math.inf)), default=None)
        if static_neural is None or grammar_order is None or structural is None or best_neural is None:
            missing = True
        baseline_best = max(
            [
                float(grammar_order.get("expanded_saved_pct", -math.inf)) if grammar_order else -math.inf,
                float(structural.get("expanded_saved_pct", -math.inf)) if structural else -math.inf,
            ]
        )
        static_saved = (
            float(static_neural.get("expanded_saved_pct", float("nan")))
            if static_neural is not None
            else float("nan")
        )
        if static_neural is None or static_saved < baseline_best:
            any_static_trails = True
        best_neural_saved = (
            float(best_neural.get("expanded_saved_pct", float("nan")))
            if best_neural is not None
            else float("nan")
        )
        if best_neural is None or best_neural_saved < baseline_best:
            all_best_neural_win = False
        best_neural_hidden = None
        if best_neural is not None:
            best_neural_hidden = (
                best_neural.get("closed_loop_hidden")
                or best_neural.get("unified_scorer_hidden")
            )
        grammar_results[grammar] = {
            "static_neural_saved_pct": static_saved,
            "best_neural_strategy": best_neural.get("selection_strategy") if best_neural else None,
            "best_neural_hidden": best_neural_hidden,
            "best_neural_saved_pct": best_neural_saved,
            "grammar_order_saved_pct": float(grammar_order.get("expanded_saved_pct", float("nan"))) if grammar_order else None,
            "structural_saved_pct": float(structural.get("expanded_saved_pct", float("nan"))) if structural else None,
            "baseline_best_saved_pct": baseline_best if baseline_best != -math.inf else None,
        }
        for label, row in (
            (f"{grammar}:neural_structural", static_neural),
            (f"{grammar}:best_neural", best_neural),
            (f"{grammar}:grammar_order", grammar_order),
            (f"{grammar}:structural_heuristic", structural),
        ):
            evidence_rows.append(
                metric_row(
                    label,
                    row,
                    [
                        "selection_strategy",
                        "closed_loop_hidden",
                        "unified_scorer_hidden",
                        "expanded_saved_pct",
                        "expanded_exact",
                        "accepted_rules",
                        "false_rejections",
                    ],
                )
            )

    result["metrics"] = grammar_results
    result["evidence_rows"] = evidence_rows
    if missing:
        result["status"] = "missing"
    elif any_static_trails:
        result["status"] = "contradicted"
    elif all_best_neural_win:
        result["status"] = "supported"
    else:
        result["status"] = "underpowered"
    if result["status"] != "supported":
        result["residual"] = claim.get("residual_when_not_supported", [])
    cand = grammar_results.get("candidate", {})
    cond = grammar_results.get("conditional_gap", {})
    if missing:
        result["summary"] = "Missing one or more fixed-budget proposer baseline rows."
    elif any_static_trails:
        result["summary"] = (
            "static neural proposer trails deterministic baselines on at least one grammar; "
            f"candidate static={num(cand.get('static_neural_saved_pct'), 2)} saved vs "
            f"grammar={num(cand.get('grammar_order_saved_pct'), 2)}, "
            f"conditional static={num(cond.get('static_neural_saved_pct'), 2)} saved vs "
            f"grammar={num(cond.get('grammar_order_saved_pct'), 2)}."
        )
    elif all_best_neural_win:
        result["summary"] = (
            "best neural proposer matches or beats deterministic baselines for every "
            "reported grammar."
        )
    else:
        result["summary"] = (
            "best neural proposer is mixed against deterministic baselines; keep the "
            "claim underpowered until budget and seed sweeps land."
        )
    return result


def evaluate_arc_ood_shared_vs_single(claim: dict[str, Any], repo: Path) -> dict[str, Any]:
    result = base_result(claim)
    source = repo / claim["evidence"]["source"]
    rows = read_csv(source)
    shared = find_first(rows, model_kind="shared")
    single = find_first(rows, model_kind="single_task")
    result["evidence_rows"] = [
        metric_row("shared", shared, ["model_kind", "seeds", "top1_mean", "nll_mean", "ece_mean", "train_updates_mean"]),
        metric_row("single_task", single, ["model_kind", "seeds", "top1_mean", "nll_mean", "ece_mean", "train_updates_mean"]),
    ]
    if shared is None or single is None:
        result["status"] = "missing"
        result["residual"] = claim.get("residual_when_not_supported", [])
        result["summary"] = "Missing shared or single_task ARC OOD aggregate row."
        return result

    shared_top1 = float(shared.get("top1_mean", float("nan")))
    single_top1 = float(single.get("top1_mean", float("nan")))
    shared_nll = float(shared.get("nll_mean", float("nan")))
    single_nll = float(single.get("nll_mean", float("nan")))
    shared_ece = float(shared.get("ece_mean", float("nan")))
    single_ece = float(single.get("ece_mean", float("nan")))
    shared_seeds = int(shared.get("seeds", 0) or 0)
    single_seeds = int(single.get("seeds", 0) or 0)
    result["metrics"] = {
        "shared_top1": shared_top1,
        "single_top1": single_top1,
        "top1_delta_pp": 100.0 * (shared_top1 - single_top1),
        "shared_nll": shared_nll,
        "single_nll": single_nll,
        "nll_delta": shared_nll - single_nll,
        "shared_ece": shared_ece,
        "single_ece": single_ece,
        "ece_delta": shared_ece - single_ece,
        "shared_seeds": shared_seeds,
        "single_seeds": single_seeds,
    }
    if shared_seeds < 3 or single_seeds < 3:
        result["status"] = "underpowered"
    elif shared_top1 > single_top1 and shared_nll < single_nll:
        result["status"] = "supported"
    else:
        result["status"] = "contradicted"
    if result["status"] != "supported":
        result["residual"] = claim.get("residual_when_not_supported", [])
    result["summary"] = (
        f"shared OOD top-1 {pct(shared_top1)} vs single {pct(single_top1)} "
        f"({100.0 * (shared_top1 - single_top1):+.2f}pp), NLL {num(shared_nll)} vs {num(single_nll)}."
    )
    result["paper_explanation"] = (
        "This supports the narrow residual-generalization claim: the shared scorer loses "
        "on the fixed ARC training task, but improves held-out same-envelope ARC ranking "
        "and log loss."
    )
    result["chart"] = {
        "kind": "delta_bar",
        "unit": "pp",
        "y_label": "shared minus single top-1",
        "rows": [
            {
                "label": "ARC OOD",
                "value": 100.0 * (shared_top1 - single_top1),
                "shared_top1": shared_top1,
                "single_top1": single_top1,
            }
        ],
    }
    return result


def evaluate_arc_gate_invariance(claim: dict[str, Any], repo: Path) -> dict[str, Any]:
    result = base_result(claim)
    source = repo / claim["evidence"]["source"]
    rows = read_csv(source)
    train = find_first(rows, split="train")
    eval_row = find_first(rows, split="eval")
    mismatch_paths = claim.get("evidence", {}).get("secondary_sources", [])
    mismatch_count = 0
    for mismatch_source in mismatch_paths:
        mismatch_path = repo / mismatch_source
        if mismatch_path.exists():
            mismatch_count += len(read_csv(mismatch_path))
        else:
            mismatch_count += 1

    result["evidence_rows"] = [
        metric_row(
            "train",
            train,
            [
                "split",
                "transform_rows",
                "loader_loaded_tasks",
                "loader_parse_errors",
                "loader_certified_tasks",
                "loader_test_correct_cells",
                "proposer_rows",
                "budgets",
            ],
        ),
        metric_row(
            "eval",
            eval_row,
            [
                "split",
                "transform_rows",
                "loader_loaded_tasks",
                "loader_parse_errors",
                "loader_certified_tasks",
                "loader_test_correct_cells",
                "proposer_rows",
                "budgets",
            ],
        ),
        {"label": "context_mismatches", "present": True, "rows": mismatch_count},
    ]
    if train is None or eval_row is None:
        result["status"] = "missing"
        result["residual"] = claim.get("residual_when_not_supported", [])
        result["summary"] = "Missing ARC gate-invariance train or eval summary row."
        return result

    train_parse_errors = as_int(train, "loader_parse_errors", -1)
    eval_parse_errors = as_int(eval_row, "loader_parse_errors", -1)
    train_certified = as_int(train, "loader_certified_tasks")
    eval_certified = as_int(eval_row, "loader_certified_tasks")
    train_cells = as_int(train, "loader_test_correct_cells")
    eval_cells = as_int(eval_row, "loader_test_correct_cells")
    train_proposer_rows = as_int(train, "proposer_rows")
    eval_proposer_rows = as_int(eval_row, "proposer_rows")
    result["metrics"] = {
        "train_certified_tasks": train_certified,
        "eval_certified_tasks": eval_certified,
        "train_correct_cells": train_cells,
        "eval_correct_cells": eval_cells,
        "train_proposer_rows": train_proposer_rows,
        "eval_proposer_rows": eval_proposer_rows,
        "mismatch_rows": mismatch_count,
    }
    if train_parse_errors != 0 or eval_parse_errors != 0:
        result["status"] = "contradicted"
    elif mismatch_count != 0:
        result["status"] = "contradicted"
    elif train_certified >= 24 and eval_certified >= 4 and train_proposer_rows > 0 and eval_proposer_rows > 0:
        result["status"] = "supported"
    else:
        result["status"] = "underpowered"
    if result["status"] != "supported":
        result["residual"] = claim.get("residual_when_not_supported", [])
    result["summary"] = (
        f"ARC gate/proposer invariance holds across scorer-context labels: "
        f"train {train_certified}/400 certified tasks, eval {eval_certified}/400, "
        f"mismatches={mismatch_count}."
    )
    result["paper_explanation"] = (
        "This separates the certified ARC gate from learned scorer choice: the public "
        "ARC gate/proposer measurements are currently scorer-independent, so scorer "
        "claims and gate claims remain cleanly separated."
    )
    result["chart"] = {
        "kind": "coverage_bar",
        "unit": "correct cells",
        "y_label": "certified correct cells",
        "rows": [
            {"label": "ARC train", "value": train_cells, "certified_tasks": train_certified},
            {"label": "ARC eval", "value": eval_cells, "certified_tasks": eval_certified},
        ],
    }
    return result


def evaluate_arc_public_evidence_surface(claim: dict[str, Any], repo: Path) -> dict[str, Any]:
    result = base_result(claim)
    source = repo / claim["evidence"]["source"]
    rows = read_csv(source)
    v10_eval = find_first(rows, version="v10_recorded", split="evaluation")
    v12_eval = find_first(rows, version="v12_recorded", split="evaluation")
    measured_evals = [
        row
        for row in rows
        if str(row.get("version", "")).endswith("_measured") and row.get("split") == "evaluation"
    ]
    measured_evals.sort(key=lambda row: version_number(str(row.get("version"))))
    current_eval = measured_evals[-1] if measured_evals else None
    current_version = str(current_eval.get("version")) if current_eval is not None else ""
    current_train = find_first(rows, version=current_version, split="training") if current_version else None
    result["evidence_rows"] = [
        metric_row("v10_eval", v10_eval, ["version", "split", "loaded_tasks", "certified_tasks", "test_correct_cells", "cell_accuracy"]),
        metric_row("v12_eval", v12_eval, ["version", "split", "loaded_tasks", "certified_tasks", "test_correct_cells", "cell_accuracy"]),
        metric_row("current_eval", current_eval, ["version", "split", "loaded_tasks", "certified_tasks", "test_correct_cells", "cell_accuracy"]),
        metric_row("current_train", current_train, ["version", "split", "loaded_tasks", "certified_tasks", "test_correct_cells", "cell_accuracy"]),
    ]
    if v10_eval is None or v12_eval is None or current_eval is None or current_train is None:
        result["status"] = "missing"
        result["residual"] = claim.get("residual_when_not_supported", [])
        result["summary"] = "Missing one or more public ARC evidence-surface rows."
        return result

    v10_eval_tasks = as_int(v10_eval, "certified_tasks")
    v12_eval_tasks = as_int(v12_eval, "certified_tasks")
    current_eval_tasks = as_int(current_eval, "certified_tasks")
    v10_eval_cells = as_int(v10_eval, "test_correct_cells")
    v12_eval_cells = as_int(v12_eval, "test_correct_cells")
    current_eval_cells = as_int(current_eval, "test_correct_cells")
    current_train_tasks = as_int(current_train, "certified_tasks")
    current_train_cells = as_int(current_train, "test_correct_cells")
    result["metrics"] = {
        "v10_eval_certified_tasks": v10_eval_tasks,
        "v12_eval_certified_tasks": v12_eval_tasks,
        "current_eval_certified_tasks": current_eval_tasks,
        "v10_eval_correct_cells": v10_eval_cells,
        "v12_eval_correct_cells": v12_eval_cells,
        "current_eval_correct_cells": current_eval_cells,
        "current_train_certified_tasks": current_train_tasks,
        "current_train_correct_cells": current_train_cells,
        "current_eval_cell_accuracy": as_float(current_eval, "cell_accuracy"),
        "current_version": current_version,
    }
    if current_eval_tasks >= v12_eval_tasks >= v10_eval_tasks and current_eval_cells >= v12_eval_cells >= v10_eval_cells:
        result["status"] = "supported"
    else:
        result["status"] = "contradicted"
    if result["status"] != "supported":
        result["residual"] = claim.get("residual_when_not_supported", [])
    result["summary"] = (
        f"public ARC evidence surface improves from v10 eval {v10_eval_tasks} tasks/"
        f"{v10_eval_cells} cells to {current_version} eval {current_eval_tasks} tasks/"
        f"{current_eval_cells} cells; {current_version} train remains "
        f"{current_train_tasks} tasks/{current_train_cells} cells."
    )
    result["paper_explanation"] = (
        "This is not a broad ARC-performance claim. It is the public-corpus parity row: "
        "the compiler can now distinguish measured ARC gate coverage, version-to-version "
        "movement, and the much larger unresolved residual."
    )
    result["chart"] = {
        "kind": "coverage_bar",
        "unit": "correct cells",
        "y_label": "evaluation correct cells",
        "rows": [
            {"label": "v10 eval", "value": v10_eval_cells, "certified_tasks": v10_eval_tasks},
            {"label": "v12 eval", "value": v12_eval_cells, "certified_tasks": v12_eval_tasks},
            {"label": f"{current_version} eval", "value": current_eval_cells, "certified_tasks": current_eval_tasks},
        ],
    }
    return result


def evaluate_arc_ood_calibration_tradeoff(claim: dict[str, Any], repo: Path) -> dict[str, Any]:
    result = base_result(claim)
    source = repo / claim["evidence"]["source"]
    rows = read_csv(source)
    fixed_shared = find_first(rows, scope="fixed_task", eval_name="arc_agi", model_kind="shared")
    fixed_single = find_first(rows, scope="fixed_task", eval_name="arc_agi", model_kind="single_task")
    ood_shared = find_first(rows, scope="arc_ood_aggregate", model_kind="shared")
    ood_single = find_first(rows, scope="arc_ood_aggregate", model_kind="single_task")
    result["evidence_rows"] = [
        metric_row("fixed_shared", fixed_shared, ["scope", "model_kind", "seeds", "top1_mean", "nll_mean", "ece_mean"]),
        metric_row("fixed_single", fixed_single, ["scope", "model_kind", "seeds", "top1_mean", "nll_mean", "ece_mean"]),
        metric_row("ood_shared", ood_shared, ["scope", "model_kind", "seeds", "top1_mean", "nll_mean", "ece_mean"]),
        metric_row("ood_single", ood_single, ["scope", "model_kind", "seeds", "top1_mean", "nll_mean", "ece_mean"]),
    ]
    if any(row is None for row in (fixed_shared, fixed_single, ood_shared, ood_single)):
        result["status"] = "missing"
        result["residual"] = claim.get("residual_when_not_supported", [])
        result["summary"] = "Missing ARC fixed-task or OOD calibration rows."
        return result

    fixed_nll_delta = as_float(fixed_shared, "nll_mean") - as_float(fixed_single, "nll_mean")
    fixed_ece_delta = as_float(fixed_shared, "ece_mean") - as_float(fixed_single, "ece_mean")
    ood_nll_delta = as_float(ood_shared, "nll_mean") - as_float(ood_single, "nll_mean")
    ood_ece_delta = as_float(ood_shared, "ece_mean") - as_float(ood_single, "ece_mean")
    fixed_seeds = min(as_int(fixed_shared, "seeds"), as_int(fixed_single, "seeds"))
    ood_seeds = min(as_int(ood_shared, "seeds"), as_int(ood_single, "seeds"))
    result["metrics"] = {
        "fixed_nll_delta_shared_minus_single": fixed_nll_delta,
        "fixed_ece_delta_shared_minus_single": fixed_ece_delta,
        "ood_nll_delta_shared_minus_single": ood_nll_delta,
        "ood_ece_delta_shared_minus_single": ood_ece_delta,
        "fixed_ece_improvement_pp": -100.0 * fixed_ece_delta,
        "ood_ece_improvement_pp": -100.0 * ood_ece_delta,
        "fixed_nll_improvement": -fixed_nll_delta,
        "ood_nll_improvement": -ood_nll_delta,
        "fixed_seeds": fixed_seeds,
        "ood_seeds": ood_seeds,
    }
    if fixed_seeds < 3 or ood_seeds < 3:
        result["status"] = "underpowered"
    elif fixed_nll_delta < 0 and fixed_ece_delta > 0 and ood_nll_delta < 0 and ood_ece_delta < 0:
        result["status"] = "supported"
    else:
        result["status"] = "contradicted"
    if result["status"] != "supported":
        result["residual"] = claim.get("residual_when_not_supported", [])
    result["summary"] = (
        "ARC shared scorer is a calibration tradeoff: fixed-task NLL improves "
        f"({num(as_float(fixed_shared, 'nll_mean'))} vs {num(as_float(fixed_single, 'nll_mean'))}) "
        f"while ECE worsens ({num(as_float(fixed_shared, 'ece_mean'))} vs {num(as_float(fixed_single, 'ece_mean'))}); "
        f"OOD improves both NLL ({num(as_float(ood_shared, 'nll_mean'))} vs {num(as_float(ood_single, 'nll_mean'))}) "
        f"and ECE ({num(as_float(ood_shared, 'ece_mean'))} vs {num(as_float(ood_single, 'ece_mean'))})."
    )
    result["paper_explanation"] = (
        "The calibration claim is explicitly conditional: shared training is not simply "
        "better calibrated everywhere. It is worse on fixed ARC ECE but better on OOD ECE, "
        "while improving log loss in both settings."
    )
    result["chart"] = {
        "kind": "delta_bar",
        "unit": "pp",
        "y_label": "shared ECE improvement vs single",
        "rows": [
            {"label": "fixed ARC", "value": -100.0 * fixed_ece_delta, "nll_improvement": -fixed_nll_delta},
            {"label": "ARC OOD", "value": -100.0 * ood_ece_delta, "nll_improvement": -ood_nll_delta},
        ],
    }
    return result


def evaluate_delayed_claim_leave_one_out(claim: dict[str, Any], repo: Path) -> dict[str, Any]:
    result = base_result(claim)
    source = repo / claim["evidence"]["source"]
    rows = read_csv(source)
    all_row = find_first(rows, training_set="all")
    claim_only = find_first(rows, training_set="claim_only")
    no_grid = find_first(rows, training_set="no_grid_path")
    no_arc = find_first(rows, training_set="no_arc_agi")
    result["evidence_rows"] = [
        metric_row("all", all_row, ["training_set", "seeds", "top1_mean", "nll_mean", "ece_mean", "top1_delta_vs_claim_only", "top1_delta_vs_all"]),
        metric_row("claim_only", claim_only, ["training_set", "seeds", "top1_mean", "nll_mean", "ece_mean", "top1_delta_vs_claim_only", "top1_delta_vs_all"]),
        metric_row("no_grid_path", no_grid, ["training_set", "seeds", "top1_mean", "nll_mean", "ece_mean", "top1_delta_vs_claim_only", "top1_delta_vs_all"]),
        metric_row("no_arc_agi", no_arc, ["training_set", "seeds", "top1_mean", "nll_mean", "ece_mean", "top1_delta_vs_claim_only", "top1_delta_vs_all"]),
    ]
    if all_row is None or claim_only is None or no_grid is None or no_arc is None:
        result["status"] = "missing"
        result["residual"] = claim.get("residual_when_not_supported", [])
        result["summary"] = "Missing delayed-claim leave-one-out comparison rows."
        return result

    all_top1 = as_float(all_row, "top1_mean")
    claim_only_top1 = as_float(claim_only, "top1_mean")
    no_grid_top1 = as_float(no_grid, "top1_mean")
    no_arc_top1 = as_float(no_arc, "top1_mean")
    seeds = min(as_int(all_row, "seeds"), as_int(claim_only, "seeds"), as_int(no_grid, "seeds"), as_int(no_arc, "seeds"))
    result["metrics"] = {
        "all_top1": all_top1,
        "claim_only_top1": claim_only_top1,
        "no_grid_path_top1": no_grid_top1,
        "no_arc_agi_top1": no_arc_top1,
        "all_vs_claim_only_delta_pp": 100.0 * (all_top1 - claim_only_top1),
        "no_grid_path_vs_all_delta_pp": 100.0 * (no_grid_top1 - all_top1),
        "no_arc_agi_vs_all_delta_pp": 100.0 * (no_arc_top1 - all_top1),
        "seeds": seeds,
    }
    if seeds < 3:
        result["status"] = "underpowered"
    elif all_top1 > claim_only_top1 and no_grid_top1 < all_top1:
        result["status"] = "supported"
    else:
        result["status"] = "contradicted"
    if result["status"] != "supported":
        result["residual"] = claim.get("residual_when_not_supported", [])
    result["summary"] = (
        f"delayed_claim all-domain top-1 {pct(all_top1)} vs claim-only {pct(claim_only_top1)} "
        f"({100.0 * (all_top1 - claim_only_top1):+.2f}pp); removing grid_path falls to "
        f"{pct(no_grid_top1)}, while removing ARC rises to {pct(no_arc_top1)}."
    )
    result["paper_explanation"] = (
        "This replaces the earlier broad positive-transfer claim with a narrower one: "
        "delayed_claim benefits from shared training in this envelope, grid_path appears "
        "to be the helpful sibling, and ARC is mild interference."
    )
    result["chart"] = {
        "kind": "delta_bar",
        "unit": "pp",
        "y_label": "top-1 delta vs claim-only",
        "rows": [
            {
                "label": str(row.get("training_set")),
                "value": 100.0 * as_float(row, "top1_delta_vs_claim_only"),
                "top1": as_float(row, "top1_mean"),
            }
            for row in rows
        ],
    }
    return result


def evaluate_multitask_capacity_sweep(claim: dict[str, Any], repo: Path) -> dict[str, Any]:
    result = base_result(claim)
    source = repo / claim["evidence"]["source"]
    rows = read_csv(source)
    arc_rows = [row for row in rows if row.get("domain") == "arc_agi"]
    claim_rows = [row for row in rows if row.get("domain") == "delayed_claim"]
    arc_rows.sort(key=lambda row: as_float(row, "parameters_mean"))
    claim_by_hidden = {str(row.get("hidden")): row for row in claim_rows}
    result["evidence_rows"] = [
        metric_row(
            f"arc:{row.get('hidden')}",
            row,
            [
                "hidden",
                "seeds",
                "parameters_mean",
                "shared_top1_mean",
                "single_top1_mean",
                "top1_delta_shared_minus_single",
                "shared_nll_mean",
                "single_nll_mean",
            ],
        )
        for row in arc_rows
    ]
    result["evidence_rows"].extend(
        metric_row(
            f"delayed_claim:{row.get('hidden')}",
            row,
            [
                "hidden",
                "seeds",
                "parameters_mean",
                "shared_top1_mean",
                "single_top1_mean",
                "top1_delta_shared_minus_single",
            ],
        )
        for row in claim_rows
    )
    if len(arc_rows) < 4 or len(claim_rows) < 4:
        result["status"] = "missing"
        result["residual"] = claim.get("residual_when_not_supported", [])
        result["summary"] = "Missing one or more capacity-sweep ARC or delayed-claim rows."
        return result

    arc_deltas = [as_float(row, "top1_delta_shared_minus_single") for row in arc_rows]
    claim_deltas = [as_float(claim_by_hidden.get(str(row.get("hidden"))), "top1_delta_shared_minus_single") for row in arc_rows]
    seeds = min(as_int(row, "seeds") for row in arc_rows + claim_rows)
    best_arc_row = max(arc_rows, key=lambda row: as_float(row, "shared_top1_mean"))
    result["metrics"] = {
        "capacity_count": len(arc_rows),
        "seeds": seeds,
        "best_arc_hidden": str(best_arc_row.get("hidden")),
        "best_arc_shared_top1": as_float(best_arc_row, "shared_top1_mean"),
        "best_arc_delta_pp": 100.0 * as_float(best_arc_row, "top1_delta_shared_minus_single"),
        "arc_delta_min_pp": 100.0 * min(arc_deltas),
        "arc_delta_max_pp": 100.0 * max(arc_deltas),
        "claim_delta_min_pp": 100.0 * min(claim_deltas),
        "claim_delta_max_pp": 100.0 * max(claim_deltas),
        "arc_delta_monotone_non_decreasing": is_monotone_non_decreasing(arc_deltas),
    }
    if seeds < 3:
        result["status"] = "underpowered"
    elif max(arc_deltas) < 0 and min(claim_deltas) >= 0 and not is_monotone_non_decreasing(arc_deltas):
        result["status"] = "supported"
    else:
        result["status"] = "contradicted"
    if result["status"] != "supported":
        result["residual"] = claim.get("residual_when_not_supported", [])
    result["summary"] = (
        "capacity sweep rules out a simple wider-is-better ARC story: best fixed-ARC shared row is "
        f"{best_arc_row.get('hidden')} at {pct(as_float(best_arc_row, 'shared_top1_mean'))}, "
        f"but every shared ARC row trails single-task; delayed_claim deltas remain "
        f"{100.0 * min(claim_deltas):+.2f}pp to {100.0 * max(claim_deltas):+.2f}pp."
    )
    result["paper_explanation"] = (
        "This keeps the architecture roadmap disciplined: capacity changes the ARC gap, "
        "but does not monotonically close it, so adapters or ordering should not be claimed "
        "until the tiny-net plateau check is done."
    )
    result["chart"] = {
        "kind": "delta_bar",
        "unit": "pp",
        "y_label": "shared minus single top-1",
        "rows": [
            {
                "label": str(row.get("hidden")),
                "value": 100.0 * as_float(row, "top1_delta_shared_minus_single"),
                "arc_shared_top1": as_float(row, "shared_top1_mean"),
                "arc_single_top1": as_float(row, "single_top1_mean"),
                "claim_delta_pp": 100.0 * as_float(claim_by_hidden.get(str(row.get("hidden"))), "top1_delta_shared_minus_single"),
            }
            for row in arc_rows
        ],
    }
    return result


def evaluate_tiny_net_plateau_audit(claim: dict[str, Any], repo: Path) -> dict[str, Any]:
    result = base_result(claim)
    source = repo / claim["evidence"]["source"]
    plateau_rows = [
        row
        for row in read_jsonl(source)
        if row.get("schema") == "crlp.multitask.tiny_net_plateau.v1"
    ]
    width_rows = secondary_jsonl(
        claim,
        repo,
        0,
        "crlp.multitask.tiny_net_width_help.v1",
    )
    shared_rows = [
        row for row in plateau_rows if row.get("model_kind") == "shared"
    ]
    core_domains = {"prime_gap", "grid_path", "delayed_claim", "arc_agi"}
    capacities = sorted({str(row.get("hidden")) for row in shared_rows})
    domains = {str(row.get("domain")) for row in shared_rows}
    arc_width = find_first(width_rows, domain="arc_agi", model_kind="shared")
    claim_width = find_first(width_rows, domain="delayed_claim", model_kind="shared")
    result["evidence_rows"] = [
        {"label": "tiny_net_plateau_rows", "present": bool(plateau_rows), "rows": len(plateau_rows)},
        {"label": "tiny_net_width_rows", "present": bool(width_rows), "rows": len(width_rows)},
        metric_row("arc_width_help", arc_width, ["best_hidden", "best_top1_mean", "largest_minus_smallest_top1", "larger_width_helped"]),
        metric_row("delayed_width_help", claim_width, ["best_hidden", "best_top1_mean", "largest_minus_smallest_top1", "larger_width_helped"]),
    ]
    if not plateau_rows or not width_rows:
        result["status"] = "missing"
        result["residual"] = claim.get("residual_when_not_supported", [])
        result["summary"] = "Missing tiny-net plateau summary or width-help rows."
        return result

    plateau_count = sum(1 for row in shared_rows if row.get("plateau_detected") is True)
    shared_count = len(shared_rows)
    min_capacities_per_domain = min(
        (
            len({str(row.get("hidden")) for row in shared_rows if row.get("domain") == domain})
            for domain in core_domains
        ),
        default=0,
    )
    result["metrics"] = {
        "plateau_rows": len(plateau_rows),
        "shared_rows": shared_count,
        "width_rows": len(width_rows),
        "capacities": capacities,
        "domains": sorted(domains),
        "min_capacities_per_core_domain": min_capacities_per_domain,
        "shared_plateau_count": plateau_count,
        "shared_plateau_rate": plateau_count / shared_count if shared_count else 0.0,
        "arc_best_hidden": arc_width.get("best_hidden") if arc_width else None,
        "arc_larger_width_helped": arc_width.get("larger_width_helped") if arc_width else None,
    }
    if not core_domains.issubset(domains) or min_capacities_per_domain < 4:
        result["status"] = "underpowered"
    elif plateau_count < shared_count:
        result["status"] = "underpowered"
    else:
        result["status"] = "supported"
    if result["status"] != "supported":
        result["residual"] = claim.get("residual_when_not_supported", [])
    result["summary"] = (
        f"tiny-net plateau audit covers {len(capacities)} capacities across "
        f"{len(domains)} domains; {plateau_count}/{shared_count} shared rows meet the "
        f"plateau threshold, and ARC shared width verdict picks "
        f"{arc_width.get('best_hidden') if arc_width else 'n/a'} as best hidden."
    )
    result["paper_explanation"] = (
        "This turns the width discussion into an audit: the paper can distinguish "
        "undertraining from representation limits using learning curves, restarts, "
        "best checkpoints, parameter counts, FLOPs, plateau flags, and width-help rows."
    )
    result["chart"] = {
        "kind": "coverage_bar",
        "unit": "top-1",
        "y_label": "best shared top-1",
        "rows": [
            {
                "label": str(row.get("hidden")),
                "value": 100.0 * as_float(row, "best_top1_mean"),
                "domain": row.get("domain"),
                "plateau_detected": bool(row.get("plateau_detected")),
            }
            for row in shared_rows
            if row.get("domain") == "arc_agi"
        ],
    }
    return result


def evaluate_rejected_candidate_reason_learning(claim: dict[str, Any], repo: Path) -> dict[str, Any]:
    result = base_result(claim)
    source = repo / claim["evidence"]["source"]
    rows = [
        row
        for row in read_jsonl(source)
        if row.get("schema") == "crlp.rejected_candidate_learning.summary.v1"
    ]
    split_rows = secondary_jsonl(
        claim,
        repo,
        0,
        "crlp.rejected_candidate_learning.split_sweep.v1",
    )
    family_rows = secondary_jsonl(
        claim,
        repo,
        1,
        "crlp.rejected_candidate_learning.reason_family_sweep.v1",
    )
    seed_rows = secondary_jsonl(
        claim,
        repo,
        2,
        "crlp.rejected_candidate_learning.seed_interval.v1",
    )
    family_seed_rows = secondary_jsonl(
        claim,
        repo,
        3,
        "crlp.rejected_candidate_learning.reason_family_seed_interval.v1",
    )

    model = find_first(rows, task="prime_gap", strategy="model_score", budget=8)
    prior = find_first(rows, task="prime_gap", strategy="reason_prior", budget=8)
    aware = find_first(rows, task="prime_gap", strategy="reason_aware", budget=8)
    residual = find_first(rows, task="prime_gap", strategy="trained_residual", budget=8)
    prime_splits = [row for row in split_rows if row.get("task") == "prime_gap" and row.get("budget") == 8]
    modular_splits = [row for row in split_rows if row.get("task") == "modular_gate" and row.get("budget") == 8]
    delayed_splits = [row for row in split_rows if row.get("task") == "delayed_claim"]

    result["evidence_rows"] = [
        metric_row("prime:model_score@8", model, ["accepted_rules", "unsafe_rules", "no_support_rules", "safe_duplicate_rules", "validation_utility", "validation_false_rejections"]),
        metric_row("prime:reason_prior@8", prior, ["accepted_rules", "unsafe_rules", "no_support_rules", "safe_duplicate_rules", "validation_utility", "validation_false_rejections"]),
        metric_row("prime:reason_aware@8", aware, ["accepted_rules", "unsafe_rules", "no_support_rules", "safe_duplicate_rules", "validation_utility", "validation_false_rejections"]),
        metric_row("prime:trained_residual@8", residual, ["accepted_rules", "unsafe_rules", "no_support_rules", "safe_duplicate_rules", "validation_utility", "validation_false_rejections"]),
        {"label": "prime_split_sweep@8", "present": bool(prime_splits), "rows": len(prime_splits)},
        {"label": "modular_split_sweep@8", "present": bool(modular_splits), "rows": len(modular_splits)},
        {"label": "reason_family_sweep", "present": bool(family_rows), "rows": len(family_rows)},
        {"label": "seed_interval", "present": bool(seed_rows), "rows": len(seed_rows)},
        {"label": "reason_family_seed_interval", "present": bool(family_seed_rows), "rows": len(family_seed_rows)},
    ]

    if model is None or prior is None or aware is None or residual is None or not prime_splits or not modular_splits:
        result["status"] = "missing"
        result["residual"] = claim.get("residual_when_not_supported", [])
        result["summary"] = "Missing rejected-candidate reason-learning rows for prime_gap or modular split sweeps."
        return result

    model_unsafe = as_int(model, "unsafe_rules")
    model_false = as_int(model, "validation_false_rejections")
    prior_false = as_int(prior, "validation_false_rejections")
    residual_false = as_int(residual, "validation_false_rejections")
    prior_utility = as_float(prior, "validation_utility")
    aware_utility = as_float(aware, "validation_utility")
    residual_utility = as_float(residual, "validation_utility")
    prime_split_pass = all(
        row.get("trained_residual_nonnegative_vs_reason_prior") is True
        and as_int(row, "trained_residual_validation_false_rejections") == 0
        for row in prime_splits
    )
    modular_split_pass = all(
        row.get("trained_residual_nonnegative_vs_reason_prior") is True
        and as_int(row, "trained_residual_validation_false_rejections") == 0
        for row in modular_splits
    )
    delayed_mixed_rows = [
        row
        for row in delayed_splits
        if row.get("trained_residual_nonnegative_vs_reason_prior") is not True
        or as_int(row, "trained_residual_validation_false_rejections") > 0
    ]
    seed_model = find_first(seed_rows, task="prime_gap", strategy="model_score", budget=8)
    seed_prior = find_first(seed_rows, task="prime_gap", strategy="reason_prior", budget=8)
    seed_residual = find_first(seed_rows, task="prime_gap", strategy="trained_residual", budget=8)

    def interval_mean(row: dict[str, Any] | None, metric: str) -> float:
        if row is None:
            return float("nan")
        value = row.get(metric)
        if isinstance(value, dict):
            return float(value.get("mean", float("nan")))
        return float("nan")

    result["metrics"] = {
        "model_score_unsafe_rules": model_unsafe,
        "model_score_validation_false_rejections": model_false,
        "reason_prior_accepted_rules": as_int(prior, "accepted_rules"),
        "reason_prior_safe_duplicate_rules": as_int(prior, "safe_duplicate_rules"),
        "reason_prior_validation_utility": prior_utility,
        "reason_prior_validation_false_rejections": prior_false,
        "reason_aware_validation_utility": aware_utility,
        "trained_residual_accepted_rules": as_int(residual, "accepted_rules"),
        "trained_residual_safe_duplicate_rules": as_int(residual, "safe_duplicate_rules"),
        "trained_residual_validation_utility": residual_utility,
        "trained_residual_validation_false_rejections": residual_false,
        "trained_residual_delta_vs_reason_prior": residual_utility - prior_utility,
        "prime_split_count": len(prime_splits),
        "prime_split_delta_min": min(as_float(row, "trained_residual_delta_validation_utility") for row in prime_splits),
        "modular_split_count": len(modular_splits),
        "modular_split_delta_min": min(as_float(row, "trained_residual_delta_validation_utility") for row in modular_splits),
        "delayed_mixed_rows": len(delayed_mixed_rows),
        "reason_family_rows": len(family_rows),
        "seed_interval_rows": len(seed_rows),
        "seed_interval_count": as_int(seed_residual, "seed_count", 0),
        "seed_interval_model_unsafe_mean": interval_mean(seed_model, "unsafe_rules"),
        "seed_interval_model_false_mean": interval_mean(seed_model, "validation_false_rejections"),
        "seed_interval_prior_false_mean": interval_mean(seed_prior, "validation_false_rejections"),
        "seed_interval_residual_false_mean": interval_mean(seed_residual, "validation_false_rejections"),
        "seed_interval_residual_utility_mean": interval_mean(seed_residual, "validation_utility"),
        "reason_family_seed_interval_rows": len(family_seed_rows),
    }

    if (
        model_unsafe > 0
        and model_false > 0
        and as_int(prior, "unsafe_rules") == 0
        and prior_false == 0
        and as_int(residual, "unsafe_rules") == 0
        and residual_false == 0
        and residual_utility > prior_utility
        and prime_split_pass
        and modular_split_pass
    ):
        result["status"] = "supported"
    else:
        result["status"] = "contradicted"
        result["residual"] = claim.get("residual_when_not_supported", [])

    result["summary"] = (
        f"reason learning repairs the prime proposer failure at budget 8: model_score selects "
        f"{model_unsafe} unsafe rows and causes {model_false} validation false rejections, "
        f"while reason_prior reaches {as_int(prior, 'accepted_rules')} accepted rules/"
        f"{num(prior_utility, 1)} utility and trained_residual reaches "
        f"{as_int(residual, 'accepted_rules')} accepted rules/{num(residual_utility, 1)} "
        f"utility with zero false rejections; prime/modular split sweeps pass, but "
        f"delayed_claim has {len(delayed_mixed_rows)} mixed split rows."
    )
    result["paper_explanation"] = (
        "This is the first substrate-level self-authoring loop: rejected verifier rows are "
        "converted into reasons, then those reasons improve the next proposer pass. The "
        "claim is scoped to the prime/modular sweeps; delayed_claim remains a stress case. "
        "Seed-interval and reason-family artifacts are reported when present so the paper "
        "can distinguish stable proposer repair from a single favorable trace."
    )
    result["chart"] = {
        "kind": "coverage_bar",
        "unit": "validation utility",
        "y_label": "budget-8 validation utility",
        "rows": [
            {"label": "model_score", "value": as_float(model, "validation_utility"), "false_rejections": model_false, "unsafe_rules": model_unsafe},
            {"label": "reason_prior", "value": prior_utility, "false_rejections": prior_false, "unsafe_rules": as_int(prior, "unsafe_rules")},
            {"label": "reason_aware", "value": aware_utility, "false_rejections": as_int(aware, "validation_false_rejections"), "unsafe_rules": as_int(aware, "unsafe_rules")},
            {"label": "trained_residual", "value": residual_utility, "false_rejections": residual_false, "unsafe_rules": as_int(residual, "unsafe_rules")},
        ],
    }
    return result


def evaluate_hypothesis_library_promotion(claim: dict[str, Any], repo: Path) -> dict[str, Any]:
    result = base_result(claim)
    source = repo / claim["evidence"]["source"]
    promotion_rows = [
        row
        for row in read_jsonl(source)
        if row.get("schema") == "crlp.hypothesis_library.promotion.v1"
    ]
    evidence = claim.get("evidence", {})
    no_prior_rows = [
        row
        for row in read_jsonl(repo / evidence["secondary_sources"][0])
        if row.get("schema") == "crlp.program_proposer.summary.v1"
    ]
    slot_prior_rows = [
        row
        for row in read_jsonl(repo / evidence["secondary_sources"][1])
        if row.get("schema") == "crlp.program_proposer.summary.v1"
    ]
    heldout_rows = secondary_jsonl(
        claim,
        repo,
        3,
        "crlp.hypothesis_library.heldout_result.v1",
    )

    promoted_rows = [row for row in promotion_rows if row.get("promoted") is True]

    def learned(rows: list[dict[str, Any]], budget: int) -> dict[str, Any] | None:
        return find_first(rows, domain="arc_agi", task="arc_loader", strategy="learned", budget=budget)

    no_prior_1 = learned(no_prior_rows, 1)
    slot_prior_1 = learned(slot_prior_rows, 1)
    no_prior_8 = learned(no_prior_rows, 8)
    slot_prior_8 = learned(slot_prior_rows, 8)

    result["evidence_rows"] = [
        metric_row("promoted_library", promoted_rows[0] if promoted_rows else None, ["template_name", "support_tasks", "covered_events", "support", "false_rejections", "library_id"]),
        metric_row("no_prior@1", no_prior_1, ["strategy", "budget", "certified_programs", "accepted_programs", "covered_events", "utility", "false_rejections"]),
        metric_row("slot_prior@1", slot_prior_1, ["strategy", "budget", "certified_programs", "accepted_programs", "covered_events", "utility", "false_rejections"]),
        metric_row("no_prior@8", no_prior_8, ["strategy", "budget", "certified_programs", "accepted_programs", "covered_events", "utility", "false_rejections"]),
        metric_row("slot_prior@8", slot_prior_8, ["strategy", "budget", "certified_programs", "accepted_programs", "covered_events", "utility", "false_rejections"]),
        {"label": "heldout_library_feedback", "present": bool(heldout_rows), "rows": len(heldout_rows)},
    ]

    if not promoted_rows or no_prior_1 is None or slot_prior_1 is None or no_prior_8 is None or slot_prior_8 is None:
        result["status"] = "missing"
        result["residual"] = claim.get("residual_when_not_supported", [])
        result["summary"] = "Missing hypothesis-library promotion or prior-ablation rows."
        return result

    promoted = promoted_rows[0]
    no_cert_1 = as_int(no_prior_1, "certified_programs")
    slot_cert_1 = as_int(slot_prior_1, "certified_programs")
    no_utility_8 = as_float(no_prior_8, "utility")
    slot_utility_8 = as_float(slot_prior_8, "utility")
    no_false_max = max(as_int(no_prior_1, "false_rejections"), as_int(no_prior_8, "false_rejections"))
    slot_false_max = max(as_int(slot_prior_1, "false_rejections"), as_int(slot_prior_8, "false_rejections"))
    heldout_primary = [
        row
        for row in heldout_rows
        if row.get("primary_mode") is True and row.get("mode") != "no_prior"
    ]
    heldout_pass_rows = [
        row
        for row in heldout_primary
        if as_int(row, "delta_learned_certified_budget1") >= 0
        and as_float(row, "delta_learned_utility_budget1", 0.0) >= 0.0
        and row.get("tight_budget_improved_or_equal") is True
    ]
    result["metrics"] = {
        "promoted_count": len(promoted_rows),
        "promoted_support_tasks": as_int(promoted, "support_tasks"),
        "promoted_covered_events": as_int(promoted, "covered_events"),
        "promoted_false_rejections": as_int(promoted, "false_rejections"),
        "no_prior_certified_at_1": no_cert_1,
        "slot_prior_certified_at_1": slot_cert_1,
        "certified_at_1_delta": slot_cert_1 - no_cert_1,
        "no_prior_utility_at_8": no_utility_8,
        "slot_prior_utility_at_8": slot_utility_8,
        "utility_at_8_delta": slot_utility_8 - no_utility_8,
        "heldout_result_rows": len(heldout_rows),
        "heldout_primary_rows": len(heldout_primary),
        "heldout_primary_pass_rows": len(heldout_pass_rows),
    }

    if (
        as_int(promoted, "support_tasks") >= 2
        and as_int(promoted, "false_rejections") == 0
        and slot_cert_1 > no_cert_1
        and slot_utility_8 > no_utility_8
        and no_false_max == 0
        and slot_false_max == 0
    ):
        result["status"] = "supported"
    else:
        result["status"] = "contradicted"
        result["residual"] = claim.get("residual_when_not_supported", [])

    result["summary"] = (
        f"certified-library promotion supports fixture-level feedback: promoted "
        f"{promoted.get('template_name')} from {as_int(promoted, 'support_tasks')} tasks/"
        f"{as_int(promoted, 'covered_events')} covered events with zero false rejections; "
        f"slot priors raise learned certified@1 from {no_cert_1} to {slot_cert_1} and "
        f"utility@8 from {num(no_utility_8)} to {num(slot_utility_8)}. "
        f"Held-out split feedback rows: {len(heldout_rows)}."
    )
    result["paper_explanation"] = (
        "This makes the hypothesis library a generated claim rather than a roadmap note. "
        "Fixture feedback and held-out feedback are separated in machine-readable rows; "
        "public ARC promotion remains the R-S.1c gap before using it as a headline ARC claim."
    )
    result["chart"] = {
        "kind": "coverage_bar",
        "unit": "certified programs",
        "y_label": "learned certified programs at budget 1",
        "rows": [
            {"label": "no_prior", "value": no_cert_1, "utility_at_8": no_utility_8},
            {"label": "slot_prior", "value": slot_cert_1, "utility_at_8": slot_utility_8},
        ],
    }
    return result


def evaluate_signal_flight_replay_control(claim: dict[str, Any], repo: Path) -> dict[str, Any]:
    result = base_result(claim)
    source = repo / claim["evidence"]["source"]
    rows = [
        row
        for row in read_jsonl(source)
        if row.get("schema") == "crlp.signal_flight.summary.v1"
        and row.get("task") == "signal_flight"
        and row.get("training_mode") == "counterfactual_utility"
    ]
    shadow_rows = secondary_jsonl(
        claim,
        repo,
        0,
        "crlp.signal_shadow_replay_audit.v1",
    )
    shadow = shadow_rows[-1] if shadow_rows else None
    row = rows[-1] if rows else None
    result["evidence_rows"] = [
        metric_row(
            "signal_flight:neural_vs_baseline",
            row,
            [
                "training_mode",
                "parameter_count",
                "forward_flops",
                "utility_groups",
                "top1",
                "top3",
                "selected_utility",
                "baseline_selected_utility",
                "regret",
                "baseline_regret",
                "selected_damage_rate",
                "baseline_damage_rate",
                "selected_death_rate",
                "baseline_death_rate",
                "selected_candidate_mask_rate",
                "target_candidate_mask_rate",
            ],
        )
    ]
    result["evidence_rows"].append(
        metric_row(
            "signal_live_shadow",
            shadow,
            [
                "shadow_frames",
                "shadow_disagreements",
                "shadow_disagreement_rate",
                "counterfactual_replay_available",
                "inference_ms_p95",
            ],
        )
    )
    if row is None:
        result["status"] = "missing"
        result["residual"] = claim.get("residual_when_not_supported", [])
        result["summary"] = "Missing Signal flight counterfactual utility summary row."
        return result

    top1 = as_float(row, "top1")
    baseline_top1 = as_float(row, "baseline_top1")
    selected_utility = as_float(row, "selected_utility")
    baseline_utility = as_float(row, "baseline_selected_utility")
    regret = as_float(row, "regret")
    baseline_regret = as_float(row, "baseline_regret")
    damage = as_float(row, "selected_damage_rate")
    baseline_damage = as_float(row, "baseline_damage_rate")
    death = as_float(row, "selected_death_rate")
    baseline_death = as_float(row, "baseline_death_rate")
    selected_mask = as_float(row, "selected_candidate_mask_rate")
    target_mask = as_float(row, "target_candidate_mask_rate")
    groups = as_int(row, "utility_groups")
    parameter_count = as_int(row, "parameter_count")
    result["metrics"] = {
        "top1": top1,
        "baseline_top1": baseline_top1,
        "top1_delta_pp": 100.0 * (top1 - baseline_top1),
        "selected_utility": selected_utility,
        "baseline_selected_utility": baseline_utility,
        "utility_delta": selected_utility - baseline_utility,
        "regret": regret,
        "baseline_regret": baseline_regret,
        "regret_delta": regret - baseline_regret,
        "regret_reduction_pct": 100.0 * (baseline_regret - regret) / baseline_regret if baseline_regret > 0 else None,
        "selected_damage_rate": damage,
        "baseline_damage_rate": baseline_damage,
        "selected_death_rate": death,
        "baseline_death_rate": baseline_death,
        "utility_groups": groups,
        "parameter_count": parameter_count,
        "shadow_frames": as_int(shadow, "shadow_frames", 0),
        "shadow_disagreement_rate": as_float(shadow, "shadow_disagreement_rate", 0.0),
        "shadow_counterfactual_replay_available": bool(shadow.get("counterfactual_replay_available")) if shadow else False,
    }

    if groups < 50:
        result["status"] = "underpowered"
    elif parameter_count > 4096:
        result["status"] = "underpowered"
    elif (
        top1 > baseline_top1
        and selected_utility > baseline_utility
        and regret < baseline_regret
        and damage <= baseline_damage
        and death <= baseline_death
        and selected_mask >= 0.999
        and target_mask >= 0.999
    ):
        result["status"] = "supported"
    else:
        result["status"] = "contradicted"
    if result["status"] != "supported":
        result["residual"] = claim.get("residual_when_not_supported", [])

    result["summary"] = (
        f"Signal offline replay ranker top-1 {pct(top1)} vs baseline {pct(baseline_top1)} "
        f"({100.0 * (top1 - baseline_top1):+.2f}pp), utility {num(selected_utility)} vs "
        f"{num(baseline_utility)}, regret {num(regret)} vs {num(baseline_regret)}, "
        f"damage {pct(damage)} vs {pct(baseline_damage)}, deaths {pct(death)} vs {pct(baseline_death)}; "
        f"live shadow frames={as_int(shadow, 'shadow_frames', 0)}."
    )
    result["paper_explanation"] = (
        "This is an application vignette, not a live-control claim: Signal contributes a "
        "replay-certified bounded-control surface where the neural component ranks nine "
        "WASD candidates and simulator replay owns the outcome labels."
    )
    result["chart"] = {
        "kind": "coverage_bar",
        "unit": "top-1",
        "y_label": "held-out top-1",
        "rows": [
            {"label": "baseline", "value": 100.0 * baseline_top1, "regret": baseline_regret},
            {"label": "neural", "value": 100.0 * top1, "regret": regret},
        ],
    }
    return result


def evaluate_signal_live_neural_active(claim: dict[str, Any], repo: Path) -> dict[str, Any]:
    result = base_result(claim)
    source = repo / claim["evidence"]["source"]
    rows = [
        row
        for row in read_jsonl(source)
        if row.get("schema") == "crlp.signal_live_active_summary.v1"
    ]
    frames = secondary_jsonl(claim, repo, 0, "crlp.signal_live_active_frame.v1")
    row = rows[-1] if rows else None
    result["evidence_rows"] = [
        metric_row(
            "signal_live_active",
            row,
            [
                "log_path",
                "rows_seen",
                "active_rows",
                "window_active_rows",
                "latest_brain_mode",
                "latest_active_source",
                "checkpoint_hash",
                "checkpoint_path",
                "feature_set",
                "dry_run_rows",
                "inference_ms_p95",
                "candidate_score_rows",
                "candidate_score_min_count",
                "safety_gate_rows",
                "replay_ready_rows",
                "damage_events",
                "death_events",
                "stale_within_log",
                "latest_row_age_at_capture_ms",
            ],
        ),
        {
            "label": "signal_live_active_frames",
            "present": bool(frames),
            "rows": len(frames),
        },
    ]
    if row is None:
        result["status"] = "missing"
        result["residual"] = claim.get("residual_when_not_supported", [])
        result["summary"] = "Missing Signal live neural-active summary row."
        return result

    active_rows = as_int(row, "active_rows")
    window_active_rows = as_int(row, "window_active_rows")
    dry_run_rows = as_int(row, "dry_run_rows")
    candidate_score_rows = as_int(row, "candidate_score_rows")
    candidate_score_min_count = as_int(row, "candidate_score_min_count")
    safety_gate_rows = as_int(row, "safety_gate_rows")
    replay_ready_rows = as_int(row, "replay_ready_rows")
    damage_events = as_int(row, "damage_events")
    death_events = as_int(row, "death_events")
    inference_p95 = as_float(row, "inference_ms_p95")
    stale_within_log = bool(row.get("stale_within_log"))
    checkpoint_hash = str(row.get("checkpoint_hash") or "")
    checkpoint_path = str(row.get("checkpoint_path") or "")
    feature_set = str(row.get("feature_set") or "")
    latest_brain_mode = str(row.get("latest_brain_mode") or "")
    latest_active_source = str(row.get("latest_active_source") or "")
    min_rows = int(claim.get("decision_rule", {}).get("min_active_rows", 50))
    max_p95_ms = float(claim.get("decision_rule", {}).get("max_inference_p95_ms", 5.0))

    result["metrics"] = {
        "active_rows": active_rows,
        "window_active_rows": window_active_rows,
        "dry_run_rows": dry_run_rows,
        "candidate_score_rows": candidate_score_rows,
        "candidate_score_min_count": candidate_score_min_count,
        "safety_gate_rows": safety_gate_rows,
        "replay_ready_rows": replay_ready_rows,
        "active_frame_rows": len(frames),
        "damage_events": damage_events,
        "death_events": death_events,
        "inference_ms_p95": inference_p95,
        "checkpoint_hash": checkpoint_hash,
        "checkpoint_hash_short": checkpoint_hash[:12],
        "checkpoint_path": checkpoint_path,
        "feature_set": feature_set,
        "latest_brain_mode": latest_brain_mode,
        "latest_active_source": latest_active_source,
        "stale_within_log": stale_within_log,
    }

    contradicted = (
        latest_brain_mode != "neural_active"
        or latest_active_source != "neural"
        or dry_run_rows != 0
        or not checkpoint_hash
        or feature_set != "signal-flight-live-v2"
        or stale_within_log
    )
    underpowered = (
        active_rows < min_rows
        or window_active_rows < min_rows
        or candidate_score_rows < min_rows
        or candidate_score_min_count < 9
        or safety_gate_rows < min_rows
        or replay_ready_rows < min_rows
        or len(frames) < min_rows
        or not math.isfinite(inference_p95)
        or inference_p95 > max_p95_ms
    )

    if contradicted:
        result["status"] = "contradicted"
    elif underpowered:
        result["status"] = "underpowered"
    else:
        result["status"] = "supported"
    if result["status"] != "supported":
        result["residual"] = claim.get("residual_when_not_supported", [])

    result["summary"] = (
        "Signal live neural-active pilot executed "
        f"{window_active_rows} recent neural-source rows from checkpoint "
        f"{checkpoint_hash[:12] or 'missing'} with p95 inference {num(inference_p95, 3)} ms, "
        f"{replay_ready_rows} replay-ready rows, safety-gate rows={safety_gate_rows}, "
        f"damage events={damage_events}, deaths={death_events}; no baseline comparison is claimed."
    )
    result["paper_explanation"] = (
        "This supports only the existence of guarded live neural-active execution: "
        "the running Signal pilot applies checkpoint-ranked WASD actions and logs "
        "the fields needed for replay. It does not yet show that active neural "
        "control improves utility or safety versus the heuristic baseline."
    )
    result["chart"] = {
        "kind": "coverage_bar",
        "unit": "rows",
        "y_label": "logged live rows",
        "rows": [
            {"label": "active", "value": window_active_rows, "p95_ms": inference_p95},
            {"label": "replay-ready", "value": replay_ready_rows, "p95_ms": inference_p95},
            {"label": "active frames", "value": len(frames), "p95_ms": inference_p95},
        ],
    }
    return result


def evaluate_trace_determinism_audit(claim: dict[str, Any], repo: Path) -> dict[str, Any]:
    result = base_result(claim)
    source = repo / claim["evidence"]["source"]
    rows = read_jsonl(source)
    audits = [
        row
        for row in rows
        if row.get("schema") == "crlp.trace_determinism_audit.v1"
    ]
    compare = find_first(rows, schema="crlp.trace_determinism_compare.v1")
    result["evidence_rows"] = [
        metric_row(
            f"audit:{Path(str(row.get('trace_path', 'unknown'))).parent.parent.name or row.get('trace_path', 'unknown')}",
            row,
            [
                "trace_path",
                "row_count",
                "invalid_rows",
                "rank_fp_sensitive_rows",
                "rank_fp_sensitive_rate",
                "rank_fp_sensitive_logged_mismatches",
                "semantic_hash",
                "score_margin_min",
                "score_margin_median",
                "score_margin_p95",
            ],
        )
        for row in audits
    ]
    result["evidence_rows"].append(
        metric_row(
            "float_vs_double_compare",
            compare,
            [
                "baseline_trace_path",
                "all_semantic_hashes_equal",
                "row_counts_equal",
                "semantic_hashes",
                "row_counts",
            ],
        )
    )
    if not audits or compare is None:
        result["status"] = "missing"
        result["residual"] = claim.get("residual_when_not_supported", [])
        result["summary"] = "Missing trace determinism audit or float/double compare row."
        return result

    invalid_rows = sum(as_int(row, "invalid_rows", 0) for row in audits)
    mismatch_rows = sum(as_int(row, "rank_fp_sensitive_logged_mismatches", 0) for row in audits)
    sensitive_rows = sum(as_int(row, "rank_fp_sensitive_rows", 0) for row in audits)
    row_count = sum(as_int(row, "row_count", 0) for row in audits)
    max_sensitive_rate = max((as_float(row, "rank_fp_sensitive_rate", 0.0) for row in audits), default=0.0)
    semantic_hashes = [
        str(row.get("semantic_hash", ""))
        for row in audits
        if row.get("semantic_hash")
    ]
    result["metrics"] = {
        "audit_count": len(audits),
        "row_count": row_count,
        "invalid_rows": invalid_rows,
        "rank_fp_sensitive_rows": sensitive_rows,
        "rank_fp_sensitive_rate_max": max_sensitive_rate,
        "rank_fp_sensitive_logged_mismatches": mismatch_rows,
        "semantic_hash": semantic_hashes[0] if semantic_hashes else "",
        "all_semantic_hashes_equal": bool(compare.get("all_semantic_hashes_equal")),
        "row_counts_equal": bool(compare.get("row_counts_equal")),
    }
    if not semantic_hashes:
        result["status"] = "missing"
    elif invalid_rows != 0 or mismatch_rows != 0:
        result["status"] = "contradicted"
    elif not compare.get("all_semantic_hashes_equal") or not compare.get("row_counts_equal"):
        result["status"] = "contradicted"
    else:
        result["status"] = "supported"
    if result["status"] != "supported":
        result["residual"] = claim.get("residual_when_not_supported", [])
    result["summary"] = (
        f"trace determinism audit covers {len(audits)} traces and {row_count} rows; "
        f"invalid_rows={invalid_rows}, rank_fp_sensitive_rows={sensitive_rows}, "
        f"logged_mismatches={mismatch_rows}, semantic_hashes_equal="
        f"{bool(compare.get('all_semantic_hashes_equal'))}."
    )
    result["paper_explanation"] = (
        "The reproducibility claim is semantic rather than bit-exact: float and double "
        "runs may differ in logits and calibration diagnostics, but the replay-semantic "
        "trace projection has the same hash and exposes every rank-sensitive row."
    )
    result["chart"] = {
        "kind": "coverage_bar",
        "unit": "rows",
        "y_label": "trace rows",
        "rows": [
            {
                "label": Path(str(row.get("trace_path", "trace"))).parent.parent.name or "trace",
                "value": as_int(row, "row_count", 0),
                "rank_fp_sensitive_rows": as_int(row, "rank_fp_sensitive_rows", 0),
                "invalid_rows": as_int(row, "invalid_rows", 0),
            }
            for row in audits
        ],
    }
    return result


EVALUATORS = {
    "prime_gate_delta": evaluate_prime_gate_delta,
    "prime_proposer_fixed_budget": evaluate_prime_proposer_fixed_budget,
    "arc_ood_shared_vs_single": evaluate_arc_ood_shared_vs_single,
    "arc_gate_invariance": evaluate_arc_gate_invariance,
    "arc_public_evidence_surface": evaluate_arc_public_evidence_surface,
    "arc_ood_calibration_tradeoff": evaluate_arc_ood_calibration_tradeoff,
    "delayed_claim_leave_one_out": evaluate_delayed_claim_leave_one_out,
    "multitask_capacity_sweep": evaluate_multitask_capacity_sweep,
    "tiny_net_plateau_audit": evaluate_tiny_net_plateau_audit,
    "rejected_candidate_reason_learning": evaluate_rejected_candidate_reason_learning,
    "hypothesis_library_promotion": evaluate_hypothesis_library_promotion,
    "signal_flight_replay_control": evaluate_signal_flight_replay_control,
    "signal_live_neural_active": evaluate_signal_live_neural_active,
    "trace_determinism_audit": evaluate_trace_determinism_audit,
}


def status_counts(claims: list[dict[str, Any]]) -> dict[str, int]:
    return {status: sum(1 for claim in claims if claim.get("status") == status) for status in STATUS_ORDER}


def typ_string(text: str) -> str:
    return "\"" + text.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n") + "\""


def typ_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return "none"
        return repr(float(value)) if isinstance(value, float) else str(value)
    if value is None:
        return "none"
    if isinstance(value, str):
        return typ_string(value)
    if isinstance(value, list):
        return "(" + ", ".join(typ_value(item) for item in value) + ("," if len(value) == 1 else "") + ")"
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if isinstance(key, str) and key.replace("_", "").replace("-", "").isalnum():
                parts.append(f"{key.replace('-', '_')}: {typ_value(item)}")
        return "(" + ", ".join(parts) + ")"
    return typ_string(str(value))


def var_name(claim_id: str, suffix: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in claim_id)
    return f"claim_{safe}_{suffix}"


def required_failures(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": claim["id"],
            "status": claim["status"],
            "title": claim["title"],
            "summary": claim["summary"],
        }
        for claim in claims
        if claim.get("required_for_submission") and claim.get("status") in FAIL_STATUSES
    ]


def limitation_candidates(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    limitations: list[dict[str, Any]] = []
    for claim in claims:
        status = claim.get("status")
        if status == "supported":
            continue
        severity = "blocking" if claim.get("required_for_submission") else "scope"
        text = claim.get("limitation_when_not_supported") or claim.get("summary") or claim.get("title")
        limitations.append(
            {
                "claim_id": claim["id"],
                "status": status,
                "severity": severity,
                "title": claim.get("title", ""),
                "text": text,
                "roadmap_items": list(claim.get("roadmap_items", [])),
                "residual": list(claim.get("residual", [])),
            }
        )
    return limitations


def claim_family_id(claim_id: str) -> str:
    return claim_id.split(".", 1)[0]


def merge_unique(existing: list[Any], additions: list[Any]) -> list[Any]:
    out = list(existing)
    seen = {json.dumps(item, sort_keys=True) for item in out}
    for item in additions:
        key = json.dumps(item, sort_keys=True)
        if key not in seen:
            out.append(item)
            seen.add(key)
    return out


def experiment_map_from_registry(
    registry: dict[str, Any],
    claims: list[dict[str, Any]],
) -> dict[str, Any]:
    family_rows = list(registry.get("claim_families", []))
    families = {row.get("id"): row for row in family_rows if row.get("id")}
    claim_by_id = {claim["id"]: claim for claim in claims}
    tracks: dict[str, dict[str, Any]] = {}

    def ensure_track(source: dict[str, Any]) -> dict[str, Any]:
        track_id = source["id"]
        track = tracks.setdefault(
            track_id,
            {
                "id": track_id,
                "title": source.get("title", ""),
                "role": source.get("role", ""),
                "claim_families": [],
                "claims": [],
                "roadmap_items": [],
                "commands": [],
                "artifacts": [],
                "agent_instructions": [],
                "success_criteria": [],
                "failure_modes": [],
            },
        )
        for key in ("title", "role"):
            if source.get(key) and not track.get(key):
                track[key] = source[key]
        family = source.get("claim_family")
        if family:
            track["claim_families"] = merge_unique(track["claim_families"], [family])
        if source.get("roadmap_items"):
            track["roadmap_items"] = merge_unique(track["roadmap_items"], list(source["roadmap_items"]))
        if source.get("command"):
            track["commands"] = merge_unique(track["commands"], [source["command"]])
        if source.get("artifacts"):
            track["artifacts"] = merge_unique(track["artifacts"], list(source["artifacts"]))
        if source.get("agent_instruction"):
            track["agent_instructions"] = merge_unique(track["agent_instructions"], [source["agent_instruction"]])
        if source.get("success_criteria"):
            track["success_criteria"] = merge_unique(track["success_criteria"], list(source["success_criteria"]))
        if source.get("failure_modes"):
            track["failure_modes"] = merge_unique(track["failure_modes"], list(source["failure_modes"]))
        return track

    def attach_claim(track: dict[str, Any], claim: dict[str, Any]) -> None:
        family = claim_family_id(claim["id"])
        track["claim_families"] = merge_unique(track["claim_families"], [family])
        claim_ref = {
            "id": claim["id"],
            "family": family,
            "title": claim.get("title", ""),
            "status": claim.get("status", ""),
            "required_for_submission": claim.get("required_for_submission", False),
            "summary": claim.get("summary", ""),
            "paper_surfaces": list(claim.get("paper_surfaces", [])),
            "residual": list(claim.get("residual", [])),
        }
        if all(existing.get("id") != claim["id"] for existing in track["claims"]):
            track["claims"].append(claim_ref)

    for claim in claims:
        for experiment in claim.get("experiments", []):
            track = ensure_track(experiment)
            attach_claim(track, claim)

    for experiment in registry.get("experiment_tracks", []):
        track = ensure_track(experiment)
        for claim_id in experiment.get("claim_ids", []):
            claim = claim_by_id.get(claim_id)
            if claim is not None:
                attach_claim(track, claim)

    for track in tracks.values():
        track["claim_family_details"] = [
            families[family_id]
            for family_id in track.get("claim_families", [])
            if family_id in families
        ]
        track["claims"].sort(key=lambda claim: claim["id"])
        track["claim_families"].sort()
        track["roadmap_items"].sort()

    return {
        "schema": "crlp.paper_claim_experiment_map.v1",
        "claim_families": family_rows,
        "experiments": [tracks[key] for key in sorted(tracks)],
    }


def roadmap_audit(
    paper_path: Path,
    claims: list[dict[str, Any]],
    family_ids: list[str] | None = None,
) -> dict[str, Any]:
    known_ids = {claim["id"] for claim in claims}
    known_families = {claim_family_id(claim["id"]) for claim in claims}
    known_families.update(family_ids or [])
    rows: list[dict[str, Any]] = []
    if not paper_path.exists():
        return {
            "paper": str(paper_path),
            "rows": [],
            "missing_links": [],
            "unregistered_claim_refs": [],
            "row_count": 0,
        }
    for line_no, line in enumerate(paper_path.read_text(encoding="utf-8").splitlines(), start=1):
        match = ROADMAP_ROW_RE.match(line)
        if not match:
            continue
        item_id, work, effort, defends = match.groups()
        refs = sorted(set(re.findall(r"\bC\d+(?:\.[A-Za-z0-9_]+)?\b", defends)))
        family_refs = sorted(ref for ref in refs if "." not in ref)
        claim_refs = sorted(ref for ref in refs if "." in ref)
        registered_family_refs = sorted(ref for ref in family_refs if ref in known_families)
        registered_claim_refs = sorted(ref for ref in claim_refs if ref in known_ids)
        unregistered_refs = sorted(
            set(refs) - set(registered_family_refs) - set(registered_claim_refs)
        )
        linked = bool(refs)
        rows.append(
            {
                "id": item_id,
                "line": line_no,
                "work": work,
                "effort": effort,
                "defends": defends,
                "claim_refs": registered_claim_refs,
                "claim_family_refs": registered_family_refs,
                "claim_tags": refs,
                "unregistered_claim_tags": unregistered_refs,
                "linked": linked,
            }
        )
    missing = [row for row in rows if not row["linked"]]
    unregistered = [row for row in rows if row["unregistered_claim_tags"]]
    return {
        "paper": str(paper_path),
        "row_count": len(rows),
        "linked_count": len(rows) - len(missing),
        "missing_count": len(missing),
        "unregistered_count": len(unregistered),
        "rows": rows,
        "missing_links": missing,
        "unregistered_claim_refs": unregistered,
    }


def write_claims_typ(
    path: Path,
    claims: list[dict[str, Any]],
    counts: dict[str, int],
    sources: dict[str, Any],
    failures: list[dict[str, Any]],
    limitations: list[dict[str, Any]],
    roadmap: dict[str, Any],
) -> None:
    source_rows = [
        {"path": source, **info}
        for source, info in sorted(sources.items())
    ]
    support_rows = [
        {
            "id": claim["id"],
            "family": claim_family_id(claim["id"]),
            "status": claim["status"],
            "required": claim["required_for_submission"],
            "title": claim["title"],
            "summary": claim["summary"],
            "explanation": claim.get("paper_explanation", ""),
            "source": claim.get("provenance", {}).get("source"),
            "chart_kind": claim.get("chart", {}).get("kind", ""),
            "chart_rows": claim.get("chart", {}).get("rows", []),
        }
        for claim in claims
    ]
    lines = [
        "// AUTOGENERATED by scripts/derive_paper_claims.py.",
        "// Do not edit by hand; rerun `make paper-claims` after results change.",
        "",
        f"#let paper_claim_status_counts = {typ_value(counts)}",
        f"#let paper_claim_support_ok = {typ_value(len(failures) == 0)}",
        f"#let paper_claim_required_failures = {typ_value(failures)}",
        f"#let paper_claim_sources = {typ_value(source_rows)}",
        f"#let paper_claim_limitations = {typ_value(limitations)}",
        f"#let paper_claim_roadmap_audit = {typ_value(roadmap)}",
        f"#let paper_claim_support_surfaces = {typ_value(support_rows)}",
        "",
        "#let paper_claims = (",
    ]
    for claim in claims:
        lines.append(
            "  ("
            f"id: {typ_value(claim['id'])}, "
            f"status: {typ_value(claim['status'])}, "
            f"required: {typ_value(claim['required_for_submission'])}, "
            f"title: {typ_value(claim['title'])}, "
            f"source: {typ_value(claim.get('provenance', {}).get('source'))}, "
            f"summary: {typ_value(claim['summary'])}"
            "),"
        )
    lines.append(")")
    lines.append("")
    for claim in claims:
        lines.append(f"#let {var_name(claim['id'], 'status')} = {typ_value(claim['status'])}")
        lines.append(f"#let {var_name(claim['id'], 'summary')} = {typ_value(claim['summary'])}")
        lines.append(f"#let {var_name(claim['id'], 'explanation')} = {typ_value(claim.get('paper_explanation', ''))}")
        lines.append(f"#let {var_name(claim['id'], 'chart')} = {typ_value(claim.get('chart', {}))}")
        lines.append(f"#let {var_name(claim['id'], 'chart_rows')} = {typ_value(claim.get('chart', {}).get('rows', []))}")
        for key, value in flatten_metrics(claim.get("metrics", {})).items():
            lines.append(f"#let {var_name(claim['id'], key)} = {typ_value(value)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        safe_key = "".join(ch if ch.isalnum() else "_" for ch in str(key))
        name = f"{prefix}_{safe_key}" if prefix else safe_key
        if isinstance(value, dict):
            out.update(flatten_metrics(value, name))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[name] = value
    return out


def write_gap_report(path: Path, claims: list[dict[str, Any]], counts: dict[str, int], sources: dict[str, Any]) -> None:
    lines = [
        "# Paper Claim Gap Report",
        "",
        "Generated by `make paper-claims`.",
        "",
        "## Status Counts",
        "",
        "| Status | Count |",
        "| --- | ---: |",
    ]
    for status in STATUS_ORDER:
        lines.append(f"| `{status}` | {counts.get(status, 0)} |")
    lines.extend(["", "## Sources", ""])
    for source, info in sorted(sources.items()):
        if not info.get("exists"):
            lines.append(f"- `{source}`: missing")
            continue
        lines.append(
            f"- `{source}`: `{info.get('sha256', 'missing')}` "
            f"({info.get('row_count')} rows, {info.get('size_bytes')} bytes)"
        )
    lines.extend(["", "## Claims", ""])
    for claim in claims:
        lines.extend(
            [
                f"### {claim['id']}: {claim['title']}",
                "",
                f"- Status: `{claim['status']}`",
                f"- Required: `{str(claim['required_for_submission']).lower()}`",
                f"- Summary: {claim['summary']}",
                f"- Surfaces: {', '.join(f'`{s}`' for s in claim.get('paper_surfaces', [])) or 'none'}",
                "",
            ]
        )
        residual = claim.get("residual", [])
        if residual:
            lines.append("Residual:")
            for item in residual:
                lines.append(f"- {item}")
            lines.append("")
        derived = claim.get("derived_claims", [])
        if derived:
            lines.append("Derived claims:")
            for item in derived:
                lines.append(f"- `{item}`")
            lines.append("")
        lines.append("Evidence rows:")
        for row in claim.get("evidence_rows", []):
            label = row.get("label", "row")
            present = "yes" if row.get("present") else "no"
            metrics = ", ".join(
                f"{key}={value}"
                for key, value in row.items()
                if key not in {"label", "present"}
            )
            lines.append(f"- `{label}` present={present}" + (f"; {metrics}" if metrics else ""))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_limitations_report(path: Path, limitations: list[dict[str, Any]]) -> None:
    lines = [
        "# Generated Paper Limitations",
        "",
        "Generated by `make paper-claims` from non-supported claim states.",
        "",
    ]
    if not limitations:
        lines.append("No generated limitation candidates.")
    for item in limitations:
        roadmap = ", ".join(f"`{entry}`" for entry in item.get("roadmap_items", [])) or "none"
        lines.extend(
            [
                f"## {item['claim_id']} ({item['status']})",
                "",
                f"- Severity: `{item['severity']}`",
                f"- Roadmap: {roadmap}",
                f"- Candidate text: {item['text']}",
                "",
            ]
        )
        residual = item.get("residual", [])
        if residual:
            lines.append("Residual work:")
            for entry in residual:
                lines.append(f"- {entry}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_roadmap_report(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Paper Roadmap Claim-Link Audit",
        "",
        "Generated by `make paper-claims` from the roadmap table in `paper/paper.typ`.",
        "",
        f"- Rows scanned: {audit.get('row_count', 0)}",
        f"- Rows with claim-family tags: {audit.get('linked_count', 0)}",
        f"- Rows without claim-family tags: {audit.get('missing_count', 0)}",
        f"- Rows with claim-family tags not yet in `paper/claims.yaml`: {audit.get('unregistered_count', 0)}",
        "",
    ]
    missing = audit.get("missing_links", [])
    if missing:
        lines.extend(["## Unlinked Rows", ""])
        for row in missing:
            lines.append(
                f"- `{row['id']}` line {row['line']}: defends `{row['defends']}`"
            )
        lines.append("")
    else:
        lines.append("All scanned roadmap rows have a claim-family tag.")
        lines.append("")
    unregistered = audit.get("unregistered_claim_refs", [])
    if unregistered:
        lines.extend(["## Claim Tags Not Yet Registered", ""])
        for row in unregistered:
            tags = ", ".join(f"`{tag}`" for tag in row["unregistered_claim_tags"])
            lines.append(f"- `{row['id']}` line {row['line']}: {tags}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_agent_instructions(path: Path, experiment_map: dict[str, Any]) -> None:
    lines = [
        "# Claim-Experiment Agent Instructions",
        "",
        "Generated by `make paper-claims` from `paper/claims.yaml`.",
        "",
        "Use each track as a research-agent brief: the experiment should either strengthen, falsify, or retire the listed claim target. Keep outputs in the named artifacts so the claim compiler can consume them later.",
        "",
    ]
    experiments = experiment_map.get("experiments", [])
    if not experiments:
        lines.append("No experiment tracks are registered.")
    for experiment in experiments:
        title = experiment.get("title") or experiment["id"]
        lines.extend(
            [
                f"## {experiment['id']}: {title}",
                "",
                f"- Role: `{experiment.get('role') or 'unspecified'}`",
            ]
        )
        claim_refs = experiment.get("claims", [])
        if claim_refs:
            rendered = ", ".join(
                f"`{claim['id']}` ({claim.get('status', 'unknown')})"
                for claim in claim_refs
            )
            lines.append(f"- Evaluated claim targets: {rendered}")
        family_details = experiment.get("claim_family_details", [])
        if family_details:
            rendered = ", ".join(
                f"`{family['id']}` {family.get('title', '').strip()}".strip()
                for family in family_details
            )
            lines.append(f"- Claim families: {rendered}")
        roadmap = experiment.get("roadmap_items", [])
        if roadmap:
            lines.append(f"- Roadmap: {', '.join(f'`{item}`' for item in roadmap)}")
        commands = experiment.get("commands", [])
        if commands:
            lines.append(f"- Command: {', '.join(f'`{command}`' for command in commands)}")
        artifacts = experiment.get("artifacts", [])
        if artifacts:
            lines.append(f"- Artifacts: {', '.join(f'`{artifact}`' for artifact in artifacts)}")
        instructions = experiment.get("agent_instructions", [])
        if instructions:
            lines.append("")
            lines.append("Agent brief:")
            for item in instructions:
                lines.append(f"- {item}")
        criteria = experiment.get("success_criteria", [])
        if criteria:
            lines.append("")
            lines.append("Success criteria:")
            for item in criteria:
                lines.append(f"- {item}")
        failure_modes = experiment.get("failure_modes", [])
        if failure_modes:
            lines.append("")
            lines.append("Failure modes to report:")
            for item in failure_modes:
                lines.append(f"- {item}")
        residuals = [
            residual
            for claim in claim_refs
            for residual in claim.get("residual", [])
        ]
        if residuals:
            lines.append("")
            lines.append("Current claim residuals:")
            for item in merge_unique([], residuals):
                lines.append(f"- {item}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_support_report(
    path: Path,
    failures: list[dict[str, Any]],
    limitations: list[dict[str, Any]],
    roadmap: dict[str, Any],
    experiment_map: dict[str, Any],
) -> None:
    lines = [
        "# Paper Support Check",
        "",
        "Generated by `make paper-support`.",
        "",
        f"- Required-claim failures: {len(failures)}",
        f"- Generated limitation candidates: {len(limitations)}",
        f"- Roadmap rows without claim-family tags: {roadmap.get('missing_count', 0)}",
        f"- Roadmap rows with unregistered claim-family tags: {roadmap.get('unregistered_count', 0)}",
        f"- Claim-linked experiment briefs: {len(experiment_map.get('experiments', []))}",
        "",
    ]
    if failures:
        lines.extend(["## Required Failures", ""])
        for failure in failures:
            lines.append(f"- `{failure['id']}` is `{failure['status']}`: {failure['summary']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", type=Path, default=Path("paper/claims.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/generated"))
    parser.add_argument("--paper", type=Path, default=Path("paper/paper.typ"))
    parser.add_argument("--fail-required", action="store_true")
    parser.add_argument("--support-check", action="store_true")
    parser.add_argument("--fail-unlinked-roadmap", action="store_true")
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    registry = read_json(repo / args.claims)
    output_dir = repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    claims: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    for claim in registry.get("claims", []):
        for evidence_source in evidence_source_paths(claim):
            if evidence_source not in sources:
                sources[evidence_source] = source_metadata(repo, evidence_source)
        evaluator_name = claim.get("builtin_evaluator")
        evaluator = EVALUATORS.get(evaluator_name)
        if evaluator is None:
            result = base_result(claim)
            result["status"] = "missing"
            result["summary"] = f"Unknown evaluator: {evaluator_name}"
        else:
            result = evaluator(claim, repo)
        attach_provenance(result, claim, sources)
        apply_staleness(result, claim, sources)
        claims.append(result)

    counts = status_counts(claims)
    failures = required_failures(claims)
    limitations = limitation_candidates(claims)
    family_ids = [
        row["id"]
        for row in registry.get("claim_families", [])
        if isinstance(row, dict) and row.get("id")
    ]
    roadmap = roadmap_audit(repo / args.paper, claims, family_ids)
    experiment_map = experiment_map_from_registry(registry, claims)
    payload = {
        "schema": "crlp.paper_claims.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "registry": str(args.claims),
        "claim_families": registry.get("claim_families", []),
        "status_counts": counts,
        "sources": sources,
        "support": {
            "ok": len(failures) == 0,
            "required_failures": failures,
            "limitation_count": len(limitations),
            "roadmap_missing_links": roadmap.get("missing_count", 0),
            "roadmap_unregistered_claim_refs": roadmap.get("unregistered_count", 0),
        },
        "roadmap_audit": roadmap,
        "experiment_map": experiment_map,
        "limitations": limitations,
        "claims": claims,
    }
    (output_dir / "claims.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_claims_typ(output_dir / "claims.typ", claims, counts, sources, failures, limitations, roadmap)
    write_gap_report(output_dir / "gap_report.md", claims, counts, sources)
    write_limitations_report(output_dir / "limitations.md", limitations)
    write_roadmap_report(output_dir / "roadmap_links.md", roadmap)
    write_agent_instructions(output_dir / "agent_instructions.md", experiment_map)
    write_support_report(output_dir / "support_report.md", failures, limitations, roadmap, experiment_map)

    print(
        "paper claims: "
        + ", ".join(f"{status}={counts.get(status, 0)}" for status in STATUS_ORDER)
    )
    print(
        f"paper support: {'ok' if not failures else 'failed'}; "
        f"limitations={len(limitations)}; "
        f"roadmap_unlinked={roadmap.get('missing_count', 0)}; "
        f"roadmap_unregistered={roadmap.get('unregistered_count', 0)}"
    )
    print(f"wrote {output_dir / 'gap_report.md'}")
    should_fail_required = args.fail_required or args.support_check
    if should_fail_required and failures:
        for failure in failures:
            print(
                f"required claim not supported: {failure['id']} ({failure['status']})",
                file=sys.stderr,
            )
        return 1
    if args.fail_unlinked_roadmap and roadmap.get("missing_count", 0):
        print(
            f"roadmap rows without known claim-family links: {roadmap['missing_count']}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
