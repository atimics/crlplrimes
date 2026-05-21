#!/usr/bin/env python3
"""Promote repeated protocol residuals into reusable operator candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


CANDIDATE_SCHEMA = "crlp.protocol.candidate_row.v1"
OUTCOME_SCHEMA = "crlp.protocol.verifier_outcome.v1"
OPERATOR_SCHEMA = "crlp.protocol.operator_promotion.v1"
SUMMARY_SCHEMA = "crlp.protocol.operator_promotion.summary.v1"
PRIOR_SCHEMA = "crlp.protocol.operator_prior.v1"


@dataclass
class EvidenceRow:
    domain: str
    stable_id: int
    status: str
    candidate_kind: str
    proof_kind: str
    source_schema: str
    source_artifact: str
    residual_class: str
    reason_code: str
    failure_stage: str
    authority: str
    utility: float
    cost: float
    false_rejections: int
    support: int
    has_frame: bool
    composition_context: str
    example_text: str


@dataclass
class OperatorBucket:
    operator_name: str
    residual_family: str
    rows: list[EvidenceRow] = field(default_factory=list)


def parse_json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def text_has(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


def join_evidence(rows: list[dict[str, Any]]) -> list[EvidenceRow]:
    candidates: dict[tuple[str, int], dict[str, Any]] = {}
    evidence: list[EvidenceRow] = []
    for row in rows:
        if row.get("schema") != CANDIDATE_SCHEMA:
            continue
        key = (str(row.get("domain", "")), as_int(row.get("stable_id")))
        candidates[key] = row
    for row in rows:
        if row.get("schema") != OUTCOME_SCHEMA:
            continue
        domain = str(row.get("domain", ""))
        stable_id = as_int(row.get("stable_id"))
        candidate = candidates.get((domain, stable_id), {})
        has_frame = any(
            candidate.get(field_name)
            for field_name in (
                "frame_sequence_id",
                "frame_previous",
                "frame_next",
                "composition_context",
            )
        )
        evidence.append(
            EvidenceRow(
                domain=domain,
                stable_id=stable_id,
                status=str(row.get("status", "")),
                candidate_kind=str(candidate.get("candidate_kind", "")),
                proof_kind=str(row.get("proof_kind", "")),
                source_schema=str(row.get("source_schema", "")),
                source_artifact=str(row.get("source_artifact", "")),
                residual_class=str(row.get("residual_class", "")),
                reason_code=str(row.get("reason_code", "")),
                failure_stage=str(row.get("failure_stage", "")),
                authority=str(row.get("authority", "")),
                utility=as_float(row.get("utility")),
                cost=as_float(row.get("cost")),
                false_rejections=as_int(row.get("false_rejections")),
                support=as_int(row.get("support")),
                has_frame=has_frame,
                composition_context=str(candidate.get("composition_context", "")),
                example_text=str(candidate.get("example_text", "")),
            )
        )
    return evidence


def operator_for_evidence(row: EvidenceRow) -> tuple[str, str]:
    text = " ".join(
        (
            row.status,
            row.candidate_kind,
            row.proof_kind,
            row.source_schema,
            row.residual_class,
            row.reason_code,
            row.failure_stage,
            row.authority,
        )
    )
    if (
        row.status == "unsafe"
        or row.false_rejections > 0
        or text_has(text, ("death", "damage", "unsafe", "false_rejection"))
    ):
        return "residual_safety_guard", "safety"
    if text_has(text, ("shape", "frame", "normalizer", "canvas", "bbox", "panel")):
        return "counterexample_shape_frame_repair", "shape_frame"
    if text_has(text, ("selector", "phase", "anchor", "object", "mask")):
        return "counterexample_selector_phase_repair", "selector_phase"
    if text_has(text, ("transform", "color", "recolor", "palette", "render")):
        return "counterexample_transform_color_repair", "transform_color"
    if row.status == "no_support" or text_has(text, ("no_support", "diagnostic")):
        return "support_gap_bootstrap", "support_gap"
    if row.status == "rejected" or text_has(text, ("rejected", "uncertified")):
        return "rejection_filter_prior", "rejection_filter"
    if row.has_frame or row.composition_context:
        return "temporal_composition_context", "temporal_composition"
    if row.source_schema == "crlp.hypothesis.evidence.v1":
        return "certified_template_prior", "certified_template"
    if row.status == "accepted":
        return "certificate_acceptance_prior", "certificate_acceptance"
    return "residual_review_prior", "residual_review"


def group_evidence(evidence_rows: list[EvidenceRow]) -> dict[str, OperatorBucket]:
    buckets: dict[str, OperatorBucket] = {}
    for row in evidence_rows:
        operator_name, residual_family = operator_for_evidence(row)
        bucket = buckets.get(operator_name)
        if bucket is None:
            bucket = OperatorBucket(operator_name, residual_family)
            buckets[operator_name] = bucket
        bucket.rows.append(row)
    return buckets


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def promotion_reason(
    *,
    support_rows: int,
    support_domains: int,
    verifier_backed_rows: int,
    min_support_rows: int,
    min_domains: int,
) -> str:
    if support_rows < min_support_rows:
        return "below support-row threshold"
    if support_domains < min_domains:
        return "below cross-domain threshold"
    if verifier_backed_rows == 0:
        return "no verifier-backed evidence"
    return "meets cross-domain verifier-backed residual threshold"


def compiler_contract(operator_name: str) -> str:
    return {
        "residual_safety_guard": "compile into a negative-cost or unsafe-candidate guard before ranking",
        "counterexample_shape_frame_repair": "compile into a shape/frame repair sketch before verifier replay",
        "counterexample_selector_phase_repair": "compile into selector and phase-binding repair slots",
        "counterexample_transform_color_repair": "compile into transform/color-binding repair slots",
        "support_gap_bootstrap": "mine missing-support examples into new candidate families before promotion",
        "rejection_filter_prior": "use as a rejection prior only after local verifier replay",
        "temporal_composition_context": "inject previous/current/next frame features into candidate scoring",
        "certified_template_prior": "seed template and slot priors for compatible typed candidates",
        "certificate_acceptance_prior": "rank candidates toward historically accepted certificate shapes",
    }.get(operator_name, "route rows to manual residual triage before promotion")


def input_slots(operator_name: str) -> list[str]:
    slots = ["domain", "candidate_kind", "residual_class", "proof_kind", "authority"]
    if operator_name == "temporal_composition_context":
        slots.extend(["frame_sequence_id", "frame_previous", "frame_next"])
    if operator_name.startswith("counterexample_"):
        slots.extend(["failure_stage", "state_hash"])
    if operator_name == "certified_template_prior":
        slots.extend(["template_name", "slot_bindings"])
    return slots


def output_slots(operator_name: str) -> list[str]:
    if operator_name == "residual_safety_guard":
        return ["unsafe_guard", "counterexample_weight"]
    if operator_name.startswith("counterexample_"):
        return ["repair_operator", "repair_slots"]
    if operator_name == "support_gap_bootstrap":
        return ["candidate_family_seed"]
    if operator_name == "temporal_composition_context":
        return ["frame_features", "composition_prior"]
    if operator_name == "certified_template_prior":
        return ["template_prior", "slot_prior"]
    return ["ranking_prior"]


def operator_record(
    bucket: OperatorBucket,
    *,
    min_support_rows: int,
    min_domains: int,
    top_examples: int,
) -> dict[str, Any]:
    rows = bucket.rows
    domains = sorted({row.domain for row in rows if row.domain})
    source_schemas = Counter(row.source_schema for row in rows if row.source_schema)
    candidate_kinds = Counter(row.candidate_kind for row in rows if row.candidate_kind)
    residual_classes = Counter(row.residual_class for row in rows if row.residual_class)
    proof_kinds = Counter(row.proof_kind for row in rows if row.proof_kind)
    authorities = Counter(row.authority for row in rows if row.authority)
    statuses = Counter(row.status for row in rows if row.status)
    verifier_backed_rows = sum(1 for row in rows if row.proof_kind or row.authority)
    accepted_rows = statuses.get("accepted", 0)
    unsafe_rows = statuses.get("unsafe", 0)
    support_rows = len(rows)
    support_domains = len(domains)
    false_rejections = sum(row.false_rejections for row in rows)
    reason = promotion_reason(
        support_rows=support_rows,
        support_domains=support_domains,
        verifier_backed_rows=verifier_backed_rows,
        min_support_rows=min_support_rows,
        min_domains=min_domains,
    )
    promoted = reason.startswith("meets ")
    evidence_basis = {
        "operator_name": bucket.operator_name,
        "rows": [
            {
                "domain": row.domain,
                "stable_id": row.stable_id,
                "status": row.status,
                "residual_class": row.residual_class,
                "source_schema": row.source_schema,
            }
            for row in rows
        ],
    }
    evidence_hash = stable_hash(evidence_basis)
    operator_id = f"operator:{bucket.residual_family}:{evidence_hash[:12]}"
    examples = [
        {
            "domain": row.domain,
            "stable_id": row.stable_id,
            "status": row.status,
            "candidate_kind": row.candidate_kind,
            "residual_class": row.residual_class,
            "reason_code": row.reason_code,
            "example_text": row.example_text,
        }
        for row in rows[:top_examples]
    ]
    return {
        "schema": OPERATOR_SCHEMA,
        "report_version": 1,
        "date": date.today().isoformat(),
        "operator_id": operator_id,
        "operator_name": bucket.operator_name,
        "residual_family": bucket.residual_family,
        "promoted": promoted,
        "promotion_policy": "cross_domain_residual_family",
        "reason_code": reason,
        "support_rows": support_rows,
        "support_domains": support_domains,
        "domains": domains,
        "verifier_backed_rows": verifier_backed_rows,
        "accepted_rows": accepted_rows,
        "unsafe_rows": unsafe_rows,
        "rejected_rows": statuses.get("rejected", 0),
        "no_support_rows": statuses.get("no_support", 0),
        "false_rejections": false_rejections,
        "support": sum(row.support for row in rows),
        "utility_mean": mean([row.utility for row in rows]),
        "cost_mean": mean([row.cost for row in rows]),
        "frame_rows": sum(1 for row in rows if row.has_frame),
        "top_source_schemas": dict(source_schemas.most_common(8)),
        "top_candidate_kinds": dict(candidate_kinds.most_common(8)),
        "top_residual_classes": dict(residual_classes.most_common(12)),
        "source_residual_class": residual_classes.most_common(1)[0][0]
        if residual_classes
        else "",
        "top_proof_kinds": dict(proof_kinds.most_common(8)),
        "top_authorities": dict(authorities.most_common(8)),
        "status_counts": dict(sorted(statuses.items())),
        "input_slots": input_slots(bucket.operator_name),
        "output_slots": output_slots(bucket.operator_name),
        "compiler_contract": compiler_contract(bucket.operator_name),
        "evidence_hash": evidence_hash,
        "examples": examples,
    }


def promotion_records(
    evidence_rows: list[EvidenceRow],
    *,
    min_support_rows: int,
    min_domains: int,
    top_examples: int,
) -> list[dict[str, Any]]:
    records = [
        operator_record(
            bucket,
            min_support_rows=min_support_rows,
            min_domains=min_domains,
            top_examples=top_examples,
        )
        for bucket in group_evidence(evidence_rows).values()
    ]
    return sorted(
        records,
        key=lambda row: (
            not bool(row["promoted"]),
            -as_int(row["support_domains"]),
            -as_int(row["support_rows"]),
            str(row["operator_name"]),
        ),
    )


def prior_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if not record.get("promoted"):
            continue
        base = {
            "schema_version": PRIOR_SCHEMA,
            "operator_id": record["operator_id"],
            "operator_name": record["operator_name"],
            "residual_family": record["residual_family"],
            "support_domains": record["support_domains"],
            "support_rows": record["support_rows"],
            "verified_rows": record["verifier_backed_rows"],
            "utility_mean": f"{float(record['utility_mean']):.8f}",
            "cost_mean": f"{float(record['cost_mean']):.8f}",
        }
        slot_items: list[tuple[str, str, str, int]] = [
            ("operator", "operator", str(record["operator_name"]), as_int(record["support_rows"])),
        ]
        for domain in record.get("domains", []):
            slot_items.append(("domain", "domain", str(domain), 1))
        for residual, count in (record.get("top_residual_classes") or {}).items():
            slot_items.append(("residual_class", "residual", str(residual), as_int(count)))
        for candidate_kind, count in (record.get("top_candidate_kinds") or {}).items():
            slot_items.append(("candidate_kind", "candidate_kind", str(candidate_kind), as_int(count)))
        for slot_name, slot_type, slot_value, count in slot_items:
            row = dict(base)
            row.update(
                {
                    "slot_name": slot_name,
                    "slot_type": slot_type,
                    "slot_value": slot_value,
                    "proposal_count": count,
                    "verified_count": min(count, as_int(record["verifier_backed_rows"])),
                }
            )
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_prior_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "schema_version",
        "operator_id",
        "operator_name",
        "residual_family",
        "slot_name",
        "slot_type",
        "slot_value",
        "proposal_count",
        "verified_count",
        "support_domains",
        "support_rows",
        "utility_mean",
        "cost_mean",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        fp.write("\t".join(fields) + "\n")
        for row in rows:
            fp.write("\t".join(str(row.get(field, "")) for field in fields) + "\n")


def summary_row(records: list[dict[str, Any]], evidence_rows: list[EvidenceRow]) -> dict[str, Any]:
    promoted = [row for row in records if row.get("promoted")]
    domains = sorted({row.domain for row in evidence_rows if row.domain})
    by_operator = {
        str(row["operator_name"]): {
            "promoted": bool(row["promoted"]),
            "support_rows": as_int(row["support_rows"]),
            "support_domains": as_int(row["support_domains"]),
            "residual_family": row["residual_family"],
        }
        for row in records
    }
    return {
        "schema": SUMMARY_SCHEMA,
        "date": date.today().isoformat(),
        "evidence_rows": len(evidence_rows),
        "operator_candidates": len(records),
        "promoted_operators": len(promoted),
        "domains": domains,
        "by_operator": by_operator,
    }


def write_markdown(path: Path, records: list[dict[str, Any]], prior_count: int) -> None:
    lines = [
        "# Protocol Operator Promotion",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "This learner reads the shared protocol candidate/outcome spine, joins each "
        "candidate to its verifier result, normalizes repeated residual classes into "
        "operator families, and promotes only verifier-backed families that meet the "
        "cross-domain support threshold.",
        "",
        f"Operator prior rows exported: `{prior_count}`",
        "",
        "| Operator | Promoted | Domains | Rows | Residual family | Compiler contract |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in records:
        lines.append(
            "| `{operator}` | {promoted} | {domains} | {rows} | `{family}` | {contract} |".format(
                operator=row["operator_name"],
                promoted="yes" if row.get("promoted") else "no",
                domains=row["support_domains"],
                rows=row["support_rows"],
                family=row["residual_family"],
                contract=row["compiler_contract"],
            )
        )
    lines.extend(
        [
            "",
            "## Trace Contract",
            "",
            "- Candidate set: `crlp.protocol.candidate_row.v1` rows joined to verifier outcomes.",
            "- Verifier source: `proof_kind`, `authority`, and `source_schema` from the protocol spine.",
            "- Promotion rule: repeated residual family, verifier-backed rows, and cross-domain support.",
            "- Output: `crlp.protocol.operator_promotion.v1` plus `crlp.protocol.operator_prior.v1` TSV rows.",
            "- Scorer role: promoted operators become reusable priors; domain adapters must still compile and replay them before using them as gates.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--min-support-rows", type=int, default=4)
    parser.add_argument("--min-domains", type=int, default=2)
    parser.add_argument("--top-examples", type=int, default=5)
    args = parser.parse_args()

    evidence_rows = join_evidence(parse_json_rows(args.input))
    records = promotion_records(
        evidence_rows,
        min_support_rows=args.min_support_rows,
        min_domains=args.min_domains,
        top_examples=args.top_examples,
    )
    priors = prior_rows(records)
    write_jsonl(args.output, records)
    write_prior_tsv(args.prior, priors)
    if args.summary:
        write_jsonl(args.summary, [summary_row(records, evidence_rows)])
    if args.markdown:
        write_markdown(args.markdown, records, len(priors))
    print(f"wrote {args.output}")
    print(f"wrote {args.prior}")
    if args.summary:
        print(f"wrote {args.summary}")
    if args.markdown:
        print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
