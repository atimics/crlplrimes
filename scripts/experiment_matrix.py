#!/usr/bin/env python3
"""Generate a structurally repetitive experiment matrix for the paper."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any


DOMAIN_ROWS = [
    {
        "domain": "prime_gap",
        "candidate_set": "action candidates plus certified modular rule filters",
        "verifier": "prime-gap replay and rule certificate checks",
        "certificate_gate": "false-rejection accounting and train/validation support",
        "scorer_role": "rank residual actions/rules inside a fixed verifier budget",
        "trace_artifact": "crlp.compare.v1; crlp.rule_trace.v1; crlp.rule_search.summary.v1",
        "supported_claim": "C1.prime_gate_delta; C5.rejected_candidate_reason_learning",
        "limitation": "static neural proposer is not superior to deterministic baselines at fixed budget",
    },
    {
        "domain": "grid_path",
        "candidate_set": "grid actions and symbolic path predicates",
        "verifier": "grid transition and path certificate checks",
        "certificate_gate": "typed symbolic rule application and false-rejection checks",
        "scorer_role": "residual action scorer after symbolic constraints",
        "trace_artifact": "crlp.compare.v1; crlp.symbolic_domain.trace.v1",
        "supported_claim": "C1/C5 substrate portability context",
        "limitation": "not the current headline domain; mainly a transfer sanity surface",
    },
    {
        "domain": "delayed_claim",
        "candidate_set": "claim actions, threshold gates, and utility-margin rules",
        "verifier": "delayed-claim rule certificate and false-rejection counters",
        "certificate_gate": "safety-first residual gate for unsafe threshold rows",
        "scorer_role": "rank utility after predicted safety dominates",
        "trace_artifact": "crlp.delayed_claim.rule_trace.v1; crlp.rejected_candidate_learning.*",
        "supported_claim": "delayed-claim is a stress case for generic reason priors",
        "limitation": "reason-prior transfer alone can fail safety; claim remains domain-scoped",
    },
    {
        "domain": "arc_agi",
        "candidate_set": "typed transform, program, and hypothesis candidates",
        "verifier": "train-pair exactness and held-out test-cell diagnostics",
        "certificate_gate": "program/hypothesis certificate rows and loader summaries",
        "scorer_role": "rank bounded candidates; scorer does not own correctness",
        "trace_artifact": "crlp.arc_loader.summary.v1; crlp.program_proposer.summary.v1; crlp.hypothesis.evidence.v1",
        "supported_claim": "C1.arc_public_evidence_surface; C5.hypothesis_library_promotion",
        "limitation": "ARC remains an evidence surface; broad score chasing is out of scope",
    },
    {
        "domain": "multitask_tiny_net",
        "candidate_set": "shared residual ranking rows across prime/grid/claim/ARC",
        "verifier": "task-local labels and certified downstream gates",
        "certificate_gate": "per-task held-out metrics, OOD ARC variants, calibration bins",
        "scorer_role": "shared residual scorer with domain-specific diagnostics",
        "trace_artifact": "crlp.multitask.summary.v1; crlp.multitask.tiny_net_plateau.v1",
        "supported_claim": "C4.arc_ood_shared_vs_single; C7.tiny_net_plateau_stop_rule when materialized",
        "limitation": "larger width is not the default explanation until plateau evidence exists",
    },
    {
        "domain": "signal_flight",
        "candidate_set": "nine discrete WASD action candidates",
        "verifier": "offline counterfactual replay; live shadow frame comparison",
        "certificate_gate": "damage/death/regret counters and live safety-mask logs",
        "scorer_role": "rank bounded control candidates beside heuristic control",
        "trace_artifact": "crlp.signal_flight.summary.v1; crlp.signal_live_brain.v1; crlp.signal_shadow_replay_audit.v1",
        "supported_claim": "C9.signal_flight_replay_control",
        "limitation": "live neural control is not claimed until replay follow-up passes",
    },
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def claim_statuses(claims_path: Path) -> dict[str, str]:
    claims = read_json(claims_path).get("claims", [])
    return {str(row.get("id")): str(row.get("status")) for row in claims}


def add_status(row: dict[str, Any], statuses: dict[str, str]) -> dict[str, Any]:
    claim_ids = [
        part.strip()
        for part in str(row["supported_claim"]).replace(";", ",").split(",")
        if part.strip().startswith("C")
    ]
    present = {claim_id: statuses.get(claim_id, "not_registered") for claim_id in claim_ids}
    return {
        "schema": "crlp.experiment_matrix.row.v1",
        "date": date.today().isoformat(),
        **row,
        "claim_statuses": present,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True) + "\n")


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Experiment Matrix",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "| Domain | Candidate set | Verifier | Gate | Scorer role | Trace artifact | Result / claim | Limitation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| `{domain}` | {candidate_set} | {verifier} | {certificate_gate} | {scorer_role} | `{trace_artifact}` | {supported_claim} | {limitation} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims-json", type=Path, default=Path("paper/generated/claims.json"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("paper/generated/experiment_matrix.jsonl"))
    parser.add_argument("--output-md", type=Path, default=Path("paper/generated/experiment_matrix.md"))
    args = parser.parse_args()

    statuses = claim_statuses(args.claims_json)
    rows = [add_status(row, statuses) for row in DOMAIN_ROWS]
    write_jsonl(args.output_jsonl, rows)
    write_markdown(args.output_md, rows)
    print(f"wrote {args.output_jsonl}")
    print(f"wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
