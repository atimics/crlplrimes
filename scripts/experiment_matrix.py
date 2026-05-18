#!/usr/bin/env python3
"""Generate the frozen, structurally repetitive experiment matrix."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any


DOMAIN_ROWS = [
    {
        "domain": "prime_gap",
        "candidate_set": "next-gap actions plus certified modular rule filters",
        "verifier": "segmented-sieve replay and modular certificate checks",
        "certificate_gate": "zero-false-rejection divisibility certificates over train/validation support",
        "scorer_role": "rank residual gaps and budgeted certified rules",
        "trace_artifact": "crlp.compare.v1; crlp.rule_trace.v1; crlp.rule_search.summary.v1",
        "result": "learned gate 68.87% top-1 vs 45.12% hand-wheel and 9.13% raw",
        "claim": "certified modular gates shrink the residual action space",
        "claim_ids": ["C1.prime_gate_delta", "C2.prime_proposer_fixed_budget"],
        "limitation": "static neural proposer is not superior to deterministic baselines at fixed budget",
    },
    {
        "domain": "grid_path",
        "candidate_set": "four moves plus symbolic local/topology/bound predicates",
        "verifier": "BFS shortest-path successor and rule certificate checks",
        "certificate_gate": "legality, trap, clear-path, and distance-dominance certificates",
        "scorer_role": "rank legal residual moves after symbolic constraints",
        "trace_artifact": "crlp.compare.v1; crlp.symbolic_domain.trace.v1",
        "result": "cheap-bound gate lifts priors-enabled raw 75.80% to 90.10%",
        "claim": "same gate contract works outside residue arithmetic",
        "claim_ids": [],
        "limitation": "supporting transfer surface, not a main contribution",
    },
    {
        "domain": "delayed_claim",
        "candidate_set": "claim actions, threshold gates, and utility-margin rules",
        "verifier": "deterministic hidden-outcome simulator and rule certificate checks",
        "certificate_gate": "finite-generator safety check plus false-rejection counters",
        "scorer_role": "rank utility only after safety dominates the objective",
        "trace_artifact": "crlp.delayed_claim.rule_trace.v1; crlp.rejected_candidate_learning.*",
        "result": "clean threshold+margin gate reduces four actions to one; noisy rows expose failures",
        "claim": "reason learning needs domain-aware safety constraints",
        "claim_ids": ["C4.delayed_claim_sibling_transfer"],
        "limitation": "reason-prior transfer alone can fail safety; claim remains domain-scoped",
    },
    {
        "domain": "rejected_candidate_learning",
        "candidate_set": "rejected verifier rows grouped by task, split, reason family, and seed",
        "verifier": "original rule/certificate checker that rejected or accepted each row",
        "certificate_gate": "false-rejection-first budget filter over reason priors and residual scores",
        "scorer_role": "learn residual ordering from rejection reasons and marginal verifier utility",
        "trace_artifact": "crlp.rejected_candidate_learning.*",
        "result": "trained residual reaches 7 accepted rules / 245 utility with zero false rejections",
        "claim": "verifier failures become training signal",
        "claim_ids": ["C5.rejected_candidate_reason_learning"],
        "limitation": "not a generic reason-code ranker; delayed-claim safety remains scoped",
    },
    {
        "domain": "hypothesis_library_feedback",
        "candidate_set": "certified hypothesis templates, slot bindings, and compiled programs",
        "verifier": "train-pair exactness, type checks, and held-out fixture split evaluation",
        "certificate_gate": "zero-false-rejection promotion before a binding enters the library",
        "scorer_role": "rank promoted slot priors and residual program candidates under tight budgets",
        "trace_artifact": "crlp.hypothesis.evidence.v1; crlp.hypothesis_library.heldout_result.v1",
        "result": "slot priors raise certified@1 from 2 to 4; held-out split rows = 30",
        "claim": "certified learning can accumulate through reusable library priors",
        "claim_ids": ["C5.hypothesis_library_promotion"],
        "limitation": "public ARC promotion remains scoped evidence, not a headline ARC score",
    },
    {
        "domain": "multitask_tiny_net",
        "candidate_set": "shared residual ranking rows across prime/grid/claim/ARC",
        "verifier": "task-local labels and certified downstream gates",
        "certificate_gate": "per-task held-out metrics, OOD ARC variants, calibration bins",
        "scorer_role": "shared residual scorer with domain-specific diagnostics",
        "trace_artifact": "crlp.multitask.summary.v1; crlp.multitask.tiny_net_plateau.v1",
        "result": "ARC OOD shared +5.57pp; capacity is not monotonic; plateau audit is underpowered",
        "claim": "representation/protocol limits dominate naive width scaling",
        "claim_ids": ["C4.arc_ood_shared_vs_single", "C4.multitask_capacity_tradeoff", "C7.tiny_net_plateau_stop_rule"],
        "limitation": "architecture-size claims stay gated by the plateau audit",
    },
    {
        "domain": "arc_agi_evidence_surface",
        "candidate_set": "cell colors, typed transforms, programs, and hypothesis-derived candidates",
        "verifier": "train-pair consistency, unique compiled prediction, and public corpus loader",
        "certificate_gate": "program/hypothesis certificate rows and scorer-invariant loader summaries",
        "scorer_role": "rank bounded candidates; scorer does not own correctness",
        "trace_artifact": "crlp.arc_loader.summary.v1; crlp.program_proposer.summary.v1; crlp.hypothesis.evidence.v1",
        "result": "public eval coverage is 7/400 tasks and 1057 cells under the measured surface",
        "claim": "ARC exposes priors, grammar coverage, and failure taxonomy",
        "claim_ids": ["C1.arc_gate_scorer_invariance", "C1.arc_public_evidence_surface", "C3.arc_ood_calibration_tradeoff"],
        "limitation": "not a broad ARC competence or benchmark-score claim",
    },
    {
        "domain": "signal_flight",
        "candidate_set": "certified control-target hypotheses plus nine WASD action candidates",
        "verifier": "offline counterfactual simulator replay and live shadow frame comparison",
        "certificate_gate": "replay utility, regret, damage/death counters, and shadow disagreement logs",
        "scorer_role": "rank bounded control candidates beside heuristic control",
        "trace_artifact": "crlp.signal_flight.summary.v1; crlp.signal_live_brain.v1; crlp.signal_shadow_replay_audit.v1",
        "result": "offline top-1 79.76% vs 67.86% baseline; live shadow frames = 22191",
        "claim": "application pressure for replay-grounded residual ranking",
        "claim_ids": ["C9.signal_flight_replay_control"],
        "limitation": "shadow inference is not active neural control",
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
    claim_source = row.get("claim_ids", row.get("claim", ""))
    if isinstance(claim_source, list):
        claim_parts = [str(part) for part in claim_source]
    else:
        claim_parts = str(claim_source).replace(";", ",").split(",")
    claim_ids = [
        part.strip()
        for part in claim_parts
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
        "# Frozen Experiment Matrix",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "The paper evidence families are frozen to these rows. New domains belong in roadmap notes until this matrix is deliberately revised.",
        "",
        "| Domain | Candidate set | Verifier | Certificate/gate | Scorer role | Trace artifact | Result | Claim | Limitation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| `{domain}` | {candidate_set} | {verifier} | {certificate_gate} | {scorer_role} | `{trace_artifact}` | {result} | {claim} | {limitation} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def typst_string(value: Any) -> str:
    return json.dumps(str(value))


def write_typst(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "// Generated by scripts/experiment_matrix.py; do not edit by hand.",
        "#let experiment_matrix_rows = (",
    ]
    keys = (
        "domain",
        "candidate_set",
        "verifier",
        "certificate_gate",
        "scorer_role",
        "trace_artifact",
        "result",
        "claim",
        "limitation",
    )
    for row in rows:
        lines.append("  (")
        for key in keys:
            lines.append(f"    {key}: {typst_string(row[key])},")
        lines.append("  ),")
    lines.append(")")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims-json", type=Path, default=Path("paper/generated/claims.json"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("paper/generated/experiment_matrix.jsonl"))
    parser.add_argument("--output-md", type=Path, default=Path("paper/generated/experiment_matrix.md"))
    parser.add_argument("--output-typ", type=Path, default=Path("paper/generated/experiment_matrix.typ"))
    args = parser.parse_args()

    statuses = claim_statuses(args.claims_json)
    rows = [add_status(row, statuses) for row in DOMAIN_ROWS]
    write_jsonl(args.output_jsonl, rows)
    write_markdown(args.output_md, rows)
    write_typst(args.output_typ, rows)
    print(f"wrote {args.output_jsonl}")
    print(f"wrote {args.output_md}")
    print(f"wrote {args.output_typ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
