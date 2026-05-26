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
        "domain_display": "arithmetic sequence filtering",
        "candidate_set": "next-prime gap actions plus certified modular filters",
        "verifier": "segmented-sieve replay and modular certificate checks",
        "certificate_gate": "zero-false-rejection divisibility certificates for the audited next-prime instance",
        "scorer_role": "rank residual gaps and budgeted certified filters",
        "trace_artifact": "crlp.compare.v1; crlp.rule_trace.v1; crlp.rule_search.summary.v1",
        "result": "learned filter 68.87% top-1 vs 45.12% hand-wheel and 9.13% raw; seed interval audited",
        "claim": "certified modular filters shrink the residual action space",
        "claim_ids": ["C1.prime_gate_delta", "C1.prime_gate_delta_ci", "C2.prime_proposer_fixed_budget"],
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
        "domain": "arc_verified_self_play",
        "candidate_set": "forward-composed visual primitive programs plus verifier-shaped negative siblings",
        "verifier": "known-correct synthetic output construction and ARC visual-pipeline verifier fields",
        "certificate_gate": "accepted rows are generated by primitive composition; public coverage still requires ARC verifier replay",
        "scorer_role": "train inverse visual-program induction from exact self-play rows",
        "trace_artifact": "crlp.arc_verified_self_play.summary.v1; crlp.arc_synthetic_shape_curriculum.v1; crlp.neural_grammar_discovery.arc_visual_pipeline_neural_summary.v1",
        "result": "self-play reports generated supervision against template=7, oracle=14, and nine zero-coverage ARC families",
        "claim": "ARC proposer training can be generated from verifier-owned primitives instead of authored task libraries",
        "claim_ids": [],
        "limitation": "distribution mismatch remains open until the generated prior increases public ARC verifier coverage",
    },
    {
        "domain": "signal_flight",
        "candidate_set": "certified control-target hypotheses plus nine WASD action candidates",
        "verifier": "offline counterfactual simulator replay, closed-loop rollout replay, live shadow frame comparison, and live active telemetry",
        "certificate_gate": "replay utility, regret, damage/death counters, certified control-target rollout rows, shadow disagreement logs, live safety gates, and protocol compatibility rows",
        "scorer_role": "rank bounded control candidates beside heuristic control",
        "trace_artifact": "crlp.signal_flight.summary.v1; crlp.signal_flight_rollout.summary.v1; crlp.signal_shadow_replay_audit.v1; crlp.signal_live_active_summary.v1",
        "result": "offline replay, rollout bridge, and guarded live neural-active evidence are reported; active baseline comparison remains open",
        "claim": "application pressure for replay-grounded residual ranking and guarded live neural-active execution",
        "claim_ids": ["C9.signal_flight_replay_control", "C9.signal_live_neural_active"],
        "limitation": "active neural versus heuristic performance is not yet claimed",
    },
]


PROTOCOL_DETAILS = {
    "prime_gap": {
        "data_domain": "next-prime gaps over segmented-sieve windows; broader family includes period-residue sequence filters",
        "grammar": "local modular filters over residues, gap classes, and wheel support",
        "examples": ["modular reject rule", "wheel support certificate", "next-prime gap action", "Pisano-period recurrence gate"],
        "takeaway": "certified modular structure is the candidate mechanism; the neural scorer ranks residual budget choices while Fibonacci/Lucas-style filters remain boundary evidence.",
    },
    "grid_path": {
        "data_domain": "grid mazes with deterministic shortest-path labels",
        "grammar": "local moves plus boundary, trap, path-clear, and distance predicates",
        "examples": ["legal move", "trap reject", "distance-dominance gate"],
        "takeaway": "the same candidate-verifier contract transfers to topology once the grammar stays domain-local.",
    },
    "delayed_claim": {
        "data_domain": "delayed reward/claim rows with hidden outcomes and margins",
        "grammar": "threshold, margin, and sibling-claim predicates",
        "examples": ["safe threshold claim", "margin reject", "unsafe reason-prior transfer"],
        "takeaway": "utility ranking must be downstream of safety when false rejection risk is domain-specific.",
    },
    "rejected_candidate_learning": {
        "data_domain": "verifier rejection rows grouped by seed, split, and reason family",
        "grammar": "the original domain grammars that produced rejected candidate rows",
        "examples": ["unsafe candidate", "no-support row", "trained residual recovery"],
        "takeaway": "rejected verifier rows are reusable supervision, not discarded failed searches.",
    },
    "hypothesis_library_feedback": {
        "data_domain": "ARC-like hypothesis fixtures split into promotion and held-out task ids",
        "grammar": "typed templates, slot bindings, and compiled program candidates",
        "examples": ["slot prior", "promoted template", "held-out certified@1 row"],
        "takeaway": "certified hypotheses can become library priors and improve tight-budget held-out certification.",
    },
    "multitask_tiny_net": {
        "data_domain": "shared residual rows from prime, grid, claim, and ARC surfaces",
        "grammar": "domain-local grammars encoded through common residual feature rows",
        "examples": ["shared trunk row", "ARC OOD row", "capacity plateau row"],
        "takeaway": "small rankers are useful around certified structure, but one flat head exposes interference.",
    },
    "arc_agi_evidence_surface": {
        "data_domain": "public and fixture ARC tasks with cell, object, transform, and hypothesis candidates",
        "grammar": "typed transforms, object/slot bindings, and bounded program templates",
        "examples": ["typed transform", "hypothesis candidate", "loader summary"],
        "takeaway": "ARC is an evidence surface for grammar coverage and failure taxonomy, not a score chase.",
    },
    "arc_verified_self_play": {
        "data_domain": "generated ARC-like tasks from forward visual-primitive composition plus existing ARC visual-pipeline attempts",
        "grammar": "visual primitives for object motion, recolor, tile/scale, crop/extract, projection/panel, counting, fill, and symmetry",
        "examples": ["forward-composed accepted program", "verifier-reason negative sibling", "inverse proposer row"],
        "takeaway": "the durable ARC asset is the verifier/primitive system; authored solution libraries are baselines, not the training source.",
    },
    "signal_flight": {
        "data_domain": "Signal flight replay frames, closed-loop rollout rows, live shadow-control comparisons, and live neural-active rows",
        "grammar": "bounded action/control-option candidates beside the heuristic controller",
        "examples": ["WASD candidate", "counterfactual utility row", "rollout controller row", "shadow disagreement row"],
        "takeaway": "Signal adds application pressure through replay, rollout, shadow, and guarded live artifacts while keeping active-control performance open.",
    },
}

AUTHORITY_DETAILS = {
    "prime_gap": {
        "grounding_authority": "exact_check",
        "authority_claim_scope": "zero-false-rejection gates and exact residual ranking inside the next-prime support",
    },
    "grid_path": {
        "grounding_authority": "exact_check",
        "authority_claim_scope": "shortest-path successor and certificate claims inside generated grids",
    },
    "delayed_claim": {
        "grounding_authority": "deterministic_replay",
        "authority_claim_scope": "simulator-certified utility and safety under the delayed-claim generator",
    },
    "rejected_candidate_learning": {
        "grounding_authority": "exact_check",
        "authority_claim_scope": "reason-code feedback on arithmetic/modular verifier rows; replay stress rows are limitations only",
    },
    "hypothesis_library_feedback": {
        "grounding_authority": "exact_check",
        "authority_claim_scope": "train-pair exactness and held-out fixture certification under the typed hypothesis grammar",
    },
    "multitask_tiny_net": {
        "grounding_authority": "mixed_diagnostic",
        "authority_claim_scope": "diagnostic scorer rows only; not aggregated into an authority-level leaderboard",
    },
    "arc_agi_evidence_surface": {
        "grounding_authority": "corpus_proxy",
        "authority_claim_scope": "public-corpus coverage and failure taxonomy, not broad exact ARC competence",
    },
    "arc_verified_self_play": {
        "grounding_authority": "exact_check",
        "authority_claim_scope": "exact synthetic supervision for inverse proposer training; public ARC coverage remains a separate verifier-replay measurement",
    },
    "signal_flight": {
        "grounding_authority": "deterministic_replay",
        "authority_claim_scope": "simulator replay, rollout replay, protocol-compatible live telemetry, and guarded safety evidence",
    },
}


ROLE_DETAILS = {
    "prime_gap": {
        "neural_scorer_role": "score residual actions and certified filters after the gate defines the target",
        "neural_proposer_role": "static proposer is a negative baseline; closed-loop and unified proposer rows must be compared to grammar order",
        "headline_baseline": "scorer baseline is the certified gate; proposer baseline is grammar order plus verifier-feedback variants",
    },
    "grid_path": {
        "neural_scorer_role": "rank legal residual moves under exact shortest-path labels",
        "neural_proposer_role": "budgeted rule-order diagnostics only",
        "headline_baseline": "BFS successor and cheap symbolic gates",
    },
    "delayed_claim": {
        "neural_scorer_role": "rank utility after simulator safety dominates the decision",
        "neural_proposer_role": "consume simulator/gate rejection reasons as scoped feedback",
        "headline_baseline": "deterministic clean threshold and margin gates",
    },
    "rejected_candidate_learning": {
        "neural_scorer_role": "train residual ordering from rejected verifier outcomes",
        "neural_proposer_role": "feed reason codes and residual classes into the next proposal pass",
        "headline_baseline": "unguided model_score versus reason_prior and trained_residual",
    },
    "hypothesis_library_feedback": {
        "neural_scorer_role": "rank bounded program candidates after template certification",
        "neural_proposer_role": "turn zero-false-rejection templates into slot priors for the next search",
        "headline_baseline": "no_prior and grammar-order tight-budget selection",
    },
    "multitask_tiny_net": {
        "neural_scorer_role": "shared residual scorer with per-domain diagnostics",
        "neural_proposer_role": "none promoted; architecture rows are blocked by the plateau audit",
        "headline_baseline": "single-task scorer and C7 plateau audit",
    },
    "arc_agi_evidence_surface": {
        "neural_scorer_role": "rank bounded candidates and expose calibration/coverage gaps",
        "neural_proposer_role": "grow certified transform/program grammar coverage rather than claim scorer capacity fixes",
        "headline_baseline": "corpus loader coverage and train-pair certified transform rows",
    },
    "arc_verified_self_play": {
        "neural_scorer_role": "none as correctness owner; ranker learns inverse program ordering from generated verifier rows",
        "neural_proposer_role": "self-play proposer trained on accepted programs and verifier-reason negative siblings",
        "headline_baseline": "authored template baseline 7 certified eval tasks, oracle ceiling 14, and nine zero-coverage miss families",
    },
    "signal_flight": {
        "neural_scorer_role": "rank bounded WASD/control candidates beside heuristic control",
        "neural_proposer_role": "control-target hypotheses remain replay-gated before action ranking",
        "headline_baseline": "heuristic controller, simulator replay, and live guardrails",
    },
}


REQUIRED_PROTOCOL_FIELDS = (
    "data_domain",
    "grounding_authority",
    "authority_claim_scope",
    "grammar",
    "candidate_set",
    "verifier",
    "certificate_gate",
    "scorer_role",
    "neural_scorer_role",
    "neural_proposer_role",
    "headline_baseline",
    "trace_artifact",
    "examples",
    "result_table",
    "takeaway",
    "limitation",
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def claim_index(claims_path: Path) -> dict[str, dict[str, Any]]:
    claims = read_json(claims_path).get("claims", [])
    return {str(row.get("id")): row for row in claims}


def add_status(row: dict[str, Any], claims: dict[str, dict[str, Any]]) -> dict[str, Any]:
    protocol = {
        "data_domain": row["domain"],
        "grounding_authority": "unclassified",
        "authority_claim_scope": "unclassified",
        "grammar": "domain-local candidate grammar",
        "examples": [],
        "result_table": row["result"],
        "takeaway": row["claim"],
        "neural_scorer_role": row.get("scorer_role", ""),
        "neural_proposer_role": "not declared",
        "headline_baseline": "not declared",
    }
    protocol.update(AUTHORITY_DETAILS.get(str(row["domain"]), {}))
    protocol.update(ROLE_DETAILS.get(str(row["domain"]), {}))
    protocol.update(PROTOCOL_DETAILS.get(str(row["domain"]), {}))
    protocol.setdefault("result_table", row["result"])
    enriched = {**protocol, **row}
    enriched.setdefault("domain_display", str(row["domain"]).replace("_", " "))
    enriched.setdefault("result_table", enriched["result"])
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
    present = {claim_id: claims.get(claim_id, {}).get("status", "not_registered") for claim_id in claim_ids}
    if row.get("domain") == "signal_flight":
        replay_summary = claims.get("C9.signal_flight_replay_control", {}).get("summary", "")
        live_summary = claims.get("C9.signal_live_neural_active", {}).get("summary", "")
        summaries = [str(item) for item in (replay_summary, live_summary) if item]
        if summaries:
            enriched["result"] = " ".join(summaries)
            live_status = claims.get("C9.signal_live_neural_active", {}).get("status", "missing")
            if live_status == "supported":
                enriched["result_table"] = (
                    "offline replay, rollout bridge, and live neural-active claims supported; "
                    "active baseline comparison remains open"
                )
            else:
                enriched["result_table"] = (
                    "offline replay and rollout bridge supported; live neural-active evidence "
                    f"{live_status} pending protocol-compatible rows; active baseline comparison remains open"
                )
    return {
        "schema": "crlp.experiment_matrix.row.v1",
        "date": date.today().isoformat(),
        **enriched,
        "claim_statuses": present,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True) + "\n")


def validate_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        missing = [field for field in REQUIRED_PROTOCOL_FIELDS if not row.get(field)]
        if missing:
            raise ValueError(f"{row.get('domain', '<unknown>')} missing protocol fields: {missing}")
        if row.get("grounding_authority") == "unclassified":
            raise ValueError(f"{row.get('domain', '<unknown>')} missing grounding authority")
        examples = row.get("examples")
        if not isinstance(examples, list) or not (2 <= len(examples) <= 4):
            raise ValueError(f"{row.get('domain', '<unknown>')} must have 2-4 protocol examples")


def text_list(value: Any, separator: str = "; ") -> str:
    if isinstance(value, list):
        return separator.join(str(item) for item in value)
    return str(value)


def md_cell(value: Any) -> str:
    return text_list(value).replace("|", "\\|")


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Frozen Experiment Matrix",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "The evidence families are frozen to these rows. New domains belong in roadmap notes until this matrix is deliberately revised.",
        "",
        "Authority is a hard reporting axis. Exact checks, deterministic replay, corpus proxies, and mixed diagnostics are not collapsed into a single score.",
        "",
        "| Domain/data | Authority | Grammar | Candidate set | Verifier/gate | Neural role split | Headline baseline | Trace artifact | Result table | Takeaway | Limitation |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| `{domain}`: {data_domain} | `{grounding_authority}`: {authority_claim_scope} | {grammar} | {candidate_set} | {verifier}; {certificate_gate} | scorer: {neural_scorer_role}; proposer: {neural_proposer_role} | {headline_baseline} | `{trace_artifact}` | {result_table} | {takeaway} | {limitation} |".format(
                **{key: md_cell(value) for key, value in row.items()}
            )
        )
    lines.extend(["", "## Authority Leaderboards", ""])
    for authority in ("exact_check", "deterministic_replay", "corpus_proxy", "human_review"):
        authority_rows = [row for row in rows if row.get("grounding_authority") == authority]
        lines.extend([f"### `{authority}`", ""])
        if authority_rows:
            for row in authority_rows:
                lines.append(
                    f"- `{row['domain']}`: {text_list(row['result_table'])}"
                )
        else:
            lines.append("- No frozen rows.")
        lines.append("")
    mixed_rows = [
        row for row in rows
        if row.get("grounding_authority") not in {
            "exact_check",
            "deterministic_replay",
            "corpus_proxy",
            "human_review",
        }
    ]
    lines.extend(["### `mixed_diagnostic`", ""])
    for row in mixed_rows:
        lines.append(f"- `{row['domain']}`: {text_list(row['authority_claim_scope'])}")
    lines.append("")
    lines.extend(["", "## Repeated Protocol Rows", ""])
    for row in rows:
        lines.extend(
            [
                f"### `{row['domain']}`",
                "",
                f"1. Domain/data: {text_list(row['data_domain'])}",
                f"2. Authority: `{row['grounding_authority']}` - {text_list(row['authority_claim_scope'])}",
                f"3. Grammar: {text_list(row['grammar'])}",
                f"4. Candidate set: {text_list(row['candidate_set'])}",
                f"5. Verifier/gate: {text_list(row['verifier'])}; {text_list(row['certificate_gate'])}",
                f"6. Neural scorer role: {text_list(row['neural_scorer_role'])}",
                f"7. Neural proposer role: {text_list(row['neural_proposer_role'])}",
                f"8. Headline baseline: {text_list(row['headline_baseline'])}",
                f"9. Examples: {text_list(row['examples'])}",
                f"10. Result table: {text_list(row['result_table'])}",
                f"11. Takeaway: {text_list(row['takeaway'])}",
                f"12. Limitation: {text_list(row['limitation'])}",
                "",
            ]
        )
    while lines and lines[-1] == "":
        lines.pop()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def typst_string(value: Any) -> str:
    return json.dumps(text_list(value))


def write_typst(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "// Generated by scripts/experiment_matrix.py; do not edit by hand.",
        "#let experiment_matrix_rows = (",
    ]
    keys = (
        "domain",
        "domain_display",
        "data_domain",
        "grounding_authority",
        "authority_claim_scope",
        "grammar",
        "candidate_set",
        "verifier",
        "certificate_gate",
        "scorer_role",
        "neural_scorer_role",
        "neural_proposer_role",
        "headline_baseline",
        "trace_artifact",
        "examples",
        "result_table",
        "result",
        "claim",
        "takeaway",
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

    claims = claim_index(args.claims_json)
    rows = [add_status(row, claims) for row in DOMAIN_ROWS]
    validate_rows(rows)
    write_jsonl(args.output_jsonl, rows)
    write_markdown(args.output_md, rows)
    write_typst(args.output_typ, rows)
    print(f"wrote {args.output_jsonl}")
    print(f"wrote {args.output_md}")
    print(f"wrote {args.output_typ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
