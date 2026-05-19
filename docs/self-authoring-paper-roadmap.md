# Self-Authoring Paper Roadmap

Date: 2026-05-17
Status: R-P.1, R-P.2, R-P.3, R-P.5, R-P.6, R-P.7, and R-P.8 implemented; R-P.4 remains open

## Purpose

The current project already emits structured experiment evidence, but the paper
is still partly hand-authored around that evidence. A self-authoring paper means:

```text
experiment binaries -> evidence rows -> claim registry -> generated paper data,
prose, limitations, and roadmap status -> PDF
```

The paper should not merely cite `results/latest`; it should be compiled from
machine-readable claims whose support, contradiction, and missing-evidence state
are derived from the code outputs.

## Current Pipeline

The existing pipeline is:

```text
make results
  -> results/latest/environment.json
  -> results/latest/comparison.jsonl
  -> results/latest/comparison.canonical.jsonl
  -> results/latest/trace_determinism.jsonl
  -> results/latest/prime_gap_trace.semantic.jsonl
  -> results/latest/summary.md

make paper-data
  -> paper/data.json
  -> paper/data.typ

make paper-support
  -> runs make trace-determinism-compare
  -> paper/generated/claims.json
  -> paper/generated/claims.typ
  -> paper/generated/gap_report.md
  -> paper/generated/limitations.md
  -> paper/generated/roadmap_links.md
  -> paper/generated/support_report.md
  -> paper/generated/agent_instructions.md
  -> fails if any required claim is stale, missing, underpowered, or contradicted

make paper-pdf
  -> runs make paper-support
  -> paper/paper.pdf
```

This is the right spine, but `paper/data.typ` is still mostly a data dump. The
paper decides which result supports which claim by hand.

## Gap Analysis

| Gap | Current state | Required state |
| --- | --- | --- |
| Claim registry | Claims live in prose, captions, and the roadmap table. | Claims have stable ids, evidence queries, decision rules, and paper surfaces. |
| Support status | The reader infers whether evidence is complete. | Each claim is marked `supported`, `contradicted`, `underpowered`, `missing`, or `stale`. |
| Hard-coded metrics | Many abstract/body/caption numbers are literal Typst text. | Numbers are rendered from generated claim variables. |
| Contradictions | A new result can silently make prose stale. | A checker fails when prose references unsupported or contradicted claims. |
| Roadmap semantics | Roadmap items list experiments by effort. | Each item names the unsupported claim it closes. |
| Limitations | Limitations are hand-maintained narrative. | Failed and underpowered claims generate limitation candidates. |
| PDF provenance | Tables cite result paths informally. | Every generated table/prose block can point back to row ids and source commands. |

## Target Artifacts

Add a claim layer between `results/latest/comparison.jsonl` and `paper/data.typ`:

```text
paper/claims.yaml
  stable claim definitions and paper-surface mapping

scripts/derive_paper_claims.py
  loads evidence rows, evaluates claim decision rules, writes generated files

paper/generated/claims.json
  complete machine-readable claim states

paper/generated/claims.typ
  Typst variables, tables, and short generated prose snippets

paper/generated/gap_report.md
  claim-by-claim code-to-PDF gap analysis

paper/generated/limitations.md
  generated limitation candidates from non-supported claims

paper/generated/roadmap_links.md
  audit of roadmap rows that are not yet linked to known claim families

paper/generated/support_report.md
  concise pass/fail report for the required paper-support gate

paper/generated/agent_instructions.md
  claim-targeted briefs for future research agents without letting them rewrite the paper by hand

.github/pull_request_template.md
  review checklist separating editorial-only edits from explicit claim changes

.github/workflows/paper.yml
  PR and main-branch gate for editorial-boundary checks, paper support checks,
  and PDF compilation

docs/editorial-agent-prs.md
  repository policy for reviewable editorial-agent work

scripts/check_editorial_pr.py
  CI enforcement for editorial-only PRs and claim-change PR audit fields
```

`paper/paper.typ` should import `paper/generated/claims.typ` in the same spirit
as `paper/data.typ`, but `claims.typ` should carry interpretation, not just
numbers.

## Claim Record Shape

Each claim should be explicit enough that a script can decide whether the paper
is allowed to say it:

```yaml
id: C1.prime_gate_delta
title: Prime learned gate improves top-1 over raw and hand-wheel gates
claim_type: gate_effect
paper_surfaces:
  - abstract
  - axis_1_table
  - cross_task_interpretation
evidence:
  source: results/latest/comparison.canonical.jsonl
  schemas:
    - crlp.compare.v1
  filters:
    task: prime_gap
  required_rows:
    - evaluation_mode: raw
    - evaluation_mode: hand_wheel_model
    - evaluation_mode: expanded_rule_model
decision_rule:
  supported_if:
    - expanded_rule_model.top1 > hand_wheel_model.top1
    - hand_wheel_model.top1 > raw.top1
    - expanded_rule_model.false_rejections == 0
staleness:
  canonical_hash: auto
outputs:
  metric_vars:
    - raw_top1
    - hand_top1
    - learned_top1
  prose_template: "On prime_gap, the certified learned gate raises top-1 from {raw_top1} to {learned_top1}."
```

## Initial Claim Families

Start with the claims already carrying the paper:

| Family | Claim ids | Evidence rows | Current risk |
| --- | --- | --- | --- |
| Gate effect | `C1.*` | `crlp.compare.v1`, `crlp.rule_search.summary.v1`, `crlp.arc_transform.summary.v1` | Some numbers are hard-coded; grid/delayed/ARC evidence maturity differs. Delayed-claim now has a production threshold+margin gate/proposer loop plus shifted/noisy stress rows; direct rules solve clean bands while indirect/noisy rows expose robustness limits. |
| Proposer usefulness | `C2.*` | `crlp.rule_search.summary.v1`, `crlp.program_proposer.summary.v1` | The current prime proposer result is mixed/negative; prose must not overclaim. |
| RL calibration | `C3.*` | `crlp.compare.v1` with ECE/reliability bins | Prime has evidence; cross-task replication is still open. |
| Multitask transfer | `C4.*` | `crlp.multitask.summary.v1`, roadmap CSVs | Same-envelope ARC OOD is positive; fixed-task ARC is negative. |
| Substrate portability | `C5.*` | `crlp.hypothesis.evidence.v1`, `crlp.arc_induction.parity.v1` | Public ARC parity row and prime-gap cross-port remain open. |
| Literature positioning | `C6.*` | bibliography entries plus manually curated summaries | This can be source-checked but not generated from C results. |
| Training completeness | `C7.*` | train-to-plateau, restart, lineage, and artifact-fed training rows | Needed before architecture/capacity claims are allowed to harden. |
| Paper compiler | `C8.*` | claim registry, support reports, generated limitations, roadmap-link audit | Framework claim; it governs whether prose is allowed to compile. |

## Roadmap Items

These items sit above the experiment roadmap. They do not replace R-E, R-X,
R-MT, R-S, or R-W items; they make those items machine-actionable.

| ID | Work | Effort | Defends |
| --- | --- | ---: | --- |
| R-P.1 | Create `paper/claims.yaml` with the first-pass claim registry for C1/C2/C4. | done | self-authoring |
| R-P.2 | Add `scripts/derive_paper_claims.py` to evaluate claims against `comparison.canonical.jsonl` and roadmap result CSVs. | done | self-authoring |
| R-P.3 | Generate `paper/generated/claims.json`, `claims.typ`, and `gap_report.md`, and import the generated claim snapshot into the paper appendix. | done | self-authoring |
| R-P.4 | Replace hard-coded abstract/body/caption metrics with generated claim variables. | 1d | self-authoring |
| R-P.5 | Add `make paper-support`, which fails on stale, missing, underpowered, or contradicted required claims. | done | self-authoring |
| R-P.6 | Link every paper roadmap item to the claim family it supports or unblocks; `roadmap_links.md` now reports zero unlinked rows. | done | self-authoring |
| R-P.7 | Generate limitation candidates from contradicted and underpowered claims. | done | self-authoring |
| R-P.8 | Add an editorial-agent PR boundary: prose-only review checklist, claim-change escape hatch, CI enforcement, paper gates, and disclosure language. | done | self-authoring |

## Submission Tiers

Minimum viable self-authoring:

- R-P.1 through R-P.3 are done for the first three load-bearing claims:
  `C1.prime_gate_delta`, `C2.prime_proposer_fixed_budget`, and
  `C4.arc_ood_shared_vs_single`.
- `gap_report.md` exists and identifies unsupported or contradicted paper claims.
- `make paper-support` exists and blocks required unsupported claims before PDF compilation.
- `limitations.md` exists and the PDF imports generated limitation candidates.
- The paper can still contain hand prose, but the claim status is generated.

Strong self-authoring:

- R-P.4.
- R-P.8.
- Main numerical claims in the abstract, headline tables, and major captions
  come from generated claim variables.
- CI or local `make check` can detect stale or contradicted required claims.
- Editorial-agent changes move through reviewable PRs whose template prevents
  prose edits from silently changing claim scope, numbers, citations, generated
  summaries, or experiment interpretation.

Full self-authoring:

- Claim registry coverage for all C1-C6 families plus generated roadmap summaries.
- A changed evidence row can automatically move a claim from body prose to a
  limitation or roadmap item without hand-auditing the whole PDF.

## Next Step

R-P.4 should replace the literal numbers in the abstract, Axis 1 table,
multitask table, and major captions with generated variables from:

1. `C1.prime_gate_delta`
2. `C2.prime_proposer_fixed_budget`
3. `C4.arc_ood_shared_vs_single`

`make paper-support` now implements the required-claim gate, and
`paper/generated/roadmap_links.md` reports zero unlinked roadmap rows. R-P.4 is
the remaining systems task: replace body/abstract literal metrics with generated
claim variables so stale evidence cannot survive as hard-coded prose.
