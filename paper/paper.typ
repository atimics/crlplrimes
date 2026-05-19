#import "@preview/cetz:0.3.4"
#import "@preview/cetz-plot:0.1.1": plot
#import "generated/claims.typ": paper_claim_limitations, paper_claim_support_ok, paper_claim_support_surfaces, paper_claim_status_counts, claim_C1_prime_gate_delta_summary, claim_C1_prime_gate_delta_ci_summary, claim_C1_prime_gate_delta_raw_top1, claim_C1_prime_gate_delta_hand_top1, claim_C1_prime_gate_delta_learned_top1, claim_C1_prime_gate_delta_learned_top10, claim_C1_prime_gate_delta_learned_vs_hand_delta_pp, claim_C1_prime_gate_delta_false_rejections, claim_C1_prime_gate_delta_sample_count, claim_C2_prime_proposer_fixed_budget_summary, claim_C1_arc_public_evidence_surface_chart_rows, claim_C1_arc_public_evidence_surface_summary, claim_C3_arc_ood_calibration_tradeoff_chart_rows, claim_C3_arc_ood_calibration_tradeoff_summary, claim_C4_delayed_claim_sibling_transfer_summary, claim_C4_multitask_capacity_tradeoff_chart_rows, claim_C4_multitask_capacity_tradeoff_summary, claim_C4_adapter_heads_shared_trunk_summary, claim_C5_rejected_candidate_reason_learning_chart_rows, claim_C5_rejected_candidate_reason_learning_summary, claim_C5_hypothesis_library_promotion_chart_rows, claim_C5_hypothesis_library_promotion_summary, claim_C7_tiny_net_plateau_stop_rule_summary, claim_C8_trace_determinism_audit_summary, claim_C9_signal_flight_replay_control_chart_rows, claim_C9_signal_flight_replay_control_summary, claim_C9_signal_live_neural_active_chart_rows, claim_C9_signal_live_neural_active_summary
#import "generated/experiment_matrix.typ": experiment_matrix_rows

#set document(
  title: "Executable Accountability for Verifier-Grounded Residual Ranking",
  author: "Anonymous authors",
)

#set page(
  paper: "us-letter",
  margin: (x: 1in, y: 1in),
  numbering: "1",
)

#set text(font: "New Computer Modern", size: 10.5pt)
#set par(justify: true, leading: 0.6em)
#set heading(numbering: "1.")
#show heading.where(level: 1): set text(size: 13pt, weight: "bold")
#show heading.where(level: 2): set text(size: 11.5pt, weight: "bold")
#show raw: set text(font: "DejaVu Sans Mono", size: 9pt)

#let pp(v) = {
  let sign = if v >= 0 { "+" } else { "-" }
  let mag = calc.round(calc.abs(v) * 100) / 100
  [#sign#mag pp]
}

#let pct(v) = [#(calc.round(v * 10000) / 100)%]

#let claim_delta_chart(rows, y-min: -20, y-max: 20, y-label: [Delta]) = cetz.canvas({
  import cetz.draw: *
  plot.plot(
    size: (10, 4.2),
    x-label: none,
    y-label: y-label,
    x-min: -0.5,
    x-max: rows.len() - 0.5,
    y-min: y-min,
    y-max: y-max,
    x-tick-step: none,
    y-tick-step: calc.round((y-max - y-min) / 4),
    x-ticks: rows.enumerate().map(((i, r)) => (i, text(size: 7.5pt, r.label))),
    {
      plot.add(
        ((-0.5, 0), (rows.len() - 0.5, 0)),
        style: (stroke: (paint: gray, thickness: 0.5pt)),
      )
      for (i, r) in rows.enumerate() {
        let fill = if r.value >= 0 { rgb("#2ca02c") } else { rgb("#d62728") }
        plot.add-bar(
          ((i, r.value),),
          bar-width: 0.58,
          style: (fill: fill, stroke: black + 0.45pt),
        )
      }
    },
  )
})

#let claim_coverage_chart(rows, y-max: 260, y-label: [Coverage]) = cetz.canvas({
  import cetz.draw: *
  plot.plot(
    size: (10, 4.2),
    x-label: none,
    y-label: y-label,
    x-min: -0.5,
    x-max: rows.len() - 0.5,
    y-min: 0,
    y-max: y-max,
    x-tick-step: none,
    y-tick-step: calc.round(y-max / 4),
    x-ticks: rows.enumerate().map(((i, r)) => (i, text(size: 7.5pt, r.label))),
    {
      for (i, r) in rows.enumerate() {
        plot.add-bar(
          ((i, r.value),),
          bar-width: 0.58,
          style: (fill: rgb("#4e79a7"), stroke: black + 0.45pt),
        )
      }
    },
  )
})

#align(center)[
  #text(size: 17pt, weight: "bold")[
    Executable Accountability for \
    Verifier-Grounded Residual Ranking
  ]

  #v(4pt)
  #text(size: 11pt)[
    Certificate-gated neural proposers under claim governance
  ]

  #v(8pt)
  #text(size: 11pt)[
    Anonymous authors
  ]
]

#v(6pt)

#align(center)[
  #box(width: 85%)[
    #set par(justify: true)
    #set text(size: 10pt)
    *Abstract.*
    This paper studies executable accountability for verifier-grounded residual ranking.
    Exact or replayable checks own admissibility and consequences; small learned models rank,
    calibrate, reuse, or learn around the residual. The contribution is not that tiny
    networks solve hard tasks, but that verifier and replay artifacts become proposal
    structure while the manuscript is constrained by generated evidence. The clean
    anchor is `prime_gap`, where the generated claim row reports
    #claim_C1_prime_gate_delta_summary The main feedback results are
    rejected-candidate reason learning and certified hypothesis-library promotion. Boundary
    rows for ARC, Signal, tiny-net capacity, and language-adjacent haiku probes test scope
    without expanding the central claim. The paper explicitly does not claim broad ARC
    competence, CDCL-equivalent learned clauses, comparative live neural-control
    improvement, semantic language understanding, or general end-to-end task solving.
  ]
]

#v(10pt)

= Introduction

Machine-learning systems increasingly operate in settings where bad actions can be checked
after the fact, but correctness should not be delegated to end-to-end model authority.
Verifier-grounded systems should separate authority from ranking. Exact checkers,
deterministic replay, and certificate tests should decide which candidates are admissible or
what consequences a candidate produced. Learned components should spend their limited
capacity on the residual: search order, calibration, reuse, and prioritization. The same
discipline should govern the paper. Empirical prose should be downstream of trace rows,
environment identity, checkpoint hashes, result summaries, and generated limitation
candidates rather than manually maintained claims.

This paper is therefore a protocol paper, not an experiment catalog. The one-sentence thesis
is:

#quote[
  Verifier-grounded systems should let exact or replayable checks own admissibility, use
  small learned models only to rank the residual, and bind empirical claims to executable
  artifact evidence.
]

The operational summary is simple: the model proposes, the verifier decides, and the trace
remembers enough to train the next proposal pass and govern the next manuscript claim.

Two empirical loops carry the result. First, rejected verifier rows are logged with reason
codes and reused as supervision for a later safety-first residual ranker. Second, repeatedly
certified hypotheses are promoted into a library and reused as proposal priors under held-out
budgets. The prime-gap gate result supplies the clean anchor; the fixed-budget proposer
failure supplies the falsifying negative row that motivates the feedback loop.

The remaining domains are boundary evidence. ARC tests grammar provenance and failure
taxonomy, not benchmark competence. Signal tests replay and guarded live execution pressure,
not comparative live-control performance. Tiny-net and adapter rows bound architecture
claims. Haiku is language-adjacent corpus-proxy grounding, not semantic understanding. New
domains stay outside the paper body until promoted through the evidence matrix.

The environment names are deliberately small. `prime_gap` tests exact arithmetic
verification; `grid_path` tests topological path constraints; delayed-claim tests hidden
outcome and safety-margin gates; ARC tests visual-spatial abstraction provenance; Signal
tests bounded WASD control under replay and live telemetry; haiku tests corpus-derived proxy
grounding.

At this artifact snapshot, the generated claim compiler reports
#paper_claim_status_counts.supported supported claims,
#paper_claim_status_counts.contradicted contradicted claims,
#paper_claim_status_counts.underpowered underpowered claim,
#paper_claim_status_counts.missing missing claims, and
#paper_claim_status_counts.stale stale claims. Contradicted and underpowered rows remain
visible because falsifiability is part of the protocol. The fixed-budget proposer row is
contradicted (#claim_C2_prime_proposer_fixed_budget_summary), and the tiny-net plateau row is
underpowered (#claim_C7_tiny_net_plateau_stop_rule_summary).

= Positioning

The formal-methods ancestry is counterexample-guided search. Sketching combines inductive
synthesis with validation-generated counterexamples @solarlezama2008sketching, and SyGuS
names the grammar-plus-specification setting @alur2013sygus. CEGAR and CDCL add the adjacent
lesson that failed traces can refine later search rather than disappear as waste
@clarke2000cegar @marquessilva1999grasp @bayardo1997csp. This paper inherits that shape but
uses failures as supervised ranking data, not as sound learned clauses.

Modern verifier-bounded learning asks models to solve or help verify reasoning traces
@cobbe2021training @lightman2023verify @uesato2022solving @hosseini2024vstar
@deepseek2025r1. AlphaGeometry-style systems combine learned proposal with formal checking
@trinh2024alphageometry @chervonyi2025alphageometry2, and theorem-proving work similarly
uses neural proposal around formal environments @polu2020generative. The artifact here is
smaller and narrower: the learned model never owns correctness.

The accountability lineage is equally important. Model Cards, Datasheets, reproducibility
checklists, and reinforcement-learning reproducibility critiques made documentation and
artifact discipline central research objects @mitchell2019modelcards @gebru2021datasheets
@pineau2021reproducibility @henderson2018deep. The claim compiler is an executable version
of that idea: claims, numbers, limitations, and support status are generated from repository
artifacts before the paper compiles.

= Protocol

Every promoted row has the same contract:

```text
Task T = {
  state distribution S_T,
  candidate enumerator A_T(s),
  observable features x_T(s, a),
  residual scorer f_theta(x_T),
  grounding operator O_T(s, a),
  grounding authority level,
  grammar G_T,
  certificate or replay checker C_T,
  trace / replay / claim protocol
}
```

The invariant is:

```text
The model proposes. The grounding operator decides. The trace remembers.
```

#figure(
  table(
    columns: (1.0fr, 2.1fr, 2.2fr),
    align: left,
    inset: 4pt,
    table.header[*Stage*][*Concrete prime-gap object*][*Artifact that survives*],
    [Candidate],
      [The proposer emits a modular gate candidate such as "reject candidate `n` when
        `n > 3` and `n % 3 == 0`" for next-prime search states.],
      [Grammar id, candidate id, split, seed, and support window.],
    [Gate check],
      [The segmented-sieve verifier checks whether the rule removes any true next-prime
        outcome under the declared support.],
      [A false-rejection certificate; accepted only when false rejections are zero.],
    [Residual rank],
      [Certified composite candidates are removed, and the scorer ranks the remaining
        action ids by residual score.],
      [Top-k metrics, stable action ids, ranking policy, and score margins.],
    [Trace emit],
      [The run writes comparison, rule-search, and certificate-trace rows.],
      [`crlp.compare.v1`, `crlp.rule_search.summary.v1`, environment and model hashes.],
    [Claim consume],
      [The claim compiler derives the paper macro from the artifact summary.],
      [`claim_C1_prime_gate_delta_summary` and generated support/limitation status.],
  ),
  caption: [Concrete prime-gap trace. The paper number is downstream of candidate
    generation, verifier certification, residual ranking, trace emission, and generated
    claim consumption.],
) <tab:prime-trace-walkthrough>

An exact verifier is the strongest grounding operator, but the protocol also admits
deterministic replay and explicitly weaker corpus proxies. Authority level determines claim
scope.

#figure(
  table(
    columns: (1.15fr, 1.55fr, 2.05fr, 2.1fr),
    align: left,
    inset: 4pt,
    table.header[*Authority*][*Operator*][*Examples*][*Allowed claim*],
    [exact check],
      [proof, lookup, type check, deterministic verifier],
      [`prime_gap`, `grid_path`, certified ARC transform rows],
      [correctness and zero-false-rejection claims inside declared support],
    [deterministic replay],
      [simulator or replay deciding candidate consequences],
      [Signal flight, delayed-claim simulator],
      [replay-certified utility, regret, damage, death, or decision metrics],
    [corpus proxy],
      [train-split grammar, adjacency, corruption benchmark],
      [haiku prototypes, public corpus coverage surfaces],
      [corpus fit and near-miss discrimination, not semantic truth],
    [human review],
      [reviewable acceptance/rejection],
      [editorial-agent pull requests],
      [review-gated assistance, not autonomous empirical authority],
  ),
  caption: [Grounding authority levels. The same residual scorer has different evidentiary
    force depending on which operator produced the outcome row.],
) <tab:grounding-authority>

The task-independent rules are concise. Candidate enumeration is complete before gating.
Features declare provenance, and grounding-derived quantities are labels, rewards,
diagnostics, or gates unless the schema explicitly permits them as inputs. Rules enter the
gate only through the task grammar and certificate or replay checker. JSONL traces use stable
candidate ids, ranking policy, score-margin metadata, environment identity, and replay
semantic hashes so reordered candidates or float ties cannot silently change the paper's
meaning.

= Evidence Matrix

The matrix is the scope boundary. A row may be interpreted in the prose only if its data,
candidate grammar, grounding operator, scorer role, trace artifact, result, claim, and
limitation are registered. The paper may tighten wording without adding experiment families;
new families require revising this matrix first.

#figure(
  text(size: 7pt)[
    #table(
      columns: (0.9fr, 2.25fr, 1.65fr, 2.45fr),
      align: left,
      inset: 4pt,
      table.header[*Family*][*Grounding*][*Learned Role*][*Claim / Boundary*],
      ..experiment_matrix_rows.map(r => (
        [#strong[#r.domain_display]],
        [
          #strong[Grounding:] #r.verifier; #r.certificate_gate
        ],
        [#r.scorer_role],
        [
          #strong[Claim:] #r.claim.
          #linebreak()
          #strong[Limit:] #r.limitation
        ],
      )).flatten()
    )
  ],
  caption: [Frozen evidence matrix. Rows use a four-column protocol view to avoid treating
    the paper as a leaderboard or a catalog.],
) <tab:frozen-evidence>

= Prime-Gap Anchor

The prime-gap row is the cleanest evidence because the verifier is exact and the gate has a
clear safety property. Candidate actions are possible prime gaps after a segmented-sieve
window. Candidate gates are modular rejection predicates. A gate is accepted only if its
certificate has zero false rejections over the declared support; the scorer then ranks the
residual candidates. The `hand wheel` baseline is a deterministic hand-coded
wheel-factorization filter over small modular divisibility rules.

#figure(
  table(
    columns: (1.5fr, 1.05fr, 1.05fr, 1.05fr),
    align: left,
    inset: 5pt,
    table.header[*Condition*][*Top-1*][*Top-10*][*False rejections*],
    [no gate], [#pct(claim_C1_prime_gate_delta_raw_top1)], [78.01%], [0],
    [hand wheel], [#pct(claim_C1_prime_gate_delta_hand_top1)], [99.77%], [0],
    [learned wheel], [#strong[#pct(claim_C1_prime_gate_delta_learned_top1)]],
      [#strong[#pct(claim_C1_prime_gate_delta_learned_top10)]],
      [#strong[#claim_C1_prime_gate_delta_false_rejections]],
  ),
  caption: [Prime-gap certified modular gate on the canonical sample of
    #claim_C1_prime_gate_delta_sample_count rows. #claim_C1_prime_gate_delta_summary
    Learned-vs-hand top-1 delta: #pp(claim_C1_prime_gate_delta_learned_vs_hand_delta_pp).],
) <tab:prime-anchor>

The seed audit closes the headline-stability gap for this anchor:
#claim_C1_prime_gate_delta_ci_summary
The zero spread is a seed-stability audit over this deterministic benchmark: seeds vary
checkpoints and proposer runs, not the declared prime-range evaluation distribution.

This result is intentionally narrow. It supports the protocol claim that certified structure
can shrink residual search sharply while preserving correctness under the declared support.
It does not imply that a static neural proposer is better than deterministic symbolic
ordering. The generated negative row says the opposite:
#claim_C2_prime_proposer_fixed_budget_summary That failure is useful because it prevents the
paper from becoming a generic "neural proposal wins" story.

The naive learned proposer fails. That is the point. Unsupported neural proposal is not
trusted; verifier feedback is recycled until proposal economics improve. In this protocol,
negative rows are not cleanup work after the fact. They are the training substrate that
separates residual ranking from admissibility.

= Feedback Loops

The main contribution is what happens after verifier feedback exists.

*Rejected-candidate learning.* Failed rule proposals are logged with task, split, seed,
reason family, false-rejection counts, and marginal verifier utility. Those rows become
targets for a later residual ranker. The supported claim is scoped to prime/modular sweeps:
#claim_C5_rejected_candidate_reason_learning_summary

#figure(
  claim_coverage_chart(
    claim_C5_rejected_candidate_reason_learning_chart_rows,
    y-max: 260,
    y-label: [accepted-rule utility],
  ),
  caption: [Rejected-candidate reason learning. Verifier failures become training signal
    without making the learned model authoritative.],
) <fig:claim-reason-learning>

*Certified hypothesis-library feedback.* Repeated zero-false-rejection hypotheses are promoted
into a reusable library, converted into slot priors, and checked on held-out tasks under the
same tight budget. The supported claim is:
#claim_C5_hypothesis_library_promotion_summary

#figure(
  claim_coverage_chart(
    claim_C5_hypothesis_library_promotion_chart_rows,
    y-max: 6,
    y-label: [certified at 1],
  ),
  caption: [Certified hypothesis-library promotion. Certified hypotheses become proposal
    priors only after passing the promotion gate.],
) <fig:claim-hypothesis-library>

Together these rows are the paper's feedback mechanism. Rejections and certifications are not
post-hoc diagnostics; they are reusable artifacts that change the next proposal pass.

= Boundary Evidence

Boundary rows keep the thesis honest without becoming separate stories.

*ARC.* ARC is a provenance and grammar-coverage stress surface. The paper reports the current
measured public-corpus surface, not a headline ARC score:
#claim_C1_arc_public_evidence_surface_summary Calibration remains conditional:
#claim_C3_arc_ood_calibration_tradeoff_summary

#figure(
  claim_coverage_chart(
    claim_C1_arc_public_evidence_surface_chart_rows,
    y-max: 1200,
    y-label: [evaluation cells],
  ),
  caption: [Public ARC evidence surface. This is measured `v15` coverage and failure
    taxonomy, not broad ARC competence.],
) <fig:claim-arc-public-surface>

*Signal.* Signal is a deterministic-replay pressure test. Offline replay scores bounded WASD
candidate actions against recorded simulator consequences, and closed-loop replay evaluates
guarded rollout rows before any live-control language is admitted. The live `neural_active`
row is narrower: a running local Signal pilot applies checkpoint-ranked WASD proposals only
through protocol checks and safety gates, and those gates may veto execution instead of
letting the neural top choice own control. The trace records active source, checkpoint hash,
latency, protocol compatibility, safety-gate rows, damage/death telemetry, and replay-ready
frames:
#claim_C9_signal_flight_replay_control_summary
#claim_C9_signal_live_neural_active_summary The paper still does not claim active neural
control improves utility, regret, damage, or death versus the heuristic baseline.

#figure(
  claim_coverage_chart(
    claim_C9_signal_live_neural_active_chart_rows,
    y-max: 260,
    y-label: [live rows],
  ),
  caption: [Signal guarded live neural-active execution. The row establishes live
    checkpoint-ranked execution and replay-ready telemetry, not comparative active-control
    superiority.],
) <fig:claim-signal-live-active>

*Architecture and language-adjacent probes.* Shared-scorer, adapter, and tiny-net rows remain
boundary evidence: #claim_C4_multitask_capacity_tradeoff_summary
#claim_C4_adapter_heads_shared_trunk_summary Haiku prototypes use corpus-derived vocabulary,
adjacency, syllable/shape constraints, and real-vs-corrupt discrimination as a
`corpus_proxy` operator. They test how the architecture behaves when the "verifier" is only
dataset grounding; they are intentionally not promoted as semantic language understanding.

= Claim Governance

The paper construction layer mirrors the protocol. Result scripts emit canonical summaries,
the claim compiler converts those summaries into Typst macros and limitation candidates, and
the PDF consumes those macros rather than hand-maintained numbers. The trace determinism row
is currently supported: #claim_C8_trace_determinism_audit_summary
This is not a claim that every sentence is cryptographically enforced. It is the narrower
claim that registered empirical claims cannot remain supported when their required artifacts
are missing, stale, contradicted, or underpowered.

#figure(
  table(
    columns: (1.2fr, 0.8fr, 2.6fr),
    align: (left, left, left),
    inset: 5pt,
    table.header[*Class*][*Count*][*Body rule*],
    [supported], [#paper_claim_status_counts.supported],
      [May be interpreted, using generated summaries and artifact-backed scope.],
    [contradicted], [#paper_claim_status_counts.contradicted],
      [Must remain negative evidence, limitation, or claim-change work.],
    [underpowered], [#paper_claim_status_counts.underpowered],
      [May motivate roadmap items, not body-strength architecture claims.],
    [missing/stale], [#paper_claim_status_counts.missing / #paper_claim_status_counts.stale],
      [Must block or remove body claims until artifacts are regenerated.],
  ),
  caption: [Generated claim-gate summary. The detailed support-surface rows remain in
    `paper/generated/claims.typ`; the body uses the counts and generated claim macros rather
    than forcing every long row into the PDF.],
) <tab:generated-claim-support>

Editorial agents fit the same governance model. They may improve wording, structure,
transitions, and clarity through reviewable pull requests. They may not change claim scope,
numeric results, generated summaries, citations, or experiment interpretation unless they
open an explicit claim-change PR. This disclosure is operational rather than decorative: AI
agents are reviewable assistants, while empirical claims are generated from repository
artifacts and checked before compilation.

= Limitations

#set enum(numbering: "1.", spacing: 6pt)

+ *Scope is deliberately narrow.* The paper supports verifier-grounded residual ranking and
  executable claim governance. It does not support broad task solving, broad ARC
  competence, semantic language understanding, or competitive theorem proving.
+ *Multi-seed intervals are still uneven.* The clean prime-gate headline now has a generated
  seed interval. Some boundary rows still need more explicit confidence intervals before
  they should be treated as stable quantitative laws.
+ *The fixed-budget proposer result is negative.* The generated contradiction is not an
  embarrassment; it is the reason rejected-candidate learning matters.
+ *Architecture claims are underpowered or contradicted.* Tiny-net plateau evidence remains
  underpowered, and adapter heads are diagnostic rather than promoted architecture.
+ *Signal is live but not comparative.* The live `neural_active` row establishes guarded
  checkpoint-ranked execution with replay-ready telemetry. It does not establish active
  neural improvement over the heuristic baseline.
+ *Corpus proxies are weaker than verifiers.* The haiku task can ground train-split
  adjacency, shape, and corruption discrimination. It cannot ground truth or semantics.
+ *Executable accountability is partial.* Claim compilation catches unsupported generated
  claims, stale evidence, and limitation candidates. It does not replace human judgment
  about interpretation, ethics, or whether a domain belongs in the paper.

#set enum(numbering: "1.")

#let paper_support_status = if paper_claim_support_ok { [passing] } else { [failing] }

#figure(
  table(
    columns: (auto, auto, 1.6fr),
    align: (left, left, left),
    table.header[*Claim*][*Status*][*Generated limitation candidate*],
    ..paper_claim_limitations.map(item => (
      [#raw(item.claim_id)],
      [#item.status],
      [#item.text],
    )).flatten(),
  ),
  caption: [Generated limitation candidates from the paper-claim compiler. The required
    claim support gate is currently #paper_support_status. Optional contradicted claims
    remain visible here instead of blocking PDF compilation.],
) <tab:generated-limitations>

= Conclusion

The result is a disciplined loop: enumerate candidates, ground them with an exact check or
replayable operator, let a small model rank the residual, preserve the trace, and bind the
paper's claims to generated evidence. Prime-gap shows the clean certified-gate anchor.
Rejected-candidate learning and hypothesis-library promotion show that verifier artifacts can
feed back into later proposal structure. ARC, Signal, tiny-net, adapter, and haiku rows mark
the boundary rather than widening the thesis.

The broader lesson is pragmatic. When correctness, safety, or auditability matter, the model
should not own the whole decision and the paper should not own the whole narrative. The
checker or replay system owns admissibility and consequence. The learned component owns
ranking pressure. The repository trace owns what the manuscript is allowed to say.

#bibliography("references.bib")
