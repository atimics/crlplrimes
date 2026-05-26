#import "@preview/cetz:0.3.4"
#import "@preview/cetz-plot:0.1.1": plot
#import "generated/claims.typ": paper_claim_limitations, paper_claim_support_ok, paper_claim_support_surfaces, paper_claim_status_counts, claim_C1_prime_gate_delta_summary, claim_C1_prime_gate_delta_ci_summary, claim_C1_prime_gate_delta_raw_top1, claim_C1_prime_gate_delta_hand_top1, claim_C1_prime_gate_delta_learned_top1, claim_C1_prime_gate_delta_learned_top10, claim_C1_prime_gate_delta_learned_vs_hand_delta_pp, claim_C1_prime_gate_delta_false_rejections, claim_C1_prime_gate_delta_sample_count, claim_C2_prime_proposer_fixed_budget_summary, claim_C1_arc_public_evidence_surface_chart_rows, claim_C1_arc_public_evidence_surface_summary, claim_C3_arc_ood_calibration_tradeoff_chart_rows, claim_C3_arc_ood_calibration_tradeoff_summary, claim_C4_delayed_claim_sibling_transfer_summary, claim_C4_multitask_capacity_tradeoff_chart_rows, claim_C4_multitask_capacity_tradeoff_summary, claim_C4_adapter_heads_shared_trunk_summary, claim_C5_rejected_candidate_reason_learning_chart_rows, claim_C5_rejected_candidate_reason_learning_summary, claim_C5_hypothesis_library_promotion_chart_rows, claim_C5_hypothesis_library_promotion_summary, claim_C7_tiny_net_plateau_stop_rule_summary, claim_C8_trace_determinism_audit_summary, claim_C8_dataset_engine_evidence_surface_chart_rows, claim_C8_dataset_engine_evidence_surface_summary, claim_C9_signal_flight_replay_control_chart_rows, claim_C9_signal_flight_replay_control_summary, claim_C9_signal_live_neural_active_chart_rows, claim_C9_signal_live_neural_active_summary
#import "generated/experiment_matrix.typ": experiment_matrix_rows

#set document(
  title: "Verified Search with Neural Ranking and Executable Claims",
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

#let claim_count_label(count, singular, plural) = if count == 1 { singular } else { plural }

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
    Verified Search with Neural Ranking \
    and Executable Claims
  ]

  #v(4pt)
  #text(size: 11pt)[
    Rules decide what is allowed; neural models rank what remains
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
    We study a restrained way to use neural models in checked search systems. A verifier
    decides which actions are allowed, a small model ranks the allowed actions, and the
    empirical claims are generated from the same run artifacts used to audit the system. The
    contribution is not that tiny networks solve hard tasks. It is that checked failures and
    checked successes become reusable training and proposal structure without giving the
    model authority over correctness. The clean anchor is arithmetic sequence filtering: in
    the audited next-prime instance (`prime_gap` in traces), the generated claim row reports
    #claim_C1_prime_gate_delta_summary The same family also admits recurrence filters such
    as Fibonacci/Lucas gates via period residues, but those remain smoke-level boundary
    evidence here. The main feedback results are rejected-candidate reason learning and
    certified hypothesis-library promotion. Boundary rows for ARC, Signal, tiny-net
    capacity, and haiku test where this protocol holds and where it weakens. The current
    artifact also exposes a dataset-first engine surface that ports several domains into one
    candidate-row contract, but that surface is infrastructure evidence rather than a new
    domain-result claim. The scope of this evidence explicitly excludes broad ARC competence,
    CDCL-equivalent learned clauses, comparative live neural-control improvement, semantic
    language understanding, and general end-to-end task solving.
  ]
]

#v(10pt)

= Introduction

Many tasks have a simple asymmetry: it is hard to know the best action in advance, but easy
to check whether a proposed action is allowed. In those settings, the learned model should
not own the final answer. A rule, verifier, certificate, or replay system should decide what
is valid. The model should only decide which valid candidate to try first.

The same discipline should govern the claims. Reported numbers should come from trace rows,
environment identity, checkpoint hashes, result summaries, and generated limitation
candidates rather than from hand-maintained prose.

The thesis is:

#quote[
  Let exact or replayable checks decide what is allowed. Let small learned models order the
  allowed choices. Let repository artifacts decide which empirical claims are admissible.
]

The operational summary is simple: the model proposes, the checker decides, and the trace
remembers enough to train the next proposal pass and constrain the next empirical claim.

Two empirical loops carry the result. First, rejected verifier rows are logged with reason
codes and reused as supervision for a later safety-first residual ranker. Second, repeatedly
certified hypotheses are promoted into a library and reused as proposal priors under held-out
budgets. The arithmetic sequence-filter result supplies the clean anchor; the fixed-budget
proposer failure supplies the falsifying negative row that motivates the feedback loop.

The remaining domains are boundary evidence. ARC tests grammar coverage and failure
taxonomy, not benchmark competence. Signal tests replay and guarded live execution pressure,
not comparative live-control performance. Tiny-net and adapter rows bound architecture
claims. Haiku is a language-adjacent corpus check, not semantic understanding. The
dataset-first runner is a shared infrastructure surface: it can show that a domain compiles to
complete candidate rows, verifier labels, traces, checkpoints, and engine evidence, but it
does not by itself promote the domain's scientific result. New domains stay outside the claim
set until promoted through the evidence matrix.

The environment names are deliberately small. The audited `prime_gap` trace is the
next-prime instance of a broader modular-filter family; smoke rows already include
Fibonacci/Lucas recurrence filters via period residues. `grid_path` tests topological path
constraints; delayed-claim tests hidden outcome and safety-margin gates; ARC tests
visual-spatial abstraction provenance; Signal tests bounded WASD control under replay and
live telemetry; haiku tests corpus-derived proxy grounding.

At this artifact snapshot, the generated claim compiler reports
#paper_claim_status_counts.supported #claim_count_label(paper_claim_status_counts.supported, [supported claim], [supported claims]),
#paper_claim_status_counts.contradicted #claim_count_label(paper_claim_status_counts.contradicted, [contradicted claim], [contradicted claims]),
#paper_claim_status_counts.underpowered #claim_count_label(paper_claim_status_counts.underpowered, [underpowered claim], [underpowered claims]),
#paper_claim_status_counts.missing #claim_count_label(paper_claim_status_counts.missing, [missing claim], [missing claims]), and
#paper_claim_status_counts.stale #claim_count_label(paper_claim_status_counts.stale, [stale claim], [stale claims]). Contradicted and underpowered rows remain
visible because falsifiability is part of the protocol. The fixed-budget proposer row is
contradicted (#claim_C2_prime_proposer_fixed_budget_summary), and the tiny-net plateau row is
underpowered (#claim_C7_tiny_net_plateau_stop_rule_summary).

= Positioning

The formal-methods ancestry is counterexample-guided search. Sketching combines inductive
synthesis with validation-generated counterexamples @solarlezama2008sketching, and SyGuS
names the grammar-plus-specification setting @alur2013sygus. CEGAR and CDCL add the adjacent
lesson that failed traces can refine later search rather than disappear as waste
@clarke2000cegar @marquessilva1999grasp @bayardo1997csp. This protocol inherits that shape
but uses failures as supervised ranking data, not as sound learned clauses.

Modern verifier-bounded learning asks models to solve or help verify reasoning traces
@cobbe2021training @lightman2023verify @uesato2022solving @hosseini2024vstar
@deepseek2025r1. AlphaGeometry-style systems combine learned proposal with formal checking
@trinh2024alphageometry @chervonyi2025alphageometry2, and theorem-proving work similarly
uses neural proposal around formal environments @polu2020generative. The artifact here is
smaller and narrower: the learned model never owns correctness.

The accountability lineage is equally important. Model Cards, Datasheets, reproducibility
checklists, and reinforcement-learning reproducibility critiques made documentation and
artifact discipline central research objects @mitchell2019modelcards @gebru2021datasheets
@pineau2021reproducibility @henderson2018deep. Here, that idea becomes a build rule:
numbers, limitations, and support status are generated from repository artifacts before
claim compilation.

= Protocol

The protocol has three jobs. First, list the candidates. Second, use a checker or replay to
decide what each candidate is allowed to mean. Third, let a small model rank only the
remaining choices. Every promoted row records those jobs in the same contract:

```text
Task T = {
  state distribution S_T,
  candidate enumerator A_T(s),
  observable features x_T(s, a),
  candidate ranker f_theta(x_T),
  checking or replay operator O_T(s, a),
  checker authority level,
  grammar G_T,
  certificate or replay checker C_T,
  trace / replay / claim protocol
}
```

The protocol's invariant is simple:

```text
The model proposes. The checker decides. The trace remembers.
```

#figure(
  table(
    columns: (1.0fr, 2.1fr, 2.2fr),
    align: left,
    inset: 4pt,
    table.header[*Stage*][*Concrete arithmetic-filter object*][*Artifact that survives*],
    [Candidate],
      [The proposer emits a modular gate candidate such as "reject candidate `n` when
        `n > 3` and `n % 3 == 0`" for next-prime search states.],
      [Grammar id, candidate id, split, seed, and support window.],
    [Gate check],
      [The segmented-sieve verifier checks whether the rule removes any true next-prime
        outcome under the declared support.],
      [A false-rejection certificate; accepted only when false rejections are zero.],
    [Residual rank],
      [Certified composite candidates are removed, and the model ranks the remaining
        action ids.],
      [Top-k metrics, stable action ids, ranking policy, and score margins.],
    [Trace emit],
      [The run writes comparison, rule-search, and certificate-trace rows.],
      [`crlp.compare.v1`, `crlp.rule_search.summary.v1`, environment and model hashes.],
    [Claim consume],
      [The claim compiler derives the registered summary from the artifact.],
      [`claim_C1_prime_gate_delta_summary` and generated support/limitation status.],
  ),
  caption: [Concrete arithmetic-filter trace. The headline number is downstream of candidate
    generation, verifier certification, model ranking, trace emission, and generated claim
    consumption.],
) <tab:prime-trace-walkthrough>

Not every domain gives the system the same kind of checker. Exact verification is strongest.
Replay is weaker but still auditable. Corpus tests are weaker again: they can show fit to a
dataset, not truth. Claim scope changes with that authority level.

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
  caption: [Checker authority levels. The same model ranking has different evidentiary force
    depending on which operator produced the outcome row.],
) <tab:grounding-authority>

The weaker operators do not inherit the authority of stronger ones. If `O_T` is learned,
sampled, corpus-derived, or human-adjudicated, the checked object is the declared estimator
or review process: its split, seed, rubric, artifact hash, and acceptance rule. The claim
compiler may then certify conformance to that operator, not correctness in the underlying
open domain. A human-reviewed claim therefore remains review-gated assistance; a
corpus-proxy claim remains corpus fit; neither becomes an exact-verifier claim by passing
through the same trace format.

The task-independent rules are short. Enumerate candidates before checking them. Say where
features came from. Treat checker-derived quantities as labels, rewards, diagnostics, or
gates unless a schema explicitly allows them as model inputs. Let rules enter the gate only
through the task grammar and certificate or replay checker. JSONL traces carry stable
candidate ids, ranking policy, score margins, environment identity, and semantic hashes so
reordered candidates or float ties cannot silently change registered claim meaning.

= Engine Substrate

The implementation has a lower-level substrate beneath the domain reports. A dataset-first
action-scoring runner consumes grouped candidate rows:

```text
split, group_id, state_hash, action_id, target, f0, ..., fN
```

Each group is one state with a complete candidate set. The target field is verifier-owned and
may mark more than one valid candidate, so equivalent macro steps, tied shortest paths,
corpus-valid continuations, or equal-utility actions need not be collapsed into an arbitrary
single gold row. Domain adapters own state generation, feature encoding, candidate
enumeration, and verifier labeling. The shared engine owns softmax ranking, trace emission,
checkpointing, numeric summaries, and common engine-evidence rows.

The row contract also has a scaling boundary. It assumes the adapter can present a bounded
candidate set for each state. Continuous-control, robotics, or multi-agent environments must
therefore add a discretizer, sampler, motion-primitive library, or replayable proposal
generator before they fit the runner. The guarantee is then over the enumerated candidate set
and declared support, not over every action in the unconstrained environment. If the adapter
cannot make that support explicit, the shared engine is instrumentation rather than a
verified-search guarantee.

The generated engine claim is:
#claim_C8_dataset_engine_evidence_surface_summary

#figure(
  claim_coverage_chart(
    claim_C8_dataset_engine_evidence_surface_chart_rows,
    y-max: 10,
    y-label: [rows],
  ),
  caption: [Dataset-first engine evidence surface. These rows prove that the registered smoke
    adapters can compile to one candidate-row protocol with replay/schema declarations; they
    do not replace richer domain-specific reports or promote broad task competence.],
) <fig:dataset-engine-surface>

The generic trace-replay path is intentionally weaker than a domain-lossless replay adapter.
When old traces do not store original state/action features, replay training uses trace-native
features such as logged score, probability, action id, candidate count, and rank. That is enough
to exercise stable action ids, multi-target labels, checkpointing, and numeric certificates.
It is not evidence that grid, ARC, haiku, Signal, or program-synthesis semantics can be
reconstructed from a score trace alone.

= Evidence Matrix

The matrix is the scope boundary. Every row answers the same questions: what produced
the candidates, what checked them, what did the model rank, what artifact proves the result,
and what claim is allowed? A row may support a claim only if those pieces and its limitation
are registered. New experiment families require revising this matrix first.

#figure(
  text(size: 7pt)[
    #table(
      columns: (1.4fr, 2.05fr, 1.5fr, 2.3fr),
      align: left,
      inset: 4pt,
      table.header[*Family*][*Grounding*][*Learned Role*][*Claim / Boundary*],
      ..experiment_matrix_rows.map(r => (
        [
          #strong[
            #if r.domain == "prime_gap" {
              [arithmetic #linebreak() sequence #linebreak() filtering]
            } else {
              [#r.domain_display]
            }
          ]
        ],
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
  caption: [Frozen evidence matrix. Each row states the checker, learned role, supported
    claim, and boundary condition.],
) <tab:frozen-evidence>

= Anchor: Arithmetic Sequence Filtering

The cleanest test case involves arithmetic sequence filtering. The model suggests
mathematical rules that filter out candidates known not to belong to a target sequence.
While the family extends to sequences such as Fibonacci and Lucas numbers through
Pisano-period residue gates, the strictly audited anchor here is a next-prime filter. A rule
is only accepted if an exact mathematical checker confirms that it never deletes the true
next-prime answer inside the declared support. The machine-readable trace id remains
`prime_gap`.

In formal terms, candidate actions are possible next-prime gaps after a segmented-sieve
window. Candidate filters are modular rejection predicates. A filter is accepted only if its
certificate has zero false rejections; the scorer then ranks the remaining candidates. The
`hand wheel` baseline is a deterministic hand-coded wheel-factorization filter over small
modular divisibility rules.

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
  caption: [Arithmetic sequence filtering on the canonical next-prime sample of
    #claim_C1_prime_gate_delta_sample_count rows: #claim_C1_prime_gate_delta_summary
    Learned-vs-hand top-1 delta: #pp(claim_C1_prime_gate_delta_learned_vs_hand_delta_pp).],
) <tab:prime-anchor>

The seed audit checks that the headline does not depend on one lucky training run:
#claim_C1_prime_gate_delta_ci_summary
The zero spread is a seed-stability audit over this deterministic benchmark: seeds vary
checkpoints and proposer runs, not the declared prime-range evaluation distribution.

This result is intentionally narrow. It shows that verified structure can shrink the search
space sharply while preserving correctness under the declared support. It does not show that
a static neural proposer beats deterministic symbolic ordering. The generated negative row
says the opposite:
#claim_C2_prime_proposer_fixed_budget_summary That failure is useful because it prevents the
claim set from becoming a generic "neural proposal wins" story.

The naive learned proposer fails. That is the point. Unsupported neural proposal is not
trusted; verifier feedback is recycled until proposal economics improve. In this protocol,
negative rows are not cleanup work after the fact. They are the training substrate that
separates model ranking from validity.

= Feedback Loops

The main contribution is what happens after the checker has said yes or no.

*Rejected-candidate learning.* When the verifier rejects a model suggestion, the system logs
exactly why it failed. Those failures are then recycled as training data, teaching the model
to rank future guesses better. In formal terms, failed rule proposals are logged with task,
split, seed, reason family, false-rejection counts, and marginal verifier utility, then used
as targets for a later ranker. The supported claim is scoped to arithmetic-sequence/modular sweeps:
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

*Certified hypothesis-library feedback.* If a generated rule is perfectly safe and verified
multiple times, it is saved as a reusable building block. The model can then use those saved
rules as proposal hints on future, unseen tasks. In formal terms, repeated
zero-false-rejection hypotheses are promoted into a reusable library, converted into slot
priors, and checked on held-out tasks under the same tight budget. The supported claim is:
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

Together these rows are the protocol's feedback mechanism. Rejections and certifications are not
after-the-fact diagnostics; they are reusable artifacts that change the next proposal pass.
They are also artifact-bounded, not an online self-modifying policy. Promotion uses frozen
artifact identity, split and seed labels, zero-false-rejection gates, held-out checks, and
semantic trace hashes. Continuous monitoring can rebuild summaries and limitation
candidates, but a changed checkpoint or trace snapshot creates a new artifact identity rather
than silently updating an existing claim. This keeps feedback useful while making policy
drift and rebuild cost visible in the evidence surface.

= Boundary Evidence

Boundary rows keep the thesis honest without becoming separate stories.

*ARC.* ARC is a provenance and grammar-coverage stress surface. The reported result is the current
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

*Signal.* To test physical control pressure, we use a simulator where bounded movement
actions have recorded consequences. The model's WASD choices are scored against replayed
simulation rows before live-control claims are admitted. This is the deterministic-replay
pressure test. The live `neural_active` row is narrower: a running local
Signal pilot applies checkpoint-ranked WASD proposals only through protocol checks and
safety gates, and those gates may veto execution instead of letting the neural top choice own
control. The trace records active source, checkpoint hash, latency, protocol compatibility,
safety-gate rows, damage/death telemetry, and replay-ready frames:
#claim_C9_signal_flight_replay_control_summary
#claim_C9_signal_live_neural_active_summary The supported claim set still does not show that active neural
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

*Architecture and language-adjacent probes.* Some rows ask how much the design depends on
model size or task-specific heads. Those are boundary checks, not architecture claims:
#claim_C4_multitask_capacity_tradeoff_summary #claim_C4_adapter_heads_shared_trunk_summary
For haiku, there is no exact checker for meaning. The system can only check whether a line
uses corpus-derived vocabulary, adjacency, syllable/shape constraints, and real-vs-corrupt
discrimination. We call that weaker checker `corpus_proxy`; it makes haiku a boundary probe,
not a semantic language-understanding result.

= Claim Governance

Empirical reporting is wired directly to experiment outputs. Result scripts write canonical
summaries, and the claim compiler turns registered source artifacts into Typst macros,
support-surface rows, and limitation candidates. The compilation layer uses those macros
rather than hand-maintained numbers. `make paper-support` refreshes support-owned audit
artifacts such as trace determinism and dataset-engine evidence before checking the claim
registry. It does not silently rerun the full promoted-result snapshot: canonical headline
rows such as `results/latest/comparison.canonical.jsonl` are refreshed by `make results` and
then checked by `make paper-support`. If an artifact is missing, stale relative to its
declared freshness prerequisite, contradicted, or underpowered, the manuscript compilation
layer will reject the registered claim as supported. The trace determinism row is currently
supported:
#claim_C8_trace_determinism_audit_summary
The dataset-engine row is governed the same way:
#claim_C8_dataset_engine_evidence_surface_summary
This does not mean every sentence is cryptographically enforced. It means the empirical
claims registered with the claim compiler must stay attached to live artifacts.

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
    `paper/generated/claims.typ`; the body uses counts and generated claim macros rather
    than forcing every long support row into the main text.],
) <tab:generated-claim-support>

Editorial agents fit the same rule. They may improve wording, structure, transitions, and
clarity through reviewable pull requests. They may not change claim scope, numeric results,
generated summaries, citations, or experiment interpretation unless they open an explicit
claim-change PR. The disclosure is operational: AI agents are assistants, while empirical
claims come from repository artifacts checked before compilation.

= Limitations

#set enum(numbering: "1.", spacing: 6pt)

+ *Scope is deliberately narrow.* The supported object is a checked search system: rules
  decide validity, neural models rank what remains, and claims stay tied to artifacts. The
  formal scope is verifier-grounded residual ranking and executable claim governance. It does
  not support broad task solving, broad ARC competence, semantic language understanding, or
  competitive theorem proving.
+ *Multi-seed intervals are still uneven.* The clean next-prime filter headline now has a generated
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
+ *Candidate-row scalability depends on bounded support.* Continuous action spaces and
  multi-agent settings need an explicit enumerator, sampler, or replayable proposal surface
  before the protocol applies. Without that surface, the engine can log and compare runs but
  cannot make verifier-grounded guarantees over the unconstrained action space.
+ *Feedback is artifact-bounded.* Rejected rows and certified hypotheses can train later
  rankers, but claim promotion depends on frozen traces, held-out checks, semantic hashes,
  and regenerated support reports. Online policy drift is evidence to audit, not a silent
  update to an existing claim.
+ *Engine ports are not domain promotions.* The dataset-first runner proves a shared row,
  replay, checkpoint, and evidence contract across smoke adapters. It does not replace the
  richer domain CLIs, rule searches, replay reports, or paper evidence needed to promote a
  domain result.
+ *Executable accountability is partial.* Claim compilation catches unsupported generated
  claims, stale evidence, and limitation candidates. It does not replace human judgment
  about interpretation, ethics, or whether a domain belongs in the supported claim set.

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
  caption: [Generated limitation candidates from the claim compiler. The required
    claim support gate is currently #paper_support_status. Optional contradicted claims
    remain visible here instead of blocking PDF compilation.],
) <tab:generated-limitations>

= Conclusion

The result is a disciplined loop. Enumerate candidates. Check which ones are allowed. Let a
small model rank the allowed choices. Preserve the trace. Bind empirical claims to the
generated evidence. Arithmetic sequence filtering supplies the clean certified-gate anchor,
with the next-prime trace as the fully audited instance. Rejected-candidate learning and
hypothesis-library promotion show that checked failures and checked successes can feed back
into later proposal structure. The dataset-first engine shows how future experiments can
enter the system as candidate rows plus verifier labels before becoming paper claims. ARC,
Signal, tiny-net, adapter, and haiku rows mark the boundary rather than widening the thesis.

The broader lesson is pragmatic. When correctness, safety, or auditability matter, the model
should not own the whole decision and prose should not outrun the evidence. The checker or
replay system owns validity and consequence. The learned component owns ranking pressure.
The repository trace owns what empirical reporting is allowed to say.

#bibliography("references.bib")
