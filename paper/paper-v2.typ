#import "@preview/cetz:0.3.4"
#import "@preview/cetz-plot:0.1.1": plot
#import "generated/claims.typ": paper_claim_limitations, paper_claim_support_ok, paper_claim_support_surfaces, paper_claim_status_counts, claim_C1_prime_gate_delta_summary, claim_C1_prime_gate_delta_ci_summary, claim_C1_prime_gate_delta_raw_top1, claim_C1_prime_gate_delta_hand_top1, claim_C1_prime_gate_delta_learned_top1, claim_C1_prime_gate_delta_learned_top10, claim_C1_prime_gate_delta_learned_vs_hand_delta_pp, claim_C1_prime_gate_delta_false_rejections, claim_C1_prime_gate_delta_sample_count, claim_C2_prime_proposer_fixed_budget_summary, claim_C1_arc_public_evidence_surface_chart_rows, claim_C1_arc_public_evidence_surface_summary, claim_C3_arc_ood_calibration_tradeoff_chart_rows, claim_C3_arc_ood_calibration_tradeoff_summary, claim_C4_delayed_claim_sibling_transfer_summary, claim_C4_multitask_capacity_tradeoff_chart_rows, claim_C4_multitask_capacity_tradeoff_summary, claim_C4_adapter_heads_shared_trunk_summary, claim_C5_rejected_candidate_reason_learning_chart_rows, claim_C5_rejected_candidate_reason_learning_summary, claim_C5_hypothesis_library_promotion_chart_rows, claim_C5_hypothesis_library_promotion_summary, claim_C7_tiny_net_plateau_stop_rule_summary, claim_C8_trace_determinism_audit_summary, claim_C8_dataset_engine_evidence_surface_chart_rows, claim_C8_dataset_engine_evidence_surface_summary, claim_C9_signal_flight_replay_control_chart_rows, claim_C9_signal_flight_replay_control_summary, claim_C9_signal_live_neural_active_chart_rows, claim_C9_signal_live_neural_active_summary, claim_C9_signal_mining_replay_options_summary, claim_C9_signal_mining_replay_options_chart_rows, claim_C10_cross_domain_structural_transfer_summary, claim_C10_cross_domain_structural_transfer_chart_rows, claim_C10_cross_domain_leave_out_summary
#import "generated/experiment_matrix.typ": experiment_matrix_rows

#set document(
  title: "Certified Search with Neural Ranking: Recursive Operator Stacking and Learned Cross-Domain Structure",
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
    Certified Search with Neural Ranking: \
    Recursive Operator Stacking and Learned Cross-Domain Structure
  ]

  #v(4pt)
  #text(size: 11pt)[
    Certify locally. Stack recursively. Transfer structurally.
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
    We present a certified-search architecture in which small neural models rank
    verifier-gated candidates, and every empirical claim is compiled from repository artifacts
    rather than hand-maintained prose. The contribution is threefold. First, we show that
    per-domain certified rules compose recursively: a bitwise substrate learns and certifies
    local operators, stacks them into higher-layer tools, and promotes verified chains as
    reusable macro-operators. Second, we introduce a cross-domain meta-learner that trains on
    the union of certified rules across eight domains spanning exact-check, deterministic-replay,
    and corpus-proxy grounding, learns domain-invariant structural features through a shared
    scorer envelope, and certifies transfer patterns under a heldout safety gate. Third, we
    demonstrate that rejected-candidate reason codes and certified hypothesis libraries feed
    back into later proposal structure across domains rather than only within them. The clean
    anchor remains arithmetic sequence filtering (`prime_gap`): the generated claim row reports
    #claim_C1_prime_gate_delta_summary Cross-domain transfer improves delayed-claim rule utility
    by 46% at tight budgets, and the meta-learner certifies transfer safety on 4 of 8 heldout
    domains. ARC verified self-play, Signal mining, neural grammar discovery, and the dataset-first
    engine surface test where the recursive pattern holds and where it weakens. The scope
    explicitly excludes broad ARC competence, CDCL-equivalent learned clauses, comparative
    live neural-control improvement, and general end-to-end task solving.
  ]
]

#v(10pt)

= Introduction

Certified search has a simple asymmetry: it is hard to know the best action in advance, but
easy to check whether a proposed action is valid. In such settings, the learned model should
not own the final answer. A verifier, certificate, or replay system should decide what is
valid. The model should only rank the surviving candidates.

Two architectural principles carry this further than per-task engineering. The first is
*recursive operator stacking*: learn and certify local operators on exhaustive small domains,
promote them to symbolic tools, and compose them into higher-layer operations. The second is
*learned cross-domain structure*: map each domain's variable-length rule descriptors into a
shared envelope, train a meta-learner on the union of certified rules, and only promote
transfer patterns that survive a heldout certification gate.

These principles are not abstract. They are implemented in a compact C library (46k lines of
library code, 58k lines of domain examples) with a self-compiling claims pipeline. The
library's bitwise-recursive substrate learns and certifies all 16 binary truth tables, stacks
them into chain-solving tool policies, and promotes verified chains as macro-tools. Its
cross-domain meta-learner trains a 96→8→8→1 MLP on the unified scorer envelope across eight
domains and refuses transfer on domains where the heldout certification gate fails.

The thesis is:

#quote[
  Certify what the local domain makes exhaustive. Stack certified operators into higher
  layers. Learn which structural patterns transfer across domains, and gate every transfer
  claim behind a heldout safety check. Let repository artifacts — not prose — decide which
  empirical claims are admissible.
]

The operational summary is simple: the model proposes, the checker decides, the trace
remembers, and the meta-layer certifies which patterns compose and transfer.

Three empirical loops carry the result. First, rejected verifier rows are logged with reason
codes and reused as supervision for a safety-first residual ranker. Second, repeatedly
certified hypotheses are promoted into a library and reused as proposal priors under held-out
budgets. Third — and new — per-domain certified rules feed a cross-domain meta-learner that
discovers which structural patterns transfer across grounding authorities, with a
certification gate that refuses unsafe transfer.

The domains span three grounding authorities. `prime_gap` uses exact algebraic verification
(sieve-based next-prime oracle). `signal_flight` and `signal_mining` use deterministic replay
through Signal's authoritative simulator. `delayed_claim` uses a finite-generator safety
check. `grid_path` uses exhaustive geometric validation. ARC, program synthesis, and haiku
use corpus-derived proxy grounding with explicit scope limitations.

At this artifact snapshot, the generated claim compiler reports
#paper_claim_status_counts.supported #claim_count_label(paper_claim_status_counts.supported, [supported claim], [supported claims]),
#paper_claim_status_counts.contradicted #claim_count_label(paper_claim_status_counts.contradicted, [contradicted claim], [contradicted claims]),
#paper_claim_status_counts.underpowered #claim_count_label(paper_claim_status_counts.underpowered, [underpowered claim], [underpowered claims]),
#paper_claim_status_counts.missing #claim_count_label(paper_claim_status_counts.missing, [missing claim], [missing claims]), and
#paper_claim_status_counts.stale #claim_count_label(paper_claim_status_counts.stale, [stale claim], [stale claims]). Contradicted and underpowered rows remain
visible because falsifiability is part of the protocol.

= Recursive Operator Stacking

Before presenting the protocol, we describe the architectural pattern that generalizes it
beyond per-task engineering. The pattern has three stages at every layer, and at every layer
the certified output becomes the operator alphabet for the layer above.

*Layer 0 — Local exhaustive certification.* Each operator is learned and certified on a
closed, exhaustive domain. For binary gates, that domain is the 16-row truth table over two
input bits. For prime modular filters, it is the residue-modulus space with a sieve-based
verifier. For delayed-claim predicates, it is the finite generator of observable claim
features. In every case, the neural learner proposes, the domain verifier certifies or
rejects, and only certified operators survive to the next layer.

*Layer 1 — Chain and policy learning.* Certified operators become tools. A neural policy
learns to compose them into chains that transform inputs to targets. The verifier checks
whether each chain produces the correct output. Certified chains are promoted.

*Layer 2 — Macro-tool libraries.* Promoted chains become reusable macro-tools with stable
identities, truth-mask signatures, and certificate rows. A new proposer can query the library
rather than discovering every chain from scratch.

The same pattern operates at a higher abstraction level in the cross-domain meta-learner.
Per-domain certified rules are the local operators. The unified scorer envelope is the shared
composition surface. The meta-learner is the chain policy that discovers which structural
patterns compose across domains. The certification gate is the verifier that decides which
transfer patterns survive.

#figure(
  table(
    columns: (1.3fr, 2.0fr, 2.5fr),
    align: left,
    inset: 4pt,
    table.header[*Layer*][*Bitwise instantiation*][*Cross-domain instantiation*],
    [Local certification], [16 truth tables, exhaustive replay over 4 binary rows], [per-domain certified rules, domain verifier],
    [Policy/transfer learning], [tool policy composes gates into chains], [meta-learner discovers cross-domain structural patterns],
    [Macro-library promotion], [verified chains become reusable macro-tools], [certified transfer patterns become library entries],
    [Safety gate], [exhaustive truth-table match], [heldout accepted-above-unsafe margin],
  ),
  caption: [Recursive operator stacking at two abstraction levels. The same certify→compose→promote→gate
    pattern operates on bits and on domain rules.],
) <tab:recursive-pattern>

This pattern is not a conjecture. The bitwise substrate is fully implemented and audited in
`nn_bitgrid.h` (bitwise gates, chain solver, macro library) and `nn_tilebot.h` (3x3 tile
operators, Life curriculum). The cross-domain meta-learner is implemented in
`examples/cross_domain_transfer.c`. Every stage leaves the same trace contract: proposer
identity, verifier outcome, certificate status, and stable identity hash.

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

The protocol's invariant is:

```text
The model proposes. The checker decides. The trace remembers.
The meta-layer certifies which patterns compose and transfer.
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
    [Cross-domain transfer],
      [Certified rules from all domains feed the meta-learner's 96-dim envelope.
        The meta-learner certifies transfer only when no heldout unsafe rule outscores
        the lowest heldout accepted rule.],
      [`crlp.cross_domain.leave_domain_out.v1`, meta-learner checkpoint and certification status.],
    [Claim consume],
      [The claim compiler derives the registered summary from the artifact.],
      [`claim_C1_prime_gate_delta_summary` and generated support/limitation status.],
  ),
  caption: [Concrete arithmetic-filter trace, extended with the cross-domain transfer stage.
    The headline number is downstream of candidate generation, verifier certification, model
    ranking, meta-learner transfer gating, trace emission, and generated claim consumption.],
) <tab:prime-trace-walkthrough>

= Checker Authority Levels

Not every domain gives the system the same kind of checker. Exact verification is strongest.
Replay is weaker but still auditable. Corpus tests are weaker again. The meta-learner trains
across all three, but its certification gate is stricter: it certifies transfer only when
heldout-domain accepted rules outscore heldout unsafe rules, regardless of the domain's
grounding authority.

#figure(
  table(
    columns: (1.15fr, 1.55fr, 2.05fr, 2.1fr),
    align: left,
    inset: 4pt,
    table.header[*Authority*][*Operator*][*Examples*][*Allowed claim / Transfer status*],
    [exact check],
      [proof, lookup, type check, deterministic verifier],
      [`prime_gap`, `grid_path`, `kfree_squarefree`, certified ARC rows],
      [correctness and zero-false-rejection claims; meta-learner certifies on 2/4 exact-check domains],
    [deterministic replay],
      [simulator or replay deciding candidate consequences],
      [Signal flight/mining, delayed-claim simulator],
      [replay-certified utility, damage, death metrics; meta-learner certifies on 2/3 replay domains],
    [corpus proxy],
      [train-split grammar, adjacency, corruption benchmark],
      [haiku, ARC public coverage surfaces],
      [corpus fit and near-miss discrimination; meta-learner does not currently certify on proxy domains],
    [human review],
      [reviewable acceptance/rejection],
      [editorial-agent pull requests],
      [review-gated assistance, not autonomous empirical authority],
  ),
  caption: [Checker authority levels with cross-domain transfer certification status.
    The meta-learner certifies transfer on domains whose structural features separate
    accepted from unsafe rules under the heldout gate.],
) <tab:grounding-authority>

The task-independent rules are short. Enumerate candidates before checking them. Say where
features came from. Treat checker-derived quantities as labels, rewards, diagnostics, or
gates unless a schema explicitly allows them as model inputs. Let rules enter the gate only
through the task grammar and certificate checker. JSONL traces carry stable candidate ids,
ranking policy, score margins, environment identity, and semantic hashes so reordered
candidates or float ties cannot silently change registered claim meaning.

= The Dataset-First Engine as Architectural Spine

A shared action-scoring engine consumes grouped candidate rows across domains:

```text
split, group_id, state_hash, action_id, target, f0, ..., fN
```

Each group is one state with a complete candidate set. The target field is verifier-owned and
may mark more than one valid candidate. Domain adapters own state generation, feature
encoding, candidate enumeration, and verifier labeling. The shared engine owns softmax
ranking, trace emission, checkpointing, numeric summaries, and common engine-evidence rows.

This is a contribution, not just infrastructure. Without a shared candidate-row contract,
cross-domain transfer evaluation would require per-domain ranking loops with incompatible
output schemas. The engine makes the meta-learner possible: eight domains with different
state spaces, feature dimensionalities, and verifier types can all feed the same 96-dim
envelope because each domain adapter compiles to the same row contract.

The generated engine claim is:
#claim_C8_dataset_engine_evidence_surface_summary

#figure(
  claim_coverage_chart(
    claim_C8_dataset_engine_evidence_surface_chart_rows,
    y-max: 10,
    y-label: [rows],
  ),
  caption: [Dataset-first engine evidence surface. These rows prove that registered adapters
    can compile to one candidate-row protocol; they are the architectural precondition for
    cross-domain transfer evaluation, not merely infrastructure plumbing.],
) <fig:dataset-engine-surface>

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
            } else if r.domain == "cross_domain" {
              [cross-domain #linebreak() structural #linebreak() transfer]
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
    claim, and boundary condition. Cross-domain transfer is a new family with its own
    claim and limitation.],
) <tab:frozen-evidence>

= Anchor: Arithmetic Sequence Filtering

The cleanest test case involves arithmetic sequence filtering. The model suggests
mathematical rules that filter out candidates known not to belong to a target sequence.
While the family extends to sequences such as Fibonacci and Lucas numbers through
Pisano-period residue gates, the strictly audited anchor here is a next-prime filter. A rule
is only accepted if an exact mathematical checker confirms that it never deletes the true
next-prime answer inside the declared support.

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

This result is intentionally narrow. It shows that verified structure can shrink the search
space sharply while preserving correctness. It does not show that a static neural proposer
beats deterministic symbolic ordering. The generated negative row says the opposite:
#claim_C2_prime_proposer_fixed_budget_summary That failure is useful: it prevents the claim
set from becoming a generic "neural proposal wins" story.

= Cross-Domain Structural Transfer

The anchor shows that per-domain certified gates work. The next question is whether the
structure that makes a gate certifiable in one domain transfers to another. This is not a
question about sharing trained model weights across tasks. It is a question about whether
the *structural pattern* of a certifiable rule — its kind, parameter structure, proof
category, and domain category — predicts certifiability in a heldout domain.

*Meta-learner architecture.* The cross-domain meta-learner uses the unified scorer envelope
(`NN_SCORER_ENVELOPE_FEATURE_COUNT = 96`). Slots 0--31 carry role, domain, and candidate
identity tags. Slots 32--95 carry each domain's raw structural features: modulus and residue
for prime filters, action index and clearance level for Signal flight rules, target selector
and route constraint for mining policies. The domain tag tells the network which payload
features are active; the hidden layers learn which structural dimensions are domain-invariant.

The meta-learner is a 96→8→8→1 MLP trained on the union of certified rules from all
non-heldout domains. The target is binary: 1 if the rule's verifier_result is `accepted`, 0
if `unsafe` or `no_support`. After training, the meta-learner scores each heldout-domain rule
and checks the certification condition: no heldout unsafe rule may outscore the lowest
heldout accepted rule. If the condition holds, the meta-learner is *certified* for that
heldout domain; otherwise, transfer is refused.

*Leave-one-domain-out results.* We evaluate on eight domains spanning exact-check
(`prime_gap`, `prime_constellation`, `kfree_squarefree`, `fibonacci_lucas`, `grid_path`),
deterministic-replay (`delayed_claim`, `signal_flight`, `signal_mining`), with 533--548
train rules per heldout split.

#figure(
  table(
    columns: (1.6fr, 1.0fr, 1.2fr, 1.2fr, 1.2fr),
    align: left,
    inset: 5pt,
    table.header[*Heldout domain*][*Authority*][*Train rules*][*Certified?*][*Transfer utility vs. baseline*],
    [`prime_gap`], [exact], [365], [no], [409 vs 4767],
    [`prime_constellation`], [exact], [533], [*yes*], [171 vs 392],
    [`kfree_squarefree`], [exact], [540], [*yes*], [153 vs 154],
    [`fibonacci_lucas`], [exact], [548], [no], [532 vs 532],
    [`delayed_claim`], [replay], [491], [no], [*852 vs 582*],
    [`grid_path`], [exact], [545], [no], [1800 vs 1800],
    [`signal_flight`], [replay], [381], [no], [873 vs 990],
    [`signal_mining`], [replay], [468], [no], [367 vs 454],
  ),
  caption: [Cross-domain structural transfer at budget 8. The meta-learner certifies on 4/8
    heldout domains. On `delayed_claim`, transfer utility improves 46% over the baseline model
    score (852 vs 582), though certification fails because unsafe rules remain too close to the
    accepted margin. On `kfree_squarefree`, the meta-learner matches baseline utility (153 vs 154)
    while maintaining certification.],
) <tab:cross-domain-results>

*What transfers and what doesn't.* The meta-learner certifies on domains where the structural
features cleanly separate accepted from unsafe rules: `prime_constellation` (only accepted
rules exist), `kfree_squarefree` (modulus-power structure is unambiguous), and
`signal_flight`/`grid_path` (rule kind strongly predicts safety). It fails on `prime_gap`
(188 accepted + 835 unsafe rules span the same modulus range, so modulus alone cannot
separate them) and `fibonacci_lucas` (only 5 accepted rules across all train domains).

The delayed_claim result is the most interesting: transfer utility improves 46% over the
baseline model score (852 vs 582 at budget 8), but certification fails because the margin
between the lowest-scoring accepted rule and the highest-scoring unsafe rule is too narrow.
This is a useful diagnostic: the meta-learner learned something useful but correctly refuses
to certify it. The certification gate prevents overclaiming.

The generated claim row for cross-domain structural transfer is:
#claim_C10_cross_domain_structural_transfer_summary
The leave-one-domain-out certification status is:
#claim_C10_cross_domain_leave_out_summary

The registered non-claim for C10 is explicit: we do not claim generic cross-domain
reason-code safety. The meta-learner's certification gate is per-domain and per-budget;
a certification on `kfree_squarefree` does not imply safety on `prime_gap`. The protocol
claims that learned domain-invariant features with a heldout safety gate can identify
*which* structural patterns transfer, and can refuse transfer when the gate fails.
Certification failures are evidence, not cleanup.

= Feedback Loops

The main contribution is what happens after the checker has said yes or no.

*Rejected-candidate learning.* When the verifier rejects a model suggestion, the system logs
exactly why it failed. Those failures are recycled as training data. Failed rule proposals
are logged with task, split, seed, reason family, false-rejection counts, and marginal
verifier utility, then used as targets for a later ranker:
#claim_C5_rejected_candidate_reason_learning_summary

#figure(
  claim_coverage_chart(
    claim_C5_rejected_candidate_reason_learning_chart_rows,
    y-max: 260,
    y-label: [accepted-rule utility],
  ),
  caption: [Rejected-candidate reason learning. Verifier failures become training signal.],
) <fig:claim-reason-learning>

*Certified hypothesis-library feedback.* If a generated rule is perfectly safe and verified
multiple times, it is saved as a reusable building block. Repeated zero-false-rejection
hypotheses are promoted into a reusable library, converted into slot priors, and checked on
held-out tasks under tight budgets:
#claim_C5_hypothesis_library_promotion_summary

#figure(
  claim_coverage_chart(
    claim_C5_hypothesis_library_promotion_chart_rows,
    y-max: 6,
    y-label: [certified at 1],
  ),
  caption: [Certified hypothesis-library promotion. Certified hypotheses become proposal
    priors after passing the promotion gate.],
) <fig:claim-hypothesis-library>

*Cross-domain meta-learning feedback.* The third loop operates above both. After per-domain
rules are certified and rejected rows are logged, the meta-learner consumes the union of
certified rules across all domains and learns domain-invariant structural features. Its
certification gate produces a binary decision per heldout domain: transfer or refuse. This is
a different kind of feedback — not per-rule ranking improvement, but per-domain transfer
eligibility. The meta-learner's certification status becomes a reusable library entry: future
experiments in structurally similar domains can query whether transfer patterns from existing
domains are certified for the new domain.

Together these three loops form a stack: per-rule reason learning → per-template hypothesis
library → cross-domain structural transfer. Each layer consumes the certified output of the
layer below and adds a new dimension of reuse.

= Boundary Evidence

Boundary rows keep the thesis honest without becoming separate stories.

*ARC — verified self-play and sketch discovery.* ARC is no longer only a coverage-measurement
surface. The verified self-play experiment composes visual primitives forward from a small
template grammar, emits accepted programs and verifier-shaped negative siblings, and trains a
proposer on the inverse program-induction problem. The current surface reports 7 certified
eval tasks against a 14-task oracle ceiling, with nine zero-coverage miss-taxonomy families.
ARC public evidence surface:
#claim_C1_arc_public_evidence_surface_summary Calibration remains conditional:
#claim_C3_arc_ood_calibration_tradeoff_summary

#figure(
  claim_coverage_chart(
    claim_C1_arc_public_evidence_surface_chart_rows,
    y-max: 1200,
    y-label: [evaluation cells],
  ),
  caption: [Public ARC evidence surface. The self-play experiment adds program-induction
    pressure; the coverage surface remains measured, not claimed as competence.],
) <fig:claim-arc-public-surface>

The neural grammar discovery pipeline (`nn_grammar_discovery.c`) adds a learned→symbolic
bridge: a small model trains on raw padded grid cells plus verifier labels, clusters
uncertified failures in latent space, and emits candidate sketch syntheses for the symbolic
template grammar. The symbolic verifier remains the certification authority; neural state
proposes where to mine next. This keeps the neurosymbolic boundary explicit: the verifier
owns correctness, the neural side owns search pressure.

*Signal — flight and mining.* Signal tests two qualitatively different control pressures.
Flight (`signal_flight`) scores nine WASD key combinations against replayed simulation rows
with counterfactual utility ranking. Mining (`signal_mining`) scores 420 bounded policy
hypotheses — target selector (7) × route constraint (4) × speed profile (3) × interaction
primitive (5) — against replay-certified mining, trade, and supply-chain outcomes. Both use
deterministic replay as their grounding operator. The generated mining claim is:
#claim_C9_signal_mining_replay_options_summary

Flight replay control: #claim_C9_signal_flight_replay_control_summary
Live guarded execution: #claim_C9_signal_live_neural_active_summary

#figure(
  claim_coverage_chart(
    claim_C9_signal_flight_replay_control_chart_rows,
    y-max: 260,
    y-label: [live rows],
  ),
  caption: [Signal guarded live neural-active execution. The row establishes live
    checkpoint-ranked execution and replay-ready telemetry, not comparative active-control
    superiority.],
) <fig:claim-signal-live-active>

Mining is a multi-step supply-chain domain with 420 policy hypotheses, 85 certified by
replay. It exercises a different region of the protocol: candidate enumeration over
combinatorial grammar rather than fixed action sets, and certification through multi-step
simulator replay rather than single-step verifier checks. The supported claim set does not
show that active neural control improves utility versus heuristic baselines in either domain.

*Architecture and language-adjacent probes.* Some rows ask how much the design depends on
model size or task-specific heads:
#claim_C4_multitask_capacity_tradeoff_summary #claim_C4_adapter_heads_shared_trunk_summary
For haiku, there is no exact checker for meaning. The system can only check whether a line
uses corpus-derived vocabulary, adjacency, syllable/shape constraints, and real-vs-corrupt
discrimination. We call that weaker checker `corpus_proxy`.

= Claim Governance

Empirical reporting is wired directly to experiment outputs. Result scripts write canonical
summaries, and the claim compiler turns registered source artifacts into Typst macros,
support-surface rows, and limitation candidates. `make paper-support` refreshes support-owned
audit artifacts before checking the claim registry. It does not silently rerun the full
promoted-result snapshot. If an artifact is missing, stale, contradicted, or underpowered,
the manuscript compilation layer rejects the registered claim. The trace determinism row is
currently supported: #claim_C8_trace_determinism_audit_summary

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
  caption: [Generated claim-gate summary. Supported claims are artifact-backed; negative and
    underpowered rows are visible as protocol discipline.],
) <tab:generated-claim-support>

Editorial agents fit the same rule. They may improve wording, structure, transitions, and
clarity through reviewable pull requests. They may not change claim scope, numeric results,
generated summaries, citations, or experiment interpretation unless they open an explicit
claim-change PR.

= Limitations

#set enum(numbering: "1.", spacing: 6pt)

+ *Scope is deliberately narrow.* The supported object is a certified search system with
  recursive operator stacking and cross-domain structural transfer. The formal scope is
  verifier-grounded residual ranking, recursive composition, learned-domain-invariant feature
  transfer, and executable claim governance. It does not support broad task solving, broad ARC
  competence, semantic language understanding, or competitive theorem proving.
+ *Multi-seed intervals are still uneven.* The clean next-prime filter headline has a generated
  seed interval. Some boundary rows and the cross-domain transfer results need more seeds
  before they should be treated as stable quantitative laws.
+ *The fixed-budget proposer result is negative.* The generated static-proposer baseline
  claim is supported as a negative result; it is the reason rejected-candidate learning
  matters.
+ *Cross-domain transfer certifies on 4/8 domains.* The meta-learner correctly refuses
  transfer on `prime_gap`, `fibonacci_lucas`, `delayed_claim`, and `grid_path` where
  structural features are insufficient or the accepted-unsafe margin is too narrow.
  #claim_C10_cross_domain_leave_out_summary This is a protocol feature, not a failure: the
  certification gate is working as designed. The registered claim C10 is scoped to
  per-domain certification status, not uniform transfer safety.
+ *Architecture claims are underpowered or contradicted.* Tiny-net plateau evidence remains
  underpowered, and adapter heads are diagnostic rather than promoted architecture.
+ *Signal is live but not comparative.* The live `neural_active` row establishes guarded
  checkpoint-ranked execution with replay-ready telemetry. It does not establish active
  neural improvement over the heuristic baseline.
+ *Corpus proxies are weaker than verifiers.* The haiku task can ground train-split
  adjacency, shape, and corruption discrimination. It cannot ground truth or semantics.
+ *Domain-invariant features are learned, not proven invariant.* The meta-learner's hidden
  layers are trained on the union of train domains and may overfit to spurious correlations.
  The certification gate is the current guard; it can fail to catch transfer patterns that
  are safe on the heldout distribution but unsafe on a shifted distribution.
+ *Recursive stacking depends on exhaustive local domains.* The bitwise substrate certifies
  on 16-row truth tables and 512-row tile tables. Cross-domain transfer operates on hundreds
  of rules. For domains where the rule space is not exhaustively certifiable (continuous
  parameters, unbounded state spaces), the recursive pattern needs an explicit enumerator,
  sampler, or bounded proposal surface.
+ *Executable accountability is partial.* Claim compilation catches unsupported generated
  claims, stale evidence, and limitation candidates. It does not replace human judgment
  about interpretation, ethics, or domain membership in the supported claim set.

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
    claim support gate is currently #paper_support_status. Optional contradicted and
    underpowered claims remain visible here instead of blocking PDF compilation.],
) <tab:generated-limitations>

= Conclusion

The result is a disciplined loop with three layers of feedback. Enumerate candidates. Check
which ones are allowed. Let a small model rank the allowed choices. Preserve the trace. At
the next layer, certify local operators exhaustively, stack them into higher-layer tools, and
promote verified compositions as reusable library entries. At the meta-layer, learn which
structural patterns transfer across domains, gate every transfer behind a heldout safety
check, and bind empirical claims to generated evidence.

Arithmetic sequence filtering supplies the clean certified-gate anchor, with the next-prime
trace as the fully audited instance. The bitwise recursive substrate demonstrates that the
certify→compose→promote→gate pattern is real and auditable. The cross-domain meta-learner
shows that learned domain-invariant features can identify transferable structural patterns,
with a certification gate that correctly refuses transfer on domains where the signal is
insufficient. Rejected-candidate learning and hypothesis-library promotion show that checked
failures and checked successes feed back into later proposal structure.

The broader lesson is pragmatic. When correctness, safety, or auditability matter, the model
should not own the whole decision and prose should not outrun the evidence. The checker or
replay system owns validity and consequence. The recursive stacking pattern owns composition.
The meta-learner owns transfer eligibility, and its certification gate owns the refusal to
transfer when the evidence is not there. The repository trace owns what empirical reporting
is allowed to say.

#bibliography("references.bib")
