#import "@preview/cetz:0.3.4"
#import "@preview/cetz-plot:0.1.1": plot, chart
#import "data.typ": learning_curves, gate_ablation, arc_epoch_eval, prime_gap_epoch_loss, multitask_shared_vs_single, multitask_rmt1, multitask_rmt2_aggregate, multitask_rmt6_capacity
#import "generated/claims.typ": paper_claim_limitations, paper_claim_support_ok, paper_claim_support_surfaces, claim_C1_arc_public_evidence_surface_chart_rows, claim_C1_arc_public_evidence_surface_summary, claim_C3_arc_ood_calibration_tradeoff_chart_rows, claim_C3_arc_ood_calibration_tradeoff_summary, claim_C4_delayed_claim_sibling_transfer_chart_rows, claim_C4_delayed_claim_sibling_transfer_summary, claim_C4_multitask_capacity_tradeoff_chart_rows, claim_C4_multitask_capacity_tradeoff_summary, claim_C5_rejected_candidate_reason_learning_chart_rows, claim_C5_rejected_candidate_reason_learning_summary, claim_C5_hypothesis_library_promotion_chart_rows, claim_C5_hypothesis_library_promotion_summary, claim_C9_signal_flight_replay_control_chart_rows, claim_C9_signal_flight_replay_control_summary, claim_C9_signal_live_neural_active_chart_rows, claim_C9_signal_live_neural_active_summary
#import "generated/experiment_matrix.typ": experiment_matrix_rows

#set document(
  title: "Certificate-Gated Neural Proposers for Verifier-Grounded Action Scoring",
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
  let sign = if v >= 0 { "+" } else { "−" }
  let mag = calc.round(calc.abs(v) * 100) / 100
  [#sign#mag pp]
}

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

#let claim_coverage_chart(rows, y-max: 240, y-label: [Coverage]) = cetz.canvas({
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
    Certificate-Gated Neural Proposers for \
    Verifier-Grounded Action Scoring
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
    This paper studies a narrow architecture for settings with exact verifiers: symbolic or
    replay-certified mechanisms define which candidates are admissible, and small neural
    components rank, calibrate, reuse, or learn around those certified hypotheses. The main
    contribution is not that tiny networks solve hard tasks. It is an artifact-feedback loop:
    rejected verifier rows become reason-coded training signal for later proposers, and
    repeated zero-false-rejection hypotheses become reusable library priors. The frozen
    evidence matrix covers certified modular gates, grid-path and delayed-claim stress
    surfaces, ARC-AGI transform coverage, a tiny-net capacity audit, and Signal replay
    control. The strongest gate anchor is `prime_gap`: with residue-sieve features removed,
    the raw scorer reaches 9.13% top-1, a fixed hand-wheel gate reaches 45.12%, and a
    certified learned-rule gate reaches 68.87% at 100% top-10. The more important substrate
    rows are C5: rejected-candidate residuals repair unsafe fixed-budget proposer rankings,
    and certified hypothesis-library priors improve tight-budget proposal coverage under a
    held-out split. ARC is an evidence surface for priors and failure taxonomy; Signal is
    application pressure under replay, shadow logging, and a guarded live-active pilot. The
    paper explicitly does not claim broad ARC competence, comparative live neural-control
    improvement, or general end-to-end task solving.
  ]
]

#v(10pt)

= Introduction

Exact verifiers are common in domains where action correctness can be checked algorithmically
even when discovery cannot: theorem proving, program synthesis, combinatorial search, formal
shortest-path problems. The dominant style in modern learned policy work treats the verifier as
a reward signal for an end-to-end neural policy. We argue this is the wrong default when the
candidate space contains many structurally impossible actions that the verifier can cheaply
eliminate.

Our thesis is the *certificate-gating loop*: a learned scorer should consume the *survivors* of
a verifier-checked symbolic filter, and the filter itself can be produced by a rule proposer
gated by a soundness-certificate verifier. The scorer ranks the residual; the gate eliminates
the rest; nothing the gate emits can violate correctness because every rule it accepts carries
a deterministic proof.

The paper's experiment-family set is now frozen. New domains and new benchmark families are
roadmap items, not body evidence, until the matrix in @tab:frozen-evidence is deliberately
revised. The frozen families are: certified modular gates, rejected-candidate learning,
certified hypothesis-library feedback, tiny-net capacity auditing, ARC evidence surfaces, and
Signal replay pressure. Grid-path and delayed-claim remain supporting stress surfaces because
they test whether the same candidate/verifier/scorer/trace contract survives outside the
prime-gap anchor.

The two main empirical contributions are:

#set enum(numbering: "(M1)", spacing: 6pt)
+ *Rejected-candidate learning.* Failed verifier proposals are not discarded. They are logged
  with reason codes, false-rejection counters, split keys, seed intervals, and marginal
  verifier utility, then reused as targets for a safety-first residual ranker. This repairs
  the fixed-budget modular proposer failure without giving the neural model authority over
  correctness.
+ *Held-out certified-library feedback.* Repeated zero-false-rejection hypotheses are promoted
  into a library, converted into slot priors, and evaluated on held-out tasks under the same
  verifier budget. This is the cumulative-learning direction: certified patterns become
  reusable proposal structure rather than one-off fixture wins.

#set enum(numbering: "1.", spacing: auto)

The older axis-style claims remain in the paper as support and boundary evidence. Certified
gates reduce the residual action space; the fixed-budget neural proposer result is negative
and becomes a limitation; verifier-RL calibration is a prime-gap-only support row; tiny-net
capacity results constrain architecture claims. Those axes are not new experiment families.
They are checks on the same loop.

Operationally, the certificate-gating loop yields an artifact-feedback mechanism that recycles
verifier outputs into future gates, priors, and ranking targets (@tab:artifact-feedback), and a
generalized task contract that keeps the same candidate/verifier/scorer/trace interface across
domains from arithmetic to replay-grounded control (@tab:schema).

The evidence is heterogeneous by design, so the paper uses a single protocol matrix rather
than a single leaderboard. The primary question in every row is not "how smart is the small
network?", but "what does the environment supply, what does the verifier own, what is the
network allowed to rank, and what reusable artifact is produced?"

#figure(
  text(size: 5.9pt)[
    #table(
      columns: (0.82fr, 1.25fr, 1.18fr, 1.22fr, 1.12fr, 1.15fr, 1.18fr, 1.05fr, 1.18fr),
      align: left,
      table.header[*Family*][*Candidate Set*][*Verifier*][*Certificate / Gate*][*Scorer Role*][*Trace Artifact*][*Result*][*Claim*][*Limitation*],
      ..experiment_matrix_rows.map(r => (
        [#raw(r.domain)],
        [#r.candidate_set],
        [#r.verifier],
        [#r.certificate_gate],
        [#r.scorer_role],
        [#r.trace_artifact],
        [#r.result],
        [#r.claim],
        [#r.limitation],
      )).flatten()
    )
  ],
  caption: [Frozen evidence matrix. Every result row uses the same structure: domain/data,
    grammar, candidate set, verifier/certificate, scorer role, trace artifact, result,
    takeaway/claim, and limitation. The paper may tighten these rows, but it should not add
    new experiment families without revising this matrix first.],
) <tab:frozen-evidence>

*Non-claims.* The paper does not claim general task solving, competitive ARC performance,
comparative live neural-control improvement, or that small networks replace scale. It does not claim that a generic
reason-code ranker is safe across domains, or that wider MLPs solve the ARC geometry failure.
It claims a narrower protocol result: when certified candidate mechanisms own admissibility,
small learned components can usefully rank residual candidates, calibrate decisions, reuse
certified hypotheses, and learn from verifier rejections under an auditable trace contract.

The artifact now emits `crlp.experiment_schema.v1` rows for every certified task with `make
experiment-schemas`, plus shared protocol rows for candidate rows, verifier outcomes, ranker
decisions, discovered rules, and library promotions. The paper should be read against that
schema: task-specific verifiers and grammars change, but candidate row shape, verifier-outcome
shape, leakage discipline, proposer baselines, trace replay, and verifier-RL comparisons are
shared.

= Related Work

== Verifier-guided learning and process supervision

DeepSeek-R1 is the closest contemporary confirmation of the verifier-first premise at
frontier scale: DeepSeek-AI report reinforcement learning from verifiable rewards (RLVR),
where math, code, and format checks provide correctness signals without a learned reward
model @deepseek2025r1. That supports the architectural direction of this paper, but it is
not the same artifact. DeepSeek-R1 couples verifiable rewards to massive language-model
sampling and opaque training infrastructure; this work keeps the verifier line auditable by
putting candidate enumeration, certificate checks, accepted rules, rejected proposals, and
scorer metrics into schema-attested C11 traces. The contrast with the standard learned-reward
line @christiano2017deep @stiennon2020learning @ouyang2022training is therefore not "RL is
unnecessary"; it is narrower: when an exact verifier exists, learned components should rank
or propose within a verifier-bounded search, not replace the correctness oracle.

The learned-verifier line starts from a different point. GSM8K-style verifier-guided
decoding trains a model to rank sampled solutions after generation @cobbe2021training;
process-supervision work trains reward models on intermediate reasoning steps rather than
only final outcomes @lightman2023verify @uesato2022solving; V-STaR adds a self-training
loop in which both correct and incorrect sampled solutions improve the verifier
@hosseini2024vstar. Those systems are natural relatives because they all use a learned
component to reduce search over candidate reasoning traces. The boundary in this paper is
that the verifier is not learned: verifier outputs may provide labels, rewards,
certificates, and diagnostics, but accepted gates require deterministic certificates and
scorer inputs are checked against a typed leakage policy. In short, learned verifiers ask
"which sampled answer should we trust?"; this work asks "which candidates remain after a
sound filter, and how should we rank that residual?"

The regularization used in Axis 3 is also borrowed, not claimed as novel. RLHF systems made
KL-to-prior and entropy-style stabilization standard for preference rewards
@stiennon2020learning @ouyang2022training, with PPO and trust-region policy optimization as
the broader policy-gradient background @schulman2017proximal @schulman2015trust. Our
contribution is to show the same stabilization pattern under exact verifier rewards: strict
policy-gradient updates inflate confidence on the prime-gap modular gate instance, while the KL/entropy branch
preserves ranking and improves calibration. Calibration itself is reported in the standard
ECE/temperature-scaling frame of @guo2017calibration.

== Neuro-symbolic search with formal checkers

AlphaGeometry is the closest structural ancestor of this work. It interleaves a neural
language model that proposes auxiliary geometric constructions with a symbolic deduction
engine that verifies whether the construction completes the proof @trinh2024alphageometry;
AlphaGeometry2 extends that line to stronger olympiad-geometry performance
@chervonyi2025alphageometry2. GPT-f is the earlier theorem-proving version of the same
pattern: a language model proposes proof steps and a formal system checks them
@polu2020generative. Our architecture is deliberately smaller and more explicit. The
grammar is task-local, the proposer sees structural rule descriptors rather than a full
natural-language proof context, and every accepted rule is emitted into a trace with
support, marginal utility, certificate status, and false-rejection counts. The comparison is
therefore not scale; it is audit posture. Their systems show that neural proposal plus
formal checking is powerful at large scale; this paper asks how much of that pattern can be
made schema-attested, reproducible, and heterogeneous across small verifier-equipped tasks.

== Program synthesis and library reuse

DreamCoder and Hypothesis Search are the closest relatives of the hypothesis layer rather
than the modular rule gate. DreamCoder learns reusable DSL abstractions through
wake-sleep Bayesian program learning @ellis2021dreamcoder; Hypothesis Search samples
natural-language hypotheses, compiles them into executable programs, and keeps the
train-consistent ones @wang2024hypothesis. `NNHypothesis` sits between those two poles:
templates and slot types are deterministic, but bindings are induced from task evidence and
compiled into canonical `NNProgram` rows that the same certificate/proposer machinery can
score. This is why the ARC and `program_synthesis` evidence rows matter: they show the
substrate's "hypothesis -> program -> certificate -> proposer summary" path outside the
current modular arithmetic grammar.

The library-feedback claim is narrower than open-ended program induction. A binding enters
the library only after a verifier accepts it with zero false rejections, and the promoted
binding is evaluated as a prior under a held-out budget. This makes the connection to
program-synthesis systems explicit without turning the paper into a new DSL benchmark.

== ARC abstraction benchmarks

ARC is used here as one hard abstraction substrate in the evidence portfolio. Chollet's ARC
framing emphasizes skill-acquisition efficiency and the need to account for priors
@chollet2019measure. The 2024 ARC Prize report and recent test-time-training work show the
current leaderboard direction: large-model or program-synthesis systems with per-instance
adaptation achieve far higher ARC scores than this artifact @chollet2024arcprize
@akyurek2024surprising. Our public ARC result should therefore be read narrowly:
#claim_C1_arc_public_evidence_surface_summary The fixed `0520fde7` transform gate gives a
clean scorer-vs-certificate contrast on one task. The research claim is not ARC
performance; it is that feature provenance, certificate-gated transforms, matched
proposer budgets, and replayable evidence rows make the priors and failure modes visible.

= Method

== The Certificate-Gating Loop

The reusable architecture is:

#figure(
  cetz.canvas(length: 1cm, {
    import cetz.draw: *
    let col_w = 2.5
    let gap = 0.55
    let step = col_w + gap
    let xs = (0, step, 2 * step, 3 * step)
    let y_top = 1.6
    let y_bot = -1.6
    let box(x, y, w, h, label, ..opts) = {
      rect((x - w/2, y - h/2), (x + w/2, y + h/2), ..opts)
      content((x, y), text(size: 8.5pt, label))
    }
    let h = 1.1
    box(xs.at(0), y_top, col_w, h, [state], stroke: 0.7pt)
    box(xs.at(1), y_top, col_w, h, align(center)[rule\ proposer], fill: rgb("#e8f0ff"), stroke: 0.7pt)
    box(xs.at(2), y_top, col_w, h, align(center)[certificate\ verifier], fill: rgb("#dfe9d6"), stroke: 0.7pt)
    box(xs.at(3), y_top, col_w, h, align(center)[accepted\ rules], stroke: 0.7pt)

    box(xs.at(0), y_bot, col_w, h, [state], stroke: 0.7pt)
    box(xs.at(1), y_bot, col_w, h, align(center)[enumerate\ candidates], stroke: 0.7pt)
    box(xs.at(2), y_bot, col_w, h, align(center)[gated\ candidates], stroke: 0.7pt)
    box(xs.at(3), y_bot, col_w, h, align(center)[learned\ scorer], fill: rgb("#e8f0ff"), stroke: 0.7pt)
    let x_softmax = 4 * step
    box(x_softmax, y_bot, col_w, h, align(center)[softmax\ + rank], stroke: 0.7pt)

    let half = col_w / 2
    let pad = 0.18
    for i in range(3) {
      line((xs.at(i) + half + pad, y_top), (xs.at(i + 1) - half - pad, y_top), mark: (end: ">"))
      line((xs.at(i) + half + pad, y_bot), (xs.at(i + 1) - half - pad, y_bot), mark: (end: ">"))
    }
    line((xs.at(3) + half + pad, y_bot), (x_softmax - half - pad, y_bot), mark: (end: ">"))

    line((xs.at(3), y_top - 0.45), (xs.at(2), y_bot + 0.45),
      stroke: (paint: rgb("#7a5b00"), thickness: 0.8pt, dash: "dashed"),
      mark: (end: ">"))
    content((xs.at(3) + 0.7, 0),
      anchor: "west",
      text(size: 7.5pt, fill: rgb("#7a5b00"), [rules gate the\ candidate set]))

    line((x_softmax, y_bot - 0.45), (x_softmax, y_bot - 1.05), mark: (end: ">"))
    content((x_softmax / 2, y_bot - 1.45),
      text(size: 8.5pt, [exact verifier $arrow.r$ correct action / cost / JSONL trace]))
  }),
  caption: [The certificate-gating loop. A rule proposer (top) produces candidate rules
    that a soundness-certificate verifier accepts or rejects. Accepted rules deterministically
    prune the candidate set; a shared neural scorer ranks the survivors; the exact verifier
    supplies the correctness signal.],
) <fig:loop>

The design invariant is *the model proposes, the verifier decides, the trace remembers.* A rule
$r$ is accepted only if a deterministic algebraic check certifies that $r$ cannot reject the
true action on any state in the support set, so soundness does not depend on the proposer being
well-trained.

#figure(
  text(size: 8.2pt)[
    #table(
      columns: (1.05fr, 1.15fr, 1.15fr, 1.25fr),
      align: left,
      table.header[*Verifier Output*][*Trace Artifact*][*Reusable Form*][*Next Ranking Use*],
      [accepted rule],
        [rule trace with support, certificate status, and false rejections],
        [gate program],
        [prune impossible candidates before scoring],
      [rejected candidate],
        [reason-coded verifier rejection row],
        [unsafe / no-support / duplicate prior],
        [avoid low-utility proposals at fixed budget],
      [certified hypothesis],
        [typed program evidence row],
        [library prior over template slots],
        [promote reusable transforms under a verifier budget],
      [replay outcome],
        [counterfactual utility, damage, and death row],
        [bounded action utility target],
        [rank control candidates without owning safety],
    )
  ],
  caption: [Artifact feedback loop. The paper's central mechanism is not raw neural
    accuracy. The environment produces certified or replayed artifacts; those artifacts
    become gates, priors, and ranking targets for the next proposal pass.],
) <tab:artifact-feedback>

== Generalized Task Schema

Every verifier-backed experiment in the artifact emits the same schema row:

```text
Task T = {
  state distribution S_T,
  candidate enumerator A_T(s),
  observable features x_T(s, a),
  neural residual scorer f_theta(x_T),
  exact verifier V_T(s, a),
  symbolic grammar G_T,
  certificate checker C_T(rule),
  rule proposer q_phi(rule | structural(rule)),
  gate executor R_T(rule, s, a),
  trace/replay/RL protocol
}
```

#figure(
  text(size: 7.2pt)[
    #table(
      columns: (auto, auto, auto, auto, auto, auto, auto),
      align: left,
      table.header[*Task*][*Candidate Set*][*Verifier*][*Certificate / Gate*][*Scorer Role*][*Trace Artifact*][*Protocol Point*],
      [`prime_gap`],
        [candidate gaps + modular rule candidates],
        [segmented sieve next-prime verifier],
        [divisibility certificate; zero false rejection],
        [rank residual gaps and budgeted rules],
        [`crlp.compare.v1`, `crlp.rule_trace.v1`],
        [certified modular gate search anchor],
      [`grid_path`],
        [four moves + legality/distance rules],
        [BFS shortest-path successor],
        [move cannot reject optimal successor],
        [rank legal residual moves],
        [`crlp.compare.v1`],
        [same gate contract outside prime residues],
      [`delayed_claim`],
        [trust / ignore / request / cautious],
        [deterministic hidden-outcome simulator],
        [finite-generator safety check],
        [rank actions under delayed labels],
        [`crlp.trace.v1`, replay rows],
        [trace replay plus production threshold+margin gate],
      [`arc_agi`],
        [cell colors + transform/program candidates],
        [held-out grid and train-pair transform check],
        [train consistency + unique compiled prediction],
        [rank cells, programs, and hypothesis-derived transforms],
        [`crlp.arc_transform.summary.v1`, `crlp.hypothesis.evidence.v1`],
        [abstraction task exposes priors and grammar coverage],
      [`program_synthesis`],
        [list/int DSL abstractions],
        [run candidate abstraction on examples],
        [monomorphic DSL type check],
        [rank reusable typed programs],
        [`crlp.hypothesis.evidence.v1`],
        [same hypothesis → program → certificate path beyond ARC],
      [`signal_flight`],
        [nine WASD control candidates],
        [authoritative Signal simulator replay + live active telemetry],
        [counterfactual utility + damage/death replay + safety-gate log],
        [rank bounded real-time control actions],
        [`signal.flight_counterfactual.v1`, `crlp.signal_flight.summary.v1`, `crlp.signal_live_active_summary.v1`],
        [application vignette for replay-grounded control],
    )
  ],
  caption: [The task instantiations share the same verifier-gated
    action-scoring contract. Each row repeats the same question: what is the
    candidate set, which checker owns correctness, what becomes a certificate,
    what the learned model is allowed to rank, which trace records the decision,
    and what the task proves about the protocol.],
) <tab:schema>

The leakage rule is shared: verifier outputs may be labels, rewards, certificates, diagnostics,
or executed gates, but they are not default scorer inputs. Proposer inputs are structural rule
descriptors; support-derived marginal utility labels and false-rejection diagnostics are targets,
not features. In the
artifact this is a typed contract, not just prose: every scorer feature declares provenance
(`state_observation`, `structural_prior`, `verifier_derived`, or `derived_from_gate`), every
task declares a leakage policy (`forbidden`, `priors_allowed`, or `legacy`), and schema emission
fails if enabled scorer features violate that policy.

== Trace and Checkpoint Contract

Every step writes a JSONL row with schema `crlp.trace.v1`, including a stable per-action
identifier, the candidate scores and probabilities, the selected and target indices, the
verifier result, and verifier cost. Replay resolves `target_index` $arrow.r$ stable `action_id`
$arrow.r$ regenerated candidate row, so reordering action ids cannot silently change replay
semantics. Checkpoints carry SHA-256 model identity, a `parent_model_hash`, a
`feature_encoder_version`, and a `feature_set`; comparison reports (`crlp.compare.v1`) carry
both hashes, the evaluated `feature_set`, and machine-readable reliability bins.

= Task Family: Certified Modular Gate Search

The prime-gap task is the current production instantiation of certified modular gate search.
The family is defined by the proof shape, not by primes: enumerate arithmetic candidates,
propose residue/period/divisibility rules from structural descriptors, accept only rules with
deterministic certificates, gate impossible candidates, and rank the residual with the learned
scorer. Prime constellations, k-free or polynomial-value filters, and Fibonacci/Lucas
recurrence gates are natural next members because they preserve that modular certificate
contract while changing the arithmetic substrate.

The artifact now includes a smoke implementation of those next members in
`examples/modular_gate_search.c`. It emits `crlp.modular_gate.summary.v1` rows for
`prime_constellation`, `kfree_squarefree`, and `fibonacci_lucas` under matched rule budgets.
Each task reports `grammar_order`, `random`, `structural_heuristic`, `learned_structural`,
`train_marginal_oracle`, and `validation_marginal_oracle` rows. The learned proposer is a
structural-only linear model trained on train-set rule coverage labels, and the smoke requires
zero held-out false rejections. These rows defend the family boundary; they do not yet replace
the production `prime_gap` learned-gate table.

== Prime-Gap Setup

The state is a prime $p$; the candidate space is gaps $g in {2, 4, ..., 256}$ plus an overflow
action; the verifier returns the true next-prime gap. The default candidate encoding is the
minimal feature set $[log p, "norm_gap", "overflow_flag"]$. A full-sieve feature set that adds
precomputed divisibility flags for $p + g$ modulo small primes is retained only as a leakage
ablation. The scorer is a shared MLP returning one scalar per candidate; softmax over
candidates produces the proposal distribution.

The default split is $p < 800,000$ train, $800,000 lt.eq p < 1,000,000$ validation,
$1,000,000 lt.eq p lt.eq 3,000,000$ test; all reported numbers use 30,000 held-out test
states unless stated otherwise.

== Why this domain

- A fast deterministic verifier at all scales (segmented sieves, wheel filters, block proofs).
- A compact totally-ordered candidate space.
- An exhausted classical literature \u{2014} we cannot beat a sieve, which sharpens the question
  to "what does the learned component contribute under perfect feedback?"
- Rich algebraic structure for the rule grammar (residue classes, divisibility certificates).

= Axis 1: Certified gates reduce the residual action space

The same trained prime-gap scorer inside the certified modular gate instance is evaluated under
three gate conditions on the canonical
30,000-sample held-out benchmark. Top-1 results:

#figure(
  cetz.canvas({
    import cetz.draw: *
    let bars = (
      (label: "no gate", value: 9.13, color: rgb("#a0a0a0")),
      (label: "hand-wheel", value: 45.12, color: rgb("#5b8def")),
      (label: "learned wheel", value: 68.87, color: rgb("#2ca02c")),
    )
    plot.plot(
      size: (9, 4.2),
      x-label: none,
      y-label: [Top-1 accuracy (%)],
      y-min: 0, y-max: 80,
      y-tick-step: 10,
      x-min: -0.5, x-max: 2.5,
      x-tick-step: none,
      x-ticks: ((0, [no gate]), (1, [hand-wheel]), (2, [learned wheel])),
      legend: none,
      {
        for (i, b) in bars.enumerate() {
          plot.add-bar(
            ((i, b.value),),
            bar-width: 0.6,
            style: (fill: b.color, stroke: black + 0.5pt),
          )
        }
      },
    )
  }),
  caption: [Top-1 accuracy by gate on the canonical 30,000-sample held-out benchmark. The
    clean raw-to-learned delta is *9.13 $arrow.r$ 68.87*. The fixed-hand-wheel-to-learned
    delta is *45.12 $arrow.r$ 68.87* (+23.8pp). The former 45.12% raw-scorer result is now a
    full-sieve-feature leakage ablation, not the headline.],
) <fig:f1>

Full metrics:

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    table.header[*Gate*][*Top-1*][*Top-10*][*NLL*][*ECE*],
    [no gate], [9.13%], [78.01%], [2.9994], [0.1076],
    [hand-wheel], [45.12%], [99.77%], [1.5902], [0.0834],
    [*learned wheel*], [*68.87%*], [*100.00%*], [*1.0056*], [0.1402],
  ),
  caption: [Gate ablation, minimal-feature scorer, identical weights across rows. Timing
    fields are excluded from the canonical comparison because they vary by runner; the
    numerical claim here is the accuracy and calibration effect of certified pruning.],
) <tab:f1>

= Axis 2: Rule search must beat fixed-budget symbolic baselines

The canonical rule grammar is `reject candidate n when n > m and n % m == r`; the certificate
verifier accepts a rule iff $gcd(m, r) > 1$ (so $n$ is necessarily composite). The matched-budget
baseline table also reports conditional gap filters (`if p % m == a then reject gap % m == b`)
because that grammar is less saturated and exposes proposer failure more clearly. Zero false
rejections is a structural property of the certificate. The proposer input is
structural-only: condition kind, moduli, reciprocal scales, residues, zero-residue indicators,
gcd ratios, and derived candidate residue. Empirical support and false-rejection counts are
diagnostics, and budgeted marginal utility is the neural target, not a proposer feature.

#figure(
  cetz.canvas(length: 1cm, {
    import cetz.draw: *
    let stages = (
      (label: "proposed",         value: 32637, color: rgb("#a0a0a0")),
      (label: "verified",         value: 4692,  color: rgb("#88b0e8")),
      (label: "accepted",         value: 53,    color: rgb("#2ca02c")),
      (label: "false\nrejections", value: 0,    color: rgb("#d62728")),
    )
    let plot_w = 12.0
    let plot_h = 5.0
    let x_origin = 0.0
    let y_origin = 0.0
    let col_w = plot_w / stages.len()
    let max_log = calc.log(32637, base: 10)
    let scale(v) = if v <= 0 { 0.0 } else {
      calc.max(0.0, calc.log(v, base: 10)) / max_log * plot_h
    }
    line((x_origin, y_origin), (x_origin + plot_w, y_origin), stroke: 0.7pt)
    line((x_origin, y_origin), (x_origin, y_origin + plot_h + 0.4), stroke: 0.7pt)
    content((x_origin - 0.9, y_origin + plot_h / 2),
      angle: 90deg, text(size: 9pt, [Rule count (log scale)]))
    for (i, s) in stages.enumerate() {
      let cx = x_origin + (i + 0.5) * col_w
      let bw = col_w * 0.55
      let h = scale(s.value)
      if h > 0.05 {
        rect((cx - bw / 2, y_origin), (cx + bw / 2, y_origin + h),
          fill: s.color, stroke: black + 0.5pt)
      } else {
        line((cx - bw / 2, y_origin), (cx + bw / 2, y_origin),
          stroke: s.color + 2.5pt)
      }
      content((cx, y_origin + h + 0.35),
        text(size: 9pt, weight: "bold", str(s.value)))
      content((cx, y_origin - 0.45), text(size: 9pt, s.label))
    }
  }),
  caption: [Rule-search acceptance funnel. Of 32,637 proposed rules, 4,692 pass the soundness
    certificate, 53 survive support and false-rejection screens, and *zero* falsely reject the
    true next-prime gap. The accepted set recovers 79.79% of wheel-equivalent verifier work.],
) <fig:f2>

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, left, right, right, right),
    table.header[*Grammar*][*Rule ordering*][*Budget*][*Exact probes*][*Saved*],
    [candidate], [neural structural], [16], [1.98], [72.49%],
    [candidate], [random, seed 20240511], [16], [3.86], [46.30%],
    [candidate], [grammar order], [16], [1.93], [73.22%],
    [candidate], [structural heuristic], [16], [1.93], [73.22%],
    [candidate], [validation-marginal oracle], [16], [1.93], [73.22%],
    [conditional gap], [neural structural], [16], [7.04], [2.15%],
    [conditional gap], [random, seed 20240511], [16], [6.94], [3.49%],
    [conditional gap], [grammar order], [16], [3.20], [55.57%],
    [conditional gap], [structural heuristic], [16], [3.20], [55.57%],
    [conditional gap], [validation-marginal oracle], [16], [3.11], [56.83%],
  ),
  caption: [Matched 16-rule-budget proposer baselines over the same split and certificate
    verifier. Lower exact probes is better.],
) <tab:f2-baselines>

*What this evidence supports.* "Certificate-gated structural rule discovery over this grammar
produces a sound, useful rule set without support-count feature leakage." The soundness comes
from the certificate; the coverage comes from the grammar plus accepted certified filters.

*What this evidence does not support.* "The neural proposer, specifically, is essential."
@tab:f2-baselines is a negative result for that stronger claim: simple deterministic orderings
beat the neural structural proposer at the fixed 16-rule budget on both reported grammars, and
the conditional gap grammar exposes a larger failure. The next neurosymbolic experiment must use
a richer grammar and show that learned ordering beats grammar-order and structural-heuristic
baselines under the same budget.

This negative row is the hinge between the gate result and the artifact-feedback result later in
the paper. A plain neural proposer trained only on structural descriptors is not enough; the
system becomes useful when verifier failures are promoted into reusable reasons. The generated
C5 reason-learning row makes that repair measurable: at budget 8, `model_score` selects 7 unsafe
rules and causes 172 validation false rejections, while reason-prior and trained-residual
orderings recover useful certified rules with zero validation false rejections
(@fig:claim-reason-learning). Thus the paper does not claim "neural proposal wins"; it claims
that verifier-grounded traces can turn failed proposals into the next proposal policy's training
signal.

= Axis 3: Verifier RL must preserve calibration

Starting from a single supervised checkpoint, we clone two RL branches: strict policy gradient
on exact rewards, and the same gradient regularized with KL-to-supervised-prior + entropy.

#figure(
  cetz.canvas({
    import cetz.draw: *
    let policies = (
      (label: "Supervised", ece: 0.0704, nll: 2.9170),
      (label: "Strict RL", ece: 0.1271, nll: 3.0611),
      (label: "Strict RL (T=1.5)", ece: 0.0613, nll: 2.9063),
      (label: "KL/ent RL", ece: 0.0604, nll: 2.9056),
      (label: "KL/ent RL (T=1.0)", ece: 0.0604, nll: 2.9056),
    )
    plot.plot(
      size: (10, 4.5),
      x-label: none,
      y-label: [ECE (lower is better)],
      y-min: 0, y-max: 0.35,
      y-tick-step: 0.05,
      x-min: -0.5, x-max: 4.5,
      x-tick-step: none,
      x-ticks: policies.enumerate().map(((i, p)) => (i, text(size: 7.5pt, p.label))),
      {
        for (i, p) in policies.enumerate() {
          let c = if p.label == "Strict RL" {
            rgb("#d62728")
          } else if p.label.starts-with("KL") {
            rgb("#2ca02c")
          } else {
            rgb("#7f7f7f")
          }
          plot.add-bar(
            ((i, p.ece),),
            bar-width: 0.55,
            style: (fill: c, stroke: black + 0.5pt),
          )
        }
      },
    )
  }),
  caption: [Expected calibration error by RL variant. Strict verifier RL degrades calibration
    (ECE 0.070 $arrow.r$ 0.127) while leaving top-1 *unchanged at 9.13%* across all variants.
    Post-hoc temperature scaling needs $T = 1.5$ to recover; KL/entropy regularization fixes it
    at training time without that crutch.],
) <fig:f3>

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    table.header[*Policy*][*T*][*Top-1*][*Top-10*][*NLL*][*ECE*],
    [Supervised], [1.00], [9.13%], [78.01%], [2.9170], [0.0704],
    [Strict verifier RL], [1.00], [9.13%], [78.01%], [3.0611], [*0.1271*],
    [Strict RL, temp-scaled], [1.50], [9.13%], [78.01%], [2.9063], [0.0613],
    [KL/entropy RL], [1.00], [9.13%], [78.01%], [2.9056], [0.0604],
    [KL/entropy RL, temp-scaled], [1.00], [9.13%], [78.01%], [2.9056], [0.0604],
  ),
  caption: [RL ablation, single cloned supervised checkpoint, default split.],
) <tab:f3>

The mechanism: strict policy gradient on a sparse exact reward signal pushes logit magnitudes
without changing their relative ordering. The KL penalty bounds movement away from the
reference policy; the entropy bonus discourages collapse to a single overconfident action.
Together they reproduce the standard RLHF training stabilization, applied here to verifier
rewards instead of preference rewards.

== Reliability Diagram

#figure(
  cetz.canvas({
    import cetz.draw: *
    plot.plot(
      size: (8, 5.5),
      x-label: [Predicted confidence],
      y-label: [Empirical accuracy],
      x-min: 0, x-max: 1,
      y-min: 0, y-max: 1,
      x-tick-step: 0.2,
      y-tick-step: 0.2,
      legend: "inner-north-west",
      {
        plot.add(
          ((0, 0), (1, 1)),
          style: (stroke: (paint: gray, dash: "dashed", thickness: 0.6pt)),
          label: [perfect calibration],
        )
        plot.add(
          (
            (0.549, 0.667),
            (0.655, 0.429),
            (0.753, 0.667),
            (0.856, 0.571),
            (0.994, 0.975),
          ),
          mark: "o",
          mark-size: 0.18,
          style: (stroke: rgb("#2ca02c") + 1.2pt),
          label: [delayed_claim, replay successor],
        )
      },
    )
  }),
  caption: [Reliability diagram for the trace-replay successor on `delayed_claim`. Bins from
    `results/latest/comparison.jsonl`. The high-confidence bin (894 of 1000 samples,
    $hat(p) = 0.994$, $hat(p)_"emp" = 0.975$) carries most of the mass; the low-population mid
    bins are noisier.],
) <fig:reliability>

= Cross-Task Learning

The same shared neural library, scorer architecture, and verifier-backed task contract
instantiate across the task families in @tab:schema. This section reports how the learned
component behaves on each. The reported numbers are produced by `make results` and live in
`results/latest/comparison.jsonl`; the figures in this section are driven by
`paper/data.typ`, autogenerated from those same artifacts by
`paper/scripts/extract_learning_curves.py` so the figures cannot drift from the underlying
runs.

== Per-task gate ablation

Different tasks instantiate the certificate-gating loop differently: the prime-gap modular
gate instance and grid-path have *gate ablations* in the strict sense (same scorer, with vs without the gate);
delayed-claim has a trace-replay successor plus a production threshold+margin gate;
ARC-AGI has a *scorer-vs-gate* ablation where the two rows are different architectures (a
neural cell scorer and a symbolic transform gate over the same task). A single common-axis
bar chart would obscure those distinctions, so @fig:cross-acc reports four small per-task
panels with row labels that name what is being compared.

#figure(
  cetz.canvas({
    import cetz.draw: *
    // Four small per-task panels. Each carries its own row labels so the
    // comparison is honest: prime/grid are gate ablations on the same scorer;
    // delayed-claim includes its production threshold+margin gate; arc_agi is scorer vs gate.
    let panels = (
      (
        title: [`prime_gap` — gate ablation (same scorer)],
        rows: (
          ("no gate (raw scorer)",   9.13, rgb("#a0a0a0")),
          ("hand-wheel gate",       45.12, rgb("#5b8def")),
          ("learned-wheel gate",    68.87, rgb("#2ca02c")),
        ),
      ),
      (
        title: [`grid_path` — gate ablation (priors disabled vs enabled)],
        rows: (
          ("raw, priors disabled",  57.40, rgb("#a0a0a0")),
          ("cheap gate, priors disabled",69.20, rgb("#5b8def")),
          ("cheap-bound, priors disabled",87.00, rgb("#9467bd")),
          ("mixed gate, priors disabled",93.50, rgb("#2ca02c")),
          ("raw, priors enabled",   75.80, rgb("#c0c0c0")),
          ("cheap gate, priors enabled", 89.70, rgb("#5b8def")),
          ("cheap-bound, priors enabled", 90.10, rgb("#9467bd")),
          ("mixed gate, priors enabled", 95.30, rgb("#2ca02c")),
        ),
      ),
      (
        title: [`delayed_claim` — threshold+margin gate],
        rows: (
          ("raw supervised parent", 93.50, rgb("#a0a0a0")),
          ("trace-replay successor",94.80, rgb("#5b8def")),
          ("production gate",      100.00, rgb("#2ca02c")),
        ),
      ),
      (
        title: [`arc_agi` (`0520fde7`) — scorer vs certified gate],
        rows: (
          ("neural cell scorer",    28.56, rgb("#a0a0a0")),
          ("certified transform",  100.00, rgb("#2ca02c")),
        ),
      ),
    )
    let cell_w = 6.4
    let cell_h = 3.0
    let col_gap = 0.6
    let row_gap = 0.7
    let cols = 2
    for (i, p) in panels.enumerate() {
      let col = calc.rem(i, cols)
      let row = calc.div-euclid(i, cols)
      let x0 = col * (cell_w + col_gap)
      let y0 = -row * (cell_h + row_gap)
      // axes
      line((x0, y0), (x0 + cell_w, y0), stroke: 0.7pt)
      line((x0, y0), (x0, y0 + cell_h), stroke: 0.7pt)
      let nrows = p.rows.len()
      let bar_band = cell_h / (nrows + 0.6)
      let bar_h = bar_band * 0.55
      let max_acc = 100.0
      for (j, r) in p.rows.enumerate() {
        let (label, value, color) = r
        let cy = y0 + cell_h - (j + 0.6) * bar_band
        let bx = x0 + value / max_acc * cell_w
        rect((x0, cy - bar_h / 2), (bx, cy + bar_h / 2),
          fill: color, stroke: black + 0.3pt)
        content((bx + 0.1, cy), anchor: "west",
          text(size: 7pt, [#calc.round(value * 10) / 10%]))
        content((x0 - 0.1, cy), anchor: "east", text(size: 7pt, label))
      }
      content((x0 + cell_w / 2, y0 + cell_h + 0.35),
        text(size: 8.5pt, weight: "bold", p.title))
      for v in (0, 50, 100) {
        let x = x0 + v / max_acc * cell_w
        line((x, y0), (x, y0 - 0.1), stroke: 0.5pt)
        content((x, y0 - 0.3), text(size: 7pt, [#v%]))
      }
    }
  }),
  caption: [Per-task accuracy by ablation row. Each panel labels its own comparison so the
    rows are read against the right baseline: the prime-gap modular gate instance and grid-path are *gate ablations*
    (same scorer, with vs without the gate; grid-path additionally varies whether the
    Manhattan structural priors are enabled in the scorer features); delayed-claim includes
    the trace-replay successor and the production threshold+margin gate over the same
    generator, with parent-hash chain attesting model identity; ARC-AGI is a *scorer-vs-gate*
    ablation where the two rows are different architectures over the same `0520fde7` task,
    not the same model with vs without a gate. The ARC `100.00%` is on nine held-out cells of
    one task; the public ARC evidence surface is the generated v15 measured row reported in
    #claim_C1_arc_public_evidence_surface_summary, with the hypothesis-induction layer in
    @sec:artifact-feedback adding transform families beyond the older v10 floor (zero baseline-rate, 4/4 induction-rate on the synthetic
    suite under `crlp.arc_induction.parity.v1`).],
) <fig:cross-acc>

== Training curves per task

The four tasks expose learning on different timescales. `grid_path` and `delayed_claim`
emit per-step training traces under `--trace-train`, so we plot rolling 50-step accuracy
across a 4000-step supervised run. `prime_gap` does not emit a per-step training trace;
its first-class binary reports per-epoch supervised loss on stdout, which decreases over
8 epochs and is what we plot. `arc_agi` trains its cell scorer in a single supervised
block and exposes per-epoch eval only by re-running the binary with different `--epochs`
values, so we plot the top-1 / NLL curve over a 4-point sweep `{1, 2, 4, 8}`. Each panel
labels its own x-axis honestly. The plotted data is imported verbatim from
`paper/data.typ`, autogenerated by `paper/scripts/extract_learning_curves.py`.

#figure(
  cetz.canvas({
    import cetz.draw: *
    let cell_w = 5.6
    let cell_h = 3.4
    let col_gap = 0.9
    let row_gap = 1.4
    let cols = 2

    // Render an XY-line panel with task-specific scaling.
    let panel(idx, title, pts, xmax, ymax, color, x_label, y_label) = {
      let col = calc.rem(idx, cols)
      let row = calc.div-euclid(idx, cols)
      let x0 = col * (cell_w + col_gap)
      let y0 = -row * (cell_h + row_gap)
      line((x0, y0), (x0 + cell_w, y0), stroke: 0.7pt)
      line((x0, y0), (x0, y0 + cell_h), stroke: 0.7pt)
      let xscale(v) = v / xmax * cell_w
      let yscale(v) = v / ymax * cell_h
      for g in (0.25, 0.5, 0.75, 1.0) {
        if g <= 1.0 {
          let y = y0 + g * cell_h
          line((x0, y), (x0 + cell_w, y),
            stroke: (paint: gray, thickness: 0.3pt, dash: "dotted"))
          content((x0 - 0.1, y), anchor: "east",
            text(size: 7pt, [#calc.round(g * ymax * 100) / 100]))
        }
      }
      let mapped = pts.map(t => (x0 + xscale(t.at(0)), y0 + yscale(t.at(1))))
      for j in range(mapped.len() - 1) {
        line(mapped.at(j), mapped.at(j + 1), stroke: color + 1.0pt)
      }
      for pt in mapped {
        circle(pt, radius: 0.06, fill: color, stroke: white + 0.3pt)
      }
      content((x0 + cell_w / 2, y0 + cell_h + 0.4), text(size: 9pt, weight: "bold", title))
      content((x0 - 0.6, y0 + cell_h / 2), angle: 90deg, text(size: 7pt, y_label))
      content((x0 + cell_w / 2, y0 - 0.55), text(size: 8pt, x_label))
      content((x0, y0 - 0.35), anchor: "west", text(size: 7pt, [#pts.first().at(0)]))
      content((x0 + cell_w, y0 - 0.35), anchor: "east", text(size: 7pt, str(xmax)))
    }

    // (0) prime_gap: per-epoch gated training loss (learned-wheel supervised).
    //     The raw scorer is structurally flat at 12% top-1 because the 129-way
    //     action space exceeds the minimal feature set's residual signal; what
    //     trains is the gated loss, plotted here.
    let pg = prime_gap_epoch_loss
    let pg_pts = pg.map(t => (t.at(0), t.at(2)))
    panel(0,
      [`prime_gap` (gated supervised loss)],
      pg_pts,
      pg_pts.last().at(0),
      0.50,
      rgb("#2ca02c"),
      [training epoch],
      [loss (lower better)])

    // (1) arc_agi: per-epoch raw cell-scorer top-1 on 0520fde7.
    let arc = arc_epoch_eval
    let arc_pts = arc.map(t => (t.at(0), t.at(1)))
    panel(1,
      [`arc_agi` (raw cell scorer top-1)],
      arc_pts,
      arc_pts.last().at(0),
      0.40,
      rgb("#9467bd"),
      [training epoch],
      [top-1 accuracy])

    // (2) grid_path: per-step rolling accuracy (4000-step training trace).
    let gp = learning_curves.at("grid_path")
    let gp_pts = gp.map(t => (t.at(0), t.at(1)))
    panel(2,
      [`grid_path` (rolling-50 train accuracy)],
      gp_pts,
      gp_pts.last().at(0),
      1.0,
      rgb("#5b8def"),
      [training step],
      [accuracy])

    // (3) delayed_claim: per-step rolling accuracy.
    let dc = learning_curves.at("delayed_claim")
    let dc_pts = dc.map(t => (t.at(0), t.at(1)))
    panel(3,
      [`delayed_claim` (rolling-50 train accuracy)],
      dc_pts,
      dc_pts.last().at(0),
      1.0,
      rgb("#d62728"),
      [training step],
      [accuracy])
  }),
  caption: [Per-task training-time curves on four different timescales, each labeled with
    its own y-axis: *`prime_gap`* plots the per-epoch supervised loss of the *gated* scorer
    (learned-wheel supervised), which decreases from 0.42 to 0.23 over 8 epochs and
    plateaus around epoch 3 — the raw scorer's top-1 is structurally flat at $approx$12%
    on this minimal feature set because the 129-way candidate space exceeds the residual
    signal, which is the whole point of the gate. *`arc_agi`* plots per-epoch top-1 of the
    raw cell scorer on `0520fde7` across the sweep $\{1, 2, 4, 8\}$ epochs; the scorer
    climbs from 7.11% to 28.56% over the first four epochs and then plateaus, after which
    additional epochs only inflate NLL (visible as ECE drift in @fig:cross-acc).
    *`grid_path`* and *`delayed_claim`* plot rolling 50-step accuracy across the
    4000-step supervised training trace produced under `--trace-train`. All four series are
    imported from `paper/data.typ`, autogenerated by
    `paper/scripts/extract_learning_curves.py` from the binaries' own outputs.],
) <fig:learning>

== Per-task interpretation

*Prime-gap modular gate instance.* @fig:f1 already shows the *9.13 → 45.12 → 68.87%* progression at final
evaluation. The training curve in @fig:learning shows that the cell scorer alone reaches
$approx$20% top-1 — well below the hand-wheel and learned-wheel gated rows. The model is
learning *something*, but on a 129-way action space the gate is the load-bearing component.

*Grid-path.* The model has access to Manhattan structural priors (`candidate_manhattan_to_goal`,
`candidate_manhattan_delta`); the *prior-disabled* rows quantify their value. With priors
enabled, the raw scorer is at 75.80%; the cheap-local gate lifts this to 89.70%
(+13.9pp), the cheap-bound gate reaches 90.10% (+14.3pp), and the mixed
topology+bound+exact-distance gate reaches 95.30% (+19.5pp). With priors
disabled at evaluation time, the cheap-local gate lifts 57.40% -> 69.20%, the
cheap-bound gate reaches 87.00%, and the mixed gate reaches 93.50%. The clean
cross-domain gate claim is now the non-exact cheap-bound family; the distance-dominance
rule remains an explicitly labeled oracle-style upper bound. The
family/budget proposer sweep is a robustness check: the default neural selector now targets
train/validation residual marginal utility, which prevents overlapping clear-path rules from
consuming the fixed budget, but it is not an "always wins" result at every tiny budget and seed.

*Delayed-claim.* The supervised parent reaches 93.5% accuracy and ECE 0.0363; the trace-replay
successor improves to 94.8% / ECE 0.0166 over the same evaluation set with a parent-hash chain
attesting identity. The production threshold+margin gate then certifies 4 rules for the
neural, closed-loop, structural, and validation-oracle orderings, with zero held-out false
rejections. The accepted threshold rules remove dominated `ignore`/`act_cautiously` actions;
the accepted utility-margin rules resolve the hard `trust` versus `request_more_evidence`
residual. On the held-out generator this leaves 1.00 candidate/action, so the gated row is
100.0% accurate with NLL 0.0000 and ECE 0.0000. The same direct gate stays exact on a shifted
seed band, but clean-observation soundness is not the same thing as noisy-observation
robustness: with observation noise 0.03 it records 23/1000 false rejections, and on a hard
uncertainty+noise band it records 104/1000. The indirect risk-margin grammar is safer but
less decisive: the structural/oracle row has zero false rejections under observation noise
0.03, but only 94.6% accuracy and 1.35 candidates. The reliability diagram in
@fig:reliability is from the replay successor before this deterministic gate.

*ARC-AGI.* On the fixed `0520fde7` task, the neural cell scorer reaches 28.56% top-1 with
*ECE 0.5574 — substantially worse calibration than any of the other three tasks*. The cause
is structural: the cell scorer must distribute probability mass over 10 candidate colors per
output cell while seeing only three training pairs and 9 held-out cells, so the softmax has
neither enough data nor enough structure to align confidence with accuracy. The certified
transform gate then sidesteps the calibration question entirely: it produces a *unique*
grammar-consistent prediction or no prediction, so the held-out top-1 is either 100% or
the gate abstains. On the public ARC corpus, #claim_C1_arc_public_evidence_surface_summary The
hypothesis-induction layer described in @sec:artifact-feedback adds transform families beyond the
older v10 grammar floor. Under the matched-baseline parity protocol `crlp.arc_induction.parity.v1`, on the
synthetic move-to-anchor suite the baseline grammar certifies 0/4 tasks and the
hypothesis-induction loader certifies 4/4 with zero regressions on the previously supported tasks;
the same harness applied to the public ARC corpus is reported below as a scoped evidence
surface rather than as a claim of competitive ARC performance.

== Cross-domain port

To show the substrate is not ARC-specific, the same `NNHypothesis` + `NNProgram` + evidence
schema is instantiated in the `program_synthesis` task on a `filter_then_reduce` template
(stages `enumerate → filter → reduce`, slots `filter.predicate_id`, `reduce.kind`,
`output.type`). The existing certificate checker for the list-DSL abstractions accepts the
compiled programs unchanged, and `make program-synthesis` emits *nine certified
`crlp.hypothesis.evidence.v1` rows* (one per abstraction in the library: `count_positive`,
`count_negative`, `count_even`, `sum_positive`, `sum_negative`, `any_positive`,
`all_positive`, `first_positive_or_zero`, `any_even`). Eight of the nine cover one
task-instance event each; `any_even` certifies but covers zero on the current
fixture-task pool — the row appears in `results/latest/summary.md` as a "Program Synthesis
Hypothesis Evidence" table. A corresponding `prime_gap` hypothesis cross-port is not yet
part of the supported evidence.

== One shared scorer across all four tasks

The cross-task panels in @fig:learning trained a separate `f_theta` for each task. As an
ablation, we replace the four scorers with *one shared scorer of the same architecture* —
a single `[96 → 64 → 32 → 1]` MLP that takes a task-tagged action row (a fixed
96-dimensional envelope with the task id as a one-hot prefix) and softmaxes over each
task's candidate set. The shared scorer is trained on a uniform-random draw across the
four tasks at matched per-task update budget; for the same total budget we also train four
single-task baselines. Numbers are means across three seeds: 16 epochs $times$ 12 000
steps per seed, for 192 000 total updates, captured from `crlp.multitask.summary.v1`
rows. *Three-seed steady state is what the paper claims; the
single-seed snapshot from the earlier capture is retained in `paper/data.typ` as
historical context only.*

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right),
    table.header[*Task*][*Cands*][*Shared top-1*][*Single top-1*][*Δ top-1 (3-seed)*][*Shared NLL*][*Single NLL*],
    ..multitask_rmt1.map(m => {
      let delta = (m.shared_top1 - m.single_top1) * 100.0
      let sign = if delta >= 0 { "+" } else { "−" }
      let mag = calc.round(calc.abs(delta) * 10) / 10
      let delta_text = if delta >= 0 { [#sign#mag pp] } else { [#sign#mag pp] }
      let highlight(v) = text(weight: "bold", v)
      let pct(v) = [#(calc.round(v * 1000) / 10)%]
      let nll(v) = [#(calc.round(v * 10000) / 10000)]
      (
        raw(m.domain),
        [#calc.round(m.candidate_count)],
        if m.shared_top1 >= m.single_top1 { highlight(pct(m.shared_top1)) } else { pct(m.shared_top1) },
        if m.single_top1 > m.shared_top1 { highlight(pct(m.single_top1)) } else { pct(m.single_top1) },
        delta_text,
        nll(m.shared_nll),
        nll(m.single_nll),
      )
    }).flatten()
  ),
  caption: [Shared *one-MLP-on-all-four-tasks* vs *single-task* baseline at three-seed
    steady state, with 192 000 total updates per seed. Same `[96 → 64 → 32 → 1]` MLP,
    same task-tagged 96-feature envelope, same learning rate, three seeds, mean
    reported. At steady state the multi-seed picture is much quieter than the single-seed
    snapshot suggested: `delayed_claim` is statistically tied (+0.1pp); `prime_gap` and
    `grid_path` are tied within sub-1pp; *`arc_agi` keeps a 16.6pp single-task advantage* on
    the trained `0520fde7` task, down from the −28.6pp gap reported at the earlier 24 000-update
    budget. The "positive transfer on delayed_claim" framing from the one-seed snapshot did
    not survive replication and is retracted; the load-bearing multitask finding instead
    comes from the OOD evaluation in @tab:multitask-ood below.],
) <tab:multitask>

*The load-bearing positive transfer signal is on OOD, not on the trained task.* In the
same-envelope OOD evaluation,
we evaluate the same two ARC scorers (shared and single-task) — both trained only on
`0520fde7` — on a held-out suite of 12 same-envelope synthetic ARC tasks (7×3 → 3×3
panel-composition tasks across families like `copy_left`, `copy_right`, `union`, `xor`,
`row_any_intersection`). The shared scorer beats the single-task scorer on this OOD
suite, even though it lost to it on the trained task in @tab:multitask. This is consistent
with the standard interpretation of single-task vs multi-task training: the single-task
scorer is over-specialising to the 270 effective training rows of `0520fde7` (3 train
pairs × 9 cells × 10 candidate colors) while the shared scorer is constrained by joint
training on three unrelated tasks to find more transferable features.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    table.header[*Scorer*][*OOD top-1 (3-seed)*][*OOD NLL*][*OOD ECE*],
    ..multitask_rmt2_aggregate.map(m => {
      let mag(v) = calc.round(v * 10000) / 10000
      let pct(v) = [#(calc.round(v * 1000) / 10)%]
      let highlight = m.model_kind == "shared"
      let row_label = if m.model_kind == "shared" {
        text(weight: "bold", [shared (multi-task)])
      } else {
        [single-task ARC]
      }
      (
        row_label,
        if highlight { text(weight: "bold", pct(m.top1)) } else { pct(m.top1) },
        [#mag(m.nll)],
        [#mag(m.ece)],
      )
    }).flatten()
  ),
  caption: [Same-envelope ARC out-of-distribution evaluation. Both scorers were
    trained only on `0520fde7`; this table evaluates each on a held-out suite of 12
    synthetic 7×3 → 3×3 panel-composition tasks with the same 96-feature action envelope
    and the same 10-way color action space. Three seeds, aggregate row. *The shared
    multi-task scorer beats single-task by +5.57pp top-1 on OOD* despite losing by 16.6pp
    on the trained task. NLL drops by 40% under shared (2.63 vs 4.37), confirming that
    single-task is producing confidently wrong predictions on unseen tasks.],
) <tab:multitask-ood>

The reading is deliberately narrow. On the fixed ARC task, shared training loses
in-distribution top-1; on same-envelope OOD, it improves top-1, NLL, and ECE. The
certificate-gating contract makes that trade acceptable because the gate owns correctness
where structure is already certified, while the scorer is most useful on the residual tasks
the gate does not cover. Under a pure-scorer system this would be a direct
memorization-vs-generalization tradeoff; under a verifier-gated system, the gate handles
covered structure and the scorer can be optimized for residual ranking.

Two boundary checks keep the claim from becoming too broad. First, the calibration audit
shows a real tradeoff: on fixed `0520fde7`, shared training lowers NLL (4.08 vs 8.28) while
worsening ECE (0.6995 vs 0.5589); on same-envelope OOD, it improves both NLL (2.63 vs
4.37) and ECE (0.4481 vs 0.4820). Second, the delayed-claim leave-one-out row shows that
transfer is sibling-specific rather than universal: all-domain training beats claim-only
by +1.40pp, removing `grid_path` hurts most, and removing ARC slightly helps.

The capacity sweep rules out the simplest architectural explanation. Wider is not
monotonically better: `[96→32→16→1]` is the closest shared ARC row to single-task
(-1.23pp), while `[96→256→128→1]` only partially recovers the `64x32` gap. Delayed-claim
stays positive or tied at every capacity. The remaining question is whether the current
small nets are undertrained, saturated, or missing cumulative protocol state.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    table.header[*Hidden*][*Params*][*ARC shared*][*ARC single*][*ARC Δ*][*Claim Δ*],
    ..multitask_rmt6_capacity.map(m => {
      let pct(v) = [#(calc.round(v * 1000) / 10)%]
      let delta(v) = {
        let pp = v * 100.0
        let sign = if pp >= 0 { "+" } else { "−" }
        let mag = calc.round(calc.abs(pp) * 100) / 100
        [#sign#mag pp]
      }
      (
        raw(m.hidden),
        [#m.parameters],
        pct(m.arc_shared_top1),
        pct(m.arc_single_top1),
        delta(m.arc_delta),
        delta(m.claim_delta),
      )
    }).flatten()
  ),
  caption: [Fixed-task capacity sweep at the same three-seed, 192 000-update
    protocol. The ARC gap is *not* monotonic in trunk size: `32x16` is closest
    to single-task on fixed `0520fde7`, while `256x128` only partially recovers the
    `64x32` baseline. Delayed-claim remains positive or tied at every capacity.],
) <tab:multitask-capacity>

The scorer-invariance audit fixes the boundary between learned scoring and certified ARC evidence. The public
ARC transform gate and budgeted proposer path currently do not consume a multitask scorer
checkpoint, so the comparison is an invariance audit over the scorer-independent path.
Across 400 public train tasks and 400 public eval tasks, accepted rule sets, prediction
hashes, loader summaries, and budgeted proposer aggregates were identical under the
`shared_scorer_context` and `single_task_scorer_context` labels (`0` exact gate/loader
mismatches and `0` proposer mismatches). This strengthens the verifier-correctness claim,
but it also narrows the scorer claim: the audit does not show that the shared scorer improves
public ARC search, only that the certified gate/proposer measurements are not contaminated
by scorer choice in the current implementation.

== Main artifact-feedback results

The two central result rows are the C5 artifact-feedback rows: rejected-candidate learning
and certified hypothesis-library feedback. The rest of the claim table is boundary evidence:
ARC is a measured evidence surface, the capacity sweep guards architecture interpretation,
and Signal is application pressure under replay, shadow, and guarded live-active logging. Each row is evaluated
against a registered result artifact before the PDF is built, so unsupported or stale
required claims fail before they become prose. The only non-blocking negative row remains
the fixed-budget prime proposer claim, which is deliberately carried as a limitation.

#figure(
  table(
    columns: (1.7fr, auto, auto, auto),
    align: (left, left, left, center),
    table.header[*Claim*][*Family*][*Status*][*Required*],
    ..paper_claim_support_surfaces.map(s => (
      [#raw(s.id)],
      [#s.family],
      if s.status == "supported" { text(weight: "bold", [supported]) } else { [#s.status] },
      if s.required { [yes] } else { [no] },
    )).flatten(),
  ),
  caption: [Generated claim status matrix. Required claims fail `make paper-support` if
    their registered evidence is missing, stale, underpowered, or contradicted.],
) <tab:generated-claim-support>

#figure(
  claim_coverage_chart(
    claim_C5_rejected_candidate_reason_learning_chart_rows,
    y-max: 260,
    y-label: [budget-8 validation utility],
  ),
  caption: [C5 rejected-candidate reason learning. #claim_C5_rejected_candidate_reason_learning_summary],
) <fig:claim-reason-learning>

#figure(
  claim_coverage_chart(
    claim_C5_hypothesis_library_promotion_chart_rows,
    y-max: 5,
    y-label: [learned certified programs at budget 1],
  ),
  caption: [C5 certified hypothesis-library promotion. #claim_C5_hypothesis_library_promotion_summary],
) <fig:claim-hypothesis-library>

The next rows are intentionally not headline performance claims. ARC is kept as an evidence
surface: it exposes which transform and hypothesis families are certified, where the grammar
misses, and whether scorer-independent gate measurements stay invariant. Signal is kept as
application pressure: it exercises replay-certified ranking, shadow logging, and live
neural-active execution without claiming that active neural control beats the heuristic.

#figure(
  claim_coverage_chart(
    claim_C1_arc_public_evidence_surface_chart_rows,
    y-max: 1200,
    y-label: [public ARC eval cells certified],
  ),
  caption: [C1 public ARC evidence surface. #claim_C1_arc_public_evidence_surface_summary
    The chart is intentionally scoped to certified covered cells, not overall ARC
    performance.],
) <fig:claim-arc-public-surface>

#figure(
  claim_delta_chart(
    claim_C3_arc_ood_calibration_tradeoff_chart_rows,
    y-min: -16,
    y-max: 6,
    y-label: [shared ECE improvement vs single (pp)],
  ),
  caption: [C3 ARC calibration tradeoff. Positive bars mean shared scoring has lower ECE
    than single-task scoring; negative bars mean worse ECE. #claim_C3_arc_ood_calibration_tradeoff_summary],
) <fig:claim-arc-calibration-tradeoff>

#figure(
  claim_delta_chart(
    claim_C4_delayed_claim_sibling_transfer_chart_rows,
    y-min: 0,
    y-max: 2.2,
    y-label: [top-1 delta vs claim-only (pp)],
  ),
  caption: [C4 delayed-claim sibling transfer. #claim_C4_delayed_claim_sibling_transfer_summary],
) <fig:claim-delayed-transfer>

#figure(
  claim_delta_chart(
    claim_C4_multitask_capacity_tradeoff_chart_rows,
    y-min: -18,
    y-max: 2,
    y-label: [ARC shared minus single top-1 (pp)],
  ),
  caption: [C4 capacity boundary. Negative bars mean the shared scorer still trails the
    single-task ARC scorer at that trunk size. #claim_C4_multitask_capacity_tradeoff_summary],
) <fig:claim-capacity-boundary>

= Application Pressure: Signal Flight

Signal is the practical reach test for the protocol, but the claim is intentionally
narrow. The current result is not "a tiny net flies the game" and not "neural control beats
the heuristic." Signal is split into generated evidence rows: an offline counterfactual
replay row, a live shadow row, and a live neural-active row. In the replay row, each sampled
state enumerates nine WASD candidates, the Signal simulator replays the candidates, and the
ranker learns from replay-certified utility, damage, and death outcomes.

#figure(
  claim_coverage_chart(
    claim_C9_signal_flight_replay_control_chart_rows,
    y-max: 90,
    y-label: [held-out top-1 (%)],
  ),
  caption: [C9 Signal replay-control vignette. #claim_C9_signal_flight_replay_control_summary
    The baseline is the primitive no-safety goal-heading WASD policy. This row supports
    bounded replay-certified ranking only; live execution is tracked by the separate
    neural-active claim.],
) <fig:claim-signal-flight>

#figure(
  claim_coverage_chart(
    claim_C9_signal_live_neural_active_chart_rows,
    y-max: 300,
    y-label: [live rows],
  ),
  caption: [C9 Signal live neural-active execution. #claim_C9_signal_live_neural_active_summary
    This row supports live guarded neural-active execution and replay-ready telemetry, not
    comparative utility or safety versus the heuristic baseline.],
) <fig:claim-signal-live-active>

This is why Signal belongs in @tab:schema even before it becomes a headline control
experiment: it changes the environment class without changing the contract. Candidate
enumeration is still explicit, the verifier is authoritative simulator replay or logged
live telemetry, the certificate is a replayed or replay-ready outcome record rather than a
theorem-like proof, and the neural component still ranks candidates rather than owning
safety. The next unsupported gate is comparative active control: summarize active neural
episodes against a heuristic baseline with utility, regret, damage, and death.

= Artifact Feedback and Audit Contract <sec:artifact-feedback>

The hybrid architecture is the contribution; the audit contract is the discipline that makes
the claims reproducible. Six pieces matter:

The generated C5 rows make the artifact-feedback loop explicit. Rejected candidate traces
become reason priors and a trained residual that repair the prime/modular proposer
economics without introducing false rejections (@fig:claim-reason-learning). Repeated
zero-false-rejection hypothesis bindings become a reusable library prior and feed that
prior back into budgeted proposal ranking (@fig:claim-hypothesis-library). In both cases,
the method is the experiment: verified claim outcomes become the next round's proposal
features, and the claim compiler decides whether the paper may say the loop helped.

- *Stable action IDs.* Trace rows carry `schema: "crlp.trace.v1"`, `trace_version: 1`, and
  `action_id_contract: "stable_unique_action_ids"`, plus a declared ranking policy and
  selected-score margin. Replay resolves `target_index` to the trace row's `action_id`,
  then maps that stable id back into the regenerated candidate row. The trace determinism
  audit projects rows onto replay-semantic fields, hashes that projection, and reports the
  count of `rank_fp_sensitive` decisions rather than requiring bit-identical logits.
- *Checkpoint identity.* SHA-256 over the exact checkpoint bytes, plus `parent_model_hash`,
  `feature_encoder_version`, and `feature_set` in metadata. Successor models declare their
  parent; comparison reports carry both hashes and the evaluated `feature_set`.
- *Hypothesis layer.* `NNHypothesis` (header `include/nn_hypothesis.h`) sits one level above
  `NNProgram`: a *template* declares a typed operator skeleton as a sequence of stages with
  named, typed slots; a *binding* fills each slot with a concrete value inferred from
  observed task evidence; `nn_hypothesis_compile` produces a composed `NNProgram` whose
  canonical id binds back to the hypothesis template, so the existing verifier and
  closed-loop policy treat it as any other program. `crlp.hypothesis.evidence.v1` rows are
  emitted by `nn_hypothesis_write_evidence_json`, a library function reused by both
  `arc_agi` and `program_synthesis`. The provenance contract holds: state-observation
  slot values are `NN_FEATURE_STATE_OBSERVATION`, gate-derived support / residual gain
  appear only on the evidence row, never on the hypothesis feature vector that a
  selection-mode proposer scores against.
- *Certificate-gated baselines.* `crlp.arc_induction.parity.v1` records the matched
  baseline-vs-induction certification delta per split (training, evaluation) on the public
  ARC corpus or any synthetic split; `crlp.program_proposer.summary.v1` records
  matched-budget rule-set comparisons across `{random, grammar-order, structural, learned,
  closed-loop, oracle}`. Both schemas attest *what was held fixed* between rows, which is
  the load-bearing thing for cross-task comparison.
- *Pure C, no third-party deps.* The neural library is dense layers, sigmoid/ReLU/tanh/linear,
  MSE and softmax cross-entropy, SGD, batched 4-by-4 tiled forward/backward, cloned networks,
  action scoring, pairwise ranking loss, and cached policy-gradient updates with optional
  KL/entropy regularization. $approx$3k lines for the core; the current certified modular
  gate implementation is split into focused C modules for data, features, rule search,
  training, evaluation, and replay.
- *CI-baseline runs.* A pinned `ubuntu-24.04` + `gcc-13` image runs `make check` and the
  ARC induction smoke on every push and PR, runs `make results` nightly, captures
  `crlp.environment.v1` for runner identity, and emits a *timing-stripped*
  `comparison.canonical.jsonl` so the diff that gates "should the paper rewrite?" cannot be
  triggered by wallclock noise. Laptop runs produce the same artifacts; drift shows up as a
  visible `cc_version` or `git_dirty` difference rather than as silent numeric drift.

= Limitations <sec:limitations>

#set enum(numbering: "L1.")
+ *Experiment-family scope is frozen for this paper.* The current evidence matrix is already
  broad enough to support the protocol claim. Additional domains would make the draft less
  falsifiable unless they replace a weaker row or pass the same candidate/verifier/gate/scorer
  trace standard. New domains belong in roadmap notes, not body evidence.
+ *Proposer learning remains weak (Axis 2).* The fixed-budget baseline shows grammar order and
  a simple structural heuristic beat the current neural proposer on both the unconditional
  candidate grammar and the conditional gap grammar. The load-bearing claim is certificate-gated
  rule discovery, not learned proposer superiority. The negative result is reported deliberately.
+ *Cross-task verifier-RL comparison missing (Axis 3).* Calibration evidence is on the
  prime-gap modular gate instance only. The pipeline supports running strict-vs-KL on
  grid-path and delayed-claim; the comparison is not in canonical results yet.
+ *Limited cross-domain gate coverage (Axis 1).* The prime-gap modular gate instance and
  grid-path have the strongest *gate ablations* in the strict sense. Delayed-claim now has
  a production threshold+margin gate that resolves its held-out deterministic generator,
  including the hard `trust` versus `request_more_evidence` residual, and it now reports
  shifted/noisy stress rows. Those rows are mixed: the direct grammar is exact under clean
  seed shift but false-rejects under observation noise, while the indirect proxy avoids
  false rejections but leaves a larger residual. This is evidence of the robustness frontier,
  not a broad noisy-claim solution.
  ARC-AGI's "gate" is a different architecture (transform grammar) over the same task, not
  the same scorer with vs without filtering, so it is reported separately in @fig:cross-acc.
+ *Statistical rigor.* Default `make prime-gap-experiments` runs 1 seed; the harness
  supports more (`PRIME_GAP_EXPERIMENT_SEEDS=…`). The 9.13 → 45.12 → 68.87% deltas, the
  73.22% baseline saving, the grid-path 75.80 -> 90.10 cheap-bound gate delta,
  the 95.30 mixed-gate upper bound, and the modular-rule-search negative
  result all need confidence intervals across $gt.eq 5$ seeds before any of them should be
  treated as a stable property.
+ *Learned-wheel calibration tradeoff* needs a reliability-diagram pass. The learned wheel
  improves ranking sharply (@tab:f1) but has higher ECE (0.1402) than the raw minimal row
  (0.1076).
+ *ARC ECE outlier.* The 0.5574 ECE on the `0520fde7` cell scorer is reported, with the
  structural explanation (10 colors × 3 train pairs is too sparse for the softmax to align
  with accuracy). It is not yet replicated across multiple ARC tasks; the structural
  argument is suggestive, not proof.
+ *Signal lacks a comparative active-control result.* The Signal rows now support offline
  replay ranking, live shadow evidence, and guarded live neural-active execution
  (@fig:claim-signal-live-active). They do not yet support the stronger claim that active
  neural control improves utility, regret, damage, or death versus the heuristic baseline.
+ *No competitive large-system baseline.* The related-work comparison now positions the
  protocol against learned verifiers, RLVR, AlphaGeometry-style formal search, DreamCoder,
  Hypothesis Search, and ARC systems, but the artifact does not empirically compare against
  those large systems. That is acceptable for the protocol claim; it would not be acceptable
  for a claim of competitive theorem proving, language-model reasoning, or ARC performance.
+ *Single-runner numbers.* All headline percentages were captured from a single CI
  baseline run (`ubuntu-24.04` + `gcc-13`, see `results/latest/environment.json`). Laptop
  runs produce different `model_ms_per_sample` and minor numerical differences in
  `nll` / `ece` at the third decimal; the canonical pipeline strips timing fields and now
  exposes a replay-semantic trace hash plus `rank_fp_sensitive` counts, but a cross-machine
  double-blind has not yet been run. Raw logits remain diagnostics rather than semantic
  trace identity.

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
    claim support gate is currently #paper_support_status. Optional contradicted claims remain visible
    here instead of blocking PDF compilation.],
) <tab:generated-limitations>

#set enum(numbering: "1.")

= Conclusion

The paper's strongest claim is narrow: certified candidate mechanisms should own
admissibility, and small learned components should rank, calibrate, reuse, or learn around
those certified hypotheses. The two main empirical contributions are the feedback loops.
Rejected candidate rows become reason-coded targets for a later safety-first residual
ranker; certified hypotheses become reusable library priors and are checked on held-out
proposal budgets. The gate, tiny-net, ARC, and Signal rows matter because they keep that
claim honest: modular gates show the protocol can sharply reduce residual search, capacity
audits block a naive wider-is-better story, ARC exposes priors and misses rather than
claiming broad competence, and Signal keeps the loop under replay-grounded application
pressure without claiming comparative live-control performance.

The negative fixed-budget proposer result is therefore not an embarrassment. It is evidence
that the protocol is falsifiable: a flat neural proposer over structural rows is weak, but
the verifier trace can still become useful training signal. The broader implication is
pragmatic. When correctness, safety, or auditability matter, end-to-end learning should not
own the whole decision. Verifier-grounded ranking substrates provide a smaller, auditable
contract: symbolic or replay checkers own admissibility, neural components spend capacity on
search and prioritization, and the trace artifacts make both success and failure reusable.

#bibliography("references.bib")
