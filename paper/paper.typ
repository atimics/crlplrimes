#import "@preview/cetz:0.3.4"
#import "@preview/cetz-plot:0.1.1": plot, chart

#set document(
  title: "Certificate-Gated Neural Proposers for Verifier-Grounded Action Scoring",
  author: "TBA",
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

#let placeholder(body) = box(
  fill: rgb("#fff7d6"),
  stroke: rgb("#c9a000") + 0.5pt,
  inset: (x: 4pt, y: 2pt),
  radius: 2pt,
  text(weight: "bold", fill: rgb("#7a5b00"), [PLACEHOLDER: ] + body),
)

#let pblock(body) = block(
  fill: rgb("#fff7d6"),
  stroke: (left: 2.5pt + rgb("#c9a000")),
  inset: (x: 8pt, y: 6pt),
  width: 100%,
  spacing: 6pt,
  [#text(weight: "bold", fill: rgb("#7a5b00"))[PLACEHOLDER \u{2014} ] #body],
)

#align(center)[
  #text(size: 17pt, weight: "bold")[
    Certificate-Gated Neural Proposers for \
    Verifier-Grounded Action Scoring
  ]

  #v(8pt)
  #text(size: 11pt)[
    #placeholder[Authors and affiliations]
  ]

  #v(6pt)
  #text(size: 10pt, style: "italic")[Working draft \u{2014} ideal-paper roadmap]
]

#v(6pt)

#align(center)[
  #box(width: 85%)[
    #set par(justify: true)
    #set text(size: 10pt)
    *Abstract.*
    When an exact verifier is available, the right architecture for action scoring is not an
    end-to-end neural policy. We study a hybrid pipeline in which a deterministic
    certificate-checked gate prunes structurally impossible candidates before a small learned
    scorer ranks the residual. On a prime-gap benchmark, gating the same scorer with a learned
    rule set lifts top-1 accuracy from 44.60% (fixed hand-wheel) to 68.87% (learned wheel) at
    100% top-10, with an $approx$8$times$ reduction in model latency. The symbolic gate itself
    is produced by a neural proposer trained against a soundness-certificate verifier, which
    accepts 148 rules with zero false rejections out of 25,122 proposed candidates and recovers
    69.40% of wheel-equivalent verifier work. Under sparse-exact-reward policy gradient on the
    same task, strict verifier RL inflates logit scale (ECE 0.074 $arrow.r$ 0.310) without
    improving ranking; a KL-toward-prior + entropy regularizer keeps ECE at 0.082 at the same
    top-k. Prime gaps are used as a substrate, not the goal; we argue the protocol is
    transferable to any exact-verifier domain.
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
a verifier-checked symbolic filter, and the filter itself can be produced by a neural proposer
gated by a soundness-certificate verifier. The scorer ranks the residual; the gate eliminates
the rest; nothing the gate emits can violate correctness because every rule it accepts carries
a deterministic proof.

We make three claims, each tied to a concrete experiment in the released artifact:

#set enum(numbering: "(C1)", spacing: 6pt)
+ *Certificate-gating dominates raw scoring.* On the canonical prime-gap benchmark, the same
  scorer reaches 44.60% top-1 under a fixed hand-wheel gate and *68.87% top-1 under a learned
  rule gate*, both at 100% top-10. The dominating effect is the gate, not the depth of the
  scorer.
+ *A certificate-gated proposer discovers sound symbolic gates.* A neural proposer over a small
  rule grammar, paired with a soundness-certificate verifier, produces 148 sound rules with
  zero false rejections, recovering 69.40% of wheel-equivalent verifier work. The soundness
  guarantee is a property of the certificate, not of training.
+ *KL-regularized verifier RL prevents calibration collapse.* Strict policy gradient on sparse
  exact rewards inflates logit scale rather than improving ranking; a standard KL-to-prior +
  entropy term restores calibration at unchanged top-k.

#set enum(numbering: "1.", spacing: auto)

We use prime gaps as the experiment domain because they offer a fast deterministic verifier at
all scales, a compact totally-ordered candidate space, an exhausted classical literature, and
rich algebraic structure for the rule grammar. *We are not claiming improved prime generation.*

= Related Work

#pblock[
  Required citations and 1-paragraph positioning against each:
  - Process supervision (Lightman et al., 2023): we use a similar verifier-as-step-checker
    pattern; their reward model is learned, ours is exact-by-construction.
  - Verifier-guided decoding / best-of-N (Cobbe et al., 2021; Hosseini et al., 2024): we gate
    *before* scoring, not after.
  - AlphaProof / AlphaGeometry symbolic-neural search (Trinh et al., 2024): our rule synthesis
    is structurally similar but verifier-grounded by certificate rather than by proof
    completion.
  - RLHF KL-regularized policy gradient (Stiennon et al., 2020; Ouyang et al., 2022): our
    Finding 3 is this objective applied to exact-verifier rewards.
  - Neural-symbolic program synthesis (NeSy literature).
]

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
  caption: [The certificate-gating loop. A learned rule proposer (top) produces candidate rules
    that a soundness-certificate verifier accepts or rejects. Accepted rules deterministically
    prune the candidate set; a shared neural scorer ranks the survivors; the exact verifier
    supplies the correctness signal.],
) <fig:loop>

The design invariant is *the model proposes, the verifier decides, the trace remembers.* A rule
$r$ is accepted only if a deterministic algebraic check certifies that $r$ cannot reject the
true action on any state in the support set, so soundness does not depend on the proposer being
well-trained.

== Trace and Checkpoint Contract

Every step writes a JSONL row with schema `crlp.trace.v1`, including a stable per-action
identifier, the candidate scores and probabilities, the selected and target indices, the
verifier result, and verifier cost. Replay resolves `target_index` $arrow.r$ stable `action_id`
$arrow.r$ regenerated candidate row, so reordering action ids cannot silently change replay
semantics. Checkpoints carry SHA-256 model identity, a `parent_model_hash`, and a
`feature_encoder_version`; comparison reports (`crlp.compare.v1`) carry both hashes plus
machine-readable reliability bins.

= Experiment Domain: Prime Gaps

== Setup

The state is a prime $p$; the candidate space is gaps $g in {2, 4, ..., 256}$ plus an overflow
action; the verifier returns the true next-prime gap. Each candidate is encoded as
$[log p, "norm_gap", "overflow_flag"]$ concatenated with divisibility flags for $p + g$ modulo
small primes. The scorer is a shared MLP returning one scalar per candidate; softmax over
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

= Finding 1: Certificate-gating dominates raw scoring

The same trained scorer is evaluated under three gate conditions on the canonical 30,000-sample
held-out benchmark. Top-1 results:

#figure(
  cetz.canvas({
    import cetz.draw: *
    let bars = (
      (label: "no gate", value: 45.12, color: rgb("#a0a0a0")),
      (label: "hand-wheel", value: 44.60, color: rgb("#5b8def")),
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
    defensible learned-gate delta is *44.60 $arrow.r$ 68.87* over the fixed hand-wheel
    (+24.3pp). The no-gate row is shown for context; the 45.12 $arrow.r$ 68.87 framing conflates
    the gate's contribution with the gain from having any gate at all.],
) <fig:f1>

Full metrics:

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    table.header[*Gate*][*Top-1*][*Top-10*][*NLL*][*ECE*][*ms/sample*],
    [no gate], [45.12%], [99.37%], [1.6691], [0.1180], [0.0415],
    [hand-wheel], [44.60%], [100.00%], [1.3775], [0.0538], [---],
    [*learned wheel*], [*68.87%*], [*100.00%*], [*1.0302*], [0.1332], [*0.0052*],
  ),
  caption: [Gate ablation, parent scorer, identical weights across rows.],
) <tab:f1>

#pblock[
  *Required to elevate C1 from "load-bearing on prime-gap" to "general":*
  - *E1.1* \u{2014} Random-rule gate at matched verifier-call budget: replace the learned rule
    set with $k$ random sound rules and report the top-1 / top-10 / NLL curve as $k$ varies.
    Resolves whether the *content* of the learned rules matters or only their count.
  - *E1.2* \u{2014} Replicate on a second domain. `grid_path` admits a natural certificate gate
    (legality + admissible-heuristic pruning). Same scorer + three gates + same table.
  - *E1.3* \u{2014} Investigate the *hand-wheel ECE anomaly* (0.054 vs no-gate 0.118 and
    learned-wheel 0.133). One of two things is true: the hand-wheel induces a special
    candidate-set structure that happens to be well-calibrated, or the learned wheel is
    inducing miscalibration that should be addressed.
  - *E1.4* \u{2014} Confidence intervals across $gt.eq 5$ seeds for the 44.60 $arrow.r$ 68.87
    delta.
]

= Finding 2: A certificate-gated proposer discovers sound symbolic gates

The rule grammar is `if p % m == a then reject gap % m == b`; the certificate verifier accepts
a rule iff $gcd(a + b, m) > 1$ (so $p + g$ is necessarily composite) and the support count
exceeds a threshold. Zero false rejections is a structural property of the certificate.

#figure(
  cetz.canvas(length: 1cm, {
    import cetz.draw: *
    let stages = (
      (label: "proposed",         value: 25122, color: rgb("#a0a0a0")),
      (label: "verified",         value: 934,   color: rgb("#88b0e8")),
      (label: "accepted",         value: 148,   color: rgb("#2ca02c")),
      (label: "false\nrejections", value: 0,    color: rgb("#d62728")),
    )
    let plot_w = 12.0
    let plot_h = 5.0
    let x_origin = 0.0
    let y_origin = 0.0
    let col_w = plot_w / stages.len()
    let max_log = calc.log(25122, base: 10)
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
  caption: [Rule-search acceptance funnel. Of 25,122 proposed rules, 934 pass the soundness
    certificate, 148 survive support and false-rejection screens, and *zero* falsely reject the
    true next-prime gap. The accepted set recovers 69.40% of wheel-equivalent verifier work.],
) <fig:f2>

*What this evidence supports.* "A certificate-gated proposer over this grammar produces a
sound, useful rule set." The soundness comes from the certificate; the coverage comes from the
combination of proposer ranking and enumeration.

*What this evidence does not yet support.* "The neural proposer, specifically, is essential."
The numbers above are consistent with both "the proposer ranked rules well" and "a near-random
proposer + the same verifier reaches the same rule set." Distinguishing the two requires E2.1
below.

#pblock[
  *Required to elevate C2 from "the verifier is doing useful work" to "the learned proposer is
  essential":*
  - *E2.1* \u{2014} Random-rule-proposer baseline at fixed verifier-call budget. Sample rules
    uniformly from the grammar; run the same certificate verifier; report accepted rules and
    verifier work saved as a function of budget. The learned proposer's claim is then "matches
    the random baseline at $approx N$x lower budget" for some defensible $N$.
  - *E2.2* \u{2014} Exhaustive enumerator at the same budget. For small enough $m$, exhaustive
    enumeration is feasible and lower-bounds the achievable coverage.
  - *E2.3* \u{2014} Generalize the grammar: extend to conjunctive rules
    (`if p % m == a and ... then reject ...`), report whether the proposer scales.
  - *E2.4* \u{2014} Show one *worked example* of an accepted rule the hand-wheel does not
    capture and explain why it generalizes.
]

= Finding 3: KL-regularized verifier RL prevents calibration collapse

Starting from a single supervised checkpoint, we clone two RL branches: strict policy gradient
on exact rewards, and the same gradient regularized with KL-to-supervised-prior + entropy.

#figure(
  cetz.canvas({
    import cetz.draw: *
    let policies = (
      (label: "Supervised", ece: 0.0743, nll: 1.5778),
      (label: "Strict RL", ece: 0.3100, nll: 2.3612),
      (label: "Strict RL (T=2.5)", ece: 0.0707, nll: 1.5816),
      (label: "KL/ent RL", ece: 0.0821, nll: 1.5878),
      (label: "KL/ent RL (T=1.2)", ece: 0.0682, nll: 1.5602),
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
  caption: [Expected calibration error by RL variant. Strict verifier RL collapses calibration
    (ECE 0.074 $arrow.r$ 0.310) while leaving top-1 *unchanged at 45.12%* across all variants.
    Post-hoc temperature scaling needs $T = 2.5$ to recover; KL/entropy regularization fixes it
    at training time without that crutch.],
) <fig:f3>

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    table.header[*Policy*][*T*][*Top-1*][*Top-10*][*NLL*][*ECE*],
    [Supervised], [1.00], [45.12%], [99.77%], [1.5778], [0.0743],
    [Strict verifier RL], [1.00], [45.12%], [99.46%], [2.3612], [*0.3100*],
    [Strict RL, temp-scaled], [2.50], [45.12%], [99.46%], [1.5816], [0.0707],
    [KL/entropy RL], [1.00], [45.12%], [99.75%], [1.5878], [0.0821],
    [KL/entropy RL, temp-scaled], [1.20], [45.12%], [99.75%], [1.5602], [0.0682],
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
    bins are noisier. #placeholder[Replace with the prime-gap reliability diagram for supervised,
    strict-RL, and KL-RL once `make prime-gap` is run with `reliability_bins` capture enabled.]],
) <fig:reliability>

#pblock[
  *Required to elevate C3 from "true on prime-gap" to "a property of sparse-exact-reward RL":*
  - *E3.1* \u{2014} Run strict-vs-regularized RL on `grid_path` and `delayed_claim` with the
    same protocol; report the analogous @tab:f3 for each.
  - *E3.2* \u{2014} Sweep the KL coefficient and entropy weight on a coarse grid for each task,
    plot the calibration-ranking Pareto frontier.
  - *E3.3* \u{2014} Per-bin reliability diagrams for *strict-RL vs KL-RL on prime-gap* (the
    current diagram uses delayed-claim as a stand-in because that's the only task whose
    `reliability_bins` are currently captured).
  - *E3.4* \u{2014} Confidence intervals across $gt.eq 5$ seeds for the strict-vs-KL ECE delta.
]

= Substrate

The hybrid architecture is the contribution; the substrate is the discipline that makes the
claims auditable. Three pieces matter and would be a brief methods section in a longer paper:

- *Stable action IDs.* Trace rows carry `schema: "crlp.trace.v1"`, `trace_version: 1`, and
  `action_id_contract: "stable_unique_action_ids"`. Replay resolves `target_index` to the
  trace row's `action_id`, then maps that stable id back into the regenerated candidate row.
- *Checkpoint identity.* SHA-256 over the exact checkpoint bytes, plus `parent_model_hash` and
  `feature_encoder_version` in metadata. Successor models declare their parent; comparison
  reports carry both hashes.
- *Pure C, no third-party deps.* The neural library is dense layers, sigmoid/ReLU/tanh/linear,
  MSE and softmax cross-entropy, SGD, batched 4-by-4 tiled forward/backward, cloned networks,
  action scoring, pairwise ranking loss, and cached policy-gradient updates with optional
  KL/entropy regularization. $approx$3k lines for the core; the prime-gap task is split into
  focused C modules for data, features, rule search, training, evaluation, and replay.

= Limitations

#set enum(numbering: "L1.")
+ *Random-proposer baseline missing (Finding 2).* Until E2.1 lands, the load-bearing claim is
  "certificate-gated rule discovery is sound and recovers most of the wheel," not "the learned
  proposer is essential."
+ *Cross-task verifier-RL comparison missing (Finding 3).* The pipeline supports it; the
  comparison is not in canonical results yet.
+ *Single-domain headline (Finding 1).* Hybrid-wins is demonstrated on prime gaps only.
+ *Statistical rigor.* Default `make prime-gap-experiments` runs 2 seeds for 1 epoch; the
  harness supports more. Deltas need confidence intervals across $gt.eq 5$ seeds.
+ *Hand-wheel ECE anomaly* is unexplained.
+ *No literature comparison* on the calibration-collapse mechanism yet.
#set enum(numbering: "1.")

= Conclusion

#pblock[
  Write the conclusion last. Three sentences max: (a) certificate-gating is the architectural
  claim, (b) verifier-grounded rule synthesis is the constructive claim, (c) KL-regularized
  verifier RL is the training-stability claim. Avoid claiming the protocol scales; that is
  future work, not a result.
]

#v(12pt)

= Roadmap to the ideal paper

This appendix tracks every experiment the current draft asks for but does not yet have. The
goal is that every #placeholder[...] marker in the body resolves to one of these rows.

#figure(
  table(
    columns: (auto, 1.4fr, auto, auto),
    align: (left, left, center, center),
    table.header[*ID*][*Experiment*][*Effort*][*Defends*],
    [E1.1], [Random-rule gate at matched verifier-call budget], [2d], [C1],
    [E1.2], [Replicate Finding 1 on `grid_path` with admissible-heuristic gate], [3-4d], [C1],
    [E1.3], [Investigate hand-wheel ECE anomaly (0.054 vs 0.118 / 0.133)], [1d], [C1],
    [E1.4], [CIs across $gt.eq 5$ seeds for 44.60 $arrow.r$ 68.87 delta], [0.5d], [C1],
    [E2.1], [Random-rule-proposer baseline at fixed budget], [2d], [C2],
    [E2.2], [Exhaustive rule enumerator at same budget (where feasible)], [1d], [C2],
    [E2.3], [Conjunctive rule grammar extension], [3-5d], [C2],
    [E2.4], [Worked example of an accepted rule the hand-wheel misses], [0.5d], [C2],
    [E3.1], [Strict-vs-regularized RL on `grid_path` and `delayed_claim`], [2-3d], [C3],
    [E3.2], [KL/entropy Pareto sweep per task], [2d], [C3],
    [E3.3], [Reliability diagrams for strict-RL and KL-RL on prime-gap], [0.5d], [C3],
    [E3.4], [CIs across $gt.eq 5$ seeds for strict-vs-KL ECE delta], [0.5d], [C3],
    [W1], [Related-work positioning paragraphs (see §2)], [1d], [all],
    [W2], [Conclusion + abstract revision after experiments], [0.5d], [all],
  ),
  caption: [Roadmap. Total estimated effort: $approx$3 weeks of focused work to take the
    current draft from "defensible as an arxiv preprint" to "submission-ready with all
    headline claims independently supported."],
) <tab:roadmap>

#v(6pt)

*Minimum viable submission.* E2.1, E1.3, E3.3, W1, W2. $approx$5 days. Leaves Findings 2 and 3
with qualified scope (prime-gap only) but every claim in the body has a defensible experiment
behind it.

*Strong submission.* Add E1.2, E2.4, E3.1, E1.4, E3.4. $approx$2 weeks. All three findings have
either a cross-task replication or a baseline strong enough to rule out the obvious alternative
explanation.

*Full ideal paper.* Everything above. $approx$3 weeks. The grammar extension (E2.3) is the
optional ambition; if conjunctive rules generalize, the paper grows a fourth finding.
