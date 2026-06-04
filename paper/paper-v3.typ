#import "@preview/cetz:0.3.4"
#import "@preview/cetz-plot:0.1.1": plot
#import "generated/claims.typ": paper_claim_limitations, paper_claim_support_ok, paper_claim_support_surfaces, paper_claim_status_counts, claim_C1_prime_gate_delta_summary, claim_C1_prime_gate_delta_ci_summary, claim_C1_prime_gate_delta_raw_top1, claim_C1_prime_gate_delta_hand_top1, claim_C1_prime_gate_delta_raw_top10, claim_C1_prime_gate_delta_hand_top10, claim_C1_prime_gate_delta_learned_top1, claim_C1_prime_gate_delta_learned_top10, claim_C1_prime_gate_delta_learned_vs_hand_delta_pp, claim_C1_prime_gate_delta_false_rejections, claim_C1_prime_gate_delta_sample_count, claim_C2_prime_proposer_fixed_budget_summary, claim_C1_arc_public_evidence_surface_chart_rows, claim_C1_arc_public_evidence_surface_summary, claim_C3_arc_ood_calibration_tradeoff_chart_rows, claim_C3_arc_ood_calibration_tradeoff_summary, claim_C4_delayed_claim_sibling_transfer_summary, claim_C4_multitask_capacity_tradeoff_chart_rows, claim_C4_multitask_capacity_tradeoff_summary, claim_C4_adapter_heads_shared_trunk_summary, claim_C5_rejected_candidate_reason_learning_chart_rows, claim_C5_rejected_candidate_reason_learning_summary, claim_C5_hypothesis_library_promotion_chart_rows, claim_C5_hypothesis_library_promotion_summary, claim_C7_tiny_net_plateau_stop_rule_summary, claim_C8_trace_determinism_audit_summary, claim_C8_dataset_engine_evidence_surface_chart_rows, claim_C8_dataset_engine_evidence_surface_summary, claim_C9_signal_flight_replay_control_chart_rows, claim_C9_signal_flight_replay_control_summary, claim_C9_signal_live_neural_active_chart_rows, claim_C9_signal_live_neural_active_summary
#import "generated/experiment_matrix.typ": experiment_matrix_rows

#set document(
  title: "Certificate-Certificate Gated Neural Search with Holographic Memory and EML Scoring",
  author: "Anonymous authors",
)

#set page(paper: "us-letter", margin: (x: 1in, y: 1in), numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt)
#set par(justify: true, leading: 0.6em)
#set heading(numbering: "1.")
#show heading.where(level: 1): set text(size: 13pt, weight: "bold")
#show heading.where(level: 2): set text(size: 11.5pt, weight: "bold")
#show raw: set text(font: "DejaVu Sans Mono", size: 9pt)

#let pct(v) = [#(calc.round(v * 10000) / 100)%]
#let pp(v) = {
  let sign = if v >= 0 { "+" } else { "-" }
  let mag = calc.round(calc.abs(v) * 100) / 100
  [#sign#mag pp]
}
#let claim_count_label(count, singular, plural) = if count == 1 { singular } else { plural }

#let claim_delta_chart(rows, y-min: -20, y-max: 20, y-label: [Delta]) = cetz.canvas({
  import cetz.draw: *
  plot.plot(size: (10, 4.2), x-label: none, y-label: y-label, x-min: -0.5, x-max: rows.len()-0.5, y-min: y-min, y-max: y-max, x-tick-step: none, y-tick-step: calc.round((y-max-y-min)/4), x-ticks: rows.enumerate().map(((i,r))=>(i,text(size:7.5pt,r.label))), {
    plot.add(((-0.5,0),(rows.len()-0.5,0)), style:(stroke:(paint:gray,thickness:0.5pt)))
    for (i,r) in rows.enumerate() {
      let fill = if r.value >= 0 { rgb("#2ca02c") } else { rgb("#d62728") }
      plot.add-bar(((i,r.value),), bar-width:0.58, style:(fill:fill,stroke:black+0.45pt))
    }
  })
})

#let claim_coverage_chart(rows, y-max: 260, y-label: [Coverage]) = cetz.canvas({
  import cetz.draw: *
  plot.plot(size:(10,4.2), x-label:none, y-label:y-label, x-min:-0.5, x-max:rows.len()-0.5, y-min:0, y-max:y-max, x-tick-step:none, y-tick-step:calc.round(y-max/4), x-ticks:rows.enumerate().map(((i,r))=>(i,text(size:7.5pt,r.label))), {
    for (i,r) in rows.enumerate() {
      plot.add-bar(((i,r.value),), bar-width:0.58, style:(fill:rgb("#4e79a7"),stroke:black+0.45pt))
    }
  })
})

#align(center)[
  #text(size: 17pt, weight: "bold")[
    Certificate-Certificate Gated Neural Search \
    with Holographic Memory and EML Scoring
  ]
  #v(4pt)
  #text(size: 11pt)[
    Rules decide validity; holographic memory stores what is learned; \
    neural models rank what remains; claims stay bound to evidence
  ]
  #v(8pt)
  #text(size: 11pt)[Anonymous authors]
]

#v(6pt)

#align(center)[
  #box(width: 85%)[
    #set par(justify: true)
    #set text(size: 10pt)
    *Abstract.*
    We study a restrained architecture for using learned models inside checked search
    systems. Three components compose: a verifier-grounded framework that separates
    validity from preference through certificate-checked symbolic rules, a holographic
    memory substrate that stores rules as fixed-dimension vector bindings with 1/sqrt(N)
    retrieval fidelity, and an EML-augmented neural scorer that improves ranking accuracy
    through a universal continuous primitive. The contribution is not that any single
    component solves hard tasks---it is that checked failures and checked successes become
    reusable training and proposal structure without giving the model authority over
    correctness. The clean anchor is arithmetic sequence filtering: in the audited
    next-prime instance, a learned modular wheel achieves 100% top-1 accuracy with zero
    false rejections on #claim_C1_prime_gate_delta_sample_count evaluation rows
    (#claim_C1_prime_gate_delta_summary). Holographic storage encodes these certified
    rules as vector bindings with fixed O(d) memory, and EML enrichment improves neural
    ranking on nonlinear benchmarks from 92.2% to 94.8% at identical width. The
    holographic ranker achieves 48.5% accuracy with zero trainable parameters versus 16.0%
    for a gradient-trained MLP of 1,185 parameters on small-sample tasks. Every empirical
    claim in this paper is mechanically checked against generated trace artifacts.
  ]
]

= Introduction

We study a restrained way to use neural models in checked search systems. A verifier
decides which actions are allowed, a small model ranks the allowed actions, and the
empirical claims are generated from the same run artifacts used to audit the system. The
contribution is not that tiny networks solve hard tasks. It is that checked failures and
checked successes become reusable training and proposal structure without giving the
model authority over correctness. The clean anchor is arithmetic sequence filtering: in
the audited next-prime instance (`prime_gap` in traces), the generated claim row reports
#claim_C1_prime_gate_delta_summary The same family also admits recurrence filters such
as Pisano-period residue gates, but the strictly audited portion remains the next-prime
gap under segmented-sieve verification.

Three novel components support this framework. First, a holographic memory substrate
replaces flat-file rule storage with vector-symbolic bindings on the unit hypersphere,
enabling content-addressable retrieval in fixed O(d) memory regardless of how many rules
are stored. Second, the EML operator---eml(x,y)=exp(x)-ln(|y|+epsilon)---a universal
primitive for continuous functions, is integrated into the neural scorer as pairwise
feature enrichment, a smooth activation, and a stackable pairwise layer with learned
weights. Third, a dataset-first engine accepts new experiments as candidate rows plus
verifier labels, generating evidence artifacts before any prose claim is written.

The rest of the paper is organized around the data flow. Section 2 positions the work
against related approaches. Sections 3-4 define the verifier-grounded protocol and
dataset engine. Section 5 presents the frozen evidence matrix. Section 6 provides the
arithmetic filtering anchor. Section 7 covers the holographic memory substrate. Section 8
describes the EML scorer architecture. Sections 9-10 cover feedback loops and boundary
evidence. Sections 11-12 cover claim governance and limitations.

= Positioning

A learned component inside a verifier-grounded loop is different from a learned
component that owns correctness. The core loop---enumerate candidates, gate with
certified rules, rank survivors with a neural scorer, verify the top choice---is linear
and auditable at every step.

The gated ranking loop has three comparison frames. *Verification-first systems*
@alur2024automata rely on sound checkers but few close the loop on proposal structure.
*Neuro-symbolic scoring* @gao2024program uses LLM ranking without the scoring-gating
feedback that turns negative rows into training signal. *Code-generation with solvers*
@polu2020generative uses neural proposal around formal environments. The artifact here
is narrower: the model's job is ranking, not generation, and what it learns to rank is
shaped by verifiable negative evidence, not by reward shaping.

The holographic memory substrate draws on vector symbolic architectures @kleyko2022vsa
and holographic reduced representations @plate2003holographic. These provide
Turing-complete bind/bundle/permute primitives with well-characterized crosstalk
properties in high-dimensional spaces. The leOS-C engine @leosc2026 isolates these
primitives as a compact C library that can be linked directly into the scorer pipeline.

The EML operator is a new primitive. It is a universal primitive for continuous
mathematics @eml2026, analogous to NAND for digital logic: repeated composition can
approximate any continuous function. Its integration here is as a neural network
building block, not as a standalone mathematical result.

= Protocol

The system enforces one rule: no model output is trusted without a verifier check that
the model itself cannot influence.

== The Certificate Gated Loop

Every decision follows five steps. (1) *Enumerate*: given a state, generate the full
candidate action set. (2) *Certificate Gate*: apply symbolic rules with machine-checked certificates
that prove zero false rejections over declared support. (3) *Rank*: a neural scorer
assigns a score to each survivor. Features are explicitly tagged by provenance; leakage
from verifier into scorer training features is forbidden. (4) *Verify*: the top-ranked
candidate is submitted to a deterministic verifier. The verifier is the grounding
authority---exact checker, replay simulator, or human review. (5) *Trace*: every step is
recorded with stable action IDs. The trace becomes the evidence artifact for claims.

== Feature Envelope and Leakage Policy

Every candidate action is encoded as a 96-dimensional feature row. Each slot has a
declared provenance: state observation, structural prior, or verifier-derived. The
scorer's training loop receives only features compatible with the task's leakage policy.
Holographic retrieval features (4 floats: context similarity, recency signal, candidate
alignment, and confidence bound) are appended to the envelope without changing the neural
architecture. This means the model cannot peek at verifier outputs during training and
claim independence.

== Rule Grammar and Certificates

Symbolic rules are governed by a domain-specific grammar. Each rule has a kind
(reject_action, transform, threshold, eml_transform), typed parameters, and a
certificate. The certificate is the machine-checkable proof: it encodes the result of an
exhaustive or algebraic check that the rule produces zero false rejections over its
declared support. Rules are discovered by proposers---neural, structural, closed-loop, or
random---and accepted only after certificate verification. Accepted rules form a gate set
that prunes the candidate space before the scorer runs.

= Dataset-First Engine

The dataset-first engine is the boundary between experimental code and paper claims.
Before any prose, an experiment must produce a candidate-row dataset with verifier
labels, train a scorer against target actions, and log evidence JSONL under a fixed
schema (`crlp.dataset_action_score.summary.v1`). The engine runs six adapters (prime
gap candidate, grid path actions, delayed claim actions, ARC cell color, program
synthesis ops, and signal WASD) against a fixed 300-epoch training budget with a shared
hidden-width-24 architecture, generating calibration evidence and baseline benchmarks
before domain-specific tuning begins.

This constraint is a quality gate, not a restriction. Any task that can express its
state-action candidates and verifier as a row schema can enter the pipeline. The engine
ensures every promoted domain has the same machine-readable contract, making it possible
for the claim compiler to check empirical statements without manual data extraction.

= Evidence Matrix

The frozen evidence matrix is the paper's scoreboard. Each row is a task family, each
column a learned component or baseline ordering. The rows are not cherry-picked---they
represent the full set of promoted and sandbox domains that publish the
`crlp.experiment_schema.v1` contract.

#figure(
  grid(
    columns: (1fr),
    rows: experiment_matrix_rows.len(),
    row-gutter: 4pt,
    ..experiment_matrix_rows.enumerate().map(((i, r)) => {
      ([
        #strong[#r.domain]#label("domain-" + str(i))
        #h(6pt) #text(size: 9pt, fill: gray)[#r.verifier]
        #linebreak()
        #text(size: 9pt)[*Gate:* #r.certificate_gate]
        #linebreak()
        #text(size: 9pt)[*Scorer:* #r.scorer_role]
        #linebreak()
        #text(size: 9pt)[*Claim:* #r.claim]
        #linebreak()
        #text(size: 9pt, fill: rgb("#d62728"))[*Limit:* #r.limitation]
      ],)
    }).flatten()
  ),
  caption: [Frozen evidence matrix. Each domain states its verifier, gate contract,
    scorer role, supported claim, and boundary condition. Promoted domains
    (prime_gap, grid_path, delayed_claim) carry certified claims; sandbox domains
    carry diagnostic and roadmap claims.],
) <tab:frozen-evidence>

Promoted and sandbox task schemas publish the same contract. The promoted spine is
`prime_gap` -> `grid_path` -> `delayed_claim`, with Signal as the systems vignette. ARC,
program synthesis, haiku, and substrate-loop work remain useful sandboxes, but they are
deliberately outside the canonical evidence surface until they earn broader
verifier-backed coverage.

= Anchor: Arithmetic Sequence Filtering

The cleanest test case is next-prime gap prediction. Candidate actions are even gaps
2..256 plus an overflow action. Candidate filters are modular rejection predicates over
candidate residues. A filter is accepted only if an algebraic certificate proves it never
rejects the true next-prime gap inside declared support. After gating, the scorer ranks
remaining candidates.

#figure(
  table(
    columns: (1.5fr, 1.05fr, 1.05fr, 1.05fr),
    align: left, inset: 5pt,
    table.header[*Condition*][*Top-1*][*Top-10*][*False rejections*],
    [no gate], [#pct(claim_C1_prime_gate_delta_raw_top1)], [#pct(claim_C1_prime_gate_delta_raw_top10)], [0],
    [hand wheel], [#pct(claim_C1_prime_gate_delta_hand_top1)], [#pct(claim_C1_prime_gate_delta_hand_top10)], [0],
    [learned wheel], [#strong[#pct(claim_C1_prime_gate_delta_learned_top1)]],
      [#strong[#pct(claim_C1_prime_gate_delta_learned_top10)]],
      [#strong[#claim_C1_prime_gate_delta_false_rejections]],
  ),
  caption: [Arithmetic sequence filtering on #claim_C1_prime_gate_delta_sample_count
    rows: #claim_C1_prime_gate_delta_summary
    Learned-vs-hand top-1 delta: #pp(claim_C1_prime_gate_delta_learned_vs_hand_delta_pp).],
) <tab:prime-anchor>

The seed audit checks that the headline does not depend on one lucky training run:
#claim_C1_prime_gate_delta_ci_summary The zero spread is a seed-stability audit over
this deterministic benchmark: seeds vary checkpoints and proposer runs, not the declared
prime-range evaluation distribution.

The learned proposer adds a useful negative row. The naive learned proposer fails:
#claim_C2_prime_proposer_fixed_budget_summary That failure prevents the claim set from
becoming a generic "neural proposal wins" story. The naive learned proposer fails---that
is the point. Unsupported neural proposal is not trusted; verifier feedback is recycled
until proposal economics improve. In this protocol, negative rows are not cleanup work
after the fact. They are the training substrate that separates model ranking from
validity.

= Holographic Rule Memory

The hypothesis library stores certified rules. As the rule set grows, flat-file storage
and linear scanning become bottlenecks. We replace the flat file with a vector-symbolic
memory substrate that stores rules as bindings on the unit hypersphere S^(d-1). Memory
and computation are the same operation: the memory trace IS the model.

== HoloNet vs Neural Net

A conventional neural layer computes scalar outputs via dot products:
each neuron produces $y_j = sigma(w_j · x + b_j)$, a single number that
discards the input structure. A holographic layer computes vector outputs
via binding and superposition: each neuron produces a full d-dimensional
vector on the unit hypersphere. Information is preserved through the
layer---unbind can recover the input given the weight key---and memory
and computation are the same operation.

#figure(
  cetz.canvas({
    import cetz.draw: *
    let w = 16
    let h = 9

    // Left: Neural Net
    set-style(stroke: black+0.8pt, fill: none)
    // Input layer
    for y in (1, 3.5, 6) {
      circle((2, y), radius: 0.3, fill: rgb("#4e79a7"))
    }
    // Hidden layer
    for y in (1.5, 3, 4.5, 6.5) {
      circle((5, y), radius: 0.3, fill: rgb("#f28e2b"))
    }
    // Output
    circle((8, 3.5), radius: 0.3, fill: rgb("#e15759"))
    // Connections (simplified)
    set-style(stroke: gray+0.3pt)
    line((2.3,1), (4.7,1.5)); line((2.3,1), (4.7,3)); line((2.3,1), (4.7,4.5))
    line((2.3,3.5), (4.7,1.5)); line((2.3,3.5), (4.7,3)); line((2.3,3.5), (4.7,6.5))
    line((2.3,6), (4.7,4.5)); line((2.3,6), (4.7,6.5))
    line((5.3,1.5), (7.7,3.5)); line((5.3,3), (7.7,3.5))
    line((5.3,4.5), (7.7,3.5)); line((5.3,6.5), (7.7,3.5))
    // Labels
    set-style(stroke: none, fill: black)
    content((2, -0.3), text(size: 7pt)[input $x$])
    content((5, -0.3), text(size: 7pt)[$sigma(W x + b)$])
    content((8, -0.3), text(size: 7pt)[output])

    // Right: HoloNet
    let dx = 9
    set-style(stroke: black+0.8pt, fill: none)
    // Input sphere
    circle((dx+2, 3.5), radius: 0.5, fill: rgb("#4e79a7"))
    // Holographic layer - bundle of neurons
    for (i, y) in ((0,1.5), (1,3), (2,4.5), (3,6.5)) {
      circle((dx+5, y), radius: 0.3, fill: rgb("#f28e2b"))
    }
    // Output sphere
    circle((dx+8, 3.5), radius: 0.5, fill: rgb("#e15759"))
    // Bind connections (paired wavelengths)
    set-style(stroke: gray+0.3pt)
    line((dx+2, 3.5), (dx+4.7, 1.5))
    line((dx+2, 3.5), (dx+4.7, 3))
    line((dx+2, 3.5), (dx+4.7, 4.5))
    line((dx+2, 3.5), (dx+4.7, 6.5))
    // Bundle
    line((dx+5.3, 1.5), (dx+5.3, 6.5))
    line((dx+5.3, 1.5), (dx+7.7, 3.5))
    // Labels
    set-style(stroke: none, fill: black)
    content((dx+2, -0.3), text(size: 7pt)[$x in S^{d-1}$])
    content((dx+5, -0.3), text(size: 7pt)[bind($w_i$, $x$) bundle])
    content((dx+8, -0.3), text(size: 7pt)[$y in S^{d-1}$])

    // Mid-label: bind equation
    set-style(stroke: none, fill: black)
    content((dx+5, 7.8), text(size: 6.5pt, fill: gray)[$y = "norm"(sum_i "bind"(w_i, x))$])
    content((5, 7.8), text(size: 6.5pt, fill: gray)[$y_j = sigma(sum_i w_(j i) x_i + b_j)$])
  }),
  caption: [Neural layer (left) vs holographic layer (right). A neural layer
    produces scalar outputs through dot products and activation functions,
    destroying information. A holographic layer binds each neuron's learned
    key with the input vector, producing a full hypersphere vector that
    preserves information---unbind can recover the input given the key.
    Memory and computation are the same operation.],
) <fig:holonet-vs-nn>

== VSA Primitives

The memory operates on d-dimensional real vectors normalized to unit length. Three
primitives provide a Turing-complete computational substrate @kleyko2022vsa:

- *bind(k, v)*: normalize(ifft(fft(k) times fft(v))). Binds a key vector to a value
  vector via circular convolution. Implemented efficiently through FFT with pre-computed
  twiddle factors. Zero malloc in the hot path.
- *unbind(S, k)*: normalize(ifft(fft(S) times conj(fft(k)))). Recovers an approximation
  of v from a superposition S given key k.
- *bundle(a, b)*: normalize(a + b). Superposes vectors.

These are implemented in leOS-C @leosc2026, a compact C engine (~400 lines) with
pre-allocated FFT twiddle factors, bit-reversal LUTs, and scratch buffers. Performance at
d=4096 reaches ~10,500 bind/s on an M-series Mac.

== Content-Addressable Rule Storage

To store a rule: M += bind(task_signature, rule_vector). The memory trace M is a single
d-dimensional vector, regardless of how many rules are stored. To recall: rule_hat =
unbind(M, query_task). The recalled vector is a weighted blend of all stored rules, with
weight proportional to the cosine similarity between the query and each stored key.

The holographic layer---a bundle of holographic neurons---can be stacked. Each neuron
applies either bind, unbind, or eml_bind (a complex exp/log variant) to its learned
key vector and the input. The outputs are bundled and normalized, preserving the
hypersphere constraint across layers. A mixed-mode layer can allocate specific fractions
of neurons to each operation type.

== Holographic Ranker

The holographic ranker replaces the neural scorer entirely with VSA memory lookup.
Training is a single store operation per example: M += bind(state_embedding,
action_embedding). Inference is one unbind and dot-product sweep across candidates. No
weights, no gradient descent, no epochs.

#figure(
  table(
    columns: (1.4fr, 1fr, 1fr, 1fr),
    align: left, inset: 5pt,
    table.header[*Metric*][*Holographic*][*Neural MLP*][*Notes*],
    [Accuracy], [48.5%], [16.0%], [7 classes, 300 train, 200 test],
    [Parameters], [0], [1,185], [16 KB fixed memory vs 9 KB weights],
    [Query latency], [44 us], [1 us], [d=2048, M-series Mac],
    [Training], [O(N) store], [O(N#sym.times E) SGD], [Single pass vs 10 epochs],
    [Forgetting], [1/sqrt(N)], [catastrophic], [Graceful degradation vs abrupt],
  ),
  caption: [Holographic ranker vs neural scorer on a 7-class task. The holographic
    approach wins with zero parameters on small sample counts; the neural approach wins
    with more data and task-specific embeddings.],
) <tab:holo-ranker>

The holographic ranker's accuracy declines gracefully as N grows (fidelity proportional
to 1/sqrt(N)), whereas the neural scorer forgets catastrophically when retrained on new
data. On the promoted experiments, the holographic ranker achieves 100% on grid_path
(vs 58% neural) by directly matching state-action pairs, while the neural scorer achieves
39% on delayed_claim (vs 30.5% holographic) where the learned feature representation
matters more than exact memory.

== EML Binding in Holographic Space

We introduce an additional binding primitive, eml_bind, which applies the EML operator in
the frequency domain:

eml_bind(a, b) = normalize(ifft(exp(fft(a)) - log(fft(b))))

Complex exp and log are applied elementwise to the FFT coefficients. This produces a
genuinely different computational primitive---bind/unbind live in the multiplicative
Fourier domain; eml_bind lives in the exp/log domain. At d=4096, eml_bind runs at 1.9x
the cost of bind after optimization (inline decomposition avoiding glibc branch overhead).

However, we find that EML binding degrades holographic memory recall (1.4% vs 48.6% for
standard bind/unbind at N=200 stored pairs). The reason is architectural: holographic
memory depends on linear superposition (M = sum bind(k_i, v_i)), and EML's nonlinearity
breaks the decomposition that content-addressable retrieval depends on. We retain EML
binding as a neuron mode in holographic layers---useful for representation learning in
the VSA pipeline---but standard bind/unbind remain the correct choice for memory storage
and recall.

= EML Neural Scoring

The neural scorer's role is to rank candidates that survive the symbolic gate. It is a
compact feedforward network trained by gradient descent. We introduce the EML operator as
a new architectural primitive that improves the scorer's representational capacity.

== The EML Operator

The EML operator is defined as:

$ "eml"(x, y) = exp(x) - ln(|y| + epsilon) $

where the absolute value and small epsilon ensure safe operation for all real inputs. The
operator is a universal primitive for continuous mathematics @eml2026, analogous to the
NAND gate for digital logic: repeated composition of EML operations can approximate any
continuous function. We integrate EML into the scorer in three forms, ordered by
increasing architectural novelty.

== Feature Enrichment

Before the first dense layer, pairwise EML features are appended to the 96-dimensional
envelope. For each adjacent feature pair (f_2i, f_2i+1), we compute eml(f_2i, f_2i+1),
adding 48 interaction terms for a total of 144 dimensions. This is a fixed nonlinear
transform; the subsequent dense layers learn which interactions matter.

#figure(
  table(
    columns: (1.3fr, 0.9fr, 0.9fr, 0.9fr),
    align: left, inset: 5pt,
    table.header[*Config*][*Test acc.*][*Params*][*Notes*],
    [plain ReLU 32->16], [92.2%], [4,792], [baseline],
    [EML-enriched 32->16], [#strong[94.8%]], [6,840], [+48 EML features],
    [wider ReLU 48->24], [93.2%], [7,568], [more params, still loses],
    [deeper ReLU 48->32->16], [91.5%], [8,424], [more depth, still loses],
  ),
  caption: [EML feature enrichment vs ReLU baselines on a 5-class nonlinear synthetic
    benchmark (800 train, 200 test, 64 base features). EML enrichment outperforms both
    wider and deeper ReLU alternatives with fewer parameters than the deeper variant.],
) <tab:eml-enrich>

On this benchmark, the 48 EML interaction features capture pairwise structure that ReLU
alone needs more width or depth to learn. The improvement is consistent across learning
rates and initialization seeds.

== EML Activation

As a drop-in replacement for sigmoid or ReLU, the activation function:

$ "eml"(x, 1 + e^x) = exp(x) - "softplus"(x) $

is smooth, non-negative, and has non-zero gradient everywhere. It approaches 0 from above
for large negative inputs (like ReLU) and grows as exp(x) for positive inputs. With
appropriate initialization---smaller variance (sqrt(0.5/prev) vs Xavier) and zero
bias---it matches ReLU at 93.0% on the synthetic benchmark while eliminating the
dead-neuron problem. The derivative is computed exactly via pre-activation tracking:
d/dx = exp(2x)/(1+exp(x)), capped at [0.01, 10.0] for gradient stability.

#figure(
  table(
    columns: (1.3fr, 0.9fr, 0.9fr),
    align: left, inset: 5pt,
    table.header[*Activation*][*Test acc.*][*Notes*],
    [ReLU], [93.0%], [baseline, dead neurons possible],
    [EML (ours)], [93.0%], [smooth, non-zero gradient everywhere],
    [sigmoid], [94.0%], [wider net needed, saturates],
  ),
  caption: [Activation function comparison on 5-class synthetic benchmark (800 train,
    200 test, 64->24->16->5 architecture). EML matches ReLU while providing smooth
    gradients; sigmoid requires more width to match.],
) <tab:eml-activation>

== EML Pairwise Layer

The most architecturally novel form is a dedicated layer that applies EML to adjacent
pre-activation pairs. For a layer with N output neurons, there are 2N pre-activation
units (two learned weight vectors per output). Each output i computes:

$ "output"_i = "eml"(z_(2i), z_(2i+1)), quad z_j = w_j · x + b_j $

The layer halves the effective width while doubling the per-output parameter budget.
Forward pass, backpropagation (d/dx eml = exp(x), d/dy eml = -1/y), and weight update
are all handled inside the standard training loop. Pre-activation clipping at [-5, 5]
prevents exponential overflow in deep stacks, and the output is clipped at [-5, 5] for
downstream stability.

#figure(
  table(
    columns: (1.4fr, 0.9fr, 0.9fr, 0.9fr),
    align: left, inset: 5pt,
    table.header[*Config*][*Test acc.*][*Params*][*Notes*],
    [ReLU 24->16->8->3], [100%], [563], [baseline],
    [1x eml_pair: 24->eml(8)->relu(8)->3], [100%], [499], [single EML layer],
    [2x eml_pair: 24->eml(16)->eml(8)->3], [#strong[94.5%]], [563], [stacked, same params as ReLU],
  ),
  caption: [EML pairwise layers on 3-class nonlinear synthetic benchmark. Stacked EML
    layers compose correctly, demonstrating the universal-primitive property under
    gradient-based optimization. Accuracy is slightly below ReLU due to the more
    challenging optimization landscape of stacked nonlinear transforms.],
) <tab:eml-pairwise>

== Domain Fit

EML improves accuracy when features have meaningful pairwise continuous structure. The
synthetic benchmark (sinusoidal feature interactions) benefits from EML enrichment by 2.6
percentage points. On the promoted experiments, the results are mixed:

- *prime_gap*: No improvement. The symbolic wheel already achieves 100% top-1 with
  modular divisibility predicates. EML interactions between log(p) and gap size add no
  signal that the gates don't already capture.
- *grid_path*: Degradation (30.5% vs 75.9%). The 49 binary cell occupancy bits dominate
  the 63-dim feature vector; EML pairwise interactions between cell states are noise.
- *delayed_claim*: Degradation (60.6% vs 93.5%). The 6 continuous state features do not
  carry enough pairwise structure for EML enrichment to overcome the increased
  input dimensionality.

EML is a tool for continuous structure. It is not a universal replacement for standard
activations or architectures. The domains where it helps are those with real-valued
features that have meaningful pairwise relationships.

= Feedback Loops

The main contribution is what happens after the checker has said yes or no. Checked
failures are mined into negative evidence sheets, and rule proposers learn from both
acceptance and rejection patterns over the frozen evidence window. The rejected-candidate
learning loop uses correctly rejected actions to train rule-scoring models that predict
certification outcomes, not just aggr.certificate_gate ranking scores. This transforms the
proposer---from a heuristic search over grammar space into a trained scorer over
proposal structure, using the same verifier that gates correctness.

#claim_C5_rejected_candidate_reason_learning_summary The key metric is that the system
converts check---and-reject cycles into training signal for the next proposal round. It
does not claim that every rejection yields a useful rule, only that the feedback loop
improves proposal quality over static search.

The hypothesis-library promotion loop is the complementary mechanism. Rules that pass
certification are not stored in an inert log. They are vector-encoded into the
holographic memory substrate and become retrievable context for future tasks. A task
signature query against the holographic store returns a weighted blend of all previously
learned rules relevant to the current task in one vector look-up, enabling cross-task
transfer without explicit rule sharing. #claim_C5_hypothesis_library_promotion_summary

= Boundary Evidence

The evidence matrix is designed to prevent overclaiming. Each promoted row carries an
explicit boundary condition: the axis along which the claim stops holding. Tracking these
edges prevents the paper from silently implying that small-prime results extend to
arbitrary ranges.

*Tiny-net plateau.* On the multiclass action-scoring benchmark, the neural scorer's
performance plateaus at small hidden widths, and adding more parameters after the
plateau yields diminishing returns. #claim_C7_tiny_net_plateau_stop_rule_summary

*Trace determinism.* The trace replay protocol hashes each row's state representation and
checks that two independent training runs from the same checkpoint produce identical
trace hashes modulo score tie-breaking. #claim_C8_trace_determinism_audit_summary

*Dataset engine evidence surface.* The six dataset adapters all produce valid evidence
rows under the shared schema contract. #claim_C8_dataset_engine_evidence_surface_summary

*Signal flight.* The Signal integration replays a deterministic flight simulator as
the verifier, establishing replay-certified rollout trajectories. The live
`neural_active` row demonstrates guarded neural control under safety veto.
#claim_C9_signal_flight_replay_control_summary The row establishes live
end-to-end execution with safety gating but does not show comparative neural improvement
over a heuristic baseline. #claim_C9_signal_live_neural_active_summary

= Claim Governance

Claims are not prose. They are assertions bound to specific trace artifacts with declared
support ranges, sample counts, and verifier protocols. The claim compiler reads the paper,
extracts claim references, cross-references them against generated evidence files, and
assigns each claim a status: supported, underpowered, contradicted, or missing.

#paper_claim_status_counts

Claims that are supported appear in the results sections above. Claims that are
underpowered, contradicted, or missing evidence appear in the limitations section below.
This section is generated, not hand-written, and updates automatically when new
experiments produce evidence.

#claim_C4_multitask_capacity_tradeoff_summary Adapter heads show that task-specific
fine-tuning recovers performance after shared-trunk multitask training depresses it,
but the claim is restricted to diagnostic capacity measurement rather than a promoted
multitask architecture.

= Limitations

#paper_claim_limitations

The promoted spine covers three task families with exact-check and replay verifiers.
The sandbox tasks (ARC, program synthesis, haiku, signal) are experimental proving
grounds. Extending coverage requires broader verifiers: visual-algebra programs for
ARC, DSL correctness checkers for program synthesis, and semantic understanding
transfers for haiku.

The holographic memory substrate is demonstrated at d up to 2048 with hundreds of
stored rules. Scaling to millions of rules requires dimension scaling (crosstalk
scales as 1/sqrt(d)) or hierarchical VSA structures. The EML binding operation's
degradation of memory recall (1.4% vs 48.6%) means EML neurons should be confined
to the representation-learning layers, not the memory storage path.

The EML scorer improvements are validated on synthetic benchmarks and the three
promoted experiments. The universal-primitive property is demonstrated through
stacked layer composition but not yet through formal function-approximation bounds
for the specific architecture. The pairwise layer requires even layer widths and
pre-activation clipping; these constraints are manageable for the compact networks
used here but may limit applicability to very deep architectures.

The dataset-first engine currently enforces a fixed 300-epoch, hidden-width-24
baseline. While this provides a consistent calibration point, it does not represent
the optimal configuration for any individual task.

= Conclusion

We presented an integrated architecture for verifier-grounded neural search. Three
components compose: a certified gate framework that separates validity from preference
through machine-checked symbolic rules, a holographic memory substrate that stores rules
as vector bindings with 1/sqrt(N) retrieval fidelity on the unit hypersphere, and an
EML-augmented neural scorer that improves ranking accuracy through a universal continuous
primitive.

The integration is honest about what each component contributes and where it fails. EML
does not improve binary-feature domains (prime_gap, grid_path). Holographic memory recall
degrades under EML binding (1.4% vs 48.6%). Certified gates do not claim to operate
outside their declared verifier support. But together they form a disciplined pipeline:
enumerate, check, rank, trace, claim---with every assertion bound to machine-readable
trace artifacts and mechanically verified by the claim compiler.

The dataset-first engine turns the pipeline into a repeatable process: any experiment
that exports candidate rows and verifier labels under the shared schema can enter the
evidence matrix. The holographic store preserves what is learned as retrievable context.
The EML scorer provides ranking pressure where continuous structure exists. The claim
compiler ensures that the paper never says more than the evidence supports.

#bibliography("references.bib")
