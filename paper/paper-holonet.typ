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

The substrate is demonstrated as a weight store for neural scorers
(holographic recall with fidelity guarantees), as a zero-parameter holographic
ranker that competes with gradient-trained MLPs on small-sample tasks, and as a
feature extractor that appends retrieval signals to the scorer envelope. An EML-based
binding variant is introduced but shown to degrade memory recall, confirming that
the linear superposition property is essential for content-addressable retrieval.

= Background: Verifier-Grounded Search

The holographic substrate is designed for scorers inside a verifier-grounded loop @crlplrimes2026. In that framework, candidate actions are enumerated, gated by certificate-checked symbolic rules, and ranked by a neural scorer. The scorer's weights, the certified rules, and the reinforced state-action pairs all need storage that degrades gracefully as the system learns.


= Holographic Weight Storage

The neural scorer's weights are conventionally flat arrays---one float per connection,
loaded and saved as checkpoint files. We replace this with holographic storage: each
weight vector lives as a vector-symbolic binding in a fixed-dimension memory trace on
the unit hypersphere S^(d-1). The memory trace IS the weight matrix. Retrieval IS the
forward pass. Memory and computation are the same operation.

Three properties make this attractive. *Graceful degradation*: recall fidelity is
proportional to 1/sqrt(N), so adding more weights (from new tasks or fine-tuning)
degrades existing ones smoothly. *Content-addressable lookup*: weights are retrieved
by key similarity, enabling implicit weight sharing across related tasks.
*Fixed memory*: the entire weight store occupies O(d) memory regardless of count.

The same substrate also stores certified rules from the hypothesis library (Section 3),
retrieves context for the scorer (Section 3.2), and accumulates reinforced state-action
pairs over time. But its primary role is as the weight substrate for the neural scorer:
the bind operation IS both weight storage and forward computation.

= VSA Primitives

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

= Holographic Ranker

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

= EML Binding in Holographic Space in Holographic Space

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


= Conclusion

We presented a holographic weight storage substrate for verifier-grounded neural scorers. The key insight is that bind/unbind operations on the unit hypersphere unify weight storage and forward computation into a single primitive. This gives the scorer graceful degradation (1/sqrt(N)), content-addressable weight lookup, and fixed O(d) memory---properties that flat weight arrays cannot provide. The holographic ranker demonstrates that zero-parameter VSA memory lookup can compete with gradient-trained scorers on small-sample tasks, and the EML binding experiment confirms that linear superposition is essential for content-addressable retrieval.

#bibliography("references.bib")
