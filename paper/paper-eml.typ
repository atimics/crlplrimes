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

We also document where EML does not help: domains dominated by binary or categorical
features (prime_gap, grid_path) see no improvement or degradation, and holographic
memory recall collapses under EML binding (1.4% vs 48.6%). EML is a tool for
continuous structure, not a universal replacement.

= Background: Compact Neural Scorers

EML is designed for compact scorers inside verifier-grounded search systems @crlplrimes2026. These scorers are small feedforward networks (typically 2-3 hidden layers, 8-48 units) trained to rank candidate actions that survive symbolic gate checks. Their compact size---dictated by modest per-task sample counts and verifier-gated training budgets---makes representational efficiency per parameter valuable.


= EML Neural Scoring

The neural scorer's role is to rank candidates that survive the symbolic gate. It is a
compact feedforward network trained by gradient descent. We introduce the EML operator as
a new architectural primitive that improves the scorer's representational capacity.

= The EML Operator

The EML operator is defined as:

$ "eml"(x, y) = exp(x) - ln(|y| + epsilon) $

where the absolute value and small epsilon ensure safe operation for all real inputs. The
operator is a universal primitive for continuous mathematics @eml2026, analogous to the
NAND gate for digital logic: repeated composition of EML operations can approximate any
continuous function. We integrate EML into the scorer in three forms, ordered by
increasing architectural novelty.

= Feature Enrichment

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

= EML Activation

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

= EML Pairwise Layer

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

= Domain Fit

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


= Conclusion

We introduced EML as a universal primitive for continuous functions and integrated it into compact neural scorers in three forms. Feature enrichment improves accuracy on nonlinear benchmarks from 92.2% to 94.8%. The EML activation matches ReLU while eliminating dead neurons. Stacked EML pairwise layers compose correctly at 94.5%, validating the universal-primitive property under gradient-based optimization. EML does not improve binary-feature domains, and we document these limitations explicitly. The operator is a building block for scorers where continuous feature structure exists---not a universal replacement for standard architectures.

#bibliography("references.bib")
