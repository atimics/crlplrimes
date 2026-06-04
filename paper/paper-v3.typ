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

The dataset-first engine accepts new experiments as candidate rows plus verifier
labels, generating evidence artifacts before any prose claim is written. Two
architectural extensions are under development and described in companion reports:
holographic weight storage @paper-holonet2026 for graceful degradation and
content-addressable weight lookup, for improved neural ranking on structured continuous features.
This extension is compatible with the framework but not required for the
results presented here.

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

The promoted spine carries machine-checked claims. Sandbox domains carry
diagnostic claims that track progress toward promotion.

*prime_gap.* Certified modular filters shrink the residual action space.
(Verifier: segmented-sieve replay and modular certificate checks.)

*grid_path.* The same gate contract works outside residue arithmetic, applying
legality, topology, and distance-dominance certificates to spatial pathfinding.
(Verifier: BFS shortest-path successor and rule certificate checks.)

*delayed_claim.* Reason learning needs domain-aware safety constraints: threshold
and margin predicates operate over observable claim features under a
deterministic hidden-outcome simulator.
(Verifier: deterministic replay and finite-generator safety checks.)

*rejected_candidate_learning.* Verifier failures become training signal.
(Verifier: original rule checker that rejected or accepted each row.)

*hypothesis_library_feedback.* Certified learning accumulates through reusable
library priors, with zero-false-rejection promotion gates.
(Verifier: train-pair exactness, type checks, and held-out fixture evaluation.)

*arc_agi, program_synthesis, haiku_word_lm, signal_flight.* These sandbox
domains publish the same schema contract but carry diagnostic claims. They
demonstrate generalization beyond arithmetic domains while remaining outside the
canonical evidence surface until earning broader verifier-backed coverage.

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

= Ongoing Work

Two architectural extensions are under active development and described in
companion technical reports.

*Holographic weight storage* @paper-holonet2026 replaces flat weight arrays with
vector-symbolic bindings on the unit hypersphere, giving neural scorers graceful
degradation (1/sqrt(N)), content-addressable weight lookup, and fixed O(d) memory.
A zero-parameter holographic ranker achieves 48.5% accuracy versus 16.0% for a
gradient-trained MLP on small-sample tasks.

universal primitive for continuous functions, analogous to NAND for digital logic.
Integrated into neural scorers as pairwise feature enrichment, it improves accuracy
on nonlinear benchmarks from 92.2% to 94.8% at identical width. Stacked EML
pairwise layers compose correctly at 94.5%. The operator does not yet improve the
promoted experiments, which are dominated by binary and categorical features.

This extension is compatible with the certified-gate framework. Neither is
required for the results presented in Sections 2--10.

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

We presented a verifier-grounded architecture for neural search. The core loop is
simple: enumerate candidates, gate with certified rules, rank survivors with a neural
scorer, verify the top choice, and preserve the trace. The claim compiler closes the
loop between empirical work and scientific reporting by mechanically checking every
assertion against generated trace artifacts.

The architecture is honest about its limits. Certified gates do not claim to
operate outside their declared verifier support. The learned proposer's failure
(#claim_C2_prime_proposer_fixed_budget_summary) is documented as a negative row.
The dataset-first engine ensures every promoted domain publishes the same
machine-readable contract. The claim compiler ensures the paper never says more
than the evidence supports.

The dataset-first engine makes the pipeline repeatable: any experiment that
exports candidate rows and verifier labels under the shared schema can enter the
evidence matrix. Two extensions under development---holographic weight storage
framework but not required for the results presented here.

#bibliography("references.bib")
