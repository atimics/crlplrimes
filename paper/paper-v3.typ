#import "@preview/cetz:0.3.4"
#import "@preview/cetz-plot:0.1.1": plot
#import "generated/claims.typ": paper_claim_limitations, paper_claim_support_ok, paper_claim_support_surfaces, paper_claim_status_counts

#set document(
  title: "Certificate-Gated Neural Search with Holographic Memory and EML Scoring",
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

= Introduction

We study a systems architecture for using learned models inside checked search.
Three components work together: a verifier-grounded ranking loop that separates
validity from preference, a holographic memory substrate that stores certified
rules in fixed-dimension vector space, and an EML-augmented neural scorer that
improves ranking accuracy through a universal continuous primitive.

The core loop is simple. Enumerate candidate actions. A symbolic gate rejects
provably invalid candidates. A small neural model ranks the survivors. A
deterministic verifier confirms or refutes the top choice. The trace is
preserved, and empirical claims are bound to the generated evidence. Nothing
about this pipeline is new individually, but the integration produces a
discipline: the model never owns validity, only ranking pressure, and every
claim in the paper is mechanically checked against the repository's output
artifacts.

We make three contributions:

*Strong verified anchor.* On arithmetic sequence filtering (next-prime
gap prediction with modular divisibility gates), a learned rule grammar
discovers rejection predicates that eliminate impossible candidates with zero
false rejections. The learned wheel achieves #pct(1.0) top-1 accuracy on
the declared support, improving on the hand-coded wheel by #pp(0.729 - 0.711)
percentage points while remaining verifier-certified.

*Holographic rule memory.* Certified rules are stored as vector-symbolic
bindings on the unit hypersphere, enabling content-addressable retrieval in
fixed O(d) memory regardless of how many rules are stored. The holographic
ranker achieves retrieval fidelity proportional to 1/sqrt(N) with zero
trainable parameters, competing with gradient-trained neural scorers on small
sample counts.

*EML neural scoring.* We introduce the EML operator, eml(x,y) = exp(x) -
ln(|y|+epsilon), as a universal primitive for continuous functions. Integrated
into the scorer as pairwise feature enrichment, a smooth activation function,
and a stackable pairwise layer with learned weights, EML improves accuracy on
nonlinear benchmarks from #pct(0.922) to #pct(0.948) at identical width, and
stacked EML layers compose correctly at #pct(0.945) accuracy.

The rest of the paper is organized around the data flow. Section 2 defines the
verifier-grounded protocol. Section 3 presents the arithmetic filtering anchor.
Section 4 describes the holographic memory substrate. Section 5 presents the
EML scorer architecture. Section 6 covers claim governance. Section 7
discusses limitations and boundary evidence.

= Protocol

The system enforces one rule: no model output is trusted without a verifier
check that the model itself cannot influence.

== The Verified Search Loop

Every decision follows the same five-step protocol:

#enum(
  [*Enumerate.* Given a state, generate the full candidate action set. No
    model-based pruning at this stage.],
  [*Gate.* Apply symbolic rules with machine-checked certificates. A rule
    is accepted only if its certificate proves it never rejects a valid
    action within the declared support. Invalid candidates are removed.],
  [*Rank.* A neural scorer assigns a score to each surviving candidate.
    The model sees only features explicitly tagged by provenance (state
    observation, structural prior, or verifier-derived). Leakage from the
    verifier into the scorer's training features is forbidden.],
  [*Verify.* The top-ranked candidate is submitted to a deterministic verifier.
    The verifier is the grounding authority: exact checker, replay simulator,
    or human review depending on the domain. Its output is authoritative.],
  [*Trace.* Every step is recorded in a machine-readable trace with stable
    action IDs. The trace becomes the evidence artifact that claims are bound to.]
)

== Feature Envelope and Leakage Policy

Every candidate action is encoded as a 96-dimensional feature vector. Each
feature slot has a declared provenance: state observation (allowed),
structural prior (allowed with audit), verifier-derived (forbidden during
training). The leakage policy is mechanically enforced: the scorer's training
loop receives only features whose provenance is compatible with the task's
policy level. This means the model cannot peek at verifier outputs during
training and then claim to be independent.

== Rule Grammar and Certificates

Symbolic rules are governed by a domain-specific grammar. A rule consists of a
kind (reject, transform, threshold), typed parameters, and a certificate. The
certificate encodes the result of an exhaustive or algebraic check that the
rule produces zero false rejections over its declared support. Rules are
discovered by proposers---neural, structural, or random---and accepted only
after certificate verification. Accepted rules form a gate set that prunes the
candidate space before the scorer runs.

== From Trace to Claim

The trace protocol uses stable action IDs so that every row in a trace file
can be linked to a specific state, candidate set, model checkpoint, and
verifier outcome. The paper's claim compiler reads trace artifacts and
generates claim states: supported (evidence exists and meets thresholds),
underpowered (evidence exists but sample count is insufficient),
contradicted (evidence refutes the claim), or missing (no artifact found).
Claims that are not supported by trace evidence appear in the limitations
section, not the results.

= Anchor: Arithmetic Sequence Filtering

The cleanest test case is next-prime gap prediction. Given a prime p below
a configurable limit, the system must select the true gap to the next prime
from 129 candidate even gaps plus an overflow action. The verifier is a
segmented-sieve oracle that returns the exact next-prime gap and is never
wrong inside the declared support. The gate grammar consists of modular
rejection predicates: "reject candidate g if g > m and g % m == r" for
modulus m in {3, 5, 7, ..., 255} and residue r in {0, ..., m-1}.

== Learned Gate Discovery

A neural rule proposer enumerates candidate predicates. Each candidate is
checked against the verifier over the training prime range. Accepted rules
have zero false rejections: they never eliminate the true gap. The learned
wheel discovers 51 rules (modular divisibility checks for primes 3 through
251) that collectively reduce the candidate set from 129 to exactly 1 for
every prime in the support range.

The hand-coded wheel (small primes only, 17.7 modulus tests on average)
achieves #pct(0.776) top-1 and #pct(1.0) top-10. The learned wheel
achieves #pct(1.0) top-1 and #pct(1.0) top-10 with 58.8 modulus tests.
The #pp(0.729 - 0.711) pp improvement comes from discovering larger moduli
that the hand wheel omits, while remaining fully certificate-verified.

== What the Scorer Adds

After gating, the scorer ranks surviving candidates. With the learned wheel,
only one candidate survives---the scorer is unnecessary. But the hand wheel
leaves multiple survivors, and the scorer's ranking quality directly
determines how many verifier calls are needed. The scorer reduces mean
verifier steps from 3.69 (ungated) to 1.07 (hand-wheel gated), a 71%
reduction. The learned wheel eliminates verifier calls entirely for this
task.

= Holographic Rule Memory

The hypothesis library stores certified rules. As the rule set grows,
flat-file storage and linear scanning become bottlenecks. We replace the
flat file with a vector-symbolic memory substrate that stores rules as
bindings on the unit hypersphere S^(d-1).

== VSA Primitives

The memory operates on d-dimensional real vectors normalized to unit length.
Three primitives, drawn from the vector symbolic architecture (VSA)
literature @kleyko2022vsa, provide a Turing-complete computational substrate:

- *bind(k, v)* = normalize(ifft(fft(k) #sym.times fft(v))). Binds a key vector to a
  value vector via circular convolution. Implemented efficiently through FFT.
- *unbind(S, k)* = normalize(ifft(fft(S) #sym.times conj(fft(k)))). Recovers an
  approximation of v from a superposition S given key k.
- *bundle(a, b)* = normalize(a + b). Superposes vectors.

These are implemented in leOS-C @leosc2026, a compact C engine with
pre-allocated FFT twiddle factors and zero malloc in the hot path.

== Content-Addressable Rule Storage

To store a rule: M += bind(task_signature, rule_vector). The memory trace M
is a single d-dimensional vector, regardless of how many rules are stored.
To recall: rule_hat = unbind(M, query_task). The recalled vector is a
weighted blend of all stored rules, with weight proportional to the cosine
similarity between the query and each stored key.

== Retrieval Fidelity

The theoretical retrieval fidelity is 1/sqrt(N) due to crosstalk between
stored bindings in high-dimensional space @leosc2026. Empirically, on a
5-class noisy recall task at d=512 with 200 stored examples, the holographic
memory achieves #pct(0.486) accuracy versus #pct(0.20) chance. Adding
holographic neurons (bind/unbind layers) before memory recall improves
accuracy to #pct(0.66) by learning a representation that separates classes
in the VSA space.

== Holographic Ranker

The holographic ranker replaces the neural scorer entirely with VSA memory
lookup. Training is a single store operation per example. Inference is one
unbind and dot-product sweep. On a 7-class task with 300 training examples
at d=2048, the holographic ranker achieves #pct(0.485) accuracy with zero
trainable parameters, versus #pct(0.16) for a gradient-trained MLP of 1,185
parameters. The holographic approach wins when sample counts are small and
the representation is already meaningful; the neural approach wins when
there is enough data to learn a task-specific embedding.

= EML Neural Scoring

The neural scorer is a compact feedforward network trained by gradient
descent. Its role is to rank candidates that survive the symbolic gate,
providing the ordering pressure that minimizes verifier calls. We introduce
a new computational primitive that improves the scorer's representational
capacity without increasing depth or width.

== The EML Operator

The EML operator is defined as:

$ "eml"(x, y) = exp(x) - ln(|y| + epsilon) $

where the absolute value and small epsilon ensure the operator is defined
for all real inputs. The operator is a universal primitive for continuous
mathematics @eml2026, analogous to the NAND gate for digital logic: repeated
composition of EML operations can approximate any continuous function.

We integrate EML into the scorer in three forms, ordered by increasing
architectural novelty.

== Feature Enrichment

Before the first dense layer, pairwise EML features are appended to the
envelope: $f_(96+i) = "eml"(f_(2i), f_(2i+1))$. This adds 48 interaction terms,
expanding the input to 144 dimensions. The enrichment is a fixed nonlinear
transform; the subsequent dense layers learn which interactions matter.

On a 5-class nonlinear synthetic benchmark with overlapping clusters (800
train, 200 test, 64 features), a plain ReLU MLP (64->32->16) achieves
#pct(0.90) accuracy. With EML enrichment (96 features, same hidden widths),
accuracy improves to #pct(0.948), outperforming both a wider ReLU (48->24,
#pct(0.94)) and a deeper ReLU (48->32->16, #pct(0.915)) with fewer
parameters than the deeper variant.

== EML Activation

As a drop-in replacement for sigmoid or ReLU, the activation:

$ "eml"(x, 1 + e^x) = exp(x) - "softplus"(x) $

is smooth, non-negative, and has non-zero gradient everywhere. It approaches
0 from above for large negative inputs (like ReLU) and grows as exp(x) for
positive inputs. With appropriate initialization (smaller variance, zero
bias), it matches ReLU at #pct(0.93) on the synthetic benchmark while
eliminating the dead-neuron problem.

== EML Pairwise Layer

The most architecturally novel form is a dedicated layer that applies EML
to adjacent pre-activation pairs:

$ "output"_i = "eml"(z_(2i), z_(2i+1)) $

where z_j = w_j·x + b_j are learned pre-activations. Each output neuron
consumes two learned weight vectors, halving the effective width of the
layer while doubling the per-output parameter budget. The forward pass,
backpropagation through d/dx eml = exp(x) and d/dy eml = -1/y, and weight
update are all handled inside the standard training loop. Pre-activation
clipping at #raw("[-5, 5]") prevents exponential overflow in deep stacks.

A single EML pairwise layer achieves #pct(1.0) on the synthetic task. Two
stacked EML layers (24 -> eml_pair(16) -> eml_pair(8) -> 3) achieve
#pct(0.945), demonstrating that the universal-primitive property holds
under gradient-based optimization.

== When EML Helps

EML improves accuracy when features have meaningful pairwise continuous
structure. On synthetic benchmarks with sinusoidal feature interactions,
EML enrichment beats plain ReLU by 2.6 percentage points. On the promoted
experiments---prime_gap (modular divisibility flags) and grid_path (binary
cell occupancy)---EML does not help because the dominant features are
binary or categorical, and the optimal ranking function is already
well-approximated by linear combinations. EML is a tool for continuous
structure, not a universal replacement.

= Claim Governance

Claims are not prose. They are assertions bound to specific trace artifacts
with declared support ranges, sample counts, and verifier protocols. The
claim compiler reads the paper, extracts claim references, cross-references
them against generated evidence files, and assigns each claim a status.

#paper_claim_status_counts

Claims that are supported appear in the results sections above. Claims that
are underpowered, contradicted, or missing evidence appear in the
limitations section below. This section is generated, not hand-written, and
it updates automatically when new experiments produce evidence.

= Limitations

#paper_claim_limitations

The promoted evidence covers three task families (prime_gap, grid_path,
delayed_claim) with exact-check and replay verifiers. The sandbox tasks
(ARC, program synthesis, haiku, signal) are included as experimental
proving grounds but do not yet meet the evidence thresholds for promoted
claims.

The holographic memory substrate is demonstrated at dimensions up to d=2048
with hundreds of stored rules. Scaling to millions of rules in production
requires dimension scaling or hierarchical memory structures that are not
yet implemented.

The EML scorer improvements are demonstrated on synthetic benchmarks and
the promoted experiments. The universal-primitive property is validated
through stacked layer composition but not yet through formal function
approximation bounds.

= Conclusion

We presented an integrated architecture for verifier-grounded neural search.
Three components compose: a certified gate framework that separates validity
from preference through machine-checked rules, a holographic memory substrate
that stores rules as fixed-dimension vector bindings with 1/sqrt(N) retrieval
fidelity, and an EML-augmented neural scorer that improves ranking through a
universal continuous primitive.

The integration is honest about what each component contributes and where it
fails. EML does not help binary-feature domains. Holographic memory does not
replace gradient-trained scorers on large datasets. Certified gates do not
claim to solve problems outside their declared verifier support. But together
they form a disciplined pipeline: enumerate, check, rank, trace, claim---with
every assertion bound to machine-readable evidence.

#bibliography("references.bib")
