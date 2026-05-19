# Neural Grammar Discovery

`scripts/neural_grammar_discovery.py` is the shared engine layer for turning
neural failure pressure into symbolic grammar-delta candidates and typed
`GrammarSketch` rows. It is deliberately not a grammar by itself. It emits
candidate deltas, sketch examples, proposer rows, verifier rows, and promotion
ledgers that stay empty until a domain adapter compiles a candidate into
executable grammar and a verifier certifies it.

Run the current adapters:

```sh
make neural-grammar-discovery HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR=/path/to/ARC-AGI/data
```

The target consumes:

- ARC failure-proof rows from
  `build/float/hypothesis_library_neural_discovery/failure_proof/task_proof_rows.jsonl`
- haiku word-generation traces from
  `build/float/haiku_word_lm_artifacts/smoke_trace.jsonl`
- haiku understanding traces from
  `build/float/haiku_understanding_artifacts/smoke_trace.jsonl`
- program-synthesis hypothesis evidence rows from
  `build/float/neural_grammar_discovery/program_synthesis_source.jsonl`
- grid-path training traces from
  `build/float/grid_path.jsonl`
- signal-mining deterministic counterfactual traces from
  `build/float/signal_mining_smoke/trace.jsonl`

Outputs:

- `build/float/neural_grammar_discovery/grammar_delta_candidates.jsonl`
- `build/float/neural_grammar_discovery/compiled_grammar_deltas.jsonl`
- `build/float/neural_grammar_discovery/grammar_delta_verification.jsonl`
- `build/float/neural_grammar_discovery/promoted_grammar_deltas.jsonl`
- `build/float/neural_grammar_discovery/arc_sketch_program_probe_sources.jsonl`
- `build/float/neural_grammar_discovery/arc_sketch_program_probe_prior.tsv`
- `build/float/neural_grammar_discovery/arc_sketch_program_probe_comparison.jsonl`
- `build/float/neural_grammar_discovery/arc_sketch_program_probe_comparison.md`
- `build/float/neural_grammar_discovery/grammar_sketches.jsonl`
- `build/float/neural_grammar_discovery/grammar_sketch_examples.jsonl`
- `build/float/neural_grammar_discovery/grammar_sketch_proposals.jsonl`
- `build/float/neural_grammar_discovery/grammar_sketch_verification.jsonl`
- `build/float/neural_grammar_discovery/promoted_grammar_sketches.jsonl`

The Python driver only assembles source rows and reports. Delta compilation,
delta replay predicates, and the neural-linear sketch proposer run through the
C runtime at `build/float/neural_grammar_discovery_runtime`, with normalized
runtime TSVs kept under `build/float/neural_grammar_discovery/runtime/`.
- `build/float/neural_grammar_discovery/summary.jsonl`
- `build/float/neural_grammar_discovery/summary.md`

## Contract

The shared engine owns common schemas:

- `crlp.neural_grammar_discovery.delta_candidate.v1`
- `crlp.neural_grammar_discovery.compiled_delta.v1`
- `crlp.neural_grammar_discovery.delta_verification.v1`
- `crlp.neural_grammar_discovery.promoted_delta.v1`
- `crlp.neural_grammar_discovery.summary.v1`
- `crlp.grammar_sketch.v1`
- `crlp.grammar_sketch.example.v1`
- `crlp.grammar_sketch.proposal.v1`
- `crlp.grammar_sketch.verification.v1`
- `crlp.grammar_sketch.promoted.v1`

Each domain adapter owns only translation from local trace rows into candidate
grammar-delta families:

- ARC maps neural failure-proof clusters to missing template-family candidates
  such as patch/lattice, separator-panel extraction, tiling/canvas growth, crop
  extraction, projection compression, and composition.
- Haiku maps word-generation bottlenecks and understanding near-miss failures to
  constraint-family candidates such as latent adjacency-class transitions,
  continuation viability, line shape, line-source coherence, topic coherence,
  and content co-occurrence.
- Program synthesis maps certified and rejected hypothesis evidence into typed
  DSL abstraction sketches.
- Grid path maps shortest-path verifier corrections into compass-move sketches.
- Signal mining maps deterministic counterfactual policy rows into tactical
  option sketches.

`GrammarSketch` rows are a typed interchange artifact, not a verifier bypass.
They contain a domain, family, compiler target, typed slots, structural atoms,
optional composition, source artifact references, and promotion status. Existing
accepted/rejected traces become `grammar_sketch_examples.jsonl` rows with
outcomes such as `accepted`, `rejected`, `unsafe`, and `no_support`.

The first proposer stage is intentionally simple:

- `frequency_sketch_proposer` ranks sketches by accepted support penalized by
  rejected, unsafe, and no-support examples.
- `nearest_neighbor_sketch_proposer` maps rejected examples to the most similar
  accepted sketch within the same domain by token overlap.
- `neural_linear_sketch_proposer` trains a tiny logistic ranker over sketch
  tokens and emits neural proposal scores over the same sketch surface.

Promotion requires all of the following:

1. the candidate has a domain compiler,
2. the compiler emits executable symbolic grammar,
3. verifier replay certifies the executable delta on train evidence,
4. held-out checks do not regress.

The first ARC/haiku promotion paths are intentionally narrow:

- ARC compiles missing-template-family candidates into replayable diagnostic gap
  certificates. Promotion requires the existing ARC failure-proof replay to show
  no accepted existing rows, missing-family evidence, a completed promotion
  replay, and zero train-consistent current-template proposals.
- ARC execution uses one generic C hypothesis template, `arc_sketch_program`,
  rather than one bespoke template per promoted family. Its slots encode a
  selector, operation, geometry, color, and repeat parameters, then a small
  interpreter materializes crop and tile-repeat sketch programs for the existing
  train verifier.
- Promoted ARC family pressure is also compiled into `arc_sketch_program` TSV
  probe rows. The TSV is loaded by `arc_agi` through the existing
  `crlp.arc_neural_discovery.template_probe.v1` and
  `crlp.arc_neural_discovery.slot_probe.v1` contracts, then compared against a
  no-prior baseline under the same verifier flags.
- Haiku generation compiles bottleneck candidates into executable trace
  constraints over strict-adjacency generation rows.
- Haiku understanding compiles corruption-family candidates into symbolic
  replay predicates and promotes only when the existing symbolic verifier selects
  `real` for every support row.

Until then, the engine reports `promoted=false` and a concrete
`blocking_reason`. This keeps the neural component in the discovery role and the
symbolic verifier in the certification role.
