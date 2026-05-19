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
- `build/float/neural_grammar_discovery/arc_visual_pipeline_attempts.jsonl`
- `build/float/neural_grammar_discovery/arc_sketch_program_heldout_candidates.jsonl`
- `build/float/neural_grammar_discovery/arc_sketch_program_heldout_structure_proposals.jsonl`
- `build/float/neural_grammar_discovery/arc_sketch_program_heldout_sources.jsonl`
- `build/float/neural_grammar_discovery/arc_sketch_program_heldout_prior.tsv`
- `build/float/neural_grammar_discovery/arc_sketch_program_heldout_comparison.jsonl`
- `build/float/neural_grammar_discovery/arc_sketch_program_heldout_comparison.md`
- `build/float/neural_grammar_discovery/arc_visual_pipeline_heldout_attempts.jsonl`
- `build/float/neural_grammar_discovery/arc_visual_pipeline_proposals.jsonl`
- `build/float/neural_grammar_discovery/arc_visual_pipeline_neural_proposals.jsonl`
- `build/float/neural_grammar_discovery/arc_visual_pipeline_neural_summary.jsonl`
- `build/float/neural_grammar_discovery/arc_visual_pipeline_proposal_summary.jsonl`
- `build/float/neural_grammar_discovery/grammar_sketches.jsonl`
- `build/float/neural_grammar_discovery/grammar_sketch_examples.jsonl`
- `build/float/neural_grammar_discovery/grammar_sketch_proposals.jsonl`
- `build/float/neural_grammar_discovery/grammar_sketch_verification.jsonl`
- `build/float/neural_grammar_discovery/promoted_grammar_sketches.jsonl`
- `build/float/neural_grammar_discovery/grammar_sketch_memory.tsv`
- `build/float/neural_grammar_discovery/grammar_sketch_memory_summary.jsonl`

The Python driver only assembles source rows and reports. Delta compilation,
delta replay predicates, and the neural-linear sketch proposer run through the
C runtime at `build/float/neural_grammar_discovery_runtime`, with normalized
runtime TSVs kept under `build/float/neural_grammar_discovery/runtime/`.
The C proposer writes a checkpoint containing bias, token vocabulary, token
weights, epochs seen, and training examples seen. Use
`--resume-sketch-memory --sketch-memory <path>` to continue from a prior
checkpoint instead of starting the scorer from zero weights.
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
  selector, operation, geometry, color, and repeat parameters, then the C sketch
  VM materializes crop, normalized crop, object-mask extraction, lattice sample,
  diagonal sparse-lattice compaction, macrocell substitute, point-line
  projection, axis projection/compression, separator-panel mapping,
  palette-scale expansion, object-color tiling, marker-count motif tiling,
  component/color-count line compression, tile-repeat, and component-summary
  sketches for the existing train verifier.
- The newest ARC sketch ops execute through a typed internal visual-program
  interpreter, so composite behaviors are recorded as pipelines such as
  `select_object_color -> render_solid_rect` and
  `count_marker_cells -> select_motif_window -> repeat_window_by_count` rather
  than remaining opaque task-shaped functions. Probe source rows include
  `visual_program` and `slot_contract` metadata for auditability.
- The newer C miners enumerate typed visual pipeline specs first, then compile
  them into the stable `arc_sketch_program` slot ABI. This keeps the discovery
  surface closer to a learnable visual algebra while preserving the existing
  verifier and TSV/JSONL contracts.
- ARC writes accepted and rejected visual-pipeline attempts as JSONL before
  promotion. The attempt surface covers both newer pipeline-native ops and
  legacy successful sketch ops such as crop, macrocell substitution, lattice
  sampling, component summary, object-mask extraction, projection/compression,
  and separator mapping. Python trains frequency and nearest-neighbor baselines
  over those attempt rows using leave-one-task-out scoring. It also writes a
  normalized runtime TSV and asks the C runtime to train a neural visual-pipeline
  ranker on source attempts, then score held-out attempts. The ARC exporter
  links accepted attempts to same-task/same-op rejected siblings whose active
  slots differ by one or two values. The runtime uses those verifier-derived
  contrast pairs for a pairwise accepted-over-rejected training pass. Attempt
  rows also carry task/slot compatibility features such as input/output shape,
  scale, period divisibility, output-size matches, and color-presence checks.
  Rejected rows include verifier-distance telemetry: train pairs checked,
  unsupported/apply-failed/shape-mismatch counts, checked cells, cell
  mismatches, mismatch ratio, first failing pair, and a dense verifier score.
  The C ranker trains on dense verifier targets and then reranks held-out
  attempts with a learned score plus verifier-proxy distance score.
  The neural summary reports top-1/top-3/top-retained accepted-attempt recovery,
  contrast pair counts, pairwise violation counts, per-op recovery, and
  calibration by verifier rejection reason.
- Promoted ARC family pressure is exported by `arc_agi` itself. Python writes
  only promoted-family task rows, invokes the C probe exporter, and consumes the
  resulting `arc_sketch_program_probe_sources.jsonl` and
  `arc_sketch_program_probe_prior.tsv`. The exporter emits TSV priors through
  the existing `crlp.arc_neural_discovery.template_probe.v1` and
  `crlp.arc_neural_discovery.slot_probe.v1` contracts only after the C verifier
  proves a sketch exact on every train pair.
- ARC also runs a held-out triage slice from public coverage rows, excluding
  the failure-proof/source tasks. This produces separate held-out candidate,
  source, prior, comparison, and log artifacts so source-slice repair is not
  mistaken for broad ARC learning. The comparison report splits newly certified
  baseline-unsolved tasks from tasks that the baseline already certified.
- Held-out ARC candidate rows now carry pre-enumeration `op_hints`. Python
  computes a shape-conditioned structure proposal from source accepted ops,
  public task shape/color/separator features, and the promoted family, then
  writes `arc_sketch_program_heldout_structure_proposals.jsonl`. The C exporter
  uses those hints only to choose which generic visual-algebra miners to run;
  every emitted prior row still has to be exact on all train pairs before it can
  enter `arc_sketch_program_heldout_prior.tsv`.
- ARC probe sources include per-family rejection summaries for unsupported
  shapes, train mismatches, duplicates, and cap hits. These are diagnostic gap
  certificates; they are not executable promoted programs.
- Haiku generation compiles bottleneck candidates into executable trace
  constraints over strict-adjacency generation rows.
- Haiku understanding compiles corruption-family candidates into symbolic
  replay predicates and promotes only when the existing symbolic verifier selects
  `real` for every support row.

Until then, the engine reports `promoted=false` and a concrete
`blocking_reason`. This keeps the neural component in the discovery role and the
symbolic verifier in the certification role.
