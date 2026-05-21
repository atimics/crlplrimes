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
make learned-grammar-coverage
make experiment-health
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
- `build/float/neural_grammar_discovery/arc_visual_pipeline_coverage_frontier.jsonl`
- `build/float/neural_grammar_discovery/arc_visual_pipeline_coverage_frontier_summary.jsonl`
- `build/float/neural_grammar_discovery/arc_sketch_program_heldout_repair_candidates.jsonl`
- `build/float/neural_grammar_discovery/arc_sketch_program_heldout_repair_sources.jsonl`
- `build/float/neural_grammar_discovery/arc_sketch_program_heldout_repair_prior.tsv`
- `build/float/neural_grammar_discovery/arc_visual_pipeline_heldout_repair_attempts.jsonl`
- `build/float/neural_grammar_discovery/arc_visual_pipeline_repair_sketches.jsonl`
- `build/float/neural_grammar_discovery/arc_visual_pipeline_repair_sketch_summary.jsonl`
- `build/float/neural_grammar_discovery/arc_visual_pipeline_repair_gain_summary.jsonl`
- `build/float/neural_grammar_discovery/arc_sketch_program_heldout_sources.jsonl`
- `build/float/neural_grammar_discovery/arc_sketch_program_heldout_prior.tsv`
- `build/float/neural_grammar_discovery/arc_sketch_program_heldout_comparison.jsonl`
- `build/float/neural_grammar_discovery/arc_sketch_program_heldout_comparison.md`
- `build/float/neural_grammar_discovery/arc_visual_pipeline_heldout_attempts.jsonl`
- `build/float/neural_grammar_discovery/arc_visual_pipeline_proposals.jsonl`
- `build/float/neural_grammar_discovery/arc_visual_pipeline_neural_proposals.jsonl`
- `build/float/neural_grammar_discovery/arc_visual_pipeline_neural_summary.jsonl`
- `build/float/neural_grammar_discovery/arc_visual_pipeline_proposal_summary.jsonl`
- `build/float/learned_grammar_coverage/arc_visual_compare.jsonl`
- `build/float/learned_grammar_coverage/arc_visual_curriculum/arc_visual_curriculum_examples.jsonl`
- `build/float/learned_grammar_coverage/arc_visual_curriculum/arc_visual_curriculum_attempts.jsonl`
- `build/float/learned_grammar_coverage/arc_visual_curriculum/arc_visual_curriculum_prior.tsv`
- `build/float/learned_grammar_coverage/arc_visual_active_curriculum/summary.jsonl`
- `build/float/learned_grammar_coverage/arc_visual_capacity_sweep/capacity_sweep.jsonl`
- `build/float/learned_grammar_coverage/arc_visual_capacity_sweep/capacity_sweep.md`
- `build/float/neural_grammar_discovery/grammar_sketches.jsonl`
- `build/float/neural_grammar_discovery/grammar_sketch_examples.jsonl`
- `build/float/neural_grammar_discovery/grammar_sketch_proposals.jsonl`
- `build/float/neural_grammar_discovery/grammar_sketch_verification.jsonl`
- `build/float/neural_grammar_discovery/promoted_grammar_sketches.jsonl`
- `build/float/neural_grammar_discovery/grammar_sketch_memory.tsv`
- `build/float/neural_grammar_discovery/grammar_sketch_memory_summary.jsonl`
- `build/float/learned_grammar_coverage/coverage.jsonl`
- `build/float/experiment_health/experiment_health.jsonl`

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
- `crlp.neural_grammar_discovery.arc_visual_pipeline_coverage_frontier.v1`
- `crlp.neural_grammar_discovery.arc_visual_pipeline_coverage_frontier_summary.v1`
- `crlp.neural_grammar_discovery.arc_visual_pipeline_repair_sketch.v1`
- `crlp.neural_grammar_discovery.arc_visual_pipeline_repair_sketch_summary.v1`
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
  and content co-occurrence. It also emits diagnostic grammar-sketch examples
  from generation and understanding traces so those pressure rows participate in
  the same temporal-frame proposer as other domains.
- Program synthesis maps certified and rejected hypothesis evidence into typed
  DSL abstraction sketches.
- Grid path maps shortest-path verifier corrections into compass-move sketches.
- Signal mining maps deterministic counterfactual policy rows into tactical
  option sketches.

`GrammarSketch` rows are a typed interchange artifact, not a verifier bypass.
They contain a domain, family, compiler target, typed slots, structural atoms,
optional composition, source artifact references, and promotion status. Existing
accepted/rejected traces become `grammar_sketch_examples.jsonl` rows with
outcomes such as `accepted`, `rejected`, `unsafe`, and `no_support`. Diagnostic
pressure rows may use `diagnostic_positive`; those rows train proposal ranking
but never promote a sketch without verifier certification.

Every sketch example also carries a shared `crlp.neural_frame.v1` context. A
frame gives the example's sequence id, sequence position, previous sketch
summary, next sketch summary, and local composition context. The runtime TSVs
pass both sketch-level `composition` tokens and example-level `frame` tokens to
the neural-linear proposer. This makes temporal influence an engine feature:
haiku line/poem transitions, grid action traces, signal option groups, and
program-composition rows all expose previous/current/future context through the
same proposal surface.

The first proposer stage is intentionally simple:

- `frequency_sketch_proposer` ranks sketches by accepted support penalized by
  rejected, unsafe, and no-support examples.
- `nearest_neighbor_sketch_proposer` maps rejected examples to the most similar
  accepted sketch within the same domain by token overlap.
- `neural_linear_sketch_proposer` trains a tiny logistic ranker over sketch,
  composition, and neural-frame tokens, then emits proposal scores over the same
  sketch surface.

Promotion requires all of the following:

1. the candidate has a domain compiler,
2. the compiler emits executable symbolic grammar,
3. verifier replay certifies the executable delta on train evidence,
4. held-out checks do not regress.

For ARC, the target ownership model is learned visual programs, not handcrafted
transform templates. Legacy transforms may remain as seed operators, examples,
or compiler backends, but they should not be the abstraction that owns new
coverage. The audit surface therefore separates `arc_sketch_program` generic
visual-program attempts from legacy template attempts. The existing regression
gate still reports whether any verified candidate exists, while the replacement
gates give credit only to generic visual programs; legacy template matches stay
visible as training evidence and replacement debt. If a selected solution is
still owned by a legacy template, `experiment_health` reports
`handcrafted_transform_dependency`; if public tasks have generated attempts but
no train-consistent generic program, it reports
`learned_visual_program_coverage_gap`. That makes the replacement pressure
explicit: learn coverage, composition, and slot synthesis over the visual
algebra, then compile through C and certify with the existing train verifier.

The learned-grammar coverage regression surface makes that sequence explicit for
ARC and haiku. `crlp.learned_grammar_coverage.v1` rows separate candidate-space
coverage from ranker quality, verifier certification, and held-out/test-pair
transfer. ARC rows are verifier-backed through `arc_agi`; haiku rows currently
use the corpus-adjacency and real-vs-corrupt benchmark traces as proxy authority,
so they may train the proposer but do not support semantic or promotion claims.
`crlp.experiment_health.v1` rows then fold coverage status, numeric certificate
rows, coverage holes, verifier deltas, and generated paper-claim support into one
per-experiment JSONL report.

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
  tolerant separator-intersection color extraction, separator-panel relation
  masks, palette-scale expansion, object-color tiling, marker-count motif
  tiling, component/color-count line compression, tile-repeat, and
  component-summary sketches for the existing train verifier.
- ARC sketch search now exposes a shared compositional grammar above the stable
  `sketch.op` ABI. Each executable attempt carries `composition_kind`,
  `normalization_frame`, `typed_stages`, and `primitive_contract`; examples are
  `region_extraction + object_bbox_frame`,
  `lattice_transform + macrocell_frame`,
  `panel_composition + separator_panel_frame`,
  `panel_composition + separator_relation_frame`, and
  `count_render + marker_count_frame`. The enum op remains only the compiler
  target that lets the existing verifier replay the sketch exactly.
- The C miners enumerate typed visual pipeline specs first, then compile them
  into the stable `arc_sketch_program` slot ABI. Shape-conditioned held-out
  hints are now grammar hints over composition kinds, normalization frames, and
  stage names rather than one-off op names. The C exporter accepts those hints
  and runs every compatible generic miner. This keeps the discovery surface a
  learnable visual algebra while preserving the existing verifier and TSV/JSONL
  contracts.
- The visual algebra has a feedback surface, not just static labels. Each
  accepted or rejected attempt is bucketed by `failure_stage` and
  `failure_detail`, for example `renderer_shape`, `selector_or_transform`, or
  `transform_or_color`. Python scores grammar hints from source attempts using
  acceptance rate, dense verifier score, and failure-stage penalties, then
  writes the selected hint evidence into
  `arc_sketch_program_heldout_structure_proposals.jsonl`.
- ARC coverage is now treated as its own frontier. After the first held-out
  export, Python groups zero-accepted tasks, keeps the best verifier-distance
  near misses, classifies residuals such as `frame_or_output_shape`,
  `selector_or_phase`, `transform_or_color`, and `cell_render_cleanup`, then
  writes `arc_visual_pipeline_coverage_frontier.jsonl`. Those rows generate a
  second-pass repair candidate file with residual-expanded grammar hints. The
  repair pass may broaden attempts, but still contributes executable probes only
  when `arc_agi` proves exact train consistency.
- Zero-accepted frontier tasks also become counterexample-guided repair
  sketches in `arc_visual_pipeline_repair_sketches.jsonl`. These rows bind the
  best failed parent attempt to C-emitted residual telemetry: first failing
  train pair, expected/produced shape, shape delta, checked cells, mismatch
  count, and mismatch bbox. Python can name typed holes such as frame
  normalization, selector phase, color binding, or residual-mask cleanup, but
  promotion still requires an executable `arc_agi` compiler/verifier path.
- Repair candidates carry provenance through the TSV handoff into C-emitted
  probe and attempt rows: `coverage_frontier_id`, parent attempt id, repair
  operator, dominant residual class/stage, and repair strategy. The fixed
  13-field prior ABI is unchanged; provenance stays on JSONL evidence rows.
  `arc_visual_pipeline_repair_gain_summary.jsonl` reports first-pass coverage,
  repair-pass accepted tasks, newly repaired frontier tasks, accepted repair
  ops, and repair rejection counts.
- Two counterexample repair compilers now live in `arc_agi` as generic
  `arc_sketch_program` ops. `recolor_crop` composes region selection with
  one-step color/mask cleanup, filtered-bbox selection, object-selector-aware
  bbox crops, and small bbox-relative framing repairs. `marker_polyline`
  composes singleton marker extraction, learned marker ordering, spatial and
  nearest-neighbor marker orderings, and orthogonal/perimeter route rendering.
  Both are enumerated from repair hints and only emit prior rows after exact
  train consistency.
- The generic ARC VM also exposes richer C-side coverage basis functions for
  repair proposals: macrocell substitution has additional motif bases such as
  cross, line, diagonal, checker, inverse-center, and inner-fill masks, and
  object-mask extraction can emit framed crops with learned width, height, and
  bbox-relative offsets instead of only tight object boxes.
- The newest generic visual-combinator compiler treats small visual holes as
  reusable programs rather than new task families. Its first two subprograms
  are `select_components -> normalize_component_origins -> emit_occupancy_grid`
  and `select_components -> count_components -> emit_count_template`. The
  latter turns component counts into verifier-checked template masks such as
  corner/center orders, which gives the learner an executable count-to-shape
  candidate without adding a task-specific rule.
- The panel-composition basis now includes executable relation sketches, not
  only panel extraction. `separator_intersection_color` handles tolerant
  row/column separator intersections that emit a single color cell, while
  `separator_panel_relation_mask` aligns the two panels around a full separator
  and emits verifier-checked OR/AND/XOR/background/left-only/right-only masks.
  These are still generic sketch compilers: they become priors only when
  `arc_agi` proves exact train consistency.
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
  mismatches, mismatch ratio, first failing pair, counterexample shape/bbox
  telemetry, and a dense verifier score.
  The C ranker trains on dense verifier targets and then reranks held-out
  attempts with a learned score plus verifier-proxy distance score.
  It also trains auxiliary heads for renderer correctness, sequence/block
  renderer correctness, and component shape/color grounding from the same
  accepted/rejected attempt rows. Those heads are blended only as tiny
  tie-breakers until they beat the main ranker without reducing covered-task
  top-1 recovery.
  The neural summary reports top-1/top-3/top-retained accepted-attempt recovery,
  contrast pair counts, pairwise violation counts, per-op recovery, and
  calibration by verifier rejection reason. Its token surface includes
  composition, normalization, typed-stage, primitive-contract, slot, task-shape,
  and verifier-distance features.
- `make learned-grammar-coverage-arc-visual-compare` now closes a small
  curriculum loop around that surface. It runs ARC with legacy transforms
  suppressed, extracts certified `arc_sketch_program` wins as positive visual
  programs, projects their op hints onto public-training visual holes, asks
  `arc_agi` to mine verifier-checked candidates for those holes, and reruns the
  visual-only comparison with the combined prior. The curriculum attempt log is
  the contrast dataset: accepted probes are positive labels; rejected probes
  keep verifier failure stages such as selector/transform, renderer shape, and
  renderer cells.
- `make arc-visual-active-curriculum` repeats the same loop across iterations.
  It stops when public-training visual-only certification stops improving.
  `make arc-visual-capacity-sweep` runs the paired sanity check: train the
  visual-pipeline ranker for different epoch counts, then rerun ARC visual-only
  selection with different hidden sizes. This separates ranker capacity from
  candidate-space coverage.
- The ARC neural runtime now trains auxiliary heads over the same executable
  attempt surface: `renderer_correctness`, `sequence_block_renderer`, and
  `component_shape_color`. `arc_agi` emits a C-owned `component_inventory`
  object on each attempt row with selected-component counts, bbox dimensions,
  shape counts, output foreground counts, and selector/layout match flags.
  Python uses that verifier-side inventory first, falling back to its cached
  parser only for older rows. The runtime also trains a
  `component_selector_layout` head over those features. These heads are blended
  conservatively into ranking as learned tie-breakers, while the verifier score
  and main visual-pipeline ranker remain dominant.
- ARC also runs the visual-pipeline ranker once before held-out repair export.
  That pre-repair pass converts high-scoring renderer, sequence/block, and
  component-shape/color predictions into grammar hints for a second C repair
  lane, but only when those hints open executable operators the residual lane
  did not already allow. The artifacts are
  `arc_visual_pipeline_pre_repair_guidance_proposals.jsonl` and
  `arc_visual_pipeline_pre_repair_guidance_summary.jsonl`.
- `visual_combinator` is the shared C-side sketch VM for learned visual
  programs. It now includes occupancy-grid, count-template, and
  component-shape/color-grid subprograms under the same strict verifier
  boundary. The neural repair lane can also open a slot-pinned
  component-mask renderer with `pin:x_anchor=3`, normalizer pins such as
  `pin:y_anchor=0..3`, the full-canvas layout pin `pin:y_anchor=6`,
  selector-predicate pins such as `pin:y_anchor=7..18`, renderer-mode pins such
  as `pin:y_offset=0..8`, and color-binding pins such as `pin:period=0..1`.
  The `y_anchor=7..12` modes select components by output-bbox match,
  rotated-bbox match, shared anchor shape, shared anchor area, shared anchor
  color, or unique component color. The `y_anchor=13..18` modes reuse those
  predicates but preserve the original input canvas frame for layout-sensitive
  renders. The `y_offset` renderer modes cover raw masks, 90/180/270-degree
  rotations, horizontal and vertical mirrors, outline-only masks, hole-filled
  masks, and interior-only masks. The `pin:period=3` component-mask color mode
  learns a compact marker-glyph-to-color binding from train pairs and stores it
  in the existing `x_offset` slot, so recolor-from-marker tasks can remain
  executable without changing the TSV ABI.
  Component selection is shared between the VM and a C-side feasibility gate,
  so impossible component-mask shapes are counted as `pruned_infeasible`
  instead of consuming full verifier replay budget.
  Component-shape/grid and component-mask attempts are still diagnostic unless
  the C verifier proves exact train consistency.
- ARC ranking reports two separate denominators. `top1_rate`, `top3_rate`, and
  `retained_rate` are ranker-recovery metrics only over held-out tasks where the
  C exporter produced at least one exact train-consistent attempt. The harder
  progress metrics are `target_task_count`, `accepted_target_task_count`,
  `no_accepted_target_task_count`, `search_space_coverage_rate`,
  `top1_all_task_rate`, and `retained_all_task_rate`; these keep the report from
  looking saturated when the attempted grammar does not contain a solution.
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
- Held-out ARC candidate rows still use the backward-compatible `op_hints`
  column, but its values are now grammar hints such as `region_extraction`,
  `normalized_bbox_frame`, `projection_transform`, or `separator_panel_frame`.
  Python computes a shape-conditioned composition proposal from learned source
  attempt feedback, public task shape/color/separator features, and the promoted
  family, then writes `arc_sketch_program_heldout_structure_proposals.jsonl`.
  When these learned hints are present, the C exporter bypasses family-specific
  branch dispatch and runs the generic compatible-miner enumerator. Every
  emitted prior row still has to be exact on all train pairs before it can enter
  `arc_sketch_program_heldout_prior.tsv`.
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
