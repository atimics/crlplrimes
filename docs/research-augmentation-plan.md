# Research Augmentation Plan — Holographic Certification, Compositional Synthesis, and Geometric Representation Learning

Date: 2026-06-06 | Updated: 2026-06-10 | Status: draft

## 0. Reference Intake: `holostuff`

The public repository `AnOversizedMooseWithSocks/holostuff` was reviewed as a
reference corpus for numpy-only holographic/VSA experiments. It should influence
the backlog, but not enter this codebase as copied source: the reviewed snapshot
did not include a visible license file. All implementations here must be
independent C code, evaluated through this repo's schema, trace, and verifier
contracts.

Applicable ideas:

- `HoloTree` / `HoloForest`: deterministic random-projection partitioning keeps
  each holographic leaf memory within capacity, then queries with a beam or
  multi-tree union.
- Self-organizing memory: rebuild a shadow prototype store, judge candidates on
  held-out examples, then swap the winner. This matches the repo's preference
  for verifier-measured promotion over hand-tuned thresholds.
- Compositional scene encoding: segment objects, bind attributes into object
  vectors, bundle them into a scene, and recover factors. For ARC, this maps to
  connected components, panel/layer identity, bounding boxes, colours, masks,
  and relative positions.

Non-applicable or parked ideas:

- Text generation and unified multimodal "mind" demos are sandbox material only.
- Image archive/compression work is peripheral unless reframed as robust
  evidence/certificate storage.
- Post-hoc score blending remains deprecated; memory should feed features into
  the existing scorer or ranker protocol.

## 1. Proven Claims

### C1: The Gated Loop Works

Enumeration → certified symbolic gate → neural ranking → verification. This loop
delivers 62/400 (15.5%) exact-match certification on the ARC public training set
under operator-valid semantics (train-consistent, holonet best-pick).

**Evidence:** `examples/arc_agi.c:evaluate_arc_transform_gate_candidates` with
holonet best-pick scoring. 30 → 62 certified tasks after adding 9 operator
families, full composition, and 8× geometric augmentation.

### C2: Holographic Certificate Store Provides Fixed-Dimension Memory

The `HolographicRanker` stores (state, action) pairs via bind and recalls via
unbind in O(dim) space regardless of stored count. LeosEngine provides
self-contained FFT-based VSA operations with zero heap allocation in the hot path.

**Evidence:** `src/nn_holographic_ranker.c`, `src/holographic/leos_engine.c`.
Linked into `arc_agi` binary. ~7k bind/s at dim=8192.

**Boundary:** This claim is about the flat trace API and its fixed memory
footprint, not unlimited capacity. Flat holographic stores still have an
interference limit; capacity partitioning is now a separate backlog item below.

### C3: Rejected Candidates Are Useful Data

The hypothesis library prior (save/restore via `--save-hypothesis-prior`) accumulates
proposal/certified counts per template. Reloading a prior from a 62-certified run
increases train-consistent proposals from 267 to 323.

**Evidence:** 6 templates accumulate 8/8 proposals/certified each per pass.

### C4: Geometric Augmentation Compounds with Induction

8× dihedral augmentation (4 rotations × 2 flips) provides the largest single
coverage boost: 33 → 60 certified tasks.

**Evidence:** `arc_transform_task` applies rot90/flip_h to all grids before evaluation.
400 tasks → 3,200 augmented evaluations per pass.

### C5: Full Composition Unlocks Multi-Step Transforms

Relaxing composition from "at least one operand must be scale" to "any composable
pair" enables ~30 composed solves among the 62 certified.

**Evidence:** `compose_tile_repeat_then_draw_line` (6), `compose_pattern_extend_then_crop` (7).

---

## 2. Architecture Gaps

### G1: No Scale Invariance

A 3×3 shape and a 9×9 version are different inputs to every operator.
**Required:** Scale-space representation or multi-resolution grid encoding.

### G2: No Object-Abstraction Layer

Every operator works on raw pixel grids. Two L-shapes at different positions
require brute-force translation sweeps.
**Required:** Slot-based or holographic object encoding.

### G3: No Learnable Grid Representation

The candidate builder enumerates fixed operator families. The 13 hypothesis
templates cannot generalize from 62 examples.
**Required:** A representation layer that learns from solved tasks. The first
implemented bridge is `--arc-proposal-trace`, which emits verifier-labeled
`crlp.arc_proposal_trace.v1` rows for ordinary transform candidates. Those rows
make family/slot proposal learning measurable without trusting the learner for
correctness.

### G4: The Holonet Is Not in the Hot Path

Certification uses `arc_structural_program_score` (hardcoded heuristic), not
the holographic ranker.
**Required:** Wire HolographicRanker into certification scoring.

### G5: Flat Holographic Capacity Is Not Managed

A single trace degrades as stored associations accumulate. Memory neurons spread
load across cells, but hot cells can still become overloaded.
**Required:** Capacity-partitioned memory: deterministic tree/forest routing
where each leaf owns a small `HolographicRanker` kept inside measured capacity.

### G6: Self-Organization Is Not Yet Verifier-Measured

Centroid learning and hypothesis priors can drift or fragment without a clear
adoption rule.
**Required:** Shadow-and-swap reorganization judged by held-out verifier utility,
false-rejection preservation, and matched baselines.

---

## 3. Research Threads

### Thread 1: Holographic Grid Encoding

**Goal:** Replace pixel-grid scoring with holographic scene vectors.

**Approach:**
- Encode each non-background cell as: `S_cell = bind(pos_x, bind(pos_y, color_vec))`
- Bundle all cells: `S_scene = bundle(S_cells)`
- Query: `recall = unbind(S_scene, pos_key)` retrieves what's at that position
- Cyclic tile: compare similarity of shifted queries to detect period

**Papers:** Plate 1995 (HRR), Gayler 2003 (VSA), Kanerva 2009 (HDC),
Frady 2020 (Resonator Networks), Kleyko 2022 (VSA survey)

### Thread 2: Scale-Invariant Grid Representation

**Goal:** Detect the same shape at different scales.

**Approach:**
- Multi-scale grid pyramid: downsample 1×/2×/3×/4× via max-pool and average-pool
- Apply operators at each scale, upsample results
- Fourier-domain scaling: FFT → truncate/pad → IFFT (preserves frequency content)

**Papers:** Henriques & Vedaldi 2016 (Warped Convolutions), Chi 2020 (FNO),
DeVries & Taylor 2017 (Cutout), Shorten & Khoshgoftaar 2019 (augmentation survey)

### Thread 3: Object-Centric Scene Encoding

**Goal:** Represent grid as a set of objects, not a pixel array.

**Approach:**
- Extract all connected components (currently only largest)
- Per-component features: bbox, area, centroid, color, shape hash
- Encode each object as VSA vector: `S_obj = bind(shape_key, bind(pos_key, color_key))`
- Operators query the scene via similarity search
- Keep the prior explicit: object-scene vectors are a declared representation
  condition (`arc_object_scene_v1`), not a hidden feature channel for exact
  claims.
- Use compositional binding for attributes such as `component_id`, `panel_role`,
  `layer_role`, `bbox`, `centroid`, `area`, `aspect`, `color_set`, and
  `relative_position`.

**Papers:** Locatello 2020 (Slot Attention), Burgess 2019 (MONet),
Santoro 2018 (Relation Networks), Battaglia 2018 (Graph Networks)

### Thread 4: Learned Program Synthesis from Templates

**Goal:** Train hypothesis templates on solved tasks to generalize to unsolved.

**Approach:**
- Multi-pass training: run 8× augmentation through hypothesis system, save priors
- Cross-task similarity: use holographic scene vectors to propagate successful
  templates from solved to similar unsolved tasks
- Trace-first proposal learning: run the public loader with
  `--arc-proposal-trace`, then train tiny family and slot heads from task
  features plus candidate slots to verifier labels. The learner proposes order;
  the transform verifier still decides train consistency and exactness.
- Baseline result: online `trace_prior` counts over family/op/slot positives
  beat structural ordering at ARC budget 8 and 64. `trace_task_prior` adds
  task-feature success prototypes and improves the budget-64 row over global
  trace counts. A flat online `trace_linear` logistic head helps only at budget
  2 and regresses at budget 64; a listwise `trace_mlp` also trails the
  prototype prior. This matches the holo pattern: memory/prototype indexing is a
  reliable substrate, and neural routing should be confidence-gated or trained
  as a residual correction.
- Implemented residual bridge: `trace_holo_residual` stores verified positive
  `(task_features, candidate_features)` pairs in `HolographicRanker`, recalls a
  task-conditioned action context once per ranking pass, and adds only a bounded
  positive residual to `trace_task_prior` after support/similarity/memory gates.
  The fixture smoke validates the wiring and summary rows. Public unique-task
  rows now show changed program-set hashes but matched utility versus
  `trace_task_prior`. The residual diagnostic schema now shows why: 20 public
  rows over five exact candidates, zero budget crossings, and the largest lifts
  remain far outside budget 64. The next step is representation/gate redesign,
  not claiming a coverage improvement. The HoloNet research pass now narrows
  that redesign: add rank-pressure memory for exact candidates missed by
  `trace_task_prior`, and route it through family or HoloForest partitions before
  tuning residual weights. The first `HoloPartitionedRanker` sandbox confirms
  why: 32 routed local traces improve 2048-d exact retrieval top-1 from about
  `0.50` flat to `0.98`, `0.85`, `0.78`, `0.63`, and `0.55` at
  64, 128, 256, 512, and 1024 stored pairs respectively.
- Current cleanup bridge: `trace_holo_bucket` keeps raw pressure residuals as a
  negative control and instead uses guarded pressure memory to open a bounded
  verifier-review window around candidates missed by `trace_task_prior`. The
  ordered public run improves at budgets 8/16/32/64 and the tight grid now
  matches at budget 1 after adding a min-budget floor. The promotion audit is
  `make arc-holo-bucket-robustness`, which emits raw ARC logs plus
  `crlp.arc_holo_bucket_robustness.run.v1`, aggregate rows, and
  `crlp.arc_holo_bucket_robustness.verdict.v1` rows across task-order,
  warmup-threshold, and review-window perturbations. The first default full public sweep keeps
  zero crossed-out exact candidates; `auto` and one-slot review pass the
  no-regression check, while two-/four-slot review windows regress tight budget
  2 and should remain exploratory larger-budget settings. The invocation gate
  is now explicit via `--arc-holo-bucket-min-pressure-updates` and
  `crlp.arc_holo_bucket_gate.v1`; the focused public `auto,1` audit opens
  776/800 rows after warmup, opens 0/800 at tight budget 1, and gives the next
  learner a direct target for deciding when the review bucket is worth spending.
  `ARC_HOLO_BUCKET_ROBUSTNESS_MIN_PRESSURE_UPDATES` may now be a comma-separated
  list; `verdict.jsonl` is the threshold-selection artifact.
  The first learned invocation gate is implemented behind
  `--arc-holo-bucket-learned-gate` and exposed through
  `ARC_HOLO_BUCKET_ROBUSTNESS_LEARNED_GATE=1`. It trains online from prior
  task-budget labels and can only close the existing fixed gate. Fixture smoke
  says threshold `0.5` is too conservative, while threshold `0.3` recovers the
  fixture uplift with zero crossed-out exact candidates; public verdict rows
  still need to promote or reject it. The learned-gate surface now records
  review-window exact hits, closed review exacts, mean target, score, and review
  margin. A bounded 40-task public smoke at threshold `0.3` found outside-budget
  exact pressure candidates but zero review-window exacts, making the pressure
  review selector/reranker the next evidence target.

**Papers:** Lake 2017 (Compositional Generalization), Bunel 2018 (Neural Program
Synthesis), Nye 2021 (Scratchpads)

### Thread 5: Spectral Methods for Grid Transforms

**Goal:** Fourier-domain operations for efficient multi-scale processing.

**Approach:**
- LeosEngine FFT for period detection (find peaks in frequency domain)
- Low-pass filter for scale/blur: zero high frequencies → smaller effective grid
- Cross-correlation via FFT multiplication for shape matching

**Papers:** Gu 2022 (S4), Cooley & Tukey 1965 (FFT)

### Thread 6: Capacity-Partitioned Holographic Memory

**Goal:** Keep holographic recall usable after a flat trace would overload.

**Approach:**
- Build a deterministic random-projection tree over state/query vectors.
- Store each leaf's associations in a small `HolographicRanker`.
- Query by descending to the best leaf, then optionally widening with beam
  backtracking or multiple independently seeded trees.
- Report recall@1, comparisons/query, build time, query time, and leaf load
  distribution against exact scan and flat trace baselines.

**Evidence target:** sandbox JSONL rows first. Promote only if the forest
improves a real downstream ranker/proposer budget without increasing unsafe
decisions.

### Thread 7: Verifier-Measured Self-Organization

**Goal:** Let memory/prototype stores reorganize only when measured utility
improves under the repo's verifier protocol.

**Approach:**
- Build shadow candidates: keep, merge duplicates, split multimodal classes,
  rebuild from recent valid experience.
- Judge candidates on held-out verifier rows or replay rows.
- Adopt the winner only after checking false-rejection / safety constraints.
- Emit rows recording candidate strategy, held-out utility, adopted strategy,
  memory size, and residual risk.

**Evidence target:** hypothesis-library or rejected-candidate memory first, not
promoted ARC coverage.

---

## 4. Phase Plan

### Phase 1: Holographic Grid Encoding and VSA Scoring (2-3 weeks)

1. Implement `grid_to_hologram`: 30×30 grid → 256-dim scene vector via LeosEngine
2. Holographic cyclic tile detector replacing broken `arc_apply_pattern_tile_candidate`
3. Wire HolographicRanker into certification scoring
4. **Observed:** full-scale raw-grid VSA persistence works (`stored_count=114`
   after shuffled train + ordered load), but ordered metrics remain 70/400.
5. **Observed:** `arc_object_scene_v1` also persists and scores cleanly
   (`stored_count=114`), but ordered metrics still remain 70/400.
6. **Next target:** keep raw-grid ARC VSA scoring parked, keep raw
   `--arc-holo-pressure-residual` as a negative control, and focus on
   `trace_holo_bucket`. It treats guarded pressure memory as a cleanup/review
   bucket rather than a scorer. The public unique-task run records utility
   improvements at budgets 8/16/32/64 and bucket diagnostics show 8 exact
   candidates crossing into budget with 0 crossing out. The tight-grid run now
   matches at budget 1 and improves at budgets 2/4/8, with 4 exact crossings in
   and 0 out. The explicit gate rows make this a conditional verifier-feedback
   memory path rather than an always-on score channel; the next promotion test
   is more task-order seeds for `auto`/`1` and a learned invocation policy.
7. **Measured substrate row:** `holo_experiments --mode partitioned-retrieval`
   now emits `crlp.holo_partitioned_retrieval.v1`; use local max-load below
   roughly 50 pairs as the first routing target for ARC residual memories.

### Phase 1B: Object-Scene VSA Representation (done; no uplift)

1. Extract connected components and panel/layer facts already available in ARC
2. Bind component attributes into object vectors and bundle a scene vector
3. Compare structural, `arc_grid_hologram_v1`, and `arc_object_scene_v1` ranker
   conditions at matched task order and budgets
4. **Observed:** no improvement over baseline under matched ordered public rows.

### Phase 1C: Scale-Space / Visual-Program Coverage (1-2 weeks)

1. Audit scale-up/down and output-size mismatch misses
2. Extend multi-scale candidate evaluation only where shape evidence justifies it
3. Inspect learned visual-program zero-candidate / coverage-gap tasks
4. **Observed:** `reflect_tile_{2,3,4}x` is verifier-backed and improves the
   focused shape-scale audit from 32 to 55 certified rows, but the capped
   400-row loader gate remains 70/400.
5. **Observed:** `--unique-task-gate` now provides a cheap all-400-file
   canonical gate. With induction, reflected tiling improves that gate from
   31 exact / 34 certified / 0.02710333 cell accuracy to 34 exact / 37 certified
   / 0.02922078 cell accuracy; exact reflected-tile tasks are `3af2c5a8`,
   `62c24649`, and `67e8384a`.
6. **Observed:** `marker_motif_tile` adds a second verifier-backed fixed-scale
   primitive: stamp the input motif into output blocks selected by a fixed,
   dominant, or rarest marker color. It improves the same gate to 36 exact /
   39 certified / 0.03207933 cell accuracy; exact additions are `c3e719e8` and
   `cce03e0d`. Use `--disable-marker-motif-tile` for ablation.
7. **Observed:** `axis_tile` adds one-axis repeat, reflect, and reverse-first
   reflect expansion. It improves the same gate to 42 exact / 45 certified /
   0.03447911 cell accuracy; exact additions are `4c4377d9`, `6d0aefbc`,
   `6fa7a44f`, `8be77c9e`, `a416b8f3`, and `c9e6f938`. Use
   `--disable-axis-tile` for ablation.
8. **Observed:** `period_extend` continues inferred row/column periods across a
   doubled axis. It improves the same gate to 43 exact / 46 certified /
   0.03606719 cell accuracy; exact addition is `963e52fc`. Use
   `--disable-period-extend` for ablation.
9. **Target:** improve proposal coverage without hand-coding task-specific
   operators, using `--unique-task-gate` as the routine coverage signal and
   reserving full augmented all-files runs for less frequent confirmation.

### Phase 1D: Capacity Partitioning Ablation (1 week)

1. Implement flat/tree/forest recall benchmark over synthetic vectors
2. Repeat on ARC/hypothesis candidate embeddings
3. Emit `crlp.holo_forest_benchmark.v1` sandbox rows
4. **Target:** forest preserves recall where flat trace degrades, at fewer
   comparisons than exact scan

### Phase 2: Scale-Space Grid Operations (2 weeks)

1. Multi-scale pyramid: downsample 1×/2×/3×/4× via max-pool/average-pool
2. Apply operators at each scale, upsample predictions
3. **Target:** 29/29 scale-up tasks solved (currently ~15/29)

### Phase 3: Object-Centric Representation (2 weeks)

1. Extract all connected components per grid
2. Encode each component with (position, color, area, aspect ratio, perimeter)
3. Component-aware operators (per-object translate, per-object recolor)
4. **Target:** object-scene ranker rows improve budgeted proposal ordering;
   exact solves remain verifier-owned

### Phase 4: Template Pre-Training and Self-Organization Loop (1 week)

1. Multi-pass training: build priors from augmented data
2. Cross-task holographic similarity for template propagation
3. Shadow split/merge/rebuild of hypothesis memories, judged on held-out
   verifier rows
4. **Target:** memory compaction or proposal-order improvement without unsafe
   false-rejection regression

---

## 5. Augmented Bibliography

### Group J: Holographic Representations and VSA

- Plate, T. A. (1995). "Holographic Reduced Representations." *IEEE TNN*.
- Gayler, R. W. (2003). "Vector Symbolic Architectures." *ICCS*.
- Kanerva, P. (2009). "Hyperdimensional Computing." *Cognitive Computation*.
- Frady, E. P. et al. (2020). "Resonator Networks." *Neural Computation*.

### Group K: Geometric Data Augmentation and Invariance

- Shorten, C. & Khoshgoftaar, T. M. (2019). "A Survey on Image Data Augmentation." *J. Big Data*.
- Henriques, J. F. & Vedaldi, A. (2016). "Warped Convolutions." *arXiv*.
- DeVries, T. & Taylor, G. W. (2017). "Cutout." *arXiv*.

### Group L: Object-Centric and Relational Learning

- Locatello, F. et al. (2020). "Object-Centric Learning with Slot Attention." *NeurIPS*.
- Burgess, C. P. et al. (2019). "MONet." *arXiv*.
- Santoro, A. et al. (2018). "Relation Networks." *NeurIPS*.

### Group M: Fourier and Spectral Methods

- Chi, L. et al. (2020). "Fourier Neural Operator." *arXiv*.
- Gu, A. et al. (2022). "S4: Structured State Spaces." *ICLR*.

### Group N: Program Synthesis and Compositional Generalization

- Lake, B. M. et al. (2017). "Building Machines That Learn and Think Like People." *BBS*.
- Bunel, R. et al. (2018). "Neural Program Synthesis." *ICLR*.

---

## 6. Evaluation Protocol

| Phase | Metric | Baseline | Target |
|-------|--------|----------|--------|
| Holo encoding/scoring | Certified tasks | 70 | 75+ |
| Capacity partitioning | Recall after flat overload | flat trace | forest advantage |
| Multi-scale | Scale-up tasks solved | ~15/29 | 29/29 |
| Object-centric | Budgeted proposer ordering | flat/structural | sandbox improvement |
| Self-organization | Held-out utility at fixed safety | keep-only memory | no unsafe regression |
| Template training | Certified tasks | 70 | 80+ only after verifier promotion |

Runs use `--induction` with 8× augmentation. DreamCoder save/restore carries priors
forward between phases. Each phase is additive on the existing operator set.
Capacity, object-scene, and self-organization rows are sandbox rows until a
paper-support gate explicitly promotes them.

---

*SPDX-License-Identifier: MIT. Copyright (c) 2026 Anonymous Authors.*
