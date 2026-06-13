# Next Phase Plan — VSA Scoring, Capacity, Scale-Space, and Object-Centric Operators

Date: 2026-06-06 | Updated: 2026-06-10 | Current coverage: 70/400 capped augmented rows; 43/400 exact on unique-task gate

## External Reference Intake

`AnOversizedMooseWithSocks/holostuff` was reviewed as a reference
implementation for numpy-only holographic/VSA mechanisms. It is relevant to
this plan as a source of design patterns, not as a code dependency: no license
file was visible in the repository snapshot reviewed, so any implementation in
this repo must be independent C code with local tests and evidence rows.

The useful additions are:

- deterministic capacity partitioning (`HoloTree` / `HoloForest`) for traces
  that would otherwise exceed flat holographic capacity;
- shadow-and-swap self-organization for splitting/merging prototype memories
  only after held-out measurement says the reorganization helps;
- compositional object/scene encoding by binding object attributes into scene
  vectors and recovering them with cleanup/resonator-style factoring.

These refine the backlog below. They do not change the evidence rule: ARC/VSA
work remains a sandbox until verifier-backed rows, matched baselines, and
freshness checks promote it.

## Where We Are

16 operator families can explain 70 ARC tasks. The remaining 330 tasks need either new
transforms we don't have, or better ways to apply the transforms we DO have. The
infrastructure is ready: LeosEngine static engine in the hot path, DreamCoder
save/restore loop, 8× geometric augmentation, full composition.

The full-scale VSA scorer probes are complete. `--vsa-ranker` now stores
verified associations and emits `crlp.arc_vsa_ranker.summary.v1`; both
`arc_grid_hologram_v1` and `arc_object_scene_v1` loaded ordered passes ended
with `stored_count=114`. Ordered public metrics stayed at the baseline 70/400
for both representation conditions. The next leverage point is scale-space and
visual-program proposal coverage, not VSA persistence, score-weight tuning, or
another ARC VSA representation.

Scale-space work has started. The generic reflected-tile primitive
(`reflect_tile_{2,3,4}x`) improved a focused shape-scale audit from 32 to 55
certified rows, but the standard capped 400-row public loader gate remains
70/400. The measurement fix is now in place: `--unique-task-gate` evaluates one
canonical orientation per public training file in about 25-29 seconds. On that
gate, reflected tiling moves induction from 31 exact / 34 certified to 34 exact
/ 37 certified, with exact reflected-tile coverage on `3af2c5a8`, `62c24649`,
and `67e8384a`. The next fixed-scale primitive, `marker_motif_tile`, stamps the
input motif into blocks selected by a fixed, dominant, or rarest marker color;
it moves the same gate to 36 exact / 39 certified and adds exact coverage for
`c3e719e8` and `cce03e0d`. `axis_tile` then handles one-axis repeat,
reflect, and reverse-first reflect expansions, moving the gate to 42 exact /
45 certified with exact additions `4c4377d9`, `6d0aefbc`, `6fa7a44f`,
`8be77c9e`, `a416b8f3`, and `c9e6f938`. `period_extend` continues inferred
row/column periods across a doubled axis and moves the gate to 43 exact /
46 certified with exact addition `963e52fc`. Full all-files augmented runs still
exceed the practical runtime window before emitting a loader summary.

The next proposal-learning surface is also in place. `--arc-proposal-trace`
writes `crlp.arc_proposal_trace.v1` rows after candidate evaluation, with task
features, candidate family/slot fields, structural scores, and verifier labels.
The full unique-task trace run produced 25,750 valid rows over 400 tasks in
about 22 seconds while preserving the 43 exact / 46 certified gate. It contains
150 train-consistent / test-supported proposal rows and 144 exact candidate
rows. This makes the next high-leverage loop trace-first: learn family and slot
proposal order from verifier rows, then use the unchanged verifier gate to
certify any resulting coverage.

The first online learners are now measured as `strategy:"trace_prior"` and
`strategy:"trace_task_prior"` in `crlp.program_proposer.summary.v1`.
`trace_prior` learns positive-evidence family/op/slot counts from previous
tasks; `trace_task_prior` adds task-feature success prototypes. On the
unique-task gate, `trace_task_prior` beats structural ordering at budget 64
(`0.02073334` utility versus `0.01635728`) and improves over the global trace
prior (`0.01986872`), but remains below the existing learned scorer
(`0.02459768`). Tiny trained `trace_linear` and listwise `trace_mlp` heads are
instrumented, but at budget 64 they reach only `0.01986872` and `0.02018634`.
That matches the `holostuff` and local HoloNet lesson: measured prototype memory
is a strong index, while learned routing can underperform on clustered stores
unless it is confidence-gated or trained on residual mistakes. The residual
step is now implemented as `strategy:"trace_holo_residual"`: verified positive
proposal rows are stored in `HolographicRanker` as `(task_features,
candidate_features)` bindings, task recall is computed once per ranking pass,
and the recalled action similarity can only add a bounded positive residual to
`trace_task_prior` when family/op support, task-feature similarity, and memory
count clear confidence gates. The ARC hypothesis fixture smoke emits
`trace_holo_residual` rows at every program budget with matching solved utility.
The public unique-task training gate now also runs: the residual changes
program-set hashes, but matches `trace_task_prior` utility at budgets 8, 16, 32,
and 64. That makes it an instrumentation and diagnostics surface, not a promoted
coverage uplift. `--arc-holo-residual-diagnostics` now emits
`crlp.arc_holo_residual_diagnostic.v1` rows for exact candidates that move by at
least eight ranks or cross a budget boundary. The first public diagnostic run
emits 20 rows over five exact candidates and zero budget crossings: the largest
shifts lift `67e8384a` reflected-tile compositions by 30-73 ranks, but only from
roughly rank 1160-1185 to 1112-1132. The residual is therefore too weak or too
coarse to move exact candidates into budget, not currently unsafe for accepted
coverage. The 2026-06-13 HoloNet research pass in
`docs/holonet-research-brief.md` changes the follow-up: do not tune the residual
weight first. Add capacity management and cleanup-style routing first, because
flat holographic traces are expected to lose discriminative power as cross-term
noise grows. The first `HoloPartitionedRanker` sandbox confirms the capacity
claim: at 2048 dimensions, 32 routed partitions lift exact retrieval top-1 from
about `0.50` flat to `0.98/0.85/0.78/0.63/0.55` at
`64/128/256/512/1024` stored pairs, with clear wins while local max load stays
below roughly 50 pairs.

## The Core Bet

The bet: **the operators we have can solve 30-40 more tasks if applied to the
right tasks in the right order.** The problem isn't only operator coverage — it
is proposal *selection*. We now have a trace-first path for learning which
families and slots deserve budget before manual audits add more operators.

## Phase Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Grid Input │ ──► │ Holographic       │ ──► │ Candidate        │
│  (30×30)    │     │ Scene Encoder     │     │ Scoring          │
└─────────────┘     │ grid → 256-dim    │     │ holo_ranker_     │
                    │ LeosEngine bind/  │     │ score()          │
                    │ bundle            │     └────────┬────────┘
                    └──────────────────┘              │
                                                      ▼
                    ┌──────────────────┐     ┌─────────────────┐
                    │ Scale-Space      │     │ Certification    │
                    │ Pyramid          │◄────│ Gate             │
                    │ downsample →     │     │ top-N scored     │
                    │ apply → upsample │     │ candidates       │
                    └──────────────────┘     └─────────────────┘
```

## Five Workstreams

### Workstream A: VSA Grid Encoder (highest leverage)

**What:** `grid_to_hologram(grid) → double[256]` as a reusable function.

**Why:** Every downstream VSA feature (scoring, similarity, period detection)
needs this. Currently the holo_tile function has inline encoding that can't be
reused.

**Implementation:**
1. Extract the encoding logic from `arc_apply_holo_tile_candidate` into a
   standalone `arc_grid_to_hologram` function.
2. Use the static LeosEngine (already exists at line ~26140).
3. Precompute position keys for all 900 cells once (`leos_engine_keygen` with
   seed = y*1000+x).
4. Precompute color keys for 10 colors once (seed = c*10000+1).
5. Store precomputed keys in static arrays so they're computed once per process.

**Target:** A single function call that encodes any grid in O(n) time where n
is the number of non-bg cells, reusing precomputed keys.

### Workstream B: VSA Candidate Scoring (wired; no raw-grid uplift)

**What:** Replace `arc_structural_program_score` with holographic similarity
scoring in the certification gate.

**Why:** Structural scores are hardcoded heuristics (color_map=120, gravity=98,
etc.). They don't learn from solved tasks. A VSA scorer would:
1. Encode each training grid as a holographic vector.
2. When a candidate succeeds on a grid, store (grid_vector, candidate_op) in the
   HolographicRanker.
3. For a new grid, recall similar grids and prefer candidates that worked on
   those grids.

This is the DreamCoder loop applied to operator *selection* rather than operator
*generation*. The HolographicRanker already has the store/recall/score API.

**Implementation:**
1. Add `HolographicRanker *holo_scorer` as a static variable.
2. In the certification gate, for each train-consistent candidate:
   - Encode the grid → `grid_vec`
   - Call `holo_ranker_score(holo_scorer, grid_vec, candidate_ops, n, scores)`
   - Use VSA score instead of structural score for best-pick
3. After certification, if the task was solved, call
   `holo_ranker_store(holo_scorer, grid_vec, winning_candidate_vec)`.
4. The ranker accumulates knowledge across tasks. Later tasks benefit from
   earlier ones.

**Current result:** The ranker wire-up works at full scale: pass 1 stored 44
verified associations, pass 2 stored another 70, and the saved trace reported
`stored_count=114`. However, the flat grid-vector condition did not improve
ordered metrics over baseline: certified tasks stayed 70, exact tasks stayed 70,
cell accuracy stayed 0.10719844, and budget-64 program proposer utility stayed
0.103249027237 for the learned strategy.

**Target:** Keep the persistence/instrumentation, but do not treat ARC VSA
scoring as the current unlock. Revisit only after candidate coverage changes.

### Workstream C: Capacity-Partitioned Holographic Memory

**What:** Add a deterministic forest index over holographic traces: random
projection splits route items into leaves small enough that each leaf's
`HolographicRanker` stays below its interference/crosstalk limit.

**Why:** The current flat trace is deliberately O(dim), but fixed-width bundles
have finite capacity. Memory neurons reduce load by sparse activation, but each
cell can still become overloaded on skewed or long-running streams. A forest
turns "one overloaded trace" into "many small traces plus a reproducible routing
rule", making capacity a measured parameter instead of a hidden cliff.

**Reference pattern:** `holostuff`'s HoloTree/HoloForest idea: split keys by
seeded random hyperplanes, store each leaf in a small holographic memory, query
with a beam or forest union, and report recall/comparisons against exact scan.
Reimplement independently in C.

**Implementation:**
1. Add a sandbox `HoloForest` type around `HolographicRanker` and `LeosEngine`.
2. Emit JSONL rows for flat trace vs tree vs forest: `N`, `dim`, `leaf_size`,
   `beam`, recall@1, comparisons/query, build time, query time.
3. Test first on synthetic vectors, then on hypothesis-library or ARC candidate
   embeddings.
4. Only wire into memory neurons after the flat-vs-forest ablation shows a real
   benefit at matched recall/cost.

**Target:** Preserve recall after the flat ranker degrades, while using fewer
comparisons than exact scan. This is a capacity/stability result, not an ARC
coverage claim by itself.

### Workstream D: Scale-Space Pyramid (reach)

**What:** Apply operators at multiple resolutions, not just single downsampling.

**Why:** The current `ARC_RULE_MULTISCALE` only tries one downsampling factor.
A proper pyramid would try factors 1, 2, 3, 4 and pick the best match.

**Implementation:**
1. For each candidate operator, try applying at original scale AND at each
   downsampled scale.
2. Upsample results and compare with output dimensions.
3. This is an augmentation of the *candidate evaluation* path, not a new operator.
4. Can be implemented as a post-processing step after `arc_apply_candidate`: if
   the output dimensions don't match, try downsampled input.

**Target:** Solve more scale-up/scale-down tasks (currently ~15/52 unsolved).

### Workstream E: Object-Centric Scene Vectors (measured; no uplift)

**What:** Represent ARC grids as sets of objects and object attributes rather
than only as raw cells.

**Why:** The ARC visual failure notes already identify objectness, panels,
component correspondence, and layer relations as missing priors. A compositional
VSA scene vector gives the ranker a way to compare tasks by object structure
without hiding those priors inside an opaque scorer.

**Reference pattern:** `holostuff`'s compositional scene coder: segment
foreground objects, assign simple colour/shape/texture/position tags, bind tags
into object vectors, and bundle objects into a scene vector. In this repo, the
initial C version should keep to ARC-native facts: connected component id,
bounding box, centroid, colour set, area/aspect ratio, panel/layer role, and
relative position.

**Implementation:**
1. Extend the ARC sandbox with an explicit object-scene encoder.
2. Use the scene vector only for candidate/proposer ordering at first; exact
   certification still comes from the symbolic transform gate.
3. Compare against flat grid hologram and current structural ordering at fixed
   candidate budgets.
4. Emit the representation condition in every row
   (`arc_grid_hologram_v1`, `arc_object_scene_v1`, etc.) so scorer gains cannot
   masquerade as verifier gains.

**Target:** Improve budgeted proposal ordering on object-motion, recolour,
panel/layer, and component relation tasks. Promote only after held-out public
rows separate ranker improvement from exact verifier coverage.

## What NOT to Do

- **Don't add more operators.** The 16 we have cover the geometric, color-mapping,
  and basic object-manipulation space. New operators should come from the
  DreamCoder composition loop (template training), not from hand-coding.
- **Don't increase augmentation beyond 8×.** Geometric symmetry covers the
  dihedral group. Color augmentation didn't help. More data won't help without
  better operator selection.
- **Don't optimize VSA speed before it proves useful.** The static engine and
  precomputed keys are enough. Optimize only after VSA scoring shows gains.
- **Don't vendor `holostuff` code.** Use it as a reference only unless a license
  grant is obtained. All promoted or sandbox implementations here should be
  independently written C with local tests.
- **Don't promote capacity or object-scene demos as ARC solves.** Forest recall,
  object vectors, and self-organization are mechanisms. They become evidence
  only when wired into the existing verifier protocol and compared to matched
  baselines.

## Success Metrics

| Metric | Current | Phase 2 Target |
|--------|---------|----------------|
| Certified tasks | 70 | 80+ |
| Unique-task certified/exact tasks | 46 / 43 after reflected + marker motif + axis tiling + period extension | 50+ / 46+ |
| VSA scoring enabled | Yes, no grid/object-scene uplift | Keep instrumented |
| Proposal trace rows | `crlp.arc_proposal_trace.v1`: 25,750 rows / 400 tasks / 144 exact candidate rows | Task-conditioned family/slot proposer |
| Trace-prior proposer | `trace_task_prior` beats structural and global trace prior at budget 64; trained `trace_linear`/`trace_mlp` trail it; default `trace_holo_residual` matches public utility and diagnostics show 0 budget crossings; guarded partitioned pressure memory now trains 461 public-gate updates but `--arc-holo-pressure-residual` remains negative, with 2 crossings in and 27 crossings out | Replace raw pressure boosts with cleanup buckets or verifier-derived residual targets |
| Flat-vs-forest recall rows | Partitioned sandbox rows exist: 32 partitions lift top-1 from ~0.50 flat to 0.98/0.85/0.78/0.63/0.55 at 64/128/256/512/1024 pairs | Add tree/forest or ARC family-partition rows |
| Focused shape-scale audit | 55 certified rows after reflected tiling; marker motif adds 2, axis tiling adds 6, and period extension adds 1 unique-task exact solve | Extend to next fixed-scale family |
| Object-scene ranker condition | Yes, no uplift | Park |
| Runtime (unique-task gate, induction) | 25-29s | <60s |

## Rollout Order

1. **Coverage measurement** — keep `--unique-task-gate` as the fast canonical
   all-400-file gate so primitives that help outside the capped augmented slice
   show up in headline metrics.
2. **Trace-first proposer learning** — use `crlp.arc_holo_residual_diagnostic.v1`
   rows to separate far-out exact lifts from true budget-window wins/losses.
   Next, train a rank-pressure HoloNet memory from exact candidates missed by
   `trace_task_prior`, routed through family or HoloForest partitions so each
   local trace stays below the flat-memory crosstalk limit.
3. **Workstream D** (scale-space pyramid) — continue from `reflect_tile` into
   other shape-evidenced scale transforms. Runtime +5-10s.
4. **Visual-program proposal coverage** — inspect zero-candidate and
   learned-visual-program coverage-gap tasks before adding more hand operators.
5. **Workstream C** (capacity partitioning) — add flat/tree/forest ablations now
   for the ARC residual proposer; the flat trace already shows useful but
   budget-irrelevant movement.
6. **Workstream A/B/E hardening** — keep the reusable encoders, save/load, and VSA
   summary rows stable; avoid more score-weight tuning until candidate coverage
   changes.
