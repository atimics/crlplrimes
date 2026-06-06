# Next Phase Plan — VSA Scoring, Scale-Space, and Object-Centric Operators

Date: 2026-06-06 | Current coverage: 70/400 (17.5%)

## Where We Are

16 operator families can explain 70 ARC tasks. The remaining 330 tasks need either new
transforms we don't have, or better ways to apply the transforms we DO have. The
infrastructure is ready: LeosEngine static engine in the hot path, DreamCoder
save/restore loop, 8× geometric augmentation, full composition.

## The Core Bet

The bet: **the operators we have can solve 30-40 more tasks if applied to the
right tasks in the right order.** The problem isn't operator coverage — it's
operator *selection*. We enumerate all candidates blindly. A VSA-based scoring
system would learn which operators work on which grid patterns and prioritize
those candidates.

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

## Three Workstreams

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

### Workstream B: VSA Candidate Scoring (the unlock)

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

**Target:** Replace structural scoring with VSA scoring. Measure whether
certified count increases.

### Workstream C: Scale-Space Pyramid (reach)

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

## What NOT to Do

- **Don't add more operators.** The 16 we have cover the geometric, color-mapping,
  and basic object-manipulation space. New operators should come from the
  DreamCoder composition loop (template training), not from hand-coding.
- **Don't increase augmentation beyond 8×.** Geometric symmetry covers the
  dihedral group. Color augmentation didn't help. More data won't help without
  better operator selection.
- **Don't optimize VSA speed before it proves useful.** The static engine and
  precomputed keys are enough. Optimize only after VSA scoring shows gains.

## Success Metrics

| Metric | Current | Phase 2 Target |
|--------|---------|----------------|
| Certified tasks | 70 | 80+ |
| VSA scoring enabled | No | Yes |
| Scale-up tasks solved | ~15/52 | 25/52 |
| Runtime (400 tasks, 8× aug, induction) | 35s | <60s |

## Rollout Order

1. **Workstream A** (VSA grid encoder) — extract reusable function, precompute keys.
   No coverage change expected. Runtime should stay same.
2. **Workstream B** (VSA scoring) — wire HolographicRanker into certification.
   Expected +5-10 solves from better operator selection. Runtime +10-20s.
3. **Workstream C** (scale-space pyramid) — if A+B doesn't reach 80, add
   multi-resolution candidate evaluation. Expected +5-10 solves from scale
   invariance. Runtime +5-10s.
