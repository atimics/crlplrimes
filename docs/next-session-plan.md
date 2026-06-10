# Next Session Plan — Scale-Space and Visual-Program Coverage

Date: 2026-06-06 | Updated: 2026-06-10 | Current coverage: 70/400 capped augmented rows; 43/400 exact on unique-task gate

## Backlog Alignment Note

The two-pass VSA experiment has now run at full public-training scale. The
later backlog remains refined by reviewing `AnOversizedMooseWithSocks/holostuff`
as reference material:

- use `holostuff` only as a conceptual reference, not copied code, because the
  reviewed snapshot had no visible license file;
- add a capacity-partitioning follow-up (`HoloForest`-style C sandbox around
  `HolographicRanker`) before relying on flat traces for long-running memory
  claims;
- harden the existing `memory_neuron.c` implementation instead of treating it
  as unwritten;
- keep object-centric scene vectors and self-organization in sandbox rows until
  verifier-backed promotion criteria are satisfied.

The VSA follow-ups are now measured: both raw grid vectors and object-scene
vectors persist and score cleanly, but neither improves ordered public ARC
coverage. Scale-space work has started with verifier-backed
`reflect_tile_{2,3,4}x`, `marker_motif_tile`, `axis_tile`, and `period_extend` primitives. The capped 400
augmented-row slice misses several tasks where they help, so the current
affordable headline gate is now `--unique-task-gate`: one canonical orientation
per public training file.

## Full-Scale VSA Results

Command family:

```
./build/float/arc_agi \
  --task-dir build/float/arc_agi_public/data/training \
  --max-tasks 400 --induction --min-induction-evidence 1 \
  --vsa-ranker /tmp/vsa_trained_final.bin \
  --loader-summary-only
```

Observed rows:

| Condition | Certified | Exact | Cell accuracy | VSA stored |
|-----|-----------|-------|---------------|------------|
| Baseline ordered | 70 | 70 | 0.10719844 | n/a |
| `arc_grid_hologram_v1` loaded ordered | 70 | 70 | 0.10719844 | 114 |
| `arc_object_scene_v1` loaded ordered | 70 | 70 | 0.10719844 | 114 |

Interpretation:

- The `--vsa-ranker` path is real now: full-scale pass 1 stored 44 verified
  associations, and pass 2 ended with `stored_count=114`.
- Loaded VSA did not move ordered public metrics relative to baseline:
  certified tasks, exact tasks, cell accuracy, and program-budget utilities are
  unchanged.
- This argues against spending the next session on blend-weight or
  representation tuning. The useful next test is whether missing coverage comes
  from scale-space and visual-program proposal gaps rather than ordering.

## Scale-Space Progress

Added two generic scale/placement transforms:

- `reflect_tile_2x`, `reflect_tile_3x`, `reflect_tile_4x`;
- output cell source coordinates mirror on alternating horizontal/vertical
  tiles;
- `marker_motif_tile`, which stamps the whole input motif into block positions
  selected by a fixed, dominant, or rarest marker color;
- `axis_tile`, which repeats, reflects, or reverse-first reflects along one
  expanded axis;
- `period_extend`, which continues inferred row/column periods across a doubled
  axis;
- verifier remains unchanged: train-pair consistency plus exact test check.

Focused shape-scale task audit:

| Gate | Certified | Exact | Cell accuracy |
|------|-----------|-------|---------------|
| Before reflected tiling | 32 | 32 | 0.18498294 |
| After reflected tiling | 55 | 55 | 0.51228669 |

Capped 400-row public loader gate:

| Gate | Certified | Exact | Cell accuracy |
|------|-----------|-------|---------------|
| Before reflected tiling | 70 | 70 | 0.10719844 |
| After reflected tiling | 70 | 70 | 0.10719844 |

Unique-task public training gate:

| Gate | Certified | Exact | Cell accuracy |
|------|-----------|-------|---------------|
| No reflected tiling, induction | 34 | 31 | 0.02710333 |
| Reflected tiling, induction | 37 | 34 | 0.02922078 |
| Reflected tiling + marker motif, induction | 39 | 36 | 0.03207933 |
| Reflected + marker motif + axis tiling, induction | 45 | 42 | 0.03447911 |
| Reflected + marker motif + axis tiling + period extension, induction | 46 | 43 | 0.03606719 |

Interpretation: the scale-space primitives now have a canonical all-400-file
signal. Reflected tiling contributes +3 exact / +3 certified tasks, marker
motif tiling contributes another +2 exact / +2 certified tasks, axis
tiling contributes another +6 exact / +6 certified tasks, and period extension
adds +1 exact / +1 certified task with no exact-set regressions. Exact reflected-tile tasks in that gate are `3af2c5a8`,
`62c24649`, and `67e8384a`; exact marker-motif additions are `c3e719e8` and
`cce03e0d`. Exact axis-tile additions are `4c4377d9`, `6d0aefbc`,
`6fa7a44f`, `8be77c9e`, `a416b8f3`, and `c9e6f938`; the exact
period-extension addition is `963e52fc`. Use `--disable-reflect-tile`,
`--disable-marker-motif-tile`, `--disable-axis-tile`, and
`--disable-period-extend` for ablations. The all-files augmented pass was
attempted twice and exceeded the current practical runtime window before
emitting a loader summary (`~116-125s`, terminated abnormally), so use
`--unique-task-gate` for fast canonical coverage tracking before paying for
full augmented sweeps.

## What's Proved

- VSA save/load works on 10-task runs: save file is 2056 bytes, load pass
  improves cell accuracy from 0.728 to 0.832.
- Stack overflow fixed (`ARCGrid best` moved to static).
- The crash on 400-task runs was stale object files, not logic errors.

## Session Goal

Improve ARC coverage by targeting known scale-space and visual-program proposal
gaps. The exact verifier remains unchanged; any new row must separate proposal
coverage from verifier correctness. Use `--unique-task-gate` as the fast
all-400-file canonical measurement while continuing scale-space and
visual-program coverage work.

## Step 1: Learnable Proposal Trace

Stop using manual task search as the main loop. The loader now accepts
`--arc-proposal-trace path` and writes `crlp.arc_proposal_trace.v1` JSONL rows
after ordinary candidate evaluation. Each row includes:

- task id and normalized task features;
- candidate family, op, name, structural and task-conditioned scores;
- candidate slots (`input_color`, `output_color`, selector kinds, render mode,
  displacement, scale, and hypothesis slots);
- verifier labels (`train_consistent`, `test_supported`, `test_exact`, checked
  cells, mismatches, cell accuracy, prediction hash, and program hash).

The emitter records all train-consistent/exact rows plus a bounded rejected
sample (`64` negatives per task by default in the current hook). This creates
supervised labels for family and slot proposal without allowing the learner to
decide correctness.

Current full unique-task trace command:

```
./build/float/arc_agi \
  --task-dir build/float/arc_agi_public/data/training \
  --unique-task-gate --induction --min-induction-evidence 1 \
  --loader-summary-only \
  --arc-proposal-trace /tmp/arc_proposal_trace_unique.jsonl
```

Result: 25,750 valid trace rows over 400 tasks in ~22s, preserving 43 exact /
46 certified unique-task coverage. The trace contains 150 train-consistent /
test-supported proposal rows and 144 exact candidate rows.

## Step 2: Train Tiny Family/Slot Proposer

The first implementations are online trace proposers. `trace_prior` learns
positive-evidence log scores over family, op, color, selector, render, and scale
slots from previous verifier rows. `trace_task_prior` adds task-conditioned
success prototypes over the same normalized task features emitted in
`crlp.arc_proposal_trace.v1`. `trace_linear` adds a tiny online logistic head
and `trace_mlp` adds a small listwise MLP trained from the same emitted rows.
All report as normal
`crlp.program_proposer.summary.v1` strategies.

Measured unique-task gate:

| Budget | Structural utility | Existing learned utility | Trace-prior utility | Trace-task-prior utility | Trace-linear utility | Trace-MLP utility |
|--------|--------------------|--------------------------|---------------------|--------------------------|----------------------|-------------------|
| 1 | 0.01635728 | 0.01635728 | 0.01423984 | 0.01423984 | 0.01423984 | not measured |
| 2 | 0.01635728 | 0.01635728 | 0.01423984 | 0.01423984 | 0.01558089 | not measured |
| 4 | 0.01635728 | 0.01635728 | 0.01551030 | 0.01621612 | 0.01589850 | not measured |
| 8 | 0.01635728 | 0.01635728 | 0.01653374 | 0.01653374 | 0.01589850 | not measured |
| 64 | 0.01635728 | 0.02459768 | 0.01986872 | 0.02073334 | 0.01986872 | 0.02018634 |

Interpretation: learned family/slot evidence beats structural ordering at
budgets 8 and 64. Task conditioning improves the global trace prior at budget 4
and 64, but it is still below the existing learned scorer at budget 64. The
flat logistic `trace_linear` head helps at budget 2 but regresses at budget 64,
and the listwise `trace_mlp` remains below `trace_task_prior`. The holo lesson
holds: a measured prototype/index memory is the best trace-derived proposer so
far; neural heads should be confidence-gated or trained on residual pairwise
mistakes before they can override it.

Use the proposal trace as the dataset for the next learner:

- family head: task features -> candidate-family priority;
- slot head: task features + family -> likely color/selector/scale/render
  slots;
- target: train-consistent first, exact when public labels are available;
- evaluation: matched budget against grammar order, structural heuristic,
  learned current scorer, random, and oracle orderings.

Acceptance target: improve `crlp.program_proposer.summary.v1` utility at tight
budgets without reducing `--unique-task-gate` exact/certified counts.

## Step 3: Scale-Space Audit

Audit unsolved tasks tagged as scale-up/scale-down or output-size mismatch.
Record:

- task id;
- input/output shape relation;
- existing candidate families present;
- whether `ARC_RULE_MULTISCALE` tries the needed factor;
- whether the failure is candidate generation or candidate application.

## Step 4: Implement Narrow Scale-Space Candidate Evaluation

Add the smallest candidate-evaluation change that covers audited failures:

- try factors 2, 3, and 4 where output dimensions justify them;
- keep factor attempts explicit in evidence rows;
- avoid new hand-coded task-specific operators.

## Step 5: Visual-Program Proposal Coverage

If scale-space is flat, move to visual-program proposal coverage:

- inspect zero-candidate / low-candidate tasks from the learned visual-program
  audit;
- add proposal mining only when it generalizes across a failure family;
- report legacy-template dependency separately from generic visual-program
  coverage.

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Capped augmented-row certified tasks | 70 | 75+ |
| Unique-task certified/exact tasks | 46 / 43 | 49+ / 46+ |
| VSA two-pass persistence | Yes, `stored_count=114` | Keep |
| Object-scene ranker rows | Yes, no uplift | Park |
| Proposal trace rows | `crlp.arc_proposal_trace.v1`: 25,750 rows / 400 tasks / 144 exact candidate rows | Tiny family/slot proposer |
| Scale-space/visual-program audit | Reflected + marker motif + axis tiling + period extension measured | Continue fixed-scale miss mining |
| Runtime (unique-task gate) | 25-29s | <60s |

## Risks

- **Stale object files:** Always clean rebuild.
- **VSA representation churn:** Already observed: both `arc_grid_hologram_v1`
  and `arc_object_scene_v1` persistence work but do not improve ordered metrics.
  Mitigation: park ARC VSA scoring until proposal coverage changes.
- **Flat trace capacity:** Do not build `HoloForest` first. Capacity matters
  after a representation shows signal or a flat trace degrades with growing
  stores.
- **Scale-space slowdown:** all-files augmented induction exceeded the current
  runtime window. Use the implemented `--unique-task-gate` before using
  all-files augmented sweeps as a routine gate.
