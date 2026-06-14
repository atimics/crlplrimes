# HoloNet Research Brief

Updated 2026-06-13 after reviewing the current HoloNet literature and
`AnOversizedMooseWithSocks/holostuff` at commit `7cb093f`.

## Scope

The word "HoloNet" is overloaded. For this repo, it should mean
capacity-managed holographic/vector-symbolic associative memory, not neural
computer-generated holography for displays. Optical HoloNet work is useful as a
calibration analogy, but it is not the algorithmic substrate we should wire into
ARC proposal learning.

## Relevant Research Line

Tony Plate's Holographic Reduced Representations remain the right root model:
fixed-width vectors represent role bindings and compositional structures through
circular convolution, approximate unbinding, and cleanup memory. The crucial
constraint is explicit in the original paper: recovered items are noisy, and
the noise grows with the number of associations in one trace. Cleanup and
capacity management are not implementation details; they are part of the model.

Modern HRR work confirms the same shape. The NeurIPS 2021 HRR paper revisits
HRRs as differentiable neuro-symbolic components and reports that numerical
conditioning matters enough to require projection. Residual/attentional VSA
architectures make the same engineering point from another angle: deep or
stacked holographic operations need residual-style conditioning to remain
trainable.

The neural-holography display papers use "HoloNet" differently: HoloNet is a
neural generator for phase-only holograms, often with camera-in-the-loop
calibration. That line does not map directly to our ARC proposer, except for one
discipline we should borrow: optimize against the real measurement loop rather
than a pretty internal proxy.

## Holostuff Takeaways

`holostuff` is reference material only because the repo has no visible license.
The useful ideas are safe to reimplement independently:

- A single trace has a hard capacity cliff. The implementation and docs treat
  partitioning as the normal fix, not an optional scale improvement.
- Routing beats global blending when the store is clustered. Random partitions,
  HoloTree/HoloForest routing, and semantic prefixes all reduce crosstalk by
  keeping each local trace small.
- Cleanup is mandatory after unbinding. Chained meaning works when noisy
  vectors snap back to known symbols between steps; direct algebraic chaining
  accumulates cross-term noise.
- Measurement beats structural priors. Several attractive mechanisms in the
  notes are recorded negatives because held-out accuracy, compression bits, or
  recall@k did not improve.
- Reflex/hot-path caches help only when query traffic is skewed and must carry
  a flux guard so uniform traffic does not pay the cache tax.

## Implications For This Engine

Our current `trace_holo_residual` is intentionally conservative, but the public
diagnostics show the flat trace is too weak and too coarse: exact candidates move
by tens of ranks yet remain far outside budget 64. The next HoloNet step should
not lower gates or add score weight. It should change the memory geometry.

The first C sandbox for that geometry is now implemented as
`HoloPartitionedRanker`, a deterministic nearest-anchor router over multiple
local `HolographicRanker` traces. It is intentionally simpler than a full
HoloForest, but it tests the same capacity claim: keep each trace small before
asking it to discriminate actions. The 2026-06-13 `holo_experiments --mode
partitioned-retrieval` run emits `crlp.holo_partitioned_retrieval.v1` rows. At
2048 dimensions, a flat trace is about `0.50` top-1 once loaded with 64-1024
pairs; 32 partitions lift top-1 to `0.98` at 64 pairs, `0.85` at 128, `0.78` at
256, `0.63` at 512, and `0.55` at 1024. The measured rule of thumb is that the
local max load needs to stay below roughly 50 pairs before the routed trace is a
clear win.

The highest-value design is a capacity-managed ARC residual proposer:

1. Keep `trace_task_prior` as the baseline and verifier-owned certification as
   the safety boundary.
2. Add a second HoloNet memory for rank pressure: exact candidates missed by the
   baseline budget are stored more strongly than generic positives.
3. Partition that pressure memory by candidate family/op or by a tiny
   deterministic HoloForest over task features, so each trace stays below the
   flat-memory crosstalk limit.
4. Score through cleanup-style candidate sets: compare recalled context against
   candidates in the routed bucket and emit rank diagnostics for exact
   candidates near budget boundaries.
5. Promote only when the diagnostic schema shows exact candidates crossing into
   budget without matching crossings out.

A first direct pressure-memory wiring was tested on 2026-06-13 and not promoted:
on the public unique-task gate it moved two exact candidates into a budget window
but moved eight out, with lower utility at budgets 8, 32, and 64. The follow-up
guarded/local implementation now trains an opt-in partitioned pressure memory
from exact candidates missed by the active budget grid, and diagnostics report
`holo_pressure_updates`. Default scoring leaves this pressure residual disabled:
the public unique-task run records 461 pressure updates, matches
`trace_task_prior` utility at budgets 8/16/32/64, and keeps 0 crossings in / 0
crossings out. Enabling `--arc-holo-pressure-residual` is still negative: 2
crossings in, 27 crossings out, with lower utility at 8, 16, and 64. The useful
failure mode is that even family/local pressure can boost related but wrong
candidates ahead of exact candidates. The follow-up `trace_holo_bucket` strategy
uses the same pressure memory as a cleanup/review bucket instead of a raw score
boost: exact candidates already inside the baseline budget are pinned, and only
a tiny pressure-retrieved outside-budget review window can enter. On the public
unique-task gate it improves utility over `trace_task_prior` at every budget
(`0.01685136/0.02036279/0.02122741/0.02265669` at 8/16/32/64), raises budget-64
certified programs from 29 to 31, and emits `crlp.arc_holo_bucket_diagnostic.v1`
with 8 crossings into budget and 0 out.
The tight-grid check uses a budget floor: `trace_holo_bucket` now matches
`trace_task_prior` at budget 1, improves utility at budgets 2, 4, and 8, and
bucket diagnostics show 4 crossings into budget and 0 out.
The prior audit reran the bucket strategy against `trace_task_prior` across
task-order perturbations, budget grids, the min-budget floor, and fixed/auto
review-window sizes. Each raw ARC JSONL log is kept, while `summary.jsonl`
records one
`crlp.arc_holo_bucket_robustness.run.v1` row per matched budget and
`aggregate.jsonl` records median/min/max deltas plus crossed-in/out totals.
The harness now also accepts comma-separated
`--min-pressure-updates` values and writes `verdict.jsonl` /
`verdict.csv` rows under `crlp.arc_holo_bucket_robustness.verdict.v1`; those
rows collapse each grid/floor/warmup/review-window configuration into a
promotion decision. The script target that produced these rows is retired until
the sweep is ported into C orchestration. This is the promotion boundary: the
bucket is a claim only if positive utility survives alternate orderings with no
budget-1 regression and no exact candidates crossed out of budget.

The first default full public robustness run uses original order plus
`shuffle:20240516`, tight and standard grids, min budget 2, and review windows
`auto/1/2/4`. It keeps zero crossed-out exact candidates across all rows. The
stable configs are `auto` and fixed review slot `1`: both preserve tight budget
1 and avoid negative utility deltas, while still producing positive median
standard-grid deltas. Fixed slots `2` and `4` are useful at budgets 4/8 and
standard-grid budgets, but they regress tight budget 2 in original order; keep
them as exploratory settings, not defaults.

The bucket invocation gate is now explicit rather than implicit in the ranker.
`--arc-holo-bucket-min-pressure-updates` controls the pressure-memory warmup,
the min-budget floor keeps budget 1 closed, and
`crlp.arc_holo_bucket_gate.v1` records gate-open state, review slots,
pressure-memory size, outside-budget pressure candidates, exact pressure
candidates, and max raw pressure. The focused public gate audit over `auto,1`
opens 776/800 task-budget rows after warmup, opens 0/800 rows at tight budget 1,
and preserves the stable no-regression result. This row is the training surface
for a sharper learned invocation policy; it does not make raw pressure residuals
safe.

The first learned invocation policy is now implemented behind
`--arc-holo-bucket-learned-gate`. It is an online logistic gate that can close,
but not bypass, the fixed budget/warmup/review-slot gate. The diagnostic row
now records fixed-gate state, learned-gate score, example count, and threshold.
Fixture smoke shows the default learned threshold `0.5` is conservative, while
`--arc-holo-bucket-learned-gate-threshold 0.3` recovers the fixture crossing
with fewer openings and zero crossed-out exact candidates. Treat it as an
experimental proposer-control policy until the robustness verdict rows pass on
public task-order sweeps. A bounded 40-task public smoke at threshold `0.3`
opened 21/320 tight and 28/320 standard task-budget rows, crossed no exact
candidates in or out, and therefore passed no-regression without showing uplift.
The learned-gate surface now emits `review_exact_candidates`,
`closed_review_exact_candidates`, mean label target, mean score, and review
margin into the robustness rows. On that public smoke there were 18 tight and 24
standard outside-budget exact pressure candidates but zero review-window exacts,
so the bottleneck is the pressure review selector/reranker rather than a missed
learned-gate opening.

The next selector probe now emits labeled candidate rows instead of hiding the
miss in aggregate gate counts. The review priority blends raw Holo pressure with
a small clamped `trace_task_prior` baseline bonus; this recovers the fixture
crossing at threshold `0.3`, but the bounded 40-task public smoke remains
non-promotable with zero selected exacts. The new
`crlp.arc_holo_bucket_review_candidate.v1` rows record bounded review-frontier
contenders plus every exact pressure candidate outside the frontier. Public
verdict rows now show 18 tight / 24 standard exact review-candidate labels, all
missed by the one-slot/auto selector, with mean exact review rank about 518 tight
and 516 standard and worst exact rank 1101. This is the handoff point from manual
pressure tuning to a learned review reranker.

`tools/arc_holo_bucket_review_selector.c` is the first offline reranker probe
over those rows, exposed as `make arc-holo-bucket-review-selector`. It trains a
pairwise linear selector from proposal-time features only; verifier labels such
as train consistency and test support are excluded from the default feature set.
On the bounded public 40-task diagnostic logs, five task-hash folds move exact
candidates inside the diagnostic candidate set from mean raw loaded rank 17.5 to
mean learned loaded rank 5.67, with 17 held-out exact rows selected top-1 versus 0
for raw review priority. This is useful signal, but not enough to wire into
runtime: the next learner needs either broader negative coverage from the review
surface or a less overfit candidate-family feature set.

## Near-Term Backlog Change

Replace "tune HoloNet residual confidence" with:

1. Build a C HoloForest/partitioning sandbox around `HolographicRanker` and
   report flat/tree/forest recall rows. The first partitioned-ranker sandbox is
   complete; the remaining step is tree/forest routing if family partitions are
   too skewed.
2. Keep `--arc-holo-pressure-residual` disabled; raw pressure boosts remain a
   negative control.
3. Use `make arc-holo-bucket-robustness` as the C promotion audit before
   expanding order, budget, warmup, and review-window perturbations. Keep
   `auto`/`1` as the current promotion candidates and use `2`/`4` only to study
   larger-budget recall.
4. Compare fixed versus learned bucket invocation through that C sweep; the
   learned gate should remain opt-in until verdict rows are both no-regression
   and positive on the public perturbation grid.
5. Learn the pressure review selector from
   `crlp.arc_holo_bucket_review_candidate.v1` rows so outside-budget exact
   pressure candidates enter the bounded review window; track
   `review_candidate_missed_exact_rows`, `mean_exact_review_rank`,
   `review_exact_candidates`, and `closed_review_exact_candidates` in
   `verdict.jsonl`. Use `make arc-holo-bucket-review-selector` as the offline
   learning check before wiring selector weights into `trace_holo_bucket`.
6. Use `crlp.arc_holo_residual_diagnostic.v1` as the promotion metric: budget
   crossings first, utility second, hash changes last.

## Sources

- Tony A. Plate, "Holographic Reduced Representations", IEEE Transactions on
  Neural Networks, 1995:
  https://redwood.berkeley.edu/wp-content/uploads/2020/08/Plate-HRR-IEEE-TransNN.pdf
- Ganesan et al., "Learning with Holographic Reduced Representations",
  NeurIPS 2021:
  https://openreview.net/forum?id=RX6PrcpXP-
- Olin-Ammentorp et al., "Residual and Attentional Architectures for
  Vector-Symbols", 2022:
  https://arxiv.org/pdf/2207.08953
- Peng et al., "Neural Holography with Camera-in-the-loop Training", 2020:
  https://www.computationalimaging.org/wp-content/uploads/2020/08/NeuralHolography_SIGAsia2020.pdf
- `AnOversizedMooseWithSocks/holostuff`, reviewed as reference-only material:
  https://github.com/AnOversizedMooseWithSocks/holostuff
