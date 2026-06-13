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
candidates ahead of exact candidates; the next version needs a cleanup candidate
set or verifier-derived residual target, not a raw score boost.

## Near-Term Backlog Change

Replace "tune HoloNet residual confidence" with:

1. Build a C HoloForest/partitioning sandbox around `HolographicRanker` and
   report flat/tree/forest recall rows. The first partitioned-ranker sandbox is
   complete; the remaining step is tree/forest routing if family partitions are
   too skewed.
2. Keep the guarded ARC pressure-memory instrument in place, but leave
   `--arc-holo-pressure-residual` disabled by default until it shows crossings
   into budget without matching crossings out.
3. Replace raw pressure score boosts with cleanup-style candidate buckets or a
   verifier-derived residual target.
4. Use `crlp.arc_holo_residual_diagnostic.v1` as the promotion metric: budget
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
