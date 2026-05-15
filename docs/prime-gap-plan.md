# Prime-Gap Experiment Plan

This document covers the prime-gap experiments specifically. The repository-level thesis is in `README.md` — hybrid neural-symbolic scoring under exact verification, with three findings (certificate-gating dominates raw scoring, the symbolic gate can be learned, KL-regularized RL prevents calibration collapse). Prime gaps are one load-bearing benchmark for that thesis, not the project goal; `grid_path` now carries the first non-prime certified gate check.

## Why Primes Are the Substrate

Prime gaps give the project four properties that most exact-verifier toy domains lack:

- A fast deterministic verifier at all sizes (segmented sieves, wheel filters, block proofs).
- A compact, totally-ordered candidate space (gaps `2..256` plus an overflow action).
- An exhausted classical baseline — the sieve cannot be beaten by a small neural net, which sharpens the question to "what does the neural component *add* under perfect feedback?"
- Rich algebraic structure (residue classes, divisibility, certificates) that a small rule grammar can exploit, supplying the learned symbolic gate in Finding 2.

We are not trying to find primes faster. We use primes as a clean exact-verifier domain for studying learned proposal policies and learned symbolic filters under perfect mathematical feedback.

## Implementation Layout

The prime-gap implementation is now a real module split under `src/prime_gap/`:

- `data.c`: sieve construction, train/validation/test split generation, action/gap mapping.
- `features.c`: candidate feature encoding, ranking helpers, wheel-verifier work accounting.
- `symbolic_wheel.c`: learned symbolic-wheel discovery, persistence, and reporting.
- `rule_search.c`: expanded rule grammar, certificate-gated selection, rule trace output.
- `block_verifier.c`: segmented block-verifier diagnostics and assisted block scans.
- `eval.c`: policy sampling, calibration metrics, model evaluation, traces, baselines.
- `train.c`: supervised, pairwise-ranking, verifier-RL, and ablation training loops.
- `run.c`: main prime-gap experiment orchestration.
- `replay_compare.c`: trace replay and checkpoint comparison CLIs.

`src/prime_gap/internal.h` is the private contract between those modules. `src/nn_tasks_prime_gap.c` is retained as a test-only module index, while production builds compile `src/prime_gap/*.c` separately.

## Avoid This Misframing

Do not frame the project as:

```text
We train a neural network to find primes faster than a sieve.
```

Use this framing instead:

```text
Primes are a compact exact-verifier domain for studying certificate-gated hybrid
scoring. The verifier is the oracle and reward source. The neural components
are evaluated by (a) ranking quality on the gated candidate set,
(b) calibration under verifier-guided training, and (c) the soundness and
coverage of the rules the rule-search proposer discovers.
```

## What We Have Learned

### Hybrid wins without residue-feature leakage (load-bearing for Finding 1)

Same scorer, three gates, default benchmark (`sample_count=30000`):

- No gate, minimal features: 9.13% top-1, 78.01% top-10, 2.9994 NLL.
- Hand-wheel gate, minimal features: 45.12% top-1, 99.77% top-10, 1.5902 NLL.
- Learned-rule gate, minimal features: 68.31% top-1, 100% top-10, 1.0172 NLL, ~4x lower model ms/sample.

The defensible deltas are now sharper: **9.13 -> 68.31** top-1 from clean raw scorer to learned-rule hybrid, and **45.12 -> 68.31** from fixed hand-wheel to learned rule program. The former shows the symbolic gate doing the work once residue flags are removed from the scorer; the latter isolates the extra value of the learned candidate-divisibility rule set over a fixed small-prime wheel. The canonical learned rules are accepted by algebraic certificate (`gcd(m, r) > 1` for `n % m == r`), not by scanning held-out primes. The old 45.12% raw-scorer number is now labeled as a full-sieve-feature leakage ablation, not a headline result.

### Certificate-gated rule discovery is sound (load-bearing for Finding 2)

`prime_rule_search` on the canonical run:

```text
proposed:        32,637 rules
verified:         4,692   ← survived verifier screens
accepted:            52   ← unconditional candidate filters
false rejections:     0
direct exact work:  7.20  → after accepted rules: 1.47 (79.64% saved)
```

The default proposer now receives only structural rule features for unconditional candidate filters: kind, moduli, residues, zero-residue indicators, gcd ratios, and derived candidate residue. The expanded grammar can still express conditional gap filters under `make prime-rule-search-conditional`, but that mode is not the canonical result. The proposer does not receive `train_rejects`, validation support, utility, or false-rejection counters as inputs; empirical support is the target it is asked to predict. Zero false rejections is a property of the certificate verifier, not of training, so the soundness guarantee does not depend on the scorer being well-trained.

The fixed-budget baseline is now explicit (`make prime-rule-search-baselines`). At a 16-rule budget on the same grammar, split, and verifier, the neural structural proposer saves 67.03% of exact work, one deterministic random ordering saves 46.30%, and grammar order / structural heuristic / validation oracle each save 73.22%. This is a negative result for the stronger neural claim: the current rule set is useful because the certificate-gated grammar is useful, not because the neural proposer is essential. The load-bearing Finding 2 claim is therefore "certificate-gated structural rule discovery is sound and recovers most of the wheel"; a future neurosymbolic claim needs a richer grammar and a proposer that beats grammar-order and structural-heuristic baselines under the same budget.

### Calibration collapse and the KL fix on prime-gap (load-bearing for Finding 3)

From the same minimal-feature supervised checkpoint:

- Supervised: ECE 0.0704, T=1.20 after rescaling.
- Strict verifier RL: ECE 0.1271 unscaled, needs T=1.50 to recover.
- KL/entropy-regularized RL: ECE 0.0604 unscaled, T=1.00 after rescaling.

Ranking (top-1, top-10) is essentially unchanged across all three. The RL story is therefore not "RL improves the policy" — it is "naive RL breaks calibration on this task; the standard RLHF-style regularizer prevents that." Whether the same failure mode and fix transfer to `grid_path` and `delayed_claim` is open work, not yet in the canonical results.

### Verification is still classical

Model-assisted exact verification does not beat wheel or block verification, because wrong proposals add proof obligations. This is expected for exact next-prime verification and clarifies the contribution: the value is in the gated-candidate-set ranking, not in replacing the verifier.

## Primary Metrics

Capability:

- top-1/top-5/top-10 gap-action accuracy
- rank of the true action, MRR, mean absolute gap error
- NLL on held-out ranges, probability assigned to the true gap
- generalization by prime-size bucket
- sample efficiency
- supervised-only vs verifier-guided fine-tuning

Calibration:

- top-1 confidence, correct vs incorrect confidence
- ECE, high-confidence coverage and accuracy
- post-hoc temperature
- reliability bins (stored in `crlp.compare.v1`; not yet plotted in README)

Symbolic gate (rule search):

- proposed / verified / accepted rule counts
- false-rejection rate (must remain zero by construction)
- verifier work saved versus hand-wheel and versus direct scan

Ablations:

- empirical gap-frequency baseline
- log-only, gap-only, candidate-residue-only scorers
- linear minimal-feature scorer (matches hidden network on top-1 under clean inputs)
- full-sieve-feature scorer (explicit leakage ablation; not a clean model)
- pairwise-ranking ablation

Verifier diagnostics (informative, not headline):

- direct scan exact primality probes
- wheel-filtered exact probes
- segmented block-verifier cost
- model-assisted verifier cost

## Open Work

Ordered roughly by priority for the writeup.

1. **Richer proposer benchmark.** The fixed-budget prime baseline now shows grammar order and a simple structural heuristic beat the current neural proposer. The next test needs a richer grammar where ordering is genuinely non-trivial, plus random seeds and matched-budget grammar/heuristic baselines.
2. **Multi-seed deltas with confidence intervals.** Default sweep currently runs 1 seed; harness supports more (`PRIME_GAP_EXPERIMENT_SEEDS=...`). Report 95% CIs on the minimal-vs-leakage, strict-vs-regularized, and hand-wheel-vs-learned-rule deltas across >=5 seeds.
3. **Reliability diagrams.** `crlp.compare.v1` already carries `reliability_bins`; render them.
4. **Calibration sweep over larger held-out ranges.** Expand the KL/entropy grid; check whether the validation-selected weights are stable across prime-size buckets.
5. **Stronger verifier-aware baselines.** Hand-coded wheel ranking, linear scorer with explicit wheel-survivor features, conditioned frequency over larger residue signatures.
6. **Richer grid_path gate baseline.** The current `grid_path` gate is a budgeted legality sanity check: a structural proposer ranks five certified edge/block rules, accepts the top three using train+validation only, and reports test false rejections separately. The raw scorer no longer receives explicit legality flags. Add admissible dead-end or distance-bound pruning rules that reduce BFS work without becoming a hand-coded shortest-path solver.
7. **Expanded ablations.** Different small-prime feature sets, remove overflow action, train on wheel survivors only.
8. **RL-only experiments.** No supervised warm start, strict-only, entropy-only, baseline/advantage variants. Useful for triangulating the calibration-collapse story.
9. **Capability-oriented task variants.** Find any prime after `p`, rank prime candidates in a fixed interval, rank wheel-surviving candidates only, predict overflow probability for larger gaps.
