# Prime-Gap Proposal Policy Plan

## Framing

This is a verifier-guided learning environment, not an attempt to beat deterministic prime generation.

The deterministic sieve/verifier is the oracle. It gives us:

- exact labels for supervised warm start
- exact rewards for verifier-guided updates
- exact evaluation of every proposal
- a known baseline for how inefficient the neural policy is

The neural network's job is to learn a proposal distribution:

```text
policy(gap_action | current_prime)
```

The primary question is whether a small C neural network can learn and generalize useful structure in prime-gap vectors under perfect mathematical feedback.

## What We Have Learned

The model learns prime-gap structure.

Across the current split sweep, supervised training moves the model from useless ranking to strong proposal quality:

- top-1 action accuracy roughly 39% to 45%
- top-5 roughly 92% to 95%
- top-10 roughly 99%
- average rank/proposal work drops from about 7 to 8 positions to about 2 to 3

That means the model is learning a real ranking prior over gap actions.

The model does not reduce exact next-prime verification work under the verifier models tested so far.

This is not a failure of the capability experiment. It clarifies the contract: exact next-prime verification requires proving all earlier candidates composite. Classical algorithms already do that proof as part of finding the prime.

## Avoid This Misframing

Do not frame the project as:

```text
We train a neural network to find primes faster than a sieve.
```

Use this framing instead:

```text
We use primes as a compact exact-verifier domain for studying learned proposal policies.
The verifier is the oracle and reward source. The neural net is evaluated by proposal quality,
generalization, calibration, and response to verifier-guided training.
```

## Primary Metrics

Capability metrics:

- top-1/top-5/top-10 gap-action accuracy
- rank of the true action
- mean reciprocal rank
- mean absolute gap error
- negative log likelihood on held-out ranges
- probability assigned to the true gap
- calibration of policy probabilities
- high-confidence coverage and accuracy
- generalization by prime-size bucket
- sample efficiency
- supervised-only vs verifier-guided fine-tuning

Ablation metrics:

- empirical gap-frequency baseline
- log-only candidate scorer
- gap-only candidate scorer
- candidate residue-only scorer
- full scorer

Verifier diagnostics:

- direct scan exact primality probes
- wheel-filtered exact probes
- segmented block-verifier cost
- model-assisted verifier cost

Verifier diagnostics are useful, but they are not the headline success criterion.

## Current Result

The learned policy is a strong ranker.

The new ablation results separate ranking from calibrated probability estimation. Candidate residue
features alone can rank well because they expose a learned soft sieve, but their probability estimates
are poorly calibrated. The full feature set improves held-out negative log likelihood and true-action
probability while preserving the strong top-k ranking behavior.

Post-hoc temperature scaling now uses a held-out validation slice from the high end of the training
range instead of refitting on the training rows. Strict verifier RL still mostly creates a logit-scale
problem: on the default split it keeps top-1 at 45.12%, but pushes ECE to 0.3100 and selects a soft
temperature of 2.50.

KL/entropy-regularized RL is the cleaner training-side fix. With both strict and regularized RL
starting from the same cloned supervised checkpoint, the regularized branch keeps top-10 at 99.75%,
reduces unscaled NLL from 2.3612 to 1.5878, and reduces unscaled ECE from 0.3100 to 0.0821.
Temperature scaling still helps, but the selected temperature falls back to 1.20 instead of 2.50,
which confirms that the regularizer prevents most of the post-RL logit inflation.

Regularization weights are now CLI arguments, and `make prime-gap-regularization-sweep` runs a
small default-range grid so the calibration/ranking tradeoff can be checked without editing code.

The deterministic verifier remains the correctness mechanism.

Model-assisted exact verification currently does not beat wheel or block verification, because wrong proposals create extra proof obligations. This is expected for exact next-prime verification.

## Next Work

1. Improve calibration.
   - expand the KL/entropy sweep across larger held-out ranges
   - add validation-selected regularization weights instead of fixed defaults

2. Add stronger non-neural baselines.
   - frequency conditioned on divisibility flags
   - hand-coded wheel ranking
   - linear/logistic scorer

3. Expand ablations.
   - different small-prime feature sets
   - remove overflow action
   - train on wheel survivors only

4. Add RL-only experiments.
   - no supervised warm start
   - strict reward only
   - entropy bonus
   - baseline/advantage variants

5. Add capability-oriented task variants.
   - find any prime after `p`
   - rank prime candidates in a fixed interval
   - rank wheel-surviving candidates only
   - predict overflow probability for larger gaps

The most promising next implementation is calibration improvement. The current metrics show the policy can rank candidates well, but confidence is not yet a reliable probability of correctness.
