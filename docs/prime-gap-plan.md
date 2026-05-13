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
- mean absolute gap error
- cross-entropy on held-out ranges
- probability assigned to the true gap
- calibration of policy probabilities
- generalization by prime range
- sample efficiency
- supervised-only vs verifier-guided fine-tuning

Verifier diagnostics:

- direct scan exact primality probes
- wheel-filtered exact probes
- segmented block-verifier cost
- model-assisted verifier cost

Verifier diagnostics are useful, but they are not the headline success criterion.

## Current Result

The learned policy is a strong ranker.

The deterministic verifier remains the correctness mechanism.

Model-assisted exact verification currently does not beat wheel or block verification, because wrong proposals create extra proof obligations. This is expected for exact next-prime verification.

## Next Work

1. Add stronger non-neural baselines.
   - empirical gap frequency
   - frequency conditioned on divisibility flags
   - hand-coded wheel ranking
   - linear/logistic scorer

2. Add ablations.
   - no `log(p)`
   - no `normalized_gap`
   - only gap frequency
   - only divisibility flags
   - different small-prime feature sets

3. Add RL-only experiments.
   - no supervised warm start
   - strict reward only
   - entropy bonus
   - baseline/advantage variants

4. Add capability-oriented task variants.
   - find any prime after `p`
   - rank prime candidates in a fixed interval
   - rank wheel-surviving candidates only
   - predict overflow probability for larger gaps

Calibration and loss reporting are now in the experiment output. The most promising next implementation is stronger baselines plus ablations. That will tell us whether the neural policy is learning beyond empirical gap frequency and small-prime divisibility.
