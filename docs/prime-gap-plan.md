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
- binned calibration curves
- temperature-scaled calibration
- generalization by prime-size bucket
- sample efficiency
- supervised-only vs verifier-guided fine-tuning

Ablation metrics:

- empirical gap-frequency baseline
- hand-coded wheel-frequency baseline
- residue-signature conditioned frequency baseline
- linear full-feature scorer
- log-only candidate scorer
- gap-only candidate scorer
- candidate residue-only scorer
- wheel-survivor-only scorer
- entropy-regularized full scorer
- full scorer

Verifier diagnostics:

- direct scan exact primality probes
- wheel-filtered exact probes
- segmented block-verifier cost
- model-assisted verifier cost

Verifier diagnostics are useful, but they are not the headline success criterion.

## Current Result

The learned policy is a strong ranker.

The ablation and baseline results separate ranking from calibrated probability estimation.
Candidate residue features alone can rank well because they expose a learned soft sieve, but their
probability estimates are poorly calibrated. The hand-coded wheel-frequency baseline is now the most
important non-neural comparison: it takes empirical gap frequencies and masks candidates eliminated
by small-prime divisibility. It reaches roughly the same top-k regime as the neural scorer on the
default split, which means a large part of the current task is explained by cheap sieve structure.

The linear full-feature scorer is also competitive with the MLP. That is useful negative pressure on
the project: before adding larger networks, we need task variants where nonlinear learning has
something meaningful to do.

The full feature set improves held-out negative log likelihood and true-action probability while
preserving strong top-k ranking behavior. Temperature scaling now uses a chronological calibration
split carved from the training range instead of reusing the fit split. Strict verifier RL tends to
increase confidence more than accuracy; post-hoc temperature scaling is therefore useful as a
calibration diagnostic. Entropy-regularized supervised training is available as a simple pressure
against overconfident policies.

The wheel-survivor task is implemented. It trains and evaluates only over candidates that survive the
small-prime feature wheel. On the current feature set this does not materially improve ranking,
which suggests that the remaining error is mostly gap-frequency ambiguity among plausible survivors,
not failure to reject obvious composites.

The deterministic verifier remains the correctness mechanism.

Model-assisted exact verification currently does not beat wheel or block verification, because wrong proposals create extra proof obligations. This is expected for exact next-prime verification.

## Next Work

1. Improve calibration.
   - tune entropy weight across prime-size buckets
   - compare temperature scaling against label smoothing
   - tune temperature by ECE as well as NLL

2. Make the task harder.
   - larger held-out ranges where prime gaps exceed the current action cap
   - explicit overflow prediction and top-k conditional on overflow
   - fixed interval candidate ranking rather than next-gap-only ranking

3. Expand ablations.
   - different small-prime feature sets
   - remove overflow action
   - compare raw residues against divisibility flags
   - compare chronological vs randomized train splits

4. Add RL-only experiments.
   - no supervised warm start
   - strict reward only
   - entropy bonus
   - baseline/advantage variants

5. Add capability-oriented task variants.
   - find any prime after `p`
   - rank prime candidates in a fixed interval
   - predict overflow probability for larger gaps

The most promising next implementation is a harder task variant. The current metrics show the policy
can rank candidates well, but the strongest classical wheel baseline explains much of that lift.
The next experiment should keep the perfect verifier but ask the model for something less reducible
to small-prime masking plus empirical gap frequency.
