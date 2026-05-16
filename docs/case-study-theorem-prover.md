# Case Study — Theorem-Prover Lemma Selection

Status: proposed case study. The engine in `include/nn_symbolic.h` is sufficient
to host this task; the implementation work is dataset wiring, a domain-specific
rule grammar, and a certificate verifier. No new neural primitives are needed.

## Why this domain

Theorem-prover lemma selection is a domain the field cares about for reasons
the prime-gap experiment never had:

- **Adoption pressure.** Modern prover work (Lean Mathlib, Coq, Isabelle/HOL)
  uses learned tactic/lemma selection routinely. A neuro-symbolic substrate
  that demonstrates matched-budget honesty would be cited.
- **Exact verifier.** The prover itself is the verifier. A proposed lemma is
  "accepted" only if the prover discharges the goal using it; this is
  exact-by-construction and external to the neural model.
- **Non-enumerable rule space.** Lemma libraries are large (Mathlib has
  ≈ 200k lemmas, Mizar has ≈ 60k). Selection-mode enumeration is infeasible at
  deployment time; the proposer's job is to allocate a verifier-call budget
  among lemma candidates. This is the **discovery-mode** regime the prime-gap
  grammars never reached.
- **Public benchmarks exist.** HOLStep, MetaMath, MizAR, LeanDojo, ProofNet —
  the field has labeled (goal, useful-lemma) pairs already.

This is the cleanest available domain where the engine's discovery-mode
abstractions can demonstrate value that selection-mode prime-gap rules
structurally cannot.

## Formal contract under `NNSymbolicDomain`

| Concept | Domain instantiation |
| --- | --- |
| state | proof obligation (goal term + local context) |
| action space | available tactic/lemma applications |
| candidate enumerator | lemma library filtered by syntactic compatibility (e.g. unifiable head symbol) |
| exact verifier | the prover: apply the lemma, check whether the obligation is discharged or reduces to a smaller obligation |
| rule grammar | lemma library entries (terms) + optional simple combinators (rewrite, specialise) |
| certificate checker | proof of well-formedness: lemma's type unifies with the goal type modulo metavariables |
| utility | discharge rate per verifier call on the training set |

The certificate is straightforward in the discovery setting: a lemma is
"admissible" if it can be applied (type-checks against the goal). The verifier
then determines whether it actually closes or shrinks the obligation.

## Rule grammar

Minimum viable grammar (Phase 1):

```text
rule := apply(lemma_id)
```

A "rule" is just a library lemma identifier. The certificate verifier checks
that the lemma's conclusion unifies with the goal's head term (or one of its
subterms) modulo a configurable depth bound. The verifier then runs `apply` and
records whether the obligation was closed, reduced (new sub-obligations of
smaller term size), or unchanged.

Extended grammar (Phase 2):

```text
rule := apply(lemma_id)
      | rewrite(lemma_id, direction)
      | specialise(lemma_id, hypothesis_id)
```

The extended grammar is non-trivially larger and benefits from a proposer that
can prioritise. Each operator has an admissibility certificate that is cheap to
check.

## Why this fits the engine

Mapping into `NNSymbolicDomain` callbacks:

- `enumerate_rules`: iterate the lemma library, optionally filtered by a
  cheap syntactic admissibility check.
- `encode_rule`: structural features of the lemma — number of premises, head
  symbol category (logical connective vs arithmetic vs set-theoretic), term
  size, presence of metavariables, declared lemma class (`simp`, `norm_cast`,
  user-tagged). For Phase 1, no goal-conditioning required.
- `encode_residual_features` (for the closed-loop / unified policy):
  similarity to already-applied lemmas in the same proof attempt, count of
  remaining sub-obligations, depth into the proof tree.
- `check_certificate`: type-unification check between lemma conclusion and
  goal. Sound by construction.
- `verify_action`: invoke the prover, return obligation discharge status.

`NNRuleCoverage` becomes a bitset over training-set proof obligations × library
lemmas. The bit is set if applying that lemma to that obligation closes or
makes measurable progress on the obligation. Greedy set-cover on this
coverage produces a "high-utility lemma set" — the same `marginal_utility`
target shape the prime-gap rule search already uses.

## Dataset

The most accessible starting point is **HOLStep** (Kaliszyk & Urban, 2017): a
released dataset of HOL Light theorems with positive/negative usefulness
labels for each premise. The format is simple enough to load in C without
running HOL Light:

- Each example: (conjecture, set of useful premises, set of distractor premises).
- ≈ 11k conjectures, ≈ 10M premise pairs.
- Verifier is **the labeled dataset**, not a live prover: a premise is
  "useful" iff the original HOL Light proof used it. This is approximate, but
  it removes the prover-integration burden from Phase 1.

Phase 2 can advance to **LeanDojo** (Yang et al.) for a live verifier; that
requires a Lean install and tactic-state IPC. Phase 1 with HOLStep is
implementable in this repo without external dependencies beyond a dataset
download.

## Matched-budget baselines

The discipline from `docs/experiment-schema.md` applies unchanged:

| Baseline | Mechanism |
| --- | --- |
| random | sample lemmas uniformly from the admissible set |
| grammar order | apply lemmas in library declaration order |
| structural heuristic | rank by hand-coded features: shorter term size first, fewer metavariables, more recently declared (mirrors prover defaults) |
| validation oracle | rank by held-out (validation-conjecture) discharge rate |
| neural (static) | scorer over structural lemma features, no goal/residual conditioning |
| closed-loop | proposer scores conditional on accepted lemmas in the current proof attempt |
| unified scorer | shared neural scorer also trained on the existing four action-scoring tasks |

The metric is **discharge rate at budget K** (e.g., for K ∈ {1, 5, 16}): of the
top-K lemmas the proposer suggests for a held-out conjecture, what fraction
discharge the conjecture. This is HOLStep's standard premise-selection metric
and is directly comparable to published numbers.

A defensible positive result is: **the unified scorer at K=16 beats the
strongest non-neural baseline by ≥ 3pp on held-out conjectures, at a soundness
guarantee equivalent to the dataset's labeling.**

## Expected leakage hazards

Same three categories as prime-gap, with new concrete forms:

1. **Goal-derived features in the lemma-side encoder.** If `encode_rule`
   includes any function of the current goal (term hash, conjecture id), the
   scorer is no longer learning a lemma-only prior — it is learning a
   conditional matcher. That may be desirable for performance but it changes
   the leakage policy from `forbidden` to `priors_allowed` (or `gate_allowed`
   for residual lemma-use features). Tag features accordingly.
2. **Library-order features.** Mathlib and HOLStep both have meaningful
   declaration order (basics → advanced). A feature like "lemma index /
   library size" implicitly encodes "more advanced lemmas come later," which
   correlates with usefulness. This is the equivalent of the prime-gap residue
   leak: a hand-engineered prior masquerading as a learned feature. Either
   exclude it or report under `priors_allowed`.
3. **Test-conjecture overlap.** HOLStep splits by theorem; some splits leak
   conjectures whose proofs share intermediate lemmas with the training set.
   Use the conditional/unconditional split from the HOLStep paper (the
   conditional split is the harder generalisation test).

The audit machinery in `nn_experiment.h` carries these correctly if features
are typed at registration time. No new policy enum needed; lemma-side
verifier-derived features go under the existing `NN_LEAKAGE_GATE_ALLOWED`
proposer category.

## Engine fit

The engine in `include/nn_symbolic.h` is sufficient as-is, except:

- `NNRule::params` currently caps at 12. A lemma id needs one slot; rewrite
  direction adds one; specialisation adds two. Phase 1 fits trivially; Phase 2
  extended grammar uses ≈ 4 slots per rule. No header change needed.
- `NN_RULE_MAX_FEATURES = 64`. Lemma structural features (premise count, term
  size, head symbol bucket, lemma class indicator, declaration index,
  metavariable counts, ≈ 16 features total) fit comfortably. Residual features
  for closed-loop add ≈ 10 more. Still under 64.
- `NNRuleCoverage` is event-indexed; for theorem-prover use, an "event" is a
  (conjecture, useful-lemma) pair on the training set. ≈ 11k conjectures × ≈
  100 useful lemmas/conjecture ≈ 1M events. Bitset is 1M / 8 = 125 KB per
  rule × ≈ 200k rules ≈ 25 GB — too large to materialise. Phase 1 must use a
  subsampled event set (≈ 4096 conjectures, the same shape as
  `max_marginal_states`) or a sparse coverage representation.

The sparse-coverage variant is the one piece of new engine work this case
study likely needs. If it is added, the same sparse representation also helps
the prime-gap conditional grammar at larger budgets.

## Phased plan

| Phase | Deliverable | Effort |
| --- | --- | --- |
| 0 | HOLStep loader, library enumeration, sparse `NNRuleCoverage` variant | 2–3 days |
| 1 | Phase 1 matched-budget grid (random, grammar order, structural, oracle, static neural, closed-loop) on HOLStep premise selection at K ∈ {1, 5, 16}. Replicate published HOLStep baselines as a sanity check. | 2 days |
| 2 | Add unified scorer (per `docs/unified-scorer-proposal.md`). Train shared scorer across prime-gap, grid-path, delayed-claim, ARC, and theorem-prover rule rows. | 1 week |
| 3 | Cross-task transfer evaluation: train shared scorer on three of {prime-gap, grid-path, claim, theorem-prover}, evaluate held-out on the fourth. | 1 week |
| 4 | Live verifier upgrade: replace HOLStep label oracle with LeanDojo. | 2+ weeks |

Phase 0–2 ship as a workshop-paper-ready result. Phase 3 is the strongest
positive cross-task claim available to this project. Phase 4 is conference-
paper ambition.

## Honest scope

This case study can be implemented in this repo with no new external runtime
dependencies through Phase 3, assuming the HOLStep dataset is downloaded as a
flat text artifact. Phase 4 requires Lean ≥ 4.0 and an IPC layer; that is out
of scope for the C-only substrate without a per-task subprocess bridge.

The published HOLStep best-baseline number at K=1 is ≈ 83% accuracy
(Kaliszyk & Urban, 2017, with neural network architectures of the period).
This is a deceptive headline: it is binary classification of (premise,
conjecture) usefulness with paired negatives, not a budget-K selection metric
on the full library. For comparability with published numbers, Phase 1 should
report both the binary classification metric **and** the matched-budget
discharge metric, with a clear note about which baseline corresponds to which
published number.

## What this changes about the paper

If Phase 1 lands, the project has a non-toy verifier-grounded case study in a
domain the field cares about. The "selection-mode prime gaps are degenerate"
critique becomes a feature: the paper presents prime-gap as the
methodology-shakedown task and theorem-prover lemma selection as the test of
whether the methodology produces useful results in a discovery-mode regime.

If Phase 3 also lands, the paper has a cross-task transfer result that the
prime-gap-only version structurally cannot produce.

## Out of scope for this case study

- Tactic synthesis (selecting *what to do* rather than *which lemma to use*).
- Reinforcement learning over proof trees; the proposal stays in
  supervised/imitation mode on greedy targets, matching the rest of the repo.
- Live integration with multiple provers; one prover (HOLStep → Lean) is
  sufficient.
- Proof-quality metrics (proof length, term size of constructed terms).
  Discharge rate at fixed budget is the headline; everything else is
  diagnostic.
