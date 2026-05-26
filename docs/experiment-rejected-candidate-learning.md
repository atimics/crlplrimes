# Experiment Proposal: Rejected-Candidate Learning From Certificate Reason Codes

Status: implemented as `the retired rejected-candidate workflow`.

## Purpose

The substrate already records rejected symbolic candidates with reason codes such
as `unsafe` and `no_support`. This experiment tests whether those rejections are
useful supervised signal for the next proposer pass, instead of being treated as
discarded search noise.

## Hypothesis

If certificate reason codes are stable across structurally similar rules, then a
lightweight reason-aware prior should move unsafe or unsupported rules out of a
small verifier budget while preserving safe-support rules.

This is a substrate-specific result: it exists because the verifier emits
structured rejection reasons and false-rejection counters.

## Protocol Shape

| Field | Instantiation |
| --- | --- |
| Candidate set | Prime modular rejection rules, modular-gate family rows, and delayed-claim threshold rows |
| Verifier | Task-local rule certificate plus train/validation support counters |
| Certificate/gate | `verifier_result`, deterministic-certificate flag, false-rejection counters |
| Scorer role | Rerank rules before budgeted verification |
| Trace artifact | `crlp.rejected_candidate_learning.summary.v1` |
| Protocol point | Rejected rows become supervised signal without becoming scorer inputs |

## Implementation

The script now runs three rejected-row surfaces:

- `prime_rule_search` over an unconditional candidate-filter grammar, split by
  `filter_modulus`.
- `modular_gate_search` over prime-constellation, squarefree/k-free, and
  Fibonacci/Lucas recurrence gates, split by rule modulus.
- `delayed_claim --rule-search` over threshold and utility-margin rules, split
  by action-balanced structural templates.

Each path writes a raw candidate trace, learns reason-code rates from the train
side of the split, and evaluates held-out candidate rows.

The current rankers are:

- `model_score`: existing neural structural proposer score.
- `reason_prior`: learned reason-code prior from structural buckets only, with
  deterministic grammar-order ties.
- `reason_aware`: learned reason-code prior first, then the existing neural
  `model_score` as an explicit residual tiebreaker.
- `trained_residual`: learned reason-code prior first, then a dedicated linear
  residual trained inside positive-prior train buckets. Its target orders
  accepted rules by marginal utility above safe duplicates, then no-support rows,
  then unsafe rows. The delayed-claim path also trains a safety head over all
  train rows and ranks predicted safe rows before utility, because unsafe
  threshold rows are the observed tight-budget failure mode.

The deterministic tiebreaker matters. The first implementation used a stable
sort, which let `reason_prior` inherit the incoming trace order. Because the
trace is already sorted by `model_score`, that made `reason_prior` and
`reason_aware` look identical. The retained evidence removes that hidden neural
leak from the prior and records top-rule details with both reason and neural
scores.

The summary also emits guard metrics against `reason_prior`: top-k overlap,
delta accepted rules, and delta validation utility. A residual can only support
the neural contribution claim when those deltas are non-negative on the relevant
budget and the selected held-out rows preserve zero validation false rejections.

Run:

```sh
make rejected-candidate-learning
make rejected-candidate-seed-audit
```

Outputs:

- `build/float/rejected_candidate_learning/prime_rule_trace.jsonl`
- `build/float/rejected_candidate_learning/modular_gate_trace.jsonl`
- `build/float/rejected_candidate_learning/delayed_claim_rule_trace.jsonl`
- `build/float/rejected_candidate_learning/reason_learning.jsonl`
- `build/float/rejected_candidate_learning/split_sweep.jsonl`
- `build/float/rejected_candidate_learning/modular_leave_one_family_out.jsonl`
- `build/float/rejected_candidate_learning/paper_table.jsonl`
- `build/float/rejected_candidate_learning/reason_family_sweep.jsonl`
- `build/float/rejected_candidate_learning/summary.md`
- `build/float/rejected_candidate_seed_audit/seed_interval_summary.jsonl`
- `build/float/rejected_candidate_seed_audit/paper_table.csv`
- `build/float/rejected_candidate_seed_audit/summary.md`

The base run also writes `crlp.rejected_candidate_learning.reason_family_sweep.v1`
rows so top-k choices can be audited by `accepted`, `safe_duplicate`, `unsafe`,
`no_support`, and `other`. The seed audit reruns the same protocol over
`REJECTED_CANDIDATE_SEEDS` and summarizes seed intervals for the paper-facing
comparison: baseline candidate score, reason prior, trained residual, accepted
count, unsafe count, validation false rejections, and utility.

## Success Criteria

At small budgets, `reason_prior`, `reason_aware`, or `trained_residual` should:

- select more `accepted` rules than `model_score`,
- select fewer `unsafe` rules than `model_score`,
- preserve zero validation false rejections, and
- improve validation marginal utility.

For the neural contribution claim, `reason_aware` also needs a top-rule ordering
or utility profile that differs from `reason_prior`; otherwise the result should
be stated as a reason-code prior result rather than a neural residual result.
The current positive residual candidate is `trained_residual`, not the legacy
`model_score` tiebreaker.

## Current Result

On the default prime-gap split (`--split-filter-modulus 61`),
`trained_residual` improves the budget-8 result from 6 accepted rules / 216
validation utility to 7 accepted rules / 245 validation utility, with zero
validation false rejections. The legacy `reason_aware` row changes ordering but
drops to 210 utility at the same budget.

The first-class split sweep (`split_sweep.jsonl`) over `43, 53, 61, 71` keeps
the trained residual non-negative relative to `reason_prior` at budget 8:

| Split | Reason prior accepted / utility | Trained residual accepted / utility | Delta utility |
| ---: | ---: | ---: | ---: |
| 43 | 7 / 306 | 8 / 342 | +36 |
| 53 | 7 / 274 | 8 / 290 | +16 |
| 61 | 6 / 216 | 7 / 245 | +29 |
| 71 | 5 / 170 | 5 / 170 | +0 |

The added modular-gate family is a second modular result. At budget 8 on the
default modulus split, `reason_prior` and `trained_residual` both select 4
accepted / 4 safe-duplicate rows with zero false rejections and 3828 validation
utility. The trained residual is non-negative against `reason_prior` across the
split grid `5, 7, 11, 13`, improving the tighter `5` and `7` splits and tying
the default `11` and `13` splits.

The delayed-claim task now uses the same rejected-row learning loop with a
safety-first residual rank key. At budget 8, pure `reason_prior` selects 5
unsafe rows and produces 1692 validation false rejections. The trained residual
selects 2 accepted / 6 safe-duplicate rows, zero unsafe rows, 3961 validation
utility, and zero validation false rejections. The action-balanced split grid
`44, 66, 88` passes the non-negative residual guard with zero false rejections.
This is a real repair of the reason-prior safety hole, although the task-local
`model_score` still has higher utility on this surface.

The modular leave-one-family-out check is mixed rather than solved. Holding out
`prime_constellation`, the trained residual improves budget-8 utility from
16365 to 25139. Holding out `fibonacci_lucas`, it ties the reason prior at
22393. Holding out `kfree_squarefree`, it selects one more accepted rule but
drops utility from 9843 to 9810, so the transfer guard fails. This is now a
useful limitation row: rejected-row learning transfers across some modular
families, but not uniformly across all held-out arithmetic families.

## Failure Modes

- Reason buckets are too domain-specific and do not transfer to held-out
  structural regions.
- The learned prior merely rediscovers a trivial certificate predicate.
- The proposer was already saturated, leaving no room for reason-code learning.
- The neural residual tiebreaker changes ordering but hurts tight-budget
  utility, which means the reusable reason-code signal is real but the residual
  scorer still needs a better target or feature set.
- The dedicated residual overfits positive-prior buckets. Treat a one-split win
  as provisional until it holds across multiple `--split-filter-modulus`
  settings.
- A second task can expose a weaker reason vocabulary: delayed-claim needed a
  safety-first residual head before the reason-prior loop became clean at tight
  budgets.
- Modular family transfer can be locally positive while still failing a held-out
  family by utility; leave-one-family-out is the current guard against
  overstating cross-family reuse.

The second failure is still informative: it shows which rejection reasons should
be promoted into an explicit structural baseline before stronger learning claims
are made.

## Paper Use

If positive, this supports a new substrate claim: rejected candidate rows are
not just audit exhaust; they are reusable supervision for budgeted search. The
safe current wording is: reason-code priors are reusable supervision, a
dedicated residual can improve tight-budget ordering on guarded splits, and a
safety head can turn unsafe rejected delayed-claim rows into useful negative
supervision. The leave-one-family-out modular result should be stated as mixed
transfer, not a uniform cross-family win.
