# Rejected-Candidate Reason Learning

Date: 2026-05-17

This experiment treats certificate result codes as reusable training signal. It now covers the original prime-gap rule search, an additional modular-gate family, and a second non-prime task using the same accepted / safe-duplicate / unsafe / no-support learning loop.

## Coverage

- `prime_gap`: original modular candidate filters from `crlp.rule_trace.v1`.
- `modular_gate`: prime-constellation, squarefree/k-free, and Fibonacci/Lucas recurrence gates from `crlp.modular_gate.rule_trace.v1`.
- `delayed_claim`: threshold and utility-margin claim gates from `crlp.delayed_claim.rule_trace.v1`.

## Ranking Discipline

- `model_score`: task-local neural or structural proposer score, with grammar-order ties.
- `reason_prior`: learned reason-code rate only, with grammar-order ties.
- `reason_aware`: learned reason-code rate first, then `model_score` as the explicit residual tiebreaker.
- `trained_residual`: learned reason-code rate first, then a dedicated linear residual trained inside positive-prior train buckets.

## Prime-Gap Modular Filters

Raw trace: `build/float/rejected_candidate_learning/prime_rule_trace.jsonl`

- Split key: `filter_modulus` <= `61` for train.
- Train rows: 1888; reason labels: {'accepted': 17, 'no_support': 464, 'safe_duplicate': 247, 'unsafe': 1160}
- Eval rows: 2862; reason labels: {'unsafe': 1740, 'no_support': 711, 'safe_duplicate': 404, 'accepted': 7}
- Trained residual rows: 172; reason labels: {'accepted': 17, 'safe_duplicate': 155}
- Stable trained-residual claim at budget 8: PASS

### Budget Results

| Strategy | Budget | Accepted | Safe duplicate | Unsafe | No support | Validation utility | Validation false rejections |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| model_score | 8 | 0 | 0 | 7 | 1 | 0 | 172 |
| reason_prior | 8 | 6 | 2 | 0 | 0 | 216 | 0 |
| reason_aware | 8 | 6 | 2 | 0 | 0 | 210 | 0 |
| trained_residual | 8 | 7 | 1 | 0 | 0 | 245 | 0 |
| model_score | 16 | 0 | 1 | 13 | 2 | 0 | 360 |
| reason_prior | 16 | 7 | 9 | 0 | 0 | 245 | 0 |
| reason_aware | 16 | 7 | 9 | 0 | 0 | 245 | 0 |
| trained_residual | 16 | 7 | 9 | 0 | 0 | 245 | 0 |
| model_score | 32 | 0 | 3 | 23 | 6 | 0 | 672 |
| reason_prior | 32 | 7 | 25 | 0 | 0 | 245 | 0 |
| reason_aware | 32 | 7 | 25 | 0 | 0 | 245 | 0 |
| trained_residual | 32 | 7 | 25 | 0 | 0 | 245 | 0 |
| model_score | 64 | 0 | 7 | 44 | 13 | 0 | 1325 |
| reason_prior | 64 | 7 | 57 | 0 | 0 | 245 | 0 |
| reason_aware | 64 | 7 | 57 | 0 | 0 | 245 | 0 |
| trained_residual | 64 | 7 | 57 | 0 | 0 | 245 | 0 |

### Split Sweep Guard

- Split grid: 43, 53, 61, 71

| Split | Reason prior accepted / utility | Trained residual accepted / utility | Delta utility | Guard |
| ---: | ---: | ---: | ---: | --- |
| 43 | 7 / 306 | 8 / 342 | +36 | pass |
| 53 | 7 / 274 | 8 / 290 | +16 | pass |
| 61 | 6 / 216 | 7 / 245 | +29 | pass |
| 71 | 5 / 170 | 5 / 170 | +0 | pass |

### Budget 8 Top Rules

| Strategy | Top rules |
| --- | --- |
| model_score | reject candidate % 97 == 55<br>reject candidate % 96 == 54<br>reject candidate % 97 == 54<br>reject candidate % 97 == 56<br>reject candidate % 97 == 53 |
| reason_prior | reject candidate % 67 == 0<br>reject candidate % 71 == 0<br>reject candidate % 73 == 0<br>reject candidate % 77 == 0<br>reject candidate % 79 == 0 |
| reason_aware | reject candidate % 97 == 0<br>reject candidate % 91 == 0<br>reject candidate % 67 == 0<br>reject candidate % 71 == 0<br>reject candidate % 73 == 0 |
| trained_residual | reject candidate % 97 == 0<br>reject candidate % 89 == 0<br>reject candidate % 83 == 0<br>reject candidate % 79 == 0<br>reject candidate % 73 == 0 |

## Additional Modular Gate Families

Raw trace: `build/float/rejected_candidate_learning/modular_gate_trace.jsonl`

- Split key: `modulus` <= `11` for train.
- Train rows: 19; reason labels: {'accepted': 16, 'safe_duplicate': 3}
- Eval rows: 43; reason labels: {'accepted': 17, 'safe_duplicate': 19, 'no_support': 7}
- Trained residual rows: 17; reason labels: {'accepted': 16, 'safe_duplicate': 1}
- Stable trained-residual claim at budget 8: PASS

### Budget Results

| Strategy | Budget | Accepted | Safe duplicate | Unsafe | No support | Validation utility | Validation false rejections |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| model_score | 8 | 4 | 4 | 0 | 0 | 8539 | 0 |
| reason_prior | 8 | 4 | 4 | 0 | 0 | 3828 | 0 |
| reason_aware | 8 | 5 | 3 | 0 | 0 | 3763 | 0 |
| trained_residual | 8 | 4 | 4 | 0 | 0 | 3828 | 0 |
| model_score | 16 | 6 | 10 | 0 | 0 | 14960 | 0 |
| reason_prior | 16 | 7 | 7 | 0 | 2 | 4052 | 0 |
| reason_aware | 16 | 7 | 7 | 0 | 2 | 4052 | 0 |
| trained_residual | 16 | 7 | 7 | 0 | 2 | 4052 | 0 |
| model_score | 32 | 16 | 16 | 0 | 0 | 21558 | 0 |
| reason_prior | 32 | 14 | 11 | 0 | 7 | 13299 | 0 |
| reason_aware | 32 | 14 | 11 | 0 | 7 | 13299 | 0 |
| trained_residual | 32 | 14 | 11 | 0 | 7 | 13299 | 0 |
| model_score | 64 | 17 | 19 | 0 | 7 | 21686 | 0 |
| reason_prior | 64 | 17 | 19 | 0 | 7 | 21686 | 0 |
| reason_aware | 64 | 17 | 19 | 0 | 7 | 21686 | 0 |
| trained_residual | 64 | 17 | 19 | 0 | 7 | 21686 | 0 |

### Split Sweep Guard

- Split grid: 5, 7, 11, 13

| Split | Reason prior accepted / utility | Trained residual accepted / utility | Delta utility | Guard |
| ---: | ---: | ---: | ---: | --- |
| 5 | 5 / 13517 | 8 / 16333 | +2816 | pass |
| 7 | 5 / 13517 | 6 / 14314 | +797 | pass |
| 11 | 4 / 3828 | 4 / 3828 | +0 | pass |
| 13 | 4 / 3828 | 4 / 3828 | +0 | pass |

### Budget 8 Top Rules

| Strategy | Top rules |
| --- | --- |
| model_score | reject twin-prime shift when n+d+0 == 0 mod 13<br>reject twin-prime shift when n+d+2 == 0 mod 13<br>reject lucas index when i == 12 mod period_7_16<br>reject fibonacci index when i == 0 mod period_7_16<br>reject lucas index when i == 4 mod period_7_16 |
| reason_prior | reject fibonacci index when i == 5 mod period_5_20<br>reject fibonacci index when i == 10 mod period_5_20<br>reject fibonacci index when i == 15 mod period_5_20<br>reject value when value == 0 mod 5^2<br>reject value when value == 0 mod 7^2 |
| reason_aware | reject fibonacci index when i == 15 mod period_5_20<br>reject fibonacci index when i == 10 mod period_5_20<br>reject fibonacci index when i == 5 mod period_5_20<br>reject value when value == 0 mod 5^2<br>reject value when value == 0 mod 7^2 |
| trained_residual | reject fibonacci index when i == 15 mod period_5_20<br>reject fibonacci index when i == 5 mod period_5_20<br>reject fibonacci index when i == 10 mod period_5_20<br>reject value when value == 0 mod 5^2<br>reject value when value == 0 mod 7^2 |

## Delayed-Claim Threshold Gates

Raw trace: `build/float/rejected_candidate_learning/delayed_claim_rule_trace.jsonl`

- Split key: `rule_template` <= `66` for train.
- Train rows: 268; reason labels: {'no_support': 28, 'unsafe': 120, 'safe_duplicate': 118, 'accepted': 2}
- Eval rows: 270; reason labels: {'unsafe': 120, 'no_support': 20, 'safe_duplicate': 128, 'accepted': 2}
- Trained residual rows: 268; reason labels: {'no_support': 28, 'unsafe': 120, 'safe_duplicate': 118, 'accepted': 2}
- Stable trained-residual claim at budget 8: FAIL

### Budget Results

| Strategy | Budget | Accepted | Safe duplicate | Unsafe | No support | Validation utility | Validation false rejections |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| model_score | 8 | 0 | 8 | 0 | 0 | 7869 | 0 |
| reason_prior | 8 | 0 | 3 | 5 | 0 | 3738 | 1692 |
| reason_aware | 8 | 0 | 5 | 3 | 0 | 4887 | 1137 |
| trained_residual | 8 | 0 | 5 | 3 | 0 | 4887 | 1137 |
| model_score | 16 | 1 | 15 | 0 | 0 | 14754 | 0 |
| reason_prior | 16 | 0 | 5 | 11 | 0 | 7372 | 2273 |
| reason_aware | 16 | 0 | 11 | 5 | 0 | 11213 | 1692 |
| trained_residual | 16 | 2 | 9 | 5 | 0 | 7148 | 1692 |
| model_score | 32 | 1 | 31 | 0 | 0 | 27549 | 0 |
| reason_prior | 32 | 0 | 5 | 25 | 2 | 13039 | 3714 |
| reason_aware | 32 | 1 | 26 | 5 | 0 | 24350 | 1692 |
| trained_residual | 32 | 2 | 19 | 5 | 6 | 15360 | 1692 |
| model_score | 64 | 1 | 63 | 0 | 0 | 49419 | 0 |
| reason_prior | 64 | 0 | 5 | 55 | 4 | 27315 | 7945 |
| reason_aware | 64 | 1 | 58 | 5 | 0 | 48541 | 1692 |
| trained_residual | 64 | 2 | 48 | 5 | 9 | 34605 | 1692 |

### Split Sweep Guard

- Split grid: 44, 66, 88

| Split | Reason prior accepted / utility | Trained residual accepted / utility | Delta utility | Guard |
| ---: | ---: | ---: | ---: | --- |
| 44 | 0 / 2643 | 2 / 3004 | +361 | pass |
| 66 | 0 / 3738 | 0 / 4887 | +1149 | fail |
| 88 | 0 / 3075 | 1 / 3169 | +94 | fail |

### Budget 8 Top Rules

| Strategy | Top rules |
| --- | --- |
| model_score | reject ignore when feature 5 <= 1<br>reject ignore when feature 4 <= 1<br>reject ignore when feature 3 <= 1<br>reject act_cautiously when feature 5 <= 1<br>reject act_cautiously when feature 4 <= 1 |
| reason_prior | reject trust when feature 3 <= 0.4<br>reject trust when feature 3 <= 0.6<br>reject trust when feature 3 <= 1<br>reject ignore when feature 3 <= 0.4<br>reject ignore when feature 3 <= 1 |
| reason_aware | reject ignore when feature 3 <= 1<br>reject act_cautiously when feature 3 <= 1<br>reject act_cautiously when feature 3 <= 0.6<br>reject ignore when feature 3 <= 0.4<br>reject act_cautiously when feature 3 <= 0.4 |
| trained_residual | reject act_cautiously when feature 3 <= 1<br>reject ignore when feature 3 <= 1<br>reject act_cautiously when feature 3 <= 0.4<br>reject act_cautiously when feature 3 <= 0.6<br>reject ignore when feature 3 <= 0.4 |

## Interpretation

The prime-gap result remains the cleanest positive case. The added modular family checks whether the same reason-code learner survives beyond next-prime gap filters, while delayed-claim checks whether the loop transfers to a non-arithmetic decision task. A neural contribution claim should still be tied to the `trained_residual` row and the split-sweep guard, not to accidental trace-order agreement between `reason_prior` and `reason_aware`.

## Trace Contract

- Candidate rows carry a task, structural descriptors, `model_score`, train/eval support counters, false-rejection counters, and `verifier_result`.
- Rejected rows are labels for the next ranker; held-out verifier results are not ranker inputs.
- Trace artifact: `crlp.rejected_candidate_learning.summary.v1` and `crlp.rejected_candidate_learning.split_sweep.v1` rows.
