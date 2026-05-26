# Verifier Trace Replay Results - 2026-05-14

Snapshot from branch `codex/verifier-trace-replay-improvements`.

## Verification

Commands run:

- `make test`
- `make delayed-claim-eval`
- `make prime-gap-eval`
- `make check`
- `make source-policy-check`
- `git diff --check`

Result: all passed.

## Delayed Claim

`make delayed-claim-eval`

| model | accuracy | nll | ece | decision utility | proposal steps | baseline steps | saved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| trained parent | 93.50% | 0.1580 | 0.0363 | 1.0432 | 1.065 | 2.406 | 55.7% |
| trace replay successor | 94.80% | 0.1188 | 0.0166 | 1.0704 | 1.052 | 2.406 | 56.3% |

Replay improves the parent on accuracy, NLL, ECE, decision utility, and proposal steps.

## Prime Gap Quick Eval

`make prime-gap-eval`

| policy | top-1 | top-10 | nll | ece | proposal steps | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| empirical-gap | 19.00% | 85.60% | 2.6585 | 0.0133 | 5.70 | frequency-only baseline |
| hand-wheel empirical | 44.60% | 100.00% | 1.3775 | 0.0538 | 1.96 | deterministic wheel filters first |
| residue-conditioned | 44.60% | 96.10% | 2.6558 | 0.3026 | 2.54 | residue-count baseline |
| hand-wheel residue-conditioned | 44.60% | 100.00% | 1.4079 | 0.0397 | 2.00 | residue counts after wheel filtering |
| supervised | 53.90% | 99.90% | 1.3291 | 0.0963 | 1.86 | full feature softmax scorer |
| pairwise-ranking | 52.00% | 100.00% | 2.9745 | 0.4642 | 1.91 | hinge ranking ablation |
| learned-wheel supervised | 89.00% | 100.00% | 0.5343 | 0.0950 | 1.13 | symbolic wheel survivor scorer |

The hand-wheel baselines show how much deterministic residue work explains before the neural scorer adds value. The pairwise-ranking ablation gets useful ranking but poor calibration.

## Frozen Prime Comparison

`prime_compare` on the default frozen benchmark, `sample_count=30000`.

| model / mode | top-1 | top-10 | nll | ece | mean rank | proposal steps | model ms/sample | verifier ms/sample |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| parent raw | 45.12% | 99.37% | 1.6691 | 0.1180 | 2.2448 | 2.3775 | 0.0415 | 0.0009 |
| parent + learned-wheel rules | 68.87% | 100.00% | 1.0302 | 0.1332 | 1.4544 | 1.4544 | 0.0052 | 0.0008 |
| replay successor raw | 0.06% | 80.47% | 6.0313 | 0.6912 | 7.9213 | 8.7432 | 0.0422 | 0.0010 |
| replay successor + learned-wheel rules | 68.87% | 100.00% | 1.0196 | 0.1323 | 1.4544 | 1.4544 | 0.0052 | 0.0008 |

Wall-clock values are representative local measurements, not stable benchmark constants.

## Trace Contract Check

Trace rows now include:

- `schema: "crlp.trace.v1"`
- `trace_version: 1`
- `action_id_contract: "stable_unique_action_ids"`

Replay resolves `target_index` to the trace row's `action_id`, then maps that stable action ID back into the regenerated candidate row. This is covered by the prime replay regression test with deliberately reordered action IDs.
