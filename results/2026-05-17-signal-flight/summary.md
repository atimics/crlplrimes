# Signal Flight Evidence Summary

Generated from Signal result JSONL artifacts.

## Offline Replay

Source: `build/float/signal_flight_experiment/summary.jsonl`.

| metric | neural ranker | no-safety baseline |
| --- | ---: | ---: |
| top-1 | 79.76% | 67.86% |
| top-3 | 94.05% | n/a |
| selected utility | 0.14317 | 0.13593 |
| regret | 0.00735 | 0.01459 |
| damage rate | 2.38% | 2.38% |
| death rate | 0.00% | 0.00% |

## Live Shadow Evidence

| metric | value |
| --- | ---: |
| shadow frames | 22191 |
| disagreement rate | 73.51% |
| inference p95 ms | 0.751 |

## Live Neural-Active Evidence

Log: `build/float/signal_live_brain/live-9093-reset-20260518.jsonl`.

| metric | value |
| --- | ---: |
| active rows | 10078 |
| window active rows | 250 |
| brain mode | `neural_active` |
| active source | `neural` |
| checkpoint | `f12b644ed9fa` |
| inference p95 ms | 0.124 |
| replay-ready rows | 249 |
| damage events | 0 |
| death events | 0 |

## Claim Boundary

Signal now has offline replay evidence, live shadow evidence, and a guarded `neural_active` live pilot artifact when the live-active row is present. These artifacts do not yet support a comparative active-control performance claim versus the heuristic baseline.
