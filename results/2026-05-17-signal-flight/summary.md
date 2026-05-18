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

## Closed-Loop Rollout Evidence

| controller | episodes | reach | death | damage eps | stuck | progress | utility | selected certified |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `goal_heading_no_safety` | 24 | 8.33% | 8.33% | 79.17% | 0.00% | 1865.130 | -1.464 | 0.00% |
| `flight_steer_to` | 24 | 0.00% | 0.00% | 20.83% | 0.00% | 699.843 | 0.160 | 0.00% |
| `neural_raw` | 24 | 12.50% | 0.00% | 62.50% | 0.00% | 694.468 | 0.092 | 40.45% |
| `neural_guarded` | 24 | 12.50% | 0.00% | 25.00% | 0.00% | 696.216 | 0.356 | 40.88% |

## Live Shadow Evidence

| metric | value |
| --- | ---: |
| shadow frames | 74 |
| disagreement rate | 75.68% |
| inference p95 ms | 0.089 |
| protocol-compatible rows | 74 |
| counterfactual replay | `False` |

## Live Neural-Active Evidence

Log: `build/float/signal_live_brain/live-protocol-20260518.jsonl`.

| metric | value |
| --- | ---: |
| active rows | 74 |
| window active rows | 74 |
| brain mode | `neural_active` |
| active source | `neural` |
| checkpoint | `f12b644ed9fa` |
| inference p95 ms | 0.089 |
| replay-ready rows | 74 |
| protocol-compatible rows | 74 |
| damage events | 2 |
| death events | 0 |

## Claim Boundary

Signal now has offline replay evidence, live shadow evidence, and a guarded `neural_active` live pilot artifact when the live-active row is present. The rollout artifact is the deterministic bridge from one-step ranking into closed-loop episodes. These artifacts still do not support a comparative active-control performance claim versus the heuristic baseline.
