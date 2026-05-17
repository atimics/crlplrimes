# Signal Flight Replay Result

Source: `build/float/signal_flight_experiment/summary.jsonl`.

This is an offline replay-controlled application vignette. It is not evidence that the live websocket pilot is neural-controlled.

| metric | neural ranker | no-safety baseline |
| --- | ---: | ---: |
| top-1 | 79.76% | 67.86% |
| top-3 | 94.05% | n/a |
| selected utility | 0.14317 | 0.13593 |
| regret | 0.00735 | 0.01459 |
| damage rate | 2.38% | 2.38% |
| death rate | 0.00% | 0.00% |

The scoped claim is that a 2113-parameter WASD ranker improves replay-certified held-out action ranking and regret relative to the primitive no-safety goal-heading baseline, while matching the baseline damage/death rates on this trace.
