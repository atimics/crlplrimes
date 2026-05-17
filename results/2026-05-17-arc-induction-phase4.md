# ARC Induction Phase 4

Git commit: `d9432421c288`
Git dirty: `false`
Task dir: `/private/tmp/ARC-AGI/data`
Program train dir: `/private/tmp/ARC-AGI/data/training`
Max tasks per split: `400`
Grammar version: `10`

| Split | Loaded | Parse Errors | Baseline Certified | Induction Certified | Delta | Added Tasks | Regressions |
|---|---:|---:|---:|---:|---:|---|---|
| train | 400 | 0 | 24 | 24 | +0 | - | - |
| eval | 400 | 0 | 4 | 4 | +0 | - | - |

## Hypothesis Selection

### train

- tasks with induction telemetry: `400`
- proposals: attempted `18124`, retained `14727`, train-consistent before cap `1`, train-consistent retained `1`, selected `1`
- heap pressure: dropped `2447`, replaced `950`
- selected by coverage `1`, score fallback `0`
- proposal-cap-hit tasks `14`, coverage-fallback tasks `0`

| Template | Attempted | Retained | Train-consistent before cap | Train-consistent retained | Heap dropped | Selected |
|---|---:|---:|---:|---:|---:|---:|
| move_to_anchor | 10403 | 7218 | 0 | 0 | 2447 | 0 |
| select_then_translate | 7721 | 7509 | 1 | 1 | 0 | 1 |

### eval

- tasks with induction telemetry: `400`
- proposals: attempted `32938`, retained `20779`, train-consistent before cap `0`, train-consistent retained `0`, selected `0`
- heap pressure: dropped `9177`, replaced `2982`
- selected by coverage `0`, score fallback `0`
- proposal-cap-hit tasks `25`, coverage-fallback tasks `0`

| Template | Attempted | Retained | Train-consistent before cap | Train-consistent retained | Heap dropped | Selected |
|---|---:|---:|---:|---:|---:|---:|
| move_to_anchor | 20298 | 10147 | 0 | 0 | 8550 | 0 |
| select_then_translate | 12640 | 10632 | 0 | 0 | 627 | 0 |

Parity holds: `true`
Eval gain holds: `false`
Load/parse clean: `true`
Phase 4 holds: `false`
