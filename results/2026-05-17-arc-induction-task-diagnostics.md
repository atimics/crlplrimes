# ARC Induction Phase 4

Git commit: `a75f1bc7265c`
Git dirty: `false`
Task dir: `/private/tmp/ARC-AGI/data`
Program train dir: `/private/tmp/ARC-AGI/data/training`
Task diagnostics: `results/2026-05-17-arc-induction-task-diagnostics.tasks.jsonl`
Max tasks per split: `400`
Grammar version: `10`

| Split | Loaded | Parse Errors | Baseline Certified | Induction Certified | Delta | Added Tasks | Regressions |
|---|---:|---:|---:|---:|---:|---|---|
| train | 400 | 0 | 24 | 24 | +0 | - | - |
| eval | 400 | 0 | 4 | 5 | +1 | 0b17323b | - |

## Hypothesis Selection

### train

- tasks with induction telemetry: `400`
- proposals: attempted `51143`, retained `4`, train-consistent before cap `4`, train-consistent retained `4`, selected `4`
- heap pressure: dropped `0`, replaced `0`
- selected by coverage `4`, score fallback `0`
- proposal-cap-hit tasks `0`, coverage-fallback tasks `0`

| Template | Attempted | Retained | Train-consistent before cap | Train-consistent retained | Heap dropped | Selected |
|---|---:|---:|---:|---:|---:|---:|
| connect_aligned | 4032 | 1 | 1 | 1 | 0 | 1 |
| extend_aligned_sequence | 2 | 0 | 0 | 0 | 0 | 0 |
| fill_enclosed | 3816 | 2 | 2 | 2 | 0 | 2 |
| move_to_anchor | 10701 | 0 | 0 | 0 | 0 | 0 |
| object_extract | 24350 | 0 | 0 | 0 | 0 | 0 |
| recolor_by_reference | 312 | 0 | 0 | 0 | 0 | 0 |
| select_then_translate | 7930 | 1 | 1 | 1 | 0 | 1 |

### eval

- tasks with induction telemetry: `400`
- proposals: attempted `81748`, retained `2`, train-consistent before cap `2`, train-consistent retained `2`, selected `1`
- heap pressure: dropped `0`, replaced `0`
- selected by coverage `1`, score fallback `0`
- proposal-cap-hit tasks `0`, coverage-fallback tasks `0`

| Template | Attempted | Retained | Train-consistent before cap | Train-consistent retained | Heap dropped | Selected |
|---|---:|---:|---:|---:|---:|---:|
| connect_aligned | 6744 | 0 | 0 | 0 | 0 | 0 |
| extend_aligned_sequence | 2 | 2 | 2 | 2 | 0 | 1 |
| fill_enclosed | 6463 | 0 | 0 | 0 | 0 | 0 |
| move_to_anchor | 20628 | 0 | 0 | 0 | 0 | 0 |
| object_extract | 34568 | 0 | 0 | 0 | 0 | 0 |
| recolor_by_reference | 442 | 0 | 0 | 0 | 0 | 0 |
| select_then_translate | 12901 | 0 | 0 | 0 | 0 | 0 |


## Task-Level Diagnostics

### train

- tasks: `400`
- certified tasks: `24`
- cell delta: `0` over `56672` labeled test cells
- improved / unchanged / worsened tasks: `0` / `400` / `0`

| Bucket | Tasks |
|---|---:|
| `certified` | 24 |
| `no_induction_attempted` | 10 |
| `no_induction_generated` | 366 |

Top cell improvements:
- -

Top cell regressions:
- -

### eval

- tasks: `400`
- certified tasks: `5`
- cell delta: `225` over `98515` labeled test cells
- improved / unchanged / worsened tasks: `1` / `399` / `0`

| Bucket | Tasks |
|---|---:|
| `certified` | 5 |
| `no_induction_attempted` | 11 |
| `no_induction_generated` | 384 |

Top cell improvements:
- `0b17323b` (+225 cells, 0 -> 225, certified)

Top cell regressions:
- -

Parity holds: `true`
Eval gain holds: `true`
Load/parse clean: `true`
Phase 4 holds: `true`
