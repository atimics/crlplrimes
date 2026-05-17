# ARC Induction Phase 4

Git commit: `c8cb3b4a6e34`
Git dirty: `false`
Task dir: `/private/tmp/ARC-AGI/data`
Program train dir: `/private/tmp/ARC-AGI/data/training`
Max tasks per split: `400`
Grammar version: `10`

| Split | Loaded | Parse Errors | Baseline Certified | Induction Certified | Delta | Added Tasks | Regressions |
|---|---:|---:|---:|---:|---:|---|---|
| train | 400 | 0 | 24 | 24 | +0 | - | - |
| eval | 400 | 0 | 4 | 5 | +1 | 0b17323b | - |

## Hypothesis Selection

### train

- tasks with induction telemetry: `400`
- proposals: attempted `42627`, retained `31967`, train-consistent before cap `4`, train-consistent retained `4`, selected `4`
- heap pressure: dropped `7785`, replaced `2875`
- selected by coverage `4`, score fallback `0`
- proposal-cap-hit tasks `45`, coverage-fallback tasks `0`

| Template | Attempted | Retained | Train-consistent before cap | Train-consistent retained | Heap dropped | Selected |
|---|---:|---:|---:|---:|---:|---:|
| connect_aligned | 2852 | 2852 | 1 | 1 | 0 | 1 |
| extend_aligned_sequence | 2 | 2 | 0 | 0 | 0 | 0 |
| fill_enclosed | 2157 | 2157 | 2 | 2 | 0 | 2 |
| move_to_anchor | 10403 | 6184 | 0 | 0 | 2571 | 0 |
| object_extract | 19262 | 13456 | 0 | 0 | 5178 | 0 |
| recolor_by_reference | 230 | 145 | 0 | 0 | 36 | 0 |
| select_then_translate | 7721 | 7171 | 1 | 1 | 0 | 1 |

### eval

- tasks with induction telemetry: `400`
- proposals: attempted `69152`, retained `41667`, train-consistent before cap `2`, train-consistent retained `2`, selected `1`
- heap pressure: dropped `20590`, replaced `6895`
- selected by coverage `1`, score fallback `0`
- proposal-cap-hit tasks `68`, coverage-fallback tasks `0`

| Template | Attempted | Retained | Train-consistent before cap | Train-consistent retained | Heap dropped | Selected |
|---|---:|---:|---:|---:|---:|---:|
| connect_aligned | 4914 | 4717 | 0 | 0 | 31 | 0 |
| extend_aligned_sequence | 2 | 2 | 2 | 2 | 0 | 1 |
| fill_enclosed | 3649 | 3589 | 0 | 0 | 60 | 0 |
| move_to_anchor | 20306 | 8144 | 0 | 0 | 8716 | 0 |
| object_extract | 27277 | 15074 | 0 | 0 | 11106 | 0 |
| recolor_by_reference | 364 | 196 | 0 | 0 | 72 | 0 |
| select_then_translate | 12640 | 9945 | 0 | 0 | 605 | 0 |

Parity holds: `true`
Eval gain holds: `true`
Load/parse clean: `true`
Phase 4 holds: `true`
