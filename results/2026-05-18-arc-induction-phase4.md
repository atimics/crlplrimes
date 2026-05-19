# ARC Induction Phase 4

Git commit: `b7ebf794ebe5`
Git dirty: `true`
Task dir: `/private/tmp/ARC-AGI/data`
Program train dir: `/private/tmp/ARC-AGI/data/training`
Task diagnostics: `results/2026-05-18-arc-induction-phase4.tasks.jsonl`
Max tasks per split: `400`
Grammar version: `15`

| Split | Loaded | Parse Errors | Baseline Certified | Induction Certified | Delta | Added Tasks | Regressions |
|---|---:|---:|---:|---:|---:|---|---|
| train | 400 | 0 | 24 | 27 | +3 | 2013d3e2, 5bd6f4ac, d10ecb37 | - |
| eval | 400 | 0 | 7 | 9 | +2 | 0b17323b, 73182012 | - |

## Hypothesis Selection

### train

- tasks with induction telemetry: `400`
- proposals: attempted `53321`, retained `53`, train-consistent before cap `53`, train-consistent retained `53`, selected `9`
- heap pressure: dropped `0`, replaced `0`
- selected by coverage `9`, score fallback `0`
- proposal-cap-hit tasks `0`, coverage-fallback tasks `0`

| Template | Attempted | Retained | Train-consistent before cap | Train-consistent retained | Heap dropped | Selected |
|---|---:|---:|---:|---:|---:|---:|
| connect_aligned | 4032 | 1 | 1 | 1 | 0 | 1 |
| extend_aligned_sequence | 2 | 0 | 0 | 0 | 0 | 0 |
| fill_enclosed | 3816 | 2 | 2 | 2 | 0 | 2 |
| move_to_anchor | 10701 | 0 | 0 | 0 | 0 | 0 |
| object_extract | 24350 | 0 | 0 | 0 | 0 | 0 |
| projection_bbox_panel_selection | 1089 | 37 | 37 | 37 | 0 | 5 |
| projection_panel_selection | 1089 | 12 | 12 | 12 | 0 | 0 |
| recolor_by_reference | 312 | 0 | 0 | 0 | 0 | 0 |
| select_then_translate | 7930 | 1 | 1 | 1 | 0 | 1 |
| tile_motif_repeat | 0 | 0 | 0 | 0 | 0 | 0 |

### eval

- tasks with induction telemetry: `400`
- proposals: attempted `82496`, retained `12`, train-consistent before cap `12`, train-consistent retained `12`, selected `3`
- heap pressure: dropped `0`, replaced `0`
- selected by coverage `3`, score fallback `0`
- proposal-cap-hit tasks `0`, coverage-fallback tasks `0`

| Template | Attempted | Retained | Train-consistent before cap | Train-consistent retained | Heap dropped | Selected |
|---|---:|---:|---:|---:|---:|---:|
| connect_aligned | 6744 | 0 | 0 | 0 | 0 | 0 |
| extend_aligned_sequence | 2 | 2 | 2 | 2 | 0 | 1 |
| fill_enclosed | 6463 | 0 | 0 | 0 | 0 | 0 |
| move_to_anchor | 20628 | 0 | 0 | 0 | 0 | 0 |
| object_extract | 34568 | 0 | 0 | 0 | 0 | 0 |
| projection_bbox_panel_selection | 369 | 9 | 9 | 9 | 0 | 1 |
| projection_panel_selection | 378 | 0 | 0 | 0 | 0 | 0 |
| recolor_by_reference | 442 | 0 | 0 | 0 | 0 | 0 |
| select_then_translate | 12901 | 0 | 0 | 0 | 0 | 0 |
| tile_motif_repeat | 1 | 1 | 1 | 1 | 0 | 1 |


## Task-Level Diagnostics

### train

- tasks: `400`
- certified tasks: `27`
- cell delta: `22` over `56672` labeled test cells
- improved / unchanged / worsened tasks: `3` / `397` / `0`

| Bucket | Tasks |
|---|---:|
| `certified` | 27 |
| `no_induction_attempted` | 9 |
| `no_induction_generated` | 362 |
| `selected_test_mismatch` | 2 |

Top cell improvements:
- `5bd6f4ac` (+9 cells, 0 -> 9, certified)
- `2013d3e2` (+9 cells, 0 -> 9, certified)
- `d10ecb37` (+4 cells, 0 -> 4, certified)

Top cell regressions:
- -

### eval

- tasks: `400`
- certified tasks: `9`
- cell delta: `241` over `98515` labeled test cells
- improved / unchanged / worsened tasks: `2` / `398` / `0`

| Bucket | Tasks |
|---|---:|
| `certified` | 9 |
| `no_induction_attempted` | 10 |
| `no_induction_generated` | 381 |

Top cell improvements:
- `0b17323b` (+225 cells, 0 -> 225, certified)
- `73182012` (+16 cells, 0 -> 16, certified)

Top cell regressions:
- -

Parity holds: `true`
Eval gain holds: `true`
Load/parse clean: `true`
Phase 4 holds: `true`
