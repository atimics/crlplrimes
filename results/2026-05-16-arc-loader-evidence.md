# ARC Loader Evidence Snapshot

Generated from a temporary public ARC-AGI checkout at `/private/tmp/ARC-AGI`.
The corpus is not vendored in this repository.

## Commands

```sh
git clone --depth 1 https://github.com/fchollet/ARC-AGI.git /private/tmp/ARC-AGI
./build/float/arc_agi --task-dir /private/tmp/ARC-AGI/data/training --max-tasks 400 --loader-summary-only
./build/float/arc_agi --task-dir /private/tmp/ARC-AGI/data/evaluation --max-tasks 400 --loader-summary-only
```

## Loader Summary

| Split | Files | Loaded | Parse errors | Train pairs | Test pairs | Max rules | Unique predictions | Test exact | Certified | Unsupported | Correct cells | Test cells | Cell accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| training | 400 | 400 | 0 | 1302 | 416 | 719 | 20 | 19 | 19 | 50 | 521 | 56672 | 0.9193% |
| evaluation | 400 | 400 | 0 | 1363 | 419 | 718 | 4 | 4 | 4 | 45 | 187 | 98515 | 0.1898% |

## Program-Proposer Summary

Grammar v4 maps ARC transforms into `NNProgram` rows, adds a connected-component
largest-object primitive, and reports matched 64-program proposer budgets through
`crlp.program_proposer.summary.v1`.

| Split | Strategy | Budget | Proposed | Certified programs | Accepted programs | Covered cells | Event cells | Utility |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| training | grammar-order | 64 | 25600 | 17 | 17 | 433 | 56672 | 0.7640% |
| training | structural heuristic | 64 | 25600 | 16 | 16 | 409 | 56672 | 0.7217% |
| training | random | 64 | 25600 | 0 | 0 | 0 | 56672 | 0.0000% |
| training | learned family prior | 64 | 25600 | 10 | 10 | 168 | 56672 | 0.2964% |
| training | oracle | 64 | 25600 | 30 | 30 | 521 | 56672 | 0.9193% |
| evaluation | grammar-order | 64 | 25600 | 4 | 4 | 187 | 98515 | 0.1898% |
| evaluation | structural heuristic | 64 | 25600 | 4 | 4 | 187 | 98515 | 0.1898% |
| evaluation | random | 64 | 25600 | 0 | 0 | 0 | 98515 | 0.0000% |
| evaluation | learned family prior | 64 | 25600 | 4 | 4 | 187 | 98515 | 0.1898% |
| evaluation | oracle | 64 | 25600 | 4 | 4 | 187 | 98515 | 0.1898% |

## Certified Tasks

| Split | Task | Rules | Train-consistent rules | Test pairs | Correct cells | Accepted rules |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| training | `0520fde7` | 718 | 2 | 1 | 9/9 | `mask_intersection_color_2`, `object_select_overlap_color_2` |
| training | `0d3d703e` | 719 | 1 | 1 | 9/9 | `grid_color_map_same_shape` |
| training | `1cf80156` | 718 | 2 | 1 | 24/24 | `crop_non_background_0`, `object_crop_largest_non_background_0` |
| training | `1f85a75f` | 718 | 1 | 1 | 16/16 | `object_crop_largest_non_background_0` |
| training | `25ff71a9` | 718 | 1 | 2 | 18/18 | `translate_nonzero_+0_+1` |
| training | `3c9b0459` | 718 | 1 | 1 | 9/9 | `rotate180` |
| training | `6150a2bd` | 718 | 1 | 1 | 9/9 | `rotate180` |
| training | `67a3c6ac` | 718 | 1 | 1 | 9/9 | `flip_horizontal` |
| training | `68b16354` | 718 | 1 | 1 | 49/49 | `flip_vertical` |
| training | `6f8cd79b` | 718 | 9 | 1 | 42/42 | `bbox_outline_*_to_8` |
| training | `74dd1130` | 718 | 1 | 1 | 9/9 | `transpose` |
| training | `9172f3a0` | 718 | 1 | 1 | 81/81 | `scale_3x` |
| training | `9dfd6313` | 718 | 1 | 1 | 36/36 | `transpose` |
| training | `b1948b0a` | 719 | 1 | 1 | 16/16 | `grid_color_map_same_shape` |
| training | `be94b721` | 718 | 1 | 1 | 12/12 | `object_crop_largest_non_background_0` |
| training | `c59eb873` | 718 | 1 | 1 | 100/100 | `scale_2x` |
| training | `c8f0f002` | 719 | 1 | 1 | 15/15 | `grid_color_map_same_shape` |
| training | `d511f180` | 719 | 1 | 1 | 16/16 | `grid_color_map_same_shape` |
| training | `ed36ccf7` | 718 | 1 | 1 | 9/9 | `rotate270` |
| evaluation | `195ba7dc` | 718 | 1 | 1 | 30/30 | `mask_union_color_1` |
| evaluation | `5d2a5c43` | 718 | 1 | 2 | 48/48 | `mask_union_color_8` |
| evaluation | `60c09cac` | 718 | 1 | 1 | 100/100 | `scale_2x` |
| evaluation | `e133d23d` | 718 | 1 | 1 | 9/9 | `mask_union_color_2` |

Interpretation: the loader and certificate path work across many ARC-format
tasks, but this grammar is intentionally narrow. It only certifies tasks that
fit the current hand-coded split-panel, recolor, crop, geometry, bbox,
small-translation, and largest connected-component families. The object primitive
adds two training certifications (`1f85a75f`, `be94b721`) and one redundant
certificate on `1cf80156`; it does not improve the public evaluation split yet.
