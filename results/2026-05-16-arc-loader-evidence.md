# ARC Loader Evidence Snapshot

Generated from a temporary public ARC-AGI checkout at `/private/tmp/ARC-AGI`.
The corpus is not vendored in this repository.

## Commands

```sh
git clone --depth 1 https://github.com/fchollet/ARC-AGI.git /private/tmp/ARC-AGI
./build/float/arc_agi --task-dir /private/tmp/ARC-AGI/data/training --max-tasks 400 --loader-summary-only --program-budget-grid
./build/float/arc_agi --program-train-dir /private/tmp/ARC-AGI/data/training --task-dir /private/tmp/ARC-AGI/data/evaluation --max-tasks 400 --loader-summary-only --program-budget-grid
```

## Loader Summary

| Split | Files | Loaded | Parse errors | Train pairs | Test pairs | Max rules | Unique predictions | Test exact | Certified | Unsupported | Correct cells | Test cells | Cell accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| training | 400 | 400 | 0 | 1302 | 416 | 719 | 20 | 19 | 19 | 50 | 521 | 56672 | 0.9193% |
| evaluation | 400 | 400 | 0 | 1363 | 419 | 718 | 4 | 4 | 4 | 45 | 187 | 98515 | 0.1898% |

## Program-Proposer Summary

Grammar v4 maps ARC transforms into `NNProgram` rows, adds a connected-component
largest-object primitive, and reports matched proposer budget curves through
`crlp.program_proposer.summary.v1`. The learned proposer uses the shared
`nn_program_encode_features()` program encoder plus ARC shape/color/object
descriptors. The evaluation rows below pretrain that proposer on the public
training split before scoring the public evaluation split.

| Split | Strategy | Budget | Proposed | Certified programs | Accepted programs | Covered cells | Event cells | Utility |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| training | grammar-order | 8 | 3200 | 0 | 0 | 0 | 56672 | 0.0000% |
| training | structural heuristic | 8 | 3200 | 13 | 13 | 228 | 56672 | 0.4023% |
| training | random | 8 | 3200 | 0 | 0 | 0 | 56672 | 0.0000% |
| training | learned program scorer | 8 | 3200 | 14 | 14 | 328 | 56672 | 0.5788% |
| training | oracle | 8 | 3200 | 29 | 29 | 521 | 56672 | 0.9193% |
| training | grammar-order | 16 | 6400 | 0 | 0 | 0 | 56672 | 0.0000% |
| training | structural heuristic | 16 | 6400 | 13 | 13 | 228 | 56672 | 0.4023% |
| training | random | 16 | 6400 | 0 | 0 | 0 | 56672 | 0.0000% |
| training | learned program scorer | 16 | 6400 | 15 | 15 | 409 | 56672 | 0.7217% |
| training | oracle | 16 | 6400 | 30 | 30 | 521 | 56672 | 0.9193% |
| training | grammar-order | 32 | 12800 | 1 | 1 | 9 | 56672 | 0.0159% |
| training | structural heuristic | 32 | 12800 | 14 | 14 | 228 | 56672 | 0.4023% |
| training | random | 32 | 12800 | 0 | 0 | 0 | 56672 | 0.0000% |
| training | learned program scorer | 32 | 12800 | 16 | 16 | 409 | 56672 | 0.7217% |
| training | oracle | 32 | 12800 | 30 | 30 | 521 | 56672 | 0.9193% |
| training | grammar-order | 64 | 25600 | 17 | 17 | 433 | 56672 | 0.7640% |
| training | structural heuristic | 64 | 25600 | 16 | 16 | 409 | 56672 | 0.7217% |
| training | random | 64 | 25600 | 0 | 0 | 0 | 56672 | 0.0000% |
| training | learned program scorer | 64 | 25600 | 17 | 17 | 433 | 56672 | 0.7640% |
| training | oracle | 64 | 25600 | 30 | 30 | 521 | 56672 | 0.9193% |
| evaluation | grammar-order | 8 | 3200 | 0 | 0 | 0 | 98515 | 0.0000% |
| evaluation | structural heuristic | 8 | 3200 | 0 | 0 | 0 | 98515 | 0.0000% |
| evaluation | random | 8 | 3200 | 0 | 0 | 0 | 98515 | 0.0000% |
| evaluation | learned program scorer | 8 | 3200 | 1 | 1 | 100 | 98515 | 0.1015% |
| evaluation | oracle | 8 | 3200 | 4 | 4 | 187 | 98515 | 0.1898% |
| evaluation | grammar-order | 16 | 6400 | 0 | 0 | 0 | 98515 | 0.0000% |
| evaluation | structural heuristic | 16 | 6400 | 1 | 1 | 9 | 98515 | 0.0091% |
| evaluation | random | 16 | 6400 | 0 | 0 | 0 | 98515 | 0.0000% |
| evaluation | learned program scorer | 16 | 6400 | 2 | 2 | 130 | 98515 | 0.1320% |
| evaluation | oracle | 16 | 6400 | 4 | 4 | 187 | 98515 | 0.1898% |
| evaluation | grammar-order | 32 | 12800 | 2 | 2 | 39 | 98515 | 0.0396% |
| evaluation | structural heuristic | 32 | 12800 | 3 | 3 | 87 | 98515 | 0.0883% |
| evaluation | random | 32 | 12800 | 0 | 0 | 0 | 98515 | 0.0000% |
| evaluation | learned program scorer | 32 | 12800 | 3 | 3 | 139 | 98515 | 0.1411% |
| evaluation | oracle | 32 | 12800 | 4 | 4 | 187 | 98515 | 0.1898% |
| evaluation | grammar-order | 64 | 25600 | 4 | 4 | 187 | 98515 | 0.1898% |
| evaluation | structural heuristic | 64 | 25600 | 4 | 4 | 187 | 98515 | 0.1898% |
| evaluation | random | 64 | 25600 | 0 | 0 | 0 | 98515 | 0.0000% |
| evaluation | learned program scorer | 64 | 25600 | 4 | 4 | 187 | 98515 | 0.1898% |
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
The learned proposer is now a real neural scorer over program/task descriptors:
it improves tight-budget ranking on both splits, but it only recovers the same
four evaluation certifications as grammar/structural order by budget 64 because
the grammar ceiling is still four tasks.
