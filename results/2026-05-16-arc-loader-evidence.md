# ARC Loader Evidence Snapshot

Generated from a temporary public ARC-AGI checkout at `/private/tmp/ARC-AGI`.
The corpus is not vendored in this repository.

## Commands

```sh
git clone --depth 1 https://github.com/fchollet/ARC-AGI.git /private/tmp/ARC-AGI
./build/float/arc_agi --task-dir /private/tmp/ARC-AGI/data/training --max-tasks 400 --program-budget-grid
./build/float/arc_agi --program-train-dir /private/tmp/ARC-AGI/data/training --task-dir /private/tmp/ARC-AGI/data/evaluation --max-tasks 400 --program-budget-grid
./build/float/arc_agi --program-train-dir /private/tmp/arc-selector-smoke --task-dir /private/tmp/arc-selector-smoke --max-tasks 1 --program-budget-grid --loader-summary-only
```

## Loader Summary

| Split | Files | Loaded | Parse errors | Train pairs | Test pairs | Max rules | Unique predictions | Test exact | Certified | Unsupported | Correct cells | Test cells | Cell accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| training | 400 | 400 | 0 | 1302 | 416 | 790 | 25 | 24 | 24 | 48 | 1187 | 56672 | 2.0945% |
| evaluation | 400 | 400 | 0 | 1363 | 419 | 790 | 4 | 4 | 4 | 44 | 187 | 98515 | 0.1898% |

## Program-Proposer Summary

Grammar v10 maps ARC transforms into `NNProgram` rows, adds deterministic
object-scene evidence, folds object translation, aligned-object connection, and
enclosed-region fill under a generic `object_relation_induced` template, induces
relation background plus input/output color parameters from train pairs, adds
selected-object selector/render parameters, and lets a trained proposer append
verifier-checked two-step scale compositions from its top primitive candidates.
The learned proposer uses the shared `nn_program_encode_features()` program
encoder plus ARC shape/color/object-relation/object-selector parameter
descriptors. The evaluation rows below pretrain that proposer on the public
training split before scoring the public evaluation split.

| Split | Strategy | Budget | Proposed | Certified programs | Accepted programs | Covered cells | Event cells | Utility |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| training | grammar-order | 8 | 3200 | 0 | 0 | 0 | 56672 | 0.0000% |
| training | structural heuristic | 8 | 3200 | 13 | 13 | 228 | 56672 | 0.4023% |
| training | random | 8 | 3200 | 0 | 0 | 0 | 56672 | 0.0000% |
| training | learned program scorer | 8 | 3200 | 13 | 13 | 319 | 56672 | 0.5629% |
| training | oracle | 8 | 3200 | 37 | 37 | 1187 | 56672 | 2.0945% |
| training | grammar-order | 16 | 6400 | 0 | 0 | 0 | 56672 | 0.0000% |
| training | structural heuristic | 16 | 6400 | 13 | 13 | 228 | 56672 | 0.4023% |
| training | random | 16 | 6400 | 0 | 0 | 0 | 56672 | 0.0000% |
| training | learned program scorer | 16 | 6400 | 16 | 16 | 409 | 56672 | 0.7217% |
| training | oracle | 16 | 6400 | 38 | 38 | 1187 | 56672 | 2.0945% |
| training | grammar-order | 32 | 12800 | 1 | 1 | 9 | 56672 | 0.0159% |
| training | structural heuristic | 32 | 12800 | 14 | 14 | 228 | 56672 | 0.4023% |
| training | random | 32 | 12800 | 0 | 0 | 0 | 56672 | 0.0000% |
| training | learned program scorer | 32 | 12800 | 16 | 16 | 409 | 56672 | 0.7217% |
| training | oracle | 32 | 12800 | 38 | 38 | 1187 | 56672 | 2.0945% |
| training | grammar-order | 64 | 25600 | 17 | 17 | 433 | 56672 | 0.7640% |
| training | structural heuristic | 64 | 25600 | 17 | 17 | 433 | 56672 | 0.7640% |
| training | random | 64 | 25600 | 2 | 2 | 58 | 56672 | 0.1023% |
| training | learned program scorer | 64 | 25600 | 17 | 17 | 433 | 56672 | 0.7640% |
| training | oracle | 64 | 25600 | 38 | 38 | 1187 | 56672 | 2.0945% |
| evaluation | grammar-order | 8 | 3200 | 0 | 0 | 0 | 98515 | 0.0000% |
| evaluation | structural heuristic | 8 | 3200 | 0 | 0 | 0 | 98515 | 0.0000% |
| evaluation | random | 8 | 3200 | 0 | 0 | 0 | 98515 | 0.0000% |
| evaluation | learned program scorer | 8 | 3200 | 1 | 1 | 100 | 98515 | 0.1015% |
| evaluation | oracle | 8 | 3200 | 8 | 8 | 187 | 98515 | 0.1898% |
| evaluation | grammar-order | 16 | 6400 | 0 | 0 | 0 | 98515 | 0.0000% |
| evaluation | structural heuristic | 16 | 6400 | 2 | 2 | 57 | 98515 | 0.0579% |
| evaluation | random | 16 | 6400 | 0 | 0 | 0 | 98515 | 0.0000% |
| evaluation | learned program scorer | 16 | 6400 | 2 | 2 | 130 | 98515 | 0.1320% |
| evaluation | oracle | 16 | 6400 | 8 | 8 | 187 | 98515 | 0.1898% |
| evaluation | grammar-order | 32 | 12800 | 2 | 2 | 39 | 98515 | 0.0396% |
| evaluation | structural heuristic | 32 | 12800 | 3 | 3 | 87 | 98515 | 0.0883% |
| evaluation | random | 32 | 12800 | 0 | 0 | 0 | 98515 | 0.0000% |
| evaluation | learned program scorer | 32 | 12800 | 3 | 3 | 139 | 98515 | 0.1411% |
| evaluation | oracle | 32 | 12800 | 8 | 8 | 187 | 98515 | 0.1898% |
| evaluation | grammar-order | 64 | 25600 | 4 | 4 | 187 | 98515 | 0.1898% |
| evaluation | structural heuristic | 64 | 25600 | 4 | 4 | 187 | 98515 | 0.1898% |
| evaluation | random | 64 | 25600 | 1 | 1 | 9 | 98515 | 0.0091% |
| evaluation | learned program scorer | 64 | 25600 | 4 | 4 | 187 | 98515 | 0.1898% |
| evaluation | oracle | 64 | 25600 | 8 | 8 | 187 | 98515 | 0.1898% |

## Certified Tasks

| Split | Task | Rules | Train-consistent rules | Test pairs | Correct cells | Accepted rules |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| training | `00d62c1b` | 719 | 1 | 1 | 400/400 | `object_relation_fill_enclosed_3_to_4` |
| training | `0520fde7` | 718 | 2 | 1 | 9/9 | `mask_intersection_color_2`, `object_select_overlap_color_2` |
| training | `0d3d703e` | 719 | 1 | 1 | 9/9 | `grid_color_map_same_shape` |
| training | `1cf80156` | 718 | 2 | 1 | 24/24 | `crop_non_background_0`, `object_crop_largest_non_background_0` |
| training | `1f85a75f` | 718 | 1 | 1 | 16/16 | `object_crop_largest_non_background_0` |
| training | `25ff71a9` | 718 | 1 | 2 | 18/18 | `translate_nonzero_+0_+1` |
| training | `3c9b0459` | 740 | 1 | 1 | 9/9 | `rotate180` |
| training | `6150a2bd` | 740 | 1 | 1 | 9/9 | `rotate180` |
| training | `67a3c6ac` | 731 | 1 | 1 | 9/9 | `flip_horizontal` |
| training | `68b16354` | 759 | 1 | 1 | 49/49 | `flip_vertical` |
| training | `6f8cd79b` | 731 | 9 | 1 | 42/42 | `bbox_outline_*_to_8` |
| training | `74dd1130` | 740 | 1 | 1 | 9/9 | `transpose` |
| training | `9172f3a0` | 762 | 1 | 1 | 81/81 | `scale_3x` |
| training | `9dfd6313` | 757 | 1 | 1 | 36/36 | `transpose` |
| training | `a5313dff` | 731 | 1 | 1 | 81/81 | `object_relation_fill_enclosed_2_to_1` |
| training | `a79310a0` | 762 | 1 | 1 | 25/25 | `object_relation_translate_bg_0_color_8_to_2_+0_+1` |
| training | `b1948b0a` | 762 | 1 | 1 | 16/16 | `grid_color_map_same_shape` |
| training | `be94b721` | 730 | 1 | 1 | 12/12 | `object_crop_largest_non_background_0` |
| training | `c59eb873` | 763 | 3 | 1 | 100/100 | `scale_2x`, `compose_scale_2x_then_crop_non_background_0`, `compose_crop_non_background_0_then_scale_2x` |
| training | `c8f0f002` | 774 | 1 | 1 | 15/15 | `grid_color_map_same_shape` |
| training | `d511f180` | 788 | 1 | 1 | 16/16 | `grid_color_map_same_shape` |
| training | `ded97339` | 751 | 1 | 1 | 100/100 | `object_relation_connect_aligned_bg_0_color_8_to_8_both` |
| training | `ed36ccf7` | 790 | 1 | 1 | 9/9 | `rotate270` |
| training | `f25fbde4` | 763 | 2 | 1 | 60/60 | `compose_scale_2x_then_crop_non_background_0`, `compose_crop_non_background_0_then_scale_2x` |
| evaluation | `195ba7dc` | 748 | 1 | 1 | 30/30 | `mask_union_color_1` |
| evaluation | `5d2a5c43` | 766 | 1 | 2 | 48/48 | `mask_union_color_8` |
| evaluation | `60c09cac` | 775 | 5 | 1 | 100/100 | `scale_2x`, `compose_scale_2x_then_crop_non_background_{1,2}`, `compose_crop_non_background_{1,2}_then_scale_2x` |
| evaluation | `e133d23d` | 785 | 1 | 1 | 9/9 | `mask_union_color_2` |

## Selector Smoke

The public corpus snapshot did not add exact certifications from the new
selected-object selectors. A one-task ARC-format smoke under
`/private/tmp/arc-selector-smoke` exercises the new path directly: the transform
gate certifies `object_select_crop_min_bg_0_color_1_to_1` and
`object_select_crop_recolor_min_bg_0_color_1_to_1`, and after pretraining on the
same selector task the learned proposer ranks both certificates inside budget
8 while grammar-order and structural heuristic rank no selected-object
certificate at budgets 8, 16, 32, or 64.

Interpretation: the loader and certificate path work across many ARC-format
tasks, but this grammar is intentionally narrow. It only certifies tasks that
fit the current hand-coded split-panel, recolor, crop, geometry, bbox,
small-translation, largest/selected connected-component, generic
object-relation induction, and learned scale-composition families. The
largest-object primitive adds two training certifications (`1f85a75f`,
`be94b721`) and one redundant certificate on `1cf80156`; learned scale
composition adds the training-only `f25fbde4` certificate. The generic
object-relation template induces the training-only `ded97339`, `00d62c1b`,
`a5313dff`, and `a79310a0` certificates by learning selector/parameter scores
over relation kind, background, colors, mode, and offset rather than treating
connection, recolored translation, and enclosed fill as separate families.
Grammar v10 adds selected-object selector parameters (`unique`, `largest`,
`smallest`, `interior`, `border`, `nearest`) plus connector-only rendering, but
the public evaluation split remains at four exact certifications. The learned
proposer is still a real neural scorer over program/task descriptors: it
improves tight-budget ranking on the public evaluation split, and the selector
smoke shows it can rank learned selected-object certificates when selector
evidence exists, but public ARC exact coverage is unchanged.
