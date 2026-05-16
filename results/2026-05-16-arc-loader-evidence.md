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
| training | 400 | 400 | 0 | 1302 | 416 | 529 | 18 | 17 | 17 | 65 | 493 | 56672 | 0.8699% |
| evaluation | 400 | 400 | 0 | 1363 | 419 | 528 | 4 | 4 | 4 | 57 | 187 | 98515 | 0.1898% |

## Certified Tasks

| Split | Task | Rules | Train-consistent rules | Test pairs | Correct cells | Accepted rules |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| training | `0520fde7` | 528 | 2 | 1 | 9/9 | `mask_intersection_color_2`, `object_select_overlap_color_2` |
| training | `0d3d703e` | 529 | 1 | 1 | 9/9 | `grid_color_map_same_shape` |
| training | `1cf80156` | 528 | 1 | 1 | 24/24 | `crop_non_background_0` |
| training | `25ff71a9` | 528 | 1 | 2 | 18/18 | `translate_nonzero_+0_+1` |
| training | `3c9b0459` | 528 | 1 | 1 | 9/9 | `rotate180` |
| training | `6150a2bd` | 528 | 1 | 1 | 9/9 | `rotate180` |
| training | `67a3c6ac` | 528 | 1 | 1 | 9/9 | `flip_horizontal` |
| training | `68b16354` | 528 | 1 | 1 | 49/49 | `flip_vertical` |
| training | `6f8cd79b` | 528 | 9 | 1 | 42/42 | `bbox_outline_*_to_8` |
| training | `74dd1130` | 528 | 1 | 1 | 9/9 | `transpose` |
| training | `9172f3a0` | 528 | 1 | 1 | 81/81 | `scale_3x` |
| training | `9dfd6313` | 528 | 1 | 1 | 36/36 | `transpose` |
| training | `b1948b0a` | 529 | 1 | 1 | 16/16 | `grid_color_map_same_shape` |
| training | `c59eb873` | 528 | 1 | 1 | 100/100 | `scale_2x` |
| training | `c8f0f002` | 529 | 1 | 1 | 15/15 | `grid_color_map_same_shape` |
| training | `d511f180` | 529 | 1 | 1 | 16/16 | `grid_color_map_same_shape` |
| training | `ed36ccf7` | 528 | 1 | 1 | 9/9 | `rotate270` |
| evaluation | `195ba7dc` | 528 | 1 | 1 | 30/30 | `mask_union_color_1` |
| evaluation | `5d2a5c43` | 528 | 1 | 2 | 48/48 | `mask_union_color_8` |
| evaluation | `60c09cac` | 528 | 1 | 1 | 100/100 | `scale_2x` |
| evaluation | `e133d23d` | 528 | 1 | 1 | 9/9 | `mask_union_color_2` |

Interpretation: the loader and certificate path work across many ARC-format
tasks, but this grammar is intentionally narrow. It only certifies tasks that
fit the current hand-coded split-panel, recolor, crop, geometry, bbox, and
small-translation families.
