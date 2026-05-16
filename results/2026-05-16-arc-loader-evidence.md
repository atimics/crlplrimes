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

| Split | Files | Loaded | Parse errors | Train pairs | Test pairs | Rules | Unique predictions | Test exact | Certified | Unsupported | Correct cells | Test cells | Cell accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| training | 400 | 400 | 0 | 1302 | 416 | 46 | 1 | 1 | 1 | 396 | 9 | 56672 | 0.0159% |
| evaluation | 400 | 400 | 0 | 1363 | 419 | 46 | 3 | 3 | 3 | 393 | 87 | 98515 | 0.0883% |

## Certified Tasks

| Split | Task | Rules | Train-consistent rules | Test pairs | Correct cells | Accepted rules |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| training | `0520fde7` | 46 | 2 | 1 | 9/9 | `mask_intersection_color_2`, `object_select_overlap_color_2` |
| evaluation | `195ba7dc` | 46 | 1 | 1 | 30/30 | `mask_union_color_1` |
| evaluation | `5d2a5c43` | 46 | 1 | 2 | 48/48 | `mask_union_color_8` |
| evaluation | `e133d23d` | 46 | 1 | 1 | 9/9 | `mask_union_color_2` |

Interpretation: the loader and certificate path work across many ARC-format
tasks, but this grammar is intentionally narrow. It only certifies tasks that
fit the current split-panel copy/mask/union family.
