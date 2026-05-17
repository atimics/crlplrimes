# Multitask ARC Failure Diagnostics

All rows evaluate fixed `0520fde7` ARC with `--eval-tasks arc --no-task-baselines`.

## Order Sweep

| Order | Top-1 | Fixed cells | NLL | Updates | Pred 0 cells | Pred 2 cells | Dominant color cells |
|---|---:|---:|---:|---:|---:|---:|---:|
| `uniform_random` | 26.17% | 40.74% | 4.0827 | 47869 | 5.67 | 3.33 | 6.33 |
| `round_robin` | 45.23% | 51.85% | 4.1008 | 48000 | 5.33 | 3.67 | 6.00 |
| `arc_heavy` | 42.80% | 66.67% | 6.5816 | 95933 | 8.00 | 1.00 | 8.00 |
| `prime_light` | 42.80% | 66.67% | 6.9045 | 57745 | 8.00 | 1.00 | 8.00 |

## Learning-Rate Sweep

| LR | Top-1 | Fixed cells | NLL | Fixed NLL | Pred 0 cells | Pred 2 cells | Dominant color cells |
|---|---:|---:|---:|---:|---:|---:|---:|
| `0.0015` | 26.17% | 40.74% | 3.3589 | 2.4426 | 5.67 | 3.33 | 6.33 |
| `0.003` | 26.17% | 40.74% | 3.6801 | 2.7699 | 5.67 | 3.33 | 6.33 |
| `0.006` | 26.17% | 40.74% | 4.0827 | 2.8758 | 5.67 | 3.33 | 6.33 |
| `0.012` | 28.53% | 44.44% | 4.1064 | 2.7575 | 6.00 | 3.00 | 6.33 |

## Epoch Prefix Sweep

| Epochs | Top-1 | Fixed cells | NLL | Fixed NLL | Pred 0 cells | Pred 2 cells |
|---|---:|---:|---:|---:|---:|---:|
| `1` | 26.17% | 40.74% | 2.8942 | 1.9886 | 5.67 | 3.33 |
| `2` | 23.77% | 37.04% | 2.9992 | 2.0830 | 5.33 | 3.67 |
| `4` | 26.13% | 40.74% | 3.1627 | 2.1576 | 5.67 | 3.33 |
| `8` | 26.13% | 40.74% | 4.0126 | 2.8284 | 5.67 | 3.33 |
| `16` | 26.17% | 40.74% | 4.0827 | 2.8758 | 5.67 | 3.33 |

## ARC Training Loss

| Scenario | First loss | Final loss | Delta | Final updates |
|---|---:|---:|---:|---:|
| `1` | 0.2875 | 0.2875 | 0.0000 | 2997 |
| `2` | 0.2875 | 0.0363 | -0.2512 | 5937 |
| `4` | 0.2875 | 0.0083 | -0.2792 | 11932 |
| `8` | 0.2875 | 0.0036 | -0.2839 | 23967 |
| `16` | 0.2875 | 0.0020 | -0.2855 | 47869 |

## Diagnosis

- Not simple starvation: uniform random gives ARC about 47.9k updates, round-robin gives exactly 48k, and `arc_heavy` gives about 95.9k. Extra ARC updates improve fixed-cell accuracy only by collapsing to a background-heavy grid, not by learning the target foreground cells.
- Not late forgetting: the epoch-prefix scan is already in the same prediction family after one epoch. ARC train loss falls from `0.2875` to `0.0020`, but fixed-cell accuracy stays near `40.74%` and no run reaches exact grid prediction.
- Not fixed by this LR range: `0.0015`, `0.003`, and `0.006` produce the same top-1 and fixed-cell accuracy; `0.012` moves the metric only slightly and still does not reach exact grid.
- Primary failure mode: fixed-ARC cell/color mode collapse. The scorer learns the relevant color set `{0,2}` and a crude background/foreground prior, but not the geometry. `arc_heavy` and `prime_light` converge to the same prediction hash across all three seeds, predicting eight zero cells and one `2` cell; all six correctly predicted cells are background cells and zero foreground cells are correct.
- Single-task reference shows the same structural problem. In `rmt_arc_fixed_grid_baseline.csv`, all three single-task ARC runs predict `[0,0,0,0,2,0,0,0,0]`: six background cells correct, zero foreground cells correct, exact grid false.

Conclusion: the blocker is the fixed ARC scorer formulation, not the core NN engine. The current 96-feature cell-color row plus three ARC training pairs is enough to overfit training loss and discover color priors, but not enough to infer the `0520fde7` panel geometry. R-MT.8 should still be useful for sampling dynamics, but it will not by itself solve fixed ARC exactness unless the ARC feature/training formulation changes.
