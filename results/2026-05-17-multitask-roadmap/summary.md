# Multitask Roadmap Experiments

Seeds: `20240516, 20240517, 20240518`
Budget: `16` epochs x `12000` steps = `192000` total shared updates.
Evaluation samples: `1000` per aggregate row.

## R-MT.1 Longer Shared And Single Runs

| Domain | Shared top-1 | Single top-1 | Delta | Shared NLL | Single NLL | Shared ECE | Single ECE | Shared updates | Single updates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `prime_gap` | 14.07% | 13.63% | +0.43pp | 2.6648 | 2.6733 | 0.0279 | 0.0410 | 48121 | 48000 |
| `grid_path` | 90.90% | 91.13% | -0.23pp | 0.3216 | 0.2999 | 0.0357 | 0.0268 | 47908 | 48000 |
| `delayed_claim` | 98.00% | 97.90% | +0.10pp | 0.0472 | 0.0412 | 0.0102 | 0.0134 | 48101 | 48000 |
| `arc_agi` | 26.17% | 42.80% | -16.63pp | 4.0827 | 8.2754 | 0.6995 | 0.5589 | 47869 | 48000 |

## R-MT.2 Same-Envelope ARC OOD

The ARC scorer is trained on `0520fde7`; evaluation uses 12 held-out 7x3 -> 3x3 panel-composition tasks inside the existing 96-feature action-row envelope.

| Model | Top-1 | NLL | ECE | Updates |
|---|---:|---:|---:|---:|
| `shared` | 48.40% | 2.6278 | 0.4481 | 47869 |
| `single_task` | 42.83% | 4.3723 | 0.4820 | 48000 |

Aggregate shared-minus-single OOD top-1 delta: `+5.57pp`.

### ARC OOD Task Breakdown

| Task | Family | Shared top-1 | Single top-1 | Delta |
|---|---|---:|---:|---:|
| `copy_left_sparse` | `copy_left` | 39.17% | 40.83% | -1.67pp |
| `copy_right_sparse` | `copy_right` | 49.17% | 37.50% | +11.67pp |
| `left_implies_right` | `left_implies_right` | 38.90% | 33.40% | +5.50pp |
| `left_only_bands` | `left_only` | 40.00% | 20.00% | +20.00pp |
| `neither_diag` | `neither` | 58.33% | 52.50% | +5.83pp |
| `not_left_motif` | `not_left` | 50.83% | 51.67% | -0.83pp |
| `not_right_motif` | `not_right` | 55.83% | 54.17% | +1.67pp |
| `right_implies_left` | `right_implies_left` | 57.13% | 71.40% | -14.27pp |
| `right_only_floor` | `right_only` | 71.67% | 62.50% | +9.17pp |
| `row_any_intersection` | `row_any_intersection` | 38.87% | 44.40% | -5.53pp |
| `union_checker` | `union` | 33.33% | 11.10% | +22.23pp |
| `xor_offset` | `xor` | 49.97% | 36.07% | +13.90pp |

## R-MT.3 Delayed-Claim Leave-One-Out

Full artifact: `rmt3_leave_one_out.md`.

| Training set | Omitted sibling | Claim top-1 | Delta vs claim-only | Delta vs all |
|---|---|---:|---:|---:|
| `all` | `none` | 98.00% | +1.40pp | +0.00pp |
| `no_prime_gap` | `prime_gap` | 97.83% | +1.23pp | -0.17pp |
| `no_grid_path` | `grid_path` | 97.20% | +0.60pp | -0.80pp |
| `no_arc_agi` | `arc_agi` | 98.50% | +1.90pp | +0.50pp |
| `claim_only` | `all_siblings` | 96.60% | +0.00pp | -1.40pp |

Interpretation: delayed-claim positive transfer survives at the long-run budget, but the sibling pattern is not uniformly beneficial. Removing `arc_agi` slightly improves delayed-claim, while removing `grid_path` is the largest loss.

## R-MT.5 Calibration Audit

Full artifact: `rmt5_calibration.md`.

| Evaluation | Shared ECE | Single ECE | Delta | Shared NLL | Single NLL | Delta |
|---|---:|---:|---:|---:|---:|---:|
| `arc_agi` | 0.6995 | 0.5589 | +0.1406 | 4.0827 | 8.2754 | -4.1927 |
| `arc_agi_ood_same_envelope` | 0.4481 | 0.4820 | -0.0339 | 2.6278 | 4.3723 | -1.7445 |

Interpretation: fixed `0520fde7` is a ranking/log-loss tradeoff, not a calibration win: shared training lowers NLL but worsens ECE. On same-envelope ARC OOD, shared improves top-1, NLL, and ECE.

## R-MT.6 Multitask Capacity Sweep

Full artifact: `rmt6_capacity_sweep.md`.

| Hidden | Params | Prime gap delta | Grid path delta | Delayed-claim delta | Fixed ARC delta |
|---|---:|---:|---:|---:|---:|
| `32x16` | 3649 | +1.63pp | -0.20pp | +0.43pp | -1.23pp |
| `64x32` | 8321 | +0.43pp | -0.23pp | +0.10pp | -16.63pp |
| `128x64` | 20737 | +0.43pp | -0.40pp | +0.13pp | -14.27pp |
| `256x128` | 57857 | +1.87pp | -0.30pp | +0.43pp | -8.30pp |

Interpretation: capacity alone does not explain fixed ARC interference. The smallest `32x16` trunk is the best fixed-ARC shared row (39.20%) and has the least negative shared-minus-single gap (-1.23pp); the largest trunk partially recovers the `64x32` gap but does not close it. Delayed-claim stays strong across capacities.

## ARC Fixed-Scorer Failure Diagnostics

Full artifact: `rmt_arc_failure_diagnostics.md`.

| Diagnostic | Best/top row | What happened |
|---|---|---|
| order sweep | `round_robin` top-1 45.23%; `arc_heavy` fixed cells 66.67% | More ARC updates help only by pushing a background-heavy mode, not exact geometry. |
| LR sweep | `0.012` top-1 28.53%; fixed cells 44.44% | The tested LR range does not solve fixed ARC. |
| epoch prefixes | 1 epoch and 16 epochs both 40.74% fixed cells | Not late forgetting; wrong prediction modes appear early while train loss keeps falling. |
| single-task reference | `[0,0,0,0,2,0,0,0,0]` for all three seeds | Single-task also learns color/background priors rather than target foreground geometry. |

Interpretation: the fixed ARC scorer failure is cell/color mode collapse under the current 96-feature row and three-pair training setup, not a core NN engine bug. The symbolic gate remains the correct path for exact `0520fde7`; the learned fixed scorer needs a better ARC training/formulation surface before it can be expected to infer the panel geometry.

## R-MT.7 ARC Gate and Proposer Invariance

Full artifact: `rmt7_arc_gate_invariance.md`.

| Split | Tasks | Certified tasks | Test-exact tasks | Correct cells | Exact gate/loader mismatches | Proposer mismatches |
|---|---:|---:|---:|---:|---:|---:|
| `train` | 400 | 24 | 24 | 1187 | 0 | 0 |
| `eval` | 400 | 4 | 4 | 187 | 0 | 0 |

Interpretation: the public ARC transform gate and budgeted proposer path do not currently consume a multitask scorer checkpoint. With that current scorer-independent path, accepted rule sets, prediction hashes, loader summaries, and budgeted proposer aggregates are invariant between the `shared_scorer_context` and `single_task_scorer_context` labels.
