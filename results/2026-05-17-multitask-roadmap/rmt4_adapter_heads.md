# Multitask Adapter-Headed Sharing

Seeds: `20240516, 20240517, 20240518`
Budget: `16` epochs x `12000` steps = `192000` total shared updates.
Evaluation samples: `1000` per domain row.

This run compares the flat shared scalar scorer, a shared trunk with per-domain scalar heads, and matched single-task baselines under the same 96-feature envelope.

## Fixed-Suite Top-1

| Domain | Shared | Task heads | Single | Heads - shared | Heads - single |
|---|---:|---:|---:|---:|---:|
| `prime_gap` | 14.07% | 12.20% | 13.63% | -1.87pp | -1.43pp |
| `grid_path` | 90.90% | 90.97% | 91.13% | +0.07pp | -0.17pp |
| `delayed_claim` | 98.00% | 97.03% | 97.90% | -0.97pp | -0.87pp |
| `arc_agi` | 26.17% | 28.57% | 42.80% | +2.40pp | -14.23pp |

## ARC OOD Top-1

| Suite | Shared | Task heads | Single | Heads - shared | Heads - single |
|---|---:|---:|---:|---:|---:|
| `arc_ood` | 48.40% | 47.50% | 42.83% | -0.90pp | +4.67pp |

## ARC And Transfer Guardrail

| Suite | Domain | Model | Top-1 | NLL | ECE | Updates | Params | FLOPs |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `fixed` | `arc_agi` | `shared` | 26.17% | 4.0827 | 0.6995 | 47869 | 8321 | 16545 |
| `fixed` | `arc_agi` | `shared_task_heads` | 28.57% | 4.8506 | 0.6880 | 47954 | 8420 | 16545 |
| `fixed` | `arc_agi` | `single_task` | 42.80% | 8.2754 | 0.5589 | 48000 | 8321 | 16545 |
| `ood` | `arc_agi` | `shared` | 48.40% | 2.6278 | 0.4481 | 47869 | 8321 | 16545 |
| `ood` | `arc_agi` | `shared_task_heads` | 47.50% | 3.1888 | 0.4652 | 47954 | 8420 | 16545 |
| `ood` | `arc_agi` | `single_task` | 42.83% | 4.3723 | 0.4820 | 48000 | 8321 | 16545 |
| `fixed` | `delayed_claim` | `shared` | 98.00% | 0.0472 | 0.0102 | 48101 | 8321 | 16545 |
| `fixed` | `delayed_claim` | `shared_task_heads` | 97.03% | 0.0653 | 0.0157 | 47821 | 8420 | 16545 |
| `fixed` | `delayed_claim` | `single_task` | 97.90% | 0.0412 | 0.0134 | 48000 | 8321 | 16545 |
| `fixed` | `grid_path` | `shared` | 90.90% | 0.3216 | 0.0357 | 47908 | 8321 | 16545 |
| `fixed` | `grid_path` | `shared_task_heads` | 90.97% | 0.3040 | 0.0235 | 48152 | 8420 | 16545 |
| `fixed` | `grid_path` | `single_task` | 91.13% | 0.2999 | 0.0268 | 48000 | 8321 | 16545 |

## Decision Rule

- Decision: `diagnostic_only`.
- Promote this architecture only if task heads improve fixed ARC versus the flat shared scorer, preserve delayed-claim transfer, avoid a material grid regression, and preserve ARC OOD versus the single-task baseline.
- If task heads help fixed ARC but hurt OOD or transfer, keep them as diagnostic evidence rather than the default architecture.
