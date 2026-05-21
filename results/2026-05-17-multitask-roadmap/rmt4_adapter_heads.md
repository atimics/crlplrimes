# Multitask Adapter-Headed Sharing

Seeds: `20240516, 20240517, 20240518`
Budget: `16` epochs x `12000` steps = `192000` total shared updates.
Evaluation samples: `1000` per domain row.

This run compares the flat shared scalar scorer, a shared trunk with per-domain scalar heads, and matched single-task baselines under the same 96-feature envelope.

## Fixed-Suite Top-1

| Domain | Shared | Task heads | Single | Heads - shared | Heads - single |
|---|---:|---:|---:|---:|---:|
| `prime_gap` | 14.07% | 11.50% | 12.20% | -2.57pp | -0.70pp |
| `grid_path` | 90.73% | 90.97% | 91.17% | +0.23pp | -0.20pp |
| `delayed_claim` | 97.67% | 97.67% | 98.00% | +0.00pp | -0.33pp |
| `arc_agi` | 26.20% | 26.20% | 42.80% | +0.00pp | -16.60pp |

## ARC OOD Top-1

| Suite | Shared | Task heads | Single | Heads - shared | Heads - single |
|---|---:|---:|---:|---:|---:|
| `arc_ood` | 50.53% | 48.70% | 42.90% | -1.83pp | +5.80pp |

## ARC And Transfer Guardrail

| Suite | Domain | Model | Top-1 | NLL | ECE | Updates | Params | FLOPs |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `fixed` | `arc_agi` | `shared` | 26.20% | 4.2288 | 0.7036 | 47869 | 8321 | 16545 |
| `fixed` | `arc_agi` | `shared_task_heads` | 26.20% | 4.6031 | 0.7139 | 47954 | 8420 | 16545 |
| `fixed` | `arc_agi` | `single_task` | 42.80% | 8.0476 | 0.5700 | 48000 | 8321 | 16545 |
| `ood` | `arc_agi` | `shared` | 50.53% | 2.6244 | 0.4269 | 47869 | 8321 | 16545 |
| `ood` | `arc_agi` | `shared_task_heads` | 48.70% | 3.0555 | 0.4422 | 47954 | 8420 | 16545 |
| `ood` | `arc_agi` | `single_task` | 42.90% | 4.2576 | 0.4800 | 48000 | 8321 | 16545 |
| `fixed` | `delayed_claim` | `shared` | 97.67% | 0.0541 | 0.0141 | 48101 | 8321 | 16545 |
| `fixed` | `delayed_claim` | `shared_task_heads` | 97.67% | 0.0523 | 0.0147 | 47821 | 8420 | 16545 |
| `fixed` | `delayed_claim` | `single_task` | 98.00% | 0.0422 | 0.0100 | 48000 | 8321 | 16545 |
| `fixed` | `grid_path` | `shared` | 90.73% | 0.3107 | 0.0280 | 47908 | 8321 | 16545 |
| `fixed` | `grid_path` | `shared_task_heads` | 90.97% | 0.2975 | 0.0181 | 48152 | 8420 | 16545 |
| `fixed` | `grid_path` | `single_task` | 91.17% | 0.2995 | 0.0253 | 48000 | 8321 | 16545 |

## Decision Rule

- Decision: `diagnostic_only`.
- Promote this architecture only if task heads improve fixed ARC versus the flat shared scorer, preserve delayed-claim transfer, avoid a material grid regression, and preserve ARC OOD versus the single-task baseline.
- If task heads help fixed ARC but hurt OOD or transfer, keep them as diagnostic evidence rather than the default architecture.
