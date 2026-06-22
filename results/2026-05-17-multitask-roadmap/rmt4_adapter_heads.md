# Multitask Adapter-Headed Sharing

Seeds: `20240516, 20240517, 20240518`
Budget: `16` epochs x `12000` steps = `192000` total shared updates.
Evaluation samples: `1000` per domain row.

This run compares the flat shared scalar scorer, a shared trunk with per-domain scalar heads, and matched single-task baselines under the same 96-feature envelope.

## Fixed-Suite Top-1

| Domain | Shared | Task heads | Single | Heads - shared | Heads - single |
|---|---:|---:|---:|---:|---:|
| `prime_gap` | 55.83% | 56.70% | 56.70% | +0.87pp | +0.00pp |
| `grid_path` | 91.00% | 90.97% | 91.10% | -0.03pp | -0.13pp |
| `delayed_claim` | 97.97% | 97.07% | 98.03% | -0.90pp | -0.97pp |
| `arc_agi` | 26.20% | 36.90% | 42.80% | +10.70pp | -5.90pp |

## ARC OOD Top-1

| Suite | Shared | Task heads | Single | Heads - shared | Heads - single |
|---|---:|---:|---:|---:|---:|
| `arc_ood` | 48.90% | 47.77% | 42.90% | -1.13pp | +4.87pp |

## ARC And Transfer Guardrail

| Suite | Domain | Model | Top-1 | NLL | ECE | Updates | Params | FLOPs |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `fixed` | `arc_agi` | `shared` | 26.20% | 3.8792 | 0.6986 | 47869 | 8321 | 16545 |
| `fixed` | `arc_agi` | `shared_task_heads` | 36.90% | 4.4355 | 0.6510 | 47954 | 8453 | 16545 |
| `fixed` | `arc_agi` | `single_task` | 42.80% | 8.0370 | 0.5702 | 48000 | 8321 | 16545 |
| `ood` | `arc_agi` | `shared` | 48.90% | 2.5472 | 0.4373 | 47869 | 8321 | 16545 |
| `ood` | `arc_agi` | `shared_task_heads` | 47.77% | 3.2086 | 0.4563 | 47954 | 8453 | 16545 |
| `ood` | `arc_agi` | `single_task` | 42.90% | 4.2592 | 0.4780 | 48000 | 8321 | 16545 |
| `fixed` | `delayed_claim` | `shared` | 97.97% | 0.0449 | 0.0143 | 48101 | 8321 | 16545 |
| `fixed` | `delayed_claim` | `shared_task_heads` | 97.07% | 0.0660 | 0.0161 | 47821 | 8453 | 16545 |
| `fixed` | `delayed_claim` | `single_task` | 98.03% | 0.0422 | 0.0099 | 48000 | 8321 | 16545 |
| `fixed` | `grid_path` | `shared` | 91.00% | 0.3152 | 0.0386 | 47908 | 8321 | 16545 |
| `fixed` | `grid_path` | `shared_task_heads` | 90.97% | 0.3018 | 0.0276 | 48152 | 8453 | 16545 |
| `fixed` | `grid_path` | `single_task` | 91.10% | 0.3013 | 0.0285 | 48000 | 8321 | 16545 |

## Decision Rule

- Decision: see `rmt4_adapter_heads_decision.jsonl`.
- Promote this architecture only if task heads improve fixed ARC versus the flat shared scorer, preserve delayed-claim transfer, avoid a material grid regression, and preserve ARC OOD versus the single-task baseline.
- If task heads help fixed ARC but hurt OOD or transfer, keep them as diagnostic evidence rather than the default architecture.
