# Fully Trained Tiny-Net Audit

Date: 2026-06-15
Seeds: `20240516, 20240517, 20240518`
Capacities: `8x4, 16x8, 32x16, 64x32`
Model variants: `shared, shared_task_heads, single_task`
Epoch checkpoints: `1, 2, 4, 8, 16, 32`
Steps per epoch: `12000`; eval samples: `1000`

## Best Checkpoints

| Hidden | Model | Prime | Grid | Delayed | ARC |
| --- | --- | ---: | ---: | ---: | ---: |
| `8x4` | `shared` | 56.70% @ 1 | 90.60% @ 16 | 98.63% @ 32 | 45.17% @ 16 |
| `8x4` | `shared_task_heads` | 56.70% @ 1 | 90.93% @ 32 | 98.77% @ 32 | 34.53% @ 8 |
| `8x4` | `single_task` | 56.70% @ 1 | 91.43% @ 32 | 97.97% @ 16 | 45.20% @ 8 |
| `16x8` | `shared` | 56.70% @ 2 | 90.87% @ 32 | 98.37% @ 32 | 45.23% @ 8 |
| `16x8` | `shared_task_heads` | 56.70% @ 1 | 91.03% @ 4 | 98.73% @ 32 | 33.30% @ 4 |
| `16x8` | `single_task` | 56.70% @ 1 | 91.03% @ 32 | 97.97% @ 16 | 45.20% @ 4 |
| `32x16` | `shared` | 56.70% @ 1 | 91.10% @ 16 | 98.77% @ 32 | 35.67% @ 1 |
| `32x16` | `shared_task_heads` | 56.70% @ 1 | 90.97% @ 4 | 98.63% @ 32 | 34.53% @ 32 |
| `32x16` | `single_task` | 56.70% @ 1 | 91.37% @ 32 | 97.97% @ 16 | 45.20% @ 4 |
| `64x32` | `shared` | 56.70% @ 1 | 91.47% @ 8 | 98.50% @ 32 | 30.93% @ 2 |
| `64x32` | `shared_task_heads` | 56.70% @ 1 | 91.33% @ 8 | 98.40% @ 32 | 36.90% @ 4 |
| `64x32` | `single_task` | 56.70% @ 1 | 91.27% @ 32 | 98.03% @ 16 | 42.80% @ 4 |

## Plateau Flags

| Hidden | Domain | Model | Last gain | Window gain | Plateau |
| --- | --- | --- | ---: | ---: | --- |
| `8x4` | `prime_gap` | `shared` | -0.90pp | -0.90pp | no |
| `8x4` | `grid_path` | `shared` | -0.57pp | +0.37pp | no |
| `8x4` | `delayed_claim` | `shared` | +0.73pp | +0.87pp | no |
| `8x4` | `arc_agi` | `shared` | -5.93pp | +0.00pp | no |
| `8x4` | `prime_gap` | `shared_task_heads` | +0.00pp | +0.00pp | no |
| `8x4` | `grid_path` | `shared_task_heads` | +0.37pp | +0.00pp | yes |
| `8x4` | `delayed_claim` | `shared_task_heads` | +2.47pp | +1.97pp | no |
| `8x4` | `arc_agi` | `shared_task_heads` | +0.00pp | -2.37pp | no |
| `8x4` | `prime_gap` | `single_task` | +0.00pp | +0.00pp | no |
| `8x4` | `grid_path` | `single_task` | +0.60pp | +0.63pp | no |
| `8x4` | `delayed_claim` | `single_task` | -1.07pp | -0.13pp | no |
| `8x4` | `arc_agi` | `single_task` | +0.00pp | +0.00pp | yes |
| `16x8` | `prime_gap` | `shared` | -0.37pp | -0.37pp | no |
| `16x8` | `grid_path` | `shared` | +0.30pp | +0.10pp | yes |
| `16x8` | `delayed_claim` | `shared` | +0.00pp | +0.57pp | no |
| `16x8` | `arc_agi` | `shared` | -8.33pp | -8.33pp | no |
| `16x8` | `prime_gap` | `shared_task_heads` | +0.00pp | +0.00pp | no |
| `16x8` | `grid_path` | `shared_task_heads` | -0.10pp | -0.03pp | no |
| `16x8` | `delayed_claim` | `shared_task_heads` | +1.00pp | +1.13pp | no |
| `16x8` | `arc_agi` | `shared_task_heads` | +5.97pp | -1.17pp | no |
| `16x8` | `prime_gap` | `single_task` | +0.00pp | +0.00pp | no |
| `16x8` | `grid_path` | `single_task` | +0.07pp | +0.17pp | yes |
| `16x8` | `delayed_claim` | `single_task` | -0.83pp | +0.00pp | no |
| `16x8` | `arc_agi` | `single_task` | +0.00pp | +0.00pp | no |
| `32x16` | `prime_gap` | `shared` | +0.00pp | +0.00pp | no |
| `32x16` | `grid_path` | `shared` | -0.43pp | +0.00pp | yes |
| `32x16` | `delayed_claim` | `shared` | +0.90pp | +0.20pp | no |
| `32x16` | `arc_agi` | `shared` | -2.37pp | +8.33pp | no |
| `32x16` | `prime_gap` | `shared_task_heads` | +0.00pp | +0.00pp | no |
| `32x16` | `grid_path` | `shared_task_heads` | +0.00pp | +0.00pp | no |
| `32x16` | `delayed_claim` | `shared_task_heads` | +0.97pp | +1.13pp | no |
| `32x16` | `arc_agi` | `shared_task_heads` | +8.33pp | +8.33pp | no |
| `32x16` | `prime_gap` | `single_task` | +0.00pp | +0.00pp | no |
| `32x16` | `grid_path` | `single_task` | +0.23pp | +0.07pp | yes |
| `32x16` | `delayed_claim` | `single_task` | -0.30pp | +0.60pp | no |
| `32x16` | `arc_agi` | `single_task` | +0.00pp | +0.00pp | no |
| `64x32` | `prime_gap` | `shared` | +0.87pp | +0.00pp | no |
| `64x32` | `grid_path` | `shared` | -0.03pp | -0.50pp | no |
| `64x32` | `delayed_claim` | `shared` | +0.53pp | +0.13pp | no |
| `64x32` | `arc_agi` | `shared` | -0.03pp | -0.03pp | no |
| `64x32` | `prime_gap` | `shared_task_heads` | +0.00pp | +0.00pp | no |
| `64x32` | `grid_path` | `shared_task_heads` | -0.07pp | -0.43pp | yes |
| `64x32` | `delayed_claim` | `shared_task_heads` | +1.33pp | +0.70pp | no |
| `64x32` | `arc_agi` | `shared_task_heads` | -2.37pp | -2.37pp | no |
| `64x32` | `prime_gap` | `single_task` | +0.00pp | +0.00pp | no |
| `64x32` | `grid_path` | `single_task` | +0.17pp | +0.20pp | yes |
| `64x32` | `delayed_claim` | `single_task` | -0.30pp | +0.53pp | no |
| `64x32` | `arc_agi` | `single_task` | +0.00pp | +0.00pp | no |

## Width Verdict

| Domain | Model | Best hidden | Largest-smallest top-1 | Larger helped |
| --- | --- | ---: | ---: | --- |
| `prime_gap` | `shared` | `8x4` | +0.00pp | no |
| `prime_gap` | `shared_task_heads` | `8x4` | +0.00pp | no |
| `prime_gap` | `single_task` | `8x4` | +0.00pp | no |
| `grid_path` | `shared` | `64x32` | +0.87pp | yes |
| `grid_path` | `shared_task_heads` | `64x32` | +0.40pp | yes |
| `grid_path` | `single_task` | `8x4` | -0.17pp | no |
| `delayed_claim` | `shared` | `32x16` | -0.13pp | yes |
| `delayed_claim` | `shared_task_heads` | `8x4` | -0.37pp | no |
| `delayed_claim` | `single_task` | `64x32` | +0.07pp | yes |
| `arc_agi` | `shared` | `16x8` | -14.23pp | yes |
| `arc_agi` | `shared_task_heads` | `64x32` | +2.37pp | yes |
| `arc_agi` | `single_task` | `8x4` | -2.40pp | no |

## Audit Verdict

| Domain | Model | Coverage | Plateau rows | Best hidden | Audit ready |
| --- | --- | --- | ---: | ---: | --- |
| `prime_gap` | `shared` | yes | 0/4 | `8x4` | no |
| `prime_gap` | `shared_task_heads` | yes | 0/4 | `8x4` | no |
| `prime_gap` | `single_task` | yes | 0/4 | `8x4` | no |
| `grid_path` | `shared` | yes | 2/4 | `64x32` | no |
| `grid_path` | `shared_task_heads` | yes | 2/4 | `64x32` | no |
| `grid_path` | `single_task` | yes | 3/4 | `8x4` | no |
| `delayed_claim` | `shared` | yes | 0/4 | `32x16` | no |
| `delayed_claim` | `shared_task_heads` | yes | 0/4 | `8x4` | no |
| `delayed_claim` | `single_task` | yes | 0/4 | `64x32` | no |
| `arc_agi` | `shared` | yes | 0/4 | `16x8` | no |
| `arc_agi` | `shared_task_heads` | yes | 0/4 | `64x32` | no |
| `arc_agi` | `single_task` | yes | 1/4 | `8x4` | no |
