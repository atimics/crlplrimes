# Multitask Capacity Sweep

Seeds: `20240516, 20240517, 20240518`
Capacities: `32x16, 64x32, 128x64, 256x128`
Budget: `16` epochs x `12000` steps = `192000` total shared updates.
Evaluation samples: `1000` per domain row.

## Fixed-Task Shared Minus Single Top-1

| Hidden | Params | Prime gap | Grid path | Delayed claim | ARC fixed |
|---|---:|---:|---:|---:|---:|
| `32x16` | 3649 | +1.63pp | -0.20pp | +0.43pp | -1.23pp |
| `64x32` | 8321 | +0.43pp | -0.23pp | +0.10pp | -16.63pp |
| `128x64` | 20737 | +0.43pp | -0.40pp | +0.13pp | -14.27pp |
| `256x128` | 57857 | +1.87pp | -0.30pp | +0.43pp | -8.30pp |

## ARC Fixed-Task Detail

| Hidden | Shared top-1 | Single top-1 | Delta | Shared NLL | Single NLL | Shared ECE | Single ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `32x16` | 39.20% | 40.43% | -1.23pp | 2.8153 | 8.8215 | 0.4819 | 0.5869 |
| `64x32` | 26.17% | 42.80% | -16.63pp | 4.0827 | 8.2754 | 0.6995 | 0.5589 |
| `128x64` | 26.17% | 40.43% | -14.27pp | 4.0758 | 8.2046 | 0.7289 | 0.5557 |
| `256x128` | 34.50% | 42.80% | -8.30pp | 3.9021 | 8.0251 | 0.6683 | 0.5576 |

## Delayed-Claim Guardrail

| Hidden | Shared top-1 | Single top-1 | Delta | Shared NLL | Single NLL |
|---|---:|---:|---:|---:|---:|
| `32x16` | 98.60% | 98.17% | +0.43pp | 0.0395 | 0.0414 |
| `64x32` | 98.00% | 97.90% | +0.10pp | 0.0472 | 0.0412 |
| `128x64` | 98.17% | 98.03% | +0.13pp | 0.0443 | 0.0412 |
| `256x128` | 98.30% | 97.87% | +0.43pp | 0.0385 | 0.0424 |

## Interpretation

- Best shared fixed-ARC top-1 is `39.20%` at `32x16`.
- Least negative fixed-ARC shared-minus-single delta is `-1.23pp` at `32x16`.
- R-MT.6 should be read as a fixed-envelope capacity diagnostic; it does not replace the same-envelope ARC OOD result from R-MT.2.
