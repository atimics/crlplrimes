# Delayed-Claim Leave-One-Out

Seeds: `20240516, 20240517, 20240518`
Budget: `16` epochs x `3000` steps per active domain = `48000` expected updates per domain.
Evaluation samples: `1000` delayed-claim states per row.

| Training set | Omitted sibling | Claim top-1 | Δ vs claim-only | Δ vs all | NLL | ECE | Claim updates |
|---|---|---:|---:|---:|---:|---:|---:|
| `all` | `none` | 98.00% | +1.40pp | +0.00pp | 0.0472 | 0.0102 | 48101 |
| `no_prime_gap` | `prime_gap` | 97.83% | +1.23pp | -0.17pp | 0.0474 | 0.0079 | 47996 |
| `no_grid_path` | `grid_path` | 97.20% | +0.60pp | -0.80pp | 0.0566 | 0.0114 | 47996 |
| `no_arc_agi` | `arc_agi` | 98.50% | +1.90pp | +0.50pp | 0.0393 | 0.0106 | 47932 |
| `claim_only` | `all_siblings` | 96.60% | +0.00pp | -1.40pp | 0.0772 | 0.0196 | 48000 |

## Interpretation

All-domain training is `+1.40pp` versus claim-only on delayed-claim top-1.
The largest removal gain versus all-domain is `no_arc_agi` at `+0.50pp`.
The largest removal loss versus all-domain is `no_grid_path` at `-0.80pp`.
This keeps a delayed-claim positive-transfer claim alive, but it should be tied to the specific sibling-removal pattern above.
