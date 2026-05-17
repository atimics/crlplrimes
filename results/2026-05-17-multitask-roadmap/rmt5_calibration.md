# Multitask Calibration Audit

Source: `raw.jsonl` rows with schema `crlp.multitask.summary.v1`.
ECE means are seed means; reliability-bin tables pool counts across seeds.

## R-MT.5 Fixed-Task Calibration

| Evaluation | Shared ECE | Single ECE | Delta | Shared NLL | Single NLL | Delta | Shared conf. | Single conf. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `arc_agi` | 0.6995 | 0.5589 | +0.1406 | 4.0827 | 8.2754 | -4.1927 | 0.9405 | 0.9600 |
| `delayed_claim` | 0.0102 | 0.0134 | -0.0033 | 0.0472 | 0.0412 | +0.0061 | 0.9791 | 0.9779 |
| `grid_path` | 0.0357 | 0.0268 | +0.0089 | 0.3216 | 0.2999 | +0.0217 | 0.9031 | 0.8914 |
| `prime_gap` | 0.0279 | 0.0410 | -0.0131 | 2.6648 | 2.6733 | -0.0085 | 0.1362 | 0.1371 |

## R-MT.5 ARC OOD Calibration

| Evaluation | Shared ECE | Single ECE | Delta | Shared NLL | Single NLL | Delta | Shared conf. | Single conf. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `arc_agi_ood_same_envelope` | 0.4481 | 0.4820 | -0.0339 | 2.6278 | 4.3723 | -1.7445 | 0.9119 | 0.9025 |

## Interpretation

On fixed `0520fde7`, shared training still hurts top-1, but it cuts ARC NLL from `8.2754` to `4.0827`; ECE is worse (`0.6995` vs `0.5589`).
On the same-envelope ARC OOD aggregate, shared training improves both ranking and calibration: NLL `2.6278` vs `4.3723`, ECE `0.4481` vs `0.4820`.
For `prime_gap`, `grid_path`, and `delayed_claim`, ECE deltas are small enough that R-MT.5 should be framed as an ARC clarification rather than a broad calibration win.

### Reliability Bins: `arc_agi`

| Model | Bin | Count | Confidence | Accuracy | Gap | ECE contribution |
|---|---:|---:|---:|---:|---:|---:|
| `shared` | 0.5-0.6 | 71 | 0.5643 | 1.0000 | -0.4357 | 0.0103 |
| `shared` | 0.6-0.7 | 71 | 0.6376 | 0.0000 | +0.6376 | 0.0151 |
| `shared` | 0.7-0.8 | 250 | 0.7584 | 0.0000 | +0.7584 | 0.0632 |
| `shared` | 0.8-0.9 | 72 | 0.8891 | 0.0000 | +0.8891 | 0.0213 |
| `shared` | 0.9-1.0 | 2536 | 0.9790 | 0.2815 | +0.6974 | 0.5896 |
| `single_task` | 0.6-0.7 | 71 | 0.6581 | 1.0000 | -0.3419 | 0.0081 |
| `single_task` | 0.7-0.8 | 144 | 0.7428 | 0.5000 | +0.2428 | 0.0117 |
| `single_task` | 0.8-0.9 | 358 | 0.8563 | 0.7989 | +0.0574 | 0.0068 |
| `single_task` | 0.9-1.0 | 2427 | 0.9971 | 0.3523 | +0.6448 | 0.5216 |

### Reliability Bins: `arc_agi_ood_same_envelope`

| Model | Bin | Count | Confidence | Accuracy | Gap | ECE contribution |
|---|---:|---:|---:|---:|---:|---:|
| `shared` | 0.5-0.6 | 138 | 0.5467 | 0.6884 | -0.1417 | 0.0065 |
| `shared` | 0.6-0.7 | 217 | 0.6606 | 0.3410 | +0.3196 | 0.0231 |
| `shared` | 0.7-0.8 | 185 | 0.7514 | 0.6486 | +0.1027 | 0.0063 |
| `shared` | 0.8-0.9 | 302 | 0.8452 | 0.4272 | +0.4181 | 0.0421 |
| `shared` | 0.9-1.0 | 2158 | 0.9836 | 0.4791 | +0.5045 | 0.3629 |
| `single_task` | 0.5-0.6 | 250 | 0.5441 | 0.4000 | +0.1441 | 0.0120 |
| `single_task` | 0.6-0.7 | 179 | 0.6561 | 0.4525 | +0.2036 | 0.0121 |
| `single_task` | 0.7-0.8 | 243 | 0.7593 | 0.6049 | +0.1544 | 0.0125 |
| `single_task` | 0.8-0.9 | 292 | 0.8594 | 0.4247 | +0.4347 | 0.0423 |
| `single_task` | 0.9-1.0 | 2036 | 0.9914 | 0.4091 | +0.5823 | 0.3952 |
