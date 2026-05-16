# Unified Scorer — Phase 3: Data-sparse target

Data-sparse marginal-utility target. Unified scorer should beat greedy-on-subsample.

See `docs/unified-scorer-proposal.md` for the full design.

## Conditional grammar

### marginal_states = 64

| Proposer | Budget | Softmax group | Cov hit/write | Params | Forward work | Exact probes | Saved % | Proposer loss | Accepted | False rej. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| closed-loop (128x64) | 16 | 148 | 3/0 | 11.6K | 23.1K | 3.144 | 56.30% | 0.0164 | 16 | 0 |
| grammar | 16 | 0 | 2/0 |  |  | 3.197 | 55.57% | 0.0000 | 16 | 0 |
| neural | 16 | 0 | 2/0 | 541 | 1.1K | 7.041 | 2.15% | 0.0104 | 16 | 0 |
| random | 16 | 0 | 2/0 |  |  | 6.944 | 3.49% | 0.0000 | 16 | 0 |
| structural | 16 | 0 | 2/0 |  |  | 3.197 | 55.57% | 0.0000 | 16 | 0 |
| unified (128x64) | 16 | 148 | 3/0 | 20.7K | 41.3K | 3.153 | 56.18% | 3.4651 | 16 | 0 |
| validation-oracle | 16 | 0 | 2/0 |  |  | 3.185 | 55.74% | 0.0000 | 16 | 0 |

_Params and Forward work are driver estimates from the declared hidden size unless the binary emitted matching `parameters` / `forward_flops` fields, in which case those are used. The estimate uses input width = 96 for unified rows and 25 for closed-loop rows; adjust `DEFAULT_FEATURE_WIDTHS` if the engine changes them._

### marginal_states = 256

| Proposer | Budget | Softmax group | Cov hit/write | Params | Forward work | Exact probes | Saved % | Proposer loss | Accepted | False rej. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| closed-loop (128x64) | 16 | 148 | 3/0 | 11.6K | 23.1K | 3.144 | 56.31% | 0.0108 | 16 | 0 |
| grammar | 16 | 0 | 2/0 |  |  | 3.197 | 55.57% | 0.0000 | 16 | 0 |
| neural | 16 | 0 | 2/0 | 541 | 1.1K | 7.041 | 2.15% | 0.0089 | 16 | 0 |
| random | 16 | 0 | 2/0 |  |  | 6.944 | 3.49% | 0.0000 | 16 | 0 |
| structural | 16 | 0 | 2/0 |  |  | 3.197 | 55.57% | 0.0000 | 16 | 0 |
| unified (128x64) | 16 | 148 | 3/0 | 20.7K | 41.3K | 3.153 | 56.18% | 3.7823 | 16 | 0 |
| validation-oracle | 16 | 0 | 2/0 |  |  | 3.112 | 56.74% | 0.0000 | 16 | 0 |

_Params and Forward work are driver estimates from the declared hidden size unless the binary emitted matching `parameters` / `forward_flops` fields, in which case those are used. The estimate uses input width = 96 for unified rows and 25 for closed-loop rows; adjust `DEFAULT_FEATURE_WIDTHS` if the engine changes them._

### marginal_states = 4096

| Proposer | Budget | Softmax group | Cov hit/write | Params | Forward work | Exact probes | Saved % | Proposer loss | Accepted | False rej. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| closed-loop (128x64) | 16 | 148 | 3/0 | 11.6K | 23.1K | 3.153 | 56.18% | 0.0073 | 16 | 0 |
| grammar | 16 | 0 | 2/0 |  |  | 3.197 | 55.57% | 0.0000 | 16 | 0 |
| neural | 16 | 0 | 2/0 | 541 | 1.1K | 7.041 | 2.15% | 0.0088 | 16 | 0 |
| random | 16 | 0 | 2/0 |  |  | 6.944 | 3.49% | 0.0000 | 16 | 0 |
| structural | 16 | 0 | 2/0 |  |  | 3.197 | 55.57% | 0.0000 | 16 | 0 |
| unified (128x64) | 16 | 148 | 3/0 | 20.7K | 41.3K | 3.153 | 56.18% | 4.0350 | 16 | 0 |
| validation-oracle | 16 | 0 | 2/0 |  |  | 3.106 | 56.83% | 0.0000 | 16 | 0 |

_Params and Forward work are driver estimates from the declared hidden size unless the binary emitted matching `parameters` / `forward_flops` fields, in which case those are used. The estimate uses input width = 96 for unified rows and 25 for closed-loop rows; adjust `DEFAULT_FEATURE_WIDTHS` if the engine changes them._

## Unconditional grammar

### marginal_states = 64

| Proposer | Budget | Softmax group | Cov hit/write | Params | Forward work | Exact probes | Saved % | Proposer loss | Accepted | False rej. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| closed-loop (128x64) | 16 | 4096 | 3/0 | 11.6K | 23.1K | 2.275 | 68.38% | 0.0036 | 9 | 0 |
| grammar | 16 | 0 | 2/0 |  |  | 1.927 | 73.22% | 0.0000 | 16 | 0 |
| neural | 16 | 0 | 2/0 | 541 | 1.1K | 1.967 | 72.66% | 0.0046 | 16 | 0 |
| random | 16 | 0 | 2/0 |  |  | 3.864 | 46.30% | 0.0000 | 16 | 0 |
| structural | 16 | 0 | 2/0 |  |  | 1.927 | 73.22% | 0.0000 | 16 | 0 |
| unified (128x64) | 16 | 4096 | 3/0 | 20.7K | 41.3K | 2.004 | 72.15% | 3.7677 | 14 | 0 |
| validation-oracle | 16 | 0 | 2/0 |  |  | 1.987 | 72.39% | 0.0000 | 16 | 0 |

_Params and Forward work are driver estimates from the declared hidden size unless the binary emitted matching `parameters` / `forward_flops` fields, in which case those are used. The estimate uses input width = 96 for unified rows and 25 for closed-loop rows; adjust `DEFAULT_FEATURE_WIDTHS` if the engine changes them._

### marginal_states = 256

| Proposer | Budget | Softmax group | Cov hit/write | Params | Forward work | Exact probes | Saved % | Proposer loss | Accepted | False rej. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| closed-loop (128x64) | 16 | 4096 | 3/0 | 11.6K | 23.1K | 2.142 | 70.23% | 0.0031 | 11 | 0 |
| grammar | 16 | 0 | 2/0 |  |  | 1.927 | 73.22% | 0.0000 | 16 | 0 |
| neural | 16 | 0 | 2/0 | 541 | 1.1K | 2.003 | 72.16% | 0.0046 | 16 | 0 |
| random | 16 | 0 | 2/0 |  |  | 3.864 | 46.30% | 0.0000 | 16 | 0 |
| structural | 16 | 0 | 2/0 |  |  | 1.927 | 73.22% | 0.0000 | 16 | 0 |
| unified (128x64) | 16 | 4096 | 3/0 | 20.7K | 41.3K | 1.961 | 72.75% | 4.0622 | 15 | 0 |
| validation-oracle | 16 | 0 | 2/0 |  |  | 1.950 | 72.90% | 0.0000 | 16 | 0 |

_Params and Forward work are driver estimates from the declared hidden size unless the binary emitted matching `parameters` / `forward_flops` fields, in which case those are used. The estimate uses input width = 96 for unified rows and 25 for closed-loop rows; adjust `DEFAULT_FEATURE_WIDTHS` if the engine changes them._

### marginal_states = 4096

| Proposer | Budget | Softmax group | Cov hit/write | Params | Forward work | Exact probes | Saved % | Proposer loss | Accepted | False rej. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| closed-loop (128x64) | 16 | 4096 | 3/0 | 11.6K | 23.1K | 2.201 | 69.40% | 0.0018 | 10 | 0 |
| grammar | 16 | 0 | 2/0 |  |  | 1.927 | 73.22% | 0.0000 | 16 | 0 |
| neural | 16 | 0 | 2/0 | 541 | 1.1K | 1.979 | 72.49% | 0.0046 | 16 | 0 |
| random | 16 | 0 | 2/0 |  |  | 3.864 | 46.30% | 0.0000 | 16 | 0 |
| structural | 16 | 0 | 2/0 |  |  | 1.927 | 73.22% | 0.0000 | 16 | 0 |
| unified (128x64) | 16 | 4096 | 3/0 | 20.7K | 41.3K | 1.961 | 72.75% | 4.0195 | 15 | 0 |
| validation-oracle | 16 | 0 | 2/0 |  |  | 1.927 | 73.22% | 0.0000 | 16 | 0 |

_Params and Forward work are driver estimates from the declared hidden size unless the binary emitted matching `parameters` / `forward_flops` fields, in which case those are used. The estimate uses input width = 96 for unified rows and 25 for closed-loop rows; adjust `DEFAULT_FEATURE_WIDTHS` if the engine changes them._

