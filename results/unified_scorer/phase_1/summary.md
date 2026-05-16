# Unified Scorer — Phase 1: Baseline parity

Unified scorer should land within ±1pp of the dedicated closed-loop at hidden=64x32.

See `docs/unified-scorer-proposal.md` for the full design.

## Conditional grammar

### marginal_states = 4096

| Proposer | Budget | Softmax group | Cov hit/write | Params | Forward work | Exact probes | Saved % | Proposer loss | Accepted | False rej. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| closed-loop (64x32) | 16 | 148 | 3/0 | 3.8K | 7.5K | 3.153 | 56.18% | 0.0073 | 16 | 0 |
| grammar | 16 | 0 | 2/0 |  |  | 3.197 | 55.57% | 0.0000 | 16 | 0 |
| neural | 16 | 0 | 2/0 | 541 | 1.1K | 7.041 | 2.15% | 0.0088 | 16 | 0 |
| random | 16 | 0 | 2/0 |  |  | 6.944 | 3.49% | 0.0000 | 16 | 0 |
| structural | 16 | 0 | 2/0 |  |  | 3.197 | 55.57% | 0.0000 | 16 | 0 |
| unified (64x32) | 16 | 148 | 3/0 | 8.3K | 16.5K | 3.197 | 55.57% | 4.0262 | 16 | 0 |
| validation-oracle | 16 | 0 | 2/0 |  |  | 3.106 | 56.83% | 0.0000 | 16 | 0 |

_Params and Forward work are driver estimates from the declared hidden size unless the binary emitted matching `parameters` / `forward_flops` fields, in which case those are used. The estimate uses input width = 96 for unified rows and 25 for closed-loop rows; adjust `DEFAULT_FEATURE_WIDTHS` if the engine changes them._

## Unconditional grammar

### marginal_states = 4096

| Proposer | Budget | Softmax group | Cov hit/write | Params | Forward work | Exact probes | Saved % | Proposer loss | Accepted | False rej. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| closed-loop (64x32) | 16 | 4096 | 3/0 | 3.8K | 7.5K | 2.201 | 69.40% | 0.0017 | 10 | 0 |
| grammar | 16 | 0 | 2/0 |  |  | 1.927 | 73.22% | 0.0000 | 16 | 0 |
| neural | 16 | 0 | 2/0 | 541 | 1.1K | 1.979 | 72.49% | 0.0046 | 16 | 0 |
| random | 16 | 0 | 2/0 |  |  | 3.864 | 46.30% | 0.0000 | 16 | 0 |
| structural | 16 | 0 | 2/0 |  |  | 1.927 | 73.22% | 0.0000 | 16 | 0 |
| unified (64x32) | 16 | 4096 | 3/0 | 8.3K | 16.5K | 1.927 | 73.22% | 4.1620 | 16 | 0 |
| validation-oracle | 16 | 0 | 2/0 |  |  | 1.927 | 73.22% | 0.0000 | 16 | 0 |

_Params and Forward work are driver estimates from the declared hidden size unless the binary emitted matching `parameters` / `forward_flops` fields, in which case those are used. The estimate uses input width = 96 for unified rows and 25 for closed-loop rows; adjust `DEFAULT_FEATURE_WIDTHS` if the engine changes them._

