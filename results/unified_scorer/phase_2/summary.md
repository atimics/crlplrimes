# Unified Scorer — Phase 2: Capacity scaling

Sweep hidden capacity; compare unified vs closed-loop at each size.

See `docs/unified-scorer-proposal.md` for the full design.

## Conditional grammar

### marginal_states = 4096

| Proposer | Budget | Softmax group | Cov hit/write | Params | Forward work | Exact probes | Saved % | Proposer loss | Accepted | False rej. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unified (128x64) | 16 | 148 | 3/0 | 20.7K | 41.3K | 3.153 | 56.18% | 4.0350 | 16 | 0 |
| unified (256x128) | 16 | 148 | 3/0 | 57.9K | 115.3K | 3.153 | 56.18% | 3.4337 | 16 | 0 |
| unified (64x32) | 16 | 148 | 3/0 | 8.3K | 16.5K | 3.197 | 55.57% | 4.0262 | 16 | 0 |

_Params and Forward work are driver estimates from the declared hidden size unless the binary emitted matching `parameters` / `forward_flops` fields, in which case those are used. The estimate uses input width = 96 for unified rows and 25 for closed-loop rows; adjust `DEFAULT_FEATURE_WIDTHS` if the engine changes them._

## Unconditional grammar

### marginal_states = 4096

| Proposer | Budget | Softmax group | Cov hit/write | Params | Forward work | Exact probes | Saved % | Proposer loss | Accepted | False rej. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unified (128x64) | 16 | 4096 | 3/0 | 20.7K | 41.3K | 1.961 | 72.75% | 4.0195 | 15 | 0 |
| unified (256x128) | 16 | 4096 | 3/0 | 57.9K | 115.3K | 1.927 | 73.22% | 3.4688 | 16 | 0 |
| unified (64x32) | 16 | 4096 | 3/0 | 8.3K | 16.5K | 1.927 | 73.22% | 4.1620 | 16 | 0 |

_Params and Forward work are driver estimates from the declared hidden size unless the binary emitted matching `parameters` / `forward_flops` fields, in which case those are used. The estimate uses input width = 96 for unified rows and 25 for closed-loop rows; adjust `DEFAULT_FEATURE_WIDTHS` if the engine changes them._

