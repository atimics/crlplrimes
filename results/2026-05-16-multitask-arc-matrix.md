# Multitask ARC Matrix Snapshot

Date: 2026-05-16

This snapshot records the first four-domain shared-envelope matrix after
removing prime residue features from `examples/multitask_domain.c`.

## Command

```sh
make multitask-domain
```

The run generated 360 raw JSON rows, 65 trained-domain summary rows, and 57
held-out summary rows.

## Artifacts

```text
build/float/multitask_ablation_matrix/raw.jsonl
build/float/multitask_ablation_matrix/trained_summary.csv
build/float/multitask_ablation_matrix/heldout_summary.csv
build/float/multitask_ablation_matrix/summary.md
build/float/multitask_ablation_matrix/commands.txt
```

## All-Domain Results

Mean trained-domain top-1 over seeds `20240516,20240517,20240518`:

| Variant | Prime | Grid | Claim | ARC |
| --- | ---: | ---: | ---: | ---: |
| Task tags | 11.77% | 90.37% | 95.50% | 26.13% |
| No task tags | 14.07% | 90.00% | 95.97% | 30.90% |

Full all-domain rows:

| Variant | Domain | Top-1 | NLL |
| --- | --- | ---: | ---: |
| Task tags | `prime_gap` | 11.77% | 2.6633 |
| Task tags | `grid_path` | 90.37% | 0.3530 |
| Task tags | `delayed_claim` | 95.50% | 0.1022 |
| Task tags | `arc_agi` | 26.13% | 3.1627 |
| No task tags | `prime_gap` | 14.07% | 2.6603 |
| No task tags | `grid_path` | 90.00% | 0.3607 |
| No task tags | `delayed_claim` | 95.97% | 0.1003 |
| No task tags | `arc_agi` | 30.90% | 2.9508 |

## Interpretation

- Prime residue predicates are gone from the shared multitask envelope.
- The prime row is weak without symbolic modular structure, which is the
  expected honest result.
- Grid and delayed-claim remain easy under the fixed-width action-row encoding.
- ARC-AGI-1 task `0520fde7` is no longer saturated, but this is still a single
  supervised cell-color sample, not an ARC solver.
- Task tags are not currently necessary for the all-domain matrix; the untagged
  variant slightly outperformed the tagged variant in this run.

## Verification

The code state used for this snapshot passed:

```sh
make source-policy-check
make build/float/multitask_domain
make test
make sanitize
make build/double/multitask_domain REAL=double
make check
```
