# Shared Multitask Domain Matrix

`examples/multitask_domain.c` is a generalization smoke test for the C scorer
library. It trains one scalar action scorer across multiple domains by mapping
every task into the same fixed-width action-row envelope.

This is intentionally separate from the certified-gate evidence. The first-class
task contract lives in `crlp.experiment_schema.v1` and currently covers
`prime_gap`, `grid_path`, `delayed_claim`, `program_synthesis`, and the fixed
`arc_agi` sample. The multitask binary asks a different question: can the same
core scorer, update path, and summary protocol train across heterogeneous action
domains in shuffled order?

A second shared-domain smoke covers symbolic rules:
`examples/symbolic_domains.c`. That binary uses `NNSymbolicDomain` instead of
the fixed 96-wide action-row envelope, and trains one certificate-status rule
classifier across prime, grid, delayed-claim, and a tiny ARC transform grammar.
It is a shared formal-interface smoke, not the closed-loop residual proposer
benchmark.

## Domains

| Domain | Action task | Candidate count | Label source | Status |
| --- | --- | ---: | --- | --- |
| `prime_gap` | rank candidate next-prime gaps | 129 | segmented-sieve label | clean scorer row, no residue flags |
| `grid_path` | rank four grid moves | 4 | BFS shortest-path successor | exact verifier label |
| `delayed_claim` | choose trust/ignore/evidence/cautious | 4 | deterministic hidden-outcome simulator | synthetic delayed-label task |
| `arc_agi` | choose output-cell color | 10 | ARC-AGI-1 task `0520fde7` output grid | first-class fixed-sample schema plus certified transform gate; shared multitask row remains action-only |

The ARC sample is reduced to balanced cell-color scoring. Given an input grid,
an output coordinate, and a candidate color action `0..9`, the scorer ranks the
candidate colors. The row exposes the flattened input grid, output coordinate,
grid sizes, candidate action, and optional task tag. It does not expose a direct
overlap/intersection rule. The standalone `make arc-agi` path uses the same
fixed task without the multitask task tag, adds a certified transform-gate row,
and is captured by `make results`.

## Feature Envelope

The shared row width is 96 features:

```text
bias
optional task one-hot
normalized action index
normalized candidate count
common state/action fields
domain-specific fields
```

The active feature-envelope names are:

| Variant | CLI flag | `feature_envelope` |
| --- | --- | --- |
| Task tagged | default | `task_tagged_action_rows_v1` |
| Untagged | `--no-task-tags` | `untagged_action_rows_v1` |

Prime residue features were removed from this envelope. That keeps the
multitask comparison honest: prime cannot quietly receive hand-coded modular
predicates that the other domains do not get.

## Run Commands

Build and run one all-domain scorer:

```sh
make multitask-domain
./build/float/multitask_domain --tasks all --eval-tasks all
```

Build and run the standalone first-class ARC sample:

```sh
make arc-agi
```

Run the ARC-format JSON loader over a local corpus checkout:

```sh
make arc-agi-loader ARC_AGI_TASK_DIR=/path/to/ARC-AGI/data/training ARC_AGI_MAX_TASKS=400
./build/float/arc_agi --task-dir /path/to/ARC-AGI/data/evaluation --max-tasks 400 --loader-summary-only
```

Build and run the shared symbolic-rule smoke:

```sh
make symbolic-domains
```

`make symbolic-domains` currently emits `crlp.symbolic_domain.summary.v1` and
`crlp.symbolic_rule_proposer.summary.v1` directly to stdout. The domain rows
include rule-feature leakage-policy fields. It is not captured by the canonical
`make results` pipeline yet.

Run the repeatable multi-seed matrix:

```sh
python3 scripts/multitask_ablation_matrix.py
```

The matrix enumerates every non-empty training subset of the configured domains
and evaluates all domains each time. With four domains this produces 15 training
sets per variant. The script scales `--steps-per-epoch` by the number of active
training domains so each trained domain gets a roughly matched update budget.

Artifacts are written to:

```text
build/float/multitask_ablation_matrix/raw.jsonl
build/float/multitask_ablation_matrix/trained_summary.csv
build/float/multitask_ablation_matrix/heldout_summary.csv
build/float/multitask_ablation_matrix/summary.md
build/float/multitask_ablation_matrix/commands.txt
```

## Summary Schema

Each evaluated domain emits one JSON row:

```json
{
  "schema": "crlp.multitask.summary.v1",
  "task": "multi_domain",
  "domain": "prime_gap",
  "task_set": "all",
  "eval_task_set": "all",
  "model_kind": "shared_action_scorer",
  "training_order": "shuffled",
  "feature_envelope": "task_tagged_action_rows_v1",
  "train_updates": 12000,
  "epochs": 4,
  "steps_per_epoch": 12000,
  "learning_rate": 0.004,
  "seed": 20240516,
  "sample_count": 1000,
  "candidate_count": 129.0,
  "top1": 0.12,
  "top5": 0.42,
  "top10": 0.78,
  "nll": 2.66,
  "mean_rank": 14.0,
  "decision_utility": 0.0,
  "baseline_steps": 0.0
}
```

`scripts/neurosymbolic_tuning.py` also recognizes
`crlp.multitask.summary.v1` and normalizes its config key as:

```text
task_set:model_kind:feature_envelope
```

There is no `prime_features` field; prime residue flags are not a multitask
feature option under this envelope.

## Current Matrix

Current all-domain trained top-1 means over three seeds
`20240516,20240517,20240518`:

| Variant | Prime | Grid | Claim | ARC |
| --- | ---: | ---: | ---: | ---: |
| Task tags | 11.77% | 90.37% | 95.50% | 26.13% |
| No task tags | 14.07% | 90.00% | 95.97% | 30.90% |

The interpretation is deliberately conservative:

- The shared scorer and training loop work across all configured domains.
- `grid_path` and `delayed_claim` remain easy under this envelope.
- `prime_gap` is weak without symbolic residue predicates.
- The harder ARC row is unsaturated under this envelope; the standalone loader
  can scan many ARC tasks, but the current certified grammar still covers only a
  narrow split-panel family rather than a general ARC solver.

## Next Steps

1. Expand the ARC loader grammar beyond the current split-panel copy/mask/union
   family.
2. Add task-family metadata so ARC tasks with different transformations can be
   evaluated without changing the scorer code.
3. Promote loader-backed symbolic ARC evidence into matched grammar/proposer
   comparisons before treating ARC as production verifier-gated ARC evidence.
4. Track transfer with held-out task families, not only held-out domains from
   the same compiled set.
