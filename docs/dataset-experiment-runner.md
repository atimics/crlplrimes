# Dataset-First Experiment Runner

The engine direction is to make most experiments data declarations plus
grounding operators, not bespoke binaries. The reusable loop is:

```text
dataset or generator -> grouped candidate rows -> verifier/target label
-> generic neural scorer -> trace/eval summary -> optional rule/macro discovery
```

`examples/dataset_action_score.c` is the first executable slice of that
direction. It can consume grouped CSV rows:

```text
split,group_id,state_hash,action_id,target,f0,...,fN
```

Each group is one state with a complete candidate set. One or more rows in the
group can have `target=1`, which lets adapters label all equivalent verifier-
valid candidates instead of forcing an arbitrary gold choice. A synthetic
generator, exact verifier, simulator replay, or corpus proxy can produce the
rows. The runner trains the shared softmax action scorer, ranks candidates,
writes optional `crlp.trace.v1` rows, saves an optional checkpoint, and emits
`crlp.dataset_action_score.summary.v1`.

It can also generate rows in-process through a named shared experiment:

```sh
make dataset-action-score-adapter-smoke
./build/float/dataset_action_score --adapter threshold --task adapter_threshold_actions
./build/float/dataset_action_score --experiment bitwise_macro_step
```

The shared dataset boundary lives in `include/nn_dataset.h` and
`src/nn_dataset.c`. `NNDatasetExperiment` is the generic contract:

```text
name + domain + feature_schema + verifier_schema
+ generate(config) -> NNDataset rows
```

`--adapter` remains a compatibility alias for `--experiment`. The `threshold`
experiment is intentionally tiny. It
implements the same row contract without a CSV file:

```text
generate state x -> enumerate actions {0,1}
-> programmatic verifier marks action 1 iff x >= 0.5
-> generic runner trains/ranks/traces
```

That proves the boundary the rest of the engine should use. Domain adapters
should generate complete candidate groups and verifier labels; the runner should
own neural training, checkpointing, metrics, traces, and summaries.

Useful commands:

```sh
make dataset-action-score-smoke
make dataset-action-score-replay-smoke
make dataset-action-score-adapter-smoke
make dataset-action-score-multitarget-smoke
make dataset-action-score-bitwise-smoke
make dataset-action-score-bitwise-macro-smoke
make dataset-action-score-haiku-smoke
./build/float/dataset_action_score \
  --dataset tests/fixtures/dataset_action_score/threshold_actions.csv \
  --features 4 \
  --task dataset_threshold_actions \
  --trace build/float/dataset_action_score_artifacts/trace.jsonl
```

This is intentionally small. The contract it proves is the important part:

- experiments can be fixtures or generated artifacts;
- candidate enumeration is explicit and complete per group;
- the verifier writes labels before training;
- groups can carry one or more verifier-positive targets;
- neural code is generic across tasks;
- trace and checkpoint machinery are shared.

`--replay-trace` is the generic replay wedge. It consumes existing
`crlp.trace.v1` rows, validates the stable action-id/ranking contract, reads
`target_indices`, and recompiles each trace record into a grouped
`NNDataset`. Because current traces do not store original domain feature
vectors, the generic replay feature set is intentionally trace-native:

```text
trace_replay_logged_score_v1:
bias, action_id_low16, action_id_high16, candidate_index,
candidate_count, logged_score, logged_probability, logged_rank,
selected_flag, top_score_delta
```

That is enough for shared replay training, trace parsing, multi-target labels,
checkpoints, and numeric certificates. It is not a substitute for
domain-lossless replay adapters; grid, delayed-claim, ARC, haiku, or Signal
can still provide richer replay by regenerating domain features from their own
state/action encodings.

`threshold_multi` is the regression wedge for multi-target groups. It labels
every action whose value clears the state threshold as valid. The generic
trainer spreads target mass across all valid rows, and evaluation treats top-k
as correct when any verifier-valid candidate is ranked in the band.

The first real domain crossover is `bitwise_truth_table`. It converts the
bitwise recursive truth-mask task into grouped candidate rows:

```text
target truth table -> enumerate 16 candidate masks
-> verifier labels the exact matching mask
-> generic dataset scorer ranks masks across train/valid/test splits
```

That is deliberately simpler than the full recursive macro learner, but it
exercises the same engine boundary: domain code generates candidates and exact
labels; shared code owns training, ranking, traces, checkpoints, and summaries.

The next crossover is `bitwise_macro_step`. It emits the same 32-feature
candidate representation used by the recursive bitwise tool policy:

```text
target truth table + current chain prefix
-> enumerate allowed tool applications over existing nodes
-> oracle shortest-chain replay labels the next macro step
-> generic dataset scorer learns/ranks those step candidates
```

This adapter is intentionally a dataset boundary, not yet a promoted macro
engine. Its verifier uses multi-target labels: a candidate is valid when adding
its result reduces the remaining certified completion distance and still fits
inside the chain budget. Top-k remains a useful early signal, but top-1 is no
longer penalized for every alternate valid chain step.

The experiment registry now carries the fuller shared spec:

```text
NNDatasetExperiment {
    name;
    domain;
    feature_schema;
    verifier_schema;
    label_mode;
    generate(config, out_dataset);
}
```

With this spec, bitwise gates, tile rules, grid tasks, haiku candidates, and
Signal replay rows can all feed the same runner through `NNDataset` rows.
Domain code should only own state generation, feature encoding, candidate
enumeration, and verifier logic. The engine should own training, ranking,
residual mining, trace writing, checkpointing, matched baselines, and summary
rows.

Multi-target groups are now supported in the runner. The remaining
generalization is domain-specific credit assignment: for equivalent macro
chains, ARC programs, or haiku repairs, adapters should label all candidates
that replay to an equivalent verified outcome rather than a single arbitrary
oracle choice.

`haiku_adjacency` is the first text-domain dataset adapter. With `--corpus`, it
loads haiku lines from the existing corpus path and derives line-specific
adjacency from observed corpus transitions, not a generic word-list gate. The
embedded mini corpus remains only as a no-file fallback. Each state is a partial
haiku line; candidates are next words; verifier-positive rows are words that
satisfy corpus adjacency, syllable budget, and continuation viability. The
runner budgets generated experiments with `--max-groups` and
`--max-candidates`, and `--rollout-passes` repeats the training pass over
generated candidate groups for rollout-style refinement.

The bitwise recursive substrate sets the promotion discipline:

```text
neural proposal -> verifier replay -> certified tool -> promoted macro
```

The dataset runner is the lower-level execution discipline:

```text
candidate rows -> verifier labels -> scorer/ranker -> trace/eval row
```

Together they move the project toward experiments that are primarily datasets
and verifiers. Bespoke binaries remain useful for new domains, but they should
compile down to this shared row protocol as quickly as possible.

## Ported Smoke Adapters

The registry now has smoke-level `NNDatasetExperiment` ports for the main
experiment families:

```sh
make dataset-action-score-rest-smoke
```

That aggregate target runs:

- `grid_path`: generated 7x7 obstacle states, four move candidates, and BFS
  shortest-successor labels. It labels every tied shortest successor as valid.
- `delayed_claim`: generated claim states, four claim actions, and deterministic
  hidden-utility argmax labels.
- `prime_gap_candidate`: odd-prime states, even gap candidates, and exact
  primality labels for the first next-prime gap.
- `arc_cell_color`: the fixed ARC 0520fde7-style split-grid cell-color task,
  with one candidate row per output color.
- `program_synthesis_ops`: fixture DSL abstraction candidates with verifier
  labels for the target abstraction row.
- `signal_wasd`: Signal-style WASD candidates over synthetic counterfactual
  utility states, including multi-target ties.

These are engine ports, not removals of the production task CLIs. The legacy
binaries still carry richer domain reports, rule searches, replay modes, and
paper evidence. The dataset ports prove that each family can compile to the
shared row protocol: complete candidates, verifier-positive labels, shared
softmax scorer, common traces, checkpoints, and summary metrics.

## Shared Evidence Contract

`NNDatasetExperiment` now declares the non-dataset engine surfaces that used to
exist only inside domain CLIs:

```text
capabilities: dataset | rule_search | replay | evidence
rule_search_schema
replay_schema
evidence_schema
```

`examples/dataset_action_score.c` accepts `--evidence PATH` and appends a
`crlp.engine_evidence.v1` row after a run. The row records the domain,
feature/verifier schema, shared capabilities, rule-search and replay schema
names, train/valid/test top-k metrics, target multiplicity, model size, and
checkpoint identity. Evidence runs are reproducible by default: if `--evidence`
is provided without `--save`, the runner derives a checkpoint path beside the
evidence file from the task name, saves the model, hashes the exact checkpoint
bytes, and writes both `model_hash` and `checkpoint_path` into the evidence row.

```sh
make dataset-action-score-engine-evidence
```

This is the generic reporting layer. Domain CLIs can still emit richer local
reports, but paper and health tooling can now consume one common evidence file
for the engine-backed smoke path instead of requiring bespoke summary parsers
for every experiment.
