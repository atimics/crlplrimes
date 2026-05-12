# Verifier-Grounded Training Substrate Roadmap

crlplrimes is evolving from a compact C neural-network experiment into a small proposal engine for
verifier-grounded learning. The core loop is:

```text
symbolic state -> candidate actions -> neural scorer -> verifier outcome -> trace -> replay
```

The model proposes and ranks actions. The verifier owns correctness. The trace corpus is the durable
memory that can train successor checkpoints.

## Milestones

1. Runtime/training split
   - Keep inference, softmax, action scoring, and future checkpoint loading usable without task code.
   - Keep training, replay, examples, and benchmarks outside the inference-oriented surface.

2. Generic action scorer
   - Public `NNActionCandidate` and row-major scoring APIs.
   - Prime-gap uses the generic scoring/training names where possible.
   - Minimal non-prime action-scoring example.

3. Checkpoints
   - Deterministic binary checkpoint format with version, scalar type, architecture, parameters, and
     metadata placeholders for lineage and feature encoders.
   - Save/load round trip and deterministic-byte tests.

4. Trace emission
   - JSONL trace schema for scorer/verifier interactions.
   - Prime-gap trace mode records candidate scores, probabilities, selected/target action, and
     verifier diagnostics.

5. Prime replay
   - Train child model from emitted traces.
   - Compare parent and child checkpoints on frozen prime-gap capability and verifier diagnostics.

6. Synthetic `gossip_truth`
   - Standalone claim-resolution task with action scoring and simple symbolic baselines.
   - Emits and replays traces without Signal integration.

7. Native/WASM inference parity
   - Frozen checkpoint fixture with native and browser/WASM score/rank comparison.
   - Symbolic server/verifier remains authoritative.

## Implemented Slices

The first implementation slice promotes action scoring to a public runtime API:

```c
int nn_score_action_rows(NeuralNet *net,
                         const nn_real *candidate_inputs,
                         size_t candidate_count,
                         nn_real *scores,
                         nn_real *probabilities);

int nn_score_actions(NeuralNet *net,
                     NNActionCandidate *candidates,
                     size_t candidate_count);
```

`nn_train_action_scorer` is the proposal-oriented alias for the existing shared softmax scorer
training path.

The second slice adds canonical model checkpoints:

```c
int nn_save_checkpoint(const NeuralNet *net, const char *path, const NNCheckpointMeta *meta);
NeuralNet *nn_load_checkpoint(const char *path, NNCheckpointMeta *out_meta);
int nn_hash_checkpoint_file(const char *path, uint8_t out_hash[32]);
```

The initial checkpoint format is binary, deterministic, and dependency-free. It includes model
architecture, scalar type, activations, loss, parameters, and fixed metadata placeholders for parent
model identity, trace roots, view roots, normalization, feature encoder version, and benchmark
report identity. The hash function computes SHA-256 over the saved checkpoint bytes.

The third slice adds trace JSONL emission:

```c
int nn_trace_write_jsonl(FILE *file, const NNTraceRecord *record);
```

Prime-gap can now run with `--trace path.jsonl`. The CLI writes one trace row per evaluated held-out
prime state after supervised training, plus a companion `path.jsonl.nnckpt` checkpoint. Each trace
row carries the companion checkpoint SHA-256 as `model_hash`, candidate action ids, scores,
probabilities, the selected top proposal, the verified target action, per-row loss, verifier result,
and prime-specific verifier-cost details.
