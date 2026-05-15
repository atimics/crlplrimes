# CRLPLRIMES: Certificate-Gated Neural Proposers for Verifier-Grounded Action Scoring

This repository is a compact C neural-network library and an exact-verifier research harness for **certified neural-symbolic action scoring**. The project is not a prime-gap solver. It is one architecture instantiated on three verifier-backed action tasks:

| Task | State | Actions | Exact verifier | Gate grammar | Certificate |
| --- | --- | --- | --- | --- | --- |
| `prime_gap` | prime `p` | candidate gaps | segmented sieve next-prime verifier | modular rejection predicates | algebraic divisibility proof |
| `grid_path` | grid, start, goal | four moves | BFS shortest-path successor | legality and admissible-pruning predicates | cannot reject an optimal move |
| `delayed_claim` | observable claim features | trust/ignore/request/cautious | deterministic hidden-outcome simulator | threshold/dominance/evidence predicates | finite-generator safety check |

Every experiment now publishes the same machine-readable task contract with:

```sh
make experiment-schemas
```

The schema is `crlp.experiment_schema.v1` and defines the state distribution, candidate enumerator, observable features, leakage policy, scorer, verifier, rule grammar, certificate checker, proposer inputs, baselines, trace protocol, replay protocol, and verifier-RL protocol. The public C contract lives in `include/nn_experiment.h`.
The working design note is [docs/experiment-schema.md](docs/experiment-schema.md).

## Generalized Experiment Schema

Each task is specified as:

```text
Task T = {
  state distribution S_T,
  candidate enumerator A_T(s),
  observable state/action features x_T(s, a),
  neural residual scorer f_theta(x_T),
  exact verifier V_T(s, a),
  symbolic rule grammar G_T,
  certificate checker C_T(rule),
  rule proposer q_phi(rule | structural(rule)),
  gate executor R_T(rule, s, a),
  trace/replay/RL protocol
}
```

The reusable architecture is:

```text
state ──▶ rule proposer ─▶ rule candidates ─▶ certificate verifier ─▶ accepted rules
                                                                        │
state ──▶ candidate enumerator ─────────────────────────────────────────┴─▶ gated candidates
                                                                                │
                                                              learned scorer ──▶ softmax ranking
                                                                                │
                                                              exact verifier ──▶ correct action / cost
                                                                                │
                                                                       JSONL trace ─▶ replay
```

Design invariant:

```text
The model proposes. The verifier decides. The trace remembers.
```

The task-independent rules are:

1. Candidate enumeration is complete before gating; the verifier is not allowed to prune the candidate set silently.
2. The scorer receives only observable state/action features. Verifier-derived facts can be labels, rewards, certificates, or executed gates, but not default scorer inputs.
3. The proposer receives structural rule descriptors only. Empirical support, utility, and false-rejection counts are targets or diagnostics, not features.
4. A rule can enter the gate only through the task's certificate checker.
5. Every task is evaluated against neural, random, grammar-order, structural-heuristic, and oracle proposer orderings at matched budgets.
6. JSONL traces use stable action ids so replay cannot silently change semantics when candidates are reordered.

## Current Evidence

The canonical result is still strongest on `prime_gap`, but it is now presented as one task row rather than the project goal. With residue-sieve features removed from the scorer input, the raw prime scorer reaches **9.13%** top-1; a fixed hand gate reaches **45.12%**; and a certified learned-rule gate reaches **68.31%** at 100% top-10. On `grid_path`, a budgeted certified legality gate improves parent accuracy from **75.80%** to **89.60%**. On `delayed_claim`, the current load-bearing result is trace replay and calibration, not gate learning: the replay successor reaches **94.80%** accuracy versus **93.50%** for the trained parent.

The negative result is equally important: the current prime rule proposer is not a neural success story. At a 16-rule budget, grammar order and a simple structural heuristic save **73.22%** of exact work, versus **67.03%** for the neural proposer and **46.30%** for one random ordering. The next neurosymbolic benchmark must make proposer learning non-trivial across all three task schemas.

## Protocol Axes and Current Results

### Axis 1 — Certified gates reduce the residual action space

Each task should report the same gate ablation: no gate, hand/spec gate, learned certified gate, and matched-budget baselines. The prime-gap task currently has the complete three-gate table:

| Gate | Top-1 | Top-10 | NLL |
| --- | ---: | ---: | ---: |
| None (raw candidate space) | 9.13% | 78.01% | 2.9994 |
| Hand-wheel (small-prime divisibility) | 45.12% | 99.77% | 1.5902 |
| **Learned rule program** (certificate-checked rules) | **68.31%** | **100.00%** | **1.0172** |

The same protocol is partially implemented on `grid_path`: raw parent accuracy is **75.80%** and budgeted legality-gate accuracy is **89.60%**. `delayed_claim` still needs its certified gate grammar implemented; until then it is a trace/replay and verifier-RL task, not evidence for gate learning. The former raw 45.12% prime figure is retained only as a full-sieve-feature leakage ablation.

### Axis 2 — Rule search must beat fixed-budget symbolic baselines

`prime_rule_search` runs the canonical neural proposer over unconditional candidate filters `reject candidate n when n > m and n % m == r` for `m=3..255`. The expanded grammar can also express conditional gap filters (`if p % m == a then reject gap % m == b`) under `prime-rule-search-conditional`, but that broader mode is not the load-bearing result. The certificate verifier only accepts rules with a deterministic soundness proof, so an accepted rule provably cannot reject the true next-prime gap.

Rule-search run from `results/latest/comparison.jsonl`:

```text
proposed:        32,637
verified:         4,692   ← rules that survive verifier screens
accepted:            52   ← all unconditional candidate filters
false rejections:     0
direct exact:      7.20   ← average exact-verifier work without rules
expanded exact:    1.47   ← average exact-verifier work after accepted rules
saved:            79.64%
```

The proposer is small (`crlp.rule_trace.v1` schema, scoring rule candidates with the same shared-scorer abstraction as gap actions). Its input feature set is structural-only: condition kind, moduli, residues, zero-residue indicators, gcd ratios, and derived candidate residue. It does not receive `train_rejects`, validation support, utility, or false-rejection counters as features; those counts are labels and verifier diagnostics. The discipline of the result is in the verifier, not the model: a rule is accepted only if a deterministic algebraic check certifies soundness, so zero false rejections is a property of the certificate, not of training.

Matched-budget proposer baselines now run in `make prime-rule-search-baselines`. At a 16-rule budget over the same grammar, split, and certificate verifier:

| Rule ordering | Exact probes | Work saved |
| --- | ---: | ---: |
| Neural structural proposer | 2.37 | 67.03% |
| Random ordering, seed 20240511 | 3.86 | 46.30% |
| Grammar order | 1.93 | 73.22% |
| Structural heuristic | 1.93 | 73.22% |
| Validation-count oracle | 1.93 | 73.22% |

**What this evidence does and does not show.** It shows that certificate-gated structural rule discovery is sound and useful on one task. It also shows the current neural proposer is not the reason this rule set works: at the fixed 16-rule budget, simple deterministic orderings beat it. The stronger neurosymbolic claim now has a concrete target: implement the same proposer-baseline grid for `grid_path` and `delayed_claim`, then learn a proposer that beats grammar-order and structural-heuristic baselines under the same grammar and budget.

### Axis 3 — Verifier RL must preserve calibration across tasks

Strict verifier policy gradient on sparse exact rewards is a known-bad recipe, and we observe the failure mode cleanly. Same supervised checkpoint, two RL branches, default split:

| Policy | Temp | Top-1 | Top-10 | NLL | ECE | Top-1 confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Supervised | 1.00 | 9.13% | 78.01% | 2.9170 | 0.0704 | 0.1617 |
| Strict verifier RL | 1.00 | 9.13% | 78.01% | 3.0611 | 0.1271 | 0.2184 |
| Strict verifier RL, temp-scaled | 1.50 | 9.13% | 78.01% | 2.9063 | 0.0613 | 0.1526 |
| KL/entropy-regularized verifier RL | 1.00 | 9.13% | 78.01% | 2.9056 | 0.0604 | 0.1517 |
| KL/entropy-regularized RL, temp-scaled | 1.00 | 9.13% | 78.01% | 2.9056 | 0.0604 | 0.1517 |

Three observations:

- **Ranking is preserved across all variants.** Strict RL does not improve top-1 — it is mostly logit inflation.
- **Strict RL degrades calibration.** Unscaled ECE rises from 0.0704 to 0.1271, and post-hoc temperature scaling needs `T=1.5` to undo it. The RL branch has effectively learned to be overconfident.
- **KL-to-prior + entropy fixes it at training time.** Unscaled ECE drops to 0.0604; the selected temperature stays at `T=1.0`, meaning the policy is already close to calibrated without rescaling. This is the standard RLHF objective applied to exact-verifier rewards, and it works here for the same reason it works there: the reference policy supplies the bulk of the probability mass and the gradient is bounded.

The finding as stated holds for prime-gap. Whether the same logit-inflation failure mode appears under verifier RL on `grid_path` and `delayed_claim`, and whether the same KL/entropy regularizer fixes it, is on the open-work list — the pipeline supports running it, but the comparison is not in the canonical results yet.

## Task Instantiation: Prime Gap

Prime gaps are used as the load-bearing benchmark because the domain has properties most exact-verifier sandboxes lack:

- **A fast deterministic verifier at all scales** (segmented sieves, wheel factorization, block proofs).
- **A compact totally-ordered candidate space** (gaps `2..256` plus an overflow action).
- **An exhausted classical literature** — there is no plausible claim that a small neural net beats a sieve, which sharpens the question to "what does the neural component contribute under perfect feedback?"
- **Rich algebraic structure** the rule grammar can exploit (residue classes, divisibility, certificates).

We are **not** claiming improved prime generation. Exact next-prime verification under any of these pipelines remains the classical algorithm's job, because a correct proof requires ruling out earlier candidates and a wrong neural proposal creates extra proof obligations. The framing is:

```text
Primes are the substrate. The contribution is the protocol: certificate-gated
hybrid scoring, learned symbolic gates, and calibration-preserving verifier RL.
```

The default training split is:

```text
train       p < 800,000
validation  800,000 <= p < 1,000,000
test        1,000,000 <= p <= 3,000,000
reported metrics use 30,000 held-out test states
```

Detailed baselines:

| Method | Temp | Top-1 | Top-10 | NLL | ECE | Top-1 confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Empirical gap baseline | 1.00 | 15.46% | 78.71% | 2.8585 | 0.0196 | 0.1743 |
| Residue-conditioned frequency | 1.00 | 40.11% | 93.04% | 2.8583 | 0.2769 | 0.1242 |
| Linear minimal scorer | 1.00 | 9.13% | 78.01% | 2.9119 | 0.0289 | 0.1202 |
| Linear minimal, temp-scaled | 0.85 | 9.13% | 78.01% | 2.9005 | 0.0486 | 0.1399 |
| Supervised minimal scorer | 1.00 | 9.13% | 78.01% | 2.9994 | 0.1076 | 0.1989 |
| Supervised minimal, temp-scaled | 1.20 | 9.13% | 78.01% | 2.8991 | 0.0459 | 0.1372 |
| Full-sieve-feature ablation (leakage) | 1.00 | 45.12% | 99.77% | 1.5778 | 0.0743 | 0.4882 |
| Supervised minimal + hand-wheel | 1.00 | 45.12% | 99.77% | 1.5902 | 0.0834 | 0.5075 |
| **Supervised minimal + learned-rule gate** | **1.00** | **68.31%** | **100.00%** | **1.0172** | 0.1379 | 0.6254 |
| Learned-rule survivor replay | 1.00 | 68.31% | 100.00% | 1.0307 | 0.1306 | 0.6689 |
| Strict verifier RL | 1.00 | 9.13% | 78.01% | 3.0611 | 0.1271 | 0.2184 |
| KL/entropy-regularized verifier RL | 1.00 | 9.13% | 78.01% | 2.9056 | 0.0604 | 0.1517 |

The interpretation is now sharper. Among raw-action-space scorers without residue leakage, the hidden network and linear scorer both collapse to gap-frequency behavior. The 45.12% full-sieve row is useful only as an upper-bound ablation showing how much signal the old residue flags handed to the model. The dominating effect is not depth of the scorer; it is the verified symbolic gate.

## Task Instantiations: Delayed Claim and Grid Path

`delayed_claim` is a delayed-label synthetic task with six state features and four actions (`trust`, `ignore`, `request_more_evidence`, `act_cautiously`). A deterministic hidden rule resolves the outcome and defines the best action. The task validates trace replay with parent-hash lineage:

| Model | Accuracy | NLL | ECE | Decision utility | Steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Trained parent | 93.50% | 0.1580 | 0.0363 | 1.0432 | 1.065 |
| Trace-replay successor | 94.80% | 0.1188 | 0.0166 | 1.0704 | 1.052 |

`grid_path` is a shortest-path task with an exact BFS verifier owning correctness. Candidate actions are all four moves, including out-of-bounds and blocked moves. The neural model sees start/goal coordinates, candidate displacement, the obstacle map, and an action one-hot; it no longer receives explicit `out_of_bounds` or `blocked_next` legality flags. BFS supplies labels and verification cost, not a pruned candidate set.

A small structural rule proposer now runs on this task too. Its grammar is the certified legality gate: reject edge moves that leave the grid, and reject moves into blocked cells. The proposer scores five certified rules from structural features, selection uses train+validation only, and a fixed budget accepts the top three before test evaluation. This is intentionally not a pathfinding solver; it is a cross-domain sanity check that certificate-gated pruning can be represented and evaluated outside prime residues.

| Model / mode | Accuracy | NLL | ECE | Candidates | Decision utility | Steps | Nodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trained parent, raw | 75.80% | 0.7154 | 0.0712 | 4.00 | 0.6975 | 1.327 | 21.46 |
| Trained parent, budgeted legality gate | 89.60% | 0.3732 | 0.1033 | 3.13 | 0.8700 | 1.116 | 21.25 |
| Trace-replay successor, raw | 74.00% | 0.7274 | 0.0602 | 4.00 | 0.6750 | 1.335 | 21.75 |
| Trace-replay successor, budgeted legality gate | 87.60% | 0.3851 | 0.0928 | 3.13 | 0.8450 | 1.137 | 21.55 |

The grid legality proposer accepts 3/5 certified rules under budget with 0 held-out false rejections, rejecting 0.87 impossible moves per held-out state on average. The raw scorer now has to infer legality from coordinates and the map; the useful signal is that a verifier-certified symbolic gate transfers outside prime residues, not that legality pruning solves shortest-path ranking.

## Limitations and Next Work

- **Proposer learning is not solved.** The fixed-budget baseline now shows grammar order and a simple structural heuristic outperform the current neural proposer at a 16-rule budget. Axis 2 should be read as a benchmark requirement, not as evidence that the neural proposer is essential. The next proposer experiment needs richer grammars across all three tasks where ordering is non-trivial and random/grammar/heuristic baselines are hard to beat.
- **Cross-task verifier-RL comparison.** Axis 3 (strict-vs-regularized RL on the prime-gap task) needs a matching run on `grid_path` and `delayed_claim` before it can be stated as a general property of sparse-exact-reward policy gradient. The pipeline supports it; the comparison is not in the canonical results yet.
- **Richer non-prime gate grammars.** `grid_path` now has a budgeted certified legality gate, but that gate is intentionally shallow. The next cross-domain test should learn admissible dead-end or distance-bound pruning rules that reduce BFS verifier work without becoming a hand-coded solver.
- **Statistical rigor.** The default `make prime-gap-experiments` config currently runs 1 seed; the harness supports more (`PRIME_GAP_EXPERIMENT_SEEDS=...`). The minimal-vs-leakage, strict-vs-regularized, and hand-wheel-vs-learned-rule deltas should be reported with confidence intervals across >=5 seeds.
- **Reliability diagrams.** The comparison schema stores `reliability_bins`; the README currently reports only ECE. Calibration claims would be stronger with a small reliability plot.
- **Learned-rule calibration.** The learned rule program improves ranking sharply, but its ECE is worse than the raw minimal and hand-wheel rows in the default run. That calibration tradeoff deserves a focused reliability-diagram pass.

## Reproducibility Substrate

The hybrid architecture is the contribution; the substrate is the discipline that makes it auditable. Three pieces matter:

- **Stable action IDs and trace contract.** Trace rows carry `schema: "crlp.trace.v1"`, `trace_version: 1`, and `action_id_contract: "stable_unique_action_ids"`. Replay resolves `target_index` to the trace row's `action_id`, then maps that stable id back into the regenerated candidate row. Reordering action ids cannot silently change replay semantics; the prime replay regression test exercises that.
- **Checkpoint identity.** Checkpoints carry a `parent_model_hash`, a `training_trace_root`, a `feature_encoder_version`, and a `feature_set` in metadata; the model hash is SHA-256 over the exact checkpoint bytes. Successor models declare their parent. Comparison reports (`crlp.compare.v1`) carry `model_hash`, `parent_hash`, `feature_set`, and a machine-readable `reliability_bins` array.
- **Pure C, no third-party deps.** The neural library is dense layers, sigmoid/ReLU/tanh/linear, MSE and softmax cross-entropy, SGD, batched 4-by-4 tiled forward/backward, cloned networks, action scoring, pairwise ranking loss, cached policy-gradient updates with optional KL/entropy regularization, deterministic checkpoint save/load, SHA-256 hashing, JSONL trace writing. The core is ~3k lines; the prime-gap task is split into focused modules under `src/prime_gap/`.

End-to-end quick path:

```sh
make
make test
make experiment-schemas
make prime-gap
make prime-gap-trace
make prime-gap-replay
make prime-gap-eval
make prime-rule-search
make delayed-claim
make grid-path
make results
```

`prime-gap-trace` writes `build/float/prime_v1.nnckpt` and `build/float/traces/prime_gap.jsonl`. `prime-gap-replay` trains `build/float/prime_v2.nnckpt` from that trace. `prime-gap-eval` compares both checkpoints on the same frozen benchmark. `prime-rule-search` writes `build/float/prime_rules_v2.crlprule` and `build/float/traces/prime_rule_search.jsonl`. `make results` writes `results/latest/summary.md`, `results/latest/comparison.jsonl`, and `results/latest/commands.txt`; bulky logs stay under ignored `results/**/logs/`.

## C Neural Network Library

Public headers split conceptually:

```text
include/nn_core.h             allocation, inspection, forward pass, softmax, action scoring, checkpoints, traces
include/nn_train.h            sample training, batch training, action-scorer training, cached policy-gradient with KL/entropy reg
include/nn_experiment.h       shared experiment-schema contract for verifier-backed action tasks
include/nn_tasks_prime_gap.h  command entry points for the prime-gap task module
include/nn.h                  compatibility umbrella for existing users
```

The prime-gap implementation is split across `src/prime_gap/`: data generation, feature encoding, symbolic-wheel discovery, expanded rule search, block-verifier accounting, evaluation, training, run orchestration, and replay/compare entry points each have their own module. Shared private declarations live in `src/prime_gap/internal.h`. `src/nn_tasks_prime_gap.c` remains as a test-only module index so private helpers can be exercised without exporting them from the public task header. The files in `examples/prime_gap.c`, `examples/prime_replay.c`, `examples/prime_compare.c`, and `examples/prime_rule_search.c` are thin command wrappers.

## Build

```sh
make
```

The default scalar type is `float`. Build with double precision when needed:

```sh
make REAL=double
```

## Run the XOR Example

```sh
make run
```

Expected output should converge toward:

```text
0 xor 0 -> close to 0
0 xor 1 -> close to 1
1 xor 0 -> close to 1
1 xor 1 -> close to 0
```

The example also prints basic performance metrics:

```text
parameters
training time
training throughput
final loss
accuracy
inference latency
inference throughput
```

## Run the Standalone Action-Scoring Example

```sh
make action-score
```

This example trains a tiny shared scorer over three candidate rows and prints deterministic ranked actions with scores and probabilities. It is independent of the prime-gap experiment.

## Run Benchmarks

```sh
make benchmark
```

The benchmark uses larger synthetic MLP shapes and reports:

```text
parameter count
estimated dense forward FLOPs
forward latency
forward throughput
estimated forward GFLOP/s
mini-batch training throughput
sample-wise SGD baseline
batched training speedup
estimated dense training GFLOP/s
```

The sample-SGD and batch-SGD lines measure throughput for different update schedules, so their losses are not expected to match after the same epoch count.

The current batch path uses internal 4-by-4 tiles for forward propagation, hidden-delta backpropagation, and gradient accumulation, with scalar fallback code for remainders and small dimensions.

Gradient buffers are allocated lazily the first time mini-batch training runs, so inference-only use and sample-wise training avoid that extra model memory.

## Running the Prime-Gap Experiments

The headline pipeline is `make prime-gap` followed by `make prime-rule-search`.

```sh
make prime-gap
```

Builds a sieve-backed prime-gap dataset and trains a shared candidate scorer:

```text
input  = [log(p), normalized_gap, overflow_flag]
score  = f(input)
policy = softmax(scores across candidate gaps 2..256 plus an overflow action)
```

The default `minimal` feature set intentionally omits precomputed residue divisibility flags. Use `--feature-set full` or `--full-features` only for the leakage/upper-bound ablation where the scorer sees the small-prime sieve flags directly. After a learned rule program has already pruned composites, those sieve flags are mostly dead on the survivor set, so they are diagnostic rather than a clean source of ranking signal. The run evaluates empirical, residue-conditioned, and linear minimal baselines, runs supervised feature ablations (log-only, gap-only, residue-only, full-sieve leakage), fits a scalar softmax temperature on a held-out validation slice, and compares strict and KL/entropy-regularized verifier-RL branches cloned from the same supervised checkpoint.

```sh
make prime-rule-search
```

Runs the prime-gap rule-grammar instantiation of Axis 2. The default neural proposer scores unconditional candidate filters of the form `reject candidate n when n > m and n % m == r` over `m=3..255`; `make prime-rule-search-baselines` reruns the same grammar with neural, random, grammar-order, structural-heuristic, and validation-oracle orderings at a fixed 16-rule budget; `make prime-rule-search-conditional` runs the broader conditional gap grammar. The neural proposer's feature set is structural-only and excludes support counts, utility, and false-rejection counters. The verifier accepts only rules with a deterministic algebraic certificate. The JSONL trace uses schema `crlp.rule_trace.v1` and records grammar version, selection strategy, proposer feature set, rule text, model score, verifier result, deterministic-certificate flag, accepted flag, support counts, and false-rejection counts.

CLI output of `make prime-gap` is split into sections:

```text
capability:    top-k accuracy, NLL, true-action probability, mean rank, MRR, gap error
calibration:   top-1 confidence, correct/incorrect confidence, ECE, high-confidence accuracy
prime buckets: capability metrics over four increasing held-out prime ranges
verifier:      exact classical work needed to verify or correct proposals
policy:        supervised, strict RL, regularized RL, and temperature-scaled rows
```

`proposal_steps` measures how quickly the true action appears in the model's proposal order. `wheel_exact` measures how many exact sieve primality probes remain after small-prime divisibility filters eliminate obvious composites. `assisted_wheel` tries the model's top-10 proposals before falling back to wheel scan. `block_cost` estimates segmented interval-proof work, and `assisted_block` tries model proposals before falling back to the block scan.

Larger sweeps:

```sh
make prime-gap-sweep
make prime-gap-regularization-sweep
make prime-gap-experiments
```

The experiment harness runs supervised epoch sweeps and KL/entropy regularization sweeps and writes parsed CSV plus a Markdown summary under `build/float/prime_gap_experiments/`. Curated result snapshots that should survive `make clean` live under `results/`. Override the default one-seed quick run with:

```sh
make prime-gap-experiments PRIME_GAP_EXPERIMENT_SEEDS=20240511,20240512,20240513
```

Optional arguments:

```sh
./build/float/prime_gap [train_limit] [test_limit] [supervised_epochs] [rl_epochs] [max_train_states] [max_test_states] [regularized_reference_weight] [regularized_entropy_weight] [rng_seed] [--feature-set minimal|full|log-only|gap-only|residue-only]
```

## Run Tests

```sh
make test
```

Run sanitizer coverage with:

```sh
make sanitize
```

Run the CI-style local check with:

```sh
make check
```

This runs the unit tests, sanitizer tests, double-precision tests, the standalone action-scoring example, delayed-claim trace/replay/eval, the expanded prime-rule-search smoke path, and a small prime-gap trace/replay/eval smoke path.

The tests compare scalar and batched forward results, verify `batch_size=1` training against sample-wise training, finite-difference check batched gradients through the tiled kernels, and check the softmax cross-entropy and shared softmax-scorer gradients.

## API

```c
const size_t layers[] = {2, 4, 1};
NeuralNet *net = nn_create(layers, 3, 42);

size_t parameters = nn_parameter_count(net);
size_t forward_flops = nn_forward_flop_count(net);

nn_real input[] = {1.0, 0.0};
nn_real target[] = {1.0};

nn_train_sample(net, input, target, 0.7);

nn_real output[1];
nn_forward(net, input, output);

nn_free(net);
```

Use `nn_create_config` for non-default activations or softmax cross-entropy:

```c
const size_t layers[] = {12, 32, 16, 1};
NNConfig config = {
    .layer_sizes = layers,
    .layer_count = 4,
    .seed = 42,
    .hidden_activation = NN_ACTIVATION_RELU,
    .output_activation = NN_ACTIVATION_LINEAR,
    .loss = NN_LOSS_MSE,
};
NeuralNet *scorer = nn_create_config(&config);
NeuralNet *copy = nn_clone(scorer);
```

With a linear output layer, `nn_forward` returns logits; call `nn_softmax` when probabilities are needed.

For a shared scalar scorer over candidate actions, train one state at a time with softmax cross-entropy across the candidate rows:

```c
nn_real scores[16];
nn_real probabilities[16];
size_t ranked[16];

nn_score_action_rows(scorer, candidate_inputs, candidate_count, scores, probabilities);
nn_rank_action_scores(scores, candidate_count, ranked);

nn_real loss = nn_train_action_scorer(scorer,
                                      candidate_inputs,
                                      candidate_count,
                                      true_candidate_index,
                                      0.005f);
```

`candidate_inputs` is row-major: `candidate_count * nn_input_size(scorer)` values. The scorer must have one scalar output. Probabilities sum to approximately `1.0`, and ranking is deterministic for equal scores.

Verifier-guided or policy-gradient experiments can reuse a cached candidate forward pass and add KL/entropy regularization against a reference policy:

```c
nn_forward_batch(scorer, candidate_inputs, scores, candidate_count);
nn_real objective = nn_train_softmax_scorer_policy_regularized_cached(
    scorer,
    candidate_count,
    sampled_index,
    advantage,
    reference_probabilities,
    1.20f,
    0.12f,
    0.001f);
```

Training a dataset uses contiguous row-major buffers:

```c
nn_real final_loss = nn_train(net, inputs, targets, sample_count, 0.7, 1000, 1);
```

Mini-batch training uses reusable batch activations, batch deltas, and gradient buffers inside the network:

```c
nn_real final_loss = nn_train_batched(net, inputs, targets, sample_count, 32, 0.05, 10, 1);
```

`inputs` must contain `sample_count * nn_input_size(net)` values. `targets` must contain `sample_count * nn_output_size(net)` values.

## Checkpoints

Checkpoints use a small canonical binary format with:

```text
magic
checkpoint_version
scalar_type
layer_count
layer_sizes
hidden_activation
output_activation
loss
weights
biases
metadata block length
metadata block
```

Metadata stores `parent_model_hash`, `training_trace_root`, `feature_encoder_version`, `feature_set`, `benchmark_report_hash`, and `notes`.

```c
NNCheckpointMeta meta = {0};
meta.checkpoint_version = 1;
meta.feature_encoder_version = 1;

nn_save_checkpoint(net, "model.nnckpt", &meta);

char hash[65];
nn_hash_checkpoint_hex("model.nnckpt", hash);

NNCheckpointMeta loaded_meta;
NeuralNet *loaded = nn_load_checkpoint("model.nnckpt", &loaded_meta);
```

The hash is SHA-256 over the exact checkpoint bytes, so changing weights or metadata changes model identity.

## Trace Schema

Trace rows are JSONL records with schema `crlp.trace.v1`:

```json
{"schema":"crlp.trace.v1","trace_version":1,"action_id_contract":"stable_unique_action_ids","task":"prime_gap","state_hash":"0000000000000000","model_hash":"unknown","feature_encoder_version":1,"candidate_count":0,"action_ids":[],"scores":[],"probabilities":[],"selected_index":0,"target_index":0,"loss":0.0,"verifier_result":"correct","verifier_cost":{},"seed":0,"feature_set":"minimal"}
```

Prime-gap traces add `p`, `true_gap`, `proposal_steps`, `wheel_exact`, `assisted_wheel`, `block_cost`, `assisted_block`, and `log_scale`. Rule-search traces use schema `crlp.rule_trace.v1`. Delayed-claim traces use the deterministic per-row seed for replay.

Use `nn_trace_parse_jsonl` to validate rows before replay. It checks the schema string, trace version, action-id contract, required scalar fields, unique action ids, score/probability array lengths, selected/target index bounds, and probability mass. Replay code resolves `target_index` to the row's stable `action_id`, then maps that action id back into the regenerated candidate rows before training. `nn_trace_json_get_u64`, `nn_trace_json_get_real`, and `nn_trace_json_get_string` are small helpers for task-specific extra fields.

Comparison reports are JSONL records with schema `crlp.compare.v1`, `report_version`, `model_hash`, `parent_hash`, `feature_set`, `feature_encoder_version`, benchmark seed, sample count, capability metrics, calibration metrics, machine-readable `reliability_bins`, and any task-specific verifier or utility fields.

Deterministic fixtures live in `tests/fixtures/` and cover a tiny action scorer, expected scores, expected probabilities, expected ranking, and an example trace row. The tests exercise those fixtures in both `float` and `double` builds.

## Delayed-Claim Task

`delayed_claim` is a small delayed-label synthetic task with six state features:

```text
source_reliability
evidence_strength
contradiction_count
claim_age
local_history_score
noise_level
```

The model scores four actions: `trust`, `ignore`, `request_more_evidence`, and `act_cautiously`. A deterministic hidden rule resolves the later outcome and defines the best action. The task reports accuracy, NLL, calibration error, and decision utility.

```sh
make delayed-claim
make delayed-claim-trace
make delayed-claim-replay
make delayed-claim-eval
```

## Grid-Path Task

`grid_path` is a second exact-verifier task. A deterministic generator creates a small obstructed grid, start, and goal. Candidate actions are all four moves. The neural model ranks moves from observable map and move features, while an exact BFS verifier owns correctness by identifying the shortest-path successor. `grid_path --rule-search` runs the cross-task budgeted legality-gate experiment: a structural proposer ranks five certified rules that reject out-of-bounds and blocked-cell moves, then accepts the top three under a fixed rule budget.

The task reports accuracy, NLL, calibration error, decision utility, proposal steps, fixed-order baseline steps, exact verifier nodes, baseline verifier nodes, and candidate counts before/after the legality gate. Trace replay stores stable move action IDs, then maps them back into the regenerated four-row candidate set before training a successor model.

```sh
make grid-path
make grid-path-rule-search
make grid-path-trace
make grid-path-replay
make grid-path-eval
```
