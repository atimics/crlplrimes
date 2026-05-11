# C Neural Network

A small feed-forward neural network in C. It implements dense layers, configurable sigmoid/ReLU/tanh/linear activations, mean-squared-error and softmax cross-entropy losses, stochastic gradient descent, layer-wise mini-batch training, tiled batch kernels, and simple CPU benchmarks.

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

## Prime-Gap Capability Experiment

This repository also includes a verifier-guided learning experiment over prime gaps. The goal is not to build a faster prime solver. The deterministic sieve/verifier is deliberately the oracle: it generates labels, supplies exact rewards, and gives us a known classical baseline so we can measure exactly how inefficient or capable the neural policy is.

The question is:

```text
Can a small C neural network learn a useful proposal distribution over next-prime gap vectors?
```

The success metrics are capability metrics first:

```text
top-1/top-5/top-10 gap-action accuracy
rank of the true gap
negative log likelihood / probability assigned to the true gap
mean reciprocal rank
calibration error and high-confidence accuracy
generalization by prime-size bucket
supervised warm start vs verifier-guided RL
```

Verifier work metrics are diagnostic, not the headline. They tell us how much classical computation the exact oracle is doing and prevent us from pretending that a learned proposal has replaced mathematical verification.

```sh
make prime-gap
```

This builds a sieve-backed prime-gap dataset and trains a shared candidate scorer:

```text
input = [log(p), normalized_gap, overflow_flag, divisibility flags for p + gap modulo small primes]
score = f(input)
policy = softmax(scores across candidate gaps 2..256 plus an overflow action)
```

The example first evaluates empirical gap-frequency, hand-coded wheel-frequency, and residue-conditioned frequency baselines, then runs supervised scorer comparisons:

```text
linear full-feature scorer
log-only
gap-only
candidate residue-only
full feature set
```

The full model trains with supervised softmax cross-entropy over all candidate actions, then runs a strict verifier-guided policy-gradient pass where reward is `1` only when the sampled action matches the exact next gap or the overflow class.

It also evaluates calibration interventions:

```text
temperature scaling selected on a chronological calibration split
entropy-regularized supervised training
raw supervised vs raw strict-RL calibration
temperature-scaled supervised vs temperature-scaled strict-RL calibration
```

The train range is split chronologically into fit and calibration subsets before evaluation on the held-out prime range. The example also includes a wheel-survivor task variant that ranks only candidate gaps not eliminated by the small-prime feature wheel; this makes the "learned proposal" question stricter by comparing against a cheap non-neural sieve prior.

It also includes a fixed-window interval task:

```text
target = uniform distribution over every prime p + g for even gaps 2..256
```

This is deliberately not next-gap prediction. It asks whether the scorer can rank all prime candidates in a local interval, with interval-frequency and interval-wheel-frequency baselines reported beside linear and MLP scorers. The same interval task is also run with raw normalized residue features instead of explicit divisibility flags, so we can test whether the network learns a sieve-like rule rather than receiving one directly. A wider raw-residue sweep repeats that diagnostic for windows `512`, `1024`, and `2048`; these wider runs cap train/test states and epochs so the default CLI remains practical.

The CLI output is split into human-readable sections:

```text
capability: top-k accuracy, NLL, true-action probability, mean rank, MRR, gap error
calibration: top-1 confidence, correct/incorrect confidence, ECE, high-confidence accuracy, binned calibration curve
prime buckets: capability metrics over four increasing held-out prime ranges
verifier diagnostics: exact classical work needed to verify or correct proposals
```

`proposal_steps` measures how quickly the true action appears in the model's proposal order. `wheel exact` measures how many exact sieve primality probes remain after small-prime divisibility filters eliminate obvious composites. `assisted_wheel` tries the model's top-10 proposals before falling back to wheel scan. `block cost` estimates segmented interval-proof work by marking composites with base primes up to `sqrt(p + g)`, and `assisted_block` tries model proposals before falling back to the block scan. These verifier metrics are reported separately because an exact next-prime verifier still has to rule out earlier candidates.

The prime-gap code is split so the experiment can keep growing without turning the CLI into the architecture:

```text
include/prime_gap.h        public experiment types and helper APIs
src/prime_gap_data.c       sieve, datasets, constants, and utility routines
src/prime_gap_features.c   gap/action features, ranking, and softmax helpers
src/prime_gap_verifier.c   exact verifier accounting and segmented scan diagnostics
src/prime_gap_eval.c       baselines, metrics, calibration, and interval-target caching
src/prime_gap_train.c      supervised, interval, verifier-RL, and ablation training loops
examples/prime_gap.c       CLI orchestration for the current experiment suite
```

Interval-prime evaluation caches verifier-generated target distributions for each sampled test window. This keeps repeated baseline/model evaluations from recomputing the same exact `is_prime` labels while preserving the verifier as the source of truth.

Current interpretation: the network learns a strong gap-ranking prior, but much of the easy top-k lift comes from small-prime divisibility structure. The wheel-frequency and linear-scorer baselines are intentionally strong checks on that. The interval-prime task shows the same pattern: the MLP improves probability quality, but top-rank behavior is still largely explained by the wheel prior. The raw-residue interval variant removes the direct composite flags while keeping compact modular information; on the default run the raw-residue MLP recovers most of the wheel behavior. In the wider sweep, the MLP keeps stronger MAP, total prime probability, and wheel-surviving top choices than the raw linear model, though top-1 prime accuracy narrows at larger windows. Exact next-prime verification remains the classical algorithm's job. That is expected and useful: the experiment is about learned proposal capability under perfect verification, not replacing the verifier.

Run a larger split sweep with:

```sh
make prime-gap-sweep
```

Optional arguments:

```sh
./build/float/prime_gap [train_limit] [test_limit] [supervised_epochs] [rl_epochs] [max_train_states] [max_test_states]
```

## Run Tests

```sh
make test
```

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
```

With a linear output layer, `nn_forward` returns logits; call `nn_softmax` when probabilities are needed.

For a shared scalar scorer over candidate actions, train one state at a time with softmax cross-entropy across the candidate rows:

```c
nn_real loss = nn_train_softmax_scorer(scorer,
                                       candidate_inputs,
                                       candidate_count,
                                       true_candidate_index,
                                       0.005f);
```

Use entropy-regularized scorer training when you want to penalize overconfident distributions:

```c
nn_real loss = nn_train_softmax_scorer_entropy(scorer,
                                               candidate_inputs,
                                               candidate_count,
                                               true_candidate_index,
                                               0.10f,
                                               0.005f);
```

Use target-distribution scorer training when more than one candidate should be treated as correct:

```c
nn_real loss = nn_train_softmax_scorer_targets(scorer,
                                               candidate_inputs,
                                               candidate_count,
                                               target_distribution,
                                               0.005f);
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
