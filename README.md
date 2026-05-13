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
cross-entropy / policy confidence
generalization across larger prime ranges
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

The example first trains with supervised softmax cross-entropy over all candidate actions, then runs a strict verifier-guided policy-gradient pass where reward is `1` only when the sampled action matches the exact next gap or the overflow class. It reports top-1/top-5/top-10 accuracy, top-1 candidate primality, mean absolute gap error, exact-action coverage, proposal rank steps, direct scan work, wheel-filtered scan work, model-assisted wheel work, segmented block-verifier work, and model-assisted block-verifier work.

`proposal_steps` measures how quickly the true action appears in the model's proposal order. `wheel exact` measures how many exact sieve primality probes remain after small-prime divisibility filters eliminate obvious composites. `assisted_wheel` tries the model's top-10 proposals before falling back to wheel scan. `block cost` estimates segmented interval-proof work by marking composites with base primes up to `sqrt(p + g)`, and `assisted_block` tries model proposals before falling back to the block scan. These verifier metrics are reported separately because an exact next-prime verifier still has to rule out earlier candidates.

Current interpretation: the network learns a strong gap-ranking prior, but exact next-prime verification remains the classical algorithm's job. That is expected and useful: the experiment is about learned proposal capability under perfect verification, not replacing the verifier.

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

Run sanitizer coverage with:

```sh
make sanitize
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

Training a dataset uses contiguous row-major buffers:

```c
nn_real final_loss = nn_train(net, inputs, targets, sample_count, 0.7, 1000, 1);
```

Mini-batch training uses reusable batch activations, batch deltas, and gradient buffers inside the network:

```c
nn_real final_loss = nn_train_batched(net, inputs, targets, sample_count, 32, 0.05, 10, 1);
```

`inputs` must contain `sample_count * nn_input_size(net)` values. `targets` must contain `sample_count * nn_output_size(net)` values.
