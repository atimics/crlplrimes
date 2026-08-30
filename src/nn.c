#include "nn.h"
#include "nn_blas.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef NN_REAL_IS_FLOAT
#define NN_REAL_IS_FLOAT 1
#endif

#if NN_REAL_IS_FLOAT
#define NN_EXP expf
#define NN_SQRT sqrtf
#define NN_LOG logf
#define NN_TANH tanhf
#else
#define NN_EXP exp
#define NN_SQRT sqrt
#define NN_LOG log
#define NN_TANH tanh
#endif

#define NN_CHECKPOINT_MAGIC "NNCKPT01"
#define NN_CHECKPOINT_MAGIC_SIZE 8u
#define NN_CHECKPOINT_VERSION 1u
#define NN_SCALAR_FLOAT32 1u
#define NN_SCALAR_FLOAT64 2u

struct NeuralNet {
    size_t layer_count;
    size_t *sizes;
    NNActivation hidden_activation;
    NNActivation output_activation;
    NNActivation *layer_activations; /* per-layer override, NULL = use hidden/output */
    NNLoss loss;
    nn_real **activations;
    nn_real **pre_activations; /* pre-activation sums (before activation fn) */
    nn_real **eml_pre_acts;     /* EML_PAIRWISE: 2N pre-act values for N outputs */
    nn_real **biases;
    nn_real **weights;
    nn_real **deltas;
    nn_real **grad_biases;
    nn_real **grad_weights;
    nn_real **batch_activations;
    nn_real **batch_deltas;
    size_t batch_capacity;
    size_t batch_cache_sample_count;
    int batch_cache_valid;
    uint32_t rng_state;
};

static uint32_t rng_next(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static nn_real rng_uniform(uint32_t *state)
{
    return (nn_real)((double)rng_next(state) / (double)UINT32_MAX);
}

static nn_real sigmoid(nn_real x)
{
    if (x >= (nn_real)0) {
        nn_real z = NN_EXP(-x);
        return (nn_real)1 / ((nn_real)1 + z);
    }

    nn_real z = NN_EXP(x);
    return z / ((nn_real)1 + z);
}

static nn_real sigmoid_derivative_from_activation(nn_real activation)
{
    return activation * ((nn_real)1 - activation);
}

static nn_real activate(nn_real x, NNActivation activation)
{
    switch (activation) {
    case NN_ACTIVATION_RELU:
        return x > (nn_real)0 ? x : (nn_real)0;
    case NN_ACTIVATION_TANH:
        return NN_TANH(x);
    case NN_ACTIVATION_LINEAR:
        return x;
    case NN_ACTIVATION_EML:
        return nn_eml_activation(x);
    case NN_ACTIVATION_SIGMOID:
    default:
        return sigmoid(x);
    }
}

/*
 * Derivative of activation function given the *activation value* (post-activation).
 * For most activations the derivative is expressible in terms of the output.
 */
static nn_real activation_derivative_from_activation(nn_real activation_value,
                                                     NNActivation activation)
{
    switch (activation) {
    case NN_ACTIVATION_RELU:
        return activation_value > (nn_real)0 ? (nn_real)1 : (nn_real)0;
    case NN_ACTIVATION_TANH:
        return (nn_real)1 - activation_value * activation_value;
    case NN_ACTIVATION_LINEAR:
        return (nn_real)1;
    case NN_ACTIVATION_EML:
        /* EML requires its pre-activation. Callers handle it separately. */
        return (nn_real)NAN;
    case NN_ACTIVATION_EML_PAIRWISE:
        /* Pairwise derivatives are expanded from the output delta separately. */
        return (nn_real)1;
    case NN_ACTIVATION_SIGMOID:
    default:
        return sigmoid_derivative_from_activation(activation_value);
    }
}

/*
 * Derivative of EML activation given the *pre-activation* value x.
 * activation = exp(x) - softplus(x)
 * d/dx = exp(x) - sigmoid(x)
 */
static nn_real eml_activation_derivative(nn_real x) {
    return nn_eml_activation_dx(x);
}

static NNActivation activation_for_layer(const NeuralNet *net, size_t layer)
{
    if (net->layer_activations != NULL && net->layer_activations[layer] != NN_ACTIVATION_SIGMOID)
        return net->layer_activations[layer];
    return layer + 1 == net->layer_count ? net->output_activation : net->hidden_activation;
}

static nn_real activation_derivative_for_neuron(const NeuralNet *net,
                                                size_t layer,
                                                size_t neuron)
{
    NNActivation activation = activation_for_layer(net, layer);
    if (activation == NN_ACTIVATION_EML) {
        return eml_activation_derivative(net->pre_activations[layer][neuron]);
    }
    return activation_derivative_from_activation(net->activations[layer][neuron],
                                                 activation);
}

static void *checked_calloc(size_t count, size_t size)
{
    if (count == 0 || size == 0) {
        return NULL;
    }

    if (count > SIZE_MAX / size) {
        return NULL;
    }

    return calloc(count, size);
}

static int write_all(FILE *fp, const void *data, size_t size)
{
    return fp != NULL && (size == 0 || fwrite(data, 1, size, fp) == size);
}

static int read_all(FILE *fp, void *data, size_t size)
{
    return fp != NULL && (size == 0 || fread(data, 1, size, fp) == size);
}

static int write_u32_le(FILE *fp, uint32_t value)
{
    unsigned char bytes[4];
    bytes[0] = (unsigned char)(value & 0xffu);
    bytes[1] = (unsigned char)((value >> 8) & 0xffu);
    bytes[2] = (unsigned char)((value >> 16) & 0xffu);
    bytes[3] = (unsigned char)((value >> 24) & 0xffu);
    return write_all(fp, bytes, sizeof(bytes));
}

static int read_u32_le(FILE *fp, uint32_t *value)
{
    unsigned char bytes[4];
    if (value == NULL || !read_all(fp, bytes, sizeof(bytes))) {
        return 0;
    }

    *value = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
             ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
    return 1;
}

static int write_u64_le(FILE *fp, uint64_t value)
{
    unsigned char bytes[8];
    for (size_t i = 0; i < sizeof(bytes); ++i) {
        bytes[i] = (unsigned char)((value >> (8u * i)) & 0xffu);
    }
    return write_all(fp, bytes, sizeof(bytes));
}

static int read_u64_le(FILE *fp, uint64_t *value)
{
    unsigned char bytes[8];
    uint64_t result = 0;
    if (value == NULL || !read_all(fp, bytes, sizeof(bytes))) {
        return 0;
    }

    for (size_t i = 0; i < sizeof(bytes); ++i) {
        result |= (uint64_t)bytes[i] << (8u * i);
    }
    *value = result;
    return 1;
}

static uint32_t scalar_type_id(void)
{
#if NN_REAL_IS_FLOAT
    return NN_SCALAR_FLOAT32;
#else
    return NN_SCALAR_FLOAT64;
#endif
}

static int sizes_are_valid(const size_t *layer_sizes, size_t layer_count)
{
    if (layer_sizes == NULL || layer_count < 2) {
        return 0;
    }

    for (size_t i = 0; i < layer_count; ++i) {
        if (layer_sizes[i] == 0) {
            return 0;
        }
    }

    return 1;
}

static int activation_is_valid(NNActivation activation)
{
    return activation == NN_ACTIVATION_SIGMOID || activation == NN_ACTIVATION_RELU ||
           activation == NN_ACTIVATION_TANH || activation == NN_ACTIVATION_LINEAR ||
           activation == NN_ACTIVATION_EML || activation == NN_ACTIVATION_EML_PAIRWISE;
}

static int loss_is_valid(NNLoss loss)
{
    return loss == NN_LOSS_MSE || loss == NN_LOSS_SOFTMAX_CROSS_ENTROPY;
}

static int config_is_valid(const NNConfig *config)
{
    if (config == NULL || !sizes_are_valid(config->layer_sizes, config->layer_count) ||
        !activation_is_valid(config->hidden_activation) ||
        !activation_is_valid(config->output_activation) || !loss_is_valid(config->loss)) {
        return 0;
    }

    if (config->loss == NN_LOSS_SOFTMAX_CROSS_ENTROPY &&
        (config->output_activation != NN_ACTIVATION_LINEAR &&
         config->output_activation != NN_ACTIVATION_EML_PAIRWISE)) {
        return 0;
    }

    return 1;
}

static void invalidate_batch_cache(NeuralNet *net)
{
    if (net == NULL) {
        return;
    }

    net->batch_cache_valid = 0;
    net->batch_cache_sample_count = 0;
}

static void mark_batch_cache_valid(NeuralNet *net, size_t sample_count)
{
    net->batch_cache_valid = 1;
    net->batch_cache_sample_count = sample_count;
}

static nn_real dot_product(const nn_real *a, const nn_real *b, size_t count)
{
    nn_real sum0 = 0;
    nn_real sum1 = 0;
    nn_real sum2 = 0;
    nn_real sum3 = 0;
    size_t i = 0;

    for (; i + 3 < count; i += 4) {
        sum0 += a[i] * b[i];
        sum1 += a[i + 1] * b[i + 1];
        sum2 += a[i + 2] * b[i + 2];
        sum3 += a[i + 3] * b[i + 3];
    }

    nn_real sum = (sum0 + sum1) + (sum2 + sum3);
    for (; i < count; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

static void axpy(nn_real *restrict y, const nn_real *restrict x, size_t count, nn_real scale)
{
    size_t i = 0;
    for (; i + 3 < count; i += 4) {
        y[i] += scale * x[i];
        y[i + 1] += scale * x[i + 1];
        y[i + 2] += scale * x[i + 2];
        y[i + 3] += scale * x[i + 3];
    }

    for (; i < count; ++i) {
        y[i] += scale * x[i];
    }
}

static void dense_forward_layer(const nn_real *restrict previous_activations,
                                const nn_real *restrict weights,
                                const nn_real *restrict biases,
                                nn_real *restrict activations,
                                nn_real *restrict pre_activations,
                                size_t previous,
                                size_t current,
                                NNActivation activation)
{
    if (activation == NN_ACTIVATION_EML_PAIRWISE) {
        /* EML pairwise: current = output count, weights has 2*current rows.
         * For each output i: z1 = dot(w[2i], x)+b[2i], z2 = dot(w[2i+1], x)+b[2i+1],
         *   activations[i] = eml(z1, z2),
         *   pre_activations stores z1/z2 interleaved if available */
        for (size_t i = 0; i < current; ++i) {
            const nn_real *w1 = &weights[(2 * i) * previous];
            const nn_real *w2 = &weights[(2 * i + 1) * previous];
            nn_real z1 = biases[2 * i] + dot_product(w1, previous_activations, previous);
            nn_real z2 = biases[2 * i + 1] + dot_product(w2, previous_activations, previous);
            if (pre_activations != NULL) {
                pre_activations[2 * i]     = z1;
                pre_activations[2 * i + 1] = z2;
            }
            activations[i] = nn_eml(z1, z2);
        }
        return;
    }

    /* standard activation: y = weights * x (matrix-vector via BLAS sgemv/dgemv) */
nn_blas_gemv(weights, previous_activations, activations,
                 current, previous, (nn_real)1, (nn_real)0);

    for (size_t neuron = 0; neuron < current; ++neuron) {
        nn_real sum = activations[neuron] + biases[neuron];
        if (pre_activations != NULL) pre_activations[neuron] = sum;
        activations[neuron] = activate(sum, activation);
    }
}

static void dense_forward_layer_batch(const nn_real *restrict previous_activations,
                                      const nn_real *restrict weights,
                                      const nn_real *restrict biases,
                                      nn_real *restrict activations,
                                      size_t batch_size,
                                      size_t previous,
                                      size_t current,
                                      NNActivation activation)
{
    if (activation == NN_ACTIVATION_EML_PAIRWISE) {
        /* EML pairwise: 2*current weight rows produce current outputs.
         * For each sample: z1=dot(w[2i],x)+b[2i], z2=dot(w[2i+1],x)+b[2i+1],
         *   out[i] = eml(z1, z2). */
        for (size_t sample = 0; sample < batch_size; ++sample) {
            const nn_real *in = &previous_activations[sample * previous];
            nn_real *out = &activations[sample * current];
            for (size_t i = 0; i < current; ++i) {
                const nn_real *w1 = &weights[(2 * i) * previous];
                const nn_real *w2 = &weights[(2 * i + 1) * previous];
                nn_real z1 = biases[2 * i] + dot_product(w1, in, previous);
                nn_real z2 = biases[2 * i + 1] + dot_product(w2, in, previous);
                out[i] = nn_eml(z1, z2);
            }
        }
        return;
    }

    /* C = X * W^T  (batch matrix-matrix multiply via BLAS sgemm/dgemm)
     * X is batch_size x previous, W is current x previous (row-major).
     * Result C is batch_size x current. */
nn_blas_gemm(previous_activations, weights, activations,
                 batch_size, previous, current, (nn_real)1, (nn_real)0);

    /* apply bias and activation */
    for (size_t sample = 0; sample < batch_size; ++sample) {
        nn_real *act = &activations[sample * current];
        for (size_t neuron = 0; neuron < current; ++neuron) {
            act[neuron] = activate(act[neuron] + biases[neuron], activation);
        }
    }
}

static void free_layer_arrays(nn_real **arrays, size_t layer_count)
{
    if (arrays == NULL) {
        return;
    }

    for (size_t layer = 0; layer < layer_count; ++layer) {
        free(arrays[layer]);
    }

    free(arrays);
}

static void free_batch_workspace(NeuralNet *net)
{
    free_layer_arrays(net->batch_activations, net->layer_count);
    free_layer_arrays(net->batch_deltas, net->layer_count);
    net->batch_activations = NULL;
    net->batch_deltas = NULL;
    net->batch_capacity = 0;
    invalidate_batch_cache(net);
}

static int ensure_batch_workspace(NeuralNet *net, size_t batch_size)
{
    if (batch_size <= net->batch_capacity) {
        return 1;
    }

    nn_real **activations = checked_calloc(net->layer_count, sizeof(*activations));
    nn_real **deltas = checked_calloc(net->layer_count, sizeof(*deltas));
    if (activations == NULL || deltas == NULL) {
        free(activations);
        free(deltas);
        return 0;
    }

    for (size_t layer = 0; layer < net->layer_count; ++layer) {
        if (batch_size > SIZE_MAX / net->sizes[layer]) {
            free_layer_arrays(activations, net->layer_count);
            free_layer_arrays(deltas, net->layer_count);
            return 0;
        }

        size_t value_count = batch_size * net->sizes[layer];
        activations[layer] = checked_calloc(value_count, sizeof(**activations));
        deltas[layer] = checked_calloc(value_count, sizeof(**deltas));
        if (activations[layer] == NULL || deltas[layer] == NULL) {
            free_layer_arrays(activations, net->layer_count);
            free_layer_arrays(deltas, net->layer_count);
            return 0;
        }
    }

    free_batch_workspace(net);
    net->batch_activations = activations;
    net->batch_deltas = deltas;
    net->batch_capacity = batch_size;
    return 1;
}

static void load_batch_inputs(NeuralNet *net,
                              const nn_real *inputs,
                              const size_t *order,
                              size_t start,
                              size_t batch_size,
                              size_t input_width)
{
    nn_real *batch_inputs = net->batch_activations[0];

    if (order == NULL) {
        memcpy(batch_inputs,
               &inputs[start * input_width],
               batch_size * input_width * sizeof(*batch_inputs));
        return;
    }

    for (size_t sample = 0; sample < batch_size; ++sample) {
        size_t index = order[start + sample];
        memcpy(&batch_inputs[sample * input_width],
               &inputs[index * input_width],
               input_width * sizeof(*batch_inputs));
    }
}

static void forward_batch_workspace(NeuralNet *net, size_t batch_size)
{
    for (size_t layer = 1; layer < net->layer_count; ++layer) {
        dense_forward_layer_batch(net->batch_activations[layer - 1],
                                  net->weights[layer],
                                  net->biases[layer],
                                  net->batch_activations[layer],
                                  batch_size,
                                  net->sizes[layer - 1],
                                  net->sizes[layer],
                                  activation_for_layer(net, layer));
    }
}

static int ensure_gradient_buffers(NeuralNet *net)
{
    if (net->grad_biases != NULL && net->grad_weights != NULL) {
        return 1;
    }

    nn_real **grad_biases = checked_calloc(net->layer_count, sizeof(*grad_biases));
    nn_real **grad_weights = checked_calloc(net->layer_count, sizeof(*grad_weights));
    if (grad_biases == NULL || grad_weights == NULL) {
        free(grad_biases);
        free(grad_weights);
        return 0;
    }

    for (size_t layer = 1; layer < net->layer_count; ++layer) {
        size_t current = net->sizes[layer];
        size_t previous = net->sizes[layer - 1];

        if (previous != 0 && current > SIZE_MAX / previous) {
            free_layer_arrays(grad_biases, net->layer_count);
            free_layer_arrays(grad_weights, net->layer_count);
            return 0;
        }

        size_t weight_count = current * previous;
        grad_biases[layer] = checked_calloc(current, sizeof(**grad_biases));
        grad_weights[layer] = checked_calloc(weight_count, sizeof(**grad_weights));

        if (grad_biases[layer] == NULL || grad_weights[layer] == NULL) {
            free_layer_arrays(grad_biases, net->layer_count);
            free_layer_arrays(grad_weights, net->layer_count);
            return 0;
        }
    }

    net->grad_biases = grad_biases;
    net->grad_weights = grad_weights;
    return 1;
}

void nn_softmax(const nn_real *scores, nn_real *probabilities, size_t count)
{
    if (scores == NULL || probabilities == NULL || count == 0) {
        return;
    }

    nn_real max_score = scores[0];
    for (size_t i = 1; i < count; ++i) {
        if (scores[i] > max_score) {
            max_score = scores[i];
        }
    }

    nn_real denominator = 0;
    for (size_t i = 0; i < count; ++i) {
        probabilities[i] = NN_EXP(scores[i] - max_score);
        denominator += probabilities[i];
    }

    if (denominator <= (nn_real)0) {
        nn_real uniform = (nn_real)1 / (nn_real)count;
        for (size_t i = 0; i < count; ++i) {
            probabilities[i] = uniform;
        }
        return;
    }

    for (size_t i = 0; i < count; ++i) {
        probabilities[i] /= denominator;
    }
}

nn_real nn_softmax_cross_entropy(const nn_real *scores, const nn_real *targets, size_t count)
{
    if (scores == NULL || targets == NULL || count == 0) {
        return (nn_real)NAN;
    }

    nn_real max_score = scores[0];
    for (size_t i = 1; i < count; ++i) {
        if (scores[i] > max_score) {
            max_score = scores[i];
        }
    }

    nn_real sum_exp = 0;
    for (size_t i = 0; i < count; ++i) {
        sum_exp += NN_EXP(scores[i] - max_score);
    }

    nn_real log_denominator = max_score + NN_LOG(sum_exp);
    nn_real loss = 0;
    for (size_t i = 0; i < count; ++i) {
        loss -= targets[i] * (scores[i] - log_denominator);
    }

    return loss;
}

static nn_real softmax_cross_entropy_index(const nn_real *scores, size_t count, size_t target_index)
{
    if (scores == NULL || count == 0 || target_index >= count) {
        return (nn_real)NAN;
    }

    nn_real max_score = scores[0];
    for (size_t i = 1; i < count; ++i) {
        if (scores[i] > max_score) {
            max_score = scores[i];
        }
    }

    nn_real sum_exp = 0;
    for (size_t i = 0; i < count; ++i) {
        sum_exp += NN_EXP(scores[i] - max_score);
    }

    return max_score + NN_LOG(sum_exp) - scores[target_index];
}

static nn_real softmax_probabilities_and_index_loss(const nn_real *scores,
                                                    nn_real *probabilities,
                                                    size_t count,
                                                    size_t target_index,
                                                    nn_real *log_denominator_out)
{
    if (scores == NULL || probabilities == NULL || count == 0 || target_index >= count) {
        return (nn_real)NAN;
    }

    nn_real max_score = scores[0];
    for (size_t i = 1; i < count; ++i) {
        if (scores[i] > max_score) {
            max_score = scores[i];
        }
    }

    nn_real denominator = 0;
    for (size_t i = 0; i < count; ++i) {
        probabilities[i] = NN_EXP(scores[i] - max_score);
        denominator += probabilities[i];
    }

    if (denominator <= (nn_real)0) {
        nn_real uniform = (nn_real)1 / (nn_real)count;
        for (size_t i = 0; i < count; ++i) {
            probabilities[i] = uniform;
        }
        if (log_denominator_out != NULL) {
            *log_denominator_out = (nn_real)NAN;
        }
        return (nn_real)NAN;
    }

    nn_real inv_denominator = (nn_real)1 / denominator;
    for (size_t i = 0; i < count; ++i) {
        probabilities[i] *= inv_denominator;
    }

    nn_real log_denominator = max_score + NN_LOG(denominator);
    if (log_denominator_out != NULL) {
        *log_denominator_out = log_denominator;
    }
    return log_denominator - scores[target_index];
}

static nn_real softmax_cross_entropy_from_log_denominator(const nn_real *scores,
                                                          const nn_real *targets,
                                                          size_t count,
                                                          nn_real log_denominator)
{
    nn_real loss = 0;
    for (size_t i = 0; i < count; ++i) {
        loss -= targets[i] * (scores[i] - log_denominator);
    }
    return loss;
}

static nn_real safe_log_probability(nn_real probability)
{
    const nn_real floor = (nn_real)1e-30;
    return NN_LOG(probability < floor ? floor : probability);
}

static nn_real compute_deltas_and_loss(NeuralNet *net, const nn_real *target)
{
    size_t output_layer = net->layer_count - 1;
    size_t output_width = net->sizes[output_layer];
    nn_real *output_activations = net->activations[output_layer];
    nn_real *output_deltas = net->deltas[output_layer];
    nn_real loss = 0;

    if (net->loss == NN_LOSS_SOFTMAX_CROSS_ENTROPY) {
        nn_softmax(output_activations, output_deltas, output_width);
        loss = nn_softmax_cross_entropy(output_activations, target, output_width);

        for (size_t neuron = 0; neuron < output_width; ++neuron) {
            output_deltas[neuron] -= target[neuron];
            output_deltas[neuron] *=
                activation_derivative_for_neuron(net, output_layer, neuron);
        }
    } else {
        for (size_t neuron = 0; neuron < output_width; ++neuron) {
            nn_real activation = output_activations[neuron];
            nn_real error = activation - target[neuron];
            loss += (nn_real)0.5 * error * error;
            output_deltas[neuron] =
                error * activation_derivative_for_neuron(net, output_layer, neuron);
        }
    }

    for (size_t layer = net->layer_count - 2; layer > 0; --layer) {
        size_t current = net->sizes[layer];
        size_t next = net->sizes[layer + 1];
        nn_real *current_deltas = net->deltas[layer];
        const nn_real *next_deltas = net->deltas[layer + 1];
        const nn_real *next_weights = net->weights[layer + 1];

        memset(current_deltas, 0, current * sizeof(*current_deltas));

        for (size_t next_neuron = 0; next_neuron < next; ++next_neuron) {
            const nn_real delta = next_deltas[next_neuron];
            const nn_real *weights = &next_weights[next_neuron * current];
            axpy(current_deltas, weights, current, delta);
        }

        for (size_t neuron = 0; neuron < current; ++neuron) {
            NNActivation act = activation_for_layer(net, layer);
            if (act == NN_ACTIVATION_EML_PAIRWISE) {
                /* derivative applied in eml_pairwise_backprop_post */
            } else {
                current_deltas[neuron] *=
                    activation_derivative_for_neuron(net, layer, neuron);
            }
        }
    }

    return loss;
}

/*
 * Post-process EML_PAIRWISE layers: expand output deltas to pre-activation
 * deltas, backprop through 2x weight rows, and accumulate into previous layer.
 * Called after compute_deltas_and_loss, before apply_sample_update.
 */
static void eml_pairwise_backprop_post(NeuralNet *net)
{
    for (size_t layer = 1; layer < net->layer_count; ++layer) {
        NNActivation act = activation_for_layer(net, layer);
        if (act != NN_ACTIVATION_EML_PAIRWISE) continue;
        if (net->pre_activations == NULL || net->pre_activations[layer] == NULL) continue;

        size_t current = net->sizes[layer];      /* output count (N) */
        size_t previous = net->sizes[layer - 1]; /* input count */
        nn_real *deltas = net->deltas[layer];     /* currently size N */
        const nn_real *pre = net->pre_activations[layer]; /* size 2N */
        nn_real *prev_deltas = net->deltas[layer - 1];
        // prev_acts not needed here; backprop flows through weights

        /* Expand N output deltas to 2N pre-activation deltas in-place.
         * We use a scratch buffer since deltas is only N elements. */
        nn_real *scratch = malloc(2 * current * sizeof(nn_real));
        if (!scratch) continue;

        for (size_t i = 0; i < current; ++i) {
            nn_real d = deltas[i];
            nn_real z1 = pre[2 * i];
            nn_real z2 = pre[2 * i + 1];
            scratch[2 * i]     = d * nn_eml_dx(z1);
            scratch[2 * i + 1] = d * nn_eml_dy(z2);
        }

        /* Backprop expanded deltas through the 2N weight rows to previous layer */
        const nn_real *weights = net->weights[layer]; /* size: 2*current * previous */
        for (size_t j = 0; j < 2 * current; ++j) {
            nn_real delta = scratch[j];
            const nn_real *wrow = &weights[j * previous];
            for (size_t k = 0; k < previous; ++k) {
                prev_deltas[k] += delta * wrow[k];
            }
        }

        free(scratch);
    }
}

static void apply_sample_update(NeuralNet *net, nn_real learning_rate)
{
    for (size_t layer = 1; layer < net->layer_count; ++layer) {
        size_t previous = net->sizes[layer - 1];
        size_t current = net->sizes[layer];
        const nn_real *previous_activations = net->activations[layer - 1];
        const nn_real *deltas = net->deltas[layer];
        nn_real *biases = net->biases[layer];
        nn_real *weights = net->weights[layer];
        NNActivation act = activation_for_layer(net, layer);

        size_t rows = current;
        if (act == NN_ACTIVATION_EML_PAIRWISE) {
            rows = current * 2;
            /* deltas were expanded in eml_pairwise_backprop_post and stored
             * in deltas[layer] — but wait, deltas is only current elements.
             * We need the expanded deltas. Store them in a temporary.
             * For now: use pre_activations to reconstruct. */
            /* Re-expand deltas from pre_activations (same as in post fn) */
            nn_real *scratch = malloc(rows * sizeof(nn_real));
            if (!scratch) continue;
            const nn_real *pre = net->pre_activations[layer];
            for (size_t i = 0; i < current; ++i) {
                nn_real d = deltas[i];
                scratch[2 * i]     = d * nn_eml_dx(pre[2 * i]);
                scratch[2 * i + 1] = d * nn_eml_dy(pre[2 * i + 1]);
            }
            for (size_t neuron = 0; neuron < rows; ++neuron) {
                nn_real step = learning_rate * scratch[neuron];
                nn_real *nw = &weights[neuron * previous];
                biases[neuron] -= step;
                for (size_t j = 0; j < previous; ++j)
                    nw[j] -= step * previous_activations[j];
            }
            free(scratch);
            continue;
        }
        for (size_t neuron = 0; neuron < rows; ++neuron) {
            nn_real step = learning_rate * deltas[neuron];
            nn_real *neuron_weights = &weights[neuron * previous];

            biases[neuron] -= step;

            for (size_t input_index = 0; input_index < previous; ++input_index) {
                neuron_weights[input_index] -= step * previous_activations[input_index];
            }
        }
    }

    invalidate_batch_cache(net);
}

static void dense_backprop_deltas_batch(const nn_real *restrict next_deltas,
                                        const nn_real *restrict next_weights,
                                        const nn_real *restrict current_activations,
                                        nn_real *restrict current_deltas,
                                        size_t batch_size,
                                        size_t current,
                                        size_t next,
                                        NNActivation activation)
{
    size_t sample = 0;

    for (; sample + 3 < batch_size; sample += 4) {
        const nn_real *next_delta0 = &next_deltas[(sample + 0) * next];
        const nn_real *next_delta1 = &next_deltas[(sample + 1) * next];
        const nn_real *next_delta2 = &next_deltas[(sample + 2) * next];
        const nn_real *next_delta3 = &next_deltas[(sample + 3) * next];
        const nn_real *activation0 = &current_activations[(sample + 0) * current];
        const nn_real *activation1 = &current_activations[(sample + 1) * current];
        const nn_real *activation2 = &current_activations[(sample + 2) * current];
        const nn_real *activation3 = &current_activations[(sample + 3) * current];
        nn_real *current_delta0 = &current_deltas[(sample + 0) * current];
        nn_real *current_delta1 = &current_deltas[(sample + 1) * current];
        nn_real *current_delta2 = &current_deltas[(sample + 2) * current];
        nn_real *current_delta3 = &current_deltas[(sample + 3) * current];
        size_t neuron = 0;

        for (; neuron + 3 < current; neuron += 4) {
            nn_real sum00 = 0;
            nn_real sum01 = 0;
            nn_real sum02 = 0;
            nn_real sum03 = 0;
            nn_real sum10 = 0;
            nn_real sum11 = 0;
            nn_real sum12 = 0;
            nn_real sum13 = 0;
            nn_real sum20 = 0;
            nn_real sum21 = 0;
            nn_real sum22 = 0;
            nn_real sum23 = 0;
            nn_real sum30 = 0;
            nn_real sum31 = 0;
            nn_real sum32 = 0;
            nn_real sum33 = 0;

            for (size_t next_neuron = 0; next_neuron < next; ++next_neuron) {
                const nn_real *weights = &next_weights[next_neuron * current + neuron];
                nn_real d0 = next_delta0[next_neuron];
                nn_real d1 = next_delta1[next_neuron];
                nn_real d2 = next_delta2[next_neuron];
                nn_real d3 = next_delta3[next_neuron];
                nn_real w0 = weights[0];
                nn_real w1 = weights[1];
                nn_real w2 = weights[2];
                nn_real w3 = weights[3];

                sum00 += d0 * w0;
                sum01 += d0 * w1;
                sum02 += d0 * w2;
                sum03 += d0 * w3;
                sum10 += d1 * w0;
                sum11 += d1 * w1;
                sum12 += d1 * w2;
                sum13 += d1 * w3;
                sum20 += d2 * w0;
                sum21 += d2 * w1;
                sum22 += d2 * w2;
                sum23 += d2 * w3;
                sum30 += d3 * w0;
                sum31 += d3 * w1;
                sum32 += d3 * w2;
                sum33 += d3 * w3;
            }

            current_delta0[neuron + 0] =
                sum00 * activation_derivative_from_activation(activation0[neuron + 0], activation);
            current_delta0[neuron + 1] =
                sum01 * activation_derivative_from_activation(activation0[neuron + 1], activation);
            current_delta0[neuron + 2] =
                sum02 * activation_derivative_from_activation(activation0[neuron + 2], activation);
            current_delta0[neuron + 3] =
                sum03 * activation_derivative_from_activation(activation0[neuron + 3], activation);
            current_delta1[neuron + 0] =
                sum10 * activation_derivative_from_activation(activation1[neuron + 0], activation);
            current_delta1[neuron + 1] =
                sum11 * activation_derivative_from_activation(activation1[neuron + 1], activation);
            current_delta1[neuron + 2] =
                sum12 * activation_derivative_from_activation(activation1[neuron + 2], activation);
            current_delta1[neuron + 3] =
                sum13 * activation_derivative_from_activation(activation1[neuron + 3], activation);
            current_delta2[neuron + 0] =
                sum20 * activation_derivative_from_activation(activation2[neuron + 0], activation);
            current_delta2[neuron + 1] =
                sum21 * activation_derivative_from_activation(activation2[neuron + 1], activation);
            current_delta2[neuron + 2] =
                sum22 * activation_derivative_from_activation(activation2[neuron + 2], activation);
            current_delta2[neuron + 3] =
                sum23 * activation_derivative_from_activation(activation2[neuron + 3], activation);
            current_delta3[neuron + 0] =
                sum30 * activation_derivative_from_activation(activation3[neuron + 0], activation);
            current_delta3[neuron + 1] =
                sum31 * activation_derivative_from_activation(activation3[neuron + 1], activation);
            current_delta3[neuron + 2] =
                sum32 * activation_derivative_from_activation(activation3[neuron + 2], activation);
            current_delta3[neuron + 3] =
                sum33 * activation_derivative_from_activation(activation3[neuron + 3], activation);
        }

        for (; neuron < current; ++neuron) {
            nn_real sum0 = 0;
            nn_real sum1 = 0;
            nn_real sum2 = 0;
            nn_real sum3 = 0;

            for (size_t next_neuron = 0; next_neuron < next; ++next_neuron) {
                nn_real weight = next_weights[next_neuron * current + neuron];
                sum0 += next_delta0[next_neuron] * weight;
                sum1 += next_delta1[next_neuron] * weight;
                sum2 += next_delta2[next_neuron] * weight;
                sum3 += next_delta3[next_neuron] * weight;
            }

            current_delta0[neuron] =
                sum0 * activation_derivative_from_activation(activation0[neuron], activation);
            current_delta1[neuron] =
                sum1 * activation_derivative_from_activation(activation1[neuron], activation);
            current_delta2[neuron] =
                sum2 * activation_derivative_from_activation(activation2[neuron], activation);
            current_delta3[neuron] =
                sum3 * activation_derivative_from_activation(activation3[neuron], activation);
        }
    }

    for (; sample < batch_size; ++sample) {
        nn_real *current_delta_row = &current_deltas[sample * current];
        const nn_real *next_delta_row = &next_deltas[sample * next];
        const nn_real *activation_row = &current_activations[sample * current];

        memset(current_delta_row, 0, current * sizeof(*current_delta_row));

        for (size_t next_neuron = 0; next_neuron < next; ++next_neuron) {
            const nn_real delta = next_delta_row[next_neuron];
            const nn_real *weights = &next_weights[next_neuron * current];
            axpy(current_delta_row, weights, current, delta);
        }

        for (size_t neuron = 0; neuron < current; ++neuron) {
            current_delta_row[neuron] *=
                activation_derivative_from_activation(activation_row[neuron], activation);
        }
    }
}

static nn_real compute_batch_deltas_and_loss(NeuralNet *net,
                                             const nn_real *targets,
                                             const size_t *order,
                                             size_t start,
                                             size_t batch_size,
                                             size_t target_width)
{
    size_t output_layer = net->layer_count - 1;
    size_t output_width = net->sizes[output_layer];
    const nn_real *output_activations = net->batch_activations[output_layer];
    nn_real *output_deltas = net->batch_deltas[output_layer];
    nn_real loss = 0;

    for (size_t sample = 0; sample < batch_size; ++sample) {
        size_t index = order == NULL ? start + sample : order[start + sample];
        const nn_real *target = &targets[index * target_width];
        const nn_real *activations = &output_activations[sample * output_width];
        nn_real *deltas = &output_deltas[sample * output_width];

        if (net->loss == NN_LOSS_SOFTMAX_CROSS_ENTROPY) {
            nn_softmax(activations, deltas, output_width);
            loss += nn_softmax_cross_entropy(activations, target, output_width);

            for (size_t neuron = 0; neuron < output_width; ++neuron) {
                deltas[neuron] -= target[neuron];
                deltas[neuron] *= activation_derivative_from_activation(
                    activations[neuron], net->output_activation);
            }
        } else {
            for (size_t neuron = 0; neuron < output_width; ++neuron) {
                nn_real error = activations[neuron] - target[neuron];
                loss += (nn_real)0.5 * error * error;
                deltas[neuron] = error * activation_derivative_from_activation(
                                             activations[neuron], net->output_activation);
            }
        }
    }

    for (size_t layer = net->layer_count - 2; layer > 0; --layer) {
        size_t current = net->sizes[layer];
        size_t next = net->sizes[layer + 1];
        nn_real *current_deltas = net->batch_deltas[layer];
        const nn_real *next_deltas = net->batch_deltas[layer + 1];
        const nn_real *next_weights = net->weights[layer + 1];
        const nn_real *current_activations = net->batch_activations[layer];

        dense_backprop_deltas_batch(next_deltas,
                                    next_weights,
                                    current_activations,
                                    current_deltas,
                                    batch_size,
                                    current,
                                    next,
                                    activation_for_layer(net, layer));
    }

    return loss;
}

static void apply_batch_gradients(NeuralNet *net, size_t batch_size, nn_real scale)
{
    for (size_t layer = 1; layer < net->layer_count; ++layer) {
        size_t previous = net->sizes[layer - 1];
        size_t current = net->sizes[layer];
        const nn_real *previous_activations = net->batch_activations[layer - 1];
        const nn_real *deltas = net->batch_deltas[layer];
        nn_real *biases = net->biases[layer];
        nn_real *weights = net->weights[layer];
        nn_real *grad_biases = net->grad_biases[layer];
        nn_real *grad_weights = net->grad_weights[layer];

        if (current == 1) {
            nn_real *neuron_grad_weights = grad_weights;
            nn_real *neuron_weights = weights;
            nn_real bias_sum = 0;
            memset(neuron_grad_weights, 0, previous * sizeof(*neuron_grad_weights));

            for (size_t sample = 0; sample < batch_size; ++sample) {
                nn_real delta = deltas[sample];
                const nn_real *previous_row = &previous_activations[sample * previous];
                bias_sum += delta;
                axpy(neuron_grad_weights, previous_row, previous, delta);
            }

            size_t input_index = 0;
            for (; input_index + 3 < previous; input_index += 4) {
                neuron_weights[input_index + 0] -= scale * neuron_grad_weights[input_index + 0];
                neuron_weights[input_index + 1] -= scale * neuron_grad_weights[input_index + 1];
                neuron_weights[input_index + 2] -= scale * neuron_grad_weights[input_index + 2];
                neuron_weights[input_index + 3] -= scale * neuron_grad_weights[input_index + 3];
            }
            for (; input_index < previous; ++input_index) {
                neuron_weights[input_index] -= scale * neuron_grad_weights[input_index];
            }

            grad_biases[0] = bias_sum;
            biases[0] -= scale * bias_sum;
            continue;
        }

        size_t neuron = 0;
        for (; neuron + 3 < current; neuron += 4) {
            nn_real bias0 = 0;
            nn_real bias1 = 0;
            nn_real bias2 = 0;
            nn_real bias3 = 0;
            size_t input_index = 0;

            for (; input_index + 3 < previous; input_index += 4) {
                nn_real sum00 = 0;
                nn_real sum01 = 0;
                nn_real sum02 = 0;
                nn_real sum03 = 0;
                nn_real sum10 = 0;
                nn_real sum11 = 0;
                nn_real sum12 = 0;
                nn_real sum13 = 0;
                nn_real sum20 = 0;
                nn_real sum21 = 0;
                nn_real sum22 = 0;
                nn_real sum23 = 0;
                nn_real sum30 = 0;
                nn_real sum31 = 0;
                nn_real sum32 = 0;
                nn_real sum33 = 0;

                for (size_t sample = 0; sample < batch_size; ++sample) {
                    const nn_real *previous_row = &previous_activations[sample * previous];
                    const nn_real *delta_row = &deltas[sample * current + neuron];
                    nn_real x0 = previous_row[input_index + 0];
                    nn_real x1 = previous_row[input_index + 1];
                    nn_real x2 = previous_row[input_index + 2];
                    nn_real x3 = previous_row[input_index + 3];
                    nn_real d0 = delta_row[0];
                    nn_real d1 = delta_row[1];
                    nn_real d2 = delta_row[2];
                    nn_real d3 = delta_row[3];

                    sum00 += d0 * x0;
                    sum01 += d0 * x1;
                    sum02 += d0 * x2;
                    sum03 += d0 * x3;
                    sum10 += d1 * x0;
                    sum11 += d1 * x1;
                    sum12 += d1 * x2;
                    sum13 += d1 * x3;
                    sum20 += d2 * x0;
                    sum21 += d2 * x1;
                    sum22 += d2 * x2;
                    sum23 += d2 * x3;
                    sum30 += d3 * x0;
                    sum31 += d3 * x1;
                    sum32 += d3 * x2;
                    sum33 += d3 * x3;
                }

                nn_real *grad0 = &grad_weights[(neuron + 0) * previous + input_index];
                nn_real *grad1 = &grad_weights[(neuron + 1) * previous + input_index];
                nn_real *grad2 = &grad_weights[(neuron + 2) * previous + input_index];
                nn_real *grad3 = &grad_weights[(neuron + 3) * previous + input_index];
                nn_real *weights0 = &weights[(neuron + 0) * previous + input_index];
                nn_real *weights1 = &weights[(neuron + 1) * previous + input_index];
                nn_real *weights2 = &weights[(neuron + 2) * previous + input_index];
                nn_real *weights3 = &weights[(neuron + 3) * previous + input_index];

                grad0[0] = sum00;
                grad0[1] = sum01;
                grad0[2] = sum02;
                grad0[3] = sum03;
                grad1[0] = sum10;
                grad1[1] = sum11;
                grad1[2] = sum12;
                grad1[3] = sum13;
                grad2[0] = sum20;
                grad2[1] = sum21;
                grad2[2] = sum22;
                grad2[3] = sum23;
                grad3[0] = sum30;
                grad3[1] = sum31;
                grad3[2] = sum32;
                grad3[3] = sum33;
                weights0[0] -= scale * sum00;
                weights0[1] -= scale * sum01;
                weights0[2] -= scale * sum02;
                weights0[3] -= scale * sum03;
                weights1[0] -= scale * sum10;
                weights1[1] -= scale * sum11;
                weights1[2] -= scale * sum12;
                weights1[3] -= scale * sum13;
                weights2[0] -= scale * sum20;
                weights2[1] -= scale * sum21;
                weights2[2] -= scale * sum22;
                weights2[3] -= scale * sum23;
                weights3[0] -= scale * sum30;
                weights3[1] -= scale * sum31;
                weights3[2] -= scale * sum32;
                weights3[3] -= scale * sum33;
            }

            for (; input_index < previous; ++input_index) {
                nn_real sum0 = 0;
                nn_real sum1 = 0;
                nn_real sum2 = 0;
                nn_real sum3 = 0;

                for (size_t sample = 0; sample < batch_size; ++sample) {
                    nn_real activation = previous_activations[sample * previous + input_index];
                    const nn_real *delta_row = &deltas[sample * current + neuron];
                    sum0 += delta_row[0] * activation;
                    sum1 += delta_row[1] * activation;
                    sum2 += delta_row[2] * activation;
                    sum3 += delta_row[3] * activation;
                }

                grad_weights[(neuron + 0) * previous + input_index] = sum0;
                grad_weights[(neuron + 1) * previous + input_index] = sum1;
                grad_weights[(neuron + 2) * previous + input_index] = sum2;
                grad_weights[(neuron + 3) * previous + input_index] = sum3;
                weights[(neuron + 0) * previous + input_index] -= scale * sum0;
                weights[(neuron + 1) * previous + input_index] -= scale * sum1;
                weights[(neuron + 2) * previous + input_index] -= scale * sum2;
                weights[(neuron + 3) * previous + input_index] -= scale * sum3;
            }

            for (size_t sample = 0; sample < batch_size; ++sample) {
                const nn_real *delta_row = &deltas[sample * current + neuron];
                bias0 += delta_row[0];
                bias1 += delta_row[1];
                bias2 += delta_row[2];
                bias3 += delta_row[3];
            }

            grad_biases[neuron + 0] = bias0;
            grad_biases[neuron + 1] = bias1;
            grad_biases[neuron + 2] = bias2;
            grad_biases[neuron + 3] = bias3;
            biases[neuron + 0] -= scale * bias0;
            biases[neuron + 1] -= scale * bias1;
            biases[neuron + 2] -= scale * bias2;
            biases[neuron + 3] -= scale * bias3;
        }

        for (; neuron < current; ++neuron) {
            nn_real *neuron_grad_weights = &grad_weights[neuron * previous];
            nn_real *neuron_weights = &weights[neuron * previous];
            nn_real bias_sum = 0;
            size_t input_index = 0;

            for (size_t sample = 0; sample < batch_size; ++sample) {
                bias_sum += deltas[sample * current + neuron];
            }

            for (; input_index + 3 < previous; input_index += 4) {
                nn_real sum0 = 0;
                nn_real sum1 = 0;
                nn_real sum2 = 0;
                nn_real sum3 = 0;

                for (size_t sample = 0; sample < batch_size; ++sample) {
                    const nn_real *previous_row = &previous_activations[sample * previous];
                    nn_real delta = deltas[sample * current + neuron];
                    sum0 += delta * previous_row[input_index + 0];
                    sum1 += delta * previous_row[input_index + 1];
                    sum2 += delta * previous_row[input_index + 2];
                    sum3 += delta * previous_row[input_index + 3];
                }

                neuron_grad_weights[input_index + 0] = sum0;
                neuron_grad_weights[input_index + 1] = sum1;
                neuron_grad_weights[input_index + 2] = sum2;
                neuron_grad_weights[input_index + 3] = sum3;
                neuron_weights[input_index + 0] -= scale * sum0;
                neuron_weights[input_index + 1] -= scale * sum1;
                neuron_weights[input_index + 2] -= scale * sum2;
                neuron_weights[input_index + 3] -= scale * sum3;
            }

            for (; input_index < previous; ++input_index) {
                nn_real sum = 0;
                for (size_t sample = 0; sample < batch_size; ++sample) {
                    sum += deltas[sample * current + neuron] *
                           previous_activations[sample * previous + input_index];
                }
                neuron_grad_weights[input_index] = sum;
                neuron_weights[input_index] -= scale * sum;
            }

            grad_biases[neuron] = bias_sum;
            biases[neuron] -= scale * bias_sum;
        }
    }

    invalidate_batch_cache(net);
}

static nn_real train_indexed_batch(NeuralNet *net,
                                   const nn_real *inputs,
                                   const nn_real *targets,
                                   const size_t *order,
                                   size_t start,
                                   size_t batch_size,
                                   size_t input_width,
                                   size_t target_width,
                                   nn_real learning_rate)
{
    invalidate_batch_cache(net);

    if (!ensure_batch_workspace(net, batch_size)) {
        return (nn_real)NAN;
    }

    if (!ensure_gradient_buffers(net)) {
        return (nn_real)NAN;
    }

    load_batch_inputs(net, inputs, order, start, batch_size, input_width);
    forward_batch_workspace(net, batch_size);
    nn_real loss =
        compute_batch_deltas_and_loss(net, targets, order, start, batch_size, target_width);
    apply_batch_gradients(net, batch_size, learning_rate / (nn_real)batch_size);
    return loss / (nn_real)batch_size;
}

NeuralNet *nn_create_config(const NNConfig *config)
{
    if (!config_is_valid(config)) {
        return NULL;
    }

    NeuralNet *net = calloc(1, sizeof(*net));
    if (net == NULL) {
        return NULL;
    }

    net->layer_count = config->layer_count;
    net->hidden_activation = config->hidden_activation;
    net->output_activation = config->output_activation;
    net->loss = config->loss;
    net->rng_state = config->seed == 0 ? 1u : config->seed;
    net->sizes = checked_calloc(config->layer_count, sizeof(*net->sizes));
    net->activations = checked_calloc(config->layer_count, sizeof(*net->activations));
    net->pre_activations = checked_calloc(config->layer_count, sizeof(*net->pre_activations));
    net->eml_pre_acts = checked_calloc(config->layer_count, sizeof(*net->eml_pre_acts));
    net->biases = checked_calloc(config->layer_count, sizeof(*net->biases));
    net->weights = checked_calloc(config->layer_count, sizeof(*net->weights));
    net->deltas = checked_calloc(config->layer_count, sizeof(*net->deltas));

    if (net->sizes == NULL || net->activations == NULL || net->pre_activations == NULL ||
        net->eml_pre_acts == NULL ||
        net->biases == NULL || net->weights == NULL || net->deltas == NULL) {
        nn_free(net);
        return NULL;
    }

    memcpy(net->sizes, config->layer_sizes, config->layer_count * sizeof(*net->sizes));

    for (size_t layer = 0; layer < config->layer_count; ++layer) {
        net->activations[layer] = checked_calloc(net->sizes[layer], sizeof(**net->activations));
        NNActivation la = layer + 1 == net->layer_count ? config->output_activation : config->hidden_activation;
        size_t pre_act_size = net->sizes[layer];
        if (la == NN_ACTIVATION_EML_PAIRWISE) {
            pre_act_size = net->sizes[layer] * 2;
        }
        net->pre_activations[layer] = checked_calloc(pre_act_size, sizeof(**net->pre_activations));
        net->deltas[layer] = checked_calloc(net->sizes[layer], sizeof(**net->deltas));

        if (net->activations[layer] == NULL || net->pre_activations[layer] == NULL || net->deltas[layer] == NULL) {
            nn_free(net);
            return NULL;
        }

        if (layer == 0) {
            continue;
        }

        size_t previous = net->sizes[layer - 1];
        size_t current = net->sizes[layer];
        /* EML_PAIRWISE: double weight count (2 pre-act sums per output) */
        size_t weight_rows = (la == NN_ACTIVATION_EML_PAIRWISE) ? current * 2 : current;
        if (previous != 0 && weight_rows > SIZE_MAX / previous) {
            nn_free(net);
            return NULL;
        }
        size_t weight_count = weight_rows * previous;

        net->biases[layer] = checked_calloc(weight_rows, sizeof(**net->biases));
        net->weights[layer] = checked_calloc(weight_count, sizeof(**net->weights));

        if (net->biases[layer] == NULL || net->weights[layer] == NULL) {
            nn_free(net);
            return NULL;
        }

        NNActivation layer_activation = la;
        nn_real scale;
        if (layer_activation == NN_ACTIVATION_RELU) {
            scale = NN_SQRT((nn_real)6 / (nn_real)previous);
        } else if (layer_activation == NN_ACTIVATION_EML) {
            /* EML: smaller init to start in the safe exp/log domain */
            scale = NN_SQRT((nn_real)0.5f / (nn_real)previous);
        } else {
            scale = NN_SQRT((nn_real)6 / (nn_real)(previous + current));
        }
        for (size_t i = 0; i < weight_count; ++i) {
            nn_real w = ((rng_uniform(&net->rng_state) * (nn_real)2) - (nn_real)1) * scale;
            /* for EML: shift mean negative so pre-acts start in safe zone */
            if (layer_activation == NN_ACTIVATION_EML) {
                w -= scale * (nn_real)0.5f;
            }
            net->weights[layer][i] = w;
        }
        /* for EML: zero-centered init with small variance */
        if (layer_activation == NN_ACTIVATION_EML || layer_activation == NN_ACTIVATION_EML_PAIRWISE) {
            for (size_t i = 0; i < weight_rows; ++i) {
                net->biases[layer][i] = (nn_real)0;
            }
        }
    }

    return net;
}

NeuralNet *nn_create(const size_t *layer_sizes, size_t layer_count, unsigned int seed)
{
    NNConfig config;
    config.layer_sizes = layer_sizes;
    config.layer_count = layer_count;
    config.seed = seed;
    config.hidden_activation = NN_ACTIVATION_SIGMOID;
    config.output_activation = NN_ACTIVATION_SIGMOID;
    config.loss = NN_LOSS_MSE;
    return nn_create_config(&config);
}

NeuralNet *nn_clone(const NeuralNet *source)
{
    if (source == NULL) {
        return NULL;
    }

    NNConfig config;
    config.layer_sizes = source->sizes;
    config.layer_count = source->layer_count;
    config.seed = source->rng_state;
    config.hidden_activation = source->hidden_activation;
    config.output_activation = source->output_activation;
    config.loss = source->loss;

    NeuralNet *copy = nn_create_config(&config);
    if (copy == NULL) {
        return NULL;
    }

    copy->rng_state = source->rng_state;
    for (size_t layer = 1; layer < source->layer_count; ++layer) {
        size_t previous = source->sizes[layer - 1];
        size_t current = source->sizes[layer];
        memcpy(copy->biases[layer],
               source->biases[layer],
               current * sizeof(*copy->biases[layer]));
        memcpy(copy->weights[layer],
               source->weights[layer],
               current * previous * sizeof(*copy->weights[layer]));
    }

    return copy;
}

void nn_free(NeuralNet *net)
{
    if (net == NULL) {
        return;
    }

    free_layer_arrays(net->activations, net->layer_count);
    free_layer_arrays(net->pre_activations, net->layer_count);
    free_layer_arrays(net->eml_pre_acts, net->layer_count);
    free_layer_arrays(net->biases, net->layer_count);
    free_layer_arrays(net->weights, net->layer_count);
    free_layer_arrays(net->deltas, net->layer_count);
    free_layer_arrays(net->grad_biases, net->layer_count);
    free_layer_arrays(net->grad_weights, net->layer_count);

    free(net->layer_activations);

    free_batch_workspace(net);

    free(net->sizes);
    free(net);
}

size_t nn_layer_count(const NeuralNet *net)
{
    return net == NULL ? 0 : net->layer_count;
}

size_t nn_layer_size(const NeuralNet *net, size_t layer_index)
{
    return net == NULL || layer_index >= net->layer_count ? 0 : net->sizes[layer_index];
}

NNActivation nn_hidden_activation(const NeuralNet *net)
{
    return net == NULL ? NN_ACTIVATION_SIGMOID : net->hidden_activation;
}

NNActivation nn_output_activation(const NeuralNet *net)
{
    return net == NULL ? NN_ACTIVATION_SIGMOID : net->output_activation;
}

NNLoss nn_loss(const NeuralNet *net)
{
    return net == NULL ? NN_LOSS_MSE : net->loss;
}

size_t nn_input_size(const NeuralNet *net)
{
    return net == NULL ? 0 : net->sizes[0];
}

size_t nn_output_size(const NeuralNet *net)
{
    return net == NULL ? 0 : net->sizes[net->layer_count - 1];
}

size_t nn_parameter_count(const NeuralNet *net)
{
    if (net == NULL) {
        return 0;
    }

    size_t total = 0;
    for (size_t layer = 1; layer < net->layer_count; ++layer) {
        size_t previous = net->sizes[layer - 1];
        size_t current = net->sizes[layer];

        if (previous != 0 && current > (SIZE_MAX - total) / previous) {
            return 0;
        }
        total += previous * current;

        if (current > SIZE_MAX - total) {
            return 0;
        }
        total += current;
    }

    return total;
}

#include "nn_numeric_certificate.inc"

int nn_neuron_contract_interval(const NeuralNet *net,
                                size_t layer_index,
                                size_t neuron_index,
                                nn_real input_min,
                                nn_real input_max,
                                NNNeuronContract *out_contract)
{
    if (out_contract != NULL) {
        memset(out_contract, 0, sizeof(*out_contract));
        snprintf(out_contract->schema,
                 sizeof(out_contract->schema),
                 "%s",
                 NN_NEURON_CONTRACT_SCHEMA);
        out_contract->report_version = 1u;
        out_contract->pre_activation_min = (nn_real)NAN;
        out_contract->pre_activation_max = (nn_real)NAN;
        out_contract->output_min = (nn_real)NAN;
        out_contract->output_max = (nn_real)NAN;
    }
    if (net == NULL || out_contract == NULL || layer_index == 0 ||
        layer_index >= net->layer_count || neuron_index >= net->sizes[layer_index] ||
        input_min > input_max || !isfinite((double)input_min) ||
        !isfinite((double)input_max)) {
        return 0;
    }

    size_t previous = net->sizes[layer_index - 1];
    const nn_real *weights = &net->weights[layer_index][neuron_index * previous];
    nn_real lower = net->biases[layer_index][neuron_index];
    nn_real upper = lower;
    nn_real weight_l1 = (nn_real)0;
    int finite = isfinite((double)lower);
    for (size_t i = 0; i < previous; ++i) {
        nn_real weight = weights[i];
        if (!isfinite((double)weight)) {
            finite = 0;
            continue;
        }
        weight_l1 += nn_abs_real(weight);
        if (weight >= (nn_real)0) {
            lower += weight * input_min;
            upper += weight * input_max;
        } else {
            lower += weight * input_max;
            upper += weight * input_min;
        }
    }

    NNActivation activation = activation_for_layer(net, layer_index);
    nn_real output_min = lower;
    nn_real output_max = upper;
    nn_real derivative_upper_bound = (nn_real)1;
    switch (activation) {
    case NN_ACTIVATION_RELU:
        derivative_upper_bound = upper <= (nn_real)0 ? (nn_real)0 : (nn_real)1;
        output_min = lower <= (nn_real)0 ? (nn_real)0 : lower;
        output_max = upper <= (nn_real)0 ? (nn_real)0 : upper;
        break;
    case NN_ACTIVATION_TANH:
        derivative_upper_bound = (nn_real)1;
        output_min = NN_TANH(lower);
        output_max = NN_TANH(upper);
        break;
    case NN_ACTIVATION_SIGMOID:
        derivative_upper_bound = (nn_real)0.25;
        output_min = sigmoid(lower);
        output_max = sigmoid(upper);
        break;
    case NN_ACTIVATION_LINEAR:
    default:
        derivative_upper_bound = (nn_real)1;
        break;
    }

    out_contract->layer_index = layer_index;
    out_contract->neuron_index = neuron_index;
    out_contract->input_width = previous;
    out_contract->input_min = input_min;
    out_contract->input_max = input_max;
    out_contract->pre_activation_min = lower;
    out_contract->pre_activation_max = upper;
    out_contract->output_min = output_min;
    out_contract->output_max = output_max;
    out_contract->weight_l1 = weight_l1;
    out_contract->derivative_upper_bound = derivative_upper_bound;
    out_contract->lipschitz_upper_bound = weight_l1 * derivative_upper_bound;
    out_contract->certified = finite && isfinite((double)lower) && isfinite((double)upper) &&
                              isfinite((double)output_min) && isfinite((double)output_max) &&
                              isfinite((double)out_contract->lipschitz_upper_bound);
    return out_contract->certified;
}

int nn_neuron_contract_write_jsonl(FILE *fp, const NNNeuronContract *contract)
{
    if (fp == NULL || contract == NULL) {
        return 0;
    }
    if (fputs("{\"schema\":", fp) == EOF ||
        !nn_write_json_string(fp,
                              contract->schema[0] == '\0' ? NN_NEURON_CONTRACT_SCHEMA
                                                           : contract->schema) ||
        fprintf(fp,
                ",\"report_version\":%u,\"layer_index\":%zu,"
                "\"neuron_index\":%zu,\"input_width\":%zu,\"certified\":%s",
                contract->report_version,
                contract->layer_index,
                contract->neuron_index,
                contract->input_width,
                contract->certified ? "true" : "false") < 0) {
        return 0;
    }

    const struct {
        const char *name;
        nn_real value;
    } fields[] = {
        {"input_min", contract->input_min},
        {"input_max", contract->input_max},
        {"pre_activation_min", contract->pre_activation_min},
        {"pre_activation_max", contract->pre_activation_max},
        {"output_min", contract->output_min},
        {"output_max", contract->output_max},
        {"weight_l1", contract->weight_l1},
        {"derivative_upper_bound", contract->derivative_upper_bound},
        {"lipschitz_upper_bound", contract->lipschitz_upper_bound},
    };
    for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); ++i) {
        if (fprintf(fp, ",\"%s\":", fields[i].name) < 0 ||
            !nn_write_json_real(fp, fields[i].value)) {
            return 0;
        }
    }
    return fputs("}\n", fp) != EOF;
}

void nn_set_layer_activation(NeuralNet *net, size_t layer, NNActivation activation)
{
    if (net == NULL || layer >= net->layer_count) return;
    if (net->layer_activations == NULL) {
        net->layer_activations = calloc(net->layer_count, sizeof(NNActivation));
        if (net->layer_activations == NULL) return;
    }
    net->layer_activations[layer] = activation;
}

size_t nn_forward_flop_count(const NeuralNet *net)
{
    if (net == NULL) {
        return 0;
    }

    size_t total = 0;
    for (size_t layer = 1; layer < net->layer_count; ++layer) {
        size_t previous = net->sizes[layer - 1];
        size_t current = net->sizes[layer];

        if (previous > (SIZE_MAX - 1) / 2) {
            return 0;
        }
        size_t flops_per_neuron = 2 * previous + 1;

        if (current > (SIZE_MAX - total) / flops_per_neuron) {
            return 0;
        }
        total += current * flops_per_neuron;
    }

    return total;
}

#ifdef NN_ENABLE_TEST_API
static nn_real *parameter_pointer(NeuralNet *net, size_t index)
{
    if (net == NULL) {
        return NULL;
    }

    for (size_t layer = 1; layer < net->layer_count; ++layer) {
        size_t previous = net->sizes[layer - 1];
        size_t current = net->sizes[layer];
        size_t weight_count = previous * current;

        if (index < weight_count) {
            return &net->weights[layer][index];
        }
        index -= weight_count;

        if (index < current) {
            return &net->biases[layer][index];
        }
        index -= current;
    }

    return NULL;
}

nn_real nn_test_get_parameter(const NeuralNet *net, size_t index)
{
    nn_real *parameter = parameter_pointer((NeuralNet *)net, index);
    return parameter == NULL ? (nn_real)NAN : *parameter;
}

int nn_test_set_parameter(NeuralNet *net, size_t index, nn_real value)
{
    nn_real *parameter = parameter_pointer(net, index);
    if (parameter == NULL) {
        return 0;
    }

    *parameter = value;
    return 1;
}

int nn_test_has_gradient_buffers(const NeuralNet *net)
{
    return net != NULL && net->grad_biases != NULL && net->grad_weights != NULL;
}
#endif

void nn_forward(NeuralNet *net, const nn_real *input, nn_real *output)
{
    if (net == NULL) {
        return;
    }

    invalidate_batch_cache(net);

    if (input == NULL) {
        return;
    }

    memcpy(net->activations[0], input, net->sizes[0] * sizeof(*input));

    for (size_t layer = 1; layer < net->layer_count; ++layer) {
        dense_forward_layer(net->activations[layer - 1],
                            net->weights[layer],
                            net->biases[layer],
                            net->activations[layer],
                            net->pre_activations[layer],
                            net->sizes[layer - 1],
                            net->sizes[layer],
                            activation_for_layer(net, layer));
    }

    if (output != NULL) {
        size_t last = net->layer_count - 1;
        memcpy(output, net->activations[last], net->sizes[last] * sizeof(*output));
    }
}

void nn_forward_batch(NeuralNet *net,
                      const nn_real *inputs,
                      nn_real *outputs,
                      size_t sample_count)
{
    if (net == NULL) {
        return;
    }

    if (inputs == NULL || outputs == NULL || sample_count == 0) {
        invalidate_batch_cache(net);
        return;
    }

    size_t input_width = nn_input_size(net);
    size_t output_width = nn_output_size(net);

    if (ensure_batch_workspace(net, sample_count)) {
        load_batch_inputs(net, inputs, NULL, 0, sample_count, input_width);
        forward_batch_workspace(net, sample_count);
        memcpy(outputs,
               net->batch_activations[net->layer_count - 1],
               sample_count * output_width * sizeof(*outputs));
        mark_batch_cache_valid(net, sample_count);
        return;
    }

    invalidate_batch_cache(net);
    for (size_t sample = 0; sample < sample_count; ++sample) {
        nn_forward(net, &inputs[sample * input_width], &outputs[sample * output_width]);
    }
}

int nn_score_action_rows(NeuralNet *net,
                         const nn_real *candidate_inputs,
                         size_t candidate_count,
                         nn_real *scores,
                         nn_real *probabilities)
{
    if (net == NULL || candidate_inputs == NULL || candidate_count == 0 || scores == NULL ||
        nn_output_size(net) != 1) {
        if (net != NULL) {
            invalidate_batch_cache(net);
        }
        return 0;
    }

    nn_forward_batch(net, candidate_inputs, scores, candidate_count);
    if (!net->batch_cache_valid || net->batch_cache_sample_count != candidate_count) {
        return 0;
    }

    if (probabilities != NULL) {
        nn_softmax(scores, probabilities, candidate_count);
    }
    return 1;
}

int nn_score_action_rows_eml(NeuralNet *net,
                             const nn_real *candidate_inputs,
                             size_t candidate_count,
                             size_t feature_count,
                             nn_real *scores,
                             nn_real *probabilities)
{
    if (net == NULL || candidate_inputs == NULL || candidate_count == 0 ||
        scores == NULL || feature_count < 2) {
        return 0;
    }

    size_t enriched_count = feature_count + feature_count / 2;
    if (nn_input_size(net) != enriched_count) {
        return 0;
    }

    /* allocate enriched feature buffer */
    nn_real *enriched = malloc(candidate_count * enriched_count * sizeof(nn_real));
    if (enriched == NULL) return 0;

    for (size_t i = 0; i < candidate_count; ++i) {
        nn_eml_enrich_features(candidate_inputs + i * feature_count,
                               feature_count,
                               enriched + i * enriched_count);
    }

    int ok = nn_score_action_rows(net, enriched, candidate_count, scores, probabilities);
    free(enriched);
    return ok;
}

const char *nn_scorer_envelope_feature_set(void)
{
    return NN_SCORER_ENVELOPE_FEATURE_SET;
}

const char *nn_scorer_envelope_slot_name(size_t slot)
{
    switch (slot) {
    case NN_SCORER_ENVELOPE_F_BIAS:
        return "bias";
    case NN_SCORER_ENVELOPE_F_ROLE_RULE:
        return "role_rule";
    case NN_SCORER_ENVELOPE_F_DOMAIN_PRIME_RULE:
        return "domain_prime_rule_propose";
    case NN_SCORER_ENVELOPE_F_ROLE_ACTION:
        return "role_action";
    case NN_SCORER_ENVELOPE_F_DOMAIN_PRIME_ACTION:
        return "domain_prime_gap_action";
    case NN_SCORER_ENVELOPE_F_CANDIDATE_INDEX:
        return "normalized_candidate_index";
    case NN_SCORER_ENVELOPE_F_CANDIDATE_COUNT:
        return "normalized_candidate_count";
    case NN_SCORER_ENVELOPE_F_DOMAIN_GRID_ACTION:
        return "domain_grid_path_action";
    case NN_SCORER_ENVELOPE_F_DOMAIN_CLAIM_ACTION:
        return "domain_delayed_claim_action";
    case NN_SCORER_ENVELOPE_F_DOMAIN_ARC_ACTION:
        return "domain_arc_agi_action";
    default:
        break;
    }
    return NULL;
}

int nn_scorer_envelope_domain_slot(NNScorerDomain domain, size_t *out_slot)
{
    if (out_slot == NULL) {
        return 0;
    }
    switch (domain) {
    case NN_SCORER_DOMAIN_PRIME_GAP_ACTION:
        *out_slot = (size_t)NN_SCORER_ENVELOPE_F_DOMAIN_PRIME_ACTION;
        return 1;
    case NN_SCORER_DOMAIN_GRID_PATH_ACTION:
        *out_slot = (size_t)NN_SCORER_ENVELOPE_F_DOMAIN_GRID_ACTION;
        return 1;
    case NN_SCORER_DOMAIN_DELAYED_CLAIM_ACTION:
        *out_slot = (size_t)NN_SCORER_ENVELOPE_F_DOMAIN_CLAIM_ACTION;
        return 1;
    case NN_SCORER_DOMAIN_ARC_AGI_ACTION:
        *out_slot = (size_t)NN_SCORER_ENVELOPE_F_DOMAIN_ARC_ACTION;
        return 1;
    case NN_SCORER_DOMAIN_PRIME_RULE_PROPOSE:
        *out_slot = (size_t)NN_SCORER_ENVELOPE_F_DOMAIN_PRIME_RULE;
        return 1;
    case NN_SCORER_DOMAIN_GO_POSITION_ACTION:
        *out_slot = (size_t)NN_SCORER_ENVELOPE_F_DOMAIN_GO_ACTION;
        return 1;
    }
    return 0;
}

int nn_scorer_envelope_fill_base(nn_real *row,
                                 size_t row_capacity,
                                 const NNScorerEnvelopeRow *spec)
{
    if (row == NULL || spec == NULL ||
        row_capacity < (size_t)NN_SCORER_ENVELOPE_FEATURE_COUNT ||
        spec->candidate_count == 0 ||
        spec->candidate_index >= spec->candidate_count) {
        return 0;
    }

    memset(row, 0, row_capacity * sizeof(*row));
    row[NN_SCORER_ENVELOPE_F_BIAS] = (nn_real)1;
    if (spec->include_role_tag) {
        switch (spec->role) {
        case NN_SCORER_ROLE_ACTION:
            row[NN_SCORER_ENVELOPE_F_ROLE_ACTION] = (nn_real)1;
            break;
        case NN_SCORER_ROLE_RULE:
            row[NN_SCORER_ENVELOPE_F_ROLE_RULE] = (nn_real)1;
            break;
        default:
            return 0;
        }
    }
    if (spec->include_domain_tag) {
        size_t domain_slot = 0;
        if (!nn_scorer_envelope_domain_slot(spec->domain, &domain_slot) ||
            domain_slot >= row_capacity) {
            return 0;
        }
        row[domain_slot] = (nn_real)1;
    }

    row[NN_SCORER_ENVELOPE_F_CANDIDATE_INDEX] =
        spec->candidate_count <= 1
            ? (nn_real)0
            : (nn_real)((double)spec->candidate_index / (double)(spec->candidate_count - 1u));
    size_t scale = spec->candidate_count_scale == 0 ? spec->candidate_count
                                                    : spec->candidate_count_scale;
    row[NN_SCORER_ENVELOPE_F_CANDIDATE_COUNT] =
        (nn_real)((double)spec->candidate_count / (double)scale);
    return 1;
}

int nn_scorer_envelope_fill_eml(nn_real *row,
                                size_t row_capacity,
                                const NNScorerEnvelopeRow *spec)
{
    /* fill the base 96 features first */
    if (!nn_scorer_envelope_fill_base(row, row_capacity, spec)) {
        return 0;
    }
    if (row_capacity < NN_SCORER_ENVELOPE_EML_FEATURE_COUNT) {
        return 0;
    }
    /* append 48 pairwise EML interaction features in slots 96..143 */
    nn_eml_pairwise(row, row + NN_SCORER_ENVELOPE_FEATURE_COUNT,
                    NN_SCORER_ENVELOPE_FEATURE_COUNT / 2);
    return 1;
}

/* Fill envelope with EML enrichment plus 4 memory-neuron context features.
 * 96 base + 48 EML pairwise + 4 memory context = 148 total.
 * row_capacity must be at least 148. memory_context may be NULL for zeros. */
int nn_scorer_envelope_fill_memory(nn_real *row,
                                   size_t row_capacity,
                                   const NNScorerEnvelopeRow *spec,
                                   const nn_real *memory_context)
{
    if (!nn_scorer_envelope_fill_base(row, row_capacity, spec)) {
        return 0;
    }
    if (row_capacity < NN_SCORER_ENVELOPE_MEMORY_FEATURE_COUNT) {
        return 0;
    }
    /* append 48 pairwise EML interaction features in slots 96..143 */
    nn_eml_pairwise(row, row + NN_SCORER_ENVELOPE_FEATURE_COUNT,
                    NN_SCORER_ENVELOPE_FEATURE_COUNT / 2);
    /* append 4 memory context features in slots 144..147 */
    if (memory_context != NULL) {
        for (size_t i = 0; i < NN_SCORER_ENVELOPE_MEMORY_CONTEXT_COUNT; i++) {
            row[NN_SCORER_ENVELOPE_EML_FEATURE_COUNT + i] = memory_context[i];
        }
    } else {
        for (size_t i = 0; i < NN_SCORER_ENVELOPE_MEMORY_CONTEXT_COUNT; i++) {
            row[NN_SCORER_ENVELOPE_EML_FEATURE_COUNT + i] = (nn_real)0.0;
        }
    }
    return 1;
}

static int rank_score_is_better(const nn_real *scores,
                                const uint32_t *action_ids,
                                size_t left,
                                size_t right,
                                nn_real score_epsilon)
{
    double delta = (double)scores[left] - (double)scores[right];
    double epsilon = score_epsilon < (nn_real)0 ? 0.0 : (double)score_epsilon;
    if (delta > epsilon) {
        return 1;
    }
    if (delta < -epsilon) {
        return 0;
    }

    uint32_t left_id = action_ids == NULL ? (uint32_t)left : action_ids[left];
    uint32_t right_id = action_ids == NULL ? (uint32_t)right : action_ids[right];
    if (left_id != right_id) {
        return left_id < right_id;
    }
    return left < right;
}

int nn_rank_action_scores_with_ids(const nn_real *scores,
                                   const uint32_t *action_ids,
                                   size_t count,
                                   size_t *ranked_indices,
                                   nn_real score_epsilon)
{
    if (scores == NULL || ranked_indices == NULL || count == 0) {
        return 0;
    }

    for (size_t i = 0; i < count; ++i) {
        if (!isfinite((double)scores[i])) {
            return 0;
        }
        ranked_indices[i] = i;
    }

    for (size_t i = 0; i < count; ++i) {
        size_t best = i;
        for (size_t j = i + 1; j < count; ++j) {
            if (rank_score_is_better(scores,
                                     action_ids,
                                     ranked_indices[j],
                                     ranked_indices[best],
                                     score_epsilon)) {
                best = j;
            }
        }

        size_t tmp = ranked_indices[i];
        ranked_indices[i] = ranked_indices[best];
        ranked_indices[best] = tmp;
    }

    return 1;
}

int nn_rank_action_scores(const nn_real *scores, size_t count, size_t *ranked_indices)
{
    return nn_rank_action_scores_with_ids(scores, NULL, count, ranked_indices, (nn_real)0);
}

nn_real nn_action_score_margin(const nn_real *scores, size_t count, size_t selected_index)
{
    if (scores == NULL || count == 0 || selected_index >= count ||
        !isfinite((double)scores[selected_index])) {
        return (nn_real)NAN;
    }

    if (count == 1) {
        return (nn_real)0;
    }

    int found_other = 0;
    nn_real best_other = (nn_real)0;
    for (size_t i = 0; i < count; ++i) {
        if (i == selected_index) {
            continue;
        }
        if (!isfinite((double)scores[i])) {
            return (nn_real)NAN;
        }
        if (!found_other || scores[i] > best_other) {
            best_other = scores[i];
            found_other = 1;
        }
    }
    return found_other ? scores[selected_index] - best_other : (nn_real)0;
}

nn_real nn_train_sample(NeuralNet *net,
                        const nn_real *input,
                        const nn_real *target,
                        nn_real learning_rate)
{
    if (net == NULL || input == NULL || target == NULL || learning_rate <= (nn_real)0) {
        return (nn_real)NAN;
    }

    nn_forward(net, input, NULL);
    nn_real loss = compute_deltas_and_loss(net, target);
    eml_pairwise_backprop_post(net);
    apply_sample_update(net, learning_rate);
    return loss;
}

nn_real nn_train_batch(NeuralNet *net,
                       const nn_real *inputs,
                       const nn_real *targets,
                       size_t batch_size,
                       nn_real learning_rate)
{
    if (net == NULL || inputs == NULL || targets == NULL || batch_size == 0 ||
        learning_rate <= (nn_real)0) {
        return (nn_real)NAN;
    }

    return train_indexed_batch(net,
                               inputs,
                               targets,
                               NULL,
                               0,
                               batch_size,
                               nn_input_size(net),
                               nn_output_size(net),
                               learning_rate);
}

static nn_real train_softmax_scorer_from_workspace(NeuralNet *net,
                                                   size_t candidate_count,
                                                   size_t selected_index,
                                                   nn_real weight,
                                                   nn_real learning_rate)
{
    if (weight == (nn_real)0) {
        return (nn_real)0;
    }

    if (!ensure_gradient_buffers(net)) {
        return (nn_real)NAN;
    }

    size_t output_layer = net->layer_count - 1;
    const nn_real *scores = net->batch_activations[output_layer];
    nn_real *output_deltas = net->batch_deltas[output_layer];
    nn_real loss =
        weight * softmax_probabilities_and_index_loss(scores,
                                                      output_deltas,
                                                      candidate_count,
                                                      selected_index,
                                                      NULL);

    if (net->output_activation == NN_ACTIVATION_LINEAR) {
        for (size_t sample = 0; sample < candidate_count; ++sample) {
            nn_real target = sample == selected_index ? (nn_real)1 : (nn_real)0;
            output_deltas[sample] = weight * (output_deltas[sample] - target);
        }
    } else {
        for (size_t sample = 0; sample < candidate_count; ++sample) {
            nn_real target = sample == selected_index ? (nn_real)1 : (nn_real)0;
            nn_real score = scores[sample];
            output_deltas[sample] = weight * (output_deltas[sample] - target) *
                                    activation_derivative_from_activation(score,
                                                                          net->output_activation);
        }
    }

    for (size_t layer = net->layer_count - 2; layer > 0; --layer) {
        size_t current = net->sizes[layer];
        size_t next = net->sizes[layer + 1];
        dense_backprop_deltas_batch(net->batch_deltas[layer + 1],
                                    net->weights[layer + 1],
                                    net->batch_activations[layer],
                                    net->batch_deltas[layer],
                                    candidate_count,
                                    current,
                                    next,
                                    activation_for_layer(net, layer));
    }

    apply_batch_gradients(net, candidate_count, learning_rate);
    return loss;
}

static nn_real train_pairwise_ranker_from_workspace(NeuralNet *net,
                                                    size_t candidate_count,
                                                    size_t target_index,
                                                    nn_real margin,
                                                    nn_real learning_rate)
{
    if (!ensure_gradient_buffers(net)) {
        return (nn_real)NAN;
    }

    size_t output_layer = net->layer_count - 1;
    const nn_real *scores = net->batch_activations[output_layer];
    nn_real *output_deltas = net->batch_deltas[output_layer];
    memset(output_deltas, 0, candidate_count * sizeof(*output_deltas));

    nn_real loss = (nn_real)0;
    nn_real target_score = scores[target_index];
    nn_real scale = (nn_real)1 / (nn_real)(candidate_count - 1);
    for (size_t sample = 0; sample < candidate_count; ++sample) {
        if (sample == target_index) {
            continue;
        }

        nn_real violation = margin - target_score + scores[sample];
        if (violation <= (nn_real)0) {
            continue;
        }

        loss += violation;
        output_deltas[target_index] -= scale;
        output_deltas[sample] += scale;
    }

    for (size_t sample = 0; sample < candidate_count; ++sample) {
        output_deltas[sample] *= activation_derivative_from_activation(
            scores[sample], net->output_activation);
    }

    for (size_t layer = net->layer_count - 2; layer > 0; --layer) {
        size_t current = net->sizes[layer];
        size_t next = net->sizes[layer + 1];
        dense_backprop_deltas_batch(net->batch_deltas[layer + 1],
                                    net->weights[layer + 1],
                                    net->batch_activations[layer],
                                    net->batch_deltas[layer],
                                    candidate_count,
                                    current,
                                    next,
                                    activation_for_layer(net, layer));
    }

    apply_batch_gradients(net, candidate_count, learning_rate);
    return loss * scale;
}

static nn_real nn_train_softmax_scorer_cached_raw(NeuralNet *net,
                                                  size_t candidate_count,
                                                  size_t selected_index,
                                                  nn_real weight,
                                                  nn_real learning_rate)
{
    if (net == NULL || candidate_count == 0 || selected_index >= candidate_count ||
        learning_rate <= (nn_real)0 || nn_output_size(net) != 1 ||
        net->output_activation != NN_ACTIVATION_LINEAR ||
        !net->batch_cache_valid || net->batch_cache_sample_count != candidate_count ||
        net->batch_capacity < candidate_count || net->batch_activations == NULL ||
        net->batch_deltas == NULL) {
        return (nn_real)NAN;
    }

    return train_softmax_scorer_from_workspace(
        net, candidate_count, selected_index, weight, learning_rate);
}

nn_real nn_train_softmax_scorer_cached(NeuralNet *net,
                                       size_t candidate_count,
                                       size_t selected_index,
                                       nn_real weight,
                                       nn_real learning_rate)
{
    if (nn_numeric_certificate_path() == NULL) {
        return nn_train_softmax_scorer_cached_raw(
            net, candidate_count, selected_index, weight, learning_rate);
    }

    if (net == NULL || candidate_count == 0 || selected_index >= candidate_count ||
        learning_rate <= (nn_real)0 || nn_output_size(net) != 1 ||
        net->output_activation != NN_ACTIVATION_LINEAR ||
        !net->batch_cache_valid || net->batch_cache_sample_count != candidate_count ||
        net->batch_capacity < candidate_count || net->batch_activations == NULL ||
        net->batch_activations[0] == NULL || weight == (nn_real)0) {
        return nn_train_softmax_scorer_cached_raw(
            net, candidate_count, selected_index, weight, learning_rate);
    }

    size_t input_width = nn_input_size(net);
    size_t input_count = candidate_count * input_width;
    nn_real *candidate_inputs = (nn_real *)malloc(input_count * sizeof(*candidate_inputs));
    if (candidate_inputs == NULL) {
        return (nn_real)NAN;
    }
    memcpy(candidate_inputs, net->batch_activations[0], input_count * sizeof(*candidate_inputs));

    NNNumericGuardConfig guard;
    nn_numeric_guard_config_from_env(&guard);
    NNTrainingStepCertificate certificate;
    int committed = nn_train_softmax_scorer_weighted_guarded(net,
                                                             candidate_inputs,
                                                             candidate_count,
                                                             selected_index,
                                                             weight,
                                                             learning_rate,
                                                             &guard,
                                                             &certificate);
    free(candidate_inputs);
    if (!nn_numeric_certificate_write_env(&certificate)) {
        return (nn_real)NAN;
    }
    return committed ? certificate.optimizer_loss : (nn_real)NAN;
}

static nn_real nn_train_softmax_scorer_policy_regularized_cached_raw(
    NeuralNet *net,
    size_t candidate_count,
    size_t selected_index,
    nn_real policy_weight,
    const nn_real *reference_probabilities,
    nn_real reference_weight,
    nn_real entropy_weight,
    nn_real learning_rate)
{
    if (net == NULL || candidate_count == 0 || selected_index >= candidate_count ||
        learning_rate <= (nn_real)0 || nn_output_size(net) != 1 ||
        net->output_activation != NN_ACTIVATION_LINEAR ||
        reference_weight < (nn_real)0 || entropy_weight < (nn_real)0 ||
        (reference_weight != (nn_real)0 && reference_probabilities == NULL) ||
        !net->batch_cache_valid || net->batch_cache_sample_count != candidate_count ||
        net->batch_capacity < candidate_count || net->batch_activations == NULL ||
        net->batch_deltas == NULL) {
        return (nn_real)NAN;
    }

    if (policy_weight == (nn_real)0 && reference_weight == (nn_real)0 &&
        entropy_weight == (nn_real)0) {
        return (nn_real)0;
    }

    if (!ensure_gradient_buffers(net)) {
        return (nn_real)NAN;
    }

    size_t output_layer = net->layer_count - 1;
    const nn_real *scores = net->batch_activations[output_layer];
    nn_real *output_deltas = net->batch_deltas[output_layer];
    nn_real log_denominator = (nn_real)0;
    nn_real selected_loss = softmax_probabilities_and_index_loss(scores,
                                                                 output_deltas,
                                                                 candidate_count,
                                                                 selected_index,
                                                                 &log_denominator);

    nn_real loss = (nn_real)0;
    if (policy_weight != (nn_real)0) {
        loss += policy_weight * selected_loss;
    }
    if (reference_weight != (nn_real)0) {
        loss += reference_weight *
                softmax_cross_entropy_from_log_denominator(scores,
                                                           reference_probabilities,
                                                           candidate_count,
                                                           log_denominator);
    }

    nn_real entropy_loss = (nn_real)0;
    if (entropy_weight != (nn_real)0) {
        for (size_t sample = 0; sample < candidate_count; ++sample) {
            nn_real probability = output_deltas[sample];
            entropy_loss += probability * safe_log_probability(probability);
        }
        loss += entropy_weight * entropy_loss;
    }

    for (size_t sample = 0; sample < candidate_count; ++sample) {
        nn_real probability = output_deltas[sample];
        nn_real delta = (nn_real)0;

        if (policy_weight != (nn_real)0) {
            nn_real target = sample == selected_index ? (nn_real)1 : (nn_real)0;
            delta += policy_weight * (probability - target);
        }
        if (reference_weight != (nn_real)0) {
            delta += reference_weight * (probability - reference_probabilities[sample]);
        }
        if (entropy_weight != (nn_real)0) {
            delta += entropy_weight * probability *
                     (safe_log_probability(probability) - entropy_loss);
        }

        if (net->output_activation == NN_ACTIVATION_LINEAR) {
            output_deltas[sample] = delta;
        } else {
            output_deltas[sample] =
                delta * activation_derivative_from_activation(scores[sample],
                                                              net->output_activation);
        }
    }

    for (size_t layer = net->layer_count - 2; layer > 0; --layer) {
        size_t current = net->sizes[layer];
        size_t next = net->sizes[layer + 1];
        dense_backprop_deltas_batch(net->batch_deltas[layer + 1],
                                    net->weights[layer + 1],
                                    net->batch_activations[layer],
                                    net->batch_deltas[layer],
                                    candidate_count,
                                    current,
                                    next,
                                    activation_for_layer(net, layer));
    }

    apply_batch_gradients(net, candidate_count, learning_rate);
    return loss;
}

nn_real nn_train_softmax_scorer_policy_regularized_cached(
    NeuralNet *net,
    size_t candidate_count,
    size_t selected_index,
    nn_real policy_weight,
    const nn_real *reference_probabilities,
    nn_real reference_weight,
    nn_real entropy_weight,
    nn_real learning_rate)
{
    if (nn_numeric_certificate_path() == NULL ||
        (policy_weight == (nn_real)0 && reference_weight == (nn_real)0 &&
         entropy_weight == (nn_real)0)) {
        return nn_train_softmax_scorer_policy_regularized_cached_raw(net,
                                                                     candidate_count,
                                                                     selected_index,
                                                                     policy_weight,
                                                                     reference_probabilities,
                                                                     reference_weight,
                                                                     entropy_weight,
                                                                     learning_rate);
    }

    if (net == NULL || candidate_count == 0 || selected_index >= candidate_count ||
        learning_rate <= (nn_real)0 || nn_output_size(net) != 1 ||
        net->output_activation != NN_ACTIVATION_LINEAR ||
        reference_weight < (nn_real)0 || entropy_weight < (nn_real)0 ||
        (reference_weight != (nn_real)0 && reference_probabilities == NULL) ||
        !net->batch_cache_valid || net->batch_cache_sample_count != candidate_count ||
        net->batch_capacity < candidate_count || net->batch_activations == NULL ||
        net->batch_activations[0] == NULL) {
        return nn_train_softmax_scorer_policy_regularized_cached_raw(net,
                                                                     candidate_count,
                                                                     selected_index,
                                                                     policy_weight,
                                                                     reference_probabilities,
                                                                     reference_weight,
                                                                     entropy_weight,
                                                                     learning_rate);
    }

    size_t input_width = nn_input_size(net);
    size_t input_count = candidate_count * input_width;
    nn_real *candidate_inputs = (nn_real *)malloc(input_count * sizeof(*candidate_inputs));
    if (candidate_inputs == NULL) {
        return (nn_real)NAN;
    }
    memcpy(candidate_inputs, net->batch_activations[0], input_count * sizeof(*candidate_inputs));

    NNNumericGuardConfig guard;
    nn_numeric_guard_config_from_env(&guard);
    NNTrainingStepCertificate certificate;
    int committed = nn_train_softmax_scorer_policy_regularized_guarded(net,
                                                                       candidate_inputs,
                                                                       candidate_count,
                                                                       selected_index,
                                                                       policy_weight,
                                                                       reference_probabilities,
                                                                       reference_weight,
                                                                       entropy_weight,
                                                                       learning_rate,
                                                                       &guard,
                                                                       &certificate);
    free(candidate_inputs);
    if (!nn_numeric_certificate_write_env(&certificate)) {
        return (nn_real)NAN;
    }
    return committed ? certificate.optimizer_loss : (nn_real)NAN;
}

int nn_train_softmax_scorer_policy_regularized_guarded(
    NeuralNet *net,
    const nn_real *candidate_inputs,
    size_t candidate_count,
    size_t selected_index,
    nn_real policy_weight,
    const nn_real *reference_probabilities,
    nn_real reference_weight,
    nn_real entropy_weight,
    nn_real learning_rate,
    const NNNumericGuardConfig *guard,
    NNTrainingStepCertificate *certificate)
{
    NNNumericGuardConfig default_guard;
    NNNumericGuardConfig active_guard;
    nn_real *before_scores = NULL;
    nn_real *after_scores = NULL;
    NeuralNet *candidate = NULL;
    int status = 0;

    if (certificate != NULL) {
        nn_training_step_certificate_init(certificate);
        certificate->attempted_learning_rate = learning_rate;
        certificate->candidate_count = candidate_count;
        certificate->selected_index = selected_index;
        certificate->parameter_count = nn_parameter_count(net);
    }

    if (net == NULL || candidate_inputs == NULL || candidate_count == 0 ||
        selected_index >= candidate_count || learning_rate <= (nn_real)0 ||
        nn_output_size(net) != 1 || net->output_activation != NN_ACTIVATION_LINEAR ||
        reference_weight < (nn_real)0 || entropy_weight < (nn_real)0 ||
        (reference_weight != (nn_real)0 && reference_probabilities == NULL) ||
        (policy_weight == (nn_real)0 && reference_weight == (nn_real)0 &&
         entropy_weight == (nn_real)0)) {
        if (certificate != NULL) {
            certificate->rejection_reason = NN_NUMERIC_STEP_REJECT_INVALID_ARGUMENT;
        }
        return 0;
    }

    if (guard == NULL) {
        nn_numeric_guard_config_init(&default_guard);
        active_guard = default_guard;
    } else {
        active_guard = *guard;
    }
    nn_numeric_guard_config_sanitize(&active_guard);

    before_scores = (nn_real *)calloc(candidate_count, sizeof(*before_scores));
    after_scores = (nn_real *)calloc(candidate_count, sizeof(*after_scores));
    if (before_scores == NULL || after_scores == NULL) {
        if (certificate != NULL) {
            certificate->rejection_reason = NN_NUMERIC_STEP_REJECT_ALLOC_FAILED;
        }
        goto done;
    }

    NNTrainingStepCertificate base;
    nn_training_step_certificate_init(&base);
    base.attempted_learning_rate = learning_rate;
    base.candidate_count = candidate_count;
    base.selected_index = selected_index;
    base.parameter_count = nn_parameter_count(net);
    nn_real ignored_loss = (nn_real)NAN;
    int before_score_finite = nn_score_metrics(net,
                                               candidate_inputs,
                                               candidate_count,
                                               selected_index,
                                               before_scores,
                                               &base.max_abs_logit_before,
                                               &base.selected_margin_before,
                                               &ignored_loss,
                                               (nn_real)1);
    base.loss_before = nn_policy_regularized_objective(before_scores,
                                                       candidate_count,
                                                       selected_index,
                                                       policy_weight,
                                                       reference_probabilities,
                                                       reference_weight,
                                                       entropy_weight);
    base.finite_before = nn_parameter_metrics(net, &base.max_abs_parameter_before) &&
                         before_score_finite && isfinite((double)base.loss_before);

    nn_real current_learning_rate = learning_rate;
    size_t max_attempt = active_guard.max_shrink_steps;
    for (size_t shrink_step = 0; shrink_step <= max_attempt; ++shrink_step) {
        NNTrainingStepCertificate attempt = base;
        attempt.shrink_steps = shrink_step;
        attempt.applied_learning_rate = current_learning_rate;
        attempt.rejection_reason = NN_NUMERIC_STEP_REJECT_NONE;
        attempt.step_committed = 0;

        candidate = nn_clone(net);
        if (candidate == NULL) {
            attempt.rejection_reason = NN_NUMERIC_STEP_REJECT_ALLOC_FAILED;
            if (certificate != NULL) {
                *certificate = attempt;
            }
            goto done;
        }

        nn_forward_batch(candidate, candidate_inputs, after_scores, candidate_count);
        nn_real update_loss =
            nn_train_softmax_scorer_policy_regularized_cached_raw(candidate,
                                                                  candidate_count,
                                                                  selected_index,
                                                                  policy_weight,
                                                                  reference_probabilities,
                                                                  reference_weight,
                                                                  entropy_weight,
                                                                  current_learning_rate);
        attempt.optimizer_loss = update_loss;
        nn_real post_ignored_loss = (nn_real)NAN;
        int after_score_finite = nn_score_metrics(candidate,
                                                  candidate_inputs,
                                                  candidate_count,
                                                  selected_index,
                                                  after_scores,
                                                  &attempt.max_abs_logit_after,
                                                  &attempt.selected_margin_after,
                                                  &post_ignored_loss,
                                                  (nn_real)1);
        attempt.loss_after = nn_policy_regularized_objective(after_scores,
                                                             candidate_count,
                                                             selected_index,
                                                             policy_weight,
                                                             reference_probabilities,
                                                             reference_weight,
                                                             entropy_weight);
        attempt.finite_after =
            nn_parameter_metrics(candidate, &attempt.max_abs_parameter_after) &&
            after_score_finite && isfinite((double)attempt.loss_after);
        attempt.max_abs_parameter_delta = nn_parameter_max_abs_delta(net, candidate);
        attempt.max_abs_logit_delta =
            nn_score_max_abs_delta(before_scores, after_scores, candidate_count);
        attempt.rejection_reason = nn_numeric_guard_check(&active_guard, &attempt);

        if (attempt.rejection_reason == NN_NUMERIC_STEP_REJECT_NONE) {
            if (!nn_copy_parameters(net, candidate)) {
                attempt.rejection_reason = NN_NUMERIC_STEP_REJECT_INVALID_ARGUMENT;
                if (certificate != NULL) {
                    *certificate = attempt;
                }
                goto done;
            }
            attempt.step_committed = 1;
            if (certificate != NULL) {
                *certificate = attempt;
            }
            status = 1;
            goto done;
        }

        if (certificate != NULL) {
            *certificate = attempt;
        }
        nn_free(candidate);
        candidate = NULL;
        current_learning_rate *= active_guard.shrink_factor;
    }

done:
    nn_free(candidate);
    free(before_scores);
    free(after_scores);
    return status;
}

static nn_real nn_train_softmax_scorer_weighted_raw(NeuralNet *net,
                                                    const nn_real *candidate_inputs,
                                                    size_t candidate_count,
                                                    size_t selected_index,
                                                    nn_real weight,
                                                    nn_real learning_rate)
{
    if (net == NULL || candidate_inputs == NULL || candidate_count == 0 ||
        selected_index >= candidate_count || learning_rate <= (nn_real)0 ||
        nn_output_size(net) != 1 || net->output_activation != NN_ACTIVATION_LINEAR) {
        return (nn_real)NAN;
    }

    invalidate_batch_cache(net);

    if (weight == (nn_real)0) {
        return (nn_real)0;
    }

    if (!ensure_batch_workspace(net, candidate_count)) {
        return (nn_real)NAN;
    }

    size_t input_width = nn_input_size(net);
    memcpy(net->batch_activations[0],
           candidate_inputs,
           candidate_count * input_width * sizeof(*candidate_inputs));
    forward_batch_workspace(net, candidate_count);
    return train_softmax_scorer_from_workspace(
        net, candidate_count, selected_index, weight, learning_rate);
}

nn_real nn_train_softmax_scorer_weighted(NeuralNet *net,
                                         const nn_real *candidate_inputs,
                                         size_t candidate_count,
                                         size_t selected_index,
                                         nn_real weight,
                                         nn_real learning_rate)
{
    if (nn_numeric_certificate_path() == NULL || weight == (nn_real)0) {
        return nn_train_softmax_scorer_weighted_raw(net,
                                                    candidate_inputs,
                                                    candidate_count,
                                                    selected_index,
                                                    weight,
                                                    learning_rate);
    }

    NNNumericGuardConfig guard;
    nn_numeric_guard_config_from_env(&guard);
    NNTrainingStepCertificate certificate;
    int committed = nn_train_softmax_scorer_weighted_guarded(net,
                                                             candidate_inputs,
                                                             candidate_count,
                                                             selected_index,
                                                             weight,
                                                             learning_rate,
                                                             &guard,
                                                             &certificate);
    if (!nn_numeric_certificate_write_env(&certificate)) {
        return (nn_real)NAN;
    }
    return committed ? certificate.optimizer_loss : (nn_real)NAN;
}

int nn_train_softmax_scorer_weighted_guarded(
    NeuralNet *net,
    const nn_real *candidate_inputs,
    size_t candidate_count,
    size_t selected_index,
    nn_real weight,
    nn_real learning_rate,
    const NNNumericGuardConfig *guard,
    NNTrainingStepCertificate *certificate)
{
    NNNumericGuardConfig default_guard;
    NNNumericGuardConfig active_guard;
    nn_real *before_scores = NULL;
    nn_real *after_scores = NULL;
    NeuralNet *candidate = NULL;
    int status = 0;

    if (certificate != NULL) {
        nn_training_step_certificate_init(certificate);
        certificate->attempted_learning_rate = learning_rate;
        certificate->candidate_count = candidate_count;
        certificate->selected_index = selected_index;
        certificate->parameter_count = nn_parameter_count(net);
    }

    if (net == NULL || candidate_inputs == NULL || candidate_count == 0 ||
        selected_index >= candidate_count || learning_rate <= (nn_real)0 ||
        nn_output_size(net) != 1 || net->output_activation != NN_ACTIVATION_LINEAR ||
        weight == (nn_real)0) {
        if (certificate != NULL) {
            certificate->rejection_reason = NN_NUMERIC_STEP_REJECT_INVALID_ARGUMENT;
        }
        return 0;
    }

    if (guard == NULL) {
        nn_numeric_guard_config_init(&default_guard);
        active_guard = default_guard;
    } else {
        active_guard = *guard;
    }
    nn_numeric_guard_config_sanitize(&active_guard);

    before_scores = (nn_real *)calloc(candidate_count, sizeof(*before_scores));
    after_scores = (nn_real *)calloc(candidate_count, sizeof(*after_scores));
    if (before_scores == NULL || after_scores == NULL) {
        if (certificate != NULL) {
            certificate->rejection_reason = NN_NUMERIC_STEP_REJECT_ALLOC_FAILED;
        }
        goto done;
    }

    NNTrainingStepCertificate base;
    nn_training_step_certificate_init(&base);
    base.attempted_learning_rate = learning_rate;
    base.candidate_count = candidate_count;
    base.selected_index = selected_index;
    base.parameter_count = nn_parameter_count(net);
    base.finite_before = nn_parameter_metrics(net, &base.max_abs_parameter_before) &&
                         nn_score_metrics(net,
                                          candidate_inputs,
                                          candidate_count,
                                          selected_index,
                                          before_scores,
                                          &base.max_abs_logit_before,
                                          &base.selected_margin_before,
                                          &base.loss_before,
                                          weight);

    nn_real current_learning_rate = learning_rate;
    size_t max_attempt = active_guard.max_shrink_steps;
    for (size_t shrink_step = 0; shrink_step <= max_attempt; ++shrink_step) {
        NNTrainingStepCertificate attempt = base;
        attempt.shrink_steps = shrink_step;
        attempt.applied_learning_rate = current_learning_rate;
        attempt.rejection_reason = NN_NUMERIC_STEP_REJECT_NONE;
        attempt.step_committed = 0;

        candidate = nn_clone(net);
        if (candidate == NULL) {
            attempt.rejection_reason = NN_NUMERIC_STEP_REJECT_ALLOC_FAILED;
            if (certificate != NULL) {
                *certificate = attempt;
            }
            goto done;
        }

        nn_real update_loss = nn_train_softmax_scorer_weighted_raw(candidate,
                                                                   candidate_inputs,
                                                                   candidate_count,
                                                                   selected_index,
                                                                   weight,
                                                                   current_learning_rate);
        attempt.optimizer_loss = update_loss;
        attempt.finite_after =
            nn_parameter_metrics(candidate, &attempt.max_abs_parameter_after) &&
            nn_score_metrics(candidate,
                             candidate_inputs,
                             candidate_count,
                             selected_index,
                             after_scores,
                             &attempt.max_abs_logit_after,
                             &attempt.selected_margin_after,
                             &attempt.loss_after,
                             weight);
        attempt.max_abs_parameter_delta = nn_parameter_max_abs_delta(net, candidate);
        attempt.max_abs_logit_delta =
            nn_score_max_abs_delta(before_scores, after_scores, candidate_count);
        attempt.rejection_reason = nn_numeric_guard_check(&active_guard, &attempt);

        if (attempt.rejection_reason == NN_NUMERIC_STEP_REJECT_NONE) {
            if (!nn_copy_parameters(net, candidate)) {
                attempt.rejection_reason = NN_NUMERIC_STEP_REJECT_INVALID_ARGUMENT;
                if (certificate != NULL) {
                    *certificate = attempt;
                }
                goto done;
            }
            attempt.step_committed = 1;
            if (certificate != NULL) {
                *certificate = attempt;
            }
            status = 1;
            goto done;
        }

        if (certificate != NULL) {
            *certificate = attempt;
        }
        nn_free(candidate);
        candidate = NULL;
        current_learning_rate *= active_guard.shrink_factor;
    }

done:
    nn_free(candidate);
    free(before_scores);
    free(after_scores);
    return status;
}

nn_real nn_train_softmax_scorer(NeuralNet *net,
                                const nn_real *candidate_inputs,
                                size_t candidate_count,
                                size_t target_index,
                                nn_real learning_rate)
{
    return nn_train_action_scorer(net,
                                  candidate_inputs,
                                  candidate_count,
                                  target_index,
                                  learning_rate);
}

nn_real nn_train_action_scorer(NeuralNet *net,
                               const nn_real *candidate_inputs,
                               size_t candidate_count,
                               size_t target_index,
                               nn_real learning_rate)
{
    return nn_train_softmax_scorer_weighted(
        net, candidate_inputs, candidate_count, target_index, (nn_real)1, learning_rate);
}

nn_real nn_train_pairwise_action_ranker(NeuralNet *net,
                                        const nn_real *candidate_inputs,
                                        size_t candidate_count,
                                        size_t target_index,
                                        nn_real margin,
                                        nn_real learning_rate)
{
    if (net == NULL || candidate_inputs == NULL || candidate_count < 2 ||
        target_index >= candidate_count || margin <= (nn_real)0 ||
        learning_rate <= (nn_real)0 || nn_output_size(net) != 1 ||
        net->output_activation != NN_ACTIVATION_LINEAR) {
        return (nn_real)NAN;
    }

    invalidate_batch_cache(net);

    if (!ensure_batch_workspace(net, candidate_count)) {
        return (nn_real)NAN;
    }

    size_t input_width = nn_input_size(net);
    memcpy(net->batch_activations[0],
           candidate_inputs,
           candidate_count * input_width * sizeof(*candidate_inputs));
    forward_batch_workspace(net, candidate_count);
    return train_pairwise_ranker_from_workspace(
        net, candidate_count, target_index, margin, learning_rate);
}

static nn_real nn_train_raw(NeuralNet *net,
                            const nn_real *inputs,
                            const nn_real *targets,
                            size_t sample_count,
                            nn_real learning_rate,
                            size_t epochs,
                            int shuffle)
{
    if (net == NULL || inputs == NULL || targets == NULL || sample_count == 0 || epochs == 0 ||
        learning_rate <= (nn_real)0) {
        return (nn_real)NAN;
    }

    size_t *order = NULL;
    if (shuffle) {
        order = checked_calloc(sample_count, sizeof(*order));
        if (order == NULL) {
            return (nn_real)NAN;
        }

        for (size_t i = 0; i < sample_count; ++i) {
            order[i] = i;
        }
    }

    size_t input_width = nn_input_size(net);
    size_t target_width = nn_output_size(net);
    nn_real epoch_loss = 0;

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        if (shuffle) {
            for (size_t i = sample_count - 1; i > 0; --i) {
                size_t j = (size_t)rng_next(&net->rng_state) % (i + 1);
                size_t tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
            }
        }

        epoch_loss = 0;
        for (size_t sample = 0; sample < sample_count; ++sample) {
            size_t index = shuffle ? order[sample] : sample;
            epoch_loss += nn_train_sample(net,
                                          &inputs[index * input_width],
                                          &targets[index * target_width],
                                          learning_rate);
        }

        epoch_loss /= (nn_real)sample_count;
    }

    free(order);
    return epoch_loss;
}

static nn_real nn_train_batched_raw(NeuralNet *net,
                                    const nn_real *inputs,
                                    const nn_real *targets,
                                    size_t sample_count,
                                    size_t batch_size,
                                    nn_real learning_rate,
                                    size_t epochs,
                                    int shuffle)
{
    if (net == NULL || inputs == NULL || targets == NULL || sample_count == 0 || batch_size == 0 ||
        epochs == 0 || learning_rate <= (nn_real)0) {
        return (nn_real)NAN;
    }

    if (batch_size > sample_count) {
        batch_size = sample_count;
    }

    size_t *order = NULL;
    if (shuffle) {
        order = checked_calloc(sample_count, sizeof(*order));
        if (order == NULL) {
            return (nn_real)NAN;
        }

        for (size_t i = 0; i < sample_count; ++i) {
            order[i] = i;
        }
    }

    size_t input_width = nn_input_size(net);
    size_t target_width = nn_output_size(net);
    nn_real epoch_loss = 0;

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        if (shuffle) {
            for (size_t i = sample_count - 1; i > 0; --i) {
                size_t j = (size_t)rng_next(&net->rng_state) % (i + 1);
                size_t tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
            }
        }

        epoch_loss = 0;
        for (size_t start = 0; start < sample_count; start += batch_size) {
            size_t remaining = sample_count - start;
            size_t current_batch = remaining < batch_size ? remaining : batch_size;
            nn_real batch_loss = train_indexed_batch(net,
                                                     inputs,
                                                     targets,
                                                     order,
                                                     start,
                                                     current_batch,
                                                     input_width,
                                                     target_width,
                                                     learning_rate);
            epoch_loss += batch_loss * (nn_real)current_batch;
        }

        epoch_loss /= (nn_real)sample_count;
    }

    free(order);
    return epoch_loss;
}

static int nn_train_supervised_guarded_internal(
    NeuralNet *net,
    const nn_real *inputs,
    const nn_real *targets,
    size_t sample_count,
    size_t batch_size,
    nn_real learning_rate,
    size_t epochs,
    int shuffle,
    int use_batches,
    const NNNumericGuardConfig *guard,
    NNTrainingStepCertificate *certificate)
{
    NNNumericGuardConfig default_guard;
    NNNumericGuardConfig active_guard;
    nn_real *before_outputs = NULL;
    nn_real *after_outputs = NULL;
    NeuralNet *candidate = NULL;
    int status = 0;

    if (certificate != NULL) {
        nn_training_step_certificate_init(certificate);
        snprintf(certificate->step_kind,
                 sizeof(certificate->step_kind),
                 "%s",
                 "batch_supervised");
        certificate->attempted_learning_rate = learning_rate;
        certificate->candidate_count = sample_count;
        certificate->selected_index = 0u;
        certificate->parameter_count = nn_parameter_count(net);
        certificate->selected_margin_before = (nn_real)0;
        certificate->selected_margin_after = (nn_real)0;
    }

    if (net == NULL || inputs == NULL || targets == NULL || sample_count == 0 ||
        epochs == 0 || learning_rate <= (nn_real)0 || nn_output_size(net) == 0) {
        if (certificate != NULL) {
            certificate->rejection_reason = NN_NUMERIC_STEP_REJECT_INVALID_ARGUMENT;
        }
        return 0;
    }

    size_t output_width = nn_output_size(net);
    if (sample_count > SIZE_MAX / output_width) {
        if (certificate != NULL) {
            certificate->rejection_reason = NN_NUMERIC_STEP_REJECT_INVALID_ARGUMENT;
        }
        return 0;
    }
    size_t output_count = sample_count * output_width;

    if (guard == NULL) {
        nn_numeric_guard_config_init(&default_guard);
        active_guard = default_guard;
    } else {
        active_guard = *guard;
    }
    nn_numeric_guard_config_sanitize(&active_guard);

    before_outputs = (nn_real *)calloc(output_count, sizeof(*before_outputs));
    after_outputs = (nn_real *)calloc(output_count, sizeof(*after_outputs));
    if (before_outputs == NULL || after_outputs == NULL) {
        if (certificate != NULL) {
            certificate->rejection_reason = NN_NUMERIC_STEP_REJECT_ALLOC_FAILED;
        }
        goto done;
    }

    NNTrainingStepCertificate base;
    nn_training_step_certificate_init(&base);
    snprintf(base.step_kind, sizeof(base.step_kind), "%s", "batch_supervised");
    base.attempted_learning_rate = learning_rate;
    base.candidate_count = sample_count;
    base.selected_index = 0u;
    base.parameter_count = nn_parameter_count(net);
    base.selected_margin_before = (nn_real)0;
    base.selected_margin_after = (nn_real)0;
    int before_output_finite = nn_dataset_metrics(net,
                                                  inputs,
                                                  targets,
                                                  sample_count,
                                                  before_outputs,
                                                  &base.loss_before,
                                                  &base.max_abs_logit_before);
    base.finite_before = nn_parameter_metrics(net, &base.max_abs_parameter_before) &&
                         before_output_finite && isfinite((double)base.loss_before);

    nn_real current_learning_rate = learning_rate;
    size_t max_attempt = active_guard.max_shrink_steps;
    for (size_t shrink_step = 0; shrink_step <= max_attempt; ++shrink_step) {
        NNTrainingStepCertificate attempt = base;
        attempt.shrink_steps = shrink_step;
        attempt.applied_learning_rate = current_learning_rate;
        attempt.rejection_reason = NN_NUMERIC_STEP_REJECT_NONE;
        attempt.step_committed = 0;

        candidate = nn_clone(net);
        if (candidate == NULL) {
            attempt.rejection_reason = NN_NUMERIC_STEP_REJECT_ALLOC_FAILED;
            if (certificate != NULL) {
                *certificate = attempt;
            }
            goto done;
        }

        nn_real update_loss = use_batches
                                  ? nn_train_batched_raw(candidate,
                                                         inputs,
                                                         targets,
                                                         sample_count,
                                                         batch_size,
                                                         current_learning_rate,
                                                         epochs,
                                                         shuffle)
                                  : nn_train_raw(candidate,
                                                 inputs,
                                                 targets,
                                                 sample_count,
                                                 current_learning_rate,
                                                 epochs,
                                                 shuffle);
        attempt.optimizer_loss = update_loss;
        int after_output_finite = nn_dataset_metrics(candidate,
                                                    inputs,
                                                    targets,
                                                    sample_count,
                                                    after_outputs,
                                                    &attempt.loss_after,
                                                    &attempt.max_abs_logit_after);
        attempt.finite_after =
            nn_parameter_metrics(candidate, &attempt.max_abs_parameter_after) &&
            after_output_finite && isfinite((double)attempt.loss_after);
        attempt.max_abs_parameter_delta = nn_parameter_max_abs_delta(net, candidate);
        attempt.max_abs_logit_delta =
            nn_score_max_abs_delta(before_outputs, after_outputs, output_count);
        attempt.selected_margin_before = (nn_real)0;
        attempt.selected_margin_after = (nn_real)0;
        attempt.rejection_reason = nn_numeric_guard_check(&active_guard, &attempt);

        if (attempt.rejection_reason == NN_NUMERIC_STEP_REJECT_NONE) {
            if (!nn_copy_parameters(net, candidate)) {
                attempt.rejection_reason = NN_NUMERIC_STEP_REJECT_INVALID_ARGUMENT;
                if (certificate != NULL) {
                    *certificate = attempt;
                }
                goto done;
            }
            attempt.step_committed = 1;
            if (certificate != NULL) {
                *certificate = attempt;
            }
            status = 1;
            goto done;
        }

        if (certificate != NULL) {
            *certificate = attempt;
        }
        nn_free(candidate);
        candidate = NULL;
        current_learning_rate *= active_guard.shrink_factor;
    }

done:
    nn_free(candidate);
    free(before_outputs);
    free(after_outputs);
    return status;
}

nn_real nn_train(NeuralNet *net,
                 const nn_real *inputs,
                 const nn_real *targets,
                 size_t sample_count,
                 nn_real learning_rate,
                 size_t epochs,
                 int shuffle)
{
    if (nn_numeric_certificate_path() == NULL) {
        return nn_train_raw(net, inputs, targets, sample_count, learning_rate, epochs, shuffle);
    }

    return nn_train_batched(net, inputs, targets, sample_count, 1u, learning_rate, epochs, shuffle);
}

nn_real nn_train_batched(NeuralNet *net,
                         const nn_real *inputs,
                         const nn_real *targets,
                         size_t sample_count,
                         size_t batch_size,
                         nn_real learning_rate,
                         size_t epochs,
                         int shuffle)
{
    if (nn_numeric_certificate_path() == NULL) {
        return nn_train_batched_raw(net,
                                    inputs,
                                    targets,
                                    sample_count,
                                    batch_size,
                                    learning_rate,
                                    epochs,
                                    shuffle);
    }

    if (net == NULL || inputs == NULL || targets == NULL || sample_count == 0 || batch_size == 0 ||
        epochs == 0 || learning_rate <= (nn_real)0) {
        return (nn_real)NAN;
    }

    if (batch_size > sample_count) {
        batch_size = sample_count;
    }

    size_t input_width = nn_input_size(net);
    size_t target_width = nn_output_size(net);
    if (input_width == 0 || target_width == 0 || batch_size > SIZE_MAX / input_width ||
        batch_size > SIZE_MAX / target_width) {
        return (nn_real)NAN;
    }

    size_t *order = NULL;
    if (shuffle) {
        order = checked_calloc(sample_count, sizeof(*order));
        if (order == NULL) {
            return (nn_real)NAN;
        }
        for (size_t i = 0; i < sample_count; ++i) {
            order[i] = i;
        }
    }

    nn_real *batch_inputs =
        checked_calloc(batch_size * input_width, sizeof(*batch_inputs));
    nn_real *batch_targets =
        checked_calloc(batch_size * target_width, sizeof(*batch_targets));
    if (batch_inputs == NULL || batch_targets == NULL) {
        free(order);
        free(batch_inputs);
        free(batch_targets);
        return (nn_real)NAN;
    }

    NNNumericGuardConfig guard;
    nn_numeric_guard_config_from_env(&guard);
    nn_real epoch_loss = (nn_real)0;

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        if (shuffle) {
            for (size_t i = sample_count - 1; i > 0; --i) {
                size_t j = (size_t)rng_next(&net->rng_state) % (i + 1);
                size_t tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
            }
        }

        epoch_loss = (nn_real)0;
        for (size_t start = 0; start < sample_count; start += batch_size) {
            size_t remaining = sample_count - start;
            size_t current_batch = remaining < batch_size ? remaining : batch_size;
            for (size_t sample = 0; sample < current_batch; ++sample) {
                size_t index = shuffle ? order[start + sample] : start + sample;
                memcpy(&batch_inputs[sample * input_width],
                       &inputs[index * input_width],
                       input_width * sizeof(*batch_inputs));
                memcpy(&batch_targets[sample * target_width],
                       &targets[index * target_width],
                       target_width * sizeof(*batch_targets));
            }

            NNTrainingStepCertificate certificate;
            int committed = nn_train_supervised_guarded_internal(net,
                                                                 batch_inputs,
                                                                 batch_targets,
                                                                 current_batch,
                                                                 current_batch,
                                                                 learning_rate,
                                                                 1u,
                                                                 0,
                                                                 1,
                                                                 &guard,
                                                                 &certificate);
            if (!nn_numeric_certificate_write_env(&certificate) || !committed ||
                !isfinite((double)certificate.optimizer_loss)) {
                free(order);
                free(batch_inputs);
                free(batch_targets);
                return (nn_real)NAN;
            }
            epoch_loss += certificate.optimizer_loss * (nn_real)current_batch;
        }
        epoch_loss /= (nn_real)sample_count;
    }

    free(order);
    free(batch_inputs);
    free(batch_targets);
    return epoch_loss;
}

#include "nn_checkpoint.inc"

#include "nn_trace.inc"
