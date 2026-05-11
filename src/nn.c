#include "nn.h"

#include <math.h>
#include <stdint.h>
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

struct NeuralNet {
    size_t layer_count;
    size_t *sizes;
    NNActivation hidden_activation;
    NNActivation output_activation;
    NNLoss loss;
    nn_real **activations;
    nn_real **biases;
    nn_real **weights;
    nn_real **deltas;
    nn_real **grad_biases;
    nn_real **grad_weights;
    nn_real **batch_activations;
    nn_real **batch_deltas;
    size_t batch_capacity;
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
    case NN_ACTIVATION_SIGMOID:
    default:
        return sigmoid(x);
    }
}

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
    case NN_ACTIVATION_SIGMOID:
    default:
        return sigmoid_derivative_from_activation(activation_value);
    }
}

static NNActivation activation_for_layer(const NeuralNet *net, size_t layer)
{
    return layer + 1 == net->layer_count ? net->output_activation : net->hidden_activation;
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
           activation == NN_ACTIVATION_TANH || activation == NN_ACTIVATION_LINEAR;
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
        config->output_activation != NN_ACTIVATION_LINEAR) {
        return 0;
    }

    return 1;
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
                                size_t previous,
                                size_t current,
                                NNActivation activation)
{
    for (size_t neuron = 0; neuron < current; ++neuron) {
        const nn_real *neuron_weights = &weights[neuron * previous];
        nn_real sum = biases[neuron] + dot_product(neuron_weights, previous_activations, previous);
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
    size_t sample = 0;
    for (; sample + 3 < batch_size; sample += 4) {
        const nn_real *previous0 = &previous_activations[(sample + 0) * previous];
        const nn_real *previous1 = &previous_activations[(sample + 1) * previous];
        const nn_real *previous2 = &previous_activations[(sample + 2) * previous];
        const nn_real *previous3 = &previous_activations[(sample + 3) * previous];
        nn_real *activations0 = &activations[(sample + 0) * current];
        nn_real *activations1 = &activations[(sample + 1) * current];
        nn_real *activations2 = &activations[(sample + 2) * current];
        nn_real *activations3 = &activations[(sample + 3) * current];
        size_t neuron = 0;

        for (; neuron + 3 < current; neuron += 4) {
            const nn_real *weights0 = &weights[(neuron + 0) * previous];
            const nn_real *weights1 = &weights[(neuron + 1) * previous];
            const nn_real *weights2 = &weights[(neuron + 2) * previous];
            const nn_real *weights3 = &weights[(neuron + 3) * previous];
            nn_real sum00 = biases[neuron + 0];
            nn_real sum01 = biases[neuron + 1];
            nn_real sum02 = biases[neuron + 2];
            nn_real sum03 = biases[neuron + 3];
            nn_real sum10 = biases[neuron + 0];
            nn_real sum11 = biases[neuron + 1];
            nn_real sum12 = biases[neuron + 2];
            nn_real sum13 = biases[neuron + 3];
            nn_real sum20 = biases[neuron + 0];
            nn_real sum21 = biases[neuron + 1];
            nn_real sum22 = biases[neuron + 2];
            nn_real sum23 = biases[neuron + 3];
            nn_real sum30 = biases[neuron + 0];
            nn_real sum31 = biases[neuron + 1];
            nn_real sum32 = biases[neuron + 2];
            nn_real sum33 = biases[neuron + 3];

            for (size_t input_index = 0; input_index < previous; ++input_index) {
                nn_real x0 = previous0[input_index];
                nn_real x1 = previous1[input_index];
                nn_real x2 = previous2[input_index];
                nn_real x3 = previous3[input_index];
                nn_real w0 = weights0[input_index];
                nn_real w1 = weights1[input_index];
                nn_real w2 = weights2[input_index];
                nn_real w3 = weights3[input_index];

                sum00 += x0 * w0;
                sum01 += x0 * w1;
                sum02 += x0 * w2;
                sum03 += x0 * w3;
                sum10 += x1 * w0;
                sum11 += x1 * w1;
                sum12 += x1 * w2;
                sum13 += x1 * w3;
                sum20 += x2 * w0;
                sum21 += x2 * w1;
                sum22 += x2 * w2;
                sum23 += x2 * w3;
                sum30 += x3 * w0;
                sum31 += x3 * w1;
                sum32 += x3 * w2;
                sum33 += x3 * w3;
            }

            activations0[neuron + 0] = activate(sum00, activation);
            activations0[neuron + 1] = activate(sum01, activation);
            activations0[neuron + 2] = activate(sum02, activation);
            activations0[neuron + 3] = activate(sum03, activation);
            activations1[neuron + 0] = activate(sum10, activation);
            activations1[neuron + 1] = activate(sum11, activation);
            activations1[neuron + 2] = activate(sum12, activation);
            activations1[neuron + 3] = activate(sum13, activation);
            activations2[neuron + 0] = activate(sum20, activation);
            activations2[neuron + 1] = activate(sum21, activation);
            activations2[neuron + 2] = activate(sum22, activation);
            activations2[neuron + 3] = activate(sum23, activation);
            activations3[neuron + 0] = activate(sum30, activation);
            activations3[neuron + 1] = activate(sum31, activation);
            activations3[neuron + 2] = activate(sum32, activation);
            activations3[neuron + 3] = activate(sum33, activation);
        }

        for (; neuron < current; ++neuron) {
            const nn_real *neuron_weights = &weights[neuron * previous];
            activations0[neuron] =
                activate(biases[neuron] + dot_product(neuron_weights, previous0, previous),
                         activation);
            activations1[neuron] =
                activate(biases[neuron] + dot_product(neuron_weights, previous1, previous),
                         activation);
            activations2[neuron] =
                activate(biases[neuron] + dot_product(neuron_weights, previous2, previous),
                         activation);
            activations3[neuron] =
                activate(biases[neuron] + dot_product(neuron_weights, previous3, previous),
                         activation);
        }
    }

    for (; sample < batch_size; ++sample) {
        dense_forward_layer(&previous_activations[sample * previous],
                            weights,
                            biases,
                            &activations[sample * current],
                            previous,
                            current,
                            activation);
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

static void zero_gradients(NeuralNet *net)
{
    for (size_t layer = 1; layer < net->layer_count; ++layer) {
        size_t current = net->sizes[layer];
        size_t previous = net->sizes[layer - 1];

        memset(net->grad_biases[layer], 0, current * sizeof(*net->grad_biases[layer]));
        memset(net->grad_weights[layer],
               0,
               current * previous * sizeof(*net->grad_weights[layer]));
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
            output_deltas[neuron] *= activation_derivative_from_activation(
                output_activations[neuron], net->output_activation);
        }
    } else {
        for (size_t neuron = 0; neuron < output_width; ++neuron) {
            nn_real activation = output_activations[neuron];
            nn_real error = activation - target[neuron];
            loss += (nn_real)0.5 * error * error;
            output_deltas[neuron] =
                error * activation_derivative_from_activation(activation, net->output_activation);
        }
    }

    for (size_t layer = net->layer_count - 2; layer > 0; --layer) {
        size_t current = net->sizes[layer];
        size_t next = net->sizes[layer + 1];
        nn_real *current_deltas = net->deltas[layer];
        const nn_real *next_deltas = net->deltas[layer + 1];
        const nn_real *next_weights = net->weights[layer + 1];
        const nn_real *current_activations = net->activations[layer];

        memset(current_deltas, 0, current * sizeof(*current_deltas));

        for (size_t next_neuron = 0; next_neuron < next; ++next_neuron) {
            const nn_real delta = next_deltas[next_neuron];
            const nn_real *weights = &next_weights[next_neuron * current];
            axpy(current_deltas, weights, current, delta);
        }

        for (size_t neuron = 0; neuron < current; ++neuron) {
            current_deltas[neuron] *= activation_derivative_from_activation(
                current_activations[neuron], activation_for_layer(net, layer));
        }
    }

    return loss;
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

        for (size_t neuron = 0; neuron < current; ++neuron) {
            nn_real step = learning_rate * deltas[neuron];
            nn_real *neuron_weights = &weights[neuron * previous];

            biases[neuron] -= step;

            for (size_t input_index = 0; input_index < previous; ++input_index) {
                neuron_weights[input_index] -= step * previous_activations[input_index];
            }
        }
    }
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

static void accumulate_batch_gradients(NeuralNet *net, size_t batch_size)
{
    for (size_t layer = 1; layer < net->layer_count; ++layer) {
        size_t previous = net->sizes[layer - 1];
        size_t current = net->sizes[layer];
        const nn_real *previous_activations = net->batch_activations[layer - 1];
        const nn_real *deltas = net->batch_deltas[layer];
        nn_real *grad_biases = net->grad_biases[layer];
        nn_real *grad_weights = net->grad_weights[layer];

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

                grad0[0] += sum00;
                grad0[1] += sum01;
                grad0[2] += sum02;
                grad0[3] += sum03;
                grad1[0] += sum10;
                grad1[1] += sum11;
                grad1[2] += sum12;
                grad1[3] += sum13;
                grad2[0] += sum20;
                grad2[1] += sum21;
                grad2[2] += sum22;
                grad2[3] += sum23;
                grad3[0] += sum30;
                grad3[1] += sum31;
                grad3[2] += sum32;
                grad3[3] += sum33;
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

                grad_weights[(neuron + 0) * previous + input_index] += sum0;
                grad_weights[(neuron + 1) * previous + input_index] += sum1;
                grad_weights[(neuron + 2) * previous + input_index] += sum2;
                grad_weights[(neuron + 3) * previous + input_index] += sum3;
            }

            for (size_t sample = 0; sample < batch_size; ++sample) {
                const nn_real *delta_row = &deltas[sample * current + neuron];
                bias0 += delta_row[0];
                bias1 += delta_row[1];
                bias2 += delta_row[2];
                bias3 += delta_row[3];
            }

            grad_biases[neuron + 0] += bias0;
            grad_biases[neuron + 1] += bias1;
            grad_biases[neuron + 2] += bias2;
            grad_biases[neuron + 3] += bias3;
        }

        for (; neuron < current; ++neuron) {
            nn_real *neuron_grad_weights = &grad_weights[neuron * previous];
            nn_real bias_sum = 0;

            for (size_t sample = 0; sample < batch_size; ++sample) {
                nn_real delta = deltas[sample * current + neuron];
                const nn_real *previous_row = &previous_activations[sample * previous];

                bias_sum += delta;
                axpy(neuron_grad_weights, previous_row, previous, delta);
            }

            grad_biases[neuron] += bias_sum;
        }
    }
}

static void apply_gradients(NeuralNet *net, nn_real scale)
{
    for (size_t layer = 1; layer < net->layer_count; ++layer) {
        size_t previous = net->sizes[layer - 1];
        size_t current = net->sizes[layer];
        size_t weight_count = current * previous;
        nn_real *biases = net->biases[layer];
        nn_real *weights = net->weights[layer];
        const nn_real *grad_biases = net->grad_biases[layer];
        const nn_real *grad_weights = net->grad_weights[layer];

        for (size_t i = 0; i < current; ++i) {
            biases[i] -= scale * grad_biases[i];
        }

        for (size_t i = 0; i < weight_count; ++i) {
            weights[i] -= scale * grad_weights[i];
        }
    }
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
    if (!ensure_batch_workspace(net, batch_size)) {
        return (nn_real)NAN;
    }

    if (!ensure_gradient_buffers(net)) {
        return (nn_real)NAN;
    }

    zero_gradients(net);

    load_batch_inputs(net, inputs, order, start, batch_size, input_width);
    forward_batch_workspace(net, batch_size);
    nn_real loss =
        compute_batch_deltas_and_loss(net, targets, order, start, batch_size, target_width);
    accumulate_batch_gradients(net, batch_size);
    apply_gradients(net, learning_rate / (nn_real)batch_size);
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
    net->biases = checked_calloc(config->layer_count, sizeof(*net->biases));
    net->weights = checked_calloc(config->layer_count, sizeof(*net->weights));
    net->deltas = checked_calloc(config->layer_count, sizeof(*net->deltas));

    if (net->sizes == NULL || net->activations == NULL || net->biases == NULL ||
        net->weights == NULL || net->deltas == NULL) {
        nn_free(net);
        return NULL;
    }

    memcpy(net->sizes, config->layer_sizes, config->layer_count * sizeof(*net->sizes));

    for (size_t layer = 0; layer < config->layer_count; ++layer) {
        net->activations[layer] = checked_calloc(net->sizes[layer], sizeof(**net->activations));
        net->deltas[layer] = checked_calloc(net->sizes[layer], sizeof(**net->deltas));

        if (net->activations[layer] == NULL || net->deltas[layer] == NULL) {
            nn_free(net);
            return NULL;
        }

        if (layer == 0) {
            continue;
        }

        size_t previous = net->sizes[layer - 1];
        size_t current = net->sizes[layer];
        if (previous != 0 && current > SIZE_MAX / previous) {
            nn_free(net);
            return NULL;
        }
        size_t weight_count = current * previous;

        net->biases[layer] = checked_calloc(current, sizeof(**net->biases));
        net->weights[layer] = checked_calloc(weight_count, sizeof(**net->weights));

        if (net->biases[layer] == NULL || net->weights[layer] == NULL) {
            nn_free(net);
            return NULL;
        }

        NNActivation layer_activation = activation_for_layer(net, layer);
        nn_real scale = layer_activation == NN_ACTIVATION_RELU
                            ? NN_SQRT((nn_real)6 / (nn_real)previous)
                            : NN_SQRT((nn_real)6 / (nn_real)(previous + current));
        for (size_t i = 0; i < weight_count; ++i) {
            net->weights[layer][i] = ((rng_uniform(&net->rng_state) * (nn_real)2) - (nn_real)1) *
                                     scale;
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

void nn_free(NeuralNet *net)
{
    if (net == NULL) {
        return;
    }

    if (net->activations != NULL) {
        for (size_t i = 0; i < net->layer_count; ++i) {
            free(net->activations[i]);
        }
    }

    if (net->biases != NULL) {
        for (size_t i = 0; i < net->layer_count; ++i) {
            free(net->biases[i]);
        }
    }

    if (net->weights != NULL) {
        for (size_t i = 0; i < net->layer_count; ++i) {
            free(net->weights[i]);
        }
    }

    if (net->deltas != NULL) {
        for (size_t i = 0; i < net->layer_count; ++i) {
            free(net->deltas[i]);
        }
    }

    if (net->grad_biases != NULL) {
        for (size_t i = 0; i < net->layer_count; ++i) {
            free(net->grad_biases[i]);
        }
    }

    if (net->grad_weights != NULL) {
        for (size_t i = 0; i < net->layer_count; ++i) {
            free(net->grad_weights[i]);
        }
    }

    free_batch_workspace(net);

    free(net->sizes);
    free(net->activations);
    free(net->biases);
    free(net->weights);
    free(net->deltas);
    free(net->grad_biases);
    free(net->grad_weights);
    free(net);
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
    if (net == NULL || input == NULL) {
        return;
    }

    memcpy(net->activations[0], input, net->sizes[0] * sizeof(*input));

    for (size_t layer = 1; layer < net->layer_count; ++layer) {
        dense_forward_layer(net->activations[layer - 1],
                            net->weights[layer],
                            net->biases[layer],
                            net->activations[layer],
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
    if (net == NULL || inputs == NULL || outputs == NULL || sample_count == 0) {
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
        return;
    }

    for (size_t sample = 0; sample < sample_count; ++sample) {
        nn_forward(net, &inputs[sample * input_width], &outputs[sample * output_width]);
    }
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
                                                   nn_real entropy_weight,
                                                   nn_real learning_rate)
{
    if (weight == (nn_real)0 && entropy_weight == (nn_real)0) {
        return (nn_real)0;
    }

    if (!ensure_gradient_buffers(net)) {
        return (nn_real)NAN;
    }

    zero_gradients(net);

    size_t output_layer = net->layer_count - 1;
    const nn_real *scores = net->batch_activations[output_layer];
    nn_real *output_deltas = net->batch_deltas[output_layer];
    nn_softmax(scores, output_deltas, candidate_count);
    nn_real entropy = 0;
    for (size_t sample = 0; sample < candidate_count; ++sample) {
        nn_real probability = output_deltas[sample];
        if (probability > (nn_real)0) {
            entropy -= probability * NN_LOG(probability);
        }
    }

    nn_real loss = weight * softmax_cross_entropy_index(scores, candidate_count, selected_index) -
                   entropy_weight * entropy;

    for (size_t sample = 0; sample < candidate_count; ++sample) {
        nn_real target = sample == selected_index ? (nn_real)1 : (nn_real)0;
        nn_real score = scores[sample];
        nn_real probability = output_deltas[sample];
        nn_real log_probability = probability > (nn_real)0 ? NN_LOG(probability) : (nn_real)-27.631021;
        nn_real entropy_delta =
            entropy_weight * probability * (log_probability + entropy);
        output_deltas[sample] = (weight * (probability - target) + entropy_delta) *
                                activation_derivative_from_activation(score, net->output_activation);
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

    accumulate_batch_gradients(net, candidate_count);
    apply_gradients(net, learning_rate);
    return loss;
}

static nn_real train_softmax_scorer_targets_from_workspace(NeuralNet *net,
                                                           size_t candidate_count,
                                                           const nn_real *target_distribution,
                                                           nn_real learning_rate)
{
    if (!ensure_gradient_buffers(net)) {
        return (nn_real)NAN;
    }

    zero_gradients(net);

    size_t output_layer = net->layer_count - 1;
    const nn_real *scores = net->batch_activations[output_layer];
    nn_real *output_deltas = net->batch_deltas[output_layer];
    nn_softmax(scores, output_deltas, candidate_count);
    nn_real target_sum = 0;
    for (size_t sample = 0; sample < candidate_count; ++sample) {
        target_sum += target_distribution[sample];
    }
    nn_real loss =
        nn_softmax_cross_entropy(scores, target_distribution, candidate_count) / target_sum;

    for (size_t sample = 0; sample < candidate_count; ++sample) {
        nn_real score = scores[sample];
        nn_real normalized_target = target_distribution[sample] / target_sum;
        output_deltas[sample] =
            (output_deltas[sample] - normalized_target) *
            activation_derivative_from_activation(score, net->output_activation);
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

    accumulate_batch_gradients(net, candidate_count);
    apply_gradients(net, learning_rate);
    return loss;
}

nn_real nn_train_softmax_scorer_cached(NeuralNet *net,
                                       size_t candidate_count,
                                       size_t selected_index,
                                       nn_real weight,
                                       nn_real learning_rate)
{
    if (net == NULL || candidate_count == 0 || selected_index >= candidate_count ||
        learning_rate <= (nn_real)0 || nn_output_size(net) != 1 ||
        net->output_activation != NN_ACTIVATION_LINEAR ||
        net->batch_capacity < candidate_count || net->batch_activations == NULL ||
        net->batch_deltas == NULL) {
        return (nn_real)NAN;
    }

    return train_softmax_scorer_from_workspace(
        net, candidate_count, selected_index, weight, (nn_real)0, learning_rate);
}

nn_real nn_train_softmax_scorer_weighted(NeuralNet *net,
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
        net, candidate_count, selected_index, weight, (nn_real)0, learning_rate);
}

nn_real nn_train_softmax_scorer(NeuralNet *net,
                                const nn_real *candidate_inputs,
                                size_t candidate_count,
                                size_t target_index,
                                nn_real learning_rate)
{
    return nn_train_softmax_scorer_weighted(
        net, candidate_inputs, candidate_count, target_index, (nn_real)1, learning_rate);
}

nn_real nn_train_softmax_scorer_entropy(NeuralNet *net,
                                        const nn_real *candidate_inputs,
                                        size_t candidate_count,
                                        size_t target_index,
                                        nn_real entropy_weight,
                                        nn_real learning_rate)
{
    if (net == NULL || candidate_inputs == NULL || candidate_count == 0 ||
        target_index >= candidate_count || entropy_weight < (nn_real)0 ||
        learning_rate <= (nn_real)0 || nn_output_size(net) != 1 ||
        net->output_activation != NN_ACTIVATION_LINEAR) {
        return (nn_real)NAN;
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
        net, candidate_count, target_index, (nn_real)1, entropy_weight, learning_rate);
}

nn_real nn_train_softmax_scorer_targets(NeuralNet *net,
                                        const nn_real *candidate_inputs,
                                        size_t candidate_count,
                                        const nn_real *target_distribution,
                                        nn_real learning_rate)
{
    if (net == NULL || candidate_inputs == NULL || target_distribution == NULL ||
        candidate_count == 0 || learning_rate <= (nn_real)0 || nn_output_size(net) != 1 ||
        net->output_activation != NN_ACTIVATION_LINEAR) {
        return (nn_real)NAN;
    }

    nn_real target_sum = 0;
    for (size_t i = 0; i < candidate_count; ++i) {
        if (target_distribution[i] < (nn_real)0) {
            return (nn_real)NAN;
        }
        target_sum += target_distribution[i];
    }
    if (target_sum <= (nn_real)0) {
        return (nn_real)NAN;
    }

    if (!ensure_batch_workspace(net, candidate_count)) {
        return (nn_real)NAN;
    }

    size_t input_width = nn_input_size(net);
    memcpy(net->batch_activations[0],
           candidate_inputs,
           candidate_count * input_width * sizeof(*candidate_inputs));
    forward_batch_workspace(net, candidate_count);
    return train_softmax_scorer_targets_from_workspace(
        net, candidate_count, target_distribution, learning_rate);
}

nn_real nn_train(NeuralNet *net,
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

nn_real nn_train_batched(NeuralNet *net,
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
