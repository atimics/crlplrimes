#include "nn.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(NN_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(NN_USE_CBLAS)
#include <cblas.h>
#endif

/*
 * Portable BLAS wrappers: dispatch to Accelerate, OpenBLAS/CBLAS, or
 * scalar fallback depending on build-time NN_BLAS setting.
 */
static void nn_blas_gemv(const nn_real *A, const nn_real *x, nn_real *y,
                         size_t rows, size_t cols,
                         nn_real alpha, nn_real beta)
{
#if defined(NN_USE_ACCELERATE) || defined(NN_USE_CBLAS)
#if NN_REAL_IS_FLOAT
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                (int)rows, (int)cols,
                (float)alpha, (const float *)A, (int)cols,
                (const float *)x, 1, (float)beta, (float *)y, 1);
#else
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                (int)rows, (int)cols,
                (double)alpha, (const double *)A, (int)cols,
                (const double *)x, 1, (double)beta, (double *)y, 1);
#endif
#else
    /* Scalar fallback: y = alpha * A * x + beta * y */
    for (size_t r = 0; r < rows; ++r) {
        nn_real dot = (nn_real)0;
        for (size_t c = 0; c < cols; ++c)
            dot += A[r * cols + c] * x[c];
        y[r] = alpha * dot + beta * y[r];
    }
#endif
}

static void nn_blas_gemm(const nn_real *A, const nn_real *B, nn_real *C,
                         size_t m, size_t k, size_t n,
                         nn_real alpha, nn_real beta)
{
    /* C(m x n) = alpha * A(m x k) * B(k x n) + beta * C.
     * For dense_forward_layer_batch: C = X * W^T where X is batch x prev,
     * W is current x prev, so A=X(m=batch, k=prev), B=W^T (k=prev, n=current).
     * Storage: A row-major, B row-major needs CblasTrans to treat it as W^T. */
#if defined(NN_USE_ACCELERATE) || defined(NN_USE_CBLAS)
#if NN_REAL_IS_FLOAT
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)m, (int)n, (int)k,
                (float)alpha, (const float *)A, (int)k,
                (const float *)B, (int)k,
                (float)beta, (float *)C, (int)n);
#else
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)m, (int)n, (int)k,
                (double)alpha, (const double *)A, (int)k,
                (const double *)B, (int)k,
                (double)beta, (double *)C, (int)n);
#endif
#else
    /* Scalar fallback: C = alpha * A * B + beta * C */
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            nn_real s = (nn_real)0;
            for (size_t p = 0; p < k; ++p)
                s += A[i * k + p] * B[j * k + p]; /* B row-major, B[j][p] */
            C[i * n + j] = alpha * s + beta * C[i * n + j];
        }
    }
#endif
}

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
        { nn_real ex = NN_EXP(x); return ex - NN_LOG((nn_real)1 + ex); }
    case NN_ACTIVATION_SIGMOID:
    default:
        return sigmoid(x);
    }
}

/* EML pairwise activation: output[i] = eml(pre[2i], pre[2i+1]).
 * pre_activations has 2*output_count elements. */
static void activate_eml_pairwise(const nn_real *pre, nn_real *out, size_t output_count)
{
    for (size_t i = 0; i < output_count; ++i) {
        out[i] = nn_eml(pre[2 * i], pre[2 * i + 1]);
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
        { nn_real a = activation_value; return a < 2 ? (a < 0 ? 0 : a) : 2; }
    case NN_ACTIVATION_SIGMOID:
    default:
        return sigmoid_derivative_from_activation(activation_value);
    }
}

/*
 * Derivative of EML activation given the *pre-activation* value x.
 * activation = exp(x) - log(1+exp(x))
 * d/dx = exp(2x) / (1+exp(x))
 */
static nn_real eml_activation_derivative(nn_real x) {
    nn_real ex = NN_EXP(x); nn_real d = (ex*ex)/(1+ex);
    return d > 10 ? 10 : (d < 0.01f ? 0.01f : d);
}

/* EML pairwise derivative for backprop.
 * Given output gradient dL/do[i] and the original pre-activations (z1, z2):
 *   dL/dz1 = dL/do[i] * exp(z1)
 *   dL/dz2 = dL/do[i] * (-1/z2)
 * pre has 2*count elements, out_grad has count elements, pre_grad has 2*count */
static void eml_pairwise_backward(const nn_real *pre, const nn_real *out_grad,
                                  nn_real *pre_grad, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        nn_real z1 = pre[2 * i];
        nn_real z2 = pre[2 * i + 1];
        nn_real og = out_grad[i];
        pre_grad[2 * i]     = og * nn_eml_dx(z1);
        pre_grad[2 * i + 1] = og * nn_eml_dy(z2);
    }
}

static NNActivation activation_for_layer(const NeuralNet *net, size_t layer)
{
    if (net->layer_activations != NULL && net->layer_activations[layer] != NN_ACTIVATION_SIGMOID)
        return net->layer_activations[layer];
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
            /* cap pre-acts to prevent exp overflow in eml */
            if (z1 > (nn_real)5) z1 = (nn_real)5;
            if (z1 < (nn_real)-5) z1 = (nn_real)-5;
            if (z2 > (nn_real)5) z2 = (nn_real)5;
            if (z2 < (nn_real)-5) z2 = (nn_real)-5;
            if (pre_activations != NULL) {
                pre_activations[2 * i]     = z1;
                pre_activations[2 * i + 1] = z2;
            }
            nn_real val = nn_eml(z1, z2);
            /* clip tightly: stacked EML layers compound exp(exp(x)) */
            if (val > (nn_real)5) val = (nn_real)5;
            if (val < (nn_real)-5) val = (nn_real)-5;
            activations[i] = val;
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
                if (z1 > (nn_real)5) z1 = (nn_real)5;
                if (z1 < (nn_real)-5) z1 = (nn_real)-5;
                if (z2 > (nn_real)5) z2 = (nn_real)5;
                if (z2 < (nn_real)-5) z2 = (nn_real)-5;
                nn_real val = nn_eml(z1, z2);
                if (val > (nn_real)5) val = (nn_real)5;
                if (val < (nn_real)-5) val = (nn_real)-5;
                out[i] = val;
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
            NNActivation act = activation_for_layer(net, layer);
            if (act == NN_ACTIVATION_EML && net->pre_activations != NULL) {
                current_deltas[neuron] *= eml_activation_derivative(
                    net->pre_activations[layer][neuron]);
            } else if (act == NN_ACTIVATION_EML_PAIRWISE) {
                /* derivative applied in eml_pairwise_backprop_post */
            } else {
                current_deltas[neuron] *= activation_derivative_from_activation(
                    current_activations[neuron], act);
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

void nn_numeric_guard_config_init(NNNumericGuardConfig *config)
{
    if (config == NULL) {
        return;
    }

    config->require_finite = 1;
    config->max_shrink_steps = 4u;
    config->shrink_factor = (nn_real)0.5;
    config->max_abs_parameter = (nn_real)0;
    config->max_abs_parameter_delta = (nn_real)0;
    config->max_abs_logit = (nn_real)0;
    config->max_abs_logit_delta = (nn_real)0;
    config->min_selected_margin = (nn_real)0;
}

void nn_training_step_certificate_init(NNTrainingStepCertificate *certificate)
{
    if (certificate == NULL) {
        return;
    }

    memset(certificate, 0, sizeof(*certificate));
    snprintf(certificate->schema,
             sizeof(certificate->schema),
             "%s",
             NN_NUMERIC_STEP_CERTIFICATE_SCHEMA);
    snprintf(certificate->step_kind, sizeof(certificate->step_kind), "%s", "candidate_scorer");
    certificate->report_version = 1u;
    certificate->rejection_reason = NN_NUMERIC_STEP_REJECT_NONE;
    certificate->optimizer_loss = (nn_real)NAN;
    certificate->loss_before = (nn_real)NAN;
    certificate->loss_after = (nn_real)NAN;
    certificate->selected_margin_before = (nn_real)NAN;
    certificate->selected_margin_after = (nn_real)NAN;
}

const char *nn_numeric_step_reject_reason_name(NNNumericStepRejectReason reason)
{
    switch (reason) {
    case NN_NUMERIC_STEP_REJECT_NONE:
        return "none";
    case NN_NUMERIC_STEP_REJECT_INVALID_ARGUMENT:
        return "invalid_argument";
    case NN_NUMERIC_STEP_REJECT_NONFINITE:
        return "nonfinite";
    case NN_NUMERIC_STEP_REJECT_PARAMETER_BOUND:
        return "parameter_bound";
    case NN_NUMERIC_STEP_REJECT_PARAMETER_DELTA_BOUND:
        return "parameter_delta_bound";
    case NN_NUMERIC_STEP_REJECT_LOGIT_BOUND:
        return "logit_bound";
    case NN_NUMERIC_STEP_REJECT_LOGIT_DELTA_BOUND:
        return "logit_delta_bound";
    case NN_NUMERIC_STEP_REJECT_MARGIN_BOUND:
        return "margin_bound";
    case NN_NUMERIC_STEP_REJECT_ALLOC_FAILED:
        return "alloc_failed";
    default:
        return "unknown";
    }
}

static nn_real nn_abs_real(nn_real value)
{
    return value < (nn_real)0 ? -value : value;
}

static int nn_parameter_metrics(const NeuralNet *net, nn_real *max_abs)
{
    if (max_abs != NULL) {
        *max_abs = (nn_real)0;
    }
    if (net == NULL) {
        return 0;
    }

    int finite = 1;
    nn_real observed = (nn_real)0;
    for (size_t layer = 1; layer < net->layer_count; ++layer) {
        size_t previous = net->sizes[layer - 1];
        size_t current = net->sizes[layer];
        size_t weight_count = previous * current;
        for (size_t i = 0; i < weight_count; ++i) {
            nn_real value = net->weights[layer][i];
            if (!isfinite((double)value)) {
                finite = 0;
            } else if (nn_abs_real(value) > observed) {
                observed = nn_abs_real(value);
            }
        }
        for (size_t i = 0; i < current; ++i) {
            nn_real value = net->biases[layer][i];
            if (!isfinite((double)value)) {
                finite = 0;
            } else if (nn_abs_real(value) > observed) {
                observed = nn_abs_real(value);
            }
        }
    }
    if (max_abs != NULL) {
        *max_abs = observed;
    }
    return finite;
}

static nn_real nn_parameter_max_abs_delta(const NeuralNet *before, const NeuralNet *after)
{
    if (before == NULL || after == NULL || before->layer_count != after->layer_count) {
        return (nn_real)NAN;
    }

    nn_real observed = (nn_real)0;
    for (size_t layer = 1; layer < before->layer_count; ++layer) {
        if (before->sizes[layer] != after->sizes[layer] ||
            before->sizes[layer - 1] != after->sizes[layer - 1]) {
            return (nn_real)NAN;
        }
        size_t previous = before->sizes[layer - 1];
        size_t current = before->sizes[layer];
        size_t weight_count = previous * current;
        for (size_t i = 0; i < weight_count; ++i) {
            nn_real delta = nn_abs_real(after->weights[layer][i] - before->weights[layer][i]);
            if (!isfinite((double)delta)) {
                return (nn_real)NAN;
            }
            if (delta > observed) {
                observed = delta;
            }
        }
        for (size_t i = 0; i < current; ++i) {
            nn_real delta = nn_abs_real(after->biases[layer][i] - before->biases[layer][i]);
            if (!isfinite((double)delta)) {
                return (nn_real)NAN;
            }
            if (delta > observed) {
                observed = delta;
            }
        }
    }
    return observed;
}

static int nn_copy_parameters(NeuralNet *dst, const NeuralNet *src)
{
    if (dst == NULL || src == NULL || dst->layer_count != src->layer_count) {
        return 0;
    }

    for (size_t layer = 1; layer < src->layer_count; ++layer) {
        if (dst->sizes[layer] != src->sizes[layer] ||
            dst->sizes[layer - 1] != src->sizes[layer - 1]) {
            return 0;
        }
        size_t previous = src->sizes[layer - 1];
        size_t current = src->sizes[layer];
        memcpy(dst->biases[layer], src->biases[layer], current * sizeof(*dst->biases[layer]));
        memcpy(dst->weights[layer],
               src->weights[layer],
               current * previous * sizeof(*dst->weights[layer]));
    }
    dst->rng_state = src->rng_state;
    invalidate_batch_cache(dst);
    return 1;
}

static int nn_score_metrics(NeuralNet *net,
                            const nn_real *candidate_inputs,
                            size_t candidate_count,
                            size_t selected_index,
                            nn_real *scores,
                            nn_real *max_abs_logit,
                            nn_real *selected_margin,
                            nn_real *loss,
                            nn_real weight)
{
    if (max_abs_logit != NULL) {
        *max_abs_logit = (nn_real)0;
    }
    if (selected_margin != NULL) {
        *selected_margin = (nn_real)NAN;
    }
    if (loss != NULL) {
        *loss = (nn_real)NAN;
    }
    if (net == NULL || candidate_inputs == NULL || candidate_count == 0 ||
        selected_index >= candidate_count || scores == NULL || nn_output_size(net) != 1) {
        return 0;
    }

    nn_forward_batch(net, candidate_inputs, scores, candidate_count);
    int finite = 1;
    nn_real observed = (nn_real)0;
    for (size_t i = 0; i < candidate_count; ++i) {
        if (!isfinite((double)scores[i])) {
            finite = 0;
            continue;
        }
        nn_real abs_score = nn_abs_real(scores[i]);
        if (abs_score > observed) {
            observed = abs_score;
        }
    }
    if (max_abs_logit != NULL) {
        *max_abs_logit = observed;
    }
    if (selected_margin != NULL) {
        *selected_margin = nn_action_score_margin(scores, candidate_count, selected_index);
    }
    if (loss != NULL) {
        *loss = weight * softmax_cross_entropy_index(scores, candidate_count, selected_index);
    }
    return finite && selected_margin != NULL && isfinite((double)*selected_margin) &&
           loss != NULL && isfinite((double)*loss);
}

static int nn_dataset_metrics(NeuralNet *net,
                              const nn_real *inputs,
                              const nn_real *targets,
                              size_t sample_count,
                              nn_real *outputs,
                              nn_real *loss,
                              nn_real *max_abs_output)
{
    if (loss != NULL) {
        *loss = (nn_real)NAN;
    }
    if (max_abs_output != NULL) {
        *max_abs_output = (nn_real)0;
    }
    if (net == NULL || inputs == NULL || targets == NULL || sample_count == 0 ||
        outputs == NULL || loss == NULL || max_abs_output == NULL) {
        return 0;
    }

    size_t output_width = nn_output_size(net);
    if (output_width == 0) {
        return 0;
    }

    nn_forward_batch(net, inputs, outputs, sample_count);
    int finite = 1;
    nn_real total_loss = (nn_real)0;
    nn_real observed = (nn_real)0;
    for (size_t sample = 0; sample < sample_count; ++sample) {
        const nn_real *row_output = &outputs[sample * output_width];
        const nn_real *row_target = &targets[sample * output_width];
        for (size_t i = 0; i < output_width; ++i) {
            if (!isfinite((double)row_output[i]) || !isfinite((double)row_target[i])) {
                finite = 0;
            } else if (nn_abs_real(row_output[i]) > observed) {
                observed = nn_abs_real(row_output[i]);
            }
        }

        if (net->loss == NN_LOSS_SOFTMAX_CROSS_ENTROPY) {
            total_loss += nn_softmax_cross_entropy(row_output, row_target, output_width);
        } else {
            for (size_t i = 0; i < output_width; ++i) {
                nn_real error = row_output[i] - row_target[i];
                total_loss += (nn_real)0.5 * error * error;
            }
        }
    }

    *loss = total_loss / (nn_real)sample_count;
    *max_abs_output = observed;
    return finite && isfinite((double)*loss);
}

static nn_real nn_policy_regularized_objective(const nn_real *scores,
                                               size_t candidate_count,
                                               size_t selected_index,
                                               nn_real policy_weight,
                                               const nn_real *reference_probabilities,
                                               nn_real reference_weight,
                                               nn_real entropy_weight)
{
    if (scores == NULL || candidate_count == 0 || selected_index >= candidate_count ||
        reference_weight < (nn_real)0 || entropy_weight < (nn_real)0 ||
        (reference_weight != (nn_real)0 && reference_probabilities == NULL)) {
        return (nn_real)NAN;
    }

    nn_real loss = (nn_real)0;
    if (policy_weight != (nn_real)0) {
        loss += policy_weight * softmax_cross_entropy_index(scores,
                                                            candidate_count,
                                                            selected_index);
    }
    if (reference_weight != (nn_real)0) {
        loss += reference_weight *
                nn_softmax_cross_entropy(scores, reference_probabilities, candidate_count);
    }
    if (entropy_weight != (nn_real)0) {
        nn_real max_score = scores[0];
        for (size_t i = 1; i < candidate_count; ++i) {
            if (scores[i] > max_score) {
                max_score = scores[i];
            }
        }
        nn_real denominator = (nn_real)0;
        for (size_t i = 0; i < candidate_count; ++i) {
            denominator += NN_EXP(scores[i] - max_score);
        }
        if (denominator <= (nn_real)0 || !isfinite((double)denominator)) {
            return (nn_real)NAN;
        }
        nn_real entropy_loss = (nn_real)0;
        for (size_t i = 0; i < candidate_count; ++i) {
            nn_real probability = NN_EXP(scores[i] - max_score) / denominator;
            entropy_loss += probability * safe_log_probability(probability);
        }
        loss += entropy_weight * entropy_loss;
    }
    return loss;
}

static nn_real nn_score_max_abs_delta(const nn_real *before,
                                      const nn_real *after,
                                      size_t count)
{
    if (before == NULL || after == NULL) {
        return (nn_real)NAN;
    }
    nn_real observed = (nn_real)0;
    for (size_t i = 0; i < count; ++i) {
        nn_real delta = nn_abs_real(after[i] - before[i]);
        if (!isfinite((double)delta)) {
            return (nn_real)NAN;
        }
        if (delta > observed) {
            observed = delta;
        }
    }
    return observed;
}

static int nn_guard_bound_active(nn_real value)
{
    return isfinite((double)value) && value > (nn_real)0;
}

static NNNumericStepRejectReason nn_numeric_guard_check(
    const NNNumericGuardConfig *guard,
    const NNTrainingStepCertificate *certificate)
{
    if (guard == NULL || certificate == NULL) {
        return NN_NUMERIC_STEP_REJECT_INVALID_ARGUMENT;
    }
    if (guard->require_finite &&
        (!certificate->finite_before || !certificate->finite_after ||
         !isfinite((double)certificate->loss_before) ||
         !isfinite((double)certificate->loss_after) ||
         !isfinite((double)certificate->optimizer_loss) ||
         !isfinite((double)certificate->max_abs_parameter_delta) ||
         !isfinite((double)certificate->max_abs_logit_delta) ||
         !isfinite((double)certificate->selected_margin_before) ||
         !isfinite((double)certificate->selected_margin_after))) {
        return NN_NUMERIC_STEP_REJECT_NONFINITE;
    }
    if (nn_guard_bound_active(guard->max_abs_parameter) &&
        certificate->max_abs_parameter_after > guard->max_abs_parameter) {
        return NN_NUMERIC_STEP_REJECT_PARAMETER_BOUND;
    }
    if (nn_guard_bound_active(guard->max_abs_parameter_delta) &&
        certificate->max_abs_parameter_delta > guard->max_abs_parameter_delta) {
        return NN_NUMERIC_STEP_REJECT_PARAMETER_DELTA_BOUND;
    }
    if (nn_guard_bound_active(guard->max_abs_logit) &&
        certificate->max_abs_logit_after > guard->max_abs_logit) {
        return NN_NUMERIC_STEP_REJECT_LOGIT_BOUND;
    }
    if (nn_guard_bound_active(guard->max_abs_logit_delta) &&
        certificate->max_abs_logit_delta > guard->max_abs_logit_delta) {
        return NN_NUMERIC_STEP_REJECT_LOGIT_DELTA_BOUND;
    }
    if (nn_guard_bound_active(guard->min_selected_margin) &&
        certificate->selected_margin_after < guard->min_selected_margin) {
        return NN_NUMERIC_STEP_REJECT_MARGIN_BOUND;
    }
    return NN_NUMERIC_STEP_REJECT_NONE;
}

static int nn_write_json_real(FILE *fp, nn_real value)
{
    if (fp == NULL) {
        return 0;
    }
    if (!isfinite((double)value)) {
        return fputs("null", fp) != EOF;
    }
    return fprintf(fp, "%.17g", (double)value) >= 0;
}

int nn_training_step_certificate_write_jsonl(FILE *fp,
                                             const NNTrainingStepCertificate *certificate)
{
    if (fp == NULL || certificate == NULL) {
        return 0;
    }

    if (fputs("{\"schema\":", fp) == EOF ||
        !nn_write_json_string(fp,
                              certificate->schema[0] == '\0'
                                  ? NN_NUMERIC_STEP_CERTIFICATE_SCHEMA
                                  : certificate->schema) ||
        fputs(",\"step_kind\":", fp) == EOF ||
        !nn_write_json_string(fp,
                              certificate->step_kind[0] == '\0'
                                  ? "candidate_scorer"
                                  : certificate->step_kind) ||
        fprintf(fp,
                ",\"report_version\":%u,\"step_committed\":%s,"
                "\"rejection_reason\":",
                certificate->report_version,
                certificate->step_committed ? "true" : "false") < 0 ||
        !nn_write_json_string(fp,
                              nn_numeric_step_reject_reason_name(
                                  certificate->rejection_reason)) ||
        fprintf(fp,
                ",\"shrink_steps\":%zu,\"parameter_count\":%zu,"
                "\"candidate_count\":%zu,\"selected_index\":%zu,"
                "\"finite_before\":%s,\"finite_after\":%s",
                certificate->shrink_steps,
                certificate->parameter_count,
                certificate->candidate_count,
                certificate->selected_index,
                certificate->finite_before ? "true" : "false",
                certificate->finite_after ? "true" : "false") < 0) {
        return 0;
    }

    const struct {
        const char *name;
        nn_real value;
    } fields[] = {
        {"attempted_learning_rate", certificate->attempted_learning_rate},
        {"applied_learning_rate", certificate->applied_learning_rate},
        {"optimizer_loss", certificate->optimizer_loss},
        {"loss_before", certificate->loss_before},
        {"loss_after", certificate->loss_after},
        {"max_abs_parameter_before", certificate->max_abs_parameter_before},
        {"max_abs_parameter_after", certificate->max_abs_parameter_after},
        {"max_abs_parameter_delta", certificate->max_abs_parameter_delta},
        {"max_abs_logit_before", certificate->max_abs_logit_before},
        {"max_abs_logit_after", certificate->max_abs_logit_after},
        {"max_abs_logit_delta", certificate->max_abs_logit_delta},
        {"selected_margin_before", certificate->selected_margin_before},
        {"selected_margin_after", certificate->selected_margin_after},
    };
    for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); ++i) {
        if (fprintf(fp, ",\"%s\":", fields[i].name) < 0 ||
            !nn_write_json_real(fp, fields[i].value)) {
            return 0;
        }
    }
    return fputs("}\n", fp) != EOF;
}

static const char *nn_numeric_certificate_path(void)
{
    const char *path = getenv("CRLP_NUMERIC_CERT_PATH");
    if (path == NULL || path[0] == '\0') {
        return NULL;
    }
    return path;
}

static int nn_env_parse_real(const char *name, nn_real *out)
{
    const char *text = getenv(name);
    if (text == NULL || text[0] == '\0' || out == NULL) {
        return 0;
    }

    errno = 0;
    char *end = NULL;
    double parsed = strtod(text, &end);
    if (end == text || errno == ERANGE || !isfinite(parsed)) {
        return 0;
    }
    while (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r') {
        ++end;
    }
    if (*end != '\0') {
        return 0;
    }
    *out = (nn_real)parsed;
    return 1;
}

static int nn_env_parse_size(const char *name, size_t *out)
{
    const char *text = getenv(name);
    if (text == NULL || text[0] == '\0' || out == NULL) {
        return 0;
    }

    errno = 0;
    char *end = NULL;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (end == text || errno == ERANGE || parsed > (unsigned long long)SIZE_MAX) {
        return 0;
    }
    while (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r') {
        ++end;
    }
    if (*end != '\0') {
        return 0;
    }
    *out = (size_t)parsed;
    return 1;
}

static void nn_numeric_guard_config_sanitize(NNNumericGuardConfig *config)
{
    if (config == NULL) {
        return;
    }
    if (config->shrink_factor <= (nn_real)0 || config->shrink_factor >= (nn_real)1 ||
        !isfinite((double)config->shrink_factor)) {
        config->shrink_factor = (nn_real)0.5;
    }
    if (config->max_abs_parameter < (nn_real)0 ||
        !isfinite((double)config->max_abs_parameter)) {
        config->max_abs_parameter = (nn_real)0;
    }
    if (config->max_abs_parameter_delta < (nn_real)0 ||
        !isfinite((double)config->max_abs_parameter_delta)) {
        config->max_abs_parameter_delta = (nn_real)0;
    }
    if (config->max_abs_logit < (nn_real)0 || !isfinite((double)config->max_abs_logit)) {
        config->max_abs_logit = (nn_real)0;
    }
    if (config->max_abs_logit_delta < (nn_real)0 ||
        !isfinite((double)config->max_abs_logit_delta)) {
        config->max_abs_logit_delta = (nn_real)0;
    }
    if (config->min_selected_margin < (nn_real)0 ||
        !isfinite((double)config->min_selected_margin)) {
        config->min_selected_margin = (nn_real)0;
    }
}

static void nn_numeric_guard_config_from_env(NNNumericGuardConfig *config)
{
    if (config == NULL) {
        return;
    }
    nn_numeric_guard_config_init(config);

    nn_real real_value = (nn_real)0;
    size_t size_value = 0u;
    if (nn_env_parse_size("CRLP_NUMERIC_MAX_SHRINK_STEPS", &size_value)) {
        config->max_shrink_steps = size_value;
    }
    if (nn_env_parse_real("CRLP_NUMERIC_SHRINK_FACTOR", &real_value)) {
        config->shrink_factor = real_value;
    }
    if (nn_env_parse_real("CRLP_NUMERIC_MAX_ABS_PARAMETER", &real_value)) {
        config->max_abs_parameter = real_value;
    }
    if (nn_env_parse_real("CRLP_NUMERIC_MAX_ABS_PARAMETER_DELTA", &real_value)) {
        config->max_abs_parameter_delta = real_value;
    }
    if (nn_env_parse_real("CRLP_NUMERIC_MAX_ABS_LOGIT", &real_value)) {
        config->max_abs_logit = real_value;
    }
    if (nn_env_parse_real("CRLP_NUMERIC_MAX_ABS_LOGIT_DELTA", &real_value)) {
        config->max_abs_logit_delta = real_value;
    }
    if (nn_env_parse_real("CRLP_NUMERIC_MIN_SELECTED_MARGIN", &real_value)) {
        config->min_selected_margin = real_value;
    }
    if (nn_env_parse_real("CRLP_NUMERIC_REQUIRE_FINITE", &real_value)) {
        config->require_finite = real_value != (nn_real)0;
    }
    nn_numeric_guard_config_sanitize(config);
}

static int nn_numeric_certificate_write_env(const NNTrainingStepCertificate *certificate)
{
    const char *path = nn_numeric_certificate_path();
    if (path == NULL) {
        return 1;
    }

    FILE *fp = fopen(path, "a");
    if (fp == NULL) {
        return 0;
    }
    int ok = nn_training_step_certificate_write_jsonl(fp, certificate);
    if (fclose(fp) != 0) {
        ok = 0;
    }
    return ok;
}

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

static size_t checkpoint_metadata_v1_size(void)
{
    return sizeof(uint32_t) + 65u + 65u + 65u + 256u;
}

static size_t checkpoint_metadata_size(void)
{
    return checkpoint_metadata_v1_size() + 32u;
}

static int checkpoint_meta_copy_text(char *dst, size_t dst_size, const char *value)
{
    if (dst == NULL || dst_size == 0) {
        return 0;
    }
    const char *src = value == NULL ? "" : value;
    size_t len = strlen(src);
    if (len >= dst_size) {
        dst[0] = '\0';
        return 0;
    }
    memcpy(dst, src, len + 1u);
    return 1;
}

void nn_checkpoint_meta_init(NNCheckpointMeta *meta, uint32_t feature_encoder_version)
{
    if (meta == NULL) {
        return;
    }
    memset(meta, 0, sizeof(*meta));
    meta->checkpoint_version = NN_CHECKPOINT_VERSION;
    meta->feature_encoder_version = feature_encoder_version;
}

int nn_checkpoint_meta_set_parent_model_hash(NNCheckpointMeta *meta, const char *value)
{
    return meta != NULL &&
           checkpoint_meta_copy_text(meta->parent_model_hash,
                                     sizeof(meta->parent_model_hash),
                                     value);
}

int nn_checkpoint_meta_set_training_trace_root(NNCheckpointMeta *meta, const char *value)
{
    return meta != NULL &&
           checkpoint_meta_copy_text(meta->training_trace_root,
                                     sizeof(meta->training_trace_root),
                                     value);
}

int nn_checkpoint_meta_set_benchmark_report_hash(NNCheckpointMeta *meta, const char *value)
{
    return meta != NULL &&
           checkpoint_meta_copy_text(meta->benchmark_report_hash,
                                     sizeof(meta->benchmark_report_hash),
                                     value);
}

int nn_checkpoint_meta_set_notes(NNCheckpointMeta *meta, const char *value)
{
    return meta != NULL &&
           checkpoint_meta_copy_text(meta->notes, sizeof(meta->notes), value);
}

int nn_checkpoint_meta_set_feature_set(NNCheckpointMeta *meta, const char *value)
{
    return meta != NULL &&
           checkpoint_meta_copy_text(meta->feature_set, sizeof(meta->feature_set), value);
}

static int write_checkpoint_metadata(FILE *fp, const NNCheckpointMeta *meta)
{
    NNCheckpointMeta local;
    memset(&local, 0, sizeof(local));
    if (meta != NULL) {
        local = *meta;
    }

    if (local.checkpoint_version == 0) {
        local.checkpoint_version = NN_CHECKPOINT_VERSION;
    }
    local.parent_model_hash[sizeof(local.parent_model_hash) - 1] = '\0';
    local.training_trace_root[sizeof(local.training_trace_root) - 1] = '\0';
    local.benchmark_report_hash[sizeof(local.benchmark_report_hash) - 1] = '\0';
    local.notes[sizeof(local.notes) - 1] = '\0';
    local.feature_set[sizeof(local.feature_set) - 1] = '\0';

    return write_u32_le(fp, local.feature_encoder_version) &&
           write_all(fp, local.parent_model_hash, sizeof(local.parent_model_hash)) &&
           write_all(fp, local.training_trace_root, sizeof(local.training_trace_root)) &&
           write_all(fp, local.benchmark_report_hash, sizeof(local.benchmark_report_hash)) &&
           write_all(fp, local.notes, sizeof(local.notes)) &&
           write_all(fp, local.feature_set, sizeof(local.feature_set));
}

static int read_checkpoint_metadata(FILE *fp,
                                    uint32_t checkpoint_version,
                                    uint64_t metadata_size,
                                    NNCheckpointMeta *out_meta)
{
    NNCheckpointMeta local;
    memset(&local, 0, sizeof(local));
    local.checkpoint_version = checkpoint_version;

    if (metadata_size != (uint64_t)checkpoint_metadata_v1_size() &&
        metadata_size != (uint64_t)checkpoint_metadata_size()) {
        return 0;
    }

    if (!read_u32_le(fp, &local.feature_encoder_version) ||
        !read_all(fp, local.parent_model_hash, sizeof(local.parent_model_hash)) ||
        !read_all(fp, local.training_trace_root, sizeof(local.training_trace_root)) ||
        !read_all(fp, local.benchmark_report_hash, sizeof(local.benchmark_report_hash)) ||
        !read_all(fp, local.notes, sizeof(local.notes))) {
        return 0;
    }
    if (metadata_size == (uint64_t)checkpoint_metadata_size() &&
        !read_all(fp, local.feature_set, sizeof(local.feature_set))) {
        return 0;
    }

    local.parent_model_hash[sizeof(local.parent_model_hash) - 1] = '\0';
    local.training_trace_root[sizeof(local.training_trace_root) - 1] = '\0';
    local.benchmark_report_hash[sizeof(local.benchmark_report_hash) - 1] = '\0';
    local.notes[sizeof(local.notes) - 1] = '\0';
    local.feature_set[sizeof(local.feature_set) - 1] = '\0';

    if (out_meta != NULL) {
        *out_meta = local;
    }
    return 1;
}

int nn_save_checkpoint(const NeuralNet *net, const char *path, const NNCheckpointMeta *meta)
{
    if (net == NULL || path == NULL) {
        return 0;
    }

    uint32_t checkpoint_version =
        meta != NULL && meta->checkpoint_version != 0 ? meta->checkpoint_version
                                                      : NN_CHECKPOINT_VERSION;
    if (checkpoint_version != NN_CHECKPOINT_VERSION) {
        return 0;
    }

    FILE *fp = fopen(path, "wb");
    if (fp == NULL) {
        return 0;
    }

    int ok = write_all(fp, NN_CHECKPOINT_MAGIC, NN_CHECKPOINT_MAGIC_SIZE) &&
             write_u32_le(fp, checkpoint_version) && write_u32_le(fp, scalar_type_id()) &&
             write_u64_le(fp, (uint64_t)net->layer_count);

    for (size_t layer = 0; ok && layer < net->layer_count; ++layer) {
        ok = write_u64_le(fp, (uint64_t)net->sizes[layer]);
    }

    ok = ok && write_u32_le(fp, (uint32_t)net->hidden_activation) &&
         write_u32_le(fp, (uint32_t)net->output_activation) &&
         write_u32_le(fp, (uint32_t)net->loss);

    for (size_t layer = 1; ok && layer < net->layer_count; ++layer) {
        size_t previous = net->sizes[layer - 1];
        size_t current = net->sizes[layer];
        if (previous != 0 && current > SIZE_MAX / previous) {
            ok = 0;
            break;
        }
        size_t weight_count = previous * current;
        ok = write_all(fp, net->weights[layer], weight_count * sizeof(*net->weights[layer])) &&
             write_all(fp, net->biases[layer], current * sizeof(*net->biases[layer]));
    }

    ok = ok && write_u64_le(fp, (uint64_t)checkpoint_metadata_size()) &&
         write_checkpoint_metadata(fp, meta);

    if (fclose(fp) != 0) {
        ok = 0;
    }

    return ok;
}

NeuralNet *nn_load_checkpoint(const char *path, NNCheckpointMeta *out_meta)
{
    if (path == NULL) {
        return NULL;
    }

    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        return NULL;
    }

    char magic[NN_CHECKPOINT_MAGIC_SIZE];
    uint32_t checkpoint_version = 0;
    uint32_t scalar_type = 0;
    uint64_t layer_count64 = 0;
    size_t *sizes = NULL;
    NeuralNet *net = NULL;
    int ok = read_all(fp, magic, sizeof(magic)) &&
             memcmp(magic, NN_CHECKPOINT_MAGIC, sizeof(magic)) == 0 &&
             read_u32_le(fp, &checkpoint_version) &&
             checkpoint_version == NN_CHECKPOINT_VERSION &&
             read_u32_le(fp, &scalar_type) && scalar_type == scalar_type_id() &&
             read_u64_le(fp, &layer_count64) && layer_count64 <= (uint64_t)SIZE_MAX;

    size_t layer_count = (size_t)layer_count64;
    if (ok) {
        sizes = checked_calloc(layer_count, sizeof(*sizes));
        ok = sizes != NULL;
    }

    for (size_t layer = 0; ok && layer < layer_count; ++layer) {
        uint64_t size64 = 0;
        ok = read_u64_le(fp, &size64) && size64 <= (uint64_t)SIZE_MAX;
        sizes[layer] = (size_t)size64;
    }

    uint32_t hidden_activation = 0;
    uint32_t output_activation = 0;
    uint32_t loss = 0;
    ok = ok && read_u32_le(fp, &hidden_activation) &&
         read_u32_le(fp, &output_activation) && read_u32_le(fp, &loss);

    if (ok) {
        NNConfig config;
        config.layer_sizes = sizes;
        config.layer_count = layer_count;
        config.seed = 1u;
        config.hidden_activation = (NNActivation)hidden_activation;
        config.output_activation = (NNActivation)output_activation;
        config.loss = (NNLoss)loss;
        net = nn_create_config(&config);
        ok = net != NULL;
    }

    for (size_t layer = 1; ok && layer < layer_count; ++layer) {
        size_t previous = net->sizes[layer - 1];
        size_t current = net->sizes[layer];
        if (previous != 0 && current > SIZE_MAX / previous) {
            ok = 0;
            break;
        }
        size_t weight_count = previous * current;
        ok = read_all(fp, net->weights[layer], weight_count * sizeof(*net->weights[layer])) &&
             read_all(fp, net->biases[layer], current * sizeof(*net->biases[layer]));
    }

    uint64_t metadata_size = 0;
    ok = ok && read_u64_le(fp, &metadata_size) &&
         read_checkpoint_metadata(fp, checkpoint_version, metadata_size, out_meta);

    if (fclose(fp) != 0) {
        ok = 0;
    }

    free(sizes);
    if (!ok) {
        nn_free(net);
        return NULL;
    }

    return net;
}

typedef struct {
    uint32_t state[8];
    uint64_t bit_count;
    unsigned char buffer[64];
    size_t buffer_count;
} NNSha256;

static uint32_t sha256_rotr(uint32_t value, uint32_t bits)
{
    return (value >> bits) | (value << (32u - bits));
}

static uint32_t sha256_load_be32(const unsigned char *bytes)
{
    return ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) |
           ((uint32_t)bytes[2] << 8) | (uint32_t)bytes[3];
}

static void sha256_store_be32(unsigned char *bytes, uint32_t value)
{
    bytes[0] = (unsigned char)((value >> 24) & 0xffu);
    bytes[1] = (unsigned char)((value >> 16) & 0xffu);
    bytes[2] = (unsigned char)((value >> 8) & 0xffu);
    bytes[3] = (unsigned char)(value & 0xffu);
}

static void sha256_transform(NNSha256 *sha, const unsigned char block[64])
{
    static const uint32_t k[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
        0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
        0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
        0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
        0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
        0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };
    uint32_t w[64];
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
    uint32_t e;
    uint32_t f;
    uint32_t g;
    uint32_t h;

    for (size_t i = 0; i < 16; ++i) {
        w[i] = sha256_load_be32(&block[i * 4]);
    }
    for (size_t i = 16; i < 64; ++i) {
        uint32_t s0 = sha256_rotr(w[i - 15], 7) ^ sha256_rotr(w[i - 15], 18) ^
                      (w[i - 15] >> 3);
        uint32_t s1 = sha256_rotr(w[i - 2], 17) ^ sha256_rotr(w[i - 2], 19) ^
                      (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    a = sha->state[0];
    b = sha->state[1];
    c = sha->state[2];
    d = sha->state[3];
    e = sha->state[4];
    f = sha->state[5];
    g = sha->state[6];
    h = sha->state[7];

    for (size_t i = 0; i < 64; ++i) {
        uint32_t s1 = sha256_rotr(e, 6) ^ sha256_rotr(e, 11) ^ sha256_rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + s1 + ch + k[i] + w[i];
        uint32_t s0 = sha256_rotr(a, 2) ^ sha256_rotr(a, 13) ^ sha256_rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = s0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    sha->state[0] += a;
    sha->state[1] += b;
    sha->state[2] += c;
    sha->state[3] += d;
    sha->state[4] += e;
    sha->state[5] += f;
    sha->state[6] += g;
    sha->state[7] += h;
}

static void sha256_init(NNSha256 *sha)
{
    sha->state[0] = 0x6a09e667u;
    sha->state[1] = 0xbb67ae85u;
    sha->state[2] = 0x3c6ef372u;
    sha->state[3] = 0xa54ff53au;
    sha->state[4] = 0x510e527fu;
    sha->state[5] = 0x9b05688cu;
    sha->state[6] = 0x1f83d9abu;
    sha->state[7] = 0x5be0cd19u;
    sha->bit_count = 0;
    sha->buffer_count = 0;
}

static void sha256_update(NNSha256 *sha, const unsigned char *data, size_t size)
{
    sha->bit_count += (uint64_t)size * 8u;

    while (size > 0) {
        size_t available = sizeof(sha->buffer) - sha->buffer_count;
        size_t take = size < available ? size : available;
        memcpy(&sha->buffer[sha->buffer_count], data, take);
        sha->buffer_count += take;
        data += take;
        size -= take;

        if (sha->buffer_count == sizeof(sha->buffer)) {
            sha256_transform(sha, sha->buffer);
            sha->buffer_count = 0;
        }
    }
}

static void sha256_final(NNSha256 *sha, uint8_t out_hash[32])
{
    uint64_t bit_count = sha->bit_count;
    unsigned char pad = 0x80u;
    unsigned char zero = 0;
    unsigned char length[8];

    sha256_update(sha, &pad, 1);
    while (sha->buffer_count != 56) {
        sha256_update(sha, &zero, 1);
    }

    for (size_t i = 0; i < sizeof(length); ++i) {
        length[7 - i] = (unsigned char)((bit_count >> (8u * i)) & 0xffu);
    }
    sha256_update(sha, length, sizeof(length));

    for (size_t i = 0; i < 8; ++i) {
        sha256_store_be32(&out_hash[i * 4], sha->state[i]);
    }
}

int nn_hash_checkpoint_file(const char *path, uint8_t out_hash[32])
{
    if (path == NULL || out_hash == NULL) {
        return 0;
    }

    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        return 0;
    }

    NNSha256 sha;
    sha256_init(&sha);

    unsigned char buffer[4096];
    int ok = 1;
    for (;;) {
        size_t count = fread(buffer, 1, sizeof(buffer), fp);
        if (count != 0) {
            sha256_update(&sha, buffer, count);
        }
        if (count < sizeof(buffer)) {
            if (ferror(fp)) {
                ok = 0;
            }
            break;
        }
    }

    if (fclose(fp) != 0) {
        ok = 0;
    }
    if (!ok) {
        return 0;
    }

    sha256_final(&sha, out_hash);
    return 1;
}

int nn_hash_checkpoint_hex(const char *path, char out_hex[65])
{
    static const char hex[] = "0123456789abcdef";
    uint8_t hash[32];

    if (out_hex == NULL || !nn_hash_checkpoint_file(path, hash)) {
        return 0;
    }

    for (size_t i = 0; i < sizeof(hash); ++i) {
        out_hex[i * 2] = hex[hash[i] >> 4];
        out_hex[i * 2 + 1] = hex[hash[i] & 0x0fu];
    }
    out_hex[64] = '\0';
    return 1;
}

static int trace_write_json_string(FILE *fp, const char *text)
{
    return nn_write_json_string(fp, text);
}

static int trace_write_real_array(FILE *fp, const nn_real *values, size_t count)
{
    if (fputc('[', fp) == EOF) {
        return 0;
    }
    for (size_t i = 0; i < count; ++i) {
        if (i != 0 && fputc(',', fp) == EOF) {
            return 0;
        }
#if NN_REAL_IS_FLOAT
        if (fprintf(fp, "%.9g", (double)values[i]) < 0) {
            return 0;
        }
#else
        if (fprintf(fp, "%.17g", (double)values[i]) < 0) {
            return 0;
        }
#endif
    }
    return fputc(']', fp) != EOF;
}

static uint32_t trace_action_id_at(const NNTraceRecord *record, size_t index)
{
    return record->action_ids == NULL ? (uint32_t)index : record->action_ids[index];
}

static int trace_target_indices_are_valid(const NNTraceRecord *record)
{
    if (record->target_indices == NULL) {
        return record->target_count == 0 || record->target_count == 1;
    }
    if (record->target_count == 0) {
        return 0;
    }
    int saw_primary = 0;
    for (size_t i = 0; i < record->target_count; ++i) {
        size_t index = record->target_indices[i];
        if (index >= record->candidate_count) {
            return 0;
        }
        saw_primary = saw_primary || index == record->target_index;
        for (size_t j = i + 1; j < record->target_count; ++j) {
            if (index == record->target_indices[j]) {
                return 0;
            }
        }
    }
    return saw_primary;
}

static int trace_record_arrays_are_valid(const NNTraceRecord *record)
{
    double probability_sum = 0.0;
    for (size_t i = 0; i < record->candidate_count; ++i) {
        if (record->action_ids == NULL && i > UINT32_MAX) {
            return 0;
        }
        uint32_t action_id = trace_action_id_at(record, i);
        for (size_t j = i + 1; j < record->candidate_count; ++j) {
            if (action_id == trace_action_id_at(record, j)) {
                return 0;
            }
        }
        if (!isfinite((double)record->scores[i]) ||
            !isfinite((double)record->probabilities[i]) ||
            record->probabilities[i] < (nn_real)0 ||
            record->probabilities[i] > (nn_real)1) {
            return 0;
        }
        probability_sum += (double)record->probabilities[i];
    }
    return fabs(probability_sum - 1.0) <= 1e-3;
}

int nn_trace_write_jsonl(FILE *fp, const NNTraceRecord *record)
{
    if (fp == NULL || record == NULL || record->candidate_count == 0 ||
        record->scores == NULL || record->probabilities == NULL ||
        record->selected_index >= record->candidate_count ||
        record->target_index >= record->candidate_count ||
        !trace_target_indices_are_valid(record) ||
        !trace_record_arrays_are_valid(record)) {
        return 0;
    }

    const char *task = record->task_name == NULL ? "unknown" : record->task_name;
    const char *model_hash =
        record->model_hash_hex == NULL ? "unknown" : record->model_hash_hex;
    const char *verifier_result =
        record->verifier_result == NULL ? "unknown" : record->verifier_result;
    const char *verifier_cost =
        record->verifier_cost_json == NULL ? "{}" : record->verifier_cost_json;
    uint32_t selected_action_id = trace_action_id_at(record, record->selected_index);
    uint32_t target_action_id = trace_action_id_at(record, record->target_index);
    size_t target_count = record->target_indices == NULL ? 1u : record->target_count;
    nn_real selected_score_margin =
        nn_action_score_margin(record->scores, record->candidate_count, record->selected_index);
    if (!isfinite((double)selected_score_margin)) {
        return 0;
    }
    size_t selected_by_policy = 0;
    for (size_t i = 1; i < record->candidate_count; ++i) {
        if (rank_score_is_better(record->scores,
                                 record->action_ids,
                                 i,
                                 selected_by_policy,
                                 NN_RANK_SCORE_EPSILON)) {
            selected_by_policy = i;
        }
    }
    if (selected_by_policy != record->selected_index) {
        return 0;
    }
    int rank_fp_sensitive = fabs((double)selected_score_margin) <=
                            (double)NN_RANK_SCORE_EPSILON;

    if (fputc('{', fp) == EOF || fputs("\"schema\":", fp) == EOF ||
        !trace_write_json_string(fp, NN_TRACE_SCHEMA) ||
        fprintf(fp, ",\"trace_version\":%u", (unsigned int)NN_TRACE_VERSION) < 0 ||
        fputs(",\"action_id_contract\":", fp) == EOF ||
        !trace_write_json_string(fp, NN_TRACE_ACTION_ID_CONTRACT) ||
        fputs(",\"task\":", fp) == EOF || !trace_write_json_string(fp, task) ||
        fputs(",\"state_hash\":\"", fp) == EOF ||
        fprintf(fp, "%016llx", (unsigned long long)record->state_hash) < 0 ||
        fputs("\",\"model_hash\":", fp) == EOF ||
        !trace_write_json_string(fp, model_hash) ||
        fprintf(fp,
                ",\"feature_encoder_version\":%u,\"candidate_count\":%zu,"
                "\"action_ids\":[",
                record->feature_encoder_version,
                record->candidate_count) < 0) {
        return 0;
    }

    for (size_t i = 0; i < record->candidate_count; ++i) {
        uint32_t action_id = record->action_ids == NULL ? (uint32_t)i : record->action_ids[i];
        if ((i != 0 && fputc(',', fp) == EOF) || fprintf(fp, "%u", action_id) < 0) {
            return 0;
        }
    }

    if (fputs("],\"scores\":", fp) == EOF ||
        !trace_write_real_array(fp, record->scores, record->candidate_count) ||
        fputs(",\"probabilities\":", fp) == EOF ||
        !trace_write_real_array(fp, record->probabilities, record->candidate_count)) {
        return 0;
    }

    if (fprintf(fp, ",\"target_count\":%zu,\"target_indices\":[", target_count) < 0) {
        return 0;
    }
    for (size_t i = 0; i < target_count; ++i) {
        size_t index = record->target_indices == NULL ? record->target_index
                                                      : record->target_indices[i];
        if ((i != 0 && fputc(',', fp) == EOF) || fprintf(fp, "%zu", index) < 0) {
            return 0;
        }
    }
    if (fputs("],\"target_action_ids\":[", fp) == EOF) {
        return 0;
    }
    for (size_t i = 0; i < target_count; ++i) {
        size_t index = record->target_indices == NULL ? record->target_index
                                                      : record->target_indices[i];
        if ((i != 0 && fputc(',', fp) == EOF) ||
            fprintf(fp, "%u", trace_action_id_at(record, index)) < 0) {
            return 0;
        }
    }
    if (fputc(']', fp) == EOF) {
        return 0;
    }

#if NN_REAL_IS_FLOAT
    const char *loss_format =
        ",\"selected_index\":%zu,\"target_index\":%zu,\"loss\":%.9g";
    const char *rank_guard_format =
        ",\"selected_action_id\":%u,\"target_action_id\":%u,"
        "\"selected_score_margin\":%.9g,\"rank_score_epsilon\":%.9g,";
#else
    const char *loss_format =
        ",\"selected_index\":%zu,\"target_index\":%zu,\"loss\":%.17g";
    const char *rank_guard_format =
        ",\"selected_action_id\":%u,\"target_action_id\":%u,"
        "\"selected_score_margin\":%.17g,\"rank_score_epsilon\":%.17g,";
#endif
    if (fprintf(fp,
                loss_format,
                record->selected_index,
                record->target_index,
                (double)record->loss) < 0 ||
        fprintf(fp,
                rank_guard_format,
                selected_action_id,
                target_action_id,
                (double)selected_score_margin,
                (double)NN_RANK_SCORE_EPSILON) < 0 ||
        fprintf(fp,
                "\"rank_fp_sensitive\":%s,\"ranking_policy\":",
                rank_fp_sensitive ? "true" : "false") < 0 ||
        !trace_write_json_string(fp, NN_RANKING_POLICY) ||
        fputs(",\"verifier_result\":", fp) == EOF ||
        !trace_write_json_string(fp, verifier_result) ||
        fputs(",\"verifier_cost\":", fp) == EOF || fputs(verifier_cost, fp) == EOF ||
        fprintf(fp, ",\"seed\":%llu", (unsigned long long)record->seed) < 0) {
        return 0;
    }

    if (record->extra_json != NULL && record->extra_json[0] != '\0') {
        if (fputc(',', fp) == EOF || fputs(record->extra_json, fp) == EOF) {
            return 0;
        }
    }

    /* Output feature vector if available */
    if (record->features != NULL && record->feature_count > 0) {
        if (fputs(",\"features\":[", fp) == EOF) return 0;
        for (size_t i = 0; i < record->feature_count; i++) {
            if (i > 0 && fputc(',', fp) == EOF) return 0;
            if (fprintf(fp, "%.9g", (double)record->features[i]) < 0) return 0;
        }
        if (fputc(']', fp) == EOF) return 0;
    }

    return fputs("}\n", fp) != EOF;
}

static const char *trace_find_key_value(const char *line, const char *key)
{
    if (line == NULL || key == NULL || key[0] == '\0') {
        return NULL;
    }

    char pattern[96];
    int written = snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    if (written <= 0 || (size_t)written >= sizeof(pattern)) {
        return NULL;
    }

    const char *start = strstr(line, pattern);
    if (start == NULL) {
        return NULL;
    }
    return start + (size_t)written;
}

int nn_trace_json_get_string(const char *line,
                             const char *key,
                             char *out_value,
                             size_t out_value_size)
{
    const char *start = trace_find_key_value(line, key);
    if (start == NULL || out_value == NULL || out_value_size == 0 || *start != '"') {
        return 0;
    }
    ++start;

    size_t output = 0;
    for (const char *p = start; *p != '\0'; ++p) {
        unsigned char ch = (unsigned char)*p;
        if (ch == '"') {
            out_value[output] = '\0';
            return 1;
        }
        if (ch == '\\') {
            ++p;
            if (*p == '\0') {
                return 0;
            }
            switch (*p) {
            case '"':
            case '\\':
            case '/':
                ch = (unsigned char)*p;
                break;
            case 'n':
                ch = '\n';
                break;
            case 'r':
                ch = '\r';
                break;
            case 't':
                ch = '\t';
                break;
            default:
                return 0;
            }
        }

        if (output + 1 >= out_value_size) {
            return 0;
        }
        out_value[output++] = (char)ch;
    }

    return 0;
}

int nn_trace_json_get_u64(const char *line, const char *key, uint64_t *out_value)
{
    const char *start = trace_find_key_value(line, key);
    if (start == NULL || out_value == NULL || *start == '-' || *start == '+') {
        return 0;
    }

    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(start, &end, 10);
    if (end == start || errno == ERANGE ||
        (*end != ',' && *end != '}' && *end != ']' && *end != '\n' && *end != '\r' &&
         *end != '\0')) {
        return 0;
    }
    *out_value = (uint64_t)value;
    return 1;
}

int nn_trace_json_get_real(const char *line, const char *key, nn_real *out_value)
{
    const char *start = trace_find_key_value(line, key);
    if (start == NULL || out_value == NULL) {
        return 0;
    }

    errno = 0;
    char *end = NULL;
    double value = strtod(start, &end);
    if (end == start || errno == ERANGE || !isfinite(value) ||
        (*end != ',' && *end != '}' && *end != ']' && *end != '\n' && *end != '\r' &&
         *end != '\0')) {
        return 0;
    }
    *out_value = (nn_real)value;
    return 1;
}

static int trace_parse_u32_array(const char *line,
                                 const char *key,
                                 uint32_t *values,
                                 size_t expected_count)
{
    const char *p = trace_find_key_value(line, key);
    if (p == NULL || values == NULL || *p != '[') {
        return 0;
    }
    ++p;

    for (size_t i = 0; i < expected_count; ++i) {
        if (*p == '-' || *p == '+') {
            return 0;
        }
        errno = 0;
        char *end = NULL;
        unsigned long value = strtoul(p, &end, 10);
        if (end == p || errno == ERANGE || value > UINT32_MAX) {
            return 0;
        }
        values[i] = (uint32_t)value;
        p = end;
        if (i + 1 < expected_count) {
            if (*p != ',') {
                return 0;
            }
            ++p;
        }
    }

    return *p == ']';
}

static int trace_parse_real_array(const char *line,
                                  const char *key,
                                  nn_real *values,
                                  size_t expected_count)
{
    const char *p = trace_find_key_value(line, key);
    if (p == NULL || values == NULL || *p != '[') {
        return 0;
    }
    ++p;

    for (size_t i = 0; i < expected_count; ++i) {
        errno = 0;
        char *end = NULL;
        double value = strtod(p, &end);
        if (end == p || errno == ERANGE || !isfinite(value)) {
            return 0;
        }
        values[i] = (nn_real)value;
        p = end;
        if (i + 1 < expected_count) {
            if (*p != ',') {
                return 0;
            }
            ++p;
        }
    }

    return *p == ']';
}

static int trace_parse_hex_u64(const char *line, const char *key, uint64_t *out_value)
{
    char text[32];
    if (!nn_trace_json_get_string(line, key, text, sizeof(text)) || strlen(text) != 16) {
        return 0;
    }

    uint64_t value = 0;
    for (size_t i = 0; i < 16; ++i) {
        char ch = text[i];
        uint64_t nibble;
        if (ch >= '0' && ch <= '9') {
            nibble = (uint64_t)(ch - '0');
        } else if (ch >= 'a' && ch <= 'f') {
            nibble = (uint64_t)(ch - 'a' + 10);
        } else if (ch >= 'A' && ch <= 'F') {
            nibble = (uint64_t)(ch - 'A' + 10);
        } else {
            return 0;
        }
        value = (value << 4) | nibble;
    }

    *out_value = value;
    return 1;
}

static int trace_validate_logged_action_id(const char *line,
                                           const char *key,
                                           uint32_t expected_action_id)
{
    uint64_t logged = 0;
    const char *value = trace_find_key_value(line, key);
    if (value == NULL) {
        return 1;
    }
    if (!nn_trace_json_get_u64(line, key, &logged) || logged > UINT32_MAX) {
        return 0;
    }
    return (uint32_t)logged == expected_action_id;
}

int nn_trace_parse_jsonl(const char *line,
                         NNParsedTraceRecord *out_record,
                         uint32_t *action_ids,
                         nn_real *scores,
                         nn_real *probabilities,
                         size_t candidate_capacity)
{
    if (line == NULL || out_record == NULL || action_ids == NULL || scores == NULL ||
        probabilities == NULL || candidate_capacity == 0) {
        return 0;
    }

    NNParsedTraceRecord parsed;
    memset(&parsed, 0, sizeof(parsed));

    uint64_t candidate_count = 0;
    uint64_t selected_index = 0;
    uint64_t target_index = 0;
    uint64_t trace_version = 0;
    uint64_t feature_encoder_version = 0;
    nn_real loss = 0;
    nn_real rank_score_epsilon = NN_RANK_SCORE_EPSILON;

    if (!nn_trace_json_get_string(line, "schema", parsed.schema, sizeof(parsed.schema)) ||
        strcmp(parsed.schema, NN_TRACE_SCHEMA) != 0 ||
        !nn_trace_json_get_u64(line, "trace_version", &trace_version) ||
        trace_version != NN_TRACE_VERSION ||
        !nn_trace_json_get_string(line,
                                  "action_id_contract",
                                  parsed.action_id_contract,
                                  sizeof(parsed.action_id_contract)) ||
        strcmp(parsed.action_id_contract, NN_TRACE_ACTION_ID_CONTRACT) != 0 ||
        !nn_trace_json_get_string(line, "task", parsed.task_name, sizeof(parsed.task_name)) ||
        !trace_parse_hex_u64(line, "state_hash", &parsed.state_hash) ||
        !nn_trace_json_get_string(
            line, "model_hash", parsed.model_hash_hex, sizeof(parsed.model_hash_hex)) ||
        !nn_trace_json_get_u64(line, "feature_encoder_version", &feature_encoder_version) ||
        feature_encoder_version > UINT32_MAX ||
        !nn_trace_json_get_u64(line, "candidate_count", &candidate_count) ||
        candidate_count == 0 || candidate_count > candidate_capacity ||
        !nn_trace_json_get_u64(line, "selected_index", &selected_index) ||
        !nn_trace_json_get_u64(line, "target_index", &target_index) ||
        selected_index >= candidate_count || target_index >= candidate_count ||
        !nn_trace_json_get_real(line, "loss", &loss) ||
        !nn_trace_json_get_string(
            line, "verifier_result", parsed.verifier_result, sizeof(parsed.verifier_result)) ||
        !nn_trace_json_get_u64(line, "seed", &parsed.seed)) {
        return 0;
    }

    parsed.feature_encoder_version = (uint32_t)feature_encoder_version;
    parsed.trace_version = (uint32_t)trace_version;
    parsed.candidate_count = (size_t)candidate_count;
    parsed.selected_index = (size_t)selected_index;
    parsed.target_index = (size_t)target_index;
    parsed.loss = loss;
    parsed.rank_score_epsilon = rank_score_epsilon;

    if (!trace_parse_u32_array(line, "action_ids", action_ids, parsed.candidate_count) ||
        !trace_parse_real_array(line, "scores", scores, parsed.candidate_count) ||
        !trace_parse_real_array(line, "probabilities", probabilities, parsed.candidate_count)) {
        return 0;
    }

    for (size_t i = 0; i < parsed.candidate_count; ++i) {
        for (size_t j = i + 1; j < parsed.candidate_count; ++j) {
            if (action_ids[i] == action_ids[j]) {
                return 0;
            }
        }
    }
    parsed.selected_action_id = action_ids[parsed.selected_index];
    parsed.target_action_id = action_ids[parsed.target_index];

    if (!trace_validate_logged_action_id(line,
                                         "selected_action_id",
                                         parsed.selected_action_id) ||
        !trace_validate_logged_action_id(line, "target_action_id", parsed.target_action_id)) {
        return 0;
    }

    nn_real logged_epsilon = 0;
    if (trace_find_key_value(line, "rank_score_epsilon") != NULL) {
        if (!nn_trace_json_get_real(line, "rank_score_epsilon", &logged_epsilon) ||
            logged_epsilon < (nn_real)0) {
            return 0;
        }
        parsed.rank_score_epsilon = logged_epsilon;
    }

    parsed.selected_score_margin =
        nn_action_score_margin(scores, parsed.candidate_count, parsed.selected_index);
    if (!isfinite((double)parsed.selected_score_margin)) {
        return 0;
    }
    nn_real logged_margin = 0;
    if (trace_find_key_value(line, "selected_score_margin") != NULL) {
        if (!nn_trace_json_get_real(line, "selected_score_margin", &logged_margin) ||
            !isfinite((double)logged_margin)) {
            return 0;
        }
        double tolerance = fabs((double)parsed.selected_score_margin) * 1e-5 + 1e-5;
        if (fabs((double)logged_margin - (double)parsed.selected_score_margin) > tolerance) {
            return 0;
        }
        parsed.selected_score_margin = logged_margin;
    }

    if (nn_trace_json_get_string(line,
                                 "ranking_policy",
                                 parsed.ranking_policy,
                                 sizeof(parsed.ranking_policy))) {
        if (strcmp(parsed.ranking_policy, NN_RANKING_POLICY) != 0) {
            return 0;
        }
        size_t selected_by_policy = 0;
        for (size_t i = 1; i < parsed.candidate_count; ++i) {
            if (rank_score_is_better(scores,
                                     action_ids,
                                     i,
                                     selected_by_policy,
                                     parsed.rank_score_epsilon)) {
                selected_by_policy = i;
            }
        }
        if (selected_by_policy != parsed.selected_index) {
            return 0;
        }
    } else {
        snprintf(parsed.ranking_policy, sizeof(parsed.ranking_policy), "legacy_score_order");
    }

    nn_real probability_sum = 0;
    for (size_t i = 0; i < parsed.candidate_count; ++i) {
        if (!isfinite((double)scores[i]) ||
            !isfinite((double)probabilities[i]) ||
            probabilities[i] < (nn_real)0 ||
            probabilities[i] > (nn_real)1) {
            return 0;
        }
        probability_sum += probabilities[i];
    }

    if (fabs((double)probability_sum - 1.0) > 1e-3) {
        return 0;
    }

    *out_record = parsed;
    return 1;
}
