#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int main(void) {
    size_t layer_sizes[] = {96, 256, 256, 128, 64, 1};
    size_t layer_count = sizeof(layer_sizes)/sizeof(layer_sizes[0]);
    unsigned int seed = 42;
    size_t batch_size = 64;
    size_t warmup = 100;
    size_t iters = 2000;

    NeuralNet *net = nn_create(layer_sizes, layer_count, seed);
    if (!net) { printf("create failed\n"); return 1; }
    
    size_t params = nn_parameter_count(net);
    size_t flops = nn_forward_flop_count(net);
    printf("Network: ");
    for (size_t i = 0; i < layer_count; i++) printf("%zu%s", layer_sizes[i], i+1<layer_count?"-":"");
    printf("\nParameters: %zu  FLOPs/fwd: %zu\n", params, flops);

    /* Single-sample forward */
    size_t input_size = nn_input_size(net);
    size_t output_size = nn_output_size(net);
    nn_real *input = calloc(input_size, sizeof(*input));
    nn_real *output = calloc(output_size, sizeof(*output));
    for (size_t i = 0; i < input_size; i++) input[i] = (nn_real)(i % 100) / 100.0f;

    /* Warmup */
    for (size_t i = 0; i < warmup; i++) nn_forward(net, input, output);
    
    /* Benchmark single-sample forward */
    double t0 = now_seconds();
    for (size_t i = 0; i < iters; i++) nn_forward(net, input, output);
    double t1 = now_seconds();
    double us_per_fwd = (t1 - t0) / (double)iters * 1e6;
    printf("Single forward: %.1f us  (%.0f fwd/s)\n", us_per_fwd, 1e6 / us_per_fwd);

    /* Batch forward */
    nn_real *batch_in = calloc(batch_size * input_size, sizeof(*batch_in));
    nn_real *batch_out = calloc(batch_size * output_size, sizeof(*batch_out));
    for (size_t i = 0; i < batch_size * input_size; i++) batch_in[i] = (nn_real)(i % 100) / 100.0f;

    /* Warmup batch */
    for (size_t i = 0; i < warmup; i++) nn_forward_batch(net, batch_in, batch_out, batch_size);

    t0 = now_seconds();
    for (size_t i = 0; i < iters; i++) nn_forward_batch(net, batch_in, batch_out, batch_size);
    t1 = now_seconds();
    double us_per_batch = (t1 - t0) / (double)iters * 1e6;
    printf("Batch %zu forward: %.1f us  (%.0f batch/s, %.1f M samples/s)\n",
           batch_size, us_per_batch, 1e6 / us_per_batch,
           (double)batch_size / us_per_batch);

    nn_free(net);
    free(input); free(output);
    free(batch_in); free(batch_out);
    return 0;
}
