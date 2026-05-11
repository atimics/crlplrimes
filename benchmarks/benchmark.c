#include "nn.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    const char *name;
    size_t layers[5];
    size_t layer_count;
    size_t samples;
    size_t batch_size;
    size_t epochs;
    size_t forward_repeats;
    nn_real learning_rate;
} BenchCase;

static double now_seconds(void)
{
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
    }
#endif
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static uint32_t rng_next(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static nn_real rng_uniform(uint32_t *state)
{
    return (nn_real)((double)rng_next(state) / (double)UINT32_MAX);
}

static double safe_rate(double work, double seconds)
{
    return seconds > 0.0 ? work / seconds : 0.0;
}

static int allocate_dataset(const BenchCase *bench,
                            nn_real **inputs_out,
                            nn_real **targets_out)
{
    size_t input_width = bench->layers[0];
    size_t output_width = bench->layers[bench->layer_count - 1];
    size_t input_count = bench->samples * input_width;
    size_t target_count = bench->samples * output_width;

    nn_real *inputs = calloc(input_count, sizeof(*inputs));
    nn_real *targets = calloc(target_count, sizeof(*targets));
    if (inputs == NULL || targets == NULL) {
        free(inputs);
        free(targets);
        return 0;
    }

    uint32_t rng = 123u;
    for (size_t i = 0; i < input_count; ++i) {
        inputs[i] = rng_uniform(&rng);
    }

    for (size_t sample = 0; sample < bench->samples; ++sample) {
        if (output_width == 1) {
            targets[sample] = (rng_next(&rng) & 1u) ? (nn_real)1 : (nn_real)0;
        } else {
            size_t class_index = (size_t)(rng_next(&rng) % (uint32_t)output_width);
            targets[sample * output_width + class_index] = (nn_real)1;
        }
    }

    *inputs_out = inputs;
    *targets_out = targets;
    return 1;
}

static double benchmark_forward(NeuralNet *net,
                                const nn_real *inputs,
                                size_t samples,
                                size_t repeats,
                                nn_real *outputs)
{
    volatile nn_real sink = 0;
    size_t output_width = nn_output_size(net);
    double started = now_seconds();

    for (size_t repeat = 0; repeat < repeats; ++repeat) {
        nn_forward_batch(net, inputs, outputs, samples);
        sink += outputs[(repeat % samples) * output_width];
    }

    double elapsed = now_seconds() - started;
    if (sink < 0) {
        fputs("unreachable benchmark guard\n", stderr);
    }
    return elapsed;
}

static int run_case(const BenchCase *bench)
{
    NeuralNet *net = nn_create(bench->layers, bench->layer_count, 42);
    if (net == NULL) {
        fprintf(stderr, "%s: failed to create network\n", bench->name);
        return 0;
    }

    nn_real *inputs = NULL;
    nn_real *targets = NULL;
    if (!allocate_dataset(bench, &inputs, &targets)) {
        fprintf(stderr, "%s: failed to allocate dataset\n", bench->name);
        nn_free(net);
        return 0;
    }

    NeuralNet *sgd_net = nn_create(bench->layers, bench->layer_count, 42);
    if (sgd_net == NULL) {
        fprintf(stderr, "%s: failed to create SGD baseline network\n", bench->name);
        free(inputs);
        free(targets);
        nn_free(net);
        return 0;
    }

    size_t output_width = nn_output_size(net);
    nn_real *outputs = calloc(bench->samples * output_width, sizeof(*outputs));
    if (outputs == NULL) {
        fprintf(stderr, "%s: failed to allocate outputs\n", bench->name);
        free(inputs);
        free(targets);
        nn_free(sgd_net);
        nn_free(net);
        return 0;
    }

    size_t parameters = nn_parameter_count(net);
    size_t forward_flops = nn_forward_flop_count(net);
    size_t forward_passes = bench->samples * bench->forward_repeats;

    nn_forward_batch(net, inputs, outputs, bench->samples);
    double forward_seconds =
        benchmark_forward(net, inputs, bench->samples, bench->forward_repeats, outputs);

    double sgd_started = now_seconds();
    nn_real sgd_loss = nn_train(sgd_net,
                                inputs,
                                targets,
                                bench->samples,
                                bench->learning_rate,
                                bench->epochs,
                                1);
    double sgd_seconds = now_seconds() - sgd_started;

    double batch_started = now_seconds();
    nn_real batch_loss = nn_train_batched(net,
                                          inputs,
                                          targets,
                                          bench->samples,
                                          bench->batch_size,
                                          bench->learning_rate,
                                          bench->epochs,
                                          1);
    double batch_seconds = now_seconds() - batch_started;
    double sample_updates = (double)bench->samples * (double)bench->epochs;
    size_t batches_per_epoch = (bench->samples + bench->batch_size - 1) / bench->batch_size;
    size_t batch_updates = batches_per_epoch * bench->epochs;

    printf("%-16s params=%zu fwd_flops=%zu samples=%zu batch=%zu epochs=%zu\n",
           bench->name,
           parameters,
           forward_flops,
           bench->samples,
           bench->batch_size,
           bench->epochs);
    printf("  forward-only:\n");
    printf("    %.6fs %zu passes %.3fus/pass %.2f Mpass/s %.2f GFLOP/s\n",
           forward_seconds,
           forward_passes,
           safe_rate(forward_seconds * 1000000.0, (double)forward_passes),
           safe_rate((double)forward_passes, forward_seconds) / 1000000.0,
           safe_rate((double)forward_passes * (double)forward_flops, forward_seconds) /
               1000000000.0);
    printf("  training-throughput:\n");
    printf("    sample-sgd: %.6fs %.0f samples/s updates=%.0f loss=%.6f est_dense=%.2f GFLOP/s\n",
           sgd_seconds,
           safe_rate(sample_updates, sgd_seconds),
           sample_updates,
           (double)sgd_loss,
           safe_rate(sample_updates * (double)forward_flops * 3.0, sgd_seconds) /
               1000000000.0);
    printf("    batch-sgd:  %.6fs %.0f samples/s updates=%zu loss=%.6f est_dense=%.2f GFLOP/s throughput_speedup=%.2fx\n\n",
           batch_seconds,
           safe_rate(sample_updates, batch_seconds),
           batch_updates,
           (double)batch_loss,
           safe_rate(sample_updates * (double)forward_flops * 3.0, batch_seconds) /
               1000000000.0,
           safe_rate(sgd_seconds, batch_seconds));

    free(outputs);
    free(inputs);
    free(targets);
    nn_free(sgd_net);
    nn_free(net);
    return 1;
}

int main(void)
{
    const BenchCase cases[] = {
        {
            "tiny-mlp",
            {32, 64, 16},
            3,
            512,
            32,
            5,
            100,
            (nn_real)0.05,
        },
        {
            "mnist-shape",
            {784, 128, 10},
            3,
            256,
            32,
            2,
            10,
            (nn_real)0.03,
        },
        {
            "wide-mlp",
            {1024, 256, 128, 10},
            4,
            128,
            16,
            1,
            5,
            (nn_real)0.02,
        },
    };

    printf("real_size=%zu bytes\n\n", sizeof(nn_real));

    int ok = 1;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        ok = run_case(&cases[i]) && ok;
    }

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
