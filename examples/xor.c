#include "nn.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum {
    INPUT_WIDTH = 2,
    OUTPUT_WIDTH = 1,
    SAMPLE_COUNT = 4,
    CHECKPOINT_EPOCHS = 4000,
    TOTAL_EPOCHS = 20000,
    INFERENCE_BENCHMARK_ROUNDS = 250000,
};

typedef struct {
    nn_real loss;
    size_t correct;
    double accuracy;
} EvalMetrics;

static double now_seconds(void)
{
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static double safe_rate(double work, double seconds)
{
    return seconds > 0.0 ? work / seconds : 0.0;
}

static EvalMetrics evaluate_xor(NeuralNet *net, const nn_real *inputs, const nn_real *targets)
{
    EvalMetrics metrics = {0, 0, 0.0};

    for (size_t i = 0; i < SAMPLE_COUNT; ++i) {
        nn_real output[OUTPUT_WIDTH];
        nn_forward(net, &inputs[i * INPUT_WIDTH], output);

        nn_real error = output[0] - targets[i];
        int prediction = output[0] >= 0.5;
        int expected = targets[i] >= 0.5;

        metrics.loss += (nn_real)0.5 * error * error;
        metrics.correct += prediction == expected ? 1u : 0u;
    }

    metrics.loss /= (nn_real)SAMPLE_COUNT;
    metrics.accuracy = (double)metrics.correct / (double)SAMPLE_COUNT;
    return metrics;
}

static double benchmark_inference(NeuralNet *net, const nn_real *inputs, size_t *forward_passes)
{
    volatile nn_real sink = 0;
    nn_real output[OUTPUT_WIDTH];
    double started = now_seconds();

    for (size_t round = 0; round < INFERENCE_BENCHMARK_ROUNDS; ++round) {
        for (size_t i = 0; i < SAMPLE_COUNT; ++i) {
            nn_forward(net, &inputs[i * INPUT_WIDTH], output);
            sink += output[0];
        }
    }

    double elapsed = now_seconds() - started;
    *forward_passes = (size_t)INFERENCE_BENCHMARK_ROUNDS * SAMPLE_COUNT;

    if (sink < 0) {
        fputs("unreachable benchmark guard\n", stderr);
    }

    return elapsed;
}

int main(void)
{
    const size_t layers[] = {INPUT_WIDTH, 4, OUTPUT_WIDTH};
    NeuralNet *net = nn_create(layers, sizeof(layers) / sizeof(layers[0]), 42);
    if (net == NULL) {
        fputs("failed to create neural network\n", stderr);
        return EXIT_FAILURE;
    }

    const nn_real inputs[] = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    };

    const nn_real targets[] = {
        0.0,
        1.0,
        1.0,
        0.0,
    };

    printf("parameters: %zu\n", nn_parameter_count(net));

    double training_seconds = 0.0;
    for (size_t epoch = CHECKPOINT_EPOCHS; epoch <= TOTAL_EPOCHS; epoch += CHECKPOINT_EPOCHS) {
        double started = now_seconds();
        nn_real loss = nn_train(net, inputs, targets, SAMPLE_COUNT, 0.7f, CHECKPOINT_EPOCHS, 1);
        double elapsed = now_seconds() - started;
        double sample_updates = (double)SAMPLE_COUNT * CHECKPOINT_EPOCHS;

        training_seconds += elapsed;
        printf("epoch %5zu loss %.6f chunk %.6fs %.0f samples/s\n",
               epoch,
               (double)loss,
               elapsed,
               safe_rate(sample_updates, elapsed));
    }

    puts("");
    for (size_t i = 0; i < SAMPLE_COUNT; ++i) {
        nn_real output[OUTPUT_WIDTH];
        nn_forward(net, &inputs[i * INPUT_WIDTH], output);
        printf("%.0f xor %.0f -> %.4f (target %.0f)\n",
               (double)inputs[i * INPUT_WIDTH],
               (double)inputs[i * INPUT_WIDTH + 1],
               (double)output[0],
               (double)targets[i]);
    }

    EvalMetrics metrics = evaluate_xor(net, inputs, targets);
    size_t forward_passes = 0;
    double inference_seconds = benchmark_inference(net, inputs, &forward_passes);
    double total_sample_updates = (double)SAMPLE_COUNT * TOTAL_EPOCHS;

    puts("");
    printf("final loss: %.6f\n", (double)metrics.loss);
    printf("accuracy: %.2f%% (%zu/%d)\n",
           metrics.accuracy * 100.0,
           metrics.correct,
           SAMPLE_COUNT);
    printf("training time: %.6fs\n", training_seconds);
    printf("training throughput: %.0f samples/s\n",
           safe_rate(total_sample_updates, training_seconds));
    printf("inference benchmark: %zu forward passes in %.6fs\n",
           forward_passes,
           inference_seconds);
    printf("inference latency: %.3fus/pass\n",
           safe_rate(inference_seconds * 1000000.0, (double)forward_passes));
    printf("inference throughput: %.0f passes/s\n",
           safe_rate((double)forward_passes, inference_seconds));

    nn_free(net);
    return EXIT_SUCCESS;
}
