#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#define FEATURES 40
#define HIDDEN0  32
#define HIDDEN1  16
#define LAYERS   4
#define TRAIN_SAMPLES 10000
#define TEST_SAMPLES  2000
#define EPOCHS 200
#define SEED 42
#define LR (nn_real)0.005

static uint64_t rng_next(uint64_t *rng) {
    *rng = (*rng * UINT64_C(6364136223846793005)) + UINT64_C(1442695040888963407);
    return *rng;
}

static double teacher_score(const double f[FEATURES]) {
    return f[11] * 0.4 + (1.0 - f[16]) * 0.3 + f[6] * 0.15 + f[13] * 0.1 + f[18] * 0.05;
}

static void random_features(double f[FEATURES], uint64_t *rng) {
    for (int i = 0; i < FEATURES; i++)
        f[i] = (double)(rng_next(rng) % 1000000u) / 1000000.0;
    f[0]  = 1.0;
    f[4]  = (rng_next(rng) % 100 < 30) ? 0.0 : 1.0;
    f[13] = (rng_next(rng) % 100 < 30) ? 1.0 : 0.0;
    f[14] = (rng_next(rng) % 100 < 30) ? 1.0 : 0.0;
    for (int i = 20; i <= 27; i++) f[i] = (rng_next(rng) % 100 < 20) ? 1.0 : 0.0;
    for (int i = 31; i < FEATURES; i++) f[i] = (rng_next(rng) % 100 < 15) ? 1.0 : 0.0;
    f[18] = 0.3 + (double)(rng_next(rng) % 700000u) / 1000000.0;
    f[12] = ((double)(rng_next(rng) % 2000u) - 1000.0) / 1000.0;
    f[30] = teacher_score(f);
}

int main(int argc, char **argv) {
    const char *save_path = "build/float/signal_contract.nnckpt";
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) save_path = argv[++i];

    const size_t layers[] = {FEATURES, HIDDEN0, HIDDEN1, 1};
    NNConfig config;
    memset(&config, 0, sizeof(config));
    config.layer_sizes = layers;
    config.layer_count = LAYERS;
    config.seed = SEED;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_SIGMOID;
    config.loss = NN_LOSS_MSE;

    NeuralNet *net = nn_create_config(&config);
    if (!net) { fprintf(stderr, "create failed\n"); return 1; }

    nn_real *train_x = calloc(TRAIN_SAMPLES * FEATURES, sizeof(nn_real));
    nn_real *train_y = calloc(TRAIN_SAMPLES, sizeof(nn_real));
    nn_real *test_x  = calloc(TEST_SAMPLES * FEATURES, sizeof(nn_real));
    nn_real *test_y  = calloc(TEST_SAMPLES, sizeof(nn_real));

    uint64_t rng = SEED;
    for (int i = 0; i < TRAIN_SAMPLES; i++) {
        double f[FEATURES]; random_features(f, &rng);
        for (int j = 0; j < FEATURES; j++) train_x[i * FEATURES + j] = (nn_real)f[j];
        train_y[i] = (nn_real)f[30];
    }
    for (int i = 0; i < TEST_SAMPLES; i++) {
        double f[FEATURES]; random_features(f, &rng);
        for (int j = 0; j < FEATURES; j++) test_x[i * FEATURES + j] = (nn_real)f[j];
        test_y[i] = (nn_real)f[30];
    }

    printf("Training %d samples, %d epochs...\n", TRAIN_SAMPLES, EPOCHS);

    free(train_x); free(train_y); free(test_x); free(test_y);
    return 0;
}
