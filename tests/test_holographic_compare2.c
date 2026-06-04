#include "nn_holographic.h"
#include "leos_engine.h"
#include "nn_core.h"
#include "nn_train.h"
#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DIM      2048
#define N_CAND      8
#define N_TRAIN    80
#define N_TEST     40
#define N_STATIC   32      /* static features: pure noise, no signal */
#define N_HOLO      4
#define N_BASE     (N_STATIC)
#define N_AUG      (N_STATIC + N_HOLO)

static double eval(int use_holographic,
                   const double *feat_base, const double *feat_aug,
                   const size_t *targets, size_t n, int train_first) {
    size_t fc = use_holographic ? N_AUG : N_BASE;
    const double *feat = use_holographic ? feat_aug : feat_base;

    const size_t layers[] = {fc, 24, 12, 1};
    NNConfig cfg = {layers, 4, (unsigned)(42 + use_holographic * 100),
                    NN_ACTIVATION_RELU, NN_ACTIVATION_LINEAR, NN_LOSS_MSE};
    NeuralNet *net = nn_create_config(&cfg);
    if (!net) return 0;

    if (train_first) {
        for (size_t ep = 0; ep < 15; ep++)
            for (size_t s = 0; s < n; s++)
                nn_train_softmax_scorer(net,
                    (nn_real*)&feat[s * N_CAND * fc], N_CAND, targets[s], 0.02);
    }

    size_t correct = 0;
    for (size_t s = 0; s < n; s++) {
        nn_real scores[N_CAND];
        nn_score_action_rows(net, (nn_real*)&feat[s * N_CAND * fc], N_CAND, scores, NULL);
        int top = 0;
        for (int i = 1; i < N_CAND; i++) if (scores[i] > scores[top]) top = i;
        if ((size_t)top == targets[s]) correct++;
    }
    nn_free(net);
    return train_first ? (double)correct / n : 0.0;
}

int main(void) {
    printf("=== Holographic Features: Strict Side-by-Side ===\n");
    printf("Static features: %d dims of PURE NOISE (zero signal)\n", N_STATIC);
    printf("Holographic: reinforced (state→correct_action) × %d train states\n\n", N_TRAIN);

    if (nn_holographic_init(DIM) != 0) { printf("FAIL: init\n"); return 1; }
    LeosEngine *eng = leos_engine_create(DIM);
    if (!eng) { printf("FAIL: engine\n"); return 1; }

    size_t n_total = N_TRAIN + N_TEST;
    double *fbase = calloc(n_total * N_CAND * N_BASE, sizeof(double));
    double *faug  = calloc(n_total * N_CAND * N_AUG,  sizeof(double));
    size_t *targets = calloc(n_total, sizeof(size_t));
    if (!fbase || !faug || !targets) { printf("FAIL: alloc\n"); return 1; }

    /* Generate: static features = noise, holographic = reinforced signal */
    uint64_t seed = 12345;
    for (size_t s = 0; s < n_total; s++) {
        size_t correct = (size_t)((seed * 7 + 13) % N_CAND);
        targets[s] = correct;
        seed = seed * 1103515245 + 12345;

        /* Generate state and action embeddings */
        double *state = malloc(DIM * sizeof(double));
        double *action = malloc(DIM * sizeof(double));
        leos_engine_keygen(eng, (uint64_t)(s * 3 + 1), state);
        leos_engine_keygen(eng, (uint64_t)(s * 3 + 2), action);

        for (int c = 0; c < N_CAND; c++) {
            double *fb = &fbase[(s * N_CAND + c) * N_BASE];
            double *fa = &faug[(s * N_CAND + c) * N_AUG];

            /* Static: pure random noise */
            for (int j = 0; j < N_STATIC; j++)
                fb[j] = (double)((seed * (j * 31 + c * 17 + 1)) % 10000) / 10000.0 - 0.5;

            memcpy(fa, fb, N_STATIC * sizeof(double));

            /* Holographic: extract features for this (state, candidate) pair */
            nn_holographic_extract_features(state, action, &fa[N_STATIC]);
        }

        /* Reinforce the correct action (train set only) */
        if (s < N_TRAIN) {
            nn_holographic_reinforce(state, action);
            nn_holographic_store_rule(state, action);
        }

        free(state); free(action);
    }

    /* Side-by-side: train on N_TRAIN, test on N_TRAIN+N_TEST */
    printf("First, train on %d states:\n", N_TRAIN);
    printf("  Model   | Train Acc | Test Acc  | Delta\n");
    printf("  --------|-----------|-----------|-------\n");

    for (int rep = 0; rep < 5; rep++) {
        double tr_base = eval(0, fbase, faug, targets, N_TRAIN, 1);
        double tr_aug  = eval(1, fbase, faug, targets, N_TRAIN, 1);
        double te_base = eval(0, fbase + N_TRAIN * N_CAND * N_BASE,
                              faug + N_TRAIN * N_CAND * N_AUG,
                              targets + N_TRAIN, N_TEST, 1);
        double te_aug  = eval(1, fbase + N_TRAIN * N_CAND * N_BASE,
                              faug + N_TRAIN * N_CAND * N_AUG,
                              targets + N_TRAIN, N_TEST, 1);
        printf("  base     | %5.1f%%    | %5.1f%%    |\n", tr_base*100, te_base*100);
        printf("  holo     | %5.1f%%    | %5.1f%%    |  %+5.1f%%\n",
               tr_aug*100, te_aug*100, (te_aug - te_base)*100);
        if (rep < 4) printf("  --------|-----------|-----------|-------\n");
    }

    printf("\n  Random baseline: %.1f%%\n", 100.0/N_CAND);
    nn_holographic_stats();

    free(fbase); free(faug); free(targets);
    leos_engine_destroy(eng);
    nn_holographic_shutdown();
    return 0;
}
