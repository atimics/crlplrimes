/*
 * Side-by-side: holographic features vs baseline.
 *
 * Simulates a simplified prime-gap scoring scenario:
 *   - 5 candidate gaps per state
 *   - 20 training states, 10 test states
 *   - One "correct" gap per state (deterministic target)
 *   - Neural scorer: 96 features static, +4 holographic
 *
 * Compares top-1 accuracy and NLL with vs without holographic features.
 */
#include "nn_holographic.h"
#include "leos_engine.h"
#include "nn_core.h"
#include "nn_train.h"
#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DIM       2048
#define N_CAND     5
#define N_TRAIN   50
#define N_TEST    50
#define N_FEAT    96
#define N_HF       4
#define N_TOTAL  (N_FEAT + N_HF)

/* ---- Simple MLP scorer ---- */
static nn_real train_scorer(const nn_real *inputs, const size_t *targets,
                            size_t n_samples, size_t feat_count) {
    const size_t layers[] = {feat_count, 32, 16, 1};
    NNConfig cfg = {layers, 4, 42, NN_ACTIVATION_RELU, NN_ACTIVATION_LINEAR, NN_LOSS_MSE};
    NeuralNet *net = nn_create_config(&cfg);
    if (!net) return NAN;

    nn_real loss = 0;
    for (size_t ep = 0; ep < 10; ep++) {
        for (size_t s = 0; s < n_samples; s++) {
            loss = nn_train_softmax_scorer(net,
                &inputs[s * N_CAND * feat_count], N_CAND, targets[s], 0.01);
        }
    }
    nn_free(net);
    return loss;
}

static double eval_scorer(const nn_real *inputs, const size_t *targets,
                          size_t n_samples, size_t feat_count, double *out_nll) {
    const size_t layers[] = {feat_count, 32, 16, 1};
    NNConfig cfg = {layers, 4, 42, NN_ACTIVATION_RELU, NN_ACTIVATION_LINEAR, NN_LOSS_MSE};
    NeuralNet *net = nn_create_config(&cfg);
    if (!net) return 0;

    /* Train */
    for (size_t ep = 0; ep < 10; ep++)
        for (size_t s = 0; s < n_samples; s++)
            nn_train_softmax_scorer(net,
                &inputs[s * N_CAND * feat_count], N_CAND, targets[s], 0.01);

    /* Evaluate */
    size_t correct = 0;
    double sum_nll = 0;
    for (size_t s = 0; s < n_samples; s++) {
        nn_real scores[N_CAND];
        nn_score_action_rows(net, &inputs[s * N_CAND * feat_count], N_CAND, scores, NULL);

        /* Softmax */
        nn_real max_s = scores[0];
        for (int i = 1; i < N_CAND; i++) if (scores[i] > max_s) max_s = scores[i];
        nn_real sum_exp = 0;
        for (int i = 0; i < N_CAND; i++) sum_exp += exp(scores[i] - max_s);
        nn_real probs[N_CAND];
        for (int i = 0; i < N_CAND; i++) probs[i] = exp(scores[i] - max_s) / sum_exp;

        /* Top-1 */
        int top = 0;
        for (int i = 1; i < N_CAND; i++) if (scores[i] > scores[top]) top = i;
        if ((size_t)top == targets[s]) correct++;

        /* NLL */
        sum_nll += -log(probs[targets[s]] + 1e-10);
    }

    nn_free(net);
    *out_nll = sum_nll / n_samples;
    return (double)correct / n_samples;
}

/* ---- Generate data ---- */
static void generate_data(double *feat96, double *feat100, size_t *targets, size_t n) {
    LeosEngine *eng = leos_engine_create(DIM);
    nn_holographic_init(DIM);

    /* Seed each state's "correct" candidate deterministically */
    uint64_t seed = 12345;
    for (size_t s = 0; s < n; s++) {
        size_t correct = (size_t)(seed % N_CAND);
        targets[s] = correct;
        seed = seed * 1103515245 + 12345;

        /* Build 96 static features (random for demo) */
        for (int c = 0; c < N_CAND; c++) {
            double *f96 = &feat96[(s * N_CAND + c) * N_FEAT];
            double *f100 = &feat100[(s * N_CAND + c) * N_TOTAL];

            for (int j = 0; j < N_FEAT; j++) {
                double v = (double)((seed * (j + 1) * (c + 1)) % 1000) / 1000.0 - 0.5;
                f96[j] = (nn_real)v;
                f100[j] = (nn_real)v;
            }

            /* Feature 0 encodes the correct answer (leakage for the demo) */
            f96[0] = (nn_real)(c == (int)correct ? 1.0 : 0.0);
            f100[0] = (nn_real)(c == (int)correct ? 1.0 : 0.0);
        }
    }

    /* For the holographic features: reinforce the correct associations */
    double *state = malloc(DIM * sizeof(double));
    double *action = malloc(DIM * sizeof(double));
    for (size_t s = 0; s < n; s++) {
        leos_engine_keygen(eng, (uint64_t)(s * 2 + 1), state);
        leos_engine_keygen(eng, (uint64_t)(s * 2 + 2), action);
        nn_holographic_reinforce(state, action);
    }
    free(state); free(action);
    leos_engine_destroy(eng);
}

/* ---- Main ---- */
int main(void) {
    printf("=== Holographic Features: Side-by-Side Comparison ===\n\n");

    size_t n_total = N_TRAIN + N_TEST;

    /* Allocate feature arrays */
    double *feat96  = calloc(n_total * N_CAND * N_FEAT, sizeof(double));
    double *feat100 = calloc(n_total * N_CAND * N_TOTAL, sizeof(double));
    size_t *targets = calloc(n_total, sizeof(size_t));
    if (!feat96 || !feat100 || !targets) { printf("FAIL: alloc\n"); return 1; }

    /* Generate data */
    generate_data(feat96, feat100, targets, n_total);

    /* Run comparison 5 times with different random seeds */
    printf("Run | Baseline (96 feat) | Holographic (100 feat) | Delta\n");
    printf("----|--------------------|------------------------|-------\n");

    for (int run = 0; run < 5; run++) {
        double nll96, nll100;
        double acc96  = eval_scorer((nn_real*)feat96,  targets, N_TRAIN, N_FEAT,  &nll96);
        double acc100 = eval_scorer((nn_real*)feat100, targets, N_TRAIN, N_TOTAL, &nll100);

        printf("  %d  |  acc=%.1f%% nll=%.3f |  acc=%.1f%% nll=%.3f  |  %+.1f%%\n",
               run + 1,
               acc96 * 100, nll96,
               acc100 * 100, nll100,
               (acc100 - acc96) * 100);
    }

    /* Also test on held-out data */
    printf("\nHeld-out test set (%d samples):\n", N_TEST);
    for (int run = 0; run < 3; run++) {
        double nll96, nll100;
        double acc96  = eval_scorer((nn_real*)(feat96 + N_TRAIN * N_CAND * N_FEAT),
                                    targets + N_TRAIN, N_TEST, N_FEAT, &nll96);
        double acc100 = eval_scorer((nn_real*)(feat100 + N_TRAIN * N_CAND * N_TOTAL),
                                    targets + N_TRAIN, N_TEST, N_TOTAL, &nll100);
        printf("  run %d: baseline=%.1f%%  holographic=%.1f%%  delta=%+.1f%%\n",
               run + 1, acc96 * 100, acc100 * 100, (acc100 - acc96) * 100);
    }

    nn_holographic_shutdown();
    free(feat96); free(feat100); free(targets);
    return 0;
}
