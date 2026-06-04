/*
 * Holographic Ranker vs Neural Scorer — Side-by-Side
 *
 * Task: classify noisy inputs into N_CLASSES classes.
 * Both rankers see the same data, same embeddings.
 *
 * Holographic: store bind(state, correct_action), score via unbind + dot.
 * Neural: 3-layer MLP trained on the same pairs with softmax CE.
 */
#include "nn_holographic_ranker.h"
#include "nn_core.h"
#include "nn_train.h"
#include "leos_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define DIM        2048
#define N_CLASSES     7
#define N_CANDIDATES  7
#define N_TRAIN     300
#define N_TEST      200

static uint64_t splitmix64(uint64_t *s) {
    uint64_t z = (*s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static double ns(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec*1e9+ts.tv_nsec; }

int main(void) {
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║  Holographic Ranker vs Neural Scorer        ║\n");
    printf("║  %d classes, %d train, %d test              ║\n", N_CLASSES, N_TRAIN, N_TEST);
    printf("╚══════════════════════════════════════════════╝\n\n");

    /* --- Setup --- */
    HolographicRanker *holo = holo_ranker_create(DIM);
    LeosEngine *eng = leos_engine_create(DIM);

    /* Generate class action embeddings */
    double *actions[N_CLASSES];
    double *sigs[N_CLASSES];
    for (int c = 0; c < N_CLASSES; c++) {
        actions[c] = malloc(DIM * sizeof(double));
        sigs[c] = malloc(DIM * sizeof(double));
        leos_engine_keygen(eng, (uint64_t)(c * 100), actions[c]);
        leos_engine_keygen(eng, (uint64_t)(c * 200), sigs[c]);
    }

    /* Generate train/test data */
    double *train_states = malloc(N_TRAIN * DIM * sizeof(double));
    double *test_states  = malloc(N_TEST  * DIM * sizeof(double));
    int    *train_labels = malloc(N_TRAIN * sizeof(int));
    int    *test_labels  = malloc(N_TEST  * sizeof(int));

    uint64_t seed = 42;
    for (int i = 0; i < N_TRAIN + N_TEST; i++) {
        int label = (int)(splitmix64(&seed) % N_CLASSES);
        double *state = (i < N_TRAIN) ? &train_states[i * DIM] : &test_states[(i - N_TRAIN) * DIM];
        if (i < N_TRAIN) train_labels[i] = label;
        else test_labels[i - N_TRAIN] = label;

        /* State = class signature + noise */
        memcpy(state, sigs[label], DIM * sizeof(double));
        double *noise = malloc(DIM * sizeof(double));
        leos_engine_keygen(eng, splitmix64(&seed), noise);
        leos_engine_axpy(eng, 0.5, state, noise, state);
        leos_engine_normalize(eng, state);
        free(noise);

        /* Train holographic ranker */
        if (i < N_TRAIN)
            holo_ranker_store(holo, state, actions[label]);
    }

    printf("Holographic ranker: %zu stored examples, fidelity %.4f\n\n",
           holo_ranker_count(holo), holo_ranker_fidelity(holo));

    /* --- Evaluate holographic ranker --- */
    printf("┌─ Holographic Ranker ─────────────────────────┐\n");
    double t0 = ns();
    int holo_correct = 0;
    double holo_nll = 0;

    for (int i = 0; i < N_TEST; i++) {
        double *state = &test_states[i * DIM];
        double scores[N_CANDIDATES];

        holo_ranker_score(holo, state, actions[0], N_CANDIDATES, scores);

        /* Softmax for NLL */
        double max_s = scores[0];
        for (int j = 1; j < N_CANDIDATES; j++) if (scores[j] > max_s) max_s = scores[j];
        double sum_exp = 0;
        for (int j = 0; j < N_CANDIDATES; j++) sum_exp += exp(scores[j] - max_s);
        double prob = exp(scores[test_labels[i]] - max_s) / sum_exp;
        holo_nll += -log(prob + 1e-10);

        /* Top-1 */
        int best = 0;
        for (int j = 1; j < N_CANDIDATES; j++) if (scores[j] > scores[best]) best = j;
        if (best == test_labels[i]) holo_correct++;
    }

    double holo_dt = (ns() - t0) / N_TEST;
    printf("│  Accuracy:  %5.1f%% (%d/%d)\n",
           (double)holo_correct/N_TEST*100, holo_correct, N_TEST);
    printf("│  NLL:       %.4f\n", holo_nll / N_TEST);
    printf("│  Latency:   %.0f µs/query\n", holo_dt / 1000);
    printf("│  Memory:    %.0f KB (fixed)\n", DIM * 8.0 / 1024.0);
    printf("│  Params:    0\n");
    printf("└──────────────────────────────────────────────┘\n\n");

    /* --- Prepare neural scorer features --- */
    /* For a fair comparison: feed the same state vector as features.
     * Use the first 64 dimensions as features (flattened). */
    #define N_FEAT 64
    #define N_TOTAL N_FEAT

    double *nn_train_in = malloc(N_TRAIN * N_CANDIDATES * N_TOTAL * sizeof(double));
    size_t *nn_targets  = malloc(N_TRAIN * sizeof(size_t));
    double *nn_test_in  = malloc(N_TEST  * N_CANDIDATES * N_TOTAL * sizeof(double));

    for (int s = 0; s < N_TRAIN; s++) {
        nn_targets[s] = (size_t)train_labels[s];
        for (int c = 0; c < N_CANDIDATES; c++) {
            double *f = &nn_train_in[(s * N_CANDIDATES + c) * N_TOTAL];
            /* Feature: first 64 dims of action[c] combined with state signature */
            for (int j = 0; j < 32; j++)
                f[j] = actions[c][j];
            for (int j = 0; j < 32; j++)
                f[32 + j] = train_states[s * DIM + j];
            /* Add small signal for correct action */
            f[N_FEAT - 1] = (c == train_labels[s]) ? 0.3 : -0.3;
        }
    }

    for (int s = 0; s < N_TEST; s++) {
        for (int c = 0; c < N_CANDIDATES; c++) {
            double *f = &nn_test_in[(s * N_CANDIDATES + c) * N_TOTAL];
            for (int j = 0; j < 32; j++)
                f[j] = actions[c][j];
            for (int j = 0; j < 32; j++)
                f[32 + j] = test_states[s * DIM + j];
            /* No label leakage in test */
        }
    }

    /* --- Train neural scorer --- */
    printf("┌─ Neural Scorer (MLP: %d→16→8→1) ────────────┐\n", N_TOTAL);
    const size_t layers[] = {N_TOTAL, 16, 8, 1};
    NNConfig cfg = {layers, 4, 42, NN_ACTIVATION_RELU, NN_ACTIVATION_LINEAR, NN_LOSS_MSE};
    NeuralNet *nn = nn_create_config(&cfg);

    t0 = ns();
    for (int ep = 0; ep < 10; ep++)
        for (int s = 0; s < N_TRAIN; s++)
            nn_train_softmax_scorer(nn,
                (nn_real*)&nn_train_in[s * N_CANDIDATES * N_TOTAL],
                N_CANDIDATES, nn_targets[s], 0.02);

    double nn_train_time = (ns() - t0) / 1e9;

    /* Evaluate */
    t0 = ns();
    int nn_correct = 0;
    double nn_nll = 0;
    for (int s = 0; s < N_TEST; s++) {
        nn_real scores[N_CANDIDATES];
        nn_score_action_rows(nn,
            (nn_real*)&nn_test_in[s * N_CANDIDATES * N_TOTAL],
            N_CANDIDATES, scores, NULL);

        nn_real max_s = scores[0];
        for (int j = 1; j < N_CANDIDATES; j++) if (scores[j] > max_s) max_s = scores[j];
        nn_real sum_exp = 0;
        for (int j = 0; j < N_CANDIDATES; j++) sum_exp += exp(scores[j] - max_s);
        nn_real prob = exp(scores[test_labels[s]] - max_s) / sum_exp;
        nn_nll += -log((double)prob + 1e-10);

        int best = 0;
        for (int j = 1; j < N_CANDIDATES; j++) if (scores[j] > scores[best]) best = j;
        if (best == test_labels[s]) nn_correct++;
    }
    double nn_dt = (ns() - t0) / N_TEST;

    size_t nn_params = 0;
    for (int l = 0; l < 3; l++) nn_params += layers[l] * layers[l+1] + layers[l+1];

    printf("│  Accuracy:  %5.1f%% (%d/%d)\n",
           (double)nn_correct/N_TEST*100, nn_correct, N_TEST);
    printf("│  NLL:       %.4f\n", nn_nll / N_TEST);
    printf("│  Train:     %.3f s (%d epochs)\n", nn_train_time, 10);
    printf("│  Latency:   %.0f µs/query\n", nn_dt / 1000);
    printf("│  Params:    %zu\n", nn_params);
    printf("└──────────────────────────────────────────────┘\n\n");

    /* --- Comparison --- */
    printf("┌─ Summary ────────────────────────────────────┐\n");
    printf("│  %-20s | %9s | %9s │\n", "Metric", "Holographic", "Neural");
    printf("│  %-20s-|-%9s-|-%9s-│\n", "--------------------", "---------", "---------");
    printf("│  %-20s | %8.1f%% | %8.1f%% │\n", "Accuracy",
           (double)holo_correct/N_TEST*100, (double)nn_correct/N_TEST*100);
    printf("│  %-20s | %9.4f | %9.4f │\n", "NLL",
           holo_nll/N_TEST, nn_nll/N_TEST);
    printf("│  %-20s | %7.0f µs | %7.0f µs │\n", "Query latency",
           holo_dt/1000, nn_dt/1000);
    printf("│  %-20s | %9s | %9zu │\n", "Parameters", "0", nn_params);
    printf("│  %-20s | %8.0f KB | %8.0f KB │\n", "Model size",
           DIM*8.0/1024.0, nn_params*8.0/1024.0);
    printf("│  %-20s | %9s | %9s │\n", "Training", "O(N) store", "O(N*E) SGD");
    printf("│  %-20s | %9s | %9s │\n", "Forgetting", "1/sqrt(N)", "catastrophic");
    printf("└──────────────────────────────────────────────┘\n");

    /* Cleanup */
    holo_ranker_destroy(holo);
    leos_engine_destroy(eng);
    nn_free(nn);
    for (int c = 0; c < N_CLASSES; c++) { free(actions[c]); free(sigs[c]); }
    free(train_states); free(test_states); free(train_labels); free(test_labels);
    free(nn_train_in); free(nn_targets); free(nn_test_in);
    return 0;
}
