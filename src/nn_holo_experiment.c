#include "nn_holo_experiment.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

static double ns(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec*1e9+ts.tv_nsec; }

HoloExperiment *holo_experiment_create(size_t dim) {
    HoloExperiment *hx = calloc(1, sizeof(HoloExperiment));
    if (!hx) return NULL;
    hx->ranker = holo_ranker_create(dim);
    if (!hx->ranker) { free(hx); return NULL; }
    hx->dim = dim;
    return hx;
}

void holo_experiment_destroy(HoloExperiment *hx) {
    if (!hx) return;
    holo_ranker_destroy(hx->ranker);
    free(hx);
}

void holo_experiment_train(HoloExperiment *hx,
                           const double *state, const double *action) {
    holo_ranker_store(hx->ranker, state, action);
    hx->train_count++;
}

void holo_experiment_score(HoloExperiment *hx,
                           const double *state,
                           const double *action_vecs,
                           size_t n_candidates,
                           double *scores_out,
                           size_t true_label) {
    double t0 = ns();
    holo_ranker_score(hx->ranker, state, action_vecs, n_candidates, scores_out);
    hx->total_time_ns += (ns() - t0);

    /* Softmax + top-1 */
    double max_s = scores_out[0];
    for (size_t i = 1; i < n_candidates; i++)
        if (scores_out[i] > max_s) max_s = scores_out[i];

    double sum_exp = 0;
    for (size_t i = 0; i < n_candidates; i++)
        sum_exp += exp(scores_out[i] - max_s);

    double prob = exp(scores_out[true_label] - max_s) / sum_exp;
    hx->sum_nll += -log(prob + 1e-10);

    size_t best = 0;
    for (size_t i = 1; i < n_candidates; i++)
        if (scores_out[i] > scores_out[best]) best = i;
    if (best == true_label) hx->correct++;

    hx->test_count++;
}

double holo_experiment_accuracy(const HoloExperiment *hx) {
    return hx->test_count > 0 ? (double)hx->correct / hx->test_count : 0;
}

double holo_experiment_nll(const HoloExperiment *hx) {
    return hx->test_count > 0 ? hx->sum_nll / hx->test_count : 0;
}

double holo_experiment_avg_latency_us(const HoloExperiment *hx) {
    return hx->test_count > 0 ? hx->total_time_ns / hx->test_count / 1000 : 0;
}

void holo_experiment_print_compare(const char *name,
                                   double neural_acc, double neural_nll,
                                   const HoloExperiment *hx) {
    double holo_acc = holo_experiment_accuracy(hx);
    double holo_nll = holo_experiment_nll(hx);
    double delta = holo_acc - neural_acc;

    printf("  %-20s | %6.1f%% | %6.1f%% | %+6.1f%% | %6.3f | %6.3f | %5.0f us\n",
           name,
           neural_acc * 100, holo_acc * 100, delta * 100,
           neural_nll, holo_nll,
           holo_experiment_avg_latency_us(hx));
}
