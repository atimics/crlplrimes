#ifndef NN_HOLO_EXPERIMENT_H
#define NN_HOLO_EXPERIMENT_H

#include "nn_holographic_ranker.h"
#include <stddef.h>

/*
 * Holographic experiment runner — drop-in for any crlplrimes task.
 *
 * Usage in experiment code:
 *
 *   HoloExperiment *hx = holo_experiment_create(2048);
 *
 *   // For each training state with known correct action:
 *   holo_experiment_train(hx, state_embedding, action_embedding);
 *
 *   // For each test state:
 *   holo_experiment_score(hx, state_embedding, action_embeddings,
 *                          n_candidates, scores);
 *
 *   // Compare against neural baseline:
 *   holo_experiment_compare(neural_accuracy, neural_nll,
 *                           holo_accuracy, holo_nll);
 */

typedef struct {
    HolographicRanker *ranker;
    size_t dim;
    size_t train_count;
    size_t test_count;
    size_t correct;
    double sum_nll;
    double total_time_ns;
} HoloExperiment;

HoloExperiment *holo_experiment_create(size_t dim);
void holo_experiment_destroy(HoloExperiment *hx);

/* Store a training example */
void holo_experiment_train(HoloExperiment *hx,
                           const double *state, const double *action);

/* Score candidates for a test state */
void holo_experiment_score(HoloExperiment *hx,
                           const double *state,
                           const double *action_vecs,
                           size_t n_candidates,
                           double *scores_out,
                           size_t true_label);

/* Get accumulated accuracy */
double holo_experiment_accuracy(const HoloExperiment *hx);
double holo_experiment_nll(const HoloExperiment *hx);
double holo_experiment_avg_latency_us(const HoloExperiment *hx);

/* Dump comparison table */
void holo_experiment_print_compare(const char *experiment_name,
                                   double neural_acc, double neural_nll,
                                   const HoloExperiment *hx);

#endif /* NN_HOLO_EXPERIMENT_H */
