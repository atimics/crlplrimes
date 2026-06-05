#ifndef NN_HOLO_RUNNER_H
#define NN_HOLO_RUNNER_H

#include "nn_holographic_ranker.h"
#include "holographic/leos_engine.h"
#include <stddef.h>
#include <stdint.h>

/*
 * Compatibility shim for sandbox experiments (haiku_word_lm) that
 * still reference the old HoloRunner API.
 *
 * The real architecture is now MemoryNeuronGrid. This wrapper keeps
 * the old HoloRunner type as a thin shell around HolographicRanker
 * so existing sandbox code compiles without a full rewrite.
 */

typedef struct HoloRunner {
    HolographicRanker *ranker;
    LeosEngine *eng;
    size_t dim;
    size_t train_count;
    size_t test_count;
    size_t correct;
    double sum_nll;
    double total_time_ns;
    double neural_acc;
    double neural_nll;
    int neural_ran;
} HoloRunner;

HoloRunner *holo_runner_create(size_t dim);
void holo_runner_destroy(HoloRunner *hr);
void holo_runner_train(HoloRunner *hr, const double *state, const double *action);
void holo_runner_eval(HoloRunner *hr, const double *state, const double *action_vecs,
                      size_t n_candidates, double *scores_out, size_t true_label);
double holo_runner_accuracy(const HoloRunner *hr);
double holo_runner_nll(const HoloRunner *hr);
int holo_runner_save(const HoloRunner *hr, const char *path);
int holo_runner_load(HoloRunner *hr, const char *path);
void holo_runner_embed_state(HoloRunner *hr, uint64_t seed, double *out);
size_t holo_runner_dim(const HoloRunner *hr);

#endif /* NN_HOLO_RUNNER_H */
