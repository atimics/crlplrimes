#ifndef NN_HOLO_RUNNER_H
#define NN_HOLO_RUNNER_H

#include <stddef.h>
#include <stdint.h>

typedef struct HoloRunner {
    struct HolographicRanker *ranker;
    struct LeosEngine *eng;
    size_t dim, train_count, test_count, correct;
    double sum_nll, total_time_ns, neural_acc, neural_nll;
    int neural_ran;
} HoloRunner;

HoloRunner *holo_runner_create(size_t dim);
void holo_runner_destroy(HoloRunner *hr);

void holo_runner_train(HoloRunner *hr, const double *state, const double *action);
void holo_runner_eval(HoloRunner *hr, const double *state, const double *action_vecs,
                      size_t n_candidates, double *scores_out, size_t true_label);

void holo_runner_neural_baseline(HoloRunner *hr,
                                 const double *train_features, const int *train_labels,
                                 const double *test_features,  const int *test_labels,
                                 size_t n_train, size_t n_test,
                                 size_t n_features, size_t n_candidates);

double holo_runner_accuracy(const HoloRunner *hr);
double holo_runner_nll(const HoloRunner *hr);
void   holo_runner_print_compare(const HoloRunner *hr);

int    holo_runner_save(const HoloRunner *hr, const char *path);
int    holo_runner_load(HoloRunner *hr, const char *path);

void   holo_runner_embed_state(HoloRunner *hr, uint64_t seed, double *out);
size_t holo_runner_dim(const HoloRunner *hr);

#endif /* NN_HOLO_RUNNER_H */
