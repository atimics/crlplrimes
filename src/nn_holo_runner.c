#include "nn_holo_runner.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

HoloRunner *holo_runner_create(size_t dim) {
    HoloRunner *hr = calloc(1, sizeof(HoloRunner));
    if (!hr) return NULL;
    hr->ranker = holo_ranker_create(dim);
    hr->eng = leos_engine_create(dim);
    hr->dim = dim;
    if (!hr->ranker || !hr->eng) { holo_runner_destroy(hr); return NULL; }
    return hr;
}

void holo_runner_destroy(HoloRunner *hr) {
    if (!hr) return;
    holo_ranker_destroy(hr->ranker);
    leos_engine_destroy(hr->eng);
    free(hr);
}

void holo_runner_train(HoloRunner *hr, const double *s, const double *a) {
    holo_ranker_store(hr->ranker, s, a);
    hr->train_count++;
}

void holo_runner_eval(HoloRunner *hr, const double *st, const double *av,
                      size_t nc, double *sc, size_t tl) {
    holo_ranker_score(hr->ranker, st, av, nc, sc);
    double mx = sc[0];
    for (size_t i = 1; i < nc; i++) if (sc[i] > mx) mx = sc[i];
    double se = 0;
    for (size_t i = 0; i < nc; i++) se += exp(sc[i] - mx);
    double pr = exp(sc[tl] - mx) / se;
    hr->sum_nll += -log(pr + 1e-10);
    size_t b = 0;
    for (size_t i = 1; i < nc; i++) if (sc[i] > sc[b]) b = i;
    if (b == tl) hr->correct++;
    hr->test_count++;
}

double holo_runner_accuracy(const HoloRunner *hr) {
    return hr->test_count ? (double)hr->correct / hr->test_count : 0;
}

double holo_runner_nll(const HoloRunner *hr) {
    return hr->test_count ? hr->sum_nll / hr->test_count : 0;
}

int holo_runner_save(const HoloRunner *hr, const char *p) {
    return holo_ranker_save(hr->ranker, p);
}

int holo_runner_load(HoloRunner *hr, const char *p) {
    return holo_ranker_load(hr->ranker, p);
}

void holo_runner_embed_state(HoloRunner *hr, uint64_t seed, double *out) {
    leos_engine_keygen(hr->eng, seed, out);
}

size_t holo_runner_dim(const HoloRunner *hr) {
    return hr->dim;
}
