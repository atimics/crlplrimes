#include <stdio.h>
#include <inttypes.h>
#include "nn_holographic_ranker.h"
#include "leos_engine.h"
#include <stdlib.h>
#include <string.h>

struct HolographicRanker {
    LeosEngine *eng;
    size_t dim;
    double *M;
    size_t stored;
    size_t acc_correct;
    size_t acc_total;
};

HolographicRanker *holo_ranker_create(size_t dim) {
    LeosEngine *eng = leos_engine_create(dim);
    if (!eng) return NULL;
    HolographicRanker *r = calloc(1, sizeof(HolographicRanker));
    if (!r) { leos_engine_destroy(eng); return NULL; }
    r->eng = eng;
    r->dim = dim;
    r->M = calloc(dim, sizeof(double));
    if (!r->M) { holo_ranker_destroy(r); return NULL; }
    return r;
}

void holo_ranker_destroy(HolographicRanker *r) {
    if (!r) return;
    leos_engine_destroy(r->eng);
    free(r->M);
    free(r);
}

void holo_ranker_store(HolographicRanker *r, const double *state, const double *action) {
    if (!r || !state || !action) return;
    double *bound = calloc(r->dim, sizeof(double));
    if (!bound) return;
    leos_engine_bind(r->eng, state, action, bound);
    leos_engine_add(r->eng, r->M, bound, r->M);
    r->stored++;
    free(bound);
}

void holo_ranker_score(HolographicRanker *r, const double *state,
                       const double *action_vecs, size_t n, double *scores) {
    if (!r || !state || !action_vecs || !scores || n == 0) return;
    double *ctx = calloc(r->dim, sizeof(double));
    if (!ctx) return;
    leos_engine_unbind(r->eng, r->M, state, ctx);
    for (size_t i = 0; i < n; i++)
        scores[i] = leos_engine_dot(r->eng, ctx, &action_vecs[i * r->dim]);
    free(ctx);
}

size_t holo_ranker_count(const HolographicRanker *r) { return r ? r->stored : 0; }
double holo_ranker_fidelity(const HolographicRanker *r) {
    return r ? (double)r->stored : 0;
}

int holo_ranker_recall(HolographicRanker *r, const double *state, double *ctx_out) {
    if (!r || !state || !ctx_out) return -1;
    leos_engine_unbind(r->eng, r->M, state, ctx_out);
    return 0;
}

/* ---- Ordinal selection ---- */

int holo_ranker_select(HolographicRanker *r, const double *state,
                       const double *action_vecs, size_t n) {
    if (!r || !state || !action_vecs || n == 0) return -1;
    double *scores = calloc(n, sizeof(double));
    if (!scores) return -1;
    holo_ranker_score(r, state, action_vecs, n, scores);
    int best = 0;
    for (size_t i = 1; i < n; i++)
        if (scores[i] > scores[best]) best = (int)i;
    free(scores);
    return best;
}

size_t holo_ranker_select_top_k(HolographicRanker *r, const double *state,
                                const double *action_vecs, size_t n, size_t k,
                                size_t *out) {
    if (!r || !state || !action_vecs || !out || n == 0 || k == 0) return 0;
    double *scores = calloc(n, sizeof(double));
    if (!scores) return 0;
    holo_ranker_score(r, state, action_vecs, n, scores);
    size_t sel = k < n ? k : n;
    for (size_t i = 0; i < sel; i++) out[i] = i;
    for (size_t i = 1; i < sel; i++) {
        size_t ki = out[i]; double kd = scores[ki]; size_t j = i;
        while (j > 0 && scores[out[j-1]] < kd) { out[j] = out[j-1]; j--; }
        out[j] = ki;
    }
    for (size_t i = sel; i < n; i++) {
        if (scores[i] <= scores[out[sel-1]]) continue;
        size_t pos = sel-1;
        while (pos > 0 && scores[out[pos-1]] < scores[i]) pos--;
        for (size_t j = sel-1; j > pos; j--) out[j] = out[j-1];
        out[pos] = i;
    }
    free(scores);
    return sel;
}

size_t holo_ranker_correct_count(const HolographicRanker *r) {
    return r ? r->acc_correct : 0;
}

void holo_ranker_record_result(HolographicRanker *r, int correct) {
    if (!r) return;
    r->acc_total++;
    if (correct) r->acc_correct++;
}

void holo_ranker_reset_accuracy(HolographicRanker *r) {
    if (!r) return;
    r->acc_correct = 0; r->acc_total = 0;
}

double holo_ranker_empirical_fidelity(const HolographicRanker *r) {
    if (!r || r->acc_total == 0) return 0;
    return (double)r->acc_correct / (double)r->acc_total;
}

/* ---- Serialization ---- */

int holo_ranker_save(const HolographicRanker *r, const char *path) {
    if (!r || !path) return -1;
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    fwrite(r->M, sizeof(double), r->dim, fp);
    uint64_t sc = (uint64_t)r->stored;
    fwrite(&sc, sizeof(sc), 1, fp);
    fclose(fp);
    return 0;
}

int holo_ranker_load(HolographicRanker *r, const char *path) {
    if (!r || !path) return -1;
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    size_t nr = fread(r->M, sizeof(double), r->dim, fp);
    if (nr != r->dim) { fclose(fp); return -1; }
    uint64_t sc = 0;
    if (fread(&sc, sizeof(sc), 1, fp) != 1) sc = 0;
    r->stored = (size_t)sc;
    fclose(fp);
    return 0;
}
