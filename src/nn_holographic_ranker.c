#include <stdio.h>
#include <inttypes.h>
#include "nn_holographic_ranker.h"
#include "leos_engine.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct HolographicRanker {
    LeosEngine *eng;
    size_t dim;
    double *M;         /* memory trace accumulator */
    size_t stored;
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

void holo_ranker_store(HolographicRanker *r,
                       const double *state, const double *action) {
    /* bind(state, action) → accumulate into M */
    double *pair = malloc(r->dim * sizeof(double));
    leos_engine_bind(r->eng, state, action, pair);
    leos_engine_add(r->eng, r->M, pair, r->M);
    free(pair);
    r->stored++;
}

void holo_ranker_score(HolographicRanker *r,
                       const double *state,
                       const double *action_vecs,
                       size_t n_candidates,
                       double *scores_out) {
    if (r->stored == 0) {
        /* No training data: uniform scores */
        for (size_t i = 0; i < n_candidates; i++) scores_out[i] = 0.0;
        return;
    }

    /* Normalize memory trace onto hypersphere */
    double *M_norm = malloc(r->dim * sizeof(double));
    memcpy(M_norm, r->M, r->dim * sizeof(double));
    leos_engine_normalize(r->eng, M_norm);

    /* Retrieve context: unbind(M_norm, state) */
    double *context = malloc(r->dim * sizeof(double));
    leos_engine_unbind(r->eng, M_norm, state, context);

    /* Score each candidate: dot(context, action_i) */
    for (size_t i = 0; i < n_candidates; i++) {
        scores_out[i] = leos_engine_dot(r->eng, context,
                                        &action_vecs[i * r->dim]);
    }

    free(M_norm);
    free(context);
}

size_t holo_ranker_count(const HolographicRanker *r) {
    return r->stored;
}

double holo_ranker_fidelity(const HolographicRanker *r) {
    if (r->stored == 0) return 1.0;
    return 1.0 / sqrt((double)r->stored);
}

int holo_ranker_save(const HolographicRanker *r, const char *path) {
    if (!r || !path) return -1;
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    /* Write header: dim, stored count */
    uint64_t header[2] = {r->dim, r->stored};
    if (fwrite(header, sizeof(uint64_t), 2, fp) != 2) { fclose(fp); return -1; }
    /* Write memory trace */
    if (fwrite(r->M, sizeof(double), r->dim, fp) != r->dim) { fclose(fp); return -1; }
    fclose(fp);
    return 0;
}

int holo_ranker_load(HolographicRanker *r, const char *path) {
    if (!r || !path) return -1;
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    uint64_t header[2];
    if (fread(header, sizeof(uint64_t), 2, fp) != 2) { fclose(fp); return -1; }
    if (header[0] != r->dim) { fclose(fp); return -1; } /* dimension mismatch */
    r->stored = (size_t)header[1];
    if (fread(r->M, sizeof(double), r->dim, fp) != r->dim) { fclose(fp); return -1; }
    fclose(fp);
    return 0;
}

int holo_ranker_recall(HolographicRanker *r,
                       const double *state,
                       double *context_out) {
    if (r->stored == 0) {
        memset(context_out, 0, r->dim * sizeof(double));
        return -1;
    }
    double *M_norm = malloc(r->dim * sizeof(double));
    memcpy(M_norm, r->M, r->dim * sizeof(double));
    leos_engine_normalize(r->eng, M_norm);
    leos_engine_unbind(r->eng, M_norm, state, context_out);
    free(M_norm);
    return 0;
}
