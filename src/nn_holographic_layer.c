#include "nn_holographic_layer.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define LAYER_DEFAULT_K 4
#define LAYER_EPSILON 0.01

struct HolographicLayer {
    double *centroids;
    size_t *counts;
    size_t centroid_count;
    size_t dim;
    size_t feature_dim;
    HolographicRanker *ranker;
    LeosEngine *eng;
    double centroid_lr;
    double *scratch;
};

static void find_k_nearest(const double *dists, size_t n, size_t k,
                           size_t *out_idx, double *out_d) {
    size_t found = 0;
    for (size_t i = 0; i < n && found < k; i++) {
        out_idx[found] = i; out_d[found] = dists[i]; found++;
    }
    if (found == 0) return;
    for (size_t i = 1; i < found; i++) {
        size_t ki = out_idx[i]; double kd = out_d[i]; size_t j = i;
        while (j > 0 && out_d[j-1] > kd) {
            out_idx[j] = out_idx[j-1]; out_d[j] = out_d[j-1]; j--;
        }
        out_idx[j] = ki; out_d[j] = kd;
    }
    for (size_t i = k; i < n; i++) {
        if (dists[i] >= out_d[k-1]) continue;
        size_t pos = k-1;
        while (pos > 0 && out_d[pos-1] > dists[i]) pos--;
        for (size_t j = k-1; j > pos; j--) {
            out_idx[j] = out_idx[j-1]; out_d[j] = out_d[j-1];
        }
        out_idx[pos] = i; out_d[pos] = dists[i];
    }
}

static double feat_dist(const double *a, const double *b, size_t n) {
    double d = 0;
    for (size_t i = 0; i < n; i++) { double df = a[i]-b[i]; d += df*df; }
    return d;
}

HolographicLayer *holo_layer_create(size_t dim, size_t centroid_count,
                                    size_t feature_dim, double lr) {
    HolographicLayer *l = calloc(1, sizeof(*l));
    if (!l) return NULL;
    l->dim = dim;
    l->feature_dim = feature_dim;
    l->centroid_count = centroid_count;
    l->centroid_lr = lr;
    l->ranker = holo_ranker_create(dim);
    l->eng = leos_engine_create(dim);
    l->centroids = calloc(centroid_count * feature_dim, sizeof(double));
    l->counts = calloc(centroid_count, sizeof(size_t));
    l->scratch = calloc(dim * 2, sizeof(double));
    if (!l->ranker || !l->eng || !l->centroids || !l->counts || !l->scratch) {
        holo_layer_destroy(l); return NULL;
    }
    /* Random centroids via leos keygen with increasing seeds */
    double *tmp = calloc(dim, sizeof(double));
    if (tmp) {
        for (size_t i = 0; i < centroid_count; i++) {
            leos_engine_keygen(l->eng, 14695981039346656037ULL ^ (i*2654435761ULL+42), tmp);
            leos_engine_normalize(l->eng, tmp);
            for (size_t j = 0; j < feature_dim && j < dim; j++)
                l->centroids[i*feature_dim + j] = tmp[j];
        }
        free(tmp);
    }
    return l;
}

void holo_layer_destroy(HolographicLayer *l) {
    if (!l) return;
    if (l->ranker) holo_ranker_destroy(l->ranker);
    if (l->eng) leos_engine_destroy(l->eng);
    free(l->centroids);
    free(l->counts);
    free(l->scratch);
    free(l);
}

void holo_layer_store(HolographicLayer *l,
                      const double *feature, const double *item) {
    if (!l || !feature || !item) return;
    size_t k = LAYER_DEFAULT_K;
    if (k > l->centroid_count) k = l->centroid_count;

    double *dists = calloc(l->centroid_count, sizeof(double));
    size_t *active = calloc(k, sizeof(size_t));
    double *ad = calloc(k, sizeof(double));
    if (!dists || !active || !ad) { free(dists); free(active); free(ad); return; }

    for (size_t i = 0; i < l->centroid_count; i++)
        dists[i] = feat_dist(feature, &l->centroids[i*l->feature_dim], l->feature_dim);
    find_k_nearest(dists, l->centroid_count, k, active, ad);
    free(dists); free(ad);

    for (size_t i = 0; i < k; i++) {
        size_t ci = active[i];
        /* Nudge centroid toward feature */
        double *c = &l->centroids[ci * l->feature_dim];
        for (size_t j = 0; j < l->feature_dim; j++)
            c[j] += l->centroid_lr * (feature[j] - c[j]);
        l->counts[ci]++;
        /* Bind feature and item, store in shared ranker */
        leos_engine_bind(l->eng, feature, item, l->scratch);
        holo_ranker_store(l->ranker, feature, l->scratch);
    }
    free(active);
}

void holo_layer_query(const HolographicLayer *l,
                      const double *query, double *recall_out) {
    if (!l || !query || !recall_out) return;
    holo_ranker_recall(l->ranker, query, recall_out);
}

void holo_layer_score(const HolographicLayer *l,
                      const double *recall,
                      const double *candidates,
                      size_t n, double *scores_out) {
    if (!l || !recall || !candidates || !scores_out) return;
    for (size_t i = 0; i < n; i++)
        scores_out[i] = leos_engine_dot(l->eng, recall, &candidates[i*l->dim]);
}

size_t holo_layer_dim(const HolographicLayer *l) { return l ? l->dim : 0; }
size_t holo_layer_centroid_count(const HolographicLayer *l) { return l ? l->centroid_count : 0; }
size_t holo_layer_stored(const HolographicLayer *l) { return l ? holo_ranker_count(l->ranker) : 0; }
double holo_layer_fidelity(const HolographicLayer *l) { return l ? holo_ranker_fidelity(l->ranker) : 0; }
const double *holo_layer_centroids(const HolographicLayer *l) { return l ? l->centroids : NULL; }
const size_t *holo_layer_counts(const HolographicLayer *l) { return l ? l->counts : NULL; }

/* ---- Ordinal selection ---- */

int holo_layer_select(const HolographicLayer *l,
                      const double *query,
                      const double *candidates,
                      size_t n) {
    if (!l || !query || !candidates || n == 0) return -1;
    double *recall = calloc(l->dim, sizeof(double));
    if (!recall) return -1;
    holo_layer_query(l, query, recall);
    double *scores = calloc(n, sizeof(double));
    if (!scores) { free(recall); return -1; }
    holo_layer_score(l, recall, candidates, n, scores);
    int best = 0;
    for (size_t i = 1; i < n; i++)
        if (scores[i] > scores[best]) best = (int)i;
    free(recall); free(scores);
    /* Record result for empirical fidelity:
     * since we don't know the target, check if best == 0
     * as a self-retrieval signal (caller should arrange candidates
     * so target is at index 0). */
    return best;
}

size_t holo_layer_select_top_k(const HolographicLayer *l,
                               const double *query,
                               const double *candidates,
                               size_t n, size_t k,
                               size_t *out) {
    if (!l || !query || !candidates || !out || n == 0 || k == 0) return 0;
    double *recall = calloc(l->dim, sizeof(double));
    if (!recall) return 0;
    holo_layer_query(l, query, recall);
    double *scores = calloc(n, sizeof(double));
    if (!scores) { free(recall); return 0; }
    holo_layer_score(l, recall, candidates, n, scores);
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
    free(recall); free(scores);
    return sel;
}

/* Accuracy tracking embedded in layer ranker */
double holo_layer_empirical_fidelity(const HolographicLayer *l) {
    return l ? holo_ranker_empirical_fidelity(l->ranker) : 0;
}
void holo_layer_record_result(HolographicLayer *l, int correct) {
    if (l) holo_ranker_record_result(l->ranker, correct);
}
void holo_layer_reset_accuracy(HolographicLayer *l) {
    if (l) holo_ranker_reset_accuracy(l->ranker);
}

