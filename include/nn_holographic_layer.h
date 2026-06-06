#ifndef NN_HOLOGRAPHIC_LAYER_H
#define NN_HOLOGRAPHIC_LAYER_H

#include "nn_holographic_ranker.h"
#include "holographic/leos_engine.h"
#include <stddef.h>

/*
 * HolographicLayer — generalized hidden layer for holographic composition.
 *
 * Generalizes the centroid-routing pattern from MemoryNeuronGrid into a
 * reusable type that can be stacked. One shared HolographicRanker stores
 * all items. C centroids route writes to the ranker by feature-space
 * proximity and drift toward stored items via online clustering.
 *
 * Unlike MemoryNeuronGrid: no two-tick split, no per-neuron context
 * extraction. Store on every call, query anytime. The layer exposes
 * raw holographic recall via unbind.
 *
 * Dimensions must be power-of-2 and consistent across stacked layers.
 */

typedef struct HolographicLayer HolographicLayer;

HolographicLayer *holo_layer_create(size_t dim,
                                    size_t centroid_count,
                                    size_t feature_dim,
                                    double centroid_lr);
void holo_layer_destroy(HolographicLayer *layer);

void holo_layer_store(HolographicLayer *layer,
                      const double *feature,
                      const double *item);

void holo_layer_query(const HolographicLayer *layer,
                      const double *query,
                      double *recall_out);

void holo_layer_score(const HolographicLayer *layer,
                      const double *recall,
                      const double *candidates,
                      size_t n,
                      double *scores_out);

size_t holo_layer_dim(const HolographicLayer *layer);
size_t holo_layer_centroid_count(const HolographicLayer *layer);
size_t holo_layer_stored(const HolographicLayer *layer);
double holo_layer_fidelity(const HolographicLayer *layer);
const double *holo_layer_centroids(const HolographicLayer *layer);
const size_t *holo_layer_counts(const HolographicLayer *layer);

#endif

/* ---- Ordinal selection ---- */

/* Select the single best candidate from n dim-dimensional vectors.
 * Returns index 0..n-1, or -1 on error. */
int holo_layer_select(const HolographicLayer *layer,
                      const double *query,
                      const double *candidates,
                      size_t n);

/* Select top-K candidates. Returns count selected (min(k,n)). */
size_t holo_layer_select_top_k(const HolographicLayer *layer,
                               const double *query,
                               const double *candidates,
                               size_t n,
                               size_t k,
                               size_t *out_indices);

/* Empirical fidelity: fraction of queries where select() returned the
 * correct target, or 0 if no queries recorded. */
double holo_layer_empirical_fidelity(const HolographicLayer *layer);
void holo_layer_record_result(HolographicLayer *layer, int correct);
void holo_layer_reset_accuracy(HolographicLayer *layer);

