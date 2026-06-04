#ifndef NN_HOLOGRAPHIC_RANKER_H
#define NN_HOLOGRAPHIC_RANKER_H

#include <stddef.h>

/*
 * Holographic Ranker — zero-parameter action scorer.
 *
 * Replaces the neural scorer (MLP) with holographic k-NN:
 *
 *   Store:   M += bind(state_embedding, action_embedding)
 *   Recall:  context = unbind(M, state_embedding)
 *   Rank:    score_i = dot(context, action_embedding_i)
 *
 * No weights. No gradient descent. No epochs.
 * The model IS the memory trace. Training IS storing examples.
 *
 * Memory: O(dim) fixed regardless of how many examples are stored.
 * Fidelity: 1/sqrt(N) theoretical retrieval quality.
 */

typedef struct HolographicRanker HolographicRanker;

/* Create a ranker with given dimension (power of 2). */
HolographicRanker *holo_ranker_create(size_t dim);

/* Destroy. */
void holo_ranker_destroy(HolographicRanker *r);

/* Store a verified (state, action) pair.
 * Both are dim-dimensional unit-norm vectors. */
void holo_ranker_store(HolographicRanker *r,
                       const double *state, const double *action);

/* Score candidates: for each candidate action embedding, compute
 *   score[i] = dot(unbind(M, state), action_i)
 *
 * state:          dim-dimensional query vector
 * action_vecs:    n_candidates * dim flat array of action embeddings
 * n_candidates:   number of candidates
 * scores_out:     n_candidates scores (higher = better match)
 */
void holo_ranker_score(HolographicRanker *r,
                       const double *state,
                       const double *action_vecs,
                       size_t n_candidates,
                       double *scores_out);

/* Number of stored examples. */
size_t holo_ranker_count(const HolographicRanker *r);

/* Estimated retrieval fidelity. */
double holo_ranker_fidelity(const HolographicRanker *r);

#endif /* NN_HOLOGRAPHIC_RANKER_H */

/* Serialization: save/load the memory trace.
 * Format: raw double array (dim * sizeof(double) bytes) + uint64_t stored_count. */
int holo_ranker_save(const HolographicRanker *r, const char *path);
int holo_ranker_load(HolographicRanker *r, const char *path);
