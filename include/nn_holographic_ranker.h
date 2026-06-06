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
 * Memory: O(dim) fixed regardless of how many examples are stored.
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

/* Stored count (used for display only; empirical fidelity is the real metric). */
double holo_ranker_fidelity(const HolographicRanker *r);

/* Recall: return the context vector from unbind(M, state).
 * Equivalent to the first two steps of holo_ranker_score.
 * context_out must point to dim doubles. Returns 0 on success. */
int holo_ranker_recall(HolographicRanker *r,
                       const double *state,
                       double *context_out);

#endif /* NN_HOLOGRAPHIC_RANKER_H */

/* Serialization: save/load the memory trace.
 * Format: raw double array (dim * sizeof(double) bytes) + uint64_t stored_count. */
int holo_ranker_save(const HolographicRanker *r, const char *path);
int holo_ranker_load(HolographicRanker *r, const char *path);

/* ---- Ordinal ranking: select, don't score ---- */

/* Select the single best candidate: returns the index (0..n-1) of the
 * highest-scoring candidate. Ordinal — no floating-point scores returned.
 * Returns -1 if no candidates. */
int holo_ranker_select(HolographicRanker *r,
                       const double *state,
                       const double *action_vecs,
                       size_t n);

/* Select top-K candidates: fills out_indices[0..k-1] with the indices
 * of the k highest-scoring candidates, in descending order.
 * Returns the number of candidates selected (min(k, n)). */
size_t holo_ranker_select_top_k(HolographicRanker *r,
                                const double *state,
                                const double *action_vecs,
                                size_t n,
                                size_t k,
                                size_t *out_indices);

/* Empirical fidelity: number of times select() returned the correct
 * target. Reset via holo_ranker_reset_accuracy(). Count-based, not
 * theoretical 1/sqrt(N). */
size_t holo_ranker_correct_count(const HolographicRanker *r);
void holo_ranker_record_result(HolographicRanker *r, int correct);
void holo_ranker_reset_accuracy(HolographicRanker *r);
double holo_ranker_empirical_fidelity(const HolographicRanker *r);

