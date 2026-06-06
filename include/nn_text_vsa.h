#ifndef NN_TEXT_VSA_H
#define NN_TEXT_VSA_H

#include "holographic/leos_engine.h"
#include <stddef.h>
#include <stdint.h>

/*
 * Text-to-VSA encoding — bag-of-tokens holographic embedding.
 *
 * Each whitespace-delimited token is FNV-1a-hashed to a uint64 seed,
 * which is passed to leos_engine_keygen to produce a deterministic unit
 * vector on the hypersphere. The text embedding is the normalized sum
 * of its token vectors. Similar texts produce similar vectors via
 * shared-token superposition; identical texts produce identical vectors.
 *
 * This is the bridge between symbolic text (claims, bibliography entries,
 * haiku lines, section headings) and the holographic bind/unbind/score
 * substrate used by HolographicRanker and MemoryNeuronGrid.
 */

/* Encode a null-terminated text string into a dim-dimensional VSA vector.
 * Tokenization: lowercase, split on whitespace and punctuation.
 * out must point to dim doubles. The output is unit-norm.
 *
 * Returns the number of tokens encoded, or 0 on empty input. */
size_t nn_text_vsa_encode(LeosEngine *eng, const char *text, double *out);

/* Encode from pre-hashed token IDs and their precomputed keygen seeds.
 * Use when encoding many texts over a fixed vocabulary — the caller
 * hashes each vocabulary entry once and reuses the seeds.
 *
 * out must point to dim doubles. The output is unit-norm.
 *
 * Returns the number of tokens encoded, or 0 on empty input. */
size_t nn_text_vsa_encode_tokens(LeosEngine *eng,
                                 const int *token_ids,
                                 size_t count,
                                 const uint64_t *token_seeds,
                                 double *out);

/* Hash a single token to a uint64 seed via FNV-1a.
 * Useful for building token-seed tables once and reusing them. */
uint64_t nn_text_vsa_hash_token(const char *token);

/* Convenience: encode text as bag-of-tokens, then bind it with a context
 * vector and store into a holographic ranker. This is the text-level
 * analog of holo_ranker_store(context, text_embedding). */
void nn_text_vsa_store(LeosEngine *eng,
                       const char *text,
                       const double *context,
                       double *scratch,
                       void *ranker,  /* HolographicRanker* */
                       void (*store_fn)(void *, const double *, const double *));

#endif /* NN_TEXT_VSA_H */

