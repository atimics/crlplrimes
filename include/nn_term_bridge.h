#ifndef NN_TERM_BRIDGE_H
#define NN_TERM_BRIDGE_H

#include <stddef.h>

/*
 * TextMutator — gated text diversification via certified synonym substitution.
 *
 * The same loop applied to text: enumerate valid term alternatives from a
 * certified thesaurus (the gate), rank them by how well they preserve topic
 * affinity, store only the top-K diversified variants alongside the original.
 *
 * Each paper gets K+1 abstract vectors: the original, plus up to K
 * term-shifted variants where domain terms are replaced with certified
 * alternatives. The topic centroid sees all surfaces and drifts to the
 * semantic intersection.
 */

typedef struct TextMutator TextMutator;

/* Create a mutator from a TSV thesaurus file.
 * Format: from<TAB>to  (one mapping per line, bidirectional implied) */
TextMutator *text_mutator_create(const char *thesaurus_path);
void text_mutator_destroy(TextMutator *m);

/* Apply all gated substitutions to produce a diversified text.
 * Returns allocated string with term substitutions applied.
 * Caller must free. */
char *text_mutator_diversify(const TextMutator *m, const char *text);

/* Number of substitution pairs in the thesaurus */
size_t text_mutator_entry_count(const TextMutator *m);

#endif
