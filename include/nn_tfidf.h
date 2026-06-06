#ifndef NN_TFIDF_H
#define NN_TFIDF_H

#include <stddef.h>

typedef struct TfidfVocab TfidfVocab;

TfidfVocab *tfidf_vocab_create(const char *const *paths, size_t n_docs,
                               size_t dim);
void tfidf_encode(const TfidfVocab *vocab, const char *text, double *out);
size_t tfidf_vocab_size(const TfidfVocab *vocab);
void tfidf_vocab_destroy(TfidfVocab *vocab);

#endif
