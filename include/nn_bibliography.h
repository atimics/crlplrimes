#ifndef NN_BIBLIOGRAPHY_H
#define NN_BIBLIOGRAPHY_H

#include "nn_holographic_ranker.h"
#include "nn_term_bridge.h"
#include "nn_tfidf.h"
#include "nn_holographic_layer.h"
#include <stddef.h>

#define BIBLIOGRAPHY_DIM 2048
#define BIB_MAX_SENTENCES 300
#define BIBLIOGRAPHY_MAX_PAPERS 64
#define BIBLIOGRAPHY_MAX_SENTENCES_PER_PAPER 256
#define BIBLIOGRAPHY_MAX_TOPICS 8
#define BIBLIOGRAPHY_BIBKEY_MAX 64

typedef struct {
    char bibkey[BIBLIOGRAPHY_BIBKEY_MAX];
    char **sentences;
    size_t sentence_count;
    double *abstract_vec;
    double *binding_vecs;      /* flat array of sentence-pair bindings */
    int *binding_indices;      /* maps binding i → original sentence index */
    size_t binding_count;      /* number of bindings stored */
    double topic_affinities[BIBLIOGRAPHY_MAX_TOPICS];
    int primary_topic;
} BibPaper;

typedef struct {
    BibPaper papers[BIBLIOGRAPHY_MAX_PAPERS];
    size_t paper_count;

    /* Per-paper holographic rankers — one item per ranker, fidelity 1.0 */
    HolographicRanker *paper_rankers[BIBLIOGRAPHY_MAX_PAPERS];
    double *paper_centroids[BIBLIOGRAPHY_MAX_PAPERS];  /* topic-space centroids */

    HolographicLayer *sentence_layers[BIBLIOGRAPHY_MAX_TOPICS];

    double **topic_vecs;
    char topic_names[BIBLIOGRAPHY_MAX_TOPICS][32];
    size_t topic_count;

    TextMutator *term_bridge;  /* term diversification bridge */
    TfidfVocab *tfidf_vocab;
    /* Claim gate: trained on TF-IDF rankings, used for holographic retrieval */
    HolographicRanker **claim_gates;
    double **gate_claim_vecs;
    size_t n_gates;
    double *gate_candidates;
    double *paper_global_bias;  /* per-paper mean gate score, for de-biasing */  /* flat array of all paper TF-IDF vectors */
    int gates_ready;          /* 1 when all claim gates are fully trained */
} BibliographyStore;

BibliographyStore *bibliography_create(const char *manifest_path,
                                       const char *text_dir,
                                       const char *const *topic_contexts,
                                       size_t topic_count);
void bibliography_destroy(BibliographyStore *bib);

int bibliography_query_claim(const BibliographyStore *bib,
                             const char *claim_text,
                             const char *domain_context,
                             char out_bibkeys[][BIBLIOGRAPHY_BIBKEY_MAX],
                             double *out_scores, size_t N);

int bibliography_encode_paper(BibliographyStore *bib,
                              const char *bibkey, const char *text_path);

size_t bibliography_paper_count(const BibliographyStore *bib);
size_t bibliography_sentence_count(const BibliographyStore *bib);
double bibliography_sentence_fidelity(const BibliographyStore *bib);
double bibliography_abstract_fidelity(const BibliographyStore *bib);
double bibliography_topic_fidelity(const BibliographyStore *bib);

/* Ordinal query: returns bibkeys in rank order (1st is best match).
 * out_bibkeys[0..N-1] filled with ranked bibkeys.
 * Returns number of candidates ranked (min(paper_count, N)). */
size_t bibliography_ordinal_query(const BibliographyStore *bib,
                                  const char *claim_text,
                                  const char *domain_context,
                                  char out_bibkeys[][BIBLIOGRAPHY_BIBKEY_MAX],
                                  size_t N);

#endif
