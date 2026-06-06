#include "nn_tfidf.h"
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Hash table for term → document frequency ---- */

#define HASH_CAP 65536
#define MAX_TERM_LEN 64

typedef struct {
    char term[MAX_TERM_LEN];
    uint32_t df;        /* document frequency */
    double idf;         /* pre-computed IDF */
    uint32_t occupied;
} HashSlot;

struct TfidfVocab {
    HashSlot *slots;
    size_t capacity;
    size_t n_terms;
    size_t dim;
    double default_idf;  /* for out-of-vocabulary terms */
};

/* djb2 hash */
static uint32_t hash_str(const char *s) {
    uint32_t h = 5381;
    while (*s) h = ((h << 5) + h) + (uint32_t)(unsigned char)*s++;
    return h;
}

/* Normalize word: lowercase, strip non-alpha, min 3 chars */
static int normalize_word(const char *raw, size_t raw_len, char *out, size_t out_cap) {
    size_t j = 0;
    for (size_t i = 0; i < raw_len && j < out_cap - 1; i++) {
        unsigned char c = (unsigned char)raw[i];
        if (isalpha(c)) out[j++] = (char)tolower(c);
    }
    out[j] = '\0';
    return j >= 3;
}

/* Common English stop words */
static int is_stop_word(const char *w) {
    static const char *stops[] = {
        "the","and","for","that","with","this","from","have","are","was",
        "not","but","they","has","had","were","can","its","all","been",
        "one","will","would","which","their","also","when","more","some",
        "than","other","into","such","about","these","only","then","them",
        "over","each","any","may","could","should","between","after","most",
        "those","both","being","under","while","during","without","through",
        "where","because","however","therefore","thus","although","though",
        "using","used","well","within","often","based","et","al","figure",
        "table","section","paper","proposed","approach","method","results",
        "model","data","set","use","two","three","many","new","first",
        "different","case","since","example","shown","given","following",
        "fig","tab","ie","eg","etc","via","per","non","due","yet"
    };
    size_t n = sizeof(stops) / sizeof(stops[0]);
    for (size_t i = 0; i < n; i++)
        if (!strcmp(w, stops[i])) return 1;
    return 0;
}

static HashSlot *find_or_insert(TfidfVocab *v, const char *term) {
    uint32_t h = hash_str(term);
    for (size_t probe = 0; probe < v->capacity; probe++) {
        size_t idx = (h + probe) % v->capacity;
        if (!v->slots[idx].occupied) {
            if (v->n_terms >= v->capacity / 2) return NULL; /* table full */
            size_t tl = strlen(term);
            if (tl >= MAX_TERM_LEN) tl = MAX_TERM_LEN - 1;
            memcpy(v->slots[idx].term, term, tl);
            v->slots[idx].term[tl] = '\0';
            v->slots[idx].df = 0;
            v->slots[idx].occupied = 1;
            v->n_terms++;
            return &v->slots[idx];
        }
        if (!strcmp(v->slots[idx].term, term))
            return &v->slots[idx];
    }
    return NULL;
}

static const HashSlot *find_term(const TfidfVocab *v, const char *term) {
    uint32_t h = hash_str(term);
    for (size_t probe = 0; probe < v->capacity; probe++) {
        size_t idx = (h + probe) % v->capacity;
        if (!v->slots[idx].occupied) return NULL;
        if (!strcmp(v->slots[idx].term, term)) return &v->slots[idx];
    }
    return NULL;
}

/* ---- Tokenize text, count term frequencies for a single document ---- */

typedef struct {
    char term[MAX_TERM_LEN];
    int count;
} TermCount;

#define MAX_DOC_TERMS 8192

static size_t tokenize_and_count(const char *text, TermCount *counts, size_t cap) {
    size_t n = 0;
    const char *p = text;
    while (*p) {
        while (*p && !isalpha((unsigned char)*p)) p++;
        if (!*p) break;
        const char *start = p;
        while (*p && (isalpha((unsigned char)*p) || *p == '-')) p++;
        size_t len = (size_t)(p - start);
        char word[MAX_TERM_LEN];
        if (normalize_word(start, len, word, sizeof(word))) {
            if (is_stop_word(word)) continue;
            /* Find existing or add */
            int found = 0;
            for (size_t i = 0; i < n; i++) {
                if (!strcmp(counts[i].term, word)) {
                    counts[i].count++;
                    found = 1;
                    break;
                }
            }
            if (!found && n < cap) {
                size_t tl = strlen(word);
                if (tl >= MAX_TERM_LEN) tl = MAX_TERM_LEN - 1;
                memcpy(counts[n].term, word, tl);
                counts[n].term[tl] = '\0';
                counts[n].count = 1;
                n++;
            }
        }
    }
    return n;
}

/* ---- Public API ---- */

TfidfVocab *tfidf_vocab_create(const char *const *paths, size_t n_docs,
                               size_t dim) {
    TfidfVocab *v = calloc(1, sizeof(TfidfVocab));
    if (!v) return NULL;
    v->capacity = HASH_CAP;
    v->dim = dim;
    v->slots = calloc(v->capacity, sizeof(HashSlot));
    if (!v->slots) { free(v); return NULL; }

    /* Pass 1: count document frequencies */
    TermCount *counts = calloc(MAX_DOC_TERMS, sizeof(TermCount));
    char *buf = NULL; size_t buf_cap = 0;

    for (size_t d = 0; d < n_docs; d++) {
        FILE *fp = fopen(paths[d], "r");
        if (!fp) continue;
        fseek(fp, 0, SEEK_END);
        long sz = ftell(fp);
        if (sz <= 0 || sz > 2000000) { fclose(fp); continue; }
        fseek(fp, 0, SEEK_SET);
        if ((size_t)sz + 1 > buf_cap) {
            buf_cap = (size_t)sz + 1;
            char *nb = realloc(buf, buf_cap);
            if (!nb) { fclose(fp); break; }
            buf = nb;
        }
        size_t nr = fread(buf, 1, (size_t)sz, fp);
        fclose(fp);
        buf[nr] = '\0';

        size_t nt = tokenize_and_count(buf, counts, MAX_DOC_TERMS);
        for (size_t i = 0; i < nt; i++) {
            HashSlot *s = find_or_insert(v, counts[i].term);
            if (s) s->df++;
        }
    }
    free(counts);
    free(buf);

    /* Pass 2: compute IDF = log(N / df) */
    double logN = log((double)n_docs);
    double sum_idf = 0;
    size_t n_valid = 0;
    for (size_t i = 0; i < v->capacity; i++) {
        if (!v->slots[i].occupied) continue;
        double idf = logN - log((double)v->slots[i].df);
        if (idf < 0) idf = 0;
        v->slots[i].idf = idf;
        sum_idf += idf;
        n_valid++;
    }
    v->default_idf = n_valid > 0 ? sum_idf / (double)n_valid : 1.0;

    return v;
}

void tfidf_encode(const TfidfVocab *vocab, const char *text, double *out) {
    if (!vocab || !text || !out) return;
    memset(out, 0, vocab->dim * sizeof(double));

    TermCount counts[MAX_DOC_TERMS];
    size_t nt = tokenize_and_count(text, counts, MAX_DOC_TERMS);

    double sum_sq = 0;
    for (size_t i = 0; i < nt; i++) {
        const HashSlot *s = find_term(vocab, counts[i].term);
        double idf = s ? s->idf : vocab->default_idf;
        double tf = 1.0 + log((double)counts[i].count);  /* log-normalized TF */
        if (tf < 0) tf = 0;
        double weight = tf * idf;

        /* Feature hashing: hash term to a dimension bin */
        uint32_t h = hash_str(counts[i].term);
        size_t bin = h % vocab->dim;
        out[bin] += weight;
        sum_sq += weight * weight;
    }

    /* L2 normalize */
    if (sum_sq > 1e-12) {
        double inv = 1.0 / sqrt(sum_sq);
        for (size_t i = 0; i < vocab->dim; i++) out[i] *= inv;
    }
}

size_t tfidf_vocab_size(const TfidfVocab *vocab) {
    return vocab ? vocab->n_terms : 0;
}

void tfidf_vocab_destroy(TfidfVocab *vocab) {
    if (!vocab) return;
    free(vocab->slots);
    free(vocab);
}
