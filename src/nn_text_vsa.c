#include "nn_text_vsa.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

uint64_t nn_text_vsa_hash_token(const char *token) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (const unsigned char *p = (const unsigned char *)token; *p != '\0'; ++p) {
        hash ^= (uint64_t)*p;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int is_token_char(int c) {
    if (c == '\0') return 0;
    if (c <= ' ') return 0;
    if (c == '.' || c == ',' || c == ';' || c == ':' ||
        c == '!' || c == '?' || c == '"' || c == '\'' ||
        c == '(' || c == ')' || c == '[' || c == ']' ||
        c == '{' || c == '}' || c == '-' || c == '_') return 0;
    return 1;
}

static const char *skip_delimiters(const char *p) {
    while (*p && !is_token_char(*p)) ++p;
    return p;
}

static size_t token_length(const char *p) {
    size_t n = 0;
    while (p[n] && is_token_char(p[n])) ++n;
    return n;
}

static void emit_token(LeosEngine *eng, const char *token, size_t tok_len,
                       const char *prev_token, size_t prev_len,
                       double *scratch, double *out) {
    uint64_t seed = nn_text_vsa_hash_token(token);
    leos_engine_keygen(eng, seed, scratch);
    leos_engine_add(eng, out, scratch, out);

    if (prev_token && prev_len > 0) {
        char bigram[256];
        size_t bn = 0;
        size_t cp = prev_len < 120 ? prev_len : 120;
        memcpy(bigram, prev_token, cp); bn += cp;
        bigram[bn++] = ' ';
        size_t ct = tok_len < (254 - bn) ? tok_len : (254 - bn);
        memcpy(bigram + bn, token, ct); bn += ct;
        bigram[bn] = '\0';
        uint64_t bseed = nn_text_vsa_hash_token(bigram);
        leos_engine_keygen(eng, bseed, scratch);
        leos_engine_add(eng, out, scratch, out);
    }
}

size_t nn_text_vsa_encode(LeosEngine *eng, const char *text, double *out) {
    size_t dim = leos_engine_dim(eng);
    leos_engine_zero(eng, out);
    if (!text || !*text) return 0;

    double *scratch = calloc(dim, sizeof(double));
    if (!scratch) return 0;

    size_t count = 0;
    const char *p = text;
    char prev_buf[128] = ""; size_t prev_len = 0;

    while (*p) {
        p = skip_delimiters(p);
        if (!*p) break;
        size_t n = token_length(p);
        if (n == 0) break;

        char token_buf[128];
        size_t copy_n = n < (sizeof(token_buf)-1) ? n : (sizeof(token_buf)-1);
        for (size_t i = 0; i < copy_n; ++i)
            token_buf[i] = (char)tolower((unsigned char)p[i]);
        token_buf[copy_n] = '\0';

        emit_token(eng, token_buf, copy_n,
                   prev_len > 0 ? prev_buf : NULL, prev_len,
                   scratch, out);
        count++;
        memcpy(prev_buf, token_buf, copy_n+1);
        prev_len = copy_n;
        p += n;
    }

    if (count > 0) leos_engine_normalize(eng, out);
    free(scratch);
    return count;
}

size_t nn_text_vsa_encode_tokens(LeosEngine *eng,
                                 const int *token_ids, size_t count,
                                 const uint64_t *token_seeds, double *out) {
    size_t dim = leos_engine_dim(eng);
    leos_engine_zero(eng, out);
    if (!token_ids || count == 0) return 0;

    double *scratch = calloc(dim, sizeof(double));
    if (!scratch) return 0;

    size_t encoded = 0;
    uint64_t prev_seed = 0; int has_prev = 0;
    for (size_t i = 0; i < count; ++i) {
        (void)token_ids;
        leos_engine_keygen(eng, token_seeds[i], scratch);
        leos_engine_add(eng, out, scratch, out);
        ++encoded;
        if (has_prev) {
            uint64_t pair[2] = {prev_seed, token_seeds[i]};
            uint64_t bseed = nn_text_vsa_hash_token((const char *)pair);
            leos_engine_keygen(eng, bseed, scratch);
            leos_engine_add(eng, out, scratch, out);
            ++encoded;
        }
        prev_seed = token_seeds[i]; has_prev = 1;
    }

    leos_engine_normalize(eng, out);
    free(scratch);
    return encoded;
}

void nn_text_vsa_store(LeosEngine *eng, const char *text,
                       const double *context, double *scratch,
                       void *ranker,
                       void (*store_fn)(void*,const double*,const double*)) {
    size_t dim = leos_engine_dim(eng);
    double *text_vec = scratch;
    double *bind_out = scratch + dim;
    nn_text_vsa_encode(eng, text, text_vec);
    leos_engine_bind(eng, context, text_vec, bind_out);
    store_fn(ranker, context, bind_out);
}
