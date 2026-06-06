#include "nn_term_bridge.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ENTRIES 1024
#define MAX_TERM_LEN 64

typedef struct { char from[MAX_TERM_LEN]; char to[MAX_TERM_LEN]; } TermEntry;

struct TextMutator {
    TermEntry entries[MAX_ENTRIES];
    size_t count;
};

TextMutator *text_mutator_create(const char *path) {
    TextMutator *m = calloc(1, sizeof(*m));
    if (!m) return NULL;
    if (!path) return m;

    FILE *fp = fopen(path, "r");
    if (!fp) return m;

    char line[256];
    while (fgets(line, sizeof(line), fp) && m->count < MAX_ENTRIES) {
        char *tab = strchr(line, '\t');
        if (!tab) continue;
        size_t flen = (size_t)(tab - line);
        if (flen >= MAX_TERM_LEN) flen = MAX_TERM_LEN - 1;
        memcpy(m->entries[m->count].from, line, flen);
        m->entries[m->count].from[flen] = '\0';

        char *rest = tab + 1;
        size_t rlen = strlen(rest);
        while (rlen > 0 && (rest[rlen-1] == '\n' || rest[rlen-1] == '\r')) rlen--;
        if (rlen >= MAX_TERM_LEN) rlen = MAX_TERM_LEN - 1;
        memcpy(m->entries[m->count].to, rest, rlen);
        m->entries[m->count].to[rlen] = '\0';

        m->count++;
    }
    fclose(fp);
    return m;
}

void text_mutator_destroy(TextMutator *m) { free(m); }
size_t text_mutator_entry_count(const TextMutator *m) { return m ? m->count : 0; }

static char *lowercase(const char *s, size_t len) {
    char *out = calloc(len + 1, 1);
    if (!out) return NULL;
    for (size_t i = 0; i < len; i++)
        out[i] = (char)tolower((unsigned char)s[i]);
    return out;
}

char *text_mutator_diversify(const TextMutator *m, const char *text) {
    if (!m || !text) return NULL;
    size_t slen = strlen(text);
    char *out = calloc(slen * 2 + 1, 1);
    if (!out) return NULL;
    size_t pos = 0;

    char *lower = lowercase(text, slen);
    if (!lower) { free(out); return NULL; }

    size_t i = 0;
    while (i < slen) {
        int matched = 0;
        for (size_t e = 0; e < m->count; e++) {
            size_t mlen = strlen(m->entries[e].from);
            if (i + mlen > slen) continue;
            if (strncmp(lower + i, m->entries[e].from, mlen) != 0) continue;

            /* Word boundary check */
            if (i > 0 && isalnum((unsigned char)lower[i-1])) continue;
            if (i + mlen < slen && isalnum((unsigned char)lower[i+mlen])) continue;

            size_t tlen = strlen(m->entries[e].to);
            memcpy(out + pos, m->entries[e].to, tlen);
            pos += tlen;
            i += mlen;
            matched = 1;
            break;
        }
        if (!matched) {
            out[pos++] = text[i];
            i++;
        }
    }
    out[pos] = '\0';
    free(lower);
    return out;
}
