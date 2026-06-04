#include "nn.h"

#include "nn_holo_runner.h"
#include "nn_holo_gate.h"
#include "nn_holographic_ranker.h"
#include "leos_engine.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    HW_FEATURE_COUNT = 32,
    HW_HOLO_DIM = 1024,
    HW_HOLO_AGENTS = 101,
    HW_HOLO_TOP_K = 5,
    HW_HOLO_PROMOTE_AT = 200,
    HW_VOCAB_HASH_SIZE = 65536,
    HW_MAX_CANDIDATES = 256,
    HW_MAX_POEMS = 8192,
    HW_MAX_LINE_WORDS = 32,
    HW_MAX_WORD_LEN = 64,
    HW_MAX_VOCAB = 24000,
    HW_MAX_ADJ_EDGES = 300000,
    HW_ADJ_BUCKETS = 524288,
    HW_MAX_GENERATED = 768,
    HW_FEATURE_ENCODER_VERSION = 1,
    HW_ACTION_NEWLINE = 1,
    HW_ACTION_EOS = 2,
    HW_ADJ_START = -1,
    HW_ADJ_BOUNDARY = -2,
};

static const int HW_TARGET_SYLLABLES[3] = {5, 7, 5};
static const int HW_MAX_LINE_CHARS[3] = {28, 34, 28};

typedef struct {
    char word[HW_MAX_WORD_LEN];
    int syllables;
    int length;
    size_t frequency;
    size_t line_frequency[3];
} HWVocabEntry;

typedef struct {
    HWVocabEntry entries[HW_MAX_VOCAB];
    int ranked_ids[HW_MAX_VOCAB];
    int hash_table[HW_VOCAB_HASH_SIZE];
    size_t count;
    size_t max_frequency;
} HWVocab;

typedef struct {
    int words[3][HW_MAX_LINE_WORDS];
    size_t line_counts[3];
} HWPoem;

typedef struct {
    HWPoem poems[HW_MAX_POEMS];
    size_t count;
    size_t ignored_stanzas;
    size_t ignored_words;
} HWCorpus;

typedef struct {
    int line;
    int prev_word_id;
    int next_word_id;
    size_t frequency;
    int hash_next;
    int out_next;
} HWAdjEdge;

typedef struct {
    HWAdjEdge edges[HW_MAX_ADJ_EDGES];
    int buckets[HW_ADJ_BUCKETS];
    int out_buckets[HW_ADJ_BUCKETS];
    size_t count;
    size_t overflow;
    size_t transitions;
    uint64_t hash;
} HWAdjacency;

typedef struct {
    const char *corpus_path;
    const char *word_list_path;
    const char *trace_path;
    const char *save_path;
    size_t epochs;
    size_t max_updates;
    size_t samples;
    size_t top_k;
    size_t max_candidates;
    double learning_rate;
    double temperature;
    unsigned int seed;
    int sample;
    int strict_adjacency;
    int use_holo_gate;
    int use_anki;
    int online;
    const char *save_holo_path;
    const char *merge_holo_path;
    size_t chunk_start;
    size_t chunk_end;
} HWConfig;

typedef enum {
    HW_CAND_WORD = 0,
    HW_CAND_NEWLINE = 1,
    HW_CAND_EOS = 2,
} HWCandidateKind;

typedef struct {
    HWCandidateKind kind;
    int word_id;
    uint32_t action_id;
} HWCandidate;

typedef struct {
    int line;
    int line_syllables;
    int line_chars;
    int line_words;
    int poem_words;
    int prev_word_id;
} HWState;

typedef struct {
    size_t rows;
    size_t skipped;
    size_t top1;
    double nll;
} HWEval;

typedef struct {
    size_t syllable;
    size_t line_chars;
    size_t zero_frequency;
    size_t adjacency;
    size_t cap;
    size_t forced_target;
    size_t dead_end;
} HWRejectStats;

typedef struct {
    size_t next_epoch;
    size_t interval;
    double last_loss;
    size_t review_count;
} AnkiEntry;

typedef struct {
    HoloRunner *runner;
    double centroid[HW_HOLO_DIM];
    size_t stored;
} HoloAgent;

static const HWVocab *g_sort_vocab;

static uint32_t rng_next(uint32_t *state)
{
    uint32_t x = *state == 0 ? 1u : *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static double rng_unit(uint32_t *state)
{
    return (double)(rng_next(state) >> 8) / (double)0x01000000u;
}

static double now_seconds(void)
{
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static int is_vowel_char(char c)
{
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y';
}

static int syllable_proxy(const char *word)
{
    int count = 0;
    int prev_vowel = 0;
    for (size_t i = 0; word[i] != '\0'; ++i) {
        int vowel = is_vowel_char(word[i]);
        if (vowel && !prev_vowel) {
            count += 1;
        }
        prev_vowel = vowel;
    }
    return count > 0 ? count : 1;
}

static int normalize_word(const char *raw, char *out, size_t out_size)
{
    size_t n = 0;
    for (size_t i = 0; raw[i] != '\0'; ++i) {
        unsigned char c = (unsigned char)raw[i];
        if (!isalpha(c)) {
            return 0;
        }
        if (n + 1u >= out_size) {
            return 0;
        }
        out[n++] = (char)tolower(c);
    }
    out[n] = '\0';
    return n > 0;
}

static void strip_line(char *line)
{
    size_t len = strlen(line);
    while (len > 0 && (line[len - 1u] == '\n' || line[len - 1u] == '\r' ||
                       isspace((unsigned char)line[len - 1u]))) {
        line[len - 1u] = '\0';
        len -= 1;
    }
}

static uint64_t vocab_hash_word(const char *word)
{
    uint64_t h = 14695981039346656037ULL;
    while (*word) { h ^= (uint64_t)(unsigned char)*word++; h *= 1099511628211ULL; }
    return h;
}

static int vocab_find(const HWVocab *vocab, const char *word)
{
    if (vocab->count == 0) return -1;
    uint64_t h = vocab_hash_word(word);
    size_t idx = (size_t)(h & (uint64_t)(HW_VOCAB_HASH_SIZE - 1));
    for (;;) {
        int vi = vocab->hash_table[idx];
        if (vi < 0) return -1;
        if (strcmp(vocab->entries[vi].word, word) == 0) return vi;
        idx = (idx + 1) & (size_t)(HW_VOCAB_HASH_SIZE - 1);
    }
}

static int vocab_add(HWVocab *vocab, const char *word)
{
    int existing = vocab_find(vocab, word);
    if (existing >= 0) return existing;
    if (vocab->count >= HW_MAX_VOCAB) return -1;
    HWVocabEntry *entry = &vocab->entries[vocab->count];
    memset(entry, 0, sizeof(*entry));
    snprintf(entry->word, sizeof(entry->word), "%s", word);
    entry->length = (int)strlen(word);
    entry->syllables = syllable_proxy(word);
    vocab->ranked_ids[vocab->count] = (int)vocab->count;
    uint64_t h = vocab_hash_word(word);
    size_t idx = (size_t)(h & (uint64_t)(HW_VOCAB_HASH_SIZE - 1));
    while (vocab->hash_table[idx] >= 0)
        idx = (idx + 1) & (size_t)(HW_VOCAB_HASH_SIZE - 1);
    vocab->hash_table[idx] = (int)vocab->count;
    vocab->count += 1;
    return (int)(vocab->count - 1u);
}

static int load_word_list(const char *path, HWVocab *vocab)
{
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "failed to open word list %s: %s\n", path, strerror(errno));
        return 0;
    }
    char line[HW_MAX_WORD_LEN * 2];
    char word[HW_MAX_WORD_LEN];
    while (fgets(line, sizeof(line), fp) != NULL) {
        strip_line(line);
        if (!normalize_word(line, word, sizeof(word))) {
            continue;
        }
        if (vocab_add(vocab, word) < 0) {
            fclose(fp);
            fprintf(stderr, "word list exceeds HW_MAX_VOCAB=%d\n", HW_MAX_VOCAB);
            return 0;
        }
    }
    fclose(fp);
    return 1;
}

static int parse_line_words(char *line,
                            HWVocab *vocab,
                            int allow_new_words,
                            int *out_words,
                            size_t *out_count)
{
    *out_count = 0;
    char *cursor = line;
    while (*cursor != '\0') {
        while (*cursor != '\0' && isspace((unsigned char)*cursor)) {
            cursor += 1;
        }
        if (*cursor == '\0') {
            break;
        }
        char *start = cursor;
        while (*cursor != '\0' && !isspace((unsigned char)*cursor)) {
            cursor += 1;
        }
        char saved = *cursor;
        *cursor = '\0';
        char word[HW_MAX_WORD_LEN];
        if (!normalize_word(start, word, sizeof(word))) {
            *cursor = saved;
            return 0;
        }
        int id = vocab_find(vocab, word);
        if (id < 0 && allow_new_words) {
            id = vocab_add(vocab, word);
        }
        if (id < 0 || *out_count >= HW_MAX_LINE_WORDS) {
            *cursor = saved;
            return 0;
        }
        out_words[*out_count] = id;
        *out_count += 1;
        *cursor = saved;
    }
    return *out_count > 0;
}

static void corpus_add_current(HWCorpus *corpus,
                               HWVocab *vocab,
                               int line_words[3][HW_MAX_LINE_WORDS],
                               const size_t *line_counts)
{
    if (line_counts[0] == 0 || line_counts[1] == 0 || line_counts[2] == 0) {
        corpus->ignored_stanzas += 1;
        return;
    }
    if (corpus->count >= HW_MAX_POEMS) {
        corpus->ignored_stanzas += 1;
        return;
    }
    HWPoem *poem = &corpus->poems[corpus->count++];
    memset(poem, 0, sizeof(*poem));
    for (int line = 0; line < 3; ++line) {
        poem->line_counts[line] = line_counts[line];
        for (size_t i = 0; i < line_counts[line]; ++i) {
            int word_id = line_words[line][i];
            poem->words[line][i] = word_id;
            vocab->entries[word_id].frequency += 1;
            vocab->entries[word_id].line_frequency[line] += 1;
            if (vocab->entries[word_id].frequency > vocab->max_frequency) {
                vocab->max_frequency = vocab->entries[word_id].frequency;
            }
        }
    }
}

static int load_corpus(const char *path, HWVocab *vocab, HWCorpus *corpus)
{
    memset(corpus, 0, sizeof(*corpus));
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "failed to open corpus %s: %s\n", path, strerror(errno));
        return 0;
    }

    const int allow_new_words = vocab->count == 0;
    int line_words[3][HW_MAX_LINE_WORDS];
    size_t line_counts[3] = {0, 0, 0};
    int current_line = 0;
    char line[512];
    while (fgets(line, sizeof(line), fp) != NULL) {
        strip_line(line);
        if (line[0] == '\0') {
            if (current_line == 3) {
                corpus_add_current(corpus, vocab, line_words, line_counts);
            } else if (current_line != 0) {
                corpus->ignored_stanzas += 1;
            }
            memset(line_words, 0, sizeof(line_words));
            memset(line_counts, 0, sizeof(line_counts));
            current_line = 0;
            continue;
        }
        if (current_line >= 3) {
            corpus->ignored_stanzas += 1;
            memset(line_words, 0, sizeof(line_words));
            memset(line_counts, 0, sizeof(line_counts));
            current_line = 0;
            continue;
        }
        if (!parse_line_words(line,
                              vocab,
                              allow_new_words,
                              line_words[current_line],
                              &line_counts[current_line])) {
            corpus->ignored_words += 1;
            memset(line_words, 0, sizeof(line_words));
            memset(line_counts, 0, sizeof(line_counts));
            current_line = 0;
            continue;
        }
        current_line += 1;
    }
    if (current_line == 3) {
        corpus_add_current(corpus, vocab, line_words, line_counts);
    } else if (current_line != 0) {
        corpus->ignored_stanzas += 1;
    }
    fclose(fp);
    return corpus->count > 0;
}

static int load_poem_stream(FILE *fp, HWVocab *vocab, HWCorpus *corpus)
{
    int line_words[3][HW_MAX_LINE_WORDS];
    size_t line_counts[3] = {0, 0, 0};
    int current_line = 0;
    char line[512];
    int poem_started = 0;
    while (fgets(line, sizeof(line), fp) != NULL) {
        strip_line(line);
        if (line[0] == '\0') {
            if (current_line == 3) {
                corpus_add_current(corpus, vocab, line_words, line_counts);
                return 1;
            }
            if (current_line > 0) {
                corpus->ignored_stanzas += 1;
                memset(line_words, 0, sizeof(line_words));
                memset(line_counts, 0, sizeof(line_counts));
                current_line = 0;
                poem_started = 0;
                continue;
            }
            continue;
        }
        poem_started = 1;
        if (current_line >= 3) {
            corpus->ignored_stanzas += 1;
            memset(line_words, 0, sizeof(line_words));
            memset(line_counts, 0, sizeof(line_counts));
            current_line = 0;
        }
        int allow_new = 1;
        if (!parse_line_words(line, vocab, allow_new, line_words[current_line], &line_counts[current_line])) {
            corpus->ignored_words += 1;
            memset(line_words, 0, sizeof(line_words));
            memset(line_counts, 0, sizeof(line_counts));
            current_line = 0;
            continue;
        }
        current_line += 1;
        if (current_line == 3) {
            corpus_add_current(corpus, vocab, line_words, line_counts);
            return 1;
        }
    }
    if (current_line == 3) {
        corpus_add_current(corpus, vocab, line_words, line_counts);
        return 1;
    }
    return 0;
}

static int compare_ranked_words(const void *a, const void *b)
{
    const HWVocab *vocab = g_sort_vocab;
    int left = *(const int *)a;
    int right = *(const int *)b;
    const HWVocabEntry *le = &vocab->entries[left];
    const HWVocabEntry *re = &vocab->entries[right];
    if (le->frequency != re->frequency) {
        return le->frequency < re->frequency ? 1 : -1;
    }
    return strcmp(le->word, re->word);
}

static void sort_vocab(HWVocab *vocab)
{
    for (size_t i = 0; i < vocab->count; ++i) {
        vocab->ranked_ids[i] = (int)i;
    }
    g_sort_vocab = vocab;
    qsort(vocab->ranked_ids, vocab->count, sizeof(vocab->ranked_ids[0]), compare_ranked_words);
    g_sort_vocab = NULL;
}

static size_t active_vocab_count(const HWVocab *vocab)
{
    size_t count = 0;
    for (size_t i = 0; i < vocab->count; ++i) {
        if (vocab->entries[i].frequency > 0) {
            count += 1;
        }
    }
    return count;
}

static uint64_t hash_adjacency_key(int line, int prev_word_id, int next_word_id)
{
    uint64_t hash = 1469598103934665603ull;
    const int values[] = {line, prev_word_id + 3, next_word_id + 3};
    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
        hash ^= (uint64_t)(uint32_t)values[i];
        hash *= 1099511628211ull;
    }
    return hash;
}





static uint64_t hash_adjacency_out_key(int line, int prev_word_id)
{
    uint64_t hash = 1469598103934665603ull;
    const int values[] = {line, prev_word_id + 3};
    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
        hash ^= (uint64_t)(uint32_t)values[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

static void adjacency_init(HWAdjacency *adjacency)
{
    adjacency->count = 0;
    adjacency->overflow = 0;
    adjacency->transitions = 0;
    adjacency->hash = 0;
    for (size_t i = 0; i < HW_ADJ_BUCKETS; ++i) {
        adjacency->buckets[i] = -1;
        adjacency->out_buckets[i] = -1;
    }
}

static HWAdjEdge *adjacency_find_edge(HWAdjacency *adjacency,
                                      int line,
                                      int prev_word_id,
                                      int next_word_id)
{
    uint64_t hash = hash_adjacency_key(line, prev_word_id, next_word_id);
    size_t bucket = (size_t)(hash & (uint64_t)(HW_ADJ_BUCKETS - 1));
    for (int edge_id = adjacency->buckets[bucket]; edge_id >= 0;
         edge_id = adjacency->edges[edge_id].hash_next) {
        HWAdjEdge *edge = &adjacency->edges[edge_id];
        if (edge->line == line && edge->prev_word_id == prev_word_id &&
            edge->next_word_id == next_word_id) {
            return edge;
        }
    }
    return NULL;
}

static const HWAdjEdge *adjacency_find_const(const HWAdjacency *adjacency,
                                             int line,
                                             int prev_word_id,
                                             int next_word_id)
{
    uint64_t hash = hash_adjacency_key(line, prev_word_id, next_word_id);
    size_t bucket = (size_t)(hash & (uint64_t)(HW_ADJ_BUCKETS - 1));
    for (int edge_id = adjacency->buckets[bucket]; edge_id >= 0;
         edge_id = adjacency->edges[edge_id].hash_next) {
        const HWAdjEdge *edge = &adjacency->edges[edge_id];
        if (edge->line == line && edge->prev_word_id == prev_word_id &&
            edge->next_word_id == next_word_id) {
            return edge;
        }
    }
    return NULL;
}

static int adjacency_add(HWAdjacency *adjacency,
                         int line,
                         int prev_word_id,
                         int next_word_id)
{
    adjacency->transitions += 1;
    HWAdjEdge *existing = adjacency_find_edge(adjacency, line, prev_word_id, next_word_id);
    if (existing != NULL) {
        existing->frequency += 1;
        return 1;
    }
    if (adjacency->count >= HW_MAX_ADJ_EDGES) {
        adjacency->overflow += 1;
        return 0;
    }
    uint64_t hash = hash_adjacency_key(line, prev_word_id, next_word_id);
    size_t bucket = (size_t)(hash & (uint64_t)(HW_ADJ_BUCKETS - 1));
    HWAdjEdge *edge = &adjacency->edges[adjacency->count];
    uint64_t out_hash = hash_adjacency_out_key(line, prev_word_id);
    size_t out_bucket = (size_t)(out_hash & (uint64_t)(HW_ADJ_BUCKETS - 1));
    edge->line = line;
    edge->prev_word_id = prev_word_id;
    edge->next_word_id = next_word_id;
    edge->frequency = 1;
    edge->hash_next = adjacency->buckets[bucket];
    edge->out_next = adjacency->out_buckets[out_bucket];
    adjacency->buckets[bucket] = (int)adjacency->count;
    adjacency->out_buckets[out_bucket] = (int)adjacency->count;
    adjacency->count += 1;
    return 1;
}

static void adjacency_finalize_hash(HWAdjacency *adjacency)
{
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < adjacency->count; ++i) {
        const HWAdjEdge *edge = &adjacency->edges[i];
        const uint64_t values[] = {
            (uint64_t)(uint32_t)edge->line,
            (uint64_t)(uint32_t)(edge->prev_word_id + 3),
            (uint64_t)(uint32_t)(edge->next_word_id + 3),
            (uint64_t)edge->frequency,
        };
        for (size_t j = 0; j < sizeof(values) / sizeof(values[0]); ++j) {
            hash ^= values[j];
            hash *= 1099511628211ull;
        }
    }
    adjacency->hash = hash;
}

static void adjacency_add_poem(HWAdjacency *adjacency, const HWPoem *poem)
{
    for (int line = 0; line < 3; ++line) {
        int prev_word_id = HW_ADJ_START;
        for (size_t i = 0; i < poem->line_counts[line]; ++i) {
            int word_id = poem->words[line][i];
            (void)adjacency_add(adjacency, line, prev_word_id, word_id);
            prev_word_id = word_id;
        }
        (void)adjacency_add(adjacency, line, prev_word_id, HW_ADJ_BOUNDARY);
    }
}

static int adjacency_allows_transition(const HWAdjacency *adjacency,
                                       const HWState *state,
                                       int next_word_id)
{
    if (adjacency == NULL || state->line < 0 || state->line > 2) {
        return 0;
    }
    int prev_word_id = state->line_words == 0 ? HW_ADJ_START : state->prev_word_id;
    return adjacency_find_const(adjacency, state->line, prev_word_id, next_word_id) != NULL;
}

static uint32_t action_id_for_candidate(const HWCandidate *candidate)
{
    if (candidate->kind == HW_CAND_NEWLINE) {
        return HW_ACTION_NEWLINE;
    }
    if (candidate->kind == HW_CAND_EOS) {
        return HW_ACTION_EOS;
    }
    return 3u + (uint32_t)candidate->word_id;
}

static uint64_t hash_state(const HWState *state)
{
    uint64_t hash = 1469598103934665603ull;
    const int values[] = {
        state->line,
        state->line_syllables,
        state->line_chars,
        state->line_words,
        state->poem_words,
        state->prev_word_id,
    };
    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
        hash ^= (uint64_t)(uint32_t)values[i];
        hash *= 1099511628211ull;
    }

    return hash;
}
static void anki_init(AnkiEntry *entries, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        entries[i].next_epoch = 0;
        entries[i].interval = 1;
        entries[i].last_loss = 1e9;
        entries[i].review_count = 0;
    }
}

static void anki_update(AnkiEntry *entry, double poem_loss, size_t current_epoch)
{
    entry->review_count += 1;
    if (poem_loss < entry->last_loss * 0.98) {
        entry->interval = entry->interval < 32 ? entry->interval * 2 : 32;
    } else if (poem_loss > entry->last_loss * 1.05) {
        entry->interval = 1;
    }
    entry->last_loss = poem_loss;
    entry->next_epoch = current_epoch + entry->interval;
}


static int boundary_allowed(const HWState *state)
{
    if (state->line < 0 || state->line > 2 || state->line_words == 0) {
        return 0;
    }
    int target = HW_TARGET_SYLLABLES[state->line];
    return state->line_syllables >= target - 1 && state->line_syllables <= target + 2;
}

static int word_allowed(const HWVocab *vocab, const HWState *state, int word_id)
{
    if (state->line < 0 || state->line > 2) {
        return 0;
    }
    const HWVocabEntry *word = &vocab->entries[word_id];
    if (word->frequency == 0) {
        return 0;
    }
    int target = HW_TARGET_SYLLABLES[state->line];
    int next_syllables = state->line_syllables + word->syllables;
    if (next_syllables > target + 2) {
        return 0;
    }
    int extra_space = state->line_words > 0 ? 1 : 0;
    int next_chars = state->line_chars + extra_space + word->length;
    if (next_chars > HW_MAX_LINE_CHARS[state->line]) {
        return 0;
    }
    return 1;
}

static int adjacency_has_legal_continuation(const HWVocab *vocab,
                                            const HWAdjacency *adjacency,
                                            const HWState *state)
{
    if (adjacency == NULL || state->line < 0 || state->line > 2) {
        return 0;
    }
    int prev_word_id = state->line_words == 0 ? HW_ADJ_START : state->prev_word_id;
    uint64_t hash = hash_adjacency_out_key(state->line, prev_word_id);
    size_t bucket = (size_t)(hash & (uint64_t)(HW_ADJ_BUCKETS - 1));
    for (int edge_id = adjacency->out_buckets[bucket]; edge_id >= 0;
         edge_id = adjacency->edges[edge_id].out_next) {
        const HWAdjEdge *edge = &adjacency->edges[edge_id];
        if (edge->line != state->line || edge->prev_word_id != prev_word_id) {
            continue;
        }
        if (edge->next_word_id == HW_ADJ_BOUNDARY && boundary_allowed(state)) {
            return 1;
        }
        if (edge->next_word_id >= 0 && word_allowed(vocab, state, edge->next_word_id)) {
            return 1;
        }
    }
    return 0;
}

static int candidate_matches(const HWCandidate *candidate, HWCandidateKind kind, int word_id)
{
    return candidate->kind == kind && (kind != HW_CAND_WORD || candidate->word_id == word_id);
}

static void apply_word(HWState *state, const HWVocab *vocab, int word_id);
static void apply_boundary(HWState *state, HWCandidateKind kind);

static size_t build_candidates(const HWVocab *vocab,
                               const HWAdjacency *adjacency,
                               const HWState *state,
                               HWCandidateKind target_kind,
                               int target_word_id,
                               size_t max_candidates,
                               int strict_adjacency,
                               HWRejectStats *rejects,
                               HWCandidate *candidates)
{
    if (max_candidates == 0 || max_candidates > HW_MAX_CANDIDATES) {
        max_candidates = HW_MAX_CANDIDATES;
    }
    size_t count = 0;
    int boundary_is_allowed = boundary_allowed(state);
    int boundary_is_adjacent =
        !strict_adjacency || adjacency_allows_transition(adjacency, state, HW_ADJ_BOUNDARY);
    int boundary_has_continuation = 1;
    if (strict_adjacency && state->line < 2 && boundary_is_allowed && boundary_is_adjacent) {
        HWState next_state = *state;
        apply_boundary(&next_state, HW_CAND_NEWLINE);
        boundary_has_continuation = adjacency_has_legal_continuation(vocab, adjacency, &next_state);
    }
    if (state->line < 2 && boundary_is_allowed && boundary_is_adjacent && boundary_has_continuation &&
        count < max_candidates) {
        candidates[count].kind = HW_CAND_NEWLINE;
        candidates[count].word_id = -1;
        candidates[count].action_id = action_id_for_candidate(&candidates[count]);
        count += 1;
    }
    if (state->line == 2 && boundary_is_allowed && boundary_is_adjacent && count < max_candidates) {
        candidates[count].kind = HW_CAND_EOS;
        candidates[count].word_id = -1;
        candidates[count].action_id = action_id_for_candidate(&candidates[count]);
        count += 1;
    }
    if (boundary_is_allowed && (!boundary_is_adjacent || !boundary_has_continuation) &&
        rejects != NULL) {
        rejects->adjacency += 1;
    }
    for (size_t r = 0; r < vocab->count && count < max_candidates; ++r) {
        int word_id = vocab->ranked_ids[r];
        const HWVocabEntry *word = &vocab->entries[word_id];
        if (word->frequency == 0) {
            if (rejects != NULL) {
                rejects->zero_frequency += 1;
            }
            continue;
        }
        if (state->line >= 0 && state->line <= 2) {
            if (state->line_syllables + word->syllables > HW_TARGET_SYLLABLES[state->line] + 2) {
                if (rejects != NULL) {
                    rejects->syllable += 1;
                }
                continue;
            }
            int extra_space = state->line_words > 0 ? 1 : 0;
            if (state->line_chars + extra_space + word->length > HW_MAX_LINE_CHARS[state->line]) {
                if (rejects != NULL) {
                    rejects->line_chars += 1;
                }
                continue;
            }
        }
        if (strict_adjacency && !adjacency_allows_transition(adjacency, state, word_id)) {
            if (rejects != NULL) {
                rejects->adjacency += 1;
            }
            continue;
        }
        if (strict_adjacency) {
            HWState next_state = *state;
            apply_word(&next_state, vocab, word_id);
            if (!adjacency_has_legal_continuation(vocab, adjacency, &next_state)) {
                if (rejects != NULL) {
                    rejects->adjacency += 1;
                }
                continue;
            }
        }
        candidates[count].kind = HW_CAND_WORD;
        candidates[count].word_id = word_id;
        candidates[count].action_id = action_id_for_candidate(&candidates[count]);
        count += 1;
    }
    if (target_kind == HW_CAND_WORD && target_word_id >= 0) {
        int found = 0;
        for (size_t i = 0; i < count; ++i) {
            if (candidate_matches(&candidates[i], target_kind, target_word_id)) {
                found = 1;
                break;
            }
        }
        if (!found && word_allowed(vocab, state, target_word_id)) {
            int target_adjacent =
                !strict_adjacency || adjacency_allows_transition(adjacency, state, target_word_id);
            if (target_adjacent && strict_adjacency) {
                HWState next_state = *state;
                apply_word(&next_state, vocab, target_word_id);
                target_adjacent = adjacency_has_legal_continuation(vocab, adjacency, &next_state);
            }
            if (target_adjacent) {
                size_t index = count < max_candidates ? count++ : max_candidates - 1u;
                candidates[index].kind = HW_CAND_WORD;
                candidates[index].word_id = target_word_id;
                candidates[index].action_id = action_id_for_candidate(&candidates[index]);
                if (rejects != NULL) {
                    rejects->forced_target += 1;
                }
            } else if (rejects != NULL) {
                rejects->adjacency += 1;
            }
        }
    }
    if (count == 0 && rejects != NULL) {
        rejects->dead_end += 1;
    }
    if (rejects != NULL && count == max_candidates) {
        rejects->cap += 1;
    }
    return count;
}

static int target_index(const HWCandidate *candidates,
                        size_t candidate_count,
                        HWCandidateKind kind,
                        int word_id,
                        size_t *out)
{
    for (size_t i = 0; i < candidate_count; ++i) {
        if (candidate_matches(&candidates[i], kind, word_id)) {
            *out = i;
            return 1;
        }
    }
    return 0;
}

static void apply_word(HWState *state, const HWVocab *vocab, int word_id)
{
    const HWVocabEntry *word = &vocab->entries[word_id];
    if (state->line_words > 0) {
        state->line_chars += 1;
    }
    state->line_chars += word->length;
    state->line_syllables += word->syllables;
    state->line_words += 1;
    state->poem_words += 1;
    state->prev_word_id = word_id;
}

static void apply_boundary(HWState *state, HWCandidateKind kind)
{
    if (kind == HW_CAND_NEWLINE) {
        state->line += 1;
        state->line_syllables = 0;
        state->line_chars = 0;
        state->line_words = 0;
        state->prev_word_id = -1;
    } else if (kind == HW_CAND_EOS) {
        state->line = 3;
    }
}

static void fill_features(const HWVocab *vocab,
                          const HWState *state,
                          const HWCandidate *candidate,
                          nn_real *features)
{
    memset(features, 0, HW_FEATURE_COUNT * sizeof(*features));
    int line = state->line >= 0 && state->line <= 2 ? state->line : 2;
    int target = HW_TARGET_SYLLABLES[line];
    features[0] = (nn_real)1;
    features[1] = (nn_real)line / (nn_real)2;
    features[2] = (nn_real)state->line_syllables / (nn_real)(target + 2);
    features[3] = (nn_real)(target - state->line_syllables) / (nn_real)7;
    features[4] = (nn_real)state->line_chars / (nn_real)HW_MAX_LINE_CHARS[line];
    features[5] = (nn_real)state->line_words / (nn_real)12;
    features[6] = (nn_real)state->poem_words / (nn_real)36;
    features[7] = state->line_words == 0 ? (nn_real)1 : (nn_real)0;
    features[8] = boundary_allowed(state) ? (nn_real)1 : (nn_real)0;
    features[9] = candidate->kind == HW_CAND_WORD ? (nn_real)1 : (nn_real)0;
    features[10] = candidate->kind == HW_CAND_NEWLINE ? (nn_real)1 : (nn_real)0;
    features[11] = candidate->kind == HW_CAND_EOS ? (nn_real)1 : (nn_real)0;
    if (candidate->kind == HW_CAND_WORD && candidate->word_id >= 0) {
        const HWVocabEntry *word = &vocab->entries[candidate->word_id];
        int extra_space = state->line_words > 0 ? 1 : 0;
        int next_chars = state->line_chars + extra_space + word->length;
        int next_syllables = state->line_syllables + word->syllables;
        double max_freq = vocab->max_frequency > 0 ? log((double)vocab->max_frequency + 1.0) : 1.0;
        features[12] = (nn_real)word->syllables / (nn_real)4;
        features[13] = (nn_real)word->length / (nn_real)16;
        features[14] = (nn_real)(log((double)word->frequency + 1.0) / max_freq);
        features[15] = next_syllables == target ? (nn_real)1 : (nn_real)0;
        features[16] = next_syllables >= target - 1 && next_syllables <= target + 2 ? (nn_real)1 : (nn_real)0;
        features[17] = (nn_real)(target + 2 - next_syllables) / (nn_real)9;
        features[18] = (nn_real)(HW_MAX_LINE_CHARS[line] - next_chars) / (nn_real)HW_MAX_LINE_CHARS[line];
        features[19] = state->prev_word_id == candidate->word_id ? (nn_real)1 : (nn_real)0;
        features[20] = (nn_real)(word->word[0] - 'a') / (nn_real)25;
        features[21] = (nn_real)(word->word[word->length - 1] - 'a') / (nn_real)25;
        features[22] = is_vowel_char(word->word[0]) ? (nn_real)1 : (nn_real)0;
        features[23] = is_vowel_char(word->word[word->length - 1]) ? (nn_real)1 : (nn_real)0;
        features[24] = (nn_real)word->line_frequency[line] / (nn_real)(word->frequency > 0 ? word->frequency : 1);
        features[25] = (nn_real)candidate->word_id / (nn_real)(vocab->count > 1 ? vocab->count - 1u : 1u);
        if (state->prev_word_id >= 0) {
            const HWVocabEntry *prev = &vocab->entries[state->prev_word_id];
            features[26] = prev->word[prev->length - 1] == word->word[0] ? (nn_real)1 : (nn_real)0;
            features[27] = (nn_real)prev->syllables / (nn_real)4;
            features[28] = (nn_real)(log((double)prev->frequency + 1.0) / max_freq);
        }
        features[29] = word->length <= 4 ? (nn_real)1 : (nn_real)0;
        features[30] = word->length >= 8 ? (nn_real)1 : (nn_real)0;
        features[31] = word->syllables == 1 ? (nn_real)1 : (nn_real)0;
    } else {
        features[12] = (nn_real)0;
        features[15] = boundary_allowed(state) ? (nn_real)1 : (nn_real)0;
        features[17] = (nn_real)(target - state->line_syllables) / (nn_real)7;
        features[18] = (nn_real)(HW_MAX_LINE_CHARS[line] - state->line_chars) / (nn_real)HW_MAX_LINE_CHARS[line];
    }
}

static void build_candidate_inputs(const HWVocab *vocab,
                                   const HWState *state,
                                   const HWCandidate *candidates,
                                   size_t candidate_count,
                                   nn_real *inputs)
{
    for (size_t i = 0; i < candidate_count; ++i) {
        fill_features(vocab, state, &candidates[i], &inputs[i * HW_FEATURE_COUNT]);
    }
}



static size_t train_split_count(const HWCorpus *corpus)
{
    if (corpus->count <= 1) return corpus->count;
    size_t train = (corpus->count * 4u) / 5u;
    return train == 0 ? 1u : train;
}

static void holo_agent_init_centroid(HoloAgent *a, int agent_idx)
{
    /* 100 agents: random centroids spread across the hypersphere.
     * Hebbian updates will pull them toward the data distribution.
     * Top-K sparse coding ensures they diversify naturally. */
    LeosEngine *eng = a->runner ? leos_engine_create(HW_HOLO_DIM) : NULL;
    if (!eng) return;
    uint64_t seed = 14695981039346656037ULL ^ ((uint64_t)agent_idx * 2654435761ULL + 42);
    leos_engine_keygen(eng, seed, a->centroid);
    leos_engine_normalize(eng, a->centroid);
    leos_engine_destroy(eng);
}

static double holo_cosine(const double *a, const double *b)
{
    double d = 0.0;
    for (size_t i = 0; i < HW_HOLO_DIM; ++i) d += a[i] * b[i];
    return d;
}

static void holo_agent_route(HoloAgent *agents, const double *state_vec,
                              int top_k, int *out_indices, double *out_sims)
{
    double sims[HW_HOLO_AGENTS];
    int indices[HW_HOLO_AGENTS];
    for (int i = 0; i < HW_HOLO_AGENTS; ++i) {
        sims[i] = holo_cosine(state_vec, agents[i].centroid);
        indices[i] = i;
    }
    for (int i = 0; i < HW_HOLO_AGENTS; ++i)
        for (int j = i + 1; j < HW_HOLO_AGENTS; ++j)
            if (sims[j] > sims[i]) {
                double ts = sims[i]; sims[i] = sims[j]; sims[j] = ts;
                int ti = indices[i]; indices[i] = indices[j]; indices[j] = ti;
            }
    for (int k = 0; k < top_k && k < HW_HOLO_AGENTS; ++k) {
        out_indices[k] = indices[k];
        out_sims[k] = sims[k];
    }
}

static void holo_agent_train(HoloAgent *agents, const double *state_vec, const double *action_vec)
{
    int top_idx[HW_HOLO_TOP_K];
    double top_sim[HW_HOLO_TOP_K];
    holo_agent_route(agents, state_vec, HW_HOLO_TOP_K, top_idx, top_sim);
    double sim_total = 0.0;
    for (int k = 0; k < HW_HOLO_TOP_K; ++k) if (top_sim[k] > 0.0) sim_total += top_sim[k];
    if (sim_total <= 0.0) sim_total = 1.0;
    for (int k = 0; k < HW_HOLO_TOP_K; ++k) {
        int ai = top_idx[k];
        double w = top_sim[k] / sim_total;
        holo_runner_train(agents[ai].runner, state_vec, action_vec);
        agents[ai].stored += 1;
        /* Hebbian centroid update: decaying lr, move toward activated state */
        double lr = 0.001 / (1.0 + (double)agents[ai].stored / 1000.0);
        for (size_t d = 0; d < HW_HOLO_DIM; ++d)
            agents[ai].centroid[d] = (1.0 - lr) * agents[ai].centroid[d] + lr * w * state_vec[d];
        /* Renormalize */
        double norm = 0.0;
        for (size_t d = 0; d < HW_HOLO_DIM; ++d) norm += agents[ai].centroid[d] * agents[ai].centroid[d];
        if (norm > 0.0) { norm = sqrt(norm); for (size_t d = 0; d < HW_HOLO_DIM; ++d) agents[ai].centroid[d] /= norm; }
    }
}

static void holo_agent_promote_if_needed(HoloAgent *agents, int child_idx, int parent_idx)
{
    if (agents[child_idx].stored < HW_HOLO_PROMOTE_AT) return;
    if (parent_idx < 0 || parent_idx >= HW_HOLO_AGENTS) return;
    /* Promote: reset child, but first reinforce parent with child's centroid */
    holo_runner_train(agents[parent_idx].runner, agents[child_idx].centroid, agents[child_idx].centroid);
    agents[parent_idx].stored += 1;
    /* Reset child: new runner, new centroid */
    holo_runner_destroy(agents[child_idx].runner);
    agents[child_idx].runner = holo_runner_create(HW_HOLO_DIM);
    holo_agent_init_centroid(&agents[child_idx], child_idx);
    agents[child_idx].stored = 0;
    printf("  promoted agent %d → %d\n", child_idx, parent_idx);
}

static nn_real train_step(NeuralNet *net,
                          const HWVocab *vocab,
                          const HWAdjacency *adjacency,
                          const HWState *state,
                          HWCandidateKind target_kind,
                          int target_word_id,
                          size_t max_candidates,
                          int strict_adjacency,
                          double learning_rate,
                          int *trained,
                          HoloAgent *agents,
                          int use_holo_gate)
{
    HWCandidate candidates[HW_MAX_CANDIDATES];
    nn_real inputs[HW_MAX_CANDIDATES * HW_FEATURE_COUNT];
    size_t candidate_count = build_candidates(vocab,
                                              adjacency, state,
                                              target_kind, target_word_id,
                                              max_candidates, strict_adjacency,
                                              NULL, candidates);
    size_t target = 0;
    *trained = 0;
    if (candidate_count == 0 ||
        !target_index(candidates, candidate_count, target_kind, target_word_id, &target))
        return (nn_real)0;
    build_candidate_inputs(vocab, state, candidates, candidate_count, inputs);
    *trained = 1;
    if (use_holo_gate) {
        double sv[HW_HOLO_DIM], av[HW_HOLO_DIM];
        uint64_t sseed = hash_state(state);
        uint64_t aseed = (uint64_t)action_id_for_candidate(&candidates[target]);
        LeosEngine *te = leos_engine_create(HW_HOLO_DIM);
        leos_engine_keygen(te, sseed, sv); leos_engine_normalize(te, sv);
        leos_engine_keygen(te, aseed, av); leos_engine_normalize(te, av);
        holo_agent_train(agents, sv, av);
        /* Promote overflow */
        for (int ai = 0; ai < 100; ++ai) {
            int parent = ai < 90 ? 90 + (ai / 10) : 100;
            if (parent != ai) holo_agent_promote_if_needed(agents, ai, parent);
        }
        leos_engine_destroy(te);
    }
    return nn_train_softmax_scorer(net, inputs, candidate_count, target, (nn_real)learning_rate);
}



static int score_step(NeuralNet *net,
                      const HWVocab *vocab,
                      const HWAdjacency *adjacency,
                      const HWState *state,
                      HWCandidateKind target_kind,
                      int target_word_id,
                      size_t max_candidates,
                      int strict_adjacency,
                      HWEval *eval,
                      HoloAgent *agents, int use_holo_gate)
{
    HWCandidate candidates[HW_MAX_CANDIDATES];
    nn_real inputs[HW_MAX_CANDIDATES * HW_FEATURE_COUNT];
    nn_real scores[HW_MAX_CANDIDATES];
    nn_real probabilities[HW_MAX_CANDIDATES];
    size_t ranked[HW_MAX_CANDIDATES];
    uint32_t action_ids[HW_MAX_CANDIDATES];
    size_t candidate_count = build_candidates(vocab,
                                              adjacency,
                                              state,
                                              target_kind,
                                              target_word_id,
                                              max_candidates,
                                              strict_adjacency,
                                              NULL,
                                              candidates);
    size_t target = 0;
    if (candidate_count == 0 ||
        !target_index(candidates, candidate_count, target_kind, target_word_id, &target)) {
        eval->skipped += 1;
        return 0;
    }
    for (size_t i = 0; i < candidate_count; ++i) {
        action_ids[i] = candidates[i].action_id;
    }
    build_candidate_inputs(vocab, state, candidates, candidate_count, inputs);
    if (!nn_score_action_rows(net, inputs, candidate_count, scores, probabilities)) {
        eval->skipped += 1;
        return 0;
    }
    if (use_holo_gate) {
            double *av = calloc(candidate_count * HW_HOLO_DIM, sizeof(double));
            double *hscores = calloc(candidate_count, sizeof(double));
            double *ensembled = calloc(candidate_count, sizeof(double));
            double total_weight = 0.0;
            for (int ai = 0; ai < 100; ++ai) {
                double fi = holo_ranker_fidelity(agents[ai].runner->ranker);
                if (fi < 0.01) continue;
                double w = fi * fi * fi;
                double sv[HW_HOLO_DIM];
                holo_runner_embed_state(agents[ai].runner, hash_state(state), sv);
                memset(av, 0, candidate_count * HW_HOLO_DIM * sizeof(double));
                for (size_t hi = 0; hi < candidate_count; ++hi)
                    holo_runner_embed_state(agents[ai].runner, (uint64_t)action_ids[hi], &av[hi * HW_HOLO_DIM]);
                holo_ranker_score(agents[ai].runner->ranker, sv, av, candidate_count, hscores);
                double mx = hscores[0]; int valid = isfinite(mx);
                for (size_t hi = 1; hi < candidate_count; ++hi) { if (!isfinite(hscores[hi])) valid = 0; if (hscores[hi] > mx) mx = hscores[hi]; }
                if (valid) {
                    double sm = 0.0;
                    for (size_t hi = 0; hi < candidate_count; ++hi) { hscores[hi] = exp(hscores[hi] - mx); sm += hscores[hi]; }
                    if (sm > 0.0) {
                        for (size_t hi = 0; hi < candidate_count; ++hi) hscores[hi] /= sm;
                        for (size_t hi = 0; hi < candidate_count; ++hi) ensembled[hi] += w * hscores[hi];
                        total_weight += w;
                    }
                }
            }
            if (total_weight > 0.0) {
                double blend_alpha = total_weight / (total_weight + 1.0);
                for (size_t hi = 0; hi < candidate_count; ++hi)
                    scores[hi] = (nn_real)(blend_alpha * ensembled[hi] / total_weight + (1.0 - blend_alpha) * (double)scores[hi]);
            }
            free(av); free(hscores); free(ensembled);
        }
    if (!nn_rank_action_scores_with_ids(scores, action_ids, candidate_count, ranked, NN_RANK_SCORE_EPSILON)) {
        eval->skipped += 1;
        return 0;
    }
    double probability = (double)probabilities[target];
    if (probability < 1e-12) {
        probability = 1e-12;
    }
    eval->nll += -log(probability);
    eval->top1 += ranked[0] == target ? 1u : 0u;
    eval->rows += 1;
    return 1;
}

static HWEval evaluate_split(NeuralNet *net,
                             const HWVocab *vocab,
                             const HWAdjacency *adjacency,
                             const HWCorpus *corpus,
                             size_t start,
                             size_t end,
                             size_t max_candidates,
                             int strict_adjacency,
                             HoloAgent *agents, int use_holo_gate)
{
    HWEval eval;
    memset(&eval, 0, sizeof(eval));
    for (size_t p = start; p < end; ++p) {
        const HWPoem *poem = &corpus->poems[p];
        HWState state;
        memset(&state, 0, sizeof(state));
        state.prev_word_id = -1;
        for (int line = 0; line < 3; ++line) {
            for (size_t i = 0; i < poem->line_counts[line]; ++i) {
                int word_id = poem->words[line][i];
                (void)score_step(net,
                                 vocab,
                                 adjacency,
                                 &state,
                                 HW_CAND_WORD,
                                 word_id,
                                 max_candidates,
                                 strict_adjacency,
                                 &eval,
                                 agents, use_holo_gate);
                apply_word(&state, vocab, word_id);
            }
            HWCandidateKind boundary = line < 2 ? HW_CAND_NEWLINE : HW_CAND_EOS;
            (void)score_step(net,
                             vocab,
                             adjacency,
                             &state,
                             boundary,
                             -1,
                             max_candidates,
                             strict_adjacency,
                             &eval,
                             agents, use_holo_gate);
            apply_boundary(&state, boundary);
        }
    }
    return eval;
}

static size_t choose_candidate(const nn_real *probabilities,
                               const size_t *ranked,
                               size_t candidate_count,
                               const HWConfig *config,
                               uint32_t *rng_state)
{
    if (!config->sample) {
        return ranked[0];
    }
    size_t top_k = config->top_k;
    if (top_k == 0 || top_k > candidate_count) {
        top_k = candidate_count;
    }
    double weights[HW_MAX_CANDIDATES];
    double total = 0.0;
    double temperature = config->temperature <= 0.0 ? 1.0 : config->temperature;
    for (size_t i = 0; i < top_k; ++i) {
        size_t index = ranked[i];
        double p = (double)probabilities[index];
        if (p < 1e-12) {
            p = 1e-12;
        }
        weights[i] = exp(log(p) / temperature);
        total += weights[i];
    }
    double pick = rng_unit(rng_state) * total;
    for (size_t i = 0; i < top_k; ++i) {
        if (pick <= weights[i]) {
            return ranked[i];
        }
        pick -= weights[i];
    }
    return ranked[top_k - 1u];
}

static int json_word(FILE *fp, const char *word)
{
    if (fputc('"', fp) == EOF) {
        return 0;
    }
    for (size_t i = 0; word[i] != '\0'; ++i) {
        if (word[i] == '"' || word[i] == '\\') {
            if (fputc('\\', fp) == EOF) {
                return 0;
            }
        }
        if (fputc(word[i], fp) == EOF) {
            return 0;
        }
    }
    return fputc('"', fp) != EOF;
}

static const char *candidate_label(const HWVocab *vocab, const HWCandidate *candidate)
{
    if (candidate->kind == HW_CAND_NEWLINE) {
        return "<newline>";
    }
    if (candidate->kind == HW_CAND_EOS) {
        return "<eos>";
    }
    return vocab->entries[candidate->word_id].word;
}

static int write_trace(FILE *fp,
                       const HWVocab *vocab,
                       const HWAdjacency *adjacency,
                       const char *model_hash,
                       const HWState *state,
                       const HWCandidate *candidates,
                       const HWRejectStats *rejects,
                       const uint32_t *action_ids,
                       const nn_real *scores,
                       const nn_real *probabilities,
                       size_t candidate_count,
                       size_t selected,
                       size_t top_ranked,
                       size_t sample_index,
                       size_t step,
                       uint32_t seed,
                       const HWConfig *config)
{
    if (fprintf(fp,
                "{\"schema\":\"crlp.haiku_word_lm.generation_trace.v1\","
                "\"task\":\"haiku_word_lm\",\"state_hash\":\"%016llx\","
                "\"model_hash\":\"%s\",\"feature_encoder_version\":%u,"
                "\"sample\":%zu,\"step\":%zu,\"line\":%d,\"line_syllables\":%d,"
                "\"line_words\":%d,\"candidate_count\":%zu,\"selected_index\":%zu,"
                "\"top_ranked_index\":%zu,\"selected_action_id\":%u,"
                "\"top_ranked_action_id\":%u,\"sampling\":\"%s\",\"top_k\":%zu,"
                "\"temperature\":%.6g,\"seed\":%u,\"strict_adjacency\":%s,"
                "\"adjacency_hash\":\"%016llx\","
                "\"rejected_candidates\":{\"syllable\":%zu,\"line_chars\":%zu,"
                "\"zero_frequency\":%zu,\"adjacency\":%zu,\"cap\":%zu,\"forced_target\":%zu,"
                "\"dead_end\":%zu},\"action_ids\":[",
                (unsigned long long)hash_state(state),
                model_hash,
                (unsigned int)HW_FEATURE_ENCODER_VERSION,
                sample_index,
                step,
                state->line,
                state->line_syllables,
                state->line_words,
                candidate_count,
                selected,
                top_ranked,
                action_ids[selected],
                action_ids[top_ranked],
                config->sample ? "top_k_sample" : "top1",
                config->top_k,
                config->temperature,
                seed,
                config->strict_adjacency ? "true" : "false",
                (unsigned long long)(adjacency != NULL ? adjacency->hash : 0ull),
                rejects != NULL ? rejects->syllable : 0,
                rejects != NULL ? rejects->line_chars : 0,
                rejects != NULL ? rejects->zero_frequency : 0,
                rejects != NULL ? rejects->adjacency : 0,
                rejects != NULL ? rejects->cap : 0,
                rejects != NULL ? rejects->forced_target : 0,
                rejects != NULL ? rejects->dead_end : 0) < 0) {
        return 0;
    }
    for (size_t i = 0; i < candidate_count; ++i) {
        if ((i != 0 && fputc(',', fp) == EOF) || fprintf(fp, "%u", action_ids[i]) < 0) {
            return 0;
        }
    }
    if (fputs("],\"candidate_tokens\":[", fp) == EOF) {
        return 0;
    }
    for (size_t i = 0; i < candidate_count; ++i) {
        if (i != 0 && fputc(',', fp) == EOF) {
            return 0;
        }
        if (!json_word(fp, candidate_label(vocab, &candidates[i]))) {
            return 0;
        }
    }
    if (fputs("],\"scores\":[", fp) == EOF) {
        return 0;
    }
    for (size_t i = 0; i < candidate_count; ++i) {
        if ((i != 0 && fputc(',', fp) == EOF) || fprintf(fp, "%.9g", (double)scores[i]) < 0) {
            return 0;
        }
    }
    if (fputs("],\"probabilities\":[", fp) == EOF) {
        return 0;
    }
    for (size_t i = 0; i < candidate_count; ++i) {
        if ((i != 0 && fputc(',', fp) == EOF) ||
            fprintf(fp, "%.9g", (double)probabilities[i]) < 0) {
            return 0;
        }
    }
    return fputs(config->strict_adjacency
                     ? "],\"verifier_cost\":{\"gates\":[\"dataset_vocab\",\"dataset_adjacency\",\"haiku_shape_proxy\"]}}\n"
                     : "],\"verifier_cost\":{\"gates\":[\"dataset_vocab\",\"haiku_shape_proxy\"]}}\n",
                 fp) != EOF;
}

static void append_word(char *generated, size_t *length, const char *word, int line_words)
{
    if (line_words > 0 && *length + 1u < HW_MAX_GENERATED) {
        generated[(*length)++] = ' ';
    }
    for (size_t i = 0; word[i] != '\0' && *length + 1u < HW_MAX_GENERATED; ++i) {
        generated[(*length)++] = word[i];
    }
    generated[*length] = '\0';
}

static int generate_one(NeuralNet *net,
                        const HWVocab *vocab,
                        const HWAdjacency *adjacency,
                        const HWConfig *config,
                        const char *model_hash,
                        FILE *trace,
                        size_t sample_index,
                        HoloAgent *agents, int use_holo_gate)
{
    char generated[HW_MAX_GENERATED];
    size_t length = 0;
    generated[0] = '\0';
    HWState state;
    memset(&state, 0, sizeof(state));
    state.prev_word_id = -1;
    uint32_t rng_state = config->seed + (uint32_t)(sample_index * 7919u) + 31u;

    for (size_t step = 0; step < 96 && state.line < 3; ++step) {
        HWCandidate candidates[HW_MAX_CANDIDATES];
        nn_real inputs[HW_MAX_CANDIDATES * HW_FEATURE_COUNT];
        nn_real scores[HW_MAX_CANDIDATES];
        nn_real probabilities[HW_MAX_CANDIDATES];
        size_t ranked[HW_MAX_CANDIDATES];
        uint32_t action_ids[HW_MAX_CANDIDATES];
        HWRejectStats rejects;
        memset(&rejects, 0, sizeof(rejects));
        size_t candidate_count = build_candidates(vocab,
                                                  adjacency,
                                                  &state,
                                                  HW_CAND_WORD,
                                                  -1,
                                                  config->max_candidates,
                                                  config->strict_adjacency,
                                                  &rejects,
                                                  candidates);
        if (candidate_count == 0) {
            break;
        }
        for (size_t i = 0; i < candidate_count; ++i) {
            action_ids[i] = candidates[i].action_id;
        }
        build_candidate_inputs(vocab, &state, candidates, candidate_count, inputs);
        if (!nn_score_action_rows(net, inputs, candidate_count, scores, probabilities)) {
            return 0;
        }
        if (use_holo_gate) {
            double *av = calloc(candidate_count * HW_HOLO_DIM, sizeof(double));
            double *hscores = calloc(candidate_count, sizeof(double));
            double *ensembled = calloc(candidate_count, sizeof(double));
            double total_weight = 0.0;
            for (int ai = 0; ai < 100; ++ai) {
                double fi = holo_ranker_fidelity(agents[ai].runner->ranker);
                if (fi < 0.01) continue;
                double w = fi * fi * fi;
                double sv[HW_HOLO_DIM];
                holo_runner_embed_state(agents[ai].runner, hash_state(&state), sv);
                memset(av, 0, candidate_count * HW_HOLO_DIM * sizeof(double));
                for (size_t hi = 0; hi < candidate_count; ++hi)
                    holo_runner_embed_state(agents[ai].runner, (uint64_t)action_ids[hi], &av[hi * HW_HOLO_DIM]);
                holo_ranker_score(agents[ai].runner->ranker, sv, av, candidate_count, hscores);
                double mx = hscores[0]; int valid = isfinite(mx);
                for (size_t hi = 1; hi < candidate_count; ++hi) { if (!isfinite(hscores[hi])) valid = 0; if (hscores[hi] > mx) mx = hscores[hi]; }
                if (valid) {
                    double sm = 0.0;
                    for (size_t hi = 0; hi < candidate_count; ++hi) { hscores[hi] = exp(hscores[hi] - mx); sm += hscores[hi]; }
                    if (sm > 0.0) {
                        for (size_t hi = 0; hi < candidate_count; ++hi) hscores[hi] /= sm;
                        for (size_t hi = 0; hi < candidate_count; ++hi) ensembled[hi] += w * hscores[hi];
                        total_weight += w;
                    }
                }
            }
            if (total_weight > 0.0) {
                double blend_alpha = total_weight / (total_weight + 1.0);
                for (size_t hi = 0; hi < candidate_count; ++hi)
                    scores[hi] = (nn_real)(blend_alpha * ensembled[hi] / total_weight + (1.0 - blend_alpha) * (double)scores[hi]);
            }
            free(av); free(hscores); free(ensembled);
        }
        if (!nn_rank_action_scores_with_ids(scores,
                                            action_ids,
                                            candidate_count,
                                            ranked,
                                            NN_RANK_SCORE_EPSILON)) {
            return 0;
        }
        size_t selected = choose_candidate(probabilities,
                                           ranked,
                                           candidate_count,
                                           config,
                                           &rng_state);
        if (trace != NULL &&
            !write_trace(trace,
                         vocab,
                         adjacency,
                         model_hash,
                         &state,
                         candidates,
                         &rejects,
                         action_ids,
                         scores,
                         probabilities,
                         candidate_count,
                         selected,
                         ranked[0],
                         sample_index,
                         step,
                         rng_state,
                         config)) {
            return 0;
        }
        HWCandidate picked = candidates[selected];
        if (picked.kind == HW_CAND_WORD) {
            append_word(generated, &length, vocab->entries[picked.word_id].word, state.line_words);
            apply_word(&state, vocab, picked.word_id);
        } else if (picked.kind == HW_CAND_NEWLINE) {
            if (length + 1u < sizeof(generated)) {
                generated[length++] = '\n';
                generated[length] = '\0';
            }
            apply_boundary(&state, picked.kind);
        } else {
            apply_boundary(&state, picked.kind);
            break;
        }
    }
    puts("generated:");
    puts("---");
    puts(generated);
    puts("---");
    return 1;
}

static void usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s [--corpus path] [--word-list path] [--epochs N]\n"
            "       [--max-updates N] [--lr R] [--samples N] [--sample 0|1]\n"
            "       [--top-k N] [--temperature R] [--max-candidates N]\n"
            "       [--strict-adjacency 0|1] [--holo-gate 0|1]\n"
            "       [--anki 0|1] [--online 0|1]\n"
            "       [--save-holo path] [--merge-holo path]\n"
            "       [--chunk-start N] [--chunk-end N]\n"
            "       [--holo-dim N] [--trace path] [--save path] [--seed N]\n",
            argv0);
}

static int parse_config(int argc, char **argv, HWConfig *config)
{
    config->corpus_path = "data/haiku_tiny.txt";
    config->word_list_path = NULL;
    config->trace_path = NULL;
    config->save_path = NULL;
    config->epochs = 4;
    config->max_updates = 0;
    config->samples = 2;
    config->top_k = 8;
    config->max_candidates = HW_MAX_CANDIDATES;
    config->learning_rate = 0.006;
    config->temperature = 0.85;
    config->seed = 20240518u;
    config->sample = 1;
    config->strict_adjacency = 1;
    config->use_holo_gate = 0;
    config->use_anki = 0;
    config->online = 0;
    config->save_holo_path = NULL;
    config->merge_holo_path = NULL;
    config->chunk_start = 0;
    config->chunk_end = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--corpus") == 0 && i + 1 < argc) {
            config->corpus_path = argv[++i];
        } else if (strcmp(argv[i], "--word-list") == 0 && i + 1 < argc) {
            config->word_list_path = argv[++i];
        } else if (strcmp(argv[i], "--trace") == 0 && i + 1 < argc) {
            config->trace_path = argv[++i];
        } else if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            config->save_path = argv[++i];
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            if (!nn_parse_size_nonnegative_text(argv[++i], &config->epochs)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--max-updates") == 0 && i + 1 < argc) {
            if (!nn_parse_size_nonnegative_text(argv[++i], &config->max_updates)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
            if (!nn_parse_size_nonnegative_text(argv[++i], &config->samples)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            if (!nn_parse_size_nonnegative_text(argv[++i], &config->top_k)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--max-candidates") == 0 && i + 1 < argc) {
            if (!nn_parse_size_nonnegative_text(argv[++i], &config->max_candidates)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            if (!nn_parse_double_text(argv[++i], &config->learning_rate)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            if (!nn_parse_double_text(argv[++i], &config->temperature)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--sample") == 0 && i + 1 < argc) {
            size_t value = 0;
            if (!nn_parse_size_nonnegative_text(argv[++i], &value)) {
                return 0;
            }
            config->sample = value != 0;
        } else if (strcmp(argv[i], "--strict-adjacency") == 0 && i + 1 < argc) {
            size_t value = 0;
            if (!nn_parse_size_nonnegative_text(argv[++i], &value)) {
                return 0;
            }
            config->strict_adjacency = value != 0;
        } else if (strcmp(argv[i], "--anki") == 0 && i + 1 < argc) {
            size_t value = 0;
            if (!nn_parse_size_nonnegative_text(argv[++i], &value)) {
                return 0;
            }
            config->use_anki = value != 0;
        } else if (strcmp(argv[i], "--holo-gate") == 0 && i + 1 < argc) {
            size_t value = 0;
            if (!nn_parse_size_nonnegative_text(argv[++i], &value)) return 0;
            config->use_holo_gate = value != 0;
        } else if (strcmp(argv[i], "--online") == 0 && i + 1 < argc) {
            size_t value = 0;
            if (!nn_parse_size_nonnegative_text(argv[++i], &value)) return 0;
            config->online = value != 0;
        } else if (strcmp(argv[i], "--save-holo") == 0 && i + 1 < argc) {
            config->save_holo_path = argv[++i];
        } else if (strcmp(argv[i], "--merge-holo") == 0 && i + 1 < argc) {
            config->merge_holo_path = argv[++i];
        } else if (strcmp(argv[i], "--chunk-start") == 0 && i + 1 < argc) {
            if (!nn_parse_size_nonnegative_text(argv[++i], &config->chunk_start)) return 0;
        } else if (strcmp(argv[i], "--chunk-end") == 0 && i + 1 < argc) {
            if (!nn_parse_size_nonnegative_text(argv[++i], &config->chunk_end)) return 0;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            size_t value = 0;
            if (!nn_parse_size_nonnegative_text(argv[++i], &value)) {
                return 0;
            }
            config->seed = (unsigned int)value;
        } else {
            return 0;
        }
    }
    return config->epochs > 0 && config->samples > 0 && config->learning_rate > 0.0 &&
           config->max_candidates > 0 && config->max_candidates <= HW_MAX_CANDIDATES;
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    HWConfig config;
    if (!parse_config(argc, argv, &config)) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    static HWVocab vocab;
    memset(&vocab, 0, sizeof(vocab));
    for (size_t hi = 0; hi < HW_VOCAB_HASH_SIZE; ++hi) vocab.hash_table[hi] = -1;
    if (config.word_list_path != NULL && !load_word_list(config.word_list_path, &vocab)) {
        return EXIT_FAILURE;
    }

    static HWCorpus corpus;
    memset(&corpus, 0, sizeof(corpus));
    FILE *corpus_fp = NULL;
    if (config.online) {
        corpus_fp = fopen(config.corpus_path, "r");
        if (corpus_fp == NULL) {
            fprintf(stderr, "failed to open corpus %s: %s\n", config.corpus_path, strerror(errno));
            return EXIT_FAILURE;
        }
        while (load_poem_stream(corpus_fp, &vocab, &corpus)) {
            if (corpus.count >= HW_MAX_POEMS) break;
        }
        fclose(corpus_fp);
    } else {
        if (!load_corpus(config.corpus_path, &vocab, &corpus)) {
            return EXIT_FAILURE;
        }
    }
    sort_vocab(&vocab);
    size_t train_count = train_split_count(&corpus);
    static HWAdjacency adjacency;
    adjacency_init(&adjacency);
    if (!config.online) {
        for (size_t p = 0; p < train_count; ++p) {
            if (config.strict_adjacency)
                adjacency_add_poem(&adjacency, &corpus.poems[p]);
        }
        adjacency_finalize_hash(&adjacency);
    }

    const size_t layers[] = {HW_FEATURE_COUNT, 256, 128, 64, 1};
    NNConfig nn_config;
    nn_config.layer_sizes = layers;
    nn_config.layer_count = sizeof(layers) / sizeof(layers[0]);
    nn_config.seed = config.seed;
    nn_config.hidden_activation = NN_ACTIVATION_RELU;
    nn_config.output_activation = NN_ACTIVATION_LINEAR;
    nn_config.loss = NN_LOSS_MSE;
    NeuralNet *net = nn_create_config(&nn_config);
    if (net == NULL) {
        fputs("failed to create haiku_word_lm network\n", stderr);
        return EXIT_FAILURE;
    }

    HoloAgent agents[HW_HOLO_AGENTS];
    memset(agents, 0, sizeof(agents));
    if (config.use_holo_gate) {
        for (int ai = 0; ai < HW_HOLO_AGENTS; ++ai) {
            agents[ai].runner = holo_runner_create(HW_HOLO_DIM);
            if (agents[ai].runner == NULL) {
                fputs("failed to create HoloRunner\n", stderr);
                return EXIT_FAILURE;
            }
            holo_agent_init_centroid(&agents[ai], ai);
        }
        printf("holo-agent swarm initialized: %d agents dim=%d top-k=%d\n", HW_HOLO_AGENTS, HW_HOLO_DIM, HW_HOLO_TOP_K);
        if (config.merge_holo_path != NULL) {
            /* merge only affects runner 0 for now */
            if (holo_runner_load(agents[0].runner, config.merge_holo_path) != 0) {
                fprintf(stderr, "failed to load holographic trace: %s\n", config.merge_holo_path);
            }
        }
    }

    printf("haiku_word_lm corpus=%s poems=%zu train=%zu valid=%zu ignored_stanzas=%zu "
           "ignored_words=%zu vocab=%zu active_vocab=%zu features=%d parameters=%zu "
           "max_candidates=%zu strict_adjacency=%d adjacency_edges=%zu adjacency_hash=%016llx\n",
           config.corpus_path,
           corpus.count,
           train_count,
           corpus.count - train_count,
           corpus.ignored_stanzas,
           corpus.ignored_words,
           vocab.count,
           active_vocab_count(&vocab),
           HW_FEATURE_COUNT,
           nn_parameter_count(net),
           config.max_candidates,
           config.strict_adjacency,
           adjacency.count,
           (unsigned long long)adjacency.hash);

    AnkiEntry *anki = NULL;
    if (config.use_anki) {
        anki = calloc(train_count, sizeof(AnkiEntry));
        if (anki == NULL) {
            fputs("failed to allocate anki schedule\n", stderr);
            return EXIT_FAILURE;
        }
        anki_init(anki, train_count);
    }

    size_t updates = 0;
    size_t skipped = 0;
    double training_seconds = 0.0;
    for (size_t epoch = 1; epoch <= config.epochs; ++epoch) {
        double started = now_seconds();
        double loss = 0.0;
        size_t epoch_updates = 0;
        size_t epoch_skipped = 0;
        size_t poems_reviewed = 0;
        size_t poems_deferred = 0;
        size_t loop_start = config.chunk_end > 0 && config.chunk_start < train_count ? config.chunk_start : 0;
        size_t loop_end = config.chunk_end > 0 && config.chunk_end <= train_count ? config.chunk_end : train_count;
        for (size_t p = loop_start; p < loop_end; ++p) {
            if (config.use_anki && anki[p].next_epoch > epoch) {
                poems_deferred += 1;
                continue;
            }
            const HWPoem *poem = &corpus.poems[p];
            if (config.strict_adjacency && anki != NULL && anki[p].review_count == 0) {
                adjacency_add_poem(&adjacency, poem);
            } else if (config.strict_adjacency && anki == NULL) {
                adjacency_add_poem(&adjacency, poem);
            }
            HWState state;
            memset(&state, 0, sizeof(state));
            state.prev_word_id = -1;
            double poem_loss = 0.0;
            size_t poem_trained = 0;
            for (int line = 0; line < 3; ++line) {
                for (size_t i = 0; i < poem->line_counts[line]; ++i) {
                    int trained = 0;
                    int word_id = poem->words[line][i];
                    nn_real step_loss = train_step(net,
                                                   &vocab,
                                                   &adjacency,
                                                   &state,
                                                   HW_CAND_WORD,
                                                   word_id,
                                                   config.max_candidates,
                                                   config.strict_adjacency,
                                                   config.learning_rate,
                                                   &trained,
                                                   agents,
                                                   config.use_holo_gate);
                    if (trained) {
                        loss += (double)step_loss;
                        poem_loss += (double)step_loss;
                        poem_trained += 1;
                        epoch_updates += 1;
                        updates += 1;
                    } else {
                        epoch_skipped += 1;
                        skipped += 1;
                    }
                    apply_word(&state, &vocab, word_id);
                    if (config.max_updates != 0 && updates >= config.max_updates) {
                        break;
                    }
                }
                if (config.max_updates != 0 && updates >= config.max_updates) {
                    break;
                }
                HWCandidateKind boundary = line < 2 ? HW_CAND_NEWLINE : HW_CAND_EOS;
                int trained = 0;
                nn_real step_loss = train_step(net,
                                               &vocab,
                                               &adjacency,
                                               &state,
                                               boundary,
                                               -1,
                                               config.max_candidates,
                                               config.strict_adjacency,
                                               config.learning_rate,
                                               &trained,
                                               agents,
                                               config.use_holo_gate);
                if (trained) {
                    loss += (double)step_loss;
                    poem_loss += (double)step_loss;
                    poem_trained += 1;
                    epoch_updates += 1;
                    updates += 1;
                } else {
                    epoch_skipped += 1;
                    skipped += 1;
                }
                apply_boundary(&state, boundary);
                if (config.max_updates != 0 && updates >= config.max_updates) {
                    break;
                }
            }
            if (config.use_anki && poem_trained > 0) {
                double avg_poem_loss = poem_loss / (double)poem_trained;
                anki_update(&anki[p], avg_poem_loss, epoch);
                poems_reviewed += 1;
            }
            if (config.max_updates != 0 && updates >= config.max_updates) {
                break;
            }
        }
        double elapsed = now_seconds() - started;
        training_seconds += elapsed;
        printf("epoch %zu loss=%.4f updates=%zu skipped=%zu",
               epoch,
               epoch_updates > 0 ? loss / (double)epoch_updates : 0.0,
               epoch_updates,
               epoch_skipped);
        if (config.use_anki) {
            printf(" reviewed=%zu deferred=%zu", poems_reviewed, poems_deferred);
        }
        printf(" %.0f rows/s\n",
               elapsed > 0.0 ? (double)epoch_updates / elapsed : 0.0);
        if (config.max_updates != 0 && updates >= config.max_updates) {
            break;
        }
    }

    adjacency_finalize_hash(&adjacency);
    HWEval train_eval = evaluate_split(net,
                                       &vocab,
                                       &adjacency,
                                       &corpus,
                                       0,
                                       train_count,
                                       config.max_candidates,
                                       config.strict_adjacency,
                                       agents, config.use_holo_gate);
    HWEval valid_eval =
        evaluate_split(net,
                       &vocab,
                       &adjacency,
                       &corpus,
                       train_count,
                       corpus.count,
                       config.max_candidates,
                       config.strict_adjacency,
                       agents, config.use_holo_gate);
    double train_acc = train_eval.rows > 0 ? (double)train_eval.top1 / (double)train_eval.rows : 0.0;
    double valid_acc = valid_eval.rows > 0 ? (double)valid_eval.top1 / (double)valid_eval.rows : 0.0;
    double train_nll = train_eval.rows > 0 ? train_eval.nll / (double)train_eval.rows : 0.0;
    double valid_nll = valid_eval.rows > 0 ? valid_eval.nll / (double)valid_eval.rows : 0.0;
    double train_coverage = train_eval.rows + train_eval.skipped > 0
                                ? (double)train_eval.rows /
                                      (double)(train_eval.rows + train_eval.skipped)
                                : 0.0;
    double valid_coverage = valid_eval.rows + valid_eval.skipped > 0
                                ? (double)valid_eval.rows /
                                      (double)(valid_eval.rows + valid_eval.skipped)
                                : 0.0;
    printf("train_eval rows=%zu top1=%.2f%% nll=%.4f skipped=%zu\n",
           train_eval.rows,
           train_acc * 100.0,
           train_nll,
           train_eval.skipped);
    printf("valid_eval rows=%zu top1=%.2f%% nll=%.4f skipped=%zu\n",
           valid_eval.rows,
           valid_acc * 100.0,
           valid_nll,
           valid_eval.skipped);
    printf("training_seconds=%.6f total_updates=%zu total_skipped=%zu\n",
           training_seconds,
           updates,
           skipped);

    char model_hash[65] = "unsaved";
    if (config.save_path != NULL) {
        NNCheckpointMeta meta;
        nn_checkpoint_meta_init(&meta, HW_FEATURE_ENCODER_VERSION);
        if (!nn_checkpoint_meta_set_notes(&meta, "haiku_word_lm word scorer") ||
            !nn_checkpoint_meta_set_feature_set(&meta, "haiku_word_v1")) {
            fprintf(stderr, "checkpoint metadata is too long\n");
            nn_free(net);
            return EXIT_FAILURE;
        }
        if (!nn_save_checkpoint(net, config.save_path, &meta) ||
            !nn_hash_checkpoint_hex(config.save_path, model_hash)) {
            fprintf(stderr, "failed to save checkpoint: %s\n", config.save_path);
            nn_free(net);
            return EXIT_FAILURE;
        }
        printf("saved checkpoint: %s hash=%s\n", config.save_path, model_hash);
    }

    FILE *trace = NULL;
    if (config.trace_path != NULL) {
        trace = fopen(config.trace_path, "w");
        if (trace == NULL) {
            fprintf(stderr, "failed to open trace %s: %s\n", config.trace_path, strerror(errno));
            nn_free(net);
            return EXIT_FAILURE;
        }
    }
    for (size_t i = 0; i < config.samples; ++i) {
        if (!generate_one(net, &vocab, &adjacency, &config, model_hash, trace, i, agents, config.use_holo_gate)) {
            if (trace != NULL) {
                fclose(trace);
            }
            nn_free(net);
            return EXIT_FAILURE;
        }
    }
    if (trace != NULL) {
        fclose(trace);
        printf("wrote trace: %s\n", config.trace_path);
    }

    printf("{\"schema\":\"crlp.haiku_word_lm.summary.v1\",\"poems\":%zu,"
           "\"train_rows\":%zu,\"valid_rows\":%zu,\"train_top1\":%.6f,"
           "\"valid_top1\":%.6f,\"train_coverage\":%.6f,"
           "\"valid_coverage\":%.6f,\"valid_nll\":%.6f,\"updates\":%zu,"
           "\"verifier_skipped\":%zu,\"vocab_entries\":%zu,\"max_candidates\":%zu,"
           "\"strict_adjacency\":%s,\"adjacency_edges\":%zu,"
           "\"adjacency_transitions\":%zu,\"adjacency_hash\":\"%016llx\","
           "\"model_hash\":\"%s\",\"holo_gate\":%s,\"holo_dim\":%zu,\"holo_stored\":%zu}\n",
           corpus.count,
           train_eval.rows,
           valid_eval.rows,
           train_acc,
           valid_acc,
           train_coverage,
           valid_coverage,
           valid_nll,
           updates,
           skipped + train_eval.skipped + valid_eval.skipped,
           vocab.count,
           config.max_candidates,
           config.strict_adjacency ? "true" : "false",
           adjacency.count,
           adjacency.transitions,
           (unsigned long long)adjacency.hash,
           model_hash,
           config.use_holo_gate ? "true" : "false",
           (size_t)HW_HOLO_DIM,
           (agents[0].runner ? holo_ranker_count(agents[0].runner->ranker) : 0));

    free(anki);
    if (config.use_holo_gate) {
        size_t total_stored = 0;
        for (int ai = 0; ai < HW_HOLO_AGENTS; ++ai) {
            total_stored += holo_ranker_count(agents[ai].runner->ranker);
        }
        printf("  agents 0-4:  ");
        for (int ai = 0; ai < 5; ++ai)
            printf("%zu ", holo_ranker_count(agents[ai].runner->ranker));
        printf("\n  agents 95-99: ");
        for (int ai = 95; ai < 100; ++ai)
            printf("%zu ", holo_ranker_count(agents[ai].runner->ranker));
        printf("\n  root(100):   %zu stored, fidelity=%.4f\n",
               holo_ranker_count(agents[100].runner->ranker),
               holo_ranker_fidelity(agents[100].runner->ranker));
        printf("  total stored: %zu\n", total_stored);
        if (config.save_holo_path != NULL) {
            if (holo_runner_save(agents[0].runner, config.save_holo_path) != 0) {
                fprintf(stderr, "failed to save holographic trace: %s\n", config.save_holo_path);
            } else {
                printf("saved holographic trace: %s\n", config.save_holo_path);
            }
        }
    }
    nn_free(net);
    return EXIT_SUCCESS;
}
