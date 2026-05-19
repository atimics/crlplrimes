#include "nn.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    HU_FEATURE_COUNT = 40,
    HU_MAX_POEMS = 10000,
    HU_MAX_LINE_WORDS = 32,
    HU_MAX_WORD_LEN = 64,
    HU_MAX_VOCAB = 30000,
    HU_ADJ_BUCKETS = 524288,
    HU_MAX_ADJ_EDGES = 400000,
    HU_FEATURE_ENCODER_VERSION = 1,
    HU_CORRUPT_LINE_SHUFFLE = 0,
    HU_CORRUPT_LINE_SPLICE = 1,
    HU_CORRUPT_SYLLABLE_BREAK = 2,
    HU_CORRUPT_REPETITION = 3,
    HU_CORRUPT_WORD_SALAD = 4,
    HU_CORRUPT_COUNT = 5,
    HU_ADJ_START = -1,
    HU_ADJ_BOUNDARY = -2,
};

static const int HU_TARGET_SYLLABLES[3] = {5, 7, 5};

typedef struct {
    char word[HU_MAX_WORD_LEN];
    int syllables;
    int length;
    size_t frequency;
    size_t line_frequency[3];
} HUVocabEntry;

typedef struct {
    HUVocabEntry entries[HU_MAX_VOCAB];
    int ranked_ids[HU_MAX_VOCAB];
    size_t count;
    size_t max_frequency;
} HUVocab;

typedef struct {
    int words[3][HU_MAX_LINE_WORDS];
    size_t line_counts[3];
} HUPoem;

typedef struct {
    HUPoem poems[HU_MAX_POEMS];
    size_t count;
    size_t ignored_stanzas;
    size_t ignored_words;
} HUCorpus;

typedef struct {
    int line;
    int prev_word_id;
    int next_word_id;
    size_t frequency;
    int hash_next;
} HUAdjEdge;

typedef struct {
    HUAdjEdge edges[HU_MAX_ADJ_EDGES];
    int buckets[HU_ADJ_BUCKETS];
    size_t count;
    size_t overflow;
    size_t transitions;
    uint64_t hash;
} HUAdjacency;

typedef struct {
    const char *corpus_path;
    const char *trace_path;
    const char *save_path;
    size_t epochs;
    size_t max_poems;
    size_t trace_limit;
    double learning_rate;
    unsigned int seed;
} HUConfig;

typedef struct {
    size_t pairs;
    size_t correct;
    size_t by_type_pairs[HU_CORRUPT_COUNT];
    size_t by_type_correct[HU_CORRUPT_COUNT];
    double nll;
} HUEval;

static const HUVocab *g_sort_vocab;

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

static void strip_line(char *line)
{
    size_t len = strlen(line);
    while (len > 0 && (line[len - 1u] == '\n' || line[len - 1u] == '\r' ||
                       isspace((unsigned char)line[len - 1u]))) {
        line[len - 1u] = '\0';
        len -= 1;
    }
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

static int vocab_find(const HUVocab *vocab, const char *word)
{
    for (size_t i = 0; i < vocab->count; ++i) {
        if (strcmp(vocab->entries[i].word, word) == 0) {
            return (int)i;
        }
    }
    return -1;
}

static int vocab_add(HUVocab *vocab, const char *word)
{
    int existing = vocab_find(vocab, word);
    if (existing >= 0) {
        return existing;
    }
    if (vocab->count >= HU_MAX_VOCAB) {
        return -1;
    }
    HUVocabEntry *entry = &vocab->entries[vocab->count];
    memset(entry, 0, sizeof(*entry));
    snprintf(entry->word, sizeof(entry->word), "%s", word);
    entry->length = (int)strlen(word);
    entry->syllables = syllable_proxy(word);
    vocab->ranked_ids[vocab->count] = (int)vocab->count;
    vocab->count += 1;
    return (int)(vocab->count - 1u);
}

static int parse_line_words(char *line, HUVocab *vocab, int *out_words, size_t *out_count)
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
        char word[HU_MAX_WORD_LEN];
        if (!normalize_word(start, word, sizeof(word))) {
            *cursor = saved;
            return 0;
        }
        int id = vocab_add(vocab, word);
        if (id < 0 || *out_count >= HU_MAX_LINE_WORDS) {
            *cursor = saved;
            return 0;
        }
        out_words[*out_count] = id;
        *out_count += 1;
        *cursor = saved;
    }
    return *out_count > 0;
}

static void corpus_add_current(HUCorpus *corpus,
                               HUVocab *vocab,
                               int line_words[3][HU_MAX_LINE_WORDS],
                               const size_t *line_counts)
{
    if (line_counts[0] == 0 || line_counts[1] == 0 || line_counts[2] == 0 ||
        corpus->count >= HU_MAX_POEMS) {
        corpus->ignored_stanzas += 1;
        return;
    }
    HUPoem *poem = &corpus->poems[corpus->count++];
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

static int load_corpus(const char *path, HUVocab *vocab, HUCorpus *corpus, size_t max_poems)
{
    memset(corpus, 0, sizeof(*corpus));
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "failed to open corpus %s: %s\n", path, strerror(errno));
        return 0;
    }

    int line_words[3][HU_MAX_LINE_WORDS];
    size_t line_counts[3] = {0, 0, 0};
    int current_line = 0;
    char line[512];
    while (fgets(line, sizeof(line), fp) != NULL) {
        strip_line(line);
        if (line[0] == '\0') {
            if (current_line == 3) {
                corpus_add_current(corpus, vocab, line_words, line_counts);
                if (max_poems != 0 && corpus->count >= max_poems) {
                    break;
                }
            } else if (current_line != 0) {
                corpus->ignored_stanzas += 1;
            }
            memset(line_words, 0, sizeof(line_words));
            memset(line_counts, 0, sizeof(line_counts));
            current_line = 0;
            continue;
        }
        if (current_line >= 3 ||
            !parse_line_words(line, vocab, line_words[current_line], &line_counts[current_line])) {
            corpus->ignored_words += 1;
            memset(line_words, 0, sizeof(line_words));
            memset(line_counts, 0, sizeof(line_counts));
            current_line = 0;
            continue;
        }
        current_line += 1;
    }
    if ((max_poems == 0 || corpus->count < max_poems) && current_line == 3) {
        corpus_add_current(corpus, vocab, line_words, line_counts);
    } else if (current_line != 0 && current_line != 3) {
        corpus->ignored_stanzas += 1;
    }
    fclose(fp);
    return corpus->count > 1;
}

static int compare_ranked_words(const void *a, const void *b)
{
    const HUVocab *vocab = g_sort_vocab;
    int left = *(const int *)a;
    int right = *(const int *)b;
    const HUVocabEntry *le = &vocab->entries[left];
    const HUVocabEntry *re = &vocab->entries[right];
    if (le->frequency != re->frequency) {
        return le->frequency < re->frequency ? 1 : -1;
    }
    return strcmp(le->word, re->word);
}

static void sort_vocab(HUVocab *vocab)
{
    for (size_t i = 0; i < vocab->count; ++i) {
        vocab->ranked_ids[i] = (int)i;
    }
    g_sort_vocab = vocab;
    qsort(vocab->ranked_ids, vocab->count, sizeof(vocab->ranked_ids[0]), compare_ranked_words);
    g_sort_vocab = NULL;
}

static uint64_t hash_mix_u64(uint64_t hash, uint64_t value)
{
    hash ^= value;
    hash *= 1099511628211ull;
    return hash;
}

static uint64_t hash_adjacency_key(int line, int prev_word_id, int next_word_id)
{
    uint64_t hash = 1469598103934665603ull;
    hash = hash_mix_u64(hash, (uint64_t)(uint32_t)line);
    hash = hash_mix_u64(hash, (uint64_t)(uint32_t)(prev_word_id + 3));
    hash = hash_mix_u64(hash, (uint64_t)(uint32_t)(next_word_id + 3));
    return hash;
}

static void adjacency_init(HUAdjacency *adjacency)
{
    adjacency->count = 0;
    adjacency->overflow = 0;
    adjacency->transitions = 0;
    adjacency->hash = 0;
    for (size_t i = 0; i < HU_ADJ_BUCKETS; ++i) {
        adjacency->buckets[i] = -1;
    }
}

static HUAdjEdge *adjacency_find_edge(HUAdjacency *adjacency,
                                      int line,
                                      int prev_word_id,
                                      int next_word_id)
{
    uint64_t hash = hash_adjacency_key(line, prev_word_id, next_word_id);
    size_t bucket = (size_t)(hash & (uint64_t)(HU_ADJ_BUCKETS - 1));
    for (int edge_id = adjacency->buckets[bucket]; edge_id >= 0;
         edge_id = adjacency->edges[edge_id].hash_next) {
        HUAdjEdge *edge = &adjacency->edges[edge_id];
        if (edge->line == line && edge->prev_word_id == prev_word_id &&
            edge->next_word_id == next_word_id) {
            return edge;
        }
    }
    return NULL;
}

static const HUAdjEdge *adjacency_find_const(const HUAdjacency *adjacency,
                                             int line,
                                             int prev_word_id,
                                             int next_word_id)
{
    uint64_t hash = hash_adjacency_key(line, prev_word_id, next_word_id);
    size_t bucket = (size_t)(hash & (uint64_t)(HU_ADJ_BUCKETS - 1));
    for (int edge_id = adjacency->buckets[bucket]; edge_id >= 0;
         edge_id = adjacency->edges[edge_id].hash_next) {
        const HUAdjEdge *edge = &adjacency->edges[edge_id];
        if (edge->line == line && edge->prev_word_id == prev_word_id &&
            edge->next_word_id == next_word_id) {
            return edge;
        }
    }
    return NULL;
}

static int adjacency_add(HUAdjacency *adjacency, int line, int prev_word_id, int next_word_id)
{
    adjacency->transitions += 1;
    HUAdjEdge *existing = adjacency_find_edge(adjacency, line, prev_word_id, next_word_id);
    if (existing != NULL) {
        existing->frequency += 1;
        return 1;
    }
    if (adjacency->count >= HU_MAX_ADJ_EDGES) {
        adjacency->overflow += 1;
        return 0;
    }
    uint64_t hash = hash_adjacency_key(line, prev_word_id, next_word_id);
    size_t bucket = (size_t)(hash & (uint64_t)(HU_ADJ_BUCKETS - 1));
    HUAdjEdge *edge = &adjacency->edges[adjacency->count];
    edge->line = line;
    edge->prev_word_id = prev_word_id;
    edge->next_word_id = next_word_id;
    edge->frequency = 1;
    edge->hash_next = adjacency->buckets[bucket];
    adjacency->buckets[bucket] = (int)adjacency->count;
    adjacency->count += 1;
    return 1;
}

static void adjacency_finalize_hash(HUAdjacency *adjacency)
{
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < adjacency->count; ++i) {
        const HUAdjEdge *edge = &adjacency->edges[i];
        hash = hash_mix_u64(hash, (uint64_t)(uint32_t)edge->line);
        hash = hash_mix_u64(hash, (uint64_t)(uint32_t)(edge->prev_word_id + 3));
        hash = hash_mix_u64(hash, (uint64_t)(uint32_t)(edge->next_word_id + 3));
        hash = hash_mix_u64(hash, (uint64_t)edge->frequency);
    }
    adjacency->hash = hash;
}

static void build_adjacency(const HUCorpus *corpus, size_t poem_count, HUAdjacency *adjacency)
{
    adjacency_init(adjacency);
    if (poem_count > corpus->count) {
        poem_count = corpus->count;
    }
    for (size_t p = 0; p < poem_count; ++p) {
        const HUPoem *poem = &corpus->poems[p];
        for (int line = 0; line < 3; ++line) {
            int prev_word_id = HU_ADJ_START;
            for (size_t i = 0; i < poem->line_counts[line]; ++i) {
                int word_id = poem->words[line][i];
                (void)adjacency_add(adjacency, line, prev_word_id, word_id);
                prev_word_id = word_id;
            }
            (void)adjacency_add(adjacency, line, prev_word_id, HU_ADJ_BOUNDARY);
        }
    }
    adjacency_finalize_hash(adjacency);
}

static int adjacency_has(const HUAdjacency *adjacency, int line, int prev_word_id, int next_word_id)
{
    return adjacency_find_const(adjacency, line, prev_word_id, next_word_id) != NULL;
}

static int word_in_list(const char *word, const char *const *list, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        if (strcmp(word, list[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

static int is_stopword(const char *word)
{
    static const char *const words[] = {
        "a",   "an",  "and", "are", "as",   "at",   "be",   "by",   "for",
        "from", "i",   "in",  "is",  "it",   "my",   "of",   "on",   "or",
        "s",   "the", "to",  "under", "with", "you", "your", "its",  "into",
    };
    return word_in_list(word, words, sizeof(words) / sizeof(words[0]));
}

static int is_nature_word(const char *word)
{
    static const char *const words[] = {
        "ash",    "autumn", "bamboo", "bay",    "bees",    "bird",   "blossom",
        "branch", "brook",  "cedar",  "cherry", "cloud",   "clouds", "cricket",
        "dawn",   "dusk",   "egret",  "fern",   "field",   "firefly", "flower",
        "fog",    "frost",  "garden", "grass",  "gull",    "harbor", "ice",
        "leaf",   "leaves", "light",  "moon",   "moss",    "mountain", "night",
        "pine",   "pond",   "rain",   "river",  "root",    "rose",   "sea",
        "shadow", "shell",  "sky",    "snow",   "soil",    "sparrow", "spring",
        "star",   "stone",  "storm",  "summer", "sun",     "thunder", "tide",
        "tree",   "valley", "water",  "wind",   "winter",  "wood",
    };
    return word_in_list(word, words, sizeof(words) / sizeof(words[0]));
}

static int is_season_word(const char *word)
{
    static const char *const words[] = {
        "autumn", "blossom", "cherry", "cold", "fall", "frost", "ice",
        "october", "rain", "snow", "spring", "summer", "winter",
    };
    return word_in_list(word, words, sizeof(words) / sizeof(words[0]));
}

static int is_abstract_word(const char *word)
{
    static const char *const words[] = {
        "alone", "beauty", "dream", "endless", "feeling", "heart", "hope",
        "life", "love", "memory", "mind", "sadness", "silence", "soul",
        "thought", "time", "truth", "void",
    };
    return word_in_list(word, words, sizeof(words) / sizeof(words[0]));
}

static size_t train_split_count(const HUCorpus *corpus)
{
    if (corpus->count <= 1) {
        return corpus->count;
    }
    size_t train = (corpus->count * 4u) / 5u;
    return train == 0 ? 1u : train;
}

static const char *corruption_name(int type)
{
    switch (type) {
        case HU_CORRUPT_LINE_SHUFFLE:
            return "line_shuffle";
        case HU_CORRUPT_LINE_SPLICE:
            return "line_splice";
        case HU_CORRUPT_SYLLABLE_BREAK:
            return "syllable_break";
        case HU_CORRUPT_REPETITION:
            return "repetition";
        case HU_CORRUPT_WORD_SALAD:
            return "word_salad";
        default:
            return "unknown";
    }
}

static void copy_poem(HUPoem *dst, const HUPoem *src)
{
    memcpy(dst, src, sizeof(*dst));
}

static void copy_line(HUPoem *dst, int dst_line, const HUPoem *src, int src_line)
{
    dst->line_counts[dst_line] = src->line_counts[src_line];
    for (size_t i = 0; i < src->line_counts[src_line]; ++i) {
        dst->words[dst_line][i] = src->words[src_line][i];
    }
}

static int ranked_vocab_id(const HUVocab *vocab, size_t offset)
{
    if (vocab->count == 0) {
        return -1;
    }
    return vocab->ranked_ids[offset % vocab->count];
}

static void make_corruption(const HUCorpus *corpus,
                            const HUVocab *vocab,
                            size_t poem_index,
                            int type,
                            unsigned int seed,
                            HUPoem *out)
{
    const HUPoem *real = &corpus->poems[poem_index];
    copy_poem(out, real);
    uint64_t mix = (uint64_t)seed + poem_index * 1103515245ull + (uint64_t)type * 7919ull;
    switch (type) {
        case HU_CORRUPT_LINE_SHUFFLE: {
            int a = (int)(mix % 3u);
            int b = (a + 1 + (int)((mix >> 8) % 2u)) % 3;
            HUPoem tmp;
            copy_poem(&tmp, out);
            copy_line(out, a, &tmp, b);
            copy_line(out, b, &tmp, a);
            break;
        }
        case HU_CORRUPT_LINE_SPLICE: {
            size_t other = (poem_index + 1u + (size_t)(mix % (corpus->count - 1u))) % corpus->count;
            int line = (int)((mix >> 12) % 3u);
            copy_line(out, line, &corpus->poems[other], line);
            break;
        }
        case HU_CORRUPT_SYLLABLE_BREAK: {
            int line = (int)(mix % 3u);
            size_t count = out->line_counts[line];
            if (count < HU_MAX_LINE_WORDS) {
                int id = ranked_vocab_id(vocab, 2u + (size_t)((mix >> 16) % 32u));
                if (id >= 0) {
                    out->words[line][count] = id;
                    out->line_counts[line] = count + 1u;
                }
            }
            break;
        }
        case HU_CORRUPT_REPETITION: {
            int line = (int)(mix % 3u);
            int id = ranked_vocab_id(vocab, (size_t)((mix >> 4) % 8u));
            size_t count = out->line_counts[line] > 0 ? out->line_counts[line] : 5u;
            if (count > HU_MAX_LINE_WORDS) {
                count = HU_MAX_LINE_WORDS;
            }
            for (size_t i = 0; i < count; ++i) {
                out->words[line][i] = id >= 0 ? id : 0;
            }
            out->line_counts[line] = count;
            break;
        }
        case HU_CORRUPT_WORD_SALAD: {
            for (int line = 0; line < 3; ++line) {
                size_t count = out->line_counts[line];
                for (size_t i = 0; i < count; ++i) {
                    int id = ranked_vocab_id(vocab, (size_t)((mix + (uint64_t)line * 17u +
                                                              (uint64_t)i * 31u) %
                                                             96u));
                    out->words[line][i] = id >= 0 ? id : out->words[line][i];
                }
            }
            break;
        }
        default:
            break;
    }
}

static uint64_t hash_poem(const HUPoem *poem)
{
    uint64_t hash = 1469598103934665603ull;
    for (int line = 0; line < 3; ++line) {
        hash = hash_mix_u64(hash, (uint64_t)poem->line_counts[line]);
        for (size_t i = 0; i < poem->line_counts[line]; ++i) {
            hash = hash_mix_u64(hash, (uint64_t)(uint32_t)poem->words[line][i]);
        }
    }
    return hash;
}

static void fill_features(const HUVocab *vocab,
                          const HUAdjacency *adjacency,
                          const HUPoem *poem,
                          nn_real *features)
{
    memset(features, 0, HU_FEATURE_COUNT * sizeof(*features));
    size_t total_words = 0;
    size_t total_chars = 0;
    size_t unique_words = 0;
    size_t stopwords = 0;
    size_t nature_words = 0;
    size_t season_words = 0;
    size_t abstract_words = 0;
    size_t rare_words = 0;
    size_t adjacent_supported = 0;
    size_t adjacent_total = 0;
    size_t bigram_supported = 0;
    size_t bigram_total = 0;
    size_t start_supported = 0;
    size_t boundary_supported = 0;
    size_t exact_lines = 0;
    size_t repeated_positions = 0;
    size_t max_line_repeat = 0;
    double log_frequency_sum = 0.0;
    double min_log_frequency = 0.0;
    int seen_any_frequency = 0;
    int seen[HU_MAX_LINE_WORDS * 3];
    size_t seen_count = 0;
    int line_syllables[3] = {0, 0, 0};
    int line_chars[3] = {0, 0, 0};

    for (int line = 0; line < 3; ++line) {
        int prev = HU_ADJ_START;
        int line_seen[HU_MAX_LINE_WORDS];
        size_t line_seen_count = 0;
        for (size_t i = 0; i < poem->line_counts[line]; ++i) {
            int word_id = poem->words[line][i];
            const HUVocabEntry *entry = &vocab->entries[word_id];
            total_words += 1;
            total_chars += (size_t)entry->length;
            line_chars[line] += entry->length + (i > 0 ? 1 : 0);
            line_syllables[line] += entry->syllables;
            if (is_stopword(entry->word)) {
                stopwords += 1;
            }
            if (is_nature_word(entry->word)) {
                nature_words += 1;
            }
            if (is_season_word(entry->word)) {
                season_words += 1;
            }
            if (is_abstract_word(entry->word)) {
                abstract_words += 1;
            }
            if (entry->frequency <= 1) {
                rare_words += 1;
            }
            double lf = log((double)entry->frequency + 1.0);
            log_frequency_sum += lf;
            if (!seen_any_frequency || lf < min_log_frequency) {
                min_log_frequency = lf;
                seen_any_frequency = 1;
            }
            int already_seen = 0;
            for (size_t s = 0; s < seen_count; ++s) {
                if (seen[s] == word_id) {
                    already_seen = 1;
                    break;
                }
            }
            if (!already_seen && seen_count < sizeof(seen) / sizeof(seen[0])) {
                seen[seen_count++] = word_id;
                unique_words += 1;
            } else if (already_seen) {
                repeated_positions += 1;
            }
            int line_repeats = 0;
            for (size_t s = 0; s < line_seen_count; ++s) {
                if (line_seen[s] == word_id) {
                    line_repeats += 1;
                }
            }
            if (line_seen_count < sizeof(line_seen) / sizeof(line_seen[0])) {
                line_seen[line_seen_count++] = word_id;
            }
            if ((size_t)line_repeats > max_line_repeat) {
                max_line_repeat = (size_t)line_repeats;
            }
            adjacent_total += 1;
            if (adjacency_has(adjacency, line, prev, word_id)) {
                adjacent_supported += 1;
                if (prev == HU_ADJ_START) {
                    start_supported += 1;
                }
            }
            if (prev != HU_ADJ_START) {
                bigram_total += 1;
                if (adjacency_has(adjacency, line, prev, word_id)) {
                    bigram_supported += 1;
                }
            }
            prev = word_id;
        }
        adjacent_total += 1;
        if (adjacency_has(adjacency, line, prev, HU_ADJ_BOUNDARY)) {
            adjacent_supported += 1;
            boundary_supported += 1;
        }
        if (line_syllables[line] == HU_TARGET_SYLLABLES[line]) {
            exact_lines += 1;
        }
    }

    int syllable_error0 = abs(line_syllables[0] - HU_TARGET_SYLLABLES[0]);
    int syllable_error1 = abs(line_syllables[1] - HU_TARGET_SYLLABLES[1]);
    int syllable_error2 = abs(line_syllables[2] - HU_TARGET_SYLLABLES[2]);
    int total_syllable_error = syllable_error0 + syllable_error1 + syllable_error2;
    double total = total_words > 0 ? (double)total_words : 1.0;
    double max_log_freq = vocab->max_frequency > 0 ? log((double)vocab->max_frequency + 1.0) : 1.0;
    double avg_log_freq = total_words > 0 ? log_frequency_sum / total : 0.0;
    double avg_chars = total_words > 0 ? (double)total_chars / total : 0.0;
    double line_balance = 0.0;
    for (int line = 0; line < 3; ++line) {
        line_balance += fabs((double)poem->line_counts[line] - total / 3.0);
    }

    features[0] = (nn_real)1;
    features[1] = (nn_real)(total_words / 40.0);
    features[2] = (nn_real)(avg_chars / 12.0);
    features[3] = (nn_real)((double)unique_words / total);
    features[4] = (nn_real)((double)repeated_positions / total);
    features[5] = (nn_real)((double)max_line_repeat / total);
    features[6] = (nn_real)((double)stopwords / total);
    features[7] = (nn_real)((double)nature_words / total);
    features[8] = (nn_real)((double)season_words / total);
    features[9] = (nn_real)((double)abstract_words / total);
    features[10] = (nn_real)((double)total_syllable_error / 12.0);
    features[11] = (nn_real)((double)exact_lines / 3.0);
    features[12] = (nn_real)((double)syllable_error0 / 7.0);
    features[13] = (nn_real)((double)syllable_error1 / 7.0);
    features[14] = (nn_real)((double)syllable_error2 / 7.0);
    features[15] = (nn_real)((double)line_syllables[0] / 9.0);
    features[16] = (nn_real)((double)line_syllables[1] / 11.0);
    features[17] = (nn_real)((double)line_syllables[2] / 9.0);
    features[18] = (nn_real)((double)poem->line_counts[0] / 12.0);
    features[19] = (nn_real)((double)poem->line_counts[1] / 14.0);
    features[20] = (nn_real)((double)poem->line_counts[2] / 12.0);
    features[21] = (nn_real)((double)line_chars[0] / 40.0);
    features[22] = (nn_real)((double)line_chars[1] / 48.0);
    features[23] = (nn_real)((double)line_chars[2] / 40.0);
    features[24] =
        (nn_real)(adjacent_total > 0 ? (double)adjacent_supported / (double)adjacent_total : 0.0);
    features[25] =
        (nn_real)(bigram_total > 0 ? (double)bigram_supported / (double)bigram_total : 0.0);
    features[26] = (nn_real)((double)start_supported / 3.0);
    features[27] = (nn_real)((double)boundary_supported / 3.0);
    features[28] = (nn_real)(avg_log_freq / max_log_freq);
    features[29] = (nn_real)(min_log_frequency / max_log_freq);
    features[30] = (nn_real)((double)rare_words / total);
    features[31] = (nn_real)(line_balance / 18.0);
    features[32] = season_words > 0 ? (nn_real)1 : (nn_real)0;
    features[33] = nature_words > 0 ? (nn_real)1 : (nn_real)0;
    features[34] = total_syllable_error == 0 ? (nn_real)1 : (nn_real)0;
    features[35] = adjacent_total == adjacent_supported ? (nn_real)1 : (nn_real)0;
    features[36] = bigram_total == bigram_supported ? (nn_real)1 : (nn_real)0;
    features[37] = stopwords > total_words / 2u ? (nn_real)1 : (nn_real)0;
    features[38] = repeated_positions == 0 ? (nn_real)1 : (nn_real)0;
    features[39] = abstract_words == 0 ? (nn_real)1 : (nn_real)0;
}

static void fill_pair_features(const HUVocab *vocab,
                               const HUAdjacency *adjacency,
                               const HUPoem *real,
                               const HUPoem *corrupt,
                               nn_real *inputs)
{
    fill_features(vocab, adjacency, real, &inputs[0]);
    fill_features(vocab, adjacency, corrupt, &inputs[HU_FEATURE_COUNT]);
}

static int score_pair(NeuralNet *net,
                      const HUVocab *vocab,
                      const HUAdjacency *adjacency,
                      const HUPoem *real,
                      const HUPoem *corrupt,
                      nn_real *scores,
                      nn_real *probabilities)
{
    nn_real inputs[HU_FEATURE_COUNT * 2];
    fill_pair_features(vocab, adjacency, real, corrupt, inputs);
    return nn_score_action_rows(net, inputs, 2, scores, probabilities);
}

static void train_pair(NeuralNet *net,
                       const HUVocab *vocab,
                       const HUAdjacency *adjacency,
                       const HUPoem *real,
                       const HUPoem *corrupt,
                       double learning_rate,
                       double *loss_sum,
                       size_t *updates)
{
    nn_real inputs[HU_FEATURE_COUNT * 2];
    fill_pair_features(vocab, adjacency, real, corrupt, inputs);
    nn_real loss = nn_train_softmax_scorer(net, inputs, 2, 0, (nn_real)learning_rate);
    if (!isnan((double)loss)) {
        *loss_sum += (double)loss;
        *updates += 1;
    }
}

static int write_trace_row(FILE *trace,
                           const HUPoem *real,
                           const HUPoem *corrupt,
                           int type,
                           size_t poem_index,
                           const nn_real *scores,
                           const nn_real *probabilities,
                           const char *model_hash,
                           const HUAdjacency *adjacency)
{
    uint64_t state_hash = hash_mix_u64(hash_poem(real), hash_poem(corrupt));
    return fprintf(trace,
                   "{\"schema\":\"crlp.haiku_understanding.trace.v1\","
                   "\"task\":\"haiku_understanding\",\"poem_index\":%zu,"
                   "\"corruption\":\"%s\",\"state_hash\":\"%016llx\","
                   "\"feature_encoder_version\":%u,\"candidate_count\":2,"
                   "\"real_score\":%.9g,\"corrupt_score\":%.9g,"
                   "\"real_probability\":%.9g,\"corrupt_probability\":%.9g,"
                   "\"selected\":\"%s\",\"model_hash\":\"%s\","
                   "\"adjacency_hash\":\"%016llx\"}\n",
                   poem_index,
                   corruption_name(type),
                   (unsigned long long)state_hash,
                   (unsigned int)HU_FEATURE_ENCODER_VERSION,
                   (double)scores[0],
                   (double)scores[1],
                   (double)probabilities[0],
                   (double)probabilities[1],
                   scores[0] >= scores[1] ? "real" : "corrupt",
                   model_hash,
                   (unsigned long long)adjacency->hash) > 0;
}

static HUEval evaluate_split(NeuralNet *net,
                             const HUVocab *vocab,
                             const HUAdjacency *adjacency,
                             const HUCorpus *corpus,
                             size_t start,
                             size_t end,
                             const HUConfig *config,
                             const char *model_hash,
                             FILE *trace)
{
    HUEval eval;
    memset(&eval, 0, sizeof(eval));
    size_t traced = 0;
    for (size_t p = start; p < end; ++p) {
        const HUPoem *real = &corpus->poems[p];
        for (int type = 0; type < HU_CORRUPT_COUNT; ++type) {
            HUPoem corrupt;
            make_corruption(corpus, vocab, p, type, config->seed, &corrupt);
            nn_real scores[2];
            nn_real probabilities[2];
            if (!score_pair(net, vocab, adjacency, real, &corrupt, scores, probabilities)) {
                continue;
            }
            double probability = (double)probabilities[0];
            if (probability < 1e-12) {
                probability = 1e-12;
            }
            eval.nll += -log(probability);
            eval.pairs += 1;
            eval.by_type_pairs[type] += 1;
            if (scores[0] >= scores[1]) {
                eval.correct += 1;
                eval.by_type_correct[type] += 1;
            }
            if (trace != NULL && (config->trace_limit == 0 || traced < config->trace_limit)) {
                (void)write_trace_row(trace,
                                      real,
                                      &corrupt,
                                      type,
                                      p,
                                      scores,
                                      probabilities,
                                      model_hash,
                                      adjacency);
                traced += 1;
            }
        }
    }
    return eval;
}

static void usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s [--corpus path] [--epochs N] [--lr R] [--seed N]\n"
            "       [--max-poems N] [--trace path] [--trace-limit N] [--save path]\n",
            argv0);
}

static int parse_size(const char *text, size_t *out)
{
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return 0;
    }
    *out = (size_t)value;
    return 1;
}

static int parse_double_arg(const char *text, double *out)
{
    char *end = NULL;
    errno = 0;
    double value = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0') {
        return 0;
    }
    *out = value;
    return 1;
}

static int parse_config(int argc, char **argv, HUConfig *config)
{
    config->corpus_path = "data/haiku_tiny.txt";
    config->trace_path = NULL;
    config->save_path = NULL;
    config->epochs = 8;
    config->max_poems = 0;
    config->trace_limit = 2000;
    config->learning_rate = 0.006;
    config->seed = 20240518u;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--corpus") == 0 && i + 1 < argc) {
            config->corpus_path = argv[++i];
        } else if (strcmp(argv[i], "--trace") == 0 && i + 1 < argc) {
            config->trace_path = argv[++i];
        } else if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            config->save_path = argv[++i];
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            if (!parse_size(argv[++i], &config->epochs)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--max-poems") == 0 && i + 1 < argc) {
            if (!parse_size(argv[++i], &config->max_poems)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--trace-limit") == 0 && i + 1 < argc) {
            if (!parse_size(argv[++i], &config->trace_limit)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            if (!parse_double_arg(argv[++i], &config->learning_rate)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            size_t value = 0;
            if (!parse_size(argv[++i], &value)) {
                return 0;
            }
            config->seed = (unsigned int)value;
        } else {
            return 0;
        }
    }
    return config->epochs > 0 && config->learning_rate > 0.0;
}

int main(int argc, char **argv)
{
    HUConfig config;
    if (!parse_config(argc, argv, &config)) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    static HUVocab vocab;
    static HUCorpus corpus;
    memset(&vocab, 0, sizeof(vocab));
    if (!load_corpus(config.corpus_path, &vocab, &corpus, config.max_poems)) {
        return EXIT_FAILURE;
    }
    sort_vocab(&vocab);
    size_t train_count = train_split_count(&corpus);
    static HUAdjacency adjacency;
    build_adjacency(&corpus, train_count, &adjacency);
    if (adjacency.overflow != 0) {
        fprintf(stderr,
                "adjacency graph exceeds HU_MAX_ADJ_EDGES=%d overflow=%zu\n",
                HU_MAX_ADJ_EDGES,
                adjacency.overflow);
        return EXIT_FAILURE;
    }

    const size_t layers[] = {HU_FEATURE_COUNT, 64, 32, 1};
    NNConfig nn_config;
    nn_config.layer_sizes = layers;
    nn_config.layer_count = sizeof(layers) / sizeof(layers[0]);
    nn_config.seed = config.seed;
    nn_config.hidden_activation = NN_ACTIVATION_RELU;
    nn_config.output_activation = NN_ACTIVATION_LINEAR;
    nn_config.loss = NN_LOSS_MSE;
    NeuralNet *net = nn_create_config(&nn_config);
    if (net == NULL) {
        fputs("failed to create haiku_understanding network\n", stderr);
        return EXIT_FAILURE;
    }

    printf("haiku_understanding corpus=%s poems=%zu train=%zu valid=%zu ignored_stanzas=%zu "
           "ignored_words=%zu vocab=%zu features=%d parameters=%zu adjacency_edges=%zu "
           "adjacency_hash=%016llx corruptions=%d\n",
           config.corpus_path,
           corpus.count,
           train_count,
           corpus.count - train_count,
           corpus.ignored_stanzas,
           corpus.ignored_words,
           vocab.count,
           HU_FEATURE_COUNT,
           nn_parameter_count(net),
           adjacency.count,
           (unsigned long long)adjacency.hash,
           HU_CORRUPT_COUNT);

    double training_seconds = 0.0;
    size_t total_updates = 0;
    for (size_t epoch = 1; epoch <= config.epochs; ++epoch) {
        double started = (double)clock() / (double)CLOCKS_PER_SEC;
        double loss_sum = 0.0;
        size_t updates = 0;
        for (size_t p = 0; p < train_count; ++p) {
            const HUPoem *real = &corpus.poems[p];
            for (int type = 0; type < HU_CORRUPT_COUNT; ++type) {
                HUPoem corrupt;
                make_corruption(&corpus, &vocab, p, type, config.seed, &corrupt);
                train_pair(net,
                           &vocab,
                           &adjacency,
                           real,
                           &corrupt,
                           config.learning_rate,
                           &loss_sum,
                           &updates);
            }
        }
        double elapsed = (double)clock() / (double)CLOCKS_PER_SEC - started;
        training_seconds += elapsed;
        total_updates += updates;
        printf("epoch %zu loss=%.4f updates=%zu %.0f pairs/s\n",
               epoch,
               updates > 0 ? loss_sum / (double)updates : 0.0,
               updates,
               elapsed > 0.0 ? (double)updates / elapsed : 0.0);
    }

    char model_hash[65] = "unsaved";
    if (config.save_path != NULL) {
        NNCheckpointMeta meta;
        memset(&meta, 0, sizeof(meta));
        meta.checkpoint_version = 1u;
        meta.feature_encoder_version = HU_FEATURE_ENCODER_VERSION;
        snprintf(meta.notes, sizeof(meta.notes), "haiku_understanding real-vs-corrupt scorer");
        snprintf(meta.feature_set, sizeof(meta.feature_set), "haiku_understanding_v1");
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

    HUEval train_eval =
        evaluate_split(net, &vocab, &adjacency, &corpus, 0, train_count, &config, model_hash, NULL);
    HUEval valid_eval = evaluate_split(net,
                                       &vocab,
                                       &adjacency,
                                       &corpus,
                                       train_count,
                                       corpus.count,
                                       &config,
                                       model_hash,
                                       trace);
    if (trace != NULL) {
        fclose(trace);
        printf("wrote trace: %s\n", config.trace_path);
    }

    double train_acc =
        train_eval.pairs > 0 ? (double)train_eval.correct / (double)train_eval.pairs : 0.0;
    double valid_acc =
        valid_eval.pairs > 0 ? (double)valid_eval.correct / (double)valid_eval.pairs : 0.0;
    double valid_nll = valid_eval.pairs > 0 ? valid_eval.nll / (double)valid_eval.pairs : 0.0;
    printf("train_eval pairs=%zu top1=%.2f%% nll=%.4f\n",
           train_eval.pairs,
           train_acc * 100.0,
           train_eval.pairs > 0 ? train_eval.nll / (double)train_eval.pairs : 0.0);
    printf("valid_eval pairs=%zu top1=%.2f%% nll=%.4f\n",
           valid_eval.pairs,
           valid_acc * 100.0,
           valid_nll);
    for (int type = 0; type < HU_CORRUPT_COUNT; ++type) {
        double acc = valid_eval.by_type_pairs[type] > 0
                         ? (double)valid_eval.by_type_correct[type] /
                               (double)valid_eval.by_type_pairs[type]
                         : 0.0;
        printf("valid_%s pairs=%zu top1=%.2f%%\n",
               corruption_name(type),
               valid_eval.by_type_pairs[type],
               acc * 100.0);
    }
    printf("training_seconds=%.6f total_updates=%zu\n", training_seconds, total_updates);

    printf("{\"schema\":\"crlp.haiku_understanding.summary.v1\",\"poems\":%zu,"
           "\"train_pairs\":%zu,\"valid_pairs\":%zu,\"train_top1\":%.6f,"
           "\"valid_top1\":%.6f,\"valid_nll\":%.6f,\"updates\":%zu,"
           "\"vocab_entries\":%zu,\"adjacency_edges\":%zu,"
           "\"adjacency_transitions\":%zu,\"adjacency_hash\":\"%016llx\","
           "\"feature_encoder_version\":%u,\"model_hash\":\"%s\","
           "\"by_corruption\":{",
           corpus.count,
           train_eval.pairs,
           valid_eval.pairs,
           train_acc,
           valid_acc,
           valid_nll,
           total_updates,
           vocab.count,
           adjacency.count,
           adjacency.transitions,
           (unsigned long long)adjacency.hash,
           (unsigned int)HU_FEATURE_ENCODER_VERSION,
           model_hash);
    for (int type = 0; type < HU_CORRUPT_COUNT; ++type) {
        double acc = valid_eval.by_type_pairs[type] > 0
                         ? (double)valid_eval.by_type_correct[type] /
                               (double)valid_eval.by_type_pairs[type]
                         : 0.0;
        printf("%s\"%s\":{\"pairs\":%zu,\"top1\":%.6f}",
               type == 0 ? "" : ",",
               corruption_name(type),
               valid_eval.by_type_pairs[type],
               acc);
    }
    printf("}}\n");

    nn_free(net);
    return EXIT_SUCCESS;
}
