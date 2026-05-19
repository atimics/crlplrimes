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
    HAIKU_FEATURE_COUNT = 46,
    HAIKU_MAX_CANDIDATES = 64,
    HAIKU_MAX_POEMS = 8192,
    HAIKU_MAX_POEM_CHARS = 512,
    HAIKU_MAX_LINE = 256,
    HAIKU_MAX_GENERATED = 256,
    HAIKU_MAX_WORD = 96,
    HAIKU_CONTEXT = 8,
    HAIKU_FEATURE_ENCODER_VERSION = 1,
    HAIKU_STAT_START = 0,
    HAIKU_STAT_EOS = 1,
    HAIKU_STAT_COUNT = 35,
};

static const char HAIKU_VOCAB[] = "\n abcdefghijklmnopqrstuvwxyz',.-;";
static const int HAIKU_TARGET_SYLLABLES[3] = {5, 7, 5};
static const int HAIKU_MAX_LINE_CHARS[3] = {28, 34, 28};

typedef struct {
    char text[HAIKU_MAX_POEM_CHARS];
    size_t length;
} HaikuPoem;

typedef struct {
    HaikuPoem poems[HAIKU_MAX_POEMS];
    size_t count;
    size_t ignored_stanzas;
} HaikuCorpus;

typedef struct {
    const char *corpus_path;
    const char *trace_path;
    const char *save_path;
    const char *word_list_path;
    size_t epochs;
    size_t max_updates;
    size_t samples;
    size_t top_k;
    double learning_rate;
    double temperature;
    unsigned int seed;
    int sample;
    int lexicon_gate;
} HaikuConfig;

typedef struct {
    int line;
    int line_chars;
    int line_syllables;
    int total_chars;
    int prev_was_vowel;
    char prev_char;
} HaikuState;

typedef struct {
    char token;
    uint32_t action_id;
} HaikuCandidate;

typedef struct {
    double unigram[HAIKU_STAT_COUNT];
    double bigram[HAIKU_STAT_COUNT][HAIKU_STAT_COUNT];
    double trigram[HAIKU_STAT_COUNT][HAIKU_STAT_COUNT][HAIKU_STAT_COUNT];
    double line_start[HAIKU_STAT_COUNT];
    double after_space[HAIKU_STAT_COUNT];
} HaikuStats;

typedef struct {
    size_t rows;
    size_t skipped;
    size_t top1;
    double nll;
} HaikuEval;

typedef struct {
    size_t child[26];
    size_t min_suffix;
    int terminal;
} HaikuTrieNode;

typedef struct {
    char **words;
    size_t count;
    size_t capacity;
    HaikuTrieNode *trie;
    size_t trie_count;
    size_t trie_capacity;
    uint64_t hash;
    const char *path;
} HaikuLexicon;

typedef struct {
    size_t words;
    size_t dictionary_words;
} HaikuLexicalScore;

typedef struct {
    size_t samples;
    size_t words;
    size_t dictionary_words;
} HaikuLexicalAggregate;

typedef struct {
    size_t shape;
    size_t lexicon_prefix;
    size_t lexicon_boundary;
    size_t lexicon_completion;
    size_t dead_end;
} HaikuRejectStats;

typedef enum {
    HAIKU_LEXICON_OK = 0,
    HAIKU_LEXICON_REJECT_PREFIX = 1,
    HAIKU_LEXICON_REJECT_BOUNDARY = 2,
    HAIKU_LEXICON_REJECT_COMPLETION = 3,
} HaikuLexiconDecision;

static HaikuStats g_stats;

static char target_at(const HaikuPoem *poem, size_t position);
static HaikuLexiconDecision lexicon_candidate_decision(const char *prefix,
                                                       size_t prefix_len,
                                                       const HaikuState *state,
                                                       const HaikuLexicon *lexicon,
                                                       char token);

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

static char *copy_string(const char *text)
{
    size_t length = strlen(text);
    char *copy = (char *)malloc(length + 1u);
    if (copy == NULL) {
        return NULL;
    }
    memcpy(copy, text, length + 1u);
    return copy;
}

static int is_vowel_char(char c)
{
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y';
}

static int is_letter_char(char c)
{
    return c >= 'a' && c <= 'z';
}

static int is_punctuation_char(char c)
{
    return c == '\'' || c == ',' || c == '.' || c == '-' || c == ';';
}

static size_t vocab_len(void)
{
    return sizeof(HAIKU_VOCAB) - 1u;
}

static int vocab_index(char c)
{
    for (size_t i = 0; i < vocab_len(); ++i) {
        if (HAIKU_VOCAB[i] == c) {
            return (int)i;
        }
    }
    return -1;
}

static uint32_t action_id_for_token(char token)
{
    return token == '\0' ? 1u : 2u + (uint32_t)(unsigned char)token;
}

static size_t stat_index_for_token(char token)
{
    if (token == '\0') {
        return HAIKU_STAT_EOS;
    }
    int index = vocab_index(token);
    if (index < 0) {
        return HAIKU_STAT_EOS;
    }
    return 2u + (size_t)index;
}

static size_t stat_index_for_prefix_char(char token)
{
    return token == '\0' ? HAIKU_STAT_START : stat_index_for_token(token);
}

static double row_probability(const double *row, size_t index)
{
    double total = 0.0;
    for (size_t i = 0; i < HAIKU_STAT_COUNT; ++i) {
        total += row[i];
    }
    return total > 0.0 ? row[index] / total : 0.0;
}

static void build_stats(const HaikuCorpus *corpus, size_t train_count)
{
    memset(&g_stats, 0, sizeof(g_stats));
    for (size_t i = 0; i < HAIKU_STAT_COUNT; ++i) {
        g_stats.unigram[i] = 0.05;
        g_stats.line_start[i] = 0.05;
        g_stats.after_space[i] = 0.05;
        for (size_t j = 0; j < HAIKU_STAT_COUNT; ++j) {
            g_stats.bigram[i][j] = 0.05;
            for (size_t k = 0; k < HAIKU_STAT_COUNT; ++k) {
                g_stats.trigram[i][j][k] = 0.05;
            }
        }
    }

    for (size_t p = 0; p < train_count; ++p) {
        const HaikuPoem *poem = &corpus->poems[p];
        size_t prev2 = HAIKU_STAT_START;
        size_t prev1 = HAIKU_STAT_START;
        int at_line_start = 1;
        int after_space = 0;
        for (size_t pos = 0; pos <= poem->length; ++pos) {
            char token = target_at(poem, pos);
            size_t index = stat_index_for_token(token);
            g_stats.unigram[index] += 1.0;
            g_stats.bigram[prev1][index] += 1.0;
            g_stats.trigram[prev2][prev1][index] += 1.0;
            if (at_line_start) {
                g_stats.line_start[index] += 1.0;
            }
            if (after_space) {
                g_stats.after_space[index] += 1.0;
            }
            if (token == '\0') {
                break;
            }
            at_line_start = token == '\n';
            after_space = token == ' ';
            prev2 = prev1;
            prev1 = stat_index_for_prefix_char(token);
        }
    }
}

static uint64_t hash_prefix(const char *prefix, size_t length)
{
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < length; ++i) {
        hash ^= (uint64_t)(unsigned char)prefix[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

static void update_state(HaikuState *state, char token)
{
    if (token == '\n') {
        if (state->line < 3) {
            state->line += 1;
        }
        state->line_chars = 0;
        state->line_syllables = 0;
        state->prev_was_vowel = 0;
        state->prev_char = '\n';
        return;
    }

    int vowel = is_vowel_char(token);
    if (vowel && !state->prev_was_vowel) {
        state->line_syllables += 1;
    }
    state->line_chars += 1;
    state->total_chars += 1;
    state->prev_was_vowel = vowel;
    state->prev_char = token;
}

static HaikuState state_from_prefix(const char *prefix, size_t length)
{
    HaikuState state;
    memset(&state, 0, sizeof(state));
    state.prev_char = '\n';
    for (size_t i = 0; i < length; ++i) {
        update_state(&state, prefix[i]);
    }
    return state;
}

static int candidate_syllable_increment(const HaikuState *state, char token)
{
    return is_vowel_char(token) && !state->prev_was_vowel ? 1 : 0;
}

static int token_allowed(const HaikuState *state, char token)
{
    if (state->line > 2) {
        return token == '\0';
    }

    int line = state->line;
    int target = HAIKU_TARGET_SYLLABLES[line];
    int max_chars = HAIKU_MAX_LINE_CHARS[line];

    if (token == '\0') {
        return line == 2 && state->line_chars >= 3 &&
               (state->line_syllables >= target - 1 || state->line_chars >= max_chars - 6);
    }

    if (token == '\n') {
        return line < 2 && state->line_chars >= 3 && state->prev_char != ' ' &&
               (state->line_syllables >= target - 1 || state->line_chars >= max_chars - 6);
    }

    if (state->line_chars >= max_chars) {
        return 0;
    }

    if (token == ' ') {
        return state->line_chars > 0 && state->prev_char != ' ' && state->prev_char != '\n';
    }

    if (is_punctuation_char(token)) {
        return state->line_chars > 0 && state->prev_char != ' ' && state->prev_char != '\n' &&
               !is_punctuation_char(state->prev_char);
    }

    if (!is_letter_char(token)) {
        return 0;
    }

    int next_syllables = state->line_syllables + candidate_syllable_increment(state, token);
    return next_syllables <= target + 2 || state->line_chars < 4;
}

static void add_reject_stats(HaikuRejectStats *dst, const HaikuRejectStats *src)
{
    if (dst == NULL || src == NULL) {
        return;
    }
    dst->shape += src->shape;
    dst->lexicon_prefix += src->lexicon_prefix;
    dst->lexicon_boundary += src->lexicon_boundary;
    dst->lexicon_completion += src->lexicon_completion;
    dst->dead_end += src->dead_end;
}

static void record_lexicon_reject(HaikuRejectStats *rejects, HaikuLexiconDecision decision)
{
    if (rejects == NULL) {
        return;
    }
    if (decision == HAIKU_LEXICON_REJECT_PREFIX) {
        rejects->lexicon_prefix += 1;
    } else if (decision == HAIKU_LEXICON_REJECT_BOUNDARY) {
        rejects->lexicon_boundary += 1;
    } else if (decision == HAIKU_LEXICON_REJECT_COMPLETION) {
        rejects->lexicon_completion += 1;
    }
}

static size_t build_candidates(const char *prefix,
                               size_t prefix_len,
                               const HaikuState *state,
                               const HaikuLexicon *lexicon,
                               int lexicon_gate,
                               HaikuRejectStats *rejects,
                               HaikuCandidate *candidates)
{
    size_t count = 0;
    const int use_lexicon = lexicon_gate && lexicon != NULL && lexicon->trie_count > 0;
    for (size_t i = 0; i <= vocab_len() && count < HAIKU_MAX_CANDIDATES; ++i) {
        char token = i == 0 ? '\0' : HAIKU_VOCAB[i - 1u];
        if (!token_allowed(state, token)) {
            if (rejects != NULL) {
                rejects->shape += 1;
            }
            continue;
        }
        if (use_lexicon) {
            HaikuLexiconDecision decision =
                lexicon_candidate_decision(prefix, prefix_len, state, lexicon, token);
            if (decision != HAIKU_LEXICON_OK) {
                record_lexicon_reject(rejects, decision);
                continue;
            }
        }
        candidates[count].token = token;
        candidates[count].action_id = action_id_for_token(token);
        count += 1;
    }

    if (count == 0) {
        if (!use_lexicon) {
            candidates[count].token = '\0';
            candidates[count].action_id = action_id_for_token('\0');
            count += 1;
        } else if (rejects != NULL) {
            rejects->dead_end += 1;
        }
    }
    return count;
}

static int target_index(const HaikuCandidate *candidates, size_t count, char target, size_t *out)
{
    for (size_t i = 0; i < count; ++i) {
        if (candidates[i].token == target) {
            *out = i;
            return 1;
        }
    }
    return 0;
}

static void fill_features(const char *prefix,
                          size_t prefix_len,
                          const HaikuState *state,
                          const HaikuCandidate *candidate,
                          nn_real *features)
{
    memset(features, 0, HAIKU_FEATURE_COUNT * sizeof(*features));
    char token = candidate->token;
    int line = state->line <= 2 ? state->line : 2;
    int target = HAIKU_TARGET_SYLLABLES[line];
    int increment = candidate_syllable_increment(state, token);
    int next_syllables = state->line_syllables + increment;

    features[0] = (nn_real)1;
    features[1] = (nn_real)line / (nn_real)2;
    features[2] = (nn_real)state->line_chars / (nn_real)34;
    features[3] = (nn_real)state->total_chars / (nn_real)90;
    features[4] = (nn_real)target / (nn_real)7;
    features[5] = (nn_real)state->line_syllables / (nn_real)target;
    features[6] = (nn_real)(target - state->line_syllables) / (nn_real)7;
    features[7] = state->line_chars == 0 ? (nn_real)1 : (nn_real)0;
    features[8] = state->prev_char == ' ' ? (nn_real)1 : (nn_real)0;
    features[9] = state->prev_char == '\n' ? (nn_real)1 : (nn_real)0;
    features[10] = is_punctuation_char(state->prev_char) ? (nn_real)1 : (nn_real)0;
    features[11] = token == '\0' ? (nn_real)0 : (nn_real)vocab_index(token) / (nn_real)vocab_len();
    features[12] = token == '\0' ? (nn_real)1 : (nn_real)0;
    features[13] = token == '\n' ? (nn_real)1 : (nn_real)0;
    features[14] = token == ' ' ? (nn_real)1 : (nn_real)0;
    features[15] = is_vowel_char(token) ? (nn_real)1 : (nn_real)0;
    features[16] = is_letter_char(token) && !is_vowel_char(token) ? (nn_real)1 : (nn_real)0;
    features[17] = is_punctuation_char(token) ? (nn_real)1 : (nn_real)0;
    features[18] = increment ? (nn_real)1 : (nn_real)0;
    features[19] = token != '\0' && token == state->prev_char ? (nn_real)1 : (nn_real)0;
    features[20] = token == '\n' || token == '\0' ? (nn_real)1 : (nn_real)0;
    features[21] = state->line_syllables >= target - 1 ? (nn_real)1 : (nn_real)0;
    features[22] = (nn_real)next_syllables / (nn_real)(target + 2);
    features[23] = (nn_real)(HAIKU_MAX_LINE_CHARS[line] - state->line_chars) / (nn_real)34;
    features[24] = (nn_real)(2 - line) / (nn_real)2;
    features[25] = prefix_len == 0 ? (nn_real)1 : (nn_real)0;
    features[26] = state->line_chars > 0 && state->line_chars < 4 ? (nn_real)1 : (nn_real)0;
    features[27] = state->line_chars >= HAIKU_MAX_LINE_CHARS[line] - 6 ? (nn_real)1 : (nn_real)0;
    features[28] = token == '-' ? (nn_real)1 : (nn_real)0;
    features[29] = token == '\'' ? (nn_real)1 : (nn_real)0;
    features[30] = token == ',' || token == '.' || token == ';' ? (nn_real)1 : (nn_real)0;
    features[31] = (state->prev_char == 'q' && token == 'u') ? (nn_real)1 : (nn_real)0;

    for (size_t i = 0; i < HAIKU_CONTEXT; ++i) {
        nn_real value = 0;
        if (prefix_len > i) {
            value = (nn_real)(unsigned char)prefix[prefix_len - 1u - i] / (nn_real)127;
        }
        features[32 + i] = value;
    }

    size_t candidate_index = stat_index_for_token(token);
    size_t prev1 = prefix_len > 0 ? stat_index_for_prefix_char(prefix[prefix_len - 1u])
                                  : HAIKU_STAT_START;
    size_t prev2 = prefix_len > 1 ? stat_index_for_prefix_char(prefix[prefix_len - 2u])
                                  : HAIKU_STAT_START;
    double unigram = row_probability(g_stats.unigram, candidate_index);
    double bigram = row_probability(g_stats.bigram[prev1], candidate_index);
    double trigram = row_probability(g_stats.trigram[prev2][prev1], candidate_index);
    double line_start = row_probability(g_stats.line_start, candidate_index);
    double after_space = row_probability(g_stats.after_space, candidate_index);
    features[40] = (nn_real)sqrt(unigram);
    features[41] = (nn_real)sqrt(bigram);
    features[42] = (nn_real)sqrt(trigram);
    features[43] = state->line_chars == 0 ? (nn_real)sqrt(line_start) : (nn_real)0;
    features[44] = state->prev_char == ' ' ? (nn_real)sqrt(after_space) : (nn_real)0;
    features[45] = token == 'u' && state->prev_char == 'q' ? (nn_real)1 : (nn_real)0;
}

static void build_candidate_inputs(const char *prefix,
                                   size_t prefix_len,
                                   const HaikuState *state,
                                   const HaikuCandidate *candidates,
                                   size_t candidate_count,
                                   nn_real *candidate_inputs)
{
    for (size_t i = 0; i < candidate_count; ++i) {
        fill_features(prefix,
                      prefix_len,
                      state,
                      &candidates[i],
                      &candidate_inputs[i * HAIKU_FEATURE_COUNT]);
    }
}

static void strip_line(char *line)
{
    size_t len = strlen(line);
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r' ||
                       isspace((unsigned char)line[len - 1]))) {
        line[len - 1] = '\0';
        len -= 1;
    }
}

static int normalize_line(const char *line, char *out, size_t out_size)
{
    size_t n = 0;
    int last_space = 1;
    for (size_t i = 0; line[i] != '\0'; ++i) {
        unsigned char raw = (unsigned char)line[i];
        char c = (char)tolower(raw);
        if (c == '\t') {
            c = ' ';
        }
        if (c == ' ') {
            if (last_space) {
                continue;
            }
            if (n + 1 >= out_size) {
                return 0;
            }
            out[n++] = c;
            last_space = 1;
            continue;
        }
        if (vocab_index(c) < 0 || c == '\n') {
            continue;
        }
        if (n + 1 >= out_size) {
            return 0;
        }
        out[n++] = c;
        last_space = 0;
    }
    while (n > 0 && out[n - 1] == ' ') {
        n -= 1;
    }
    out[n] = '\0';
    return n > 0;
}

static size_t poem_line_count(const char *text, size_t length)
{
    if (length == 0) {
        return 0;
    }
    size_t lines = 1;
    for (size_t i = 0; i < length; ++i) {
        if (text[i] == '\n') {
            lines += 1;
        }
    }
    return lines;
}

static void corpus_add_current(HaikuCorpus *corpus, char *current, size_t *current_len)
{
    while (*current_len > 0 && current[*current_len - 1] == '\n') {
        current[*current_len - 1] = '\0';
        *current_len -= 1;
    }
    if (*current_len == 0) {
        return;
    }
    if (poem_line_count(current, *current_len) != 3) {
        corpus->ignored_stanzas += 1;
        *current_len = 0;
        current[0] = '\0';
        return;
    }
    if (corpus->count < HAIKU_MAX_POEMS) {
        HaikuPoem *poem = &corpus->poems[corpus->count++];
        memcpy(poem->text, current, *current_len + 1u);
        poem->length = *current_len;
    }
    *current_len = 0;
    current[0] = '\0';
}

static int load_corpus(const char *path, HaikuCorpus *corpus)
{
    memset(corpus, 0, sizeof(*corpus));
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "failed to open corpus %s: %s\n", path, strerror(errno));
        return 0;
    }

    char line[HAIKU_MAX_LINE];
    char normalized[HAIKU_MAX_LINE];
    char current[HAIKU_MAX_POEM_CHARS];
    size_t current_len = 0;
    current[0] = '\0';

    while (fgets(line, sizeof(line), fp) != NULL) {
        strip_line(line);
        if (!normalize_line(line, normalized, sizeof(normalized))) {
            corpus_add_current(corpus, current, &current_len);
            continue;
        }
        size_t len = strlen(normalized);
        if (current_len + len + 2u >= sizeof(current)) {
            corpus->ignored_stanzas += 1;
            current_len = 0;
            current[0] = '\0';
            continue;
        }
        if (current_len > 0) {
            current[current_len++] = '\n';
        }
        memcpy(&current[current_len], normalized, len + 1u);
        current_len += len;
    }
    corpus_add_current(corpus, current, &current_len);
    fclose(fp);
    return corpus->count > 0;
}

static int normalize_word(const char *line, char *out, size_t out_size)
{
    size_t n = 0;
    for (size_t i = 0; line[i] != '\0'; ++i) {
        unsigned char raw = (unsigned char)line[i];
        if (raw == '\n' || raw == '\r') {
            break;
        }
        if (!isalpha(raw)) {
            return 0;
        }
        if (n + 1u >= out_size) {
            return 0;
        }
        out[n++] = (char)tolower(raw);
    }
    out[n] = '\0';
    return n > 0;
}

static int compare_words(const void *a, const void *b)
{
    const char *left = *(const char *const *)a;
    const char *right = *(const char *const *)b;
    return strcmp(left, right);
}

static uint64_t hash_update(uint64_t hash, const char *bytes, size_t length)
{
    for (size_t i = 0; i < length; ++i) {
        hash ^= (uint64_t)(unsigned char)bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

static void free_lexicon(HaikuLexicon *lexicon)
{
    if (lexicon == NULL) {
        return;
    }
    for (size_t i = 0; i < lexicon->count; ++i) {
        free(lexicon->words[i]);
    }
    free(lexicon->words);
    free(lexicon->trie);
    memset(lexicon, 0, sizeof(*lexicon));
}

static int lexicon_add_trie_node(HaikuLexicon *lexicon, size_t *out)
{
    if (lexicon->trie_count == lexicon->trie_capacity) {
        size_t next_capacity = lexicon->trie_capacity == 0 ? 1024u : lexicon->trie_capacity * 2u;
        HaikuTrieNode *next =
            (HaikuTrieNode *)realloc(lexicon->trie, next_capacity * sizeof(*next));
        if (next == NULL) {
            return 0;
        }
        lexicon->trie = next;
        lexicon->trie_capacity = next_capacity;
    }
    memset(&lexicon->trie[lexicon->trie_count], 0, sizeof(lexicon->trie[lexicon->trie_count]));
    *out = lexicon->trie_count;
    lexicon->trie_count += 1;
    return 1;
}

static int lexicon_add_word(HaikuLexicon *lexicon, const char *word)
{
    if (lexicon->count == lexicon->capacity) {
        size_t next_capacity = lexicon->capacity == 0 ? 1024u : lexicon->capacity * 2u;
        char **next = (char **)realloc(lexicon->words, next_capacity * sizeof(*next));
        if (next == NULL) {
            return 0;
        }
        lexicon->words = next;
        lexicon->capacity = next_capacity;
    }
    lexicon->words[lexicon->count] = copy_string(word);
    if (lexicon->words[lexicon->count] == NULL) {
        return 0;
    }
    lexicon->count += 1;
    return 1;
}

static void lexicon_deduplicate_and_hash(HaikuLexicon *lexicon)
{
    if (lexicon->count == 0) {
        lexicon->hash = 0;
        return;
    }
    qsort(lexicon->words, lexicon->count, sizeof(*lexicon->words), compare_words);

    size_t write = 0;
    for (size_t read = 0; read < lexicon->count; ++read) {
        if (write > 0 && strcmp(lexicon->words[read], lexicon->words[write - 1u]) == 0) {
            free(lexicon->words[read]);
            continue;
        }
        lexicon->words[write++] = lexicon->words[read];
    }
    lexicon->count = write;

    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < lexicon->count; ++i) {
        hash = hash_update(hash, lexicon->words[i], strlen(lexicon->words[i]));
        hash = hash_update(hash, "\n", 1u);
    }
    lexicon->hash = hash;
}

static int lexicon_insert_trie_word(HaikuLexicon *lexicon, const char *word)
{
    size_t node = 0;
    for (size_t i = 0; word[i] != '\0'; ++i) {
        unsigned char raw = (unsigned char)word[i];
        if (raw < 'a' || raw > 'z') {
            return 0;
        }
        size_t slot = (size_t)(raw - 'a');
        if (lexicon->trie[node].child[slot] == 0) {
            size_t next = 0;
            if (!lexicon_add_trie_node(lexicon, &next)) {
                return 0;
            }
            lexicon->trie[node].child[slot] = next;
        }
        node = lexicon->trie[node].child[slot];
    }
    lexicon->trie[node].terminal = 1;
    return 1;
}

static int lexicon_build_trie(HaikuLexicon *lexicon)
{
    free(lexicon->trie);
    lexicon->trie = NULL;
    lexicon->trie_count = 0;
    lexicon->trie_capacity = 0;

    size_t root = 0;
    if (!lexicon_add_trie_node(lexicon, &root) || root != 0) {
        return 0;
    }
    for (size_t i = 0; i < lexicon->count; ++i) {
        if (!lexicon_insert_trie_word(lexicon, lexicon->words[i])) {
            return 0;
        }
    }
    return 1;
}

static size_t lexicon_compute_min_suffix(HaikuLexicon *lexicon, size_t node)
{
    size_t best = lexicon->trie[node].terminal ? 0u : HAIKU_MAX_WORD;
    for (size_t i = 0; i < 26; ++i) {
        size_t child = lexicon->trie[node].child[i];
        if (child == 0) {
            continue;
        }
        size_t suffix = lexicon_compute_min_suffix(lexicon, child);
        if (suffix + 1u < best) {
            best = suffix + 1u;
        }
    }
    lexicon->trie[node].min_suffix = best;
    return best;
}

static int load_lexicon(const char *path, HaikuLexicon *lexicon)
{
    memset(lexicon, 0, sizeof(*lexicon));
    lexicon->path = path;
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "failed to open word list %s: %s\n", path, strerror(errno));
        return 0;
    }

    char line[HAIKU_MAX_LINE];
    char word[HAIKU_MAX_WORD];
    while (fgets(line, sizeof(line), fp) != NULL) {
        strip_line(line);
        if (!normalize_word(line, word, sizeof(word))) {
            continue;
        }
        if (!lexicon_add_word(lexicon, word)) {
            fclose(fp);
            free_lexicon(lexicon);
            return 0;
        }
    }
    fclose(fp);

    lexicon_deduplicate_and_hash(lexicon);
    if (!lexicon_build_trie(lexicon)) {
        free_lexicon(lexicon);
        return 0;
    }
    (void)lexicon_compute_min_suffix(lexicon, 0);
    return lexicon->count > 0;
}

static int lexicon_contains(const HaikuLexicon *lexicon, const char *word)
{
    size_t low = 0;
    size_t high = lexicon->count;
    while (low < high) {
        size_t mid = low + (high - low) / 2u;
        int cmp = strcmp(word, lexicon->words[mid]);
        if (cmp == 0) {
            return 1;
        }
        if (cmp < 0) {
            high = mid;
        } else {
            low = mid + 1u;
        }
    }
    return 0;
}

static int lexicon_follow_prefix(const HaikuLexicon *lexicon,
                                 const char *word,
                                 size_t length,
                                 size_t *out_node)
{
    size_t node = 0;
    for (size_t i = 0; i < length; ++i) {
        unsigned char raw = (unsigned char)word[i];
        if (raw < 'a' || raw > 'z') {
            return 0;
        }
        size_t next = lexicon->trie[node].child[(size_t)(raw - 'a')];
        if (next == 0) {
            return 0;
        }
        node = next;
    }
    *out_node = node;
    return 1;
}

static int current_word_prefix(const char *prefix,
                               size_t prefix_len,
                               char *word,
                               size_t word_size,
                               size_t *out_len)
{
    size_t start = prefix_len;
    while (start > 0 && isalpha((unsigned char)prefix[start - 1u])) {
        start -= 1;
    }
    size_t length = prefix_len - start;
    if (length + 1u > word_size) {
        return 0;
    }
    for (size_t i = 0; i < length; ++i) {
        word[i] = (char)tolower((unsigned char)prefix[start + i]);
    }
    word[length] = '\0';
    *out_len = length;
    return 1;
}

static HaikuLexiconDecision lexicon_candidate_decision(const char *prefix,
                                                       size_t prefix_len,
                                                       const HaikuState *state,
                                                       const HaikuLexicon *lexicon,
                                                       char token)
{
    char word[HAIKU_MAX_WORD];
    size_t word_len = 0;
    if (!current_word_prefix(prefix, prefix_len, word, sizeof(word), &word_len)) {
        return HAIKU_LEXICON_REJECT_PREFIX;
    }

    if (is_letter_char(token)) {
        size_t node = 0;
        if (!lexicon_follow_prefix(lexicon, word, word_len, &node)) {
            return HAIKU_LEXICON_REJECT_PREFIX;
        }
        size_t child = lexicon->trie[node].child[(size_t)(token - 'a')];
        if (child == 0) {
            return HAIKU_LEXICON_REJECT_PREFIX;
        }
        int line = state->line <= 2 ? state->line : 2;
        int remaining = HAIKU_MAX_LINE_CHARS[line] - state->line_chars - 1;
        if (remaining < 0 || lexicon->trie[child].min_suffix > (size_t)remaining) {
            return HAIKU_LEXICON_REJECT_COMPLETION;
        }
        return HAIKU_LEXICON_OK;
    }

    if (word_len == 0) {
        return HAIKU_LEXICON_OK;
    }

    size_t node = 0;
    if (!lexicon_follow_prefix(lexicon, word, word_len, &node)) {
        return HAIKU_LEXICON_REJECT_BOUNDARY;
    }
    return lexicon->trie[node].terminal ? HAIKU_LEXICON_OK : HAIKU_LEXICON_REJECT_BOUNDARY;
}

static void finish_lexical_word(const HaikuLexicon *lexicon,
                                const char *word,
                                size_t length,
                                int overflow,
                                HaikuLexicalScore *score)
{
    if (length == 0) {
        return;
    }
    score->words += 1;
    if (!overflow && lexicon_contains(lexicon, word)) {
        score->dictionary_words += 1;
    }
}

static HaikuLexicalScore score_lexicon_v1(const char *generated, const HaikuLexicon *lexicon)
{
    HaikuLexicalScore score;
    memset(&score, 0, sizeof(score));

    char word[HAIKU_MAX_WORD];
    size_t length = 0;
    int overflow = 0;
    for (size_t i = 0;; ++i) {
        unsigned char raw = (unsigned char)generated[i];
        if (isalpha(raw)) {
            if (length + 1u < sizeof(word)) {
                word[length++] = (char)tolower(raw);
                word[length] = '\0';
            } else {
                overflow = 1;
            }
            continue;
        }
        finish_lexical_word(lexicon, word, length, overflow, &score);
        length = 0;
        overflow = 0;
        word[0] = '\0';
        if (raw == '\0') {
            break;
        }
    }
    return score;
}

static char target_at(const HaikuPoem *poem, size_t position)
{
    return position < poem->length ? poem->text[position] : '\0';
}

static size_t train_split_count(const HaikuCorpus *corpus)
{
    if (corpus->count <= 1) {
        return corpus->count;
    }
    size_t train = (corpus->count * 4u) / 5u;
    return train == 0 ? 1u : train;
}

static nn_real train_one_position(NeuralNet *net,
                                  const HaikuPoem *poem,
                                  size_t position,
                                  const HaikuLexicon *lexicon,
                                  int lexicon_gate,
                                  double learning_rate,
                                  int *trained)
{
    HaikuState state = state_from_prefix(poem->text, position);
    HaikuCandidate candidates[HAIKU_MAX_CANDIDATES];
    nn_real candidate_inputs[HAIKU_MAX_CANDIDATES * HAIKU_FEATURE_COUNT];
    size_t candidate_count = build_candidates(poem->text,
                                              position,
                                              &state,
                                              lexicon,
                                              lexicon_gate,
                                              NULL,
                                              candidates);
    size_t target = 0;
    *trained = 0;
    if (candidate_count == 0) {
        return (nn_real)0;
    }
    if (!target_index(candidates, candidate_count, target_at(poem, position), &target)) {
        return (nn_real)0;
    }
    build_candidate_inputs(poem->text, position, &state, candidates, candidate_count, candidate_inputs);
    *trained = 1;
    return nn_train_softmax_scorer(net,
                                   candidate_inputs,
                                   candidate_count,
                                   target,
                                   (nn_real)learning_rate);
}

static HaikuEval evaluate_split(NeuralNet *net,
                                const HaikuCorpus *corpus,
                                const HaikuLexicon *lexicon,
                                int lexicon_gate,
                                size_t start,
                                size_t end)
{
    HaikuEval eval;
    memset(&eval, 0, sizeof(eval));

    for (size_t p = start; p < end; ++p) {
        const HaikuPoem *poem = &corpus->poems[p];
        for (size_t pos = 0; pos <= poem->length; ++pos) {
            HaikuState state = state_from_prefix(poem->text, pos);
            HaikuCandidate candidates[HAIKU_MAX_CANDIDATES];
            nn_real candidate_inputs[HAIKU_MAX_CANDIDATES * HAIKU_FEATURE_COUNT];
            nn_real scores[HAIKU_MAX_CANDIDATES];
            nn_real probabilities[HAIKU_MAX_CANDIDATES];
            size_t ranked[HAIKU_MAX_CANDIDATES];
            size_t candidate_count = build_candidates(poem->text,
                                                      pos,
                                                      &state,
                                                      lexicon,
                                                      lexicon_gate,
                                                      NULL,
                                                      candidates);
            size_t target = 0;
            if (candidate_count == 0) {
                eval.skipped += 1;
                continue;
            }
            if (!target_index(candidates, candidate_count, target_at(poem, pos), &target)) {
                eval.skipped += 1;
                continue;
            }
            build_candidate_inputs(poem->text, pos, &state, candidates, candidate_count, candidate_inputs);
            uint32_t action_ids[HAIKU_MAX_CANDIDATES];
            for (size_t i = 0; i < candidate_count; ++i) {
                action_ids[i] = candidates[i].action_id;
            }
            if (!nn_score_action_rows(net, candidate_inputs, candidate_count, scores, probabilities) ||
                !nn_rank_action_scores_with_ids(scores,
                                                action_ids,
                                                candidate_count,
                                                ranked,
                                                NN_RANK_SCORE_EPSILON)) {
                continue;
            }
            double probability = (double)probabilities[target];
            if (probability < 1e-12) {
                probability = 1e-12;
            }
            eval.nll += -log(probability);
            eval.top1 += ranked[0] == target ? 1u : 0u;
            eval.rows += 1;
        }
    }
    return eval;
}

static size_t choose_candidate(const nn_real *probabilities,
                               const size_t *ranked,
                               size_t candidate_count,
                               const HaikuConfig *config,
                               uint32_t *rng_state)
{
    if (!config->sample) {
        return ranked[0];
    }
    size_t top_k = config->top_k;
    if (top_k == 0 || top_k > candidate_count) {
        top_k = candidate_count;
    }
    double weights[HAIKU_MAX_CANDIDATES];
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

static void print_generated(const char *generated)
{
    puts("generated:");
    puts("---");
    puts(generated);
    puts("---");
}

static int trace_write_token(FILE *fp, char token)
{
    if (token == '\0') {
        return fputs("\"<eos>\"", fp) != EOF;
    }
    if (token == '\n') {
        return fputs("\"\\n\"", fp) != EOF;
    }
    if (token == '"' || token == '\\') {
        return fprintf(fp, "\"\\%c\"", token) >= 0;
    }
    return fprintf(fp, "\"%c\"", token) >= 0;
}

static int write_haiku_generation_trace(FILE *fp,
                                        const char *model_hash,
                                        const char *prefix,
                                        size_t prefix_len,
                                        const HaikuState *state,
                                        const HaikuCandidate *candidates,
                                        const HaikuRejectStats *rejects,
                                        const uint32_t *action_ids,
                                        const nn_real *scores,
                                        const nn_real *probabilities,
                                        size_t candidate_count,
                                        size_t selected,
                                        size_t top_ranked,
                                        size_t sample_index,
                                        size_t step,
                                        uint32_t seed,
                                        const HaikuConfig *config)
{
    if (fprintf(fp,
                "{\"schema\":\"crlp.haiku_lm.generation_trace.v1\","
                "\"task\":\"haiku_lm\",\"state_hash\":\"%016llx\","
                "\"model_hash\":\"%s\",\"feature_encoder_version\":%u,"
                "\"sample\":%zu,\"step\":%zu,\"line\":%d,\"line_syllables\":%d,"
                "\"candidate_count\":%zu,\"selected_index\":%zu,\"top_ranked_index\":%zu,"
                "\"selected_action_id\":%u,\"top_ranked_action_id\":%u,"
                "\"sampling\":\"%s\",\"top_k\":%zu,\"temperature\":%.6g,\"seed\":%u,"
                "\"rejected_candidates\":{\"shape\":%zu,\"lexicon_prefix\":%zu,"
                "\"lexicon_boundary\":%zu,\"lexicon_completion\":%zu,\"dead_end\":%zu},"
                "\"action_ids\":[",
                (unsigned long long)hash_prefix(prefix, prefix_len),
                model_hash,
                (unsigned int)HAIKU_FEATURE_ENCODER_VERSION,
                sample_index,
                step,
                state->line,
                state->line_syllables,
                candidate_count,
                selected,
                top_ranked,
                action_ids[selected],
                action_ids[top_ranked],
                config->sample ? "top_k_sample" : "top1",
                config->top_k,
                config->temperature,
                seed,
                rejects != NULL ? rejects->shape : 0,
                rejects != NULL ? rejects->lexicon_prefix : 0,
                rejects != NULL ? rejects->lexicon_boundary : 0,
                rejects != NULL ? rejects->lexicon_completion : 0,
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
        if (!trace_write_token(fp, candidates[i].token)) {
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
    if (fputs("],\"verifier_result\":\"", fp) == EOF ||
        fputs(candidates[selected].token == '\0' ? "eos" : "accepted", fp) == EOF ||
        fprintf(fp,
                "\",\"verifier_cost\":{\"gates\":%s}}\n",
                config->lexicon_gate ? "[\"haiku_shape_proxy\",\"lexicon_trie_v1\"]"
                                     : "[\"haiku_shape_proxy\"]") < 0) {
        return 0;
    }
    return 1;
}

static int generate_one(NeuralNet *net,
                        const HaikuConfig *config,
                        const char *model_hash,
                        FILE *trace,
                        const HaikuLexicon *lexicon,
                        HaikuLexicalAggregate *lexical,
                        HaikuRejectStats *generation_rejects,
                        size_t sample_index)
{
    char generated[HAIKU_MAX_GENERATED];
    size_t length = 0;
    generated[0] = '\0';
    uint32_t rng_state = config->seed + (uint32_t)(sample_index * 7919u) + 17u;

    for (size_t step = 0; step + 1u < sizeof(generated); ++step) {
        HaikuState state = state_from_prefix(generated, length);
        HaikuCandidate candidates[HAIKU_MAX_CANDIDATES];
        nn_real candidate_inputs[HAIKU_MAX_CANDIDATES * HAIKU_FEATURE_COUNT];
        nn_real scores[HAIKU_MAX_CANDIDATES];
        nn_real probabilities[HAIKU_MAX_CANDIDATES];
        size_t ranked[HAIKU_MAX_CANDIDATES];
        uint32_t action_ids[HAIKU_MAX_CANDIDATES];
        HaikuRejectStats rejects;
        memset(&rejects, 0, sizeof(rejects));
        size_t candidate_count = build_candidates(generated,
                                                  length,
                                                  &state,
                                                  lexicon,
                                                  config->lexicon_gate,
                                                  &rejects,
                                                  candidates);
        add_reject_stats(generation_rejects, &rejects);
        if (candidate_count == 0) {
            break;
        }
        for (size_t i = 0; i < candidate_count; ++i) {
            action_ids[i] = candidates[i].action_id;
        }
        build_candidate_inputs(generated, length, &state, candidates, candidate_count, candidate_inputs);
        if (!nn_score_action_rows(net, candidate_inputs, candidate_count, scores, probabilities) ||
            !nn_rank_action_scores_with_ids(scores,
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
        if (trace != NULL) {
            if (!write_haiku_generation_trace(trace,
                                              model_hash,
                                              generated,
                                              length,
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
        }
        char token = candidates[selected].token;
        if (token == '\0') {
            generated[length] = '\0';
            break;
        }
        generated[length++] = token;
        generated[length] = '\0';
    }
    print_generated(generated);
    if (lexicon != NULL && lexicon->count > 0) {
        HaikuLexicalScore score = score_lexicon_v1(generated, lexicon);
        double ratio = score.words > 0
                           ? (double)score.dictionary_words / (double)score.words
                           : 0.0;
        if (lexical != NULL) {
            lexical->samples += 1;
            lexical->words += score.words;
            lexical->dictionary_words += score.dictionary_words;
        }
        printf("lexicon_v1 sample=%zu words=%zu dictionary_words=%zu dictionary_ratio=%.2f%% "
               "lexicon_entries=%zu lexicon_hash=%016llx\n",
               sample_index,
               score.words,
               score.dictionary_words,
               ratio * 100.0,
               lexicon->count,
               (unsigned long long)lexicon->hash);
    }
    return 1;
}

static void usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s [--corpus path] [--epochs N] [--max-updates N] [--lr R]\n"
            "       [--samples N] [--sample 0|1] [--top-k N] [--temperature R]\n"
            "       [--trace path] [--save path] [--word-list path]\n"
            "       [--lexicon-gate 0|1] [--seed N]\n",
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

static int parse_config(int argc, char **argv, HaikuConfig *config)
{
    config->corpus_path = "data/haiku_tiny.txt";
    config->trace_path = NULL;
    config->save_path = NULL;
    config->word_list_path = NULL;
    config->epochs = 8;
    config->max_updates = 0;
    config->samples = 2;
    config->top_k = 5;
    config->learning_rate = 0.006;
    config->temperature = 0.85;
    config->seed = 20240518u;
    config->sample = 1;
    config->lexicon_gate = 1;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--corpus") == 0 && i + 1 < argc) {
            config->corpus_path = argv[++i];
        } else if (strcmp(argv[i], "--trace") == 0 && i + 1 < argc) {
            config->trace_path = argv[++i];
        } else if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            config->save_path = argv[++i];
        } else if (strcmp(argv[i], "--word-list") == 0 && i + 1 < argc) {
            config->word_list_path = argv[++i];
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            if (!parse_size(argv[++i], &config->epochs)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--max-updates") == 0 && i + 1 < argc) {
            if (!parse_size(argv[++i], &config->max_updates)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
            if (!parse_size(argv[++i], &config->samples)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            if (!parse_size(argv[++i], &config->top_k)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            if (!parse_double_arg(argv[++i], &config->learning_rate)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            if (!parse_double_arg(argv[++i], &config->temperature)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--sample") == 0 && i + 1 < argc) {
            size_t value = 0;
            if (!parse_size(argv[++i], &value)) {
                return 0;
            }
            config->sample = value != 0;
        } else if (strcmp(argv[i], "--lexicon-gate") == 0 && i + 1 < argc) {
            size_t value = 0;
            if (!parse_size(argv[++i], &value)) {
                return 0;
            }
            config->lexicon_gate = value != 0;
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
    return config->epochs > 0 && config->samples > 0 && config->learning_rate > 0.0;
}

int main(int argc, char **argv)
{
    HaikuConfig config;
    if (!parse_config(argc, argv, &config)) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    HaikuCorpus corpus;
    if (!load_corpus(config.corpus_path, &corpus)) {
        return EXIT_FAILURE;
    }

    HaikuLexicon lexicon;
    memset(&lexicon, 0, sizeof(lexicon));
    if (config.word_list_path != NULL) {
        if (!load_lexicon(config.word_list_path, &lexicon)) {
            return EXIT_FAILURE;
        }
        printf("lexicon_v1 word_list=%s entries=%zu trie_nodes=%zu hash=%016llx gate=%d\n",
               config.word_list_path,
               lexicon.count,
               lexicon.trie_count,
               (unsigned long long)lexicon.hash,
               config.lexicon_gate);
    }

    const size_t layers[] = {HAIKU_FEATURE_COUNT, 64, 32, 1};
    NNConfig nn_config;
    nn_config.layer_sizes = layers;
    nn_config.layer_count = sizeof(layers) / sizeof(layers[0]);
    nn_config.seed = config.seed;
    nn_config.hidden_activation = NN_ACTIVATION_RELU;
    nn_config.output_activation = NN_ACTIVATION_LINEAR;
    nn_config.loss = NN_LOSS_MSE;
    NeuralNet *net = nn_create_config(&nn_config);
    if (net == NULL) {
        fputs("failed to create haiku_lm network\n", stderr);
        free_lexicon(&lexicon);
        return EXIT_FAILURE;
    }

    size_t train_count = train_split_count(&corpus);
    build_stats(&corpus, train_count);
    size_t updates = 0;
    size_t skipped = 0;
    double training_seconds = 0.0;
    printf("haiku_lm corpus=%s poems=%zu train=%zu valid=%zu ignored=%zu vocab=%zu features=%d parameters=%zu\n",
           config.corpus_path,
           corpus.count,
           train_count,
           corpus.count - train_count,
           corpus.ignored_stanzas,
           vocab_len() + 1u,
           HAIKU_FEATURE_COUNT,
           nn_parameter_count(net));

    for (size_t epoch = 1; epoch <= config.epochs; ++epoch) {
        double started = now_seconds();
        double loss = 0.0;
        size_t epoch_updates = 0;
        size_t epoch_skipped = 0;
        for (size_t p = 0; p < train_count; ++p) {
            const HaikuPoem *poem = &corpus.poems[p];
            for (size_t pos = 0; pos <= poem->length; ++pos) {
                int trained = 0;
                nn_real step_loss = train_one_position(net,
                                                       poem,
                                                       pos,
                                                       lexicon.count > 0 ? &lexicon : NULL,
                                                       config.lexicon_gate,
                                                       config.learning_rate,
                                                       &trained);
                if (trained) {
                    loss += (double)step_loss;
                    epoch_updates += 1;
                    updates += 1;
                } else {
                    epoch_skipped += 1;
                    skipped += 1;
                }
                if (config.max_updates != 0 && updates >= config.max_updates) {
                    break;
                }
            }
            if (config.max_updates != 0 && updates >= config.max_updates) {
                break;
            }
        }
        double elapsed = now_seconds() - started;
        training_seconds += elapsed;
        double mean_loss = epoch_updates > 0 ? loss / (double)epoch_updates : 0.0;
        printf("epoch %zu loss=%.4f updates=%zu skipped=%zu %.0f rows/s\n",
               epoch,
               mean_loss,
               epoch_updates,
               epoch_skipped,
               elapsed > 0.0 ? (double)epoch_updates / elapsed : 0.0);
        if (config.max_updates != 0 && updates >= config.max_updates) {
            break;
        }
    }

    HaikuEval train_eval = evaluate_split(net,
                                          &corpus,
                                          lexicon.count > 0 ? &lexicon : NULL,
                                          config.lexicon_gate,
                                          0,
                                          train_count);
    HaikuEval valid_eval = evaluate_split(net,
                                          &corpus,
                                          lexicon.count > 0 ? &lexicon : NULL,
                                          config.lexicon_gate,
                                          train_count,
                                          corpus.count);
    double train_acc = train_eval.rows > 0 ? (double)train_eval.top1 / (double)train_eval.rows : 0.0;
    double valid_acc = valid_eval.rows > 0 ? (double)valid_eval.top1 / (double)valid_eval.rows : 0.0;
    double train_nll = train_eval.rows > 0 ? train_eval.nll / (double)train_eval.rows : 0.0;
    double valid_nll = valid_eval.rows > 0 ? valid_eval.nll / (double)valid_eval.rows : 0.0;

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
        memset(&meta, 0, sizeof(meta));
        meta.checkpoint_version = 1u;
        meta.feature_encoder_version = HAIKU_FEATURE_ENCODER_VERSION;
        snprintf(meta.notes, sizeof(meta.notes), "haiku_lm char scorer");
        snprintf(meta.feature_set, sizeof(meta.feature_set), "haiku_char_v1");
        if (!nn_save_checkpoint(net, config.save_path, &meta) ||
            !nn_hash_checkpoint_hex(config.save_path, model_hash)) {
            fprintf(stderr, "failed to save checkpoint: %s\n", config.save_path);
            nn_free(net);
            free_lexicon(&lexicon);
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
            free_lexicon(&lexicon);
            return EXIT_FAILURE;
        }
    }

    HaikuLexicalAggregate lexical;
    memset(&lexical, 0, sizeof(lexical));
    HaikuRejectStats generation_rejects;
    memset(&generation_rejects, 0, sizeof(generation_rejects));
    for (size_t i = 0; i < config.samples; ++i) {
        if (!generate_one(net,
                          &config,
                          model_hash,
                          trace,
                          lexicon.count > 0 ? &lexicon : NULL,
                          &lexical,
                          &generation_rejects,
                          i)) {
            if (trace != NULL) {
                fclose(trace);
            }
            nn_free(net);
            free_lexicon(&lexicon);
            return EXIT_FAILURE;
        }
    }
    if (trace != NULL) {
        fclose(trace);
        printf("wrote trace: %s\n", config.trace_path);
    }

    double lexical_ratio = lexical.words > 0
                               ? (double)lexical.dictionary_words / (double)lexical.words
                               : 0.0;
    printf("{\"schema\":\"crlp.haiku_lm.summary.v1\",\"poems\":%zu,\"train_rows\":%zu,"
           "\"valid_rows\":%zu,\"train_top1\":%.6f,\"valid_top1\":%.6f,"
           "\"valid_nll\":%.6f,\"updates\":%zu,\"verifier_skipped\":%zu,"
           "\"model_hash\":\"%s\",\"lexicon_enabled\":%d,\"lexicon_gate_enabled\":%d,"
           "\"lexicon_entries\":%zu,\"lexicon_trie_nodes\":%zu,"
           "\"lexicon_hash\":\"%016llx\",\"generated_words\":%zu,"
           "\"generated_dictionary_words\":%zu,\"generated_dictionary_ratio\":%.6f,"
           "\"generated_reject_shape\":%zu,\"generated_reject_lexicon_prefix\":%zu,"
           "\"generated_reject_lexicon_boundary\":%zu,"
           "\"generated_reject_lexicon_completion\":%zu,"
           "\"generated_reject_dead_end\":%zu}\n",
           corpus.count,
           train_eval.rows,
           valid_eval.rows,
           train_acc,
           valid_acc,
           valid_nll,
           updates,
           skipped + train_eval.skipped + valid_eval.skipped,
           model_hash,
           lexicon.count > 0 ? 1 : 0,
           lexicon.count > 0 && config.lexicon_gate ? 1 : 0,
           lexicon.count,
           lexicon.trie_count,
           (unsigned long long)lexicon.hash,
           lexical.words,
           lexical.dictionary_words,
           lexical_ratio,
           generation_rejects.shape,
           generation_rejects.lexicon_prefix,
           generation_rejects.lexicon_boundary,
           generation_rejects.lexicon_completion,
           generation_rejects.dead_end);

    nn_free(net);
    free_lexicon(&lexicon);
    return EXIT_SUCCESS;
}
