#include "nn.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    HR_MAX_POEMS = 10000,
    HR_MAX_LINE_WORDS = 32,
    HR_MAX_WORD_LEN = 64,
    HR_MAX_VOCAB = 30000,
    HR_MAX_ADJ_EDGES = 400000,
    HR_ADJ_BUCKETS = 524288,
    HR_DEFAULT_MAX_CANDIDATES = 96,
    HR_DEFAULT_MAX_RULES = 32,
    HR_DEFAULT_MAX_STATES = 2000,
    HR_RULE_FEATURES = 12,
    HR_ACTION_NEWLINE = 1,
    HR_ACTION_EOS = 2,
    HR_ADJ_START = -1,
    HR_ADJ_BOUNDARY = -2,
};

static const int HR_TARGET_SYLLABLES[3] = {5, 7, 5};
static const int HR_MAX_LINE_CHARS[3] = {28, 34, 28};

typedef enum {
    HR_CAND_WORD = 0,
    HR_CAND_NEWLINE = 1,
    HR_CAND_EOS = 2,
} HRCandidateKind;

typedef enum {
    HR_RULE_SYLLABLE_OVERFLOW = 0,
    HR_RULE_LINE_CHAR_OVERFLOW = 1,
    HR_RULE_EARLY_NEWLINE = 2,
    HR_RULE_EOS_BEFORE_COMPLETE = 3,
    HR_RULE_ADJACENCY_MISSING = 4,
    HR_RULE_UNSEEN_LINE_WORD = 5,
    HR_RULE_KIND_COUNT = 6,
} HRRuleKind;

typedef enum {
    HR_PROPOSER_GRAMMAR = 0,
    HR_PROPOSER_STRUCTURAL = 1,
    HR_PROPOSER_RANDOM = 2,
    HR_PROPOSER_CLOSED_LOOP = 3,
} HRProposer;

static const NNFeatureSchema HR_RULE_FEATURE_SCHEMA[HR_RULE_FEATURES] = {
    {"rule_kind", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"train_rejects_scaled", NN_FEATURE_DERIVED_FROM_GATE, 1},
    {"validation_rejects_scaled", NN_FEATURE_DERIVED_FROM_GATE, 1},
    {"test_rejects_scaled", NN_FEATURE_DERIVED_FROM_GATE, 1},
    {"residual_gain_scaled", NN_FEATURE_DERIVED_FROM_GATE, 1},
    {"accepted_count_scaled", NN_FEATURE_DERIVED_FROM_GATE, 1},
    {"is_adjacency_rule", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"is_syllable_rule", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"is_line_char_rule", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"is_newline_rule", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"is_eos_rule", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"covered_fraction", NN_FEATURE_DERIVED_FROM_GATE, 1},
};

typedef struct {
    char word[HR_MAX_WORD_LEN];
    int syllables;
    int length;
    size_t frequency;
    size_t line_frequency[3];
} HRVocabEntry;

typedef struct {
    HRVocabEntry entries[HR_MAX_VOCAB];
    int ranked_ids[HR_MAX_VOCAB];
    size_t count;
} HRVocab;

typedef struct {
    int words[3][HR_MAX_LINE_WORDS];
    size_t line_counts[3];
} HRPoem;

typedef struct {
    HRPoem poems[HR_MAX_POEMS];
    size_t count;
    size_t ignored_stanzas;
    size_t ignored_words;
} HRCorpus;

typedef struct {
    int line;
    int prev_word_id;
    int next_word_id;
    int hash_next;
    size_t frequency;
} HRAdjEdge;

typedef struct {
    HRAdjEdge edges[HR_MAX_ADJ_EDGES];
    int buckets[HR_ADJ_BUCKETS];
    size_t count;
    size_t overflow;
    uint64_t hash;
} HRAdjacency;

typedef struct {
    int line;
    int line_syllables;
    int line_chars;
    int line_words;
    int poem_words;
    int prev_word_id;
    uint64_t state_hash;
} HRState;

typedef struct {
    HRCandidateKind kind;
    int word_id;
} HRCandidate;

typedef struct {
    HRState state;
    HRCandidate candidate;
    int target;
} HREvent;

typedef struct {
    HRRuleKind kind;
    int param;
    NNRule nn_rule;
    char text[128];
} HRRule;

typedef struct {
    const char *corpus_path;
    const char *trace_path;
    const char *save_path;
    size_t max_candidates;
    size_t max_rules;
    size_t max_states;
    HRProposer proposer;
    uint32_t proposer_seed;
} HRConfig;

typedef struct {
    const HRRule *rules;
    size_t rule_count;
} HRClosedLoopContext;

static uint64_t mix_u64(uint64_t hash, uint64_t value)
{
    hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    return hash;
}

static uint32_t rng_next(uint32_t *state)
{
    uint32_t x = *state == 0 ? 1u : *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
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

static int vocab_find(const HRVocab *vocab, const char *word)
{
    for (size_t i = 0; i < vocab->count; ++i) {
        if (strcmp(vocab->entries[i].word, word) == 0) {
            return (int)i;
        }
    }
    return -1;
}

static int vocab_add(HRVocab *vocab, const char *word)
{
    int existing = vocab_find(vocab, word);
    if (existing >= 0) {
        return existing;
    }
    if (vocab->count >= HR_MAX_VOCAB) {
        return -1;
    }
    HRVocabEntry *entry = &vocab->entries[vocab->count];
    memset(entry, 0, sizeof(*entry));
    snprintf(entry->word, sizeof(entry->word), "%s", word);
    entry->length = (int)strlen(word);
    entry->syllables = syllable_proxy(word);
    vocab->ranked_ids[vocab->count] = (int)vocab->count;
    vocab->count += 1;
    return (int)(vocab->count - 1u);
}

static int parse_line_words(char *line, HRVocab *vocab, int *out_words, size_t *out_count)
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
        char word[HR_MAX_WORD_LEN];
        if (!normalize_word(start, word, sizeof(word))) {
            *cursor = saved;
            return 0;
        }
        int id = vocab_add(vocab, word);
        if (id < 0 || *out_count >= HR_MAX_LINE_WORDS) {
            *cursor = saved;
            return 0;
        }
        out_words[(*out_count)++] = id;
        *cursor = saved;
    }
    return *out_count > 0;
}

static void update_vocab_stats(HRVocab *vocab, const HRPoem *poem)
{
    for (int line = 0; line < 3; ++line) {
        for (size_t i = 0; i < poem->line_counts[line]; ++i) {
            int id = poem->words[line][i];
            vocab->entries[id].frequency += 1;
            vocab->entries[id].line_frequency[line] += 1;
        }
    }
}

static int load_corpus(const char *path, HRVocab *vocab, HRCorpus *corpus)
{
    memset(vocab, 0, sizeof(*vocab));
    memset(corpus, 0, sizeof(*corpus));
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "failed to open corpus %s: %s\n", path, strerror(errno));
        return 0;
    }

    char raw[3][512];
    size_t pending = 0;
    char line[512];
    while (fgets(line, sizeof(line), fp) != NULL) {
        strip_line(line);
        if (line[0] == '\0') {
            if (pending != 0) {
                corpus->ignored_stanzas += 1;
                pending = 0;
            }
            continue;
        }
        if (pending < 3) {
            snprintf(raw[pending], sizeof(raw[pending]), "%s", line);
        }
        pending += 1;
        if (pending == 3) {
            if (corpus->count >= HR_MAX_POEMS) {
                break;
            }
            HRPoem poem;
            memset(&poem, 0, sizeof(poem));
            int ok = 1;
            for (int i = 0; i < 3; ++i) {
                char copy[512];
                snprintf(copy, sizeof(copy), "%s", raw[i]);
                ok = ok && parse_line_words(copy, vocab, poem.words[i], &poem.line_counts[i]);
            }
            if (ok) {
                corpus->poems[corpus->count++] = poem;
                update_vocab_stats(vocab, &poem);
            } else {
                corpus->ignored_words += 1;
            }
            pending = 0;
        } else if (pending > 3) {
            corpus->ignored_stanzas += 1;
            pending = 0;
        }
    }
    if (pending != 0) {
        corpus->ignored_stanzas += 1;
    }
    fclose(fp);
    return corpus->count > 0 && vocab->count > 0;
}

static const HRVocab *g_sort_vocab;

static int compare_ranked_vocab(const void *lhs, const void *rhs)
{
    int a = *(const int *)lhs;
    int b = *(const int *)rhs;
    size_t fa = g_sort_vocab->entries[a].frequency;
    size_t fb = g_sort_vocab->entries[b].frequency;
    if (fa != fb) {
        return fa > fb ? -1 : 1;
    }
    return strcmp(g_sort_vocab->entries[a].word, g_sort_vocab->entries[b].word);
}

static void sort_vocab(HRVocab *vocab)
{
    g_sort_vocab = vocab;
    qsort(vocab->ranked_ids, vocab->count, sizeof(vocab->ranked_ids[0]), compare_ranked_vocab);
    g_sort_vocab = NULL;
}

static uint32_t adj_hash_key(int line, int prev_word_id, int next_word_id)
{
    uint32_t h = 2166136261u;
    h = (h ^ (uint32_t)(line + 17)) * 16777619u;
    h = (h ^ (uint32_t)(prev_word_id + 65537)) * 16777619u;
    h = (h ^ (uint32_t)(next_word_id + 131071)) * 16777619u;
    return h;
}

static void adjacency_add(HRAdjacency *adjacency, int line, int prev_word_id, int next_word_id)
{
    uint32_t bucket = adj_hash_key(line, prev_word_id, next_word_id) % HR_ADJ_BUCKETS;
    for (int e = adjacency->buckets[bucket]; e >= 0; e = adjacency->edges[e].hash_next) {
        HRAdjEdge *edge = &adjacency->edges[e];
        if (edge->line == line && edge->prev_word_id == prev_word_id &&
            edge->next_word_id == next_word_id) {
            edge->frequency += 1;
            return;
        }
    }
    if (adjacency->count >= HR_MAX_ADJ_EDGES) {
        adjacency->overflow += 1;
        return;
    }
    HRAdjEdge *edge = &adjacency->edges[adjacency->count];
    edge->line = line;
    edge->prev_word_id = prev_word_id;
    edge->next_word_id = next_word_id;
    edge->frequency = 1;
    edge->hash_next = adjacency->buckets[bucket];
    adjacency->buckets[bucket] = (int)adjacency->count;
    adjacency->count += 1;
    adjacency->hash = mix_u64(adjacency->hash, adj_hash_key(line, prev_word_id, next_word_id));
}

static void build_adjacency(const HRCorpus *corpus, size_t poem_begin, size_t poem_end, HRAdjacency *adjacency)
{
    for (size_t i = 0; i < HR_ADJ_BUCKETS; ++i) {
        adjacency->buckets[i] = -1;
    }
    adjacency->count = 0;
    adjacency->overflow = 0;
    adjacency->hash = 1469598103934665603ULL;
    if (poem_end > corpus->count) {
        poem_end = corpus->count;
    }
    for (size_t p = poem_begin; p < poem_end; ++p) {
        const HRPoem *poem = &corpus->poems[p];
        for (int line = 0; line < 3; ++line) {
            int prev = line == 0 ? HR_ADJ_START : HR_ADJ_BOUNDARY;
            for (size_t w = 0; w < poem->line_counts[line]; ++w) {
                int next = poem->words[line][w];
                adjacency_add(adjacency, line, prev, next);
                prev = next;
            }
        }
    }
}

static int adjacency_has(const HRAdjacency *adjacency, int line, int prev_word_id, int next_word_id)
{
    uint32_t bucket = adj_hash_key(line, prev_word_id, next_word_id) % HR_ADJ_BUCKETS;
    for (int e = adjacency->buckets[bucket]; e >= 0; e = adjacency->edges[e].hash_next) {
        const HRAdjEdge *edge = &adjacency->edges[e];
        if (edge->line == line && edge->prev_word_id == prev_word_id &&
            edge->next_word_id == next_word_id) {
            return 1;
        }
    }
    return 0;
}

static int line_char_count(const HRVocab *vocab, const HRPoem *poem, int line, size_t upto)
{
    int total = 0;
    if (upto > poem->line_counts[line]) {
        upto = poem->line_counts[line];
    }
    for (size_t i = 0; i < upto; ++i) {
        if (i > 0) {
            total += 1;
        }
        total += vocab->entries[poem->words[line][i]].length;
    }
    return total;
}

static int line_syllable_count(const HRVocab *vocab, const HRPoem *poem, int line, size_t upto)
{
    int total = 0;
    if (upto > poem->line_counts[line]) {
        upto = poem->line_counts[line];
    }
    for (size_t i = 0; i < upto; ++i) {
        total += vocab->entries[poem->words[line][i]].syllables;
    }
    return total;
}

static size_t poem_word_count(const HRPoem *poem, int through_line, size_t upto)
{
    size_t total = 0;
    for (int line = 0; line < through_line; ++line) {
        total += poem->line_counts[line];
    }
    total += upto;
    return total;
}

static HRState state_from_prefix(const HRVocab *vocab, const HRPoem *poem, int line, size_t upto)
{
    HRState state;
    memset(&state, 0, sizeof(state));
    state.line = line;
    state.line_syllables = line_syllable_count(vocab, poem, line, upto);
    state.line_chars = line_char_count(vocab, poem, line, upto);
    state.line_words = (int)upto;
    state.poem_words = (int)poem_word_count(poem, line, upto);
    state.prev_word_id = upto == 0 ? (line == 0 ? HR_ADJ_START : HR_ADJ_BOUNDARY)
                                   : poem->words[line][upto - 1u];
    state.state_hash = 1469598103934665603ULL;
    state.state_hash = mix_u64(state.state_hash, (uint64_t)line);
    state.state_hash = mix_u64(state.state_hash, (uint64_t)upto);
    state.state_hash = mix_u64(state.state_hash, (uint64_t)state.line_syllables);
    state.state_hash = mix_u64(state.state_hash, (uint64_t)state.prev_word_id);
    return state;
}

static HRCandidate target_candidate(const HRPoem *poem, int line, size_t upto)
{
    HRCandidate candidate;
    if (upto < poem->line_counts[line]) {
        candidate.kind = HR_CAND_WORD;
        candidate.word_id = poem->words[line][upto];
    } else if (line < 2) {
        candidate.kind = HR_CAND_NEWLINE;
        candidate.word_id = -1;
    } else {
        candidate.kind = HR_CAND_EOS;
        candidate.word_id = -1;
    }
    return candidate;
}

static int same_candidate(HRCandidate a, HRCandidate b)
{
    return a.kind == b.kind && a.word_id == b.word_id;
}

static int append_event(HREvent **events,
                        size_t *count,
                        size_t *capacity,
                        HRState state,
                        HRCandidate candidate,
                        int target)
{
    if (*count >= *capacity) {
        size_t next_capacity = *capacity == 0 ? 1024 : *capacity * 2u;
        HREvent *next = (HREvent *)realloc(*events, next_capacity * sizeof((*events)[0]));
        if (next == NULL) {
            return 0;
        }
        *events = next;
        *capacity = next_capacity;
    }
    (*events)[*count].state = state;
    (*events)[*count].candidate = candidate;
    (*events)[*count].target = target;
    *count += 1;
    return 1;
}

static int candidate_present(const HRCandidate *candidates, size_t count, HRCandidate candidate)
{
    for (size_t i = 0; i < count; ++i) {
        if (same_candidate(candidates[i], candidate)) {
            return 1;
        }
    }
    return 0;
}

static int build_events_for_split(const HRVocab *vocab,
                                  const HRCorpus *corpus,
                                  size_t begin,
                                  size_t end,
                                  size_t max_states,
                                  size_t max_candidates,
                                  HREvent **events_out,
                                  size_t *count_out,
                                  size_t *state_count_out)
{
    HREvent *events = NULL;
    size_t count = 0;
    size_t capacity = 0;
    size_t states = 0;
    if (end > corpus->count) {
        end = corpus->count;
    }
    if (max_candidates > vocab->count) {
        max_candidates = vocab->count;
    }
    for (size_t p = begin; p < end && (max_states == 0 || states < max_states); ++p) {
        const HRPoem *poem = &corpus->poems[p];
        for (int line = 0; line < 3 && (max_states == 0 || states < max_states); ++line) {
            for (size_t upto = 0; upto <= poem->line_counts[line] &&
                                  (max_states == 0 || states < max_states);
                 ++upto) {
                HRState state = state_from_prefix(vocab, poem, line, upto);
                HRCandidate target = target_candidate(poem, line, upto);
                HRCandidate local[HR_DEFAULT_MAX_CANDIDATES + 3];
                size_t local_count = 0;
                for (size_t r = 0; r < max_candidates && local_count < HR_DEFAULT_MAX_CANDIDATES; ++r) {
                    local[local_count].kind = HR_CAND_WORD;
                    local[local_count].word_id = vocab->ranked_ids[r];
                    local_count += 1;
                }
                if (target.kind == HR_CAND_WORD && !candidate_present(local, local_count, target) &&
                    local_count < sizeof(local) / sizeof(local[0])) {
                    local[local_count++] = target;
                }
                if (local_count < sizeof(local) / sizeof(local[0])) {
                    local[local_count].kind = HR_CAND_NEWLINE;
                    local[local_count].word_id = -1;
                    local_count += 1;
                }
                if (local_count < sizeof(local) / sizeof(local[0])) {
                    local[local_count].kind = HR_CAND_EOS;
                    local[local_count].word_id = -1;
                    local_count += 1;
                }
                for (size_t i = 0; i < local_count; ++i) {
                    if (!append_event(&events, &count, &capacity, state, local[i],
                                      same_candidate(local[i], target))) {
                        free(events);
                        return 0;
                    }
                }
                states += 1;
            }
        }
    }
    *events_out = events;
    *count_out = count;
    *state_count_out = states;
    return 1;
}

static int rule_rejects(const HRRule *rule,
                        const HRVocab *vocab,
                        const HRAdjacency *adjacency,
                        const HREvent *event)
{
    const HRState *state = &event->state;
    HRCandidate candidate = event->candidate;
    if (rule->kind == HR_RULE_SYLLABLE_OVERFLOW && candidate.kind == HR_CAND_WORD) {
        int next = state->line_syllables + vocab->entries[candidate.word_id].syllables;
        return next > HR_TARGET_SYLLABLES[state->line];
    }
    if (rule->kind == HR_RULE_LINE_CHAR_OVERFLOW && candidate.kind == HR_CAND_WORD) {
        int next = state->line_chars + vocab->entries[candidate.word_id].length +
                   (state->line_words > 0 ? 1 : 0);
        return next > HR_MAX_LINE_CHARS[state->line];
    }
    if (rule->kind == HR_RULE_EARLY_NEWLINE && candidate.kind == HR_CAND_NEWLINE) {
        return state->line < 2 && state->line_syllables < HR_TARGET_SYLLABLES[state->line];
    }
    if (rule->kind == HR_RULE_EOS_BEFORE_COMPLETE && candidate.kind == HR_CAND_EOS) {
        return state->line < 2 || state->line_syllables < HR_TARGET_SYLLABLES[2];
    }
    if (rule->kind == HR_RULE_ADJACENCY_MISSING && candidate.kind == HR_CAND_WORD) {
        return !adjacency_has(adjacency, state->line, state->prev_word_id, candidate.word_id);
    }
    if (rule->kind == HR_RULE_UNSEEN_LINE_WORD && candidate.kind == HR_CAND_WORD) {
        return vocab->entries[candidate.word_id].line_frequency[state->line] == 0;
    }
    (void)rule;
    return 0;
}

static void init_rule(HRRule *rule, HRRuleKind kind, const char *name, const char *text)
{
    memset(rule, 0, sizeof(*rule));
    rule->kind = kind;
    snprintf(rule->text, sizeof(rule->text), "%s", text);
    snprintf(rule->nn_rule.domain, sizeof(rule->nn_rule.domain), "haiku_word_lm");
    snprintf(rule->nn_rule.name, sizeof(rule->nn_rule.name), "%s", name);
    rule->nn_rule.kind = NN_RULE_KIND_REJECT_ACTION;
    rule->nn_rule.template_id = (uint32_t)kind;
    rule->nn_rule.structural_feature_count = HR_RULE_FEATURES;
    rule->nn_rule.structural_features[0] = (nn_real)kind;
    rule->nn_rule.structural_features[1] = 1.0f;
}

static size_t enumerate_rules(HRRule *rules, size_t capacity)
{
    size_t count = 0;
    if (count < capacity) {
        init_rule(&rules[count++], HR_RULE_SYLLABLE_OVERFLOW, "syllable_overflow",
                  "reject word if it would exceed the 5/7/5 syllable budget");
    }
    if (count < capacity) {
        init_rule(&rules[count++], HR_RULE_LINE_CHAR_OVERFLOW, "line_char_overflow",
                  "reject word if it would exceed the line character budget");
    }
    if (count < capacity) {
        init_rule(&rules[count++], HR_RULE_EARLY_NEWLINE, "early_newline",
                  "reject newline before the current line reaches its syllable target");
    }
    if (count < capacity) {
        init_rule(&rules[count++], HR_RULE_EOS_BEFORE_COMPLETE, "eos_before_complete",
                  "reject eos before line three reaches its syllable target");
    }
    if (count < capacity) {
        init_rule(&rules[count++], HR_RULE_ADJACENCY_MISSING, "adjacency_missing",
                  "reject word transitions absent from train-split adjacency");
    }
    if (count < capacity) {
        init_rule(&rules[count++], HR_RULE_UNSEEN_LINE_WORD, "unseen_line_word",
                  "reject a word never observed in the current line position");
    }
    return count;
}

static void score_rules(HRRule *rules,
                        size_t rule_count,
                        const HRVocab *vocab,
                        const HRAdjacency *adjacency,
                        const HREvent *train_events,
                        size_t train_event_count,
                        const HREvent *valid_events,
                        size_t valid_event_count,
                        const HREvent *test_events,
                        size_t test_event_count,
                        size_t *verified_out,
                        size_t *unsafe_out,
                        size_t *no_support_out)
{
    size_t verified = 0;
    size_t unsafe = 0;
    size_t no_support = 0;
    for (size_t r = 0; r < rule_count; ++r) {
        HRRule *rule = &rules[r];
        size_t checked = 0;
        size_t support = 0;
        size_t false_rejections = 0;
        for (size_t i = 0; i < train_event_count; ++i) {
            int rejects = rule_rejects(rule, vocab, adjacency, &train_events[i]);
            if (train_events[i].target) {
                checked += 1;
                if (rejects) {
                    false_rejections += 1;
                }
            } else if (rejects) {
                support += 1;
            }
        }
        rule->nn_rule.train_rejects = support;
        rule->nn_rule.train_false_rejections = false_rejections;
        for (size_t i = 0; i < valid_event_count; ++i) {
            int rejects = rule_rejects(rule, vocab, adjacency, &valid_events[i]);
            if (valid_events[i].target && rejects) {
                rule->nn_rule.validation_false_rejections += 1;
            } else if (!valid_events[i].target && rejects) {
                rule->nn_rule.validation_rejects += 1;
            }
        }
        for (size_t i = 0; i < test_event_count; ++i) {
            int rejects = rule_rejects(rule, vocab, adjacency, &test_events[i]);
            if (test_events[i].target && rejects) {
                rule->nn_rule.test_false_rejections += 1;
            } else if (!test_events[i].target && rejects) {
                rule->nn_rule.test_rejects += 1;
            }
        }
        rule->nn_rule.utility = (double)support;
        rule->nn_rule.structural_features[2] = (nn_real)support;
        rule->nn_rule.structural_features[3] = (nn_real)rule->nn_rule.validation_rejects;
        rule->nn_rule.structural_features[4] = (nn_real)false_rejections;
        if (false_rejections > 0) {
            nn_certificate_reject(&rule->nn_rule.certificate,
                                  "haiku_train_positive_replay",
                                  "rule rejects observed train next actions",
                                  checked,
                                  false_rejections);
            unsafe += 1;
        } else if (support == 0) {
            nn_certificate_reject(&rule->nn_rule.certificate,
                                  "haiku_negative_candidate_replay",
                                  "rule has no train negative support",
                                  checked,
                                  false_rejections);
            no_support += 1;
        } else {
            nn_certificate_accept(&rule->nn_rule.certificate,
                                  "haiku_train_positive_replay",
                                  "zero false rejections on train positives",
                                  checked,
                                  support);
            verified += 1;
        }
    }
    *verified_out = verified;
    *unsafe_out = unsafe;
    *no_support_out = no_support;
}

static int build_negative_coverage(const HRRule *rules,
                                   size_t rule_count,
                                   const HRVocab *vocab,
                                   const HRAdjacency *adjacency,
                                   const HREvent *events,
                                   size_t event_count,
                                   NNRuleCoverage *coverage,
                                   size_t *negative_count_out)
{
    size_t negatives = 0;
    for (size_t i = 0; i < event_count; ++i) {
        if (!events[i].target) {
            negatives += 1;
        }
    }
    if (!nn_rule_coverage_init(coverage, rule_count, negatives)) {
        return 0;
    }
    size_t negative_index = 0;
    for (size_t e = 0; e < event_count; ++e) {
        if (events[e].target) {
            continue;
        }
        for (size_t r = 0; r < rule_count; ++r) {
            if (rules[r].nn_rule.certificate.status == NN_CERTIFICATE_CERTIFIED &&
                rule_rejects(&rules[r], vocab, adjacency, &events[e])) {
                nn_rule_coverage_set(coverage, r, negative_index);
            }
        }
        negative_index += 1;
    }
    *negative_count_out = negatives;
    return 1;
}

static int closed_loop_features(size_t rule_index,
                                const size_t *accepted_indices,
                                size_t accepted_count,
                                const NNRuleCoverage *coverage,
                                const uint64_t *covered,
                                void *context,
                                nn_real *features,
                                size_t capacity)
{
    if (capacity < HR_RULE_FEATURES || context == NULL) {
        return 0;
    }
    const HRClosedLoopContext *ctx = (const HRClosedLoopContext *)context;
    const HRRule *rule = &ctx->rules[rule_index];
    memset(features, 0, HR_RULE_FEATURES * sizeof(features[0]));
    features[0] = (nn_real)rule->kind / (nn_real)HR_RULE_KIND_COUNT;
    features[1] = (nn_real)rule->nn_rule.train_rejects / 1000.0f;
    features[2] = (nn_real)rule->nn_rule.validation_rejects / 1000.0f;
    features[3] = (nn_real)rule->nn_rule.test_rejects / 1000.0f;
    features[4] = (nn_real)nn_rule_coverage_gain(coverage, rule_index, covered) / 1000.0f;
    features[5] = (nn_real)accepted_count / 32.0f;
    features[6] = rule->kind == HR_RULE_ADJACENCY_MISSING ? 1.0f : 0.0f;
    features[7] = rule->kind == HR_RULE_SYLLABLE_OVERFLOW ? 1.0f : 0.0f;
    features[8] = rule->kind == HR_RULE_LINE_CHAR_OVERFLOW ? 1.0f : 0.0f;
    features[9] = rule->kind == HR_RULE_EARLY_NEWLINE ? 1.0f : 0.0f;
    features[10] = rule->kind == HR_RULE_EOS_BEFORE_COMPLETE ? 1.0f : 0.0f;
    features[11] = (nn_real)nn_rule_coverage_count_covered(covered, coverage->word_count) /
                   (nn_real)(coverage->event_count == 0 ? 1 : coverage->event_count);
    (void)accepted_indices;
    return 1;
}

static int candidate_allowed(size_t rule_index,
                             const size_t *accepted_indices,
                             size_t accepted_count,
                             void *context)
{
    const HRClosedLoopContext *ctx = (const HRClosedLoopContext *)context;
    if (ctx == NULL || rule_index >= ctx->rule_count) {
        return 0;
    }
    for (size_t i = 0; i < accepted_count; ++i) {
        if (accepted_indices[i] == rule_index) {
            return 0;
        }
    }
    return ctx->rules[rule_index].nn_rule.certificate.status == NN_CERTIFICATE_CERTIFIED;
}

static int keep_accepted(size_t accepted_rule_index, size_t candidate_rule_index, void *context)
{
    (void)accepted_rule_index;
    (void)candidate_rule_index;
    (void)context;
    return 1;
}

static int select_rules(const HRConfig *config,
                        HRRule *rules,
                        size_t rule_count,
                        const NNRuleCoverage *coverage,
                        size_t *accepted,
                        size_t *accepted_count_out,
                        size_t *proposer_calls_out,
                        double *proposer_loss_out)
{
    *accepted_count_out = 0;
    *proposer_calls_out = 0;
    *proposer_loss_out = 0.0;
    if (config->proposer == HR_PROPOSER_CLOSED_LOOP) {
        HRClosedLoopContext ctx = {.rules = rules, .rule_count = rule_count};
        NNClosedLoopRulePolicyConfig policy_config;
        nn_closed_loop_rule_policy_config_defaults(&policy_config, HR_RULE_FEATURES);
        policy_config.max_rules = config->max_rules;
        policy_config.hidden1 = 32;
        policy_config.hidden2 = 16;
        policy_config.epochs = 12;
        policy_config.batch_size = 16;
        policy_config.learning_rate = 0.01f;
        policy_config.seed = config->proposer_seed;
        policy_config.feature_schema = HR_RULE_FEATURE_SCHEMA;
        policy_config.feature_schema_count =
            sizeof(HR_RULE_FEATURE_SCHEMA) / sizeof(HR_RULE_FEATURE_SCHEMA[0]);
        NNClosedLoopRulePolicy policy;
        memset(&policy, 0, sizeof(policy));
        nn_real scores[HR_RULE_KIND_COUNT];
        int ok = nn_closed_loop_rule_policy_train(coverage,
                                                  &policy_config,
                                                  closed_loop_features,
                                                  candidate_allowed,
                                                  keep_accepted,
                                                  &ctx,
                                                  &policy) &&
                 nn_closed_loop_rule_policy_select(coverage,
                                                   &policy_config,
                                                   closed_loop_features,
                                                   candidate_allowed,
                                                   keep_accepted,
                                                   &ctx,
                                                   &policy,
                                                   scores,
                                                   accepted,
                                                   config->max_rules,
                                                   accepted_count_out,
                                                   proposer_calls_out);
        *proposer_loss_out = policy.loss;
        nn_closed_loop_rule_policy_free(&policy);
        return ok;
    }

    uint64_t *covered = (uint64_t *)calloc(coverage->word_count, sizeof(uint64_t));
    int *selected = (int *)calloc(rule_count, sizeof(int));
    if (covered == NULL || selected == NULL) {
        free(covered);
        free(selected);
        return 0;
    }
    uint32_t rng = config->proposer_seed;
    while (*accepted_count_out < config->max_rules) {
        size_t best = (size_t)-1;
        double best_score = -INFINITY;
        for (size_t r = 0; r < rule_count; ++r) {
            if (selected[r] || rules[r].nn_rule.certificate.status != NN_CERTIFICATE_CERTIFIED) {
                continue;
            }
            size_t gain = nn_rule_coverage_gain(coverage, r, covered);
            if (gain == 0) {
                continue;
            }
            double score = 0.0;
            if (config->proposer == HR_PROPOSER_RANDOM) {
                score = (double)(rng_next(&rng) & 0xffffffu) / (double)0x1000000u;
            } else if (config->proposer == HR_PROPOSER_STRUCTURAL) {
                score = (double)gain + 0.25 * (double)rules[r].nn_rule.validation_rejects;
            } else {
                score = -(double)r;
            }
            if (score > best_score) {
                best_score = score;
                best = r;
            }
        }
        *proposer_calls_out += 1;
        if (best == (size_t)-1) {
            break;
        }
        selected[best] = 1;
        accepted[(*accepted_count_out)++] = best;
        nn_rule_coverage_add_rule_to_residual(coverage, best, covered);
    }
    free(covered);
    free(selected);
    return 1;
}

static const char *proposer_name(HRProposer proposer)
{
    switch (proposer) {
    case HR_PROPOSER_STRUCTURAL:
        return "structural";
    case HR_PROPOSER_RANDOM:
        return "random";
    case HR_PROPOSER_CLOSED_LOOP:
        return "closed_loop";
    case HR_PROPOSER_GRAMMAR:
    default:
        return "grammar";
    }
}

static const char *rule_kind_name(HRRuleKind kind)
{
    switch (kind) {
    case HR_RULE_SYLLABLE_OVERFLOW:
        return "syllable_overflow";
    case HR_RULE_LINE_CHAR_OVERFLOW:
        return "line_char_overflow";
    case HR_RULE_EARLY_NEWLINE:
        return "early_newline";
    case HR_RULE_EOS_BEFORE_COMPLETE:
        return "eos_before_complete";
    case HR_RULE_ADJACENCY_MISSING:
        return "adjacency_missing";
    case HR_RULE_UNSEEN_LINE_WORD:
        return "unseen_line_word";
    default:
        return "unknown";
    }
}

static uint64_t rule_set_hash(const HRRule *rules, const size_t *accepted, size_t accepted_count)
{
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < accepted_count; ++i) {
        const HRRule *rule = &rules[accepted[i]];
        hash = mix_u64(hash, (uint64_t)rule->kind);
        hash = mix_u64(hash, (uint64_t)rule->nn_rule.train_rejects);
        hash = mix_u64(hash, (uint64_t)rule->nn_rule.validation_rejects);
    }
    return hash;
}

static int write_trace(const char *path,
                       const HRConfig *config,
                       const HRRule *rules,
                       const size_t *accepted,
                       size_t accepted_count)
{
    if (path == NULL) {
        return 1;
    }
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        fprintf(stderr, "failed to open trace %s: %s\n", path, strerror(errno));
        return 0;
    }
    const char *grammar = "{\"name\":\"haiku_word_rule_search\",\"version\":1}";
    for (size_t i = 0; i < accepted_count; ++i) {
        const HRRule *rule = &rules[accepted[i]];
        NNRuleTraceRecord record;
        memset(&record, 0, sizeof(record));
        record.trace_version = 1;
        record.task_name = "haiku_rule_search";
        record.grammar_version_json = grammar;
        record.selection_strategy = proposer_name(config->proposer);
        record.proposer_feature_set = "haiku_rule_features_v1";
        record.proposer_target = "negative_candidate_marginal_rejects";
        record.rule_id = accepted[i];
        record.rule_rank = i + 1u;
        record.rule_kind = rule_kind_name(rule->kind);
        record.accepted = 1;
        record.verifier_result = "certified";
        record.certificate = rule->nn_rule.certificate.reason;
        record.model_score = 0.0;
        record.structural_score = rule->nn_rule.utility;
        record.utility = rule->nn_rule.utility;
        record.train_rejects = rule->nn_rule.train_rejects;
        record.train_false_rejections = rule->nn_rule.train_false_rejections;
        record.validation_rejects = rule->nn_rule.validation_rejects;
        record.validation_false_rejections = rule->nn_rule.validation_false_rejections;
        record.test_rejects = rule->nn_rule.test_rejects;
        record.test_false_rejections = rule->nn_rule.test_false_rejections;
        record.train_marginal_utility = (double)rule->nn_rule.train_rejects;
        record.validation_marginal_utility = (double)rule->nn_rule.validation_rejects;
        record.rule_text = rule->text;
        if (!nn_rule_trace_write_json(fp, &record)) {
            fclose(fp);
            return 0;
        }
    }
    fclose(fp);
    return 1;
}

static int write_rules(const char *path,
                       const HRRule *rules,
                       const size_t *accepted,
                       size_t accepted_count)
{
    if (path == NULL) {
        return 1;
    }
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        fprintf(stderr, "failed to open rules %s: %s\n", path, strerror(errno));
        return 0;
    }
    for (size_t i = 0; i < accepted_count; ++i) {
        const HRRule *rule = &rules[accepted[i]];
        fprintf(fp,
                "{\"schema\":\"crlp.rule.v1\",\"domain\":\"haiku_word_lm\","
                "\"rule_id\":%zu,\"name\":\"%s\",\"kind\":\"%s\","
                "\"certificate\":\"%s\",\"train_rejects\":%zu,"
                "\"validation_rejects\":%zu,\"test_rejects\":%zu,"
                "\"rule_text\":\"%s\"}\n",
                accepted[i],
                rule->nn_rule.name,
                rule_kind_name(rule->kind),
                rule->nn_rule.certificate.reason,
                rule->nn_rule.train_rejects,
                rule->nn_rule.validation_rejects,
                rule->nn_rule.test_rejects,
                rule->text);
    }
    fclose(fp);
    return 1;
}

static void print_usage(FILE *stream)
{
    fputs("usage: haiku_rule_search [--corpus path] [--max-candidates N] [--max-states N]\n"
          "                         [--max-rules N] [--proposer grammar|structural|random|closed-loop]\n"
          "                         [--proposer-seed N] [--trace path] [--save path]\n",
          stream);
}

static int parse_config(int argc, char **argv, HRConfig *config)
{
    config->corpus_path = "data/haiku_tiny.txt";
    config->trace_path = NULL;
    config->save_path = NULL;
    config->max_candidates = HR_DEFAULT_MAX_CANDIDATES;
    config->max_rules = HR_DEFAULT_MAX_RULES;
    config->max_states = HR_DEFAULT_MAX_STATES;
    config->proposer = HR_PROPOSER_CLOSED_LOOP;
    config->proposer_seed = 20240518u;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--corpus") == 0 && i + 1 < argc) {
            config->corpus_path = argv[++i];
        } else if (strcmp(argv[i], "--trace") == 0 && i + 1 < argc) {
            config->trace_path = argv[++i];
        } else if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            config->save_path = argv[++i];
        } else if (strcmp(argv[i], "--max-candidates") == 0 && i + 1 < argc) {
            if (!nn_parse_size_nonnegative_text(argv[++i], &config->max_candidates)) {
                return 0;
            }
            if (config->max_candidates > HR_DEFAULT_MAX_CANDIDATES) {
                config->max_candidates = HR_DEFAULT_MAX_CANDIDATES;
            }
        } else if (strcmp(argv[i], "--max-rules") == 0 && i + 1 < argc) {
            if (!nn_parse_size_nonnegative_text(argv[++i], &config->max_rules)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--max-states") == 0 && i + 1 < argc) {
            if (!nn_parse_size_nonnegative_text(argv[++i], &config->max_states)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--proposer-seed") == 0 && i + 1 < argc) {
            unsigned long seed = strtoul(argv[++i], NULL, 10);
            config->proposer_seed = (uint32_t)seed;
        } else if (strcmp(argv[i], "--proposer") == 0 && i + 1 < argc) {
            const char *name = argv[++i];
            if (strcmp(name, "grammar") == 0) {
                config->proposer = HR_PROPOSER_GRAMMAR;
            } else if (strcmp(name, "structural") == 0) {
                config->proposer = HR_PROPOSER_STRUCTURAL;
            } else if (strcmp(name, "random") == 0) {
                config->proposer = HR_PROPOSER_RANDOM;
            } else if (strcmp(name, "closed-loop") == 0 || strcmp(name, "closed_loop") == 0) {
                config->proposer = HR_PROPOSER_CLOSED_LOOP;
            } else {
                return 0;
            }
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(stdout);
            exit(EXIT_SUCCESS);
        } else {
            return 0;
        }
    }
    if (config->max_rules > HR_RULE_KIND_COUNT) {
        config->max_rules = HR_RULE_KIND_COUNT;
    }
    return config->max_candidates > 0 && config->max_rules > 0;
}

int main(int argc, char **argv)
{
    HRConfig config;
    if (!parse_config(argc, argv, &config)) {
        print_usage(stderr);
        return EXIT_FAILURE;
    }

    HRVocab *vocab = (HRVocab *)calloc(1, sizeof(*vocab));
    HRCorpus *corpus = (HRCorpus *)calloc(1, sizeof(*corpus));
    HRAdjacency *adjacency = (HRAdjacency *)calloc(1, sizeof(*adjacency));
    if (vocab == NULL || corpus == NULL || adjacency == NULL) {
        free(vocab);
        free(corpus);
        free(adjacency);
        return EXIT_FAILURE;
    }
    if (!load_corpus(config.corpus_path, vocab, corpus)) {
        free(vocab);
        free(corpus);
        free(adjacency);
        return EXIT_FAILURE;
    }
    sort_vocab(vocab);
    size_t train_end = (corpus->count * 70u) / 100u;
    size_t valid_end = (corpus->count * 85u) / 100u;
    if (train_end == 0) {
        train_end = corpus->count;
    }
    if (valid_end < train_end) {
        valid_end = train_end;
    }

    build_adjacency(corpus, 0, train_end, adjacency);

    HREvent *train_events = NULL;
    HREvent *valid_events = NULL;
    HREvent *test_events = NULL;
    size_t train_event_count = 0;
    size_t valid_event_count = 0;
    size_t test_event_count = 0;
    size_t train_states = 0;
    size_t valid_states = 0;
    size_t test_states = 0;
    if (!build_events_for_split(vocab,
                                corpus,
                                0,
                                train_end,
                                config.max_states,
                                config.max_candidates,
                                &train_events,
                                &train_event_count,
                                &train_states) ||
        !build_events_for_split(vocab,
                                corpus,
                                train_end,
                                valid_end,
                                config.max_states / 2u,
                                config.max_candidates,
                                &valid_events,
                                &valid_event_count,
                                &valid_states) ||
        !build_events_for_split(vocab,
                                corpus,
                                valid_end,
                                corpus->count,
                                config.max_states / 2u,
                                config.max_candidates,
                                &test_events,
                                &test_event_count,
                                &test_states)) {
        free(train_events);
        free(valid_events);
        free(test_events);
        free(vocab);
        free(corpus);
        free(adjacency);
        return EXIT_FAILURE;
    }

    HRRule rules[HR_RULE_KIND_COUNT];
    size_t rule_count = enumerate_rules(rules, HR_RULE_KIND_COUNT);
    size_t verified = 0;
    size_t unsafe = 0;
    size_t no_support = 0;
    score_rules(rules,
                rule_count,
                vocab,
                adjacency,
                train_events,
                train_event_count,
                valid_events,
                valid_event_count,
                test_events,
                test_event_count,
                &verified,
                &unsafe,
                &no_support);

    NNRuleCoverage coverage;
    memset(&coverage, 0, sizeof(coverage));
    size_t train_negatives = 0;
    if (!build_negative_coverage(rules,
                                 rule_count,
                                 vocab,
                                 adjacency,
                                 train_events,
                                 train_event_count,
                                 &coverage,
                                 &train_negatives)) {
        free(train_events);
        free(valid_events);
        free(test_events);
        free(vocab);
        free(corpus);
        free(adjacency);
        return EXIT_FAILURE;
    }

    size_t accepted[HR_RULE_KIND_COUNT];
    size_t accepted_count = 0;
    size_t proposer_calls = 0;
    double proposer_loss = 0.0;
    int ok = select_rules(&config,
                          rules,
                          rule_count,
                          &coverage,
                          accepted,
                          &accepted_count,
                          &proposer_calls,
                          &proposer_loss);
    if (!ok) {
        nn_rule_coverage_free(&coverage);
        free(train_events);
        free(valid_events);
        free(test_events);
        free(vocab);
        free(corpus);
        free(adjacency);
        return EXIT_FAILURE;
    }
    for (size_t i = 0; i < accepted_count; ++i) {
        rules[accepted[i]].nn_rule.accepted = 1;
    }

    size_t train_rejects = 0;
    size_t validation_rejects = 0;
    size_t test_rejects = 0;
    for (size_t i = 0; i < accepted_count; ++i) {
        const NNRule *rule = &rules[accepted[i]].nn_rule;
        train_rejects += rule->train_rejects;
        validation_rejects += rule->validation_rejects;
        test_rejects += rule->test_rejects;
    }

    uint64_t hash = rule_set_hash(rules, accepted, accepted_count);
    char hash_text[32];
    snprintf(hash_text, sizeof(hash_text), "%016llx", (unsigned long long)hash);
    char extra[512];
    snprintf(extra,
             sizeof(extra),
             "\"poems\":%zu,\"vocab\":%zu,\"train_states\":%zu,"
             "\"validation_states\":%zu,\"test_states\":%zu,"
             "\"train_negative_events\":%zu,\"adjacency_edges\":%zu,"
             "\"adjacency_overflow\":%zu,\"adjacency_hash\":\"%016llx\"",
             corpus->count,
             vocab->count,
             train_states,
             valid_states,
             test_states,
             train_negatives,
             adjacency->count,
             adjacency->overflow,
             (unsigned long long)adjacency->hash);

    NNRuleSearchSummary summary;
    memset(&summary, 0, sizeof(summary));
    summary.report_version = 1;
    summary.task_name = "haiku_rule_search";
    summary.grammar_version_json = "{\"name\":\"haiku_word_rule_search\",\"version\":1}";
    summary.rule_set_hash = hash_text;
    summary.train_limit = train_end;
    summary.test_limit = corpus->count - valid_end;
    summary.sample_count = train_states + valid_states + test_states;
    summary.max_rules = config.max_rules;
    summary.max_marginal_states = config.max_states;
    summary.closed_loop_hidden_json = "[32,16]";
    summary.unified_scorer_hidden_json = "[]";
    summary.selection_strategy = proposer_name(config.proposer);
    summary.selection_seed = config.proposer_seed;
    summary.proposed_rules = rule_count;
    summary.proposer_calls = proposer_calls;
    summary.verified_rules = verified;
    summary.unsafe_rules = unsafe;
    summary.no_support_rules = no_support;
    summary.uncertified_rules = unsafe + no_support;
    summary.accepted_rules = accepted_count;
    summary.false_rejections = 0;
    summary.proposer_feature_set = "haiku_rule_features_v1";
    summary.proposer_target = "negative_candidate_marginal_rejects";
    summary.residual_train_samples = train_negatives;
    summary.train_marginal_samples = train_states;
    summary.validation_marginal_samples = valid_states;
    summary.train_marginal_events = train_event_count;
    summary.validation_marginal_events = valid_event_count;
    summary.train_rejects = train_rejects;
    summary.validation_rejects = validation_rejects;
    summary.test_rejects = test_rejects;
    summary.train_marginal_rejects = train_rejects;
    summary.validation_marginal_rejects = validation_rejects;
    summary.test_marginal_rejects = test_rejects;
    summary.parameters = config.proposer == HR_PROPOSER_CLOSED_LOOP ? (HR_RULE_FEATURES * 32u + 32u * 16u + 16u) : 0u;
    summary.proposer_loss = proposer_loss;
    summary.extra_json = extra;

    ok = write_trace(config.trace_path, &config, rules, accepted, accepted_count) &&
         write_rules(config.save_path, rules, accepted, accepted_count) &&
         nn_rule_search_summary_write_json(stdout, &summary);

    nn_rule_coverage_free(&coverage);
    free(train_events);
    free(valid_events);
    free(test_events);
    free(vocab);
    free(corpus);
    free(adjacency);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
