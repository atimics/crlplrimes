#include "nn_grammar_discovery.h"
#include "nn_protocol.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define NN_GD_LINE_MAX 65536
#define NN_GD_MAX_FIELDS 96
#define NN_GD_COUNTEREXAMPLE_LIMIT 8
#define NN_GD_ARC_RUNTIME_FIELD_COUNT 85
#define NN_GD_ARC_RANKER_CACHE_VERSION 5U
#define NN_GD_ARC_RANKER_CACHE_MAX_STRING_BYTES (16U * 1024U * 1024U)
#define NN_GD_ARC_RANKER_CACHE_MAX_VOCAB 1000000U
#define NN_GD_ARC_RANKER_CACHE_MAX_FIELD_STRINGS 5000000U
#define NN_GD_ARC_RANKER_CACHE_MAX_ATTEMPTS 10000000U
#define NN_GD_ARC_RANKER_CACHE_EMPTY_STRING_REF UINT32_MAX

typedef enum {
    NN_GD_ARC_RANKER_CACHE_TSV = 1,
    NN_GD_ARC_RANKER_CACHE_JSONL = 2
} NNGDArcRankerCacheMode;

typedef struct {
    char **items;
    size_t count;
    size_t capacity;
} NNGDStringList;

static double nn_gd_wall_seconds(void) {
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
    }
#endif
    return (double)time(NULL);
}

typedef struct {
    size_t *slots;
    size_t slot_count;
} NNGDStringTable;

typedef enum {
    NN_GD_CE_ARC,
    NN_GD_CE_HAIKU_GENERATION,
    NN_GD_CE_HAIKU_UNDERSTANDING
} NNGDCounterexampleKind;

typedef struct {
    NNGDCounterexampleKind kind;
    char *task_id;
    char *split;
    char *example_id;
    long sample;
    long step;
    long candidate_count;
    long poem_index;
    char *corruption;
    char *symbolic_selected;
    NNGDStringList failed_predicates;
} NNGDCounterexample;

typedef struct {
    char *candidate_id;
    char *domain;
    char *family;
    char *source_artifacts;
    char *feature_summary_json;
    size_t support_examples;
    NNGDStringList task_ids;
    NNGDStringList adjacency_hashes;
    double promotion_proposals_generated_sum;
    long mutation_train_consistent_rows;
    NNGDCounterexample counterexamples[NN_GD_COUNTEREXAMPLE_LIMIT];
    size_t counterexamples_stored;
    size_t counterexamples_total;
} NNGDDeltaCandidate;

typedef struct {
    NNGDDeltaCandidate *items;
    size_t count;
    size_t capacity;
} NNGDDeltaCandidateList;

typedef struct {
    char *sketch_id;
    char *domain;
    char *family;
    NNGDStringList tokens;
} NNGDSketch;

typedef struct {
    NNGDSketch *items;
    size_t count;
    size_t capacity;
} NNGDSketchList;

typedef struct {
    size_t sketch_index;
    double label;
} NNGDLabel;

typedef struct {
    NNGDLabel *items;
    size_t count;
    size_t capacity;
} NNGDLabelList;

typedef struct {
    size_t sketch_index;
    double score;
} NNGDScoredSketch;

typedef struct {
    size_t sketch_index;
    double static_score;
    double dynamic_score;
    double evidence_mean;
    double overlap_penalty;
    double family_penalty;
    double domain_novelty_bonus;
    size_t same_domain_selected;
    size_t same_family_selected;
    size_t selected_before;
} NNGDClosedLoopSketchChoice;

typedef struct {
    char *split_name;
    char *family;
    char *task_id;
    char *split;
    char *attempt_id;
    char *library_id;
    long attempt_index;
    char *pipeline_kind;
    char *op_name;
    char *visual_program;
    char *slot_contract;
    char *composition_kind;
    char *normalization_frame;
    char *typed_stages;
    char *primitive_contract;
    char *slot_tokens;
    char *rejection_reason;
    char *failure_stage;
    char *failure_detail;
    char *contrast_accepted_attempt_id;
    char *contrast_pair_kind;
    int accepted;
    long proposal_count;
    long support_tasks;
    double utility_mean;
    long contrast_slot_distance;
    long contrast_rejected_sibling_count;
    long verifier_train_pairs;
    long verifier_checked_cells;
    long verifier_mismatches;
    long verifier_unsupported_pairs;
    long verifier_apply_failed_pairs;
    long verifier_shape_mismatch_pairs;
    long verifier_missing_output_pairs;
    long verifier_first_failing_pair;
    double verifier_mismatch_ratio;
    double verifier_score;
    long selector;
    long x_anchor;
    long y_anchor;
    long x_offset;
    long y_offset;
    long width;
    long height;
    long period;
    long input_width;
    long input_height;
    long output_width;
    long output_height;
    long shape_scale_x;
    long shape_scale_y;
    int period_divides_input_width;
    int period_divides_input_height;
    int slot_width_matches_output;
    int slot_height_matches_output;
    size_t contrast_accepted_index;
    NNGDStringList tokens;
    size_t token_count;
    size_t *token_indices;
    size_t token_index_count;
} NNGDArcPipelineAttempt;

typedef struct {
    NNGDArcPipelineAttempt *items;
    size_t count;
    size_t capacity;
} NNGDArcPipelineAttemptList;

typedef struct {
    char *scope;
    char *path;
    size_t rows;
} NNGDArcHoleSourceLog;

typedef struct {
    NNGDArcHoleSourceLog *items;
    size_t count;
    size_t capacity;
} NNGDArcHoleSourceLogList;

typedef struct {
    char *task_id;
    char *split;
    char *source_scope;
    char *source_path;
    char *kind;
    long attempted;
    long train_consistent_pre_cap;
    long train_consistent_retained;
    long selected;
} NNGDArcHoleTarget;

typedef struct {
    NNGDArcHoleTarget *items;
    size_t count;
    size_t capacity;
} NNGDArcHoleTargetList;

typedef struct {
    char *fields[13];
} NNGDArcHoleCandidate;

typedef struct {
    NNGDArcHoleCandidate *items;
    size_t count;
    size_t capacity;
} NNGDArcHoleCandidateList;

typedef struct {
    char *key;
    size_t count;
} NNGDArcHoleCounter;

typedef struct {
    NNGDArcHoleCounter *items;
    size_t count;
    size_t capacity;
} NNGDArcHoleCounterList;

typedef struct {
    long first_input_width;
    long first_input_height;
    long first_output_width;
    long first_output_height;
    int same_output_shape;
    int crop_like;
    int grow_like;
    int line_like;
    long scale;
    long train_pairs;
    int loaded;
} NNGDArcTaskShapes;

typedef struct {
    char *family;
    char *task_id;
    char *split;
    char *library_id;
    char *rejection_reason;
    int accepted;
    double verifier_score;
} NNGDArcAttemptDedupeFields;

typedef struct {
    char *family;
    char *task_id;
    char *split;
    char *library_id;
    char *rejection_reason;
    int accepted;
    double verifier_score;
    size_t source_index;
    fpos_t position;
    uint64_t hash;
} NNGDArcAttemptDedupeRecord;

typedef struct {
    NNGDArcAttemptDedupeRecord *items;
    size_t count;
    size_t capacity;
    size_t *slots;
    size_t slot_count;
} NNGDArcAttemptDedupeTable;

typedef struct {
    const char *key;
    size_t key_len;
    const char *value;
} NNGDJsonKeyValue;

typedef struct {
    NNGDJsonKeyValue items[192];
    size_t count;
} NNGDJsonIndex;

typedef struct {
    size_t attempt_index;
    double score;
    double model_score;
    double renderer_score;
    double sequence_block_score;
    double component_score;
    double component_selector_layout_score;
} NNGDScoredArcAttempt;

typedef enum {
    NN_GD_ARC_HEAD_RENDERER,
    NN_GD_ARC_HEAD_SEQUENCE_BLOCK_RENDERER,
    NN_GD_ARC_HEAD_COMPONENT_SHAPE_COLOR,
    NN_GD_ARC_HEAD_COMPONENT_SELECTOR_LAYOUT
} NNGDArcAuxHeadKind;

#define NNGD_ARC_AUX_HEAD_COUNT 4U

typedef struct {
    const char *name;
    NNGDArcAuxHeadKind kind;
    double *weights;
    double bias;
    size_t train_rows;
    size_t positive_rows;
    double label_sum;
    double final_loss;
    double target_score_sum;
    size_t target_rows;
} NNGDArcAuxHead;

typedef struct {
    int loaded;
    unsigned epochs_seen_before;
    unsigned epochs_seen_after;
    size_t training_examples_seen_before;
    size_t training_examples_seen_after;
    size_t restored_weights;
    size_t ignored_weights;
} NNGDMemoryState;

typedef enum {
    NN_GD_HAIKU_METRIC_REAL_PROBABILITY,
    NN_GD_HAIKU_METRIC_CORRUPT_PROBABILITY,
    NN_GD_HAIKU_METRIC_PLAN_SCORE_DELTA,
    NN_GD_HAIKU_METRIC_TOPIC_TIGHTNESS_DELTA,
    NN_GD_HAIKU_METRIC_MIN_LINE_SIMILARITY_DELTA,
    NN_GD_HAIKU_METRIC_CORRUPT_ADJACENCY_RATIO,
    NN_GD_HAIKU_METRIC_CORRUPT_BIGRAM_RATIO,
    NN_GD_HAIKU_METRIC_CORRUPT_EXACT_SYLLABLE_RATIO,
    NN_GD_HAIKU_METRIC_REPLACEMENT_LINE_SIMILARITY,
    NN_GD_HAIKU_METRIC_LOCAL_NEIGHBOR_FIT_DELTA,
    NN_GD_HAIKU_METRIC_LEFT_NEIGHBOR_SIMILARITY_DELTA,
    NN_GD_HAIKU_METRIC_RIGHT_NEIGHBOR_SIMILARITY_DELTA,
    NN_GD_HAIKU_METRIC_LINE_WORD_DELTA,
    NN_GD_HAIKU_METRIC_LINE_SYLLABLE_DELTA,
    NN_GD_HAIKU_METRIC_COUNT
} NNGDHaikuMetric;

typedef struct {
    char *corruption;
    char *state_hash;
    char *adjacency_hash;
    char *selected;
    char *symbolic_selected;
    long poem_index;
    long corrupt_line;
    long source_poem_index;
    long source_line;
    int neural_available;
    int symbolic_available;
    int neural_failed;
    int symbolic_failed;
    int topic_coherence_family;
    int plausible_negative;
    double metrics[NN_GD_HAIKU_METRIC_COUNT];
    double difficulty_score;
} NNGDHaikuCoherenceRow;

typedef struct {
    NNGDHaikuCoherenceRow *items;
    size_t count;
    size_t capacity;
    size_t source_trace_rows;
} NNGDHaikuCoherenceRows;

typedef struct {
    const char *corruption;
    size_t rows;
    size_t neural_available_rows;
    size_t symbolic_available_rows;
    size_t neural_failure_rows;
    size_t symbolic_failure_rows;
    double top1;
    double symbolic_top1;
    int top1_available;
    int symbolic_top1_available;
    double metric_sums[NN_GD_HAIKU_METRIC_COUNT];
} NNGDHaikuCorruptionSummary;

static const NNGDSketch *nn_gd_sort_sketches = NULL;
static const NNGDArcPipelineAttempt *nn_gd_sort_arc_attempts = NULL;

static uint64_t nn_gd_fnv1a_update(uint64_t hash, const char *value);

static char *nn_gd_strdup(const char *value) {
    size_t len;
    char *copy;
    if (value == NULL) {
        value = "";
    }
    len = strlen(value);
    copy = (char *)malloc(len + 1U);
    if (copy == NULL) {
        return NULL;
    }
    memcpy(copy, value, len + 1U);
    return copy;
}

static int nn_gd_reserve_strings(NNGDStringList *list, size_t needed) {
    char **items;
    size_t capacity;
    if (needed <= list->capacity) {
        return 0;
    }
    capacity = list->capacity == 0U ? 8U : list->capacity * 2U;
    while (capacity < needed) {
        capacity *= 2U;
    }
    items = (char **)realloc(list->items, capacity * sizeof(char *));
    if (items == NULL) {
        return -1;
    }
    list->items = items;
    list->capacity = capacity;
    return 0;
}

static int nn_gd_string_list_contains(const NNGDStringList *list, const char *value) {
    size_t i;
    for (i = 0; i < list->count; ++i) {
        if (strcmp(list->items[i], value) == 0) {
            return 1;
        }
    }
    return 0;
}

static int nn_gd_string_list_add(NNGDStringList *list, const char *value) {
    char *copy;
    if (value == NULL || value[0] == '\0') {
        return 0;
    }
    if (nn_gd_reserve_strings(list, list->count + 1U) != 0) {
        return -1;
    }
    copy = nn_gd_strdup(value);
    if (copy == NULL) {
        return -1;
    }
    list->items[list->count++] = copy;
    return 0;
}

static int nn_gd_string_list_add_unique(NNGDStringList *list, const char *value) {
    if (value == NULL || value[0] == '\0') {
        return 0;
    }
    if (nn_gd_string_list_contains(list, value)) {
        return 0;
    }
    return nn_gd_string_list_add(list, value);
}

static void nn_gd_string_list_free(NNGDStringList *list) {
    size_t i;
    for (i = 0; i < list->count; ++i) {
        free(list->items[i]);
    }
    free(list->items);
    list->items = NULL;
    list->count = 0U;
    list->capacity = 0U;
}

static int nn_gd_compare_strings(const void *a, const void *b) {
    const char *const *sa = (const char *const *)a;
    const char *const *sb = (const char *const *)b;
    return strcmp(*sa, *sb);
}

static void nn_gd_string_list_sort(NNGDStringList *list) {
    if (list->count > 1U) {
        qsort(list->items, list->count, sizeof(char *), nn_gd_compare_strings);
    }
}

static uint64_t nn_gd_string_hash(const char *value) {
    return nn_gd_fnv1a_update(UINT64_C(1469598103934665603),
                              value == NULL ? "" : value);
}

static void nn_gd_string_table_free(NNGDStringTable *table) {
    if (table == NULL) {
        return;
    }
    free(table->slots);
    table->slots = NULL;
    table->slot_count = 0U;
}

static size_t nn_gd_string_table_slot(const NNGDStringTable *table,
                                      const NNGDStringList *strings,
                                      const char *value,
                                      uint64_t hash) {
    size_t slot = (size_t)(hash & (uint64_t)(table->slot_count - 1U));
    for (;;) {
        size_t string_index = table->slots[slot];
        if (string_index == 0U ||
            strcmp(strings->items[string_index - 1U], value) == 0) {
            return slot;
        }
        slot = (slot + 1U) & (table->slot_count - 1U);
    }
}

static int nn_gd_string_table_rehash(const NNGDStringList *strings,
                                     NNGDStringTable *table,
                                     size_t needed) {
    NNGDStringTable rebuilt = {0};
    size_t i;
    if (needed == 0U) {
        return 0;
    }
    rebuilt.slot_count = table->slot_count == 0U ? 512U : table->slot_count;
    while (needed * 10U > rebuilt.slot_count * 7U) {
        rebuilt.slot_count *= 2U;
    }
    if (rebuilt.slot_count == table->slot_count) {
        return 0;
    }
    rebuilt.slots = (size_t *)calloc(rebuilt.slot_count, sizeof(size_t));
    if (rebuilt.slots == NULL) {
        return -1;
    }
    for (i = 0U; i < strings->count; ++i) {
        uint64_t hash = nn_gd_string_hash(strings->items[i]);
        size_t slot = nn_gd_string_table_slot(&rebuilt, strings, strings->items[i], hash);
        rebuilt.slots[slot] = i + 1U;
    }
    nn_gd_string_table_free(table);
    *table = rebuilt;
    return 0;
}

static int nn_gd_string_table_add_unique(NNGDStringList *strings,
                                         NNGDStringTable *table,
                                         const char *value) {
    uint64_t hash;
    size_t slot;
    if (value == NULL || value[0] == '\0') {
        return 0;
    }
    if (nn_gd_string_table_rehash(strings, table, strings->count + 1U) != 0) {
        return -1;
    }
    hash = nn_gd_string_hash(value);
    slot = nn_gd_string_table_slot(table, strings, value, hash);
    if (table->slots[slot] != 0U) {
        return 0;
    }
    if (nn_gd_string_list_add(strings, value) != 0) {
        return -1;
    }
    table->slots[slot] = strings->count;
    return 0;
}

static long nn_gd_string_table_find(const NNGDStringTable *table,
                                    const NNGDStringList *strings,
                                    const char *value) {
    uint64_t hash;
    size_t slot;
    if (table == NULL || strings == NULL || table->slot_count == 0U ||
        value == NULL || value[0] == '\0') {
        return -1L;
    }
    hash = nn_gd_string_hash(value);
    slot = (size_t)(hash & (uint64_t)(table->slot_count - 1U));
    for (;;) {
        size_t string_index = table->slots[slot];
        if (string_index == 0U) {
            return -1L;
        }
        if (strcmp(strings->items[string_index - 1U], value) == 0) {
            return (long)(string_index - 1U);
        }
        slot = (slot + 1U) & (table->slot_count - 1U);
    }
}

static long nn_gd_parse_long(const char *value) {
    char *end = NULL;
    long parsed;
    if (value == NULL || value[0] == '\0') {
        return 0L;
    }
    parsed = strtol(value, &end, 10);
    if (end == value) {
        return 0L;
    }
    return parsed;
}

static double nn_gd_parse_double(const char *value) {
    char *end = NULL;
    double parsed;
    if (value == NULL || value[0] == '\0') {
        return 0.0;
    }
    parsed = strtod(value, &end);
    if (end == value) {
        return 0.0;
    }
    return parsed;
}

static int nn_gd_parse_bool(const char *value) {
    if (value == NULL || value[0] == '\0') {
        return 0;
    }
    return strcmp(value, "1") == 0 || strcmp(value, "true") == 0 ||
           strcmp(value, "True") == 0;
}

static size_t nn_gd_split_tabs(char *line, char **fields, size_t max_fields) {
    char *p;
    size_t count = 0U;
    if (max_fields == 0U) {
        return 0U;
    }
    fields[count++] = line;
    for (p = line; *p != '\0'; ++p) {
        if (*p == '\n' || *p == '\r') {
            *p = '\0';
            break;
        }
        if (*p == '\t') {
            *p = '\0';
            if (count < max_fields) {
                fields[count++] = p + 1;
            }
        }
    }
    return count;
}

static const char *nn_gd_field(char **fields, size_t count, size_t index) {
    if (index >= count || fields[index] == NULL) {
        return "";
    }
    return fields[index];
}

static void nn_gd_json_write_string(FILE *fp, const char *value) {
    const unsigned char *p;
    fputc('"', fp);
    if (value == NULL) {
        value = "";
    }
    for (p = (const unsigned char *)value; *p != '\0'; ++p) {
        switch (*p) {
            case '"':
                fputs("\\\"", fp);
                break;
            case '\\':
                fputs("\\\\", fp);
                break;
            case '\b':
                fputs("\\b", fp);
                break;
            case '\f':
                fputs("\\f", fp);
                break;
            case '\n':
                fputs("\\n", fp);
                break;
            case '\r':
                fputs("\\r", fp);
                break;
            case '\t':
                fputs("\\t", fp);
                break;
            default:
                if (*p < 0x20U) {
                    fprintf(fp, "\\u%04x", (unsigned)*p);
                } else {
                    fputc((int)*p, fp);
                }
                break;
        }
    }
    fputc('"', fp);
}

static void nn_gd_json_write_string_n(FILE *fp, const char *value, size_t len) {
    char *tmp = (char *)malloc(len + 1U);
    if (tmp == NULL) {
        nn_gd_json_write_string(fp, "");
        return;
    }
    memcpy(tmp, value, len);
    tmp[len] = '\0';
    nn_gd_json_write_string(fp, tmp);
    free(tmp);
}

static void nn_gd_json_write_string_list(FILE *fp, const NNGDStringList *list) {
    size_t i;
    fputc('[', fp);
    for (i = 0; i < list->count; ++i) {
        if (i > 0U) {
            fputc(',', fp);
        }
        nn_gd_json_write_string(fp, list->items[i]);
    }
    fputc(']', fp);
}

static void nn_gd_json_write_pipe_list(FILE *fp, const char *values) {
    const char *start;
    const char *p;
    int first = 1;
    fputc('[', fp);
    if (values != NULL && values[0] != '\0') {
        start = values;
        for (p = values; ; ++p) {
            if (*p == '|' || *p == '\0') {
                if (p > start) {
                    if (!first) {
                        fputc(',', fp);
                    }
                    nn_gd_json_write_string_n(fp, start, (size_t)(p - start));
                    first = 0;
                }
                if (*p == '\0') {
                    break;
                }
                start = p + 1;
            }
        }
    }
    fputc(']', fp);
}

static int nn_gd_family_is_haiku_generation(const char *family) {
    return strcmp(family, "haiku.latent_adjacency_class_transition") == 0 ||
           strcmp(family, "haiku.continuation_viability_constraint") == 0 ||
           strcmp(family, "haiku.line_shape_transition_constraint") == 0;
}

static int nn_gd_family_is_haiku_understanding(const char *family) {
    return strcmp(family, "haiku.line_order_transition_constraint") == 0 ||
           strcmp(family, "haiku.line_source_consistency_constraint") == 0 ||
           strcmp(family, "haiku.syllable_path_constraint") == 0 ||
           strcmp(family, "haiku.repetition_constraint") == 0 ||
           strcmp(family, "haiku.content_cooccurrence_constraint") == 0 ||
           strcmp(family, "haiku.topic_coherence_transition_constraint") == 0;
}

static int nn_gd_corruption_allowed(const char *family, const char *corruption) {
    if (strcmp(family, "haiku.line_order_transition_constraint") == 0) {
        return strcmp(corruption, "line_shuffle") == 0;
    }
    if (strcmp(family, "haiku.line_source_consistency_constraint") == 0) {
        return strcmp(corruption, "line_splice") == 0;
    }
    if (strcmp(family, "haiku.syllable_path_constraint") == 0) {
        return strcmp(corruption, "syllable_break") == 0;
    }
    if (strcmp(family, "haiku.repetition_constraint") == 0) {
        return strcmp(corruption, "repetition") == 0;
    }
    if (strcmp(family, "haiku.content_cooccurrence_constraint") == 0) {
        return strcmp(corruption, "word_salad") == 0;
    }
    if (strcmp(family, "haiku.topic_coherence_transition_constraint") == 0) {
        return strcmp(corruption, "topic_splice") == 0 ||
               strcmp(corruption, "same_season_splice") == 0 ||
               strcmp(corruption, "plausible_topic_splice") == 0;
    }
    return 0;
}

static const char *nn_gd_compiler_for_candidate(const NNGDDeltaCandidate *candidate) {
    if (strcmp(candidate->domain, "arc_agi") == 0) {
        return "arc_missing_family_gap_compiler";
    }
    if (strcmp(candidate->domain, "haiku") == 0) {
        if (strcmp(candidate->family, "haiku.latent_adjacency_class_transition") == 0 ||
            strcmp(candidate->family, "haiku.continuation_viability_constraint") == 0) {
            return "haiku_generation_trace_constraint_compiler";
        }
        if (strcmp(candidate->family, "haiku.line_shape_transition_constraint") == 0) {
            return "haiku_shape_trace_constraint_compiler";
        }
        if (nn_gd_family_is_haiku_understanding(candidate->family)) {
            return "haiku_understanding_symbolic_constraint_compiler";
        }
    }
    return "unsupported";
}

static int nn_gd_candidate_list_reserve(NNGDDeltaCandidateList *list, size_t needed) {
    NNGDDeltaCandidate *items;
    size_t capacity;
    if (needed <= list->capacity) {
        return 0;
    }
    capacity = list->capacity == 0U ? 8U : list->capacity * 2U;
    while (capacity < needed) {
        capacity *= 2U;
    }
    items = (NNGDDeltaCandidate *)realloc(list->items,
                                          capacity * sizeof(NNGDDeltaCandidate));
    if (items == NULL) {
        return -1;
    }
    list->items = items;
    list->capacity = capacity;
    return 0;
}

static NNGDDeltaCandidate *nn_gd_find_candidate(NNGDDeltaCandidateList *list,
                                                const char *candidate_id) {
    size_t i;
    for (i = 0; i < list->count; ++i) {
        if (strcmp(list->items[i].candidate_id, candidate_id) == 0) {
            return &list->items[i];
        }
    }
    return NULL;
}

static NNGDDeltaCandidate *nn_gd_add_candidate(NNGDDeltaCandidateList *list,
                                               const char *candidate_id,
                                               const char *domain,
                                               const char *family,
                                               const char *source_artifacts,
                                               const char *feature_summary_json) {
    NNGDDeltaCandidate *candidate;
    if (nn_gd_candidate_list_reserve(list, list->count + 1U) != 0) {
        return NULL;
    }
    candidate = &list->items[list->count];
    memset(candidate, 0, sizeof(*candidate));
    candidate->candidate_id = nn_gd_strdup(candidate_id);
    candidate->domain = nn_gd_strdup(domain);
    candidate->family = nn_gd_strdup(family);
    candidate->source_artifacts = nn_gd_strdup(source_artifacts);
    candidate->feature_summary_json = nn_gd_strdup(feature_summary_json);
    if (candidate->candidate_id == NULL || candidate->domain == NULL ||
        candidate->family == NULL || candidate->source_artifacts == NULL ||
        candidate->feature_summary_json == NULL) {
        return NULL;
    }
    ++list->count;
    return candidate;
}

static void nn_gd_counterexample_free(NNGDCounterexample *counterexample) {
    free(counterexample->task_id);
    free(counterexample->split);
    free(counterexample->example_id);
    free(counterexample->corruption);
    free(counterexample->symbolic_selected);
    nn_gd_string_list_free(&counterexample->failed_predicates);
    memset(counterexample, 0, sizeof(*counterexample));
}

static void nn_gd_candidate_list_free(NNGDDeltaCandidateList *list) {
    size_t i;
    size_t j;
    for (i = 0; i < list->count; ++i) {
        NNGDDeltaCandidate *candidate = &list->items[i];
        free(candidate->candidate_id);
        free(candidate->domain);
        free(candidate->family);
        free(candidate->source_artifacts);
        free(candidate->feature_summary_json);
        nn_gd_string_list_free(&candidate->task_ids);
        nn_gd_string_list_free(&candidate->adjacency_hashes);
        for (j = 0; j < candidate->counterexamples_stored; ++j) {
            nn_gd_counterexample_free(&candidate->counterexamples[j]);
        }
    }
    free(list->items);
    list->items = NULL;
    list->count = 0U;
    list->capacity = 0U;
}

static int nn_gd_candidate_add_counterexample(NNGDDeltaCandidate *candidate,
                                              NNGDCounterexampleKind kind,
                                              const char *task_id,
                                              const char *split,
                                              const char *example_id,
                                              long sample,
                                              long step,
                                              long candidate_count,
                                              long poem_index,
                                              const char *corruption,
                                              const char *symbolic_selected,
                                              NNGDStringList *failed_predicates) {
    NNGDCounterexample *counterexample;
    ++candidate->counterexamples_total;
    if (candidate->counterexamples_stored >= NN_GD_COUNTEREXAMPLE_LIMIT) {
        nn_gd_string_list_free(failed_predicates);
        return 0;
    }
    counterexample = &candidate->counterexamples[candidate->counterexamples_stored];
    memset(counterexample, 0, sizeof(*counterexample));
    counterexample->kind = kind;
    counterexample->task_id = nn_gd_strdup(task_id);
    counterexample->split = nn_gd_strdup(split);
    counterexample->example_id = nn_gd_strdup(example_id);
    counterexample->sample = sample;
    counterexample->step = step;
    counterexample->candidate_count = candidate_count;
    counterexample->poem_index = poem_index;
    counterexample->corruption = nn_gd_strdup(corruption);
    counterexample->symbolic_selected = nn_gd_strdup(symbolic_selected);
    if (counterexample->task_id == NULL || counterexample->split == NULL ||
        counterexample->example_id == NULL || counterexample->corruption == NULL ||
        counterexample->symbolic_selected == NULL) {
        return -1;
    }
    counterexample->failed_predicates = *failed_predicates;
    failed_predicates->items = NULL;
    failed_predicates->count = 0U;
    failed_predicates->capacity = 0U;
    ++candidate->counterexamples_stored;
    return 0;
}

static int nn_gd_process_candidate_row(NNGDDeltaCandidate *candidate,
                                       char **fields,
                                       size_t field_count) {
    const char *family = candidate->family;
    const char *task_id = nn_gd_field(fields, field_count, 5U);
    const char *split = nn_gd_field(fields, field_count, 6U);
    const char *example_id = nn_gd_field(fields, field_count, 7U);
    long sample = nn_gd_parse_long(nn_gd_field(fields, field_count, 8U));
    long step = nn_gd_parse_long(nn_gd_field(fields, field_count, 9U));
    long candidate_count = nn_gd_parse_long(nn_gd_field(fields, field_count, 10U));
    int strict_adjacency = nn_gd_parse_bool(nn_gd_field(fields, field_count, 11U));
    long rejected_adjacency = nn_gd_parse_long(nn_gd_field(fields, field_count, 12U));
    long rejected_dead_end = nn_gd_parse_long(nn_gd_field(fields, field_count, 13U));
    long rejected_syllable = nn_gd_parse_long(nn_gd_field(fields, field_count, 14U));
    long rejected_line_chars = nn_gd_parse_long(nn_gd_field(fields, field_count, 15U));
    long accepted_existing_rows = nn_gd_parse_long(nn_gd_field(fields, field_count, 16U));
    int missing_family_evidence = nn_gd_parse_bool(nn_gd_field(fields, field_count, 17U));
    int promotion_replayed = nn_gd_parse_bool(nn_gd_field(fields, field_count, 18U));
    long promotion_train_consistent =
        nn_gd_parse_long(nn_gd_field(fields, field_count, 19U));
    double promotion_generated =
        nn_gd_parse_double(nn_gd_field(fields, field_count, 20U));
    long mutation_rows = nn_gd_parse_long(nn_gd_field(fields, field_count, 21U));
    long poem_index = nn_gd_parse_long(nn_gd_field(fields, field_count, 22U));
    const char *corruption = nn_gd_field(fields, field_count, 23U);
    const char *symbolic_selected = nn_gd_field(fields, field_count, 24U);
    const char *adjacency_hash = nn_gd_field(fields, field_count, 25U);
    NNGDStringList failed_predicates = {0};
    int covered = 1;

    ++candidate->support_examples;
    candidate->promotion_proposals_generated_sum += promotion_generated;
    candidate->mutation_train_consistent_rows += mutation_rows;
    if (nn_gd_string_list_add_unique(&candidate->task_ids, task_id) != 0 ||
        nn_gd_string_list_add_unique(&candidate->adjacency_hashes, adjacency_hash) != 0) {
        return -1;
    }

    if (strcmp(candidate->domain, "arc_agi") == 0) {
        if (accepted_existing_rows != 0L &&
            nn_gd_string_list_add(&failed_predicates, "accepted_existing_rows") != 0) {
            return -1;
        }
        if (!missing_family_evidence &&
            nn_gd_string_list_add(&failed_predicates, "missing_family_evidence") != 0) {
            return -1;
        }
        if (!promotion_replayed &&
            nn_gd_string_list_add(&failed_predicates, "promotion_replayed") != 0) {
            return -1;
        }
        if (promotion_train_consistent != 0L &&
            nn_gd_string_list_add(&failed_predicates,
                                  "promotion_proposals_train_consistent_pre_cap") != 0) {
            return -1;
        }
        if (failed_predicates.count > 0U) {
            if (nn_gd_candidate_add_counterexample(candidate,
                                                   NN_GD_CE_ARC,
                                                   task_id,
                                                   split,
                                                   "",
                                                   0L,
                                                   0L,
                                                   0L,
                                                   0L,
                                                   "",
                                                   "",
                                                   &failed_predicates) != 0) {
                return -1;
            }
        } else {
            nn_gd_string_list_free(&failed_predicates);
        }
        return 0;
    }

    if (strcmp(candidate->domain, "haiku") != 0) {
        return 0;
    }

    if (strcmp(family, "haiku.latent_adjacency_class_transition") == 0) {
        covered = strict_adjacency && candidate_count <= 2L && rejected_adjacency > 0L;
    } else if (strcmp(family, "haiku.continuation_viability_constraint") == 0) {
        covered = candidate_count <= 1L || rejected_dead_end > 0L;
    } else if (strcmp(family, "haiku.line_shape_transition_constraint") == 0) {
        covered = rejected_syllable + rejected_line_chars > 0L;
    } else if (nn_gd_family_is_haiku_understanding(family)) {
        if (!nn_gd_corruption_allowed(family, corruption) &&
            nn_gd_string_list_add(&failed_predicates, "corruption_family") != 0) {
            return -1;
        }
        if (strcmp(symbolic_selected, "real") != 0 &&
            nn_gd_string_list_add(&failed_predicates, "symbolic_selected") != 0) {
            return -1;
        }
        if (failed_predicates.count > 0U) {
            if (nn_gd_candidate_add_counterexample(candidate,
                                                   NN_GD_CE_HAIKU_UNDERSTANDING,
                                                   "",
                                                   "",
                                                   example_id,
                                                   0L,
                                                   0L,
                                                   0L,
                                                   poem_index,
                                                   corruption,
                                                   symbolic_selected,
                                                   &failed_predicates) != 0) {
                return -1;
            }
        } else {
            nn_gd_string_list_free(&failed_predicates);
        }
        return 0;
    }

    if (!covered) {
        if (nn_gd_candidate_add_counterexample(candidate,
                                               NN_GD_CE_HAIKU_GENERATION,
                                               "",
                                               "",
                                               example_id,
                                               sample,
                                               step,
                                               candidate_count,
                                               0L,
                                               "",
                                               "",
                                               &failed_predicates) != 0) {
            return -1;
        }
    } else {
        nn_gd_string_list_free(&failed_predicates);
    }
    return 0;
}

static int nn_gd_read_candidate_rows(const char *path, NNGDDeltaCandidateList *candidates) {
    FILE *fp;
    char line[NN_GD_LINE_MAX];
    char *fields[NN_GD_MAX_FIELDS];
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    while (fgets(line, sizeof(line), fp) != NULL) {
        size_t field_count = nn_gd_split_tabs(line, fields, NN_GD_MAX_FIELDS);
        const char *candidate_id = nn_gd_field(fields, field_count, 0U);
        NNGDDeltaCandidate *candidate;
        if (candidate_id[0] == '\0' || strcmp(candidate_id, "candidate_id") == 0) {
            continue;
        }
        candidate = nn_gd_find_candidate(candidates, candidate_id);
        if (candidate == NULL) {
            candidate = nn_gd_add_candidate(candidates,
                                            candidate_id,
                                            nn_gd_field(fields, field_count, 1U),
                                            nn_gd_field(fields, field_count, 2U),
                                            nn_gd_field(fields, field_count, 3U),
                                            nn_gd_field(fields, field_count, 4U));
            if (candidate == NULL) {
                fclose(fp);
                return -1;
            }
        }
        if (nn_gd_process_candidate_row(candidate, fields, field_count) != 0) {
            fclose(fp);
            return -1;
        }
    }
    fclose(fp);
    return 0;
}

static void nn_gd_write_string_list_literal(FILE *fp,
                                            const char *const *items,
                                            size_t count) {
    size_t i;
    fputc('[', fp);
    for (i = 0; i < count; ++i) {
        if (i > 0U) {
            fputc(',', fp);
        }
        nn_gd_json_write_string(fp, items[i]);
    }
    fputc(']', fp);
}

static void nn_gd_write_corruption_list(FILE *fp, const char *family) {
    static const char *topic[] = {
        "plausible_topic_splice",
        "same_season_splice",
        "topic_splice",
    };
    if (strcmp(family, "haiku.line_order_transition_constraint") == 0) {
        static const char *items[] = {"line_shuffle"};
        nn_gd_write_string_list_literal(fp, items, 1U);
    } else if (strcmp(family, "haiku.line_source_consistency_constraint") == 0) {
        static const char *items[] = {"line_splice"};
        nn_gd_write_string_list_literal(fp, items, 1U);
    } else if (strcmp(family, "haiku.syllable_path_constraint") == 0) {
        static const char *items[] = {"syllable_break"};
        nn_gd_write_string_list_literal(fp, items, 1U);
    } else if (strcmp(family, "haiku.repetition_constraint") == 0) {
        static const char *items[] = {"repetition"};
        nn_gd_write_string_list_literal(fp, items, 1U);
    } else if (strcmp(family, "haiku.content_cooccurrence_constraint") == 0) {
        static const char *items[] = {"word_salad"};
        nn_gd_write_string_list_literal(fp, items, 1U);
    } else {
        nn_gd_write_string_list_literal(fp, topic, 3U);
    }
}

static void nn_gd_write_compiled_program(FILE *fp, const NNGDDeltaCandidate *candidate) {
    const char *arc_predicates[] = {
        "accepted_existing_rows == 0",
        "missing_family_evidence == true",
        "promotion_replayed == true",
        "promotion_proposals_train_consistent_pre_cap == 0",
    };
    const char *generation_gates[] = {
        "dataset_vocab",
        "dataset_adjacency",
        "haiku_shape_proxy",
    };
    const char *shape_gates[] = {"haiku_shape_proxy"};
    const char *understanding_predicates[] = {"symbolic_selected == real"};

    if (strcmp(candidate->domain, "arc_agi") == 0) {
        fputs("{\"runtime\":\"c_trace_predicate_v1\",\"verifier\":", fp);
        nn_gd_json_write_string(fp, "arc_failure_proof_gap_replay");
        fputs(",\"family\":", fp);
        nn_gd_json_write_string(fp, candidate->family);
        fputs(",\"required_predicates\":", fp);
        nn_gd_write_string_list_literal(fp, arc_predicates, 4U);
        fputs(",\"support_task_ids\":", fp);
        nn_gd_json_write_string_list(fp, &candidate->task_ids);
        fputs(",\"promotion_scope\":\"diagnostic_gap_certificate\"}", fp);
        return;
    }

    if (strcmp(candidate->domain, "haiku") != 0) {
        fputs("{}", fp);
        return;
    }

    if (strcmp(candidate->family, "haiku.latent_adjacency_class_transition") == 0 ||
        strcmp(candidate->family, "haiku.continuation_viability_constraint") == 0) {
        fputs("{\"runtime\":\"c_trace_predicate_v1\",\"verifier\":", fp);
        nn_gd_json_write_string(fp, "haiku_word_generation_trace_replay");
        fputs(",\"family\":", fp);
        nn_gd_json_write_string(fp, candidate->family);
        fputs(",\"adjacency_hashes\":", fp);
        nn_gd_json_write_string_list(fp, &candidate->adjacency_hashes);
        fputs(",\"required_gates\":", fp);
        nn_gd_write_string_list_literal(fp, generation_gates, 3U);
        fputc('}', fp);
    } else if (strcmp(candidate->family, "haiku.line_shape_transition_constraint") == 0) {
        fputs("{\"runtime\":\"c_trace_predicate_v1\",\"verifier\":", fp);
        nn_gd_json_write_string(fp, "haiku_word_generation_trace_replay");
        fputs(",\"family\":", fp);
        nn_gd_json_write_string(fp, candidate->family);
        fputs(",\"required_gates\":", fp);
        nn_gd_write_string_list_literal(fp, shape_gates, 1U);
        fputc('}', fp);
    } else if (nn_gd_family_is_haiku_understanding(candidate->family)) {
        fputs("{\"runtime\":\"c_trace_predicate_v1\",\"verifier\":", fp);
        nn_gd_json_write_string(fp, "haiku_understanding_symbolic_replay");
        fputs(",\"family\":", fp);
        nn_gd_json_write_string(fp, candidate->family);
        fputs(",\"corruptions\":", fp);
        nn_gd_write_corruption_list(fp, candidate->family);
        fputs(",\"required_predicates\":", fp);
        nn_gd_write_string_list_literal(fp, understanding_predicates, 1U);
        fputc('}', fp);
    } else {
        fputs("{}", fp);
    }
}

static int nn_gd_write_compiled_outputs(FILE *fp, NNGDDeltaCandidateList *candidates) {
    size_t i;
    for (i = 0; i < candidates->count; ++i) {
        NNGDDeltaCandidate *candidate = &candidates->items[i];
        const char *compiler = nn_gd_compiler_for_candidate(candidate);
        int executable = strcmp(compiler, "unsupported") != 0;
        nn_gd_string_list_sort(&candidate->task_ids);
        nn_gd_string_list_sort(&candidate->adjacency_hashes);
        fputs("{\"schema\":\"crlp.neural_grammar_discovery.compiled_delta.v1\","
              "\"report_version\":1,\"candidate_id\":",
              fp);
        nn_gd_json_write_string(fp, candidate->candidate_id);
        fputs(",\"domain\":", fp);
        nn_gd_json_write_string(fp, candidate->domain);
        fputs(",\"family\":", fp);
        nn_gd_json_write_string(fp, candidate->family);
        fputs(",\"compiler\":", fp);
        nn_gd_json_write_string(fp, compiler);
        fputs(",\"executable\":", fp);
        fputs(executable ? "true" : "false", fp);
        fputs(",\"program\":", fp);
        nn_gd_write_compiled_program(fp, candidate);
        fputs(",\"source_artifacts\":", fp);
        nn_gd_json_write_pipe_list(fp, candidate->source_artifacts);
        fputs("}\n", fp);
    }
    return ferror(fp) ? -1 : 0;
}

static void nn_gd_summary_comma(FILE *fp, int *first) {
    if (!*first) {
        fputc(',', fp);
    }
    *first = 0;
}

static void nn_gd_write_feature_summary_fields(FILE *fp, const char *json, int *first) {
    const char *start;
    const char *end;
    if (json == NULL) {
        return;
    }
    start = json;
    while (isspace((unsigned char)*start)) {
        ++start;
    }
    if (*start != '{') {
        return;
    }
    ++start;
    end = json + strlen(json);
    while (end > start && isspace((unsigned char)end[-1])) {
        --end;
    }
    if (end <= start || end[-1] != '}') {
        return;
    }
    --end;
    while (end > start && isspace((unsigned char)end[-1])) {
        --end;
    }
    if (end <= start) {
        return;
    }
    nn_gd_summary_comma(fp, first);
    fwrite(start, 1U, (size_t)(end - start), fp);
}

static void nn_gd_write_counterexample(FILE *fp, const NNGDCounterexample *counterexample) {
    if (counterexample->kind == NN_GD_CE_ARC) {
        fputs("{\"task_id\":", fp);
        nn_gd_json_write_string(fp, counterexample->task_id);
        fputs(",\"split\":", fp);
        nn_gd_json_write_string(fp, counterexample->split);
        fputs(",\"failed_predicates\":", fp);
        nn_gd_json_write_string_list(fp, &counterexample->failed_predicates);
        fputc('}', fp);
        return;
    }
    if (counterexample->kind == NN_GD_CE_HAIKU_GENERATION) {
        fputs("{\"example_id\":", fp);
        nn_gd_json_write_string(fp, counterexample->example_id);
        fprintf(fp,
                ",\"sample\":%ld,\"step\":%ld,\"candidate_count\":%ld}",
                counterexample->sample,
                counterexample->step,
                counterexample->candidate_count);
        return;
    }
    fputs("{\"example_id\":", fp);
    nn_gd_json_write_string(fp, counterexample->example_id);
    fprintf(fp, ",\"poem_index\":%ld,\"corruption\":", counterexample->poem_index);
    nn_gd_json_write_string(fp, counterexample->corruption);
    fputs(",\"symbolic_selected\":", fp);
    nn_gd_json_write_string(fp, counterexample->symbolic_selected);
    fputs(",\"failed_predicates\":", fp);
    nn_gd_json_write_string_list(fp, &counterexample->failed_predicates);
    fputc('}', fp);
}

static void nn_gd_write_counterexample_rows(FILE *fp, const NNGDDeltaCandidate *candidate) {
    size_t i;
    fputc('[', fp);
    for (i = 0; i < candidate->counterexamples_stored; ++i) {
        if (i > 0U) {
            fputc(',', fp);
        }
        nn_gd_write_counterexample(fp, &candidate->counterexamples[i]);
    }
    fputc(']', fp);
}

static void nn_gd_write_verification_summary(FILE *fp,
                                             const NNGDDeltaCandidate *candidate,
                                             const char *verifier,
                                             const char *promotion_scope,
                                             int include_arc_metrics) {
    int first = 1;
    double mean_generated = 0.0;
    fputc('{', fp);
    nn_gd_write_feature_summary_fields(fp, candidate->feature_summary_json, &first);
    nn_gd_summary_comma(fp, &first);
    fprintf(fp, "\"support_examples\":%zu", candidate->support_examples);
    nn_gd_summary_comma(fp, &first);
    fprintf(fp, "\"counterexamples\":%zu", candidate->counterexamples_total);
    nn_gd_summary_comma(fp, &first);
    fputs("\"counterexample_rows\":", fp);
    nn_gd_write_counterexample_rows(fp, candidate);
    if (promotion_scope != NULL && promotion_scope[0] != '\0') {
        nn_gd_summary_comma(fp, &first);
        fputs("\"promotion_scope\":", fp);
        nn_gd_json_write_string(fp, promotion_scope);
    }
    if (include_arc_metrics) {
        if (candidate->support_examples > 0U) {
            mean_generated = candidate->promotion_proposals_generated_sum /
                             (double)candidate->support_examples;
        }
        nn_gd_summary_comma(fp, &first);
        fprintf(fp, "\"mean_promotion_proposals_generated\":%.17g", mean_generated);
        nn_gd_summary_comma(fp, &first);
        fprintf(fp,
                "\"mutation_train_consistent_rows\":%ld",
                candidate->mutation_train_consistent_rows);
    }
    if (verifier != NULL && verifier[0] != '\0') {
        nn_gd_summary_comma(fp, &first);
        fputs("\"verifier\":", fp);
        nn_gd_json_write_string(fp, verifier);
    }
    fputc('}', fp);
}

static void nn_gd_write_verification_record(FILE *fp, const NNGDDeltaCandidate *candidate) {
    const char *compiler = nn_gd_compiler_for_candidate(candidate);
    const char *verifier_status = "unknown_domain";
    const char *blocking_reason = "no domain verifier adapter";
    const char *verifier = "";
    const char *promotion_scope = "";
    int include_arc_metrics = 0;
    int certified = 0;
    int executable = strcmp(compiler, "unsupported") != 0;

    if (!executable) {
        verifier_status = "no_executable_compiler";
        blocking_reason = "no executable compiler for proposed grammar delta";
    } else if (strcmp(candidate->domain, "arc_agi") == 0) {
        include_arc_metrics = 1;
        promotion_scope = "diagnostic_gap_certificate";
        if (candidate->counterexamples_total > 0U) {
            verifier_status = "arc_gap_certificate_counterexamples";
            blocking_reason = "compiled ARC diagnostic gap program failed replay predicates";
        } else {
            verifier_status = "arc_gap_certificate_replayed";
            blocking_reason = "";
            certified = 1;
        }
    } else if (strcmp(candidate->domain, "haiku") == 0 &&
               nn_gd_family_is_haiku_generation(candidate->family)) {
        verifier = "haiku_word_generation_trace_replay";
        if (candidate->counterexamples_total > 0U) {
            verifier_status = "haiku_generation_constraint_counterexamples";
            blocking_reason =
                "compiled haiku generation constraint did not cover every support row";
        } else {
            verifier_status = "haiku_generation_constraint_replayed";
            blocking_reason = "";
            certified = 1;
        }
    } else if (strcmp(candidate->domain, "haiku") == 0 &&
               nn_gd_family_is_haiku_understanding(candidate->family)) {
        verifier = "haiku_understanding_symbolic_replay";
        if (candidate->counterexamples_total > 0U) {
            verifier_status = "haiku_understanding_symbolic_counterexamples";
            blocking_reason = "existing symbolic verifier does not certify every support row";
        } else {
            verifier_status = "haiku_understanding_symbolic_replayed";
            blocking_reason = "";
            certified = 1;
        }
    }

    fputs("{\"schema\":\"crlp.neural_grammar_discovery.delta_verification.v1\","
          "\"report_version\":1,\"candidate_id\":",
          fp);
    nn_gd_json_write_string(fp, candidate->candidate_id);
    fputs(",\"domain\":", fp);
    nn_gd_json_write_string(fp, candidate->domain);
    fputs(",\"family\":", fp);
    nn_gd_json_write_string(fp, candidate->family);
    fputs(",\"compiled_delta_schema\":\"crlp.neural_grammar_discovery.compiled_delta.v1\","
          "\"compiler\":",
          fp);
    nn_gd_json_write_string(fp, compiler);
    fputs(",\"verifier_status\":", fp);
    nn_gd_json_write_string(fp, verifier_status);
    fputs(",\"certified\":", fp);
    fputs(certified ? "true" : "false", fp);
    fputs(",\"promoted\":", fp);
    fputs(certified ? "true" : "false", fp);
    fprintf(fp, ",\"support_examples\":%zu,\"source_artifacts\":", candidate->support_examples);
    nn_gd_json_write_pipe_list(fp, candidate->source_artifacts);
    fputs(",\"blocking_reason\":", fp);
    nn_gd_json_write_string(fp, blocking_reason);
    fputs(",\"verification_summary\":", fp);
    nn_gd_write_verification_summary(fp,
                                     candidate,
                                     verifier,
                                     promotion_scope,
                                     include_arc_metrics);
    fputs("}\n", fp);
}

static int nn_gd_write_verification_outputs(FILE *fp, const NNGDDeltaCandidateList *candidates) {
    size_t i;
    for (i = 0; i < candidates->count; ++i) {
        nn_gd_write_verification_record(fp, &candidates->items[i]);
    }
    return ferror(fp) ? -1 : 0;
}

int nn_grammar_discovery_write_delta_outputs(const char *candidate_rows_path,
                                             const char *compiled_output_path,
                                             const char *verification_output_path) {
    NNGDDeltaCandidateList candidates = {0};
    FILE *compiled = NULL;
    FILE *verification = NULL;
    int status = -1;

    if (nn_gd_read_candidate_rows(candidate_rows_path, &candidates) != 0) {
        goto done;
    }
    compiled = fopen(compiled_output_path, "w");
    if (compiled == NULL) {
        goto done;
    }
    verification = fopen(verification_output_path, "w");
    if (verification == NULL) {
        goto done;
    }
    if (nn_gd_write_compiled_outputs(compiled, &candidates) != 0 ||
        nn_gd_write_verification_outputs(verification, &candidates) != 0) {
        goto done;
    }
    status = 0;

done:
    if (compiled != NULL) {
        fclose(compiled);
    }
    if (verification != NULL) {
        fclose(verification);
    }
    nn_gd_candidate_list_free(&candidates);
    return status;
}

static int nn_gd_add_token_pair(NNGDStringList *tokens,
                                const char *prefix,
                                const char *value) {
    size_t prefix_len;
    size_t value_len;
    char *token;
    if (value == NULL || value[0] == '\0') {
        return 0;
    }
    prefix_len = strlen(prefix);
    value_len = strlen(value);
    token = (char *)malloc(prefix_len + value_len + 1U);
    if (token == NULL) {
        return -1;
    }
    memcpy(token, prefix, prefix_len);
    memcpy(token + prefix_len, value, value_len + 1U);
    if (nn_gd_string_list_add_unique(tokens, token) != 0) {
        free(token);
        return -1;
    }
    free(token);
    return 0;
}

static int nn_gd_add_pipe_context_tokens(NNGDStringList *tokens,
                                         const char *prefix,
                                         const char *values) {
    char *copy;
    char *start;
    char *p;
    if (values == NULL || values[0] == '\0') {
        return 0;
    }
    copy = nn_gd_strdup(values);
    if (copy == NULL) {
        return -1;
    }
    start = copy;
    for (p = copy; ; ++p) {
        if (*p == '|' || *p == '\0') {
            char saved = *p;
            *p = '\0';
            if (start[0] != '\0' && nn_gd_add_token_pair(tokens, prefix, start) != 0) {
                free(copy);
                return -1;
            }
            if (saved == '\0') {
                break;
            }
            start = p + 1;
        }
    }
    free(copy);
    return 0;
}

static int nn_gd_add_atom_tokens(NNGDStringList *tokens, const char *atoms) {
    char *copy;
    char *start;
    char *p;
    if (atoms == NULL || atoms[0] == '\0') {
        return 0;
    }
    copy = nn_gd_strdup(atoms);
    if (copy == NULL) {
        return -1;
    }
    start = copy;
    for (p = copy; ; ++p) {
        if (*p == '|' || *p == '\0') {
            char saved = *p;
            *p = '\0';
            if (start[0] != '\0' && nn_gd_add_token_pair(tokens, "atom:", start) != 0) {
                free(copy);
                return -1;
            }
            if (saved == '\0') {
                break;
            }
            start = p + 1;
        }
    }
    free(copy);
    return 0;
}

static int nn_gd_add_slot_tokens(NNGDStringList *tokens, const char *slots) {
    char *copy;
    char *start;
    char *p;
    if (slots == NULL || slots[0] == '\0') {
        return 0;
    }
    copy = nn_gd_strdup(slots);
    if (copy == NULL) {
        return -1;
    }
    start = copy;
    for (p = copy; ; ++p) {
        if (*p == '|' || *p == '\0') {
            char saved = *p;
            char *equals;
            *p = '\0';
            equals = strchr(start, '=');
            if (equals != NULL) {
                *equals = '\0';
            }
            if (start[0] != '\0') {
                if (nn_gd_add_token_pair(tokens, "slot:", start) != 0) {
                    free(copy);
                    return -1;
                }
                if (equals != NULL && equals[1] != '\0') {
                    char *pair;
                    size_t name_len = strlen(start);
                    size_t value_len = strlen(equals + 1);
                    pair = (char *)malloc(name_len + value_len + 2U);
                    if (pair == NULL) {
                        free(copy);
                        return -1;
                    }
                    memcpy(pair, start, name_len);
                    pair[name_len] = '=';
                    memcpy(pair + name_len + 1U, equals + 1, value_len + 1U);
                    if (nn_gd_add_token_pair(tokens, "slot:", pair) != 0) {
                        free(pair);
                        free(copy);
                        return -1;
                    }
                    free(pair);
                }
            }
            if (saved == '\0') {
                break;
            }
            start = p + 1;
        }
    }
    free(copy);
    return 0;
}

static int nn_gd_sketch_list_reserve(NNGDSketchList *list, size_t needed) {
    NNGDSketch *items;
    size_t capacity;
    if (needed <= list->capacity) {
        return 0;
    }
    capacity = list->capacity == 0U ? 16U : list->capacity * 2U;
    while (capacity < needed) {
        capacity *= 2U;
    }
    items = (NNGDSketch *)realloc(list->items, capacity * sizeof(NNGDSketch));
    if (items == NULL) {
        return -1;
    }
    list->items = items;
    list->capacity = capacity;
    return 0;
}

static int nn_gd_sketch_list_add(NNGDSketchList *list,
                                 const char *sketch_id,
                                 const char *domain,
                                 const char *family,
                                 const char *compiler_target,
                                 const char *atoms,
                                 const char *slots,
                                 const char *composition) {
    NNGDSketch *sketch;
    if (nn_gd_sketch_list_reserve(list, list->count + 1U) != 0) {
        return -1;
    }
    sketch = &list->items[list->count];
    memset(sketch, 0, sizeof(*sketch));
    sketch->sketch_id = nn_gd_strdup(sketch_id);
    sketch->domain = nn_gd_strdup(domain);
    sketch->family = nn_gd_strdup(family);
    if (sketch->sketch_id == NULL || sketch->domain == NULL || sketch->family == NULL) {
        return -1;
    }
    if (nn_gd_add_token_pair(&sketch->tokens, "domain:", domain) != 0 ||
        nn_gd_add_token_pair(&sketch->tokens, "family:", family) != 0 ||
        nn_gd_add_token_pair(&sketch->tokens, "compiler:", compiler_target) != 0 ||
        nn_gd_add_atom_tokens(&sketch->tokens, atoms) != 0 ||
        nn_gd_add_slot_tokens(&sketch->tokens, slots) != 0 ||
        nn_gd_add_pipe_context_tokens(&sketch->tokens,
                                      "composition:",
                                      composition) != 0) {
        return -1;
    }
    ++list->count;
    return 0;
}

static void nn_gd_sketch_list_free(NNGDSketchList *list) {
    size_t i;
    for (i = 0; i < list->count; ++i) {
        free(list->items[i].sketch_id);
        free(list->items[i].domain);
        free(list->items[i].family);
        nn_gd_string_list_free(&list->items[i].tokens);
    }
    free(list->items);
    list->items = NULL;
    list->count = 0U;
    list->capacity = 0U;
}

static int nn_gd_read_sketches(const char *path, NNGDSketchList *sketches) {
    FILE *fp;
    char line[NN_GD_LINE_MAX];
    char *fields[NN_GD_MAX_FIELDS];
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    while (fgets(line, sizeof(line), fp) != NULL) {
        size_t field_count = nn_gd_split_tabs(line, fields, NN_GD_MAX_FIELDS);
        const char *sketch_id = nn_gd_field(fields, field_count, 0U);
        if (sketch_id[0] == '\0' || strcmp(sketch_id, "sketch_id") == 0) {
            continue;
        }
        if (nn_gd_sketch_list_add(sketches,
                                  sketch_id,
                                  nn_gd_field(fields, field_count, 1U),
                                  nn_gd_field(fields, field_count, 2U),
                                  nn_gd_field(fields, field_count, 3U),
                                  nn_gd_field(fields, field_count, 4U),
                                  nn_gd_field(fields, field_count, 5U),
                                  nn_gd_field(fields, field_count, 6U)) != 0) {
            fclose(fp);
            return -1;
        }
    }
    fclose(fp);
    return 0;
}

static long nn_gd_find_sketch(const NNGDSketchList *sketches, const char *sketch_id) {
    size_t i;
    for (i = 0; i < sketches->count; ++i) {
        if (strcmp(sketches->items[i].sketch_id, sketch_id) == 0) {
            return (long)i;
        }
    }
    return -1L;
}

static int nn_gd_label_list_reserve(NNGDLabelList *list, size_t needed) {
    NNGDLabel *items;
    size_t capacity;
    if (needed <= list->capacity) {
        return 0;
    }
    capacity = list->capacity == 0U ? 32U : list->capacity * 2U;
    while (capacity < needed) {
        capacity *= 2U;
    }
    items = (NNGDLabel *)realloc(list->items, capacity * sizeof(NNGDLabel));
    if (items == NULL) {
        return -1;
    }
    list->items = items;
    list->capacity = capacity;
    return 0;
}

static int nn_gd_label_list_add(NNGDLabelList *labels, size_t sketch_index, double label) {
    if (nn_gd_label_list_reserve(labels, labels->count + 1U) != 0) {
        return -1;
    }
    labels->items[labels->count].sketch_index = sketch_index;
    labels->items[labels->count].label = label;
    ++labels->count;
    return 0;
}

static void nn_gd_label_list_free(NNGDLabelList *labels) {
    free(labels->items);
    labels->items = NULL;
    labels->count = 0U;
    labels->capacity = 0U;
}

static int nn_gd_read_labels(const char *path,
                             const NNGDSketchList *sketches,
                             NNGDLabelList *labels) {
    FILE *fp;
    char line[NN_GD_LINE_MAX];
    char *fields[NN_GD_MAX_FIELDS];
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    while (fgets(line, sizeof(line), fp) != NULL) {
        size_t field_count = nn_gd_split_tabs(line, fields, NN_GD_MAX_FIELDS);
        const char *sketch_id = nn_gd_field(fields, field_count, 0U);
        const char *outcome = nn_gd_field(fields, field_count, 1U);
        long sketch_index;
        double label;
        if (sketch_id[0] == '\0' || strcmp(sketch_id, "sketch_id") == 0) {
            continue;
        }
        if (strcmp(outcome, "accepted") == 0) {
            label = 1.0;
        } else if (strcmp(outcome, "diagnostic_positive") == 0) {
            label = 0.75;
        } else if (strcmp(outcome, "rejected") == 0 ||
                   strcmp(outcome, "unsafe") == 0 ||
                   strcmp(outcome, "no_support") == 0) {
            label = 0.0;
        } else {
            continue;
        }
        sketch_index = nn_gd_find_sketch(sketches, sketch_id);
        if (sketch_index < 0L) {
            continue;
        }
        if (nn_gd_add_pipe_context_tokens(&sketches->items[sketch_index].tokens,
                                          "frame:",
                                          nn_gd_field(fields, field_count, 2U)) != 0) {
            fclose(fp);
            return -1;
        }
        if (nn_gd_label_list_add(labels, (size_t)sketch_index, label) != 0) {
            fclose(fp);
            return -1;
        }
    }
    fclose(fp);
    return 0;
}

static long nn_gd_token_index(const NNGDStringList *vocab, const char *token) {
    size_t low = 0U;
    size_t high = vocab->count;
    while (low < high) {
        size_t mid = low + (high - low) / 2U;
        int cmp = strcmp(vocab->items[mid], token);
        if (cmp == 0) {
            return (long)mid;
        }
        if (cmp < 0) {
            low = mid + 1U;
        } else {
            high = mid;
        }
    }
    return -1L;
}

static double nn_gd_sigmoid(double z) {
    if (z > 30.0) {
        z = 30.0;
    } else if (z < -30.0) {
        z = -30.0;
    }
    return 1.0 / (1.0 + exp(-z));
}

static double nn_gd_clamp_unit(double value) {
    if (value < 0.0) {
        return 0.0;
    }
    if (value > 1.0) {
        return 1.0;
    }
    return value;
}

static double nn_gd_sketch_logit(const NNGDSketch *sketch,
                                 const NNGDStringList *vocab,
                                 const double *weights,
                                 double bias) {
    size_t i;
    double z = bias;
    for (i = 0; i < sketch->tokens.count; ++i) {
        long index = nn_gd_token_index(vocab, sketch->tokens.items[i]);
        if (index >= 0L) {
            z += weights[index];
        }
    }
    return z;
}

static int nn_gd_compare_scored_sketches(const void *a, const void *b) {
    const NNGDScoredSketch *sa = (const NNGDScoredSketch *)a;
    const NNGDScoredSketch *sb = (const NNGDScoredSketch *)b;
    const NNGDSketch *ka = &nn_gd_sort_sketches[sa->sketch_index];
    const NNGDSketch *kb = &nn_gd_sort_sketches[sb->sketch_index];
    int cmp;
    if (sa->score < sb->score) {
        return 1;
    }
    if (sa->score > sb->score) {
        return -1;
    }
    cmp = strcmp(ka->domain, kb->domain);
    if (cmp != 0) {
        return cmp;
    }
    return strcmp(ka->sketch_id, kb->sketch_id);
}

static uint64_t nn_gd_fnv1a_update(uint64_t hash, const char *value) {
    const unsigned char *p;
    for (p = (const unsigned char *)value; *p != '\0'; ++p) {
        hash ^= (uint64_t)*p;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t nn_gd_fnv1a_update_separator(uint64_t hash) {
    hash ^= UINT64_C(255);
    hash *= UINT64_C(1099511628211);
    return hash;
}

static void nn_gd_arc_attempt_dedupe_fields_free(
    NNGDArcAttemptDedupeFields *fields) {
    if (fields == NULL) {
        return;
    }
    free(fields->family);
    free(fields->task_id);
    free(fields->split);
    free(fields->library_id);
    free(fields->rejection_reason);
    memset(fields, 0, sizeof(*fields));
}

static void nn_gd_arc_attempt_dedupe_record_free(
    NNGDArcAttemptDedupeRecord *record) {
    if (record == NULL) {
        return;
    }
    free(record->family);
    free(record->task_id);
    free(record->split);
    free(record->library_id);
    free(record->rejection_reason);
    memset(record, 0, sizeof(*record));
}

static void nn_gd_arc_attempt_dedupe_table_free(
    NNGDArcAttemptDedupeTable *table) {
    size_t i;
    if (table == NULL) {
        return;
    }
    for (i = 0U; i < table->count; ++i) {
        nn_gd_arc_attempt_dedupe_record_free(&table->items[i]);
    }
    free(table->items);
    free(table->slots);
    memset(table, 0, sizeof(*table));
}

static uint64_t nn_gd_arc_attempt_dedupe_hash(
    const NNGDArcAttemptDedupeFields *fields) {
    uint64_t hash = UINT64_C(1469598103934665603);
    hash = nn_gd_fnv1a_update(hash, fields->family);
    hash = nn_gd_fnv1a_update_separator(hash);
    hash = nn_gd_fnv1a_update(hash, fields->task_id);
    hash = nn_gd_fnv1a_update_separator(hash);
    hash = nn_gd_fnv1a_update(hash, fields->split);
    hash = nn_gd_fnv1a_update_separator(hash);
    hash = nn_gd_fnv1a_update(hash, fields->library_id);
    hash = nn_gd_fnv1a_update_separator(hash);
    hash = nn_gd_fnv1a_update(hash, fields->rejection_reason);
    return hash;
}

static int nn_gd_arc_attempt_dedupe_keys_equal(
    const NNGDArcAttemptDedupeRecord *record,
    const NNGDArcAttemptDedupeFields *fields) {
    return strcmp(record->family, fields->family) == 0 &&
           strcmp(record->task_id, fields->task_id) == 0 &&
           strcmp(record->split, fields->split) == 0 &&
           strcmp(record->library_id, fields->library_id) == 0 &&
           strcmp(record->rejection_reason, fields->rejection_reason) == 0;
}

static int nn_gd_arc_attempt_dedupe_reserve_items(
    NNGDArcAttemptDedupeTable *table,
    size_t needed) {
    NNGDArcAttemptDedupeRecord *items;
    size_t capacity;
    if (needed <= table->capacity) {
        return 0;
    }
    capacity = table->capacity == 0U ? 256U : table->capacity * 2U;
    while (capacity < needed) {
        capacity *= 2U;
    }
    items = (NNGDArcAttemptDedupeRecord *)realloc(
        table->items,
        capacity * sizeof(NNGDArcAttemptDedupeRecord));
    if (items == NULL) {
        return -1;
    }
    table->items = items;
    table->capacity = capacity;
    return 0;
}

static size_t nn_gd_arc_attempt_dedupe_slot(
    const NNGDArcAttemptDedupeTable *table,
    const NNGDArcAttemptDedupeFields *fields,
    uint64_t hash) {
    size_t slot = (size_t)(hash & (uint64_t)(table->slot_count - 1U));
    for (;;) {
        size_t record_index = table->slots[slot];
        if (record_index == 0U ||
            (table->items[record_index - 1U].hash == hash &&
             nn_gd_arc_attempt_dedupe_keys_equal(&table->items[record_index - 1U],
                                                 fields))) {
            return slot;
        }
        slot = (slot + 1U) & (table->slot_count - 1U);
    }
}

static int nn_gd_arc_attempt_dedupe_rehash(
    NNGDArcAttemptDedupeTable *table,
    size_t needed) {
    size_t slot_count = table->slot_count == 0U ? 512U : table->slot_count;
    size_t *slots;
    size_t i;
    while (needed * 10U > slot_count * 7U) {
        slot_count *= 2U;
    }
    if (slot_count == table->slot_count) {
        return 0;
    }
    slots = (size_t *)calloc(slot_count, sizeof(size_t));
    if (slots == NULL) {
        return -1;
    }
    free(table->slots);
    table->slots = slots;
    table->slot_count = slot_count;
    for (i = 0U; i < table->count; ++i) {
        NNGDArcAttemptDedupeFields fields;
        size_t slot;
        memset(&fields, 0, sizeof(fields));
        fields.family = table->items[i].family;
        fields.task_id = table->items[i].task_id;
        fields.split = table->items[i].split;
        fields.library_id = table->items[i].library_id;
        fields.rejection_reason = table->items[i].rejection_reason;
        slot = nn_gd_arc_attempt_dedupe_slot(table, &fields, table->items[i].hash);
        table->slots[slot] = i + 1U;
    }
    return 0;
}

static int nn_gd_arc_attempt_dedupe_better(
    int accepted,
    double verifier_score,
    const NNGDArcAttemptDedupeRecord *existing) {
    if (accepted != existing->accepted) {
        return accepted > existing->accepted;
    }
    return verifier_score > existing->verifier_score;
}

static int nn_gd_arc_attempt_dedupe_upsert(
    NNGDArcAttemptDedupeTable *table,
    NNGDArcAttemptDedupeFields *fields,
    size_t source_index,
    const fpos_t *position) {
    uint64_t hash = nn_gd_arc_attempt_dedupe_hash(fields);
    size_t slot;
    size_t record_index;
    NNGDArcAttemptDedupeRecord *record;
    if (nn_gd_arc_attempt_dedupe_rehash(table, table->count + 1U) != 0) {
        return -1;
    }
    slot = nn_gd_arc_attempt_dedupe_slot(table, fields, hash);
    record_index = table->slots[slot];
    if (record_index != 0U) {
        record = &table->items[record_index - 1U];
        if (nn_gd_arc_attempt_dedupe_better(fields->accepted,
                                            fields->verifier_score,
                                            record)) {
            record->accepted = fields->accepted;
            record->verifier_score = fields->verifier_score;
            record->source_index = source_index;
            record->position = *position;
        }
        nn_gd_arc_attempt_dedupe_fields_free(fields);
        return 0;
    }
    if (nn_gd_arc_attempt_dedupe_reserve_items(table, table->count + 1U) != 0) {
        return -1;
    }
    record = &table->items[table->count];
    memset(record, 0, sizeof(*record));
    record->family = fields->family;
    record->task_id = fields->task_id;
    record->split = fields->split;
    record->library_id = fields->library_id;
    record->rejection_reason = fields->rejection_reason;
    record->accepted = fields->accepted;
    record->verifier_score = fields->verifier_score;
    record->source_index = source_index;
    record->position = *position;
    record->hash = hash;
    fields->family = NULL;
    fields->task_id = NULL;
    fields->split = NULL;
    fields->library_id = NULL;
    fields->rejection_reason = NULL;
    table->slots[slot] = table->count + 1U;
    ++table->count;
    return 0;
}

#include "nn_grammar_discovery_json.inc"

#include "nn_grammar_discovery_haiku.inc"
done:
    if (hard_rows != NULL) {
        fclose(hard_rows);
    }
    if (summary != NULL) {
        fclose(summary);
    }
    if (markdown != NULL) {
        fclose(markdown);
    }
    nn_gd_haiku_rows_free(&rows);
    return status;
}

static int nn_gd_parse_arc_attempt_dedupe_fields(
    const char *line,
    NNGDArcAttemptDedupeFields *fields) {
    char *schema;
    const char *cursor = nn_gd_json_skip_space(line);
    memset(fields, 0, sizeof(*fields));
    if (cursor == NULL || *cursor != '{') {
        return 1;
    }
    schema = nn_gd_json_dup_key_string(line, "schema");
    if (schema == NULL) {
        return -1;
    }
    if (strcmp(schema,
               "crlp.neural_grammar_discovery.arc_visual_pipeline_attempt.v1") != 0) {
        free(schema);
        return 1;
    }
    free(schema);
    fields->family = nn_gd_json_dup_key_string(line, "family");
    fields->task_id = nn_gd_json_dup_key_string(line, "task_id");
    fields->split = nn_gd_json_dup_key_string(line, "split");
    fields->library_id = nn_gd_json_dup_key_string(line, "library_id");
    fields->rejection_reason = nn_gd_json_dup_key_string(line, "rejection_reason");
    fields->accepted = nn_gd_json_get_bool(line, "accepted");
    fields->verifier_score = nn_gd_json_get_double_or_zero(line, "verifier_score");
    if (fields->family == NULL || fields->task_id == NULL || fields->split == NULL ||
        fields->library_id == NULL || fields->rejection_reason == NULL) {
        nn_gd_arc_attempt_dedupe_fields_free(fields);
        return -1;
    }
    return 0;
}

int nn_grammar_discovery_dedupe_arc_visual_pipeline_attempts(
    const char *const *input_paths,
    size_t input_count,
    const char *output_path) {
    FILE **inputs = NULL;
    FILE *output = NULL;
    NNGDArcAttemptDedupeTable table = {0};
    int status = -1;
    size_t input_index;
    size_t record_index;
    if (input_paths == NULL || input_count == 0U || output_path == NULL) {
        return -1;
    }
    inputs = (FILE **)calloc(input_count, sizeof(FILE *));
    if (inputs == NULL) {
        return -1;
    }
    for (input_index = 0U; input_index < input_count; ++input_index) {
        inputs[input_index] = fopen(input_paths[input_index], "r");
        if (inputs[input_index] == NULL) {
            goto done;
        }
        for (;;) {
            fpos_t position;
            char *line;
            NNGDArcAttemptDedupeFields fields;
            int parse_status;
            if (fgetpos(inputs[input_index], &position) != 0) {
                goto done;
            }
            line = nn_gd_read_line_dynamic(inputs[input_index]);
            if (line == NULL) {
                break;
            }
            parse_status = nn_gd_parse_arc_attempt_dedupe_fields(line, &fields);
            free(line);
            if (parse_status < 0) {
                goto done;
            }
            if (parse_status == 0 &&
                nn_gd_arc_attempt_dedupe_upsert(&table,
                                                &fields,
                                                input_index,
                                                &position) != 0) {
                nn_gd_arc_attempt_dedupe_fields_free(&fields);
                goto done;
            }
        }
    }
    output = fopen(output_path, "w");
    if (output == NULL) {
        goto done;
    }
    for (record_index = 0U; record_index < table.count; ++record_index) {
        const NNGDArcAttemptDedupeRecord *record = &table.items[record_index];
        char *line;
        size_t line_len;
        if (record->source_index >= input_count ||
            fsetpos(inputs[record->source_index], &record->position) != 0) {
            goto done;
        }
        line = nn_gd_read_line_dynamic(inputs[record->source_index]);
        if (line == NULL) {
            goto done;
        }
        line_len = strlen(line);
        if (fputs(line, output) == EOF) {
            free(line);
            goto done;
        }
        if (line_len == 0U || line[line_len - 1U] != '\n') {
            if (fputc('\n', output) == EOF) {
                free(line);
                goto done;
            }
        }
        free(line);
    }
    if (ferror(output)) {
        goto done;
    }
    status = 0;

done:
    if (output != NULL) {
        fclose(output);
    }
    if (inputs != NULL) {
        for (input_index = 0U; input_index < input_count; ++input_index) {
            if (inputs[input_index] != NULL) {
                fclose(inputs[input_index]);
            }
        }
        free(inputs);
    }
    nn_gd_arc_attempt_dedupe_table_free(&table);
    return status;
}

static uint64_t nn_gd_proposal_hash(const char *sketch_id) {
    uint64_t hash = UINT64_C(1469598103934665603);
    hash = nn_gd_fnv1a_update(hash, "neural-linear");
    hash ^= 0U;
    hash *= UINT64_C(1099511628211);
    hash = nn_gd_fnv1a_update(hash, sketch_id);
    return hash;
}

static uint64_t nn_gd_closed_loop_proposal_hash(const char *sketch_id,
                                                size_t step) {
    char step_buffer[32];
    uint64_t hash = UINT64_C(1469598103934665603);
    snprintf(step_buffer, sizeof(step_buffer), "%zu", step);
    hash = nn_gd_fnv1a_update(hash, "closed-loop-sketch-composer");
    hash = nn_gd_fnv1a_update_separator(hash);
    hash = nn_gd_fnv1a_update(hash, sketch_id);
    hash = nn_gd_fnv1a_update_separator(hash);
    hash = nn_gd_fnv1a_update(hash, step_buffer);
    return hash;
}

static int nn_gd_build_vocab(const NNGDSketchList *sketches, NNGDStringList *vocab) {
    size_t i;
    size_t j;
    for (i = 0; i < sketches->count; ++i) {
        const NNGDSketch *sketch = &sketches->items[i];
        for (j = 0; j < sketch->tokens.count; ++j) {
            if (nn_gd_string_list_add_unique(vocab, sketch->tokens.items[j]) != 0) {
                return -1;
            }
        }
    }
    nn_gd_string_list_sort(vocab);
    return 0;
}

static int nn_gd_load_neural_checkpoint(const char *path,
                                        const NNGDStringList *vocab,
                                        double *weights,
                                        double *bias,
                                        NNGDMemoryState *memory) {
    FILE *fp;
    char line[NN_GD_LINE_MAX];
    char *fields[NN_GD_MAX_FIELDS];
    if (path == NULL || path[0] == '\0') {
        return 0;
    }
    fp = fopen(path, "r");
    if (fp == NULL) {
        return 0;
    }
    memory->loaded = 1;
    while (fgets(line, sizeof(line), fp) != NULL) {
        size_t field_count = nn_gd_split_tabs(line, fields, NN_GD_MAX_FIELDS);
        const char *kind = nn_gd_field(fields, field_count, 0U);
        if (strcmp(kind, "bias") == 0) {
            *bias = nn_gd_parse_double(nn_gd_field(fields, field_count, 1U));
        } else if (strcmp(kind, "epochs_seen") == 0) {
            memory->epochs_seen_before =
                (unsigned)nn_gd_parse_long(nn_gd_field(fields, field_count, 1U));
        } else if (strcmp(kind, "training_examples_seen") == 0) {
            long parsed = nn_gd_parse_long(nn_gd_field(fields, field_count, 1U));
            memory->training_examples_seen_before = parsed < 0L ? 0U : (size_t)parsed;
        } else if (strcmp(kind, "token") == 0) {
            const char *token = nn_gd_field(fields, field_count, 1U);
            long index = nn_gd_token_index(vocab, token);
            if (index >= 0L) {
                weights[index] = nn_gd_parse_double(nn_gd_field(fields, field_count, 2U));
                ++memory->restored_weights;
            } else {
                ++memory->ignored_weights;
            }
        }
    }
    fclose(fp);
    return 0;
}

static int nn_gd_save_neural_checkpoint(const char *path,
                                        const NNGDStringList *vocab,
                                        const double *weights,
                                        double bias,
                                        const NNGDMemoryState *memory) {
    FILE *fp;
    size_t i;
    if (path == NULL || path[0] == '\0') {
        return 0;
    }
    fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    fputs("schema\tcrlp.grammar_sketch.memory.v1\n", fp);
    fprintf(fp, "bias\t%.17g\n", bias);
    fprintf(fp, "epochs_seen\t%u\n", memory->epochs_seen_after);
    fprintf(fp, "training_examples_seen\t%zu\n", memory->training_examples_seen_after);
    fprintf(fp, "vocab_size\t%zu\n", vocab->count);
    for (i = 0; i < vocab->count; ++i) {
        fputs("token\t", fp);
        fputs(vocab->items[i], fp);
        fprintf(fp, "\t%.17g\n", weights[i]);
    }
    if (ferror(fp)) {
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
}

static int nn_gd_write_neural_rows(FILE *fp,
                                   const NNGDSketchList *sketches,
                                   const NNGDLabelList *labels,
                                   const NNGDStringList *vocab,
                                   const double *weights,
                                   double bias,
                                   unsigned epochs,
                                   double learning_rate,
                                   const NNGDMemoryState *memory) {
    NNGDScoredSketch *scores;
    size_t i;
    scores = (NNGDScoredSketch *)calloc(sketches->count, sizeof(NNGDScoredSketch));
    if (scores == NULL && sketches->count > 0U) {
        return -1;
    }
    for (i = 0; i < sketches->count; ++i) {
        scores[i].sketch_index = i;
        scores[i].score = nn_gd_sigmoid(nn_gd_sketch_logit(&sketches->items[i],
                                                           vocab,
                                                           weights,
                                                           bias));
    }
    nn_gd_sort_sketches = sketches->items;
    qsort(scores, sketches->count, sizeof(NNGDScoredSketch), nn_gd_compare_scored_sketches);
    nn_gd_sort_sketches = NULL;
    for (i = 0; i < sketches->count; ++i) {
        const NNGDSketch *sketch = &sketches->items[scores[i].sketch_index];
        uint64_t proposal_hash = nn_gd_proposal_hash(sketch->sketch_id);
        fputs("{\"schema\":\"crlp.grammar_sketch.proposal.v1\",\"report_version\":1,"
              "\"proposal_id\":\"gsp:",
              fp);
        fprintf(fp, "%016" PRIx64, proposal_hash);
        fputs("\",\"proposer\":\"neural_linear_sketch_proposer\",\"sketch_id\":", fp);
        nn_gd_json_write_string(fp, sketch->sketch_id);
        fputs(",\"domain\":", fp);
        nn_gd_json_write_string(fp, sketch->domain);
        fputs(",\"family\":", fp);
        nn_gd_json_write_string(fp, sketch->family);
        fprintf(fp,
                ",\"score\":%.8f,\"rank_features\":{\"epochs\":%u,"
                "\"learning_rate\":%.17g,\"vocab_size\":%zu,"
                "\"training_examples\":%zu,\"checkpoint_loaded\":%s,"
                "\"checkpoint_epochs_seen_before\":%u,"
                "\"checkpoint_epochs_seen_after\":%u,"
                "\"checkpoint_training_examples_seen_before\":%zu,"
                "\"checkpoint_training_examples_seen_after\":%zu,"
                "\"checkpoint_restored_weights\":%zu,"
                "\"checkpoint_ignored_weights\":%zu},\"rank\":%zu}\n",
                scores[i].score,
                epochs,
                learning_rate,
                vocab->count,
                labels->count,
                memory->loaded ? "true" : "false",
                memory->epochs_seen_before,
                memory->epochs_seen_after,
                memory->training_examples_seen_before,
                memory->training_examples_seen_after,
                memory->restored_weights,
                memory->ignored_weights,
                i + 1U);
    }
    free(scores);
    return ferror(fp) ? -1 : 0;
}

static int nn_gd_read_static_sketch_scores(const char *path,
                                           const NNGDSketchList *sketches,
                                           double *scores,
                                           int *seen,
                                           size_t *seen_count) {
    FILE *fp;
    char line[NN_GD_LINE_MAX];
    if (seen_count != NULL) {
        *seen_count = 0U;
    }
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    while (fgets(line, sizeof(line), fp) != NULL) {
        NNGDJsonIndex index;
        char *schema;
        char *sketch_id;
        long sketch_index;
        if (strstr(line, "\"schema\":\"crlp.grammar_sketch.proposal.v1\"") == NULL) {
            continue;
        }
        nn_gd_json_index_line(line, &index);
        schema = nn_gd_json_dup_index_string(&index, "schema");
        sketch_id = nn_gd_json_dup_index_string(&index, "sketch_id");
        if (schema == NULL || sketch_id == NULL ||
            strcmp(schema, "crlp.grammar_sketch.proposal.v1") != 0) {
            free(schema);
            free(sketch_id);
            continue;
        }
        sketch_index = nn_gd_find_sketch(sketches, sketch_id);
        if (sketch_index >= 0L) {
            size_t index_value = (size_t)sketch_index;
            scores[index_value] = nn_gd_json_index_double_or_zero(&index, "score");
            if (!seen[index_value] && seen_count != NULL) {
                ++*seen_count;
            }
            seen[index_value] = 1;
        }
        free(schema);
        free(sketch_id);
    }
    fclose(fp);
    return 0;
}

static double nn_gd_label_mean_for_sketch(const NNGDLabelList *labels,
                                          size_t sketch_index) {
    size_t i;
    size_t count = 0U;
    double sum = 0.0;
    for (i = 0U; i < labels->count; ++i) {
        if (labels->items[i].sketch_index == sketch_index) {
            sum += labels->items[i].label;
            ++count;
        }
    }
    return count == 0U ? 0.5 : sum / (double)count;
}

static int nn_gd_sketch_token_selected(const NNGDSketch *candidate,
                                       const NNGDSketch *selected) {
    size_t i;
    size_t j;
    for (i = 0U; i < candidate->tokens.count; ++i) {
        for (j = 0U; j < selected->tokens.count; ++j) {
            if (strcmp(candidate->tokens.items[i], selected->tokens.items[j]) == 0) {
                return 1;
            }
        }
    }
    return 0;
}

static double nn_gd_selected_token_overlap(const NNGDSketchList *sketches,
                                           const int *selected,
                                           size_t candidate_index) {
    const NNGDSketch *candidate = &sketches->items[candidate_index];
    size_t matched = 0U;
    size_t i;
    if (candidate->tokens.count == 0U) {
        return 0.0;
    }
    for (i = 0U; i < sketches->count; ++i) {
        if (!selected[i]) {
            continue;
        }
        matched += nn_gd_sketch_token_selected(candidate, &sketches->items[i]) ? 1U : 0U;
    }
    if (matched > candidate->tokens.count) {
        matched = candidate->tokens.count;
    }
    return (double)matched / (double)candidate->tokens.count;
}

static size_t nn_gd_selected_same_domain(const NNGDSketchList *sketches,
                                         const int *selected,
                                         size_t candidate_index) {
    const char *domain = sketches->items[candidate_index].domain;
    size_t i;
    size_t count = 0U;
    for (i = 0U; i < sketches->count; ++i) {
        if (selected[i] && strcmp(sketches->items[i].domain, domain) == 0) {
            ++count;
        }
    }
    return count;
}

static size_t nn_gd_selected_same_family(const NNGDSketchList *sketches,
                                         const int *selected,
                                         size_t candidate_index) {
    const char *family = sketches->items[candidate_index].family;
    size_t i;
    size_t count = 0U;
    for (i = 0U; i < sketches->count; ++i) {
        if (selected[i] && strcmp(sketches->items[i].family, family) == 0) {
            ++count;
        }
    }
    return count;
}

static size_t nn_gd_selected_domain_count(const NNGDSketchList *sketches,
                                          const int *selected) {
    size_t i;
    size_t j;
    size_t count = 0U;
    for (i = 0U; i < sketches->count; ++i) {
        int seen = 0;
        if (!selected[i]) {
            continue;
        }
        for (j = 0U; j < i; ++j) {
            if (selected[j] &&
                strcmp(sketches->items[i].domain, sketches->items[j].domain) == 0) {
                seen = 1;
                break;
            }
        }
        if (!seen) {
            ++count;
        }
    }
    return count;
}

static int nn_gd_pick_closed_loop_sketch(const NNGDSketchList *sketches,
                                         const NNGDLabelList *labels,
                                         const double *static_scores,
                                         const int *static_seen,
                                         const int *selected,
                                         size_t selected_count,
                                         NNGDClosedLoopSketchChoice *choice) {
    size_t i;
    int found = 0;
    NNGDClosedLoopSketchChoice best;
    memset(&best, 0, sizeof(best));
    for (i = 0U; i < sketches->count; ++i) {
        double evidence_mean;
        double overlap;
        double static_score;
        double domain_novelty_bonus;
        double family_penalty;
        double overlap_penalty;
        double dynamic_score;
        size_t same_domain;
        size_t same_family;
        if (selected[i]) {
            continue;
        }
        evidence_mean = nn_gd_label_mean_for_sketch(labels, i);
        overlap = nn_gd_selected_token_overlap(sketches, selected, i);
        same_domain = nn_gd_selected_same_domain(sketches, selected, i);
        same_family = nn_gd_selected_same_family(sketches, selected, i);
        static_score = static_seen[i] ? static_scores[i] : 0.5;
        domain_novelty_bonus = same_domain == 0U ? 0.01 : 0.0;
        family_penalty = 0.04 * (double)same_family;
        overlap_penalty = 0.03 * overlap;
        dynamic_score = static_score + 0.035 * evidence_mean +
                        domain_novelty_bonus - family_penalty - overlap_penalty;
        if (!found || dynamic_score > best.dynamic_score ||
            (dynamic_score == best.dynamic_score &&
             strcmp(sketches->items[i].sketch_id,
                    sketches->items[best.sketch_index].sketch_id) < 0)) {
            best.sketch_index = i;
            best.static_score = static_score;
            best.dynamic_score = dynamic_score;
            best.evidence_mean = evidence_mean;
            best.overlap_penalty = overlap_penalty;
            best.family_penalty = family_penalty;
            best.domain_novelty_bonus = domain_novelty_bonus;
            best.same_domain_selected = same_domain;
            best.same_family_selected = same_family;
            best.selected_before = selected_count;
            found = 1;
        }
    }
    if (!found) {
        return -1;
    }
    *choice = best;
    return 0;
}

static int nn_gd_write_closed_loop_sketch_summary(
    const char *path,
    size_t static_proposal_count,
    const NNGDSketchList *sketches,
    const int *selected,
    size_t selected_count,
    double static_score_sum,
    double dynamic_score_sum,
    size_t redundancy_penalties) {
    FILE *fp;
    if (path == NULL || path[0] == '\0') {
        return 0;
    }
    fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    fputs("{\"schema\":\"crlp.grammar_sketch.closed_loop_summary.v1\","
          "\"report_version\":1,\"proposer\":\"closed_loop_sketch_composer\","
          "\"source_proposer\":\"neural_linear_sketch_proposer\",",
          fp);
    fprintf(fp,
            "\"source_proposal_count\":%zu,\"proposal_count\":%zu,"
            "\"residual_update_count\":%zu,\"unique_domains_covered\":%zu,"
            "\"redundancy_penalties_applied\":%zu,\"mean_static_score\":%.8f,"
            "\"mean_dynamic_score\":%.8f,"
            "\"verifier_role\":\"proposals_only_verifier_still_gates_promotion\"}\n",
            static_proposal_count,
            selected_count,
            selected_count,
            nn_gd_selected_domain_count(sketches, selected),
            redundancy_penalties,
            selected_count == 0U ? 0.0 : static_score_sum / (double)selected_count,
            selected_count == 0U ? 0.0 : dynamic_score_sum / (double)selected_count);
    if (ferror(fp)) {
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
}

static int nn_gd_write_closed_loop_sketch_rows(
    const char *path,
    const NNGDSketchList *sketches,
    const NNGDLabelList *labels,
    const double *static_scores,
    const int *static_seen,
    size_t static_proposal_count,
    const char *summary_output_path) {
    FILE *fp;
    int *selected = NULL;
    size_t selected_count = 0U;
    size_t redundancy_penalties = 0U;
    double static_score_sum = 0.0;
    double dynamic_score_sum = 0.0;
    int status = -1;
    fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    selected = (int *)calloc(sketches->count == 0U ? 1U : sketches->count,
                             sizeof(int));
    if (selected == NULL) {
        goto done;
    }
    while (selected_count < sketches->count) {
        NNGDClosedLoopSketchChoice choice;
        const NNGDSketch *sketch;
        uint64_t proposal_hash;
        double emitted_score;
        if (nn_gd_pick_closed_loop_sketch(sketches,
                                          labels,
                                          static_scores,
                                          static_seen,
                                          selected,
                                          selected_count,
                                          &choice) != 0) {
            goto done;
        }
        sketch = &sketches->items[choice.sketch_index];
        emitted_score = nn_gd_clamp_unit(choice.dynamic_score);
        proposal_hash = nn_gd_closed_loop_proposal_hash(sketch->sketch_id,
                                                        selected_count);
        if (choice.overlap_penalty > 0.0 || choice.family_penalty > 0.0) {
            ++redundancy_penalties;
        }
        fputs("{\"schema\":\"crlp.grammar_sketch.proposal.v1\","
              "\"report_version\":2,\"proposal_id\":\"gspcl:",
              fp);
        fprintf(fp, "%016" PRIx64, proposal_hash);
        fputs("\",\"proposer\":\"closed_loop_sketch_composer\","
              "\"static_proposer\":\"neural_linear_sketch_proposer\","
              "\"sketch_id\":",
              fp);
        nn_gd_json_write_string(fp, sketch->sketch_id);
        fputs(",\"domain\":", fp);
        nn_gd_json_write_string(fp, sketch->domain);
        fputs(",\"family\":", fp);
        nn_gd_json_write_string(fp, sketch->family);
        fprintf(fp,
                ",\"score\":%.8f,\"static_score\":%.8f,"
                "\"rank_features\":{\"source_score\":%.8f,"
                "\"dynamic_score_raw\":%.8f,"
                "\"evidence_mean\":%.8f,\"domain_novelty_bonus\":%.8f,"
                "\"family_penalty\":%.8f,\"overlap_penalty\":%.8f,"
                "\"same_domain_selected\":%zu,\"same_family_selected\":%zu,"
                "\"selected_before\":%zu,\"source_proposal_seen\":%s},"
                "\"closed_loop_context\":{\"step\":%zu,"
                "\"residual_update\":\"select_then_recompute\","
                "\"conditioning\":\"accepted_sketches_and_residual_overlap\","
                "\"verifier_gate\":\"promotion_requires_domain_replay\"},"
                "\"rank\":%zu}\n",
                emitted_score,
                choice.static_score,
                choice.static_score,
                choice.dynamic_score,
                choice.evidence_mean,
                choice.domain_novelty_bonus,
                choice.family_penalty,
                choice.overlap_penalty,
                choice.same_domain_selected,
                choice.same_family_selected,
                choice.selected_before,
                static_seen[choice.sketch_index] ? "true" : "false",
                selected_count + 1U,
                selected_count + 1U);
        selected[choice.sketch_index] = 1;
        ++selected_count;
        static_score_sum += choice.static_score;
        dynamic_score_sum += emitted_score;
    }
    if (ferror(fp) ||
        nn_gd_write_closed_loop_sketch_summary(summary_output_path,
                                               static_proposal_count,
                                               sketches,
                                               selected,
                                               selected_count,
                                               static_score_sum,
                                               dynamic_score_sum,
                                               redundancy_penalties) != 0) {
        goto done;
    }
    status = 0;

done:
    free(selected);
    fclose(fp);
    return status;
}

#include "nn_grammar_discovery_arc.inc"

static char *nn_gd_read_file_dynamic_path(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb");
    size_t capacity = 65536U;
    size_t used = 0U;
    char *buffer;
    if (len_out != NULL) {
        *len_out = 0U;
    }
    if (fp == NULL) {
        return NULL;
    }
    buffer = (char *)malloc(capacity);
    if (buffer == NULL) {
        fclose(fp);
        return NULL;
    }
    for (;;) {
        size_t room = capacity - used;
        size_t read_count;
        if (room <= 1U) {
            size_t next_capacity = capacity * 2U;
            char *next = (char *)realloc(buffer, next_capacity);
            if (next == NULL) {
                free(buffer);
                fclose(fp);
                return NULL;
            }
            buffer = next;
            capacity = next_capacity;
            room = capacity - used;
        }
        read_count = fread(buffer + used, 1U, room - 1U, fp);
        used += read_count;
        if (read_count < room - 1U) {
            if (ferror(fp)) {
                free(buffer);
                fclose(fp);
                return NULL;
            }
            break;
        }
    }
    fclose(fp);
    buffer[used] = '\0';
    if (len_out != NULL) {
        *len_out = used;
    }
    return buffer;
}

static const char *nn_gd_json_find_key_value_range(const char *start,
                                                   const char *end,
                                                   const char *key) {
    char pattern[128];
    int written;
    const char *cursor;
    if (start == NULL || end == NULL || key == NULL || start >= end) {
        return NULL;
    }
    written = snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    if (written <= 0 || (size_t)written >= sizeof(pattern)) {
        return NULL;
    }
    cursor = start;
    while (cursor < end) {
        const char *found = strstr(cursor, pattern);
        if (found == NULL || found >= end) {
            return NULL;
        }
        if (found + written <= end) {
            return found + written;
        }
        cursor = found + 1;
    }
    return NULL;
}

static int nn_gd_arc_parse_grid_shape(const char **cursor_ref,
                                      long *width_out,
                                      long *height_out) {
    const char *cursor = nn_gd_json_skip_space(cursor_ref == NULL ? NULL : *cursor_ref);
    long height = 0L;
    long first_width = 0L;
    if (cursor == NULL || *cursor != '[') {
        return -1;
    }
    ++cursor;
    for (;;) {
        long row_width = 0L;
        cursor = nn_gd_json_skip_space(cursor);
        if (cursor == NULL) {
            return -1;
        }
        if (*cursor == ']') {
            ++cursor;
            break;
        }
        if (*cursor != '[') {
            return -1;
        }
        ++cursor;
        for (;;) {
            char *endptr = NULL;
            cursor = nn_gd_json_skip_space(cursor);
            if (cursor == NULL) {
                return -1;
            }
            if (*cursor == ']') {
                ++cursor;
                break;
            }
            (void)strtol(cursor, &endptr, 10);
            if (endptr == cursor) {
                return -1;
            }
            ++row_width;
            cursor = nn_gd_json_skip_space(endptr);
            if (cursor == NULL) {
                return -1;
            }
            if (*cursor == ',') {
                ++cursor;
                continue;
            }
            if (*cursor == ']') {
                ++cursor;
                break;
            }
            return -1;
        }
        ++height;
        if (height == 1L) {
            first_width = row_width;
        }
        cursor = nn_gd_json_skip_space(cursor);
        if (cursor == NULL) {
            return -1;
        }
        if (*cursor == ',') {
            ++cursor;
            continue;
        }
        if (*cursor == ']') {
            ++cursor;
            break;
        }
        return -1;
    }
    *width_out = first_width;
    *height_out = height;
    if (cursor_ref != NULL) {
        *cursor_ref = cursor;
    }
    return 0;
}

static int nn_gd_arc_public_task_path(const char *public_arc_dir,
                                      const char *split,
                                      const char *task_id,
                                      char *path,
                                      size_t path_size) {
    const char *subdir =
        nn_gd_arc_hole_contains_ci(split, "eval") ? "evaluation" : "training";
    int written = snprintf(path,
                           path_size,
                           "%s/%s/%s.json",
                           public_arc_dir,
                           subdir,
                           task_id);
    return written < 0 || (size_t)written >= path_size ? -1 : 0;
}

static int nn_gd_arc_load_task_shapes(const char *public_arc_dir,
                                      const char *split,
                                      const char *task_id,
                                      NNGDArcTaskShapes *shapes) {
    char path[4096];
    char *content;
    const char *train;
    const char *cursor;
    long scale_candidate = 0L;
    int stable_scale = 0;
    memset(shapes, 0, sizeof(*shapes));
    if (nn_gd_arc_public_task_path(public_arc_dir,
                                   split,
                                   task_id,
                                   path,
                                   sizeof(path)) != 0) {
        return -1;
    }
    content = nn_gd_read_file_dynamic_path(path, NULL);
    if (content == NULL) {
        return 1;
    }
    train = nn_gd_json_find_key_value(content, "train");
    cursor = nn_gd_json_skip_space(train);
    if (cursor == NULL || *cursor != '[') {
        free(content);
        return -1;
    }
    ++cursor;
    for (;;) {
        const char *object_start;
        const char *object_end;
        const char *input_cursor;
        const char *output_cursor;
        long input_width;
        long input_height;
        long output_width;
        long output_height;
        cursor = nn_gd_json_skip_space(cursor);
        if (cursor == NULL) {
            free(content);
            return -1;
        }
        if (*cursor == ']') {
            break;
        }
        if (*cursor != '{') {
            free(content);
            return -1;
        }
        object_start = cursor;
        object_end = nn_gd_arc_hole_json_object_end(object_start);
        if (object_end == NULL || object_end <= object_start) {
            free(content);
            return -1;
        }
        input_cursor = nn_gd_json_find_key_value_range(object_start,
                                                       object_end,
                                                       "input");
        output_cursor = nn_gd_json_find_key_value_range(object_start,
                                                        object_end,
                                                        "output");
        if (input_cursor == NULL || output_cursor == NULL ||
            nn_gd_arc_parse_grid_shape(&input_cursor, &input_width, &input_height) != 0 ||
            nn_gd_arc_parse_grid_shape(&output_cursor,
                                       &output_width,
                                       &output_height) != 0) {
            free(content);
            return -1;
        }
        if (shapes->train_pairs == 0L) {
            shapes->first_input_width = input_width;
            shapes->first_input_height = input_height;
            shapes->first_output_width = output_width;
            shapes->first_output_height = output_height;
            shapes->same_output_shape = 1;
            shapes->crop_like = 1;
            shapes->grow_like = 1;
            if (input_width > 0L && input_height > 0L &&
                output_width % input_width == 0L &&
                output_height % input_height == 0L) {
                long scale_x = output_width / input_width;
                long scale_y = output_height / input_height;
                if (scale_x > 1L && scale_x == scale_y) {
                    scale_candidate = scale_x;
                    stable_scale = 1;
                }
            }
        } else if (output_width != shapes->first_output_width ||
                   output_height != shapes->first_output_height) {
            shapes->same_output_shape = 0;
        }
        if (stable_scale &&
            (input_width * scale_candidate != output_width ||
             input_height * scale_candidate != output_height)) {
            stable_scale = 0;
        }
        if (output_width > input_width || output_height > input_height) {
            shapes->crop_like = 0;
        }
        if (output_width < input_width || output_height < input_height) {
            shapes->grow_like = 0;
        }
        if (output_width == 1L || output_height == 1L) {
            shapes->line_like = 1;
        }
        ++shapes->train_pairs;
        cursor = object_end;
        cursor = nn_gd_json_skip_space(cursor);
        if (cursor == NULL) {
            free(content);
            return -1;
        }
        if (*cursor == ',') {
            ++cursor;
            continue;
        }
        if (*cursor == ']') {
            break;
        }
        free(content);
        return -1;
    }
    if (stable_scale) {
        shapes->scale = scale_candidate;
    }
    shapes->loaded = shapes->train_pairs > 0L;
    free(content);
    return 0;
}

static int nn_gd_arc_synthetic_shape_pin_output(NNGDStringList *tokens,
                                                const NNGDArcTaskShapes *shapes) {
    if (shapes->same_output_shape && shapes->first_output_width > 0L &&
        shapes->first_output_height > 0L) {
        if (nn_gd_arc_hole_add_pin(tokens, "width", shapes->first_output_width) != 0 ||
            nn_gd_arc_hole_add_pin(tokens, "height", shapes->first_output_height) != 0) {
            return -1;
        }
    }
    return 0;
}

static int nn_gd_arc_synthetic_shape_pin_scale(NNGDStringList *tokens,
                                               const NNGDArcTaskShapes *shapes) {
    if (shapes->scale > 1L) {
        return nn_gd_arc_hole_add_pin(tokens, "period", shapes->scale);
    }
    return 0;
}

static int nn_gd_arc_synthetic_candidate_duplicate(
    const NNGDArcHoleCandidateList *list,
    const char *family,
    const char *task_id,
    const char *split,
    const char *op_hints) {
    size_t i;
    for (i = 0U; i < list->count; ++i) {
        const NNGDArcHoleCandidate *candidate = &list->items[i];
        if (strcmp(candidate->fields[0], family) == 0 &&
            strcmp(candidate->fields[1], task_id) == 0 &&
            strcmp(candidate->fields[2], split) == 0 &&
            strcmp(candidate->fields[6], op_hints) == 0) {
            return 1;
        }
    }
    return 0;
}

static int nn_gd_arc_synthetic_candidate_add(NNGDArcHoleCandidateList *candidates,
                                             const char *family,
                                             const char *task_id,
                                             const char *split,
                                             const NNGDStringList *tokens,
                                             const char *parent,
                                             const char *stage,
                                             const char *strategy,
                                             double utility,
                                             long proposal_count) {
    char *op_hints = nn_gd_arc_hole_compact_tokens(tokens);
    char proposal_buffer[32];
    char utility_buffer[64];
    const char *fields[13];
    NNGDArcHoleCandidate *candidate;
    size_t i;
    if (op_hints == NULL) {
        return -1;
    }
    if (family[0] == '\0' || task_id[0] == '\0' || split[0] == '\0' ||
        op_hints[0] == '\0') {
        free(op_hints);
        return 0;
    }
    if (nn_gd_arc_synthetic_candidate_duplicate(candidates,
                                                family,
                                                task_id,
                                                split,
                                                op_hints)) {
        free(op_hints);
        return 0;
    }
    snprintf(proposal_buffer,
             sizeof(proposal_buffer),
             "%ld",
             proposal_count < 1L ? 1L : proposal_count);
    snprintf(utility_buffer, sizeof(utility_buffer), "%.8f", utility);
    fields[0] = family;
    fields[1] = task_id;
    fields[2] = split;
    fields[3] = "1";
    fields[4] = proposal_buffer;
    fields[5] = utility_buffer;
    fields[6] = op_hints;
    fields[7] = "arc_synthetic_shape_probe";
    fields[8] = parent;
    fields[9] = "synthetic_shape_probe";
    fields[10] = "candidate_generation_gap";
    fields[11] = stage;
    fields[12] = strategy;
    if (nn_gd_arc_hole_candidate_list_reserve(candidates,
                                              candidates->count + 1U) != 0) {
        free(op_hints);
        return -1;
    }
    candidate = &candidates->items[candidates->count];
    memset(candidate, 0, sizeof(*candidate));
    for (i = 0U; i < 13U; ++i) {
        candidate->fields[i] = nn_gd_strdup(fields[i]);
        if (candidate->fields[i] == NULL) {
            nn_gd_arc_hole_candidate_free(candidate);
            free(op_hints);
            return -1;
        }
    }
    ++candidates->count;
    free(op_hints);
    return 0;
}

static int nn_gd_arc_synthetic_top_op_in(const char *top_op,
                                         const char *a,
                                         const char *b,
                                         const char *c,
                                         const char *d) {
    return (a != NULL && strcmp(top_op, a) == 0) ||
           (b != NULL && strcmp(top_op, b) == 0) ||
           (c != NULL && strcmp(top_op, c) == 0) ||
           (d != NULL && strcmp(top_op, d) == 0);
}

static int nn_gd_arc_synthetic_emit_strategy_rows(
    NNGDArcHoleCandidateList *candidates,
    const char *family,
    const char *task_id,
    const char *split,
    const char *top_op,
    const char *stage,
    const char *reason,
    const char *parent,
    long candidate_count,
    const NNGDArcTaskShapes *shapes,
    unsigned max_rows,
    int broad_profile,
    size_t *rows_for_example) {
    unsigned emitted = 0U;
    const char *stage_value = stage[0] != '\0' ? stage : (reason[0] != '\0' ? reason : "candidate_generation_gap");
    *rows_for_example = 0U;
    if (max_rows == 0U) {
        return 0;
    }
#define NN_GD_SYNTHETIC_MAYBE_ADD(tokens_expr, strategy_name, utility_value) \
    do { \
        NNGDStringList tokens = {0}; \
        int token_status = (tokens_expr); \
        if (token_status != 0 || \
            nn_gd_arc_synthetic_candidate_add(candidates, \
                                              family, \
                                              task_id, \
                                              split, \
                                              &tokens, \
                                              parent, \
                                              stage_value, \
                                              strategy_name, \
                                              utility_value, \
                                              64L) != 0) { \
            nn_gd_string_list_free(&tokens); \
            return -1; \
        } \
        if (tokens.count > 0U) { \
            ++*rows_for_example; \
        } \
        nn_gd_string_list_free(&tokens); \
        ++emitted; \
    } while (0)

    if (top_op[0] != '\0' && (broad_profile || strcmp(top_op, "visual_combinator") != 0) &&
        emitted < max_rows) {
        NN_GD_SYNTHETIC_MAYBE_ADD(
            nn_gd_string_list_add_unique(&tokens, top_op) != 0 ||
                    (broad_profile &&
                     nn_gd_arc_hole_add_stage_tokens(&tokens, stage_value) != 0) ||
                    nn_gd_arc_synthetic_shape_pin_output(&tokens, shapes) != 0
                ? -1
                : 0,
            "replay_top_failed_op_with_synthetic_shape_hints",
            1.0);
    }
    if ((broad_profile || strstr(family, "crop_object_selection") != NULL ||
         shapes->crop_like ||
         nn_gd_arc_synthetic_top_op_in(top_op,
                                       "crop",
                                       "masked_or_normalized_crop",
                                       "recolor_crop",
                                       "object_mask_extract")) &&
        emitted < max_rows) {
        NN_GD_SYNTHETIC_MAYBE_ADD(
            nn_gd_string_list_add_unique(&tokens, "crop") != 0 ||
                    nn_gd_string_list_add_unique(&tokens, "masked_or_normalized_crop") != 0 ||
                    nn_gd_string_list_add_unique(&tokens, "recolor_crop") != 0 ||
                    nn_gd_string_list_add_unique(&tokens, "object_mask_extract") != 0 ||
                    nn_gd_string_list_add_unique(&tokens,
                                                 "counterexample_shape_frame_repair") != 0 ||
                    nn_gd_arc_synthetic_shape_pin_output(&tokens, shapes) != 0
                ? -1
                : 0,
            "exhaust_crop_and_normalized_shape_frames",
            0.95);
    }
    if ((broad_profile || strstr(family, "patch_lattice") != NULL ||
         strstr(family, "tiling_canvas") != NULL || shapes->grow_like ||
         nn_gd_arc_synthetic_top_op_in(top_op,
                                       "macrocell_substitute",
                                       "palette_scale",
                                       "lattice_sample",
                                       NULL)) &&
        emitted < max_rows) {
        NN_GD_SYNTHETIC_MAYBE_ADD(
            nn_gd_string_list_add_unique(&tokens, "macrocell_substitute") != 0 ||
                    nn_gd_string_list_add_unique(&tokens, "palette_scale") != 0 ||
                    nn_gd_string_list_add_unique(&tokens, "lattice_sample") != 0 ||
                    nn_gd_string_list_add_unique(&tokens, "point_line_project") != 0 ||
                    nn_gd_string_list_add_unique(&tokens, "object_lattice_frame") != 0 ||
                    nn_gd_string_list_add_unique(&tokens,
                                                 "normalizer:object_lattice") != 0 ||
                    nn_gd_arc_synthetic_shape_pin_scale(&tokens, shapes) != 0
                ? -1
                : 0,
            "exhaust_macro_lattice_shape_frames",
            0.92);
    }
    if ((broad_profile ||
         (strcmp(top_op, "object_mask_extract") == 0 && candidate_count < 5000L)) &&
        emitted < max_rows) {
        NN_GD_SYNTHETIC_MAYBE_ADD(
            nn_gd_string_list_add_unique(&tokens, "visual_combinator") != 0 ||
                    nn_gd_string_list_add_unique(&tokens, "object_mask_extract") != 0 ||
                    nn_gd_string_list_add_unique(&tokens, "emitter:component_mask") != 0 ||
                    nn_gd_string_list_add_unique(&tokens, "normalizer:component_mask") != 0 ||
                    nn_gd_string_list_add_unique(&tokens, "selector:component_predicate") != 0 ||
                    nn_gd_string_list_add_unique(
                        &tokens,
                        "normalizer:component_selector_layout") != 0 ||
                    nn_gd_arc_synthetic_shape_pin_output(&tokens, shapes) != 0
                ? -1
                : 0,
            "exhaust_component_mask_render_shapes",
            0.90);
    }
    if ((broad_profile || strstr(family, "projection_compression") != NULL ||
         shapes->line_like ||
         nn_gd_arc_synthetic_top_op_in(top_op,
                                       "axis_projection_compress",
                                       "component_count_line",
                                       NULL,
                                       NULL)) &&
        emitted < max_rows) {
        NN_GD_SYNTHETIC_MAYBE_ADD(
            nn_gd_string_list_add_unique(&tokens, "axis_projection_compress") != 0 ||
                    nn_gd_string_list_add_unique(&tokens, "component_count_line") != 0 ||
                    nn_gd_string_list_add_unique(&tokens, "marker_count_motif_tile") != 0 ||
                    nn_gd_string_list_add_unique(&tokens, "component_color_summary") != 0 ||
                    nn_gd_arc_synthetic_shape_pin_output(&tokens, shapes) != 0
                ? -1
                : 0,
            "exhaust_projection_and_count_shapes",
            0.86);
    }
    if ((broad_profile || strstr(family, "separator_panel") != NULL ||
         nn_gd_arc_synthetic_top_op_in(top_op,
                                       "separator_panel_map",
                                       "separator_intersection_color",
                                       "separator_panel_relation_mask",
                                       NULL)) &&
        emitted < max_rows) {
        NN_GD_SYNTHETIC_MAYBE_ADD(
            nn_gd_string_list_add_unique(&tokens, "separator_panel_map") != 0 ||
                    nn_gd_string_list_add_unique(&tokens,
                                                 "separator_intersection_color") != 0 ||
                    nn_gd_string_list_add_unique(&tokens,
                                                 "separator_panel_relation_mask") != 0 ||
                    nn_gd_string_list_add_unique(&tokens, "normalizer:intersection") != 0 ||
                    nn_gd_string_list_add_unique(&tokens, "separator_relation_frame") != 0
                ? -1
                : 0,
            "exhaust_separator_panel_shape_frames",
            0.82);
    }
#undef NN_GD_SYNTHETIC_MAYBE_ADD
    return 0;
}

static int nn_gd_arc_synthetic_write_example(FILE *fp,
                                             const char *task_id,
                                             const char *split,
                                             const char *family,
                                             const char *failure_bucket,
                                             const char *top_op,
                                             const char *reason,
                                             const char *stage,
                                             const NNGDArcTaskShapes *shapes,
                                             size_t candidate_rows) {
    fprintf(fp, "{\"candidate_rows\":%zu,\"dominant_candidate_failure_stage\":",
            candidate_rows);
    nn_gd_json_write_string(fp, stage);
    fputs(",\"dominant_candidate_rejection_reason\":", fp);
    nn_gd_json_write_string(fp, reason);
    fputs(",\"failure_bucket\":", fp);
    nn_gd_json_write_string(fp, failure_bucket);
    fputs(",\"family\":", fp);
    nn_gd_json_write_string(fp, family);
    fputs(",\"report_version\":1,\"schema\":"
          "\"crlp.arc_synthetic_shape_probe_candidates.v1\",\"split\":",
          fp);
    nn_gd_json_write_string(fp, split);
    fputs(",\"task_id\":", fp);
    nn_gd_json_write_string(fp, task_id);
    fprintf(fp,
            ",\"task_shapes\":{\"crop_like\":%s,\"first_input_height\":%ld,"
            "\"first_input_width\":%ld,\"first_output_height\":%ld,"
            "\"first_output_width\":%ld,\"grow_like\":%s,\"line_like\":%s,"
            "\"same_output_shape\":%s,\"scale\":%ld,\"train_pairs\":%ld},"
            "\"top_op_name\":",
            shapes->crop_like ? "true" : "false",
            shapes->first_input_height,
            shapes->first_input_width,
            shapes->first_output_height,
            shapes->first_output_width,
            shapes->grow_like ? "true" : "false",
            shapes->line_like ? "true" : "false",
            shapes->same_output_shape ? "true" : "false",
            shapes->scale,
            shapes->train_pairs);
    nn_gd_json_write_string(fp, top_op);
    fputs("}\n", fp);
    return ferror(fp) ? -1 : 0;
}

static int nn_gd_arc_synthetic_write_summary(
    const char *path,
    const char *failure_report_path,
    const char *public_arc_dir,
    const char *candidate_tsv_path,
    const char *examples_path,
    size_t input_failure_rows,
    size_t selected_task_rows,
    const NNGDArcHoleCandidateList *candidates,
    size_t missing_task_files,
    int include_solved,
    const char *profile,
    unsigned max_rows_per_task,
    unsigned max_tasks,
    double min_top_score,
    long max_candidate_count) {
    FILE *fp = fopen(path, "w");
    NNGDArcHoleCounterList by_strategy = {0};
    NNGDArcHoleCounterList by_stage = {0};
    size_t i;
    int status = -1;
    if (fp == NULL) {
        return -1;
    }
    for (i = 0U; i < candidates->count; ++i) {
        if (nn_gd_arc_hole_counter_add(&by_strategy,
                                       candidates->items[i].fields[12]) != 0 ||
            nn_gd_arc_hole_counter_add(&by_stage,
                                       candidates->items[i].fields[11]) != 0) {
            goto done;
        }
    }
    fprintf(fp, "{\"candidate_rows\":%zu,\"candidate_tsv\":",
            candidates->count);
    nn_gd_json_write_string(fp, candidate_tsv_path);
    fputs(",\"examples\":", fp);
    nn_gd_json_write_string(fp, examples_path);
    fputs(",\"failure_report\":", fp);
    nn_gd_json_write_string(fp, failure_report_path);
    fprintf(fp,
            ",\"include_solved\":%s,\"input_failure_rows\":%zu,"
            "\"max_candidate_count\":%ld,\"max_rows_per_task\":%u,"
            "\"max_tasks\":%u,\"min_top_score\":%.8f,"
            "\"missing_task_files\":%zu,\"profile\":",
            include_solved ? "true" : "false",
            input_failure_rows,
            max_candidate_count,
            max_rows_per_task,
            max_tasks,
            selected_task_rows == 0U ? 0.0 : min_top_score,
            missing_task_files);
    nn_gd_json_write_string(fp, profile);
    fputs(",\"public_arc_dir\":", fp);
    nn_gd_json_write_string(fp, public_arc_dir);
    fputs(",\"report_version\":1,\"rows_by_stage\":", fp);
    if (nn_gd_arc_hole_write_counter_object(fp, &by_stage) != 0) {
        goto done;
    }
    fputs(",\"rows_by_strategy\":", fp);
    if (nn_gd_arc_hole_write_counter_object(fp, &by_strategy) != 0) {
        goto done;
    }
    fprintf(fp,
            ",\"schema\":\"crlp.arc_synthetic_shape_probe_candidates.v1\","
            "\"selected_task_rows\":%zu}\n",
            selected_task_rows);
    status = ferror(fp) ? -1 : 0;

done:
    nn_gd_arc_hole_counter_list_free(&by_strategy);
    nn_gd_arc_hole_counter_list_free(&by_stage);
    if (fclose(fp) != 0) {
        status = -1;
    }
    return status;
}

int nn_grammar_discovery_write_arc_synthetic_shape_probe_candidates(
    const char *failure_report_path,
    const char *public_arc_dir,
    const char *candidate_tsv_path,
    const char *examples_path,
    const char *summary_path,
    unsigned max_rows_per_task,
    unsigned max_tasks,
    const char *profile,
    int include_solved) {
    FILE *fp = NULL;
    FILE *examples_fp = NULL;
    NNGDArcHoleCandidateList candidates = {0};
    size_t input_failure_rows = 0U;
    size_t selected_task_rows = 0U;
    size_t missing_task_files = 0U;
    double min_top_score = 0.0;
    long max_candidate_count = 0L;
    int broad_profile;
    int status = -1;
    if (failure_report_path == NULL || public_arc_dir == NULL ||
        candidate_tsv_path == NULL || examples_path == NULL ||
        summary_path == NULL || profile == NULL ||
        (strcmp(profile, "focused") != 0 && strcmp(profile, "broad") != 0)) {
        return -1;
    }
    if (max_rows_per_task == 0U) {
        return -1;
    }
    broad_profile = strcmp(profile, "broad") == 0;
    fp = fopen(failure_report_path, "r");
    if (fp == NULL) {
        return -1;
    }
    examples_fp = fopen(examples_path, "w");
    if (examples_fp == NULL) {
        goto done;
    }
    for (;;) {
        char *line = nn_gd_read_line_dynamic(fp);
        NNGDJsonIndex index;
        char *schema = NULL;
        char *failure_bucket = NULL;
        char *family = NULL;
        char *task_id = NULL;
        char *split = NULL;
        char *top_op = NULL;
        char *stage = NULL;
        char *reason = NULL;
        char *parent = NULL;
        long candidate_count;
        double top_score;
        NNGDArcTaskShapes shapes;
        size_t rows_for_example = 0U;
        size_t before_count;
        int load_status;
        if (line == NULL) {
            break;
        }
        nn_gd_json_index_line(line, &index);
        schema = nn_gd_json_dup_index_string(&index, "schema");
        if (schema == NULL) {
            free(line);
            goto done;
        }
        if (strcmp(schema, "crlp.arc_ranker_failure_task.v1") != 0) {
            free(schema);
            free(line);
            continue;
        }
        ++input_failure_rows;
        failure_bucket = nn_gd_json_dup_index_string(&index, "failure_bucket");
        if (failure_bucket == NULL) {
            free(schema);
            free(line);
            goto done;
        }
        if (!include_solved && strcmp(failure_bucket, "candidate_generation_gap") != 0) {
            free(failure_bucket);
            free(schema);
            free(line);
            continue;
        }
        if (max_tasks > 0U && selected_task_rows >= max_tasks) {
            free(failure_bucket);
            free(schema);
            free(line);
            continue;
        }
        family = nn_gd_json_dup_index_string(&index, "family");
        task_id = nn_gd_json_dup_index_string(&index, "task_id");
        split = nn_gd_json_dup_index_string(&index, "split");
        top_op = nn_gd_json_dup_index_string(&index, "top_op_name");
        stage = nn_gd_json_dup_index_string(&index, "dominant_candidate_failure_stage");
        reason = nn_gd_json_dup_index_string(&index,
                                             "dominant_candidate_rejection_reason");
        parent = nn_gd_json_dup_index_string(&index, "top_attempt_id");
        if (family == NULL || task_id == NULL || split == NULL || top_op == NULL ||
            stage == NULL || reason == NULL || parent == NULL) {
            free(parent);
            free(reason);
            free(stage);
            free(top_op);
            free(split);
            free(task_id);
            free(family);
            free(failure_bucket);
            free(schema);
            free(line);
            goto done;
        }
        if (stage[0] == '\0') {
            free(stage);
            stage = nn_gd_json_dup_index_string(&index, "top_failure_stage");
            if (stage == NULL) {
                goto done;
            }
        }
        if (reason[0] == '\0') {
            free(reason);
            reason = nn_gd_json_dup_index_string(&index, "top_rejection_reason");
            if (reason == NULL) {
                goto done;
            }
        }
        candidate_count = nn_gd_json_index_long_or(&index, "candidate_count", 0L);
        top_score = nn_gd_json_index_double_or_zero(&index, "top_score");
        load_status = nn_gd_arc_load_task_shapes(public_arc_dir, split, task_id, &shapes);
        if (load_status > 0) {
            ++missing_task_files;
        } else if (load_status < 0) {
            free(parent);
            free(reason);
            free(stage);
            free(top_op);
            free(split);
            free(task_id);
            free(family);
            free(failure_bucket);
            free(schema);
            free(line);
            goto done;
        }
        before_count = candidates.count;
        if (nn_gd_arc_synthetic_emit_strategy_rows(&candidates,
                                                   family,
                                                   task_id,
                                                   split,
                                                   top_op,
                                                   stage,
                                                   reason,
                                                   parent,
                                                   candidate_count,
                                                   &shapes,
                                                   max_rows_per_task,
                                                   broad_profile,
                                                   &rows_for_example) != 0 ||
            nn_gd_arc_synthetic_write_example(examples_fp,
                                              task_id,
                                              split,
                                              family,
                                              failure_bucket,
                                              top_op,
                                              reason,
                                              stage,
                                              &shapes,
                                              rows_for_example) != 0) {
            free(parent);
            free(reason);
            free(stage);
            free(top_op);
            free(split);
            free(task_id);
            free(family);
            free(failure_bucket);
            free(schema);
            free(line);
            goto done;
        }
        (void)before_count;
        if (selected_task_rows == 0U || top_score < min_top_score) {
            min_top_score = top_score;
        }
        if (candidate_count > max_candidate_count) {
            max_candidate_count = candidate_count;
        }
        ++selected_task_rows;
        free(parent);
        free(reason);
        free(stage);
        free(top_op);
        free(split);
        free(task_id);
        free(family);
        free(failure_bucket);
        free(schema);
        free(line);
    }
    if (fclose(examples_fp) != 0) {
        examples_fp = NULL;
        goto done;
    }
    examples_fp = NULL;
    if (nn_gd_arc_hole_write_candidate_tsv(candidate_tsv_path, &candidates) != 0 ||
        nn_gd_arc_synthetic_write_summary(summary_path,
                                          failure_report_path,
                                          public_arc_dir,
                                          candidate_tsv_path,
                                          examples_path,
                                          input_failure_rows,
                                          selected_task_rows,
                                          &candidates,
                                          missing_task_files,
                                          include_solved,
                                          profile,
                                          max_rows_per_task,
                                          max_tasks,
                                          min_top_score,
                                          max_candidate_count) != 0) {
        goto done;
    }
    printf("arc-synthetic-shape-probe-candidates tasks=%zu candidate_rows=%zu tsv=%s\n",
           selected_task_rows,
           candidates.count,
           candidate_tsv_path);
    status = 0;

done:
    if (examples_fp != NULL) {
        fclose(examples_fp);
    }
    if (fp != NULL) {
        fclose(fp);
    }
    nn_gd_arc_hole_candidate_list_free(&candidates);
    return status;
}

static const char *NN_GD_ARC_APPEND_RUNTIME_HEADER =
    "split_name\tfamily\ttask_id\tsplit\tattempt_id\tlibrary_id\tattempt_index\t"
    "pipeline_kind\top_name\tvisual_program\tslot_contract\tcomposition_kind\t"
    "normalization_frame\ttyped_stages\tprimitive_contract\tslot_tokens\t"
    "accepted\trejection_reason\tfailure_stage\tfailure_detail\tproposal_count\t"
    "support_tasks\tutility_mean\tselector\tbackground\tcolor\toutput_color\t"
    "selector_kind\tinclude_background\torientation\tcount_mode\tx_align\ty_align\t"
    "x_anchor\ty_anchor\tx_offset\ty_offset\twidth\theight\tperiod\t"
    "contrast_accepted_attempt_id\tcontrast_slot_distance\tcontrast_pair_kind\t"
    "contrast_rejected_sibling_count\tinput_width\tinput_height\toutput_width\t"
    "output_height\tshape_scale_x\tshape_scale_y\tperiod_divides_input_width\t"
    "period_divides_input_height\tslot_width_matches_output\t"
    "slot_height_matches_output\tcolor_in_input\tcolor_in_output\t"
    "output_color_in_output\tbackground_in_input\tverifier_train_pairs\t"
    "verifier_checked_cells\tverifier_mismatches\tverifier_unsupported_pairs\t"
    "verifier_apply_failed_pairs\tverifier_shape_mismatch_pairs\t"
    "verifier_missing_output_pairs\tverifier_first_failing_pair\t"
    "verifier_mismatch_ratio\tverifier_score\tinput_component_count\t"
    "selected_component_count\tselected_component_area_min\t"
    "selected_component_area_max\tselected_component_area_total\t"
    "selected_component_same_shape\tselected_component_shape_count\t"
    "selected_component_bbox_width\tselected_component_bbox_height\t"
    "selected_bbox_matches_output_shape\tselected_bbox_rotated_output_shape\t"
    "output_foreground_cell_count\toutput_component_count\t"
    "output_non_background_color_count\tselected_count_matches_output_components\t"
    "selected_count_matches_output_foreground\tselected_area_matches_output_foreground";

static void nn_gd_strip_line_end(char *line) {
    size_t len;
    if (line == NULL) {
        return;
    }
    len = strlen(line);
    while (len > 0U && (line[len - 1U] == '\n' || line[len - 1U] == '\r')) {
        line[--len] = '\0';
    }
}

static int nn_gd_append_key_part(char **buffer,
                                 size_t *used,
                                 size_t *capacity,
                                 const char *value,
                                 size_t len) {
    size_t i;
    for (i = 0U; i < len; ++i) {
        char ch = value[i];
        if (ch == '\r' || ch == '\n') {
            break;
        }
        if (nn_gd_json_append_char(buffer, used, capacity, ch) != 0) {
            return -1;
        }
    }
    return 0;
}

static char *nn_gd_arc_append_key_from_parts(const char *split_name,
                                             const char *family,
                                             const char *task_id,
                                             const char *split,
                                             const char *library_id) {
    const char *parts[5];
    size_t capacity = 128U;
    size_t used = 0U;
    char *key = (char *)malloc(capacity);
    size_t i;
    if (key == NULL) {
        return NULL;
    }
    parts[0] = split_name == NULL ? "" : split_name;
    parts[1] = family == NULL ? "" : family;
    parts[2] = task_id == NULL ? "" : task_id;
    parts[3] = split == NULL ? "" : split;
    parts[4] = library_id == NULL ? "" : library_id;
    for (i = 0U; i < 5U; ++i) {
        if (i > 0U && nn_gd_json_append_char(&key, &used, &capacity, '\x1f') != 0) {
            free(key);
            return NULL;
        }
        if (nn_gd_append_key_part(&key,
                                  &used,
                                  &capacity,
                                  parts[i],
                                  strlen(parts[i])) != 0) {
            free(key);
            return NULL;
        }
    }
    key[used] = '\0';
    return key;
}

static char *nn_gd_arc_append_key_from_tsv_line(const char *line) {
    static const int wanted[5] = {0, 1, 2, 3, 5};
    const char *parts[5] = {"", "", "", "", ""};
    size_t lens[5] = {0U, 0U, 0U, 0U, 0U};
    const char *cursor = line;
    int field = 0;
    int want = 0;
    size_t capacity = 128U;
    size_t used = 0U;
    char *key = (char *)malloc(capacity);
    size_t i;
    if (key == NULL) {
        return NULL;
    }
    while (cursor != NULL && *cursor != '\0' && want < 5) {
        const char *start = cursor;
        const char *end;
        while (*cursor != '\0' && *cursor != '\t' &&
               *cursor != '\n' && *cursor != '\r') {
            ++cursor;
        }
        end = cursor;
        if (field == wanted[want]) {
            parts[want] = start;
            lens[want] = (size_t)(end - start);
            ++want;
        }
        if (*cursor != '\t') {
            break;
        }
        ++cursor;
        ++field;
    }
    for (i = 0U; i < 5U; ++i) {
        if (i > 0U && nn_gd_json_append_char(&key, &used, &capacity, '\x1f') != 0) {
            free(key);
            return NULL;
        }
        if (nn_gd_append_key_part(&key, &used, &capacity, parts[i], lens[i]) != 0) {
            free(key);
            return NULL;
        }
    }
    key[used] = '\0';
    return key;
}

static const char *nn_gd_json_find_object_field_value(const char *line,
                                                      const char *object_key,
                                                      const char *field_key) {
    const char *object_start = nn_gd_json_skip_space(
        nn_gd_json_find_key_value(line, object_key));
    const char *object_end;
    if (object_start == NULL || *object_start != '{') {
        return NULL;
    }
    object_end = nn_gd_arc_hole_json_object_end(object_start);
    if (object_end == NULL || object_end <= object_start) {
        return NULL;
    }
    return nn_gd_json_find_key_value_range(object_start, object_end, field_key);
}

static char *nn_gd_json_dup_object_scalar_tsv(const char *line,
                                              const char *object_key,
                                              const char *field_key,
                                              int bool_as_int) {
    const char *cursor =
        nn_gd_json_find_object_field_value(line, object_key, field_key);
    return nn_gd_json_dup_scalar_tsv_value_at_stable_number(&cursor, bool_as_int);
}

static char *nn_gd_json_dup_top_or_object_scalar_tsv(const char *line,
                                                     const NNGDJsonIndex *index,
                                                     const char *top_key,
                                                     const char *object_key,
                                                     const char *field_key,
                                                     int bool_as_int) {
    char *value = nn_gd_json_dup_index_scalar_tsv(index, top_key, bool_as_int);
    if (value == NULL || value[0] != '\0') {
        return value;
    }
    free(value);
    return nn_gd_json_dup_object_scalar_tsv(line,
                                            object_key,
                                            field_key,
                                            bool_as_int);
}

static char *nn_gd_arc_append_key_from_json(const NNGDJsonIndex *index,
                                            const char *split_name) {
    char *family = nn_gd_json_dup_index_scalar_tsv(index, "family", 1);
    char *task_id = nn_gd_json_dup_index_scalar_tsv(index, "task_id", 1);
    char *split = nn_gd_json_dup_index_scalar_tsv(index, "split", 1);
    char *library_id = nn_gd_json_dup_index_scalar_tsv(index, "library_id", 1);
    char *key = NULL;
    if (family != NULL && task_id != NULL && split != NULL && library_id != NULL) {
        key = nn_gd_arc_append_key_from_parts(split_name,
                                              family,
                                              task_id,
                                              split,
                                              library_id);
    }
    free(family);
    free(task_id);
    free(split);
    free(library_id);
    return key;
}

static int nn_gd_arc_append_write_probe_attempt(FILE *target,
                                                const char *line,
                                                const char *split_name) {
    NNGDJsonIndex index;
    char *fields[85];
    size_t i;
    int status = -1;
    memset(fields, 0, sizeof(fields));
    nn_gd_json_index_line(line, &index);
    fields[0] = nn_gd_strdup(split_name == NULL ? "" : split_name);
    fields[1] = nn_gd_json_dup_index_scalar_tsv(&index, "family", 1);
    fields[2] = nn_gd_json_dup_index_scalar_tsv(&index, "task_id", 1);
    fields[3] = nn_gd_json_dup_index_scalar_tsv(&index, "split", 1);
    fields[4] = nn_gd_json_dup_index_scalar_tsv(&index, "attempt_id", 1);
    fields[5] = nn_gd_json_dup_index_scalar_tsv(&index, "library_id", 1);
    fields[6] = nn_gd_json_dup_index_scalar_tsv(&index, "attempt_index", 1);
    fields[7] = nn_gd_json_dup_index_scalar_tsv(&index, "pipeline_kind", 1);
    fields[8] = nn_gd_json_dup_index_scalar_tsv(&index, "op_name", 1);
    fields[9] = nn_gd_json_dup_index_scalar_tsv(&index, "visual_program", 1);
    fields[10] = nn_gd_json_dup_index_scalar_tsv(&index, "slot_contract", 1);
    fields[11] = nn_gd_json_dup_top_or_object_scalar_tsv(
        line, &index, "composition_kind", "pipeline", "composition_kind", 1);
    fields[12] = nn_gd_json_dup_top_or_object_scalar_tsv(
        line, &index, "normalization_frame", "pipeline", "normalization_frame", 1);
    fields[13] = nn_gd_json_dup_top_or_object_scalar_tsv(
        line, &index, "typed_stages", "pipeline", "typed_stages", 1);
    fields[14] = nn_gd_json_dup_top_or_object_scalar_tsv(
        line, &index, "primitive_contract", "pipeline", "primitive_contract", 1);
    fields[15] = nn_gd_json_dup_slot_tokens_at_ex(
        nn_gd_json_index_find(&index, "slot_values"), 1, 0);
    fields[16] = nn_gd_json_dup_index_scalar_tsv(&index, "accepted", 1);
    fields[17] = nn_gd_json_dup_index_scalar_tsv(&index, "rejection_reason", 1);
    fields[18] = nn_gd_json_dup_index_scalar_tsv(&index, "failure_stage", 1);
    fields[19] = nn_gd_json_dup_index_scalar_tsv(&index, "failure_detail", 1);
    fields[20] = nn_gd_json_dup_index_scalar_tsv(&index, "proposal_count", 1);
    fields[21] = nn_gd_json_dup_index_scalar_tsv(&index, "support_tasks", 1);
    fields[22] =
        nn_gd_json_dup_index_scalar_tsv_stable_number(&index, "utility_mean", 1);
    fields[23] = nn_gd_json_dup_object_scalar_tsv(line, "pipeline", "selector", 1);
    fields[24] = nn_gd_json_dup_object_scalar_tsv(line, "pipeline", "background", 1);
    fields[25] = nn_gd_json_dup_object_scalar_tsv(line, "pipeline", "color", 1);
    fields[26] = nn_gd_json_dup_object_scalar_tsv(line, "pipeline", "output_color", 1);
    fields[27] = nn_gd_json_dup_object_scalar_tsv(line, "pipeline", "selector_kind", 1);
    fields[28] =
        nn_gd_json_dup_object_scalar_tsv(line, "pipeline", "include_background", 1);
    fields[29] = nn_gd_json_dup_object_scalar_tsv(line, "pipeline", "orientation", 1);
    fields[30] = nn_gd_json_dup_object_scalar_tsv(line, "pipeline", "count_mode", 1);
    fields[31] = nn_gd_json_dup_object_scalar_tsv(line, "pipeline", "x_align", 1);
    fields[32] = nn_gd_json_dup_object_scalar_tsv(line, "pipeline", "y_align", 1);
    fields[33] = nn_gd_json_dup_object_scalar_tsv(line, "pipeline", "x_anchor", 1);
    fields[34] = nn_gd_json_dup_object_scalar_tsv(line, "pipeline", "y_anchor", 1);
    fields[35] = nn_gd_json_dup_object_scalar_tsv(line, "pipeline", "x_offset", 1);
    fields[36] = nn_gd_json_dup_object_scalar_tsv(line, "pipeline", "y_offset", 1);
    fields[37] = nn_gd_json_dup_object_scalar_tsv(line, "pipeline", "width", 1);
    fields[38] = nn_gd_json_dup_object_scalar_tsv(line, "pipeline", "height", 1);
    fields[39] = nn_gd_json_dup_object_scalar_tsv(line, "pipeline", "period", 1);
    fields[40] =
        nn_gd_json_dup_index_scalar_tsv(&index, "contrast_accepted_attempt_id", 1);
    fields[41] =
        nn_gd_json_dup_index_scalar_tsv(&index, "contrast_slot_distance", 1);
    fields[42] = nn_gd_json_dup_index_scalar_tsv(&index, "contrast_pair_kind", 1);
    fields[43] =
        nn_gd_json_dup_index_scalar_tsv(&index, "contrast_rejected_sibling_count", 1);
    fields[44] =
        nn_gd_json_dup_object_scalar_tsv(line, "task_slot_features", "input_width", 1);
    fields[45] =
        nn_gd_json_dup_object_scalar_tsv(line, "task_slot_features", "input_height", 1);
    fields[46] =
        nn_gd_json_dup_object_scalar_tsv(line, "task_slot_features", "output_width", 1);
    fields[47] =
        nn_gd_json_dup_object_scalar_tsv(line, "task_slot_features", "output_height", 1);
    fields[48] =
        nn_gd_json_dup_object_scalar_tsv(line, "task_slot_features", "shape_scale_x", 1);
    fields[49] =
        nn_gd_json_dup_object_scalar_tsv(line, "task_slot_features", "shape_scale_y", 1);
    fields[50] = nn_gd_json_dup_object_scalar_tsv(
        line, "task_slot_features", "period_divides_input_width", 1);
    fields[51] = nn_gd_json_dup_object_scalar_tsv(
        line, "task_slot_features", "period_divides_input_height", 1);
    fields[52] = nn_gd_json_dup_object_scalar_tsv(
        line, "task_slot_features", "slot_width_matches_output", 1);
    fields[53] = nn_gd_json_dup_object_scalar_tsv(
        line, "task_slot_features", "slot_height_matches_output", 1);
    fields[54] =
        nn_gd_json_dup_object_scalar_tsv(line, "task_slot_features", "color_in_input", 1);
    fields[55] =
        nn_gd_json_dup_object_scalar_tsv(line, "task_slot_features", "color_in_output", 1);
    fields[56] = nn_gd_json_dup_object_scalar_tsv(
        line, "task_slot_features", "output_color_in_output", 1);
    fields[57] = nn_gd_json_dup_object_scalar_tsv(
        line, "task_slot_features", "background_in_input", 1);
    fields[58] =
        nn_gd_json_dup_object_scalar_tsv(line, "verifier_distance", "train_pairs", 1);
    fields[59] =
        nn_gd_json_dup_object_scalar_tsv(line, "verifier_distance", "checked_cells", 1);
    fields[60] =
        nn_gd_json_dup_object_scalar_tsv(line, "verifier_distance", "mismatches", 1);
    fields[61] = nn_gd_json_dup_object_scalar_tsv(
        line, "verifier_distance", "unsupported_pairs", 1);
    fields[62] = nn_gd_json_dup_object_scalar_tsv(
        line, "verifier_distance", "apply_failed_pairs", 1);
    fields[63] = nn_gd_json_dup_object_scalar_tsv(
        line, "verifier_distance", "shape_mismatch_pairs", 1);
    fields[64] = nn_gd_json_dup_object_scalar_tsv(
        line, "verifier_distance", "missing_output_pairs", 1);
    fields[65] = nn_gd_json_dup_object_scalar_tsv(
        line, "verifier_distance", "first_failing_pair", 1);
    fields[66] = nn_gd_json_dup_object_scalar_tsv(
        line, "verifier_distance", "mismatch_ratio", 1);
    fields[67] = nn_gd_json_dup_object_scalar_tsv(
        line, "verifier_distance", "verifier_score", 1);
    fields[68] = nn_gd_json_dup_object_scalar_tsv(
        line, "component_inventory", "input_component_count", 1);
    fields[69] = nn_gd_json_dup_object_scalar_tsv(
        line, "component_inventory", "selected_component_count", 1);
    fields[70] = nn_gd_json_dup_object_scalar_tsv(
        line, "component_inventory", "selected_component_area_min", 1);
    fields[71] = nn_gd_json_dup_object_scalar_tsv(
        line, "component_inventory", "selected_component_area_max", 1);
    fields[72] = nn_gd_json_dup_object_scalar_tsv(
        line, "component_inventory", "selected_component_area_total", 1);
    fields[73] = nn_gd_json_dup_object_scalar_tsv(
        line, "component_inventory", "selected_component_same_shape", 1);
    fields[74] = nn_gd_json_dup_object_scalar_tsv(
        line, "component_inventory", "selected_component_shape_count", 1);
    fields[75] = nn_gd_json_dup_object_scalar_tsv(
        line, "component_inventory", "selected_component_bbox_width", 1);
    fields[76] = nn_gd_json_dup_object_scalar_tsv(
        line, "component_inventory", "selected_component_bbox_height", 1);
    fields[77] = nn_gd_json_dup_object_scalar_tsv(
        line, "component_inventory", "selected_bbox_matches_output_shape", 1);
    fields[78] = nn_gd_json_dup_object_scalar_tsv(
        line, "component_inventory", "selected_bbox_rotated_output_shape", 1);
    fields[79] = nn_gd_json_dup_object_scalar_tsv(
        line, "component_inventory", "output_foreground_cell_count", 1);
    fields[80] = nn_gd_json_dup_object_scalar_tsv(
        line, "component_inventory", "output_component_count", 1);
    fields[81] = nn_gd_json_dup_object_scalar_tsv(
        line, "component_inventory", "output_non_background_color_count", 1);
    fields[82] = nn_gd_json_dup_object_scalar_tsv(
        line, "component_inventory", "selected_count_matches_output_components", 1);
    fields[83] = nn_gd_json_dup_object_scalar_tsv(
        line, "component_inventory", "selected_count_matches_output_foreground", 1);
    fields[84] = nn_gd_json_dup_object_scalar_tsv(
        line, "component_inventory", "selected_area_matches_output_foreground", 1);
    for (i = 0U; i < 85U; ++i) {
        if (fields[i] == NULL) {
            goto done;
        }
    }
    for (i = 0U; i < 85U; ++i) {
        if (i > 0U) {
            fputc('\t', target);
        }
        fputs(fields[i], target);
    }
    fputc('\n', target);
    status = ferror(target) ? -1 : 0;

done:
    for (i = 0U; i < 85U; ++i) {
        free(fields[i]);
    }
    return status;
}

static int nn_gd_arc_append_write_summary(
    const char *summary_path,
    const char *base_attempts_tsv_path,
    const char *probe_attempts_jsonl_path,
    const char *combined_attempts_tsv_path,
    const char *split_name,
    int accepted_only,
    int dedupe,
    size_t base_rows,
    size_t appended_rows,
    size_t skipped_duplicate_rows,
    size_t skipped_rejected_rows,
    NNGDArcHoleCounterList *appended_by_acceptance,
    NNGDArcHoleCounterList *appended_by_reason,
    NNGDArcHoleCounterList *appended_rows_by_task) {
    FILE *fp = fopen(summary_path, "w");
    int status = -1;
    if (fp == NULL) {
        return -1;
    }
    fputs("{\"accepted_only\":", fp);
    fputs(accepted_only ? "true" : "false", fp);
    fputs(",\"appended_by_acceptance\":", fp);
    if (nn_gd_arc_hole_write_counter_object(fp, appended_by_acceptance) != 0) {
        goto done;
    }
    fputs(",\"appended_by_reason\":", fp);
    if (nn_gd_arc_hole_write_counter_object(fp, appended_by_reason) != 0) {
        goto done;
    }
    fprintf(fp, ",\"appended_rows\":%zu,\"appended_rows_by_task\":", appended_rows);
    if (nn_gd_arc_hole_write_counter_object(fp, appended_rows_by_task) != 0) {
        goto done;
    }
    fprintf(fp, ",\"appended_task_count\":%zu,\"base_attempts_tsv\":",
            appended_rows_by_task->count);
    nn_gd_json_write_string(fp, base_attempts_tsv_path);
    fprintf(fp, ",\"base_rows\":%zu,\"combined_attempts_tsv\":", base_rows);
    nn_gd_json_write_string(fp, combined_attempts_tsv_path);
    fputs(",\"dedupe\":", fp);
    fputs(dedupe ? "true" : "false", fp);
    fputs(",\"probe_attempts_jsonl\":", fp);
    nn_gd_json_write_string(fp, probe_attempts_jsonl_path);
    fprintf(fp,
            ",\"report_version\":1,\"schema\":\"crlp.arc_probe_attempts_tsv_append.v1\","
            "\"skipped_duplicate_rows\":%zu,\"skipped_rejected_rows\":%zu,"
            "\"split_name\":",
            skipped_duplicate_rows,
            skipped_rejected_rows);
    nn_gd_json_write_string(fp, split_name);
    fputs("}\n", fp);
    status = ferror(fp) ? -1 : 0;

done:
    if (fclose(fp) != 0) {
        status = -1;
    }
    return status;
}

int nn_grammar_discovery_append_arc_probe_attempts_tsv(
    const char *base_attempts_tsv_path,
    const char *probe_attempts_jsonl_path,
    const char *combined_attempts_tsv_path,
    const char *summary_path,
    const char *split_name,
    int accepted_only,
    int dedupe) {
    FILE *base = NULL;
    FILE *probe = NULL;
    FILE *target = NULL;
    NNGDStringList seen_keys = {0};
    NNGDStringTable seen_table = {0};
    NNGDArcHoleCounterList appended_by_acceptance = {0};
    NNGDArcHoleCounterList appended_by_reason = {0};
    NNGDArcHoleCounterList appended_rows_by_task = {0};
    size_t base_rows = 0U;
    size_t appended_rows = 0U;
    size_t skipped_duplicate_rows = 0U;
    size_t skipped_rejected_rows = 0U;
    int status = -1;
    if (base_attempts_tsv_path == NULL || probe_attempts_jsonl_path == NULL ||
        combined_attempts_tsv_path == NULL || summary_path == NULL) {
        return -1;
    }
    if (split_name == NULL) {
        split_name = "target";
    }
    base = fopen(base_attempts_tsv_path, "r");
    if (base == NULL) {
        return -1;
    }
    probe = fopen(probe_attempts_jsonl_path, "r");
    if (probe == NULL) {
        goto done;
    }
    target = fopen(combined_attempts_tsv_path, "w");
    if (target == NULL) {
        goto done;
    }
    {
        char *header = nn_gd_read_line_dynamic(base);
        char *header_check;
        if (header == NULL) {
            goto done;
        }
        header_check = nn_gd_strdup(header);
        if (header_check == NULL) {
            free(header);
            goto done;
        }
        nn_gd_strip_line_end(header_check);
        if (strcmp(header_check, NN_GD_ARC_APPEND_RUNTIME_HEADER) != 0) {
            free(header_check);
            free(header);
            goto done;
        }
        fputs(header, target);
        free(header_check);
        free(header);
    }
    for (;;) {
        char *line = nn_gd_read_line_dynamic(base);
        char *key;
        if (line == NULL) {
            break;
        }
        fputs(line, target);
        key = nn_gd_arc_append_key_from_tsv_line(line);
        if (key == NULL ||
            nn_gd_string_table_add_unique(&seen_keys, &seen_table, key) != 0) {
            free(key);
            free(line);
            goto done;
        }
        free(key);
        free(line);
        ++base_rows;
    }
    for (;;) {
        char *line = nn_gd_read_line_dynamic(probe);
        const char *stripped;
        NNGDJsonIndex index;
        char *schema = NULL;
        char *accepted_value = NULL;
        char *key = NULL;
        char *task_id = NULL;
        char *reason = NULL;
        int accepted;
        if (line == NULL) {
            break;
        }
        stripped = nn_gd_json_skip_space(line);
        if (stripped == NULL || *stripped != '{') {
            free(line);
            continue;
        }
        nn_gd_json_index_line(line, &index);
        schema = nn_gd_json_dup_index_string(&index, "schema");
        if (schema == NULL) {
            free(line);
            goto done;
        }
        if (strcmp(schema,
                   "crlp.neural_grammar_discovery.arc_visual_pipeline_attempt.v1") != 0) {
            free(schema);
            free(line);
            continue;
        }
        accepted_value = nn_gd_json_dup_index_scalar_tsv(&index, "accepted", 1);
        if (accepted_value == NULL) {
            free(schema);
            free(line);
            goto done;
        }
        accepted = strcmp(accepted_value, "1") == 0;
        if (accepted_only && !accepted) {
            ++skipped_rejected_rows;
            free(accepted_value);
            free(schema);
            free(line);
            continue;
        }
        key = nn_gd_arc_append_key_from_json(&index, split_name);
        if (key == NULL) {
            free(accepted_value);
            free(schema);
            free(line);
            goto done;
        }
        if (dedupe && nn_gd_string_table_find(&seen_table, &seen_keys, key) >= 0L) {
            ++skipped_duplicate_rows;
            free(key);
            free(accepted_value);
            free(schema);
            free(line);
            continue;
        }
        if (nn_gd_string_table_add_unique(&seen_keys, &seen_table, key) != 0) {
            free(key);
            free(accepted_value);
            free(schema);
            free(line);
            goto done;
        }
        free(key);
        if (nn_gd_arc_append_write_probe_attempt(target, line, split_name) != 0) {
            free(accepted_value);
            free(schema);
            free(line);
            goto done;
        }
        ++appended_rows;
        if (nn_gd_arc_hole_counter_add(&appended_by_acceptance,
                                       accepted ? "accepted" : "rejected") != 0) {
            free(accepted_value);
            free(schema);
            free(line);
            goto done;
        }
        reason = nn_gd_json_dup_index_scalar_tsv(&index, "rejection_reason", 1);
        task_id = nn_gd_json_dup_index_scalar_tsv(&index, "task_id", 1);
        if (reason == NULL || task_id == NULL ||
            nn_gd_arc_hole_counter_add(&appended_by_reason,
                                       reason[0] == '\0' ? "unknown" : reason) != 0 ||
            nn_gd_arc_hole_counter_add(&appended_rows_by_task, task_id) != 0) {
            free(task_id);
            free(reason);
            free(accepted_value);
            free(schema);
            free(line);
            goto done;
        }
        free(task_id);
        free(reason);
        free(accepted_value);
        free(schema);
        free(line);
    }
    if (fclose(target) != 0) {
        target = NULL;
        goto done;
    }
    target = NULL;
    if (nn_gd_arc_append_write_summary(summary_path,
                                       base_attempts_tsv_path,
                                       probe_attempts_jsonl_path,
                                       combined_attempts_tsv_path,
                                       split_name,
                                       accepted_only,
                                       dedupe,
                                       base_rows,
                                       appended_rows,
                                       skipped_duplicate_rows,
                                       skipped_rejected_rows,
                                       &appended_by_acceptance,
                                       &appended_by_reason,
                                       &appended_rows_by_task) != 0) {
        goto done;
    }
    printf("arc-append-probe-attempts-tsv base_rows=%zu appended=%zu combined=%s\n",
           base_rows,
           appended_rows,
           combined_attempts_tsv_path);
    status = 0;

done:
    if (target != NULL) {
        fclose(target);
    }
    if (probe != NULL) {
        fclose(probe);
    }
    if (base != NULL) {
        fclose(base);
    }
    nn_gd_string_table_free(&seen_table);
    nn_gd_string_list_free(&seen_keys);
    nn_gd_arc_hole_counter_list_free(&appended_by_acceptance);
    nn_gd_arc_hole_counter_list_free(&appended_by_reason);
    nn_gd_arc_hole_counter_list_free(&appended_rows_by_task);
    return status;
}

typedef struct {
    const char *label;
    char proposals_path[4096];
    char summary_path[4096];
    char proposals_hash[32];
    char summary_hash[32];
    char *epochs;
    char *source_attempt_count;
    char *target_attempt_count;
    char *target_task_count;
    char *accepted_target_task_count;
    char *top1_rate;
    char *top1_all_task_rate;
    char *retained_rate;
    char *vocab_size;
    char *proposal_count;
    double elapsed_seconds;
} NNGDArcRankerBenchmarkRun;

static void nn_gd_arc_ranker_benchmark_run_free(
    NNGDArcRankerBenchmarkRun *run) {
    if (run == NULL) {
        return;
    }
    free(run->epochs);
    free(run->source_attempt_count);
    free(run->target_attempt_count);
    free(run->target_task_count);
    free(run->accepted_target_task_count);
    free(run->top1_rate);
    free(run->top1_all_task_rate);
    free(run->retained_rate);
    free(run->vocab_size);
    free(run->proposal_count);
    memset(run, 0, sizeof(*run));
}

static int nn_gd_join_path(char *out,
                           size_t out_size,
                           const char *dir,
                           const char *leaf) {
    int written;
    if (out == NULL || out_size == 0U || dir == NULL || leaf == NULL) {
        return -1;
    }
    written = snprintf(out,
                       out_size,
                       "%s%s%s",
                       dir,
                       dir[0] != '\0' && dir[strlen(dir) - 1U] == '/' ? "" : "/",
                       leaf);
    return written < 0 || (size_t)written >= out_size ? -1 : 0;
}

static long long nn_gd_file_size_bytes(const char *path) {
    struct stat st;
    if (path == NULL || stat(path, &st) != 0) {
        return 0LL;
    }
    return (long long)st.st_size;
}

static uint64_t nn_gd_fnv1a_update_bytes(uint64_t hash,
                                         const unsigned char *bytes,
                                         size_t count) {
    size_t i;
    for (i = 0U; i < count; ++i) {
        hash ^= (uint64_t)bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int nn_gd_file_hash_hex(const char *path, char out[32]) {
    FILE *fp = fopen(path, "rb");
    uint64_t hash = UINT64_C(1469598103934665603);
    unsigned char buffer[8192];
    if (out == NULL) {
        return -1;
    }
    out[0] = '\0';
    if (fp == NULL) {
        return -1;
    }
    for (;;) {
        size_t count = fread(buffer, 1U, sizeof(buffer), fp);
        if (count > 0U) {
            hash = nn_gd_fnv1a_update_bytes(hash, buffer, count);
        }
        if (count < sizeof(buffer)) {
            if (ferror(fp)) {
                fclose(fp);
                return -1;
            }
            break;
        }
    }
    if (fclose(fp) != 0) {
        return -1;
    }
    snprintf(out, 32U, "fnv64:%016" PRIx64, hash);
    return 0;
}

static int nn_gd_files_equal(const char *a_path, const char *b_path) {
    FILE *a = fopen(a_path, "rb");
    FILE *b = fopen(b_path, "rb");
    unsigned char a_buffer[8192];
    unsigned char b_buffer[8192];
    int equal = 0;
    if (a == NULL || b == NULL) {
        if (a != NULL) {
            fclose(a);
        }
        if (b != NULL) {
            fclose(b);
        }
        return 0;
    }
    for (;;) {
        size_t a_count = fread(a_buffer, 1U, sizeof(a_buffer), a);
        size_t b_count = fread(b_buffer, 1U, sizeof(b_buffer), b);
        if (a_count != b_count || memcmp(a_buffer, b_buffer, a_count) != 0) {
            break;
        }
        if (a_count < sizeof(a_buffer)) {
            equal = !ferror(a) && !ferror(b);
            break;
        }
    }
    fclose(a);
    fclose(b);
    return equal;
}

static char *nn_gd_json_dup_first_row_scalar(const char *path,
                                             const char *key) {
    FILE *fp = fopen(path, "r");
    char *line;
    char *value;
    NNGDJsonIndex index;
    if (fp == NULL) {
        return nn_gd_strdup("");
    }
    line = nn_gd_read_line_dynamic(fp);
    fclose(fp);
    if (line == NULL) {
        return nn_gd_strdup("");
    }
    nn_gd_json_index_line(line, &index);
    value = nn_gd_json_dup_index_scalar_tsv_stable_number(&index, key, 1);
    free(line);
    return value;
}

static int nn_gd_arc_ranker_benchmark_run(
    const char *attempt_rows_path,
    const char *cache_path,
    const char *output_dir,
    const char *label,
    unsigned epochs,
    double learning_rate,
    unsigned max_per_task,
    NNGDArcRankerBenchmarkRun *run) {
    char leaf[128];
    clock_t start;
    clock_t end;
    memset(run, 0, sizeof(*run));
    run->label = label;
    if (snprintf(leaf, sizeof(leaf), "%s.proposals.jsonl", label) < 0 ||
        nn_gd_join_path(run->proposals_path,
                        sizeof(run->proposals_path),
                        output_dir,
                        leaf) != 0 ||
        snprintf(leaf, sizeof(leaf), "%s.summary.jsonl", label) < 0 ||
        nn_gd_join_path(run->summary_path,
                        sizeof(run->summary_path),
                        output_dir,
                        leaf) != 0) {
        return -1;
    }
    start = clock();
    if (nn_grammar_discovery_write_arc_visual_pipeline_ranker_with_cache(
            attempt_rows_path,
            cache_path,
            run->proposals_path,
            run->summary_path,
            epochs,
            learning_rate,
            max_per_task) != 0) {
        return -1;
    }
    end = clock();
    run->elapsed_seconds = (double)(end - start) / (double)CLOCKS_PER_SEC;
    if (nn_gd_file_hash_hex(run->summary_path, run->summary_hash) != 0 ||
        nn_gd_file_hash_hex(run->proposals_path, run->proposals_hash) != 0) {
        return -1;
    }
    run->epochs = nn_gd_json_dup_first_row_scalar(run->summary_path, "epochs");
    run->source_attempt_count =
        nn_gd_json_dup_first_row_scalar(run->summary_path, "source_attempt_count");
    run->target_attempt_count =
        nn_gd_json_dup_first_row_scalar(run->summary_path, "target_attempt_count");
    run->target_task_count =
        nn_gd_json_dup_first_row_scalar(run->summary_path, "target_task_count");
    run->accepted_target_task_count = nn_gd_json_dup_first_row_scalar(
        run->summary_path,
        "accepted_target_task_count");
    run->top1_rate = nn_gd_json_dup_first_row_scalar(run->summary_path, "top1_rate");
    run->top1_all_task_rate =
        nn_gd_json_dup_first_row_scalar(run->summary_path, "top1_all_task_rate");
    run->retained_rate =
        nn_gd_json_dup_first_row_scalar(run->summary_path, "retained_rate");
    run->vocab_size = nn_gd_json_dup_first_row_scalar(run->summary_path, "vocab_size");
    run->proposal_count =
        nn_gd_json_dup_first_row_scalar(run->summary_path, "proposal_count");
    if (run->epochs == NULL || run->source_attempt_count == NULL ||
        run->target_attempt_count == NULL || run->target_task_count == NULL ||
        run->accepted_target_task_count == NULL || run->top1_rate == NULL ||
        run->top1_all_task_rate == NULL || run->retained_rate == NULL ||
        run->vocab_size == NULL || run->proposal_count == NULL) {
        return -1;
    }
    return 0;
}

static void nn_gd_json_write_raw_or_null(FILE *fp, const char *value) {
    if (value == NULL || value[0] == '\0') {
        fputs("null", fp);
    } else {
        fputs(value, fp);
    }
}

static int nn_gd_arc_ranker_benchmark_write_run(
    FILE *fp,
    const NNGDArcRankerBenchmarkRun *run) {
    fprintf(fp, "{\"label\":");
    nn_gd_json_write_string(fp, run->label);
    fprintf(fp, ",\"elapsed_seconds\":%.6f,\"epochs\":", run->elapsed_seconds);
    nn_gd_json_write_raw_or_null(fp, run->epochs);
    fputs(",\"summary_path\":", fp);
    nn_gd_json_write_string(fp, run->summary_path);
    fputs(",\"proposals_path\":", fp);
    nn_gd_json_write_string(fp, run->proposals_path);
    fputs(",\"summary_hash\":", fp);
    nn_gd_json_write_string(fp, run->summary_hash);
    fputs(",\"proposals_hash\":", fp);
    nn_gd_json_write_string(fp, run->proposals_hash);
    fputs(",\"source_attempt_count\":", fp);
    nn_gd_json_write_raw_or_null(fp, run->source_attempt_count);
    fputs(",\"target_attempt_count\":", fp);
    nn_gd_json_write_raw_or_null(fp, run->target_attempt_count);
    fputs(",\"target_task_count\":", fp);
    nn_gd_json_write_raw_or_null(fp, run->target_task_count);
    fputs(",\"accepted_target_task_count\":", fp);
    nn_gd_json_write_raw_or_null(fp, run->accepted_target_task_count);
    fputs(",\"top1_rate\":", fp);
    nn_gd_json_write_raw_or_null(fp, run->top1_rate);
    fputs(",\"top1_all_task_rate\":", fp);
    nn_gd_json_write_raw_or_null(fp, run->top1_all_task_rate);
    fputs(",\"retained_rate\":", fp);
    nn_gd_json_write_raw_or_null(fp, run->retained_rate);
    fputs(",\"vocab_size\":", fp);
    nn_gd_json_write_raw_or_null(fp, run->vocab_size);
    fputs(",\"proposal_count\":", fp);
    nn_gd_json_write_raw_or_null(fp, run->proposal_count);
    fputc('}', fp);
    return ferror(fp) ? -1 : 0;
}

static int nn_gd_arc_ranker_benchmark_write_report(
    const char *report_path,
    const char *attempt_rows_path,
    const char *output_dir,
    const char *cache_path,
    unsigned epochs,
    unsigned max_per_task,
    int cold_warm_summary_equal,
    int cold_warm_proposals_equal,
    const NNGDArcRankerBenchmarkRun *cold,
    const NNGDArcRankerBenchmarkRun *warm,
    const NNGDArcRankerBenchmarkRun *warm_default) {
    FILE *fp = fopen(report_path, "w");
    int status = -1;
    if (fp == NULL) {
        return -1;
    }
    fputs("{\"schema\":\"crlp.arc_ranker_benchmark.v1\","
          "\"report_version\":2,\"runtime\":\"in_process\","
          "\"attempts_tsv\":",
          fp);
    nn_gd_json_write_string(fp, attempt_rows_path);
    fprintf(fp,
            ",\"attempts_tsv_size_bytes\":%lld,\"output_dir\":",
            nn_gd_file_size_bytes(attempt_rows_path));
    nn_gd_json_write_string(fp, output_dir);
    fputs(",\"cache_path\":", fp);
    nn_gd_json_write_string(fp, cache_path);
    fprintf(fp,
            ",\"cache_size_bytes\":%lld,\"cache_size_mib\":%.3f,"
            "\"epochs\":%u,\"max_per_task\":%u,"
            "\"cold_warm_summary_equal\":%s,"
            "\"cold_warm_proposals_equal\":%s,\"runs\":[",
            nn_gd_file_size_bytes(cache_path),
            (double)nn_gd_file_size_bytes(cache_path) / (1024.0 * 1024.0),
            epochs,
            max_per_task,
            cold_warm_summary_equal ? "true" : "false",
            cold_warm_proposals_equal ? "true" : "false");
    if (nn_gd_arc_ranker_benchmark_write_run(fp, cold) != 0) {
        goto done;
    }
    fputc(',', fp);
    if (nn_gd_arc_ranker_benchmark_write_run(fp, warm) != 0) {
        goto done;
    }
    fputc(',', fp);
    if (nn_gd_arc_ranker_benchmark_write_run(fp, warm_default) != 0) {
        goto done;
    }
    fputs("]}\n", fp);
    status = ferror(fp) ? -1 : 0;

done:
    if (fclose(fp) != 0) {
        status = -1;
    }
    return status;
}

int nn_grammar_discovery_run_arc_ranker_benchmark(
    const char *attempt_rows_path,
    const char *output_dir,
    const char *report_path,
    unsigned epochs,
    double learning_rate,
    unsigned max_per_task) {
    char cache_path[4096];
    char cold_label[64];
    char warm_label[64];
    NNGDArcRankerBenchmarkRun cold = {0};
    NNGDArcRankerBenchmarkRun warm = {0};
    NNGDArcRankerBenchmarkRun warm_default = {0};
    int cold_warm_summary_equal;
    int cold_warm_proposals_equal;
    int status = -1;
    if (attempt_rows_path == NULL || output_dir == NULL || report_path == NULL ||
        epochs == 0U || max_per_task == 0U ||
        snprintf(cold_label, sizeof(cold_label), "cold_epochs%u", epochs) < 0 ||
        snprintf(warm_label, sizeof(warm_label), "warm_epochs%u", epochs) < 0 ||
        nn_gd_join_path(cache_path,
                        sizeof(cache_path),
                        output_dir,
                        "arc_ranker.cache.bin") != 0) {
        return -1;
    }
    if (mkdir(output_dir, 0777) != 0 && errno != EEXIST) {
        return -1;
    }
    if (remove(cache_path) != 0 && errno != ENOENT) {
        return -1;
    }
    if (nn_gd_arc_ranker_benchmark_run(attempt_rows_path,
                                       cache_path,
                                       output_dir,
                                       cold_label,
                                       epochs,
                                       learning_rate,
                                       max_per_task,
                                       &cold) != 0 ||
        nn_gd_arc_ranker_benchmark_run(attempt_rows_path,
                                       cache_path,
                                       output_dir,
                                       warm_label,
                                       epochs,
                                       learning_rate,
                                       max_per_task,
                                       &warm) != 0 ||
        nn_gd_arc_ranker_benchmark_run(attempt_rows_path,
                                       cache_path,
                                       output_dir,
                                       "warm_default",
                                       64U,
                                       learning_rate,
                                       max_per_task,
                                       &warm_default) != 0) {
        goto done;
    }
    cold_warm_summary_equal = nn_gd_files_equal(cold.summary_path, warm.summary_path);
    cold_warm_proposals_equal =
        nn_gd_files_equal(cold.proposals_path, warm.proposals_path);
    if (!cold_warm_summary_equal || !cold_warm_proposals_equal) {
        goto done;
    }
    if (nn_gd_arc_ranker_benchmark_write_report(report_path,
                                                attempt_rows_path,
                                                output_dir,
                                                cache_path,
                                                epochs,
                                                max_per_task,
                                                cold_warm_summary_equal,
                                                cold_warm_proposals_equal,
                                                &cold,
                                                &warm,
                                                &warm_default) != 0) {
        goto done;
    }
    printf("arc-ranker-benchmark cold=%.3fs warm=%.3fs warm_default=%.3fs "
           "cache=%.1fMiB report=%s\n",
           cold.elapsed_seconds,
           warm.elapsed_seconds,
           warm_default.elapsed_seconds,
           (double)nn_gd_file_size_bytes(cache_path) / (1024.0 * 1024.0),
           report_path);
    status = 0;

done:
    nn_gd_arc_ranker_benchmark_run_free(&cold);
    nn_gd_arc_ranker_benchmark_run_free(&warm);
    nn_gd_arc_ranker_benchmark_run_free(&warm_default);
    return status;
}

#include "nn_grammar_discovery_arc_failure.inc"

#include "nn_grammar_discovery_arc_synthetic_curriculum.inc"
#include "nn_grammar_discovery_arc_ranker.inc"

#include "nn_grammar_discovery_report.inc"
#include "nn_grammar_discovery_health.inc"

static int nn_gd_arc_neural_write_prior(const char *path) {
    return nn_gd_ngd_write_text(
        path,
        "schema_version\ttemplate_id\ttemplate_name\tlibrary_id\tslot_name\tslot_type\tslot_value\tslot_integer\tproposal_count\tcertified_count\tsupport_tasks\tcovered_events\tutility_mean\n"
        "crlp.arc_neural_discovery.template_probe.v1\t1028\tfill_enclosed\tneural:c_ngd_fill_enclosed\tprobe.template\ttemplate\tfill_enclosed\t1028\t20\t0\t1\t1\t1.0\n"
        "crlp.arc_neural_discovery.slot_probe.v1\t1028\tfill_enclosed\tneural:c_ngd_fill_enclosed:slot\tobject_scene.background\tcolor\t0.00000000\t0\t20\t0\t1\t1\t1.0\n"
        "crlp.arc_neural_discovery.slot_probe.v1\t1028\tfill_enclosed\tneural:c_ngd_fill_enclosed:slot\tfill.border_color\tcolor\t1.00000000\t1\t20\t0\t1\t1\t1.0\n"
        "crlp.arc_neural_discovery.slot_probe.v1\t1028\tfill_enclosed\tneural:c_ngd_fill_enclosed:slot\tfill.output_color\tcolor\t2.00000000\t2\t20\t0\t1\t1\t1.0\n");
}

#include "nn_grammar_discovery_arc_neural_discovery.inc"
