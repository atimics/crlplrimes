#include "nn_grammar_discovery.h"

#include <ctype.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NN_GD_LINE_MAX 65536
#define NN_GD_MAX_FIELDS 96
#define NN_GD_COUNTEREXAMPLE_LIMIT 8
#define NN_GD_ARC_RUNTIME_FIELD_COUNT 85

typedef struct {
    char **items;
    size_t count;
    size_t capacity;
} NNGDStringList;

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
    size_t contrast_accepted_index;
    NNGDStringList tokens;
    size_t *token_indices;
    size_t token_index_count;
} NNGDArcPipelineAttempt;

typedef struct {
    NNGDArcPipelineAttempt *items;
    size_t count;
    size_t capacity;
} NNGDArcPipelineAttemptList;

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

static const NNGDSketch *nn_gd_sort_sketches = NULL;
static const NNGDArcPipelineAttempt *nn_gd_sort_arc_attempts = NULL;

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

static char *nn_gd_read_line_dynamic(FILE *fp) {
    size_t capacity = 4096U;
    size_t used = 0U;
    char *line = (char *)malloc(capacity);
    if (line == NULL) {
        return NULL;
    }
    line[0] = '\0';
    for (;;) {
        if (fgets(line + used, (int)(capacity - used), fp) == NULL) {
            if (used == 0U) {
                free(line);
                return NULL;
            }
            return line;
        }
        used += strlen(line + used);
        if (used > 0U && line[used - 1U] == '\n') {
            return line;
        }
        if (capacity - used <= 1U) {
            size_t next_capacity = capacity * 2U;
            char *next = (char *)realloc(line, next_capacity);
            if (next == NULL) {
                free(line);
                return NULL;
            }
            line = next;
            capacity = next_capacity;
        }
    }
}

static const char *nn_gd_json_find_key_value(const char *line, const char *key) {
    char pattern[128];
    int written;
    const char *start;
    if (line == NULL || key == NULL || key[0] == '\0') {
        return NULL;
    }
    written = snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    if (written <= 0 || (size_t)written >= sizeof(pattern)) {
        return NULL;
    }
    start = strstr(line, pattern);
    return start == NULL ? NULL : start + (size_t)written;
}

static const char *nn_gd_json_skip_space(const char *cursor) {
    while (cursor != NULL && isspace((unsigned char)*cursor)) {
        ++cursor;
    }
    return cursor;
}

static int nn_gd_json_append_char(char **buffer,
                                  size_t *used,
                                  size_t *capacity,
                                  char value) {
    if (*used + 1U >= *capacity) {
        size_t next_capacity = *capacity * 2U;
        char *next = (char *)realloc(*buffer, next_capacity);
        if (next == NULL) {
            return -1;
        }
        *buffer = next;
        *capacity = next_capacity;
    }
    (*buffer)[(*used)++] = value;
    return 0;
}

static char *nn_gd_json_dup_string_value(const char *cursor) {
    size_t capacity = 64U;
    size_t used = 0U;
    char *value;
    cursor = nn_gd_json_skip_space(cursor);
    if (cursor == NULL || *cursor != '"') {
        return nn_gd_strdup("");
    }
    ++cursor;
    value = (char *)malloc(capacity);
    if (value == NULL) {
        return NULL;
    }
    while (*cursor != '\0' && *cursor != '"') {
        unsigned char ch = (unsigned char)*cursor++;
        if (ch == '\\') {
            ch = (unsigned char)*cursor++;
            switch (ch) {
                case '"':
                case '\\':
                case '/':
                    break;
                case 'b':
                    ch = '\b';
                    break;
                case 'f':
                    ch = '\f';
                    break;
                case 'n':
                    ch = '\n';
                    break;
                case 'r':
                    ch = '\r';
                    break;
                case 't':
                    ch = '\t';
                    break;
                case 'u': {
                    size_t i;
                    for (i = 0U; i < 4U; ++i) {
                        if (!isxdigit((unsigned char)cursor[i])) {
                            free(value);
                            return NULL;
                        }
                    }
                    cursor += 4U;
                    ch = '?';
                    break;
                }
                default:
                    free(value);
                    return NULL;
            }
        }
        if (nn_gd_json_append_char(&value, &used, &capacity, (char)ch) != 0) {
            free(value);
            return NULL;
        }
    }
    if (*cursor != '"') {
        free(value);
        return NULL;
    }
    value[used] = '\0';
    return value;
}

static char *nn_gd_json_dup_string_value_advance(const char **cursor_ref) {
    const char *cursor;
    char *value;
    cursor = nn_gd_json_skip_space(cursor_ref == NULL ? NULL : *cursor_ref);
    if (cursor == NULL || *cursor != '"') {
        if (cursor_ref != NULL) {
            *cursor_ref = cursor;
        }
        return nn_gd_strdup("");
    }
    value = nn_gd_json_dup_string_value(cursor);
    if (value == NULL) {
        return NULL;
    }
    ++cursor;
    while (*cursor != '\0' && *cursor != '"') {
        if (*cursor == '\\' && cursor[1] != '\0') {
            cursor += 2;
        } else {
            ++cursor;
        }
    }
    if (*cursor == '"') {
        ++cursor;
    }
    if (cursor_ref != NULL) {
        *cursor_ref = cursor;
    }
    return value;
}

static void nn_gd_sanitize_tsv_value(char *value) {
    char *p;
    if (value == NULL) {
        return;
    }
    for (p = value; *p != '\0'; ++p) {
        if (*p == '\t' || *p == '\r' || *p == '\n') {
            *p = ' ';
        }
    }
}

static char *nn_gd_json_dup_key_string(const char *line, const char *key) {
    return nn_gd_json_dup_string_value(nn_gd_json_find_key_value(line, key));
}

static int nn_gd_json_get_bool(const char *line, const char *key) {
    const char *cursor = nn_gd_json_skip_space(nn_gd_json_find_key_value(line, key));
    if (cursor == NULL) {
        return 0;
    }
    return strncmp(cursor, "true", 4U) == 0 || strncmp(cursor, "1", 1U) == 0;
}

static double nn_gd_json_get_double_or_zero(const char *line, const char *key) {
    const char *cursor = nn_gd_json_skip_space(nn_gd_json_find_key_value(line, key));
    char *end = NULL;
    double value;
    if (cursor == NULL) {
        return 0.0;
    }
    value = strtod(cursor, &end);
    if (end == cursor || !isfinite(value)) {
        return 0.0;
    }
    return value;
}

static char *nn_gd_json_dup_scalar_tsv_value_at(const char **cursor_ref,
                                                int bool_as_int,
                                                int null_as_python) {
    const char *cursor = nn_gd_json_skip_space(cursor_ref == NULL ? NULL : *cursor_ref);
    const char *end;
    char *value;
    size_t len;
    if (cursor == NULL) {
        return nn_gd_strdup("");
    }
    if (*cursor == '"') {
        value = nn_gd_json_dup_string_value_advance(&cursor);
        nn_gd_sanitize_tsv_value(value);
        if (cursor_ref != NULL) {
            *cursor_ref = cursor;
        }
        return value;
    }
    if (strncmp(cursor, "true", 4U) == 0) {
        if (cursor_ref != NULL) {
            *cursor_ref = cursor + 4U;
        }
        return nn_gd_strdup(bool_as_int ? "1" : "True");
    }
    if (strncmp(cursor, "false", 5U) == 0) {
        if (cursor_ref != NULL) {
            *cursor_ref = cursor + 5U;
        }
        return nn_gd_strdup(bool_as_int ? "0" : "False");
    }
    if (strncmp(cursor, "null", 4U) == 0) {
        if (cursor_ref != NULL) {
            *cursor_ref = cursor + 4U;
        }
        return nn_gd_strdup(null_as_python ? "None" : "");
    }
    end = cursor;
    while (*end != '\0' && *end != ',' && *end != '}' && *end != ']' &&
           !isspace((unsigned char)*end)) {
        ++end;
    }
    len = (size_t)(end - cursor);
    value = (char *)malloc(len + 1U);
    if (value == NULL) {
        return NULL;
    }
    memcpy(value, cursor, len);
    value[len] = '\0';
    if (cursor_ref != NULL) {
        *cursor_ref = end;
    }
    return value;
}

static const char *nn_gd_json_skip_string_literal(const char *cursor) {
    if (cursor == NULL || *cursor != '"') {
        return cursor;
    }
    ++cursor;
    while (*cursor != '\0') {
        if (*cursor == '\\' && cursor[1] != '\0') {
            cursor += 2;
            continue;
        }
        if (*cursor == '"') {
            return cursor + 1;
        }
        ++cursor;
    }
    return cursor;
}

static void nn_gd_json_index_line(const char *line, NNGDJsonIndex *index) {
    const char *cursor = line;
    if (index == NULL) {
        return;
    }
    index->count = 0U;
    while (cursor != NULL && *cursor != '\0') {
        const char *key_start;
        const char *key_end;
        const char *after_key;
        const char *value;
        while (*cursor != '\0' && *cursor != '"') {
            ++cursor;
        }
        if (*cursor == '\0') {
            break;
        }
        key_start = cursor + 1;
        after_key = nn_gd_json_skip_string_literal(cursor);
        if (after_key == NULL || after_key == cursor || after_key[-1] != '"') {
            break;
        }
        key_end = after_key - 1;
        value = nn_gd_json_skip_space(after_key);
        if (value != NULL && *value == ':' && index->count < 192U) {
            ++value;
            index->items[index->count].key = key_start;
            index->items[index->count].key_len = (size_t)(key_end - key_start);
            index->items[index->count].value = value;
            ++index->count;
        }
        cursor = after_key;
    }
}

static const char *nn_gd_json_index_find(const NNGDJsonIndex *index,
                                         const char *key) {
    size_t i;
    size_t key_len;
    if (index == NULL || key == NULL) {
        return NULL;
    }
    key_len = strlen(key);
    for (i = 0U; i < index->count; ++i) {
        if (index->items[i].key_len == key_len &&
            strncmp(index->items[i].key, key, key_len) == 0) {
            return index->items[i].value;
        }
    }
    return NULL;
}

static char *nn_gd_json_dup_index_tsv_string(const NNGDJsonIndex *index,
                                             const char *key) {
    char *value = nn_gd_json_dup_string_value(nn_gd_json_index_find(index, key));
    nn_gd_sanitize_tsv_value(value);
    return value;
}

static char *nn_gd_json_dup_index_scalar_tsv(const NNGDJsonIndex *index,
                                             const char *key,
                                             int bool_as_int) {
    const char *cursor = nn_gd_json_index_find(index, key);
    return nn_gd_json_dup_scalar_tsv_value_at(&cursor, bool_as_int, 0);
}

static int nn_gd_append_sanitized_value(char **buffer,
                                        size_t *used,
                                        size_t *capacity,
                                        const char *value) {
    const char *p;
    if (value == NULL) {
        return 0;
    }
    for (p = value; *p != '\0'; ++p) {
        char ch = *p;
        if (ch == '|') {
            ch = '/';
        } else if (ch == '\t' || ch == '\r' || ch == '\n') {
            ch = ' ';
        }
        if (nn_gd_json_append_char(buffer, used, capacity, ch) != 0) {
            return -1;
        }
    }
    return 0;
}

static char *nn_gd_make_slot_pair(const char *key, const char *value) {
    size_t capacity = 64U;
    size_t used = 0U;
    char *pair = (char *)malloc(capacity);
    if (pair == NULL) {
        return NULL;
    }
    if (nn_gd_append_sanitized_value(&pair, &used, &capacity, key) != 0 ||
        nn_gd_json_append_char(&pair, &used, &capacity, '=') != 0 ||
        nn_gd_append_sanitized_value(&pair, &used, &capacity, value) != 0) {
        free(pair);
        return NULL;
    }
    pair[used] = '\0';
    return pair;
}

static char *nn_gd_json_dup_slot_tokens_at(const char *cursor) {
    NNGDStringList pairs = {0};
    size_t capacity = 64U;
    size_t used = 0U;
    char *joined = NULL;
    size_t i;
    cursor = nn_gd_json_skip_space(cursor);
    if (cursor == NULL || *cursor != '{') {
        return nn_gd_strdup("");
    }
    ++cursor;
    for (;;) {
        char *key;
        char *value;
        char *pair;
        cursor = nn_gd_json_skip_space(cursor);
        if (cursor == NULL) {
            nn_gd_string_list_free(&pairs);
            return NULL;
        }
        if (*cursor == '}') {
            break;
        }
        key = nn_gd_json_dup_string_value_advance(&cursor);
        if (key == NULL) {
            nn_gd_string_list_free(&pairs);
            return NULL;
        }
        cursor = nn_gd_json_skip_space(cursor);
        if (cursor == NULL || *cursor != ':') {
            free(key);
            nn_gd_string_list_free(&pairs);
            return NULL;
        }
        ++cursor;
        value = nn_gd_json_dup_scalar_tsv_value_at(&cursor, 0, 1);
        if (value == NULL) {
            free(key);
            nn_gd_string_list_free(&pairs);
            return NULL;
        }
        pair = nn_gd_make_slot_pair(key, value);
        free(key);
        free(value);
        if (pair == NULL) {
            nn_gd_string_list_free(&pairs);
            return NULL;
        }
        if (nn_gd_string_list_add(&pairs, pair) != 0) {
            free(pair);
            nn_gd_string_list_free(&pairs);
            return NULL;
        }
        free(pair);
        cursor = nn_gd_json_skip_space(cursor);
        if (cursor == NULL) {
            nn_gd_string_list_free(&pairs);
            return NULL;
        }
        if (*cursor == ',') {
            ++cursor;
            continue;
        }
        if (*cursor == '}') {
            break;
        }
        nn_gd_string_list_free(&pairs);
        return NULL;
    }
    nn_gd_string_list_sort(&pairs);
    joined = (char *)malloc(capacity);
    if (joined == NULL) {
        nn_gd_string_list_free(&pairs);
        return NULL;
    }
    for (i = 0U; i < pairs.count; ++i) {
        if (i > 0U && nn_gd_json_append_char(&joined, &used, &capacity, '|') != 0) {
            free(joined);
            nn_gd_string_list_free(&pairs);
            return NULL;
        }
        if (nn_gd_append_sanitized_value(&joined,
                                         &used,
                                         &capacity,
                                         pairs.items[i]) != 0) {
            free(joined);
            nn_gd_string_list_free(&pairs);
            return NULL;
        }
    }
    joined[used] = '\0';
    nn_gd_string_list_free(&pairs);
    return joined;
}

static char *nn_gd_json_dup_index_slot_tokens(const NNGDJsonIndex *index) {
    return nn_gd_json_dup_slot_tokens_at(nn_gd_json_index_find(index, "slot_values"));
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

static int nn_gd_arc_attempt_list_reserve(NNGDArcPipelineAttemptList *list,
                                          size_t needed) {
    NNGDArcPipelineAttempt *items;
    size_t capacity;
    if (needed <= list->capacity) {
        return 0;
    }
    capacity = list->capacity == 0U ? 64U : list->capacity * 2U;
    while (capacity < needed) {
        capacity *= 2U;
    }
    items = (NNGDArcPipelineAttempt *)realloc(list->items,
                                              capacity * sizeof(NNGDArcPipelineAttempt));
    if (items == NULL) {
        return -1;
    }
    list->items = items;
    list->capacity = capacity;
    return 0;
}

static void nn_gd_arc_attempt_free(NNGDArcPipelineAttempt *attempt) {
    free(attempt->split_name);
    free(attempt->family);
    free(attempt->task_id);
    free(attempt->split);
    free(attempt->attempt_id);
    free(attempt->library_id);
    free(attempt->pipeline_kind);
    free(attempt->op_name);
    free(attempt->visual_program);
    free(attempt->slot_contract);
    free(attempt->composition_kind);
    free(attempt->normalization_frame);
    free(attempt->typed_stages);
    free(attempt->primitive_contract);
    free(attempt->slot_tokens);
    free(attempt->rejection_reason);
    free(attempt->failure_stage);
    free(attempt->failure_detail);
    free(attempt->contrast_accepted_attempt_id);
    free(attempt->contrast_pair_kind);
    free(attempt->token_indices);
    nn_gd_string_list_free(&attempt->tokens);
}

static void nn_gd_arc_attempt_list_free(NNGDArcPipelineAttemptList *list) {
    size_t i;
    for (i = 0; i < list->count; ++i) {
        nn_gd_arc_attempt_free(&list->items[i]);
    }
    free(list->items);
    list->items = NULL;
    list->count = 0U;
    list->capacity = 0U;
}

static int nn_gd_add_pipe_tokens_with_prefix(NNGDStringList *tokens,
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

static int nn_gd_add_named_value_tokens(NNGDStringList *tokens,
                                        const char *name,
                                        const char *value) {
    char token[256];
    char bucket[256];
    long parsed;
    if (name == NULL || name[0] == '\0' || value == NULL || value[0] == '\0') {
        return 0;
    }
    snprintf(token, sizeof(token), "%s=%s", name, value);
    if (nn_gd_add_token_pair(tokens, "field:", token) != 0) {
        return -1;
    }
    if (!isdigit((unsigned char)value[0]) && value[0] != '-') {
        return 0;
    }
    parsed = nn_gd_parse_long(value);
    if (parsed == 0L) {
        snprintf(bucket, sizeof(bucket), "%s:zero", name);
    } else if (parsed == 1L) {
        snprintf(bucket, sizeof(bucket), "%s:one", name);
    } else if (parsed == 2L) {
        snprintf(bucket, sizeof(bucket), "%s:two", name);
    } else if (parsed > 0L && parsed <= 4L) {
        snprintf(bucket, sizeof(bucket), "%s:small", name);
    } else if (parsed > 0L && parsed <= 8L) {
        snprintf(bucket, sizeof(bucket), "%s:medium", name);
    } else if (parsed > 8L) {
        snprintf(bucket, sizeof(bucket), "%s:large", name);
    } else {
        snprintf(bucket, sizeof(bucket), "%s:negative", name);
    }
    return nn_gd_add_token_pair(tokens, "field_bucket:", bucket);
}

static int nn_gd_arc_attempt_add_tokens(NNGDArcPipelineAttempt *attempt,
                                        char **fields,
                                        size_t field_count) {
    static const char *numeric_names[] = {
        "selector",
        "background",
        "color",
        "output_color",
        "selector_kind",
        "include_background",
        "orientation",
        "count_mode",
        "x_align",
        "y_align",
        "x_anchor",
        "y_anchor",
        "x_offset",
        "y_offset",
        "width",
        "height",
        "period",
        "proposal_count",
        "support_tasks",
        "input_width",
        "input_height",
        "output_width",
        "output_height",
        "shape_scale_x",
        "shape_scale_y",
        "period_divides_input_width",
        "period_divides_input_height",
        "slot_width_matches_output",
        "slot_height_matches_output",
        "color_in_input",
        "color_in_output",
        "output_color_in_output",
        "background_in_input",
        "verifier_train_pairs",
        "verifier_checked_cells",
        "verifier_mismatches",
        "verifier_unsupported_pairs",
        "verifier_apply_failed_pairs",
        "verifier_shape_mismatch_pairs",
        "verifier_missing_output_pairs",
        "verifier_first_failing_pair",
        "input_component_count",
        "selected_component_count",
        "selected_component_area_min",
        "selected_component_area_max",
        "selected_component_area_total",
        "selected_component_same_shape",
        "selected_component_shape_count",
        "selected_component_bbox_width",
        "selected_component_bbox_height",
        "selected_bbox_matches_output_shape",
        "selected_bbox_rotated_output_shape",
        "output_foreground_cell_count",
        "output_component_count",
        "output_non_background_color_count",
        "selected_count_matches_output_components",
        "selected_count_matches_output_foreground",
        "selected_area_matches_output_foreground",
    };
    static const size_t numeric_indices[] = {
        23U, 24U, 25U, 26U, 27U, 28U, 29U, 30U, 31U, 32U,
        33U, 34U, 35U, 36U, 37U, 38U, 39U, 20U, 21U,
        44U, 45U, 46U, 47U, 48U, 49U, 50U, 51U, 52U, 53U,
        54U, 55U, 56U, 57U,
        58U, 59U, 60U, 61U, 62U, 63U, 64U, 65U,
        68U, 69U, 70U, 71U, 72U, 73U, 74U, 75U, 76U, 77U,
        78U, 79U, 80U, 81U, 82U, 83U, 84U,
    };
    size_t i;
    if (nn_gd_add_token_pair(&attempt->tokens, "family:", attempt->family) != 0 ||
        nn_gd_add_token_pair(&attempt->tokens, "split:", attempt->split) != 0 ||
        nn_gd_add_token_pair(&attempt->tokens, "pipeline:", attempt->pipeline_kind) != 0 ||
        nn_gd_add_token_pair(&attempt->tokens, "op:", attempt->op_name) != 0 ||
        nn_gd_add_token_pair(&attempt->tokens,
                             "composition:",
                             attempt->composition_kind) != 0 ||
        nn_gd_add_token_pair(&attempt->tokens,
                             "normalizer:",
                             attempt->normalization_frame) != 0 ||
        nn_gd_add_token_pair(&attempt->tokens,
                             "visual_program:",
                             attempt->visual_program) != 0 ||
        nn_gd_add_pipe_tokens_with_prefix(&attempt->tokens,
                                          "stage:",
                                          attempt->typed_stages) != 0 ||
        nn_gd_add_pipe_tokens_with_prefix(&attempt->tokens,
                                          "contract:",
                                          attempt->slot_contract) != 0 ||
        nn_gd_add_token_pair(&attempt->tokens,
                             "primitive_contract:",
                             attempt->primitive_contract) != 0 ||
        nn_gd_add_token_pair(&attempt->tokens,
                             "failure_stage:",
                             attempt->failure_stage) != 0 ||
        nn_gd_add_token_pair(&attempt->tokens,
                             "failure_detail:",
                             attempt->failure_detail) != 0 ||
        nn_gd_add_slot_tokens(&attempt->tokens, attempt->slot_tokens) != 0) {
        return -1;
    }
    for (i = 0U; i < sizeof(numeric_names) / sizeof(numeric_names[0]); ++i) {
        if (nn_gd_add_named_value_tokens(&attempt->tokens,
                                         numeric_names[i],
                                         nn_gd_field(fields,
                                                     field_count,
                                                     numeric_indices[i])) != 0) {
            return -1;
        }
    }
    return 0;
}

static int nn_gd_arc_attempt_list_add(NNGDArcPipelineAttemptList *list,
                                      char **fields,
                                      size_t field_count) {
    NNGDArcPipelineAttempt *attempt;
    if (nn_gd_arc_attempt_list_reserve(list, list->count + 1U) != 0) {
        return -1;
    }
    attempt = &list->items[list->count];
    memset(attempt, 0, sizeof(*attempt));
    attempt->split_name = nn_gd_strdup(nn_gd_field(fields, field_count, 0U));
    attempt->family = nn_gd_strdup(nn_gd_field(fields, field_count, 1U));
    attempt->task_id = nn_gd_strdup(nn_gd_field(fields, field_count, 2U));
    attempt->split = nn_gd_strdup(nn_gd_field(fields, field_count, 3U));
    attempt->attempt_id = nn_gd_strdup(nn_gd_field(fields, field_count, 4U));
    attempt->library_id = nn_gd_strdup(nn_gd_field(fields, field_count, 5U));
    attempt->attempt_index = nn_gd_parse_long(nn_gd_field(fields, field_count, 6U));
    attempt->pipeline_kind = nn_gd_strdup(nn_gd_field(fields, field_count, 7U));
    attempt->op_name = nn_gd_strdup(nn_gd_field(fields, field_count, 8U));
    attempt->visual_program = nn_gd_strdup(nn_gd_field(fields, field_count, 9U));
    attempt->slot_contract = nn_gd_strdup(nn_gd_field(fields, field_count, 10U));
    attempt->composition_kind = nn_gd_strdup(nn_gd_field(fields, field_count, 11U));
    attempt->normalization_frame = nn_gd_strdup(nn_gd_field(fields, field_count, 12U));
    attempt->typed_stages = nn_gd_strdup(nn_gd_field(fields, field_count, 13U));
    attempt->primitive_contract = nn_gd_strdup(nn_gd_field(fields, field_count, 14U));
    attempt->slot_tokens = nn_gd_strdup(nn_gd_field(fields, field_count, 15U));
    attempt->accepted = nn_gd_parse_bool(nn_gd_field(fields, field_count, 16U));
    attempt->rejection_reason = nn_gd_strdup(nn_gd_field(fields, field_count, 17U));
    attempt->failure_stage = nn_gd_strdup(nn_gd_field(fields, field_count, 18U));
    attempt->failure_detail = nn_gd_strdup(nn_gd_field(fields, field_count, 19U));
    attempt->proposal_count = nn_gd_parse_long(nn_gd_field(fields, field_count, 20U));
    attempt->support_tasks = nn_gd_parse_long(nn_gd_field(fields, field_count, 21U));
    attempt->utility_mean = nn_gd_parse_double(nn_gd_field(fields, field_count, 22U));
    attempt->contrast_accepted_attempt_id =
        nn_gd_strdup(nn_gd_field(fields, field_count, 40U));
    attempt->contrast_slot_distance =
        nn_gd_parse_long(nn_gd_field(fields, field_count, 41U));
    attempt->contrast_pair_kind = nn_gd_strdup(nn_gd_field(fields, field_count, 42U));
    attempt->contrast_rejected_sibling_count =
        nn_gd_parse_long(nn_gd_field(fields, field_count, 43U));
    attempt->verifier_train_pairs =
        nn_gd_parse_long(nn_gd_field(fields, field_count, 58U));
    attempt->verifier_checked_cells =
        nn_gd_parse_long(nn_gd_field(fields, field_count, 59U));
    attempt->verifier_mismatches =
        nn_gd_parse_long(nn_gd_field(fields, field_count, 60U));
    attempt->verifier_unsupported_pairs =
        nn_gd_parse_long(nn_gd_field(fields, field_count, 61U));
    attempt->verifier_apply_failed_pairs =
        nn_gd_parse_long(nn_gd_field(fields, field_count, 62U));
    attempt->verifier_shape_mismatch_pairs =
        nn_gd_parse_long(nn_gd_field(fields, field_count, 63U));
    attempt->verifier_missing_output_pairs =
        nn_gd_parse_long(nn_gd_field(fields, field_count, 64U));
    attempt->verifier_first_failing_pair =
        nn_gd_parse_long(nn_gd_field(fields, field_count, 65U));
    attempt->verifier_mismatch_ratio =
        nn_gd_parse_double(nn_gd_field(fields, field_count, 66U));
    attempt->verifier_score =
        nn_gd_parse_double(nn_gd_field(fields, field_count, 67U));
    attempt->contrast_accepted_index = SIZE_MAX;
    if (attempt->split_name == NULL || attempt->family == NULL ||
        attempt->task_id == NULL || attempt->split == NULL ||
        attempt->attempt_id == NULL || attempt->library_id == NULL ||
        attempt->pipeline_kind == NULL || attempt->op_name == NULL ||
        attempt->visual_program == NULL || attempt->slot_contract == NULL ||
        attempt->composition_kind == NULL || attempt->normalization_frame == NULL ||
        attempt->typed_stages == NULL || attempt->primitive_contract == NULL ||
        attempt->slot_tokens == NULL || attempt->rejection_reason == NULL ||
        attempt->failure_stage == NULL || attempt->failure_detail == NULL ||
        attempt->contrast_accepted_attempt_id == NULL ||
        attempt->contrast_pair_kind == NULL ||
        nn_gd_arc_attempt_add_tokens(attempt, fields, field_count) != 0) {
        nn_gd_arc_attempt_free(attempt);
        memset(attempt, 0, sizeof(*attempt));
        return -1;
    }
    ++list->count;
    return 0;
}

static int nn_gd_read_arc_pipeline_attempts(const char *path,
                                            NNGDArcPipelineAttemptList *attempts) {
    FILE *fp;
    char line[NN_GD_LINE_MAX];
    char *fields[NN_GD_MAX_FIELDS];
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    while (fgets(line, sizeof(line), fp) != NULL) {
        size_t field_count = nn_gd_split_tabs(line, fields, NN_GD_MAX_FIELDS);
        const char *split_name = nn_gd_field(fields, field_count, 0U);
        const char *attempt_id = nn_gd_field(fields, field_count, 4U);
        if (split_name[0] == '\0' || strcmp(split_name, "split_name") == 0 ||
            attempt_id[0] == '\0') {
            continue;
        }
        if (nn_gd_arc_attempt_list_add(attempts, fields, field_count) != 0) {
            fclose(fp);
            return -1;
        }
    }
    fclose(fp);
    return 0;
}

static void nn_gd_free_arc_pipeline_field_buffer(char **fields, size_t field_count) {
    size_t i;
    for (i = 0U; i < field_count; ++i) {
        free(fields[i]);
        fields[i] = NULL;
    }
}

static int nn_gd_arc_pipeline_attempt_fields_from_json(const char *line,
                                                       const char *split_name,
                                                       char **fields,
                                                       size_t field_count) {
    NNGDJsonIndex index;
    char *schema;
    size_t i;
    if (field_count < NN_GD_ARC_RUNTIME_FIELD_COUNT) {
        return -1;
    }
    for (i = 0U; i < field_count; ++i) {
        fields[i] = NULL;
    }
    nn_gd_json_index_line(line, &index);
    schema = nn_gd_json_dup_index_tsv_string(&index, "schema");
    if (schema == NULL) {
        return -1;
    }
    if (strcmp(schema,
               "crlp.neural_grammar_discovery.arc_visual_pipeline_attempt.v1") != 0) {
        free(schema);
        return 1;
    }
    free(schema);
    fields[0] = nn_gd_strdup(split_name);
    fields[1] = nn_gd_json_dup_index_tsv_string(&index, "family");
    fields[2] = nn_gd_json_dup_index_tsv_string(&index, "task_id");
    fields[3] = nn_gd_json_dup_index_tsv_string(&index, "split");
    fields[4] = nn_gd_json_dup_index_tsv_string(&index, "attempt_id");
    fields[5] = nn_gd_json_dup_index_tsv_string(&index, "library_id");
    fields[6] = nn_gd_json_dup_index_scalar_tsv(&index, "attempt_index", 1);
    fields[7] = nn_gd_json_dup_index_tsv_string(&index, "pipeline_kind");
    fields[8] = nn_gd_json_dup_index_tsv_string(&index, "op_name");
    fields[9] = nn_gd_json_dup_index_tsv_string(&index, "visual_program");
    fields[10] = nn_gd_json_dup_index_tsv_string(&index, "slot_contract");
    fields[11] = nn_gd_json_dup_index_tsv_string(&index, "composition_kind");
    fields[12] = nn_gd_json_dup_index_tsv_string(&index, "normalization_frame");
    fields[13] = nn_gd_json_dup_index_tsv_string(&index, "typed_stages");
    fields[14] = nn_gd_json_dup_index_tsv_string(&index, "primitive_contract");
    fields[15] = nn_gd_json_dup_index_slot_tokens(&index);
    fields[16] = nn_gd_json_dup_index_scalar_tsv(&index, "accepted", 1);
    fields[17] = nn_gd_json_dup_index_tsv_string(&index, "rejection_reason");
    fields[18] = nn_gd_json_dup_index_tsv_string(&index, "failure_stage");
    fields[19] = nn_gd_json_dup_index_tsv_string(&index, "failure_detail");
    fields[20] = nn_gd_json_dup_index_scalar_tsv(&index, "proposal_count", 1);
    fields[21] = nn_gd_json_dup_index_scalar_tsv(&index, "support_tasks", 1);
    fields[22] = nn_gd_json_dup_index_scalar_tsv(&index, "utility_mean", 1);
    fields[23] = nn_gd_json_dup_index_scalar_tsv(&index, "selector", 1);
    fields[24] = nn_gd_json_dup_index_scalar_tsv(&index, "background", 1);
    fields[25] = nn_gd_json_dup_index_scalar_tsv(&index, "color", 1);
    fields[26] = nn_gd_json_dup_index_scalar_tsv(&index, "output_color", 1);
    fields[27] = nn_gd_json_dup_index_scalar_tsv(&index, "selector_kind", 1);
    fields[28] = nn_gd_json_dup_index_scalar_tsv(&index, "include_background", 1);
    fields[29] = nn_gd_json_dup_index_scalar_tsv(&index, "orientation", 1);
    fields[30] = nn_gd_json_dup_index_scalar_tsv(&index, "count_mode", 1);
    fields[31] = nn_gd_json_dup_index_scalar_tsv(&index, "x_align", 1);
    fields[32] = nn_gd_json_dup_index_scalar_tsv(&index, "y_align", 1);
    fields[33] = nn_gd_json_dup_index_scalar_tsv(&index, "x_anchor", 1);
    fields[34] = nn_gd_json_dup_index_scalar_tsv(&index, "y_anchor", 1);
    fields[35] = nn_gd_json_dup_index_scalar_tsv(&index, "x_offset", 1);
    fields[36] = nn_gd_json_dup_index_scalar_tsv(&index, "y_offset", 1);
    fields[37] = nn_gd_json_dup_index_scalar_tsv(&index, "width", 1);
    fields[38] = nn_gd_json_dup_index_scalar_tsv(&index, "height", 1);
    fields[39] = nn_gd_json_dup_index_scalar_tsv(&index, "period", 1);
    fields[40] = nn_gd_json_dup_index_tsv_string(&index, "contrast_accepted_attempt_id");
    fields[41] = nn_gd_json_dup_index_scalar_tsv(&index, "contrast_slot_distance", 1);
    fields[42] = nn_gd_json_dup_index_tsv_string(&index, "contrast_pair_kind");
    fields[43] =
        nn_gd_json_dup_index_scalar_tsv(&index, "contrast_rejected_sibling_count", 1);
    fields[44] = nn_gd_json_dup_index_scalar_tsv(&index, "input_width", 1);
    fields[45] = nn_gd_json_dup_index_scalar_tsv(&index, "input_height", 1);
    fields[46] = nn_gd_json_dup_index_scalar_tsv(&index, "output_width", 1);
    fields[47] = nn_gd_json_dup_index_scalar_tsv(&index, "output_height", 1);
    fields[48] = nn_gd_json_dup_index_scalar_tsv(&index, "shape_scale_x", 1);
    fields[49] = nn_gd_json_dup_index_scalar_tsv(&index, "shape_scale_y", 1);
    fields[50] =
        nn_gd_json_dup_index_scalar_tsv(&index, "period_divides_input_width", 1);
    fields[51] =
        nn_gd_json_dup_index_scalar_tsv(&index, "period_divides_input_height", 1);
    fields[52] = nn_gd_json_dup_index_scalar_tsv(&index, "slot_width_matches_output", 1);
    fields[53] = nn_gd_json_dup_index_scalar_tsv(&index, "slot_height_matches_output", 1);
    fields[54] = nn_gd_json_dup_index_scalar_tsv(&index, "color_in_input", 1);
    fields[55] = nn_gd_json_dup_index_scalar_tsv(&index, "color_in_output", 1);
    fields[56] = nn_gd_json_dup_index_scalar_tsv(&index, "output_color_in_output", 1);
    fields[57] = nn_gd_json_dup_index_scalar_tsv(&index, "background_in_input", 1);
    fields[58] = nn_gd_json_dup_index_scalar_tsv(&index, "train_pairs", 1);
    fields[59] = nn_gd_json_dup_index_scalar_tsv(&index, "checked_cells", 1);
    fields[60] = nn_gd_json_dup_index_scalar_tsv(&index, "mismatches", 1);
    fields[61] = nn_gd_json_dup_index_scalar_tsv(&index, "unsupported_pairs", 1);
    fields[62] = nn_gd_json_dup_index_scalar_tsv(&index, "apply_failed_pairs", 1);
    fields[63] = nn_gd_json_dup_index_scalar_tsv(&index, "shape_mismatch_pairs", 1);
    fields[64] = nn_gd_json_dup_index_scalar_tsv(&index, "missing_output_pairs", 1);
    fields[65] = nn_gd_json_dup_index_scalar_tsv(&index, "first_failing_pair", 1);
    fields[66] = nn_gd_json_dup_index_scalar_tsv(&index, "mismatch_ratio", 1);
    fields[67] = nn_gd_json_dup_index_scalar_tsv(&index, "verifier_score", 1);
    fields[68] = nn_gd_json_dup_index_scalar_tsv(&index, "input_component_count", 1);
    fields[69] = nn_gd_json_dup_index_scalar_tsv(&index, "selected_component_count", 1);
    fields[70] = nn_gd_json_dup_index_scalar_tsv(&index, "selected_component_area_min", 1);
    fields[71] = nn_gd_json_dup_index_scalar_tsv(&index, "selected_component_area_max", 1);
    fields[72] =
        nn_gd_json_dup_index_scalar_tsv(&index, "selected_component_area_total", 1);
    fields[73] =
        nn_gd_json_dup_index_scalar_tsv(&index, "selected_component_same_shape", 1);
    fields[74] =
        nn_gd_json_dup_index_scalar_tsv(&index, "selected_component_shape_count", 1);
    fields[75] =
        nn_gd_json_dup_index_scalar_tsv(&index, "selected_component_bbox_width", 1);
    fields[76] =
        nn_gd_json_dup_index_scalar_tsv(&index, "selected_component_bbox_height", 1);
    fields[77] =
        nn_gd_json_dup_index_scalar_tsv(&index, "selected_bbox_matches_output_shape", 1);
    fields[78] =
        nn_gd_json_dup_index_scalar_tsv(&index, "selected_bbox_rotated_output_shape", 1);
    fields[79] =
        nn_gd_json_dup_index_scalar_tsv(&index, "output_foreground_cell_count", 1);
    fields[80] = nn_gd_json_dup_index_scalar_tsv(&index, "output_component_count", 1);
    fields[81] =
        nn_gd_json_dup_index_scalar_tsv(&index, "output_non_background_color_count", 1);
    fields[82] = nn_gd_json_dup_index_scalar_tsv(
        &index,
        "selected_count_matches_output_components",
        1);
    fields[83] = nn_gd_json_dup_index_scalar_tsv(
        &index,
        "selected_count_matches_output_foreground",
        1);
    fields[84] =
        nn_gd_json_dup_index_scalar_tsv(&index, "selected_area_matches_output_foreground", 1);
    for (i = 0U; i < NN_GD_ARC_RUNTIME_FIELD_COUNT; ++i) {
        if (fields[i] == NULL) {
            nn_gd_free_arc_pipeline_field_buffer(fields, NN_GD_ARC_RUNTIME_FIELD_COUNT);
            return -1;
        }
    }
    return 0;
}

static int nn_gd_read_arc_pipeline_attempts_jsonl(const char *path,
                                                  const char *split_name,
                                                  NNGDArcPipelineAttemptList *attempts) {
    FILE *fp;
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    for (;;) {
        char *line = nn_gd_read_line_dynamic(fp);
        char *fields[NN_GD_ARC_RUNTIME_FIELD_COUNT];
        int parse_status;
        if (line == NULL) {
            break;
        }
        parse_status = nn_gd_arc_pipeline_attempt_fields_from_json(
            line,
            split_name,
            fields,
            NN_GD_ARC_RUNTIME_FIELD_COUNT);
        free(line);
        if (parse_status < 0) {
            fclose(fp);
            return -1;
        }
        if (parse_status == 0) {
            if (nn_gd_arc_attempt_list_add(attempts,
                                           fields,
                                           NN_GD_ARC_RUNTIME_FIELD_COUNT) != 0) {
                nn_gd_free_arc_pipeline_field_buffer(fields,
                                                     NN_GD_ARC_RUNTIME_FIELD_COUNT);
                fclose(fp);
                return -1;
            }
            nn_gd_free_arc_pipeline_field_buffer(fields, NN_GD_ARC_RUNTIME_FIELD_COUNT);
        }
    }
    fclose(fp);
    return 0;
}

static void nn_gd_resolve_arc_contrast_pairs(NNGDArcPipelineAttemptList *attempts,
                                             size_t *source_pair_count,
                                             size_t *target_pair_count) {
    size_t *accepted_indices;
    size_t accepted_count = 0U;
    size_t i;
    size_t j;
    if (source_pair_count != NULL) {
        *source_pair_count = 0U;
    }
    if (target_pair_count != NULL) {
        *target_pair_count = 0U;
    }
    accepted_indices = (size_t *)calloc(attempts->count == 0U ? 1U : attempts->count,
                                        sizeof(size_t));
    if (accepted_indices == NULL) {
        return;
    }
    for (i = 0U; i < attempts->count; ++i) {
        if (attempts->items[i].accepted) {
            accepted_indices[accepted_count++] = i;
        }
    }
    for (i = 0U; i < attempts->count; ++i) {
        NNGDArcPipelineAttempt *attempt = &attempts->items[i];
        attempt->contrast_accepted_index = SIZE_MAX;
        if (attempt->accepted || attempt->contrast_accepted_attempt_id == NULL ||
            attempt->contrast_accepted_attempt_id[0] == '\0') {
            continue;
        }
        for (j = 0U; j < accepted_count; ++j) {
            const NNGDArcPipelineAttempt *candidate = &attempts->items[accepted_indices[j]];
            if (strcmp(candidate->attempt_id, attempt->contrast_accepted_attempt_id) == 0) {
                attempt->contrast_accepted_index = accepted_indices[j];
                if (strcmp(attempt->split_name, "source") == 0 && source_pair_count != NULL) {
                    ++*source_pair_count;
                } else if (strcmp(attempt->split_name, "target") == 0 &&
                           target_pair_count != NULL) {
                    ++*target_pair_count;
                }
                break;
            }
        }
    }
    free(accepted_indices);
}

static int nn_gd_build_arc_vocab(const NNGDArcPipelineAttemptList *attempts,
                                 NNGDStringList *vocab) {
    size_t i;
    size_t j;
    for (i = 0U; i < attempts->count; ++i) {
        const NNGDArcPipelineAttempt *attempt = &attempts->items[i];
        for (j = 0U; j < attempt->tokens.count; ++j) {
            if (nn_gd_string_list_add_unique(vocab, attempt->tokens.items[j]) != 0) {
                return -1;
            }
        }
    }
    nn_gd_string_list_sort(vocab);
    return 0;
}

static int nn_gd_index_arc_attempt_tokens(NNGDArcPipelineAttemptList *attempts,
                                          const NNGDStringList *vocab) {
    size_t i;
    if (attempts == NULL || vocab == NULL) {
        return -1;
    }
    for (i = 0U; i < attempts->count; ++i) {
        NNGDArcPipelineAttempt *attempt = &attempts->items[i];
        size_t j;
        free(attempt->token_indices);
        attempt->token_indices = NULL;
        attempt->token_index_count = 0U;
        if (attempt->tokens.count == 0U) {
            continue;
        }
        attempt->token_indices =
            (size_t *)calloc(attempt->tokens.count, sizeof(size_t));
        if (attempt->token_indices == NULL) {
            return -1;
        }
        for (j = 0U; j < attempt->tokens.count; ++j) {
            long index = nn_gd_token_index(vocab, attempt->tokens.items[j]);
            if (index >= 0L) {
                attempt->token_indices[attempt->token_index_count++] = (size_t)index;
            }
        }
    }
    return 0;
}

static double nn_gd_arc_attempt_logit(const NNGDArcPipelineAttempt *attempt,
                                      const NNGDStringList *vocab,
                                      const double *weights,
                                      double bias) {
    size_t i;
    double z = bias;
    (void)vocab;
    for (i = 0U; i < attempt->token_index_count; ++i) {
        z += weights[attempt->token_indices[i]];
    }
    return z;
}

static double nn_gd_clamp01(double value) {
    if (value < 0.0) {
        return 0.0;
    }
    if (value > 1.0) {
        return 1.0;
    }
    return value;
}

static int nn_gd_str_contains(const char *haystack, const char *needle) {
    return haystack != NULL && needle != NULL && needle[0] != '\0' &&
           strstr(haystack, needle) != NULL;
}

static int nn_gd_arc_attempt_is_sequence_block_renderer(
    const NNGDArcPipelineAttempt *attempt) {
    if (attempt == NULL) {
        return 0;
    }
    return nn_gd_str_contains(attempt->visual_program, "repeat") ||
           nn_gd_str_contains(attempt->visual_program, "count") ||
           nn_gd_str_contains(attempt->visual_program, "template") ||
           nn_gd_str_contains(attempt->typed_stages, "repeat") ||
           nn_gd_str_contains(attempt->typed_stages, "count") ||
           nn_gd_str_contains(attempt->typed_stages, "template") ||
           strcmp(attempt->op_name, "macrocell_substitute") == 0 ||
           strcmp(attempt->op_name, "marker_count_motif_tile") == 0 ||
           strcmp(attempt->op_name, "component_count_line") == 0 ||
           strcmp(attempt->op_name, "visual_combinator") == 0;
}

static int nn_gd_arc_attempt_is_component_selector_layout(
    const NNGDArcPipelineAttempt *attempt) {
    if (attempt == NULL) {
        return 0;
    }
    return nn_gd_str_contains(attempt->primitive_contract, "selector:components") ||
           nn_gd_str_contains(attempt->primitive_contract, "selector:object") ||
           nn_gd_str_contains(attempt->normalization_frame, "object") ||
           nn_gd_str_contains(attempt->normalization_frame, "component") ||
           nn_gd_str_contains(attempt->typed_stages, "select_components") ||
           strcmp(attempt->op_name, "visual_combinator") == 0 ||
           strcmp(attempt->op_name, "object_mask_extract") == 0 ||
           strcmp(attempt->op_name, "object_color_tile") == 0 ||
           strcmp(attempt->op_name, "component_color_summary") == 0;
}

static int nn_gd_arc_aux_head_applies(const NNGDArcPipelineAttempt *attempt,
                                      NNGDArcAuxHeadKind kind) {
    if (attempt == NULL) {
        return 0;
    }
    if (kind == NN_GD_ARC_HEAD_SEQUENCE_BLOCK_RENDERER) {
        return nn_gd_arc_attempt_is_sequence_block_renderer(attempt);
    }
    if (kind == NN_GD_ARC_HEAD_COMPONENT_SELECTOR_LAYOUT) {
        return nn_gd_arc_attempt_is_component_selector_layout(attempt);
    }
    return 1;
}

static double nn_gd_arc_renderer_label(const NNGDArcPipelineAttempt *attempt) {
    double score;
    if (attempt == NULL) {
        return 0.0;
    }
    if (attempt->accepted) {
        return 1.0;
    }
    score = nn_gd_clamp01(attempt->verifier_score);
    if (strcmp(attempt->failure_stage, "renderer_cells") == 0) {
        return score;
    }
    if (strcmp(attempt->failure_stage, "transform_or_color") == 0) {
        return 0.65 * score;
    }
    if (strcmp(attempt->failure_stage, "renderer_shape") == 0) {
        return 0.35 * score;
    }
    return 0.0;
}

static double nn_gd_arc_component_label(const NNGDArcPipelineAttempt *attempt) {
    double score;
    if (attempt == NULL) {
        return 0.0;
    }
    if (attempt->accepted) {
        return 1.0;
    }
    score = nn_gd_clamp01(attempt->verifier_score);
    if (strcmp(attempt->failure_stage, "selector_or_transform") == 0 ||
        strcmp(attempt->failure_stage, "unsupported_shape") == 0 ||
        attempt->verifier_apply_failed_pairs > 0L) {
        return 0.0;
    }
    if (strcmp(attempt->failure_stage, "renderer_cells") == 0) {
        return nn_gd_clamp01(0.45 + 0.55 * score);
    }
    if (strcmp(attempt->failure_stage, "transform_or_color") == 0) {
        return 0.45 * score;
    }
    if (strcmp(attempt->failure_stage, "renderer_shape") == 0) {
        return 0.25 * score;
    }
    return 0.25 * score;
}

static double nn_gd_arc_component_selector_layout_label(
    const NNGDArcPipelineAttempt *attempt) {
    double score;
    if (attempt == NULL) {
        return 0.0;
    }
    if (attempt->accepted) {
        return 1.0;
    }
    score = nn_gd_clamp01(attempt->verifier_score);
    if (strcmp(attempt->failure_stage, "selector_or_transform") == 0 ||
        attempt->verifier_apply_failed_pairs > 0L ||
        attempt->verifier_checked_cells <= 0L) {
        return 0.0;
    }
    if (strcmp(attempt->failure_stage, "renderer_shape") == 0) {
        return 0.20 * score;
    }
    if (strcmp(attempt->failure_stage, "transform_or_color") == 0) {
        return 0.55 * score;
    }
    if (strcmp(attempt->failure_stage, "renderer_cells") == 0) {
        return nn_gd_clamp01(0.50 + 0.50 * score);
    }
    return 0.35 * score;
}

static double nn_gd_arc_aux_label(const NNGDArcPipelineAttempt *attempt,
                                  NNGDArcAuxHeadKind kind) {
    switch (kind) {
        case NN_GD_ARC_HEAD_RENDERER:
            return nn_gd_arc_renderer_label(attempt);
        case NN_GD_ARC_HEAD_SEQUENCE_BLOCK_RENDERER:
            return nn_gd_arc_renderer_label(attempt);
        case NN_GD_ARC_HEAD_COMPONENT_SHAPE_COLOR:
            return nn_gd_arc_component_label(attempt);
        case NN_GD_ARC_HEAD_COMPONENT_SELECTOR_LAYOUT:
            return nn_gd_arc_component_selector_layout_label(attempt);
        default:
            return 0.0;
    }
}

static int nn_gd_train_arc_aux_head(const NNGDArcPipelineAttemptList *attempts,
                                    const NNGDStringList *vocab,
                                    NNGDArcAuxHead *head,
                                    unsigned epochs,
                                    double learning_rate) {
    unsigned epoch;
    size_t i;
    if (attempts == NULL || vocab == NULL || head == NULL || head->weights == NULL) {
        return -1;
    }
    for (epoch = 0U; epoch < epochs; ++epoch) {
        double loss = 0.0;
        size_t rows = 0U;
        for (i = 0U; i < attempts->count; ++i) {
            const NNGDArcPipelineAttempt *attempt = &attempts->items[i];
            double label;
            double pred;
            double error;
            double active;
            double scale;
            size_t j;
            if (strcmp(attempt->split_name, "source") != 0 ||
                !nn_gd_arc_aux_head_applies(attempt, head->kind)) {
                continue;
            }
            label = nn_gd_arc_aux_label(attempt, head->kind);
            pred = nn_gd_sigmoid(
                nn_gd_arc_attempt_logit(attempt, vocab, head->weights, head->bias));
            error = pred - label;
            active = attempt->tokens.count == 0U ? 1.0 : (double)attempt->tokens.count;
            scale = learning_rate * error / active;
            head->bias -= scale;
            for (j = 0U; j < attempt->token_index_count; ++j) {
                head->weights[attempt->token_indices[j]] -= scale;
            }
            loss += error * error;
            ++rows;
            if (epoch == 0U) {
                ++head->train_rows;
                if (label >= 0.95) {
                    ++head->positive_rows;
                }
                head->label_sum += label;
            }
        }
        head->final_loss = rows == 0U ? 0.0 : loss / (double)rows;
    }
    return 0;
}

static double nn_gd_arc_aux_head_score_applied(const NNGDArcPipelineAttempt *attempt,
                                               const NNGDStringList *vocab,
                                               const NNGDArcAuxHead *head,
                                               int applies) {
    if (attempt == NULL || vocab == NULL || head == NULL || head->weights == NULL ||
        !applies) {
        return 0.5;
    }
    return nn_gd_sigmoid(
        nn_gd_arc_attempt_logit(attempt, vocab, head->weights, head->bias));
}

static int nn_gd_compare_arc_scored_attempts(const void *a, const void *b) {
    const NNGDScoredArcAttempt *sa = (const NNGDScoredArcAttempt *)a;
    const NNGDScoredArcAttempt *sb = (const NNGDScoredArcAttempt *)b;
    const NNGDArcPipelineAttempt *aa =
        &nn_gd_sort_arc_attempts[sa->attempt_index];
    const NNGDArcPipelineAttempt *ab =
        &nn_gd_sort_arc_attempts[sb->attempt_index];
    int cmp = strcmp(aa->family, ab->family);
    if (cmp != 0) {
        return cmp;
    }
    cmp = strcmp(aa->task_id, ab->task_id);
    if (cmp != 0) {
        return cmp;
    }
    cmp = strcmp(aa->split, ab->split);
    if (cmp != 0) {
        return cmp;
    }
    if (sa->score < sb->score) {
        return 1;
    }
    if (sa->score > sb->score) {
        return -1;
    }
    return strcmp(aa->attempt_id, ab->attempt_id);
}

static uint64_t nn_gd_arc_pipeline_proposal_hash(const char *attempt_id) {
    uint64_t hash = UINT64_C(1469598103934665603);
    hash = nn_gd_fnv1a_update(hash, "arc-visual-pipeline-neural");
    hash ^= 0U;
    hash *= UINT64_C(1099511628211);
    hash = nn_gd_fnv1a_update(hash, attempt_id);
    return hash;
}

typedef struct {
    char *name;
    long count;
    long accepted;
    long evaluated_tasks;
    long top1_hits;
    long top3_hits;
    long retained_hits;
    double score_sum;
} NNGDArcNamedMetric;

typedef struct {
    NNGDArcNamedMetric *items;
    size_t count;
    size_t capacity;
} NNGDArcMetricList;

static int nn_gd_arc_metric_list_reserve(NNGDArcMetricList *list, size_t needed) {
    NNGDArcNamedMetric *items;
    size_t capacity;
    if (needed <= list->capacity) {
        return 0;
    }
    capacity = list->capacity == 0U ? 8U : list->capacity * 2U;
    while (capacity < needed) {
        capacity *= 2U;
    }
    items = (NNGDArcNamedMetric *)realloc(list->items,
                                          capacity * sizeof(NNGDArcNamedMetric));
    if (items == NULL) {
        return -1;
    }
    list->items = items;
    list->capacity = capacity;
    return 0;
}

static NNGDArcNamedMetric *nn_gd_arc_metric_get(NNGDArcMetricList *list,
                                                const char *name) {
    size_t i;
    if (name == NULL || name[0] == '\0') {
        name = "(none)";
    }
    for (i = 0U; i < list->count; ++i) {
        if (strcmp(list->items[i].name, name) == 0) {
            return &list->items[i];
        }
    }
    if (nn_gd_arc_metric_list_reserve(list, list->count + 1U) != 0) {
        return NULL;
    }
    memset(&list->items[list->count], 0, sizeof(list->items[list->count]));
    list->items[list->count].name = nn_gd_strdup(name);
    if (list->items[list->count].name == NULL) {
        return NULL;
    }
    return &list->items[list->count++];
}

static void nn_gd_arc_metric_list_free(NNGDArcMetricList *list) {
    size_t i;
    for (i = 0U; i < list->count; ++i) {
        free(list->items[i].name);
    }
    free(list->items);
    list->items = NULL;
    list->count = 0U;
    list->capacity = 0U;
}

static void nn_gd_write_arc_op_recovery(FILE *fp, const NNGDArcMetricList *metrics) {
    size_t i;
    fputc('{', fp);
    for (i = 0U; i < metrics->count; ++i) {
        const NNGDArcNamedMetric *metric = &metrics->items[i];
        if (i > 0U) {
            fputc(',', fp);
        }
        nn_gd_json_write_string(fp, metric->name);
        fprintf(fp,
                ":{\"evaluated_tasks\":%ld,\"top1_hits\":%ld,"
                "\"top3_hits\":%ld,\"retained_hits\":%ld,"
                "\"top1_rate\":%.8f,\"top3_rate\":%.8f,"
                "\"retained_rate\":%.8f}",
                metric->evaluated_tasks,
                metric->top1_hits,
                metric->top3_hits,
                metric->retained_hits,
                metric->evaluated_tasks == 0L
                    ? 0.0
                    : (double)metric->top1_hits / (double)metric->evaluated_tasks,
                metric->evaluated_tasks == 0L
                    ? 0.0
                    : (double)metric->top3_hits / (double)metric->evaluated_tasks,
                metric->evaluated_tasks == 0L
                    ? 0.0
                    : (double)metric->retained_hits / (double)metric->evaluated_tasks);
    }
    fputc('}', fp);
}

static void nn_gd_write_arc_reason_calibration(FILE *fp,
                                               const NNGDArcMetricList *metrics) {
    size_t i;
    fputc('{', fp);
    for (i = 0U; i < metrics->count; ++i) {
        const NNGDArcNamedMetric *metric = &metrics->items[i];
        if (i > 0U) {
            fputc(',', fp);
        }
        nn_gd_json_write_string(fp, metric->name);
        fprintf(fp,
                ":{\"retained\":%ld,\"accepted\":%ld,\"mean_score\":%.8f,"
                "\"acceptance_rate\":%.8f}",
                metric->count,
                metric->accepted,
                metric->count == 0L ? 0.0 : metric->score_sum / (double)metric->count,
                metric->count == 0L ? 0.0 : (double)metric->accepted / (double)metric->count);
    }
    fputc('}', fp);
}

static int nn_gd_arc_group_has_accepted_op(const NNGDScoredArcAttempt *scores,
                                           size_t start,
                                           size_t end,
                                           const NNGDArcPipelineAttemptList *attempts,
                                           const char *op_name,
                                           size_t limit) {
    size_t i;
    size_t capped = start + limit;
    if (capped > end) {
        capped = end;
    }
    for (i = start; i < capped; ++i) {
        const NNGDArcPipelineAttempt *attempt =
            &attempts->items[scores[i].attempt_index];
        if (attempt->accepted && strcmp(attempt->op_name, op_name) == 0) {
            return 1;
        }
    }
    return 0;
}

int nn_grammar_discovery_write_neural_sketch_proposals(
    const char *sketches_path,
    const char *examples_path,
    const char *proposals_output_path,
    const char *checkpoint_input_path,
    const char *checkpoint_output_path,
    unsigned epochs,
    double learning_rate) {
    NNGDSketchList sketches = {0};
    NNGDLabelList labels = {0};
    NNGDStringList vocab = {0};
    NNGDMemoryState memory = {0};
    double *weights = NULL;
    double bias = 0.0;
    FILE *output = NULL;
    int status = -1;
    unsigned epoch;
    size_t i;

    if (nn_gd_read_sketches(sketches_path, &sketches) != 0 ||
        nn_gd_read_labels(examples_path, &sketches, &labels) != 0 ||
        nn_gd_build_vocab(&sketches, &vocab) != 0) {
        goto done;
    }
    weights = (double *)calloc(vocab.count == 0U ? 1U : vocab.count, sizeof(double));
    if (weights == NULL) {
        goto done;
    }
    if (nn_gd_load_neural_checkpoint(checkpoint_input_path,
                                     &vocab,
                                     weights,
                                     &bias,
                                     &memory) != 0) {
        goto done;
    }
    for (epoch = 0U; epoch < epochs; ++epoch) {
        for (i = 0; i < labels.count; ++i) {
            const NNGDLabel *label = &labels.items[i];
            const NNGDSketch *sketch = &sketches.items[label->sketch_index];
            double z = nn_gd_sketch_logit(sketch, &vocab, weights, bias);
            double pred = nn_gd_sigmoid(z);
            double error = pred - label->label;
            double active = sketch->tokens.count == 0U ? 1.0 : (double)sketch->tokens.count;
            double scale = learning_rate * error / active;
            size_t j;
            bias -= learning_rate * error;
            for (j = 0; j < sketch->tokens.count; ++j) {
                long index = nn_gd_token_index(&vocab, sketch->tokens.items[j]);
                if (index >= 0L) {
                    weights[index] -= scale;
                }
            }
        }
    }
    memory.epochs_seen_after = memory.epochs_seen_before + epochs;
    memory.training_examples_seen_after =
        memory.training_examples_seen_before + labels.count;
    if (nn_gd_save_neural_checkpoint(checkpoint_output_path,
                                     &vocab,
                                     weights,
                                     bias,
                                     &memory) != 0) {
        goto done;
    }
    output = fopen(proposals_output_path, "w");
    if (output == NULL) {
        goto done;
    }
    if (nn_gd_write_neural_rows(output,
                                &sketches,
                                &labels,
                                &vocab,
                                weights,
                                bias,
                                epochs,
                                learning_rate,
                                &memory) != 0) {
        goto done;
    }
    status = 0;

done:
    if (output != NULL) {
        fclose(output);
    }
    free(weights);
    nn_gd_string_list_free(&vocab);
    nn_gd_label_list_free(&labels);
    nn_gd_sketch_list_free(&sketches);
    return status;
}

static int nn_gd_write_arc_visual_pipeline_ranker_loaded(
    NNGDArcPipelineAttemptList *attempts,
    const char *proposals_output_path,
    const char *summary_output_path,
    unsigned epochs,
    double learning_rate,
    unsigned max_per_task) {
    NNGDStringList vocab = {0};
    NNGDScoredArcAttempt *target_scores = NULL;
    double *weights = NULL;
    double bias = 0.0;
    NNGDArcAuxHead aux_heads[NNGD_ARC_AUX_HEAD_COUNT] = {
        {"renderer_correctness", NN_GD_ARC_HEAD_RENDERER, NULL, 0.0, 0U, 0U, 0.0, 0.0, 0.0, 0U},
        {"sequence_block_renderer", NN_GD_ARC_HEAD_SEQUENCE_BLOCK_RENDERER, NULL, 0.0, 0U, 0U, 0.0, 0.0, 0.0, 0U},
        {"component_shape_color", NN_GD_ARC_HEAD_COMPONENT_SHAPE_COLOR, NULL, 0.0, 0U, 0U, 0.0, 0.0, 0.0, 0U},
        {"component_selector_layout", NN_GD_ARC_HEAD_COMPONENT_SELECTOR_LAYOUT, NULL, 0.0, 0U, 0U, 0.0, 0.0, 0.0, 0U},
    };
    FILE *proposals = NULL;
    FILE *summary = NULL;
    int status = -1;
    size_t source_count = 0U;
    size_t source_accepted = 0U;
    size_t source_rejected = 0U;
    size_t target_count = 0U;
    size_t target_accepted = 0U;
    size_t source_contrast_pair_count = 0U;
    size_t target_contrast_pair_count = 0U;
    size_t pairwise_training_updates = 0U;
    size_t pairwise_last_epoch_violations = 0U;
    double positive_weight = 1.0;
    unsigned epoch;
    unsigned aux_epochs;
    size_t i;
    size_t scored_count = 0U;
    long target_task_count = 0L;
    long evaluated_tasks = 0L;
    long no_accepted_target_task_count = 0L;
    long top1_hits = 0L;
    long top3_hits = 0L;
    long retained_hits = 0L;
    long proposal_count = 0L;
    NNGDArcMetricList op_recovery = {0};
    NNGDArcMetricList reason_calibration = {0};
    size_t head_index;

    if (max_per_task == 0U) {
        max_per_task = 1U;
    }
    aux_epochs = epochs > 16U ? 16U : epochs;
    if (nn_gd_build_arc_vocab(attempts, &vocab) != 0 ||
        nn_gd_index_arc_attempt_tokens(attempts, &vocab) != 0) {
        goto done;
    }
    nn_gd_resolve_arc_contrast_pairs(attempts,
                                     &source_contrast_pair_count,
                                     &target_contrast_pair_count);
    for (i = 0U; i < attempts->count; ++i) {
        const NNGDArcPipelineAttempt *attempt = &attempts->items[i];
        if (strcmp(attempt->split_name, "source") == 0) {
            ++source_count;
            if (attempt->accepted) {
                ++source_accepted;
            } else {
                ++source_rejected;
            }
        } else if (strcmp(attempt->split_name, "target") == 0) {
            ++target_count;
            if (attempt->accepted) {
                ++target_accepted;
            }
        }
    }
    if (source_accepted > 0U && source_rejected > source_accepted) {
        positive_weight = (double)source_rejected / (double)source_accepted;
        if (positive_weight > 64.0) {
            positive_weight = 64.0;
        }
    }
    weights = (double *)calloc(vocab.count == 0U ? 1U : vocab.count, sizeof(double));
    target_scores = (NNGDScoredArcAttempt *)calloc(target_count == 0U ? 1U : target_count,
                                                   sizeof(NNGDScoredArcAttempt));
    if (weights == NULL || target_scores == NULL) {
        goto done;
    }
    for (head_index = 0U;
         head_index < NNGD_ARC_AUX_HEAD_COUNT;
         ++head_index) {
        aux_heads[head_index].weights =
            (double *)calloc(vocab.count == 0U ? 1U : vocab.count, sizeof(double));
        if (aux_heads[head_index].weights == NULL) {
            goto done;
        }
    }
    for (epoch = 0U; epoch < epochs; ++epoch) {
        for (i = 0U; i < attempts->count; ++i) {
            const NNGDArcPipelineAttempt *attempt = &attempts->items[i];
            double label;
            double z;
            double pred;
            double error;
            double active;
            double row_weight;
            double scale;
            size_t j;
            if (strcmp(attempt->split_name, "source") != 0) {
                continue;
            }
            label = attempt->accepted ? 1.0 : attempt->verifier_score;
            if (label < 0.0) {
                label = 0.0;
            } else if (label > 1.0) {
                label = 1.0;
            }
            z = nn_gd_arc_attempt_logit(attempt, &vocab, weights, bias);
            pred = nn_gd_sigmoid(z);
            row_weight = attempt->accepted ? positive_weight : 1.0 + label;
            error = (pred - label) * row_weight;
            active = attempt->tokens.count == 0U ? 1.0 : (double)attempt->tokens.count;
            scale = learning_rate * error / active;
            bias -= scale;
            for (j = 0U; j < attempt->token_index_count; ++j) {
                weights[attempt->token_indices[j]] -= scale;
            }
        }
        pairwise_last_epoch_violations = 0U;
        for (i = 0U; i < attempts->count; ++i) {
            const NNGDArcPipelineAttempt *negative = &attempts->items[i];
            const NNGDArcPipelineAttempt *positive;
            double pos_z;
            double neg_z;
            double pred;
            double pair_scale;
            double pos_active;
            double neg_active;
            size_t j;
            if (strcmp(negative->split_name, "source") != 0 ||
                negative->accepted ||
                negative->contrast_accepted_index == SIZE_MAX ||
                negative->contrast_accepted_index >= attempts->count) {
                continue;
            }
            positive = &attempts->items[negative->contrast_accepted_index];
            if (!positive->accepted || strcmp(positive->split_name, "source") != 0) {
                continue;
            }
            pos_z = nn_gd_arc_attempt_logit(positive, &vocab, weights, bias);
            neg_z = nn_gd_arc_attempt_logit(negative, &vocab, weights, bias);
            if (pos_z <= neg_z) {
                ++pairwise_last_epoch_violations;
            }
            pred = nn_gd_sigmoid(pos_z - neg_z);
            pair_scale = 0.5 * learning_rate * (pred - 1.0);
            pos_active =
                positive->tokens.count == 0U ? 1.0 : (double)positive->tokens.count;
            neg_active =
                negative->tokens.count == 0U ? 1.0 : (double)negative->tokens.count;
            for (j = 0U; j < positive->token_index_count; ++j) {
                weights[positive->token_indices[j]] -= pair_scale / pos_active;
            }
            for (j = 0U; j < negative->token_index_count; ++j) {
                weights[negative->token_indices[j]] += pair_scale / neg_active;
            }
            ++pairwise_training_updates;
        }
    }
    for (head_index = 0U;
         head_index < NNGD_ARC_AUX_HEAD_COUNT;
         ++head_index) {
        if (nn_gd_train_arc_aux_head(attempts,
                                     &vocab,
                                     &aux_heads[head_index],
                                     aux_epochs,
                                     learning_rate) != 0) {
            goto done;
        }
    }
    for (i = 0U; i < attempts->count; ++i) {
        NNGDArcPipelineAttempt *attempt = &attempts->items[i];
        double aux_scores[NNGD_ARC_AUX_HEAD_COUNT];
        int aux_applies[NNGD_ARC_AUX_HEAD_COUNT];
        if (strcmp(attempt->split_name, "target") != 0) {
            continue;
        }
        for (head_index = 0U; head_index < NNGD_ARC_AUX_HEAD_COUNT; ++head_index) {
            aux_applies[head_index] =
                nn_gd_arc_aux_head_applies(attempt, aux_heads[head_index].kind);
            aux_scores[head_index] =
                nn_gd_arc_aux_head_score_applied(attempt,
                                                 &vocab,
                                                 &aux_heads[head_index],
                                                 aux_applies[head_index]);
        }
        target_scores[scored_count].attempt_index = i;
        target_scores[scored_count].model_score =
            nn_gd_sigmoid(nn_gd_arc_attempt_logit(attempt, &vocab, weights, bias));
        target_scores[scored_count].renderer_score = aux_scores[0];
        target_scores[scored_count].sequence_block_score = aux_scores[1];
        target_scores[scored_count].component_score = aux_scores[2];
        target_scores[scored_count].component_selector_layout_score = aux_scores[3];
        target_scores[scored_count].score =
            0.646 * target_scores[scored_count].model_score +
            0.350 * attempt->verifier_score +
            0.001 * target_scores[scored_count].renderer_score +
            0.001 * target_scores[scored_count].component_score +
            0.001 * target_scores[scored_count].component_selector_layout_score +
            0.001 * target_scores[scored_count].sequence_block_score;
        for (head_index = 0U;
             head_index < NNGD_ARC_AUX_HEAD_COUNT;
             ++head_index) {
            if (aux_applies[head_index]) {
                aux_heads[head_index].target_score_sum += aux_scores[head_index];
                ++aux_heads[head_index].target_rows;
            }
        }
        ++scored_count;
    }
    nn_gd_sort_arc_attempts = attempts->items;
    qsort(target_scores,
          scored_count,
          sizeof(NNGDScoredArcAttempt),
          nn_gd_compare_arc_scored_attempts);
    nn_gd_sort_arc_attempts = NULL;

    proposals = fopen(proposals_output_path, "w");
    summary = fopen(summary_output_path, "w");
    if (proposals == NULL || summary == NULL) {
        goto done;
    }
    for (i = 0U; i < scored_count;) {
        size_t start = i;
        size_t end;
        size_t rank;
        int group_has_accepted = 0;
        int group_top1_hit = 0;
        int group_top3_hit = 0;
        int group_retained_hit = 0;
        NNGDStringList accepted_ops = {0};
        const NNGDArcPipelineAttempt *first_attempt =
            &attempts->items[target_scores[start].attempt_index];
        for (end = start; end < scored_count; ++end) {
            const NNGDArcPipelineAttempt *attempt =
                &attempts->items[target_scores[end].attempt_index];
            if (strcmp(first_attempt->family, attempt->family) != 0 ||
                strcmp(first_attempt->task_id, attempt->task_id) != 0 ||
                strcmp(first_attempt->split, attempt->split) != 0) {
                break;
            }
            if (attempt->accepted) {
                group_has_accepted = 1;
                if (nn_gd_string_list_add_unique(&accepted_ops, attempt->op_name) != 0) {
                    nn_gd_string_list_free(&accepted_ops);
                    goto done;
                }
            }
        }
        for (rank = 0U; rank < max_per_task && start + rank < end; ++rank) {
            const NNGDScoredArcAttempt *score = &target_scores[start + rank];
            const NNGDArcPipelineAttempt *attempt =
                &attempts->items[score->attempt_index];
            NNGDArcNamedMetric *reason_metric =
                nn_gd_arc_metric_get(&reason_calibration, attempt->rejection_reason);
            uint64_t proposal_hash = nn_gd_arc_pipeline_proposal_hash(attempt->attempt_id);
            if (reason_metric == NULL) {
                nn_gd_string_list_free(&accepted_ops);
                goto done;
            }
            ++reason_metric->count;
            reason_metric->accepted += attempt->accepted ? 1L : 0L;
            reason_metric->score_sum += score->score;
            ++proposal_count;
            if (attempt->accepted) {
                group_retained_hit = 1;
                if (rank == 0U) {
                    group_top1_hit = 1;
                }
                if (rank < 3U) {
                    group_top3_hit = 1;
                }
            }
            fputs("{\"schema\":\"crlp.neural_grammar_discovery.arc_visual_pipeline_proposal.v1\","
                  "\"report_version\":1,\"proposal_id\":\"avpn:",
                  proposals);
            fprintf(proposals, "%016" PRIx64, proposal_hash);
            fputs("\",\"proposer\":\"neural_visual_pipeline_ranker\",\"family\":", proposals);
            nn_gd_json_write_string(proposals, attempt->family);
            fputs(",\"task_id\":", proposals);
            nn_gd_json_write_string(proposals, attempt->task_id);
            fputs(",\"split\":", proposals);
            nn_gd_json_write_string(proposals, attempt->split);
            fputs(",\"target_attempt_id\":", proposals);
            nn_gd_json_write_string(proposals, attempt->attempt_id);
            fputs(",\"library_id\":", proposals);
            nn_gd_json_write_string(proposals, attempt->library_id);
            fputs(",\"pipeline_kind\":", proposals);
            nn_gd_json_write_string(proposals, attempt->pipeline_kind);
            fputs(",\"op_name\":", proposals);
            nn_gd_json_write_string(proposals, attempt->op_name);
            fputs(",\"composition_kind\":", proposals);
            nn_gd_json_write_string(proposals, attempt->composition_kind);
            fputs(",\"normalization_frame\":", proposals);
            nn_gd_json_write_string(proposals, attempt->normalization_frame);
            fputs(",\"typed_stages\":", proposals);
            nn_gd_json_write_string(proposals, attempt->typed_stages);
            fputs(",\"primitive_contract\":", proposals);
            nn_gd_json_write_string(proposals, attempt->primitive_contract);
            fprintf(proposals,
                    ",\"score\":%.8f,\"target_accepted\":%s,"
                    "\"target_rejection_reason\":",
                    score->score,
                    attempt->accepted ? "true" : "false");
            nn_gd_json_write_string(proposals, attempt->rejection_reason);
            fputs(",\"failure_stage\":", proposals);
            nn_gd_json_write_string(proposals, attempt->failure_stage);
            fputs(",\"failure_detail\":", proposals);
            nn_gd_json_write_string(proposals, attempt->failure_detail);
            fprintf(proposals,
                    ",\"rank_features\":{\"epochs\":%u,\"learning_rate\":%.17g,"
                    "\"vocab_size\":%zu,\"source_attempt_count\":%zu,"
                    "\"source_accepted_count\":%zu,\"source_rejected_count\":%zu,"
                    "\"positive_weight\":%.8f,\"rank_within_task\":%zu,"
                    "\"tokens\":%zu,\"source_contrast_pairs\":%zu,"
                    "\"target_contrast_pairs\":%zu,"
                    "\"contrast_slot_distance\":%ld,"
                    "\"contrast_rejected_sibling_count\":%ld,"
                    "\"model_score\":%.8f,\"verifier_proxy_score\":%.8f,"
                    "\"renderer_model_score\":%.8f,"
                    "\"sequence_block_renderer_score\":%.8f,"
                    "\"component_shape_color_score\":%.8f,"
                    "\"component_selector_layout_score\":%.8f,"
                    "\"verifier_train_pairs\":%ld,"
                    "\"verifier_checked_cells\":%ld,"
                    "\"verifier_mismatches\":%ld,"
                    "\"verifier_unsupported_pairs\":%ld,"
                    "\"verifier_apply_failed_pairs\":%ld,"
                    "\"verifier_shape_mismatch_pairs\":%ld,"
                    "\"verifier_missing_output_pairs\":%ld,"
                    "\"verifier_first_failing_pair\":%ld,"
                    "\"verifier_mismatch_ratio\":%.8f,"
                    "\"proposal_count\":%ld,\"support_tasks\":%ld,"
                    "\"utility_mean\":%.8f},\"rank\":%zu}\n",
                    epochs,
                    learning_rate,
                    vocab.count,
                    source_count,
                    source_accepted,
                    source_rejected,
                    positive_weight,
                    rank + 1U,
                    attempt->tokens.count,
                    source_contrast_pair_count,
                    target_contrast_pair_count,
                    attempt->contrast_slot_distance,
                    attempt->contrast_rejected_sibling_count,
                    score->model_score,
                    attempt->verifier_score,
                    score->renderer_score,
                    score->sequence_block_score,
                    score->component_score,
                    score->component_selector_layout_score,
                    attempt->verifier_train_pairs,
                    attempt->verifier_checked_cells,
                    attempt->verifier_mismatches,
                    attempt->verifier_unsupported_pairs,
                    attempt->verifier_apply_failed_pairs,
                    attempt->verifier_shape_mismatch_pairs,
                    attempt->verifier_missing_output_pairs,
                    attempt->verifier_first_failing_pair,
                    attempt->verifier_mismatch_ratio,
                    attempt->proposal_count,
                    attempt->support_tasks,
                    attempt->utility_mean,
                    rank + 1U);
        }
        ++target_task_count;
        if (group_has_accepted) {
            size_t op_index;
            ++evaluated_tasks;
            top1_hits += group_top1_hit ? 1L : 0L;
            top3_hits += group_top3_hit ? 1L : 0L;
            retained_hits += group_retained_hit ? 1L : 0L;
            for (op_index = 0U; op_index < accepted_ops.count; ++op_index) {
                const char *op_name = accepted_ops.items[op_index];
                NNGDArcNamedMetric *metric = nn_gd_arc_metric_get(&op_recovery, op_name);
                if (metric == NULL) {
                    nn_gd_string_list_free(&accepted_ops);
                    goto done;
                }
                ++metric->evaluated_tasks;
                metric->top1_hits +=
                    nn_gd_arc_group_has_accepted_op(target_scores,
                                                    start,
                                                    end,
                                                    attempts,
                                                    op_name,
                                                    1U)
                        ? 1L
                        : 0L;
                metric->top3_hits +=
                    nn_gd_arc_group_has_accepted_op(target_scores,
                                                    start,
                                                    end,
                                                    attempts,
                                                    op_name,
                                                    3U)
                        ? 1L
                        : 0L;
                metric->retained_hits +=
                    nn_gd_arc_group_has_accepted_op(target_scores,
                                                    start,
                                                    end,
                                                    attempts,
                                                    op_name,
                                                    max_per_task)
                        ? 1L
                        : 0L;
            }
        } else {
            ++no_accepted_target_task_count;
        }
        nn_gd_string_list_free(&accepted_ops);
        i = end;
    }
    fputs("{\"schema\":\"crlp.neural_grammar_discovery.arc_visual_pipeline_neural_summary.v1\","
          "\"report_version\":1,\"proposer\":\"neural_visual_pipeline_ranker\",",
          summary);
    fprintf(summary,
            "\"source_attempt_count\":%zu,\"source_accepted_count\":%zu,"
            "\"source_rejected_count\":%zu,\"target_attempt_count\":%zu,"
            "\"target_accepted_count\":%zu,\"proposal_count\":%ld,"
            "\"target_task_count\":%ld,"
            "\"accepted_target_task_count\":%ld,"
            "\"no_accepted_target_task_count\":%ld,"
            "\"search_space_coverage_rate\":%.8f,"
            "\"evaluated_tasks\":%ld,"
            "\"top1_hits\":%ld,\"top3_hits\":%ld,\"retained_hits\":%ld,"
            "\"top1_rate\":%.8f,\"top3_rate\":%.8f,\"retained_rate\":%.8f,"
            "\"top1_all_task_rate\":%.8f,"
            "\"top3_all_task_rate\":%.8f,"
            "\"retained_all_task_rate\":%.8f,"
            "\"epochs\":%u,\"learning_rate\":%.17g,\"vocab_size\":%zu,"
            "\"positive_weight\":%.8f,\"source_contrast_pair_count\":%zu,"
            "\"target_contrast_pair_count\":%zu,"
            "\"pairwise_training_updates\":%zu,"
            "\"pairwise_last_epoch_violations\":%zu,"
            "\"auxiliary_head_epochs\":%u,"
            "\"rerank_model_weight\":0.646,\"rerank_verifier_proxy_weight\":0.35,"
            "\"rerank_renderer_weight\":0.001,"
            "\"rerank_component_shape_color_weight\":0.001,"
            "\"rerank_component_selector_layout_weight\":0.001,"
            "\"rerank_sequence_block_weight\":0.001,"
            "\"training_mode\":\"source_to_target_visual_pipeline_attempts_with_dense_verifier_targets_and_auxiliary_renderer_component_heads\",",
            source_count,
            source_accepted,
            source_rejected,
            target_count,
            target_accepted,
            proposal_count,
            target_task_count,
            evaluated_tasks,
            no_accepted_target_task_count,
            target_task_count == 0L
                ? 0.0
                : (double)evaluated_tasks / (double)target_task_count,
            evaluated_tasks,
            top1_hits,
            top3_hits,
            retained_hits,
            evaluated_tasks == 0L ? 0.0 : (double)top1_hits / (double)evaluated_tasks,
            evaluated_tasks == 0L ? 0.0 : (double)top3_hits / (double)evaluated_tasks,
            evaluated_tasks == 0L ? 0.0 : (double)retained_hits / (double)evaluated_tasks,
            target_task_count == 0L ? 0.0 : (double)top1_hits / (double)target_task_count,
            target_task_count == 0L ? 0.0 : (double)top3_hits / (double)target_task_count,
            target_task_count == 0L
                ? 0.0
                : (double)retained_hits / (double)target_task_count,
            epochs,
            learning_rate,
            vocab.count,
            positive_weight,
            source_contrast_pair_count,
            target_contrast_pair_count,
            pairwise_training_updates,
            pairwise_last_epoch_violations,
            aux_epochs);
    fputs("\"auxiliary_heads\":{", summary);
    for (head_index = 0U;
         head_index < NNGD_ARC_AUX_HEAD_COUNT;
         ++head_index) {
        const NNGDArcAuxHead *head = &aux_heads[head_index];
        if (head_index > 0U) {
            fputc(',', summary);
        }
        nn_gd_json_write_string(summary, head->name);
        fprintf(summary,
                ":{\"train_rows\":%zu,\"positive_rows\":%zu,"
                "\"mean_label\":%.8f,\"final_loss\":%.8f,"
                "\"target_rows\":%zu,\"target_mean_score\":%.8f}",
                head->train_rows,
                head->positive_rows,
                head->train_rows == 0U ? 0.0 : head->label_sum / (double)head->train_rows,
                head->final_loss,
                head->target_rows,
                head->target_rows == 0U
                    ? 0.0
                    : head->target_score_sum / (double)head->target_rows);
    }
    fputs("},", summary);
    fputs("\"by_proposer\":{\"neural_visual_pipeline_ranker\":{", summary);
    fprintf(summary,
            "\"target_task_count\":%ld,"
            "\"accepted_target_task_count\":%ld,"
            "\"no_accepted_target_task_count\":%ld,"
            "\"search_space_coverage_rate\":%.8f,"
            "\"evaluated_tasks\":%ld,\"top1_hits\":%ld,\"top3_hits\":%ld,"
            "\"retained_hits\":%ld,\"top1_rate\":%.8f,\"top3_rate\":%.8f,"
            "\"retained_rate\":%.8f,\"top1_all_task_rate\":%.8f,"
            "\"top3_all_task_rate\":%.8f,\"retained_all_task_rate\":%.8f}},"
            "\"by_accepted_op\":",
            target_task_count,
            evaluated_tasks,
            no_accepted_target_task_count,
            target_task_count == 0L
                ? 0.0
                : (double)evaluated_tasks / (double)target_task_count,
            evaluated_tasks,
            top1_hits,
            top3_hits,
            retained_hits,
            evaluated_tasks == 0L ? 0.0 : (double)top1_hits / (double)evaluated_tasks,
            evaluated_tasks == 0L ? 0.0 : (double)top3_hits / (double)evaluated_tasks,
            evaluated_tasks == 0L ? 0.0 : (double)retained_hits / (double)evaluated_tasks,
            target_task_count == 0L ? 0.0 : (double)top1_hits / (double)target_task_count,
            target_task_count == 0L ? 0.0 : (double)top3_hits / (double)target_task_count,
            target_task_count == 0L
                ? 0.0
                : (double)retained_hits / (double)target_task_count);
    nn_gd_write_arc_op_recovery(summary, &op_recovery);
    fputs(",\"calibration_by_rejection_reason\":", summary);
    nn_gd_write_arc_reason_calibration(summary, &reason_calibration);
    fputs("}\n", summary);
    if (ferror(proposals) || ferror(summary)) {
        goto done;
    }
    status = 0;

done:
    if (proposals != NULL) {
        fclose(proposals);
    }
    if (summary != NULL) {
        fclose(summary);
    }
    nn_gd_arc_metric_list_free(&reason_calibration);
    nn_gd_arc_metric_list_free(&op_recovery);
    free(target_scores);
    for (head_index = 0U;
         head_index < NNGD_ARC_AUX_HEAD_COUNT;
         ++head_index) {
        free(aux_heads[head_index].weights);
    }
    free(weights);
    nn_gd_string_list_free(&vocab);
    return status;
}

int nn_grammar_discovery_write_arc_visual_pipeline_ranker(
    const char *attempt_rows_path,
    const char *proposals_output_path,
    const char *summary_output_path,
    unsigned epochs,
    double learning_rate,
    unsigned max_per_task) {
    NNGDArcPipelineAttemptList attempts = {0};
    int status = -1;
    if (nn_gd_read_arc_pipeline_attempts(attempt_rows_path, &attempts) != 0) {
        goto done;
    }
    status = nn_gd_write_arc_visual_pipeline_ranker_loaded(&attempts,
                                                           proposals_output_path,
                                                           summary_output_path,
                                                           epochs,
                                                           learning_rate,
                                                           max_per_task);

done:
    nn_gd_arc_attempt_list_free(&attempts);
    return status;
}

int nn_grammar_discovery_write_arc_visual_pipeline_ranker_from_jsonl(
    const char *const *source_paths,
    size_t source_count,
    const char *const *target_paths,
    size_t target_count,
    const char *proposals_output_path,
    const char *summary_output_path,
    unsigned epochs,
    double learning_rate,
    unsigned max_per_task) {
    NNGDArcPipelineAttemptList attempts = {0};
    int status = -1;
    size_t i;
    if (source_paths == NULL || target_paths == NULL || source_count == 0U ||
        target_count == 0U) {
        return -1;
    }
    for (i = 0U; i < source_count; ++i) {
        if (nn_gd_read_arc_pipeline_attempts_jsonl(source_paths[i],
                                                   "source",
                                                   &attempts) != 0) {
            goto done;
        }
    }
    for (i = 0U; i < target_count; ++i) {
        if (nn_gd_read_arc_pipeline_attempts_jsonl(target_paths[i],
                                                   "target",
                                                   &attempts) != 0) {
            goto done;
        }
    }
    status = nn_gd_write_arc_visual_pipeline_ranker_loaded(&attempts,
                                                           proposals_output_path,
                                                           summary_output_path,
                                                           epochs,
                                                           learning_rate,
                                                           max_per_task);

done:
    nn_gd_arc_attempt_list_free(&attempts);
    return status;
}
