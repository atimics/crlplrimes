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
#define NN_GD_MAX_FIELDS 64
#define NN_GD_COUNTEREXAMPLE_LIMIT 8

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

static const NNGDSketch *nn_gd_sort_sketches = NULL;

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
                                 const char *slots) {
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
        nn_gd_add_slot_tokens(&sketch->tokens, slots) != 0) {
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
                                  nn_gd_field(fields, field_count, 5U)) != 0) {
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

static int nn_gd_write_neural_rows(FILE *fp,
                                   const NNGDSketchList *sketches,
                                   const NNGDLabelList *labels,
                                   const NNGDStringList *vocab,
                                   const double *weights,
                                   double bias,
                                   unsigned epochs,
                                   double learning_rate) {
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
                "\"training_examples\":%zu},\"rank\":%zu}\n",
                scores[i].score,
                epochs,
                learning_rate,
                vocab->count,
                labels->count,
                i + 1U);
    }
    free(scores);
    return ferror(fp) ? -1 : 0;
}

int nn_grammar_discovery_write_neural_sketch_proposals(
    const char *sketches_path,
    const char *examples_path,
    const char *proposals_output_path,
    unsigned epochs,
    double learning_rate) {
    NNGDSketchList sketches = {0};
    NNGDLabelList labels = {0};
    NNGDStringList vocab = {0};
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
                                learning_rate) != 0) {
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
