#include "nn_grammar_discovery.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define SP_LINE_MAX 65536
#define SP_MAX_FIELDS 128

typedef struct {
    char *key;
    size_t count;
} SPCounter;

typedef struct {
    SPCounter *items;
    size_t count;
    size_t capacity;
} SPCounterList;

typedef struct {
    char **items;
    size_t count;
    size_t capacity;
} SPStringList;

typedef struct {
    const char *source_op;
    int op_id;
    const char *op_name;
} SPOpMap;

typedef struct {
    int op_id;
    const char *op_name;
    char scope[16];
    long input_width;
    long input_height;
    long output_width;
    long output_height;
    long scale_x;
    long scale_y;
    long input_components;
    long output_components;
    size_t accepted;
    size_t rejected;
    size_t near_miss;
    size_t hard_reject;
    double verifier_score_sum;
} SPGenerativeGroup;

typedef struct {
    SPGenerativeGroup *items;
    size_t count;
    size_t capacity;
} SPGenerativeGroups;

typedef struct {
    size_t input_rows;
    size_t accepted_rows;
    size_t rejected_rows;
    size_t mapped_accepted_rows;
    size_t skipped_unaccepted_rows;
    size_t skipped_unmapped_accepted_rows;
    size_t skipped_unverified_accepted_rows;
    size_t template_rows;
    size_t slot_rows;
    size_t prior_rows;
    size_t generative_rows;
    SPCounterList mapped_ops;
    SPCounterList unmapped_ops;
    SPCounterList source_ops;
    SPCounterList runtime_ops;
    SPCounterList rejection_stages;
    SPStringList synthetic_hashes;
} SPExportStats;

typedef struct {
    int available;
    char path[4096];
    long loaded_tasks;
    long certified_tasks;
    long test_exact_tasks;
    long test_cells;
    long test_correct_cells;
    double cell_accuracy;
    long selection_rows;
    long runtime_mined_sketches;
    long runtime_generative_prior_rows;
    long runtime_generative_prior_ops;
    long runtime_generative_prior_mined_sketches;
    long trace_rows;
    long trace_emitted_sketches;
    SPCounterList trace_decisions;
    SPStringList state_hashes;
} SPLogStats;

typedef struct {
    long synthetic_rows;
    long accepted_rows;
    long rejected_rows;
    long max_accepted;
    long max_grid_size;
    double elapsed_seconds;
    char *stop_reason;
} SPSyntheticSummary;

typedef struct {
    long source_attempt_count;
    long source_accepted_count;
    long target_task_count;
    long accepted_target_task_count;
    long proposal_count;
    double top1_all_task_rate;
    double retained_all_task_rate;
} SPRankerSummary;

typedef struct {
    int accepted;
    int has_translate_fields;
    int has_specs;
    char attempt_id[128];
    char task_id[128];
    char failure_stage[64];
    char rejection_reason[128];
    char source_path[4096];
    double verifier_score;
    long selector;
    long color;
    long output_color;
    long dx;
    long dy;
    long render_mode;
    long attempted_specs;
    long accepted_specs;
} LSOExample;

typedef struct {
    char family[128];
    char operator_name[256];
    char operator_id[64];
    char compiler_target[64];
    char composition_kind[128];
    char normalization_frame[128];
    char typed_stages[256];
    char primitive_contract[256];
    char slot_signature[256];
    char visual_program[256];
    char learned_from[128];
    char compiler_contract[256];
    size_t accepted_count;
    size_t rejected_count;
    size_t near_miss_count;
    size_t hard_rejected_count;
    double verifier_score_sum;
    double best_verifier_score;
    SPStringList source_tasks;
    SPStringList accepted_tasks;
    SPCounterList failure_stages;
    SPCounterList rejection_reasons;
    LSOExample examples[5];
    size_t example_count;
} LSOOperator;

typedef struct {
    LSOOperator *items;
    size_t count;
    size_t capacity;
} LSOOperatorList;

typedef struct {
    size_t attempt_rows;
    size_t promoted_operators;
    size_t candidate_operators;
    size_t blocked_operators;
    SPCounterList family_counts;
    SPCounterList learned_from_counts;
    SPCounterList status_counts;
} LSOSummary;

static const SPOpMap SP_OP_MAPS[] = {
    {"crop", 0, "crop"},
    {"tile_scale", 1, "tile_repeat"},
    {"macrocell_substitute", 2, "macrocell_substitute"},
    {"component_color_summary", 4, "component_color_summary"},
    {"component_color_summary", 13, "component_count_line"},
    {"masked_or_normalized_crop", 5, "masked_or_normalized_crop"},
    {"masked_or_normalized_crop", 7, "object_mask_extract"},
    {"recolor", 10, "palette_scale"},
    {"recolor", 11, "object_color_tile"},
    {"recolor_crop", 14, "recolor_crop"},
    {"fill_or_complete", 15, "marker_polyline"},
    {"symmetry", 17, "separator_panel_relation_mask"},
    {"visual_combinator", 18, "visual_combinator"},
    {"separator_panel_map", 9, "separator_panel_map"},
    {"object_motion", 20, "object_translate"},
    {"object_relation_translate_to_anchor", 20, "object_translate"},
};

static char *sp_strdup(const char *value) {
    size_t len = value == NULL ? 0U : strlen(value);
    char *copy = (char *)malloc(len + 1U);
    if (copy == NULL) {
        return NULL;
    }
    if (len > 0U) {
        memcpy(copy, value, len);
    }
    copy[len] = '\0';
    return copy;
}

static size_t sp_split_tabs(char *line, char **fields, size_t capacity) {
    size_t count = 0U;
    char *cursor = line;
    if (line == NULL || fields == NULL || capacity == 0U) {
        return 0U;
    }
    line[strcspn(line, "\r\n")] = '\0';
    while (count < capacity) {
        char *tab;
        fields[count++] = cursor;
        tab = strchr(cursor, '\t');
        if (tab == NULL) {
            break;
        }
        *tab = '\0';
        cursor = tab + 1;
    }
    return count;
}

static const char *sp_field(char **fields, size_t count, size_t index) {
    return index < count && fields[index] != NULL ? fields[index] : "";
}

static long sp_long(const char *value) {
    char *end = NULL;
    long parsed;
    if (value == NULL || value[0] == '\0') {
        return 0L;
    }
    parsed = strtol(value, &end, 10);
    return end == value ? 0L : parsed;
}

static double sp_double(const char *value) {
    char *end = NULL;
    double parsed;
    if (value == NULL || value[0] == '\0') {
        return 0.0;
    }
    parsed = strtod(value, &end);
    return end == value || !isfinite(parsed) ? 0.0 : parsed;
}

static int sp_bool_text(const char *value) {
    return value != NULL &&
           (strcmp(value, "1") == 0 || strcmp(value, "true") == 0 ||
            strcmp(value, "True") == 0);
}

static int sp_mkdirs(const char *dir) {
    char buffer[4096];
    size_t len;
    size_t i;
    if (dir == NULL || dir[0] == '\0') {
        return 0;
    }
    len = strlen(dir);
    if (len >= sizeof(buffer)) {
        return -1;
    }
    memcpy(buffer, dir, len + 1U);
    for (i = 1U; i < len; ++i) {
        if (buffer[i] == '/') {
            buffer[i] = '\0';
            if (buffer[0] != '\0' && mkdir(buffer, 0777) != 0 &&
                errno != EEXIST) {
                return -1;
            }
            buffer[i] = '/';
        }
    }
    if (mkdir(buffer, 0777) != 0 && errno != EEXIST) {
        return -1;
    }
    return 0;
}

static int sp_mkdir_parent(const char *path) {
    char buffer[4096];
    char *slash;
    if (path == NULL || path[0] == '\0') {
        return -1;
    }
    if (strlen(path) >= sizeof(buffer)) {
        return -1;
    }
    strcpy(buffer, path);
    slash = strrchr(buffer, '/');
    if (slash == NULL) {
        return 0;
    }
    *slash = '\0';
    return sp_mkdirs(buffer);
}

static int sp_counter_reserve(SPCounterList *list, size_t needed) {
    SPCounter *items;
    size_t capacity;
    if (needed <= list->capacity) {
        return 0;
    }
    capacity = list->capacity == 0U ? 16U : list->capacity * 2U;
    while (capacity < needed) {
        capacity *= 2U;
    }
    items = (SPCounter *)realloc(list->items, capacity * sizeof(*list->items));
    if (items == NULL) {
        return -1;
    }
    list->items = items;
    list->capacity = capacity;
    return 0;
}

static int sp_counter_add(SPCounterList *list, const char *key) {
    size_t i;
    if (key == NULL || key[0] == '\0') {
        key = "unknown";
    }
    for (i = 0U; i < list->count; ++i) {
        if (strcmp(list->items[i].key, key) == 0) {
            ++list->items[i].count;
            return 0;
        }
    }
    if (sp_counter_reserve(list, list->count + 1U) != 0) {
        return -1;
    }
    list->items[list->count].key = sp_strdup(key);
    if (list->items[list->count].key == NULL) {
        return -1;
    }
    list->items[list->count].count = 1U;
    ++list->count;
    return 0;
}

static int sp_counter_cmp(const void *a, const void *b) {
    const SPCounter *ca = (const SPCounter *)a;
    const SPCounter *cb = (const SPCounter *)b;
    return strcmp(ca->key, cb->key);
}

static void sp_counter_write(FILE *fp, SPCounterList *list) {
    size_t i;
    if (list->count > 1U) {
        qsort(list->items, list->count, sizeof(list->items[0]), sp_counter_cmp);
    }
    fputc('{', fp);
    for (i = 0U; i < list->count; ++i) {
        if (i > 0U) {
            fputc(',', fp);
        }
        fputc('"', fp);
        for (const unsigned char *p = (const unsigned char *)list->items[i].key;
             *p != '\0';
             ++p) {
            if (*p == '"' || *p == '\\') {
                fputc('\\', fp);
            }
            fputc((int)*p, fp);
        }
        fprintf(fp, "\":%zu", list->items[i].count);
    }
    fputc('}', fp);
}

static void sp_counter_free(SPCounterList *list) {
    size_t i;
    for (i = 0U; i < list->count; ++i) {
        free(list->items[i].key);
    }
    free(list->items);
    memset(list, 0, sizeof(*list));
}

static int sp_string_reserve(SPStringList *list, size_t needed) {
    char **items;
    size_t capacity;
    if (needed <= list->capacity) {
        return 0;
    }
    capacity = list->capacity == 0U ? 32U : list->capacity * 2U;
    while (capacity < needed) {
        capacity *= 2U;
    }
    items = (char **)realloc(list->items, capacity * sizeof(*list->items));
    if (items == NULL) {
        return -1;
    }
    list->items = items;
    list->capacity = capacity;
    return 0;
}

static int sp_string_contains(const SPStringList *list, const char *value) {
    size_t i;
    for (i = 0U; i < list->count; ++i) {
        if (strcmp(list->items[i], value) == 0) {
            return 1;
        }
    }
    return 0;
}

static int sp_string_add_unique(SPStringList *list, const char *value) {
    if (value == NULL || value[0] == '\0' || sp_string_contains(list, value)) {
        return 0;
    }
    if (sp_string_reserve(list, list->count + 1U) != 0) {
        return -1;
    }
    list->items[list->count] = sp_strdup(value);
    if (list->items[list->count] == NULL) {
        return -1;
    }
    ++list->count;
    return 0;
}

static void sp_string_list_free(SPStringList *list) {
    size_t i;
    for (i = 0U; i < list->count; ++i) {
        free(list->items[i]);
    }
    free(list->items);
    memset(list, 0, sizeof(*list));
}

static void sp_json_string(FILE *fp, const char *value) {
    const unsigned char *p = (const unsigned char *)(value == NULL ? "" : value);
    fputc('"', fp);
    for (; *p != '\0'; ++p) {
        switch (*p) {
        case '"':
        case '\\':
            fputc('\\', fp);
            fputc((int)*p, fp);
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

static const char *sp_json_value(const char *line, const char *key) {
    char needle[128];
    const char *found;
    if (line == NULL || key == NULL) {
        return NULL;
    }
    snprintf(needle, sizeof(needle), "\"%s\":", key);
    found = strstr(line, needle);
    return found == NULL ? NULL : found + strlen(needle);
}

static double sp_json_double(const char *line, const char *key) {
    const char *cursor = sp_json_value(line, key);
    char *end = NULL;
    double value;
    if (cursor == NULL) {
        return 0.0;
    }
    while (isspace((unsigned char)*cursor)) {
        ++cursor;
    }
    value = strtod(cursor, &end);
    return end == cursor || !isfinite(value) ? 0.0 : value;
}

static char *sp_json_string_dup(const char *line, const char *key) {
    const char *cursor = sp_json_value(line, key);
    char *out;
    size_t used = 0U;
    size_t capacity = 64U;
    if (cursor == NULL) {
        return sp_strdup("");
    }
    while (isspace((unsigned char)*cursor)) {
        ++cursor;
    }
    if (*cursor != '"') {
        return sp_strdup("");
    }
    ++cursor;
    out = (char *)malloc(capacity);
    if (out == NULL) {
        return NULL;
    }
    while (*cursor != '\0' && *cursor != '"') {
        char ch = *cursor++;
        if (ch == '\\' && *cursor != '\0') {
            ch = *cursor++;
        }
        if (used + 1U >= capacity) {
            char *next;
            capacity *= 2U;
            next = (char *)realloc(out, capacity);
            if (next == NULL) {
                free(out);
                return NULL;
            }
            out = next;
        }
        out[used++] = ch;
    }
    out[used] = '\0';
    return out;
}

static size_t sp_maps_for(const char *source_op, const SPOpMap **maps, size_t cap) {
    size_t count = 0U;
    size_t i;
    for (i = 0U; i < sizeof(SP_OP_MAPS) / sizeof(SP_OP_MAPS[0]); ++i) {
        if (strcmp(SP_OP_MAPS[i].source_op, source_op) == 0) {
            if (count < cap) {
                maps[count] = &SP_OP_MAPS[i];
            }
            ++count;
        }
    }
    return count;
}

static int sp_group_reserve(SPGenerativeGroups *groups, size_t needed) {
    SPGenerativeGroup *items;
    size_t capacity;
    if (needed <= groups->capacity) {
        return 0;
    }
    capacity = groups->capacity == 0U ? 64U : groups->capacity * 2U;
    while (capacity < needed) {
        capacity *= 2U;
    }
    items = (SPGenerativeGroup *)realloc(groups->items,
                                         capacity * sizeof(*groups->items));
    if (items == NULL) {
        return -1;
    }
    groups->items = items;
    groups->capacity = capacity;
    return 0;
}

static int sp_group_add(SPGenerativeGroups *groups,
                        const SPOpMap *map,
                        const char *scope,
                        long input_width,
                        long input_height,
                        long output_width,
                        long output_height,
                        long scale_x,
                        long scale_y,
                        long input_components,
                        long output_components,
                        int accepted,
                        const char *failure_stage,
                        double verifier_score) {
    size_t i;
    SPGenerativeGroup *group = NULL;
    for (i = 0U; i < groups->count; ++i) {
        SPGenerativeGroup *candidate = &groups->items[i];
        if (candidate->op_id == map->op_id &&
            strcmp(candidate->scope, scope) == 0 &&
            candidate->input_width == input_width &&
            candidate->input_height == input_height &&
            candidate->output_width == output_width &&
            candidate->output_height == output_height &&
            candidate->scale_x == scale_x &&
            candidate->scale_y == scale_y &&
            candidate->input_components == input_components &&
            candidate->output_components == output_components) {
            group = candidate;
            break;
        }
    }
    if (group == NULL) {
        if (sp_group_reserve(groups, groups->count + 1U) != 0) {
            return -1;
        }
        group = &groups->items[groups->count++];
        memset(group, 0, sizeof(*group));
        group->op_id = map->op_id;
        group->op_name = map->op_name;
        snprintf(group->scope, sizeof(group->scope), "%s", scope);
        group->input_width = input_width;
        group->input_height = input_height;
        group->output_width = output_width;
        group->output_height = output_height;
        group->scale_x = scale_x;
        group->scale_y = scale_y;
        group->input_components = input_components;
        group->output_components = output_components;
    }
    if (accepted) {
        ++group->accepted;
    } else {
        ++group->rejected;
        if (strcmp(failure_stage, "transform_or_color") == 0 ||
            strcmp(failure_stage, "renderer_cells") == 0 ||
            strcmp(failure_stage, "verifier_distance") == 0) {
            ++group->near_miss;
        } else {
            ++group->hard_reject;
        }
    }
    group->verifier_score_sum += verifier_score;
    return 0;
}

static const char *sp_slot_type(const char *slot_name) {
    if (strcmp(slot_name, "__template__") == 0) {
        return "template";
    }
    if (strcmp(slot_name, "sketch.background") == 0 ||
        strcmp(slot_name, "sketch.color") == 0 ||
        strcmp(slot_name, "sketch.output_color") == 0) {
        return "color";
    }
    if (strcmp(slot_name, "sketch.x_offset") == 0 ||
        strcmp(slot_name, "sketch.y_offset") == 0) {
        return "offset";
    }
    return "integer";
}

static int sp_write_prior_row(FILE *fp,
                              const char *schema,
                              const char *library_id,
                              const char *slot_name,
                              long slot_value,
                              long proposal_count,
                              long certified_count,
                              long support_tasks,
                              long covered_events,
                              double utility) {
    return fprintf(fp,
                   "%s\t1036\tarc_sketch_program\t%s\t%s\t%s\t%.8f\t%ld\t%ld\t%ld\t%ld\t%ld\t%.8f\n",
                   schema,
                   library_id,
                   slot_name,
                   sp_slot_type(slot_name),
                   (double)slot_value,
                   slot_value,
                   proposal_count,
                   certified_count,
                   support_tasks,
                   covered_events,
                   utility) < 0
               ? -1
               : 0;
}

static int sp_write_slot_rows(FILE *prior,
                              char **fields,
                              size_t field_count,
                              const char *library_id,
                              int op_id) {
    const char *names[] = {
        "sketch.selector", "sketch.op", "sketch.background",
        "sketch.color",    "sketch.output_color", "sketch.x_anchor",
        "sketch.y_anchor", "sketch.x_offset", "sketch.y_offset",
        "sketch.width",    "sketch.height", "sketch.period",
    };
    long values[] = {
        sp_long(sp_field(fields, field_count, 23U)),
        (long)op_id,
        sp_long(sp_field(fields, field_count, 24U)),
        sp_long(sp_field(fields, field_count, 25U)),
        sp_long(sp_field(fields, field_count, 26U)),
        sp_long(sp_field(fields, field_count, 33U)),
        sp_long(sp_field(fields, field_count, 34U)),
        sp_long(sp_field(fields, field_count, 35U)),
        sp_long(sp_field(fields, field_count, 36U)),
        sp_long(sp_field(fields, field_count, 37U)),
        sp_long(sp_field(fields, field_count, 38U)),
        sp_long(sp_field(fields, field_count, 39U)),
    };
    long proposal_count = sp_long(sp_field(fields, field_count, 20U));
    long support_tasks = sp_long(sp_field(fields, field_count, 21U));
    double utility = sp_double(sp_field(fields, field_count, 22U));
    size_t i;
    if (proposal_count <= 0L) {
        proposal_count = 64L;
    }
    if (support_tasks <= 0L) {
        support_tasks = 1L;
    }
    if (utility <= 0.0) {
        utility = 1.0;
    }
    for (i = 0U; i < sizeof(names) / sizeof(names[0]); ++i) {
        if (sp_write_prior_row(prior,
                               "crlp.hypothesis_library.prior.v1",
                               library_id,
                               names[i],
                               values[i],
                               proposal_count,
                               1L,
                               support_tasks,
                               1L,
                               utility) != 0) {
            return -1;
        }
    }
    return 0;
}

static int sp_add_attempt_groups(SPGenerativeGroups *groups,
                                 char **fields,
                                 size_t field_count,
                                 const SPOpMap *map,
                                 int accepted) {
    long input_width = sp_long(sp_field(fields, field_count, 44U));
    long input_height = sp_long(sp_field(fields, field_count, 45U));
    long output_width = sp_long(sp_field(fields, field_count, 46U));
    long output_height = sp_long(sp_field(fields, field_count, 47U));
    long scale_x = sp_long(sp_field(fields, field_count, 48U));
    long scale_y = sp_long(sp_field(fields, field_count, 49U));
    long input_components = sp_long(sp_field(fields, field_count, 68U));
    long output_components = sp_long(sp_field(fields, field_count, 80U));
    double verifier_score = sp_double(sp_field(fields, field_count, 67U));
    const char *stage = sp_field(fields, field_count, 18U);
    return sp_group_add(groups, map, "global", 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, accepted, stage, verifier_score) != 0 ||
                   sp_group_add(groups, map, "shape", input_width, input_height, output_width, output_height, 0L, 0L, 0L, 0L, accepted, stage, verifier_score) != 0 ||
                   sp_group_add(groups, map, "scale", 0L, 0L, 0L, 0L, scale_x, scale_y, 0L, 0L, accepted, stage, verifier_score) != 0 ||
                   sp_group_add(groups, map, "components", 0L, 0L, 0L, 0L, 0L, 0L, input_components, output_components, accepted, stage, verifier_score) != 0
               ? -1
               : 0;
}

static int sp_export_attempts(const char *attempts_path,
                              const char *prior_path,
                              const char *prior_summary_path,
                              const char *generative_prior_path,
                              const char *generative_summary_path,
                              SPExportStats *stats) {
    FILE *attempts = NULL;
    FILE *prior = NULL;
    FILE *gen = NULL;
    FILE *summary = NULL;
    FILE *gen_summary = NULL;
    SPGenerativeGroups groups = {0};
    char line[SP_LINE_MAX];
    int status = -1;
    if (sp_mkdir_parent(prior_path) != 0 ||
        sp_mkdir_parent(generative_prior_path) != 0) {
        goto done;
    }
    attempts = fopen(attempts_path, "r");
    prior = fopen(prior_path, "w");
    if (attempts == NULL || prior == NULL) {
        goto done;
    }
    fputs("schema_version\ttemplate_id\ttemplate_name\tlibrary_id\tslot_name\t"
          "slot_type\tslot_value\tslot_integer\tproposal_count\tcertified_count\t"
          "support_tasks\tcovered_events\tutility_mean\n",
          prior);
    while (fgets(line, sizeof(line), attempts) != NULL) {
        char *fields[SP_MAX_FIELDS];
        size_t field_count = sp_split_tabs(line, fields, SP_MAX_FIELDS);
        const char *split_name = sp_field(fields, field_count, 0U);
        const char *op_name = sp_field(fields, field_count, 8U);
        int accepted = sp_bool_text(sp_field(fields, field_count, 16U));
        const SPOpMap *maps[8];
        size_t map_count;
        size_t map_i;
        if (strcmp(split_name, "split_name") == 0 || split_name[0] == '\0') {
            continue;
        }
        ++stats->input_rows;
        if (accepted) {
            ++stats->accepted_rows;
        } else {
            ++stats->rejected_rows;
            ++stats->skipped_unaccepted_rows;
            if (sp_counter_add(&stats->rejection_stages,
                               sp_field(fields, field_count, 18U)) != 0) {
                goto done;
            }
        }
        if (sp_counter_add(&stats->source_ops, op_name) != 0) {
            goto done;
        }
        map_count = sp_maps_for(op_name, maps, sizeof(maps) / sizeof(maps[0]));
        if (map_count == 0U) {
            if (accepted) {
                ++stats->skipped_unmapped_accepted_rows;
            }
            if (sp_counter_add(&stats->unmapped_ops, op_name) != 0) {
                goto done;
            }
            continue;
        }
        for (map_i = 0U; map_i < map_count && map_i < sizeof(maps) / sizeof(maps[0]);
             ++map_i) {
            if (sp_counter_add(&stats->runtime_ops, maps[map_i]->op_name) != 0 ||
                sp_add_attempt_groups(&groups,
                                      fields,
                                      field_count,
                                      maps[map_i],
                                      accepted) != 0) {
                goto done;
            }
        }
        if (!accepted) {
            continue;
        }
        if (sp_double(sp_field(fields, field_count, 67U)) < 0.999) {
            ++stats->skipped_unverified_accepted_rows;
            continue;
        }
        ++stats->mapped_accepted_rows;
        if (sp_counter_add(&stats->mapped_ops, op_name) != 0 ||
            sp_string_add_unique(&stats->synthetic_hashes,
                                 field_count > 85U
                                     ? sp_field(fields, field_count, 85U)
                                     : sp_field(fields, field_count, 4U)) != 0) {
            goto done;
        }
        for (map_i = 0U; map_i < map_count && map_i < sizeof(maps) / sizeof(maps[0]);
             ++map_i) {
            char library_id[256];
            long proposal_count = sp_long(sp_field(fields, field_count, 20U));
            long support_tasks = sp_long(sp_field(fields, field_count, 21U));
            double utility = sp_double(sp_field(fields, field_count, 22U));
            if (proposal_count <= 0L) {
                proposal_count = 64L;
            }
            if (support_tasks <= 0L) {
                support_tasks = 1L;
            }
            if (utility <= 0.0) {
                utility = 1.0;
            }
            snprintf(library_id,
                     sizeof(library_id),
                     "%s:op%d",
                     sp_field(fields, field_count, 4U),
                     maps[map_i]->op_id);
            if (sp_write_prior_row(prior,
                                   "crlp.hypothesis_library.template_prior.v1",
                                   library_id,
                                   "__template__",
                                   0L,
                                   proposal_count,
                                   1L,
                                   support_tasks,
                                   1L,
                                   utility) != 0 ||
                sp_write_slot_rows(prior,
                                   fields,
                                   field_count,
                                   library_id,
                                   maps[map_i]->op_id) != 0) {
                goto done;
            }
            ++stats->template_rows;
            stats->slot_rows += 12U;
        }
    }
    if (ferror(attempts) || fclose(prior) != 0) {
        prior = NULL;
        goto done;
    }
    prior = NULL;
    gen = fopen(generative_prior_path, "w");
    if (gen == NULL) {
        goto done;
    }
    fputs("schema_version\top_id\top_name\tmatch_scope\tinput_width\tinput_height\t"
          "output_width\toutput_height\tshape_scale_x\tshape_scale_y\t"
          "input_component_count\toutput_component_count\taccepted_count\t"
          "rejected_count\tnear_miss_rejected_count\thard_rejected_count\t"
          "mean_verifier_score\trejection_penalty\tscore\n",
          gen);
    for (size_t i = 0U; i < groups.count; ++i) {
        SPGenerativeGroup *g = &groups.items[i];
        size_t total = g->accepted + g->rejected;
        double mean = total == 0U ? 0.0 : g->verifier_score_sum / (double)total;
        double penalty = 1.5 * (double)g->hard_reject + 0.3 * (double)g->near_miss;
        double score =
            ((double)g->accepted + 0.25 * (double)g->near_miss + 1.0) /
            ((double)g->accepted + penalty + 2.0);
        if (fprintf(gen,
                    "crlp.arc_generative_prior.v1\t%d\t%s\t%s\t%ld\t%ld\t%ld\t%ld\t%ld\t%ld\t%ld\t%ld\t%zu\t%zu\t%zu\t%zu\t%.8f\t%.8f\t%.8f\n",
                    g->op_id,
                    g->op_name,
                    g->scope,
                    g->input_width,
                    g->input_height,
                    g->output_width,
                    g->output_height,
                    g->scale_x,
                    g->scale_y,
                    g->input_components,
                    g->output_components,
                    g->accepted,
                    g->rejected,
                    g->near_miss,
                    g->hard_reject,
                    mean,
                    penalty,
                    score) < 0) {
            goto done;
        }
        ++stats->generative_rows;
    }
    if (fclose(gen) != 0) {
        gen = NULL;
        goto done;
    }
    gen = NULL;
    stats->prior_rows = stats->template_rows + stats->slot_rows;
    if (sp_mkdir_parent(prior_summary_path) != 0 ||
        sp_mkdir_parent(generative_summary_path) != 0) {
        goto done;
    }
    summary = fopen(prior_summary_path, "w");
    gen_summary = fopen(generative_summary_path, "w");
    if (summary == NULL || gen_summary == NULL) {
        goto done;
    }
    fputs("{\"schema\":\"crlp.arc_verified_self_play.prior.v1\","
          "\"report_version\":2,\"attempts\":",
          summary);
    sp_json_string(summary, attempts_path);
    fputs(",\"prior\":", summary);
    sp_json_string(summary, prior_path);
    fprintf(summary,
            ",\"input_attempt_rows\":%zu,\"accepted_attempt_rows\":%zu,"
            "\"mapped_accepted_attempt_rows\":%zu,"
            "\"skipped_unaccepted_rows\":%zu,"
            "\"skipped_unmapped_accepted_rows\":%zu,"
            "\"skipped_unverified_accepted_rows\":%zu,"
            "\"template_prior_rows\":%zu,\"slot_prior_rows\":%zu,"
            "\"prior_rows\":%zu,\"op_id_contract\":\"ARC_SKETCH_OP ids from examples/arc_agi.c\","
            "\"mapped_ops\":",
            stats->input_rows,
            stats->accepted_rows,
            stats->mapped_accepted_rows,
            stats->skipped_unaccepted_rows,
            stats->skipped_unmapped_accepted_rows,
            stats->skipped_unverified_accepted_rows,
            stats->template_rows,
            stats->slot_rows,
            stats->prior_rows);
    sp_counter_write(summary, &stats->mapped_ops);
    fputs(",\"unmapped_ops\":", summary);
    sp_counter_write(summary, &stats->unmapped_ops);
    fputs("}\n", summary);
    fputs("{\"schema\":\"crlp.arc_generative_prior.summary.v1\","
          "\"report_version\":2,\"claim_scope\":\"rejection-aware task-conditioned runtime miner scheduling only; candidates still require train-pair verifier acceptance\","
          "\"attempts\":",
          gen_summary);
    sp_json_string(gen_summary, attempts_path);
    fputs(",\"generative_prior\":", gen_summary);
    sp_json_string(gen_summary, generative_prior_path);
    fprintf(gen_summary,
            ",\"input_attempt_rows\":%zu,\"accepted_attempt_rows\":%zu,"
            "\"rejected_attempt_rows\":%zu,\"generative_prior_rows\":%zu,"
            "\"op_id_contract\":\"ARC_SKETCH_OP ids from examples/arc_agi.c\","
            "\"source_ops\":",
            stats->input_rows,
            stats->accepted_rows,
            stats->rejected_rows,
            stats->generative_rows);
    sp_counter_write(gen_summary, &stats->source_ops);
    fputs(",\"mapped_runtime_ops\":", gen_summary);
    sp_counter_write(gen_summary, &stats->runtime_ops);
    fputs(",\"unmapped_ops\":", gen_summary);
    sp_counter_write(gen_summary, &stats->unmapped_ops);
    fputs(",\"rejection_failure_stages\":", gen_summary);
    sp_counter_write(gen_summary, &stats->rejection_stages);
    fputs(",\"rejection_training\":{\"near_miss_failure_stages\":[\"renderer_cells\",\"transform_or_color\",\"verifier_distance\"],"
          "\"hard_rejection_failure_stages\":[\"compiler\",\"normalizer_shape\",\"renderer_shape\",\"selector_or_transform\",\"unsupported_shape\"],"
          "\"score_formula\":\"(accepted + 0.25 * near_miss + 1) / (accepted + weighted_rejections + 2)\"}}\n",
          gen_summary);
    if (fclose(summary) != 0 || fclose(gen_summary) != 0) {
        summary = NULL;
        gen_summary = NULL;
        goto done;
    }
    summary = NULL;
    gen_summary = NULL;
    status = 0;
done:
    if (attempts != NULL) {
        fclose(attempts);
    }
    if (prior != NULL) {
        fclose(prior);
    }
    if (gen != NULL) {
        fclose(gen);
    }
    if (summary != NULL) {
        fclose(summary);
    }
    if (gen_summary != NULL) {
        fclose(gen_summary);
    }
    free(groups.items);
    return status;
}

static int sp_read_log(const char *path, SPLogStats *stats) {
    FILE *fp;
    char line[SP_LINE_MAX];
    if (path == NULL || path[0] == '\0') {
        return 0;
    }
    snprintf(stats->path, sizeof(stats->path), "%s", path);
    fp = fopen(path, "r");
    if (fp == NULL) {
        return 0;
    }
    while (fgets(line, sizeof(line), fp) != NULL) {
        if (strstr(line, "\"schema\":\"crlp.arc_loader.summary.v1\"") != NULL) {
            stats->available = 1;
            stats->loaded_tasks = (long)sp_json_double(line, "loaded_tasks");
            stats->certified_tasks = (long)sp_json_double(line, "certified_tasks");
            stats->test_exact_tasks = (long)sp_json_double(line, "test_exact_tasks");
            stats->test_cells = (long)sp_json_double(line, "test_cells");
            stats->test_correct_cells = (long)sp_json_double(line, "test_correct_cells");
            stats->cell_accuracy = sp_json_double(line, "cell_accuracy");
        } else if (strstr(line, "\"schema\":\"crlp.arc_hypothesis.selection.v1\"") != NULL) {
            ++stats->selection_rows;
            stats->runtime_mined_sketches +=
                (long)sp_json_double(line, "runtime_mined_sketches");
            stats->runtime_generative_prior_rows =
                (long)sp_json_double(line, "runtime_generative_prior_rows");
            stats->runtime_generative_prior_ops +=
                (long)sp_json_double(line, "runtime_generative_prior_ops");
            stats->runtime_generative_prior_mined_sketches +=
                (long)sp_json_double(line, "runtime_generative_prior_mined_sketches");
        } else if (strstr(line, "\"schema\":\"crlp.arc_generative_prior.runtime_trace.v1\"") != NULL) {
            char *decision = sp_json_string_dup(line, "decision");
            ++stats->trace_rows;
            stats->trace_emitted_sketches +=
                (long)sp_json_double(line, "emitted_sketches");
            if (decision != NULL) {
                if (sp_counter_add(&stats->trace_decisions, decision) != 0) {
                    free(decision);
                    fclose(fp);
                    return -1;
                }
                free(decision);
            }
        }
        if (strstr(line, "\"state_hash\"") != NULL) {
            char *hash = sp_json_string_dup(line, "state_hash");
            if (hash != NULL) {
                if (sp_string_add_unique(&stats->state_hashes, hash) != 0) {
                    free(hash);
                    fclose(fp);
                    return -1;
                }
                free(hash);
            }
        }
        if (strstr(line, "\"task_instance\"") != NULL) {
            char *task_instance = sp_json_string_dup(line, "task_instance");
            if (task_instance != NULL) {
                if (sp_string_add_unique(&stats->state_hashes, task_instance) != 0) {
                    free(task_instance);
                    fclose(fp);
                    return -1;
                }
                free(task_instance);
            }
        }
    }
    if (ferror(fp)) {
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
}

static void sp_log_free(SPLogStats *stats) {
    sp_counter_free(&stats->trace_decisions);
    sp_string_list_free(&stats->state_hashes);
}

static int sp_read_synthetic_summary(const char *path, SPSyntheticSummary *summary) {
    FILE *fp;
    char line[SP_LINE_MAX];
    if (path == NULL || path[0] == '\0') {
        return 0;
    }
    fp = fopen(path, "r");
    if (fp == NULL) {
        return 0;
    }
    if (fgets(line, sizeof(line), fp) != NULL) {
        summary->synthetic_rows = (long)sp_json_double(line, "synthetic_rows");
        summary->accepted_rows = (long)sp_json_double(line, "synthetic_accepted_rows");
        summary->rejected_rows = (long)sp_json_double(line, "synthetic_rejected_rows");
        summary->max_accepted = (long)sp_json_double(line, "max_accepted");
        summary->max_grid_size = (long)sp_json_double(line, "max_grid_size");
        summary->elapsed_seconds = sp_json_double(line, "elapsed_seconds");
        summary->stop_reason = sp_json_string_dup(line, "stop_reason");
    }
    fclose(fp);
    return 0;
}

static int sp_read_ranker_summary(const char *path, SPRankerSummary *summary) {
    FILE *fp;
    char line[SP_LINE_MAX];
    if (path == NULL || path[0] == '\0') {
        return 0;
    }
    fp = fopen(path, "r");
    if (fp == NULL) {
        return 0;
    }
    if (fgets(line, sizeof(line), fp) != NULL) {
        summary->source_attempt_count =
            (long)sp_json_double(line, "source_attempt_count");
        summary->source_accepted_count =
            (long)sp_json_double(line, "source_accepted_count");
        summary->target_task_count = (long)sp_json_double(line, "target_task_count");
        summary->accepted_target_task_count =
            (long)sp_json_double(line, "accepted_target_task_count");
        summary->proposal_count = (long)sp_json_double(line, "proposal_count");
        summary->top1_all_task_rate = sp_json_double(line, "top1_all_task_rate");
        summary->retained_all_task_rate = sp_json_double(line, "retained_all_task_rate");
    }
    fclose(fp);
    return 0;
}

static int sp_hashes_disjoint(const SPStringList *lhs, const SPStringList *rhs) {
    size_t i;
    if (lhs->count == 0U || rhs->count == 0U) {
        return 0;
    }
    for (i = 0U; i < lhs->count; ++i) {
        if (sp_string_contains(rhs, lhs->items[i])) {
            return 0;
        }
    }
    return 1;
}

static void sp_write_loader(FILE *fp, const SPLogStats *stats) {
    fputs("{\"available\":", fp);
    fputs(stats->available ? "true" : "false", fp);
    fputs(",\"summary\":", fp);
    sp_json_string(fp, stats->path);
    fprintf(fp,
            ",\"loaded_tasks\":%ld,\"certified_tasks\":",
            stats->loaded_tasks);
    if (stats->available) {
        fprintf(fp, "%ld", stats->certified_tasks);
    } else {
        fputs("null", fp);
    }
    fprintf(fp,
            ",\"test_exact_tasks\":%ld,\"certified_cells\":%ld,"
            "\"test_cells\":%ld,\"cell_accuracy\":%.8f}",
            stats->test_exact_tasks,
            stats->test_correct_cells,
            stats->test_cells,
            stats->cell_accuracy);
}

static void sp_write_runtime(FILE *fp, const SPLogStats *stats) {
    fprintf(fp,
            "{\"available\":%s,\"task_rows\":%ld,"
            "\"runtime_mined_sketches\":%ld,"
            "\"runtime_generative_prior_rows\":%ld,"
            "\"runtime_generative_prior_ops\":%ld,"
            "\"runtime_generative_prior_mined_sketches\":%ld}",
            stats->selection_rows > 0L ? "true" : "false",
            stats->selection_rows,
            stats->runtime_mined_sketches,
            stats->runtime_generative_prior_rows,
            stats->runtime_generative_prior_ops,
            stats->runtime_generative_prior_mined_sketches);
}

static void sp_write_trace(FILE *fp, SPLogStats *stats) {
    fprintf(fp,
            "{\"available\":%s,\"trace_rows\":%ld,"
            "\"emitted_sketches\":%ld,\"decision_counts\":",
            stats->trace_rows > 0L ? "true" : "false",
            stats->trace_rows,
            stats->trace_emitted_sketches);
    sp_counter_write(fp, &stats->trace_decisions);
    fputc('}', fp);
}

static int sp_write_summary(const char *summary_path,
                            const char *markdown_path,
                            const char *synthetic_summary_path,
                            const char *ranker_summary_path,
                            const char *prior_path,
                            const char *prior_summary_path,
                            const char *generative_prior_path,
                            const char *generative_summary_path,
                            const SPSyntheticSummary *synthetic,
                            const SPRankerSummary *ranker,
                            SPExportStats *export_stats,
                            SPLogStats *train_baseline,
                            SPLogStats *eval_baseline,
                            SPLogStats *train_visual,
                            SPLogStats *eval_visual,
                            SPLogStats *train_self,
                            SPLogStats *eval_self,
                            long template_baseline,
                            long oracle_ceiling,
                            long seed_count) {
    FILE *fp;
    FILE *md;
    long self_eval_cert = eval_self->available ? eval_self->certified_tasks : -1L;
    long full_delta = eval_self->available && eval_baseline->available
                          ? eval_self->certified_tasks - eval_baseline->certified_tasks
                          : 0L;
    long visual_delta = eval_self->available && eval_visual->available
                            ? eval_self->certified_tasks - eval_visual->certified_tasks
                            : 0L;
    int public_eval = eval_self->available;
    int fixed_budget = eval_self->available && eval_baseline->available &&
                       eval_visual->available;
    int eval_hashes = eval_self->state_hashes.count > 0U;
    int leakage = eval_hashes &&
                  sp_hashes_disjoint(&export_stats->synthetic_hashes,
                                     &eval_self->state_hashes);
    long opened_zero_families = 0L;
    const char *claim_status =
        self_eval_cert >= template_baseline && opened_zero_families >= 1L &&
                seed_count >= 3L && public_eval && fixed_budget && leakage
            ? "supported"
            : "underpowered";
    if (sp_mkdir_parent(summary_path) != 0 || sp_mkdir_parent(markdown_path) != 0) {
        return -1;
    }
    fp = fopen(summary_path, "w");
    if (fp == NULL) {
        return -1;
    }
    fputs("{\"schema\":\"crlp.arc_verified_self_play.summary.v1\","
          "\"report_version\":7,\"claim_id\":\"C5.arc_verified_self_play\","
          "\"training_loop\":\"verified_self_play_inverse_program_induction\","
          "\"supervision_source\":\"forward primitive composition with exact verifier-compatible labels\","
          "\"authored_task_library\":false,"
          "\"claim_scope\":\"self-play rows provide exact supervision for inverse proposer training; ARC competence is only measured after replay through the public ARC verifier\",",
          fp);
    fprintf(fp,
            "\"claim_certified_eval_tasks\":%ld,"
            "\"claim_template_baseline\":%ld,"
            "\"claim_oracle_ceiling\":%ld,"
            "\"claim_opened_zero_coverage_families\":%ld,"
            "\"claim_seed_count\":%ld,\"claim_minimum_seed_count\":3,"
            "\"claim_public_eval_replay_available\":%s,"
            "\"claim_fixed_budget_ablation_available\":%s,"
            "\"claim_leakage_gate_passed\":%s,"
            "\"claim_status\":",
            self_eval_cert,
            template_baseline,
            oracle_ceiling,
            opened_zero_families,
            seed_count,
            public_eval ? "true" : "false",
            fixed_budget ? "true" : "false",
            leakage ? "true" : "false");
    sp_json_string(fp, claim_status);
    fprintf(fp,
            ",\"synthesis_budget\":{\"max_accepted\":%ld,\"max_grid_size\":%ld,"
            "\"elapsed_seconds\":%.3f,\"stop_reason\":",
            synthetic->max_accepted,
            synthetic->max_grid_size,
            synthetic->elapsed_seconds);
    sp_json_string(fp, synthetic->stop_reason == NULL ? "" : synthetic->stop_reason);
    fprintf(fp,
            "},\"synthetic_rows\":%ld,\"synthetic_accepted_rows\":%ld,"
            "\"synthetic_rejected_rows\":%ld,\"curriculum_summary\":",
            synthetic->synthetic_rows,
            synthetic->accepted_rows,
            synthetic->rejected_rows);
    sp_json_string(fp, synthetic_summary_path);
    fputs(",\"ranker_summary\":", fp);
    sp_json_string(fp, ranker_summary_path);
    fprintf(fp,
            ",\"inverse_ranker\":{\"source_attempt_count\":%ld,"
            "\"source_accepted_count\":%ld,\"proposal_count\":%ld,"
            "\"target_task_count\":%ld,\"accepted_target_task_count\":%ld,"
            "\"top1_all_task_rate\":%.8f,\"retained_all_task_rate\":%.8f},",
            ranker->source_attempt_count,
            ranker->source_accepted_count,
            ranker->proposal_count,
            ranker->target_task_count,
            ranker->accepted_target_task_count,
            ranker->top1_all_task_rate,
            ranker->retained_all_task_rate);
    fputs("\"self_play_prior\":", fp);
    sp_json_string(fp, prior_path);
    fputs(",\"prior_summary\":", fp);
    sp_json_string(fp, prior_summary_path);
    fprintf(fp,
            ",\"self_play_prior_export\":{\"available\":true,"
            "\"mapped_accepted_attempt_rows\":%zu,\"prior_rows\":%zu,"
            "\"template_prior_rows\":%zu,\"slot_prior_rows\":%zu,"
            "\"mapped_ops\":",
            export_stats->mapped_accepted_rows,
            export_stats->prior_rows,
            export_stats->template_rows,
            export_stats->slot_rows);
    sp_counter_write(fp, &export_stats->mapped_ops);
    fputs(",\"unmapped_ops\":", fp);
    sp_counter_write(fp, &export_stats->unmapped_ops);
    fputs("},\"generative_prior\":", fp);
    sp_json_string(fp, generative_prior_path);
    fputs(",\"generative_prior_summary\":", fp);
    sp_json_string(fp, generative_summary_path);
    fprintf(fp,
            ",\"generative_prior_export\":{\"available\":true,"
            "\"generative_prior_rows\":%zu,\"source_ops\":",
            export_stats->generative_rows);
    sp_counter_write(fp, &export_stats->source_ops);
    fputs(",\"mapped_runtime_ops\":", fp);
    sp_counter_write(fp, &export_stats->runtime_ops);
    fputs(",\"unmapped_ops\":", fp);
    sp_counter_write(fp, &export_stats->unmapped_ops);
    fputs("},\"public_train_baseline\":", fp);
    sp_write_loader(fp, train_baseline);
    fputs(",\"public_train_visual_baseline\":", fp);
    sp_write_loader(fp, train_visual);
    fputs(",\"public_train\":", fp);
    sp_write_loader(fp, train_self);
    fputs(",\"public_eval_baseline\":", fp);
    sp_write_loader(fp, eval_baseline);
    fputs(",\"public_eval_visual_baseline\":", fp);
    sp_write_loader(fp, eval_visual);
    fputs(",\"public_eval\":", fp);
    sp_write_loader(fp, eval_self);
    fputs(",\"public_train_runtime\":", fp);
    sp_write_runtime(fp, train_self);
    fputs(",\"public_eval_runtime\":", fp);
    sp_write_runtime(fp, eval_self);
    fputs(",\"public_train_generative_prior_trace\":", fp);
    sp_write_trace(fp, train_self);
    fputs(",\"public_eval_generative_prior_trace\":", fp);
    sp_write_trace(fp, eval_self);
    fprintf(fp,
            ",\"public_replay\":{\"available\":%s,\"baseline_available\":%s,"
            "\"visual_baseline_available\":%s,"
            "\"matched_baseline_kind\":\"visual_only_no_prior\","
            "\"eval_full_baseline_certified_tasks\":",
            public_eval ? "true" : "false",
            eval_baseline->available ? "true" : "false",
            eval_visual->available ? "true" : "false");
    if (eval_baseline->available) {
        fprintf(fp, "%ld", eval_baseline->certified_tasks);
    } else {
        fputs("null", fp);
    }
    fputs(",\"eval_visual_baseline_certified_tasks\":", fp);
    if (eval_visual->available) {
        fprintf(fp, "%ld", eval_visual->certified_tasks);
    } else {
        fputs("null", fp);
    }
    fputs(",\"eval_self_play_certified_tasks\":", fp);
    if (eval_self->available) {
        fprintf(fp, "%ld", eval_self->certified_tasks);
    } else {
        fputs("null", fp);
    }
    fprintf(fp,
            ",\"delta_vs_full_baseline_certified_tasks\":%ld,"
            "\"delta_vs_visual_baseline_certified_tasks\":%ld,"
            "\"delta_vs_template_baseline_certified_tasks\":%ld,"
            "\"gap_to_oracle_ceiling_certified_tasks\":%ld,"
            "\"coverage_regressed_vs_baseline\":%s,"
            "\"promotion_status\":",
            full_delta,
            visual_delta,
            self_eval_cert < 0L ? 0L : self_eval_cert - template_baseline,
            self_eval_cert < 0L ? oracle_ceiling : oracle_ceiling - self_eval_cert,
            eval_visual->available && eval_self->available &&
                    eval_self->certified_tasks < eval_visual->certified_tasks
                ? "true"
                : "false");
    sp_json_string(fp,
                   public_eval ? "diagnostic_public_replay_under_claim_gates"
                               : "diagnostic_only_no_public_replay");
    fprintf(fp,
            "},\"leakage_audit\":{\"available\":true,"
            "\"state_hash_contract_ready\":%s,\"accepted_rows\":%ld,"
            "\"synthetic_rows\":%ld,\"unique_state_hashes\":%zu,"
            "\"public_eval_identifier_source\":\"state_hash_or_task_instance\","
            "\"public_eval_precondition_satisfied\":%s,"
            "\"eval_hashes_available\":%s,\"eval_state_hash_count\":%zu,"
            "\"disjoint_from_public_eval\":",
            export_stats->synthetic_hashes.count > 0U ? "true" : "false",
            synthetic->accepted_rows,
            synthetic->synthetic_rows,
            export_stats->synthetic_hashes.count,
            public_eval ? "true" : "false",
            eval_hashes ? "true" : "false",
            eval_self->state_hashes.count);
    if (eval_hashes) {
        fputs(leakage ? "true" : "false", fp);
    } else {
        fputs("null", fp);
    }
    fprintf(fp,
            "},\"yardsticks\":{\"template_baseline_certified_eval_tasks\":%ld,"
            "\"oracle_ceiling_certified_eval_tasks\":%ld,"
            "\"zero_coverage_families\":[\"object_motion_or_copy_extend\","
            "\"relational_layout_or_counting\",\"projection_or_panel_selection\","
            "\"color_remap_or_mode_collapse\",\"scale_tile_or_expand\","
            "\"unsupported_parse_or_shape\",\"shape_completion_or_fill\","
            "\"segmentation_or_object_extract\",\"symmetry_or_geometry\"],"
            "\"zero_coverage_family_count\":9}}\n",
            template_baseline,
            oracle_ceiling);
    if (fclose(fp) != 0) {
        return -1;
    }
    md = fopen(markdown_path, "w");
    if (md == NULL) {
        return -1;
    }
    fprintf(md,
            "# ARC Verified Self-Play\n\n"
            "- Synthetic rows: %ld accepted, %ld rejected (%ld total; stop=%s).\n"
            "- Public eval baseline: full=%ld certified, visual-only=%ld certified.\n"
            "- Public eval self-play: %ld certified (%+ld vs visual-only, %+ld vs full).\n"
            "- Claim gates: public replay=%s, fixed-budget ablation=%s, leakage=%s, seeds=%ld/3, template baseline=%ld, oracle=%ld.\n"
            "- Status: %s. This remains a search-efficiency diagnostic, not a new primitive-capability claim.\n",
            synthetic->accepted_rows,
            synthetic->rejected_rows,
            synthetic->synthetic_rows,
            synthetic->stop_reason == NULL ? "" : synthetic->stop_reason,
            eval_baseline->available ? eval_baseline->certified_tasks : -1L,
            eval_visual->available ? eval_visual->certified_tasks : -1L,
            self_eval_cert,
            visual_delta,
            full_delta,
            public_eval ? "yes" : "no",
            fixed_budget ? "yes" : "no",
            leakage ? "yes" : "no",
            seed_count,
            template_baseline,
            oracle_ceiling,
            claim_status);
    return fclose(md) == 0 ? 0 : -1;
}

int nn_grammar_discovery_write_arc_verified_self_play_report(
    const char *attempts_tsv_path,
    const char *synthetic_summary_path,
    const char *ranker_summary_path,
    const char *public_train_baseline_log,
    const char *public_eval_baseline_log,
    const char *public_train_visual_baseline_log,
    const char *public_eval_visual_baseline_log,
    const char *public_train_self_play_log,
    const char *public_eval_self_play_log,
    const char *prior_path,
    const char *prior_summary_path,
    const char *generative_prior_path,
    const char *generative_prior_summary_path,
    const char *summary_path,
    const char *markdown_path,
    long template_baseline,
    long oracle_ceiling,
    long seed_count) {
    SPExportStats export_stats;
    SPSyntheticSummary synthetic;
    SPRankerSummary ranker;
    SPLogStats train_baseline;
    SPLogStats eval_baseline;
    SPLogStats train_visual;
    SPLogStats eval_visual;
    SPLogStats train_self;
    SPLogStats eval_self;
    int status = -1;
    memset(&export_stats, 0, sizeof(export_stats));
    memset(&synthetic, 0, sizeof(synthetic));
    memset(&ranker, 0, sizeof(ranker));
    memset(&train_baseline, 0, sizeof(train_baseline));
    memset(&eval_baseline, 0, sizeof(eval_baseline));
    memset(&train_visual, 0, sizeof(train_visual));
    memset(&eval_visual, 0, sizeof(eval_visual));
    memset(&train_self, 0, sizeof(train_self));
    memset(&eval_self, 0, sizeof(eval_self));
    if (attempts_tsv_path == NULL || attempts_tsv_path[0] == '\0' ||
        prior_path == NULL || prior_path[0] == '\0' ||
        prior_summary_path == NULL || prior_summary_path[0] == '\0' ||
        generative_prior_path == NULL || generative_prior_path[0] == '\0' ||
        generative_prior_summary_path == NULL ||
        generative_prior_summary_path[0] == '\0' ||
        summary_path == NULL || summary_path[0] == '\0' ||
        markdown_path == NULL || markdown_path[0] == '\0') {
        goto done;
    }
    if (sp_export_attempts(attempts_tsv_path,
                           prior_path,
                           prior_summary_path,
                           generative_prior_path,
                           generative_prior_summary_path,
                           &export_stats) != 0 ||
        sp_read_synthetic_summary(synthetic_summary_path, &synthetic) != 0 ||
        sp_read_ranker_summary(ranker_summary_path, &ranker) != 0 ||
        sp_read_log(public_train_baseline_log, &train_baseline) != 0 ||
        sp_read_log(public_eval_baseline_log, &eval_baseline) != 0 ||
        sp_read_log(public_train_visual_baseline_log, &train_visual) != 0 ||
        sp_read_log(public_eval_visual_baseline_log, &eval_visual) != 0 ||
        sp_read_log(public_train_self_play_log, &train_self) != 0 ||
        sp_read_log(public_eval_self_play_log, &eval_self) != 0) {
        goto done;
    }
    if (synthetic.synthetic_rows == 0L) {
        synthetic.synthetic_rows = (long)export_stats.input_rows;
        synthetic.accepted_rows = (long)export_stats.accepted_rows;
        synthetic.rejected_rows = (long)export_stats.rejected_rows;
    }
    if (sp_write_summary(summary_path,
                         markdown_path,
                         synthetic_summary_path,
                         ranker_summary_path,
                         prior_path,
                         prior_summary_path,
                         generative_prior_path,
                         generative_prior_summary_path,
                         &synthetic,
                         &ranker,
                         &export_stats,
                         &train_baseline,
                         &eval_baseline,
                         &train_visual,
                         &eval_visual,
                         &train_self,
                         &eval_self,
                         template_baseline,
                         oracle_ceiling,
                         seed_count) != 0) {
        goto done;
    }
    printf("arc-verified-self-play summary=%s eval_self_play=%ld eval_visual=%ld prior_rows=%zu generative_rows=%zu\n",
           summary_path,
           eval_self.available ? eval_self.certified_tasks : -1L,
           eval_visual.available ? eval_visual.certified_tasks : -1L,
           export_stats.prior_rows,
           export_stats.generative_rows);
    status = 0;
done:
    free(synthetic.stop_reason);
    sp_counter_free(&export_stats.mapped_ops);
    sp_counter_free(&export_stats.unmapped_ops);
    sp_counter_free(&export_stats.source_ops);
    sp_counter_free(&export_stats.runtime_ops);
    sp_counter_free(&export_stats.rejection_stages);
    sp_string_list_free(&export_stats.synthetic_hashes);
    sp_log_free(&train_baseline);
    sp_log_free(&eval_baseline);
    sp_log_free(&train_visual);
    sp_log_free(&eval_visual);
    sp_log_free(&train_self);
    sp_log_free(&eval_self);
    return status;
}

static void lso_copy(char *dst, size_t dst_size, const char *src) {
    if (dst == NULL || dst_size == 0U) {
        return;
    }
    snprintf(dst, dst_size, "%s", src == NULL ? "" : src);
}

static int lso_counter_add_nonempty(SPCounterList *list, const char *key) {
    if (key == NULL || key[0] == '\0') {
        return 0;
    }
    return sp_counter_add(list, key);
}

static void lso_today(char out[11]) {
    time_t now = time(NULL);
    struct tm *tm_value = localtime(&now);
    if (tm_value == NULL || strftime(out, 11U, "%Y-%m-%d", tm_value) == 0U) {
        strcpy(out, "1970-01-01");
    }
}

static unsigned long long lso_fnv1a48(const char *text) {
    unsigned long long hash = 1469598103934665603ULL;
    const unsigned char *p = (const unsigned char *)(text == NULL ? "" : text);
    while (*p != '\0') {
        hash ^= (unsigned long long)*p++;
        hash *= 1099511628211ULL;
    }
    return hash & 0xffffffffffffULL;
}

static const char *lso_known_operator_suffix(const char *operator_name) {
    struct KnownId {
        const char *name;
        const char *suffix;
    };
    static const struct KnownId ids[] = {
        {"arc.fill_or_complete.shape_completion.boundary_fill_frame",
         "2c4cc9b1eb23"},
        {"arc.object_motion.object_motion.offset_frame", "3f7a743b2ae9"},
        {"arc.recolor.color_rewrite.mask_color_frame", "0dac0ae7b2ce"},
        {"arc.tile_scale.tile_scale.motif_scale_frame", "fbfaf50866b5"},
        {"arc.component_color_summary.component_summary.component_color_frame",
         "a645695a5994"},
        {"arc.crop.object_render.bbox_frame", "3f92aaa0ddf6"},
        {"arc.macrocell_substitute.template_render.macrocell_frame",
         "75d9e2453cf2"},
        {"arc.masked_or_normalized_crop.object_render.normalized_bbox_frame",
         "08faa2d8c1a7"},
        {"arc.recolor_crop.object_render.bbox_color_frame", "81c438fff9ef"},
        {"arc.separator_panel_map.panel_composition.separator_panel_frame",
         "a6b18bb0e848"},
        {"arc.symmetry.symmetry_geometry.axis_geometry_frame", "22d4e2d79fb9"},
        {"arc.visual_combinator.combinator_render.component_relation_frame",
         "3e21963f52a1"},
        {"arc.tile_repeat.lattice_transform.task_bbox", "f406fb43c3ff"},
        {"arc.reference_recolor_mask", "c8fd08924126"},
        {"arc.object_relation_motion_refinement", "b53118f28348"},
        {"arc.crop.lattice_transform.task_bbox", "ffe41601241c"},
    };
    size_t i;
    for (i = 0U; i < sizeof(ids) / sizeof(ids[0]); ++i) {
        if (strcmp(ids[i].name, operator_name) == 0) {
            return ids[i].suffix;
        }
    }
    return NULL;
}

static void lso_operator_id(const char *operator_name, char out[64]) {
    const char *known = lso_known_operator_suffix(operator_name);
    if (known != NULL) {
        snprintf(out, 64U, "arc_sketch_op:%s", known);
    } else {
        snprintf(out,
                 64U,
                 "arc_sketch_op:%012llx",
                 lso_fnv1a48(operator_name));
    }
}

static void lso_slot_signature_from_contract(const char *contract,
                                             char out[256]) {
    const char *cursor = contract == NULL ? "" : contract;
    size_t used = 0U;
    int saw_assignment = strchr(cursor, '=') != NULL;
    if (!saw_assignment) {
        lso_copy(out, 256U, cursor);
        return;
    }
    out[0] = '\0';
    while (*cursor != '\0' && used + 1U < 256U) {
        const char *key_start;
        const char *key_end;
        while (*cursor == '|' || *cursor == ',' || isspace((unsigned char)*cursor)) {
            ++cursor;
        }
        if (*cursor == '\0') {
            break;
        }
        key_start = cursor;
        while (*cursor != '\0' && *cursor != '=' && *cursor != '|' &&
               *cursor != ',') {
            ++cursor;
        }
        key_end = cursor;
        if (*cursor == '=') {
            while (*cursor != '\0' && *cursor != '|' && *cursor != ',') {
                ++cursor;
            }
        }
        if (key_end > key_start) {
            size_t len = (size_t)(key_end - key_start);
            if (used > 0U && used + 1U < 256U) {
                out[used++] = '|';
            }
            if (used + len >= 256U) {
                len = 255U - used;
            }
            memcpy(out + used, key_start, len);
            used += len;
            out[used] = '\0';
        }
    }
}

static int lso_is_attempt_near_miss(const char *failure_stage) {
    return failure_stage != NULL &&
           (strcmp(failure_stage, "renderer_cells") == 0 ||
            strcmp(failure_stage, "transform_or_color") == 0 ||
            strcmp(failure_stage, "verifier_distance") == 0);
}

static int lso_is_attempt_hard_reject(const char *failure_stage) {
    return failure_stage != NULL &&
           (strcmp(failure_stage, "compiler") == 0 ||
            strcmp(failure_stage, "normalizer_shape") == 0 ||
            strcmp(failure_stage, "renderer_shape") == 0 ||
            strcmp(failure_stage, "selector_or_transform") == 0 ||
            strcmp(failure_stage, "unsupported_shape") == 0);
}

static size_t lso_support_tasks(const LSOOperator *op) {
    return op->accepted_tasks.count > 0U ? op->accepted_tasks.count
                                         : op->source_tasks.count;
}

static size_t lso_proposal_count(const LSOOperator *op) {
    return op->accepted_count + op->near_miss_count;
}

static const char *lso_status(const LSOOperator *op,
                              long min_accepted,
                              long min_tasks) {
    if ((long)op->accepted_count >= min_accepted &&
        (long)op->accepted_tasks.count >= min_tasks) {
        return "promoted";
    }
    if (op->near_miss_count > 0U) {
        return "candidate";
    }
    return "blocked";
}

static const char *lso_reason_code(const LSOOperator *op,
                                   long min_accepted,
                                   long min_tasks) {
    const char *status = lso_status(op, min_accepted, min_tasks);
    if (strcmp(status, "promoted") == 0) {
        return "meets verifier-accepted support threshold";
    }
    if (strcmp(status, "candidate") == 0) {
        return "diagnostic near-miss only; requires compiler and verifier replay";
    }
    return "no accepted or near-miss evidence";
}

static int lso_operator_reserve(LSOOperatorList *list, size_t needed) {
    LSOOperator *items;
    size_t capacity;
    if (needed <= list->capacity) {
        return 0;
    }
    capacity = list->capacity == 0U ? 16U : list->capacity * 2U;
    while (capacity < needed) {
        capacity *= 2U;
    }
    items = (LSOOperator *)realloc(list->items, capacity * sizeof(*list->items));
    if (items == NULL) {
        return -1;
    }
    list->items = items;
    list->capacity = capacity;
    return 0;
}

static void lso_compiler_contract(const char *operator_name,
                                  const char *family,
                                  const char *compiler_target,
                                  char out[256]) {
    if (strcmp(operator_name, "arc.reference_recolor_mask") == 0) {
        lso_copy(out,
                 256U,
                 "learn relative mask/color binding from reference object, "
                 "recolor target object, then replay ARC verifier");
    } else if (strcmp(operator_name, "arc.object_relation_motion_refinement") ==
               0) {
        lso_copy(out,
                 256U,
                 "learn relation-conditioned offset binding before replaying "
                 "object-translate verifier");
    } else if (strcmp(compiler_target, "visual_combinator") == 0) {
        lso_copy(out,
                 256U,
                 "compile learned visual-program stages through the generic "
                 "ARC sketch VM, then replay train-pair verifier");
    } else {
        snprintf(out,
                 256U,
                 "compile typed slots into ARC_SKETCH_OP_%s, then replay "
                 "train-pair verifier",
                 family == NULL ? "unknown" : family);
    }
}

static LSOOperator *lso_find_or_add(LSOOperatorList *list,
                                    const char *family,
                                    const char *operator_name,
                                    const char *compiler_target,
                                    const char *composition_kind,
                                    const char *normalization_frame,
                                    const char *typed_stages,
                                    const char *primitive_contract,
                                    const char *slot_signature,
                                    const char *visual_program,
                                    const char *learned_from) {
    size_t i;
    for (i = 0U; i < list->count; ++i) {
        LSOOperator *op = &list->items[i];
        if (strcmp(op->operator_name, operator_name) == 0 &&
            strcmp(op->learned_from, learned_from) == 0) {
            return op;
        }
    }
    if (lso_operator_reserve(list, list->count + 1U) != 0) {
        return NULL;
    }
    LSOOperator *op = &list->items[list->count++];
    memset(op, 0, sizeof(*op));
    lso_copy(op->family, sizeof(op->family), family);
    lso_copy(op->operator_name, sizeof(op->operator_name), operator_name);
    lso_operator_id(op->operator_name, op->operator_id);
    lso_copy(op->compiler_target, sizeof(op->compiler_target), compiler_target);
    lso_copy(op->composition_kind, sizeof(op->composition_kind), composition_kind);
    lso_copy(op->normalization_frame,
             sizeof(op->normalization_frame),
             normalization_frame);
    lso_copy(op->typed_stages, sizeof(op->typed_stages), typed_stages);
    lso_copy(op->primitive_contract,
             sizeof(op->primitive_contract),
             primitive_contract);
    lso_copy(op->slot_signature, sizeof(op->slot_signature), slot_signature);
    lso_copy(op->visual_program, sizeof(op->visual_program), visual_program);
    lso_copy(op->learned_from, sizeof(op->learned_from), learned_from);
    lso_compiler_contract(op->operator_name,
                          op->family,
                          op->compiler_target,
                          op->compiler_contract);
    return op;
}

static void lso_add_example(LSOOperator *op, const LSOExample *example) {
    if (op == NULL || example == NULL || op->example_count >= 5U) {
        return;
    }
    op->examples[op->example_count++] = *example;
}

static int lso_add_observation(LSOOperator *op,
                               const char *task_id,
                               int accepted,
                               int near_miss,
                               int hard_reject,
                               double verifier_score,
                               const char *failure_stage,
                               const char *rejection_reason,
                               const LSOExample *example) {
    if (op == NULL) {
        return -1;
    }
    if (sp_string_add_unique(&op->source_tasks, task_id) != 0) {
        return -1;
    }
    if (accepted) {
        ++op->accepted_count;
        if (sp_string_add_unique(&op->accepted_tasks, task_id) != 0) {
            return -1;
        }
    } else {
        ++op->rejected_count;
        if (near_miss) {
            ++op->near_miss_count;
        } else if (hard_reject) {
            ++op->hard_rejected_count;
        }
    }
    op->verifier_score_sum += verifier_score;
    if (op->accepted_count + op->rejected_count == 1U ||
        verifier_score > op->best_verifier_score) {
        op->best_verifier_score = verifier_score;
    }
    if (lso_counter_add_nonempty(&op->failure_stages, failure_stage) != 0 ||
        lso_counter_add_nonempty(&op->rejection_reasons, rejection_reason) != 0) {
        return -1;
    }
    lso_add_example(op, example);
    return 0;
}

static LSOOperator *lso_attempt_operator(LSOOperatorList *list,
                                         char **fields,
                                         size_t field_count) {
    const char *family = sp_field(fields, field_count, 8U);
    const char *composition_kind = sp_field(fields, field_count, 11U);
    const char *normalization_frame = sp_field(fields, field_count, 12U);
    const char *typed_stages = sp_field(fields, field_count, 13U);
    const char *primitive_contract = sp_field(fields, field_count, 14U);
    const char *visual_program = sp_field(fields, field_count, 9U);
    const char *compiler_target =
        strcmp(family, "visual_combinator") == 0 ? "visual_combinator"
                                                 : "arc_sketch_program";
    char operator_name[256];
    char slot_signature[256];
    snprintf(operator_name,
             sizeof(operator_name),
             "arc.%s.%s.%s",
             family,
             composition_kind,
             normalization_frame);
    lso_slot_signature_from_contract(sp_field(fields, field_count, 10U),
                                     slot_signature);
    return lso_find_or_add(list,
                           family,
                           operator_name,
                           compiler_target,
                           composition_kind,
                           normalization_frame,
                           typed_stages,
                           primitive_contract,
                           slot_signature,
                           visual_program,
                           "attempt_tsv");
}

static int lso_read_attempts(const char *attempts_tsv_path,
                             LSOOperatorList *operators,
                             size_t *attempt_rows,
                             double near_miss_score) {
    FILE *fp;
    char line[SP_LINE_MAX];
    (void)near_miss_score;
    fp = fopen(attempts_tsv_path, "r");
    if (fp == NULL) {
        return -1;
    }
    while (fgets(line, sizeof(line), fp) != NULL) {
        char *fields[SP_MAX_FIELDS];
        size_t field_count = sp_split_tabs(line, fields, SP_MAX_FIELDS);
        const char *split_name = sp_field(fields, field_count, 0U);
        const char *op_name = sp_field(fields, field_count, 8U);
        const char *task_id = sp_field(fields, field_count, 2U);
        int accepted = sp_bool_text(sp_field(fields, field_count, 16U));
        const char *failure_stage = sp_field(fields, field_count, 18U);
        const char *rejection_reason = sp_field(fields, field_count, 17U);
        double verifier_score = sp_double(sp_field(fields, field_count, 67U));
        int near_miss = !accepted && lso_is_attempt_near_miss(failure_stage);
        int hard_reject =
            !accepted && lso_is_attempt_hard_reject(failure_stage);
        LSOOperator *op;
        LSOExample example;
        if (strcmp(split_name, "split_name") == 0 || split_name[0] == '\0' ||
            op_name[0] == '\0') {
            continue;
        }
        ++*attempt_rows;
        op = lso_attempt_operator(operators, fields, field_count);
        if (op == NULL) {
            fclose(fp);
            return -1;
        }
        memset(&example, 0, sizeof(example));
        example.accepted = accepted;
        lso_copy(example.attempt_id,
                 sizeof(example.attempt_id),
                 sp_field(fields, field_count, 4U));
        lso_copy(example.task_id, sizeof(example.task_id), task_id);
        lso_copy(example.failure_stage,
                 sizeof(example.failure_stage),
                 failure_stage);
        lso_copy(example.rejection_reason,
                 sizeof(example.rejection_reason),
                 rejection_reason);
        lso_copy(example.source_path, sizeof(example.source_path), attempts_tsv_path);
        example.verifier_score = verifier_score;
        if (lso_add_observation(op,
                                task_id,
                                accepted,
                                near_miss,
                                hard_reject,
                                verifier_score,
                                failure_stage,
                                rejection_reason,
                                &example) != 0) {
            fclose(fp);
            return -1;
        }
    }
    if (ferror(fp)) {
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
}

static LSOOperator *lso_reference_recolor_operator(LSOOperatorList *operators) {
    return lso_find_or_add(operators,
                           "reference_recolor_mask",
                           "arc.reference_recolor_mask",
                           "visual_combinator",
                           "reference_conditioned_recolor",
                           "reference_target_bbox",
                           "select_reference|derive_relative_mask|select_target|recolor_overlay",
                           "compiler_replay",
                           "reference_selector|target_selector|source_color|output_color|relative_mask",
                           "reference_recolor_mask",
                           "object_translate_trace|compiler_replay_trace");
}

static LSOOperator *lso_motion_refinement_operator(LSOOperatorList *operators) {
    return lso_find_or_add(operators,
                           "object_relation_motion_refinement",
                           "arc.object_relation_motion_refinement",
                           "visual_combinator",
                           "reference_conditioned_motion",
                           "object_relation_offset",
                           "select_reference|bind_offset|select_target|move_or_copy",
                           "diagnostic_near_miss",
                           "reference_selector|target_selector|dx|dy|render_mode",
                           "object_relation_motion_refinement",
                           "object_translate_trace");
}

static int lso_add_object_translate_trace(LSOOperatorList *operators,
                                          const char *line,
                                          const char *trace_path,
                                          double diagnostic_score) {
    char *reason = sp_json_string_dup(line, "best_rejected_reason");
    char *task_id = sp_json_string_dup(line, "task_instance");
    double score = sp_json_double(line, "best_rejected_verifier_score");
    long dx = (long)sp_json_double(line, "best_rejected_dx");
    long dy = (long)sp_json_double(line, "best_rejected_dy");
    LSOOperator *op = NULL;
    LSOExample example;
    int status = 0;
    if (reason == NULL || task_id == NULL) {
        status = -1;
        goto done;
    }
    if (strcmp(reason, "no_train_match") != 0 || score < diagnostic_score) {
        goto done;
    }
    op = dx == 0L && dy == 0L ? lso_reference_recolor_operator(operators)
                              : lso_motion_refinement_operator(operators);
    if (op == NULL) {
        status = -1;
        goto done;
    }
    memset(&example, 0, sizeof(example));
    example.accepted = 0;
    example.has_translate_fields = 1;
    lso_copy(example.task_id, sizeof(example.task_id), task_id);
    lso_copy(example.failure_stage,
             sizeof(example.failure_stage),
             "diagnostic_near_miss");
    lso_copy(example.rejection_reason, sizeof(example.rejection_reason), reason);
    lso_copy(example.source_path, sizeof(example.source_path), trace_path);
    example.verifier_score = score;
    example.selector = (long)sp_json_double(line, "best_rejected_selector");
    example.color = (long)sp_json_double(line, "best_rejected_color");
    example.output_color =
        (long)sp_json_double(line, "best_rejected_output_color");
    example.dx = dx;
    example.dy = dy;
    example.render_mode =
        (long)sp_json_double(line, "best_rejected_render_mode");
    if (lso_add_observation(op,
                            task_id,
                            0,
                            1,
                            0,
                            score,
                            "diagnostic_near_miss",
                            reason,
                            &example) != 0) {
        status = -1;
    }
done:
    free(reason);
    free(task_id);
    return status;
}

static int lso_add_reference_recolor_trace(LSOOperatorList *operators,
                                           const char *line,
                                           const char *trace_path) {
    long accepted_specs = (long)sp_json_double(line, "accepted_specs");
    long attempted_specs = (long)sp_json_double(line, "attempted_specs");
    char *task_id = sp_json_string_dup(line, "task_instance");
    LSOOperator *op;
    LSOExample example;
    int status = 0;
    if (task_id == NULL) {
        return -1;
    }
    if (accepted_specs <= 0L) {
        goto done;
    }
    op = lso_reference_recolor_operator(operators);
    if (op == NULL) {
        status = -1;
        goto done;
    }
    memset(&example, 0, sizeof(example));
    example.accepted = 1;
    example.has_specs = 1;
    example.accepted_specs = accepted_specs;
    example.attempted_specs = attempted_specs;
    lso_copy(example.task_id, sizeof(example.task_id), task_id);
    lso_copy(example.failure_stage, sizeof(example.failure_stage), "accepted");
    lso_copy(example.source_path, sizeof(example.source_path), trace_path);
    example.verifier_score = 1.0;
    if (lso_add_observation(op,
                            task_id,
                            1,
                            0,
                            0,
                            1.0,
                            "accepted",
                            "",
                            &example) != 0) {
        status = -1;
    }
done:
    free(task_id);
    return status;
}

static int lso_read_trace(const char *trace_path,
                          LSOOperatorList *operators,
                          double diagnostic_score) {
    FILE *fp;
    char line[SP_LINE_MAX];
    if (trace_path == NULL || trace_path[0] == '\0') {
        return 0;
    }
    fp = fopen(trace_path, "r");
    if (fp == NULL) {
        return 0;
    }
    while (fgets(line, sizeof(line), fp) != NULL) {
        if (strstr(line,
                   "\"schema\":\"crlp.arc_object_translate_miner.trace.v1\"") !=
            NULL) {
            if (lso_add_object_translate_trace(operators,
                                               line,
                                               trace_path,
                                               diagnostic_score) != 0) {
                fclose(fp);
                return -1;
            }
        } else if (strstr(line,
                          "\"schema\":\"crlp.arc_reference_recolor_mask_miner.trace.v1\"") !=
                   NULL) {
            if (lso_add_reference_recolor_trace(operators, line, trace_path) !=
                0) {
                fclose(fp);
                return -1;
            }
        }
    }
    if (ferror(fp)) {
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
}

static int lso_sort_rank(const LSOOperator *op) {
    if (op->accepted_count > 0U) {
        return 0;
    }
    if (op->near_miss_count > 0U) {
        return 1;
    }
    return 2;
}

static int lso_operator_cmp(const void *a, const void *b) {
    const LSOOperator *oa = (const LSOOperator *)a;
    const LSOOperator *ob = (const LSOOperator *)b;
    int ra = lso_sort_rank(oa);
    int rb = lso_sort_rank(ob);
    size_t pa = lso_proposal_count(oa);
    size_t pb = lso_proposal_count(ob);
    if (ra != rb) {
        return ra < rb ? -1 : 1;
    }
    if (pa != pb) {
        return pa > pb ? -1 : 1;
    }
    if (oa->accepted_count != ob->accepted_count) {
        return oa->accepted_count > ob->accepted_count ? -1 : 1;
    }
    if (oa->best_verifier_score != ob->best_verifier_score) {
        return oa->best_verifier_score > ob->best_verifier_score ? -1 : 1;
    }
    return strcmp(oa->operator_name, ob->operator_name);
}

static void lso_write_example(FILE *fp, const LSOExample *example) {
    fputs("{\"accepted\":", fp);
    fputs(example->accepted ? "true" : "false", fp);
    if (example->attempt_id[0] != '\0') {
        fputs(",\"attempt_id\":", fp);
        sp_json_string(fp, example->attempt_id);
    }
    if (example->has_translate_fields) {
        fprintf(fp,
                ",\"color\":%ld,\"dx\":%ld,\"dy\":%ld",
                example->color,
                example->dx,
                example->dy);
    }
    if (example->has_specs) {
        fprintf(fp,
                ",\"accepted_specs\":%ld,\"attempted_specs\":%ld",
                example->accepted_specs,
                example->attempted_specs);
    }
    fputs(",\"failure_stage\":", fp);
    sp_json_string(fp, example->failure_stage);
    if (example->has_translate_fields) {
        fprintf(fp, ",\"output_color\":%ld", example->output_color);
    }
    fputs(",\"rejection_reason\":", fp);
    sp_json_string(fp, example->rejection_reason);
    if (example->has_translate_fields) {
        fprintf(fp,
                ",\"render_mode\":%ld,\"selector\":%ld",
                example->render_mode,
                example->selector);
    }
    fputs(",\"source_path\":", fp);
    sp_json_string(fp, example->source_path);
    fputs(",\"task_id\":", fp);
    sp_json_string(fp, example->task_id);
    fprintf(fp, ",\"verifier_score\":%.8f}", example->verifier_score);
}

static int lso_write_operator(FILE *fp,
                              LSOOperator *op,
                              const char *date,
                              long min_accepted,
                              long min_tasks) {
    size_t i;
    size_t total = op->accepted_count + op->rejected_count;
    double mean = total == 0U ? 0.0 : op->verifier_score_sum / (double)total;
    fputs("{\"accepted_count\":", fp);
    fprintf(fp, "%zu,\"accepted_task_count\":%zu", op->accepted_count,
            op->accepted_tasks.count);
    fprintf(fp, ",\"best_verifier_score\":%.8f", op->best_verifier_score);
    fputs(",\"compiler_contract\":", fp);
    sp_json_string(fp, op->compiler_contract);
    fputs(",\"compiler_target\":", fp);
    sp_json_string(fp, op->compiler_target);
    fputs(",\"composition_kind\":", fp);
    sp_json_string(fp, op->composition_kind);
    fputs(",\"date\":", fp);
    sp_json_string(fp, date);
    fputs(",\"family\":", fp);
    sp_json_string(fp, op->family);
    fprintf(fp,
            ",\"hard_rejected_count\":%zu,\"learned_from\":",
            op->hard_rejected_count);
    sp_json_string(fp, op->learned_from);
    fprintf(fp, ",\"mean_verifier_score\":%.8f", mean);
    fprintf(fp, ",\"near_miss_count\":%zu", op->near_miss_count);
    fputs(",\"normalization_frame\":", fp);
    sp_json_string(fp, op->normalization_frame);
    fputs(",\"operator_id\":", fp);
    sp_json_string(fp, op->operator_id);
    fputs(",\"operator_name\":", fp);
    sp_json_string(fp, op->operator_name);
    fputs(",\"primitive_contract\":", fp);
    sp_json_string(fp, op->primitive_contract);
    fputs(",\"promotion_policy\":\"accepted verifier rows promote; diagnostic near-misses remain candidates\",",
          fp);
    fputs("\"promotion_status\":", fp);
    sp_json_string(fp, lso_status(op, min_accepted, min_tasks));
    fputs(",\"reason_code\":", fp);
    sp_json_string(fp, lso_reason_code(op, min_accepted, min_tasks));
    fprintf(fp,
            ",\"rejected_count\":%zu,\"report_version\":1,"
            "\"schema\":\"crlp.arc_learned_sketch_op.v1\",",
            op->rejected_count);
    fputs("\"slot_signature\":", fp);
    sp_json_string(fp, op->slot_signature);
    fputs(",\"source_examples\":[", fp);
    for (i = 0U; i < op->example_count; ++i) {
        if (i > 0U) {
            fputc(',', fp);
        }
        lso_write_example(fp, &op->examples[i]);
    }
    fprintf(fp,
            "],\"source_task_count\":%zu,\"support_tasks\":%zu,",
            op->source_tasks.count,
            lso_support_tasks(op));
    fputs("\"top_failure_stages\":", fp);
    sp_counter_write(fp, &op->failure_stages);
    fputs(",\"top_rejection_reasons\":", fp);
    sp_counter_write(fp, &op->rejection_reasons);
    fputs(",\"typed_stages\":", fp);
    sp_json_string(fp, op->typed_stages);
    fputs(",\"verifier_gate\":", fp);
    sp_json_string(fp, op->primitive_contract);
    fputs(",\"visual_program\":", fp);
    sp_json_string(fp, op->visual_program);
    fputs("}\n", fp);
    return ferror(fp) ? -1 : 0;
}

static int lso_write_prior_row(FILE *fp,
                               const LSOOperator *op,
                               long min_accepted,
                               long min_tasks) {
    size_t total = op->accepted_count + op->rejected_count;
    double mean = total == 0U ? 0.0 : op->verifier_score_sum / (double)total;
    return fprintf(fp,
                   "crlp.arc_learned_sketch_op_prior.v1\t%s\t%s\t%s\t%s\t%s\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\t%.8f\t%.8f\t%s\t%s\t%s\t%s\t%s\t%s\n",
                   op->operator_id,
                   op->operator_name,
                   op->family,
                   op->compiler_target,
                   lso_status(op, min_accepted, min_tasks),
                   lso_proposal_count(op),
                   op->accepted_count,
                   op->rejected_count,
                   op->near_miss_count,
                   op->hard_rejected_count,
                   lso_support_tasks(op),
                   mean,
                   op->best_verifier_score,
                   op->composition_kind,
                   op->normalization_frame,
                   op->typed_stages,
                   op->primitive_contract,
                   op->slot_signature,
                   op->learned_from) < 0
               ? -1
               : 0;
}

static int lso_summary_add_counts(LSOSummary *summary,
                                  LSOOperatorList *operators,
                                  long min_accepted,
                                  long min_tasks) {
    size_t i;
    for (i = 0U; i < operators->count; ++i) {
        LSOOperator *op = &operators->items[i];
        const char *status = lso_status(op, min_accepted, min_tasks);
        if (sp_counter_add(&summary->family_counts, op->family) != 0 ||
            sp_counter_add(&summary->learned_from_counts, op->learned_from) !=
                0 ||
            sp_counter_add(&summary->status_counts, status) != 0) {
            return -1;
        }
        if (strcmp(status, "promoted") == 0) {
            ++summary->promoted_operators;
        } else if (strcmp(status, "candidate") == 0) {
            ++summary->candidate_operators;
        } else {
            ++summary->blocked_operators;
        }
    }
    return 0;
}

static void lso_write_top_candidate(FILE *fp, const LSOOperator *op,
                                    long min_accepted, long min_tasks) {
    fputs("{\"accepted_count\":", fp);
    fprintf(fp,
            "%zu,\"best_verifier_score\":%.8f,\"compiler_target\":",
            op->accepted_count,
            op->best_verifier_score);
    sp_json_string(fp, op->compiler_target);
    fputs(",\"family\":", fp);
    sp_json_string(fp, op->family);
    fprintf(fp, ",\"near_miss_count\":%zu,\"operator_name\":",
            op->near_miss_count);
    sp_json_string(fp, op->operator_name);
    fputs(",\"promotion_status\":", fp);
    sp_json_string(fp, lso_status(op, min_accepted, min_tasks));
    fprintf(fp, ",\"support_tasks\":%zu}", lso_support_tasks(op));
}

static int lso_write_summary_json(FILE *fp,
                                  const char *attempts_tsv_path,
                                  const char *trace_path,
                                  LSOOperatorList *operators,
                                  LSOSummary *summary,
                                  const char *date,
                                  long min_accepted,
                                  long min_tasks) {
    size_t i;
    size_t top_count = operators->count < 12U ? operators->count : 12U;
    fputs("{\"attempt_paths\":[", fp);
    sp_json_string(fp, attempts_tsv_path);
    fputs("],\"attempt_rows\":", fp);
    fprintf(fp, "%zu,\"blocked_operators\":%zu,\"candidate_operators\":%zu,",
            summary->attempt_rows,
            summary->blocked_operators,
            summary->candidate_operators);
    fputs("\"claim_scope\":\"learned sketch-op priors only; all candidates still require domain compiler replay before they can act as verifier gates\",",
          fp);
    fputs("\"compiler_trace_paths\":[", fp);
    if (trace_path != NULL && trace_path[0] != '\0') {
        sp_json_string(fp, trace_path);
    }
    fputs("],\"date\":", fp);
    sp_json_string(fp, date);
    fputs(",\"family_counts\":", fp);
    sp_counter_write(fp, &summary->family_counts);
    fputs(",\"learned_from_counts\":", fp);
    sp_counter_write(fp, &summary->learned_from_counts);
    fprintf(fp,
            ",\"operator_candidates\":%zu,\"operator_status_counts\":",
            operators->count);
    sp_counter_write(fp, &summary->status_counts);
    fprintf(fp,
            ",\"promoted_operators\":%zu,\"report_version\":1,"
            "\"schema\":\"crlp.arc_learned_sketch_ops.summary.v1\","
            "\"top_candidates\":[",
            summary->promoted_operators);
    for (i = 0U; i < top_count; ++i) {
        if (i > 0U) {
            fputc(',', fp);
        }
        lso_write_top_candidate(fp, &operators->items[i], min_accepted, min_tasks);
    }
    fputs("]}\n", fp);
    return ferror(fp) ? -1 : 0;
}

static int lso_write_markdown(FILE *md,
                              LSOOperatorList *operators,
                              LSOSummary *summary,
                              const char *date,
                              long min_accepted,
                              long min_tasks) {
    size_t i;
    fprintf(md,
            "# ARC Learned Sketch Operators\n\n"
            "Date: %s\n\n"
            "This pass mines typed ARC visual-pipeline attempt rows, diagnostic traces, and compiler-replay traces into learned sketch-op families. Accepted verifier rows can promote an operator prior; diagnostic near-misses remain candidates until a compiler emits executable sketches and the ARC verifier accepts them.\n\n"
            "Attempt rows: `%zu`\n"
            "Operator candidates: `%zu`\n"
            "Promoted operators: `%zu`\n"
            "Diagnostic/candidate operators: `%zu`\n\n"
            "| Operator | Status | Family | Support tasks | Accepted | Near misses | Best score | Compiler |\n"
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |\n",
            date,
            summary->attempt_rows,
            operators->count,
            summary->promoted_operators,
            summary->candidate_operators);
    for (i = 0U; i < operators->count; ++i) {
        const LSOOperator *op = &operators->items[i];
        fprintf(md,
                "| `%s` | `%s` | `%s` | %zu | %zu | %zu | %.3f | `%s` |\n",
                op->operator_name,
                lso_status(op, min_accepted, min_tasks),
                op->family,
                lso_support_tasks(op),
                op->accepted_count,
                op->near_miss_count,
                op->best_verifier_score,
                op->compiler_target);
    }
    fputs("\n## Contract\n\n"
          "- Output schema: `crlp.arc_learned_sketch_op.v1`.\n"
          "- Prior schema: `crlp.arc_learned_sketch_op_prior.v1`.\n"
          "- Promotion requires accepted verifier support; near-miss diagnostics are proposal pressure only.\n"
          "- Runtime use still requires a domain compiler and ARC verifier replay.\n",
          md);
    return ferror(md) ? -1 : 0;
}

static int lso_write_outputs(const char *attempts_tsv_path,
                             const char *trace_path,
                             const char *operators_path,
                             const char *prior_path,
                             const char *summary_path,
                             const char *markdown_path,
                             LSOOperatorList *operators,
                             LSOSummary *summary,
                             long min_accepted,
                             long min_tasks) {
    FILE *ops = NULL;
    FILE *prior = NULL;
    FILE *summary_fp = NULL;
    FILE *md = NULL;
    char date[11];
    size_t i;
    int status = -1;
    lso_today(date);
    if (sp_mkdir_parent(operators_path) != 0 ||
        sp_mkdir_parent(prior_path) != 0 ||
        sp_mkdir_parent(summary_path) != 0 ||
        sp_mkdir_parent(markdown_path) != 0) {
        goto done;
    }
    if (operators->count > 1U) {
        qsort(operators->items,
              operators->count,
              sizeof(operators->items[0]),
              lso_operator_cmp);
    }
    if (lso_summary_add_counts(summary, operators, min_accepted, min_tasks) !=
        0) {
        goto done;
    }
    ops = fopen(operators_path, "w");
    prior = fopen(prior_path, "w");
    summary_fp = fopen(summary_path, "w");
    md = fopen(markdown_path, "w");
    if (ops == NULL || prior == NULL || summary_fp == NULL || md == NULL) {
        goto done;
    }
    fputs("schema_version\toperator_id\toperator_name\tfamily\tcompiler_target\t"
          "promotion_status\tproposal_count\taccepted_count\trejected_count\t"
          "near_miss_count\thard_rejected_count\tsupport_tasks\t"
          "mean_verifier_score\tbest_verifier_score\tcomposition_kind\t"
          "normalization_frame\ttyped_stages\tprimitive_contract\t"
          "slot_signature\tlearned_from\n",
          prior);
    for (i = 0U; i < operators->count; ++i) {
        if (lso_write_operator(ops,
                               &operators->items[i],
                               date,
                               min_accepted,
                               min_tasks) != 0 ||
            lso_write_prior_row(prior,
                                &operators->items[i],
                                min_accepted,
                                min_tasks) != 0) {
            goto done;
        }
    }
    if (lso_write_summary_json(summary_fp,
                               attempts_tsv_path,
                               trace_path,
                               operators,
                               summary,
                               date,
                               min_accepted,
                               min_tasks) != 0 ||
        lso_write_markdown(md,
                           operators,
                           summary,
                           date,
                           min_accepted,
                           min_tasks) != 0) {
        goto done;
    }
    if (fclose(ops) != 0 || fclose(prior) != 0 || fclose(summary_fp) != 0 ||
        fclose(md) != 0) {
        ops = NULL;
        prior = NULL;
        summary_fp = NULL;
        md = NULL;
        goto done;
    }
    ops = NULL;
    prior = NULL;
    summary_fp = NULL;
    md = NULL;
    status = 0;
done:
    if (ops != NULL) {
        fclose(ops);
    }
    if (prior != NULL) {
        fclose(prior);
    }
    if (summary_fp != NULL) {
        fclose(summary_fp);
    }
    if (md != NULL) {
        fclose(md);
    }
    return status;
}

static void lso_free_operators(LSOOperatorList *operators) {
    size_t i;
    for (i = 0U; i < operators->count; ++i) {
        sp_string_list_free(&operators->items[i].source_tasks);
        sp_string_list_free(&operators->items[i].accepted_tasks);
        sp_counter_free(&operators->items[i].failure_stages);
        sp_counter_free(&operators->items[i].rejection_reasons);
    }
    free(operators->items);
    memset(operators, 0, sizeof(*operators));
}

int nn_grammar_discovery_write_arc_learned_sketch_ops(
    const char *attempts_tsv_path,
    const char *trace_path,
    const char *operators_path,
    const char *prior_path,
    const char *summary_path,
    const char *markdown_path,
    long min_accepted,
    long min_tasks,
    double near_miss_score,
    double diagnostic_score) {
    LSOOperatorList operators;
    LSOSummary summary;
    int status = -1;
    memset(&operators, 0, sizeof(operators));
    memset(&summary, 0, sizeof(summary));
    if (attempts_tsv_path == NULL || attempts_tsv_path[0] == '\0' ||
        operators_path == NULL || operators_path[0] == '\0' ||
        prior_path == NULL || prior_path[0] == '\0' ||
        summary_path == NULL || summary_path[0] == '\0' ||
        markdown_path == NULL || markdown_path[0] == '\0') {
        goto done;
    }
    if (min_accepted <= 0L) {
        min_accepted = 1L;
    }
    if (min_tasks <= 0L) {
        min_tasks = 1L;
    }
    if (near_miss_score <= 0.0) {
        near_miss_score = 0.75;
    }
    if (diagnostic_score <= 0.0) {
        diagnostic_score = 0.90;
    }
    if (lso_read_attempts(attempts_tsv_path,
                          &operators,
                          &summary.attempt_rows,
                          near_miss_score) != 0 ||
        lso_read_trace(trace_path, &operators, diagnostic_score) != 0 ||
        lso_write_outputs(attempts_tsv_path,
                          trace_path,
                          operators_path,
                          prior_path,
                          summary_path,
                          markdown_path,
                          &operators,
                          &summary,
                          min_accepted,
                          min_tasks) != 0) {
        goto done;
    }
    printf("arc-learned-sketch-ops operators=%zu promoted=%zu candidates=%zu blocked=%zu summary=%s\n",
           operators.count,
           summary.promoted_operators,
           summary.candidate_operators,
           summary.blocked_operators,
           summary_path);
    status = 0;
done:
    lso_free_operators(&operators);
    sp_counter_free(&summary.family_counts);
    sp_counter_free(&summary.learned_from_counts);
    sp_counter_free(&summary.status_counts);
    return status;
}
