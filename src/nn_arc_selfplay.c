#include "nn_grammar_discovery.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

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
