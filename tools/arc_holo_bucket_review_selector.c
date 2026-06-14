#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <glob.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define INPUT_SCHEMA "crlp.arc_holo_bucket_review_candidate.v1"
#define SUMMARY_SCHEMA "crlp.arc_holo_bucket_review_selector.summary.v1"
#define WEIGHT_SCHEMA "crlp.arc_holo_bucket_review_selector.weights.v1"
#define LINE_CAP 65536
#define PATH_CAP 4096
#define NAME_CAP 256
#define TASK_CAP 160
#define SOURCE_CAP 512
#define BASE_FEATURES 38
#define MAX_HASH_BINS 256
#define MAX_FEATURES (BASE_FEATURES + MAX_HASH_BINS)

typedef struct {
    char source[SOURCE_CAP];
    char task_instance[TASK_CAP];
    char candidate_name[NAME_CAP];
    int budget;
    int review_slots;
    int baseline_rank;
    int review_rank;
    int op;
    int input_color;
    int output_color;
    int background_color;
    int selector_kind;
    int output_selector_kind;
    int reference_color;
    int render_mode;
    int step_count;
    int hypothesis_template_id;
    int hypothesis_slot_count;
    double pressure_raw;
    double review_priority;
    double review_frontier_priority;
    double priority_deficit;
    double baseline_score;
    double structural_score;
    double holo_pressure_updates;
    double positive_rows;
    double dx;
    double dy;
    double scale;
    bool exact_label;
    bool selected_for_review;
    bool would_review_under_fixed_gate;
    bool gate_open;
    int split;
    double raw_score;
    double learned_score;
    double features[MAX_FEATURES];
} Row;

typedef struct {
    Row *items;
    size_t count;
    size_t cap;
} Rows;

typedef struct {
    int *items;
    size_t count;
    size_t cap;
} IntList;

typedef struct {
    IntList *items;
    size_t count;
    size_t cap;
} Groups;

typedef struct {
    char **items;
    size_t count;
    size_t cap;
} Paths;

typedef struct {
    size_t groups;
    size_t groups_with_exact;
    size_t exact_rows;
    size_t actual_selected_exact_rows;
    size_t raw_loaded_selected_exact_rows;
    size_t learned_loaded_selected_exact_rows;
    size_t groups_with_actual_hit;
    size_t groups_with_raw_loaded_hit;
    size_t groups_with_learned_loaded_hit;
    double mean_original_exact_review_rank;
    double mean_raw_loaded_exact_rank;
    double mean_learned_loaded_exact_rank;
    int max_original_exact_review_rank;
    int max_learned_loaded_exact_rank;
} Metrics;

static uint64_t fnv1a(const char *text)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    const unsigned char *p = (const unsigned char *)text;
    while (*p) {
        hash ^= (uint64_t)*p++;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void die(const char *message)
{
    fprintf(stderr, "%s\n", message);
    exit(2);
}

static char *xstrdup(const char *value)
{
    size_t n = strlen(value) + 1u;
    char *copy = (char *)malloc(n);
    if (copy == NULL) {
        die("out of memory");
    }
    memcpy(copy, value, n);
    return copy;
}

static void paths_push(Paths *paths, const char *path)
{
    for (size_t i = 0; i < paths->count; ++i) {
        if (strcmp(paths->items[i], path) == 0) {
            return;
        }
    }
    if (paths->count == paths->cap) {
        size_t next = paths->cap ? paths->cap * 2u : 16u;
        char **items = (char **)realloc(paths->items, next * sizeof(*items));
        if (items == NULL) {
            die("out of memory");
        }
        paths->items = items;
        paths->cap = next;
    }
    paths->items[paths->count++] = xstrdup(path);
}

static bool has_jsonl_suffix(const char *name)
{
    size_t n = strlen(name);
    return n >= 6u && strcmp(name + n - 6u, ".jsonl") == 0;
}

static void scan_jsonl_dir(Paths *paths, const char *dir_path)
{
    DIR *dir = opendir(dir_path);
    if (dir == NULL) {
        return;
    }
    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        if (!has_jsonl_suffix(ent->d_name)) {
            continue;
        }
        char path[PATH_CAP];
        snprintf(path, sizeof(path), "%s/%s", dir_path, ent->d_name);
        paths_push(paths, path);
    }
    closedir(dir);
}

static void expand_input(Paths *paths, const char *input)
{
    if (strpbrk(input, "*?[]") != NULL) {
        glob_t g;
        if (glob(input, 0, NULL, &g) == 0) {
            for (size_t i = 0; i < g.gl_pathc; ++i) {
                paths_push(paths, g.gl_pathv[i]);
            }
        }
        globfree(&g);
        return;
    }
    struct stat st;
    if (stat(input, &st) == 0 && S_ISDIR(st.st_mode)) {
        char raw[PATH_CAP];
        snprintf(raw, sizeof(raw), "%s/raw", input);
        if (stat(raw, &st) == 0 && S_ISDIR(st.st_mode)) {
            scan_jsonl_dir(paths, raw);
        } else {
            scan_jsonl_dir(paths, input);
        }
        return;
    }
    paths_push(paths, input);
}

static const char *json_key_value(const char *line, const char *key)
{
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = strstr(line, needle);
    if (p == NULL) {
        return NULL;
    }
    p += strlen(needle);
    while (*p && isspace((unsigned char)*p)) {
        ++p;
    }
    if (*p != ':') {
        return NULL;
    }
    ++p;
    while (*p && isspace((unsigned char)*p)) {
        ++p;
    }
    return p;
}

static bool json_get_string(const char *line, const char *key, char *out, size_t out_cap)
{
    const char *p = json_key_value(line, key);
    if (p == NULL || *p != '"') {
        return false;
    }
    ++p;
    size_t used = 0;
    while (*p && *p != '"') {
        char ch = *p++;
        if (ch == '\\' && *p) {
            ch = *p++;
            if (ch == 'u') {
                for (int i = 0; i < 4 && isxdigit((unsigned char)*p); ++i) {
                    ++p;
                }
                ch = '?';
            }
        }
        if (used + 1u < out_cap) {
            out[used++] = ch;
        }
    }
    if (*p != '"') {
        return false;
    }
    if (out_cap > 0) {
        out[used] = '\0';
    }
    return true;
}

static double json_get_number(const char *line, const char *key, double fallback)
{
    const char *p = json_key_value(line, key);
    if (p == NULL) {
        return fallback;
    }
    errno = 0;
    char *end = NULL;
    double value = strtod(p, &end);
    if (end == p || errno == ERANGE || !isfinite(value)) {
        return fallback;
    }
    return value;
}

static bool json_get_bool(const char *line, const char *key)
{
    const char *p = json_key_value(line, key);
    if (p == NULL) {
        return false;
    }
    return strncmp(p, "true", 4) == 0 || strncmp(p, "1", 1) == 0;
}

static void rows_push(Rows *rows, const Row *row)
{
    if (rows->count == rows->cap) {
        size_t next = rows->cap ? rows->cap * 2u : 1024u;
        Row *items = (Row *)realloc(rows->items, next * sizeof(*items));
        if (items == NULL) {
            die("out of memory");
        }
        rows->items = items;
        rows->cap = next;
    }
    rows->items[rows->count++] = *row;
}

static void add_feature(double *features, size_t index, double value)
{
    if (index < MAX_FEATURES && isfinite(value)) {
        features[index] += value;
    }
}

static void compute_features(Row *row, int hash_bins)
{
    memset(row->features, 0, sizeof(row->features));
    const double baseline_rank = row->baseline_rank > 0 ? row->baseline_rank : 1.0;
    const double review_rank = row->review_rank > 0 ? row->review_rank : 1.0;
    const double budget = row->budget > 0 ? row->budget : 1.0;
    add_feature(row->features, 0, 1.0);
    add_feature(row->features, 1, log1p(baseline_rank) / 8.0);
    add_feature(row->features, 2, 1.0 / baseline_rank);
    add_feature(row->features, 3, log1p(review_rank) / 8.0);
    add_feature(row->features, 4, 1.0 / review_rank);
    add_feature(row->features, 5, log1p(budget) / 6.0);
    add_feature(row->features, 6, row->pressure_raw);
    add_feature(row->features, 7, row->review_priority / 8.0);
    add_feature(row->features, 8, row->review_frontier_priority / 8.0);
    add_feature(row->features, 9, row->priority_deficit / 8.0);
    add_feature(row->features, 10, (row->review_priority - row->review_frontier_priority) / 8.0);
    add_feature(row->features, 11, row->baseline_score / 160.0);
    add_feature(row->features, 12, row->structural_score / 160.0);
    add_feature(row->features, 13, (row->baseline_score - row->structural_score) / 80.0);
    add_feature(row->features, 14, row->holo_pressure_updates / 128.0);
    add_feature(row->features, 15, row->positive_rows / 128.0);
    add_feature(row->features, 16, row->would_review_under_fixed_gate ? 1.0 : 0.0);
    add_feature(row->features, 17, row->gate_open ? 1.0 : 0.0);
    add_feature(row->features, 18, row->dx / 16.0);
    add_feature(row->features, 19, row->dy / 16.0);
    add_feature(row->features, 20, row->scale / 16.0);
    int cats[] = {row->op, row->input_color, row->output_color, row->background_color,
                  row->selector_kind, row->output_selector_kind, row->reference_color,
                  row->render_mode, row->step_count, row->hypothesis_template_id,
                  row->hypothesis_slot_count};
    for (size_t i = 0; i < sizeof(cats) / sizeof(cats[0]); ++i) {
        uint64_t h = (uint64_t)(i + 1u) * UINT64_C(1315423911) ^ (uint64_t)(unsigned)cats[i];
        add_feature(row->features, 21u + (h % 17u), 1.0);
    }
    if (hash_bins > MAX_HASH_BINS) {
        hash_bins = MAX_HASH_BINS;
    }
    if (hash_bins > 0 && row->candidate_name[0] != '\0') {
        char token[NAME_CAP];
        size_t used = 0;
        for (const char *p = row->candidate_name;; ++p) {
            if (*p != '\0' && *p != '-' && *p != '_') {
                if (used + 1u < sizeof(token)) {
                    token[used++] = *p;
                }
            } else {
                if (used > 0) {
                    token[used] = '\0';
                    add_feature(row->features, BASE_FEATURES + (fnv1a(token) % (uint64_t)hash_bins), 1.0);
                    used = 0;
                }
                if (*p == '\0') {
                    break;
                }
            }
        }
    }
}

static bool parse_row(const char *line, const char *source, Row *row, int hash_bins)
{
    char schema[128] = {0};
    if (!json_get_string(line, "schema", schema, sizeof(schema)) ||
        strcmp(schema, INPUT_SCHEMA) != 0) {
        return false;
    }
    memset(row, 0, sizeof(*row));
    snprintf(row->source, sizeof(row->source), "%s", source);
    json_get_string(line, "task_instance", row->task_instance, sizeof(row->task_instance));
    json_get_string(line, "candidate_name", row->candidate_name, sizeof(row->candidate_name));
    row->budget = (int)json_get_number(line, "budget", 0.0);
    row->review_slots = (int)json_get_number(line, "review_slots", 1.0);
    row->baseline_rank = (int)json_get_number(line, "baseline_rank", 1.0);
    row->review_rank = (int)json_get_number(line, "review_rank", 1.0);
    row->op = (int)json_get_number(line, "op", 0.0);
    row->input_color = (int)json_get_number(line, "input_color", 0.0);
    row->output_color = (int)json_get_number(line, "output_color", 0.0);
    row->background_color = (int)json_get_number(line, "background_color", 0.0);
    row->selector_kind = (int)json_get_number(line, "selector_kind", 0.0);
    row->output_selector_kind = (int)json_get_number(line, "output_selector_kind", 0.0);
    row->reference_color = (int)json_get_number(line, "reference_color", 0.0);
    row->render_mode = (int)json_get_number(line, "render_mode", 0.0);
    row->step_count = (int)json_get_number(line, "step_count", 0.0);
    row->hypothesis_template_id = (int)json_get_number(line, "hypothesis_template_id", 0.0);
    row->hypothesis_slot_count = (int)json_get_number(line, "hypothesis_slot_count", 0.0);
    row->pressure_raw = json_get_number(line, "pressure_raw", 0.0);
    row->review_priority = json_get_number(line, "review_priority", 0.0);
    row->review_frontier_priority = json_get_number(line, "review_frontier_priority", 0.0);
    row->priority_deficit = json_get_number(line, "priority_deficit", 0.0);
    row->baseline_score = json_get_number(line, "baseline_score", 0.0);
    row->structural_score = json_get_number(line, "structural_score", 0.0);
    row->holo_pressure_updates = json_get_number(line, "holo_pressure_updates", 0.0);
    row->positive_rows = json_get_number(line, "positive_rows", 0.0);
    row->dx = json_get_number(line, "dx", 0.0);
    row->dy = json_get_number(line, "dy", 0.0);
    row->scale = json_get_number(line, "scale", 0.0);
    row->exact_label = json_get_bool(line, "exact_label");
    row->selected_for_review = json_get_bool(line, "selected_for_review");
    row->would_review_under_fixed_gate = json_get_bool(line, "would_review_under_fixed_gate");
    row->gate_open = json_get_bool(line, "gate_open");
    row->raw_score = row->review_priority;
    compute_features(row, hash_bins);
    return true;
}

static void load_file(Rows *rows, const char *path, int hash_bins)
{
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
        exit(2);
    }
    char line[LINE_CAP];
    while (fgets(line, sizeof(line), fp) != NULL) {
        Row row;
        if (parse_row(line, path, &row, hash_bins)) {
            rows_push(rows, &row);
        }
    }
    fclose(fp);
}

static void int_list_push(IntList *list, int value)
{
    if (list->count == list->cap) {
        size_t next = list->cap ? list->cap * 2u : 16u;
        int *items = (int *)realloc(list->items, next * sizeof(*items));
        if (items == NULL) {
            die("out of memory");
        }
        list->items = items;
        list->cap = next;
    }
    list->items[list->count++] = value;
}

static void groups_push(Groups *groups, IntList list)
{
    if (groups->count == groups->cap) {
        size_t next = groups->cap ? groups->cap * 2u : 64u;
        IntList *items = (IntList *)realloc(groups->items, next * sizeof(*items));
        if (items == NULL) {
            die("out of memory");
        }
        groups->items = items;
        groups->cap = next;
    }
    groups->items[groups->count++] = list;
}

static bool same_group(const Row *a, const Row *b)
{
    return a->budget == b->budget &&
           strcmp(a->source, b->source) == 0 &&
           strcmp(a->task_instance, b->task_instance) == 0;
}

static int compare_row_group(const void *lhs, const void *rhs)
{
    const Row *a = (const Row *)lhs;
    const Row *b = (const Row *)rhs;
    int c = strcmp(a->source, b->source);
    if (c != 0) {
        return c;
    }
    c = strcmp(a->task_instance, b->task_instance);
    if (c != 0) {
        return c;
    }
    return (a->budget > b->budget) - (a->budget < b->budget);
}

static void assign_splits_and_groups(Rows *rows, Groups *train, Groups *test,
                                     int holdout_mod, int holdout_bucket)
{
    qsort(rows->items, rows->count, sizeof(rows->items[0]), compare_row_group);
    size_t i = 0;
    while (i < rows->count) {
        size_t j = i + 1u;
        while (j < rows->count && same_group(&rows->items[i], &rows->items[j])) {
            ++j;
        }
        int split = 0;
        if (holdout_mod > 1) {
            split = (int)(fnv1a(rows->items[i].task_instance) % (uint64_t)holdout_mod) == holdout_bucket;
        }
        IntList list = {0};
        for (size_t k = i; k < j; ++k) {
            rows->items[k].split = split;
            int_list_push(&list, (int)k);
        }
        groups_push(split ? test : train, list);
        i = j;
    }
}

static double dot(const double *weights, const double *features, int feature_count)
{
    double sum = 0.0;
    for (int i = 0; i < feature_count; ++i) {
        sum += weights[i] * features[i];
    }
    return sum;
}

static void train_pairwise(Rows *rows, Groups *train, double *weights,
                           int feature_count, int epochs, double learning_rate,
                           double l2, int negative_limit)
{
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double rate = learning_rate / sqrt((double)epoch + 1.0);
        for (size_t gi = 0; gi < train->count; ++gi) {
            IntList *group = &train->items[gi];
            int used_negatives = 0;
            for (size_t pi = 0; pi < group->count; ++pi) {
                Row *pos = &rows->items[group->items[pi]];
                if (!pos->exact_label) {
                    continue;
                }
                used_negatives = 0;
                for (size_t ni = 0; ni < group->count; ++ni) {
                    Row *neg = &rows->items[group->items[ni]];
                    if (neg->exact_label) {
                        continue;
                    }
                    if (negative_limit > 0 && used_negatives++ >= negative_limit) {
                        break;
                    }
                    double margin = dot(weights, pos->features, feature_count) -
                                    dot(weights, neg->features, feature_count);
                    double coeff = margin > 40.0 ? 0.0 :
                                   margin < -40.0 ? 1.0 :
                                   1.0 / (1.0 + exp(margin));
                    for (int f = 0; f < feature_count; ++f) {
                        double diff = pos->features[f] - neg->features[f];
                        weights[f] *= (1.0 - rate * l2);
                        weights[f] += rate * coeff * diff;
                    }
                }
            }
        }
    }
}

static int cmp_raw_desc(const void *lhs, const void *rhs, void *ctx)
{
    const Rows *rows = (const Rows *)ctx;
    const Row *a = &rows->items[*(const int *)lhs];
    const Row *b = &rows->items[*(const int *)rhs];
    return (b->raw_score > a->raw_score) - (b->raw_score < a->raw_score);
}

static int cmp_learned_desc(const void *lhs, const void *rhs, void *ctx)
{
    const Rows *rows = (const Rows *)ctx;
    const Row *a = &rows->items[*(const int *)lhs];
    const Row *b = &rows->items[*(const int *)rhs];
    return (b->learned_score > a->learned_score) - (b->learned_score < a->learned_score);
}

static void sort_indices(int *items, size_t count, int (*cmp)(const void *, const void *, void *), Rows *rows)
{
    for (size_t i = 1; i < count; ++i) {
        int key = items[i];
        size_t j = i;
        while (j > 0 && cmp(&items[j - 1u], &key, rows) > 0) {
            items[j] = items[j - 1u];
            --j;
        }
        items[j] = key;
    }
}

static Metrics evaluate(Rows *rows, Groups *groups, const double *weights, int feature_count)
{
    Metrics m = {0};
    double original_rank_sum = 0.0;
    double raw_rank_sum = 0.0;
    double learned_rank_sum = 0.0;
    size_t raw_rank_count = 0;
    size_t learned_rank_count = 0;
    m.groups = groups->count;
    for (size_t gi = 0; gi < groups->count; ++gi) {
        IntList *group = &groups->items[gi];
        bool has_exact = false;
        bool actual_hit = false;
        bool raw_hit = false;
        bool learned_hit = false;
        int review_slots = 1;
        if (group->count > 0 && rows->items[group->items[0]].review_slots > 0) {
            review_slots = rows->items[group->items[0]].review_slots;
        }
        for (size_t i = 0; i < group->count; ++i) {
            Row *row = &rows->items[group->items[i]];
            row->learned_score = dot(weights, row->features, feature_count);
            row->raw_score = row->review_priority;
            if (row->exact_label) {
                has_exact = true;
                ++m.exact_rows;
                original_rank_sum += row->review_rank;
                if (row->review_rank > m.max_original_exact_review_rank) {
                    m.max_original_exact_review_rank = row->review_rank;
                }
                if (row->selected_for_review) {
                    ++m.actual_selected_exact_rows;
                    actual_hit = true;
                }
            }
        }
        if (has_exact) {
            ++m.groups_with_exact;
        }
        int *order = (int *)malloc(group->count * sizeof(*order));
        if (order == NULL && group->count > 0) {
            die("out of memory");
        }
        memcpy(order, group->items, group->count * sizeof(*order));
        sort_indices(order, group->count, cmp_raw_desc, rows);
        for (size_t rank = 0; rank < group->count; ++rank) {
            Row *row = &rows->items[order[rank]];
            if (row->exact_label) {
                raw_rank_sum += (double)rank + 1.0;
                ++raw_rank_count;
                if ((int)rank < review_slots) {
                    ++m.raw_loaded_selected_exact_rows;
                    raw_hit = true;
                }
            }
        }
        memcpy(order, group->items, group->count * sizeof(*order));
        sort_indices(order, group->count, cmp_learned_desc, rows);
        for (size_t rank = 0; rank < group->count; ++rank) {
            Row *row = &rows->items[order[rank]];
            if (row->exact_label) {
                learned_rank_sum += (double)rank + 1.0;
                ++learned_rank_count;
                if ((int)rank + 1 > m.max_learned_loaded_exact_rank) {
                    m.max_learned_loaded_exact_rank = (int)rank + 1;
                }
                if ((int)rank < review_slots) {
                    ++m.learned_loaded_selected_exact_rows;
                    learned_hit = true;
                }
            }
        }
        free(order);
        if (actual_hit) {
            ++m.groups_with_actual_hit;
        }
        if (raw_hit) {
            ++m.groups_with_raw_loaded_hit;
        }
        if (learned_hit) {
            ++m.groups_with_learned_loaded_hit;
        }
    }
    if (m.exact_rows > 0) {
        m.mean_original_exact_review_rank = original_rank_sum / (double)m.exact_rows;
    }
    if (raw_rank_count > 0) {
        m.mean_raw_loaded_exact_rank = raw_rank_sum / (double)raw_rank_count;
    }
    if (learned_rank_count > 0) {
        m.mean_learned_loaded_exact_rank = learned_rank_sum / (double)learned_rank_count;
    }
    return m;
}

static void write_metrics(FILE *fp, const char *split, Metrics m, int input_files,
                          size_t rows, int epochs, double lr, double l2,
                          int negative_limit, int holdout_mod, int holdout_bucket)
{
    fprintf(fp,
            "{\"schema\":\"%s\",\"split\":\"%s\",\"input_files\":%d,\"rows\":%zu,"
            "\"epochs\":%d,\"learning_rate\":%.17g,\"l2\":%.17g,\"negative_limit\":%d,"
            "\"holdout_mod\":%d,\"holdout_bucket\":%d,"
            "\"groups\":%zu,\"groups_with_exact\":%zu,\"exact_rows\":%zu,"
            "\"actual_selected_exact_rows\":%zu,\"raw_loaded_selected_exact_rows\":%zu,"
            "\"learned_loaded_selected_exact_rows\":%zu,\"groups_with_actual_hit\":%zu,"
            "\"groups_with_raw_loaded_hit\":%zu,\"groups_with_learned_loaded_hit\":%zu,"
            "\"mean_original_exact_review_rank\":%.17g,\"mean_raw_loaded_exact_rank\":%.17g,"
            "\"mean_learned_loaded_exact_rank\":%.17g,\"max_original_exact_review_rank\":%d,"
            "\"max_learned_loaded_exact_rank\":%d}\n",
            SUMMARY_SCHEMA, split, input_files, rows, epochs, lr, l2, negative_limit,
            holdout_mod, holdout_bucket, m.groups, m.groups_with_exact, m.exact_rows,
            m.actual_selected_exact_rows, m.raw_loaded_selected_exact_rows,
            m.learned_loaded_selected_exact_rows, m.groups_with_actual_hit,
            m.groups_with_raw_loaded_hit, m.groups_with_learned_loaded_hit,
            m.mean_original_exact_review_rank, m.mean_raw_loaded_exact_rank,
            m.mean_learned_loaded_exact_rank, m.max_original_exact_review_rank,
            m.max_learned_loaded_exact_rank);
}

static void usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s <inputs...> --out PATH [--epochs N] [--learning-rate X] "
            "[--l2 X] [--negative-limit N] [--holdout-mod N] [--holdout-bucket N] "
            "[--hash-bins N]\n",
            argv0);
}

int main(int argc, char **argv)
{
    const char *out_path = "build/float/arc_holo_bucket_review_selector/summary.jsonl";
    int epochs = 80;
    double learning_rate = 0.05;
    double l2 = 0.001;
    int negative_limit = 24;
    int holdout_mod = 5;
    int holdout_bucket = 0;
    int hash_bins = 64;
    Paths inputs = {0};
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--learning-rate") == 0 && i + 1 < argc) {
            learning_rate = strtod(argv[++i], NULL);
        } else if (strcmp(argv[i], "--l2") == 0 && i + 1 < argc) {
            l2 = strtod(argv[++i], NULL);
        } else if (strcmp(argv[i], "--negative-limit") == 0 && i + 1 < argc) {
            negative_limit = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--holdout-mod") == 0 && i + 1 < argc) {
            holdout_mod = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--holdout-bucket") == 0 && i + 1 < argc) {
            holdout_bucket = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hash-bins") == 0 && i + 1 < argc) {
            hash_bins = atoi(argv[++i]);
        } else if (argv[i][0] == '-') {
            usage(argv[0]);
            return 2;
        } else {
            expand_input(&inputs, argv[i]);
        }
    }
    if (inputs.count == 0) {
        die("no input JSONL files found");
    }
    if (hash_bins < 0) {
        hash_bins = 0;
    }
    if (hash_bins > MAX_HASH_BINS) {
        hash_bins = MAX_HASH_BINS;
    }
    Rows rows = {0};
    for (size_t i = 0; i < inputs.count; ++i) {
        load_file(&rows, inputs.items[i], hash_bins);
    }
    if (rows.count == 0) {
        die("no review candidate rows found");
    }
    Groups train = {0};
    Groups test = {0};
    assign_splits_and_groups(&rows, &train, &test, holdout_mod, holdout_bucket);
    int feature_count = BASE_FEATURES + hash_bins;
    double weights[MAX_FEATURES] = {0};
    train_pairwise(&rows, &train, weights, feature_count, epochs, learning_rate, l2, negative_limit);
    for (size_t i = 0; i < rows.count; ++i) {
        rows.items[i].learned_score = dot(weights, rows.items[i].features, feature_count);
    }
    char out_dir[PATH_CAP];
    snprintf(out_dir, sizeof(out_dir), "%s", out_path);
    char *slash = strrchr(out_dir, '/');
    if (slash != NULL) {
        *slash = '\0';
        char cmd[PATH_CAP + 32];
        snprintf(cmd, sizeof(cmd), "mkdir -p '%s'", out_dir);
        if (system(cmd) != 0) {
            die("failed to create output directory");
        }
    }
    FILE *out = fopen(out_path, "w");
    if (out == NULL) {
        fprintf(stderr, "cannot write %s: %s\n", out_path, strerror(errno));
        return 2;
    }
    Metrics train_m = evaluate(&rows, &train, weights, feature_count);
    Metrics test_m = evaluate(&rows, &test, weights, feature_count);
    Groups all = {0};
    for (size_t i = 0; i < train.count; ++i) {
        groups_push(&all, train.items[i]);
    }
    for (size_t i = 0; i < test.count; ++i) {
        groups_push(&all, test.items[i]);
    }
    Metrics all_m = evaluate(&rows, &all, weights, feature_count);
    size_t train_rows = 0;
    for (size_t i = 0; i < train.count; ++i) {
        train_rows += train.items[i].count;
    }
    size_t test_rows = rows.count - train_rows;
    write_metrics(out, "train", train_m, (int)inputs.count, train_rows, epochs, learning_rate, l2, negative_limit, holdout_mod, holdout_bucket);
    write_metrics(stdout, "train", train_m, (int)inputs.count, train_rows, epochs, learning_rate, l2, negative_limit, holdout_mod, holdout_bucket);
    write_metrics(out, "test", test_m, (int)inputs.count, test_rows, epochs, learning_rate, l2, negative_limit, holdout_mod, holdout_bucket);
    write_metrics(stdout, "test", test_m, (int)inputs.count, test_rows, epochs, learning_rate, l2, negative_limit, holdout_mod, holdout_bucket);
    write_metrics(out, "all", all_m, (int)inputs.count, rows.count, epochs, learning_rate, l2, negative_limit, holdout_mod, holdout_bucket);
    write_metrics(stdout, "all", all_m, (int)inputs.count, rows.count, epochs, learning_rate, l2, negative_limit, holdout_mod, holdout_bucket);
    fprintf(out, "{\"schema\":\"%s\",\"feature_count\":%d,\"top_weights\":[", WEIGHT_SCHEMA, feature_count);
    fprintf(stdout, "{\"schema\":\"%s\",\"feature_count\":%d,\"top_weights\":[", WEIGHT_SCHEMA, feature_count);
    int emitted = 0;
    for (int pass = 0; pass < 40; ++pass) {
        int best = -1;
        double best_abs = -1.0;
        for (int f = 0; f < feature_count; ++f) {
            double a = fabs(weights[f]);
            if (a > best_abs) {
                best_abs = a;
                best = f;
            }
        }
        if (best < 0 || best_abs == 0.0) {
            break;
        }
        if (emitted) {
            fputc(',', out);
            fputc(',', stdout);
        }
        fprintf(out, "{\"feature\":\"f%d\",\"weight\":%.17g}", best, weights[best]);
        fprintf(stdout, "{\"feature\":\"f%d\",\"weight\":%.17g}", best, weights[best]);
        weights[best] = 0.0;
        ++emitted;
    }
    fprintf(out, "]}\n");
    fprintf(stdout, "]}\n");
    fclose(out);
    return 0;
}
