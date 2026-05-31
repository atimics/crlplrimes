#define _POSIX_C_SOURCE 200809L

#include "nn_paper.h"

#include "nn.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#ifdef NN_ENABLE_TEST_API
#include <utime.h>
#endif

#define PAPER_CURVE_WINDOW 50U
#define PAPER_ARC_EPOCH_COUNT 4U
#define PAPER_PRIME_EPOCHS 8U

typedef struct {
    long step;
    double accuracy;
    double loss;
} PaperCurvePoint;

typedef struct {
    PaperCurvePoint *items;
    size_t count;
    size_t capacity;
} PaperCurve;

typedef struct {
    unsigned epoch;
    double top1;
    double nll;
} PaperArcEpochPoint;

typedef struct {
    PaperArcEpochPoint items[PAPER_ARC_EPOCH_COUNT];
    size_t count;
} PaperArcEpochSeries;

typedef struct {
    unsigned epoch;
    double raw_loss;
    double gated_loss;
} PaperPrimeEpochPoint;

typedef struct {
    PaperPrimeEpochPoint items[64];
    size_t count;
} PaperPrimeEpochSeries;

typedef struct {
    char *mode;
    int has_accuracy;
    int has_nll;
    int has_ece;
    double accuracy;
    double nll;
    double ece;
    char *search_saved;
    char model_hash[11];
    char parent_hash[11];
} PaperGateRow;

typedef struct {
    PaperGateRow *items;
    size_t count;
    size_t capacity;
} PaperGateRows;

typedef struct {
    char **items;
    size_t count;
    size_t capacity;
} PaperRawRows;

typedef struct {
    char **items;
    size_t count;
    size_t capacity;
} PaperLineList;

typedef struct {
    int present;
    int has_top1;
    int has_nll;
    int has_ece;
    int has_candidate_count;
    int has_sample_count;
    double top1;
    double nll;
    double ece;
    double candidate_count;
    long sample_count;
    char *hidden;
} PaperMultitaskRow;

typedef struct {
    PaperGateRows gates[4];
    char *reliability[4];
    PaperRawRows arc_hypothesis;
    PaperMultitaskRow shared[4];
    PaperMultitaskRow single[4];
} PaperComparison;

static const char *const PAPER_DOMAINS[4] = {
    "prime_gap",
    "grid_path",
    "delayed_claim",
    "arc_agi",
};

static const char *const PAPER_RELIABILITY_MODES[4] = {
    "expanded_rule_model",
    "budgeted_legality_gate_model",
    "delayed_claim",
    "neural_cell_scorer",
};

static char *paper_strdup(const char *value) {
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

static char *paper_strndup(const char *value, size_t len) {
    char *copy = (char *)malloc(len + 1U);
    if (copy == NULL) {
        return NULL;
    }
    memcpy(copy, value, len);
    copy[len] = '\0';
    return copy;
}

static int paper_file_exists(const char *path) {
    struct stat st;
    return path != NULL && stat(path, &st) == 0;
}

static int paper_mkdirs(const char *dir) {
    char buffer[4096];
    char *p;
    size_t len;
    if (dir == NULL || dir[0] == '\0') {
        return 0;
    }
    len = strlen(dir);
    if (len >= sizeof(buffer)) {
        return -1;
    }
    memcpy(buffer, dir, len + 1U);
    for (p = buffer + 1; *p != '\0'; ++p) {
        if (*p == '/') {
            *p = '\0';
            if (buffer[0] != '\0' && mkdir(buffer, 0777) != 0 &&
                errno != EEXIST) {
                return -1;
            }
            *p = '/';
        }
    }
    if (mkdir(buffer, 0777) != 0 && errno != EEXIST) {
        return -1;
    }
    return 0;
}

static int paper_mkdir_parent(const char *path) {
    char buffer[4096];
    char *slash;
    size_t len;
    if (path == NULL) {
        return -1;
    }
    len = strlen(path);
    if (len == 0U || len >= sizeof(buffer)) {
        return -1;
    }
    memcpy(buffer, path, len + 1U);
    slash = strrchr(buffer, '/');
    if (slash == NULL || slash == buffer) {
        return 0;
    }
    *slash = '\0';
    return paper_mkdirs(buffer);
}

static int paper_curve_append(PaperCurve *curve,
                              long step,
                              double accuracy,
                              double loss) {
    PaperCurvePoint *items;
    size_t capacity;
    if (curve->count + 1U > curve->capacity) {
        capacity = curve->capacity == 0U ? 64U : curve->capacity * 2U;
        items = (PaperCurvePoint *)realloc(curve->items,
                                           capacity * sizeof(PaperCurvePoint));
        if (items == NULL) {
            return -1;
        }
        curve->items = items;
        curve->capacity = capacity;
    }
    curve->items[curve->count].step = step;
    curve->items[curve->count].accuracy = accuracy;
    curve->items[curve->count].loss = loss;
    ++curve->count;
    return 0;
}

static void paper_curve_free(PaperCurve *curve) {
    if (curve == NULL) {
        return;
    }
    free(curve->items);
    memset(curve, 0, sizeof(*curve));
}

static int paper_raw_append(PaperRawRows *rows, const char *value) {
    char **items;
    size_t capacity;
    if (rows->count + 1U > rows->capacity) {
        capacity = rows->capacity == 0U ? 64U : rows->capacity * 2U;
        items = (char **)realloc(rows->items, capacity * sizeof(char *));
        if (items == NULL) {
            return -1;
        }
        rows->items = items;
        rows->capacity = capacity;
    }
    rows->items[rows->count] = paper_strdup(value);
    if (rows->items[rows->count] == NULL) {
        return -1;
    }
    ++rows->count;
    return 0;
}

static void paper_raw_rows_free(PaperRawRows *rows) {
    size_t i;
    if (rows == NULL) {
        return;
    }
    for (i = 0U; i < rows->count; ++i) {
        free(rows->items[i]);
    }
    free(rows->items);
    memset(rows, 0, sizeof(*rows));
}

static void paper_gate_row_free(PaperGateRow *row) {
    if (row == NULL) {
        return;
    }
    free(row->mode);
    free(row->search_saved);
    memset(row, 0, sizeof(*row));
}

static int paper_gate_rows_append(PaperGateRows *rows, PaperGateRow *row) {
    PaperGateRow *items;
    size_t capacity;
    if (rows->count + 1U > rows->capacity) {
        capacity = rows->capacity == 0U ? 8U : rows->capacity * 2U;
        items = (PaperGateRow *)realloc(rows->items,
                                        capacity * sizeof(PaperGateRow));
        if (items == NULL) {
            return -1;
        }
        rows->items = items;
        rows->capacity = capacity;
    }
    rows->items[rows->count] = *row;
    memset(row, 0, sizeof(*row));
    ++rows->count;
    return 0;
}

static void paper_gate_rows_free(PaperGateRows *rows) {
    size_t i;
    if (rows == NULL) {
        return;
    }
    for (i = 0U; i < rows->count; ++i) {
        paper_gate_row_free(&rows->items[i]);
    }
    free(rows->items);
    memset(rows, 0, sizeof(*rows));
}

static void paper_multitask_row_free(PaperMultitaskRow *row) {
    if (row == NULL) {
        return;
    }
    free(row->hidden);
    memset(row, 0, sizeof(*row));
}

static void paper_comparison_free(PaperComparison *comparison) {
    size_t i;
    if (comparison == NULL) {
        return;
    }
    for (i = 0U; i < 4U; ++i) {
        paper_gate_rows_free(&comparison->gates[i]);
        free(comparison->reliability[i]);
        paper_multitask_row_free(&comparison->shared[i]);
        paper_multitask_row_free(&comparison->single[i]);
    }
    paper_raw_rows_free(&comparison->arc_hypothesis);
    memset(comparison, 0, sizeof(*comparison));
}

static char *paper_read_line_dynamic(FILE *fp) {
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

static void paper_chomp(char *line) {
    size_t len;
    if (line == NULL) {
        return;
    }
    len = strlen(line);
    while (len > 0U && (line[len - 1U] == '\n' || line[len - 1U] == '\r')) {
        line[--len] = '\0';
    }
}

static const char *paper_skip_ws(const char *cursor) {
    while (cursor != NULL && isspace((unsigned char)*cursor)) {
        ++cursor;
    }
    return cursor;
}

static const char *paper_json_find_key_value(const char *line, const char *key) {
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

static char *paper_json_dup_string_value(const char *cursor) {
    size_t capacity = 64U;
    size_t used = 0U;
    char *value;
    cursor = paper_skip_ws(cursor);
    if (cursor == NULL || *cursor != '"') {
        return paper_strdup("");
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
        if (used + 1U >= capacity) {
            size_t next_capacity = capacity * 2U;
            char *next = (char *)realloc(value, next_capacity);
            if (next == NULL) {
                free(value);
                return NULL;
            }
            value = next;
            capacity = next_capacity;
        }
        value[used++] = (char)ch;
    }
    if (*cursor != '"') {
        free(value);
        return NULL;
    }
    value[used] = '\0';
    return value;
}

static char *paper_json_dup_key_string(const char *line, const char *key) {
    return paper_json_dup_string_value(paper_json_find_key_value(line, key));
}

static const char *paper_json_value_end(const char *cursor) {
    int depth = 0;
    char open;
    char close;
    cursor = paper_skip_ws(cursor);
    if (cursor == NULL) {
        return NULL;
    }
    if (*cursor == '"') {
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
        return NULL;
    }
    if (*cursor == '[' || *cursor == '{') {
        open = *cursor;
        close = open == '[' ? ']' : '}';
        (void)close;
        while (*cursor != '\0') {
            if (*cursor == '"') {
                ++cursor;
                while (*cursor != '\0') {
                    if (*cursor == '\\' && cursor[1] != '\0') {
                        cursor += 2;
                        continue;
                    }
                    if (*cursor == '"') {
                        ++cursor;
                        break;
                    }
                    ++cursor;
                }
                continue;
            }
            if (*cursor == '[' || *cursor == '{') {
                ++depth;
            } else if (*cursor == ']' || *cursor == '}') {
                --depth;
                if (depth == 0) {
                    return cursor + 1;
                }
            }
            ++cursor;
        }
        return NULL;
    }
    while (*cursor != '\0' && *cursor != ',' && *cursor != ']' && *cursor != '}' &&
           !isspace((unsigned char)*cursor)) {
        ++cursor;
    }
    return cursor;
}

static char *paper_json_dup_raw_key(const char *line, const char *key) {
    const char *start = paper_skip_ws(paper_json_find_key_value(line, key));
    const char *end;
    if (start == NULL) {
        return NULL;
    }
    end = paper_json_value_end(start);
    if (end == NULL || end < start) {
        return NULL;
    }
    return paper_strndup(start, (size_t)(end - start));
}

static int paper_json_get_double(const char *line, const char *key, double *out) {
    const char *cursor = paper_skip_ws(paper_json_find_key_value(line, key));
    char *end = NULL;
    double value;
    if (cursor == NULL || strncmp(cursor, "null", 4U) == 0) {
        return 0;
    }
    errno = 0;
    value = strtod(cursor, &end);
    if (end == cursor || errno == ERANGE || !isfinite(value)) {
        return 0;
    }
    *out = value;
    return 1;
}

static int paper_json_get_long(const char *line, const char *key, long *out) {
    const char *cursor = paper_skip_ws(paper_json_find_key_value(line, key));
    char *end = NULL;
    long value;
    if (cursor == NULL || strncmp(cursor, "null", 4U) == 0) {
        return 0;
    }
    errno = 0;
    value = strtol(cursor, &end, 10);
    if (end == cursor || errno == ERANGE) {
        return 0;
    }
    *out = value;
    return 1;
}

static int paper_raw_array_nonempty(const char *raw) {
    const char *cursor = paper_skip_ws(raw);
    if (cursor == NULL || *cursor != '[') {
        return 0;
    }
    ++cursor;
    cursor = paper_skip_ws(cursor);
    return cursor != NULL && *cursor != ']';
}

static int paper_domain_index(const char *domain) {
    size_t i;
    if (domain == NULL) {
        return -1;
    }
    for (i = 0U; i < 4U; ++i) {
        if (strcmp(domain, PAPER_DOMAINS[i]) == 0) {
            return (int)i;
        }
    }
    return -1;
}

static void paper_first_ten(char out[11], const char *value) {
    size_t len;
    if (value == NULL) {
        value = "";
    }
    len = strlen(value);
    if (len > 10U) {
        len = 10U;
    }
    memcpy(out, value, len);
    out[len] = '\0';
}

static int paper_gate_has_mode(const PaperGateRows *rows, const char *mode) {
    size_t i;
    for (i = 0U; i < rows->count; ++i) {
        if (strcmp(rows->items[i].mode, mode) == 0) {
            return 1;
        }
    }
    return 0;
}

static int paper_read_trace_curve(const char *path, PaperCurve *curve) {
    FILE *fp;
    char *line;
    double correct_window[PAPER_CURVE_WINDOW] = {0.0};
    double loss_window[PAPER_CURVE_WINDOW] = {0.0};
    double correct_sum = 0.0;
    double loss_sum = 0.0;
    double last_accuracy = 0.0;
    double last_loss = 0.0;
    long selected;
    long target;
    double loss;
    size_t row_count = 0U;
    long last_appended_step = 0;
    if (!paper_file_exists(path)) {
        return 0;
    }
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    while ((line = paper_read_line_dynamic(fp)) != NULL) {
        if (paper_json_get_long(line, "selected_index", &selected) &&
            paper_json_get_long(line, "target_index", &target) &&
            paper_json_get_double(line, "loss", &loss)) {
            size_t slot = row_count % PAPER_CURVE_WINDOW;
            double correct = selected == target ? 1.0 : 0.0;
            size_t denom;
            if (row_count >= PAPER_CURVE_WINDOW) {
                correct_sum -= correct_window[slot];
                loss_sum -= loss_window[slot];
            }
            correct_window[slot] = correct;
            loss_window[slot] = loss;
            correct_sum += correct;
            loss_sum += loss;
            denom = row_count + 1U < PAPER_CURVE_WINDOW ? row_count + 1U
                                                        : PAPER_CURVE_WINDOW;
            last_accuracy = correct_sum / (double)denom;
            last_loss = loss_sum / (double)denom;
            if (row_count % 25U == 0U) {
                if (paper_curve_append(curve,
                                       (long)row_count + 1L,
                                       last_accuracy,
                                       last_loss) != 0) {
                    free(line);
                    fclose(fp);
                    return -1;
                }
                last_appended_step = (long)row_count + 1L;
            }
            ++row_count;
        }
        free(line);
    }
    fclose(fp);
    if (row_count > 0U && last_appended_step != (long)row_count) {
        return paper_curve_append(curve, (long)row_count, last_accuracy, last_loss);
    }
    return 0;
}

static int paper_shell_quote(const char *value, char **out) {
    size_t len = 2U;
    const char *p;
    char *quoted;
    char *q;
    if (value == NULL) {
        value = "";
    }
    for (p = value; *p != '\0'; ++p) {
        len += *p == '\'' ? 4U : 1U;
    }
    quoted = (char *)malloc(len + 1U);
    if (quoted == NULL) {
        return -1;
    }
    q = quoted;
    *q++ = '\'';
    for (p = value; *p != '\0'; ++p) {
        if (*p == '\'') {
            memcpy(q, "'\\''", 4U);
            q += 4U;
        } else {
            *q++ = *p;
        }
    }
    *q++ = '\'';
    *q = '\0';
    *out = quoted;
    return 0;
}

static char *paper_format_command(const char *binary, const char *args) {
    char *quoted = NULL;
    char *command = NULL;
    size_t len;
    if (paper_shell_quote(binary, &quoted) != 0) {
        return NULL;
    }
    len = strlen(quoted) + 1U + strlen(args) + 1U;
    command = (char *)malloc(len);
    if (command != NULL) {
        snprintf(command, len, "%s %s", quoted, args);
    }
    free(quoted);
    return command;
}

static int paper_read_arc_epoch_eval(const char *arc_agi_path,
                                     PaperArcEpochSeries *series) {
    const unsigned epochs[PAPER_ARC_EPOCH_COUNT] = {1U, 2U, 4U, 8U};
    size_t i;
    if (!paper_file_exists(arc_agi_path)) {
        return 0;
    }
    for (i = 0U; i < PAPER_ARC_EPOCH_COUNT; ++i) {
        char args[128];
        char *command;
        FILE *pipe;
        char *line;
        snprintf(args,
                 sizeof(args),
                 "--epochs %u --test-samples 900 --seed 20240516",
                 epochs[i]);
        command = paper_format_command(arc_agi_path, args);
        if (command == NULL) {
            return -1;
        }
        pipe = popen(command, "r");
        free(command);
        if (pipe == NULL) {
            return -1;
        }
        while ((line = paper_read_line_dynamic(pipe)) != NULL) {
            char *mode;
            paper_chomp(line);
            if (paper_skip_ws(line)[0] != '{') {
                free(line);
                continue;
            }
            mode = paper_json_dup_key_string(line, "evaluation_mode");
            if (mode != NULL && strcmp(mode, "neural_cell_scorer") == 0 &&
                series->count < PAPER_ARC_EPOCH_COUNT) {
                double top1 = 0.0;
                double nll = 0.0;
                (void)paper_json_get_double(line, "top1", &top1);
                (void)paper_json_get_double(line, "nll", &nll);
                series->items[series->count].epoch = epochs[i];
                series->items[series->count].top1 = top1;
                series->items[series->count].nll = nll;
                ++series->count;
                free(mode);
                free(line);
                break;
            }
            free(mode);
            free(line);
        }
        (void)pclose(pipe);
    }
    return 0;
}

static int paper_prime_epoch_index(PaperPrimeEpochSeries *series, unsigned epoch) {
    size_t i;
    for (i = 0U; i < series->count; ++i) {
        if (series->items[i].epoch == epoch) {
            return (int)i;
        }
    }
    if (series->count >= sizeof(series->items) / sizeof(series->items[0])) {
        return -1;
    }
    series->items[series->count].epoch = epoch;
    series->items[series->count].raw_loss = NAN;
    series->items[series->count].gated_loss = NAN;
    ++series->count;
    return (int)(series->count - 1U);
}

static int paper_read_prime_gap_epoch_loss(const char *prime_gap_path,
                                           PaperPrimeEpochSeries *series) {
    char *command;
    FILE *pipe;
    char *line;
    if (!paper_file_exists(prime_gap_path)) {
        return 0;
    }
    command = paper_format_command(
        prime_gap_path,
        "100000 200000 8 0 8000 1500 1.20 0.12 20240511");
    if (command == NULL) {
        return -1;
    }
    pipe = popen(command, "r");
    free(command);
    if (pipe == NULL) {
        return -1;
    }
    while ((line = paper_read_line_dynamic(pipe)) != NULL) {
        unsigned epoch = 0U;
        double loss = 0.0;
        int index;
        if (sscanf(line, "model supervised epoch %u loss=%lf", &epoch, &loss) == 2) {
            index = paper_prime_epoch_index(series, epoch);
            if (index >= 0) {
                series->items[index].raw_loss = loss;
            }
        } else if (sscanf(line,
                          "model learned-wheel supervised epoch %u loss=%lf",
                          &epoch,
                          &loss) == 2) {
            index = paper_prime_epoch_index(series, epoch);
            if (index >= 0) {
                series->items[index].gated_loss = loss;
            }
        }
        free(line);
    }
    (void)pclose(pipe);
    return 0;
}

static int paper_parse_gate_row(const char *line,
                                const char *task,
                                PaperGateRows *rows) {
    PaperGateRow row;
    char *mode = paper_json_dup_key_string(line, "evaluation_mode");
    char *model_hash = NULL;
    char *parent_hash = NULL;
    memset(&row, 0, sizeof(row));
    if (mode == NULL || mode[0] == '\0') {
        free(mode);
        mode = paper_strdup(task);
    }
    if (mode == NULL) {
        return -1;
    }
    if (paper_gate_has_mode(rows, mode)) {
        free(mode);
        return 0;
    }
    row.mode = mode;
    row.has_accuracy = paper_json_get_double(line, "top1", &row.accuracy);
    if (!row.has_accuracy) {
        row.has_accuracy = paper_json_get_double(line, "accuracy", &row.accuracy);
    }
    row.has_nll = paper_json_get_double(line, "nll", &row.nll);
    row.has_ece = paper_json_get_double(line, "ece", &row.ece);
    if (!row.has_accuracy && !row.has_nll && !row.has_ece) {
        paper_gate_row_free(&row);
        return 0;
    }
    row.search_saved = paper_json_dup_raw_key(line, "search_saved");
    model_hash = paper_json_dup_key_string(line, "model_hash");
    parent_hash = paper_json_dup_key_string(line, "parent_hash");
    paper_first_ten(row.model_hash, model_hash);
    paper_first_ten(row.parent_hash, parent_hash);
    free(model_hash);
    free(parent_hash);
    if (paper_gate_rows_append(rows, &row) != 0) {
        paper_gate_row_free(&row);
        return -1;
    }
    return 0;
}

static void paper_store_multitask_row(const char *line, PaperMultitaskRow *row) {
    char *hidden;
    paper_multitask_row_free(row);
    memset(row, 0, sizeof(*row));
    row->present = 1;
    row->has_top1 = paper_json_get_double(line, "top1", &row->top1);
    row->has_nll = paper_json_get_double(line, "nll", &row->nll);
    row->has_ece = paper_json_get_double(line, "ece", &row->ece);
    row->has_candidate_count =
        paper_json_get_double(line, "candidate_count", &row->candidate_count);
    {
        long samples = 0L;
        row->has_sample_count = paper_json_get_long(line, "sample_count", &samples);
        row->sample_count = samples;
    }
    hidden = paper_json_dup_key_string(line, "hidden");
    row->hidden = hidden == NULL ? paper_strdup("") : hidden;
}

static int paper_read_comparison(const char *path, PaperComparison *comparison) {
    FILE *fp;
    char *line;
    if (!paper_file_exists(path)) {
        return 0;
    }
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    while ((line = paper_read_line_dynamic(fp)) != NULL) {
        char *schema;
        const char *trimmed = paper_skip_ws(line);
        if (trimmed == NULL || *trimmed != '{') {
            free(line);
            continue;
        }
        paper_chomp(line);
        schema = paper_json_dup_key_string(line, "schema");
        if (schema == NULL) {
            free(line);
            continue;
        }
        if (strcmp(schema, "crlp.compare.v1") == 0) {
            char *task = paper_json_dup_key_string(line, "task");
            int task_index = paper_domain_index(task);
            if (task_index >= 0) {
                char *mode = paper_json_dup_key_string(line, "evaluation_mode");
                if (paper_parse_gate_row(line,
                                         PAPER_DOMAINS[task_index],
                                         &comparison->gates[task_index]) != 0) {
                    free(mode);
                    free(task);
                    free(schema);
                    free(line);
                    fclose(fp);
                    return -1;
                }
                if (comparison->reliability[task_index] == NULL && mode != NULL &&
                    strcmp(mode, PAPER_RELIABILITY_MODES[task_index]) == 0) {
                    char *raw = paper_json_dup_raw_key(line, "reliability_bins");
                    if (paper_raw_array_nonempty(raw)) {
                        comparison->reliability[task_index] = raw;
                    } else {
                        free(raw);
                    }
                }
                free(mode);
            }
            free(task);
        } else if (strcmp(schema, "crlp.hypothesis.evidence.v1") == 0) {
            if (paper_raw_append(&comparison->arc_hypothesis, line) != 0) {
                free(schema);
                free(line);
                fclose(fp);
                return -1;
            }
        } else if (strcmp(schema, "crlp.multitask.summary.v1") == 0) {
            char *domain = paper_json_dup_key_string(line, "domain");
            char *kind = paper_json_dup_key_string(line, "model_kind");
            int domain_index = paper_domain_index(domain);
            if (domain_index >= 0 && kind != NULL) {
                if (strcmp(kind, "shared") == 0) {
                    paper_store_multitask_row(line, &comparison->shared[domain_index]);
                } else if (strcmp(kind, "single_task") == 0) {
                    paper_store_multitask_row(line, &comparison->single[domain_index]);
                }
            }
            free(domain);
            free(kind);
        }
        free(schema);
        free(line);
    }
    fclose(fp);
    return 0;
}

static void paper_write_json_string(FILE *fp, const char *value) {
    const unsigned char *p;
    fputc('"', fp);
    if (value != NULL) {
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
    }
    fputc('"', fp);
}

static void paper_write_json_number(FILE *fp, double value) {
    if (isnan(value)) {
        fputs("NaN", fp);
    } else {
        fprintf(fp, "%.15g", value);
    }
}

static void paper_write_curve_json(FILE *fp, const PaperCurve *curve) {
    size_t i;
    fputs("[", fp);
    if (curve->count > 0U) {
        fputc('\n', fp);
    }
    for (i = 0U; i < curve->count; ++i) {
        fputs("      [\n        ", fp);
        fprintf(fp, "%ld,\n        ", curve->items[i].step);
        paper_write_json_number(fp, curve->items[i].accuracy);
        fputs(",\n        ", fp);
        paper_write_json_number(fp, curve->items[i].loss);
        fputs("\n      ]", fp);
        fputs(i + 1U == curve->count ? "\n" : ",\n", fp);
    }
    fputs("    ]", fp);
}

static void paper_write_gate_json(FILE *fp, const PaperGateRows *rows) {
    size_t i;
    fputs("[", fp);
    if (rows->count > 0U) {
        fputc('\n', fp);
    }
    for (i = 0U; i < rows->count; ++i) {
        const PaperGateRow *row = &rows->items[i];
        fputs("      {\n        \"accuracy\": ", fp);
        if (row->has_accuracy) {
            paper_write_json_number(fp, row->accuracy);
        } else {
            fputs("null", fp);
        }
        fputs(",\n        \"ece\": ", fp);
        if (row->has_ece) {
            paper_write_json_number(fp, row->ece);
        } else {
            fputs("null", fp);
        }
        fputs(",\n        \"mode\": ", fp);
        paper_write_json_string(fp, row->mode);
        fputs(",\n        \"model_hash\": ", fp);
        paper_write_json_string(fp, row->model_hash);
        fputs(",\n        \"nll\": ", fp);
        if (row->has_nll) {
            paper_write_json_number(fp, row->nll);
        } else {
            fputs("null", fp);
        }
        fputs(",\n        \"parent_hash\": ", fp);
        paper_write_json_string(fp, row->parent_hash);
        fputs(",\n        \"search_saved\": ", fp);
        fputs(row->search_saved == NULL ? "null" : row->search_saved, fp);
        fputs("\n      }", fp);
        fputs(i + 1U == rows->count ? "\n" : ",\n", fp);
    }
    fputs("    ]", fp);
}

static void paper_write_raw_array_json(FILE *fp, const PaperRawRows *rows) {
    size_t i;
    fputs("[", fp);
    if (rows->count > 0U) {
        fputc('\n', fp);
    }
    for (i = 0U; i < rows->count; ++i) {
        fputs("    ", fp);
        fputs(rows->items[i], fp);
        fputs(i + 1U == rows->count ? "\n" : ",\n", fp);
    }
    fputs("  ]", fp);
}

static void paper_write_multitask_json(FILE *fp,
                                       const PaperMultitaskRow *shared,
                                       const PaperMultitaskRow *single) {
    size_t emitted = 0U;
    size_t i;
    fputs("[", fp);
    for (i = 0U; i < 4U; ++i) {
        if (!shared[i].present || !single[i].present) {
            continue;
        }
        fputs(emitted == 0U ? "\n" : ",\n", fp);
        fputs("    {\n      \"candidate_count\": ", fp);
        paper_write_json_number(fp, shared[i].candidate_count);
        fputs(",\n      \"domain\": ", fp);
        paper_write_json_string(fp, PAPER_DOMAINS[i]);
        fputs(",\n      \"hidden\": ", fp);
        paper_write_json_string(fp, shared[i].hidden == NULL ? "" : shared[i].hidden);
        fputs(",\n      \"samples\": ", fp);
        fprintf(fp, "%ld", shared[i].sample_count);
        fputs(",\n      \"shared_ece\": ", fp);
        if (shared[i].has_ece) {
            paper_write_json_number(fp, shared[i].ece);
        } else {
            fputs("null", fp);
        }
        fputs(",\n      \"shared_nll\": ", fp);
        paper_write_json_number(fp, shared[i].nll);
        fputs(",\n      \"shared_top1\": ", fp);
        paper_write_json_number(fp, shared[i].top1);
        fputs(",\n      \"single_ece\": ", fp);
        if (single[i].has_ece) {
            paper_write_json_number(fp, single[i].ece);
        } else {
            fputs("null", fp);
        }
        fputs(",\n      \"single_nll\": ", fp);
        paper_write_json_number(fp, single[i].nll);
        fputs(",\n      \"single_top1\": ", fp);
        paper_write_json_number(fp, single[i].top1);
        fputs("\n    }", fp);
        ++emitted;
    }
    fputs(emitted == 0U ? "]" : "\n  ]", fp);
}

static int paper_write_data_json(const char *path,
                                 const PaperCurve curves[3],
                                 const PaperArcEpochSeries *arc_epochs,
                                 const PaperPrimeEpochSeries *prime_epochs,
                                 const PaperComparison *comparison) {
    FILE *fp;
    size_t i;
    if (paper_mkdir_parent(path) != 0) {
        return -1;
    }
    fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    fputs("{\n  \"arc_epoch_eval\": [", fp);
    for (i = 0U; i < arc_epochs->count; ++i) {
        fputs(i == 0U ? "\n" : ",\n", fp);
        fputs("    [\n      ", fp);
        fprintf(fp, "%u,\n      ", arc_epochs->items[i].epoch);
        paper_write_json_number(fp, arc_epochs->items[i].top1);
        fputs(",\n      ", fp);
        paper_write_json_number(fp, arc_epochs->items[i].nll);
        fputs("\n    ]", fp);
    }
    fputs(arc_epochs->count == 0U ? "],\n" : "\n  ],\n", fp);
    fputs("  \"arc_hypothesis\": ", fp);
    paper_write_raw_array_json(fp, &comparison->arc_hypothesis);
    fputs(",\n  \"gate_ablation\": {\n", fp);
    for (i = 0U; i < 4U; ++i) {
        size_t idx = i == 0U ? 3U : i - 1U;
        fputs("    ", fp);
        paper_write_json_string(fp, PAPER_DOMAINS[idx]);
        fputs(": ", fp);
        paper_write_gate_json(fp, &comparison->gates[idx]);
        fputs(i + 1U == 4U ? "\n" : ",\n", fp);
    }
    fputs("  },\n  \"learning_curves\": {\n", fp);
    fputs("    \"delayed_claim\": ", fp);
    paper_write_curve_json(fp, &curves[2]);
    fputs(",\n    \"grid_path\": ", fp);
    paper_write_curve_json(fp, &curves[1]);
    fputs(",\n    \"prime_gap\": ", fp);
    paper_write_curve_json(fp, &curves[0]);
    fputs("\n  },\n  \"multitask_shared_vs_single\": ", fp);
    paper_write_multitask_json(fp, comparison->shared, comparison->single);
    fputs(",\n  \"prime_gap_epoch_loss\": [", fp);
    for (i = 0U; i < prime_epochs->count; ++i) {
        fputs(i == 0U ? "\n" : ",\n", fp);
        fputs("    [\n      ", fp);
        fprintf(fp, "%u,\n      ", prime_epochs->items[i].epoch);
        paper_write_json_number(fp, prime_epochs->items[i].raw_loss);
        fputs(",\n      ", fp);
        paper_write_json_number(fp, prime_epochs->items[i].gated_loss);
        fputs("\n    ]", fp);
    }
    fputs(prime_epochs->count == 0U ? "],\n" : "\n  ],\n", fp);
    fputs("  \"reliability\": {\n", fp);
    for (i = 0U; i < 4U; ++i) {
        size_t idx = i == 0U ? 3U : i - 1U;
        fputs("    ", fp);
        paper_write_json_string(fp, PAPER_DOMAINS[idx]);
        fputs(": ", fp);
        fputs(comparison->reliability[idx] == NULL ? "[]" : comparison->reliability[idx],
              fp);
        fputs(i + 1U == 4U ? "\n" : ",\n", fp);
    }
    fputs("  }\n}\n", fp);
    fclose(fp);
    return 0;
}

static void paper_write_typ_float_or_none(FILE *fp, int present, double value) {
    if (present) {
        fprintf(fp, "%.6f", value);
    } else {
        fputs("none", fp);
    }
}

static int paper_write_data_typ(const char *path,
                                const PaperCurve curves[3],
                                const PaperArcEpochSeries *arc_epochs,
                                const PaperPrimeEpochSeries *prime_epochs,
                                const PaperComparison *comparison) {
    FILE *fp;
    size_t i;
    if (paper_mkdir_parent(path) != 0) {
        return -1;
    }
    fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    fputs("// AUTOGENERATED by the C neural grammar discovery runtime.\n", fp);
    fputs("// Do not edit by hand; rerun `make paper-data` after results change.\n\n",
          fp);
    fputs("#let learning_curves = (\n", fp);
    for (i = 0U; i < 3U; ++i) {
        size_t j;
        fputs("  \"", fp);
        fputs(PAPER_DOMAINS[i], fp);
        fputs("\": (\n", fp);
        for (j = 0U; j < curves[i].count; ++j) {
            fprintf(fp,
                    "    (%ld, %.6f, %.6f),\n",
                    curves[i].items[j].step,
                    curves[i].items[j].accuracy,
                    curves[i].items[j].loss);
        }
        fputs("  ),\n", fp);
    }
    fputs(")\n\n#let arc_epoch_eval = (\n", fp);
    for (i = 0U; i < arc_epochs->count; ++i) {
        fprintf(fp,
                "  (%u, %.6f, %.6f),\n",
                arc_epochs->items[i].epoch,
                arc_epochs->items[i].top1,
                arc_epochs->items[i].nll);
    }
    fputs(")\n\n#let prime_gap_epoch_loss = (\n", fp);
    for (i = 0U; i < prime_epochs->count; ++i) {
        const char *raw = isnan(prime_epochs->items[i].raw_loss) ? "0.0" : NULL;
        const char *gated = isnan(prime_epochs->items[i].gated_loss) ? "0.0" : NULL;
        fprintf(fp, "  (%u, ", prime_epochs->items[i].epoch);
        if (raw != NULL) {
            fputs(raw, fp);
        } else {
            fprintf(fp, "%.6f", prime_epochs->items[i].raw_loss);
        }
        fputs(", ", fp);
        if (gated != NULL) {
            fputs(gated, fp);
        } else {
            fprintf(fp, "%.6f", prime_epochs->items[i].gated_loss);
        }
        fputs("),\n", fp);
    }
    fputs(")\n\n", fp);
    fputs("#let multitask_shared_vs_single = (\n", fp);
    for (i = 0U; i < 4U; ++i) {
        if (!comparison->shared[i].present || !comparison->single[i].present) {
            continue;
        }
        fprintf(fp,
                "  (domain: \"%s\", shared_top1: %.6f, single_top1: %.6f, "
                "shared_nll: %.6f, single_nll: %.6f, shared_ece: ",
                PAPER_DOMAINS[i],
                comparison->shared[i].top1,
                comparison->single[i].top1,
                comparison->shared[i].nll,
                comparison->single[i].nll);
        paper_write_typ_float_or_none(fp,
                                      comparison->shared[i].has_ece,
                                      comparison->shared[i].ece);
        fputs(", single_ece: ", fp);
        paper_write_typ_float_or_none(fp,
                                      comparison->single[i].has_ece,
                                      comparison->single[i].ece);
        fprintf(fp,
                ", hidden: \"%s\", candidate_count: %.0f, samples: %ld),\n",
                comparison->shared[i].hidden == NULL ? "" : comparison->shared[i].hidden,
                comparison->shared[i].candidate_count,
                comparison->shared[i].sample_count);
    }
    fputs(")\n\n#let gate_ablation = (\n", fp);
    for (i = 0U; i < 4U; ++i) {
        size_t j;
        fputs("  \"", fp);
        fputs(PAPER_DOMAINS[i], fp);
        fputs("\": (\n", fp);
        for (j = 0U; j < comparison->gates[i].count; ++j) {
            const PaperGateRow *row = &comparison->gates[i].items[j];
            fputs("    (mode: \"", fp);
            fputs(row->mode, fp);
            fputs("\", accuracy: ", fp);
            paper_write_typ_float_or_none(fp, row->has_accuracy, row->accuracy);
            fputs(", nll: ", fp);
            paper_write_typ_float_or_none(fp, row->has_nll, row->nll);
            fputs(", ece: ", fp);
            paper_write_typ_float_or_none(fp, row->has_ece, row->ece);
            fputs("),\n", fp);
        }
        fputs("  ),\n", fp);
    }
    fputs(")\n", fp);
    fclose(fp);
    return 0;
}

int nn_paper_write_data(const char *json_output_path,
                        const char *typ_output_path,
                        const char *comparison_path,
                        const char *prime_gap_trace_path,
                        const char *grid_path_trace_path,
                        const char *delayed_claim_trace_path,
                        const char *arc_agi_path,
                        const char *prime_gap_path) {
    PaperCurve curves[3] = {{0}, {0}, {0}};
    PaperArcEpochSeries arc_epochs = {{0}, 0U};
    PaperPrimeEpochSeries prime_epochs = {{0}, 0U};
    PaperComparison comparison;
    int status = -1;
    memset(&comparison, 0, sizeof(comparison));
    if (paper_read_trace_curve(prime_gap_trace_path, &curves[0]) != 0 ||
        paper_read_trace_curve(grid_path_trace_path, &curves[1]) != 0 ||
        paper_read_trace_curve(delayed_claim_trace_path, &curves[2]) != 0 ||
        paper_read_comparison(comparison_path, &comparison) != 0 ||
        paper_read_arc_epoch_eval(arc_agi_path, &arc_epochs) != 0 ||
        paper_read_prime_gap_epoch_loss(prime_gap_path, &prime_epochs) != 0) {
        goto done;
    }
    if (paper_write_data_json(json_output_path,
                              curves,
                              &arc_epochs,
                              &prime_epochs,
                              &comparison) != 0 ||
        paper_write_data_typ(typ_output_path,
                             curves,
                             &arc_epochs,
                             &prime_epochs,
                             &comparison) != 0) {
        goto done;
    }
    status = 0;

done:
    paper_curve_free(&curves[0]);
    paper_curve_free(&curves[1]);
    paper_curve_free(&curves[2]);
    paper_comparison_free(&comparison);
    return status;
}

#define PAPER_CLAIM_COUNT 20U
#define PAPER_CLAIM_SUMMARY_SIZE 1536U
#define PAPER_CLAIM_EXPLANATION_SIZE 1024U
#define PAPER_CLAIM_CHART_SIZE 4096U

typedef struct {
    const char *id;
    const char *family;
    const char *title;
    const char *source;
    int required;
    const char *claim_text;
    const char *explanation_text;
    const char *chart_kind;
    const char *chart_unit;
    const char *chart_y_label;
    const char *freshness_prerequisite;
    const char *refresh_command;
} PaperClaimSpec;

typedef struct {
    const PaperClaimSpec *spec;
    char status[24];
    char summary[PAPER_CLAIM_SUMMARY_SIZE];
    char explanation[PAPER_CLAIM_EXPLANATION_SIZE];
    char chart_rows[PAPER_CLAIM_CHART_SIZE];
    double raw_top1;
    double hand_top1;
    double learned_top1;
    double learned_top10;
    double learned_vs_hand_delta_pp;
    long false_rejections;
    long sample_count;
} PaperClaimResult;

typedef struct {
    char headers[80][128];
    char values[80][256];
    size_t count;
} PaperCsvRecord;

static const PaperClaimSpec PAPER_CLAIM_SPECS[PAPER_CLAIM_COUNT] = {
    {
        "C1.prime_gate_delta",
        "C1",
        "Audited next-prime filter improves top-1 over raw and hand-wheel gates",
        "results/latest/comparison.canonical.jsonl",
        1,
        "The verified learned next-prime filter improves top-1 ranking over both raw and hand-wheel baselines without verifier-counted true-answer deletions.",
        "",
        "",
        "",
        "",
        "results/latest/commands.txt",
        "make results",
    },
    {
        "C1.prime_gate_delta_ci",
        "C1",
        "Audited next-prime filter headline is stable across seeds",
        "results/latest/prime_gate_delta_seed_interval.jsonl",
        1,
        "The next-prime learned-filter headline remains stable across the registered seed interval.",
        "",
        "",
        "",
        "",
        "results/latest/comparison.canonical.jsonl",
        "make prime-gate-seed-audit",
    },
    {
        "C2.prime_proposer_fixed_budget",
        "C2",
        "Next-prime static proposer is a fixed-budget negative baseline",
        "results/latest/comparison.canonical.jsonl",
        0,
        "The static neural next-prime proposer does not beat deterministic symbolic orderings under matched budgets.",
        "",
        "",
        "",
        "",
        "results/latest/commands.txt",
        "make results",
    },
    {
        "C4.arc_ood_shared_vs_single",
        "C4",
        "Shared scorer beats single-task scorer on same-envelope ARC OOD",
        "results/2026-05-17-multitask-roadmap/rmt2_aggregate.csv",
        1,
        "The shared scorer beats the single-task scorer on same-envelope ARC OOD top-1 and log loss.",
        "This supports the narrow residual-generalization claim: the shared scorer loses on the fixed ARC training task, but improves held-out same-envelope ARC ranking and log loss.",
        "delta_bar",
        "pp",
        "shared minus single top-1",
        "",
        "",
    },
    {
        "C1.arc_gate_scorer_invariance",
        "C1",
        "Public ARC certified gate is invariant to scorer context labels",
        "results/2026-05-17-multitask-roadmap/rmt7_arc_gate_invariance_summary.csv",
        1,
        "The public ARC certified gate and proposer summaries are invariant to scorer context labels.",
        "This separates the certified ARC gate from learned scorer choice: the public ARC gate/proposer measurements are currently scorer-independent, so scorer claims and gate claims remain cleanly separated.",
        "coverage_bar",
        "correct cells",
        "certified correct cells",
        "",
        "",
    },
    {
        "C1.arc_public_evidence_surface",
        "C1",
        "Public ARC corpus parity row is measured and scoped",
        "results/2026-05-17-arc-evidence-surface/arc_evidence_summary.csv",
        1,
        "The public ARC row is a measured coverage surface, not a broad ARC competence claim.",
        "This is not a broad ARC-performance claim. It is the public-corpus parity row: the compiler can now distinguish measured ARC gate coverage, version-to-version movement, and the much larger unresolved residual.",
        "coverage_bar",
        "correct cells",
        "evaluation correct cells",
        "",
        "",
    },
    {
        "C3.arc_ood_calibration_tradeoff",
        "C3",
        "ARC shared scorer has a fixed-task versus OOD calibration tradeoff",
        "results/2026-05-17-multitask-roadmap/rmt5_calibration_summary.csv",
        1,
        "Shared ARC scoring improves some calibration surfaces while worsening others.",
        "The calibration claim is explicitly conditional: shared training is not simply better calibrated everywhere. It is worse on fixed ARC ECE but better on OOD ECE, while improving log loss in both settings.",
        "delta_bar",
        "pp",
        "shared ECE improvement vs single",
        "",
        "",
    },
    {
        "C4.delayed_claim_sibling_transfer",
        "C4",
        "Delayed-claim transfer gain is sibling-specific",
        "results/2026-05-17-multitask-roadmap/rmt3_leave_one_out.csv",
        1,
        "Delayed-claim transfer is positive in the registered shared envelope, but sibling-specific.",
        "This replaces the earlier broad positive-transfer claim with a narrower one: delayed_claim benefits from shared training in this envelope, grid_path appears to be the helpful sibling, and ARC is mild interference.",
        "delta_bar",
        "pp",
        "top-1 delta vs claim-only",
        "",
        "",
    },
    {
        "C4.multitask_capacity_tradeoff",
        "C4",
        "Capacity sweep does not explain away the ARC fixed-task gap",
        "results/2026-05-17-multitask-roadmap/rmt6_capacity_sweep_deltas.csv",
        0,
        "The capacity sweep remains diagnostic until the C7 plateau audit is ready.",
        "C7 is the blocking gate for capacity claims: width deltas stay in limitations until the affected task and variant rows are audit-ready.",
        "delta_bar",
        "pp",
        "shared minus single top-1",
        "",
        "",
    },
    {
        "C4.adapter_heads_shared_trunk",
        "C4",
        "Shared trunk plus domain heads is gated as architecture evidence",
        "results/2026-05-17-multitask-roadmap/rmt4_adapter_heads_deltas.csv",
        0,
        "Shared trunk plus task heads must pass C7 before being presented as the default recipe.",
        "The adapter-head result is treated as an architecture gate, not a broad one-model claim. C7 must be ready before a shared trunk plus domain heads can be promoted.",
        "delta_bar",
        "pp",
        "task-head top-1 delta",
        "",
        "",
    },
    {
        "C5.rejected_candidate_reason_learning",
        "C5",
        "Rejected verifier candidates become useful reason labels",
        "results/2026-05-17-substrate-extensions/rejected_candidate_reason_learning.jsonl",
        1,
        "Rejected verifier candidates can become reason labels that repair later proposals.",
        "This is the first feedback loop: rejected verifier rows become reason labels, and those labels improve the next proposal pass. The claim is scoped to the arithmetic-sequence/modular sweeps; delayed_claim remains a stress case.",
        "coverage_bar",
        "validation utility",
        "budget-8 validation utility",
        "",
        "",
    },
    {
        "C5.hypothesis_library_promotion",
        "C5",
        "Held-out hypothesis-library promotion is gated by certificates",
        "results/2026-05-17-substrate-extensions/hypothesis_library_promotions.jsonl",
        1,
        "Repeatedly safe hypotheses can be promoted into a library that helps future held-out tasks.",
        "This is the success-side feedback loop: repeatedly certified hypotheses become proposal priors only after the verifier has already accepted them.",
        "coverage_bar",
        "budget-1 certifications",
        "certified tasks",
        "",
        "",
    },
    {
        "C7.tiny_net_plateau_stop_rule",
        "C7",
        "Tiny-net architecture claims are gated by plateau audits",
        "results/2026-05-17-multitask-roadmap/tiny_net_plateau_summary.jsonl",
        0,
        "Capacity and architecture claims remain underpowered until tiny-net rows reach plateau-ready verdicts.",
        "Diagnostics separate weak signal, seed variance, non-plateaued rows, and width help before architecture claims can be promoted.",
        "",
        "",
        "",
        "",
        "",
    },
    {
        "C8.trace_determinism_audit",
        "C8",
        "Trace determinism audit catches replay-sensitive claim drift",
        "results/latest/trace_determinism.jsonl",
        1,
        "The trace determinism audit verifies that replay meaning survives numeric/runtime changes.",
        "",
        "",
        "",
        "",
        "build/double/traces/prime_gap.jsonl",
        "make trace-determinism-compare",
    },
    {
        "C8.dataset_engine_evidence_surface",
        "C8",
        "Dataset-first engine ports expose shared evidence without promoting domain claims",
        "build/float/dataset_action_score_artifacts/engine_evidence.jsonl",
        1,
        "The dataset-first action-scoring engine exposes complete candidate rows, verifier labels, shared replay/schema declarations, and engine evidence across the registered smoke adapters.",
        "This is an infrastructure claim: the rows prove that domains compile to one auditable engine surface, not that the smoke adapters replace richer domain-specific evidence.",
        "coverage_bar",
        "rows",
        "engine evidence rows",
        "build/float/dataset_action_score",
        "make dataset-action-score-engine-evidence",
    },
    {
        "C9.signal_flight_replay_control",
        "C9",
        "Signal replay model improves bounded offline control metrics",
        "results/2026-05-17-signal-flight/summary.jsonl",
        1,
        "Offline Signal replay uses bounded neural rankers without claiming ownership of safety.",
        "",
        "coverage_bar",
        "utility",
        "selected utility",
        "",
        "",
    },
    {
        "C9.signal_live_neural_active",
        "C9",
        "Live Signal pilot uses guarded checkpoint-ranked neural actions",
        "results/2026-05-17-signal-flight/live_active_summary.jsonl",
        1,
        "The live Signal pilot ran checkpoint-ranked neural actions only behind guards.",
        "",
        "coverage_bar",
        "rows",
        "live rows",
        "",
        "",
    },
    {
        "C5.arc_verified_self_play",
        "C5",
        "ARC verified self-play is a fixed-budget search-efficiency claim",
        "build/float/arc_verified_self_play/summary.jsonl",
        0,
        "Verified self-play may improve ARC proposer search only when fixed-budget public replay recovers certified oracle coverage.",
        "This claim is registered as underpowered until public eval replay, fixed-budget with/without self-play ablation, leakage disjointness, zero-family opening, and seed-count gates are all present.",
        "coverage_bar",
        "certified tasks",
        "public eval certified tasks",
        "",
        "",
    },
    {
        "C10.cross_domain_structural_transfer",
        "C10",
        "Cross-domain meta-learner certifies structural transfer with gated refusal",
        "build/float/traces/cross_domain_leave_out.jsonl",
        0,
        "Cross-domain meta-learner with typed payload regions certifies transfer on 9/11 domains.",
        "",
        "",
        "",
        "",
        "",
        "",
    },
    {
        "C9.signal_mining_replay_options",
        "C9",
        "Signal mining option-ranker evaluates bounded policy hypotheses through replay",
        "results/latest/comparison.canonical.jsonl",
        0,
        "Signal mining evaluates 420 policy grammar hypotheses through replay.",
        "",
        "",
        "",
        "",
        "",
        "",
    },
};

static void paper_appendf(char *buffer,
                          size_t buffer_size,
                          const char *fmt,
                          ...) {
    size_t used;
    va_list args;
    if (buffer == NULL || buffer_size == 0U) {
        return;
    }
    used = strlen(buffer);
    if (used >= buffer_size - 1U) {
        return;
    }
    va_start(args, fmt);
    (void)vsnprintf(buffer + used, buffer_size - used, fmt, args);
    va_end(args);
}

static void paper_claim_set_missing(PaperClaimResult *claim) {
    snprintf(claim->status, sizeof(claim->status), "%s", "missing");
    snprintf(claim->summary,
             sizeof(claim->summary),
             "missing required source artifact %s.",
             claim->spec->source);
    claim->chart_rows[0] = '\0';
}

static void paper_claim_init(PaperClaimResult *claim, const PaperClaimSpec *spec) {
    memset(claim, 0, sizeof(*claim));
    claim->spec = spec;
    paper_claim_set_missing(claim);
    snprintf(claim->explanation, sizeof(claim->explanation), "%s", spec->explanation_text);
}

static int paper_claim_has_text(const char *text) {
    return text != NULL && text[0] != '\0';
}

static int paper_path_mtime(const char *path, time_t *out_mtime) {
    struct stat st;
    if (!paper_claim_has_text(path) || out_mtime == NULL || stat(path, &st) != 0) {
        return 0;
    }
    *out_mtime = st.st_mtime;
    return 1;
}

static void paper_claim_set_stale(PaperClaimResult *claim,
                                  const char *freshness_prerequisite,
                                  const char *refresh_command,
                                  int prerequisite_missing) {
    const char *command = paper_claim_has_text(refresh_command)
                              ? refresh_command
                              : "the registered refresh command";
    snprintf(claim->status, sizeof(claim->status), "%s", "stale");
    if (prerequisite_missing) {
        snprintf(claim->summary,
                 sizeof(claim->summary),
                 "cannot verify source freshness for %s because prerequisite %s is missing; rerun `%s`.",
                 claim->spec->source,
                 freshness_prerequisite,
                 command);
    } else {
        snprintf(claim->summary,
                 sizeof(claim->summary),
                 "source artifact %s is older than freshness prerequisite %s; rerun `%s`.",
                 claim->spec->source,
                 freshness_prerequisite,
                 command);
    }
    claim->chart_rows[0] = '\0';
}

static void paper_apply_staleness_gate(PaperClaimResult claims[PAPER_CLAIM_COUNT]) {
    const double mtime_slop_seconds = 1.0;
    size_t i;
    for (i = 0U; i < PAPER_CLAIM_COUNT; ++i) {
        time_t source_mtime;
        time_t prerequisite_mtime;
        const PaperClaimSpec *spec = claims[i].spec;
        if (spec == NULL ||
            !paper_claim_has_text(spec->freshness_prerequisite) ||
            strcmp(claims[i].status, "supported") != 0) {
            continue;
        }
        if (!paper_path_mtime(spec->source, &source_mtime)) {
            paper_claim_set_missing(&claims[i]);
            continue;
        }
        if (!paper_path_mtime(spec->freshness_prerequisite, &prerequisite_mtime)) {
            paper_claim_set_stale(&claims[i],
                                  spec->freshness_prerequisite,
                                  spec->refresh_command,
                                  1);
            continue;
        }
        if (difftime(source_mtime, prerequisite_mtime) < -mtime_slop_seconds) {
            paper_claim_set_stale(&claims[i],
                                  spec->freshness_prerequisite,
                                  spec->refresh_command,
                                  0);
        }
    }
}

typedef struct {
    char *source;
    char *command;
    char *freshness_prerequisite;
} PaperClaimRegistryEvidence;

static void paper_claim_registry_evidence_free(PaperClaimRegistryEvidence *evidence) {
    if (evidence == NULL) {
        return;
    }
    free(evidence->source);
    free(evidence->command);
    free(evidence->freshness_prerequisite);
    memset(evidence, 0, sizeof(*evidence));
}

static char *paper_read_text_file(const char *path) {
    FILE *fp;
    char *content;
    size_t capacity = 4096U;
    size_t used = 0U;
    if (!paper_claim_has_text(path)) {
        return NULL;
    }
    fp = fopen(path, "rb");
    if (fp == NULL) {
        return NULL;
    }
    content = (char *)malloc(capacity);
    if (content == NULL) {
        fclose(fp);
        return NULL;
    }
    for (;;) {
        size_t available = capacity - used - 1U;
        size_t read_count;
        char *next;
        if (available == 0U) {
            capacity *= 2U;
            next = (char *)realloc(content, capacity);
            if (next == NULL) {
                free(content);
                fclose(fp);
                return NULL;
            }
            content = next;
            available = capacity - used - 1U;
        }
        read_count = fread(content + used, 1U, available, fp);
        used += read_count;
        if (read_count < available) {
            if (ferror(fp)) {
                free(content);
                fclose(fp);
                return NULL;
            }
            break;
        }
    }
    content[used] = '\0';
    fclose(fp);
    return content;
}

static const char *paper_json_find_key_value_bounded(const char *start,
                                                     const char *end,
                                                     const char *key) {
    char pattern[128];
    int written;
    const char *cursor;
    if (start == NULL || end == NULL || start > end || key == NULL || key[0] == '\0') {
        return NULL;
    }
    written = snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    if (written <= 0 || (size_t)written >= sizeof(pattern)) {
        return NULL;
    }
    cursor = start;
    while (cursor < end) {
        cursor = strstr(cursor, pattern);
        if (cursor == NULL || cursor >= end) {
            return NULL;
        }
        if (cursor + (size_t)written <= end) {
            return cursor + (size_t)written;
        }
        ++cursor;
    }
    return NULL;
}

static int paper_registry_find_claim_object(const char *content,
                                            const char *claim_id,
                                            const char **out_start,
                                            const char **out_end) {
    const char *content_end;
    const char *cursor;
    if (content == NULL || claim_id == NULL || out_start == NULL || out_end == NULL) {
        return -1;
    }
    content_end = content + strlen(content);
    cursor = content;
    while (cursor < content_end) {
        const char *id_value = paper_json_find_key_value_bounded(cursor, content_end, "id");
        char *parsed_id;
        const char *object_start;
        const char *object_end;
        if (id_value == NULL) {
            return 0;
        }
        parsed_id = paper_json_dup_string_value(id_value);
        if (parsed_id == NULL) {
            return -1;
        }
        if (strcmp(parsed_id, claim_id) != 0) {
            free(parsed_id);
            cursor = id_value + 1;
            continue;
        }
        free(parsed_id);
        object_start = id_value;
        while (object_start > content && *object_start != '{') {
            --object_start;
        }
        if (*object_start != '{') {
            return -1;
        }
        object_end = paper_json_value_end(object_start);
        if (object_end == NULL || object_end > content_end) {
            return -1;
        }
        *out_start = object_start;
        *out_end = object_end;
        return 1;
    }
    return 0;
}

static int paper_registry_read_claim_evidence(const char *content,
                                              const char *claim_id,
                                              PaperClaimRegistryEvidence *out) {
    const char *claim_start;
    const char *claim_end;
    const char *evidence_start;
    const char *evidence_end;
    int found;
    if (out == NULL) {
        return -1;
    }
    memset(out, 0, sizeof(*out));
    found = paper_registry_find_claim_object(content, claim_id, &claim_start, &claim_end);
    if (found <= 0) {
        return found;
    }
    evidence_start = paper_skip_ws(
        paper_json_find_key_value_bounded(claim_start, claim_end, "evidence"));
    if (evidence_start == NULL || *evidence_start != '{') {
        return -1;
    }
    evidence_end = paper_json_value_end(evidence_start);
    if (evidence_end == NULL || evidence_end > claim_end) {
        return -1;
    }
    out->source = paper_json_dup_string_value(
        paper_json_find_key_value_bounded(evidence_start, evidence_end, "source"));
    out->command = paper_json_dup_string_value(
        paper_json_find_key_value_bounded(evidence_start, evidence_end, "command"));
    out->freshness_prerequisite = paper_json_dup_string_value(
        paper_json_find_key_value_bounded(evidence_start,
                                          evidence_end,
                                          "freshness_prerequisite"));
    if (out->source == NULL || out->command == NULL || out->freshness_prerequisite == NULL) {
        paper_claim_registry_evidence_free(out);
        return -1;
    }
    return paper_claim_has_text(out->source) && paper_claim_has_text(out->command) ? 1 : -1;
}

static int paper_validate_claim_registry(const char *claims_path,
                                         char *error,
                                         size_t error_size) {
    char *content;
    size_t i;
    if (error != NULL && error_size > 0U) {
        error[0] = '\0';
    }
    content = paper_read_text_file(claims_path);
    if (content == NULL) {
        snprintf(error, error_size, "cannot read claim registry %s", claims_path);
        return -1;
    }
    for (i = 0U; i < PAPER_CLAIM_COUNT; ++i) {
        PaperClaimRegistryEvidence evidence;
        const PaperClaimSpec *spec = &PAPER_CLAIM_SPECS[i];
        int found = paper_registry_read_claim_evidence(content, spec->id, &evidence);
        if (found <= 0) {
            snprintf(error,
                     error_size,
                     "claim registry is missing evidence metadata for %s",
                     spec->id);
            free(content);
            return -1;
        }
        if (strcmp(evidence.source, spec->source) != 0) {
            snprintf(error,
                     error_size,
                     "claim %s source mismatch: registry=%s runtime=%s",
                     spec->id,
                     evidence.source,
                     spec->source);
            paper_claim_registry_evidence_free(&evidence);
            free(content);
            return -1;
        }
        if (paper_claim_has_text(spec->refresh_command) &&
            strcmp(evidence.command, spec->refresh_command) != 0) {
            snprintf(error,
                     error_size,
                     "claim %s refresh command mismatch: registry=%s runtime=%s",
                     spec->id,
                     evidence.command,
                     spec->refresh_command);
            paper_claim_registry_evidence_free(&evidence);
            free(content);
            return -1;
        }
        if (paper_claim_has_text(spec->freshness_prerequisite) &&
            strcmp(evidence.freshness_prerequisite, spec->freshness_prerequisite) != 0) {
            snprintf(error,
                     error_size,
                     "claim %s freshness prerequisite mismatch: registry=%s runtime=%s",
                     spec->id,
                     evidence.freshness_prerequisite,
                     spec->freshness_prerequisite);
            paper_claim_registry_evidence_free(&evidence);
            free(content);
            return -1;
        }
        paper_claim_registry_evidence_free(&evidence);
    }
    free(content);
    return 0;
}

static int paper_json_get_bool(const char *line, const char *key, int *out) {
    const char *cursor = paper_skip_ws(paper_json_find_key_value(line, key));
    if (cursor == NULL) {
        return 0;
    }
    if (strncmp(cursor, "true", 4U) == 0) {
        *out = 1;
        return 1;
    }
    if (strncmp(cursor, "false", 5U) == 0) {
        *out = 0;
        return 1;
    }
    return 0;
}

static void paper_line_list_free(PaperLineList *list) {
    size_t i;
    if (list == NULL) {
        return;
    }
    for (i = 0U; i < list->count; ++i) {
        free(list->items[i]);
    }
    free(list->items);
    memset(list, 0, sizeof(*list));
}

static int paper_line_list_append_owned(PaperLineList *list, char *line) {
    char **items;
    size_t capacity;
    if (list == NULL || line == NULL) {
        free(line);
        return -1;
    }
    if (list->count + 1U > list->capacity) {
        capacity = list->capacity == 0U ? 256U : list->capacity * 2U;
        items = (char **)realloc(list->items, capacity * sizeof(char *));
        if (items == NULL) {
            free(line);
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = line;
    return 0;
}

static int paper_dyn_appendf(char **buffer,
                             size_t *used,
                             size_t *capacity,
                             const char *fmt,
                             ...) {
    va_list args;
    int needed;
    char *next;
    for (;;) {
        if (*buffer == NULL || *capacity == 0U) {
            *capacity = 256U;
            *used = 0U;
            *buffer = (char *)malloc(*capacity);
            if (*buffer == NULL) {
                return -1;
            }
            (*buffer)[0] = '\0';
        }
        va_start(args, fmt);
        needed = vsnprintf(*buffer + *used, *capacity - *used, fmt, args);
        va_end(args);
        if (needed < 0) {
            return -1;
        }
        if (*used + (size_t)needed < *capacity) {
            *used += (size_t)needed;
            return 0;
        }
        while (*used + (size_t)needed >= *capacity) {
            *capacity *= 2U;
        }
        next = (char *)realloc(*buffer, *capacity);
        if (next == NULL) {
            return -1;
        }
        *buffer = next;
    }
}

static int paper_trace_canonical_append(PaperLineList *lines,
                                        const NNParsedTraceRecord *record,
                                        const uint32_t *action_ids) {
    char *line = NULL;
    size_t used = 0U;
    size_t capacity = 0U;
    size_t i;
    int sensitive = fabs((double)record->selected_score_margin) <=
                    (double)record->rank_score_epsilon;
    if (paper_dyn_appendf(&line,
                          &used,
                          &capacity,
                          "{\"schema\":\"crlp.trace.semantic.v1\","
                          "\"source_schema\":\"crlp.trace.v1\","
                          "\"row_index\":%zu,"
                          "\"task\":\"%s\","
                          "\"trace_version\":%u,"
                          "\"action_id_contract\":\"%s\","
                          "\"state_hash\":\"%016llx\","
                          "\"feature_encoder_version\":%u,"
                          "\"candidate_count\":%zu,"
                          "\"action_ids\":[",
                          lines->count,
                          record->task_name,
                          (unsigned)record->trace_version,
                          record->action_id_contract,
                          (unsigned long long)record->state_hash,
                          (unsigned)record->feature_encoder_version,
                          record->candidate_count) != 0) {
        free(line);
        return -1;
    }
    for (i = 0U; i < record->candidate_count; ++i) {
        if (paper_dyn_appendf(&line,
                              &used,
                              &capacity,
                              "%s%u",
                              i == 0U ? "" : ",",
                              (unsigned)action_ids[i]) != 0) {
            free(line);
            return -1;
        }
    }
    if (paper_dyn_appendf(&line,
                          &used,
                          &capacity,
                          "],\"selected_action_id\":%u,"
                          "\"target_action_id\":%u,"
                          "\"verifier_result\":\"%s\","
                          "\"seed\":%llu,"
                          "\"ranking_policy\":\"%s\","
                          "\"rank_fp_sensitive\":%s}",
                          (unsigned)record->selected_action_id,
                          (unsigned)record->target_action_id,
                          record->verifier_result,
                          (unsigned long long)record->seed,
                          record->ranking_policy,
                          sensitive ? "true" : "false") != 0) {
        free(line);
        return -1;
    }
    return paper_line_list_append_owned(lines, line);
}

static uint64_t paper_trace_hash_lines(const PaperLineList *lines) {
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t i;
    for (i = 0U; i < lines->count; ++i) {
        const unsigned char *p = (const unsigned char *)lines->items[i];
        while (*p != '\0') {
            hash ^= (uint64_t)(*p++);
            hash *= UINT64_C(1099511628211);
        }
        hash ^= (uint64_t)'\n';
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int paper_trace_lines_equal(const PaperLineList *a, const PaperLineList *b) {
    size_t i;
    if (a->count != b->count) {
        return 0;
    }
    for (i = 0U; i < a->count; ++i) {
        if (strcmp(a->items[i], b->items[i]) != 0) {
            return 0;
        }
    }
    return 1;
}

static int paper_trace_write_canonical_file(const char *path,
                                            const PaperLineList *lines) {
    FILE *fp;
    size_t i;
    if (path == NULL || path[0] == '\0') {
        return 0;
    }
    if (paper_mkdir_parent(path) != 0) {
        return -1;
    }
    fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    for (i = 0U; i < lines->count; ++i) {
        if (fprintf(fp, "%s\n", lines->items[i]) < 0) {
            fclose(fp);
            return -1;
        }
    }
    if (fclose(fp) != 0) {
        return -1;
    }
    return 0;
}

static int paper_trace_audit_one(const char *path,
                                 PaperLineList *canonical_lines,
                                 long *row_count_out,
                                 long *invalid_rows_out,
                                 long *sensitive_rows_out,
                                 long *logged_mismatch_out,
                                 double *min_margin_out,
                                 double *max_epsilon_out) {
    FILE *fp;
    char *line;
    long row_count = 0;
    long invalid_rows = 0;
    long sensitive_rows = 0;
    long logged_mismatch = 0;
    double min_margin = 0.0;
    double max_epsilon = 0.0;
    int have_margin = 0;
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    while ((line = paper_read_line_dynamic(fp)) != NULL) {
        long candidate_count_long = 0;
        size_t candidate_count;
        uint32_t *action_ids = NULL;
        nn_real *scores = NULL;
        nn_real *probabilities = NULL;
        NNParsedTraceRecord record;
        int logged_sensitive;
        int computed_sensitive;
        paper_chomp(line);
        if (line[0] != '{') {
            free(line);
            continue;
        }
        if (!paper_json_get_long(line, "candidate_count", &candidate_count_long) ||
            candidate_count_long <= 0) {
            ++invalid_rows;
            free(line);
            continue;
        }
        candidate_count = (size_t)candidate_count_long;
        action_ids = (uint32_t *)calloc(candidate_count, sizeof(uint32_t));
        scores = (nn_real *)calloc(candidate_count, sizeof(nn_real));
        probabilities = (nn_real *)calloc(candidate_count, sizeof(nn_real));
        if (action_ids == NULL || scores == NULL || probabilities == NULL ||
            !nn_trace_parse_jsonl(line,
                                  &record,
                                  action_ids,
                                  scores,
                                  probabilities,
                                  candidate_count)) {
            ++invalid_rows;
            free(action_ids);
            free(scores);
            free(probabilities);
            free(line);
            continue;
        }
        computed_sensitive = fabs((double)record.selected_score_margin) <=
                             (double)record.rank_score_epsilon;
        if (computed_sensitive) {
            ++sensitive_rows;
        }
        if (paper_json_get_bool(line, "rank_fp_sensitive", &logged_sensitive) &&
            logged_sensitive != computed_sensitive) {
            ++logged_mismatch;
        }
        if (!have_margin || (double)record.selected_score_margin < min_margin) {
            min_margin = (double)record.selected_score_margin;
            have_margin = 1;
        }
        if ((double)record.rank_score_epsilon > max_epsilon) {
            max_epsilon = (double)record.rank_score_epsilon;
        }
        if (paper_trace_canonical_append(canonical_lines, &record, action_ids) != 0) {
            free(action_ids);
            free(scores);
            free(probabilities);
            free(line);
            fclose(fp);
            return -1;
        }
        ++row_count;
        free(action_ids);
        free(scores);
        free(probabilities);
        free(line);
    }
    if (fclose(fp) != 0) {
        return -1;
    }
    *row_count_out = row_count;
    *invalid_rows_out = invalid_rows;
    *sensitive_rows_out = sensitive_rows;
    *logged_mismatch_out = logged_mismatch;
    *min_margin_out = have_margin ? min_margin : 0.0;
    *max_epsilon_out = max_epsilon;
    return 0;
}

int nn_paper_write_trace_determinism(const char *const *trace_paths,
                                     size_t trace_path_count,
                                     const char *output_path,
                                     const char *canonical_output_path,
                                     int gate) {
    FILE *fp;
    PaperLineList *canonical = NULL;
    long *row_counts = NULL;
    uint64_t *hashes = NULL;
    long total_invalid = 0;
    long total_sensitive = 0;
    long total_logged_mismatch = 0;
    int all_equal = 1;
    int row_counts_equal = 1;
    int status = -1;
    size_t i;
    if (trace_paths == NULL || trace_path_count == 0U || output_path == NULL) {
        return -1;
    }
    if (canonical_output_path != NULL && canonical_output_path[0] != '\0' &&
        trace_path_count != 1U) {
        return -1;
    }
    canonical = (PaperLineList *)calloc(trace_path_count, sizeof(PaperLineList));
    row_counts = (long *)calloc(trace_path_count, sizeof(long));
    hashes = (uint64_t *)calloc(trace_path_count, sizeof(uint64_t));
    if (canonical == NULL || row_counts == NULL || hashes == NULL ||
        paper_mkdir_parent(output_path) != 0) {
        goto cleanup;
    }
    fp = fopen(output_path, "w");
    if (fp == NULL) {
        goto cleanup;
    }
    for (i = 0U; i < trace_path_count; ++i) {
        long invalid = 0;
        long sensitive = 0;
        long logged_mismatch = 0;
        double min_margin = 0.0;
        double max_epsilon = 0.0;
        if (paper_trace_audit_one(trace_paths[i],
                                  &canonical[i],
                                  &row_counts[i],
                                  &invalid,
                                  &sensitive,
                                  &logged_mismatch,
                                  &min_margin,
                                  &max_epsilon) != 0) {
            fclose(fp);
            goto cleanup;
        }
        hashes[i] = paper_trace_hash_lines(&canonical[i]);
        total_invalid += invalid;
        total_sensitive += sensitive;
        total_logged_mismatch += logged_mismatch;
        fputs("{\"schema\":\"crlp.trace_determinism_audit.v1\",", fp);
        fputs("\"trace_path\":", fp);
        paper_write_json_string(fp, trace_paths[i]);
        fprintf(fp,
                ",\"row_count\":%ld,"
                "\"semantic_hash\":\"%016llx\","
                "\"semantic_schema\":\"crlp.trace.semantic.v1\","
                "\"rank_fp_sensitive_rows\":%ld,"
                "\"rank_fp_sensitive_rate\":%.15g,"
                "\"rank_fp_sensitive_logged_mismatches\":%ld,"
                "\"negative_margin_rows\":%ld,"
                "\"score_margin_min\":%.15g,"
                "\"rank_score_epsilon_max\":%.15g,"
                "\"invalid_rows\":%ld}\n",
                row_counts[i],
                (unsigned long long)hashes[i],
                sensitive,
                row_counts[i] == 0 ? 0.0 : (double)sensitive / (double)row_counts[i],
                logged_mismatch,
                min_margin < 0.0 ? 1L : 0L,
                min_margin,
                max_epsilon,
                invalid);
    }
    if (trace_path_count > 1U) {
        for (i = 1U; i < trace_path_count; ++i) {
            if (!paper_trace_lines_equal(&canonical[0], &canonical[i])) {
                all_equal = 0;
            }
            if (row_counts[i] != row_counts[0]) {
                row_counts_equal = 0;
            }
        }
        fputs("{\"schema\":\"crlp.trace_determinism_compare.v1\",", fp);
        fputs("\"baseline_trace_path\":", fp);
        paper_write_json_string(fp, trace_paths[0]);
        fputs(",\"trace_paths\":[", fp);
        for (i = 0U; i < trace_path_count; ++i) {
            if (i > 0U) {
                fputc(',', fp);
            }
            paper_write_json_string(fp, trace_paths[i]);
        }
        fputs("],\"semantic_hashes\":{", fp);
        for (i = 0U; i < trace_path_count; ++i) {
            if (i > 0U) {
                fputc(',', fp);
            }
            paper_write_json_string(fp, trace_paths[i]);
            fprintf(fp, ":\"%016llx\"", (unsigned long long)hashes[i]);
        }
        fputs("},\"row_counts\":{", fp);
        for (i = 0U; i < trace_path_count; ++i) {
            if (i > 0U) {
                fputc(',', fp);
            }
            paper_write_json_string(fp, trace_paths[i]);
            fprintf(fp, ":%ld", row_counts[i]);
        }
        fprintf(fp,
                "},\"all_semantic_hashes_equal\":%s,"
                "\"row_counts_equal\":%s,"
                "\"differing_paths\":[]}\n",
                all_equal ? "true" : "false",
                row_counts_equal ? "true" : "false");
    }
    if (gate) {
        int passed = total_invalid == 0 && total_sensitive == 0 &&
                     total_logged_mismatch == 0 && all_equal && row_counts_equal &&
                     row_counts[0] > 0;
        fprintf(fp,
                "{\"schema\":\"crlp.trace_determinism_gate.v1\","
                "\"passed\":%s,\"failure_count\":%d}\n",
                passed ? "true" : "false",
                passed ? 0 : 1);
        if (!passed) {
            fclose(fp);
            goto cleanup;
        }
    }
    if (fclose(fp) != 0) {
        goto cleanup;
    }
    if (canonical_output_path != NULL && canonical_output_path[0] != '\0' &&
        paper_trace_write_canonical_file(canonical_output_path, &canonical[0]) != 0) {
        goto cleanup;
    }
    status = 0;
cleanup:
    if (canonical != NULL) {
        for (i = 0U; i < trace_path_count; ++i) {
            paper_line_list_free(&canonical[i]);
        }
    }
    free(canonical);
    free(row_counts);
    free(hashes);
    return status;
}

static int paper_json_string_equals(const char *line,
                                    const char *key,
                                    const char *expected) {
    char *value = paper_json_dup_key_string(line, key);
    int equal;
    if (value == NULL) {
        return 0;
    }
    equal = strcmp(value, expected) == 0;
    free(value);
    return equal;
}

static int paper_json_get_nested_double(const char *line,
                                        const char *outer_key,
                                        const char *inner_key,
                                        double *out) {
    const char *cursor = paper_json_find_key_value(line, outer_key);
    if (cursor == NULL) {
        return 0;
    }
    return paper_json_get_double(cursor, inner_key, out);
}

static int paper_read_first_jsonl(const char *path, char **out_line) {
    FILE *fp;
    char *line;
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    while ((line = paper_read_line_dynamic(fp)) != NULL) {
        paper_chomp(line);
        if (line[0] != '\0') {
            fclose(fp);
            *out_line = line;
            return 1;
        }
        free(line);
    }
    fclose(fp);
    return 0;
}

static long paper_count_jsonl_rows(const char *path) {
    FILE *fp;
    char *line;
    long count = 0;
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    while ((line = paper_read_line_dynamic(fp)) != NULL) {
        paper_chomp(line);
        if (line[0] != '\0') {
            ++count;
        }
        free(line);
    }
    fclose(fp);
    return count;
}

static char *paper_trim_in_place(char *value) {
    char *end;
    while (*value != '\0' && isspace((unsigned char)*value)) {
        ++value;
    }
    end = value + strlen(value);
    while (end > value && isspace((unsigned char)end[-1])) {
        *--end = '\0';
    }
    return value;
}

static size_t paper_csv_split_in_place(char *line, char **fields, size_t max_fields) {
    char *read;
    char *write;
    int in_quote = 0;
    size_t count = 0U;
    if (max_fields == 0U) {
        return 0U;
    }
    fields[count++] = line;
    write = line;
    for (read = line; *read != '\0'; ++read) {
        if (*read == '"') {
            in_quote = !in_quote;
            continue;
        }
        if (*read == ',' && !in_quote) {
            *write++ = '\0';
            if (count < max_fields) {
                fields[count++] = write;
            }
            continue;
        }
        *write++ = *read;
    }
    *write = '\0';
    return count;
}

static void paper_csv_record_copy(PaperCsvRecord *record,
                                  char **headers,
                                  size_t header_count,
                                  char **values,
                                  size_t value_count) {
    size_t i;
    record->count = header_count < 80U ? header_count : 80U;
    for (i = 0U; i < record->count; ++i) {
        char *header = paper_trim_in_place(headers[i]);
        char *value = i < value_count ? paper_trim_in_place(values[i]) : "";
        snprintf(record->headers[i], sizeof(record->headers[i]), "%s", header);
        snprintf(record->values[i], sizeof(record->values[i]), "%s", value);
    }
}

static const char *paper_csv_record_get(const PaperCsvRecord *record,
                                        const char *name) {
    size_t i;
    for (i = 0U; i < record->count; ++i) {
        if (strcmp(record->headers[i], name) == 0) {
            return record->values[i];
        }
    }
    return NULL;
}

static int paper_csv_record_get_double(const PaperCsvRecord *record,
                                       const char *name,
                                       double *out) {
    const char *value = paper_csv_record_get(record, name);
    char *end = NULL;
    double parsed;
    if (value == NULL || value[0] == '\0') {
        return 0;
    }
    errno = 0;
    parsed = strtod(value, &end);
    if (end == value || errno == ERANGE || !isfinite(parsed)) {
        return 0;
    }
    *out = parsed;
    return 1;
}

static int paper_csv_record_get_long(const PaperCsvRecord *record,
                                     const char *name,
                                     long *out) {
    const char *value = paper_csv_record_get(record, name);
    char *end = NULL;
    long parsed;
    if (value == NULL || value[0] == '\0') {
        return 0;
    }
    errno = 0;
    parsed = strtol(value, &end, 10);
    if (end == value || errno == ERANGE) {
        return 0;
    }
    *out = parsed;
    return 1;
}

static int paper_csv_load_matching_row(const char *path,
                                       const char **pred_cols,
                                       const char **pred_vals,
                                       size_t pred_count,
                                       PaperCsvRecord *record) {
    FILE *fp;
    char *header_line;
    char *line;
    char *headers[80];
    char *values[80];
    size_t header_count;
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    header_line = paper_read_line_dynamic(fp);
    if (header_line == NULL) {
        fclose(fp);
        return 0;
    }
    paper_chomp(header_line);
    header_count = paper_csv_split_in_place(header_line, headers, 80U);
    while ((line = paper_read_line_dynamic(fp)) != NULL) {
        size_t value_count;
        size_t i;
        int matched = 1;
        paper_chomp(line);
        if (line[0] == '\0') {
            free(line);
            continue;
        }
        value_count = paper_csv_split_in_place(line, values, 80U);
        paper_csv_record_copy(record, headers, header_count, values, value_count);
        for (i = 0U; i < pred_count; ++i) {
            const char *actual = paper_csv_record_get(record, pred_cols[i]);
            if (actual == NULL || strcmp(actual, pred_vals[i]) != 0) {
                matched = 0;
                break;
            }
        }
        free(line);
        if (matched) {
            free(header_line);
            fclose(fp);
            return 1;
        }
    }
    free(header_line);
    fclose(fp);
    return 0;
}

static long paper_csv_count_rows(const char *path) {
    FILE *fp;
    char *line;
    long count = 0;
    int saw_header = 0;
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    while ((line = paper_read_line_dynamic(fp)) != NULL) {
        paper_chomp(line);
        if (line[0] != '\0') {
            if (saw_header) {
                ++count;
            } else {
                saw_header = 1;
            }
        }
        free(line);
    }
    fclose(fp);
    return count;
}

static void paper_claim_slug(const char *id, char out[96]) {
    size_t i;
    size_t j = 0U;
    for (i = 0U; id[i] != '\0' && j + 1U < 96U; ++i) {
        unsigned char ch = (unsigned char)id[i];
        out[j++] = isalnum(ch) ? (char)ch : '_';
    }
    out[j] = '\0';
}

static void paper_write_typ_string(FILE *fp, const char *value) {
    const unsigned char *cursor = (const unsigned char *)(value == NULL ? "" : value);
    fputc('"', fp);
    while (*cursor != '\0') {
        if (*cursor == '"' || *cursor == '\\') {
            fputc('\\', fp);
            fputc((int)*cursor, fp);
        } else if (*cursor == '\n') {
            fputs("\\n", fp);
        } else if (*cursor == '\r') {
            fputs("\\r", fp);
        } else if (*cursor == '\t') {
            fputs("\\t", fp);
        } else {
            fputc((int)*cursor, fp);
        }
        ++cursor;
    }
    fputc('"', fp);
}

static void paper_write_typ_float(FILE *fp, double value) {
    if (!isfinite(value)) {
        fputs("none", fp);
    } else {
        fprintf(fp, "%.15g", value);
    }
}

static int paper_write_text_file(const char *path, const char *text) {
    FILE *fp;
    if (paper_mkdir_parent(path) != 0) {
        return -1;
    }
    fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    fputs(text, fp);
    fclose(fp);
    return 0;
}

#ifdef NN_ENABLE_TEST_API
static int paper_test_set_mtime(const char *path, time_t mtime) {
    struct utimbuf times;
    times.actime = mtime;
    times.modtime = mtime;
    return utime(path, &times) == 0;
}

static int paper_test_write_timed_file(const char *path, time_t mtime) {
    return paper_write_text_file(path, "paper claim governance test fixture\n") == 0 &&
           paper_test_set_mtime(path, mtime);
}

static int paper_test_staleness_case(const char *source_path,
                                     const char *freshness_prerequisite,
                                     const char *initial_status,
                                     const char *expected_status,
                                     const char *expected_summary_substring) {
    PaperClaimResult claims[PAPER_CLAIM_COUNT];
    PaperClaimSpec spec = {
        .id = "T.paper_staleness_fixture",
        .family = "T",
        .title = "Paper staleness fixture",
        .source = source_path,
        .required = 1,
        .claim_text = "fixture",
        .explanation_text = "",
        .chart_kind = "",
        .chart_unit = "",
        .chart_y_label = "",
        .freshness_prerequisite = freshness_prerequisite,
        .refresh_command = "make test-refresh",
    };
    memset(claims, 0, sizeof(claims));
    paper_claim_init(&claims[0], &spec);
    snprintf(claims[0].status, sizeof(claims[0].status), "%s", initial_status);
    snprintf(claims[0].summary, sizeof(claims[0].summary), "%s", "supported fixture");
    paper_apply_staleness_gate(claims);
    return strcmp(claims[0].status, expected_status) == 0 &&
           (expected_summary_substring == NULL ||
            strstr(claims[0].summary, expected_summary_substring) != NULL);
}

int nn_paper_test_claim_governance_contract(const char *work_dir,
                                            const char *claims_path) {
    char source_old[4096];
    char source_new[4096];
    char prereq_old[4096];
    char prereq_new[4096];
    char prereq_missing[4096];
    char registry_error[512];
    int written;
    if (!paper_claim_has_text(work_dir) || !paper_claim_has_text(claims_path)) {
        return 0;
    }
    if (paper_mkdirs(work_dir) != 0) {
        return 0;
    }
    written = snprintf(source_old, sizeof(source_old), "%s/source_old.txt", work_dir);
    if (written <= 0 || (size_t)written >= sizeof(source_old)) {
        return 0;
    }
    written = snprintf(source_new, sizeof(source_new), "%s/source_new.txt", work_dir);
    if (written <= 0 || (size_t)written >= sizeof(source_new)) {
        return 0;
    }
    written = snprintf(prereq_old, sizeof(prereq_old), "%s/prereq_old.txt", work_dir);
    if (written <= 0 || (size_t)written >= sizeof(prereq_old)) {
        return 0;
    }
    written = snprintf(prereq_new, sizeof(prereq_new), "%s/prereq_new.txt", work_dir);
    if (written <= 0 || (size_t)written >= sizeof(prereq_new)) {
        return 0;
    }
    written = snprintf(prereq_missing,
                       sizeof(prereq_missing),
                       "%s/prereq_missing.txt",
                       work_dir);
    if (written <= 0 || (size_t)written >= sizeof(prereq_missing)) {
        return 0;
    }
    (void)remove(prereq_missing);
    if (!paper_test_write_timed_file(source_old, (time_t)1000) ||
        !paper_test_write_timed_file(source_new, (time_t)1010) ||
        !paper_test_write_timed_file(prereq_old, (time_t)1000) ||
        !paper_test_write_timed_file(prereq_new, (time_t)1005)) {
        return 0;
    }
    registry_error[0] = '\0';
    return paper_validate_claim_registry(claims_path,
                                         registry_error,
                                         sizeof(registry_error)) == 0 &&
           paper_test_staleness_case(source_old,
                                     prereq_new,
                                     "supported",
                                     "stale",
                                     "older than freshness prerequisite") &&
           paper_test_staleness_case(source_new,
                                     prereq_old,
                                     "supported",
                                     "supported",
                                     "supported fixture") &&
           paper_test_staleness_case(source_new,
                                     prereq_missing,
                                     "supported",
                                     "stale",
                                     "cannot verify source freshness") &&
           paper_test_staleness_case(source_old,
                                     prereq_new,
                                     "contradicted",
                                     "contradicted",
                                     "supported fixture");
}
#endif

static void paper_eval_prime_gate_delta(PaperClaimResult *claim) {
    FILE *fp;
    char *line;
    int have_raw = 0;
    int have_hand = 0;
    int have_learned = 0;
    fp = fopen(claim->spec->source, "r");
    if (fp == NULL) {
        return;
    }
    while ((line = paper_read_line_dynamic(fp)) != NULL) {
        char *mode = NULL;
        if (!paper_json_string_equals(line, "schema", "crlp.compare.v1") ||
            !paper_json_string_equals(line, "task", "prime_gap") ||
            !paper_json_string_equals(line, "source_target", "prime-gap-eval") ||
            !paper_json_string_equals(line, "feature_set", "minimal")) {
            free(line);
            continue;
        }
        mode = paper_json_dup_key_string(line, "evaluation_mode");
        if (mode != NULL && strcmp(mode, "raw") == 0 && !have_raw) {
            have_raw = paper_json_get_double(line, "top1", &claim->raw_top1);
        } else if (mode != NULL && strcmp(mode, "hand_wheel_model") == 0 &&
                   !have_hand) {
            have_hand = paper_json_get_double(line, "top1", &claim->hand_top1);
        } else if (mode != NULL && strcmp(mode, "expanded_rule_model") == 0 &&
                   !have_learned) {
            long false_rejections = 0;
            long sample_count = 0;
            have_learned = paper_json_get_double(line, "top1", &claim->learned_top1);
            (void)paper_json_get_double(line, "top10", &claim->learned_top10);
            (void)paper_json_get_long(line, "false_rejections", &false_rejections);
            (void)paper_json_get_long(line, "sample_count", &sample_count);
            claim->false_rejections = false_rejections;
            claim->sample_count = sample_count;
        }
        free(mode);
        free(line);
        if (have_raw && have_hand && have_learned) {
            break;
        }
    }
    fclose(fp);
    if (!(have_raw && have_hand && have_learned)) {
        return;
    }
    claim->learned_vs_hand_delta_pp = 100.0 * (claim->learned_top1 - claim->hand_top1);
    snprintf(claim->status,
             sizeof(claim->status),
             "%s",
             claim->learned_top1 > claim->hand_top1 && claim->hand_top1 > claim->raw_top1 &&
                     claim->false_rejections == 0
                 ? "supported"
                 : "contradicted");
    snprintf(claim->summary,
             sizeof(claim->summary),
             "the verified learned filter raises top-1 ranking to %.2f%% from %.2f%% for the hand-coded wheel and %.2f%% raw, with %ld verifier-counted true-answer deletions.",
             100.0 * claim->learned_top1,
             100.0 * claim->hand_top1,
             100.0 * claim->raw_top1,
             claim->false_rejections);
}

static void paper_eval_prime_gate_delta_ci(PaperClaimResult *claim) {
    FILE *fp;
    char *line;
    fp = fopen(claim->spec->source, "r");
    if (fp == NULL) {
        return;
    }
    while ((line = paper_read_line_dynamic(fp)) != NULL) {
        if (paper_json_string_equals(line, "row_type", "summary")) {
            double raw_mean = 0.0;
            double raw_std = 0.0;
            double hand_mean = 0.0;
            double hand_std = 0.0;
            double learned_mean = 0.0;
            double learned_std = 0.0;
            double delta_min = 0.0;
            double learned_top10_min = 0.0;
            long seed_count = 0;
            long false_rejections_max = 0;
            long sample_count_min = 0;
            (void)paper_json_get_long(line, "seed_count", &seed_count);
            (void)paper_json_get_nested_double(line, "raw_top1", "mean", &raw_mean);
            (void)paper_json_get_nested_double(line, "raw_top1", "stdev", &raw_std);
            (void)paper_json_get_nested_double(line, "hand_top1", "mean", &hand_mean);
            (void)paper_json_get_nested_double(line, "hand_top1", "stdev", &hand_std);
            (void)paper_json_get_nested_double(line, "learned_top1", "mean", &learned_mean);
            (void)paper_json_get_nested_double(line, "learned_top1", "stdev", &learned_std);
            (void)paper_json_get_nested_double(
                line, "learned_vs_hand_delta_pp", "min", &delta_min);
            (void)paper_json_get_double(line, "learned_top10_min", &learned_top10_min);
            (void)paper_json_get_long(line, "false_rejections_max", &false_rejections_max);
            (void)paper_json_get_long(line, "sample_count_min", &sample_count_min);
            claim->raw_top1 = raw_mean;
            claim->hand_top1 = hand_mean;
            claim->learned_top1 = learned_mean;
            claim->learned_top10 = learned_top10_min;
            claim->learned_vs_hand_delta_pp = delta_min;
            claim->false_rejections = false_rejections_max;
            claim->sample_count = sample_count_min;
            snprintf(claim->status,
                     sizeof(claim->status),
                     "%s",
                     seed_count >= 5 && delta_min > 0.0 && learned_top10_min >= 1.0 &&
                             false_rejections_max == 0
                         ? "supported"
                         : "underpowered");
            snprintf(claim->summary,
                     sizeof(claim->summary),
                     "across %ld seeds, the learned filter stays at %.2f%% +/- %.2fpp, versus %.2f%% +/- %.2fpp for the hand wheel and %.2f%% +/- %.2fpp raw; the worst learned-vs-hand gain is +%.2fpp, learned top-10 never falls below %.2f%%, and false rejections never exceed %ld; promotion requires 5 seeds.",
                     seed_count,
                     100.0 * learned_mean,
                     100.0 * learned_std,
                     100.0 * hand_mean,
                     100.0 * hand_std,
                     100.0 * raw_mean,
                     100.0 * raw_std,
                     delta_min,
                     100.0 * learned_top10_min,
                     false_rejections_max);
            free(line);
            fclose(fp);
            return;
        }
        free(line);
    }
    fclose(fp);
}

static int paper_string_seen(char values[][64], size_t count, const char *value) {
    size_t i;
    for (i = 0U; i < count; ++i) {
        if (strcmp(values[i], value) == 0) {
            return 1;
        }
    }
    return 0;
}

static void paper_eval_dataset_engine(PaperClaimResult *claim) {
    FILE *fp;
    char *line;
    char domains[16][64];
    long rows = 0;
    long all_capability_rows = 0;
    long promoted_smoke_rows = 0;
    long candidate_contract_rows = 0;
    long verifier_contract_rows = 0;
    long checkpoint_identity_rows = 0;
    long multi_target_rows = 0;
    size_t domain_count = 0U;
    double min_test_top1 = 1.0;
    int have_test = 0;
    fp = fopen(claim->spec->source, "r");
    if (fp == NULL) {
        return;
    }
    while ((line = paper_read_line_dynamic(fp)) != NULL) {
        if (paper_json_string_equals(line, "schema", "crlp.engine_evidence.v1")) {
            char *domain = paper_json_dup_key_string(line, "domain");
            double test_top1 = 0.0;
            double mean_targets = 0.0;
            int dataset_cap = 0;
            int rule_cap = 0;
            int replay_cap = 0;
            int evidence_cap = 0;
            int promoted_smoke = 0;
            int candidate_contract = 0;
            int verifier_contract = 0;
            int checkpoint_identity = 0;
            int has_dataset = paper_json_get_bool(line, "dataset", &dataset_cap);
            int has_rule = paper_json_get_bool(line, "rule_search", &rule_cap);
            int has_replay = paper_json_get_bool(line, "replay", &replay_cap);
            int has_evidence = paper_json_get_bool(line, "evidence", &evidence_cap);
            int has_promoted_smoke =
                paper_json_get_bool(line, "smoke_promotion_ready", &promoted_smoke);
            int has_candidate_contract =
                paper_json_get_bool(line,
                                    "complete_candidate_groups",
                                    &candidate_contract);
            int has_verifier_contract =
                paper_json_get_bool(line, "has_verifier_targets", &verifier_contract);
            int has_checkpoint_identity =
                paper_json_get_bool(line,
                                    "has_checkpoint_identity",
                                    &checkpoint_identity);
            ++rows;
            if (domain != NULL && domain[0] != '\0' && domain_count < 16U &&
                !paper_string_seen(domains, domain_count, domain)) {
                snprintf(domains[domain_count], sizeof(domains[domain_count]), "%s", domain);
                ++domain_count;
            }
            if (has_dataset && dataset_cap && has_rule && rule_cap && has_replay &&
                replay_cap && has_evidence && evidence_cap) {
                ++all_capability_rows;
            }
            if (has_promoted_smoke && promoted_smoke) {
                ++promoted_smoke_rows;
            }
            if (has_candidate_contract && candidate_contract) {
                ++candidate_contract_rows;
            }
            if (has_verifier_contract && verifier_contract) {
                ++verifier_contract_rows;
            }
            if (has_checkpoint_identity && checkpoint_identity) {
                ++checkpoint_identity_rows;
            }
            if (paper_json_get_double(line, "test_mean_targets", &mean_targets) &&
                mean_targets > 1.0) {
                ++multi_target_rows;
            }
            if (paper_json_get_double(line, "test_top1", &test_top1)) {
                if (!have_test || test_top1 < min_test_top1) {
                    min_test_top1 = test_top1;
                }
                have_test = 1;
            }
            free(domain);
        }
        free(line);
    }
    fclose(fp);
    claim->raw_top1 = have_test ? min_test_top1 : 0.0;
    claim->learned_top1 = rows > 0 ? (double)all_capability_rows / (double)rows : 0.0;
    claim->false_rejections = rows - all_capability_rows;
    claim->sample_count = rows;
    snprintf(claim->status,
             sizeof(claim->status),
             "%s",
             rows >= 8 && domain_count >= 7U && all_capability_rows == rows &&
                     promoted_smoke_rows == rows && candidate_contract_rows == rows &&
                     verifier_contract_rows == rows && checkpoint_identity_rows == rows
                 ? "supported"
                 : "underpowered");
    snprintf(claim->summary,
             sizeof(claim->summary),
             "the dataset-first engine evidence surface has %ld promoted-smoke rows across %zu domains; %ld/%ld rows declare dataset, rule-search, replay, and evidence capabilities, %ld/%ld pass candidate-group, verifier-target, and checkpoint-identity contracts, %ld rows expose multi-target verifier labels, and the weakest smoke test top-1 is %.2f%%. These rows are infrastructure evidence, not promoted domain-result claims.",
             rows,
             domain_count,
             all_capability_rows,
             rows,
             promoted_smoke_rows < candidate_contract_rows
                 ? promoted_smoke_rows
                 : (candidate_contract_rows < verifier_contract_rows
                        ? candidate_contract_rows
                        : (verifier_contract_rows < checkpoint_identity_rows
                               ? verifier_contract_rows
                               : checkpoint_identity_rows)),
             rows,
             multi_target_rows,
             have_test ? 100.0 * min_test_top1 : 0.0);
    snprintf(claim->chart_rows,
             sizeof(claim->chart_rows),
             "((label: \"engine rows\", value: %ld), (label: \"domains\", value: %zu), (label: \"promoted smoke\", value: %ld), (label: \"all caps\", value: %ld), (label: \"multi-target\", value: %ld))",
             rows,
             domain_count,
             promoted_smoke_rows,
             all_capability_rows,
             multi_target_rows);
}

static void paper_eval_prime_proposer(PaperClaimResult *claim) {
    FILE *fp;
    char *line;
    double candidate_static = NAN;
    double candidate_grammar = NAN;
    double candidate_structural = NAN;
    double conditional_static = NAN;
    double conditional_grammar = NAN;
    double conditional_structural = NAN;
    fp = fopen(claim->spec->source, "r");
    if (fp == NULL) {
        return;
    }
    while ((line = paper_read_line_dynamic(fp)) != NULL) {
        long max_rules = 0;
        int include_candidate = 0;
        int include_gap = 0;
        double saved = 0.0;
        char *strategy = NULL;
        int is_candidate;
        int is_conditional;
        if (!paper_json_string_equals(line, "schema", "crlp.rule_search.summary.v1") ||
            !paper_json_string_equals(line, "task", "prime_gap") ||
            !paper_json_get_long(line, "max_rules", &max_rules) ||
            max_rules != 16 ||
            !paper_json_get_double(line, "expanded_saved_pct", &saved)) {
            free(line);
            continue;
        }
        (void)paper_json_get_bool(line, "include_candidate_filters", &include_candidate);
        (void)paper_json_get_bool(line, "include_gap_filters", &include_gap);
        strategy = paper_json_dup_key_string(line, "selection_strategy");
        is_candidate = include_candidate != 0 && include_gap == 0;
        is_conditional = include_candidate == 0 && include_gap != 0;
        if (is_candidate) {
            if (strategy != NULL && strcmp(strategy, "neural_structural") == 0) {
                candidate_static = saved;
            }
            if (strategy != NULL && strcmp(strategy, "grammar_order") == 0) {
                candidate_grammar = saved;
            }
            if (strategy != NULL && strcmp(strategy, "structural_heuristic") == 0) {
                candidate_structural = saved;
            }
        } else if (is_conditional) {
            if (strategy != NULL && strcmp(strategy, "neural_structural") == 0) {
                conditional_static = saved;
            }
            if (strategy != NULL && strcmp(strategy, "grammar_order") == 0) {
                conditional_grammar = saved;
            }
            if (strategy != NULL && strcmp(strategy, "structural_heuristic") == 0) {
                conditional_structural = saved;
            }
        }
        free(strategy);
        free(line);
    }
    fclose(fp);
    if (!isfinite(candidate_static) || !isfinite(candidate_grammar) ||
        !isfinite(candidate_structural) || !isfinite(conditional_static) ||
        !isfinite(conditional_grammar) || !isfinite(conditional_structural)) {
        return;
    }
    {
        double candidate_deterministic =
            fmax(candidate_grammar, candidate_structural);
        double conditional_deterministic =
            fmax(conditional_grammar, conditional_structural);
        int static_not_superior =
            candidate_static <= candidate_deterministic &&
            conditional_static <= conditional_deterministic;
        int static_superior =
            candidate_static > candidate_deterministic &&
            conditional_static > conditional_deterministic;
        const char *interpretation =
            static_not_superior
                ? "is a supported negative baseline"
                : (static_superior ? "contradicts the registered negative baseline"
                                   : "has mixed fixed-budget evidence");
        snprintf(claim->status,
                 sizeof(claim->status),
                 "%s",
                 static_not_superior ? "supported"
                                      : (static_superior ? "contradicted"
                                                         : "underpowered"));
        snprintf(claim->summary,
                 sizeof(claim->summary),
                 "the static fixed-budget proposer %s: candidate static saves %.2f vs best deterministic %.2f, and conditional static saves %.2f vs best deterministic %.2f.",
                 interpretation,
                 candidate_static,
                 candidate_deterministic,
                 conditional_static,
                 conditional_deterministic);
    }
}

static void paper_eval_arc_ood(PaperClaimResult *claim) {
    const char *cols[1] = {"model_kind"};
    const char *shared_val[1] = {"shared"};
    const char *single_val[1] = {"single_task"};
    PaperCsvRecord shared;
    PaperCsvRecord single;
    double shared_top1;
    double single_top1;
    double shared_nll;
    double single_nll;
    long shared_seeds = 0;
    long single_seeds = 0;
    if (paper_csv_load_matching_row(claim->spec->source, cols, shared_val, 1U, &shared) <= 0 ||
        paper_csv_load_matching_row(claim->spec->source, cols, single_val, 1U, &single) <= 0 ||
        !paper_csv_record_get_double(&shared, "top1_mean", &shared_top1) ||
        !paper_csv_record_get_double(&single, "top1_mean", &single_top1) ||
        !paper_csv_record_get_double(&shared, "nll_mean", &shared_nll) ||
        !paper_csv_record_get_double(&single, "nll_mean", &single_nll)) {
        return;
    }
    (void)paper_csv_record_get_long(&shared, "seeds", &shared_seeds);
    (void)paper_csv_record_get_long(&single, "seeds", &single_seeds);
    snprintf(claim->status,
             sizeof(claim->status),
             "%s",
             shared_seeds >= 3 && single_seeds >= 3 && shared_top1 > single_top1 &&
                     shared_nll < single_nll
                 ? "supported"
                 : "contradicted");
    snprintf(claim->summary,
             sizeof(claim->summary),
             "shared OOD top-1 %.2f%% vs single %.2f%% (+%.2fpp), NLL %.4f vs %.4f.",
             100.0 * shared_top1,
             100.0 * single_top1,
             100.0 * (shared_top1 - single_top1),
             shared_nll,
             single_nll);
    snprintf(claim->chart_rows,
             sizeof(claim->chart_rows),
             "((label: \"ARC OOD\", value: %.15g, shared_top1: %.15g, single_top1: %.15g),)",
             100.0 * (shared_top1 - single_top1),
             shared_top1,
             single_top1);
}

static void paper_eval_arc_gate_invariance(PaperClaimResult *claim) {
    const char *cols[1] = {"split"};
    const char *train_val[1] = {"train"};
    const char *eval_val[1] = {"eval"};
    PaperCsvRecord train;
    PaperCsvRecord eval;
    long train_loaded = 0;
    long eval_loaded = 0;
    long train_parse = 1;
    long eval_parse = 1;
    long train_cert = 0;
    long eval_cert = 0;
    long train_cells = 0;
    long eval_cells = 0;
    long mismatch_rows = paper_csv_count_rows(
        "results/2026-05-17-multitask-roadmap/rmt7_arc_gate_invariance_mismatches.csv");
    if (paper_csv_load_matching_row(claim->spec->source, cols, train_val, 1U, &train) <= 0 ||
        paper_csv_load_matching_row(claim->spec->source, cols, eval_val, 1U, &eval) <= 0) {
        return;
    }
    (void)paper_csv_record_get_long(&train, "loader_loaded_tasks", &train_loaded);
    (void)paper_csv_record_get_long(&eval, "loader_loaded_tasks", &eval_loaded);
    (void)paper_csv_record_get_long(&train, "loader_parse_errors", &train_parse);
    (void)paper_csv_record_get_long(&eval, "loader_parse_errors", &eval_parse);
    (void)paper_csv_record_get_long(&train, "loader_certified_tasks", &train_cert);
    (void)paper_csv_record_get_long(&eval, "loader_certified_tasks", &eval_cert);
    (void)paper_csv_record_get_long(&train, "loader_test_correct_cells", &train_cells);
    (void)paper_csv_record_get_long(&eval, "loader_test_correct_cells", &eval_cells);
    if (mismatch_rows < 0) {
        mismatch_rows = 0;
    }
    snprintf(claim->status,
             sizeof(claim->status),
             "%s",
             train_parse == 0 && eval_parse == 0 && train_cert > 0 && eval_cert > 0 &&
                     mismatch_rows == 0
                 ? "supported"
                 : "contradicted");
    snprintf(claim->summary,
             sizeof(claim->summary),
             "ARC gate/proposer invariance holds across scorer-context labels: train %ld/%ld certified tasks, eval %ld/%ld, mismatches=%ld.",
             train_cert,
             train_loaded,
             eval_cert,
             eval_loaded,
             mismatch_rows);
    snprintf(claim->chart_rows,
             sizeof(claim->chart_rows),
             "((label: \"ARC train\", value: %ld, certified_tasks: %ld), (label: \"ARC eval\", value: %ld, certified_tasks: %ld))",
             train_cells,
             train_cert,
             eval_cells,
             eval_cert);
}

static void paper_eval_arc_public(PaperClaimResult *claim) {
    const char *cols[2] = {"version", "split"};
    const char *v10_eval[2] = {"v10_recorded", "evaluation"};
    const char *v12_eval[2] = {"v12_recorded", "evaluation"};
    const char *cur_eval[2] = {"v15_measured", "evaluation"};
    const char *cur_train[2] = {"v15_measured", "training"};
    PaperCsvRecord r10;
    PaperCsvRecord r12;
    PaperCsvRecord reval;
    PaperCsvRecord rtrain;
    long v10_cert = 0;
    long v12_cert = 0;
    long eval_cert = 0;
    long train_cert = 0;
    long v10_cells = 0;
    long v12_cells = 0;
    long eval_cells = 0;
    long train_cells = 0;
    if (paper_csv_load_matching_row(claim->spec->source, cols, v10_eval, 2U, &r10) <= 0 ||
        paper_csv_load_matching_row(claim->spec->source, cols, v12_eval, 2U, &r12) <= 0 ||
        paper_csv_load_matching_row(claim->spec->source, cols, cur_eval, 2U, &reval) <= 0 ||
        paper_csv_load_matching_row(claim->spec->source, cols, cur_train, 2U, &rtrain) <= 0) {
        return;
    }
    (void)paper_csv_record_get_long(&r10, "certified_tasks", &v10_cert);
    (void)paper_csv_record_get_long(&r12, "certified_tasks", &v12_cert);
    (void)paper_csv_record_get_long(&reval, "certified_tasks", &eval_cert);
    (void)paper_csv_record_get_long(&rtrain, "certified_tasks", &train_cert);
    (void)paper_csv_record_get_long(&r10, "test_correct_cells", &v10_cells);
    (void)paper_csv_record_get_long(&r12, "test_correct_cells", &v12_cells);
    (void)paper_csv_record_get_long(&reval, "test_correct_cells", &eval_cells);
    (void)paper_csv_record_get_long(&rtrain, "test_correct_cells", &train_cells);
    snprintf(claim->status,
             sizeof(claim->status),
             "%s",
             eval_cert >= v12_cert && train_cert > 0 ? "supported" : "contradicted");
    snprintf(claim->summary,
             sizeof(claim->summary),
             "the ARC row measures coverage, not competence: public eval coverage moves from v10 %ld tasks/%ld cells to v15_measured %ld tasks/%ld cells, while train coverage is %ld tasks/%ld cells.",
             v10_cert,
             v10_cells,
             eval_cert,
             eval_cells,
             train_cert,
             train_cells);
    snprintf(claim->chart_rows,
             sizeof(claim->chart_rows),
             "((label: \"v10 eval\", value: %ld, certified_tasks: %ld), (label: \"v12 eval\", value: %ld, certified_tasks: %ld), (label: \"v15_measured eval\", value: %ld, certified_tasks: %ld))",
             v10_cells,
             v10_cert,
             v12_cells,
             v12_cert,
             eval_cells,
             eval_cert);
}

static void paper_eval_arc_calibration(PaperClaimResult *claim) {
    const char *cols[2] = {"scope", "model_kind"};
    const char *fixed_shared_v[2] = {"fixed_task", "shared"};
    const char *fixed_single_v[2] = {"fixed_task", "single_task"};
    const char *ood_shared_v[2] = {"arc_ood_aggregate", "shared"};
    const char *ood_single_v[2] = {"arc_ood_aggregate", "single_task"};
    PaperCsvRecord fixed_shared;
    PaperCsvRecord fixed_single;
    PaperCsvRecord ood_shared;
    PaperCsvRecord ood_single;
    double fs_nll;
    double fi_nll;
    double os_nll;
    double oi_nll;
    double fs_ece;
    double fi_ece;
    double os_ece;
    double oi_ece;
    if (paper_csv_load_matching_row(claim->spec->source, cols, fixed_shared_v, 2U, &fixed_shared) <= 0 ||
        paper_csv_load_matching_row(claim->spec->source, cols, fixed_single_v, 2U, &fixed_single) <= 0 ||
        paper_csv_load_matching_row(claim->spec->source, cols, ood_shared_v, 2U, &ood_shared) <= 0 ||
        paper_csv_load_matching_row(claim->spec->source, cols, ood_single_v, 2U, &ood_single) <= 0 ||
        !paper_csv_record_get_double(&fixed_shared, "nll_mean", &fs_nll) ||
        !paper_csv_record_get_double(&fixed_single, "nll_mean", &fi_nll) ||
        !paper_csv_record_get_double(&ood_shared, "nll_mean", &os_nll) ||
        !paper_csv_record_get_double(&ood_single, "nll_mean", &oi_nll) ||
        !paper_csv_record_get_double(&fixed_shared, "ece_mean", &fs_ece) ||
        !paper_csv_record_get_double(&fixed_single, "ece_mean", &fi_ece) ||
        !paper_csv_record_get_double(&ood_shared, "ece_mean", &os_ece) ||
        !paper_csv_record_get_double(&ood_single, "ece_mean", &oi_ece)) {
        return;
    }
    snprintf(claim->status,
             sizeof(claim->status),
             "%s",
             fs_nll < fi_nll && os_nll < oi_nll && os_ece < oi_ece ? "supported"
                                                                   : "contradicted");
    snprintf(claim->summary,
             sizeof(claim->summary),
             "ARC shared scorer is a calibration tradeoff: fixed-task NLL improves (%.4f vs %.4f) while ECE worsens (%.4f vs %.4f); OOD improves both NLL (%.4f vs %.4f) and ECE (%.4f vs %.4f).",
             fs_nll,
             fi_nll,
             fs_ece,
             fi_ece,
             os_nll,
             oi_nll,
             os_ece,
             oi_ece);
    snprintf(claim->chart_rows,
             sizeof(claim->chart_rows),
             "((label: \"fixed ARC\", value: %.15g, nll_improvement: %.15g), (label: \"ARC OOD\", value: %.15g, nll_improvement: %.15g))",
             100.0 * (fi_ece - fs_ece),
             fi_nll - fs_nll,
             100.0 * (oi_ece - os_ece),
             oi_nll - os_nll);
}

static void paper_eval_delayed_transfer(PaperClaimResult *claim) {
    const char *cols[1] = {"training_set"};
    const char *all_v[1] = {"all"};
    const char *no_grid_v[1] = {"no_grid_path"};
    const char *no_arc_v[1] = {"no_arc_agi"};
    const char *claim_v[1] = {"claim_only"};
    PaperCsvRecord all;
    PaperCsvRecord no_grid;
    PaperCsvRecord no_arc;
    PaperCsvRecord claim_only;
    double all_top1;
    double no_grid_top1;
    double no_arc_top1;
    double claim_top1;
    if (paper_csv_load_matching_row(claim->spec->source, cols, all_v, 1U, &all) <= 0 ||
        paper_csv_load_matching_row(claim->spec->source, cols, no_grid_v, 1U, &no_grid) <= 0 ||
        paper_csv_load_matching_row(claim->spec->source, cols, no_arc_v, 1U, &no_arc) <= 0 ||
        paper_csv_load_matching_row(claim->spec->source, cols, claim_v, 1U, &claim_only) <= 0 ||
        !paper_csv_record_get_double(&all, "top1_mean", &all_top1) ||
        !paper_csv_record_get_double(&no_grid, "top1_mean", &no_grid_top1) ||
        !paper_csv_record_get_double(&no_arc, "top1_mean", &no_arc_top1) ||
        !paper_csv_record_get_double(&claim_only, "top1_mean", &claim_top1)) {
        return;
    }
    snprintf(claim->status,
             sizeof(claim->status),
             "%s",
             all_top1 > claim_top1 ? "supported" : "contradicted");
    snprintf(claim->summary,
             sizeof(claim->summary),
             "delayed_claim all-domain top-1 %.2f%% vs claim-only %.2f%% (+%.2fpp); removing grid_path falls to %.2f%%, while removing ARC rises to %.2f%%.",
             100.0 * all_top1,
             100.0 * claim_top1,
             100.0 * (all_top1 - claim_top1),
             100.0 * no_grid_top1,
             100.0 * no_arc_top1);
    snprintf(claim->chart_rows,
             sizeof(claim->chart_rows),
             "((label: \"all\", value: %.15g, top1: %.15g), (label: \"no_grid_path\", value: %.15g, top1: %.15g), (label: \"no_arc_agi\", value: %.15g, top1: %.15g), (label: \"claim_only\", value: 0.0, top1: %.15g))",
             100.0 * (all_top1 - claim_top1),
             all_top1,
             100.0 * (no_grid_top1 - claim_top1),
             no_grid_top1,
             100.0 * (no_arc_top1 - claim_top1),
             no_arc_top1,
             claim_top1);
}

static int paper_tiny_net_plateau_audit_ready(long *rows_out,
                                              long *plateau_rows_out,
                                              long *verdict_rows_out,
                                              long *audit_ready_out) {
    FILE *fp;
    char *line;
    long rows = 0;
    long plateau_rows = 0;
    long verdict_rows = paper_count_jsonl_rows(
        "results/2026-05-17-multitask-roadmap/tiny_net_audit_verdict.jsonl");
    long audit_ready = 0;

    fp = fopen("results/2026-05-17-multitask-roadmap/tiny_net_plateau_summary.jsonl",
               "r");
    if (fp != NULL) {
        while ((line = paper_read_line_dynamic(fp)) != NULL) {
            int plateau = 0;
            ++rows;
            if (paper_json_get_bool(line, "plateau_detected", &plateau) && plateau) {
                ++plateau_rows;
            }
            free(line);
        }
        fclose(fp);
    }

    fp = fopen("results/2026-05-17-multitask-roadmap/tiny_net_audit_verdict.jsonl",
               "r");
    if (fp != NULL) {
        while ((line = paper_read_line_dynamic(fp)) != NULL) {
            int ready = 0;
            if (paper_json_get_bool(line, "audit_ready", &ready) && ready) {
                ++audit_ready;
            }
            free(line);
        }
        fclose(fp);
    }

    if (rows_out != NULL) {
        *rows_out = rows;
    }
    if (plateau_rows_out != NULL) {
        *plateau_rows_out = plateau_rows;
    }
    if (verdict_rows_out != NULL) {
        *verdict_rows_out = verdict_rows < 0 ? 0 : verdict_rows;
    }
    if (audit_ready_out != NULL) {
        *audit_ready_out = audit_ready;
    }
    return rows > 0 && plateau_rows == rows && verdict_rows > 0 &&
           audit_ready == verdict_rows;
}

static void paper_tiny_net_audit_shape(long *capacity_count_out,
                                       long *restart_count_out,
                                       long *checkpoint_count_out,
                                       long *max_epoch_out) {
    FILE *fp;
    char *line;
    long capacity_count = 0;
    long restart_count = 0;
    long checkpoint_count = 0;
    long max_epoch = 0;

    fp = fopen("results/2026-05-17-multitask-roadmap/tiny_net_audit_verdict.jsonl",
               "r");
    if (fp != NULL) {
        line = paper_read_line_dynamic(fp);
        if (line != NULL) {
            (void)paper_json_get_long(line, "capacity_count", &capacity_count);
            (void)paper_json_get_long(line, "restart_count", &restart_count);
            (void)paper_json_get_long(line, "epoch_checkpoint_count", &checkpoint_count);
            free(line);
        }
        fclose(fp);
    }

    fp = fopen("results/2026-05-17-multitask-roadmap/tiny_net_plateau_summary.jsonl",
               "r");
    if (fp != NULL) {
        while ((line = paper_read_line_dynamic(fp)) != NULL) {
            long last_epochs = 0;
            if (paper_json_get_long(line, "last_epochs", &last_epochs) &&
                last_epochs > max_epoch) {
                max_epoch = last_epochs;
            }
            free(line);
        }
        fclose(fp);
    }

    if (capacity_count_out != NULL) {
        *capacity_count_out = capacity_count;
    }
    if (restart_count_out != NULL) {
        *restart_count_out = restart_count;
    }
    if (checkpoint_count_out != NULL) {
        *checkpoint_count_out = checkpoint_count;
    }
    if (max_epoch_out != NULL) {
        *max_epoch_out = max_epoch;
    }
}

static void paper_eval_capacity(PaperClaimResult *claim) {
    FILE *fp;
    char *header_line;
    char *line;
    char *headers[80];
    char *values[80];
    size_t header_count;
    double best_arc_shared = -1.0;
    double best_arc_delta = -1.0e9;
    char best_hidden[64] = "";
    double claim_delta_min = 1.0e9;
    double claim_delta_max = -1.0e9;
    int arc_rows = 0;
    int all_arc_trail = 1;
    long plateau_rows_total = 0;
    long plateau_rows_ready = 0;
    long verdict_rows_total = 0;
    long verdict_rows_ready = 0;
    int plateau_ready = paper_tiny_net_plateau_audit_ready(&plateau_rows_total,
                                                           &plateau_rows_ready,
                                                           &verdict_rows_total,
                                                           &verdict_rows_ready);
    claim->chart_rows[0] = '\0';
    fp = fopen(claim->spec->source, "r");
    if (fp == NULL) {
        return;
    }
    header_line = paper_read_line_dynamic(fp);
    if (header_line == NULL) {
        fclose(fp);
        return;
    }
    paper_chomp(header_line);
    header_count = paper_csv_split_in_place(header_line, headers, 80U);
    strcpy(claim->chart_rows, "(");
    while ((line = paper_read_line_dynamic(fp)) != NULL) {
        PaperCsvRecord row;
        size_t value_count;
        const char *domain;
        const char *hidden;
        double shared = 0.0;
        double single = 0.0;
        double delta = 0.0;
        paper_chomp(line);
        value_count = paper_csv_split_in_place(line, values, 80U);
        paper_csv_record_copy(&row, headers, header_count, values, value_count);
        domain = paper_csv_record_get(&row, "domain");
        hidden = paper_csv_record_get(&row, "hidden");
        if (domain != NULL && strcmp(domain, "arc_agi") == 0 &&
            paper_csv_record_get_double(&row, "shared_top1_mean", &shared) &&
            paper_csv_record_get_double(&row, "single_top1_mean", &single) &&
            paper_csv_record_get_double(&row, "top1_delta_shared_minus_single", &delta)) {
            ++arc_rows;
            if (delta >= 0.0) {
                all_arc_trail = 0;
            }
            if (shared > best_arc_shared) {
                best_arc_shared = shared;
                best_arc_delta = delta;
                snprintf(best_hidden, sizeof(best_hidden), "%s", hidden == NULL ? "" : hidden);
            }
            paper_appendf(claim->chart_rows,
                          sizeof(claim->chart_rows),
                          "(label: \"%s\", value: %.15g, arc_shared_top1: %.15g, arc_single_top1: %.15g, claim_delta_pp: ",
                          hidden == NULL ? "" : hidden,
                          100.0 * delta,
                          shared,
                          single);
            free(line);
            continue;
        }
        if (domain != NULL && strcmp(domain, "delayed_claim") == 0 &&
            paper_csv_record_get_double(&row, "top1_delta_shared_minus_single", &delta)) {
            if (delta < claim_delta_min) {
                claim_delta_min = delta;
            }
            if (delta > claim_delta_max) {
                claim_delta_max = delta;
            }
        }
        free(line);
    }
    free(header_line);
    fclose(fp);
    if (arc_rows == 0 || !isfinite(claim_delta_min) || claim_delta_min > 1.0e8) {
        return;
    }
    /* Fill the chart with a second pass so each ARC row can carry the matching
       delayed-claim delta for the same hidden size. */
    claim->chart_rows[0] = '\0';
    strcpy(claim->chart_rows, "(");
    {
        const char *hiddens[4] = {"32x16", "64x32", "128x64", "256x128"};
        size_t i;
        for (i = 0U; i < 4U; ++i) {
            const char *pred_cols[2] = {"hidden", "domain"};
            const char *arc_vals[2] = {hiddens[i], "arc_agi"};
            const char *claim_vals[2] = {hiddens[i], "delayed_claim"};
            PaperCsvRecord arc;
            PaperCsvRecord delayed;
            double arc_shared = 0.0;
            double arc_single = 0.0;
            double arc_delta = 0.0;
            double claim_delta = 0.0;
            if (paper_csv_load_matching_row(claim->spec->source,
                                            pred_cols,
                                            arc_vals,
                                            2U,
                                            &arc) > 0 &&
                paper_csv_load_matching_row(claim->spec->source,
                                            pred_cols,
                                            claim_vals,
                                            2U,
                                            &delayed) > 0 &&
                paper_csv_record_get_double(&arc, "shared_top1_mean", &arc_shared) &&
                paper_csv_record_get_double(&arc, "single_top1_mean", &arc_single) &&
                paper_csv_record_get_double(&arc,
                                            "top1_delta_shared_minus_single",
                                            &arc_delta) &&
                paper_csv_record_get_double(&delayed,
                                            "top1_delta_shared_minus_single",
                                            &claim_delta)) {
                paper_appendf(claim->chart_rows,
                              sizeof(claim->chart_rows),
                              "(label: \"%s\", value: %.15g, arc_shared_top1: %.15g, arc_single_top1: %.15g, claim_delta_pp: %.15g), ",
                              hiddens[i],
                              100.0 * arc_delta,
                              arc_shared,
                              arc_single,
                              100.0 * claim_delta);
            }
        }
    }
    paper_appendf(claim->chart_rows, sizeof(claim->chart_rows), ")");
    snprintf(claim->status,
             sizeof(claim->status),
             "%s",
             plateau_ready ?
                 (all_arc_trail && claim_delta_min > 0.0 ? "supported" : "contradicted") :
                 "underpowered");
    if (plateau_ready) {
        snprintf(claim->summary,
                 sizeof(claim->summary),
                 "making the shared model wider does not simply fix ARC: the best fixed-ARC shared row is %s at %.2f%%, but every shared ARC row trails single-task; delayed_claim deltas remain +%.2fpp to +%.2fpp.",
                 best_hidden,
                 100.0 * best_arc_shared,
                 100.0 * claim_delta_min,
                 100.0 * claim_delta_max);
    } else {
        snprintf(claim->summary,
                 sizeof(claim->summary),
                 "C7 blocks this capacity claim until plateau audit is ready: the sweep's best fixed-ARC shared row is %s at %.2f%% and delayed_claim deltas are +%.2fpp to +%.2fpp, but only %ld/%ld plateau rows and %ld/%ld domain/variant verdicts are audit-ready.",
                 best_hidden,
                 100.0 * best_arc_shared,
                 100.0 * claim_delta_min,
                 100.0 * claim_delta_max,
                 plateau_rows_ready,
                 plateau_rows_total,
                 verdict_rows_ready,
                 verdict_rows_total);
    }
    (void)best_arc_delta;
}

static void paper_eval_adapter(PaperClaimResult *claim) {
    const char *cols[2] = {"eval_suite", "domain"};
    const char *fixed_arc_v[2] = {"fixed", "arc_agi"};
    const char *delayed_v[2] = {"fixed", "delayed_claim"};
    const char *grid_v[2] = {"fixed", "grid_path"};
    const char *ood_arc_v[2] = {"ood", "arc_agi"};
    PaperCsvRecord fixed_arc;
    PaperCsvRecord delayed;
    PaperCsvRecord grid;
    PaperCsvRecord ood_arc;
    double fixed_delta = 0.0;
    double delayed_delta = 0.0;
    double grid_delta = 0.0;
    double ood_delta = 0.0;
    double ood_nll_delta = 0.0;
    long plateau_rows_total = 0;
    long plateau_rows_ready = 0;
    long verdict_rows_total = 0;
    long verdict_rows_ready = 0;
    int plateau_ready = paper_tiny_net_plateau_audit_ready(&plateau_rows_total,
                                                           &plateau_rows_ready,
                                                           &verdict_rows_total,
                                                           &verdict_rows_ready);
    if (paper_csv_load_matching_row(claim->spec->source, cols, fixed_arc_v, 2U, &fixed_arc) <= 0 ||
        paper_csv_load_matching_row(claim->spec->source, cols, delayed_v, 2U, &delayed) <= 0 ||
        paper_csv_load_matching_row(claim->spec->source, cols, grid_v, 2U, &grid) <= 0 ||
        paper_csv_load_matching_row(claim->spec->source, cols, ood_arc_v, 2U, &ood_arc) <= 0 ||
        !paper_csv_record_get_double(&fixed_arc,
                                     "top1_delta_headed_minus_shared",
                                     &fixed_delta) ||
        !paper_csv_record_get_double(&delayed,
                                     "top1_delta_headed_minus_shared",
                                     &delayed_delta) ||
        !paper_csv_record_get_double(&grid,
                                     "top1_delta_headed_minus_shared",
                                     &grid_delta) ||
        !paper_csv_record_get_double(&ood_arc,
                                     "top1_delta_headed_minus_single",
                                     &ood_delta) ||
        !paper_csv_record_get_double(&ood_arc,
                                     "nll_delta_headed_minus_single",
                                     &ood_nll_delta)) {
        return;
    }
    snprintf(claim->status,
             sizeof(claim->status),
             "%s",
             plateau_ready ?
                 (fixed_delta > 0.0 && delayed_delta >= -0.005 &&
                          grid_delta >= -0.005 && ood_delta >= 0.0
                      ? "supported"
                      : "contradicted") :
                 "underpowered");
    if (plateau_ready) {
        snprintf(claim->summary,
                 sizeof(claim->summary),
                 "task-specific heads help some rows and hurt others: fixed ARC heads-flat delta %+.2fpp; delayed_claim heads-flat %+.2fpp; grid_path heads-flat %+.2fpp; ARC OOD heads-single %+.2fpp top-1 and %.4f NLL.",
                 100.0 * fixed_delta,
                 100.0 * delayed_delta,
                 100.0 * grid_delta,
                 100.0 * ood_delta,
                 ood_nll_delta);
    } else {
        snprintf(claim->summary,
                 sizeof(claim->summary),
                 "C7 blocks this adapter-head architecture claim until plateau audit is ready: fixed ARC heads-flat delta %+.2fpp; delayed_claim heads-flat %+.2fpp; grid_path heads-flat %+.2fpp; ARC OOD heads-single %+.2fpp top-1 and %.4f NLL, with only %ld/%ld plateau rows and %ld/%ld domain/variant verdicts audit-ready.",
                 100.0 * fixed_delta,
                 100.0 * delayed_delta,
                 100.0 * grid_delta,
                 100.0 * ood_delta,
                 ood_nll_delta,
                 plateau_rows_ready,
                 plateau_rows_total,
                 verdict_rows_ready,
                 verdict_rows_total);
    }
    snprintf(claim->chart_rows,
             sizeof(claim->chart_rows),
             "((label: \"fixed ARC vs flat\", value: %.15g), (label: \"delayed vs flat\", value: %.15g), (label: \"grid vs flat\", value: %.15g), (label: \"ARC OOD vs single\", value: %.15g))",
             100.0 * fixed_delta,
             100.0 * delayed_delta,
             100.0 * grid_delta,
             100.0 * ood_delta);
}

static void paper_eval_rejected_reason(PaperClaimResult *claim) {
    FILE *fp;
    char *line;
    long model_unsafe = 0;
    long model_false = 0;
    long reason_accepted = 0;
    long reason_false = 0;
    long trained_accepted = 0;
    long trained_false = 0;
    double reason_utility = 0.0;
    double aware_utility = 0.0;
    double trained_utility = 0.0;
    int have_model = 0;
    int have_reason = 0;
    int have_aware = 0;
    int have_trained = 0;
    fp = fopen(claim->spec->source, "r");
    if (fp == NULL) {
        return;
    }
    while ((line = paper_read_line_dynamic(fp)) != NULL) {
        long budget = 0;
        long split_filter_modulus = 61;
        char *strategy = NULL;
        if (!paper_json_string_equals(line, "task", "prime_gap") ||
            !paper_json_get_long(line, "budget", &budget) || budget != 8) {
            free(line);
            continue;
        }
        (void)paper_json_get_long(line, "split_filter_modulus", &split_filter_modulus);
        if (split_filter_modulus != 61) {
            free(line);
            continue;
        }
        strategy = paper_json_dup_key_string(line, "strategy");
        if (strategy != NULL && strcmp(strategy, "model_score") == 0 && !have_model) {
            have_model = 1;
            (void)paper_json_get_long(line, "unsafe_rules", &model_unsafe);
            (void)paper_json_get_long(line, "validation_false_rejections", &model_false);
        } else if (strategy != NULL && strcmp(strategy, "reason_prior") == 0 &&
                   !have_reason) {
            have_reason = 1;
            (void)paper_json_get_long(line, "accepted_rules", &reason_accepted);
            (void)paper_json_get_long(line, "validation_false_rejections", &reason_false);
            (void)paper_json_get_double(line, "validation_utility", &reason_utility);
        } else if (strategy != NULL && strcmp(strategy, "reason_aware") == 0 &&
                   !have_aware) {
            have_aware = 1;
            (void)paper_json_get_double(line, "validation_utility", &aware_utility);
        } else if (strategy != NULL && strcmp(strategy, "trained_residual") == 0 &&
                   !have_trained) {
            have_trained = 1;
            (void)paper_json_get_long(line, "accepted_rules", &trained_accepted);
            (void)paper_json_get_long(line, "validation_false_rejections", &trained_false);
            (void)paper_json_get_double(line, "validation_utility", &trained_utility);
        }
        free(strategy);
        free(line);
    }
    fclose(fp);
    if (!(have_model && have_reason && have_aware && have_trained)) {
        return;
    }
    snprintf(claim->status,
             sizeof(claim->status),
             "%s",
             model_false > 0 && reason_false == 0 && trained_false == 0 &&
                     trained_utility >= reason_utility
                 ? "supported"
                 : "contradicted");
    snprintf(claim->summary,
             sizeof(claim->summary),
             "logging why the verifier rejected a guess repairs the next-prime proposer at budget 8: the unguided model chooses %ld unsafe rows and causes %ld validation false rejections, while the reason prior reaches %ld accepted rules/%.1f utility and the trained repair reaches %ld accepted rules/%.1f utility with zero false rejections; arithmetic-sequence/modular split sweeps pass, but the delayed-claim stress case has 0 mixed split rows.",
             model_unsafe,
             model_false,
             reason_accepted,
             reason_utility,
             trained_accepted,
             trained_utility);
    snprintf(claim->chart_rows,
             sizeof(claim->chart_rows),
             "((label: \"model_score\", value: 0.0, false_rejections: %ld, unsafe_rules: %ld), (label: \"reason_prior\", value: %.15g, false_rejections: %ld, unsafe_rules: 0), (label: \"reason_aware\", value: %.15g, false_rejections: 0, unsafe_rules: 0), (label: \"trained_residual\", value: %.15g, false_rejections: %ld, unsafe_rules: 0))",
             model_false,
             model_unsafe,
             reason_utility,
             reason_false,
             aware_utility,
             trained_utility,
             trained_false);
}

static void paper_eval_hypothesis_promotion(PaperClaimResult *claim) {
    char *promotion = NULL;
    long support_tasks = 0;
    long covered_events = 0;
    long false_rejections = 1;
    long heldout_rows = paper_count_jsonl_rows(
        "results/2026-05-17-substrate-extensions/hypothesis_library_heldout_result.jsonl");
    double no_prior_utility = 0.0;
    double slot_utility = 0.0;
    long no_prior_cert = 0;
    long slot_cert = 0;
    if (paper_read_first_jsonl(claim->spec->source, &promotion) <= 0) {
        return;
    }
    (void)paper_json_get_long(promotion, "support_tasks", &support_tasks);
    (void)paper_json_get_long(promotion, "covered_events", &covered_events);
    (void)paper_json_get_long(promotion, "false_rejections", &false_rejections);
    free(promotion);
    {
        char *line2 = NULL;
        FILE *fp = fopen("results/2026-05-17-substrate-extensions/hypothesis_library_heldout_result.jsonl",
                         "r");
        if (fp != NULL) {
            while ((line2 = paper_read_line_dynamic(fp)) != NULL) {
                char *mode = paper_json_dup_key_string(line2, "mode");
                if (mode != NULL && strcmp(mode, "no_prior") == 0) {
                    (void)paper_json_get_long(line2,
                                              "learned_certified_budget1",
                                              &no_prior_cert);
                    (void)paper_json_get_double(line2,
                                                "learned_utility_budget1",
                                                &no_prior_utility);
                } else if (mode != NULL && strcmp(mode, "slot_only") == 0) {
                    (void)paper_json_get_long(line2,
                                              "learned_certified_budget1",
                                              &slot_cert);
                    (void)paper_json_get_double(line2,
                                                "learned_utility_budget1",
                                                &slot_utility);
                }
                free(mode);
                free(line2);
            }
            fclose(fp);
        }
    }
    snprintf(claim->status,
             sizeof(claim->status),
             "%s",
             support_tasks > 0 && false_rejections == 0 && slot_cert >= no_prior_cert
                 ? "supported"
                 : "contradicted");
    snprintf(claim->summary,
             sizeof(claim->summary),
             "saving repeatedly safe rules helps future tasks: promoted move_to_anchor from %ld tasks/%ld covered events with zero false rejections; slot priors raise first-choice certifications from %ld to %ld and budget-8 utility from %.4f to %.4f. Held-out split feedback rows: %ld.",
             support_tasks,
             covered_events,
             no_prior_cert,
             slot_cert,
             no_prior_utility,
             slot_utility,
             heldout_rows < 0 ? 0 : heldout_rows);
    snprintf(claim->chart_rows,
             sizeof(claim->chart_rows),
             "((label: \"no_prior\", value: %ld, utility: %.15g), (label: \"slot_only\", value: %ld, utility: %.15g))",
             no_prior_cert,
             no_prior_utility,
             slot_cert,
	             slot_utility);
}

static void paper_eval_arc_verified_self_play(PaperClaimResult *claim) {
    char *line = NULL;
    long certified = -1;
    long template_baseline = 7;
    long oracle_ceiling = 14;
    long opened_zero_families = 0;
    long seed_count = 0;
    int public_eval = 0;
    int fixed_budget = 0;
    int leakage_gate = 0;
    if (paper_read_first_jsonl(claim->spec->source, &line) <= 0) {
        snprintf(claim->status, sizeof(claim->status), "%s", "underpowered");
        snprintf(claim->summary,
                 sizeof(claim->summary),
                 "ARC verified self-play is registered but underpowered: missing %s, so no public replay, fixed-budget ablation, leakage audit, or zero-family opening can be credited.",
                 claim->spec->source);
        snprintf(claim->chart_rows,
                 sizeof(claim->chart_rows),
                 "((label: \"template\", value: 7), (label: \"self-play\", value: 0), (label: \"oracle\", value: 14))");
        return;
    }
    (void)paper_json_get_long(line, "claim_certified_eval_tasks", &certified);
    (void)paper_json_get_long(line, "claim_template_baseline", &template_baseline);
    (void)paper_json_get_long(line, "claim_oracle_ceiling", &oracle_ceiling);
    (void)paper_json_get_long(line,
                              "claim_opened_zero_coverage_families",
                              &opened_zero_families);
    (void)paper_json_get_long(line, "claim_seed_count", &seed_count);
    (void)paper_json_get_bool(line, "claim_public_eval_replay_available", &public_eval);
    (void)paper_json_get_bool(line,
                              "claim_fixed_budget_ablation_available",
                              &fixed_budget);
    (void)paper_json_get_bool(line, "claim_leakage_gate_passed", &leakage_gate);
    snprintf(claim->status,
             sizeof(claim->status),
             "%s",
             certified >= template_baseline && opened_zero_families >= 1 &&
                     seed_count >= 3 && public_eval && fixed_budget && leakage_gate
                 ? "supported"
                 : "underpowered");
    claim->sample_count = seed_count;
    claim->learned_top1 =
        certified >= 0 && oracle_ceiling > 0 ? (double)certified / (double)oracle_ceiling
                                             : 0.0;
    claim->hand_top1 =
        oracle_ceiling > 0 ? (double)template_baseline / (double)oracle_ceiling : 0.0;
    snprintf(claim->summary,
             sizeof(claim->summary),
             "ARC verified self-play is %s: certified eval=%ld, template=%ld, oracle=%ld, opened zero-coverage families=%ld, seeds=%ld, public replay=%s, fixed-budget ablation=%s, leakage gate=%s. The claim remains L1 search efficiency, not new primitive capability.",
             strcmp(claim->status, "supported") == 0 ? "supported" : "underpowered",
             certified,
             template_baseline,
             oracle_ceiling,
             opened_zero_families,
             seed_count,
             public_eval ? "yes" : "no",
             fixed_budget ? "yes" : "no",
             leakage_gate ? "yes" : "no");
    snprintf(claim->chart_rows,
             sizeof(claim->chart_rows),
             "((label: \"template\", value: %ld), (label: \"self-play\", value: %ld), (label: \"oracle\", value: %ld), (label: \"opened families\", value: %ld))",
             template_baseline,
             certified < 0 ? 0 : certified,
             oracle_ceiling,
             opened_zero_families);
    free(line);
}

static void paper_eval_tiny_net(PaperClaimResult *claim) {
    long rows = 0;
    long plateau_rows = 0;
    long verdict_rows = 0;
    long audit_ready = 0;
    long capacity_count = 0;
    long restart_count = 0;
    long checkpoint_count = 0;
    long max_epoch = 0;
    int ready = paper_tiny_net_plateau_audit_ready(&rows,
                                                   &plateau_rows,
                                                   &verdict_rows,
                                                   &audit_ready);
    paper_tiny_net_audit_shape(&capacity_count,
                               &restart_count,
                               &checkpoint_count,
                               &max_epoch);
    if (rows == 0) {
        return;
    }
    snprintf(claim->status,
             sizeof(claim->status),
             "%s",
             ready ? "supported" : "underpowered");
    snprintf(claim->summary,
             sizeof(claim->summary),
             "tiny-net plateau audit covers %ld capacities, %ld restarts, and %ld checkpoints through %ld epochs across %ld domain/variant verdicts; %ld/%ld required rows meet the plateau threshold and %ld/%ld verdicts are audit-ready.",
             capacity_count > 0 ? capacity_count : 4,
             restart_count > 0 ? restart_count : 0,
             checkpoint_count > 0 ? checkpoint_count : 0,
             max_epoch > 0 ? max_epoch : 0,
             verdict_rows < 0 ? 0 : verdict_rows,
             plateau_rows,
             rows,
             audit_ready,
             verdict_rows < 0 ? 0 : verdict_rows);
}

static void paper_eval_trace_determinism(PaperClaimResult *claim) {
    FILE *fp;
    char *line;
    long traces = 0;
    long rows = 0;
    long invalid = 0;
    long rank_sensitive = 0;
    long mismatches = 0;
    int same_hash = 0;
    fp = fopen(claim->spec->source, "r");
    if (fp == NULL) {
        return;
    }
    while ((line = paper_read_line_dynamic(fp)) != NULL) {
        if (paper_json_string_equals(line, "schema", "crlp.trace_determinism_audit.v1")) {
            long value = 0;
            ++traces;
            if (paper_json_get_long(line, "row_count", &value)) {
                rows += value;
            }
            if (paper_json_get_long(line, "invalid_rows", &value)) {
                invalid += value;
            }
            if (paper_json_get_long(line, "rank_fp_sensitive_rows", &value)) {
                rank_sensitive += value;
            }
            if (paper_json_get_long(line, "rank_fp_sensitive_logged_mismatches", &value)) {
                mismatches += value;
            }
        } else if (paper_json_string_equals(line,
                                           "schema",
                                           "crlp.trace_determinism_compare.v1")) {
            (void)paper_json_get_bool(line, "all_semantic_hashes_equal", &same_hash);
        }
        free(line);
    }
    fclose(fp);
    snprintf(claim->status,
             sizeof(claim->status),
             "%s",
             traces > 0 && rows > 0 && invalid == 0 && rank_sensitive == 0 &&
                     mismatches == 0 && same_hash
                 ? "supported"
                 : "contradicted");
    snprintf(claim->summary,
             sizeof(claim->summary),
             "the trace audit checks whether replay meaning survives numeric/runtime changes: %ld traces and %ld rows, %ld invalid rows, %ld rank-sensitive rows, %ld logged mismatches, same semantic hash: %s.",
             traces,
             rows,
             invalid,
             rank_sensitive,
             mismatches,
             same_hash ? "yes" : "no");
}

static void paper_eval_signal_replay(PaperClaimResult *claim) {
    char *line = NULL;
    double top1 = 0.0;
    double baseline_top1 = 0.0;
    double utility = 0.0;
    double baseline_utility = 0.0;
    double regret = 0.0;
    double baseline_regret = 0.0;
    double damage = 0.0;
    double baseline_damage = 0.0;
    double death = 0.0;
    double baseline_death = 0.0;
    long live_rows = paper_count_jsonl_rows(
        "results/2026-05-17-signal-flight/live_active_frames.jsonl");
    if (paper_read_first_jsonl(claim->spec->source, &line) <= 0) {
        return;
    }
    (void)paper_json_get_double(line, "top1", &top1);
    (void)paper_json_get_double(line, "baseline_top1", &baseline_top1);
    (void)paper_json_get_double(line, "selected_utility", &utility);
    (void)paper_json_get_double(line, "baseline_selected_utility", &baseline_utility);
    (void)paper_json_get_double(line, "regret", &regret);
    (void)paper_json_get_double(line, "baseline_regret", &baseline_regret);
    (void)paper_json_get_double(line, "selected_damage_rate", &damage);
    (void)paper_json_get_double(line, "baseline_damage_rate", &baseline_damage);
    (void)paper_json_get_double(line, "selected_death_rate", &death);
    (void)paper_json_get_double(line, "baseline_death_rate", &baseline_death);
    free(line);
    snprintf(claim->status,
             sizeof(claim->status),
             "%s",
             top1 > baseline_top1 && utility > baseline_utility &&
                     death <= baseline_death
                 ? "supported"
                 : "contradicted");
    snprintf(claim->summary,
             sizeof(claim->summary),
             "in offline Signal replay, the model ranks recorded WASD choices at top-1 %.2f%% vs baseline %.2f%% (+%.2fpp), utility %.4f vs %.4f, regret %.4f vs %.4f, damage %.2f%% vs %.2f%%, deaths %.2f%% vs %.2f%%; live shadow frames=%ld.",
             100.0 * top1,
             100.0 * baseline_top1,
             100.0 * (top1 - baseline_top1),
             utility,
             baseline_utility,
             regret,
             baseline_regret,
             100.0 * damage,
             100.0 * baseline_damage,
             100.0 * death,
             100.0 * baseline_death,
             live_rows < 0 ? 0 : live_rows);
    snprintf(claim->chart_rows,
             sizeof(claim->chart_rows),
             "((label: \"neural replay\", value: %.15g, top1: %.15g), (label: \"baseline\", value: %.15g, top1: %.15g))",
             utility,
             top1,
             baseline_utility,
             baseline_top1);
}

static void paper_eval_signal_live(PaperClaimResult *claim) {
    char *line = NULL;
    long active_rows = 0;
    long replay_ready = 0;
    long protocol = 0;
    long safety = 0;
    long damage = 0;
    long deaths = 0;
    double p95 = 0.0;
    char *checkpoint = NULL;
    if (paper_read_first_jsonl(claim->spec->source, &line) <= 0) {
        return;
    }
    (void)paper_json_get_long(line, "active_rows", &active_rows);
    (void)paper_json_get_long(line, "replay_ready_rows", &replay_ready);
    (void)paper_json_get_long(line, "protocol_compatible_rows", &protocol);
    (void)paper_json_get_long(line, "safety_gate_rows", &safety);
    (void)paper_json_get_long(line, "damage_events", &damage);
    (void)paper_json_get_long(line, "death_events", &deaths);
    (void)paper_json_get_double(line, "inference_ms_p95", &p95);
    checkpoint = paper_json_dup_key_string(line, "checkpoint_hash");
    snprintf(claim->status,
             sizeof(claim->status),
             "%s",
             active_rows > 0 && protocol == active_rows && replay_ready > 0 && deaths == 0
                 ? "supported"
                 : "contradicted");
    snprintf(claim->summary,
             sizeof(claim->summary),
             "the live Signal pilot did run checkpoint-ranked neural actions, but only behind guards: %ld recent neural-source rows from checkpoint %.10s, p95 inference %.3f ms, %ld replay-ready rows, protocol-compatible rows=%ld, safety-gate rows=%ld, damage events=%ld, deaths=%ld; no baseline comparison is claimed.",
             active_rows,
             checkpoint == NULL ? "" : checkpoint,
             p95,
             replay_ready,
             protocol,
             safety,
             damage,
             deaths);
    snprintf(claim->chart_rows,
             sizeof(claim->chart_rows),
             "((label: \"active\", value: %ld), (label: \"replay-ready\", value: %ld), (label: \"protocol-compatible\", value: %ld), (label: \"safety-gated\", value: %ld))",
             active_rows,
             replay_ready,
             protocol,
             safety);
    free(checkpoint);
    free(line);
}


static void paper_eval_cross_domain_transfer(PaperClaimResult *claim) {
    snprintf(claim->status, sizeof(claim->status), "%s", "underpowered");
    snprintf(claim->summary, sizeof(claim->summary),
             "Cross-domain transfer claim registered as underpowered pending claim compiler schema reader.");
}

static void paper_eval_signal_mining(PaperClaimResult *claim) {
    snprintf(claim->status, sizeof(claim->status), "%s", "underpowered");
    snprintf(claim->summary, sizeof(claim->summary),
             "Signal mining claim registered as underpowered pending summary artifact generation.");
}

static void paper_evaluate_claims(PaperClaimResult claims[PAPER_CLAIM_COUNT]) {
    size_t i;
    for (i = 0U; i < PAPER_CLAIM_COUNT; ++i) {
        paper_claim_init(&claims[i], &PAPER_CLAIM_SPECS[i]);
    }
    paper_eval_prime_gate_delta(&claims[0]);
    paper_eval_prime_gate_delta_ci(&claims[1]);
    paper_eval_prime_proposer(&claims[2]);
    paper_eval_arc_ood(&claims[3]);
    paper_eval_arc_gate_invariance(&claims[4]);
    paper_eval_arc_public(&claims[5]);
    paper_eval_arc_calibration(&claims[6]);
    paper_eval_delayed_transfer(&claims[7]);
    paper_eval_capacity(&claims[8]);
    paper_eval_adapter(&claims[9]);
    paper_eval_rejected_reason(&claims[10]);
    paper_eval_hypothesis_promotion(&claims[11]);
    paper_eval_tiny_net(&claims[12]);
    paper_eval_trace_determinism(&claims[13]);
    paper_eval_dataset_engine(&claims[14]);
    paper_eval_signal_replay(&claims[15]);
    paper_eval_signal_live(&claims[16]);
    paper_eval_arc_verified_self_play(&claims[17]);
    paper_eval_cross_domain_transfer(&claims[18]);
    paper_eval_signal_mining(&claims[19]);
    paper_apply_staleness_gate(claims);
}

static void paper_count_claim_statuses(const PaperClaimResult claims[PAPER_CLAIM_COUNT],
                                       long *supported,
                                       long *contradicted,
                                       long *underpowered,
                                       long *missing,
                                       long *stale) {
    size_t i;
    *supported = 0;
    *contradicted = 0;
    *underpowered = 0;
    *missing = 0;
    *stale = 0;
    for (i = 0U; i < PAPER_CLAIM_COUNT; ++i) {
        if (strcmp(claims[i].status, "supported") == 0) {
            ++*supported;
        } else if (strcmp(claims[i].status, "contradicted") == 0) {
            ++*contradicted;
        } else if (strcmp(claims[i].status, "underpowered") == 0) {
            ++*underpowered;
        } else if (strcmp(claims[i].status, "stale") == 0) {
            ++*stale;
        } else {
            ++*missing;
        }
    }
}

static int paper_claim_support_ok(const PaperClaimResult claims[PAPER_CLAIM_COUNT]) {
    size_t i;
    for (i = 0U; i < PAPER_CLAIM_COUNT; ++i) {
        if (claims[i].spec->required && strcmp(claims[i].status, "supported") != 0) {
            return 0;
        }
    }
    return 1;
}

static int paper_write_claims_json(const char *path,
                                   const PaperClaimResult claims[PAPER_CLAIM_COUNT]) {
    FILE *fp;
    size_t i;
    long supported;
    long contradicted;
    long underpowered;
    long missing;
    long stale;
    int ok = paper_claim_support_ok(claims);
    paper_count_claim_statuses(claims,
                               &supported,
                               &contradicted,
                               &underpowered,
                               &missing,
                               &stale);
    if (paper_mkdir_parent(path) != 0) {
        return -1;
    }
    fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    fputs("{\n  \"schema\":\"crlp.paper_claim_support.c.v1\",\n", fp);
    fputs("  \"generated_by\":\"C neural grammar discovery runtime\",\n", fp);
    fprintf(fp,
            "  \"status_counts\":{\"supported\":%ld,\"contradicted\":%ld,\"underpowered\":%ld,\"missing\":%ld,\"stale\":%ld},\n",
            supported,
            contradicted,
            underpowered,
            missing,
            stale);
    fprintf(fp,
            "  \"support\":{\"ok\":%s,\"required_failures\":[",
            ok ? "true" : "false");
    {
        int first = 1;
        for (i = 0U; i < PAPER_CLAIM_COUNT; ++i) {
            if (claims[i].spec->required && strcmp(claims[i].status, "supported") != 0) {
                if (!first) {
                    fputc(',', fp);
                }
                paper_write_json_string(fp, claims[i].spec->id);
                first = 0;
            }
        }
    }
    fputs("],\"surfaced_unsupported_failures\":[]},\n", fp);
    fputs("  \"claims\":[\n", fp);
    for (i = 0U; i < PAPER_CLAIM_COUNT; ++i) {
        fprintf(fp, "    {\"id\":");
        paper_write_json_string(fp, claims[i].spec->id);
        fputs(",\"family\":", fp);
        paper_write_json_string(fp, claims[i].spec->family);
        fputs(",\"title\":", fp);
        paper_write_json_string(fp, claims[i].spec->title);
        fputs(",\"claim_text\":", fp);
        paper_write_json_string(fp, claims[i].spec->claim_text);
        fputs(",\"summary\":", fp);
        paper_write_json_string(fp, claims[i].summary);
        fputs(",\"status\":", fp);
        paper_write_json_string(fp, claims[i].status);
        fprintf(fp,
                ",\"required_for_submission\":%s,\"source\":",
                claims[i].spec->required ? "true" : "false");
        paper_write_json_string(fp, claims[i].spec->source);
        fputs(",\"freshness_prerequisite\":", fp);
        paper_write_json_string(fp,
                                paper_claim_has_text(claims[i].spec->freshness_prerequisite)
                                    ? claims[i].spec->freshness_prerequisite
                                    : "");
        fputs(",\"refresh_command\":", fp);
        paper_write_json_string(fp,
                                paper_claim_has_text(claims[i].spec->refresh_command)
                                    ? claims[i].spec->refresh_command
                                    : "");
        fputs(",\"metrics\":{\"raw_top1\":", fp);
        paper_write_json_number(fp, claims[i].raw_top1);
        fputs(",\"hand_top1\":", fp);
        paper_write_json_number(fp, claims[i].hand_top1);
        fputs(",\"learned_top1\":", fp);
        paper_write_json_number(fp, claims[i].learned_top1);
        fputs(",\"learned_top10\":", fp);
        paper_write_json_number(fp, claims[i].learned_top10);
        fprintf(fp,
                ",\"learned_vs_hand_delta_pp\":%.12g,\"false_rejections\":%ld,\"sample_count\":%ld}",
                claims[i].learned_vs_hand_delta_pp,
                claims[i].false_rejections,
                claims[i].sample_count);
        fputs("}", fp);
        fputs(i + 1U == PAPER_CLAIM_COUNT ? "\n" : ",\n", fp);
    }
    fputs("  ],\n", fp);
    fputs("  \"limitations\":[\n", fp);
    {
        int first = 1;
        for (i = 0U; i < PAPER_CLAIM_COUNT; ++i) {
            if (strcmp(claims[i].status, "supported") == 0) {
                continue;
            }
            if (!first) {
                fputs(",\n", fp);
            }
            fputs("    {\"claim_id\":", fp);
            paper_write_json_string(fp, claims[i].spec->id);
            fputs(",\"status\":", fp);
            paper_write_json_string(fp, claims[i].status);
            fputs(",\"title\":", fp);
            paper_write_json_string(fp, claims[i].spec->title);
            fputs(",\"text\":", fp);
            paper_write_json_string(fp, claims[i].summary);
            fputc('}', fp);
            first = 0;
        }
    }
    fputs("\n  ]\n}\n", fp);
    fclose(fp);
    return 0;
}

static int paper_write_claims_typ(const char *path,
                                  const PaperClaimResult claims[PAPER_CLAIM_COUNT]) {
    FILE *fp;
    size_t i;
    long supported;
    long contradicted;
    long underpowered;
    long missing;
    long stale;
    int ok = paper_claim_support_ok(claims);
    paper_count_claim_statuses(claims,
                               &supported,
                               &contradicted,
                               &underpowered,
                               &missing,
                               &stale);
    if (paper_mkdir_parent(path) != 0) {
        return -1;
    }
    fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    fputs("// AUTOGENERATED by the C neural grammar discovery runtime.\n", fp);
    fputs("// Do not edit by hand; rerun `make paper-claims` after results change.\n\n",
          fp);
    fprintf(fp,
            "#let paper_claim_status_counts = (supported: %ld, contradicted: %ld, underpowered: %ld, missing: %ld, stale: %ld)\n",
            supported,
            contradicted,
            underpowered,
            missing,
            stale);
    fprintf(fp, "#let paper_claim_support_ok = %s\n", ok ? "true" : "false");
    fputs("#let paper_claim_required_failures = (", fp);
    for (i = 0U; i < PAPER_CLAIM_COUNT; ++i) {
        if (claims[i].spec->required && strcmp(claims[i].status, "supported") != 0) {
            paper_write_typ_string(fp, claims[i].spec->id);
            fputs(", ", fp);
        }
    }
    fputs(")\n", fp);
    fputs("#let paper_claim_surfaced_unsupported_failures = ()\n", fp);
    fputs("#let paper_claim_limitations = (", fp);
    for (i = 0U; i < PAPER_CLAIM_COUNT; ++i) {
        if (strcmp(claims[i].status, "supported") == 0) {
            continue;
        }
        fputs("(claim_id: ", fp);
        paper_write_typ_string(fp, claims[i].spec->id);
        fputs(", status: ", fp);
        paper_write_typ_string(fp, claims[i].status);
        fputs(", severity: \"scope\", title: ", fp);
        paper_write_typ_string(fp, claims[i].spec->title);
        fputs(", text: ", fp);
        paper_write_typ_string(fp, claims[i].summary);
        fputs(", roadmap_items: (), residual: ()), ", fp);
    }
    fputs(")\n", fp);
    fputs("#let paper_claim_support_surfaces = (", fp);
    for (i = 0U; i < PAPER_CLAIM_COUNT; ++i) {
        fputs("(id: ", fp);
        paper_write_typ_string(fp, claims[i].spec->id);
        fputs(", family: ", fp);
        paper_write_typ_string(fp, claims[i].spec->family);
        fputs(", status: ", fp);
        paper_write_typ_string(fp, claims[i].status);
        fprintf(fp, ", required: %s, title: ", claims[i].spec->required ? "true" : "false");
        paper_write_typ_string(fp, claims[i].spec->title);
        fputs(", summary: ", fp);
        paper_write_typ_string(fp, claims[i].summary);
        fputs(", source: ", fp);
        paper_write_typ_string(fp, claims[i].spec->source);
        fputs(", freshness_prerequisite: ", fp);
        paper_write_typ_string(fp,
                               paper_claim_has_text(claims[i].spec->freshness_prerequisite)
                                   ? claims[i].spec->freshness_prerequisite
                                   : "");
        fputs(", refresh_command: ", fp);
        paper_write_typ_string(fp,
                               paper_claim_has_text(claims[i].spec->refresh_command)
                                   ? claims[i].spec->refresh_command
                                   : "");
        fputs(", chart_kind: ", fp);
        paper_write_typ_string(fp, claims[i].spec->chart_kind);
        fputs(", chart_rows: ", fp);
        fputs(claims[i].chart_rows[0] == '\0' ? "()" : claims[i].chart_rows, fp);
        fputs("), ", fp);
    }
    fputs(")\n\n", fp);
    for (i = 0U; i < PAPER_CLAIM_COUNT; ++i) {
        char slug[96];
        paper_claim_slug(claims[i].spec->id, slug);
        fprintf(fp, "#let claim_%s_status = ", slug);
        paper_write_typ_string(fp, claims[i].status);
        fputs("\n#let claim_", fp);
        fputs(slug, fp);
        fputs("_summary = ", fp);
        paper_write_typ_string(fp, claims[i].summary);
        fputs("\n#let claim_", fp);
        fputs(slug, fp);
        fputs("_explanation = ", fp);
        paper_write_typ_string(fp, claims[i].explanation);
        fputs("\n#let claim_", fp);
        fputs(slug, fp);
        fputs("_chart_rows = ", fp);
        fputs(claims[i].chart_rows[0] == '\0' ? "()" : claims[i].chart_rows, fp);
        fputs("\n#let claim_", fp);
        fputs(slug, fp);
        fputs("_chart = (kind: ", fp);
        paper_write_typ_string(fp, claims[i].spec->chart_kind);
        fputs(", unit: ", fp);
        paper_write_typ_string(fp, claims[i].spec->chart_unit);
        fputs(", y_label: ", fp);
        paper_write_typ_string(fp, claims[i].spec->chart_y_label);
        fputs(", rows: ", fp);
        fputs(claims[i].chart_rows[0] == '\0' ? "()" : claims[i].chart_rows, fp);
        fputs(")\n\n", fp);
    }
    fputs("#let claim_C1_prime_gate_delta_raw_top1 = ", fp);
    paper_write_typ_float(fp, claims[0].raw_top1);
    fputs("\n#let claim_C1_prime_gate_delta_hand_top1 = ", fp);
    paper_write_typ_float(fp, claims[0].hand_top1);
    fputs("\n#let claim_C1_prime_gate_delta_learned_top1 = ", fp);
    paper_write_typ_float(fp, claims[0].learned_top1);
    fputs("\n#let claim_C1_prime_gate_delta_learned_top10 = ", fp);
    paper_write_typ_float(fp, claims[0].learned_top10);
    fputs("\n#let claim_C1_prime_gate_delta_learned_vs_hand_delta_pp = ", fp);
    paper_write_typ_float(fp, claims[0].learned_vs_hand_delta_pp);
    fprintf(fp,
            "\n#let claim_C1_prime_gate_delta_false_rejections = %ld\n#let claim_C1_prime_gate_delta_sample_count = %ld\n",
            claims[0].false_rejections,
            claims[0].sample_count);
    fclose(fp);
    return 0;
}

static int paper_write_claim_reports(const char *output_dir,
                                     const PaperClaimResult claims[PAPER_CLAIM_COUNT]) {
    char path[4096];
    char text[8192];
    size_t i;
    int ok = paper_claim_support_ok(claims);
    text[0] = '\0';
    snprintf(text,
             sizeof(text),
             "# Paper Support Report\n\nGenerated by `make paper-support` using the C neural grammar discovery runtime.\n\nStatus: %s\n\n",
             ok ? "passing" : "failing");
    for (i = 0U; i < PAPER_CLAIM_COUNT; ++i) {
        paper_appendf(text,
                      sizeof(text),
                      "- `%s`: %s - %s\n",
                      claims[i].spec->id,
                      claims[i].status,
                      claims[i].summary);
    }
    snprintf(path, sizeof(path), "%s/support_report.md", output_dir);
    if (paper_write_text_file(path, text) != 0) {
        return -1;
    }
    text[0] = '\0';
    snprintf(text,
             sizeof(text),
             "# Paper Claim Gap Report\n\nGenerated by `make paper-claims`.\n\n");
    for (i = 0U; i < PAPER_CLAIM_COUNT; ++i) {
        if (strcmp(claims[i].status, "supported") != 0) {
            paper_appendf(text,
                          sizeof(text),
                          "- `%s`: %s - %s\n",
                          claims[i].spec->id,
                          claims[i].status,
                          claims[i].summary);
        }
    }
    snprintf(path, sizeof(path), "%s/gap_report.md", output_dir);
    if (paper_write_text_file(path, text) != 0) {
        return -1;
    }
    text[0] = '\0';
    snprintf(text,
             sizeof(text),
             "# Generated Limitations\n\nGenerated by `make paper-claims` from non-supported claim states.\n\n");
    for (i = 0U; i < PAPER_CLAIM_COUNT; ++i) {
        if (strcmp(claims[i].status, "supported") != 0) {
            paper_appendf(text,
                          sizeof(text),
                          "- `%s`: %s\n",
                          claims[i].spec->id,
                          claims[i].summary);
        }
    }
    snprintf(path, sizeof(path), "%s/limitations.md", output_dir);
    if (paper_write_text_file(path, text) != 0) {
        return -1;
    }
    snprintf(path, sizeof(path), "%s/roadmap_links.md", output_dir);
    if (paper_write_text_file(path,
                              "# Roadmap Links\n\nGenerated by `make paper-claims`; roadmap link enforcement is handled by the C claim gate.\n") != 0) {
        return -1;
    }
    snprintf(path, sizeof(path), "%s/agent_instructions.md", output_dir);
    if (paper_write_text_file(path,
                              "# Agent Instructions\n\nGenerated by `make paper-claims` from the C claim compiler. LLMs may propose text; the C verifier decides claim status.\n") != 0) {
        return -1;
    }
    snprintf(path, sizeof(path), "%s/paper_agent_trace.md", output_dir);
    if (paper_write_text_file(path,
                              "# Paper Agent Trace\n\nGenerated by `make paper-claims`. External paper-agent proposals are intentionally not required for the deterministic support gate.\n") != 0) {
        return -1;
    }
    snprintf(path, sizeof(path), "%s/paper_agent_trace.jsonl", output_dir);
    if (paper_write_text_file(path,
                              "{\"schema\":\"crlp.paper_agent_candidate.v1\",\"proposer_kind\":\"claim_compiler\",\"verifier_status\":\"accepted\",\"candidate_type\":\"audit_summary\",\"text\":\"C claim compiler generated deterministic paper support artifacts.\"}\n") != 0) {
        return -1;
    }
    return 0;
}

typedef struct {
    const char *domain;
    const char *domain_display;
    const char *data_domain;
    const char *grounding_authority;
    const char *authority_claim_scope;
    const char *grammar;
    const char *candidate_set;
    const char *verifier;
    const char *certificate_gate;
    const char *scorer_role;
    const char *neural_scorer_role;
    const char *neural_proposer_role;
    const char *headline_baseline;
    const char *trace_artifact;
    const char *examples[4];
    size_t example_count;
    const char *result_table;
    const char *result;
    const char *claim;
    const char *takeaway;
    const char *limitation;
    const char *claim_ids[4];
    size_t claim_id_count;
} PaperExperimentMatrixRow;

static const PaperExperimentMatrixRow PAPER_EXPERIMENT_MATRIX_ROWS[] = {
    {
        "prime_gap",
        "arithmetic sequence filtering",
        "next-prime gaps over segmented-sieve windows; broader family includes period-residue sequence filters",
        "exact_check",
        "zero-false-rejection gates and exact residual ranking inside the next-prime support",
        "local modular filters over residues, gap classes, and wheel support",
        "next-prime gap actions plus certified modular filters",
        "segmented-sieve replay and modular certificate checks",
        "zero-false-rejection divisibility certificates for the audited next-prime instance",
        "rank residual gaps and budgeted certified filters",
        "score residual actions and certified filters after the gate defines the target",
        "static proposer is a negative baseline; closed-loop and unified proposer rows must be compared to grammar order",
        "scorer baseline is the certified gate; proposer baseline is grammar order plus verifier-feedback variants",
        "crlp.compare.v1; crlp.rule_trace.v1; crlp.rule_search.summary.v1",
        {"modular reject rule",
         "wheel support certificate",
         "next-prime gap action",
         "Pisano-period recurrence gate"},
        4U,
        "learned filter 68.87% top-1 vs 45.12% hand-wheel and 9.13% raw; seed interval audited",
        "learned filter 68.87% top-1 vs 45.12% hand-wheel and 9.13% raw; seed interval audited",
        "certified modular filters shrink the residual action space",
        "certified modular structure is the candidate mechanism; the neural scorer ranks residual budget choices while Fibonacci/Lucas-style filters remain boundary evidence.",
        "static neural proposer is not superior to deterministic baselines at fixed budget",
        {"C1.prime_gate_delta", "C1.prime_gate_delta_ci", "C2.prime_proposer_fixed_budget", NULL},
        3U,
    },
    {
        "grid_path",
        "grid path",
        "grid mazes with deterministic shortest-path labels",
        "exact_check",
        "shortest-path successor and certificate claims inside generated grids",
        "local moves plus boundary, trap, path-clear, and distance predicates",
        "four moves plus symbolic local/topology/bound predicates",
        "BFS shortest-path successor and rule certificate checks",
        "legality, trap, clear-path, and distance-dominance certificates",
        "rank legal residual moves after symbolic constraints",
        "rank legal residual moves under exact shortest-path labels",
        "budgeted rule-order diagnostics only",
        "BFS successor and cheap symbolic gates",
        "crlp.compare.v1; crlp.symbolic_domain.trace.v1",
        {"legal move", "trap reject", "distance-dominance gate", NULL},
        3U,
        "cheap-bound gate lifts priors-enabled raw 75.80% to 90.10%",
        "cheap-bound gate lifts priors-enabled raw 75.80% to 90.10%",
        "same gate contract works outside residue arithmetic",
        "the same candidate-verifier contract transfers to topology once the grammar stays domain-local.",
        "supporting transfer surface, not a main contribution",
        {NULL, NULL, NULL, NULL},
        0U,
    },
    {
        "delayed_claim",
        "delayed claim",
        "delayed reward/claim rows with hidden outcomes and margins",
        "deterministic_replay",
        "simulator-certified utility and safety under the delayed-claim generator",
        "threshold, margin, and sibling-claim predicates",
        "claim actions, threshold gates, and utility-margin rules",
        "deterministic hidden-outcome simulator and rule certificate checks",
        "finite-generator safety check plus false-rejection counters",
        "rank utility only after safety dominates the objective",
        "rank utility after simulator safety dominates the decision",
        "consume simulator/gate rejection reasons as scoped feedback",
        "deterministic clean threshold and margin gates",
        "crlp.delayed_claim.rule_trace.v1; crlp.rejected_candidate_learning.*",
        {"safe threshold claim", "margin reject", "unsafe reason-prior transfer", NULL},
        3U,
        "clean threshold+margin gate reduces four actions to one; noisy rows expose failures",
        "clean threshold+margin gate reduces four actions to one; noisy rows expose failures",
        "reason learning needs domain-aware safety constraints",
        "utility ranking must be downstream of safety when false rejection risk is domain-specific.",
        "reason-prior transfer alone can fail safety; claim remains domain-scoped",
        {"C4.delayed_claim_sibling_transfer", NULL, NULL, NULL},
        1U,
    },
    {
        "rejected_candidate_learning",
        "rejected candidate learning",
        "verifier rejection rows grouped by seed, split, and reason family",
        "exact_check",
        "reason-code feedback on arithmetic/modular verifier rows; replay stress rows are limitations only",
        "the original domain grammars that produced rejected candidate rows",
        "rejected verifier rows grouped by task, split, reason family, and seed",
        "original rule/certificate checker that rejected or accepted each row",
        "false-rejection-first budget filter over reason priors and residual scores",
        "learn residual ordering from rejection reasons and marginal verifier utility",
        "train residual ordering from rejected verifier outcomes",
        "feed reason codes and residual classes into the next proposal pass",
        "unguided model_score versus reason_prior and trained_residual",
        "crlp.rejected_candidate_learning.*",
        {"unsafe candidate", "no-support row", "trained residual recovery", NULL},
        3U,
        "trained residual reaches 7 accepted rules / 245 utility with zero false rejections",
        "trained residual reaches 7 accepted rules / 245 utility with zero false rejections",
        "verifier failures become training signal",
        "rejected verifier rows are reusable supervision, not discarded failed searches.",
        "not a generic reason-code ranker; delayed-claim safety remains scoped",
        {"C5.rejected_candidate_reason_learning", NULL, NULL, NULL},
        1U,
    },
    {
        "hypothesis_library_feedback",
        "hypothesis library feedback",
        "ARC-like hypothesis fixtures split into promotion and held-out task ids",
        "exact_check",
        "train-pair exactness and held-out fixture certification under the typed hypothesis grammar",
        "typed templates, slot bindings, and compiled program candidates",
        "certified hypothesis templates, slot bindings, and compiled programs",
        "train-pair exactness, type checks, and held-out fixture split evaluation",
        "zero-false-rejection promotion before a binding enters the library",
        "rank promoted slot priors and residual program candidates under tight budgets",
        "rank bounded program candidates after template certification",
        "turn zero-false-rejection templates into slot priors for the next search",
        "no_prior and grammar-order tight-budget selection",
        "crlp.hypothesis.evidence.v1; crlp.hypothesis_library.heldout_result.v1",
        {"slot prior", "promoted template", "held-out certified@1 row", NULL},
        3U,
        "slot priors raise certified@1 from 2 to 4; held-out split rows = 30",
        "slot priors raise certified@1 from 2 to 4; held-out split rows = 30",
        "certified learning can accumulate through reusable library priors",
        "certified hypotheses can become library priors and improve tight-budget held-out certification.",
        "public ARC promotion remains scoped evidence, not a headline ARC score",
        {"C5.hypothesis_library_promotion", NULL, NULL, NULL},
        1U,
    },
    {
        "multitask_tiny_net",
        "multitask tiny net",
        "shared residual rows from prime, grid, claim, and ARC surfaces",
        "mixed_diagnostic",
        "diagnostic scorer rows only; not aggregated into an authority-level leaderboard",
        "domain-local grammars encoded through common residual feature rows",
        "shared residual ranking rows across prime/grid/claim/ARC",
        "task-local labels and certified downstream gates",
        "per-task held-out metrics, OOD ARC variants, calibration bins",
        "shared residual scorer with domain-specific diagnostics",
        "shared residual scorer with per-domain diagnostics",
        "none promoted; architecture rows are blocked by the plateau audit",
        "single-task scorer and C7 plateau audit",
        "crlp.multitask.summary.v1; crlp.multitask.tiny_net_plateau.v1; crlp.multitask.tiny_net_diagnostic.v1",
        {"shared trunk row", "ARC OOD row", "capacity plateau row", NULL},
        3U,
        "ARC OOD shared +5.57pp; capacity is not monotonic; 128-epoch plateau audit is underpowered",
        "ARC OOD shared +5.57pp; capacity is not monotonic; 128-epoch plateau audit is underpowered",
        "representation/protocol limits dominate naive width scaling",
        "small rankers are useful around certified structure, but one flat head exposes interference.",
        "architecture-size claims stay gated by plateau-ready diagnostics",
        {"C4.arc_ood_shared_vs_single", "C4.multitask_capacity_tradeoff", "C7.tiny_net_plateau_stop_rule", NULL},
        3U,
    },
    {
        "arc_agi_evidence_surface",
        "arc agi evidence surface",
        "public and fixture ARC tasks with cell, object, transform, and hypothesis candidates",
        "corpus_proxy",
        "public-corpus coverage and failure taxonomy, not broad exact ARC competence",
        "typed transforms, object/slot bindings, and bounded program templates",
        "cell colors, typed transforms, programs, and hypothesis-derived candidates",
        "train-pair consistency, unique compiled prediction, and public corpus loader",
        "program/hypothesis certificate rows and scorer-invariant loader summaries",
        "rank bounded candidates; scorer does not own correctness",
        "rank bounded candidates and expose calibration/coverage gaps",
        "grow certified transform/program grammar coverage rather than claim scorer capacity fixes",
        "corpus loader coverage and train-pair certified transform rows",
        "crlp.arc_loader.summary.v1; crlp.program_proposer.summary.v1; crlp.hypothesis.evidence.v1",
        {"typed transform", "hypothesis candidate", "loader summary", NULL},
        3U,
        "public eval coverage is 7/400 tasks and 1057 cells under the measured surface",
        "public eval coverage is 7/400 tasks and 1057 cells under the measured surface",
        "ARC exposes priors, grammar coverage, and failure taxonomy",
        "ARC is an evidence surface for grammar coverage and failure taxonomy, not a score chase.",
        "not a broad ARC competence or benchmark-score claim",
        {"C1.arc_gate_scorer_invariance", "C1.arc_public_evidence_surface", "C3.arc_ood_calibration_tradeoff", NULL},
        3U,
    },
    {
        "arc_verified_self_play",
        "arc verified self play",
        "generated ARC-like tasks from forward visual-primitive composition plus existing ARC visual-pipeline attempts",
        "exact_check",
        "exact synthetic supervision for inverse proposer training; support requires state-hash leakage disjointness and public ARC fixed-budget replay against no-prior baseline logs",
        "visual primitives for object motion, anchor-relative translation, recolor, tile/scale, crop/extract, projection/panel, counting, fill, and symmetry",
        "forward-composed visual primitive programs, verifier-shaped negative siblings, and exported compiler-backed visual priors",
        "known-correct synthetic output construction, ARC visual-pipeline verifier fields, and full/matched visual-only public ARC replay logs",
        "accepted rows are generated by primitive composition; public coverage requires ARC verifier replay against full and matched visual-only baseline logs",
        "train inverse visual-program induction from exact self-play rows",
        "none as correctness owner; ranker learns inverse program ordering from generated verifier rows",
        "self-play proposer trained on accepted programs and verifier-reason negative siblings, then exported as compiler-backed visual priors; synthetic top-1 is in-distribution only",
        "full no-prior public replay, matched visual-only no-prior public replay, authored template baseline 7 certified eval tasks, oracle ceiling 14, and nine zero-coverage miss families",
        "crlp.arc_verified_self_play.summary.v1; crlp.arc_verified_self_play.prior.v1; crlp.arc_synthetic_shape_curriculum.v1; crlp.protocol.verified_selfplay_sample.v1; crlp.neural_grammar_discovery.arc_visual_pipeline_neural_summary.v1",
        {"verified self-play sample", "forward-composed accepted program", "fixed-budget public ablation", NULL},
        3U,
        "self-play is registered as an underpowered L1 search-efficiency surface until fixed-budget public replay, leakage disjointness, seed count, and zero-family opening gates pass",
        "self-play is registered as an underpowered L1 search-efficiency surface until fixed-budget public replay, leakage disjointness, seed count, and zero-family opening gates pass",
        "ARC proposer training can be generated from verifier-owned primitives, but support requires fixed-budget public replay rather than synthetic in-distribution top-1",
        "the durable ARC asset is the verifier/primitive system; authored solution libraries are baselines, not the training source.",
        "distribution mismatch, leakage audit, and public fixed-budget attribution remain open before self-play can be credited",
        {"C5.arc_verified_self_play", NULL, NULL, NULL},
        1U,
    },
    {
        "signal_flight",
        "signal flight",
        "Signal flight replay frames, closed-loop rollout rows, live shadow-control comparisons, and live neural-active rows",
        "deterministic_replay",
        "simulator replay, rollout replay, protocol-compatible live telemetry, and guarded safety evidence",
        "bounded action/control-option candidates beside the heuristic controller",
        "certified control-target hypotheses plus nine WASD action candidates",
        "offline counterfactual simulator replay, closed-loop rollout replay, live shadow frame comparison, and live active telemetry",
        "replay utility, regret, damage/death counters, certified control-target rollout rows, shadow disagreement logs, live safety gates, and protocol compatibility rows",
        "rank bounded control candidates beside heuristic control",
        "rank bounded WASD/control candidates beside heuristic control",
        "control-target hypotheses remain replay-gated before action ranking",
        "heuristic controller, simulator replay, and live guardrails",
        "crlp.signal_flight.summary.v1; crlp.signal_flight_rollout.summary.v1; crlp.signal_shadow_replay_audit.v1; crlp.signal_live_active_summary.v1",
        {"WASD candidate", "counterfactual utility row", "rollout controller row", "shadow disagreement row"},
        4U,
        "offline replay, rollout bridge, and live neural-active claims supported; active baseline comparison remains open",
        "offline replay, rollout bridge, and guarded live neural-active evidence are reported; active baseline comparison remains open",
        "application pressure for replay-grounded residual ranking and guarded live neural-active execution",
        "Signal adds application pressure through replay, rollout, shadow, and guarded live artifacts while keeping active-control performance open.",
        "active neural versus heuristic performance is not yet claimed",
        {"C9.signal_flight_replay_control", "C9.signal_live_neural_active", NULL, NULL},
        2U,
    },
};

static const size_t PAPER_EXPERIMENT_MATRIX_ROW_COUNT =
    sizeof(PAPER_EXPERIMENT_MATRIX_ROWS) / sizeof(PAPER_EXPERIMENT_MATRIX_ROWS[0]);

static const PaperClaimResult *paper_find_claim(
    const PaperClaimResult claims[PAPER_CLAIM_COUNT],
    const char *id) {
    size_t i;
    for (i = 0U; i < PAPER_CLAIM_COUNT; ++i) {
        if (strcmp(claims[i].spec->id, id) == 0) {
            return &claims[i];
        }
    }
    return NULL;
}

static void paper_today_iso(char out[11]) {
    time_t now;
    struct tm tm_value;
    now = time(NULL);
    if (now == (time_t)-1) {
        strcpy(out, "unknown");
        return;
    }
    if (localtime_r(&now, &tm_value) == NULL ||
        strftime(out, 11U, "%Y-%m-%d", &tm_value) == 0U) {
        strcpy(out, "unknown");
    }
}

static void paper_matrix_join_examples(const PaperExperimentMatrixRow *row,
                                       char *out,
                                       size_t out_size) {
    size_t i;
    out[0] = '\0';
    for (i = 0U; i < row->example_count; ++i) {
        if (i > 0U) {
            paper_appendf(out, out_size, "; ");
        }
        paper_appendf(out, out_size, "%s", row->examples[i]);
    }
}

static void paper_matrix_effective_result(
    const PaperExperimentMatrixRow *row,
    const PaperClaimResult claims[PAPER_CLAIM_COUNT],
    char *result,
    size_t result_size,
    char *result_table,
    size_t result_table_size) {
    const PaperClaimResult *replay;
    const PaperClaimResult *live;
    snprintf(result, result_size, "%s", row->result);
    snprintf(result_table, result_table_size, "%s", row->result_table);
    if (strcmp(row->domain, "signal_flight") != 0) {
        return;
    }
    replay = paper_find_claim(claims, "C9.signal_flight_replay_control");
    live = paper_find_claim(claims, "C9.signal_live_neural_active");
    if (replay != NULL && replay->summary[0] != '\0' &&
        live != NULL && live->summary[0] != '\0') {
        snprintf(result,
                 result_size,
                 "%s %s",
                 replay->summary,
                 live->summary);
    }
    if (live != NULL && strcmp(live->status, "supported") == 0) {
        snprintf(result_table,
                 result_table_size,
                 "offline replay, rollout bridge, and live neural-active claims supported; active baseline comparison remains open");
    } else if (live != NULL) {
        snprintf(result_table,
                 result_table_size,
                 "offline replay and rollout bridge supported; live neural-active evidence %s pending protocol-compatible rows; active baseline comparison remains open",
                 live->status);
    }
}

static int paper_matrix_validate_row(const PaperExperimentMatrixRow *row) {
    if (row->domain == NULL || row->domain[0] == '\0' ||
        row->data_domain == NULL || row->data_domain[0] == '\0' ||
        row->grounding_authority == NULL ||
        row->grounding_authority[0] == '\0' ||
        strcmp(row->grounding_authority, "unclassified") == 0 ||
        row->authority_claim_scope == NULL ||
        row->authority_claim_scope[0] == '\0' ||
        row->grammar == NULL || row->grammar[0] == '\0' ||
        row->candidate_set == NULL || row->candidate_set[0] == '\0' ||
        row->verifier == NULL || row->verifier[0] == '\0' ||
        row->certificate_gate == NULL || row->certificate_gate[0] == '\0' ||
        row->scorer_role == NULL || row->scorer_role[0] == '\0' ||
        row->neural_scorer_role == NULL ||
        row->neural_scorer_role[0] == '\0' ||
        row->neural_proposer_role == NULL ||
        row->neural_proposer_role[0] == '\0' ||
        row->headline_baseline == NULL ||
        row->headline_baseline[0] == '\0' ||
        row->trace_artifact == NULL || row->trace_artifact[0] == '\0' ||
        row->result_table == NULL || row->result_table[0] == '\0' ||
        row->takeaway == NULL || row->takeaway[0] == '\0' ||
        row->limitation == NULL || row->limitation[0] == '\0') {
        return -1;
    }
    if (row->example_count < 2U || row->example_count > 4U) {
        return -1;
    }
    return 0;
}

static void paper_write_json_string_array(FILE *fp,
                                          const char *const *items,
                                          size_t count) {
    size_t i;
    fputc('[', fp);
    for (i = 0U; i < count; ++i) {
        if (i > 0U) {
            fputs(", ", fp);
        }
        paper_write_json_string(fp, items[i]);
    }
    fputc(']', fp);
}

static void paper_write_matrix_claim_statuses(
    FILE *fp,
    const PaperExperimentMatrixRow *row,
    const PaperClaimResult claims[PAPER_CLAIM_COUNT]) {
    size_t i;
    fputc('{', fp);
    for (i = 0U; i < row->claim_id_count; ++i) {
        const PaperClaimResult *claim = paper_find_claim(claims, row->claim_ids[i]);
        if (i > 0U) {
            fputs(", ", fp);
        }
        paper_write_json_string(fp, row->claim_ids[i]);
        fputs(": ", fp);
        paper_write_json_string(fp, claim == NULL ? "not_registered" : claim->status);
    }
    fputc('}', fp);
}

static int paper_write_experiment_matrix_jsonl(
    const char *path,
    const PaperClaimResult claims[PAPER_CLAIM_COUNT]) {
    FILE *fp;
    char date[11];
    size_t i;
    if (paper_mkdir_parent(path) != 0) {
        return -1;
    }
    fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    paper_today_iso(date);
    for (i = 0U; i < PAPER_EXPERIMENT_MATRIX_ROW_COUNT; ++i) {
        const PaperExperimentMatrixRow *row = &PAPER_EXPERIMENT_MATRIX_ROWS[i];
        char result[4096];
        char result_table[1024];
        paper_matrix_effective_result(row,
                                      claims,
                                      result,
                                      sizeof(result),
                                      result_table,
                                      sizeof(result_table));
        fputs("{\"authority_claim_scope\": ", fp);
        paper_write_json_string(fp, row->authority_claim_scope);
        fputs(", \"candidate_set\": ", fp);
        paper_write_json_string(fp, row->candidate_set);
        fputs(", \"certificate_gate\": ", fp);
        paper_write_json_string(fp, row->certificate_gate);
        fputs(", \"claim\": ", fp);
        paper_write_json_string(fp, row->claim);
        fputs(", \"claim_ids\": ", fp);
        paper_write_json_string_array(fp, row->claim_ids, row->claim_id_count);
        fputs(", \"claim_statuses\": ", fp);
        paper_write_matrix_claim_statuses(fp, row, claims);
        fputs(", \"data_domain\": ", fp);
        paper_write_json_string(fp, row->data_domain);
        fputs(", \"date\": ", fp);
        paper_write_json_string(fp, date);
        fputs(", \"domain\": ", fp);
        paper_write_json_string(fp, row->domain);
        fputs(", \"domain_display\": ", fp);
        paper_write_json_string(fp, row->domain_display);
        fputs(", \"examples\": ", fp);
        paper_write_json_string_array(fp, row->examples, row->example_count);
        fputs(", \"grammar\": ", fp);
        paper_write_json_string(fp, row->grammar);
        fputs(", \"grounding_authority\": ", fp);
        paper_write_json_string(fp, row->grounding_authority);
        fputs(", \"headline_baseline\": ", fp);
        paper_write_json_string(fp, row->headline_baseline);
        fputs(", \"limitation\": ", fp);
        paper_write_json_string(fp, row->limitation);
        fputs(", \"neural_proposer_role\": ", fp);
        paper_write_json_string(fp, row->neural_proposer_role);
        fputs(", \"neural_scorer_role\": ", fp);
        paper_write_json_string(fp, row->neural_scorer_role);
        fputs(", \"result\": ", fp);
        paper_write_json_string(fp, result);
        fputs(", \"result_table\": ", fp);
        paper_write_json_string(fp, result_table);
        fputs(", \"schema\": ", fp);
        paper_write_json_string(fp, "crlp.experiment_matrix.row.v1");
        fputs(", \"scorer_role\": ", fp);
        paper_write_json_string(fp, row->scorer_role);
        fputs(", \"takeaway\": ", fp);
        paper_write_json_string(fp, row->takeaway);
        fputs(", \"trace_artifact\": ", fp);
        paper_write_json_string(fp, row->trace_artifact);
        fputs(", \"verifier\": ", fp);
        paper_write_json_string(fp, row->verifier);
        fputs("}\n", fp);
    }
    fclose(fp);
    return 0;
}

static void paper_write_md_escaped(FILE *fp, const char *value) {
    const char *cursor = value == NULL ? "" : value;
    while (*cursor != '\0') {
        if (*cursor == '|') {
            fputs("\\|", fp);
        } else if (*cursor == '\n' || *cursor == '\r') {
            fputc(' ', fp);
        } else {
            fputc((unsigned char)*cursor, fp);
        }
        ++cursor;
    }
}

static void paper_write_md_examples(FILE *fp, const PaperExperimentMatrixRow *row) {
    char examples[512];
    paper_matrix_join_examples(row, examples, sizeof(examples));
    paper_write_md_escaped(fp, examples);
}

static int paper_write_experiment_matrix_markdown(
    const char *path,
    const PaperClaimResult claims[PAPER_CLAIM_COUNT]) {
    FILE *fp;
    char date[11];
    size_t i;
    static const char *const authorities[] = {
        "exact_check",
        "deterministic_replay",
        "corpus_proxy",
        "human_review",
    };
    if (paper_mkdir_parent(path) != 0) {
        return -1;
    }
    fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    paper_today_iso(date);
    fprintf(fp,
            "# Frozen Experiment Matrix\n\nDate: %s\n\n"
            "The evidence families are frozen to these rows. New domains belong in roadmap notes until this matrix is deliberately revised.\n\n"
            "Authority is a hard reporting axis. Exact checks, deterministic replay, corpus proxies, and mixed diagnostics are not collapsed into a single score.\n\n"
            "| Domain/data | Authority | Grammar | Candidate set | Verifier/gate | Neural role split | Headline baseline | Trace artifact | Result table | Takeaway | Limitation |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n",
            date);
    for (i = 0U; i < PAPER_EXPERIMENT_MATRIX_ROW_COUNT; ++i) {
        const PaperExperimentMatrixRow *row = &PAPER_EXPERIMENT_MATRIX_ROWS[i];
        char result[4096];
        char result_table[1024];
        paper_matrix_effective_result(row,
                                      claims,
                                      result,
                                      sizeof(result),
                                      result_table,
                                      sizeof(result_table));
        fputs("| `", fp);
        paper_write_md_escaped(fp, row->domain);
        fputs("`: ", fp);
        paper_write_md_escaped(fp, row->data_domain);
        fputs(" | `", fp);
        paper_write_md_escaped(fp, row->grounding_authority);
        fputs("`: ", fp);
        paper_write_md_escaped(fp, row->authority_claim_scope);
        fputs(" | ", fp);
        paper_write_md_escaped(fp, row->grammar);
        fputs(" | ", fp);
        paper_write_md_escaped(fp, row->candidate_set);
        fputs(" | ", fp);
        paper_write_md_escaped(fp, row->verifier);
        fputs("; ", fp);
        paper_write_md_escaped(fp, row->certificate_gate);
        fputs(" | scorer: ", fp);
        paper_write_md_escaped(fp, row->neural_scorer_role);
        fputs("; proposer: ", fp);
        paper_write_md_escaped(fp, row->neural_proposer_role);
        fputs(" | ", fp);
        paper_write_md_escaped(fp, row->headline_baseline);
        fputs(" | `", fp);
        paper_write_md_escaped(fp, row->trace_artifact);
        fputs("` | ", fp);
        paper_write_md_escaped(fp, result_table);
        fputs(" | ", fp);
        paper_write_md_escaped(fp, row->takeaway);
        fputs(" | ", fp);
        paper_write_md_escaped(fp, row->limitation);
        fputs(" |\n", fp);
    }
    fputs("\n## Authority Leaderboards\n\n", fp);
    for (i = 0U; i < sizeof(authorities) / sizeof(authorities[0]); ++i) {
        size_t j;
        int found = 0;
        fprintf(fp, "### `%s`\n\n", authorities[i]);
        for (j = 0U; j < PAPER_EXPERIMENT_MATRIX_ROW_COUNT; ++j) {
            const PaperExperimentMatrixRow *row = &PAPER_EXPERIMENT_MATRIX_ROWS[j];
            char result[4096];
            char result_table[1024];
            if (strcmp(row->grounding_authority, authorities[i]) != 0) {
                continue;
            }
            paper_matrix_effective_result(row,
                                          claims,
                                          result,
                                          sizeof(result),
                                          result_table,
                                          sizeof(result_table));
            fputs("- `", fp);
            paper_write_md_escaped(fp, row->domain);
            fputs("`: ", fp);
            paper_write_md_escaped(fp, result_table);
            fputc('\n', fp);
            found = 1;
        }
        if (!found) {
            fputs("- No frozen rows.\n", fp);
        }
        fputc('\n', fp);
    }
    fputs("### `mixed_diagnostic`\n\n", fp);
    for (i = 0U; i < PAPER_EXPERIMENT_MATRIX_ROW_COUNT; ++i) {
        const PaperExperimentMatrixRow *row = &PAPER_EXPERIMENT_MATRIX_ROWS[i];
        if (strcmp(row->grounding_authority, "exact_check") == 0 ||
            strcmp(row->grounding_authority, "deterministic_replay") == 0 ||
            strcmp(row->grounding_authority, "corpus_proxy") == 0 ||
            strcmp(row->grounding_authority, "human_review") == 0) {
            continue;
        }
        fputs("- `", fp);
        paper_write_md_escaped(fp, row->domain);
        fputs("`: ", fp);
        paper_write_md_escaped(fp, row->authority_claim_scope);
        fputc('\n', fp);
    }
    fputs("\n\n## Repeated Protocol Rows\n\n", fp);
    for (i = 0U; i < PAPER_EXPERIMENT_MATRIX_ROW_COUNT; ++i) {
        const PaperExperimentMatrixRow *row = &PAPER_EXPERIMENT_MATRIX_ROWS[i];
        char result[4096];
        char result_table[1024];
        paper_matrix_effective_result(row,
                                      claims,
                                      result,
                                      sizeof(result),
                                      result_table,
                                      sizeof(result_table));
        fprintf(fp, "### `%s`\n\n", row->domain);
        fputs("1. Domain/data: ", fp);
        paper_write_md_escaped(fp, row->data_domain);
        fputs("\n2. Authority: `", fp);
        paper_write_md_escaped(fp, row->grounding_authority);
        fputs("` - ", fp);
        paper_write_md_escaped(fp, row->authority_claim_scope);
        fputs("\n3. Grammar: ", fp);
        paper_write_md_escaped(fp, row->grammar);
        fputs("\n4. Candidate set: ", fp);
        paper_write_md_escaped(fp, row->candidate_set);
        fputs("\n5. Verifier/gate: ", fp);
        paper_write_md_escaped(fp, row->verifier);
        fputs("; ", fp);
        paper_write_md_escaped(fp, row->certificate_gate);
        fputs("\n6. Neural scorer role: ", fp);
        paper_write_md_escaped(fp, row->neural_scorer_role);
        fputs("\n7. Neural proposer role: ", fp);
        paper_write_md_escaped(fp, row->neural_proposer_role);
        fputs("\n8. Headline baseline: ", fp);
        paper_write_md_escaped(fp, row->headline_baseline);
        fputs("\n9. Examples: ", fp);
        paper_write_md_examples(fp, row);
        fputs("\n10. Result table: ", fp);
        paper_write_md_escaped(fp, result_table);
        fputs("\n11. Takeaway: ", fp);
        paper_write_md_escaped(fp, row->takeaway);
        fputs("\n12. Limitation: ", fp);
        paper_write_md_escaped(fp, row->limitation);
        if (i + 1U < PAPER_EXPERIMENT_MATRIX_ROW_COUNT) {
            fputs("\n\n", fp);
        } else {
            fputc('\n', fp);
        }
    }
    fclose(fp);
    return 0;
}

static int paper_write_experiment_matrix_typ(
    const char *path,
    const PaperClaimResult claims[PAPER_CLAIM_COUNT]) {
    FILE *fp;
    size_t i;
    if (paper_mkdir_parent(path) != 0) {
        return -1;
    }
    fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    fputs("// Generated by the C paper runtime; do not edit by hand.\n",
          fp);
    fputs("#let experiment_matrix_rows = (\n", fp);
    for (i = 0U; i < PAPER_EXPERIMENT_MATRIX_ROW_COUNT; ++i) {
        const PaperExperimentMatrixRow *row = &PAPER_EXPERIMENT_MATRIX_ROWS[i];
        char examples[512];
        char result[4096];
        char result_table[1024];
        paper_matrix_join_examples(row, examples, sizeof(examples));
        paper_matrix_effective_result(row,
                                      claims,
                                      result,
                                      sizeof(result),
                                      result_table,
                                      sizeof(result_table));
        fputs("  (\n    domain: ", fp);
        paper_write_typ_string(fp, row->domain);
        fputs(",\n    domain_display: ", fp);
        paper_write_typ_string(fp, row->domain_display);
        fputs(",\n    data_domain: ", fp);
        paper_write_typ_string(fp, row->data_domain);
        fputs(",\n    grounding_authority: ", fp);
        paper_write_typ_string(fp, row->grounding_authority);
        fputs(",\n    authority_claim_scope: ", fp);
        paper_write_typ_string(fp, row->authority_claim_scope);
        fputs(",\n    grammar: ", fp);
        paper_write_typ_string(fp, row->grammar);
        fputs(",\n    candidate_set: ", fp);
        paper_write_typ_string(fp, row->candidate_set);
        fputs(",\n    verifier: ", fp);
        paper_write_typ_string(fp, row->verifier);
        fputs(",\n    certificate_gate: ", fp);
        paper_write_typ_string(fp, row->certificate_gate);
        fputs(",\n    scorer_role: ", fp);
        paper_write_typ_string(fp, row->scorer_role);
        fputs(",\n    neural_scorer_role: ", fp);
        paper_write_typ_string(fp, row->neural_scorer_role);
        fputs(",\n    neural_proposer_role: ", fp);
        paper_write_typ_string(fp, row->neural_proposer_role);
        fputs(",\n    headline_baseline: ", fp);
        paper_write_typ_string(fp, row->headline_baseline);
        fputs(",\n    trace_artifact: ", fp);
        paper_write_typ_string(fp, row->trace_artifact);
        fputs(",\n    examples: ", fp);
        paper_write_typ_string(fp, examples);
        fputs(",\n    result_table: ", fp);
        paper_write_typ_string(fp, result_table);
        fputs(",\n    result: ", fp);
        paper_write_typ_string(fp, result);
        fputs(",\n    claim: ", fp);
        paper_write_typ_string(fp, row->claim);
        fputs(",\n    takeaway: ", fp);
        paper_write_typ_string(fp, row->takeaway);
        fputs(",\n    limitation: ", fp);
        paper_write_typ_string(fp, row->limitation);
        fputs(",\n  ),\n", fp);
    }
    fputs(")\n", fp);
    fclose(fp);
    return 0;
}

static int paper_write_experiment_matrix(
    const char *output_dir,
    const PaperClaimResult claims[PAPER_CLAIM_COUNT]) {
    char jsonl_path[4096];
    char md_path[4096];
    char typ_path[4096];
    size_t i;
    for (i = 0U; i < PAPER_EXPERIMENT_MATRIX_ROW_COUNT; ++i) {
        if (paper_matrix_validate_row(&PAPER_EXPERIMENT_MATRIX_ROWS[i]) != 0) {
            return -1;
        }
    }
    snprintf(jsonl_path, sizeof(jsonl_path), "%s/experiment_matrix.jsonl", output_dir);
    snprintf(md_path, sizeof(md_path), "%s/experiment_matrix.md", output_dir);
    snprintf(typ_path, sizeof(typ_path), "%s/experiment_matrix.typ", output_dir);
    if (paper_write_experiment_matrix_jsonl(jsonl_path, claims) != 0 ||
        paper_write_experiment_matrix_markdown(md_path, claims) != 0 ||
        paper_write_experiment_matrix_typ(typ_path, claims) != 0) {
        return -1;
    }
    return 0;
}

int nn_paper_write_claims(const char *claims_path,
                          const char *output_dir,
                          const char *paper_path,
                          const char *agent_candidates_dir,
                          int support_check,
                          int fail_required,
                          int fail_unlinked_roadmap) {
    PaperClaimResult claims[PAPER_CLAIM_COUNT];
    char json_path[4096];
    char typ_path[4096];
    char registry_error[512];
    int ok;
    (void)paper_path;
    (void)agent_candidates_dir;
    (void)fail_unlinked_roadmap;
    if (claims_path == NULL || claims_path[0] == '\0') {
        claims_path = "paper/claims.yaml";
    }
    if (output_dir == NULL || output_dir[0] == '\0') {
        output_dir = "paper/generated";
    }
    if (paper_mkdirs(output_dir) != 0) {
        return -1;
    }
    if (paper_validate_claim_registry(claims_path,
                                      registry_error,
                                      sizeof(registry_error)) != 0) {
        fprintf(stderr, "paper claim registry validation failed: %s\n", registry_error);
        return -1;
    }
    paper_evaluate_claims(claims);
    snprintf(json_path, sizeof(json_path), "%s/claims.json", output_dir);
    snprintf(typ_path, sizeof(typ_path), "%s/claims.typ", output_dir);
    if (paper_write_claims_json(json_path, claims) != 0 ||
        paper_write_claims_typ(typ_path, claims) != 0 ||
        paper_write_claim_reports(output_dir, claims) != 0 ||
        paper_write_experiment_matrix(output_dir, claims) != 0) {
        return -1;
    }
    ok = paper_claim_support_ok(claims);
    printf("paper claims: support=%s\n", ok ? "passing" : "failing");
    if ((support_check || fail_required) && !ok) {
        return 1;
    }
    return 0;
}
