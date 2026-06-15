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

#define PAPER_CLAIM_COUNT 25U
#define PAPER_CLAIM_SUMMARY_SIZE 1536U
#define PAPER_CLAIM_EXPLANATION_SIZE 1024U
#define PAPER_CLAIM_CHART_SIZE 4096U

#include "nn_paper_claim_registry.inc"
#include "nn_paper_test_api.inc"

#include "nn_paper_claim_evaluators.inc"
static const char *paper_claim_origin_column(const PaperClaimSpec *spec) {
    if (spec == NULL || spec->evidence_origin == NULL || spec->evidence_origin[0] == '\0') {
        return "measured_run";
    }
    return spec->evidence_origin;
}

int paper_write_claim_reports(const char *output_dir,
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
                      "- `%s`: %s - %s (origin: %s)\n",
                      claims[i].spec->id,
                      claims[i].status,
                      claims[i].summary,
                      paper_claim_origin_column(claims[i].spec));
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

#include "nn_paper_experiment_matrix.inc"


#include "nn_paper_evidence_origin.inc"
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
