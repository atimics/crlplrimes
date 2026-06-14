#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define LINE_CAP 65536
#define PATH_CAP 4096
#define NAME_CAP 128
#define MAX_ITEMS 64
#define MAX_RUNS 4096
#define MAX_AGG 1024

typedef struct {
    char value[NAME_CAP];
} Item;

typedef struct {
    Item items[MAX_ITEMS];
    size_t count;
} ItemList;

typedef struct {
    char grid[NAME_CAP];
    char order[NAME_CAP];
    char review_slots[NAME_CAP];
    int min_budget;
    int budget;
    int crossed_into_budget;
    int crossed_out_of_budget;
    int baseline_certified;
    int bucket_certified;
    double baseline_utility;
    double bucket_utility;
    double certified_delta;
    double utility_delta;
} RunRow;

typedef struct {
    char grid[NAME_CAP];
    char review_slots[NAME_CAP];
    int min_budget;
    int budget;
    int run_count;
    int positive_runs;
    int negative_runs;
    int zero_runs;
    int crossed_into_budget;
    int crossed_out_of_budget;
    bool passes_no_regression;
    double min_utility_delta;
    double max_utility_delta;
    double median_utility_delta;
    double median_certified_delta;
    double utility_values[MAX_ITEMS];
    double certified_values[MAX_ITEMS];
} AggRow;

static void die(const char *message)
{
    fprintf(stderr, "%s\n", message);
    exit(2);
}

static void usage(FILE *stream)
{
    fputs("usage: arc_holo_bucket_robustness --arc-bin path --task-dir path --out-dir path\n"
          "  [--max-tasks n] [--orders original,shuffle:seed] [--grids tight,standard]\n"
          "  [--min-budgets n[,n]] [--review-slots auto,1] [--min-pressure-updates n]\n"
          "  [--learned-gate] [--learned-gate-min-examples n] [--learned-gate-threshold r]\n",
          stream);
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
    return p != NULL && strncmp(p, "true", 4) == 0;
}

static void parse_list(ItemList *list, const char *text)
{
    list->count = 0;
    const char *start = text;
    while (*start) {
        const char *end = strchr(start, ',');
        size_t len = end == NULL ? strlen(start) : (size_t)(end - start);
        while (len > 0 && isspace((unsigned char)start[0])) {
            ++start;
            --len;
        }
        while (len > 0 && isspace((unsigned char)start[len - 1u])) {
            --len;
        }
        if (len > 0) {
            if (list->count >= MAX_ITEMS) {
                die("too many list items");
            }
            if (len >= NAME_CAP) {
                len = NAME_CAP - 1u;
            }
            memcpy(list->items[list->count].value, start, len);
            list->items[list->count].value[len] = '\0';
            ++list->count;
        }
        if (end == NULL) {
            break;
        }
        start = end + 1;
    }
}

static void mkdir_p(const char *path)
{
    char cmd[PATH_CAP + 32];
    snprintf(cmd, sizeof(cmd), "mkdir -p \"%s\"", path);
    if (system(cmd) != 0) {
        die("mkdir failed");
    }
}

static void shell_quote(char *out, size_t out_cap, const char *in)
{
    size_t used = 0;
    if (used + 1u < out_cap) {
        out[used++] = '\'';
    }
    for (const char *p = in; *p; ++p) {
        if (*p == '\'') {
            const char *q = "'\\''";
            while (*q && used + 1u < out_cap) {
                out[used++] = *q++;
            }
        } else if (used + 1u < out_cap) {
            out[used++] = *p;
        }
    }
    if (used + 1u < out_cap) {
        out[used++] = '\'';
    }
    if (out_cap > 0) {
        out[used < out_cap ? used : out_cap - 1u] = '\0';
    }
}

static void artifact_token(char *out, size_t out_cap, const char *in)
{
    size_t used = 0;
    for (const unsigned char *p = (const unsigned char *)in; *p; ++p) {
        char ch = (char)*p;
        if (!isalnum(*p) && ch != '-' && ch != '_') {
            ch = '_';
        }
        if (used + 1u < out_cap) {
            out[used++] = ch;
        }
    }
    if (out_cap > 0) {
        out[used] = '\0';
    }
}

static bool read_summary(const char *path, int budget, int *base_cert, double *base_util,
                         int *bucket_cert, double *bucket_util)
{
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
        return false;
    }
    bool have_base = false;
    bool have_bucket = false;
    char line[LINE_CAP];
    while (fgets(line, sizeof(line), fp) != NULL) {
        char schema[128] = {0};
        char strategy[128] = {0};
        if (!json_get_string(line, "schema", schema, sizeof(schema)) ||
            strcmp(schema, "crlp.program_proposer.summary.v1") != 0 ||
            !json_get_string(line, "strategy", strategy, sizeof(strategy))) {
            continue;
        }
        int row_budget = (int)json_get_number(line, "budget", -1.0);
        if (row_budget != budget) {
            continue;
        }
        if (strcmp(strategy, "trace_task_prior") == 0) {
            *base_cert = (int)json_get_number(line, "certified_programs", 0.0);
            *base_util = json_get_number(line, "utility", 0.0);
            have_base = true;
        } else if (strcmp(strategy, "trace_holo_bucket") == 0) {
            *bucket_cert = (int)json_get_number(line, "certified_programs", 0.0);
            *bucket_util = json_get_number(line, "utility", 0.0);
            have_bucket = true;
        }
    }
    fclose(fp);
    return have_base && have_bucket;
}

static void read_diagnostics(const char *path, int budget, int *cross_in, int *cross_out)
{
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        return;
    }
    char line[LINE_CAP];
    while (fgets(line, sizeof(line), fp) != NULL) {
        char schema[128] = {0};
        if (!json_get_string(line, "schema", schema, sizeof(schema)) ||
            strcmp(schema, "crlp.arc_holo_bucket_diagnostic.v1") != 0) {
            continue;
        }
        int row_budget = (int)json_get_number(line, "budget", -1.0);
        if (row_budget != budget) {
            continue;
        }
        if (json_get_bool(line, "crossed_into_budget")) {
            ++*cross_in;
        }
        if (json_get_bool(line, "crossed_out_of_budget")) {
            ++*cross_out;
        }
    }
    fclose(fp);
}

static int budgets_for_grid(const char *grid, int *out, size_t cap)
{
    if (strcmp(grid, "tight") == 0) {
        int values[] = {1, 2, 4, 8};
        size_t n = sizeof(values) / sizeof(values[0]);
        if (cap < n) {
            return 0;
        }
        memcpy(out, values, sizeof(values));
        return (int)n;
    }
    int values[] = {8, 16, 32, 64};
    size_t n = sizeof(values) / sizeof(values[0]);
    if (cap < n) {
        return 0;
    }
    memcpy(out, values, sizeof(values));
    return (int)n;
}

static int cmp_double(const void *lhs, const void *rhs)
{
    double a = *(const double *)lhs;
    double b = *(const double *)rhs;
    return (a > b) - (a < b);
}

static double median(double *values, int count)
{
    if (count <= 0) {
        return 0.0;
    }
    qsort(values, (size_t)count, sizeof(values[0]), cmp_double);
    if ((count & 1) != 0) {
        return values[count / 2];
    }
    return 0.5 * (values[count / 2 - 1] + values[count / 2]);
}

static AggRow *find_or_add_agg(AggRow *aggs, size_t *agg_count, const RunRow *run)
{
    for (size_t i = 0; i < *agg_count; ++i) {
        AggRow *agg = &aggs[i];
        if (agg->budget == run->budget &&
            agg->min_budget == run->min_budget &&
            strcmp(agg->grid, run->grid) == 0 &&
            strcmp(agg->review_slots, run->review_slots) == 0) {
            return agg;
        }
    }
    if (*agg_count >= MAX_AGG) {
        die("too many aggregate rows");
    }
    AggRow *agg = &aggs[(*agg_count)++];
    memset(agg, 0, sizeof(*agg));
    snprintf(agg->grid, sizeof(agg->grid), "%s", run->grid);
    snprintf(agg->review_slots, sizeof(agg->review_slots), "%s", run->review_slots);
    agg->budget = run->budget;
    agg->min_budget = run->min_budget;
    agg->passes_no_regression = true;
    agg->min_utility_delta = run->utility_delta;
    agg->max_utility_delta = run->utility_delta;
    return agg;
}

static void print_json_string(FILE *fp, const char *text)
{
    fputc('"', fp);
    for (const unsigned char *p = (const unsigned char *)text; *p; ++p) {
        if (*p == '"' || *p == '\\') {
            fputc('\\', fp);
            fputc(*p, fp);
        } else if (*p >= 0x20 && *p < 0x7f) {
            fputc(*p, fp);
        } else {
            fputc('?', fp);
        }
    }
    fputc('"', fp);
}

int main(int argc, char **argv)
{
    const char *arc_bin = NULL;
    const char *task_dir = NULL;
    const char *out_dir = NULL;
    const char *orders_text = "original,shuffle:20240516";
    const char *grids_text = "tight,standard";
    const char *min_budgets_text = "2";
    const char *review_slots_text = "auto,1,2,4";
    int max_tasks = 400;
    int min_pressure_updates = 4;
    int learned_gate = 0;
    int learned_gate_min_examples = 32;
    double learned_gate_threshold = 0.5;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--arc-bin") == 0 && i + 1 < argc) {
            arc_bin = argv[++i];
        } else if (strcmp(argv[i], "--task-dir") == 0 && i + 1 < argc) {
            task_dir = argv[++i];
        } else if (strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (strcmp(argv[i], "--max-tasks") == 0 && i + 1 < argc) {
            max_tasks = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--orders") == 0 && i + 1 < argc) {
            orders_text = argv[++i];
        } else if (strcmp(argv[i], "--grids") == 0 && i + 1 < argc) {
            grids_text = argv[++i];
        } else if (strcmp(argv[i], "--min-budgets") == 0 && i + 1 < argc) {
            min_budgets_text = argv[++i];
        } else if (strcmp(argv[i], "--review-slots") == 0 && i + 1 < argc) {
            review_slots_text = argv[++i];
        } else if (strcmp(argv[i], "--min-pressure-updates") == 0 && i + 1 < argc) {
            min_pressure_updates = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--learned-gate") == 0) {
            learned_gate = 1;
        } else if (strcmp(argv[i], "--learned-gate-min-examples") == 0 && i + 1 < argc) {
            learned_gate_min_examples = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--learned-gate-threshold") == 0 && i + 1 < argc) {
            learned_gate_threshold = strtod(argv[++i], NULL);
        } else if (strcmp(argv[i], "--help") == 0) {
            usage(stdout);
            return 0;
        } else {
            usage(stderr);
            return 2;
        }
    }
    if (arc_bin == NULL || task_dir == NULL || out_dir == NULL) {
        usage(stderr);
        return 2;
    }

    ItemList orders;
    ItemList grids;
    ItemList min_budgets;
    ItemList review_slots;
    parse_list(&orders, orders_text);
    parse_list(&grids, grids_text);
    parse_list(&min_budgets, min_budgets_text);
    parse_list(&review_slots, review_slots_text);
    if (orders.count == 0 || grids.count == 0 || min_budgets.count == 0 ||
        review_slots.count == 0) {
        die("empty sweep dimension");
    }

    mkdir_p(out_dir);
    char raw_dir[PATH_CAP];
    snprintf(raw_dir, sizeof(raw_dir), "%s/raw", out_dir);
    mkdir_p(raw_dir);

    RunRow runs[MAX_RUNS];
    size_t run_count = 0;
    int failed = 0;
    char arc_q[PATH_CAP * 2];
    char task_q[PATH_CAP * 2];
    shell_quote(arc_q, sizeof(arc_q), arc_bin);
    shell_quote(task_q, sizeof(task_q), task_dir);

    for (size_t gi = 0; gi < grids.count; ++gi) {
        const char *grid = grids.items[gi].value;
        int budgets[8];
        int budget_count = budgets_for_grid(grid, budgets, 8);
        if (budget_count <= 0) {
            die("unknown grid");
        }
        for (size_t oi = 0; oi < orders.count; ++oi) {
            const char *order = orders.items[oi].value;
            for (size_t mi = 0; mi < min_budgets.count; ++mi) {
                int min_budget = atoi(min_budgets.items[mi].value);
                for (size_t ri = 0; ri < review_slots.count; ++ri) {
                    const char *review = review_slots.items[ri].value;
                    int review_value = strcmp(review, "auto") == 0 ? 0 : atoi(review);
                    char order_token[NAME_CAP];
                    artifact_token(order_token, sizeof(order_token), order);
                    char base[NAME_CAP * 4];
                    snprintf(base,
                             sizeof(base),
                             "%s_%s_min%d_review%s%s",
                             grid,
                             order_token,
                             min_budget,
                             review,
                             learned_gate ? "_learnedgate" : "");
                    char out_path[PATH_CAP];
                    char err_path[PATH_CAP];
                    snprintf(out_path, sizeof(out_path), "%s/%s.jsonl", raw_dir, base);
                    snprintf(err_path, sizeof(err_path), "%s/%s.stderr", raw_dir, base);
                    char out_q[PATH_CAP * 2];
                    char err_q[PATH_CAP * 2];
                    shell_quote(out_q, sizeof(out_q), out_path);
                    shell_quote(err_q, sizeof(err_q), err_path);

                    char order_args[256] = "";
                    if (strncmp(order, "shuffle:", 8) == 0) {
                        snprintf(order_args, sizeof(order_args), " --shuffle --seed %s", order + 8);
                    } else if (strcmp(order, "original") != 0) {
                        die("order must be original or shuffle:seed");
                    }
                    char gate_args[256] = "";
                    if (learned_gate) {
                        snprintf(gate_args,
                                 sizeof(gate_args),
                                 " --arc-holo-bucket-learned-gate --arc-holo-bucket-learned-gate-min-examples %d --arc-holo-bucket-learned-gate-threshold %.8f",
                                 learned_gate_min_examples,
                                 learned_gate_threshold);
                    }
                    char cmd[PATH_CAP * 6];
                    snprintf(cmd,
                             sizeof(cmd),
                             "%s --task-dir %s --max-tasks %d --loader-summary-only --unique-task-gate %s --arc-holo-residual-diagnostics --arc-holo-bucket-min-budget %d --arc-holo-bucket-review-slots %d --arc-holo-bucket-min-pressure-updates %d%s%s > %s 2> %s",
                             arc_q,
                             task_q,
                             max_tasks,
                             strcmp(grid, "tight") == 0 ? "--program-budget-tight-grid" : "--program-budget-grid",
                             min_budget,
                             review_value,
                             min_pressure_updates,
                             gate_args,
                             order_args,
                             out_q,
                             err_q);
                    fprintf(stderr, "arc-holo-bucket-robustness: %s\n", base);
                    if (system(cmd) != 0) {
                        fprintf(stderr, "run failed: %s\n", base);
                        failed = 1;
                        continue;
                    }
                    for (int bi = 0; bi < budget_count; ++bi) {
                        int base_cert = 0;
                        int bucket_cert = 0;
                        double base_util = 0.0;
                        double bucket_util = 0.0;
                        if (!read_summary(out_path,
                                          budgets[bi],
                                          &base_cert,
                                          &base_util,
                                          &bucket_cert,
                                          &bucket_util)) {
                            fprintf(stderr, "missing summary rows for %s budget %d\n",
                                    out_path,
                                    budgets[bi]);
                            failed = 1;
                            continue;
                        }
                        if (run_count >= MAX_RUNS) {
                            die("too many run rows");
                        }
                        RunRow *run = &runs[run_count++];
                        memset(run, 0, sizeof(*run));
                        snprintf(run->grid, sizeof(run->grid), "%s", grid);
                        snprintf(run->order, sizeof(run->order), "%s", order);
                        snprintf(run->review_slots, sizeof(run->review_slots), "%s", review);
                        run->min_budget = min_budget;
                        run->budget = budgets[bi];
                        run->baseline_certified = base_cert;
                        run->bucket_certified = bucket_cert;
                        run->baseline_utility = base_util;
                        run->bucket_utility = bucket_util;
                        run->certified_delta = (double)bucket_cert - (double)base_cert;
                        run->utility_delta = bucket_util - base_util;
                        read_diagnostics(out_path,
                                         budgets[bi],
                                         &run->crossed_into_budget,
                                         &run->crossed_out_of_budget);
                    }
                }
            }
        }
    }

    char run_path[PATH_CAP];
    char aggregate_path[PATH_CAP];
    char verdict_path[PATH_CAP];
    snprintf(run_path, sizeof(run_path), "%s/runs.jsonl", out_dir);
    snprintf(aggregate_path, sizeof(aggregate_path), "%s/aggregate.jsonl", out_dir);
    snprintf(verdict_path, sizeof(verdict_path), "%s/verdict.jsonl", out_dir);
    FILE *run_fp = fopen(run_path, "w");
    FILE *agg_fp = fopen(aggregate_path, "w");
    FILE *verdict_fp = fopen(verdict_path, "w");
    if (run_fp == NULL || agg_fp == NULL || verdict_fp == NULL) {
        die("cannot open output reports");
    }

    AggRow aggs[MAX_AGG];
    size_t agg_count = 0;
    for (size_t i = 0; i < run_count; ++i) {
        RunRow *run = &runs[i];
        fprintf(run_fp,
                "{\"schema\":\"crlp.arc_holo_bucket_robustness.run.v1\","
                "\"grid\":");
        print_json_string(run_fp, run->grid);
        fprintf(run_fp, ",\"order\":");
        print_json_string(run_fp, run->order);
        fprintf(run_fp, ",\"review_slots\":");
        print_json_string(run_fp, run->review_slots);
        fprintf(run_fp,
                ",\"min_budget\":%d,\"budget\":%d,"
                "\"baseline_certified\":%d,\"holo_bucket_certified\":%d,"
                "\"certified_delta\":%.8f,"
                "\"baseline_utility\":%.12g,\"holo_bucket_utility\":%.12g,"
                "\"utility_delta\":%.12g,"
                "\"crossed_into_budget\":%d,\"crossed_out_of_budget\":%d,"
                "\"passes_no_regression\":%s}\n",
                run->min_budget,
                run->budget,
                run->baseline_certified,
                run->bucket_certified,
                run->certified_delta,
                run->baseline_utility,
                run->bucket_utility,
                run->utility_delta,
                run->crossed_into_budget,
                run->crossed_out_of_budget,
                (run->utility_delta >= -1e-12 && run->crossed_out_of_budget == 0)
                    ? "true"
                    : "false");
        AggRow *agg = find_or_add_agg(aggs, &agg_count, run);
        if (agg->run_count < MAX_ITEMS) {
            agg->utility_values[agg->run_count] = run->utility_delta;
            agg->certified_values[agg->run_count] = run->certified_delta;
        }
        ++agg->run_count;
        agg->crossed_into_budget += run->crossed_into_budget;
        agg->crossed_out_of_budget += run->crossed_out_of_budget;
        if (run->utility_delta > 1e-12) {
            ++agg->positive_runs;
        } else if (run->utility_delta < -1e-12) {
            ++agg->negative_runs;
        } else {
            ++agg->zero_runs;
        }
        if (run->utility_delta < agg->min_utility_delta) {
            agg->min_utility_delta = run->utility_delta;
        }
        if (run->utility_delta > agg->max_utility_delta) {
            agg->max_utility_delta = run->utility_delta;
        }
        if (run->utility_delta < -1e-12 || run->crossed_out_of_budget != 0) {
            agg->passes_no_regression = false;
        }
    }

    bool all_pass = !failed;
    for (size_t i = 0; i < agg_count; ++i) {
        AggRow *agg = &aggs[i];
        int capped = agg->run_count > MAX_ITEMS ? MAX_ITEMS : agg->run_count;
        agg->median_utility_delta = median(agg->utility_values, capped);
        agg->median_certified_delta = median(agg->certified_values, capped);
        fprintf(agg_fp,
                "{\"schema\":\"crlp.arc_holo_bucket_robustness.aggregate.v1\","
                "\"grid\":");
        print_json_string(agg_fp, agg->grid);
        fprintf(agg_fp, ",\"review_slots\":");
        print_json_string(agg_fp, agg->review_slots);
        fprintf(agg_fp,
                ",\"min_budget\":%d,\"budget\":%d,\"run_count\":%d,"
                "\"positive_runs\":%d,\"negative_runs\":%d,\"zero_runs\":%d,"
                "\"crossed_into_budget\":%d,\"crossed_out_of_budget\":%d,"
                "\"min_utility_delta\":%.12g,\"max_utility_delta\":%.12g,"
                "\"median_utility_delta\":%.12g,"
                "\"median_certified_delta\":%.8f,"
                "\"passes_no_regression\":%s}\n",
                agg->min_budget,
                agg->budget,
                agg->run_count,
                agg->positive_runs,
                agg->negative_runs,
                agg->zero_runs,
                agg->crossed_into_budget,
                agg->crossed_out_of_budget,
                agg->min_utility_delta,
                agg->max_utility_delta,
                agg->median_utility_delta,
                agg->median_certified_delta,
                agg->passes_no_regression ? "true" : "false");
        if (!agg->passes_no_regression) {
            all_pass = false;
        }
    }
    fprintf(verdict_fp,
            "{\"schema\":\"crlp.arc_holo_bucket_robustness.verdict.v1\","
            "\"run_count\":%zu,\"aggregate_count\":%zu,"
            "\"passed\":%s,\"failed_runs\":%d,"
            "\"gate\":\"utility_delta_nonnegative_and_zero_crossed_out\"}\n",
            run_count,
            agg_count,
            all_pass ? "true" : "false",
            failed);

    fclose(run_fp);
    fclose(agg_fp);
    fclose(verdict_fp);
    return all_pass ? 0 : 1;
}
