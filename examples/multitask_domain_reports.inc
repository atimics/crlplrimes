typedef struct {
    const char *eval_suite;
    const char *domain;
    const char *model_kind;
    int present;
    long seeds;
    long rows;
    double top1_mean;
    double top1_stdev;
    double nll_mean;
    double nll_stdev;
    double ece_mean;
    double ece_stdev;
    double decision_utility_mean;
    double train_updates_mean;
    double parameters_mean;
    double forward_flops_mean;
} MTReportSummary;

typedef struct {
    const char *eval_suite;
    const char *domain;
    int present;
    long seeds;
    double headed_top1_mean;
    double shared_top1_mean;
    double single_top1_mean;
    double top1_delta_headed_minus_shared;
    double top1_delta_headed_minus_single;
    double headed_nll_mean;
    double shared_nll_mean;
    double single_nll_mean;
    double nll_delta_headed_minus_shared;
    double nll_delta_headed_minus_single;
    double headed_ece_mean;
    double shared_ece_mean;
    double single_ece_mean;
    double ece_delta_headed_minus_shared;
    double ece_delta_headed_minus_single;
    double headed_parameters_mean;
    double shared_parameters_mean;
    double single_parameters_mean;
    double headed_forward_flops_mean;
    double shared_forward_flops_mean;
    double single_forward_flops_mean;
    double headed_train_updates_mean;
    double shared_train_updates_mean;
    double single_train_updates_mean;
} MTReportDelta;

static double mt_nan(void)
{
    return NAN;
}

static int mt_mkdir_p(const char *path)
{
    char buffer[4096];
    size_t len;
    if (path == NULL || path[0] == '\0') {
        return 0;
    }
    len = strlen(path);
    if (len >= sizeof(buffer)) {
        return -1;
    }
    memcpy(buffer, path, len + 1U);
    for (size_t i = 1U; i < len; ++i) {
        if (buffer[i] == '/') {
            buffer[i] = '\0';
            if (buffer[0] != '\0' && mkdir(buffer, 0755) != 0 && errno != EEXIST) {
                return -1;
            }
            buffer[i] = '/';
        }
    }
    if (mkdir(buffer, 0755) != 0 && errno != EEXIST) {
        return -1;
    }
    return 0;
}

static int mt_join_path(char *out, size_t out_size, const char *dir, const char *name)
{
    int written;
    if (out == NULL || out_size == 0U || dir == NULL || name == NULL) {
        return -1;
    }
    written = snprintf(out,
                       out_size,
                       "%s%s%s",
                       dir,
                       dir[0] != '\0' && dir[strlen(dir) - 1U] == '/' ? "" : "/",
                       name);
    return written < 0 || (size_t)written >= out_size ? -1 : 0;
}

static int mt_file_has_content(const char *path)
{
    struct stat st;
    return path != NULL && stat(path, &st) == 0 && st.st_size > 0;
}

static int mt_copy_file_to(FILE *out, const char *path)
{
    char buffer[16384];
    FILE *fp;
    if (out == NULL || path == NULL) {
        return -1;
    }
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    for (;;) {
        size_t n = fread(buffer, 1U, sizeof(buffer), fp);
        if (n > 0U && fwrite(buffer, 1U, n, out) != n) {
            fclose(fp);
            return -1;
        }
        if (n < sizeof(buffer)) {
            if (ferror(fp)) {
                fclose(fp);
                return -1;
            }
            break;
        }
    }
    fclose(fp);
    return 0;
}

static void mt_today_iso(char out[11])
{
    time_t now = time(NULL);
    struct tm tm_value;
#if defined(_POSIX_VERSION)
    if (localtime_r(&now, &tm_value) == NULL) {
        strcpy(out, "1970-01-01");
        return;
    }
#else
    struct tm *tmp = localtime(&now);
    if (tmp == NULL) {
        strcpy(out, "1970-01-01");
        return;
    }
    tm_value = *tmp;
#endif
    if (strftime(out, 11U, "%Y-%m-%d", &tm_value) == 0U) {
        strcpy(out, "1970-01-01");
    }
}

static int mt_parse_seed_list(const char *text, uint64_t *seeds, size_t capacity, size_t *count)
{
    const char *cursor = text == NULL || text[0] == '\0' ? "20240516,20240517,20240518" : text;
    size_t used = 0U;
    while (*cursor != '\0') {
        char *end = NULL;
        unsigned long long parsed;
        while (*cursor == ' ' || *cursor == '\t' || *cursor == ',') {
            ++cursor;
        }
        if (*cursor == '\0') {
            break;
        }
        errno = 0;
        parsed = strtoull(cursor, &end, 10);
        if (end == cursor || errno != 0 || used == capacity) {
            return 0;
        }
        seeds[used++] = (uint64_t)parsed;
        cursor = end;
        while (*cursor == ' ' || *cursor == '\t') {
            ++cursor;
        }
        if (*cursor == ':') {
            char *range_end = NULL;
            unsigned long long last;
            ++cursor;
            errno = 0;
            last = strtoull(cursor, &range_end, 10);
            if (range_end == cursor || errno != 0 || last < parsed) {
                return 0;
            }
            cursor = range_end;
            while (parsed < last) {
                if (used == capacity) {
                    return 0;
                }
                ++parsed;
                seeds[used++] = (uint64_t)parsed;
            }
        }
        if (*cursor == ',') {
            ++cursor;
        } else if (*cursor != '\0') {
            return 0;
        }
    }
    *count = used;
    return used > 0U;
}

static int mt_seed_seen(uint64_t *seeds, size_t count, uint64_t seed)
{
    for (size_t i = 0U; i < count; ++i) {
        if (seeds[i] == seed) {
            return 1;
        }
    }
    return 0;
}

static double mt_report_stdev(double sum, double sum_sq, long rows)
{
    if (rows < 2L) {
        return 0.0;
    }
    double mean = sum / (double)rows;
    double variance = (sum_sq - (double)rows * mean * mean) / (double)(rows - 1L);
    return variance <= 0.0 ? 0.0 : sqrt(variance);
}

static int mt_summary_collect(const MTReportRows *rows,
                              const char *eval_suite,
                              const char *domain,
                              const char *model_kind,
                              MTReportSummary *summary)
{
    uint64_t seeds[128];
    size_t seed_count = 0U;
    double top1_sum = 0.0, top1_sq = 0.0;
    double nll_sum = 0.0, nll_sq = 0.0;
    double ece_sum = 0.0, ece_sq = 0.0;
    double utility_sum = 0.0;
    double updates_sum = 0.0;
    double params_sum = 0.0;
    double flops_sum = 0.0;
    long count = 0L;
    memset(summary, 0, sizeof(*summary));
    summary->eval_suite = eval_suite;
    summary->domain = domain;
    summary->model_kind = model_kind;
    for (size_t i = 0U; rows != NULL && i < rows->count; ++i) {
        const MTReportRow *row = &rows->items[i];
        if (strcmp(row->eval_suite, eval_suite) != 0 ||
            strcmp(row->domain, domain) != 0 ||
            strcmp(row->model_kind, model_kind) != 0) {
            continue;
        }
        ++count;
        top1_sum += row->top1;
        top1_sq += row->top1 * row->top1;
        nll_sum += row->nll;
        nll_sq += row->nll * row->nll;
        ece_sum += row->ece;
        ece_sq += row->ece * row->ece;
        utility_sum += row->decision_utility;
        updates_sum += row->train_updates;
        params_sum += row->parameters;
        flops_sum += row->forward_flops;
        if (!mt_seed_seen(seeds, seed_count, row->seed) && seed_count < sizeof(seeds) / sizeof(seeds[0])) {
            seeds[seed_count++] = row->seed;
        }
    }
    if (count == 0L) {
        return 0;
    }
    summary->present = 1;
    summary->rows = count;
    summary->seeds = (long)seed_count;
    summary->top1_mean = top1_sum / (double)count;
    summary->top1_stdev = mt_report_stdev(top1_sum, top1_sq, count);
    summary->nll_mean = nll_sum / (double)count;
    summary->nll_stdev = mt_report_stdev(nll_sum, nll_sq, count);
    summary->ece_mean = ece_sum / (double)count;
    summary->ece_stdev = mt_report_stdev(ece_sum, ece_sq, count);
    summary->decision_utility_mean = utility_sum / (double)count;
    summary->train_updates_mean = updates_sum / (double)count;
    summary->parameters_mean = params_sum / (double)count;
    summary->forward_flops_mean = flops_sum / (double)count;
    return 1;
}

static const MTReportSummary *mt_find_summary(const MTReportSummary *summaries,
                                              size_t count,
                                              const char *eval_suite,
                                              const char *domain,
                                              const char *model_kind)
{
    for (size_t i = 0U; i < count; ++i) {
        if (summaries[i].present &&
            strcmp(summaries[i].eval_suite, eval_suite) == 0 &&
            strcmp(summaries[i].domain, domain) == 0 &&
            strcmp(summaries[i].model_kind, model_kind) == 0) {
            return &summaries[i];
        }
    }
    return NULL;
}

static double mt_summary_value(const MTReportSummary *summary, double value)
{
    return summary == NULL ? mt_nan() : value;
}

static MTReportDelta mt_delta_build(const MTReportSummary *summaries,
                                    size_t summary_count,
                                    const char *eval_suite,
                                    const char *domain)
{
    const MTReportSummary *headed =
        mt_find_summary(summaries, summary_count, eval_suite, domain, "shared_task_heads");
    const MTReportSummary *shared =
        mt_find_summary(summaries, summary_count, eval_suite, domain, "shared");
    const MTReportSummary *single =
        mt_find_summary(summaries, summary_count, eval_suite, domain, "single_task");
    MTReportDelta delta;
    memset(&delta, 0, sizeof(delta));
    delta.eval_suite = eval_suite;
    delta.domain = domain;
    if (headed == NULL) {
        return delta;
    }
    delta.present = 1;
    delta.seeds = headed->seeds;
    delta.headed_top1_mean = headed->top1_mean;
    delta.shared_top1_mean = mt_summary_value(shared, shared == NULL ? 0.0 : shared->top1_mean);
    delta.single_top1_mean = mt_summary_value(single, single == NULL ? 0.0 : single->top1_mean);
    delta.top1_delta_headed_minus_shared =
        shared == NULL ? mt_nan() : headed->top1_mean - shared->top1_mean;
    delta.top1_delta_headed_minus_single =
        single == NULL ? mt_nan() : headed->top1_mean - single->top1_mean;
    delta.headed_nll_mean = headed->nll_mean;
    delta.shared_nll_mean = mt_summary_value(shared, shared == NULL ? 0.0 : shared->nll_mean);
    delta.single_nll_mean = mt_summary_value(single, single == NULL ? 0.0 : single->nll_mean);
    delta.nll_delta_headed_minus_shared =
        shared == NULL ? mt_nan() : headed->nll_mean - shared->nll_mean;
    delta.nll_delta_headed_minus_single =
        single == NULL ? mt_nan() : headed->nll_mean - single->nll_mean;
    delta.headed_ece_mean = headed->ece_mean;
    delta.shared_ece_mean = mt_summary_value(shared, shared == NULL ? 0.0 : shared->ece_mean);
    delta.single_ece_mean = mt_summary_value(single, single == NULL ? 0.0 : single->ece_mean);
    delta.ece_delta_headed_minus_shared =
        shared == NULL ? mt_nan() : headed->ece_mean - shared->ece_mean;
    delta.ece_delta_headed_minus_single =
        single == NULL ? mt_nan() : headed->ece_mean - single->ece_mean;
    delta.headed_parameters_mean = headed->parameters_mean;
    delta.shared_parameters_mean = mt_summary_value(shared, shared == NULL ? 0.0 : shared->parameters_mean);
    delta.single_parameters_mean = mt_summary_value(single, single == NULL ? 0.0 : single->parameters_mean);
    delta.headed_forward_flops_mean = headed->forward_flops_mean;
    delta.shared_forward_flops_mean =
        mt_summary_value(shared, shared == NULL ? 0.0 : shared->forward_flops_mean);
    delta.single_forward_flops_mean =
        mt_summary_value(single, single == NULL ? 0.0 : single->forward_flops_mean);
    delta.headed_train_updates_mean = headed->train_updates_mean;
    delta.shared_train_updates_mean =
        mt_summary_value(shared, shared == NULL ? 0.0 : shared->train_updates_mean);
    delta.single_train_updates_mean =
        mt_summary_value(single, single == NULL ? 0.0 : single->train_updates_mean);
    return delta;
}

static const MTReportDelta *mt_find_delta(const MTReportDelta *deltas,
                                          size_t count,
                                          const char *eval_suite,
                                          const char *domain)
{
    for (size_t i = 0U; i < count; ++i) {
        if (deltas[i].present &&
            strcmp(deltas[i].eval_suite, eval_suite) == 0 &&
            strcmp(deltas[i].domain, domain) == 0) {
            return &deltas[i];
        }
    }
    return NULL;
}

static void mt_csv_double(FILE *fp, double value)
{
    if (!isnan(value)) {
        fprintf(fp, "%.17g", value);
    }
}

static void mt_json_number_or_null(FILE *fp, double value)
{
    if (isnan(value)) {
        fputs("null", fp);
    } else {
        fprintf(fp, "%.17g", value);
    }
}

static void mt_format_pct(char *out, size_t out_size, double value)
{
    if (isnan(value)) {
        snprintf(out, out_size, "-");
    } else {
        snprintf(out, out_size, "%.2f%%", 100.0 * value);
    }
}

static void mt_format_delta(char *out, size_t out_size, double value)
{
    if (isnan(value)) {
        snprintf(out, out_size, "-");
    } else {
        snprintf(out, out_size, "%+.2fpp", 100.0 * value);
    }
}

static void mt_format_float(char *out, size_t out_size, double value)
{
    if (isnan(value)) {
        snprintf(out, out_size, "-");
    } else {
        snprintf(out, out_size, "%.4f", value);
    }
}

static int mt_write_summary_csv(const char *path,
                                const MTReportSummary *summaries,
                                size_t count)
{
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    fputs("eval_suite,domain,model_kind,seeds,rows,top1_mean,top1_stdev,nll_mean,nll_stdev,ece_mean,ece_stdev,decision_utility_mean,train_updates_mean,parameters_mean,forward_flops_mean\n",
          fp);
    for (size_t i = 0U; i < count; ++i) {
        const MTReportSummary *row = &summaries[i];
        if (!row->present) {
            continue;
        }
        fprintf(fp, "%s,%s,%s,%ld,%ld,", row->eval_suite, row->domain, row->model_kind, row->seeds, row->rows);
        mt_csv_double(fp, row->top1_mean); fputc(',', fp);
        mt_csv_double(fp, row->top1_stdev); fputc(',', fp);
        mt_csv_double(fp, row->nll_mean); fputc(',', fp);
        mt_csv_double(fp, row->nll_stdev); fputc(',', fp);
        mt_csv_double(fp, row->ece_mean); fputc(',', fp);
        mt_csv_double(fp, row->ece_stdev); fputc(',', fp);
        mt_csv_double(fp, row->decision_utility_mean); fputc(',', fp);
        mt_csv_double(fp, row->train_updates_mean); fputc(',', fp);
        mt_csv_double(fp, row->parameters_mean); fputc(',', fp);
        mt_csv_double(fp, row->forward_flops_mean); fputc('\n', fp);
    }
    fclose(fp);
    return 0;
}

static int mt_write_deltas_csv(const char *path,
                               const MTReportDelta *deltas,
                               size_t count)
{
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    fputs("eval_suite,domain,seeds,headed_top1_mean,shared_top1_mean,single_top1_mean,top1_delta_headed_minus_shared,top1_delta_headed_minus_single,headed_nll_mean,shared_nll_mean,single_nll_mean,nll_delta_headed_minus_shared,nll_delta_headed_minus_single,headed_ece_mean,shared_ece_mean,single_ece_mean,ece_delta_headed_minus_shared,ece_delta_headed_minus_single,headed_parameters_mean,shared_parameters_mean,single_parameters_mean,headed_forward_flops_mean,shared_forward_flops_mean,single_forward_flops_mean,headed_train_updates_mean,shared_train_updates_mean,single_train_updates_mean\n",
          fp);
    for (size_t i = 0U; i < count; ++i) {
        const MTReportDelta *row = &deltas[i];
        if (!row->present) {
            continue;
        }
        fprintf(fp, "%s,%s,%ld,", row->eval_suite, row->domain, row->seeds);
        mt_csv_double(fp, row->headed_top1_mean); fputc(',', fp);
        mt_csv_double(fp, row->shared_top1_mean); fputc(',', fp);
        mt_csv_double(fp, row->single_top1_mean); fputc(',', fp);
        mt_csv_double(fp, row->top1_delta_headed_minus_shared); fputc(',', fp);
        mt_csv_double(fp, row->top1_delta_headed_minus_single); fputc(',', fp);
        mt_csv_double(fp, row->headed_nll_mean); fputc(',', fp);
        mt_csv_double(fp, row->shared_nll_mean); fputc(',', fp);
        mt_csv_double(fp, row->single_nll_mean); fputc(',', fp);
        mt_csv_double(fp, row->nll_delta_headed_minus_shared); fputc(',', fp);
        mt_csv_double(fp, row->nll_delta_headed_minus_single); fputc(',', fp);
        mt_csv_double(fp, row->headed_ece_mean); fputc(',', fp);
        mt_csv_double(fp, row->shared_ece_mean); fputc(',', fp);
        mt_csv_double(fp, row->single_ece_mean); fputc(',', fp);
        mt_csv_double(fp, row->ece_delta_headed_minus_shared); fputc(',', fp);
        mt_csv_double(fp, row->ece_delta_headed_minus_single); fputc(',', fp);
        mt_csv_double(fp, row->headed_parameters_mean); fputc(',', fp);
        mt_csv_double(fp, row->shared_parameters_mean); fputc(',', fp);
        mt_csv_double(fp, row->single_parameters_mean); fputc(',', fp);
        mt_csv_double(fp, row->headed_forward_flops_mean); fputc(',', fp);
        mt_csv_double(fp, row->shared_forward_flops_mean); fputc(',', fp);
        mt_csv_double(fp, row->single_forward_flops_mean); fputc(',', fp);
        mt_csv_double(fp, row->headed_train_updates_mean); fputc(',', fp);
        mt_csv_double(fp, row->shared_train_updates_mean); fputc(',', fp);
        mt_csv_double(fp, row->single_train_updates_mean); fputc('\n', fp);
    }
    fclose(fp);
    return 0;
}

static int mt_write_decision_jsonl(const char *path,
                                   const MTReportDelta *deltas,
                                   size_t delta_count,
                                   double margin)
{
    const MTReportDelta *fixed_arc = mt_find_delta(deltas, delta_count, "fixed", "arc_agi");
    const MTReportDelta *ood_arc = mt_find_delta(deltas, delta_count, "ood", "arc_agi");
    const MTReportDelta *delayed = mt_find_delta(deltas, delta_count, "fixed", "delayed_claim");
    const MTReportDelta *grid = mt_find_delta(deltas, delta_count, "fixed", "grid_path");
    int complete = fixed_arc != NULL && ood_arc != NULL && delayed != NULL && grid != NULL;
    long seeds = 0L;
    double fixed_arc_gain = fixed_arc == NULL ? mt_nan() : fixed_arc->top1_delta_headed_minus_shared;
    double delayed_delta = delayed == NULL ? mt_nan() : delayed->top1_delta_headed_minus_shared;
    double grid_delta = grid == NULL ? mt_nan() : grid->top1_delta_headed_minus_shared;
    double ood_top1 = ood_arc == NULL ? mt_nan() : ood_arc->top1_delta_headed_minus_single;
    double ood_nll = ood_arc == NULL ? mt_nan() : ood_arc->nll_delta_headed_minus_single;
    int fixed_gate = !isnan(fixed_arc_gain) && fixed_arc_gain > 0.0;
    int delayed_gate = !isnan(delayed_delta) && delayed_delta >= -margin;
    int grid_gate = !isnan(grid_delta) && grid_delta >= -margin;
    int ood_gate = !isnan(ood_top1) && !isnan(ood_nll) && ood_top1 >= 0.0 && ood_nll <= 0.0;
    int promote;
    char date[11];
    FILE *fp;
    complete = complete && !isnan(fixed_arc_gain) && !isnan(delayed_delta) &&
               !isnan(grid_delta) && !isnan(ood_top1) && !isnan(ood_nll);
    if (complete) {
        seeds = fixed_arc->seeds;
        if (ood_arc->seeds < seeds) seeds = ood_arc->seeds;
        if (delayed->seeds < seeds) seeds = delayed->seeds;
        if (grid->seeds < seeds) seeds = grid->seeds;
    }
    promote = complete && seeds >= 3L && fixed_gate && delayed_gate && grid_gate && ood_gate;
    mt_today_iso(date);
    fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    fputs("{\"complete_required_rows\": ", fp);
    fputs(complete ? "true" : "false", fp);
    fprintf(fp, ", \"date\": \"%s\", \"decision\": \"%s\", ", date, promote ? "promote_shared_trunk_heads" : "diagnostic_only");
    fputs("\"delayed_claim_top1_delta_heads_minus_flat_shared\": ", fp); mt_json_number_or_null(fp, delayed_delta);
    fputs(", \"delayed_claim_transfer_preserved\": ", fp); fputs(delayed_gate ? "true" : "false", fp);
    fputs(", \"fixed_arc_improves_vs_flat_shared\": ", fp); fputs(fixed_gate ? "true" : "false", fp);
    fputs(", \"fixed_arc_top1_delta_heads_minus_flat_shared\": ", fp); mt_json_number_or_null(fp, fixed_arc_gain);
    fputs(", \"grid_path_not_materially_regressed\": ", fp); fputs(grid_gate ? "true" : "false", fp);
    fputs(", \"grid_path_top1_delta_heads_minus_flat_shared\": ", fp); mt_json_number_or_null(fp, grid_delta);
    fputs(", \"ood_arc_nll_delta_heads_minus_single_task\": ", fp); mt_json_number_or_null(fp, ood_nll);
    fputs(", \"ood_arc_preserved_vs_single_task\": ", fp); fputs(ood_gate ? "true" : "false", fp);
    fputs(", \"ood_arc_top1_delta_heads_minus_single_task\": ", fp); mt_json_number_or_null(fp, ood_top1);
    fputs(", \"promote_shared_trunk_heads\": ", fp); fputs(promote ? "true" : "false", fp);
    fprintf(fp,
            ", \"report_version\": 1, \"schema\": \"crlp.multitask.adapter_heads_decision.v1\", \"seeds\": %ld, \"transfer_regression_margin\": %.17g}\n",
            seeds,
            margin);
    fclose(fp);
    return 0;
}

static void mt_markdown_top1_row(FILE *fp,
                                 const char *label,
                                 const MTReportSummary *shared,
                                 const MTReportSummary *headed,
                                 const MTReportSummary *single,
                                 const MTReportDelta *delta)
{
    char shared_pct[32], headed_pct[32], single_pct[32], delta_shared[32], delta_single[32];
    mt_format_pct(shared_pct, sizeof(shared_pct), shared == NULL ? mt_nan() : shared->top1_mean);
    mt_format_pct(headed_pct, sizeof(headed_pct), headed == NULL ? mt_nan() : headed->top1_mean);
    mt_format_pct(single_pct, sizeof(single_pct), single == NULL ? mt_nan() : single->top1_mean);
    mt_format_delta(delta_shared,
                    sizeof(delta_shared),
                    delta == NULL ? mt_nan() : delta->top1_delta_headed_minus_shared);
    mt_format_delta(delta_single,
                    sizeof(delta_single),
                    delta == NULL ? mt_nan() : delta->top1_delta_headed_minus_single);
    fprintf(fp,
            "| `%s` | %s | %s | %s | %s | %s |\n",
            label,
            shared_pct,
            headed_pct,
            single_pct,
            delta_shared,
            delta_single);
}

static int mt_write_adapter_report_md(const char *path,
                                      const uint64_t *seeds,
                                      size_t seed_count,
                                      const MTConfig *base,
                                      const MTReportSummary *summaries,
                                      size_t summary_count,
                                      const MTReportDelta *deltas,
                                      size_t delta_count)
{
    FILE *fp = fopen(path, "w");
    char seed_text[512];
    size_t offset = 0U;
    const char *fixed_domains[] = {"prime_gap", "grid_path", "delayed_claim", "arc_agi"};
    const char *models[] = {"shared", "shared_task_heads", "single_task"};
    const MTReportDelta *decision_delta = mt_find_delta(deltas, delta_count, "fixed", "arc_agi");
    (void)decision_delta;
    if (fp == NULL) {
        return -1;
    }
    seed_text[0] = '\0';
    for (size_t i = 0U; i < seed_count; ++i) {
        int written = snprintf(seed_text + offset,
                               sizeof(seed_text) - offset,
                               "%s%llu",
                               i == 0U ? "" : ", ",
                               (unsigned long long)seeds[i]);
        if (written < 0 || (size_t)written >= sizeof(seed_text) - offset) {
            break;
        }
        offset += (size_t)written;
    }
    fprintf(fp,
            "# Multitask Adapter-Headed Sharing\n\n"
            "Seeds: `%s`\n"
            "Budget: `%zu` epochs x `%zu` steps = `%zu` total shared updates.\n"
            "Evaluation samples: `%zu` per domain row.\n\n"
            "This run compares the flat shared scalar scorer, a shared trunk with per-domain scalar heads, and matched single-task baselines under the same 96-feature envelope.\n\n"
            "## Fixed-Suite Top-1\n\n"
            "| Domain | Shared | Task heads | Single | Heads - shared | Heads - single |\n"
            "|---|---:|---:|---:|---:|---:|\n",
            seed_text,
            base->epochs,
            base->steps_per_epoch,
            base->epochs * base->steps_per_epoch,
            base->eval_samples);
    for (size_t i = 0U; i < sizeof(fixed_domains) / sizeof(fixed_domains[0]); ++i) {
        const char *domain = fixed_domains[i];
        mt_markdown_top1_row(fp,
                             domain,
                             mt_find_summary(summaries, summary_count, "fixed", domain, "shared"),
                             mt_find_summary(summaries, summary_count, "fixed", domain, "shared_task_heads"),
                             mt_find_summary(summaries, summary_count, "fixed", domain, "single_task"),
                             mt_find_delta(deltas, delta_count, "fixed", domain));
    }
    fputs("\n## ARC OOD Top-1\n\n"
          "| Suite | Shared | Task heads | Single | Heads - shared | Heads - single |\n"
          "|---|---:|---:|---:|---:|---:|\n",
          fp);
    mt_markdown_top1_row(fp,
                         "arc_ood",
                         mt_find_summary(summaries, summary_count, "ood", "arc_agi", "shared"),
                         mt_find_summary(summaries, summary_count, "ood", "arc_agi", "shared_task_heads"),
                         mt_find_summary(summaries, summary_count, "ood", "arc_agi", "single_task"),
                         mt_find_delta(deltas, delta_count, "ood", "arc_agi"));
    fputs("\n## ARC And Transfer Guardrail\n\n"
          "| Suite | Domain | Model | Top-1 | NLL | ECE | Updates | Params | FLOPs |\n"
          "|---|---|---|---:|---:|---:|---:|---:|---:|\n",
          fp);
    for (size_t pair = 0U; pair < 4U; ++pair) {
        const char *suite = pair == 1U ? "ood" : "fixed";
        const char *domain = pair == 2U ? "delayed_claim" : pair == 3U ? "grid_path" : "arc_agi";
        for (size_t model_index = 0U; model_index < sizeof(models) / sizeof(models[0]); ++model_index) {
            const MTReportSummary *row =
                mt_find_summary(summaries, summary_count, suite, domain, models[model_index]);
            char top1[32], nll[32], ece[32];
            if (row == NULL) {
                continue;
            }
            mt_format_pct(top1, sizeof(top1), row->top1_mean);
            mt_format_float(nll, sizeof(nll), row->nll_mean);
            mt_format_float(ece, sizeof(ece), row->ece_mean);
            fprintf(fp,
                    "| `%s` | `%s` | `%s` | %s | %s | %s | %.0f | %.0f | %.0f |\n",
                    suite,
                    domain,
                    row->model_kind,
                    top1,
                    nll,
                    ece,
                    row->train_updates_mean,
                    row->parameters_mean,
                    row->forward_flops_mean);
        }
    }
    fputs("\n## Decision Rule\n\n"
          "- Decision: see `rmt4_adapter_heads_decision.jsonl`.\n"
          "- Promote this architecture only if task heads improve fixed ARC versus the flat shared scorer, preserve delayed-claim transfer, avoid a material grid regression, and preserve ARC OOD versus the single-task baseline.\n"
          "- If task heads help fixed ARC but hurt OOD or transfer, keep them as diagnostic evidence rather than the default architecture.\n",
          fp);
    fclose(fp);
    return 0;
}

typedef struct {
    size_t hidden1;
    size_t hidden2;
    char label[32];
} MTTinyCapacity;

typedef struct {
    char hidden[32];
    const char *domain;
    const char *model_kind;
    size_t epochs;
    int present;
    long seeds;
    long rows;
    double top1_mean;
    double top1_stdev;
    double top1_min;
    double top1_max;
    double nll_mean;
    double nll_stdev;
    double ece_mean;
    double ece_stdev;
    double parameters_mean;
    double forward_flops_mean;
} MTTinyCurve;

typedef struct {
    char hidden[32];
    const char *domain;
    const char *model_kind;
    int present;
    size_t best_epochs;
    double best_top1_mean;
    double best_top1_stdev;
    size_t last_epochs;
    double last_top1_mean;
    double last_gain;
    double last_window_gain;
    double best_gain_from_first;
    double plateau_threshold;
    size_t plateau_window;
    int best_epoch_in_last_window;
    int plateau_detected;
    double parameters_mean;
    double forward_flops_mean;
} MTTinyPlateau;

typedef struct {
    const char *domain;
    const char *model_kind;
    int present;
    char smallest_hidden[32];
    char largest_hidden[32];
    char best_hidden[32];
    double best_top1_mean;
    double largest_minus_smallest_top1;
    int larger_width_helped;
} MTTinyWidth;

typedef struct {
    const char *domain;
    const char *model_kind;
    int present;
    long capacity_count;
    int capacity_coverage_complete;
    long restart_count;
    long epoch_checkpoint_count;
    long plateau_count;
    int all_plateau_detected;
    int width_verdict_present;
    char best_hidden[32];
    double best_top1_mean;
    int larger_width_helped;
    int audit_ready;
} MTTinyVerdict;

static int mt_parse_capacity_list(const char *text,
                                  MTTinyCapacity *capacities,
                                  size_t capacity,
                                  size_t *count)
{
    const char *cursor = text == NULL || text[0] == '\0' ? "8x4,16x8,32x16,64x32" : text;
    size_t used = 0U;
    while (*cursor != '\0') {
        char token[64];
        size_t len = 0U;
        size_t hidden1 = 0U, hidden2 = 0U;
        while (*cursor == ' ' || *cursor == '\t' || *cursor == ',') {
            ++cursor;
        }
        if (*cursor == '\0') {
            break;
        }
        while (cursor[len] != '\0' && cursor[len] != ',') {
            ++len;
        }
        while (len > 0U && (cursor[len - 1U] == ' ' || cursor[len - 1U] == '\t')) {
            --len;
        }
        if (len == 0U || len >= sizeof(token) || used == capacity) {
            return 0;
        }
        memcpy(token, cursor, len);
        token[len] = '\0';
        if (!mt_parse_hidden_text(token, &hidden1, &hidden2)) {
            return 0;
        }
        capacities[used].hidden1 = hidden1;
        capacities[used].hidden2 = hidden2;
        snprintf(capacities[used].label, sizeof(capacities[used].label), "%zux%zu", hidden1, hidden2);
        ++used;
        cursor += len;
        while (*cursor == ' ' || *cursor == '\t') {
            ++cursor;
        }
        if (*cursor == ',') {
            ++cursor;
        } else if (*cursor != '\0') {
            return 0;
        }
    }
    *count = used;
    return used > 0U;
}

static int mt_parse_size_list(const char *text, size_t *values, size_t capacity, size_t *count)
{
    const char *cursor = text == NULL || text[0] == '\0' ? "1,2,4,8,16,32" : text;
    size_t used = 0U;
    while (*cursor != '\0') {
        char *end = NULL;
        unsigned long long parsed;
        while (*cursor == ' ' || *cursor == '\t' || *cursor == ',') {
            ++cursor;
        }
        if (*cursor == '\0') {
            break;
        }
        errno = 0;
        parsed = strtoull(cursor, &end, 10);
        if (end == cursor || errno != 0 || parsed == 0ULL || used == capacity) {
            return 0;
        }
        values[used++] = (size_t)parsed;
        cursor = end;
        while (*cursor == ' ' || *cursor == '\t') {
            ++cursor;
        }
        if (*cursor == ':') {
            char *range_end = NULL;
            unsigned long long last;
            ++cursor;
            errno = 0;
            last = strtoull(cursor, &range_end, 10);
            if (range_end == cursor || errno != 0 || last < parsed) {
                return 0;
            }
            cursor = range_end;
            while (parsed < last) {
                if (used == capacity) {
                    return 0;
                }
                ++parsed;
                values[used++] = (size_t)parsed;
            }
        }
        if (*cursor == ',') {
            ++cursor;
        } else if (*cursor != '\0') {
            return 0;
        }
    }
    *count = used;
    return used > 0U;
}

static int mt_tiny_curve_matches(const MTTinyCurve *curve,
                                 const MTReportRow *row)
{
    return strcmp(curve->hidden, row->hidden) == 0 &&
           strcmp(curve->domain, row->domain) == 0 &&
           strcmp(curve->model_kind, row->model_kind) == 0 &&
           curve->epochs == row->epochs;
}

static int mt_tiny_collect_curves(const MTReportRows *rows,
                                  MTTinyCurve **curves_out,
                                  size_t *curve_count_out)
{
    MTTinyCurve *curves;
    size_t curve_count = 0U;
    if (rows == NULL || curves_out == NULL || curve_count_out == NULL) {
        return -1;
    }
    curves = (MTTinyCurve *)calloc(rows->count == 0U ? 1U : rows->count, sizeof(*curves));
    if (curves == NULL) {
        return -1;
    }
    for (size_t row_index = 0U; row_index < rows->count; ++row_index) {
        const MTReportRow *row = &rows->items[row_index];
        int found = 0;
        for (size_t i = 0U; i < curve_count; ++i) {
            if (mt_tiny_curve_matches(&curves[i], row)) {
                found = 1;
                break;
            }
        }
        if (!found) {
            MTTinyCurve *curve = &curves[curve_count++];
            curve->present = 1;
            snprintf(curve->hidden, sizeof(curve->hidden), "%s", row->hidden);
            curve->domain = row->domain;
            curve->model_kind = row->model_kind;
            curve->epochs = row->epochs;
        }
    }
    for (size_t i = 0U; i < curve_count; ++i) {
        MTTinyCurve *curve = &curves[i];
        double top1_sum = 0.0, top1_sq = 0.0;
        double nll_sum = 0.0, nll_sq = 0.0;
        double ece_sum = 0.0, ece_sq = 0.0;
        double params_sum = 0.0, flops_sum = 0.0;
        uint64_t seeds[128];
        size_t seed_count = 0U;
        long count = 0L;
        curve->top1_min = 0.0;
        curve->top1_max = 0.0;
        for (size_t row_index = 0U; row_index < rows->count; ++row_index) {
            const MTReportRow *row = &rows->items[row_index];
            if (mt_tiny_curve_matches(curve, row)) {
                if (count == 0L || row->top1 < curve->top1_min) {
                    curve->top1_min = row->top1;
                }
                if (count == 0L || row->top1 > curve->top1_max) {
                    curve->top1_max = row->top1;
                }
                ++count;
                top1_sum += row->top1;
                top1_sq += row->top1 * row->top1;
                nll_sum += row->nll;
                nll_sq += row->nll * row->nll;
                ece_sum += row->ece;
                ece_sq += row->ece * row->ece;
                params_sum += row->parameters;
                flops_sum += row->forward_flops;
                if (!mt_seed_seen(seeds, seed_count, row->seed) &&
                    seed_count < sizeof(seeds) / sizeof(seeds[0])) {
                    seeds[seed_count++] = row->seed;
                }
            }
        }
        curve->rows = count;
        curve->seeds = (long)seed_count;
        if (count > 0L) {
            curve->top1_mean = top1_sum / (double)count;
            curve->top1_stdev = mt_report_stdev(top1_sum, top1_sq, count);
            curve->nll_mean = nll_sum / (double)count;
            curve->nll_stdev = mt_report_stdev(nll_sum, nll_sq, count);
            curve->ece_mean = ece_sum / (double)count;
            curve->ece_stdev = mt_report_stdev(ece_sum, ece_sq, count);
            curve->parameters_mean = params_sum / (double)count;
            curve->forward_flops_mean = flops_sum / (double)count;
        }
    }
    *curves_out = curves;
    *curve_count_out = curve_count;
    return 0;
}

static int mt_tiny_same_group(const MTTinyCurve *curve,
                              const char *hidden,
                              const char *domain,
                              const char *model_kind)
{
    return strcmp(curve->hidden, hidden) == 0 &&
           strcmp(curve->domain, domain) == 0 &&
           strcmp(curve->model_kind, model_kind) == 0;
}

static void mt_tiny_sort_curve_group(const MTTinyCurve **group, size_t count)
{
    for (size_t i = 0U; i < count; ++i) {
        for (size_t j = i + 1U; j < count; ++j) {
            if (group[j]->epochs < group[i]->epochs) {
                const MTTinyCurve *tmp = group[i];
                group[i] = group[j];
                group[j] = tmp;
            }
        }
    }
}

static int mt_tiny_plateau_group(const MTTinyCurve *curves,
                                 size_t curve_count,
                                 const char *hidden,
                                 const char *domain,
                                 const char *model_kind,
                                 double threshold,
                                 size_t plateau_window,
                                 MTTinyPlateau *out)
{
    const MTTinyCurve **group =
        (const MTTinyCurve **)calloc(curve_count == 0U ? 1U : curve_count, sizeof(*group));
    size_t count = 0U;
    size_t window;
    const MTTinyCurve *first;
    const MTTinyCurve *last;
    const MTTinyCurve *prev;
    const MTTinyCurve *recent_first;
    const MTTinyCurve *best;
    if (group == NULL) {
        return -1;
    }
    for (size_t i = 0U; i < curve_count; ++i) {
        if (mt_tiny_same_group(&curves[i], hidden, domain, model_kind)) {
            group[count++] = &curves[i];
        }
    }
    if (count == 0U) {
        free(group);
        return 0;
    }
    mt_tiny_sort_curve_group(group, count);
    window = plateau_window == 0U ? 1U : plateau_window;
    if (window > count) {
        window = count;
    }
    first = group[0];
    last = group[count - 1U];
    prev = count > 1U ? group[count - 2U] : last;
    recent_first = group[count - window];
    best = first;
    for (size_t i = 1U; i < count; ++i) {
        if (group[i]->top1_mean > best->top1_mean) {
            best = group[i];
        }
    }
    memset(out, 0, sizeof(*out));
    out->present = 1;
    snprintf(out->hidden, sizeof(out->hidden), "%s", hidden);
    out->domain = domain;
    out->model_kind = model_kind;
    out->best_epochs = best->epochs;
    out->best_top1_mean = best->top1_mean;
    out->best_top1_stdev = best->top1_stdev;
    out->last_epochs = last->epochs;
    out->last_top1_mean = last->top1_mean;
    out->last_gain = last->top1_mean - prev->top1_mean;
    out->last_window_gain = last->top1_mean - recent_first->top1_mean;
    out->best_gain_from_first = best->top1_mean - first->top1_mean;
    out->plateau_threshold = threshold;
    out->plateau_window = window;
    out->best_epoch_in_last_window = best->epochs >= recent_first->epochs;
    out->plateau_detected = fabs(out->last_gain) <= threshold &&
                            fabs(out->last_window_gain) <= threshold &&
                            out->best_epoch_in_last_window;
    out->parameters_mean = last->parameters_mean;
    out->forward_flops_mean = last->forward_flops_mean;
    free(group);
    return 1;
}

static int mt_tiny_collect_plateaus(const MTTinyCurve *curves,
                                    size_t curve_count,
                                    double threshold,
                                    size_t plateau_window,
                                    MTTinyPlateau **plateaus_out,
                                    size_t *plateau_count_out)
{
    MTTinyPlateau *plateaus =
        (MTTinyPlateau *)calloc(curve_count == 0U ? 1U : curve_count, sizeof(*plateaus));
    size_t plateau_count = 0U;
    if (plateaus == NULL) {
        return -1;
    }
    for (size_t i = 0U; i < curve_count; ++i) {
        int seen = 0;
        for (size_t j = 0U; j < plateau_count; ++j) {
            if (strcmp(plateaus[j].hidden, curves[i].hidden) == 0 &&
                strcmp(plateaus[j].domain, curves[i].domain) == 0 &&
                strcmp(plateaus[j].model_kind, curves[i].model_kind) == 0) {
                seen = 1;
                break;
            }
        }
        if (!seen) {
            int rc = mt_tiny_plateau_group(curves,
                                           curve_count,
                                           curves[i].hidden,
                                           curves[i].domain,
                                           curves[i].model_kind,
                                           threshold,
                                           plateau_window,
                                           &plateaus[plateau_count]);
            if (rc < 0) {
                free(plateaus);
                return -1;
            }
            if (rc > 0) {
                ++plateau_count;
            }
        }
    }
    *plateaus_out = plateaus;
    *plateau_count_out = plateau_count;
    return 0;
}

static const MTTinyPlateau *mt_tiny_find_plateau(const MTTinyPlateau *plateaus,
                                                 size_t plateau_count,
                                                 const char *hidden,
                                                 const char *domain,
                                                 const char *model_kind)
{
    for (size_t i = 0U; i < plateau_count; ++i) {
        if (plateaus[i].present &&
            strcmp(plateaus[i].hidden, hidden) == 0 &&
            strcmp(plateaus[i].domain, domain) == 0 &&
            strcmp(plateaus[i].model_kind, model_kind) == 0) {
            return &plateaus[i];
        }
    }
    return NULL;
}

static int mt_tiny_collect_widths(const MTTinyPlateau *plateaus,
                                  size_t plateau_count,
                                  const MTTinyCapacity *capacities,
                                  size_t capacity_count,
                                  const char *const *domains,
                                  size_t domain_count,
                                  const char *const *models,
                                  size_t model_count,
                                  MTTinyWidth **widths_out,
                                  size_t *width_count_out)
{
    size_t capacity = domain_count * model_count;
    MTTinyWidth *widths = (MTTinyWidth *)calloc(capacity == 0U ? 1U : capacity, sizeof(*widths));
    size_t width_count = 0U;
    if (widths == NULL) {
        return -1;
    }
    for (size_t domain_index = 0U; domain_index < domain_count; ++domain_index) {
        for (size_t model_index = 0U; model_index < model_count; ++model_index) {
            const MTTinyPlateau *first = NULL;
            const MTTinyPlateau *last = NULL;
            const MTTinyPlateau *best = NULL;
            for (size_t cap_index = 0U; cap_index < capacity_count; ++cap_index) {
                const MTTinyPlateau *row = mt_tiny_find_plateau(plateaus,
                                                                plateau_count,
                                                                capacities[cap_index].label,
                                                                domains[domain_index],
                                                                models[model_index]);
                if (row == NULL) {
                    continue;
                }
                if (first == NULL) {
                    first = row;
                }
                last = row;
                if (best == NULL || row->best_top1_mean > best->best_top1_mean) {
                    best = row;
                }
            }
            if (first != NULL && last != NULL && best != NULL) {
                MTTinyWidth *width = &widths[width_count++];
                width->present = 1;
                width->domain = domains[domain_index];
                width->model_kind = models[model_index];
                snprintf(width->smallest_hidden, sizeof(width->smallest_hidden), "%s", first->hidden);
                snprintf(width->largest_hidden, sizeof(width->largest_hidden), "%s", last->hidden);
                snprintf(width->best_hidden, sizeof(width->best_hidden), "%s", best->hidden);
                width->best_top1_mean = best->best_top1_mean;
                width->largest_minus_smallest_top1 =
                    last->best_top1_mean - first->best_top1_mean;
                width->larger_width_helped = strcmp(best->hidden, first->hidden) != 0;
            }
        }
    }
    *widths_out = widths;
    *width_count_out = width_count;
    return 0;
}

static const MTTinyWidth *mt_tiny_find_width(const MTTinyWidth *widths,
                                             size_t width_count,
                                             const char *domain,
                                             const char *model_kind)
{
    for (size_t i = 0U; i < width_count; ++i) {
        if (widths[i].present &&
            strcmp(widths[i].domain, domain) == 0 &&
            strcmp(widths[i].model_kind, model_kind) == 0) {
            return &widths[i];
        }
    }
    return NULL;
}

static const MTTinyCurve *mt_tiny_find_curve(const MTTinyCurve *curves,
                                             size_t curve_count,
                                             const char *hidden,
                                             const char *domain,
                                             const char *model_kind,
                                             size_t epochs)
{
    for (size_t i = 0U; i < curve_count; ++i) {
        if (curves[i].present &&
            curves[i].epochs == epochs &&
            strcmp(curves[i].hidden, hidden) == 0 &&
            strcmp(curves[i].domain, domain) == 0 &&
            strcmp(curves[i].model_kind, model_kind) == 0) {
            return &curves[i];
        }
    }
    return NULL;
}

static int mt_tiny_collect_verdicts(const MTTinyPlateau *plateaus,
                                    size_t plateau_count,
                                    const MTTinyWidth *widths,
                                    size_t width_count,
                                    const MTTinyCapacity *capacities,
                                    size_t capacity_count,
                                    const uint64_t *seeds,
                                    size_t seed_count,
                                    const size_t *epoch_grid,
                                    size_t epoch_count,
                                    const char *const *domains,
                                    size_t domain_count,
                                    const char *const *models,
                                    size_t model_count,
                                    MTTinyVerdict **verdicts_out,
                                    size_t *verdict_count_out)
{
    size_t capacity = domain_count * model_count;
    MTTinyVerdict *verdicts =
        (MTTinyVerdict *)calloc(capacity == 0U ? 1U : capacity, sizeof(*verdicts));
    size_t verdict_count = 0U;
    if (verdicts == NULL) {
        return -1;
    }
    for (size_t domain_index = 0U; domain_index < domain_count; ++domain_index) {
        for (size_t model_index = 0U; model_index < model_count; ++model_index) {
            const MTTinyWidth *width =
                mt_tiny_find_width(widths, width_count, domains[domain_index], models[model_index]);
            MTTinyVerdict *verdict = &verdicts[verdict_count++];
            long group_count = 0L;
            long plateau_ready_count = 0L;
            int coverage_complete = 1;
            verdict->present = 1;
            verdict->domain = domains[domain_index];
            verdict->model_kind = models[model_index];
            verdict->restart_count = (long)seed_count;
            verdict->epoch_checkpoint_count = (long)epoch_count;
            for (size_t cap_index = 0U; cap_index < capacity_count; ++cap_index) {
                const MTTinyPlateau *row = mt_tiny_find_plateau(plateaus,
                                                                plateau_count,
                                                                capacities[cap_index].label,
                                                                domains[domain_index],
                                                                models[model_index]);
                if (row == NULL) {
                    coverage_complete = 0;
                    continue;
                }
                ++group_count;
                if (row->plateau_detected) {
                    ++plateau_ready_count;
                }
            }
            verdict->capacity_count = group_count;
            verdict->capacity_coverage_complete = coverage_complete;
            verdict->plateau_count = plateau_ready_count;
            verdict->all_plateau_detected = group_count > 0L && plateau_ready_count == group_count;
            verdict->width_verdict_present = width != NULL;
            if (width != NULL) {
                snprintf(verdict->best_hidden, sizeof(verdict->best_hidden), "%s", width->best_hidden);
                verdict->best_top1_mean = width->best_top1_mean;
                verdict->larger_width_helped = width->larger_width_helped;
            }
            verdict->audit_ready = verdict->capacity_coverage_complete &&
                                   verdict->all_plateau_detected &&
                                   verdict->width_verdict_present;
        }
    }
    (void)seeds;
    (void)epoch_grid;
    *verdicts_out = verdicts;
    *verdict_count_out = verdict_count;
    return 0;
}

static int mt_write_tiny_learning_curve_csv(const char *path,
                                            const MTTinyCurve *curves,
                                            size_t count)
{
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    fputs("hidden,domain,model_kind,epochs,seeds,top1_mean,top1_stdev,top1_min,top1_max,nll_mean,nll_stdev,ece_mean,ece_stdev,parameters_mean,forward_flops_mean\n",
          fp);
    for (size_t i = 0U; i < count; ++i) {
        const MTTinyCurve *row = &curves[i];
        fprintf(fp, "%s,%s,%s,%zu,%ld,", row->hidden, row->domain, row->model_kind, row->epochs, row->seeds);
        mt_csv_double(fp, row->top1_mean); fputc(',', fp);
        mt_csv_double(fp, row->top1_stdev); fputc(',', fp);
        mt_csv_double(fp, row->top1_min); fputc(',', fp);
        mt_csv_double(fp, row->top1_max); fputc(',', fp);
        mt_csv_double(fp, row->nll_mean); fputc(',', fp);
        mt_csv_double(fp, row->nll_stdev); fputc(',', fp);
        mt_csv_double(fp, row->ece_mean); fputc(',', fp);
        mt_csv_double(fp, row->ece_stdev); fputc(',', fp);
        mt_csv_double(fp, row->parameters_mean); fputc(',', fp);
        mt_csv_double(fp, row->forward_flops_mean); fputc('\n', fp);
    }
    fclose(fp);
    return 0;
}

static int mt_write_tiny_plateau_csv(const char *path,
                                     const MTTinyPlateau *plateaus,
                                     size_t count)
{
    char date[11];
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    mt_today_iso(date);
    fputs("schema,report_version,date,hidden,domain,model_kind,best_epochs,best_top1_mean,best_top1_stdev,last_epochs,last_top1_mean,last_gain,last_window_gain,best_gain_from_first,plateau_threshold,plateau_window,plateau_method,best_epoch_in_last_window,plateau_detected,parameters_mean,forward_flops_mean\n",
          fp);
    for (size_t i = 0U; i < count; ++i) {
        const MTTinyPlateau *row = &plateaus[i];
        fprintf(fp,
                "crlp.multitask.tiny_net_plateau.v1,2,%s,%s,%s,%s,%zu,",
                date,
                row->hidden,
                row->domain,
                row->model_kind,
                row->best_epochs);
        mt_csv_double(fp, row->best_top1_mean); fputc(',', fp);
        mt_csv_double(fp, row->best_top1_stdev); fprintf(fp, ",%zu,", row->last_epochs);
        mt_csv_double(fp, row->last_top1_mean); fputc(',', fp);
        mt_csv_double(fp, row->last_gain); fputc(',', fp);
        mt_csv_double(fp, row->last_window_gain); fputc(',', fp);
        mt_csv_double(fp, row->best_gain_from_first); fputc(',', fp);
        mt_csv_double(fp, row->plateau_threshold);
        fprintf(fp,
                ",%zu,last_step_and_window_gain,%s,%s,",
                row->plateau_window,
                row->best_epoch_in_last_window ? "true" : "false",
                row->plateau_detected ? "true" : "false");
        mt_csv_double(fp, row->parameters_mean); fputc(',', fp);
        mt_csv_double(fp, row->forward_flops_mean); fputc('\n', fp);
    }
    fclose(fp);
    return 0;
}

static int mt_write_tiny_plateau_jsonl(const char *path,
                                       const MTTinyPlateau *plateaus,
                                       size_t count)
{
    char date[11];
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    mt_today_iso(date);
    for (size_t i = 0U; i < count; ++i) {
        const MTTinyPlateau *row = &plateaus[i];
        fputs("{\"schema\":\"crlp.multitask.tiny_net_plateau.v1\",\"report_version\":2,\"date\":", fp);
        mt_json_write_string(fp, date);
        fputs(",\"hidden\":", fp); mt_json_write_string(fp, row->hidden);
        fputs(",\"domain\":", fp); mt_json_write_string(fp, row->domain);
        fputs(",\"model_kind\":", fp); mt_json_write_string(fp, row->model_kind);
        fprintf(fp, ",\"best_epochs\":%zu,\"best_top1_mean\":", row->best_epochs);
        mt_json_number_or_null(fp, row->best_top1_mean);
        fputs(",\"best_top1_stdev\":", fp); mt_json_number_or_null(fp, row->best_top1_stdev);
        fprintf(fp, ",\"last_epochs\":%zu,\"last_top1_mean\":", row->last_epochs);
        mt_json_number_or_null(fp, row->last_top1_mean);
        fputs(",\"last_gain\":", fp); mt_json_number_or_null(fp, row->last_gain);
        fputs(",\"last_window_gain\":", fp); mt_json_number_or_null(fp, row->last_window_gain);
        fputs(",\"best_gain_from_first\":", fp); mt_json_number_or_null(fp, row->best_gain_from_first);
        fputs(",\"plateau_threshold\":", fp); mt_json_number_or_null(fp, row->plateau_threshold);
        fprintf(fp,
                ",\"plateau_window\":%zu,\"plateau_method\":\"last_step_and_window_gain\","
                "\"best_epoch_in_last_window\":%s,\"plateau_detected\":%s,"
                "\"parameters_mean\":",
                row->plateau_window,
                row->best_epoch_in_last_window ? "true" : "false",
                row->plateau_detected ? "true" : "false");
        mt_json_number_or_null(fp, row->parameters_mean);
        fputs(",\"forward_flops_mean\":", fp); mt_json_number_or_null(fp, row->forward_flops_mean);
        fputs("}\n", fp);
    }
    fclose(fp);
    return 0;
}

static void mt_write_capacity_ladder_json(FILE *fp,
                                          const MTTinyCapacity *capacities,
                                          size_t count)
{
    fputc('[', fp);
    for (size_t i = 0U; i < count; ++i) {
        if (i != 0U) {
            fputc(',', fp);
        }
        mt_json_write_string(fp, capacities[i].label);
    }
    fputc(']', fp);
}

static int mt_write_tiny_width_jsonl(const char *path,
                                     const MTTinyWidth *widths,
                                     size_t count,
                                     const MTTinyCapacity *capacities,
                                     size_t capacity_count)
{
    char date[11];
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    mt_today_iso(date);
    for (size_t i = 0U; i < count; ++i) {
        const MTTinyWidth *row = &widths[i];
        fputs("{\"schema\":\"crlp.multitask.tiny_net_width_help.v1\",\"report_version\":1,\"date\":", fp);
        mt_json_write_string(fp, date);
        fputs(",\"domain\":", fp); mt_json_write_string(fp, row->domain);
        fputs(",\"model_kind\":", fp); mt_json_write_string(fp, row->model_kind);
        fputs(",\"capacity_ladder\":", fp); mt_write_capacity_ladder_json(fp, capacities, capacity_count);
        fputs(",\"smallest_hidden\":", fp); mt_json_write_string(fp, row->smallest_hidden);
        fputs(",\"largest_hidden\":", fp); mt_json_write_string(fp, row->largest_hidden);
        fputs(",\"best_hidden\":", fp); mt_json_write_string(fp, row->best_hidden);
        fputs(",\"best_top1_mean\":", fp); mt_json_number_or_null(fp, row->best_top1_mean);
        fputs(",\"largest_minus_smallest_top1\":", fp); mt_json_number_or_null(fp, row->largest_minus_smallest_top1);
        fprintf(fp, ",\"larger_width_helped\":%s}\n", row->larger_width_helped ? "true" : "false");
    }
    fclose(fp);
    return 0;
}

static int mt_write_tiny_verdict_jsonl(const char *path,
                                       const MTTinyVerdict *verdicts,
                                       size_t count,
                                       const MTTinyCapacity *capacities,
                                       size_t capacity_count,
                                       const size_t *epoch_grid,
                                       size_t epoch_count,
                                       double plateau_threshold,
                                       size_t plateau_window)
{
    char date[11];
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    mt_today_iso(date);
    for (size_t i = 0U; i < count; ++i) {
        const MTTinyVerdict *row = &verdicts[i];
        fputs("{\"schema\":\"crlp.multitask.tiny_net_audit_verdict.v1\",\"report_version\":1,\"date\":", fp);
        mt_json_write_string(fp, date);
        fputs(",\"domain\":", fp); mt_json_write_string(fp, row->domain);
        fputs(",\"model_kind\":", fp); mt_json_write_string(fp, row->model_kind);
        fputs(",\"capacity_ladder\":", fp); mt_write_capacity_ladder_json(fp, capacities, capacity_count);
        fprintf(fp,
                ",\"capacity_count\":%ld,\"capacity_coverage_complete\":%s,"
                "\"restart_count\":%ld,\"epoch_checkpoints\":[",
                row->capacity_count,
                row->capacity_coverage_complete ? "true" : "false",
                row->restart_count);
        for (size_t j = 0U; j < epoch_count; ++j) {
            if (j != 0U) {
                fputc(',', fp);
            }
            fprintf(fp, "%zu", epoch_grid[j]);
        }
        fprintf(fp,
                "],\"epoch_checkpoint_count\":%ld,\"plateau_threshold\":%.17g,"
                "\"plateau_window\":%zu,\"plateau_count\":%ld,"
                "\"all_plateau_detected\":%s,\"width_verdict_present\":%s,"
                "\"best_hidden\":",
                row->epoch_checkpoint_count,
                plateau_threshold,
                plateau_window,
                row->plateau_count,
                row->all_plateau_detected ? "true" : "false",
                row->width_verdict_present ? "true" : "false");
        mt_json_write_string(fp, row->best_hidden[0] == '\0' ? "" : row->best_hidden);
        fputs(",\"best_top1_mean\":", fp); mt_json_number_or_null(fp, row->best_top1_mean);
        fprintf(fp,
                ",\"larger_width_helped\":%s,\"audit_ready\":%s}\n",
                row->larger_width_helped ? "true" : "false",
                row->audit_ready ? "true" : "false");
    }
    fclose(fp);
    return 0;
}

static int mt_write_tiny_diagnostics_jsonl(const char *path,
                                           const MTTinyCurve *curves,
                                           size_t curve_count,
                                           const MTTinyPlateau *plateaus,
                                           size_t plateau_count,
                                           const MTTinyWidth *widths,
                                           size_t width_count,
                                           const MTTinyVerdict *verdicts,
                                           size_t verdict_count,
                                           const MTTinyCapacity *capacities,
                                           size_t capacity_count)
{
    const double seed_stdev_threshold = 0.03;
    char date[11];
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    mt_today_iso(date);
    for (size_t i = 0U; i < verdict_count; ++i) {
        const MTTinyVerdict *verdict = &verdicts[i];
        const MTTinyWidth *width =
            mt_tiny_find_width(widths, width_count, verdict->domain, verdict->model_kind);
        const MTTinyPlateau *best = NULL;
        const MTTinyPlateau *smallest = NULL;
        const MTTinyPlateau *largest = NULL;
        const char *diagnosis = "mixed_unready";
        const char *next_step = "inspect_learning_curves";
        long observed_capacity_count = 0L;
        long non_plateau_count = 0L;
        long late_best_not_plateau_count = 0L;
        long high_seed_stdev_count = 0L;
        double max_seed_stdev = 0.0;
        double max_best_seed_spread = 0.0;
        double max_abs_last_gain = 0.0;
        double max_abs_last_window_gain = 0.0;
        double width_delta = 0.0;

        for (size_t cap_index = 0U; cap_index < capacity_count; ++cap_index) {
            const MTTinyPlateau *row = mt_tiny_find_plateau(plateaus,
                                                            plateau_count,
                                                            capacities[cap_index].label,
                                                            verdict->domain,
                                                            verdict->model_kind);
            if (row == NULL) {
                continue;
            }
            ++observed_capacity_count;
            if (smallest == NULL) {
                smallest = row;
            }
            largest = row;
            if (best == NULL || row->best_top1_mean > best->best_top1_mean) {
                best = row;
            }
            if (!row->plateau_detected) {
                ++non_plateau_count;
                if (row->best_epoch_in_last_window) {
                    ++late_best_not_plateau_count;
                }
            }
            if (row->best_top1_stdev >= seed_stdev_threshold) {
                ++high_seed_stdev_count;
            }
            if (row->best_top1_stdev > max_seed_stdev) {
                max_seed_stdev = row->best_top1_stdev;
            }
            if (fabs(row->last_gain) > max_abs_last_gain) {
                max_abs_last_gain = fabs(row->last_gain);
            }
            if (fabs(row->last_window_gain) > max_abs_last_window_gain) {
                max_abs_last_window_gain = fabs(row->last_window_gain);
            }
            {
                const MTTinyCurve *curve = mt_tiny_find_curve(curves,
                                                              curve_count,
                                                              row->hidden,
                                                              row->domain,
                                                              row->model_kind,
                                                              row->best_epochs);
                if (curve != NULL) {
                    double spread = curve->top1_max - curve->top1_min;
                    if (spread > max_best_seed_spread) {
                        max_best_seed_spread = spread;
                    }
                }
            }
        }

        if (width != NULL) {
            width_delta = width->largest_minus_smallest_top1;
        }
        if (verdict->audit_ready) {
            diagnosis = "audit_ready";
            next_step = "eligible_for_claim_review";
        } else if (best == NULL) {
            diagnosis = "missing_coverage";
            next_step = "finish_or_resume_audit_cells";
        } else if (best->best_top1_mean < 0.25) {
            diagnosis = "weak_signal";
            next_step = "inspect_feature_or_label_encoding";
        } else if (high_seed_stdev_count > 0L &&
                   max_seed_stdev >= fabs(width_delta)) {
            diagnosis = "seed_variance_dominates_width";
            next_step = "increase_restarts_before_width_claim";
        } else if (observed_capacity_count > 0L &&
                   non_plateau_count == observed_capacity_count) {
            diagnosis = "not_plateaued";
            next_step = "extend_or_stabilize_training";
        } else if (width != NULL && !width->larger_width_helped) {
            diagnosis = "not_width_limited";
            next_step = "try_protocol_artifacts_or_sampling";
        } else if (width != NULL && width->larger_width_helped &&
                   non_plateau_count > 0L) {
            diagnosis = "capacity_helped_but_unsettled";
            next_step = "run_stronger_baseline_ladder";
        }

        fputs("{\"schema\":\"crlp.multitask.tiny_net_diagnostic.v1\","
              "\"report_version\":1,\"date\":",
              fp);
        mt_json_write_string(fp, date);
        fputs(",\"domain\":", fp); mt_json_write_string(fp, verdict->domain);
        fputs(",\"model_kind\":", fp); mt_json_write_string(fp, verdict->model_kind);
        fputs(",\"capacity_ladder\":", fp); mt_write_capacity_ladder_json(fp, capacities, capacity_count);
        fprintf(fp,
                ",\"capacity_count\":%ld,\"observed_capacity_count\":%ld,"
                "\"restart_count\":%ld,\"epoch_checkpoint_count\":%ld,"
                "\"plateau_count\":%ld,\"non_plateau_count\":%ld,"
                "\"late_best_not_plateau_count\":%ld,"
                "\"high_seed_stdev_count\":%ld,\"audit_ready\":%s,"
                "\"best_hidden\":",
                verdict->capacity_count,
                observed_capacity_count,
                verdict->restart_count,
                verdict->epoch_checkpoint_count,
                verdict->plateau_count,
                non_plateau_count,
                late_best_not_plateau_count,
                high_seed_stdev_count,
                verdict->audit_ready ? "true" : "false");
        mt_json_write_string(fp, best == NULL ? "" : best->hidden);
        if (best != NULL) {
            fprintf(fp, ",\"best_epochs\":%zu,\"best_top1_mean\":", best->best_epochs);
            mt_json_number_or_null(fp, best->best_top1_mean);
            fputs(",\"best_top1_stdev\":", fp); mt_json_number_or_null(fp, best->best_top1_stdev);
        } else {
            fputs(",\"best_epochs\":null,\"best_top1_mean\":null,\"best_top1_stdev\":null", fp);
        }
        fputs(",\"smallest_hidden\":", fp);
        mt_json_write_string(fp, smallest == NULL ? "" : smallest->hidden);
        fputs(",\"smallest_best_top1_mean\":", fp);
        mt_json_number_or_null(fp, smallest == NULL ? NAN : smallest->best_top1_mean);
        fputs(",\"largest_hidden\":", fp);
        mt_json_write_string(fp, largest == NULL ? "" : largest->hidden);
        fputs(",\"largest_best_top1_mean\":", fp);
        mt_json_number_or_null(fp, largest == NULL ? NAN : largest->best_top1_mean);
        fputs(",\"largest_minus_smallest_top1\":", fp);
        mt_json_number_or_null(fp, width == NULL ? NAN : width->largest_minus_smallest_top1);
        fprintf(fp, ",\"larger_width_helped\":%s,\"max_seed_stdev\":",
                width != NULL && width->larger_width_helped ? "true" : "false");
        mt_json_number_or_null(fp, max_seed_stdev);
        fputs(",\"max_best_seed_spread\":", fp); mt_json_number_or_null(fp, max_best_seed_spread);
        fputs(",\"max_abs_last_gain\":", fp); mt_json_number_or_null(fp, max_abs_last_gain);
        fputs(",\"max_abs_last_window_gain\":", fp); mt_json_number_or_null(fp, max_abs_last_window_gain);
        fputs(",\"seed_stdev_threshold\":", fp); mt_json_number_or_null(fp, seed_stdev_threshold);
        fputs(",\"diagnosis\":", fp); mt_json_write_string(fp, diagnosis);
        fputs(",\"next_step\":", fp); mt_json_write_string(fp, next_step);
        fputs("}\n", fp);
    }
    fclose(fp);
    return 0;
}

static int mt_write_tiny_report_md(const char *path,
                                   const uint64_t *seeds,
                                   size_t seed_count,
                                   const MTTinyCapacity *capacities,
                                   size_t capacity_count,
                                   const size_t *epoch_grid,
                                   size_t epoch_count,
                                   const MTConfig *base,
                                   const MTTinyPlateau *plateaus,
                                   size_t plateau_count,
                                   const MTTinyWidth *widths,
                                   size_t width_count,
                                   const MTTinyVerdict *verdicts,
                                   size_t verdict_count,
                                   const char *const *domains,
                                   size_t domain_count,
                                   const char *const *models,
                                   size_t model_count)
{
    FILE *fp = fopen(path, "w");
    char date[11];
    if (fp == NULL) {
        return -1;
    }
    mt_today_iso(date);
    fprintf(fp, "# Fully Trained Tiny-Net Audit\n\nDate: %s\nSeeds: `", date);
    for (size_t i = 0U; i < seed_count; ++i) {
        fprintf(fp, "%s%llu", i == 0U ? "" : ", ", (unsigned long long)seeds[i]);
    }
    fputs("`\nCapacities: `", fp);
    for (size_t i = 0U; i < capacity_count; ++i) {
        fprintf(fp, "%s%s", i == 0U ? "" : ", ", capacities[i].label);
    }
    fputs("`\nModel variants: `", fp);
    for (size_t i = 0U; i < model_count; ++i) {
        fprintf(fp, "%s%s", i == 0U ? "" : ", ", models[i]);
    }
    fputs("`\nEpoch checkpoints: `", fp);
    for (size_t i = 0U; i < epoch_count; ++i) {
        fprintf(fp, "%s%zu", i == 0U ? "" : ", ", epoch_grid[i]);
    }
    fprintf(fp,
            "`\nSteps per epoch: `%zu`; eval samples: `%zu`\n\n"
            "## Best Checkpoints\n\n"
            "| Hidden | Model | Prime | Grid | Delayed | ARC |\n"
            "| --- | --- | ---: | ---: | ---: | ---: |\n",
            base->steps_per_epoch,
            base->eval_samples);
    for (size_t cap_index = 0U; cap_index < capacity_count; ++cap_index) {
        for (size_t model_index = 0U; model_index < model_count; ++model_index) {
            fprintf(fp, "| `%s` | `%s` |", capacities[cap_index].label, models[model_index]);
            for (size_t domain_index = 0U; domain_index < domain_count; ++domain_index) {
                const MTTinyPlateau *row = mt_tiny_find_plateau(plateaus,
                                                                plateau_count,
                                                                capacities[cap_index].label,
                                                                domains[domain_index],
                                                                models[model_index]);
                char pct[32];
                if (row == NULL) {
                    fputs(" - |", fp);
                } else {
                    mt_format_pct(pct, sizeof(pct), row->best_top1_mean);
                    fprintf(fp, " %s @ %zu |", pct, row->best_epochs);
                }
            }
            fputc('\n', fp);
        }
    }
    fputs("\n## Plateau Flags\n\n"
          "| Hidden | Domain | Model | Last gain | Window gain | Plateau |\n"
          "| --- | --- | --- | ---: | ---: | --- |\n",
          fp);
    for (size_t i = 0U; i < plateau_count; ++i) {
        char gain[32], window_gain[32];
        mt_format_delta(gain, sizeof(gain), plateaus[i].last_gain);
        mt_format_delta(window_gain, sizeof(window_gain), plateaus[i].last_window_gain);
        fprintf(fp,
                "| `%s` | `%s` | `%s` | %s | %s | %s |\n",
                plateaus[i].hidden,
                plateaus[i].domain,
                plateaus[i].model_kind,
                gain,
                window_gain,
                plateaus[i].plateau_detected ? "yes" : "no");
    }
    fputs("\n## Width Verdict\n\n"
          "| Domain | Model | Best hidden | Largest-smallest top-1 | Larger helped |\n"
          "| --- | --- | ---: | ---: | --- |\n",
          fp);
    for (size_t i = 0U; i < width_count; ++i) {
        char delta[32];
        mt_format_delta(delta, sizeof(delta), widths[i].largest_minus_smallest_top1);
        fprintf(fp,
                "| `%s` | `%s` | `%s` | %s | %s |\n",
                widths[i].domain,
                widths[i].model_kind,
                widths[i].best_hidden,
                delta,
                widths[i].larger_width_helped ? "yes" : "no");
    }
    fputs("\n## Audit Verdict\n\n"
          "| Domain | Model | Coverage | Plateau rows | Best hidden | Audit ready |\n"
          "| --- | --- | --- | ---: | ---: | --- |\n",
          fp);
    for (size_t i = 0U; i < verdict_count; ++i) {
        fprintf(fp,
                "| `%s` | `%s` | %s | %ld/%ld | `%s` | %s |\n",
                verdicts[i].domain,
                verdicts[i].model_kind,
                verdicts[i].capacity_coverage_complete ? "yes" : "no",
                verdicts[i].plateau_count,
                verdicts[i].capacity_count,
                verdicts[i].best_hidden[0] == '\0' ? "-" : verdicts[i].best_hidden,
                verdicts[i].audit_ready ? "yes" : "no");
    }
    fclose(fp);
    return 0;
}

static size_t mt_epoch_grid_max(const size_t *epoch_grid, size_t epoch_count)
{
    size_t max_epoch = 0U;
    for (size_t i = 0U; i < epoch_count; ++i) {
        if (epoch_grid[i] > max_epoch) {
            max_epoch = epoch_grid[i];
        }
    }
    return max_epoch;
}

static int mt_epoch_grid_contains(const size_t *epoch_grid,
                                  size_t epoch_count,
                                  size_t epoch)
{
    for (size_t i = 0U; i < epoch_count; ++i) {
        if (epoch_grid[i] == epoch) {
            return 1;
        }
    }
    return 0;
}

static void mt_tiny_record_task_checkpoint(MTScorer *scorer,
                                           const MTData *data,
                                           const MTConfig *run,
                                           const char *model_kind,
                                           const char *training_order,
                                           const size_t updates[MT_TASK_COUNT],
                                           size_t epoch,
                                           MTTask task)
{
    MTConfig checkpoint = *run;
    checkpoint.epochs = epoch;
    MTMetrics metrics =
        mt_evaluate_task(scorer, data, &checkpoint, task, checkpoint.eval_samples);
    mt_print_metrics_json(scorer,
                          model_kind,
                          training_order,
                          task,
                          updates[task],
                          &checkpoint,
                          metrics);
    if (task == MT_TASK_ARC_AGI) {
        mt_print_arc_ood_task_breakdown(scorer,
                                        data,
                                        &checkpoint,
                                        model_kind,
                                        training_order,
                                        updates[task]);
    }
}

static void mt_tiny_record_checkpoint(MTScorer *scorer,
                                      const MTData *data,
                                      const MTConfig *run,
                                      const char *model_kind,
                                      const char *training_order,
                                      const size_t updates[MT_TASK_COUNT],
                                      size_t epoch)
{
    MTConfig checkpoint = *run;
    MTTask eval_tasks[MT_TASK_COUNT];
    size_t eval_count;
    checkpoint.epochs = epoch;
    eval_count = mt_fill_active_tasks(checkpoint.eval_task_mask,
                                      eval_tasks,
                                      MT_TASK_COUNT);
    for (size_t task_index = 0U; task_index < eval_count; ++task_index) {
        MTTask task = eval_tasks[task_index];
        mt_tiny_record_task_checkpoint(scorer,
                                       data,
                                       &checkpoint,
                                       model_kind,
                                       training_order,
                                       updates,
                                       epoch,
                                       task);
    }
}

static int mt_tiny_cell_path(char *out,
                             size_t out_size,
                             const char *cell_dir,
                             const MTTinyCapacity *capacity,
                             uint64_t seed)
{
    char filename[128];
    int written;
    if (out == NULL || cell_dir == NULL || capacity == NULL) {
        return -1;
    }
    written = snprintf(filename,
                       sizeof(filename),
                       "tiny_net_plateau.%s.seed-%llu.jsonl",
                       capacity->label,
                       (unsigned long long)seed);
    if (written < 0 || (size_t)written >= sizeof(filename)) {
        return -1;
    }
    return mt_join_path(out, out_size, cell_dir, filename);
}

static uint64_t mt_fnv1a_update_bytes(uint64_t hash, const void *data, size_t size)
{
    const unsigned char *bytes = (const unsigned char *)data;
    for (size_t i = 0U; i < size; ++i) {
        hash ^= (uint64_t)bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t mt_fnv1a_update_text(uint64_t hash, const char *text)
{
    const char *value = text == NULL ? "" : text;
    return mt_fnv1a_update_bytes(hash, value, strlen(value));
}

static uint64_t mt_tiny_audit_signature(const MTConfig *base,
                                        const size_t *epoch_grid,
                                        size_t epoch_count)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    char buffer[128];
    char task_set[128];
    char eval_task_set[128];
    mt_task_set_name(base->task_mask, task_set, sizeof(task_set));
    mt_task_set_name(base->eval_task_mask, eval_task_set, sizeof(eval_task_set));
    hash = mt_fnv1a_update_text(hash, "tiny-net-plateau-v2;");
    hash = mt_fnv1a_update_text(hash, task_set);
    hash = mt_fnv1a_update_text(hash, ";eval=");
    hash = mt_fnv1a_update_text(hash, eval_task_set);
    snprintf(buffer,
             sizeof(buffer),
             ";steps=%zu;eval_samples=%zu;lr=%.17g;order=%s;tags=%d;heads=%d;baselines=%d;",
             base->steps_per_epoch,
             base->eval_samples,
             (double)base->learning_rate,
             mt_training_order_name((MTTrainingOrder)base->training_order),
             base->use_task_tags,
             base->run_task_head_model,
             base->run_task_baselines);
    hash = mt_fnv1a_update_text(hash, buffer);
    for (size_t i = 0U; i < epoch_count; ++i) {
        snprintf(buffer, sizeof(buffer), "%zu,", epoch_grid[i]);
        hash = mt_fnv1a_update_text(hash, buffer);
    }
    return hash;
}

static int mt_tiny_shard_selected(size_t cell_index,
                                  size_t shard_index,
                                  size_t shard_count)
{
    if (shard_count <= 1U) {
        return 1;
    }
    return cell_index % shard_count == shard_index - 1U;
}

static int mt_run_tiny_net_plateau_cell(const MTConfig *base,
                                        const MTTinyCapacity *capacity,
                                        uint64_t seed,
                                        const size_t *epoch_grid,
                                        size_t epoch_count,
                                        MTReportRows *rows,
                                        FILE *raw)
{
    uint32_t all_task_mask = mt_task_bit(MT_TASK_PRIME_GAP) |
                             mt_task_bit(MT_TASK_GRID_PATH) |
                             mt_task_bit(MT_TASK_DELAYED_CLAIM) |
                             mt_task_bit(MT_TASK_ARC_AGI);
    size_t max_epoch;
    MTConfig run;
    MTData data;
    int data_ready = 0;
    int status = 0;
    nn_real *candidate_inputs = NULL;
    NeuralNet *shared = NULL;
    MTTaskHeadedScorer *task_heads = NULL;

    if (base == NULL || capacity == NULL || epoch_grid == NULL || epoch_count == 0U ||
        rows == NULL || raw == NULL) {
        return 0;
    }
    max_epoch = mt_epoch_grid_max(epoch_grid, epoch_count);
    if (max_epoch == 0U) {
        return 0;
    }
    run = *base;

    run.tiny_net_plateau_audit = 0;
    run.adapter_heads_report = 0;
    run.hidden1 = capacity->hidden1;
    run.hidden2 = capacity->hidden2;
    run.seed = seed;
    run.epochs = max_epoch;
    run.task_mask = all_task_mask;
    run.eval_task_mask = base->eval_task_mask == 0U ? all_task_mask : base->eval_task_mask;
    run.arc_eval_suite = MT_ARC_EVAL_0520FDE7;
    run.arc_ood_task = -1;
    run.report_arc_ood_tasks = 0;
    run.quiet = 1;
    run.roadmap_item = "E-MT.9";
    run.adapter_eval_suite = NULL;
    run.json_output = raw;
    run.report_rows = rows;

    if (!mt_prepare_data(&data)) {
        fputs("failed to prepare multitask datasets\n", stderr);
        goto done;
    }
    data_ready = 1;

    candidate_inputs =
        calloc((size_t)MT_MAX_CANDIDATES * MT_FEATURE_COUNT, sizeof(*candidate_inputs));
    if (candidate_inputs == NULL) {
        goto done;
    }

    shared = mt_create_model(&run, (unsigned int)(run.seed & 0xffffffffu));
    if (shared == NULL) {
        goto done;
    }
    {
        uint32_t rng =
            (uint32_t)(run.seed ^ (run.seed >> 32) ^ UINT64_C(0x31415926));
        size_t shared_updates[MT_TASK_COUNT] = {0, 0, 0, 0};
        MTScorer shared_scorer;
        shared_scorer.kind = MT_SCORER_NN;
        shared_scorer.net = shared;
        shared_scorer.task_heads = NULL;
        for (size_t epoch = 0U; epoch < max_epoch; ++epoch) {
            if (!mt_train_mixed_one_epoch(shared,
                                          &data,
                                          &run,
                                          epoch,
                                          &rng,
                                          shared_updates,
                                          candidate_inputs)) {
                goto done;
            }
            if (mt_epoch_grid_contains(epoch_grid, epoch_count, epoch + 1U)) {
                mt_tiny_record_checkpoint(&shared_scorer,
                                          &data,
                                          &run,
                                          "shared",
                                          mt_training_order_name(
                                              (MTTrainingOrder)run.training_order),
                                          shared_updates,
                                          epoch + 1U);
            }
        }
    }

    if (run.run_task_head_model) {
        task_heads =
            mt_task_heads_create(&run,
                                 (unsigned int)((run.seed + UINT64_C(303)) &
                                                UINT64_C(0xffffffff)));
        if (task_heads == NULL) {
            goto done;
        }
        uint32_t rng =
            (uint32_t)(run.seed ^ (run.seed >> 32) ^ UINT64_C(0x27182818));
        size_t task_head_updates[MT_TASK_COUNT] = {0, 0, 0, 0};
        MTScorer task_head_scorer;
        task_head_scorer.kind = MT_SCORER_TASK_HEADS;
        task_head_scorer.net = NULL;
        task_head_scorer.task_heads = task_heads;
        for (size_t epoch = 0U; epoch < max_epoch; ++epoch) {
            if (!mt_train_mixed_task_heads_one_epoch(task_heads,
                                                     &data,
                                                     &run,
                                                     epoch,
                                                     &rng,
                                                     task_head_updates,
                                                     candidate_inputs)) {
                goto done;
            }
            if (mt_epoch_grid_contains(epoch_grid, epoch_count, epoch + 1U)) {
                mt_tiny_record_checkpoint(&task_head_scorer,
                                          &data,
                                          &run,
                                          "shared_task_heads",
                                          mt_training_order_name(
                                              (MTTrainingOrder)run.training_order),
                                          task_head_updates,
                                          epoch + 1U);
            }
        }
        mt_task_heads_free(task_heads);
        task_heads = NULL;
    }

    if (run.run_task_baselines) {
        MTTask active_tasks[MT_TASK_COUNT];
        size_t active_count = mt_fill_active_tasks(run.task_mask,
                                                   active_tasks,
                                                   MT_TASK_COUNT);
        if (active_count == 0U) {
            goto done;
        }
        for (size_t task_index = 0U; task_index < active_count; ++task_index) {
            MTTask task = active_tasks[task_index];
            if ((run.eval_task_mask & mt_task_bit(task)) == 0) {
                continue;
            }
            NeuralNet *single =
                mt_create_model(&run,
                                (unsigned int)((run.seed + 101u + task_index) &
                                               0xffffffffu));
            if (single == NULL) {
                goto done;
            }
            uint32_t rng =
                (uint32_t)(run.seed ^ (run.seed >> 32) ^
                           (0x9e3779b9u + (uint32_t)task));
            size_t single_updates[MT_TASK_COUNT] = {0, 0, 0, 0};
            MTScorer single_scorer;
            single_scorer.kind = MT_SCORER_NN;
            single_scorer.net = single;
            single_scorer.task_heads = NULL;
            for (size_t epoch = 0U; epoch < max_epoch; ++epoch) {
                size_t target_updates = ((epoch + 1U) * run.steps_per_epoch) / active_count;
                if (target_updates == 0U) {
                    target_updates = 1U;
                }
                if (target_updates > single_updates[task]) {
                    double ignored_loss = 0.0;
                    if (!mt_train_single_task_steps(single,
                                                    &data,
                                                    &run,
                                                    task,
                                                    &rng,
                                                    candidate_inputs,
                                                    target_updates - single_updates[task],
                                                    &ignored_loss)) {
                        nn_free(single);
                        goto done;
                    }
                    single_updates[task] = target_updates;
                }
                if (mt_epoch_grid_contains(epoch_grid, epoch_count, epoch + 1U)) {
                    run.epochs = epoch + 1U;
                    mt_tiny_record_task_checkpoint(&single_scorer,
                                                   &data,
                                                   &run,
                                                   "single_task",
                                                   "domain_only_random",
                                                   single_updates,
                                                   epoch + 1U,
                                                   task);
                }
            }
            nn_free(single);
        }
    }

    status = 1;
done:
    mt_task_heads_free(task_heads);
    nn_free(shared);
    free(candidate_inputs);
    if (data_ready) {
        mt_free_data(&data);
    }
    return status;
}

static int mt_run_tiny_net_plateau_audit(const MTConfig *base)
{
    uint32_t all_task_mask = mt_task_bit(MT_TASK_PRIME_GAP) |
                             mt_task_bit(MT_TASK_GRID_PATH) |
                             mt_task_bit(MT_TASK_DELAYED_CLAIM) |
                             mt_task_bit(MT_TASK_ARC_AGI);
    MTTask audit_tasks[MT_TASK_COUNT];
    const char *domains[MT_TASK_COUNT];
    const char *models[3];
    size_t domain_count = 0U;
    size_t model_count = 0U;
    uint64_t seeds[128];
    MTTinyCapacity capacities[64];
    size_t epoch_grid[128];
    size_t seed_count = 0U, capacity_count = 0U, epoch_count = 0U;
    MTReportRows rows;
    MTTinyCurve *curves = NULL;
    MTTinyPlateau *plateaus = NULL;
    MTTinyWidth *widths = NULL;
    MTTinyVerdict *verdicts = NULL;
    size_t curve_count = 0U, plateau_count = 0U, width_count = 0U, verdict_count = 0U;
    size_t present_cell_count = 0U, missing_cell_count = 0U;
    char raw_path[4096], commands_path[4096], curve_path[4096], plateau_csv_path[4096];
    char plateau_jsonl_path[4096], width_path[4096], verdict_path[4096], markdown_path[4096];
    char diagnostics_path[4096];
    char cell_dir_path[4096], cell_dir_name[128];
    FILE *raw = NULL;
    FILE *commands = NULL;
    int status = EXIT_FAILURE;
    memset(&rows, 0, sizeof(rows));
    domain_count = mt_fill_active_tasks(base->eval_task_mask == 0U ? all_task_mask : base->eval_task_mask,
                                        audit_tasks,
                                        MT_TASK_COUNT);
    if (domain_count == 0U) {
        fputs("tiny-net audit has no eval domains\n", stderr);
        return EXIT_FAILURE;
    }
    for (size_t i = 0U; i < domain_count; ++i) {
        domains[i] = mt_task_name(audit_tasks[i]);
    }
    models[model_count++] = "shared";
    if (base->run_task_head_model) {
        models[model_count++] = "shared_task_heads";
    }
    if (base->run_task_baselines) {
        models[model_count++] = "single_task";
    }
    if (!mt_parse_seed_list(base->report_seeds, seeds, sizeof(seeds) / sizeof(seeds[0]), &seed_count)) {
        fputs("invalid --seeds list\n", stderr);
        return EXIT_FAILURE;
    }
    if (!mt_parse_capacity_list(base->tiny_net_capacities,
                                capacities,
                                sizeof(capacities) / sizeof(capacities[0]),
                                &capacity_count)) {
        fputs("invalid --capacities list\n", stderr);
        return EXIT_FAILURE;
    }
    if (!mt_parse_size_list(base->tiny_net_epoch_grid,
                            epoch_grid,
                            sizeof(epoch_grid) / sizeof(epoch_grid[0]),
                            &epoch_count)) {
        fputs("invalid --epoch-grid list\n", stderr);
        return EXIT_FAILURE;
    }
    {
        uint64_t signature = mt_tiny_audit_signature(base, epoch_grid, epoch_count);
        int written = snprintf(cell_dir_name,
                               sizeof(cell_dir_name),
                               "tiny_net_plateau_cells/%016llx",
                               (unsigned long long)signature);
        if (written < 0 || (size_t)written >= sizeof(cell_dir_name)) {
            fputs("failed to prepare tiny-net cell cache path\n", stderr);
            return EXIT_FAILURE;
        }
    }
    if (mt_mkdir_p(base->output_dir) != 0 ||
        mt_join_path(cell_dir_path, sizeof(cell_dir_path), base->output_dir, cell_dir_name) != 0 ||
        mt_mkdir_p(cell_dir_path) != 0 ||
        mt_join_path(raw_path, sizeof(raw_path), base->output_dir, "tiny_net_plateau.raw.jsonl") != 0 ||
        mt_join_path(commands_path, sizeof(commands_path), base->output_dir, "tiny_net_plateau.commands.txt") != 0 ||
        mt_join_path(curve_path, sizeof(curve_path), base->output_dir, "tiny_net_learning_curve.csv") != 0 ||
        mt_join_path(plateau_csv_path, sizeof(plateau_csv_path), base->output_dir, "tiny_net_plateau_summary.csv") != 0 ||
        mt_join_path(plateau_jsonl_path, sizeof(plateau_jsonl_path), base->output_dir, "tiny_net_plateau_summary.jsonl") != 0 ||
        mt_join_path(width_path, sizeof(width_path), base->output_dir, "tiny_net_width_help.jsonl") != 0 ||
        mt_join_path(verdict_path, sizeof(verdict_path), base->output_dir, "tiny_net_audit_verdict.jsonl") != 0 ||
        mt_join_path(diagnostics_path, sizeof(diagnostics_path), base->output_dir, "tiny_net_diagnostics.jsonl") != 0 ||
        mt_join_path(markdown_path, sizeof(markdown_path), base->output_dir, "tiny_net_plateau_audit.md") != 0) {
        fputs("failed to prepare tiny-net output paths\n", stderr);
        return EXIT_FAILURE;
    }
    commands = fopen(commands_path, "w");
    if (commands == NULL) {
        fputs("failed to open tiny-net command file\n", stderr);
        goto done;
    }
    fprintf(commands, "# tiny-net cell cache: %s\n", cell_dir_path);
    fprintf(commands,
            "# resume=%s shard=%zu/%zu\n",
            base->tiny_net_resume ? "true" : "false",
            base->tiny_net_shard_index,
            base->tiny_net_shard_count);
    for (size_t cap_index = 0U; cap_index < capacity_count; ++cap_index) {
        for (size_t seed_index = 0U; seed_index < seed_count; ++seed_index) {
            MTConfig command_config = *base;
            char cell_path[4096];
            char tmp_cell_path[4096];
            char eval_task_set[128];
            size_t cell_index = cap_index * seed_count + seed_index;
            int selected = mt_tiny_shard_selected(cell_index,
                                                  base->tiny_net_shard_index,
                                                  base->tiny_net_shard_count);
            command_config.eval_task_mask =
                base->eval_task_mask == 0U ? all_task_mask : base->eval_task_mask;
            mt_task_set_name(command_config.eval_task_mask,
                             eval_task_set,
                             sizeof(eval_task_set));
            if (mt_tiny_cell_path(cell_path,
                                  sizeof(cell_path),
                                  cell_dir_path,
                                  &capacities[cap_index],
                                  seeds[seed_index]) != 0) {
                goto done;
            }
            fprintf(commands,
                    "multitask_domain --tiny-net-plateau-audit --output-dir \"%s\" --tasks all --eval-tasks %s --capacities %s --seeds %llu --epoch-grid %s --steps-per-epoch %zu --eval-samples %zu",
                    base->output_dir,
                    eval_task_set,
                    capacities[cap_index].label,
                    (unsigned long long)seeds[seed_index],
                    base->tiny_net_epoch_grid,
                    base->steps_per_epoch,
                    base->eval_samples);
            if (base->run_task_head_model) {
                fputs(" --task-heads", commands);
            } else {
                fputs(" --no-task-heads", commands);
            }
            if (!base->run_task_baselines) {
                fputs(" --no-task-baselines", commands);
            }
            fputc('\n', commands);
            if (selected) {
                int has_cell = mt_file_has_content(cell_path);
                if (base->tiny_net_resume && has_cell) {
                    printf("resumed tiny-net cell hidden=%s seed=%llu from %s\n",
                           capacities[cap_index].label,
                           (unsigned long long)seeds[seed_index],
                           cell_path);
                } else {
                    MTReportRows scratch_rows;
                    FILE *cell_raw;
                    int ok;
                    int written = snprintf(tmp_cell_path,
                                           sizeof(tmp_cell_path),
                                           "%s.tmp",
                                           cell_path);
                    if (written < 0 || (size_t)written >= sizeof(tmp_cell_path)) {
                        goto done;
                    }
                    memset(&scratch_rows, 0, sizeof(scratch_rows));
                    cell_raw = fopen(tmp_cell_path, "w");
                    if (cell_raw == NULL) {
                        free(scratch_rows.items);
                        goto done;
                    }
                    printf("running tiny-net cell hidden=%s seed=%llu -> %s\n",
                           capacities[cap_index].label,
                           (unsigned long long)seeds[seed_index],
                           cell_path);
                    ok = mt_run_tiny_net_plateau_cell(base,
                                                      &capacities[cap_index],
                                                      seeds[seed_index],
                                                      epoch_grid,
                                                      epoch_count,
                                                      &scratch_rows,
                                                      cell_raw);
                    if (fclose(cell_raw) != 0) {
                        ok = 0;
                    }
                    free(scratch_rows.items);
                    if (!ok) {
                        remove(tmp_cell_path);
                        goto done;
                    }
                    if (rename(tmp_cell_path, cell_path) != 0) {
                        remove(tmp_cell_path);
                        goto done;
                    }
                }
            }
            if (mt_file_has_content(cell_path)) {
                ++present_cell_count;
                if (mt_report_rows_load_jsonl(&rows, cell_path) != 0) {
                    fprintf(stderr, "failed to load tiny-net cell rows from %s\n", cell_path);
                    goto done;
                }
            } else {
                ++missing_cell_count;
            }
        }
    }
    fclose(commands);
    commands = NULL;
    raw = fopen(raw_path, "w");
    if (raw == NULL) {
        goto done;
    }
    for (size_t cap_index = 0U; cap_index < capacity_count; ++cap_index) {
        for (size_t seed_index = 0U; seed_index < seed_count; ++seed_index) {
            char cell_path[4096];
            if (mt_tiny_cell_path(cell_path,
                                  sizeof(cell_path),
                                  cell_dir_path,
                                  &capacities[cap_index],
                                  seeds[seed_index]) != 0) {
                goto done;
            }
            if (mt_file_has_content(cell_path) &&
                mt_copy_file_to(raw, cell_path) != 0) {
                goto done;
            }
        }
    }
    fclose(raw);
    raw = NULL;
    if (mt_tiny_collect_curves(&rows, &curves, &curve_count) != 0 ||
        mt_tiny_collect_plateaus(curves,
                                 curve_count,
                                 base->plateau_threshold,
                                 base->plateau_window,
                                 &plateaus,
                                 &plateau_count) != 0 ||
        mt_tiny_collect_widths(plateaus,
                               plateau_count,
                               capacities,
                               capacity_count,
                               domains,
                               domain_count,
                               models,
                               model_count,
                               &widths,
                               &width_count) != 0 ||
        mt_tiny_collect_verdicts(plateaus,
                                 plateau_count,
                                 widths,
                                 width_count,
                                 capacities,
                                 capacity_count,
                                 seeds,
                                 seed_count,
                                 epoch_grid,
                                 epoch_count,
                                 domains,
                                 domain_count,
                                 models,
                                 model_count,
                                 &verdicts,
                                 &verdict_count) != 0) {
        goto done;
    }
    if (missing_cell_count > 0U) {
        for (size_t i = 0U; i < verdict_count; ++i) {
            verdicts[i].capacity_coverage_complete = 0;
            verdicts[i].audit_ready = 0;
        }
        printf("tiny-net audit partial cell coverage: %zu/%zu cells present\n",
               present_cell_count,
               present_cell_count + missing_cell_count);
    }
    if (mt_write_tiny_learning_curve_csv(curve_path, curves, curve_count) != 0 ||
        mt_write_tiny_plateau_csv(plateau_csv_path, plateaus, plateau_count) != 0 ||
        mt_write_tiny_plateau_jsonl(plateau_jsonl_path, plateaus, plateau_count) != 0 ||
        mt_write_tiny_width_jsonl(width_path, widths, width_count, capacities, capacity_count) != 0 ||
        mt_write_tiny_verdict_jsonl(verdict_path,
                                    verdicts,
                                    verdict_count,
                                    capacities,
                                    capacity_count,
                                    epoch_grid,
                                    epoch_count,
                                    base->plateau_threshold,
                                    base->plateau_window) != 0 ||
        mt_write_tiny_diagnostics_jsonl(diagnostics_path,
                                        curves,
                                        curve_count,
                                        plateaus,
                                        plateau_count,
                                        widths,
                                        width_count,
                                        verdicts,
                                        verdict_count,
                                        capacities,
                                        capacity_count) != 0 ||
        mt_write_tiny_report_md(markdown_path,
                                seeds,
                                seed_count,
                                capacities,
                                capacity_count,
                                epoch_grid,
                                epoch_count,
                                base,
                                plateaus,
                                plateau_count,
                                widths,
                                width_count,
                                verdicts,
                                verdict_count,
                                domains,
                                domain_count,
                                models,
                                model_count) != 0) {
        goto done;
    }
    printf("wrote %s\n", markdown_path);
    printf("wrote %s\n", verdict_path);
    printf("wrote %s\n", diagnostics_path);
    status = EXIT_SUCCESS;
done:
    if (raw != NULL) {
        fclose(raw);
    }
    if (commands != NULL) {
        fclose(commands);
    }
    free(rows.items);
    free(curves);
    free(plateaus);
    free(widths);
    free(verdicts);
    return status;
}

static int mt_run_adapter_heads_report(const MTConfig *base)
{
    static const struct {
        const char *label;
        int suite;
    } suites[] = {
        {"fixed", MT_ARC_EVAL_0520FDE7},
        {"ood", MT_ARC_EVAL_OOD_SAME_ENVELOPE},
    };
    static const char *summary_domains[] = {"arc_agi", "delayed_claim", "grid_path", "prime_gap"};
    static const char *delta_domains[] = {"prime_gap", "grid_path", "delayed_claim", "arc_agi"};
    static const char *models[] = {"shared", "shared_task_heads", "single_task"};
    uint64_t seeds[64];
    size_t seed_count = 0U;
    MTReportRows rows;
    MTReportSummary summaries[32];
    size_t summary_count = 0U;
    MTReportDelta deltas[8];
    size_t delta_count = 0U;
    char raw_path[4096], commands_path[4096], summary_path[4096], delta_path[4096];
    char decision_path[4096], markdown_path[4096];
    FILE *raw = NULL;
    FILE *commands = NULL;
    int status = EXIT_FAILURE;
    memset(&rows, 0, sizeof(rows));
    if (!mt_parse_seed_list(base->report_seeds, seeds, sizeof(seeds) / sizeof(seeds[0]), &seed_count)) {
        fputs("invalid --seeds list\n", stderr);
        return EXIT_FAILURE;
    }
    if (mt_mkdir_p(base->output_dir) != 0 ||
        mt_join_path(raw_path, sizeof(raw_path), base->output_dir, "rmt4_adapter_heads.raw.jsonl") != 0 ||
        mt_join_path(commands_path, sizeof(commands_path), base->output_dir, "rmt4_adapter_heads.commands.txt") != 0 ||
        mt_join_path(summary_path, sizeof(summary_path), base->output_dir, "rmt4_adapter_heads_summary.csv") != 0 ||
        mt_join_path(delta_path, sizeof(delta_path), base->output_dir, "rmt4_adapter_heads_deltas.csv") != 0 ||
        mt_join_path(decision_path, sizeof(decision_path), base->output_dir, "rmt4_adapter_heads_decision.jsonl") != 0 ||
        mt_join_path(markdown_path, sizeof(markdown_path), base->output_dir, "rmt4_adapter_heads.md") != 0) {
        fputs("failed to prepare adapter-head output paths\n", stderr);
        return EXIT_FAILURE;
    }
    raw = fopen(raw_path, "w");
    commands = fopen(commands_path, "w");
    if (raw == NULL || commands == NULL) {
        fputs("failed to open adapter-head output files\n", stderr);
        goto done;
    }
    for (size_t seed_index = 0U; seed_index < seed_count; ++seed_index) {
        for (size_t suite_index = 0U; suite_index < sizeof(suites) / sizeof(suites[0]); ++suite_index) {
            MTConfig run = *base;
            run.adapter_heads_report = 0;
            run.seed = seeds[seed_index];
            run.arc_eval_suite = suites[suite_index].suite;
            run.task_mask = mt_task_bit(MT_TASK_PRIME_GAP) |
                            mt_task_bit(MT_TASK_GRID_PATH) |
                            mt_task_bit(MT_TASK_DELAYED_CLAIM) |
                            mt_task_bit(MT_TASK_ARC_AGI);
            run.eval_task_mask = run.task_mask;
            run.run_task_head_model = 1;
            run.quiet = 1;
            run.roadmap_item = "R-MT.4";
            run.adapter_eval_suite = suites[suite_index].label;
            run.json_output = raw;
            run.report_rows = &rows;
            fprintf(commands,
                    "multitask_domain --tasks all --eval-tasks all --epochs %zu --steps-per-epoch %zu --eval-samples %zu --hidden %zux%zu --seed %llu --arc-eval-suite %s --task-heads",
                    run.epochs,
                    run.steps_per_epoch,
                    run.eval_samples,
                    run.hidden1,
                    run.hidden2,
                    (unsigned long long)run.seed,
                    suites[suite_index].suite == MT_ARC_EVAL_OOD_SAME_ENVELOPE ? "ood" : "0520fde7");
            if (suites[suite_index].suite == MT_ARC_EVAL_OOD_SAME_ENVELOPE) {
                fputs(" --arc-ood-task ", commands);
                if (run.arc_ood_task < 0) {
                    fputs("all", commands);
                } else {
                    fprintf(commands, "%d", run.arc_ood_task);
                }
            }
            if (!run.run_task_baselines) {
                fputs(" --no-task-baselines", commands);
            }
            fputc('\n', commands);
            if (mt_run_once(&run) != EXIT_SUCCESS) {
                goto done;
            }
        }
    }
    fclose(raw);
    fclose(commands);
    raw = NULL;
    commands = NULL;
    for (size_t suite_index = 0U; suite_index < sizeof(suites) / sizeof(suites[0]); ++suite_index) {
        size_t domain_count =
            strcmp(suites[suite_index].label, "fixed") == 0 ? sizeof(summary_domains) / sizeof(summary_domains[0]) : 1U;
        for (size_t domain_index = 0U; domain_index < domain_count; ++domain_index) {
            const char *domain = strcmp(suites[suite_index].label, "fixed") == 0 ? summary_domains[domain_index] : "arc_agi";
            for (size_t model_index = 0U; model_index < sizeof(models) / sizeof(models[0]); ++model_index) {
                MTReportSummary summary;
                if (mt_summary_collect(&rows,
                                       suites[suite_index].label,
                                       domain,
                                       models[model_index],
                                       &summary)) {
                    summaries[summary_count++] = summary;
                }
            }
        }
    }
    for (size_t i = 0U; i < sizeof(delta_domains) / sizeof(delta_domains[0]); ++i) {
        MTReportDelta delta = mt_delta_build(summaries, summary_count, "fixed", delta_domains[i]);
        if (delta.present) {
            deltas[delta_count++] = delta;
        }
    }
    {
        MTReportDelta delta = mt_delta_build(summaries, summary_count, "ood", "arc_agi");
        if (delta.present) {
            deltas[delta_count++] = delta;
        }
    }
    if (mt_write_summary_csv(summary_path, summaries, summary_count) != 0 ||
        mt_write_deltas_csv(delta_path, deltas, delta_count) != 0 ||
        mt_write_decision_jsonl(decision_path,
                                deltas,
                                delta_count,
                                base->transfer_regression_margin) != 0 ||
        mt_write_adapter_report_md(markdown_path,
                                   seeds,
                                   seed_count,
                                   base,
                                   summaries,
                                   summary_count,
                                   deltas,
                                   delta_count) != 0) {
        goto done;
    }
    printf("wrote %s\n", markdown_path);
    status = EXIT_SUCCESS;
done:
    if (raw != NULL) {
        fclose(raw);
    }
    if (commands != NULL) {
        fclose(commands);
    }
    free(rows.items);
    return status;
}
