static const char *nn_gd_haiku_metric_name(NNGDHaikuMetric metric) {
    static const char *names[NN_GD_HAIKU_METRIC_COUNT] = {
        "real_probability",
        "corrupt_probability",
        "plan_score_delta",
        "topic_tightness_delta",
        "min_line_similarity_delta",
        "corrupt_adjacency_ratio",
        "corrupt_bigram_ratio",
        "corrupt_exact_syllable_ratio",
        "replacement_line_similarity",
        "local_neighbor_fit_delta",
        "left_neighbor_similarity_delta",
        "right_neighbor_similarity_delta",
        "line_word_delta",
        "line_syllable_delta",
    };
    return names[(size_t)metric];
}

static char *nn_gd_json_dup_index_string(const NNGDJsonIndex *index,
                                         const char *key) {
    return nn_gd_json_dup_string_value(nn_gd_json_index_find(index, key));
}

static long nn_gd_json_index_long_or(const NNGDJsonIndex *index,
                                     const char *key,
                                     long default_value) {
    const char *cursor = nn_gd_json_skip_space(nn_gd_json_index_find(index, key));
    char *end = NULL;
    long value;
    if (cursor == NULL || strncmp(cursor, "null", 4U) == 0) {
        return default_value;
    }
    value = strtol(cursor, &end, 10);
    return end == cursor ? default_value : value;
}

static double nn_gd_json_index_double_or_zero(const NNGDJsonIndex *index,
                                              const char *key) {
    const char *cursor = nn_gd_json_skip_space(nn_gd_json_index_find(index, key));
    char *end = NULL;
    double value;
    if (cursor == NULL || strncmp(cursor, "null", 4U) == 0) {
        return 0.0;
    }
    value = strtod(cursor, &end);
    if (end == cursor || !isfinite(value)) {
        return 0.0;
    }
    return value;
}

static int nn_gd_haiku_is_hard_negative_corruption(const char *corruption) {
    return strcmp(corruption, "line_splice") == 0 ||
           strcmp(corruption, "word_salad") == 0 ||
           strcmp(corruption, "topic_splice") == 0 ||
           strcmp(corruption, "same_season_splice") == 0 ||
           strcmp(corruption, "plausible_topic_splice") == 0;
}

static int nn_gd_haiku_is_topic_corruption(const char *corruption) {
    return strcmp(corruption, "topic_splice") == 0 ||
           strcmp(corruption, "same_season_splice") == 0 ||
           strcmp(corruption, "plausible_topic_splice") == 0;
}

static int nn_gd_haiku_is_available_selection(const char *selection) {
    return strcmp(selection, "real") == 0 || strcmp(selection, "corrupt") == 0;
}

static double nn_gd_haiku_difficulty_score(const double *metrics,
                                           int neural_failed,
                                           int symbolic_failed) {
    double score = 1.0 - metrics[NN_GD_HAIKU_METRIC_REAL_PROBABILITY];
    if (neural_failed) {
        score += 1.0;
    }
    if (symbolic_failed) {
        score += 0.5;
    }
    if (metrics[NN_GD_HAIKU_METRIC_PLAN_SCORE_DELTA] < 0.0) {
        score += fabs(metrics[NN_GD_HAIKU_METRIC_PLAN_SCORE_DELTA]);
    }
    if (metrics[NN_GD_HAIKU_METRIC_TOPIC_TIGHTNESS_DELTA] < 0.0) {
        score += fabs(metrics[NN_GD_HAIKU_METRIC_TOPIC_TIGHTNESS_DELTA]);
    }
    if (metrics[NN_GD_HAIKU_METRIC_MIN_LINE_SIMILARITY_DELTA] < 0.0) {
        score += fabs(metrics[NN_GD_HAIKU_METRIC_MIN_LINE_SIMILARITY_DELTA]);
    }
    if (metrics[NN_GD_HAIKU_METRIC_CORRUPT_ADJACENCY_RATIO] >= 0.5) {
        score += 0.25 * metrics[NN_GD_HAIKU_METRIC_CORRUPT_ADJACENCY_RATIO];
    }
    if (metrics[NN_GD_HAIKU_METRIC_CORRUPT_BIGRAM_RATIO] >= 0.5) {
        score += 0.25 * metrics[NN_GD_HAIKU_METRIC_CORRUPT_BIGRAM_RATIO];
    }
    if (metrics[NN_GD_HAIKU_METRIC_LOCAL_NEIGHBOR_FIT_DELTA] < 0.0) {
        score += fabs(metrics[NN_GD_HAIKU_METRIC_LOCAL_NEIGHBOR_FIT_DELTA]);
    }
    return score;
}

static int nn_gd_haiku_rows_reserve(NNGDHaikuCoherenceRows *rows, size_t needed) {
    NNGDHaikuCoherenceRow *items;
    size_t capacity;
    if (needed <= rows->capacity) {
        return 0;
    }
    capacity = rows->capacity == 0U ? 64U : rows->capacity * 2U;
    while (capacity < needed) {
        capacity *= 2U;
    }
    items = (NNGDHaikuCoherenceRow *)realloc(rows->items,
                                             capacity * sizeof(*rows->items));
    if (items == NULL) {
        return -1;
    }
    rows->items = items;
    rows->capacity = capacity;
    return 0;
}

static void nn_gd_haiku_rows_free(NNGDHaikuCoherenceRows *rows) {
    size_t i;
    for (i = 0U; i < rows->count; ++i) {
        free(rows->items[i].corruption);
        free(rows->items[i].state_hash);
        free(rows->items[i].adjacency_hash);
        free(rows->items[i].selected);
        free(rows->items[i].symbolic_selected);
    }
    free(rows->items);
    memset(rows, 0, sizeof(*rows));
}

static int nn_gd_read_haiku_coherence_trace_row(const char *line,
                                                NNGDHaikuCoherenceRows *rows) {
    NNGDJsonIndex index;
    NNGDHaikuCoherenceRow *row;
    char *schema = NULL;
    char *corruption = NULL;
    size_t i;
    const char *cursor = nn_gd_json_skip_space(line);
    if (cursor == NULL || *cursor != '{') {
        return 0;
    }
    nn_gd_json_index_line(line, &index);
    schema = nn_gd_json_dup_index_string(&index, "schema");
    if (schema == NULL) {
        return -1;
    }
    if (strcmp(schema, "crlp.haiku_understanding.trace.v1") != 0) {
        free(schema);
        return 0;
    }
    ++rows->source_trace_rows;
    corruption = nn_gd_json_dup_index_string(&index, "corruption");
    if (corruption == NULL) {
        free(schema);
        return -1;
    }
    if (!nn_gd_haiku_is_hard_negative_corruption(corruption)) {
        free(corruption);
        free(schema);
        return 0;
    }
    if (nn_gd_haiku_rows_reserve(rows, rows->count + 1U) != 0) {
        free(corruption);
        free(schema);
        return -1;
    }
    row = &rows->items[rows->count];
    memset(row, 0, sizeof(*row));
    row->corruption = corruption;
    row->selected = nn_gd_json_dup_index_string(&index, "selected");
    row->symbolic_selected = nn_gd_json_dup_index_string(&index, "symbolic_selected");
    row->state_hash = nn_gd_json_dup_index_string(&index, "state_hash");
    row->adjacency_hash = nn_gd_json_dup_index_string(&index, "adjacency_hash");
    if (row->selected == NULL || row->symbolic_selected == NULL ||
        row->state_hash == NULL || row->adjacency_hash == NULL) {
        free(schema);
        return -1;
    }
    row->poem_index = nn_gd_json_index_long_or(&index, "poem_index", 0L);
    row->corrupt_line = nn_gd_json_index_long_or(&index, "corrupt_line", -1L);
    row->source_poem_index =
        nn_gd_json_index_long_or(&index, "source_poem_index", -1L);
    row->source_line = nn_gd_json_index_long_or(&index, "source_line", -1L);
    for (i = 0U; i < (size_t)NN_GD_HAIKU_METRIC_COUNT; ++i) {
        row->metrics[i] = nn_gd_json_index_double_or_zero(
            &index,
            nn_gd_haiku_metric_name((NNGDHaikuMetric)i));
    }
    row->neural_available = nn_gd_haiku_is_available_selection(row->selected);
    row->symbolic_available =
        nn_gd_haiku_is_available_selection(row->symbolic_selected);
    row->neural_failed = row->neural_available && strcmp(row->selected, "real") != 0;
    row->symbolic_failed =
        row->symbolic_available && strcmp(row->symbolic_selected, "real") != 0;
    row->topic_coherence_family = nn_gd_haiku_is_topic_corruption(row->corruption);
    row->plausible_negative = strcmp(row->corruption, "plausible_topic_splice") == 0;
    row->difficulty_score = nn_gd_haiku_difficulty_score(row->metrics,
                                                         row->neural_failed,
                                                         row->symbolic_failed);
    ++rows->count;
    free(schema);
    return 0;
}

static int nn_gd_read_haiku_coherence_trace(const char *trace_path,
                                            NNGDHaikuCoherenceRows *rows) {
    FILE *fp;
    char *line;
    int status = 0;
    fp = fopen(trace_path, "r");
    if (fp == NULL) {
        return -1;
    }
    while ((line = nn_gd_read_line_dynamic(fp)) != NULL) {
        if (nn_gd_read_haiku_coherence_trace_row(line, rows) != 0) {
            status = -1;
            free(line);
            break;
        }
        free(line);
    }
    if (ferror(fp)) {
        status = -1;
    }
    fclose(fp);
    return status;
}

static void nn_gd_write_json_bool(FILE *fp, int value) {
    fputs(value ? "true" : "false", fp);
}

static void nn_gd_write_json_double(FILE *fp, double value) {
    if (!isfinite(value)) {
        fputs("0", fp);
    } else {
        fprintf(fp, "%.12g", value);
    }
}

static void nn_gd_write_json_nullable_ratio(FILE *fp,
                                            size_t numerator,
                                            size_t denominator) {
    if (denominator == 0U) {
        fputs("null", fp);
    } else {
        nn_gd_write_json_double(fp, (double)numerator / (double)denominator);
    }
}

static void nn_gd_write_haiku_metric_pack(FILE *fp,
                                          const NNGDHaikuCoherenceRow *row) {
    size_t i;
    fputc('{', fp);
    for (i = 0U; i < (size_t)NN_GD_HAIKU_METRIC_COUNT; ++i) {
        if (i > 0U) {
            fputc(',', fp);
        }
        nn_gd_json_write_string(fp, nn_gd_haiku_metric_name((NNGDHaikuMetric)i));
        fputc(':', fp);
        nn_gd_write_json_double(fp, row->metrics[i]);
    }
    fputc('}', fp);
}

static int nn_gd_write_haiku_hard_negative_rows(FILE *fp,
                                                const char *trace_path,
                                                const NNGDHaikuCoherenceRows *rows) {
    size_t i;
    for (i = 0U; i < rows->count; ++i) {
        const NNGDHaikuCoherenceRow *row = &rows->items[i];
        fputs("{\"schema\":\"crlp.haiku_coherence_hard_negative.v1\","
              "\"report_version\":1,\"task\":\"haiku_understanding\","
              "\"source_trace\":",
              fp);
        nn_gd_json_write_string(fp, trace_path);
        fputs(",\"source_schema\":\"crlp.haiku_understanding.trace.v1\","
              "\"paper_claim_support\":\"diagnostic_only\","
              "\"promotion_status\":\"diagnostic_only_no_verifier_certification\","
              "\"poem_index\":",
              fp);
        fprintf(fp, "%ld,\"corrupt_line\":%ld,\"source_poem_index\":%ld,"
                    "\"source_line\":%ld,\"corruption\":",
                row->poem_index,
                row->corrupt_line,
                row->source_poem_index,
                row->source_line);
        nn_gd_json_write_string(fp, row->corruption);
        fputs(",\"state_hash\":", fp);
        nn_gd_json_write_string(fp, row->state_hash);
        fputs(",\"adjacency_hash\":", fp);
        nn_gd_json_write_string(fp, row->adjacency_hash);
        fputs(",\"selected\":", fp);
        nn_gd_json_write_string(fp, row->selected);
        fputs(",\"symbolic_selected\":", fp);
        nn_gd_json_write_string(fp, row->symbolic_selected);
        fputs(",\"neural_available\":", fp);
        nn_gd_write_json_bool(fp, row->neural_available);
        fputs(",\"symbolic_available\":", fp);
        nn_gd_write_json_bool(fp, row->symbolic_available);
        fputs(",\"neural_failed\":", fp);
        nn_gd_write_json_bool(fp, row->neural_failed);
        fputs(",\"symbolic_failed\":", fp);
        nn_gd_write_json_bool(fp, row->symbolic_failed);
        fputs(",\"hard_negative\":true,\"topic_coherence_family\":", fp);
        nn_gd_write_json_bool(fp, row->topic_coherence_family);
        fputs(",\"plausible_negative\":", fp);
        nn_gd_write_json_bool(fp, row->plausible_negative);
        fputs(",\"difficulty_score\":", fp);
        nn_gd_write_json_double(fp, row->difficulty_score);
        fputs(",\"metrics\":", fp);
        nn_gd_write_haiku_metric_pack(fp, row);
        fputs("}\n", fp);
    }
    return ferror(fp) ? -1 : 0;
}

static void nn_gd_haiku_corruption_summary(const NNGDHaikuCoherenceRows *rows,
                                           const char *corruption,
                                           NNGDHaikuCorruptionSummary *summary) {
    size_t i;
    size_t metric;
    memset(summary, 0, sizeof(*summary));
    summary->corruption = corruption;
    for (i = 0U; i < rows->count; ++i) {
        const NNGDHaikuCoherenceRow *row = &rows->items[i];
        if (strcmp(row->corruption, corruption) != 0) {
            continue;
        }
        ++summary->rows;
        if (row->neural_available) {
            ++summary->neural_available_rows;
        }
        if (row->symbolic_available) {
            ++summary->symbolic_available_rows;
        }
        if (row->neural_failed) {
            ++summary->neural_failure_rows;
        }
        if (row->symbolic_failed) {
            ++summary->symbolic_failure_rows;
        }
        for (metric = 0U; metric < (size_t)NN_GD_HAIKU_METRIC_COUNT; ++metric) {
            summary->metric_sums[metric] += row->metrics[metric];
        }
    }
    if (summary->neural_available_rows > 0U) {
        summary->top1_available = 1;
        summary->top1 = (double)(summary->neural_available_rows -
                                 summary->neural_failure_rows) /
                        (double)summary->neural_available_rows;
    }
    if (summary->symbolic_available_rows > 0U) {
        summary->symbolic_top1_available = 1;
        summary->symbolic_top1 =
            (double)(summary->symbolic_available_rows -
                     summary->symbolic_failure_rows) /
            (double)summary->symbolic_available_rows;
    }
}

static void nn_gd_write_haiku_corruption_summary_object(
    FILE *fp,
    const NNGDHaikuCorruptionSummary *summary) {
    fputs("{\"rows\":", fp);
    fprintf(fp, "%zu,\"neural_available_rows\":%zu,\"symbolic_available_rows\":%zu,"
                "\"top1\":",
            summary->rows,
            summary->neural_available_rows,
            summary->symbolic_available_rows);
    if (summary->top1_available) {
        nn_gd_write_json_double(fp, summary->top1);
    } else {
        fputs("null", fp);
    }
    fputs(",\"symbolic_top1\":", fp);
    if (summary->symbolic_top1_available) {
        nn_gd_write_json_double(fp, summary->symbolic_top1);
    } else {
        fputs("null", fp);
    }
    fprintf(fp,
            ",\"neural_failure_rows\":%zu,\"symbolic_failure_rows\":%zu,"
            "\"avg_real_probability\":",
            summary->neural_failure_rows,
            summary->symbolic_failure_rows);
    nn_gd_write_json_double(fp,
                            summary->rows == 0U
                                ? 0.0
                                : summary->metric_sums[NN_GD_HAIKU_METRIC_REAL_PROBABILITY] /
                                      (double)summary->rows);
    fputs(",\"avg_plan_score_delta\":", fp);
    nn_gd_write_json_double(
        fp,
        summary->rows == 0U
            ? 0.0
            : summary->metric_sums[NN_GD_HAIKU_METRIC_PLAN_SCORE_DELTA] /
                  (double)summary->rows);
    fputs(",\"avg_topic_tightness_delta\":", fp);
    nn_gd_write_json_double(
        fp,
        summary->rows == 0U
            ? 0.0
            : summary->metric_sums[NN_GD_HAIKU_METRIC_TOPIC_TIGHTNESS_DELTA] /
                  (double)summary->rows);
    fputs(",\"avg_min_line_similarity_delta\":", fp);
    nn_gd_write_json_double(
        fp,
        summary->rows == 0U
            ? 0.0
            : summary->metric_sums[NN_GD_HAIKU_METRIC_MIN_LINE_SIMILARITY_DELTA] /
                  (double)summary->rows);
    fputs(",\"avg_corrupt_adjacency_ratio\":", fp);
    nn_gd_write_json_double(
        fp,
        summary->rows == 0U
            ? 0.0
            : summary->metric_sums[NN_GD_HAIKU_METRIC_CORRUPT_ADJACENCY_RATIO] /
                  (double)summary->rows);
    fputs(",\"avg_corrupt_bigram_ratio\":", fp);
    nn_gd_write_json_double(
        fp,
        summary->rows == 0U
            ? 0.0
            : summary->metric_sums[NN_GD_HAIKU_METRIC_CORRUPT_BIGRAM_RATIO] /
                  (double)summary->rows);
    fputs(",\"avg_replacement_line_similarity\":", fp);
    nn_gd_write_json_double(
        fp,
        summary->rows == 0U
            ? 0.0
            : summary->metric_sums[NN_GD_HAIKU_METRIC_REPLACEMENT_LINE_SIMILARITY] /
                  (double)summary->rows);
    fputs(",\"avg_local_neighbor_fit_delta\":", fp);
    nn_gd_write_json_double(
        fp,
        summary->rows == 0U
            ? 0.0
            : summary->metric_sums[NN_GD_HAIKU_METRIC_LOCAL_NEIGHBOR_FIT_DELTA] /
                  (double)summary->rows);
    fputs(",\"avg_line_word_delta\":", fp);
    nn_gd_write_json_double(
        fp,
        summary->rows == 0U
            ? 0.0
            : summary->metric_sums[NN_GD_HAIKU_METRIC_LINE_WORD_DELTA] /
                  (double)summary->rows);
    fputs(",\"avg_line_syllable_delta\":", fp);
    nn_gd_write_json_double(
        fp,
        summary->rows == 0U
            ? 0.0
            : summary->metric_sums[NN_GD_HAIKU_METRIC_LINE_SYLLABLE_DELTA] /
                  (double)summary->rows);
    fputc('}', fp);
}

static int nn_gd_haiku_row_in_diagnostic_split(const NNGDHaikuCoherenceRow *row,
                                                size_t split_index) {
    if (split_index == 0U) {
        return strcmp(row->corruption, "line_splice") == 0 &&
               row->metrics[NN_GD_HAIKU_METRIC_LOCAL_NEIGHBOR_FIT_DELTA] > 0.0;
    }
    if (split_index == 1U) {
        return strcmp(row->corruption, "line_splice") == 0 &&
               row->metrics[NN_GD_HAIKU_METRIC_LOCAL_NEIGHBOR_FIT_DELTA] <= 0.0;
    }
    if (split_index == 2U) {
        return strcmp(row->corruption, "word_salad") == 0 &&
               row->metrics[NN_GD_HAIKU_METRIC_CORRUPT_ADJACENCY_RATIO] >= 0.5;
    }
    return strcmp(row->corruption, "word_salad") == 0 &&
           row->metrics[NN_GD_HAIKU_METRIC_CORRUPT_ADJACENCY_RATIO] < 0.5;
}

static void nn_gd_write_haiku_diagnostic_splits(
    FILE *fp,
    const NNGDHaikuCoherenceRows *rows) {
    static const char *corruptions[] = {
        "line_splice",
        "line_splice",
        "word_salad",
        "word_salad",
    };
    static const char *splits[] = {
        "local_neighbor_fit_delta > 0",
        "local_neighbor_fit_delta <= 0",
        "corrupt_adjacency_ratio >= 0.5",
        "corrupt_adjacency_ratio < 0.5",
    };
    size_t i;
    fputc('[', fp);
    for (i = 0U; i < 4U; ++i) {
        size_t row_index;
        size_t split_rows = 0U;
        size_t available = 0U;
        size_t failures = 0U;
        if (i > 0U) {
            fputc(',', fp);
        }
        for (row_index = 0U; row_index < rows->count; ++row_index) {
            const NNGDHaikuCoherenceRow *row = &rows->items[row_index];
            if (!nn_gd_haiku_row_in_diagnostic_split(row, i)) {
                continue;
            }
            ++split_rows;
            if (row->neural_available) {
                ++available;
            }
            if (row->neural_failed) {
                ++failures;
            }
        }
        fputs("{\"corruption\":", fp);
        nn_gd_json_write_string(fp, corruptions[i]);
        fputs(",\"split\":", fp);
        nn_gd_json_write_string(fp, splits[i]);
        fprintf(fp, ",\"rows\":%zu,\"neural_available_rows\":%zu,"
                    "\"top1\":",
                split_rows,
                available);
        nn_gd_write_json_nullable_ratio(fp, available - failures, available);
        fprintf(fp, ",\"neural_failure_rows\":%zu}", failures);
    }
    fputc(']', fp);
}

static int nn_gd_haiku_summary_better(const NNGDHaikuCorruptionSummary *candidate,
                                      const NNGDHaikuCorruptionSummary *best) {
    double candidate_top1 = candidate->top1_available ? candidate->top1 : 1.0;
    double best_top1 = best->top1_available ? best->top1 : 1.0;
    if (candidate_top1 < best_top1) {
        return 1;
    }
    if (candidate_top1 > best_top1) {
        return 0;
    }
    if (candidate->neural_failure_rows > best->neural_failure_rows) {
        return 1;
    }
    if (candidate->neural_failure_rows < best->neural_failure_rows) {
        return 0;
    }
    return strcmp(candidate->corruption, best->corruption) < 0;
}

static int nn_gd_haiku_row_failure_ref_better(const NNGDHaikuCoherenceRow *candidate,
                                               const NNGDHaikuCoherenceRow *best) {
    if (candidate->difficulty_score > best->difficulty_score) {
        return 1;
    }
    if (candidate->difficulty_score < best->difficulty_score) {
        return 0;
    }
    return strcmp(candidate->corruption, best->corruption) < 0;
}

static int nn_gd_index_selected(const size_t *indices, size_t count, size_t value) {
    size_t i;
    for (i = 0U; i < count; ++i) {
        if (indices[i] == value) {
            return 1;
        }
    }
    return 0;
}

static void nn_gd_write_haiku_top_failure_refs(
    FILE *fp,
    const NNGDHaikuCoherenceRows *rows) {
    size_t selected[16];
    size_t selected_count = 0U;
    fputc('[', fp);
    while (selected_count < 16U) {
        size_t i;
        size_t best_index = 0U;
        int found = 0;
        for (i = 0U; i < rows->count; ++i) {
            const NNGDHaikuCoherenceRow *row = &rows->items[i];
            if (!(row->neural_failed || row->symbolic_failed) ||
                nn_gd_index_selected(selected, selected_count, i)) {
                continue;
            }
            if (!found ||
                nn_gd_haiku_row_failure_ref_better(row, &rows->items[best_index])) {
                best_index = i;
                found = 1;
            }
        }
        if (!found) {
            break;
        }
        if (selected_count > 0U) {
            fputc(',', fp);
        }
        selected[selected_count++] = best_index;
        {
            const NNGDHaikuCoherenceRow *row = &rows->items[best_index];
            fprintf(fp, "{\"poem_index\":%ld,\"corruption\":",
                    row->poem_index);
            nn_gd_json_write_string(fp, row->corruption);
            fprintf(fp,
                    ",\"corrupt_line\":%ld,\"source_poem_index\":%ld,"
                    "\"source_line\":%ld,\"state_hash\":",
                    row->corrupt_line,
                    row->source_poem_index,
                    row->source_line);
            nn_gd_json_write_string(fp, row->state_hash);
            fputs(",\"selected\":", fp);
            nn_gd_json_write_string(fp, row->selected);
            fputs(",\"symbolic_selected\":", fp);
            nn_gd_json_write_string(fp, row->symbolic_selected);
            fputs(",\"difficulty_score\":", fp);
            nn_gd_write_json_double(fp, row->difficulty_score);
            fputs(",\"metrics\":", fp);
            nn_gd_write_haiku_metric_pack(fp, row);
            fputc('}', fp);
        }
    }
    fputc(']', fp);
}

static int nn_gd_write_haiku_summary_jsonl(
    FILE *fp,
    const char *trace_path,
    const NNGDHaikuCoherenceRows *rows) {
    static const char *corruptions[] = {
        "line_splice",
        "plausible_topic_splice",
        "same_season_splice",
        "topic_splice",
        "word_salad",
    };
    NNGDHaikuCorruptionSummary summaries[5];
    size_t i;
    size_t neural_available = 0U;
    size_t neural_failures = 0U;
    size_t symbolic_available = 0U;
    size_t symbolic_failures = 0U;
    size_t joint_failures = 0U;
    size_t topic_rows = 0U;
    size_t topic_available = 0U;
    size_t topic_failures = 0U;
    size_t topic_symbolic_available = 0U;
    size_t topic_symbolic_failures = 0U;
    size_t plausible_rows = 0U;
    size_t plausible_available = 0U;
    size_t plausible_failures = 0U;
    for (i = 0U; i < 5U; ++i) {
        nn_gd_haiku_corruption_summary(rows, corruptions[i], &summaries[i]);
    }
    for (i = 0U; i < rows->count; ++i) {
        const NNGDHaikuCoherenceRow *row = &rows->items[i];
        if (row->neural_available) {
            ++neural_available;
        }
        if (row->neural_failed) {
            ++neural_failures;
        }
        if (row->symbolic_available) {
            ++symbolic_available;
        }
        if (row->symbolic_failed) {
            ++symbolic_failures;
        }
        if (row->neural_failed && row->symbolic_failed) {
            ++joint_failures;
        }
        if (row->topic_coherence_family) {
            ++topic_rows;
            if (row->neural_available) {
                ++topic_available;
            }
            if (row->neural_failed) {
                ++topic_failures;
            }
            if (row->symbolic_available) {
                ++topic_symbolic_available;
            }
            if (row->symbolic_failed) {
                ++topic_symbolic_failures;
            }
        }
        if (row->plausible_negative) {
            ++plausible_rows;
            if (row->neural_available) {
                ++plausible_available;
            }
            if (row->neural_failed) {
                ++plausible_failures;
            }
        }
    }
    fputs("{\"schema\":\"crlp.haiku_coherence_hard_negative_summary.v1\","
          "\"report_version\":1,\"source_trace\":",
          fp);
    nn_gd_json_write_string(fp, trace_path);
    fprintf(fp,
            ",\"source_trace_rows\":%zu,\"hard_negative_rows\":%zu,"
            "\"paper_claim_support\":\"diagnostic_only\","
            "\"promotion_status\":\"diagnostic_only_no_verifier_certification\","
            "\"hard_negative_top1\":",
            rows->source_trace_rows,
            rows->count);
    nn_gd_write_json_nullable_ratio(fp, neural_available - neural_failures, neural_available);
    fputs(",\"hard_negative_symbolic_top1\":", fp);
    nn_gd_write_json_nullable_ratio(fp,
                                    symbolic_available - symbolic_failures,
                                    symbolic_available);
    fprintf(fp, ",\"topic_coherence_rows\":%zu,\"topic_coherence_top1\":", topic_rows);
    nn_gd_write_json_nullable_ratio(fp, topic_available - topic_failures, topic_available);
    fputs(",\"topic_coherence_symbolic_top1\":", fp);
    nn_gd_write_json_nullable_ratio(fp,
                                    topic_symbolic_available - topic_symbolic_failures,
                                    topic_symbolic_available);
    fprintf(fp,
            ",\"plausible_negative_rows\":%zu,\"plausible_negative_top1\":",
            plausible_rows);
    nn_gd_write_json_nullable_ratio(fp,
                                    plausible_available - plausible_failures,
                                    plausible_available);
    fprintf(fp,
            ",\"neural_failure_rows\":%zu,\"symbolic_failure_rows\":%zu,"
            "\"joint_failure_rows\":%zu,\"corruption_counts\":{",
            neural_failures,
            symbolic_failures,
            joint_failures);
    {
        int first = 1;
        for (i = 0U; i < 5U; ++i) {
            if (summaries[i].rows == 0U) {
                continue;
            }
            if (!first) {
                fputc(',', fp);
            }
            first = 0;
            nn_gd_json_write_string(fp, summaries[i].corruption);
            fprintf(fp, ":%zu", summaries[i].rows);
        }
    }
    fputs("},\"by_corruption\":{", fp);
    {
        int first = 1;
        for (i = 0U; i < 5U; ++i) {
            if (summaries[i].rows == 0U) {
                continue;
            }
            if (!first) {
                fputc(',', fp);
            }
            first = 0;
            nn_gd_json_write_string(fp, summaries[i].corruption);
            fputc(':', fp);
            nn_gd_write_haiku_corruption_summary_object(fp, &summaries[i]);
        }
    }
    fputs("},\"diagnostic_splits\":", fp);
    nn_gd_write_haiku_diagnostic_splits(fp, rows);
    fputs(",\"weakest_corruptions\":[", fp);
    {
        size_t selected[5];
        size_t selected_count = 0U;
        while (selected_count < 5U) {
            size_t best = 0U;
            int found = 0;
            for (i = 0U; i < 5U; ++i) {
                if (summaries[i].rows == 0U ||
                    nn_gd_index_selected(selected, selected_count, i)) {
                    continue;
                }
                if (!found || nn_gd_haiku_summary_better(&summaries[i], &summaries[best])) {
                    best = i;
                    found = 1;
                }
            }
            if (!found) {
                break;
            }
            if (selected_count > 0U) {
                fputc(',', fp);
            }
            selected[selected_count++] = best;
            fputs("{\"corruption\":", fp);
            nn_gd_json_write_string(fp, summaries[best].corruption);
            fprintf(fp, ",\"rows\":%zu,\"top1\":", summaries[best].rows);
            if (summaries[best].top1_available) {
                nn_gd_write_json_double(fp, summaries[best].top1);
            } else {
                fputs("null", fp);
            }
            fprintf(fp,
                    ",\"neural_failure_rows\":%zu,"
                    "\"symbolic_failure_rows\":%zu}",
                    summaries[best].neural_failure_rows,
                    summaries[best].symbolic_failure_rows);
        }
    }
    fputs("],\"top_failure_refs\":", fp);
    nn_gd_write_haiku_top_failure_refs(fp, rows);
    fputs("}\n", fp);
    return ferror(fp) ? -1 : 0;
}

static void nn_gd_write_pct(FILE *fp, size_t numerator, size_t denominator) {
    if (denominator == 0U) {
        fputs("n/a", fp);
    } else {
        fprintf(fp, "%.2f%%", 100.0 * (double)numerator / (double)denominator);
    }
}

static void nn_gd_write_pct_from_summary(FILE *fp,
                                          const NNGDHaikuCorruptionSummary *summary,
                                          int symbolic) {
    if (symbolic) {
        if (!summary->symbolic_top1_available) {
            fputs("n/a", fp);
        } else {
            fprintf(fp, "%.2f%%", 100.0 * summary->symbolic_top1);
        }
    } else if (!summary->top1_available) {
        fputs("n/a", fp);
    } else {
        fprintf(fp, "%.2f%%", 100.0 * summary->top1);
    }
}

static void nn_gd_write_avg_metric(FILE *fp,
                                    const NNGDHaikuCorruptionSummary *summary,
                                    NNGDHaikuMetric metric) {
    if (summary->rows == 0U) {
        fputs("n/a", fp);
    } else {
        fprintf(fp, "%.4f", summary->metric_sums[metric] / (double)summary->rows);
    }
}

static int nn_gd_write_haiku_summary_markdown(
    FILE *fp,
    const char *trace_path,
    const NNGDHaikuCoherenceRows *rows) {
    static const char *corruptions[] = {
        "line_splice",
        "plausible_topic_splice",
        "same_season_splice",
        "topic_splice",
        "word_salad",
    };
    NNGDHaikuCorruptionSummary summaries[5];
    size_t i;
    size_t neural_available = 0U;
    size_t neural_failures = 0U;
    size_t topic_available = 0U;
    size_t topic_failures = 0U;
    size_t plausible_available = 0U;
    size_t plausible_failures = 0U;
    for (i = 0U; i < 5U; ++i) {
        nn_gd_haiku_corruption_summary(rows, corruptions[i], &summaries[i]);
    }
    for (i = 0U; i < rows->count; ++i) {
        const NNGDHaikuCoherenceRow *row = &rows->items[i];
        if (row->neural_available) {
            ++neural_available;
        }
        if (row->neural_failed) {
            ++neural_failures;
        }
        if (row->topic_coherence_family) {
            if (row->neural_available) {
                ++topic_available;
            }
            if (row->neural_failed) {
                ++topic_failures;
            }
        }
        if (row->plausible_negative) {
            if (row->neural_available) {
                ++plausible_available;
            }
            if (row->neural_failed) {
                ++plausible_failures;
            }
        }
    }
    fputs("# Haiku Coherence Hard Negatives\n\nSource trace: `", fp);
    fputs(trace_path, fp);
    fputs("`\n\n- Paper-claim support: `diagnostic_only`\n"
          "- Promotion status: `diagnostic_only_no_verifier_certification`\n"
          "- Hard-negative rows: `",
          fp);
    fprintf(fp, "%zu`\n- Hard-negative top-1: `", rows->count);
    nn_gd_write_pct(fp, neural_available - neural_failures, neural_available);
    fputs("`\n- Topic-coherence top-1: `", fp);
    nn_gd_write_pct(fp, topic_available - topic_failures, topic_available);
    fputs("`\n- Plausible-negative top-1: `", fp);
    nn_gd_write_pct(fp, plausible_available - plausible_failures, plausible_available);
    fputs("`\n\n| corruption | rows | top1 | symbolic top1 | neural failures | "
          "symbolic failures | avg real p | avg plan delta | avg topic delta | "
          "avg local delta | avg repl sim |\n"
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n",
          fp);
    for (i = 0U; i < 5U; ++i) {
        if (summaries[i].rows == 0U) {
            continue;
        }
        fprintf(fp, "| %s | %zu | ", summaries[i].corruption, summaries[i].rows);
        nn_gd_write_pct_from_summary(fp, &summaries[i], 0);
        fputs(" | ", fp);
        nn_gd_write_pct_from_summary(fp, &summaries[i], 1);
        fprintf(fp,
                " | %zu | %zu | ",
                summaries[i].neural_failure_rows,
                summaries[i].symbolic_failure_rows);
        nn_gd_write_avg_metric(fp, &summaries[i], NN_GD_HAIKU_METRIC_REAL_PROBABILITY);
        fputs(" | ", fp);
        nn_gd_write_avg_metric(fp, &summaries[i], NN_GD_HAIKU_METRIC_PLAN_SCORE_DELTA);
        fputs(" | ", fp);
        nn_gd_write_avg_metric(fp, &summaries[i], NN_GD_HAIKU_METRIC_TOPIC_TIGHTNESS_DELTA);
        fputs(" | ", fp);
        nn_gd_write_avg_metric(fp, &summaries[i], NN_GD_HAIKU_METRIC_LOCAL_NEIGHBOR_FIT_DELTA);
        fputs(" | ", fp);
        nn_gd_write_avg_metric(fp,
                               &summaries[i],
                               NN_GD_HAIKU_METRIC_REPLACEMENT_LINE_SIMILARITY);
        fputs(" |\n", fp);
    }
    fputs("\n## Diagnostic Splits\n\n"
          "| corruption | split | rows | top1 | neural failures |\n"
          "| --- | --- | --- | --- | --- |\n",
          fp);
    for (i = 0U; i < 4U; ++i) {
        static const char *split_corruptions[] = {
            "line_splice",
            "line_splice",
            "word_salad",
            "word_salad",
        };
        static const char *split_names[] = {
            "local_neighbor_fit_delta > 0",
            "local_neighbor_fit_delta <= 0",
            "corrupt_adjacency_ratio >= 0.5",
            "corrupt_adjacency_ratio < 0.5",
        };
        size_t row_index;
        size_t split_rows = 0U;
        size_t available = 0U;
        size_t failures = 0U;
        for (row_index = 0U; row_index < rows->count; ++row_index) {
            const NNGDHaikuCoherenceRow *row = &rows->items[row_index];
            if (!nn_gd_haiku_row_in_diagnostic_split(row, i)) {
                continue;
            }
            ++split_rows;
            if (row->neural_available) {
                ++available;
            }
            if (row->neural_failed) {
                ++failures;
            }
        }
        fprintf(fp, "| %s | %s | %zu | ", split_corruptions[i], split_names[i], split_rows);
        nn_gd_write_pct(fp, available - failures, available);
        fprintf(fp, " | %zu |\n", failures);
    }
    fputs("\n## Top Failure Refs\n\n", fp);
    if (rows->count == 0U) {
        fputs("No hard-negative failures were found in this trace.\n", fp);
    } else {
        size_t selected[16];
        size_t selected_count = 0U;
        fputs("| poem | corruption | line | source | selected | symbolic | difficulty | state |\n"
              "| --- | --- | --- | --- | --- | --- | --- | --- |\n",
              fp);
        while (selected_count < 16U) {
            size_t row_index;
            size_t best_index = 0U;
            int found = 0;
            for (row_index = 0U; row_index < rows->count; ++row_index) {
                const NNGDHaikuCoherenceRow *row = &rows->items[row_index];
                if (!(row->neural_failed || row->symbolic_failed) ||
                    nn_gd_index_selected(selected, selected_count, row_index)) {
                    continue;
                }
                if (!found ||
                    nn_gd_haiku_row_failure_ref_better(row, &rows->items[best_index])) {
                    best_index = row_index;
                    found = 1;
                }
            }
            if (!found) {
                break;
            }
            selected[selected_count++] = best_index;
            {
                const NNGDHaikuCoherenceRow *row = &rows->items[best_index];
                fprintf(fp,
                        "| %ld | %s | %ld | %ld:%ld | %s | %s | %.4f | %s |\n",
                        row->poem_index,
                        row->corruption,
                        row->corrupt_line,
                        row->source_poem_index,
                        row->source_line,
                        row->selected,
                        row->symbolic_selected,
                        row->difficulty_score,
                        row->state_hash);
            }
        }
        if (selected_count == 0U) {
            fputs("No hard-negative failures were found in this trace.\n", fp);
        }
    }
    return ferror(fp) ? -1 : 0;
}

int nn_grammar_discovery_write_haiku_coherence_hard_negatives(
    const char *trace_path,
    const char *output_dir) {
    NNGDHaikuCoherenceRows rows = {0};
    FILE *hard_rows = NULL;
    FILE *summary = NULL;
    FILE *markdown = NULL;
    char hard_rows_path[4096];
    char summary_path[4096];
    char markdown_path[4096];
    int written;
    int status = -1;
    if (trace_path == NULL || trace_path[0] == '\0' ||
        output_dir == NULL || output_dir[0] == '\0') {
        return -1;
    }
    written = snprintf(hard_rows_path,
                       sizeof(hard_rows_path),
                       "%s/hard_negative_rows.jsonl",
                       output_dir);
    if (written < 0 || (size_t)written >= sizeof(hard_rows_path)) {
        return -1;
    }
    written = snprintf(summary_path,
                       sizeof(summary_path),
                       "%s/summary.jsonl",
                       output_dir);
    if (written < 0 || (size_t)written >= sizeof(summary_path)) {
        return -1;
    }
    written = snprintf(markdown_path,
                       sizeof(markdown_path),
                       "%s/summary.md",
                       output_dir);
    if (written < 0 || (size_t)written >= sizeof(markdown_path)) {
        return -1;
    }
    if (nn_gd_read_haiku_coherence_trace(trace_path, &rows) != 0) {
        goto done;
    }
    hard_rows = fopen(hard_rows_path, "w");
    if (hard_rows == NULL) {
        goto done;
    }
    if (nn_gd_write_haiku_hard_negative_rows(hard_rows, trace_path, &rows) != 0) {
        goto done;
    }
    summary = fopen(summary_path, "w");
    if (summary == NULL) {
        goto done;
    }
    if (nn_gd_write_haiku_summary_jsonl(summary, trace_path, &rows) != 0) {
        goto done;
    }
    markdown = fopen(markdown_path, "w");
    if (markdown == NULL) {
        goto done;
    }
    if (nn_gd_write_haiku_summary_markdown(markdown, trace_path, &rows) != 0) {
        goto done;
    }
    printf("wrote %zu hard-negative rows to %s from %s\n",
           rows.count,
           output_dir,
           trace_path);
    status = 0;
