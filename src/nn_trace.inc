static int trace_write_json_string(FILE *fp, const char *text)
{
    return nn_write_json_string(fp, text);
}

static int trace_write_real_array(FILE *fp, const nn_real *values, size_t count)
{
    if (fputc('[', fp) == EOF) {
        return 0;
    }
    for (size_t i = 0; i < count; ++i) {
        if (i != 0 && fputc(',', fp) == EOF) {
            return 0;
        }
#if NN_REAL_IS_FLOAT
        if (fprintf(fp, "%.9g", (double)values[i]) < 0) {
            return 0;
        }
#else
        if (fprintf(fp, "%.17g", (double)values[i]) < 0) {
            return 0;
        }
#endif
    }
    return fputc(']', fp) != EOF;
}

static uint32_t trace_action_id_at(const NNTraceRecord *record, size_t index)
{
    return record->action_ids == NULL ? (uint32_t)index : record->action_ids[index];
}

static int trace_target_indices_are_valid(const NNTraceRecord *record)
{
    if (record->target_indices == NULL) {
        return record->target_count == 0 || record->target_count == 1;
    }
    if (record->target_count == 0) {
        return 0;
    }
    int saw_primary = 0;
    for (size_t i = 0; i < record->target_count; ++i) {
        size_t index = record->target_indices[i];
        if (index >= record->candidate_count) {
            return 0;
        }
        saw_primary = saw_primary || index == record->target_index;
        for (size_t j = i + 1; j < record->target_count; ++j) {
            if (index == record->target_indices[j]) {
                return 0;
            }
        }
    }
    return saw_primary;
}

static int trace_record_arrays_are_valid(const NNTraceRecord *record)
{
    double probability_sum = 0.0;
    for (size_t i = 0; i < record->candidate_count; ++i) {
        if (record->action_ids == NULL && i > UINT32_MAX) {
            return 0;
        }
        uint32_t action_id = trace_action_id_at(record, i);
        for (size_t j = i + 1; j < record->candidate_count; ++j) {
            if (action_id == trace_action_id_at(record, j)) {
                return 0;
            }
        }
        if (!isfinite((double)record->scores[i]) ||
            !isfinite((double)record->probabilities[i]) ||
            record->probabilities[i] < (nn_real)0 ||
            record->probabilities[i] > (nn_real)1) {
            return 0;
        }
        probability_sum += (double)record->probabilities[i];
    }
    return fabs(probability_sum - 1.0) <= 1e-3;
}

int nn_trace_write_jsonl(FILE *fp, const NNTraceRecord *record)
{
    if (fp == NULL || record == NULL || record->candidate_count == 0 ||
        record->scores == NULL || record->probabilities == NULL ||
        record->selected_index >= record->candidate_count ||
        record->target_index >= record->candidate_count ||
        !trace_target_indices_are_valid(record) ||
        !trace_record_arrays_are_valid(record)) {
        return 0;
    }

    const char *task = record->task_name == NULL ? "unknown" : record->task_name;
    const char *model_hash =
        record->model_hash_hex == NULL ? "unknown" : record->model_hash_hex;
    const char *verifier_result =
        record->verifier_result == NULL ? "unknown" : record->verifier_result;
    const char *verifier_cost =
        record->verifier_cost_json == NULL ? "{}" : record->verifier_cost_json;
    uint32_t selected_action_id = trace_action_id_at(record, record->selected_index);
    uint32_t target_action_id = trace_action_id_at(record, record->target_index);
    size_t target_count = record->target_indices == NULL ? 1u : record->target_count;
    nn_real selected_score_margin =
        nn_action_score_margin(record->scores, record->candidate_count, record->selected_index);
    if (!isfinite((double)selected_score_margin)) {
        return 0;
    }
    size_t selected_by_policy = 0;
    for (size_t i = 1; i < record->candidate_count; ++i) {
        if (rank_score_is_better(record->scores,
                                 record->action_ids,
                                 i,
                                 selected_by_policy,
                                 NN_RANK_SCORE_EPSILON)) {
            selected_by_policy = i;
        }
    }
    if (selected_by_policy != record->selected_index) {
        return 0;
    }
    int rank_fp_sensitive = fabs((double)selected_score_margin) <=
                            (double)NN_RANK_SCORE_EPSILON;

    if (fputc('{', fp) == EOF || fputs("\"schema\":", fp) == EOF ||
        !trace_write_json_string(fp, NN_TRACE_SCHEMA) ||
        fprintf(fp, ",\"trace_version\":%u", (unsigned int)NN_TRACE_VERSION) < 0 ||
        fputs(",\"action_id_contract\":", fp) == EOF ||
        !trace_write_json_string(fp, NN_TRACE_ACTION_ID_CONTRACT) ||
        fputs(",\"task\":", fp) == EOF || !trace_write_json_string(fp, task) ||
        fputs(",\"state_hash\":\"", fp) == EOF ||
        fprintf(fp, "%016llx", (unsigned long long)record->state_hash) < 0 ||
        fputs("\",\"model_hash\":", fp) == EOF ||
        !trace_write_json_string(fp, model_hash) ||
        fprintf(fp,
                ",\"feature_encoder_version\":%u,\"candidate_count\":%zu,"
                "\"action_ids\":[",
                record->feature_encoder_version,
                record->candidate_count) < 0) {
        return 0;
    }

    for (size_t i = 0; i < record->candidate_count; ++i) {
        uint32_t action_id = record->action_ids == NULL ? (uint32_t)i : record->action_ids[i];
        if ((i != 0 && fputc(',', fp) == EOF) || fprintf(fp, "%u", action_id) < 0) {
            return 0;
        }
    }

    if (fputs("],\"scores\":", fp) == EOF ||
        !trace_write_real_array(fp, record->scores, record->candidate_count) ||
        fputs(",\"probabilities\":", fp) == EOF ||
        !trace_write_real_array(fp, record->probabilities, record->candidate_count)) {
        return 0;
    }

    if (fprintf(fp, ",\"target_count\":%zu,\"target_indices\":[", target_count) < 0) {
        return 0;
    }
    for (size_t i = 0; i < target_count; ++i) {
        size_t index = record->target_indices == NULL ? record->target_index
                                                      : record->target_indices[i];
        if ((i != 0 && fputc(',', fp) == EOF) || fprintf(fp, "%zu", index) < 0) {
            return 0;
        }
    }
    if (fputs("],\"target_action_ids\":[", fp) == EOF) {
        return 0;
    }
    for (size_t i = 0; i < target_count; ++i) {
        size_t index = record->target_indices == NULL ? record->target_index
                                                      : record->target_indices[i];
        if ((i != 0 && fputc(',', fp) == EOF) ||
            fprintf(fp, "%u", trace_action_id_at(record, index)) < 0) {
            return 0;
        }
    }
    if (fputc(']', fp) == EOF) {
        return 0;
    }

#if NN_REAL_IS_FLOAT
    const char *loss_format =
        ",\"selected_index\":%zu,\"target_index\":%zu,\"loss\":%.9g";
    const char *rank_guard_format =
        ",\"selected_action_id\":%u,\"target_action_id\":%u,"
        "\"selected_score_margin\":%.9g,\"rank_score_epsilon\":%.9g,";
#else
    const char *loss_format =
        ",\"selected_index\":%zu,\"target_index\":%zu,\"loss\":%.17g";
    const char *rank_guard_format =
        ",\"selected_action_id\":%u,\"target_action_id\":%u,"
        "\"selected_score_margin\":%.17g,\"rank_score_epsilon\":%.17g,";
#endif
    if (fprintf(fp,
                loss_format,
                record->selected_index,
                record->target_index,
                (double)record->loss) < 0 ||
        fprintf(fp,
                rank_guard_format,
                selected_action_id,
                target_action_id,
                (double)selected_score_margin,
                (double)NN_RANK_SCORE_EPSILON) < 0 ||
        fprintf(fp,
                "\"rank_fp_sensitive\":%s,\"ranking_policy\":",
                rank_fp_sensitive ? "true" : "false") < 0 ||
        !trace_write_json_string(fp, NN_RANKING_POLICY) ||
        fputs(",\"verifier_result\":", fp) == EOF ||
        !trace_write_json_string(fp, verifier_result) ||
        fputs(",\"verifier_cost\":", fp) == EOF || fputs(verifier_cost, fp) == EOF ||
        fprintf(fp, ",\"seed\":%llu", (unsigned long long)record->seed) < 0) {
        return 0;
    }

    if (record->extra_json != NULL && record->extra_json[0] != '\0') {
        if (fputc(',', fp) == EOF || fputs(record->extra_json, fp) == EOF) {
            return 0;
        }
    }

    /* Output feature vector if available */
    if (record->features != NULL && record->feature_count > 0) {
        if (fputs(",\"features\":[", fp) == EOF) return 0;
        for (size_t i = 0; i < record->feature_count; i++) {
            if (i > 0 && fputc(',', fp) == EOF) return 0;
            if (fprintf(fp, "%.9g", (double)record->features[i]) < 0) return 0;
        }
        if (fputc(']', fp) == EOF) return 0;
    }

    return fputs("}\n", fp) != EOF;
}

static const char *trace_find_key_value(const char *line, const char *key)
{
    if (line == NULL || key == NULL || key[0] == '\0') {
        return NULL;
    }

    char pattern[96];
    int written = snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    if (written <= 0 || (size_t)written >= sizeof(pattern)) {
        return NULL;
    }

    const char *start = strstr(line, pattern);
    if (start == NULL) {
        return NULL;
    }
    return start + (size_t)written;
}

int nn_trace_json_get_string(const char *line,
                             const char *key,
                             char *out_value,
                             size_t out_value_size)
{
    const char *start = trace_find_key_value(line, key);
    if (start == NULL || out_value == NULL || out_value_size == 0 || *start != '"') {
        return 0;
    }
    ++start;

    size_t output = 0;
    for (const char *p = start; *p != '\0'; ++p) {
        unsigned char ch = (unsigned char)*p;
        if (ch == '"') {
            out_value[output] = '\0';
            return 1;
        }
        if (ch == '\\') {
            ++p;
            if (*p == '\0') {
                return 0;
            }
            switch (*p) {
            case '"':
            case '\\':
            case '/':
                ch = (unsigned char)*p;
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
            default:
                return 0;
            }
        }

        if (output + 1 >= out_value_size) {
            return 0;
        }
        out_value[output++] = (char)ch;
    }

    return 0;
}

int nn_trace_json_get_u64(const char *line, const char *key, uint64_t *out_value)
{
    const char *start = trace_find_key_value(line, key);
    if (start == NULL || out_value == NULL || *start == '-' || *start == '+') {
        return 0;
    }

    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(start, &end, 10);
    if (end == start || errno == ERANGE ||
        (*end != ',' && *end != '}' && *end != ']' && *end != '\n' && *end != '\r' &&
         *end != '\0')) {
        return 0;
    }
    *out_value = (uint64_t)value;
    return 1;
}

int nn_trace_json_get_real(const char *line, const char *key, nn_real *out_value)
{
    const char *start = trace_find_key_value(line, key);
    if (start == NULL || out_value == NULL) {
        return 0;
    }

    errno = 0;
    char *end = NULL;
    double value = strtod(start, &end);
    if (end == start || errno == ERANGE || !isfinite(value) ||
        (*end != ',' && *end != '}' && *end != ']' && *end != '\n' && *end != '\r' &&
         *end != '\0')) {
        return 0;
    }
    *out_value = (nn_real)value;
    return 1;
}

static int trace_parse_u32_array(const char *line,
                                 const char *key,
                                 uint32_t *values,
                                 size_t expected_count)
{
    const char *p = trace_find_key_value(line, key);
    if (p == NULL || values == NULL || *p != '[') {
        return 0;
    }
    ++p;

    for (size_t i = 0; i < expected_count; ++i) {
        if (*p == '-' || *p == '+') {
            return 0;
        }
        errno = 0;
        char *end = NULL;
        unsigned long value = strtoul(p, &end, 10);
        if (end == p || errno == ERANGE || value > UINT32_MAX) {
            return 0;
        }
        values[i] = (uint32_t)value;
        p = end;
        if (i + 1 < expected_count) {
            if (*p != ',') {
                return 0;
            }
            ++p;
        }
    }

    return *p == ']';
}

static int trace_parse_real_array(const char *line,
                                  const char *key,
                                  nn_real *values,
                                  size_t expected_count)
{
    const char *p = trace_find_key_value(line, key);
    if (p == NULL || values == NULL || *p != '[') {
        return 0;
    }
    ++p;

    for (size_t i = 0; i < expected_count; ++i) {
        errno = 0;
        char *end = NULL;
        double value = strtod(p, &end);
        if (end == p || errno == ERANGE || !isfinite(value)) {
            return 0;
        }
        values[i] = (nn_real)value;
        p = end;
        if (i + 1 < expected_count) {
            if (*p != ',') {
                return 0;
            }
            ++p;
        }
    }

    return *p == ']';
}

static int trace_parse_hex_u64(const char *line, const char *key, uint64_t *out_value)
{
    char text[32];
    if (!nn_trace_json_get_string(line, key, text, sizeof(text)) || strlen(text) != 16) {
        return 0;
    }

    uint64_t value = 0;
    for (size_t i = 0; i < 16; ++i) {
        char ch = text[i];
        uint64_t nibble;
        if (ch >= '0' && ch <= '9') {
            nibble = (uint64_t)(ch - '0');
        } else if (ch >= 'a' && ch <= 'f') {
            nibble = (uint64_t)(ch - 'a' + 10);
        } else if (ch >= 'A' && ch <= 'F') {
            nibble = (uint64_t)(ch - 'A' + 10);
        } else {
            return 0;
        }
        value = (value << 4) | nibble;
    }

    *out_value = value;
    return 1;
}

static int trace_validate_logged_action_id(const char *line,
                                           const char *key,
                                           uint32_t expected_action_id)
{
    uint64_t logged = 0;
    const char *value = trace_find_key_value(line, key);
    if (value == NULL) {
        return 1;
    }
    if (!nn_trace_json_get_u64(line, key, &logged) || logged > UINT32_MAX) {
        return 0;
    }
    return (uint32_t)logged == expected_action_id;
}

int nn_trace_parse_jsonl(const char *line,
                         NNParsedTraceRecord *out_record,
                         uint32_t *action_ids,
                         nn_real *scores,
                         nn_real *probabilities,
                         size_t candidate_capacity)
{
    if (line == NULL || out_record == NULL || action_ids == NULL || scores == NULL ||
        probabilities == NULL || candidate_capacity == 0) {
        return 0;
    }

    NNParsedTraceRecord parsed;
    memset(&parsed, 0, sizeof(parsed));

    uint64_t candidate_count = 0;
    uint64_t selected_index = 0;
    uint64_t target_index = 0;
    uint64_t trace_version = 0;
    uint64_t feature_encoder_version = 0;
    nn_real loss = 0;
    nn_real rank_score_epsilon = NN_RANK_SCORE_EPSILON;

    if (!nn_trace_json_get_string(line, "schema", parsed.schema, sizeof(parsed.schema)) ||
        strcmp(parsed.schema, NN_TRACE_SCHEMA) != 0 ||
        !nn_trace_json_get_u64(line, "trace_version", &trace_version) ||
        trace_version != NN_TRACE_VERSION ||
        !nn_trace_json_get_string(line,
                                  "action_id_contract",
                                  parsed.action_id_contract,
                                  sizeof(parsed.action_id_contract)) ||
        strcmp(parsed.action_id_contract, NN_TRACE_ACTION_ID_CONTRACT) != 0 ||
        !nn_trace_json_get_string(line, "task", parsed.task_name, sizeof(parsed.task_name)) ||
        !trace_parse_hex_u64(line, "state_hash", &parsed.state_hash) ||
        !nn_trace_json_get_string(
            line, "model_hash", parsed.model_hash_hex, sizeof(parsed.model_hash_hex)) ||
        !nn_trace_json_get_u64(line, "feature_encoder_version", &feature_encoder_version) ||
        feature_encoder_version > UINT32_MAX ||
        !nn_trace_json_get_u64(line, "candidate_count", &candidate_count) ||
        candidate_count == 0 || candidate_count > candidate_capacity ||
        !nn_trace_json_get_u64(line, "selected_index", &selected_index) ||
        !nn_trace_json_get_u64(line, "target_index", &target_index) ||
        selected_index >= candidate_count || target_index >= candidate_count ||
        !nn_trace_json_get_real(line, "loss", &loss) ||
        !nn_trace_json_get_string(
            line, "verifier_result", parsed.verifier_result, sizeof(parsed.verifier_result)) ||
        !nn_trace_json_get_u64(line, "seed", &parsed.seed)) {
        return 0;
    }

    parsed.feature_encoder_version = (uint32_t)feature_encoder_version;
    parsed.trace_version = (uint32_t)trace_version;
    parsed.candidate_count = (size_t)candidate_count;
    parsed.selected_index = (size_t)selected_index;
    parsed.target_index = (size_t)target_index;
    parsed.loss = loss;
    parsed.rank_score_epsilon = rank_score_epsilon;

    if (!trace_parse_u32_array(line, "action_ids", action_ids, parsed.candidate_count) ||
        !trace_parse_real_array(line, "scores", scores, parsed.candidate_count) ||
        !trace_parse_real_array(line, "probabilities", probabilities, parsed.candidate_count)) {
        return 0;
    }

    for (size_t i = 0; i < parsed.candidate_count; ++i) {
        for (size_t j = i + 1; j < parsed.candidate_count; ++j) {
            if (action_ids[i] == action_ids[j]) {
                return 0;
            }
        }
    }
    parsed.selected_action_id = action_ids[parsed.selected_index];
    parsed.target_action_id = action_ids[parsed.target_index];

    if (!trace_validate_logged_action_id(line,
                                         "selected_action_id",
                                         parsed.selected_action_id) ||
        !trace_validate_logged_action_id(line, "target_action_id", parsed.target_action_id)) {
        return 0;
    }

    nn_real logged_epsilon = 0;
    if (trace_find_key_value(line, "rank_score_epsilon") != NULL) {
        if (!nn_trace_json_get_real(line, "rank_score_epsilon", &logged_epsilon) ||
            logged_epsilon < (nn_real)0) {
            return 0;
        }
        parsed.rank_score_epsilon = logged_epsilon;
    }

    parsed.selected_score_margin =
        nn_action_score_margin(scores, parsed.candidate_count, parsed.selected_index);
    if (!isfinite((double)parsed.selected_score_margin)) {
        return 0;
    }
    nn_real logged_margin = 0;
    if (trace_find_key_value(line, "selected_score_margin") != NULL) {
        if (!nn_trace_json_get_real(line, "selected_score_margin", &logged_margin) ||
            !isfinite((double)logged_margin)) {
            return 0;
        }
        double tolerance = fabs((double)parsed.selected_score_margin) * 1e-5 + 1e-5;
        if (fabs((double)logged_margin - (double)parsed.selected_score_margin) > tolerance) {
            return 0;
        }
        parsed.selected_score_margin = logged_margin;
    }

    if (nn_trace_json_get_string(line,
                                 "ranking_policy",
                                 parsed.ranking_policy,
                                 sizeof(parsed.ranking_policy))) {
        if (strcmp(parsed.ranking_policy, NN_RANKING_POLICY) != 0) {
            return 0;
        }
        size_t selected_by_policy = 0;
        for (size_t i = 1; i < parsed.candidate_count; ++i) {
            if (rank_score_is_better(scores,
                                     action_ids,
                                     i,
                                     selected_by_policy,
                                     parsed.rank_score_epsilon)) {
                selected_by_policy = i;
            }
        }
        if (selected_by_policy != parsed.selected_index) {
            return 0;
        }
    } else {
        snprintf(parsed.ranking_policy, sizeof(parsed.ranking_policy), "legacy_score_order");
    }

    nn_real probability_sum = 0;
    for (size_t i = 0; i < parsed.candidate_count; ++i) {
        if (!isfinite((double)scores[i]) ||
            !isfinite((double)probabilities[i]) ||
            probabilities[i] < (nn_real)0 ||
            probabilities[i] > (nn_real)1) {
            return 0;
        }
        probability_sum += probabilities[i];
    }

    if (fabs((double)probability_sum - 1.0) > 1e-3) {
        return 0;
    }

    *out_record = parsed;
    return 1;
}
