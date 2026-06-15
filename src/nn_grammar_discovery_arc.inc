static int nn_gd_arc_attempt_list_reserve(NNGDArcPipelineAttemptList *list,
                                          size_t needed) {
    NNGDArcPipelineAttempt *items;
    size_t capacity;
    if (needed <= list->capacity) {
        return 0;
    }
    capacity = list->capacity == 0U ? 64U : list->capacity * 2U;
    while (capacity < needed) {
        capacity *= 2U;
    }
    items = (NNGDArcPipelineAttempt *)realloc(list->items,
                                              capacity * sizeof(NNGDArcPipelineAttempt));
    if (items == NULL) {
        return -1;
    }
    list->items = items;
    list->capacity = capacity;
    return 0;
}

static void nn_gd_arc_attempt_free(NNGDArcPipelineAttempt *attempt) {
    free(attempt->split_name);
    free(attempt->family);
    free(attempt->task_id);
    free(attempt->split);
    free(attempt->attempt_id);
    free(attempt->library_id);
    free(attempt->pipeline_kind);
    free(attempt->op_name);
    free(attempt->visual_program);
    free(attempt->slot_contract);
    free(attempt->composition_kind);
    free(attempt->normalization_frame);
    free(attempt->typed_stages);
    free(attempt->primitive_contract);
    free(attempt->slot_tokens);
    free(attempt->rejection_reason);
    free(attempt->failure_stage);
    free(attempt->failure_detail);
    free(attempt->contrast_accepted_attempt_id);
    free(attempt->contrast_pair_kind);
    free(attempt->token_indices);
    nn_gd_string_list_free(&attempt->tokens);
}

static void nn_gd_arc_attempt_list_free(NNGDArcPipelineAttemptList *list) {
    size_t i;
    for (i = 0; i < list->count; ++i) {
        nn_gd_arc_attempt_free(&list->items[i]);
    }
    free(list->items);
    list->items = NULL;
    list->count = 0U;
    list->capacity = 0U;
}

static int nn_gd_add_pipe_tokens_with_prefix(NNGDStringList *tokens,
                                             const char *prefix,
                                             const char *values) {
    char *copy;
    char *start;
    char *p;
    if (values == NULL || values[0] == '\0') {
        return 0;
    }
    copy = nn_gd_strdup(values);
    if (copy == NULL) {
        return -1;
    }
    start = copy;
    for (p = copy; ; ++p) {
        if (*p == '|' || *p == '\0') {
            char saved = *p;
            *p = '\0';
            if (start[0] != '\0' && nn_gd_add_token_pair(tokens, prefix, start) != 0) {
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

static int nn_gd_add_named_value_tokens(NNGDStringList *tokens,
                                        const char *name,
                                        const char *value) {
    char token[256];
    char bucket[256];
    long parsed;
    if (name == NULL || name[0] == '\0' || value == NULL || value[0] == '\0') {
        return 0;
    }
    snprintf(token, sizeof(token), "%s=%s", name, value);
    if (nn_gd_add_token_pair(tokens, "field:", token) != 0) {
        return -1;
    }
    if (!isdigit((unsigned char)value[0]) && value[0] != '-') {
        return 0;
    }
    parsed = nn_gd_parse_long(value);
    if (parsed == 0L) {
        snprintf(bucket, sizeof(bucket), "%s:zero", name);
    } else if (parsed == 1L) {
        snprintf(bucket, sizeof(bucket), "%s:one", name);
    } else if (parsed == 2L) {
        snprintf(bucket, sizeof(bucket), "%s:two", name);
    } else if (parsed > 0L && parsed <= 4L) {
        snprintf(bucket, sizeof(bucket), "%s:small", name);
    } else if (parsed > 0L && parsed <= 8L) {
        snprintf(bucket, sizeof(bucket), "%s:medium", name);
    } else if (parsed > 8L) {
        snprintf(bucket, sizeof(bucket), "%s:large", name);
    } else {
        snprintf(bucket, sizeof(bucket), "%s:negative", name);
    }
    return nn_gd_add_token_pair(tokens, "field_bucket:", bucket);
}

static int nn_gd_arc_attempt_add_tokens(NNGDArcPipelineAttempt *attempt,
                                        char **fields,
                                        size_t field_count) {
    static const char *numeric_names[] = {
        "selector",
        "background",
        "color",
        "output_color",
        "selector_kind",
        "include_background",
        "orientation",
        "count_mode",
        "x_align",
        "y_align",
        "x_anchor",
        "y_anchor",
        "x_offset",
        "y_offset",
        "width",
        "height",
        "period",
        "proposal_count",
        "support_tasks",
        "input_width",
        "input_height",
        "output_width",
        "output_height",
        "shape_scale_x",
        "shape_scale_y",
        "period_divides_input_width",
        "period_divides_input_height",
        "slot_width_matches_output",
        "slot_height_matches_output",
        "color_in_input",
        "color_in_output",
        "output_color_in_output",
        "background_in_input",
        "verifier_train_pairs",
        "verifier_checked_cells",
        "verifier_mismatches",
        "verifier_unsupported_pairs",
        "verifier_apply_failed_pairs",
        "verifier_shape_mismatch_pairs",
        "verifier_missing_output_pairs",
        "verifier_first_failing_pair",
        "input_component_count",
        "selected_component_count",
        "selected_component_area_min",
        "selected_component_area_max",
        "selected_component_area_total",
        "selected_component_same_shape",
        "selected_component_shape_count",
        "selected_component_bbox_width",
        "selected_component_bbox_height",
        "selected_bbox_matches_output_shape",
        "selected_bbox_rotated_output_shape",
        "output_foreground_cell_count",
        "output_component_count",
        "output_non_background_color_count",
        "selected_count_matches_output_components",
        "selected_count_matches_output_foreground",
        "selected_area_matches_output_foreground",
    };
    static const size_t numeric_indices[] = {
        23U, 24U, 25U, 26U, 27U, 28U, 29U, 30U, 31U, 32U,
        33U, 34U, 35U, 36U, 37U, 38U, 39U, 20U, 21U,
        44U, 45U, 46U, 47U, 48U, 49U, 50U, 51U, 52U, 53U,
        54U, 55U, 56U, 57U,
        58U, 59U, 60U, 61U, 62U, 63U, 64U, 65U,
        68U, 69U, 70U, 71U, 72U, 73U, 74U, 75U, 76U, 77U,
        78U, 79U, 80U, 81U, 82U, 83U, 84U,
    };
    size_t i;
    if (nn_gd_add_token_pair(&attempt->tokens, "family:", attempt->family) != 0 ||
        nn_gd_add_token_pair(&attempt->tokens, "split:", attempt->split) != 0 ||
        nn_gd_add_token_pair(&attempt->tokens, "pipeline:", attempt->pipeline_kind) != 0 ||
        nn_gd_add_token_pair(&attempt->tokens, "op:", attempt->op_name) != 0 ||
        nn_gd_add_token_pair(&attempt->tokens,
                             "composition:",
                             attempt->composition_kind) != 0 ||
        nn_gd_add_token_pair(&attempt->tokens,
                             "normalizer:",
                             attempt->normalization_frame) != 0 ||
        nn_gd_add_token_pair(&attempt->tokens,
                             "visual_program:",
                             attempt->visual_program) != 0 ||
        nn_gd_add_pipe_tokens_with_prefix(&attempt->tokens,
                                          "stage:",
                                          attempt->typed_stages) != 0 ||
        nn_gd_add_pipe_tokens_with_prefix(&attempt->tokens,
                                          "contract:",
                                          attempt->slot_contract) != 0 ||
        nn_gd_add_token_pair(&attempt->tokens,
                             "primitive_contract:",
                             attempt->primitive_contract) != 0 ||
        nn_gd_add_token_pair(&attempt->tokens,
                             "failure_stage:",
                             attempt->failure_stage) != 0 ||
        nn_gd_add_token_pair(&attempt->tokens,
                             "failure_detail:",
                             attempt->failure_detail) != 0 ||
        nn_gd_add_slot_tokens(&attempt->tokens, attempt->slot_tokens) != 0) {
        return -1;
    }
    for (i = 0U; i < sizeof(numeric_names) / sizeof(numeric_names[0]); ++i) {
        if (nn_gd_add_named_value_tokens(&attempt->tokens,
                                         numeric_names[i],
                                         nn_gd_field(fields,
                                                     field_count,
                                                     numeric_indices[i])) != 0) {
            return -1;
        }
    }
    return 0;
}

static int nn_gd_arc_attempt_list_add(NNGDArcPipelineAttemptList *list,
                                      char **fields,
                                      size_t field_count) {
    NNGDArcPipelineAttempt *attempt;
    if (nn_gd_arc_attempt_list_reserve(list, list->count + 1U) != 0) {
        return -1;
    }
    attempt = &list->items[list->count];
    memset(attempt, 0, sizeof(*attempt));
    attempt->split_name = nn_gd_strdup(nn_gd_field(fields, field_count, 0U));
    attempt->family = nn_gd_strdup(nn_gd_field(fields, field_count, 1U));
    attempt->task_id = nn_gd_strdup(nn_gd_field(fields, field_count, 2U));
    attempt->split = nn_gd_strdup(nn_gd_field(fields, field_count, 3U));
    attempt->attempt_id = nn_gd_strdup(nn_gd_field(fields, field_count, 4U));
    attempt->library_id = nn_gd_strdup(nn_gd_field(fields, field_count, 5U));
    attempt->attempt_index = nn_gd_parse_long(nn_gd_field(fields, field_count, 6U));
    attempt->pipeline_kind = nn_gd_strdup(nn_gd_field(fields, field_count, 7U));
    attempt->op_name = nn_gd_strdup(nn_gd_field(fields, field_count, 8U));
    attempt->visual_program = nn_gd_strdup(nn_gd_field(fields, field_count, 9U));
    attempt->slot_contract = nn_gd_strdup(nn_gd_field(fields, field_count, 10U));
    attempt->composition_kind = nn_gd_strdup(nn_gd_field(fields, field_count, 11U));
    attempt->normalization_frame = nn_gd_strdup(nn_gd_field(fields, field_count, 12U));
    attempt->typed_stages = nn_gd_strdup(nn_gd_field(fields, field_count, 13U));
    attempt->primitive_contract = nn_gd_strdup(nn_gd_field(fields, field_count, 14U));
    attempt->slot_tokens = nn_gd_strdup(nn_gd_field(fields, field_count, 15U));
    attempt->accepted = nn_gd_parse_bool(nn_gd_field(fields, field_count, 16U));
    attempt->rejection_reason = nn_gd_strdup(nn_gd_field(fields, field_count, 17U));
    attempt->failure_stage = nn_gd_strdup(nn_gd_field(fields, field_count, 18U));
    attempt->failure_detail = nn_gd_strdup(nn_gd_field(fields, field_count, 19U));
    attempt->proposal_count = nn_gd_parse_long(nn_gd_field(fields, field_count, 20U));
    attempt->support_tasks = nn_gd_parse_long(nn_gd_field(fields, field_count, 21U));
    attempt->utility_mean = nn_gd_parse_double(nn_gd_field(fields, field_count, 22U));
    attempt->selector = nn_gd_parse_long(nn_gd_field(fields, field_count, 23U));
    attempt->x_anchor = nn_gd_parse_long(nn_gd_field(fields, field_count, 33U));
    attempt->y_anchor = nn_gd_parse_long(nn_gd_field(fields, field_count, 34U));
    attempt->x_offset = nn_gd_parse_long(nn_gd_field(fields, field_count, 35U));
    attempt->y_offset = nn_gd_parse_long(nn_gd_field(fields, field_count, 36U));
    attempt->width = nn_gd_parse_long(nn_gd_field(fields, field_count, 37U));
    attempt->height = nn_gd_parse_long(nn_gd_field(fields, field_count, 38U));
    attempt->period = nn_gd_parse_long(nn_gd_field(fields, field_count, 39U));
    attempt->contrast_accepted_attempt_id =
        nn_gd_strdup(nn_gd_field(fields, field_count, 40U));
    attempt->contrast_slot_distance =
        nn_gd_parse_long(nn_gd_field(fields, field_count, 41U));
    attempt->contrast_pair_kind = nn_gd_strdup(nn_gd_field(fields, field_count, 42U));
    attempt->contrast_rejected_sibling_count =
        nn_gd_parse_long(nn_gd_field(fields, field_count, 43U));
    attempt->input_width = nn_gd_parse_long(nn_gd_field(fields, field_count, 44U));
    attempt->input_height = nn_gd_parse_long(nn_gd_field(fields, field_count, 45U));
    attempt->output_width = nn_gd_parse_long(nn_gd_field(fields, field_count, 46U));
    attempt->output_height = nn_gd_parse_long(nn_gd_field(fields, field_count, 47U));
    attempt->shape_scale_x = nn_gd_parse_long(nn_gd_field(fields, field_count, 48U));
    attempt->shape_scale_y = nn_gd_parse_long(nn_gd_field(fields, field_count, 49U));
    attempt->period_divides_input_width =
        nn_gd_parse_bool(nn_gd_field(fields, field_count, 50U));
    attempt->period_divides_input_height =
        nn_gd_parse_bool(nn_gd_field(fields, field_count, 51U));
    attempt->slot_width_matches_output =
        nn_gd_parse_bool(nn_gd_field(fields, field_count, 52U));
    attempt->slot_height_matches_output =
        nn_gd_parse_bool(nn_gd_field(fields, field_count, 53U));
    attempt->verifier_train_pairs =
        nn_gd_parse_long(nn_gd_field(fields, field_count, 58U));
    attempt->verifier_checked_cells =
        nn_gd_parse_long(nn_gd_field(fields, field_count, 59U));
    attempt->verifier_mismatches =
        nn_gd_parse_long(nn_gd_field(fields, field_count, 60U));
    attempt->verifier_unsupported_pairs =
        nn_gd_parse_long(nn_gd_field(fields, field_count, 61U));
    attempt->verifier_apply_failed_pairs =
        nn_gd_parse_long(nn_gd_field(fields, field_count, 62U));
    attempt->verifier_shape_mismatch_pairs =
        nn_gd_parse_long(nn_gd_field(fields, field_count, 63U));
    attempt->verifier_missing_output_pairs =
        nn_gd_parse_long(nn_gd_field(fields, field_count, 64U));
    attempt->verifier_first_failing_pair =
        nn_gd_parse_long(nn_gd_field(fields, field_count, 65U));
    attempt->verifier_mismatch_ratio =
        nn_gd_parse_double(nn_gd_field(fields, field_count, 66U));
    attempt->verifier_score =
        nn_gd_parse_double(nn_gd_field(fields, field_count, 67U));
    attempt->contrast_accepted_index = SIZE_MAX;
    if (attempt->split_name == NULL || attempt->family == NULL ||
        attempt->task_id == NULL || attempt->split == NULL ||
        attempt->attempt_id == NULL || attempt->library_id == NULL ||
        attempt->pipeline_kind == NULL || attempt->op_name == NULL ||
        attempt->visual_program == NULL || attempt->slot_contract == NULL ||
        attempt->composition_kind == NULL || attempt->normalization_frame == NULL ||
        attempt->typed_stages == NULL || attempt->primitive_contract == NULL ||
        attempt->slot_tokens == NULL || attempt->rejection_reason == NULL ||
        attempt->failure_stage == NULL || attempt->failure_detail == NULL ||
        attempt->contrast_accepted_attempt_id == NULL ||
        attempt->contrast_pair_kind == NULL ||
        nn_gd_arc_attempt_add_tokens(attempt, fields, field_count) != 0) {
        nn_gd_arc_attempt_free(attempt);
        memset(attempt, 0, sizeof(*attempt));
        return -1;
    }
    attempt->token_count = attempt->tokens.count;
    ++list->count;
    return 0;
}

static int nn_gd_read_arc_pipeline_attempts(const char *path,
                                            NNGDArcPipelineAttemptList *attempts) {
    FILE *fp;
    char line[NN_GD_LINE_MAX];
    char *fields[NN_GD_MAX_FIELDS];
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    while (fgets(line, sizeof(line), fp) != NULL) {
        size_t field_count = nn_gd_split_tabs(line, fields, NN_GD_MAX_FIELDS);
        const char *split_name = nn_gd_field(fields, field_count, 0U);
        const char *attempt_id = nn_gd_field(fields, field_count, 4U);
        if (split_name[0] == '\0' || strcmp(split_name, "split_name") == 0 ||
            attempt_id[0] == '\0') {
            continue;
        }
        if (nn_gd_arc_attempt_list_add(attempts, fields, field_count) != 0) {
            fclose(fp);
            return -1;
        }
    }
    fclose(fp);
    return 0;
}

static void nn_gd_free_arc_pipeline_field_buffer(char **fields, size_t field_count) {
    size_t i;
    for (i = 0U; i < field_count; ++i) {
        free(fields[i]);
        fields[i] = NULL;
    }
}

static int nn_gd_arc_pipeline_attempt_fields_from_json(const char *line,
                                                       const char *split_name,
                                                       char **fields,
                                                       size_t field_count) {
    NNGDJsonIndex index;
    char *schema;
    size_t i;
    if (field_count < NN_GD_ARC_RUNTIME_FIELD_COUNT) {
        return -1;
    }
    for (i = 0U; i < field_count; ++i) {
        fields[i] = NULL;
    }
    nn_gd_json_index_line(line, &index);
    schema = nn_gd_json_dup_index_tsv_string(&index, "schema");
    if (schema == NULL) {
        return -1;
    }
    if (strcmp(schema,
               "crlp.neural_grammar_discovery.arc_visual_pipeline_attempt.v1") != 0) {
        free(schema);
        return 1;
    }
    free(schema);
    fields[0] = nn_gd_strdup(split_name);
    fields[1] = nn_gd_json_dup_index_tsv_string(&index, "family");
    fields[2] = nn_gd_json_dup_index_tsv_string(&index, "task_id");
    fields[3] = nn_gd_json_dup_index_tsv_string(&index, "split");
    fields[4] = nn_gd_json_dup_index_tsv_string(&index, "attempt_id");
    fields[5] = nn_gd_json_dup_index_tsv_string(&index, "library_id");
    fields[6] = nn_gd_json_dup_index_scalar_tsv(&index, "attempt_index", 1);
    fields[7] = nn_gd_json_dup_index_tsv_string(&index, "pipeline_kind");
    fields[8] = nn_gd_json_dup_index_tsv_string(&index, "op_name");
    fields[9] = nn_gd_json_dup_index_tsv_string(&index, "visual_program");
    fields[10] = nn_gd_json_dup_index_tsv_string(&index, "slot_contract");
    fields[11] = nn_gd_json_dup_index_tsv_string(&index, "composition_kind");
    fields[12] = nn_gd_json_dup_index_tsv_string(&index, "normalization_frame");
    fields[13] = nn_gd_json_dup_index_tsv_string(&index, "typed_stages");
    fields[14] = nn_gd_json_dup_index_tsv_string(&index, "primitive_contract");
    fields[15] = nn_gd_json_dup_index_slot_tokens(&index);
    fields[16] = nn_gd_json_dup_index_scalar_tsv(&index, "accepted", 1);
    fields[17] = nn_gd_json_dup_index_tsv_string(&index, "rejection_reason");
    fields[18] = nn_gd_json_dup_index_tsv_string(&index, "failure_stage");
    fields[19] = nn_gd_json_dup_index_tsv_string(&index, "failure_detail");
    fields[20] = nn_gd_json_dup_index_scalar_tsv(&index, "proposal_count", 1);
    fields[21] = nn_gd_json_dup_index_scalar_tsv(&index, "support_tasks", 1);
    fields[22] = nn_gd_json_dup_index_scalar_tsv(&index, "utility_mean", 1);
    fields[23] = nn_gd_json_dup_index_scalar_tsv(&index, "selector", 1);
    fields[24] = nn_gd_json_dup_index_scalar_tsv(&index, "background", 1);
    fields[25] = nn_gd_json_dup_index_scalar_tsv(&index, "color", 1);
    fields[26] = nn_gd_json_dup_index_scalar_tsv(&index, "output_color", 1);
    fields[27] = nn_gd_json_dup_index_scalar_tsv(&index, "selector_kind", 1);
    fields[28] = nn_gd_json_dup_index_scalar_tsv(&index, "include_background", 1);
    fields[29] = nn_gd_json_dup_index_scalar_tsv(&index, "orientation", 1);
    fields[30] = nn_gd_json_dup_index_scalar_tsv(&index, "count_mode", 1);
    fields[31] = nn_gd_json_dup_index_scalar_tsv(&index, "x_align", 1);
    fields[32] = nn_gd_json_dup_index_scalar_tsv(&index, "y_align", 1);
    fields[33] = nn_gd_json_dup_index_scalar_tsv(&index, "x_anchor", 1);
    fields[34] = nn_gd_json_dup_index_scalar_tsv(&index, "y_anchor", 1);
    fields[35] = nn_gd_json_dup_index_scalar_tsv(&index, "x_offset", 1);
    fields[36] = nn_gd_json_dup_index_scalar_tsv(&index, "y_offset", 1);
    fields[37] = nn_gd_json_dup_index_scalar_tsv(&index, "width", 1);
    fields[38] = nn_gd_json_dup_index_scalar_tsv(&index, "height", 1);
    fields[39] = nn_gd_json_dup_index_scalar_tsv(&index, "period", 1);
    fields[40] = nn_gd_json_dup_index_tsv_string(&index, "contrast_accepted_attempt_id");
    fields[41] = nn_gd_json_dup_index_scalar_tsv(&index, "contrast_slot_distance", 1);
    fields[42] = nn_gd_json_dup_index_tsv_string(&index, "contrast_pair_kind");
    fields[43] =
        nn_gd_json_dup_index_scalar_tsv(&index, "contrast_rejected_sibling_count", 1);
    fields[44] = nn_gd_json_dup_index_scalar_tsv(&index, "input_width", 1);
    fields[45] = nn_gd_json_dup_index_scalar_tsv(&index, "input_height", 1);
    fields[46] = nn_gd_json_dup_index_scalar_tsv(&index, "output_width", 1);
    fields[47] = nn_gd_json_dup_index_scalar_tsv(&index, "output_height", 1);
    fields[48] = nn_gd_json_dup_index_scalar_tsv(&index, "shape_scale_x", 1);
    fields[49] = nn_gd_json_dup_index_scalar_tsv(&index, "shape_scale_y", 1);
    fields[50] =
        nn_gd_json_dup_index_scalar_tsv(&index, "period_divides_input_width", 1);
    fields[51] =
        nn_gd_json_dup_index_scalar_tsv(&index, "period_divides_input_height", 1);
    fields[52] = nn_gd_json_dup_index_scalar_tsv(&index, "slot_width_matches_output", 1);
    fields[53] = nn_gd_json_dup_index_scalar_tsv(&index, "slot_height_matches_output", 1);
    fields[54] = nn_gd_json_dup_index_scalar_tsv(&index, "color_in_input", 1);
    fields[55] = nn_gd_json_dup_index_scalar_tsv(&index, "color_in_output", 1);
    fields[56] = nn_gd_json_dup_index_scalar_tsv(&index, "output_color_in_output", 1);
    fields[57] = nn_gd_json_dup_index_scalar_tsv(&index, "background_in_input", 1);
    fields[58] = nn_gd_json_dup_index_scalar_tsv(&index, "train_pairs", 1);
    fields[59] = nn_gd_json_dup_index_scalar_tsv(&index, "checked_cells", 1);
    fields[60] = nn_gd_json_dup_index_scalar_tsv(&index, "mismatches", 1);
    fields[61] = nn_gd_json_dup_index_scalar_tsv(&index, "unsupported_pairs", 1);
    fields[62] = nn_gd_json_dup_index_scalar_tsv(&index, "apply_failed_pairs", 1);
    fields[63] = nn_gd_json_dup_index_scalar_tsv(&index, "shape_mismatch_pairs", 1);
    fields[64] = nn_gd_json_dup_index_scalar_tsv(&index, "missing_output_pairs", 1);
    fields[65] = nn_gd_json_dup_index_scalar_tsv(&index, "first_failing_pair", 1);
    fields[66] = nn_gd_json_dup_index_scalar_tsv(&index, "mismatch_ratio", 1);
    fields[67] = nn_gd_json_dup_index_scalar_tsv(&index, "verifier_score", 1);
    fields[68] = nn_gd_json_dup_index_scalar_tsv(&index, "input_component_count", 1);
    fields[69] = nn_gd_json_dup_index_scalar_tsv(&index, "selected_component_count", 1);
    fields[70] = nn_gd_json_dup_index_scalar_tsv(&index, "selected_component_area_min", 1);
    fields[71] = nn_gd_json_dup_index_scalar_tsv(&index, "selected_component_area_max", 1);
    fields[72] =
        nn_gd_json_dup_index_scalar_tsv(&index, "selected_component_area_total", 1);
    fields[73] =
        nn_gd_json_dup_index_scalar_tsv(&index, "selected_component_same_shape", 1);
    fields[74] =
        nn_gd_json_dup_index_scalar_tsv(&index, "selected_component_shape_count", 1);
    fields[75] =
        nn_gd_json_dup_index_scalar_tsv(&index, "selected_component_bbox_width", 1);
    fields[76] =
        nn_gd_json_dup_index_scalar_tsv(&index, "selected_component_bbox_height", 1);
    fields[77] =
        nn_gd_json_dup_index_scalar_tsv(&index, "selected_bbox_matches_output_shape", 1);
    fields[78] =
        nn_gd_json_dup_index_scalar_tsv(&index, "selected_bbox_rotated_output_shape", 1);
    fields[79] =
        nn_gd_json_dup_index_scalar_tsv(&index, "output_foreground_cell_count", 1);
    fields[80] = nn_gd_json_dup_index_scalar_tsv(&index, "output_component_count", 1);
    fields[81] =
        nn_gd_json_dup_index_scalar_tsv(&index, "output_non_background_color_count", 1);
    fields[82] = nn_gd_json_dup_index_scalar_tsv(
        &index,
        "selected_count_matches_output_components",
        1);
    fields[83] = nn_gd_json_dup_index_scalar_tsv(
        &index,
        "selected_count_matches_output_foreground",
        1);
    fields[84] =
        nn_gd_json_dup_index_scalar_tsv(&index, "selected_area_matches_output_foreground", 1);
    for (i = 0U; i < NN_GD_ARC_RUNTIME_FIELD_COUNT; ++i) {
        if (fields[i] == NULL) {
            nn_gd_free_arc_pipeline_field_buffer(fields, NN_GD_ARC_RUNTIME_FIELD_COUNT);
            return -1;
        }
    }
    return 0;
}

static int nn_gd_read_arc_pipeline_attempts_jsonl(const char *path,
                                                  const char *split_name,
                                                  NNGDArcPipelineAttemptList *attempts) {
    FILE *fp;
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    for (;;) {
        char *line = nn_gd_read_line_dynamic(fp);
        char *fields[NN_GD_ARC_RUNTIME_FIELD_COUNT];
        int parse_status;
        if (line == NULL) {
            break;
        }
        parse_status = nn_gd_arc_pipeline_attempt_fields_from_json(
            line,
            split_name,
            fields,
            NN_GD_ARC_RUNTIME_FIELD_COUNT);
        free(line);
        if (parse_status < 0) {
            fclose(fp);
            return -1;
        }
        if (parse_status == 0) {
            if (nn_gd_arc_attempt_list_add(attempts,
                                           fields,
                                           NN_GD_ARC_RUNTIME_FIELD_COUNT) != 0) {
                nn_gd_free_arc_pipeline_field_buffer(fields,
                                                     NN_GD_ARC_RUNTIME_FIELD_COUNT);
                fclose(fp);
                return -1;
            }
            nn_gd_free_arc_pipeline_field_buffer(fields, NN_GD_ARC_RUNTIME_FIELD_COUNT);
        }
    }
    fclose(fp);
    return 0;
}

static int nn_gd_arc_hole_source_log_list_reserve(NNGDArcHoleSourceLogList *list,
                                                   size_t needed) {
    NNGDArcHoleSourceLog *items;
    size_t capacity;
    if (needed <= list->capacity) {
        return 0;
    }
    capacity = list->capacity == 0U ? 4U : list->capacity * 2U;
    while (capacity < needed) {
        capacity *= 2U;
    }
    items = (NNGDArcHoleSourceLog *)realloc(list->items,
                                            capacity * sizeof(*list->items));
    if (items == NULL) {
        return -1;
    }
    list->items = items;
    list->capacity = capacity;
    return 0;
}

static void nn_gd_arc_hole_source_log_list_free(NNGDArcHoleSourceLogList *list) {
    size_t i;
    for (i = 0U; i < list->count; ++i) {
        free(list->items[i].scope);
        free(list->items[i].path);
    }
    free(list->items);
    memset(list, 0, sizeof(*list));
}

static int nn_gd_arc_hole_target_list_reserve(NNGDArcHoleTargetList *list,
                                               size_t needed) {
    NNGDArcHoleTarget *items;
    size_t capacity;
    if (needed <= list->capacity) {
        return 0;
    }
    capacity = list->capacity == 0U ? 16U : list->capacity * 2U;
    while (capacity < needed) {
        capacity *= 2U;
    }
    items = (NNGDArcHoleTarget *)realloc(list->items,
                                         capacity * sizeof(*list->items));
    if (items == NULL) {
        return -1;
    }
    list->items = items;
    list->capacity = capacity;
    return 0;
}

static void nn_gd_arc_hole_target_free(NNGDArcHoleTarget *target) {
    free(target->task_id);
    free(target->split);
    free(target->source_scope);
    free(target->source_path);
    free(target->kind);
    memset(target, 0, sizeof(*target));
}

static void nn_gd_arc_hole_target_list_free(NNGDArcHoleTargetList *list) {
    size_t i;
    for (i = 0U; i < list->count; ++i) {
        nn_gd_arc_hole_target_free(&list->items[i]);
    }
    free(list->items);
    memset(list, 0, sizeof(*list));
}

static int nn_gd_arc_hole_target_copy(NNGDArcHoleTarget *dst,
                                      const NNGDArcHoleTarget *src) {
    memset(dst, 0, sizeof(*dst));
    dst->task_id = nn_gd_strdup(src->task_id);
    dst->split = nn_gd_strdup(src->split);
    dst->source_scope = nn_gd_strdup(src->source_scope);
    dst->source_path = nn_gd_strdup(src->source_path);
    dst->kind = nn_gd_strdup(src->kind);
    dst->attempted = src->attempted;
    dst->train_consistent_pre_cap = src->train_consistent_pre_cap;
    dst->train_consistent_retained = src->train_consistent_retained;
    dst->selected = src->selected;
    if (dst->task_id == NULL || dst->split == NULL || dst->source_scope == NULL ||
        dst->source_path == NULL || dst->kind == NULL) {
        nn_gd_arc_hole_target_free(dst);
        return -1;
    }
    return 0;
}

static int nn_gd_arc_hole_target_list_add(NNGDArcHoleTargetList *list,
                                          const NNGDArcHoleTarget *target) {
    if (nn_gd_arc_hole_target_list_reserve(list, list->count + 1U) != 0) {
        return -1;
    }
    if (nn_gd_arc_hole_target_copy(&list->items[list->count], target) != 0) {
        return -1;
    }
    ++list->count;
    return 0;
}

static int nn_gd_arc_hole_target_list_upsert(NNGDArcHoleTargetList *list,
                                             const NNGDArcHoleTarget *target) {
    size_t i;
    for (i = 0U; i < list->count; ++i) {
        if (strcmp(list->items[i].task_id, target->task_id) == 0) {
            NNGDArcHoleTarget replacement;
            if (nn_gd_arc_hole_target_copy(&replacement, target) != 0) {
                return -1;
            }
            nn_gd_arc_hole_target_free(&list->items[i]);
            list->items[i] = replacement;
            return 0;
        }
    }
    return nn_gd_arc_hole_target_list_add(list, target);
}

static int nn_gd_arc_hole_compare_targets(const void *a, const void *b) {
    const NNGDArcHoleTarget *ta = (const NNGDArcHoleTarget *)a;
    const NNGDArcHoleTarget *tb = (const NNGDArcHoleTarget *)b;
    return strcmp(ta->task_id, tb->task_id);
}

static int nn_gd_arc_hole_candidate_list_reserve(NNGDArcHoleCandidateList *list,
                                                 size_t needed) {
    NNGDArcHoleCandidate *items;
    size_t capacity;
    if (needed <= list->capacity) {
        return 0;
    }
    capacity = list->capacity == 0U ? 32U : list->capacity * 2U;
    while (capacity < needed) {
        capacity *= 2U;
    }
    items = (NNGDArcHoleCandidate *)realloc(list->items,
                                            capacity * sizeof(*list->items));
    if (items == NULL) {
        return -1;
    }
    list->items = items;
    list->capacity = capacity;
    return 0;
}

static void nn_gd_arc_hole_candidate_free(NNGDArcHoleCandidate *candidate) {
    size_t i;
    for (i = 0U; i < 13U; ++i) {
        free(candidate->fields[i]);
        candidate->fields[i] = NULL;
    }
}

static void nn_gd_arc_hole_candidate_list_free(NNGDArcHoleCandidateList *list) {
    size_t i;
    for (i = 0U; i < list->count; ++i) {
        nn_gd_arc_hole_candidate_free(&list->items[i]);
    }
    free(list->items);
    memset(list, 0, sizeof(*list));
}

static int nn_gd_arc_hole_counter_list_reserve(NNGDArcHoleCounterList *list,
                                               size_t needed) {
    NNGDArcHoleCounter *items;
    size_t capacity;
    if (needed <= list->capacity) {
        return 0;
    }
    capacity = list->capacity == 0U ? 16U : list->capacity * 2U;
    while (capacity < needed) {
        capacity *= 2U;
    }
    items = (NNGDArcHoleCounter *)realloc(list->items,
                                          capacity * sizeof(*list->items));
    if (items == NULL) {
        return -1;
    }
    list->items = items;
    list->capacity = capacity;
    return 0;
}

static void nn_gd_arc_hole_counter_list_free(NNGDArcHoleCounterList *list) {
    size_t i;
    for (i = 0U; i < list->count; ++i) {
        free(list->items[i].key);
    }
    free(list->items);
    memset(list, 0, sizeof(*list));
}

static int nn_gd_arc_hole_counter_add(NNGDArcHoleCounterList *list,
                                      const char *key) {
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
    if (nn_gd_arc_hole_counter_list_reserve(list, list->count + 1U) != 0) {
        return -1;
    }
    list->items[list->count].key = nn_gd_strdup(key);
    if (list->items[list->count].key == NULL) {
        return -1;
    }
    list->items[list->count].count = 1U;
    ++list->count;
    return 0;
}

static int nn_gd_arc_hole_counter_compare_key(const void *a, const void *b) {
    const NNGDArcHoleCounter *ca = (const NNGDArcHoleCounter *)a;
    const NNGDArcHoleCounter *cb = (const NNGDArcHoleCounter *)b;
    return strcmp(ca->key, cb->key);
}

static const char *nn_gd_arc_hole_counter_dominant(
    const NNGDArcHoleCounterList *list,
    const char *fallback) {
    size_t i;
    size_t best = SIZE_MAX;
    for (i = 0U; i < list->count; ++i) {
        if (best == SIZE_MAX || list->items[i].count > list->items[best].count) {
            best = i;
        }
    }
    return best == SIZE_MAX ? fallback : list->items[best].key;
}

static int nn_gd_arc_hole_counter_add_candidates(
    NNGDArcHoleCounterList *strategy_counts,
    NNGDArcHoleCounterList *stage_counts,
    NNGDArcHoleCounterList *task_counts,
    const NNGDArcHoleCandidateList *candidates) {
    size_t i;
    for (i = 0U; i < candidates->count; ++i) {
        const NNGDArcHoleCandidate *candidate = &candidates->items[i];
        if (nn_gd_arc_hole_counter_add(strategy_counts, candidate->fields[12]) != 0 ||
            nn_gd_arc_hole_counter_add(stage_counts, candidate->fields[11]) != 0 ||
            nn_gd_arc_hole_counter_add(task_counts, candidate->fields[1]) != 0) {
            return -1;
        }
    }
    return 0;
}

static int nn_gd_arc_hole_scope_matches(const char *scope,
                                        const char *const *target_scopes,
                                        size_t target_scope_count) {
    size_t i;
    for (i = 0U; i < target_scope_count; ++i) {
        if (strcmp(scope, target_scopes[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

static int nn_gd_arc_hole_contains_ci(const char *value, const char *needle) {
    size_t needle_len;
    const char *p;
    if (value == NULL || needle == NULL) {
        return 0;
    }
    needle_len = strlen(needle);
    for (p = value; *p != '\0'; ++p) {
        size_t i;
        for (i = 0U; i < needle_len; ++i) {
            if (p[i] == '\0' ||
                tolower((unsigned char)p[i]) !=
                    tolower((unsigned char)needle[i])) {
                break;
            }
        }
        if (i == needle_len) {
            return 1;
        }
    }
    return 0;
}

static const char *nn_gd_arc_hole_split_for_scope(const char *scope) {
    if (nn_gd_arc_hole_contains_ci(scope, "train")) {
        return "public_training";
    }
    if (nn_gd_arc_hole_contains_ci(scope, "eval")) {
        return "public_evaluation";
    }
    return NULL;
}

static char *nn_gd_arc_hole_default_scope(size_t index) {
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "arc_visual_only_%zu", index + 1U);
    return nn_gd_strdup(buffer);
}

static int nn_gd_arc_hole_parse_scoped_path(const char *raw,
                                            size_t index,
                                            char **scope_out,
                                            char **path_out) {
    const char *eq;
    *scope_out = NULL;
    *path_out = NULL;
    if (raw == NULL || raw[0] == '\0') {
        return -1;
    }
    eq = strchr(raw, '=');
    if (eq != NULL && eq > raw && eq[1] != '\0') {
        size_t scope_len = (size_t)(eq - raw);
        *scope_out = (char *)malloc(scope_len + 1U);
        if (*scope_out == NULL) {
            return -1;
        }
        memcpy(*scope_out, raw, scope_len);
        (*scope_out)[scope_len] = '\0';
        *path_out = nn_gd_strdup(eq + 1);
    } else {
        *scope_out = nn_gd_arc_hole_default_scope(index);
        *path_out = nn_gd_strdup(raw);
    }
    if (*scope_out == NULL || *path_out == NULL) {
        free(*scope_out);
        free(*path_out);
        *scope_out = NULL;
        *path_out = NULL;
        return -1;
    }
    return 0;
}

static const char *nn_gd_arc_hole_json_object_end(const char *start) {
    const char *p;
    unsigned depth = 0U;
    if (start == NULL || *start != '{') {
        return NULL;
    }
    for (p = start; *p != '\0'; ++p) {
        if (*p == '"') {
            p = nn_gd_json_skip_string_literal(p);
            if (p == NULL || *p == '\0') {
                return p;
            }
            --p;
            continue;
        }
        if (*p == '{') {
            ++depth;
        } else if (*p == '}') {
            --depth;
            if (depth == 0U) {
                return p + 1;
            }
        }
    }
    return NULL;
}

static int nn_gd_arc_hole_parse_visual_template_stats(const char *line,
                                                       long *attempted,
                                                       long *train_consistent_pre_cap,
                                                       long *train_consistent_retained,
                                                       long *selected) {
    const char *name = strstr(line, "\"name\":\"arc_sketch_program\"");
    const char *start;
    const char *end;
    char *object;
    NNGDJsonIndex index;
    size_t len;
    *attempted = 0L;
    *train_consistent_pre_cap = 0L;
    *train_consistent_retained = 0L;
    *selected = 0L;
    if (name == NULL) {
        name = strstr(line, "arc_sketch_program");
    }
    if (name == NULL) {
        return 0;
    }
    start = name;
    while (start > line && *start != '{') {
        --start;
    }
    if (*start != '{') {
        return -1;
    }
    end = nn_gd_arc_hole_json_object_end(start);
    if (end == NULL || end <= start) {
        return -1;
    }
    len = (size_t)(end - start);
    object = (char *)malloc(len + 1U);
    if (object == NULL) {
        return -1;
    }
    memcpy(object, start, len);
    object[len] = '\0';
    nn_gd_json_index_line(object, &index);
    *attempted = nn_gd_json_index_long_or(&index, "attempted", 0L);
    *train_consistent_pre_cap =
        nn_gd_json_index_long_or(&index, "train_consistent_pre_cap", 0L);
    *train_consistent_retained =
        nn_gd_json_index_long_or(&index, "train_consistent_retained", 0L);
    *selected = nn_gd_json_index_long_or(&index, "selected", 0L);
    free(object);
    return 0;
}

static const char *nn_gd_arc_hole_task_kind(long selected,
                                            long train_consistent_pre_cap,
                                            long attempted) {
    if (selected > 0L) {
        return "visual_target_already_selected";
    }
    if (train_consistent_pre_cap > 0L) {
        return "visual_target_train_consistent_not_selected";
    }
    if (attempted > 0L) {
        return "visual_target_no_train_match";
    }
    return "visual_target_no_attempts";
}

static int nn_gd_arc_hole_slot_value_long(const char *slot_tokens,
                                          const char *slot_name,
                                          long *value_out) {
    const char *p = slot_tokens;
    size_t name_len = strlen(slot_name);
    if (slot_tokens == NULL || slot_name == NULL) {
        return 0;
    }
    while (*p != '\0') {
        const char *end = strchr(p, '|');
        size_t len = end == NULL ? strlen(p) : (size_t)(end - p);
        if (len > name_len + 1U && strncmp(p, slot_name, name_len) == 0 &&
            p[name_len] == '=') {
            char buffer[64];
            size_t value_len = len - name_len - 1U;
            if (value_len >= sizeof(buffer)) {
                value_len = sizeof(buffer) - 1U;
            }
            memcpy(buffer, p + name_len + 1U, value_len);
            buffer[value_len] = '\0';
            *value_out = nn_gd_parse_long(buffer);
            return 1;
        }
        if (end == NULL) {
            break;
        }
        p = end + 1;
    }
    return 0;
}

static int nn_gd_arc_hole_op_id_from_name(const char *name) {
    static const char *names[] = {
        "crop",
        "tile_repeat",
        "macrocell_substitute",
        "lattice_sample",
        "component_color_summary",
        "masked_or_normalized_crop",
        "point_line_project",
        "object_mask_extract",
        "axis_projection_compress",
        "separator_panel_map",
        "palette_scale",
        "object_color_tile",
        "marker_count_motif_tile",
        "component_count_line",
        "recolor_crop",
        "marker_polyline",
        "separator_intersection_color",
        "separator_panel_relation_mask",
        "visual_combinator",
    };
    size_t i;
    if (name == NULL) {
        return -1;
    }
    for (i = 0U; i < sizeof(names) / sizeof(names[0]); ++i) {
        if (strcmp(name, names[i]) == 0) {
            return (int)i;
        }
    }
    return -1;
}

static const char *nn_gd_arc_hole_op_name_from_id(long op_id) {
    static const char *names[] = {
        "crop",
        "tile_repeat",
        "macrocell_substitute",
        "lattice_sample",
        "component_color_summary",
        "masked_or_normalized_crop",
        "point_line_project",
        "object_mask_extract",
        "axis_projection_compress",
        "separator_panel_map",
        "palette_scale",
        "object_color_tile",
        "marker_count_motif_tile",
        "component_count_line",
        "recolor_crop",
        "marker_polyline",
        "separator_intersection_color",
        "separator_panel_relation_mask",
        "visual_combinator",
    };
    if (op_id < 0L || (size_t)op_id >= sizeof(names) / sizeof(names[0])) {
        return "";
    }
    return names[op_id];
}

static int nn_gd_arc_hole_attempt_op_id(const NNGDArcPipelineAttempt *attempt) {
    long value = -1L;
    if (nn_gd_arc_hole_slot_value_long(attempt->slot_tokens, "sketch.op", &value) &&
        value >= 0L) {
        return (int)value;
    }
    return nn_gd_arc_hole_op_id_from_name(attempt->op_name);
}

static int nn_gd_arc_hole_add_pin(NNGDStringList *tokens,
                                  const char *pin_name,
                                  long value) {
    char buffer[96];
    snprintf(buffer, sizeof(buffer), "pin:%s=%ld", pin_name, value);
    return nn_gd_string_list_add_unique(tokens, buffer);
}

static int nn_gd_arc_hole_add_op_tokens(NNGDStringList *tokens,
                                        const NNGDArcPipelineAttempt *attempt,
                                        int strict) {
    int op_id;
    if (attempt->op_name != NULL && attempt->op_name[0] != '\0' &&
        nn_gd_string_list_add_unique(tokens, attempt->op_name) != 0) {
        return -1;
    }
    op_id = nn_gd_arc_hole_attempt_op_id(attempt);
    if (strict && op_id >= 0) {
        return nn_gd_arc_hole_add_pin(tokens, "op", (long)op_id);
    }
    return 0;
}

static int nn_gd_arc_hole_add_structural_pins(NNGDStringList *tokens,
                                              const NNGDArcPipelineAttempt *attempt) {
    return nn_gd_arc_hole_add_pin(tokens, "selector", attempt->selector) != 0 ||
                   nn_gd_arc_hole_add_pin(tokens, "x_anchor", attempt->x_anchor) != 0 ||
                   nn_gd_arc_hole_add_pin(tokens, "y_anchor", attempt->y_anchor) != 0 ||
                   nn_gd_arc_hole_add_pin(tokens, "x_offset", attempt->x_offset) != 0 ||
                   nn_gd_arc_hole_add_pin(tokens, "y_offset", attempt->y_offset) != 0 ||
                   nn_gd_arc_hole_add_pin(tokens, "period", attempt->period) != 0
               ? -1
               : 0;
}

static int nn_gd_arc_hole_add_geometry_pins(NNGDStringList *tokens,
                                            const NNGDArcPipelineAttempt *attempt) {
    return nn_gd_arc_hole_add_structural_pins(tokens, attempt) != 0 ||
                   nn_gd_arc_hole_add_pin(tokens, "width", attempt->width) != 0 ||
                   nn_gd_arc_hole_add_pin(tokens, "height", attempt->height) != 0
               ? -1
               : 0;
}

static int nn_gd_arc_hole_add_stage_tokens(NNGDStringList *tokens,
                                           const char *stage) {
    if (strcmp(stage, "selector_or_transform") == 0) {
        return nn_gd_string_list_add_unique(tokens,
                                            "counterexample_selector_phase_repair") != 0 ||
                       nn_gd_string_list_add_unique(tokens,
                                                    "selector:component_predicate") != 0 ||
                       nn_gd_string_list_add_unique(
                           tokens,
                           "normalizer:component_selector_layout") != 0
                   ? -1
                   : 0;
    }
    if (strcmp(stage, "renderer_shape") == 0) {
        return nn_gd_string_list_add_unique(tokens,
                                            "counterexample_shape_frame_repair") != 0 ||
                       nn_gd_string_list_add_unique(tokens,
                                                    "normalizer:object_lattice") != 0 ||
                       nn_gd_string_list_add_unique(tokens, "object_lattice_frame") != 0
                   ? -1
                   : 0;
    }
    if (strcmp(stage, "renderer_cells") == 0) {
        return nn_gd_string_list_add_unique(
                   tokens,
                   "counterexample_cell_mask_cleanup_repair") != 0 ||
                       nn_gd_string_list_add_unique(tokens, "emitter:component_mask") != 0 ||
                       nn_gd_string_list_add_unique(tokens,
                                                    "normalizer:component_mask") != 0
                   ? -1
                   : 0;
    }
    if (strcmp(stage, "transform_or_color") == 0) {
        return nn_gd_string_list_add_unique(
                   tokens,
                   "counterexample_transform_color_repair") != 0 ||
                       nn_gd_string_list_add_unique(tokens, "emitter:component_mask") != 0
                   ? -1
                   : 0;
    }
    if (strcmp(stage, "unsupported_shape") == 0) {
        return nn_gd_string_list_add_unique(tokens,
                                            "counterexample_shape_frame_repair");
    }
    return 0;
}

static int nn_gd_arc_hole_add_shape_hint_tokens(
    NNGDStringList *tokens,
    const NNGDArcPipelineAttempt *features) {
    if (features == NULL) {
        return 0;
    }
    if (features->shape_scale_x > 1L &&
        features->shape_scale_x == features->shape_scale_y) {
        if (nn_gd_string_list_add_unique(tokens, "macrocell_substitute") != 0 ||
            nn_gd_string_list_add_unique(tokens, "palette_scale") != 0 ||
            nn_gd_string_list_add_unique(tokens, "lattice_sample") != 0 ||
            nn_gd_arc_hole_add_pin(tokens, "period", features->shape_scale_x) != 0) {
            return -1;
        }
    }
    if (features->input_width > 0L && features->input_height > 0L &&
        features->output_width > 0L && features->output_height > 0L) {
        if (features->output_width < features->input_width ||
            features->output_height < features->input_height) {
            if (nn_gd_string_list_add_unique(tokens, "crop") != 0 ||
                nn_gd_string_list_add_unique(tokens,
                                             "masked_or_normalized_crop") != 0 ||
                nn_gd_string_list_add_unique(tokens, "object_mask_extract") != 0 ||
                nn_gd_string_list_add_unique(tokens, "lattice_sample") != 0 ||
                nn_gd_string_list_add_unique(tokens,
                                             "axis_projection_compress") != 0) {
                return -1;
            }
        }
        if (features->output_width == 1L || features->output_height == 1L) {
            if (nn_gd_string_list_add_unique(tokens,
                                             "axis_projection_compress") != 0 ||
                nn_gd_string_list_add_unique(tokens, "component_count_line") != 0 ||
                nn_gd_string_list_add_unique(tokens,
                                             "component_color_summary") != 0) {
                return -1;
            }
        }
    }
    return 0;
}

static char *nn_gd_arc_hole_compact_tokens(const NNGDStringList *tokens) {
    size_t capacity = 256U;
    size_t used = 0U;
    char *joined = (char *)malloc(capacity);
    size_t i;
    if (joined == NULL) {
        return NULL;
    }
    joined[0] = '\0';
    for (i = 0U; i < tokens->count; ++i) {
        const char *token = tokens->items[i];
        size_t token_len;
        size_t extra;
        if (token == NULL || token[0] == '\0') {
            continue;
        }
        token_len = strlen(token);
        extra = token_len + (used == 0U ? 0U : 1U);
        if (used + extra > 240U) {
            continue;
        }
        if (used + extra + 1U > capacity) {
            size_t next_capacity = capacity * 2U;
            char *next;
            while (used + extra + 1U > next_capacity) {
                next_capacity *= 2U;
            }
            next = (char *)realloc(joined, next_capacity);
            if (next == NULL) {
                free(joined);
                return NULL;
            }
            joined = next;
            capacity = next_capacity;
        }
        if (used > 0U) {
            joined[used++] = ',';
        }
        memcpy(joined + used, token, token_len);
        used += token_len;
        joined[used] = '\0';
    }
    return joined;
}

static char *nn_gd_arc_hole_frontier(const char *source_scope) {
    size_t len = strlen("arc_visual_hole:") + strlen(source_scope) + 1U;
    char *value = (char *)malloc(len);
    if (value == NULL) {
        return NULL;
    }
    snprintf(value, len, "arc_visual_hole:%s", source_scope);
    return value;
}

static const char *nn_gd_arc_hole_attempt_id(const NNGDArcPipelineAttempt *attempt) {
    if (attempt->attempt_id != NULL && attempt->attempt_id[0] != '\0') {
        return attempt->attempt_id;
    }
    return attempt->library_id == NULL ? "" : attempt->library_id;
}

static int nn_gd_arc_hole_candidate_duplicate(
    const NNGDArcHoleCandidateList *list,
    const char *task_id,
    const char *split,
    const char *op_hints,
    const char *strategy) {
    size_t i;
    for (i = 0U; i < list->count; ++i) {
        const NNGDArcHoleCandidate *candidate = &list->items[i];
        if (strcmp(candidate->fields[1], task_id) == 0 &&
            strcmp(candidate->fields[2], split) == 0 &&
            strcmp(candidate->fields[6], op_hints) == 0 &&
            strcmp(candidate->fields[12], strategy) == 0) {
            return 1;
        }
    }
    return 0;
}

static int nn_gd_arc_hole_candidate_add(NNGDArcHoleCandidateList *list,
                                        const NNGDArcHoleTarget *target,
                                        const NNGDStringList *tokens,
                                        const char *frontier,
                                        const char *parent,
                                        const char *operator_name,
                                        const char *stage,
                                        const char *strategy,
                                        long proposal_count,
                                        double utility) {
    static const char *family = "arc.visual_hole_proposer_family";
    char *op_hints = nn_gd_arc_hole_compact_tokens(tokens);
    char support_tasks[32];
    char proposal_buffer[32];
    char utility_buffer[64];
    const char *fields[13];
    NNGDArcHoleCandidate *candidate;
    size_t i;
    if (op_hints == NULL) {
        return -1;
    }
    if (target->task_id[0] == '\0' || target->split[0] == '\0' ||
        op_hints[0] == '\0') {
        free(op_hints);
        return 0;
    }
    if (nn_gd_arc_hole_candidate_duplicate(list,
                                           target->task_id,
                                           target->split,
                                           op_hints,
                                           strategy)) {
        free(op_hints);
        return 0;
    }
    if (proposal_count < 1L) {
        proposal_count = 1L;
    }
    snprintf(support_tasks, sizeof(support_tasks), "%d", 1);
    snprintf(proposal_buffer, sizeof(proposal_buffer), "%ld", proposal_count);
    snprintf(utility_buffer, sizeof(utility_buffer), "%.8f", utility);
    fields[0] = family;
    fields[1] = target->task_id;
    fields[2] = target->split;
    fields[3] = support_tasks;
    fields[4] = proposal_buffer;
    fields[5] = utility_buffer;
    fields[6] = op_hints;
    fields[7] = frontier;
    fields[8] = parent == NULL ? "" : parent;
    fields[9] = operator_name;
    fields[10] = target->kind;
    fields[11] = stage;
    fields[12] = strategy;
    if (nn_gd_arc_hole_candidate_list_reserve(list, list->count + 1U) != 0) {
        free(op_hints);
        return -1;
    }
    candidate = &list->items[list->count];
    memset(candidate, 0, sizeof(*candidate));
    for (i = 0U; i < 13U; ++i) {
        candidate->fields[i] = nn_gd_strdup(fields[i]);
        if (candidate->fields[i] == NULL) {
            nn_gd_arc_hole_candidate_free(candidate);
            free(op_hints);
            return -1;
        }
    }
    ++list->count;
    free(op_hints);
    return 0;
}

static int nn_gd_arc_hole_write_target_example(FILE *fp,
                                               const NNGDArcHoleTarget *target) {
    fputs("{\"schema\":\"crlp.neural_grammar_discovery.arc_visual_hole_example.v1\","
          "\"report_version\":1,\"kind\":",
          fp);
    nn_gd_json_write_string(fp, target->kind);
    fputs(",\"source_scope\":", fp);
    nn_gd_json_write_string(fp, target->source_scope);
    fputs(",\"source_path\":", fp);
    nn_gd_json_write_string(fp, target->source_path);
    fputs(",\"split\":", fp);
    nn_gd_json_write_string(fp, target->split);
    fputs(",\"task_id\":", fp);
    nn_gd_json_write_string(fp, target->task_id);
    fprintf(fp,
            ",\"attempted\":%ld,\"train_consistent_pre_cap\":%ld,"
            "\"train_consistent_retained\":%ld,\"selected\":%ld}\n",
            target->attempted,
            target->train_consistent_pre_cap,
            target->train_consistent_retained,
            target->selected);
    return ferror(fp) ? -1 : 0;
}

static int nn_gd_arc_hole_write_nearest_example(
    FILE *fp,
    const NNGDArcHoleTarget *target,
    const NNGDArcPipelineAttempt *source,
    unsigned rank,
    double distance,
    const NNGDArcHoleCandidate *candidate) {
    fputs("{\"schema\":\"crlp.neural_grammar_discovery.arc_visual_hole_example.v1\","
          "\"report_version\":1,\"kind\":\"nearest_positive_signature\","
          "\"target_task_id\":",
          fp);
    nn_gd_json_write_string(fp, target->task_id);
    fputs(",\"source_task_id\":", fp);
    nn_gd_json_write_string(fp, source->task_id);
    fputs(",\"source_attempt_id\":", fp);
    nn_gd_json_write_string(fp, nn_gd_arc_hole_attempt_id(source));
    fprintf(fp, ",\"rank\":%u,\"distance\":%.8f,\"op_hints\":", rank, distance);
    nn_gd_json_write_string(fp, candidate->fields[6]);
    fputs("}\n", fp);
    return ferror(fp) ? -1 : 0;
}

static int nn_gd_arc_hole_write_rejected_example(
    FILE *fp,
    const NNGDArcHoleTarget *target,
    const NNGDArcPipelineAttempt *attempt,
    unsigned rank,
    const char *stage) {
    fputs("{\"schema\":\"crlp.neural_grammar_discovery.arc_visual_hole_example.v1\","
          "\"report_version\":1,\"kind\":\"rejected_attempt_seed\","
          "\"target_task_id\":",
          fp);
    nn_gd_json_write_string(fp, target->task_id);
    fputs(",\"attempt_id\":", fp);
    nn_gd_json_write_string(fp, nn_gd_arc_hole_attempt_id(attempt));
    fprintf(fp, ",\"rank\":%u,\"op_name\":", rank);
    nn_gd_json_write_string(fp, attempt->op_name);
    fputs(",\"failure_stage\":", fp);
    nn_gd_json_write_string(fp, stage);
    fprintf(fp,
            ",\"attempt_score\":[%.8f,%ld,%.8f,%ld],\"slot_tokens\":",
            attempt->verifier_score,
            attempt->verifier_checked_cells,
            -attempt->verifier_mismatch_ratio,
            -attempt->verifier_unsupported_pairs);
    nn_gd_json_write_string(fp, attempt->slot_tokens);
    fputs("}\n", fp);
    return ferror(fp) ? -1 : 0;
}

static const NNGDArcPipelineAttempt *nn_gd_arc_hole_task_features(
    const NNGDArcPipelineAttemptList *attempts,
    const char *task_id) {
    size_t i;
    for (i = 0U; i < attempts->count; ++i) {
        if (strcmp(attempts->items[i].task_id, task_id) == 0) {
            return &attempts->items[i];
        }
    }
    return NULL;
}

static double nn_gd_arc_hole_feature_distance(
    const NNGDArcPipelineAttempt *target,
    const NNGDArcPipelineAttempt *source) {
    double distance = 0.0;
    if (target == NULL || source == NULL) {
        return 1000.0;
    }
    distance += fabs((double)target->input_width - (double)source->input_width);
    distance += fabs((double)target->input_height - (double)source->input_height);
    distance += fabs((double)target->output_width - (double)source->output_width);
    distance += fabs((double)target->output_height - (double)source->output_height);
    distance += fabs((double)target->shape_scale_x - (double)source->shape_scale_x);
    distance += fabs((double)target->shape_scale_y - (double)source->shape_scale_y);
    if (target->slot_width_matches_output != source->slot_width_matches_output) {
        distance += 1.0;
    }
    if (target->slot_height_matches_output != source->slot_height_matches_output) {
        distance += 1.0;
    }
    if (target->period_divides_input_width != source->period_divides_input_width) {
        distance += 1.0;
    }
    if (target->period_divides_input_height != source->period_divides_input_height) {
        distance += 1.0;
    }
    return distance;
}

static int nn_gd_arc_hole_attempt_score_better(
    const NNGDArcPipelineAttempt *candidate,
    const NNGDArcPipelineAttempt *best) {
    if (best == NULL) {
        return 1;
    }
    if (candidate->verifier_score != best->verifier_score) {
        return candidate->verifier_score > best->verifier_score;
    }
    if (candidate->verifier_checked_cells != best->verifier_checked_cells) {
        return candidate->verifier_checked_cells > best->verifier_checked_cells;
    }
    if (candidate->verifier_mismatch_ratio != best->verifier_mismatch_ratio) {
        return candidate->verifier_mismatch_ratio < best->verifier_mismatch_ratio;
    }
    return candidate->verifier_unsupported_pairs < best->verifier_unsupported_pairs;
}

static int nn_gd_arc_hole_index_selected(const size_t *indices,
                                         size_t count,
                                         size_t index) {
    size_t i;
    for (i = 0U; i < count; ++i) {
        if (indices[i] == index) {
            return 1;
        }
    }
    return 0;
}

static int nn_gd_arc_hole_write_candidate_tsv(
    const char *path,
    const NNGDArcHoleCandidateList *candidates) {
    static const char *header[13] = {
        "family",
        "task_id",
        "split",
        "support_tasks",
        "proposal_count",
        "utility_mean",
        "op_hints",
        "coverage_frontier_id",
        "parent_attempt_id",
        "repair_operator",
        "dominant_residual_class",
        "dominant_failure_stage",
        "repair_strategy",
    };
    FILE *fp = fopen(path, "w");
    size_t i;
    if (fp == NULL) {
        return -1;
    }
    for (i = 0U; i < 13U; ++i) {
        if (i > 0U) {
            fputc('\t', fp);
        }
        fputs(header[i], fp);
    }
    fputc('\n', fp);
    for (i = 0U; i < candidates->count; ++i) {
        size_t field_index;
        for (field_index = 0U; field_index < 13U; ++field_index) {
            if (field_index > 0U) {
                fputc('\t', fp);
            }
            fputs(candidates->items[i].fields[field_index], fp);
        }
        fputc('\n', fp);
    }
    if (fclose(fp) != 0) {
        return -1;
    }
    return 0;
}

static int nn_gd_arc_hole_write_counter_object(FILE *fp,
                                               NNGDArcHoleCounterList *list) {
    size_t i;
    if (list->count > 1U) {
        qsort(list->items,
              list->count,
              sizeof(list->items[0]),
              nn_gd_arc_hole_counter_compare_key);
    }
    fputc('{', fp);
    for (i = 0U; i < list->count; ++i) {
        if (i > 0U) {
            fputc(',', fp);
        }
        nn_gd_json_write_string(fp, list->items[i].key);
        fprintf(fp, ":%zu", list->items[i].count);
    }
    fputc('}', fp);
    return ferror(fp) ? -1 : 0;
}

static int nn_gd_arc_hole_write_string_array(FILE *fp,
                                             const char *const *values,
                                             size_t count) {
    size_t i;
    fputc('[', fp);
    for (i = 0U; i < count; ++i) {
        if (i > 0U) {
            fputc(',', fp);
        }
        nn_gd_json_write_string(fp, values[i]);
    }
    fputc(']', fp);
    return ferror(fp) ? -1 : 0;
}

static int nn_gd_arc_hole_write_summary(
    const char *path,
    const NNGDArcHoleSourceLogList *source_logs,
    const char *const *attempt_logs,
    size_t attempt_log_count,
    const char *const *target_scopes,
    size_t target_scope_count,
    const char *candidate_tsv_path,
    const char *examples_path,
    size_t target_task_count,
    const NNGDArcPipelineAttemptList *attempts,
    const NNGDArcHoleCandidateList *candidates) {
    FILE *fp = fopen(path, "w");
    NNGDArcHoleCounterList accepted_ops = {0};
    NNGDArcHoleCounterList rejected_stages = {0};
    NNGDArcHoleCounterList rows_by_strategy = {0};
    NNGDArcHoleCounterList rows_by_stage = {0};
    NNGDArcHoleCounterList rows_by_task = {0};
    size_t accepted_count = 0U;
    size_t rejected_count = 0U;
    size_t i;
    int status = -1;
    if (fp == NULL) {
        return -1;
    }
    for (i = 0U; i < attempts->count; ++i) {
        const NNGDArcPipelineAttempt *attempt = &attempts->items[i];
        if (attempt->accepted) {
            ++accepted_count;
            if (nn_gd_arc_hole_counter_add(&accepted_ops, attempt->op_name) != 0) {
                goto done;
            }
        } else {
            const char *stage = attempt->failure_stage[0] != '\0'
                                    ? attempt->failure_stage
                                    : (attempt->rejection_reason[0] != '\0'
                                           ? attempt->rejection_reason
                                           : "rejected");
            ++rejected_count;
            if (nn_gd_arc_hole_counter_add(&rejected_stages, stage) != 0) {
                goto done;
            }
        }
    }
    if (nn_gd_arc_hole_counter_add_candidates(&rows_by_strategy,
                                              &rows_by_stage,
                                              &rows_by_task,
                                              candidates) != 0) {
        goto done;
    }
    fputs("{\"schema\":\"crlp.neural_grammar_discovery.arc_visual_hole_summary.v1\","
          "\"report_version\":1,\"source_logs\":[",
          fp);
    for (i = 0U; i < source_logs->count; ++i) {
        const NNGDArcHoleSourceLog *source = &source_logs->items[i];
        if (i > 0U) {
            fputc(',', fp);
        }
        fputs("{\"scope\":", fp);
        nn_gd_json_write_string(fp, source->scope);
        fputs(",\"path\":", fp);
        nn_gd_json_write_string(fp, source->path);
        fprintf(fp, ",\"rows\":%zu}", source->rows);
    }
    fputs("],\"attempt_logs\":", fp);
    if (nn_gd_arc_hole_write_string_array(fp, attempt_logs, attempt_log_count) != 0) {
        goto done;
    }
    fputs(",\"target_scopes\":", fp);
    if (nn_gd_arc_hole_write_string_array(fp, target_scopes, target_scope_count) != 0) {
        goto done;
    }
    fprintf(fp,
            ",\"target_task_count\":%zu,\"candidate_count\":%zu,"
            "\"candidate_tsv\":",
            target_task_count,
            candidates->count);
    nn_gd_json_write_string(fp, candidate_tsv_path);
    fputs(",\"example_jsonl\":", fp);
    nn_gd_json_write_string(fp, examples_path);
    fprintf(fp,
            ",\"accepted_attempts\":%zu,\"accepted_ops\":",
            accepted_count);
    if (nn_gd_arc_hole_write_counter_object(fp, &accepted_ops) != 0) {
        goto done;
    }
    fprintf(fp, ",\"rejected_attempts\":%zu,\"rejected_stages\":", rejected_count);
    if (nn_gd_arc_hole_write_counter_object(fp, &rejected_stages) != 0) {
        goto done;
    }
    fputs(",\"candidate_rows_by_strategy\":", fp);
    if (nn_gd_arc_hole_write_counter_object(fp, &rows_by_strategy) != 0) {
        goto done;
    }
    fputs(",\"candidate_rows_by_stage\":", fp);
    if (nn_gd_arc_hole_write_counter_object(fp, &rows_by_stage) != 0) {
        goto done;
    }
    fputs(",\"candidate_rows_by_task\":", fp);
    if (nn_gd_arc_hole_write_counter_object(fp, &rows_by_task) != 0) {
        goto done;
    }
    fputs("}\n", fp);
    status = ferror(fp) ? -1 : 0;

done:
    nn_gd_arc_hole_counter_list_free(&accepted_ops);
    nn_gd_arc_hole_counter_list_free(&rejected_stages);
    nn_gd_arc_hole_counter_list_free(&rows_by_strategy);
    nn_gd_arc_hole_counter_list_free(&rows_by_stage);
    nn_gd_arc_hole_counter_list_free(&rows_by_task);
    if (fclose(fp) != 0) {
        status = -1;
    }
    return status;
}

static int nn_gd_arc_hole_read_visual_log(const char *scope,
                                          const char *path,
                                          const char *const *target_scopes,
                                          size_t target_scope_count,
                                          int include_selected,
                                          NNGDArcHoleTargetList *examples,
                                          NNGDArcHoleTargetList *targets,
                                          size_t *rows_out) {
    FILE *fp = fopen(path, "r");
    const char *split = nn_gd_arc_hole_split_for_scope(scope);
    int target_scope = nn_gd_arc_hole_scope_matches(scope,
                                                    target_scopes,
                                                    target_scope_count);
    if (fp == NULL) {
        return -1;
    }
    *rows_out = 0U;
    for (;;) {
        char *line = nn_gd_read_line_dynamic(fp);
        char *schema;
        char *task_id = NULL;
        NNGDArcHoleTarget target;
        long attempted;
        long train_consistent_pre_cap;
        long train_consistent_retained;
        long selected;
        if (line == NULL) {
            break;
        }
        if (line[0] != '\0' && line[0] != '\n' && line[0] != '\r') {
            ++*rows_out;
        }
        if (!target_scope || split == NULL) {
            free(line);
            continue;
        }
        schema = nn_gd_json_dup_key_string(line, "schema");
        if (schema == NULL) {
            free(line);
            fclose(fp);
            return -1;
        }
        if (strcmp(schema, "crlp.arc_hypothesis.selection.v1") != 0) {
            free(schema);
            free(line);
            continue;
        }
        free(schema);
        if (nn_gd_arc_hole_parse_visual_template_stats(line,
                                                       &attempted,
                                                       &train_consistent_pre_cap,
                                                       &train_consistent_retained,
                                                       &selected) != 0) {
            free(line);
            fclose(fp);
            return -1;
        }
        task_id = nn_gd_json_dup_key_string(line, "task_instance");
        if (task_id == NULL) {
            free(line);
            fclose(fp);
            return -1;
        }
        memset(&target, 0, sizeof(target));
        target.task_id = task_id;
        target.split = (char *)split;
        target.source_scope = (char *)scope;
        target.source_path = (char *)path;
        target.kind = (char *)nn_gd_arc_hole_task_kind(selected,
                                                       train_consistent_pre_cap,
                                                       attempted);
        target.attempted = attempted;
        target.train_consistent_pre_cap = train_consistent_pre_cap;
        target.train_consistent_retained = train_consistent_retained;
        target.selected = selected;
        if (nn_gd_arc_hole_target_list_add(examples, &target) != 0) {
            free(task_id);
            free(line);
            fclose(fp);
            return -1;
        }
        if ((include_selected || strcmp(target.kind,
                                        "visual_target_already_selected") != 0) &&
            nn_gd_arc_hole_target_list_upsert(targets, &target) != 0) {
            free(task_id);
            free(line);
            fclose(fp);
            return -1;
        }
        free(task_id);
        free(line);
    }
    if (fclose(fp) != 0) {
        return -1;
    }
    return 0;
}

static int nn_gd_arc_hole_load_sources(const char *const *visual_only_logs,
                                       size_t visual_only_log_count,
                                       const char *const *target_scopes,
                                       size_t target_scope_count,
                                       int include_selected,
                                       NNGDArcHoleSourceLogList *sources,
                                       NNGDArcHoleTargetList *examples,
                                       NNGDArcHoleTargetList *targets) {
    size_t i;
    for (i = 0U; i < visual_only_log_count; ++i) {
        char *scope = NULL;
        char *path = NULL;
        size_t rows = 0U;
        if (nn_gd_arc_hole_parse_scoped_path(visual_only_logs[i],
                                             i,
                                             &scope,
                                             &path) != 0) {
            return -1;
        }
        if (nn_gd_arc_hole_read_visual_log(scope,
                                           path,
                                           target_scopes,
                                           target_scope_count,
                                           include_selected,
                                           examples,
                                           targets,
                                           &rows) != 0) {
            free(scope);
            free(path);
            return -1;
        }
        if (nn_gd_arc_hole_source_log_list_reserve(sources,
                                                   sources->count + 1U) != 0) {
            free(scope);
            free(path);
            return -1;
        }
        sources->items[sources->count].scope = scope;
        sources->items[sources->count].path = path;
        sources->items[sources->count].rows = rows;
        ++sources->count;
    }
    return 0;
}

static int nn_gd_arc_hole_load_attempts(const char *const *attempt_logs,
                                        size_t attempt_log_count,
                                        NNGDArcPipelineAttemptList *attempts) {
    size_t i;
    for (i = 0U; i < attempt_log_count; ++i) {
        if (nn_gd_read_arc_pipeline_attempts_jsonl(attempt_logs[i],
                                                   attempt_logs[i],
                                                   attempts) != 0) {
            return -1;
        }
    }
    return 0;
}

static int nn_gd_arc_hole_emit_shape_candidate(
    NNGDArcHoleCandidateList *candidates,
    const NNGDArcHoleTarget *target,
    const NNGDArcPipelineAttempt *features,
    const char *frontier) {
    NNGDStringList tokens = {0};
    int status = 0;
    if (nn_gd_arc_hole_add_shape_hint_tokens(&tokens, features) != 0) {
        status = -1;
        goto done;
    }
    if (tokens.count > 0U &&
        nn_gd_arc_hole_candidate_add(candidates,
                                     target,
                                     &tokens,
                                     frontier,
                                     "",
                                     "shape_feature_frontier",
                                     "shape_feature_gap",
                                     "route_shape_scale_and_compression_ops",
                                     64L,
                                     0.95) != 0) {
        status = -1;
    }

done:
    nn_gd_string_list_free(&tokens);
    return status;
}

static int nn_gd_arc_hole_emit_transfer_candidates(
    NNGDArcHoleCandidateList *candidates,
    FILE *examples_fp,
    const NNGDArcHoleTarget *target,
    const NNGDArcPipelineAttempt *target_features,
    const NNGDArcPipelineAttemptList *attempts,
    const char *frontier,
    unsigned nearest_signatures) {
    size_t *selected = NULL;
    unsigned rank;
    int status = -1;
    if (nearest_signatures == 0U) {
        return 0;
    }
    selected = (size_t *)calloc(nearest_signatures, sizeof(*selected));
    if (selected == NULL) {
        return -1;
    }
    for (rank = 0U; rank < nearest_signatures; ++rank) {
        const NNGDArcPipelineAttempt *best = NULL;
        size_t best_index = SIZE_MAX;
        double best_distance = 0.0;
        size_t i;
        for (i = 0U; i < attempts->count; ++i) {
            const NNGDArcPipelineAttempt *attempt = &attempts->items[i];
            double distance;
            if (!attempt->accepted || nn_gd_arc_hole_index_selected(selected, rank, i)) {
                continue;
            }
            distance = nn_gd_arc_hole_feature_distance(target_features, attempt);
            if (best == NULL || distance < best_distance) {
                best = attempt;
                best_index = i;
                best_distance = distance;
            }
        }
        if (best == NULL) {
            break;
        }
        selected[rank] = best_index;
        {
            NNGDStringList tokens = {0};
            size_t before_count = candidates->count;
            if (nn_gd_arc_hole_add_op_tokens(&tokens, best, 1) != 0 ||
                nn_gd_arc_hole_add_structural_pins(&tokens, best) != 0 ||
                nn_gd_arc_hole_candidate_add(candidates,
                                             target,
                                             &tokens,
                                             frontier,
                                             nn_gd_arc_hole_attempt_id(best),
                                             "nearest_positive_slot_transfer",
                                             "nearest_positive_transfer",
                                             "transfer_verified_structural_slots",
                                             32L,
                                             1.0 / (double)(rank + 1U)) != 0) {
                nn_gd_string_list_free(&tokens);
                goto done;
            }
            nn_gd_string_list_free(&tokens);
            if (candidates->count > before_count &&
                nn_gd_arc_hole_write_nearest_example(
                    examples_fp,
                    target,
                    best,
                    rank + 1U,
                    best_distance,
                    &candidates->items[candidates->count - 1U]) != 0) {
                goto done;
            }
        }
    }
    status = 0;

done:
    free(selected);
    return status;
}

static int nn_gd_arc_hole_collect_rejected_surface(
    const NNGDArcPipelineAttemptList *attempts,
    const char *task_id,
    NNGDArcHoleCounterList *stage_counts,
    NNGDArcHoleCounterList *op_counts) {
    size_t i;
    for (i = 0U; i < attempts->count; ++i) {
        const NNGDArcPipelineAttempt *attempt = &attempts->items[i];
        const char *stage;
        if (attempt->accepted || strcmp(attempt->task_id, task_id) != 0) {
            continue;
        }
        stage = attempt->failure_stage[0] != '\0'
                    ? attempt->failure_stage
                    : (attempt->rejection_reason[0] != '\0' ? attempt->rejection_reason
                                                             : "rejected");
        if (nn_gd_arc_hole_counter_add(stage_counts, stage) != 0 ||
            nn_gd_arc_hole_counter_add(op_counts,
                                       attempt->op_name[0] == '\0' ? "unknown"
                                                                   : attempt->op_name) != 0) {
            return -1;
        }
    }
    return 0;
}

static int nn_gd_arc_hole_add_dominant_ops(NNGDStringList *tokens,
                                           const NNGDArcHoleCounterList *op_counts,
                                           unsigned max_ops) {
    size_t selected[32];
    unsigned rank;
    if (max_ops > sizeof(selected) / sizeof(selected[0])) {
        max_ops = (unsigned)(sizeof(selected) / sizeof(selected[0]));
    }
    for (rank = 0U; rank < max_ops; ++rank) {
        size_t i;
        size_t best = SIZE_MAX;
        for (i = 0U; i < op_counts->count; ++i) {
            if (nn_gd_arc_hole_index_selected(selected, rank, i)) {
                continue;
            }
            if (best == SIZE_MAX ||
                op_counts->items[i].count > op_counts->items[best].count) {
                best = i;
            }
        }
        if (best == SIZE_MAX) {
            break;
        }
        selected[rank] = best;
        if (nn_gd_arc_hole_op_id_from_name(op_counts->items[best].key) >= 0 &&
            nn_gd_string_list_add_unique(tokens, op_counts->items[best].key) != 0) {
            return -1;
        }
    }
    return 0;
}

static int nn_gd_arc_hole_emit_special_shape_repairs(
    NNGDArcHoleCandidateList *candidates,
    const NNGDArcHoleTarget *target,
    const NNGDArcPipelineAttempt *attempt,
    const char *frontier,
    unsigned rank,
    const char *stage) {
    NNGDStringList tokens = {0};
    int status = -1;
    if (strcmp(stage, "renderer_shape") != 0 ||
        nn_gd_arc_hole_attempt_op_id(attempt) != 18) {
        return 0;
    }
    if (nn_gd_arc_hole_add_op_tokens(&tokens, attempt, 1) != 0 ||
        nn_gd_arc_hole_add_structural_pins(&tokens, attempt) != 0 ||
        nn_gd_string_list_add_unique(&tokens, "counterexample_shape_frame_repair") != 0 ||
        nn_gd_string_list_add_unique(&tokens, "normalizer:object_lattice") != 0 ||
        nn_gd_string_list_add_unique(&tokens, "object_lattice_frame") != 0 ||
        nn_gd_arc_hole_add_pin(&tokens, "width", 0L) != 0 ||
        nn_gd_arc_hole_add_pin(&tokens, "height", 0L) != 0 ||
        nn_gd_arc_hole_candidate_add(candidates,
                                     target,
                                     &tokens,
                                     frontier,
                                     nn_gd_arc_hole_attempt_id(attempt),
                                     "input_frame_shape_repair",
                                     stage,
                                     "replace_fixed_frame_with_input_frame",
                                     32L,
                                     0.68 / (double)rank) != 0) {
        goto done;
    }
    nn_gd_string_list_free(&tokens);
    memset(&tokens, 0, sizeof(tokens));
    if (nn_gd_arc_hole_add_op_tokens(&tokens, attempt, 1) != 0 ||
        nn_gd_string_list_add_unique(&tokens,
                                     "counterexample_cell_mask_cleanup_repair") != 0 ||
        nn_gd_string_list_add_unique(&tokens, "emitter:component_mask") != 0 ||
        nn_gd_string_list_add_unique(&tokens, "normalizer:component_mask") != 0 ||
        nn_gd_arc_hole_add_pin(&tokens, "selector", 3L) != 0 ||
        nn_gd_arc_hole_add_pin(&tokens, "x_anchor", 3L) != 0 ||
        nn_gd_arc_hole_add_pin(&tokens, "y_anchor", 6L) != 0 ||
        nn_gd_arc_hole_add_pin(&tokens, "y_offset", 7L) != 0 ||
        nn_gd_arc_hole_add_pin(&tokens, "period", 0L) != 0 ||
        nn_gd_arc_hole_add_pin(&tokens, "width", 0L) != 0 ||
        nn_gd_arc_hole_add_pin(&tokens, "height", 0L) != 0 ||
        nn_gd_arc_hole_candidate_add(candidates,
                                     target,
                                     &tokens,
                                     frontier,
                                     nn_gd_arc_hole_attempt_id(attempt),
                                     "component_mask_fill_repair",
                                     "renderer_cells",
                                     "preserve_mask_and_fill_enclosed_voids",
                                     32L,
                                     0.66 / (double)rank) != 0) {
        goto done;
    }
    nn_gd_string_list_free(&tokens);
    memset(&tokens, 0, sizeof(tokens));
    if (nn_gd_arc_hole_add_op_tokens(&tokens, attempt, 1) != 0 ||
        nn_gd_string_list_add_unique(&tokens, "counterexample_cell_span_repair") != 0 ||
        nn_gd_string_list_add_unique(&tokens, "emitter:aligned_span") != 0 ||
        nn_gd_string_list_add_unique(&tokens, "composer:aligned_span") != 0 ||
        nn_gd_arc_hole_add_pin(&tokens, "selector", 3L) != 0 ||
        nn_gd_arc_hole_add_pin(&tokens, "x_anchor", 4L) != 0 ||
        nn_gd_arc_hole_add_pin(&tokens, "y_anchor", 2L) != 0 ||
        nn_gd_arc_hole_add_pin(&tokens, "period", 0L) != 0 ||
        nn_gd_arc_hole_add_pin(&tokens, "width", 0L) != 0 ||
        nn_gd_arc_hole_add_pin(&tokens, "height", 0L) != 0 ||
        nn_gd_arc_hole_candidate_add(candidates,
                                     target,
                                     &tokens,
                                     frontier,
                                     nn_gd_arc_hole_attempt_id(attempt),
                                     "aligned_span_repair",
                                     "renderer_cells",
                                     "fill_aligned_marker_spans",
                                     32L,
                                     0.64 / (double)rank) != 0) {
        goto done;
    }
    status = 0;

done:
    nn_gd_string_list_free(&tokens);
    return status;
}

static int nn_gd_arc_hole_emit_residual_candidates(
    NNGDArcHoleCandidateList *candidates,
    FILE *examples_fp,
    const NNGDArcHoleTarget *target,
    const NNGDArcPipelineAttempt *target_features,
    const NNGDArcPipelineAttemptList *attempts,
    const char *frontier,
    unsigned max_attempt_seeds,
    unsigned max_ops_per_task,
    int emit_broad_repair) {
    NNGDArcHoleCounterList stage_counts = {0};
    NNGDArcHoleCounterList op_counts = {0};
    size_t *selected = NULL;
    const char *dominant_stage;
    unsigned rank;
    int status = -1;
    if (nn_gd_arc_hole_collect_rejected_surface(attempts,
                                                target->task_id,
                                                &stage_counts,
                                                &op_counts) != 0) {
        goto done;
    }
    dominant_stage = nn_gd_arc_hole_counter_dominant(&stage_counts, "coverage");
    if (emit_broad_repair) {
        NNGDStringList tokens = {0};
        if (nn_gd_arc_hole_add_dominant_ops(&tokens,
                                            &op_counts,
                                            max_ops_per_task) != 0 ||
            nn_gd_arc_hole_add_stage_tokens(&tokens, dominant_stage) != 0 ||
            nn_gd_arc_hole_add_shape_hint_tokens(&tokens, target_features) != 0 ||
            (tokens.count > 0U &&
             nn_gd_arc_hole_candidate_add(candidates,
                                          target,
                                          &tokens,
                                          frontier,
                                          "",
                                          "dominant_residual_op_repair",
                                          dominant_stage,
                                          "rank_ops_by_rejected_attempt_surface",
                                          64L,
                                          1.0) != 0)) {
            nn_gd_string_list_free(&tokens);
            goto done;
        }
        nn_gd_string_list_free(&tokens);
    }
    if (max_attempt_seeds == 0U) {
        status = 0;
        goto done;
    }
    selected = (size_t *)calloc(max_attempt_seeds, sizeof(*selected));
    if (selected == NULL) {
        goto done;
    }
    for (rank = 0U; rank < max_attempt_seeds; ++rank) {
        const NNGDArcPipelineAttempt *best = NULL;
        size_t best_index = SIZE_MAX;
        size_t i;
        const char *stage;
        for (i = 0U; i < attempts->count; ++i) {
            const NNGDArcPipelineAttempt *attempt = &attempts->items[i];
            if (attempt->accepted || strcmp(attempt->task_id, target->task_id) != 0 ||
                nn_gd_arc_hole_index_selected(selected, rank, i)) {
                continue;
            }
            if (nn_gd_arc_hole_attempt_score_better(attempt, best)) {
                best = attempt;
                best_index = i;
            }
        }
        if (best == NULL) {
            break;
        }
        selected[rank] = best_index;
        stage = best->failure_stage[0] != '\0'
                    ? best->failure_stage
                    : (best->rejection_reason[0] != '\0' ? best->rejection_reason
                                                          : dominant_stage);
        {
            NNGDStringList tokens = {0};
            if (nn_gd_arc_hole_add_op_tokens(&tokens, best, 1) != 0) {
                nn_gd_string_list_free(&tokens);
                goto done;
            }
            if (strcmp(stage, "renderer_cells") == 0 ||
                strcmp(stage, "transform_or_color") == 0) {
                if (nn_gd_arc_hole_add_geometry_pins(&tokens, best) != 0) {
                    nn_gd_string_list_free(&tokens);
                    goto done;
                }
            } else if (strcmp(stage, "renderer_shape") == 0 ||
                       strcmp(stage, "selector_or_transform") == 0) {
                if (nn_gd_arc_hole_add_pin(&tokens, "selector", best->selector) != 0) {
                    nn_gd_string_list_free(&tokens);
                    goto done;
                }
            }
            if (best->slot_width_matches_output &&
                nn_gd_arc_hole_add_pin(&tokens, "width", best->width) != 0) {
                nn_gd_string_list_free(&tokens);
                goto done;
            }
            if (best->slot_height_matches_output &&
                nn_gd_arc_hole_add_pin(&tokens, "height", best->height) != 0) {
                nn_gd_string_list_free(&tokens);
                goto done;
            }
            if (nn_gd_arc_hole_candidate_add(candidates,
                                             target,
                                             &tokens,
                                             frontier,
                                             nn_gd_arc_hole_attempt_id(best),
                                             "verifier_distance_slot_repair",
                                             stage,
                                             "pin_high_score_attempt_geometry",
                                             32L,
                                             0.75 / (double)(rank + 1U)) != 0) {
                nn_gd_string_list_free(&tokens);
                goto done;
            }
            nn_gd_string_list_free(&tokens);
        }
        if (emit_broad_repair) {
            NNGDStringList tokens = {0};
            if (nn_gd_arc_hole_add_op_tokens(&tokens, best, 0) != 0 ||
                nn_gd_arc_hole_add_structural_pins(&tokens, best) != 0 ||
                nn_gd_arc_hole_add_stage_tokens(&tokens, stage) != 0 ||
                nn_gd_arc_hole_add_shape_hint_tokens(&tokens, target_features) != 0 ||
                nn_gd_arc_hole_candidate_add(candidates,
                                             target,
                                             &tokens,
                                             frontier,
                                             nn_gd_arc_hole_attempt_id(best),
                                             "verifier_distance_slot_repair",
                                             stage,
                                             "repair_high_score_attempt_slots",
                                             32L,
                                             0.70 / (double)(rank + 1U)) != 0) {
                nn_gd_string_list_free(&tokens);
                goto done;
            }
            nn_gd_string_list_free(&tokens);
        }
        if (nn_gd_arc_hole_emit_special_shape_repairs(candidates,
                                                      target,
                                                      best,
                                                      frontier,
                                                      rank + 1U,
                                                      stage) != 0 ||
            nn_gd_arc_hole_write_rejected_example(examples_fp,
                                                  target,
                                                  best,
                                                  rank + 1U,
                                                  stage) != 0) {
            goto done;
        }
    }
    status = 0;

done:
    free(selected);
    nn_gd_arc_hole_counter_list_free(&stage_counts);
    nn_gd_arc_hole_counter_list_free(&op_counts);
    return status;
}

static const char *nn_gd_arc_curriculum_selector_name(long selector_id) {
    static const char *names[] = {
        "grid_panel",
        "foreground_bbox_panel",
        "color_bbox",
        "lattice",
    };
    if (selector_id < 0L || (size_t)selector_id >= sizeof(names) / sizeof(names[0])) {
        return "";
    }
    return names[selector_id];
}

static long nn_gd_arc_curriculum_slot_integer(const char *line,
                                              const char *slot_name,
                                              long default_value) {
    const char *name = strstr(line, slot_name);
    const char *start;
    const char *end;
    char *object;
    NNGDJsonIndex index;
    long value;
    size_t len;
    if (name == NULL) {
        return default_value;
    }
    start = name;
    while (start > line && *start != '{') {
        --start;
    }
    if (*start != '{') {
        return default_value;
    }
    end = nn_gd_arc_hole_json_object_end(start);
    if (end == NULL || end <= start) {
        return default_value;
    }
    len = (size_t)(end - start);
    object = (char *)malloc(len + 1U);
    if (object == NULL) {
        return default_value;
    }
    memcpy(object, start, len);
    object[len] = '\0';
    nn_gd_json_index_line(object, &index);
    value = nn_gd_json_index_long_or(&index, "integer", default_value);
    free(object);
    return value;
}

static int nn_gd_arc_curriculum_line_is_positive(const char *line) {
    char *schema = nn_gd_json_dup_key_string(line, "schema");
    char *template_name = NULL;
    char *certificate_status = NULL;
    int positive = 0;
    if (schema == NULL) {
        return 0;
    }
    if (strcmp(schema, "crlp.hypothesis.evidence.v1") != 0) {
        free(schema);
        return 0;
    }
    free(schema);
    template_name = nn_gd_json_dup_key_string(line, "template_name");
    certificate_status = nn_gd_json_dup_key_string(line, "certificate_status");
    positive = template_name != NULL && certificate_status != NULL &&
               strcmp(template_name, "arc_sketch_program") == 0 &&
               strcmp(certificate_status, "certified") == 0;
    free(template_name);
    free(certificate_status);
    return positive;
}

static int nn_gd_arc_curriculum_collect_positive(const char *line,
                                                 NNGDArcHoleCounterList *op_counts,
                                                 NNGDStringList *positive_tasks,
                                                 size_t *positive_count) {
    char *task_id;
    long op_id;
    const char *op_name;
    if (!nn_gd_arc_curriculum_line_is_positive(line)) {
        return 0;
    }
    op_id = nn_gd_arc_curriculum_slot_integer(line, "sketch.op", -1L);
    op_name = nn_gd_arc_hole_op_name_from_id(op_id);
    if (op_name[0] != '\0' && nn_gd_arc_hole_counter_add(op_counts, op_name) != 0) {
        return -1;
    }
    task_id = nn_gd_json_dup_key_string(line, "task_instance");
    if (task_id == NULL) {
        return -1;
    }
    if (task_id[0] != '\0' &&
        nn_gd_string_list_add_unique(positive_tasks, task_id) != 0) {
        free(task_id);
        return -1;
    }
    free(task_id);
    ++*positive_count;
    return 0;
}

static int nn_gd_arc_curriculum_scan_positives(const char *path,
                                               NNGDArcHoleCounterList *op_counts,
                                               NNGDStringList *positive_tasks,
                                               size_t *positive_count,
                                               size_t *rows_out) {
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    *rows_out = 0U;
    for (;;) {
        char *line = nn_gd_read_line_dynamic(fp);
        if (line == NULL) {
            break;
        }
        if (line[0] != '\0' && line[0] != '\n' && line[0] != '\r') {
            ++*rows_out;
        }
        if (nn_gd_arc_curriculum_collect_positive(line,
                                                  op_counts,
                                                  positive_tasks,
                                                  positive_count) != 0) {
            free(line);
            fclose(fp);
            return -1;
        }
        free(line);
    }
    if (fclose(fp) != 0) {
        return -1;
    }
    return 0;
}

static int nn_gd_arc_curriculum_build_op_hints(
    const NNGDArcHoleCounterList *op_counts,
    unsigned max_op_hints,
    NNGDStringList *op_hints) {
    size_t selected[32];
    unsigned rank;
    if (max_op_hints > sizeof(selected) / sizeof(selected[0])) {
        max_op_hints = (unsigned)(sizeof(selected) / sizeof(selected[0]));
    }
    for (rank = 0U; rank < max_op_hints; ++rank) {
        size_t i;
        size_t best = SIZE_MAX;
        for (i = 0U; i < op_counts->count; ++i) {
            if (nn_gd_arc_hole_index_selected(selected, rank, i)) {
                continue;
            }
            if (best == SIZE_MAX ||
                op_counts->items[i].count > op_counts->items[best].count) {
                best = i;
            }
        }
        if (best == SIZE_MAX) {
            break;
        }
        selected[rank] = best;
        if (nn_gd_string_list_add(op_hints, op_counts->items[best].key) != 0) {
            return -1;
        }
    }
    nn_gd_string_list_sort(op_hints);
    return 0;
}

static char *nn_gd_arc_curriculum_join_hints(const NNGDStringList *op_hints) {
    return nn_gd_arc_hole_compact_tokens(op_hints);
}

static int nn_gd_arc_curriculum_candidate_add(
    NNGDArcHoleCandidateList *candidates,
    const NNGDArcHoleTarget *target,
    const char *op_hints,
    const char *frontier) {
    NNGDArcHoleCandidate *candidate;
    const char *fields[13];
    char support_tasks[32];
    char proposal_count[32];
    size_t i;
    if (target->task_id[0] == '\0' || target->split[0] == '\0' ||
        op_hints == NULL || op_hints[0] == '\0') {
        return 0;
    }
    snprintf(support_tasks, sizeof(support_tasks), "%d", 1);
    snprintf(proposal_count, sizeof(proposal_count), "%d", 64);
    fields[0] = "arc.visual_curriculum_transfer_family";
    fields[1] = target->task_id;
    fields[2] = target->split;
    fields[3] = support_tasks;
    fields[4] = proposal_count;
    fields[5] = "1.00000000";
    fields[6] = op_hints;
    fields[7] = frontier;
    fields[8] = "";
    fields[9] = "visual_only_positive_op_transfer";
    fields[10] = target->kind;
    fields[11] = "coverage";
    fields[12] = "replay_verified_visual_ops";
    if (nn_gd_arc_hole_candidate_list_reserve(candidates,
                                              candidates->count + 1U) != 0) {
        return -1;
    }
    candidate = &candidates->items[candidates->count];
    memset(candidate, 0, sizeof(*candidate));
    for (i = 0U; i < 13U; ++i) {
        candidate->fields[i] = nn_gd_strdup(fields[i]);
        if (candidate->fields[i] == NULL) {
            nn_gd_arc_hole_candidate_free(candidate);
            return -1;
        }
    }
    ++candidates->count;
    return 0;
}

static int nn_gd_arc_curriculum_write_slot_values(FILE *fp, const char *line) {
    static const char *slot_names[] = {
        "sketch.background",
        "sketch.color",
        "sketch.height",
        "sketch.op",
        "sketch.output_color",
        "sketch.period",
        "sketch.selector",
        "sketch.width",
        "sketch.x_anchor",
        "sketch.x_offset",
        "sketch.y_anchor",
        "sketch.y_offset",
    };
    size_t i;
    fputc('{', fp);
    for (i = 0U; i < sizeof(slot_names) / sizeof(slot_names[0]); ++i) {
        if (i > 0U) {
            fputc(',', fp);
        }
        nn_gd_json_write_string(fp, slot_names[i]);
        fprintf(fp,
                ":%ld",
                nn_gd_arc_curriculum_slot_integer(line, slot_names[i], 0L));
    }
    fputc('}', fp);
    return ferror(fp) ? -1 : 0;
}

static int nn_gd_arc_curriculum_write_positive_example(FILE *fp,
                                                       const char *scope,
                                                       const char *path,
                                                       const char *line) {
    char *task_id = nn_gd_json_dup_key_string(line, "task_instance");
    char *program_hash = nn_gd_json_dup_key_string(line, "program_canonical_hash");
    long op_id = nn_gd_arc_curriculum_slot_integer(line, "sketch.op", -1L);
    long selector_id =
        nn_gd_arc_curriculum_slot_integer(line, "sketch.selector", -1L);
    long support = nn_gd_json_get_double_or_zero(line, "support");
    long event_count = nn_gd_json_get_double_or_zero(line, "event_count");
    long covered_events = nn_gd_json_get_double_or_zero(line, "covered_events");
    double utility = nn_gd_json_get_double_or_zero(line, "utility");
    const char *split = nn_gd_arc_hole_split_for_scope(scope);
    int status = -1;
    if (task_id == NULL || program_hash == NULL) {
        free(task_id);
        free(program_hash);
        return -1;
    }
    fputs("{\"schema\":\"crlp.neural_grammar_discovery.arc_visual_curriculum_example.v1\","
          "\"report_version\":1,\"kind\":\"verified_visual_positive\","
          "\"source_scope\":",
          fp);
    nn_gd_json_write_string(fp, scope);
    fputs(",\"source_path\":", fp);
    nn_gd_json_write_string(fp, path);
    fputs(",\"split\":", fp);
    nn_gd_json_write_string(fp, split == NULL ? "" : split);
    fputs(",\"task_id\":", fp);
    nn_gd_json_write_string(fp, task_id);
    fputs(",\"template_name\":\"arc_sketch_program\"", fp);
    fprintf(fp, ",\"op_id\":%ld,\"op_name\":", op_id);
    nn_gd_json_write_string(fp, nn_gd_arc_hole_op_name_from_id(op_id));
    fprintf(fp, ",\"selector_id\":%ld,\"selector_name\":", selector_id);
    nn_gd_json_write_string(fp, nn_gd_arc_curriculum_selector_name(selector_id));
    fputs(",\"program_canonical_hash\":", fp);
    nn_gd_json_write_string(fp, program_hash);
    fprintf(fp,
            ",\"support\":%ld,\"event_count\":%ld,\"covered_events\":%ld,"
            "\"utility\":%.8f,\"slot_values\":",
            support,
            event_count,
            covered_events,
            utility);
    if (nn_gd_arc_curriculum_write_slot_values(fp, line) != 0) {
        goto done;
    }
    fputs("}\n", fp);
    status = ferror(fp) ? -1 : 0;

done:
    free(task_id);
    free(program_hash);
    return status;
}

static int nn_gd_arc_curriculum_write_target_example(
    FILE *fp,
    const NNGDArcHoleTarget *target,
    const NNGDStringList *op_hints) {
    size_t i;
    fputs("{\"schema\":\"crlp.neural_grammar_discovery.arc_visual_curriculum_example.v1\","
          "\"report_version\":1,\"kind\":",
          fp);
    nn_gd_json_write_string(fp, target->kind);
    fputs(",\"source_scope\":", fp);
    nn_gd_json_write_string(fp, target->source_scope);
    fputs(",\"source_path\":", fp);
    nn_gd_json_write_string(fp, target->source_path);
    fputs(",\"split\":", fp);
    nn_gd_json_write_string(fp, target->split);
    fputs(",\"task_id\":", fp);
    nn_gd_json_write_string(fp, target->task_id);
    fputs(",\"template_name\":\"arc_sketch_program\"", fp);
    fprintf(fp,
            ",\"attempted\":%ld,\"train_consistent_pre_cap\":%ld,"
            "\"train_consistent_retained\":%ld,\"selected\":%ld,"
            "\"proposed_op_hints\":[",
            target->attempted,
            target->train_consistent_pre_cap,
            target->train_consistent_retained,
            target->selected);
    for (i = 0U; i < op_hints->count; ++i) {
        if (i > 0U) {
            fputc(',', fp);
        }
        nn_gd_json_write_string(fp, op_hints->items[i]);
    }
    fputs("]}\n", fp);
    return ferror(fp) ? -1 : 0;
}

static int nn_gd_arc_curriculum_scan_write_positives(FILE *fp,
                                                     const char *scope,
                                                     const char *path) {
    FILE *in = fopen(path, "r");
    if (in == NULL) {
        return -1;
    }
    for (;;) {
        char *line = nn_gd_read_line_dynamic(in);
        if (line == NULL) {
            break;
        }
        if (nn_gd_arc_curriculum_line_is_positive(line) &&
            nn_gd_arc_curriculum_write_positive_example(fp, scope, path, line) != 0) {
            free(line);
            fclose(in);
            return -1;
        }
        free(line);
    }
    if (fclose(in) != 0) {
        return -1;
    }
    return 0;
}

static int nn_gd_arc_curriculum_scan_targets(FILE *fp,
                                             const char *scope,
                                             const char *path,
                                             const char *const *target_scopes,
                                             size_t target_scope_count,
                                             const NNGDStringList *op_hints,
                                             const char *joined_hints,
                                             NNGDArcHoleCandidateList *candidates,
                                             size_t *target_task_count) {
    FILE *in = fopen(path, "r");
    const char *split = nn_gd_arc_hole_split_for_scope(scope);
    int target_scope = nn_gd_arc_hole_scope_matches(scope,
                                                    target_scopes,
                                                    target_scope_count);
    if (in == NULL) {
        return -1;
    }
    for (;;) {
        char *line = nn_gd_read_line_dynamic(in);
        char *schema;
        char *task_id = NULL;
        long attempted;
        long train_consistent_pre_cap;
        long train_consistent_retained;
        long selected;
        NNGDArcHoleTarget target;
        char *frontier = NULL;
        if (line == NULL) {
            break;
        }
        schema = nn_gd_json_dup_key_string(line, "schema");
        if (schema == NULL) {
            free(line);
            fclose(in);
            return -1;
        }
        if (strcmp(schema, "crlp.arc_hypothesis.selection.v1") != 0) {
            free(schema);
            free(line);
            continue;
        }
        free(schema);
        if (nn_gd_arc_hole_parse_visual_template_stats(line,
                                                       &attempted,
                                                       &train_consistent_pre_cap,
                                                       &train_consistent_retained,
                                                       &selected) != 0) {
            free(line);
            fclose(in);
            return -1;
        }
        task_id = nn_gd_json_dup_key_string(line, "task_instance");
        if (task_id == NULL) {
            free(line);
            fclose(in);
            return -1;
        }
        memset(&target, 0, sizeof(target));
        target.task_id = task_id;
        target.split = (char *)(split == NULL ? "" : split);
        target.source_scope = (char *)scope;
        target.source_path = (char *)path;
        target.kind = (char *)nn_gd_arc_hole_task_kind(selected,
                                                       train_consistent_pre_cap,
                                                       attempted);
        target.attempted = attempted;
        target.train_consistent_pre_cap = train_consistent_pre_cap;
        target.train_consistent_retained = train_consistent_retained;
        target.selected = selected;
        if (nn_gd_arc_curriculum_write_target_example(fp, &target, op_hints) != 0) {
            free(task_id);
            free(line);
            fclose(in);
            return -1;
        }
        if (target_scope && split != NULL && joined_hints != NULL &&
            joined_hints[0] != '\0') {
            size_t frontier_len = strlen("arc_visual_curriculum:") + strlen(scope) + 1U;
            frontier = (char *)malloc(frontier_len);
            if (frontier == NULL) {
                free(task_id);
                free(line);
                fclose(in);
                return -1;
            }
            snprintf(frontier, frontier_len, "arc_visual_curriculum:%s", scope);
            if (nn_gd_arc_curriculum_candidate_add(candidates,
                                                   &target,
                                                   joined_hints,
                                                   frontier) != 0) {
                free(frontier);
                free(task_id);
                free(line);
                fclose(in);
                return -1;
            }
            ++*target_task_count;
            free(frontier);
        }
        free(task_id);
        free(line);
    }
    if (fclose(in) != 0) {
        return -1;
    }
    return 0;
}

static int nn_gd_arc_curriculum_write_attempt_summary(FILE *fp,
                                                      const char *attempt_log) {
    NNGDArcPipelineAttemptList attempts = {0};
    NNGDArcHoleCounterList accepted_by_op = {0};
    NNGDArcHoleCounterList rejected_by_reason = {0};
    NNGDStringList accepted_tasks = {0};
    size_t accepted = 0U;
    size_t rejected = 0U;
    size_t i;
    int status = -1;
    if (attempt_log == NULL || attempt_log[0] == '\0') {
        fputs("{}", fp);
        return 0;
    }
    if (nn_gd_read_arc_pipeline_attempts_jsonl(attempt_log,
                                               attempt_log,
                                               &attempts) != 0) {
        fputs("{}", fp);
        return 0;
    }
    for (i = 0U; i < attempts.count; ++i) {
        NNGDArcPipelineAttempt *attempt = &attempts.items[i];
        if (attempt->accepted) {
            ++accepted;
            if (nn_gd_arc_hole_counter_add(&accepted_by_op, attempt->op_name) != 0 ||
                nn_gd_string_list_add_unique(&accepted_tasks, attempt->task_id) != 0) {
                goto done;
            }
        } else {
            const char *reason = attempt->failure_stage[0] != '\0'
                                     ? attempt->failure_stage
                                     : (attempt->rejection_reason[0] != '\0'
                                            ? attempt->rejection_reason
                                            : "rejected");
            ++rejected;
            if (nn_gd_arc_hole_counter_add(&rejected_by_reason, reason) != 0) {
                goto done;
            }
        }
    }
    nn_gd_string_list_sort(&accepted_tasks);
    fputs("{\"attempt_log\":", fp);
    nn_gd_json_write_string(fp, attempt_log);
    fprintf(fp, ",\"accepted_attempts\":%zu,\"rejected_attempts\":%zu", accepted, rejected);
    fputs(",\"accepted_tasks\":", fp);
    nn_gd_json_write_string_list(fp, &accepted_tasks);
    fputs(",\"accepted_by_op\":", fp);
    if (nn_gd_arc_hole_write_counter_object(fp, &accepted_by_op) != 0) {
        goto done;
    }
    fputs(",\"rejected_by_reason\":", fp);
    if (nn_gd_arc_hole_write_counter_object(fp, &rejected_by_reason) != 0) {
        goto done;
    }
    fputc('}', fp);
    status = ferror(fp) ? -1 : 0;

done:
    nn_gd_string_list_free(&accepted_tasks);
    nn_gd_arc_hole_counter_list_free(&accepted_by_op);
    nn_gd_arc_hole_counter_list_free(&rejected_by_reason);
    nn_gd_arc_attempt_list_free(&attempts);
    return status;
}

static int nn_gd_arc_curriculum_write_summary(
    const char *path,
    const NNGDArcHoleSourceLogList *sources,
    const char *const *target_scopes,
    size_t target_scope_count,
    const NNGDArcHoleCounterList *op_counts_in,
    const NNGDStringList *op_hints,
    size_t positive_count,
    size_t positive_task_count,
    size_t target_task_count,
    const NNGDArcHoleCandidateList *candidates,
    const char *candidate_tsv_path,
    const char *examples_path,
    const char *attempt_log) {
    FILE *fp = fopen(path, "w");
    NNGDArcHoleCounterList op_counts = {0};
    size_t i;
    int status = -1;
    if (fp == NULL) {
        return -1;
    }
    for (i = 0U; i < op_counts_in->count; ++i) {
        size_t j;
        for (j = 0U; j < op_counts_in->items[i].count; ++j) {
            if (nn_gd_arc_hole_counter_add(&op_counts, op_counts_in->items[i].key) != 0) {
                goto done;
            }
        }
    }
    fputs("{\"schema\":\"crlp.neural_grammar_discovery.arc_visual_curriculum_summary.v1\","
          "\"report_version\":1,\"source_logs\":[",
          fp);
    for (i = 0U; i < sources->count; ++i) {
        if (i > 0U) {
            fputc(',', fp);
        }
        fputs("{\"scope\":", fp);
        nn_gd_json_write_string(fp, sources->items[i].scope);
        fputs(",\"path\":", fp);
        nn_gd_json_write_string(fp, sources->items[i].path);
        fprintf(fp, ",\"rows\":%zu}", sources->items[i].rows);
    }
    fputs("],\"target_scopes\":", fp);
    if (nn_gd_arc_hole_write_string_array(fp, target_scopes, target_scope_count) != 0) {
        goto done;
    }
    fprintf(fp,
            ",\"positive_count\":%zu,\"positive_task_count\":%zu,"
            "\"positive_ops\":",
            positive_count,
            positive_task_count);
    if (nn_gd_arc_hole_write_counter_object(fp, &op_counts) != 0) {
        goto done;
    }
    fputs(",\"op_hints\":", fp);
    nn_gd_json_write_string_list(fp, op_hints);
    fprintf(fp,
            ",\"target_task_count\":%zu,\"candidate_count\":%zu,"
            "\"candidate_tsv\":",
            target_task_count,
            candidates->count);
    nn_gd_json_write_string(fp, candidate_tsv_path);
    fputs(",\"example_jsonl\":", fp);
    nn_gd_json_write_string(fp, examples_path);
    fputs(",\"attempt_summary\":", fp);
    if (nn_gd_arc_curriculum_write_attempt_summary(fp, attempt_log) != 0) {
        goto done;
    }
    fputs("}\n", fp);
    status = ferror(fp) ? -1 : 0;

done:
    nn_gd_arc_hole_counter_list_free(&op_counts);
    if (fclose(fp) != 0) {
        status = -1;
    }
    return status;
}

int nn_grammar_discovery_write_arc_visual_curriculum(
    const char *const *visual_only_logs,
    size_t visual_only_log_count,
    const char *const *target_scopes,
    size_t target_scope_count,
    const char *attempt_log,
    const char *candidate_tsv_path,
    const char *examples_path,
    const char *summary_path,
    unsigned max_op_hints) {
    NNGDArcHoleSourceLogList sources = {0};
    NNGDArcHoleCounterList op_counts = {0};
    NNGDStringList positive_tasks = {0};
    NNGDStringList op_hints = {0};
    NNGDArcHoleCandidateList candidates = {0};
    FILE *examples_fp = NULL;
    char *joined_hints = NULL;
    size_t positive_count = 0U;
    size_t target_task_count = 0U;
    size_t i;
    int status = -1;
    if (visual_only_logs == NULL || visual_only_log_count == 0U ||
        candidate_tsv_path == NULL || candidate_tsv_path[0] == '\0' ||
        examples_path == NULL || examples_path[0] == '\0' ||
        summary_path == NULL || summary_path[0] == '\0') {
        return -1;
    }
    for (i = 0U; i < visual_only_log_count; ++i) {
        char *scope = NULL;
        char *path = NULL;
        size_t rows = 0U;
        if (nn_gd_arc_hole_parse_scoped_path(visual_only_logs[i],
                                             i,
                                             &scope,
                                             &path) != 0) {
            goto done;
        }
        if (nn_gd_arc_curriculum_scan_positives(path,
                                                &op_counts,
                                                &positive_tasks,
                                                &positive_count,
                                                &rows) != 0) {
            free(scope);
            free(path);
            goto done;
        }
        if (nn_gd_arc_hole_source_log_list_reserve(&sources,
                                                   sources.count + 1U) != 0) {
            free(scope);
            free(path);
            goto done;
        }
        sources.items[sources.count].scope = scope;
        sources.items[sources.count].path = path;
        sources.items[sources.count].rows = rows;
        ++sources.count;
    }
    if (nn_gd_arc_curriculum_build_op_hints(&op_counts,
                                            max_op_hints == 0U ? 8U : max_op_hints,
                                            &op_hints) != 0) {
        goto done;
    }
    joined_hints = nn_gd_arc_curriculum_join_hints(&op_hints);
    if (joined_hints == NULL) {
        goto done;
    }
    examples_fp = fopen(examples_path, "w");
    if (examples_fp == NULL) {
        goto done;
    }
    for (i = 0U; i < sources.count; ++i) {
        if (nn_gd_arc_curriculum_scan_write_positives(examples_fp,
                                                      sources.items[i].scope,
                                                      sources.items[i].path) != 0) {
            goto done;
        }
    }
    for (i = 0U; i < sources.count; ++i) {
        if (nn_gd_arc_curriculum_scan_targets(examples_fp,
                                              sources.items[i].scope,
                                              sources.items[i].path,
                                              target_scopes,
                                              target_scope_count,
                                              &op_hints,
                                              joined_hints,
                                              &candidates,
                                              &target_task_count) != 0) {
            goto done;
        }
    }
    if (fclose(examples_fp) != 0) {
        examples_fp = NULL;
        goto done;
    }
    examples_fp = NULL;
    if (nn_gd_arc_hole_write_candidate_tsv(candidate_tsv_path, &candidates) != 0 ||
        nn_gd_arc_curriculum_write_summary(summary_path,
                                           &sources,
                                           target_scopes,
                                           target_scope_count,
                                           &op_counts,
                                           &op_hints,
                                           positive_count,
                                           positive_tasks.count,
                                           target_task_count,
                                           &candidates,
                                           candidate_tsv_path,
                                           examples_path,
                                           attempt_log) != 0) {
        goto done;
    }
    printf("wrote %zu ARC visual curriculum candidates from %zu positives to %s\n",
           candidates.count,
           positive_count,
           candidate_tsv_path);
    status = 0;

done:
    if (examples_fp != NULL) {
        fclose(examples_fp);
    }
    free(joined_hints);
    nn_gd_arc_hole_candidate_list_free(&candidates);
    nn_gd_string_list_free(&op_hints);
    nn_gd_string_list_free(&positive_tasks);
    nn_gd_arc_hole_counter_list_free(&op_counts);
    nn_gd_arc_hole_source_log_list_free(&sources);
    return status;
}

int nn_grammar_discovery_write_arc_visual_hole_proposer(
    const char *const *visual_only_logs,
    size_t visual_only_log_count,
    const char *const *target_scopes,
    size_t target_scope_count,
    const char *const *attempt_logs,
    size_t attempt_log_count,
    const char *candidate_tsv_path,
    const char *examples_path,
    const char *summary_path,
    unsigned nearest_signatures,
    unsigned max_attempt_seeds,
    unsigned max_ops_per_task,
    int emit_broad_repair,
    int include_selected) {
    NNGDArcHoleSourceLogList sources = {0};
    NNGDArcHoleTargetList target_examples = {0};
    NNGDArcHoleTargetList targets = {0};
    NNGDArcPipelineAttemptList attempts = {0};
    NNGDArcHoleCandidateList candidates = {0};
    FILE *examples_fp = NULL;
    size_t i;
    int status = -1;
    if (visual_only_logs == NULL || visual_only_log_count == 0U ||
        candidate_tsv_path == NULL || candidate_tsv_path[0] == '\0' ||
        examples_path == NULL || examples_path[0] == '\0' ||
        summary_path == NULL || summary_path[0] == '\0') {
        return -1;
    }
    if (nn_gd_arc_hole_load_sources(visual_only_logs,
                                    visual_only_log_count,
                                    target_scopes,
                                    target_scope_count,
                                    include_selected,
                                    &sources,
                                    &target_examples,
                                    &targets) != 0 ||
        nn_gd_arc_hole_load_attempts(attempt_logs,
                                     attempt_log_count,
                                     &attempts) != 0) {
        goto done;
    }
    if (targets.count > 1U) {
        qsort(targets.items,
              targets.count,
              sizeof(targets.items[0]),
              nn_gd_arc_hole_compare_targets);
    }
    examples_fp = fopen(examples_path, "w");
    if (examples_fp == NULL) {
        goto done;
    }
    for (i = 0U; i < target_examples.count; ++i) {
        if (nn_gd_arc_hole_write_target_example(examples_fp,
                                                &target_examples.items[i]) != 0) {
            goto done;
        }
    }
    for (i = 0U; i < targets.count; ++i) {
        const NNGDArcHoleTarget *target = &targets.items[i];
        const NNGDArcPipelineAttempt *features =
            nn_gd_arc_hole_task_features(&attempts, target->task_id);
        char *frontier = nn_gd_arc_hole_frontier(target->source_scope);
        if (frontier == NULL) {
            goto done;
        }
        if (nn_gd_arc_hole_emit_shape_candidate(&candidates,
                                                target,
                                                features,
                                                frontier) != 0 ||
            nn_gd_arc_hole_emit_transfer_candidates(&candidates,
                                                    examples_fp,
                                                    target,
                                                    features,
                                                    &attempts,
                                                    frontier,
                                                    nearest_signatures) != 0 ||
            nn_gd_arc_hole_emit_residual_candidates(&candidates,
                                                    examples_fp,
                                                    target,
                                                    features,
                                                    &attempts,
                                                    frontier,
                                                    max_attempt_seeds,
                                                    max_ops_per_task,
                                                    emit_broad_repair) != 0) {
            free(frontier);
            goto done;
        }
        free(frontier);
    }
    if (fclose(examples_fp) != 0) {
        examples_fp = NULL;
        goto done;
    }
    examples_fp = NULL;
    if (nn_gd_arc_hole_write_candidate_tsv(candidate_tsv_path, &candidates) != 0 ||
        nn_gd_arc_hole_write_summary(summary_path,
                                     &sources,
                                     attempt_logs,
                                     attempt_log_count,
                                     target_scopes,
                                     target_scope_count,
                                     candidate_tsv_path,
                                     examples_path,
                                     targets.count,
                                     &attempts,
                                     &candidates) != 0) {
        goto done;
    }
    printf("wrote %zu ARC visual hole candidates for %zu target tasks to %s\n",
           candidates.count,
           targets.count,
           candidate_tsv_path);
    status = 0;

done:
    if (examples_fp != NULL) {
        fclose(examples_fp);
    }
    nn_gd_arc_hole_candidate_list_free(&candidates);
    nn_gd_arc_attempt_list_free(&attempts);
    nn_gd_arc_hole_target_list_free(&targets);
    nn_gd_arc_hole_target_list_free(&target_examples);
    nn_gd_arc_hole_source_log_list_free(&sources);
    return status;
}
