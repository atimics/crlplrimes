static int nn_gd_cache_write_u64(FILE *fp, uint64_t value) {
    return fwrite(&value, sizeof(value), 1U, fp) == 1U ? 0 : -1;
}

static int nn_gd_cache_read_u64(FILE *fp, uint64_t *value) {
    return fread(value, sizeof(*value), 1U, fp) == 1U ? 0 : -1;
}

static int nn_gd_cache_write_u32(FILE *fp, uint32_t value) {
    return fwrite(&value, sizeof(value), 1U, fp) == 1U ? 0 : -1;
}

static int nn_gd_cache_read_u32(FILE *fp, uint32_t *value) {
    return fread(value, sizeof(*value), 1U, fp) == 1U ? 0 : -1;
}

static int nn_gd_cache_write_u16(FILE *fp, uint16_t value) {
    return fwrite(&value, sizeof(value), 1U, fp) == 1U ? 0 : -1;
}

static int nn_gd_cache_read_u16(FILE *fp, uint16_t *value) {
    return fread(value, sizeof(*value), 1U, fp) == 1U ? 0 : -1;
}

static int nn_gd_cache_write_s64(FILE *fp, int64_t value) {
    return fwrite(&value, sizeof(value), 1U, fp) == 1U ? 0 : -1;
}

static int nn_gd_cache_read_s64(FILE *fp, int64_t *value) {
    return fread(value, sizeof(*value), 1U, fp) == 1U ? 0 : -1;
}

static int nn_gd_cache_write_double(FILE *fp, double value) {
    return fwrite(&value, sizeof(value), 1U, fp) == 1U ? 0 : -1;
}

static int nn_gd_cache_read_double(FILE *fp, double *value) {
    return fread(value, sizeof(*value), 1U, fp) == 1U ? 0 : -1;
}

static int nn_gd_cache_file_signature(const char *path,
                                      uint64_t *size,
                                      int64_t *mtime) {
    struct stat st;
    if (path == NULL || stat(path, &st) != 0 || st.st_size < 0) {
        return -1;
    }
    *size = (uint64_t)st.st_size;
    *mtime = (int64_t)st.st_mtime;
    return 0;
}

static int nn_gd_cache_write_string(FILE *fp, const char *value) {
    uint64_t length;
    if (value == NULL) {
        value = "";
    }
    length = (uint64_t)strlen(value);
    if (nn_gd_cache_write_u64(fp, length) != 0) {
        return -1;
    }
    if (length == 0U) {
        return 0;
    }
    return fwrite(value, 1U, (size_t)length, fp) == (size_t)length ? 0 : -1;
}

static char *nn_gd_cache_read_string(FILE *fp) {
    uint64_t length;
    char *value;
    if (nn_gd_cache_read_u64(fp, &length) != 0 ||
        length > NN_GD_ARC_RANKER_CACHE_MAX_STRING_BYTES ||
        length > (uint64_t)(SIZE_MAX - 1U)) {
        return NULL;
    }
    value = (char *)malloc((size_t)length + 1U);
    if (value == NULL) {
        return NULL;
    }
    if (length > 0U &&
        fread(value, 1U, (size_t)length, fp) != (size_t)length) {
        free(value);
        return NULL;
    }
    value[length] = '\0';
    return value;
}

static int nn_gd_cache_write_input(FILE *fp, const char *path) {
    uint64_t size;
    int64_t mtime;
    if (nn_gd_cache_file_signature(path, &size, &mtime) != 0 ||
        nn_gd_cache_write_string(fp, path) != 0 ||
        nn_gd_cache_write_u64(fp, size) != 0 ||
        nn_gd_cache_write_s64(fp, mtime) != 0) {
        return -1;
    }
    return 0;
}

static int nn_gd_cache_match_input(FILE *fp, const char *expected_path) {
    char *path = nn_gd_cache_read_string(fp);
    uint64_t cached_size;
    uint64_t actual_size;
    int64_t cached_mtime;
    int64_t actual_mtime;
    int matches = 0;
    if (path == NULL ||
        nn_gd_cache_read_u64(fp, &cached_size) != 0 ||
        nn_gd_cache_read_s64(fp, &cached_mtime) != 0 ||
        nn_gd_cache_file_signature(expected_path, &actual_size, &actual_mtime) != 0) {
        free(path);
        return -1;
    }
    matches = strcmp(path, expected_path) == 0 &&
              cached_size == actual_size &&
              cached_mtime == actual_mtime;
    free(path);
    return matches ? 0 : -1;
}

static int nn_gd_cache_write_inputs(FILE *fp,
                                    const char *const *source_paths,
                                    size_t source_count,
                                    const char *const *target_paths,
                                    size_t target_count) {
    size_t i;
    if (nn_gd_cache_write_u64(fp, (uint64_t)source_count) != 0 ||
        nn_gd_cache_write_u64(fp, (uint64_t)target_count) != 0) {
        return -1;
    }
    for (i = 0U; i < source_count; ++i) {
        if (nn_gd_cache_write_input(fp, source_paths[i]) != 0) {
            return -1;
        }
    }
    for (i = 0U; i < target_count; ++i) {
        if (nn_gd_cache_write_input(fp, target_paths[i]) != 0) {
            return -1;
        }
    }
    return 0;
}

static int nn_gd_cache_match_inputs(FILE *fp,
                                    const char *const *source_paths,
                                    size_t source_count,
                                    const char *const *target_paths,
                                    size_t target_count) {
    uint64_t cached_source_count;
    uint64_t cached_target_count;
    size_t i;
    if (nn_gd_cache_read_u64(fp, &cached_source_count) != 0 ||
        nn_gd_cache_read_u64(fp, &cached_target_count) != 0 ||
        cached_source_count != (uint64_t)source_count ||
        cached_target_count != (uint64_t)target_count) {
        return -1;
    }
    for (i = 0U; i < source_count; ++i) {
        if (nn_gd_cache_match_input(fp, source_paths[i]) != 0) {
            return -1;
        }
    }
    for (i = 0U; i < target_count; ++i) {
        if (nn_gd_cache_match_input(fp, target_paths[i]) != 0) {
            return -1;
        }
    }
    return 0;
}

static int nn_gd_cache_write_string_list(FILE *fp, const NNGDStringList *strings) {
    size_t i;
    if (strings == NULL || nn_gd_cache_write_u64(fp, (uint64_t)strings->count) != 0) {
        return -1;
    }
    for (i = 0U; i < strings->count; ++i) {
        if (nn_gd_cache_write_string(fp, strings->items[i]) != 0) {
            return -1;
        }
    }
    return 0;
}

static int nn_gd_cache_read_string_list(FILE *fp,
                                        NNGDStringList *strings,
                                        uint64_t max_count) {
    uint64_t count;
    size_t i;
    if (nn_gd_cache_read_u64(fp, &count) != 0 ||
        count > max_count ||
        count > (uint64_t)SIZE_MAX ||
        nn_gd_reserve_strings(strings, (size_t)count) != 0) {
        return -1;
    }
    for (i = 0U; i < (size_t)count; ++i) {
        char *value = nn_gd_cache_read_string(fp);
        if (value == NULL) {
            return -1;
        }
        strings->items[strings->count++] = value;
    }
    return 0;
}

static int nn_gd_cache_write_vocab(FILE *fp, const NNGDStringList *vocab) {
    return nn_gd_cache_write_string_list(fp, vocab);
}

static int nn_gd_cache_read_vocab(FILE *fp, NNGDStringList *vocab) {
    return nn_gd_cache_read_string_list(fp,
                                        vocab,
                                        NN_GD_ARC_RANKER_CACHE_MAX_VOCAB);
}

static int nn_gd_cache_write_long(FILE *fp, long value) {
    int32_t stored;
    if (value < (long)INT32_MIN || value > (long)INT32_MAX) {
        return -1;
    }
    stored = (int32_t)value;
    return fwrite(&stored, sizeof(stored), 1U, fp) == 1U ? 0 : -1;
}

static int nn_gd_cache_read_long(FILE *fp, long *value) {
    int32_t stored;
    if (fread(&stored, sizeof(stored), 1U, fp) != 1U) {
        return -1;
    }
    *value = (long)stored;
    return 0;
}

static int nn_gd_cache_add_field_string(NNGDStringList *strings,
                                        NNGDStringTable *table,
                                        const char *value) {
    if (value == NULL) {
        value = "";
    }
    return nn_gd_string_table_add_unique(strings, table, value);
}

static int nn_gd_cache_add_arc_attempt_field_strings(
    NNGDStringList *strings,
    NNGDStringTable *table,
    const NNGDArcPipelineAttempt *attempt) {
    const char *values[] = {
        attempt->split_name,
        attempt->family,
        attempt->task_id,
        attempt->split,
        attempt->attempt_id,
        attempt->library_id,
        attempt->pipeline_kind,
        attempt->op_name,
        attempt->composition_kind,
        attempt->normalization_frame,
        attempt->typed_stages,
        attempt->primitive_contract,
        attempt->rejection_reason,
        attempt->failure_stage,
        attempt->failure_detail,
        attempt->contrast_accepted_attempt_id,
        attempt->contrast_pair_kind,
    };
    size_t i;
    for (i = 0U; i < sizeof(values) / sizeof(values[0]); ++i) {
        if (nn_gd_cache_add_field_string(strings, table, values[i]) != 0) {
            return -1;
        }
    }
    return 0;
}

static int nn_gd_cache_build_arc_field_strings(
    const NNGDArcPipelineAttemptList *attempts,
    NNGDStringList *strings,
    NNGDStringTable *table) {
    size_t i;
    if (attempts == NULL) {
        return -1;
    }
    for (i = 0U; i < attempts->count; ++i) {
        if (nn_gd_cache_add_arc_attempt_field_strings(strings,
                                                      table,
                                                      &attempts->items[i]) != 0) {
            return -1;
        }
    }
    return strings->count <= NN_GD_ARC_RANKER_CACHE_MAX_FIELD_STRINGS ? 0 : -1;
}

static int nn_gd_cache_write_string_ref(FILE *fp,
                                        const NNGDStringList *strings,
                                        const NNGDStringTable *table,
                                        const char *value) {
    long index;
    if (value == NULL || value[0] == '\0') {
        return nn_gd_cache_write_u32(fp, NN_GD_ARC_RANKER_CACHE_EMPTY_STRING_REF);
    }
    index = nn_gd_string_table_find(table, strings, value);
    if (index < 0L || (uint64_t)index >= NN_GD_ARC_RANKER_CACHE_EMPTY_STRING_REF) {
        return -1;
    }
    return nn_gd_cache_write_u32(fp, (uint32_t)index);
}

static char *nn_gd_cache_read_string_ref(FILE *fp,
                                         const NNGDStringList *strings) {
    uint32_t index;
    if (strings == NULL ||
        nn_gd_cache_read_u32(fp, &index) != 0 ||
        (index != NN_GD_ARC_RANKER_CACHE_EMPTY_STRING_REF &&
         (size_t)index >= strings->count)) {
        return NULL;
    }
    if (index == NN_GD_ARC_RANKER_CACHE_EMPTY_STRING_REF) {
        return nn_gd_strdup("");
    }
    return nn_gd_strdup(strings->items[index]);
}

static int nn_gd_cache_write_arc_attempt(FILE *fp,
                                         const NNGDStringList *field_strings,
                                         const NNGDStringTable *field_table,
                                         uint32_t token_index_width,
                                         const NNGDArcPipelineAttempt *attempt) {
    size_t i;
    if (nn_gd_cache_write_string_ref(fp,
                                     field_strings,
                                     field_table,
                                     attempt->split_name) != 0 ||
        nn_gd_cache_write_string_ref(fp, field_strings, field_table, attempt->family) != 0 ||
        nn_gd_cache_write_string_ref(fp, field_strings, field_table, attempt->task_id) != 0 ||
        nn_gd_cache_write_string_ref(fp, field_strings, field_table, attempt->split) != 0 ||
        nn_gd_cache_write_string_ref(fp,
                                     field_strings,
                                     field_table,
                                     attempt->attempt_id) != 0 ||
        nn_gd_cache_write_string_ref(fp,
                                     field_strings,
                                     field_table,
                                     attempt->library_id) != 0 ||
        nn_gd_cache_write_long(fp, attempt->attempt_index) != 0 ||
        nn_gd_cache_write_string_ref(fp,
                                     field_strings,
                                     field_table,
                                     attempt->pipeline_kind) != 0 ||
        nn_gd_cache_write_string_ref(fp, field_strings, field_table, attempt->op_name) != 0 ||
        nn_gd_cache_write_string_ref(fp,
                                     field_strings,
                                     field_table,
                                     attempt->composition_kind) != 0 ||
        nn_gd_cache_write_string_ref(fp,
                                     field_strings,
                                     field_table,
                                     attempt->normalization_frame) != 0 ||
        nn_gd_cache_write_string_ref(fp,
                                     field_strings,
                                     field_table,
                                     attempt->typed_stages) != 0 ||
        nn_gd_cache_write_string_ref(fp,
                                     field_strings,
                                     field_table,
                                     attempt->primitive_contract) != 0 ||
        nn_gd_cache_write_u32(fp, attempt->accepted ? 1U : 0U) != 0 ||
        nn_gd_cache_write_string_ref(fp,
                                     field_strings,
                                     field_table,
                                     attempt->rejection_reason) != 0 ||
        nn_gd_cache_write_string_ref(fp,
                                     field_strings,
                                     field_table,
                                     attempt->failure_stage) != 0 ||
        nn_gd_cache_write_string_ref(fp,
                                     field_strings,
                                     field_table,
                                     attempt->failure_detail) != 0 ||
        nn_gd_cache_write_long(fp, attempt->proposal_count) != 0 ||
        nn_gd_cache_write_long(fp, attempt->support_tasks) != 0 ||
        nn_gd_cache_write_double(fp, attempt->utility_mean) != 0 ||
        nn_gd_cache_write_string_ref(fp,
                                     field_strings,
                                     field_table,
                                     attempt->contrast_accepted_attempt_id) != 0 ||
        nn_gd_cache_write_long(fp, attempt->contrast_slot_distance) != 0 ||
        nn_gd_cache_write_string_ref(fp,
                                     field_strings,
                                     field_table,
                                     attempt->contrast_pair_kind) != 0 ||
        nn_gd_cache_write_long(fp, attempt->contrast_rejected_sibling_count) != 0 ||
        nn_gd_cache_write_long(fp, attempt->verifier_train_pairs) != 0 ||
        nn_gd_cache_write_long(fp, attempt->verifier_checked_cells) != 0 ||
        nn_gd_cache_write_long(fp, attempt->verifier_mismatches) != 0 ||
        nn_gd_cache_write_long(fp, attempt->verifier_unsupported_pairs) != 0 ||
        nn_gd_cache_write_long(fp, attempt->verifier_apply_failed_pairs) != 0 ||
        nn_gd_cache_write_long(fp, attempt->verifier_shape_mismatch_pairs) != 0 ||
        nn_gd_cache_write_long(fp, attempt->verifier_missing_output_pairs) != 0 ||
        nn_gd_cache_write_long(fp, attempt->verifier_first_failing_pair) != 0 ||
        nn_gd_cache_write_double(fp, attempt->verifier_mismatch_ratio) != 0 ||
        nn_gd_cache_write_double(fp, attempt->verifier_score) != 0 ||
        attempt->token_count > UINT32_MAX ||
        attempt->token_index_count > UINT32_MAX ||
        nn_gd_cache_write_u32(fp, (uint32_t)attempt->token_count) != 0 ||
        nn_gd_cache_write_u32(fp, (uint32_t)attempt->token_index_count) != 0) {
        return -1;
    }
    for (i = 0U; i < attempt->token_index_count; ++i) {
        if (token_index_width == 2U) {
            if (attempt->token_indices[i] > UINT16_MAX ||
                nn_gd_cache_write_u16(fp, (uint16_t)attempt->token_indices[i]) != 0) {
                return -1;
            }
        } else if (token_index_width == 4U) {
            if (attempt->token_indices[i] > UINT32_MAX ||
                nn_gd_cache_write_u32(fp, (uint32_t)attempt->token_indices[i]) != 0) {
                return -1;
            }
        } else {
            return -1;
        }
    }
    return 0;
}

static int nn_gd_cache_read_arc_attempt(FILE *fp,
                                        NNGDArcPipelineAttemptList *attempts,
                                        size_t vocab_count,
                                        const NNGDStringList *field_strings,
                                        uint32_t token_index_width) {
    NNGDArcPipelineAttempt *attempt;
    uint32_t accepted;
    uint32_t token_count;
    uint32_t token_index_count;
    size_t i;
    if (nn_gd_arc_attempt_list_reserve(attempts, attempts->count + 1U) != 0) {
        return -1;
    }
    attempt = &attempts->items[attempts->count];
    memset(attempt, 0, sizeof(*attempt));
    attempt->split_name = nn_gd_cache_read_string_ref(fp, field_strings);
    attempt->family = nn_gd_cache_read_string_ref(fp, field_strings);
    attempt->task_id = nn_gd_cache_read_string_ref(fp, field_strings);
    attempt->split = nn_gd_cache_read_string_ref(fp, field_strings);
    attempt->attempt_id = nn_gd_cache_read_string_ref(fp, field_strings);
    attempt->library_id = nn_gd_cache_read_string_ref(fp, field_strings);
    if (nn_gd_cache_read_long(fp, &attempt->attempt_index) != 0) {
        goto fail;
    }
    attempt->pipeline_kind = nn_gd_cache_read_string_ref(fp, field_strings);
    attempt->op_name = nn_gd_cache_read_string_ref(fp, field_strings);
    attempt->visual_program = nn_gd_strdup("");
    attempt->slot_contract = nn_gd_strdup("");
    attempt->composition_kind = nn_gd_cache_read_string_ref(fp, field_strings);
    attempt->normalization_frame = nn_gd_cache_read_string_ref(fp, field_strings);
    attempt->typed_stages = nn_gd_cache_read_string_ref(fp, field_strings);
    attempt->primitive_contract = nn_gd_cache_read_string_ref(fp, field_strings);
    attempt->slot_tokens = nn_gd_strdup("");
    if (nn_gd_cache_read_u32(fp, &accepted) != 0) {
        goto fail;
    }
    attempt->accepted = accepted != 0;
    attempt->rejection_reason = nn_gd_cache_read_string_ref(fp, field_strings);
    attempt->failure_stage = nn_gd_cache_read_string_ref(fp, field_strings);
    attempt->failure_detail = nn_gd_cache_read_string_ref(fp, field_strings);
    if (nn_gd_cache_read_long(fp, &attempt->proposal_count) != 0 ||
        nn_gd_cache_read_long(fp, &attempt->support_tasks) != 0 ||
        nn_gd_cache_read_double(fp, &attempt->utility_mean) != 0) {
        goto fail;
    }
    attempt->contrast_accepted_attempt_id =
        nn_gd_cache_read_string_ref(fp, field_strings);
    if (nn_gd_cache_read_long(fp, &attempt->contrast_slot_distance) != 0) {
        goto fail;
    }
    attempt->contrast_pair_kind = nn_gd_cache_read_string_ref(fp, field_strings);
    if (nn_gd_cache_read_long(fp, &attempt->contrast_rejected_sibling_count) != 0 ||
        nn_gd_cache_read_long(fp, &attempt->verifier_train_pairs) != 0 ||
        nn_gd_cache_read_long(fp, &attempt->verifier_checked_cells) != 0 ||
        nn_gd_cache_read_long(fp, &attempt->verifier_mismatches) != 0 ||
        nn_gd_cache_read_long(fp, &attempt->verifier_unsupported_pairs) != 0 ||
        nn_gd_cache_read_long(fp, &attempt->verifier_apply_failed_pairs) != 0 ||
        nn_gd_cache_read_long(fp, &attempt->verifier_shape_mismatch_pairs) != 0 ||
        nn_gd_cache_read_long(fp, &attempt->verifier_missing_output_pairs) != 0 ||
        nn_gd_cache_read_long(fp, &attempt->verifier_first_failing_pair) != 0 ||
        nn_gd_cache_read_double(fp, &attempt->verifier_mismatch_ratio) != 0 ||
        nn_gd_cache_read_double(fp, &attempt->verifier_score) != 0 ||
        nn_gd_cache_read_u32(fp, &token_count) != 0 ||
        nn_gd_cache_read_u32(fp, &token_index_count) != 0 ||
        token_index_count > token_count) {
        goto fail;
    }
    attempt->token_count = (size_t)token_count;
    attempt->token_index_count = (size_t)token_index_count;
    if (attempt->token_index_count > 0U) {
        attempt->token_indices =
            (size_t *)calloc(attempt->token_index_count, sizeof(size_t));
        if (attempt->token_indices == NULL) {
            goto fail;
        }
        for (i = 0U; i < attempt->token_index_count; ++i) {
            size_t token_index;
            if (token_index_width == 2U) {
                uint16_t stored;
                if (nn_gd_cache_read_u16(fp, &stored) != 0) {
                    goto fail;
                }
                token_index = (size_t)stored;
            } else if (token_index_width == 4U) {
                uint32_t stored;
                if (nn_gd_cache_read_u32(fp, &stored) != 0) {
                    goto fail;
                }
                token_index = (size_t)stored;
            } else {
                goto fail;
            }
            if (token_index >= vocab_count) {
                goto fail;
            }
            attempt->token_indices[i] = token_index;
        }
    }
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
        attempt->contrast_pair_kind == NULL) {
        goto fail;
    }
    attempt->contrast_accepted_index = SIZE_MAX;
    ++attempts->count;
    return 0;

fail:
    nn_gd_arc_attempt_free(attempt);
    memset(attempt, 0, sizeof(*attempt));
    return -1;
}

static int nn_gd_cache_write_arc_attempts(FILE *fp,
                                          const NNGDArcPipelineAttemptList *attempts,
                                          const NNGDStringList *field_strings,
                                          const NNGDStringTable *field_table,
                                          uint32_t token_index_width) {
    size_t i;
    if (attempts == NULL ||
        nn_gd_cache_write_u64(fp, (uint64_t)attempts->count) != 0) {
        return -1;
    }
    for (i = 0U; i < attempts->count; ++i) {
        if (nn_gd_cache_write_arc_attempt(fp,
                                          field_strings,
                                          field_table,
                                          token_index_width,
                                          &attempts->items[i]) != 0) {
            return -1;
        }
    }
    return 0;
}

static int nn_gd_cache_read_arc_attempts(FILE *fp,
                                         NNGDArcPipelineAttemptList *attempts,
                                         size_t vocab_count,
                                         const NNGDStringList *field_strings,
                                         uint32_t token_index_width) {
    uint64_t count;
    size_t i;
    if (nn_gd_cache_read_u64(fp, &count) != 0 ||
        count > NN_GD_ARC_RANKER_CACHE_MAX_ATTEMPTS ||
        count > (uint64_t)SIZE_MAX) {
        return -1;
    }
    for (i = 0U; i < (size_t)count; ++i) {
        if (nn_gd_cache_read_arc_attempt(fp,
                                         attempts,
                                         vocab_count,
                                         field_strings,
                                         token_index_width) != 0) {
            return -1;
        }
    }
    return 0;
}

static int nn_gd_write_arc_ranker_cache(
    const char *cache_path,
    NNGDArcRankerCacheMode mode,
    const char *const *source_paths,
    size_t source_count,
    const char *const *target_paths,
    size_t target_count,
    const NNGDArcPipelineAttemptList *attempts,
    const NNGDStringList *vocab) {
    static const unsigned char magic[8] = {'C', 'R', 'L', 'P', 'A', 'R', 'C', '1'};
    NNGDStringList field_strings = {0};
    NNGDStringTable field_table = {0};
    FILE *fp;
    int status = -1;
    uint32_t token_index_width;
    if (cache_path == NULL || cache_path[0] == '\0') {
        return 0;
    }
    if (nn_gd_cache_build_arc_field_strings(attempts,
                                            &field_strings,
                                            &field_table) != 0) {
        goto done;
    }
    fp = fopen(cache_path, "wb");
    if (fp == NULL) {
        goto done;
    }
    token_index_width = vocab != NULL && vocab->count <= UINT16_MAX ? 2U : 4U;
    if (fwrite(magic, 1U, sizeof(magic), fp) != sizeof(magic) ||
        nn_gd_cache_write_u64(fp, NN_GD_ARC_RANKER_CACHE_VERSION) != 0 ||
        nn_gd_cache_write_u64(fp, (uint64_t)mode) != 0 ||
        nn_gd_cache_write_inputs(fp,
                                 source_paths,
                                 source_count,
                                 target_paths,
                                 target_count) != 0 ||
        nn_gd_cache_write_vocab(fp, vocab) != 0 ||
        nn_gd_cache_write_string_list(fp, &field_strings) != 0 ||
        nn_gd_cache_write_u32(fp, token_index_width) != 0 ||
        nn_gd_cache_write_arc_attempts(fp,
                                       attempts,
                                       &field_strings,
                                       &field_table,
                                       token_index_width) != 0 ||
        ferror(fp)) {
        goto close_file;
    }
    status = 0;

close_file:
    if (fclose(fp) != 0) {
        status = -1;
    }
done:
    nn_gd_string_table_free(&field_table);
    nn_gd_string_list_free(&field_strings);
    return status;
}

static int nn_gd_read_arc_ranker_cache(
    const char *cache_path,
    NNGDArcRankerCacheMode mode,
    const char *const *source_paths,
    size_t source_count,
    const char *const *target_paths,
    size_t target_count,
    NNGDArcPipelineAttemptList *attempts,
    NNGDStringList *vocab) {
    static const unsigned char expected_magic[8] = {'C', 'R', 'L', 'P', 'A', 'R', 'C', '1'};
    unsigned char magic[8];
    uint64_t version;
    uint64_t cached_mode;
    NNGDStringList field_strings = {0};
    uint32_t token_index_width;
    FILE *fp;
    int status = -1;
    if (cache_path == NULL || cache_path[0] == '\0') {
        return -1;
    }
    fp = fopen(cache_path, "rb");
    if (fp == NULL) {
        return -1;
    }
    if (fread(magic, 1U, sizeof(magic), fp) != sizeof(magic) ||
        memcmp(magic, expected_magic, sizeof(magic)) != 0 ||
        nn_gd_cache_read_u64(fp, &version) != 0 ||
        version != NN_GD_ARC_RANKER_CACHE_VERSION ||
        nn_gd_cache_read_u64(fp, &cached_mode) != 0 ||
        cached_mode != (uint64_t)mode ||
        nn_gd_cache_match_inputs(fp,
                                 source_paths,
                                 source_count,
                                 target_paths,
                                 target_count) != 0 ||
        nn_gd_cache_read_vocab(fp, vocab) != 0 ||
        nn_gd_cache_read_string_list(fp,
                                     &field_strings,
                                     NN_GD_ARC_RANKER_CACHE_MAX_FIELD_STRINGS) != 0 ||
        nn_gd_cache_read_u32(fp, &token_index_width) != 0 ||
        (token_index_width != 2U && token_index_width != 4U) ||
        nn_gd_cache_read_arc_attempts(fp,
                                      attempts,
                                      vocab->count,
                                      &field_strings,
                                      token_index_width) != 0 ||
        ferror(fp)) {
        goto done;
    }
    status = 0;

done:
    fclose(fp);
    if (status != 0) {
        nn_gd_arc_attempt_list_free(attempts);
        nn_gd_string_list_free(vocab);
    }
    nn_gd_string_list_free(&field_strings);
    return status;
}

static void nn_gd_resolve_arc_contrast_pairs(NNGDArcPipelineAttemptList *attempts,
                                             size_t *source_pair_count,
                                             size_t *target_pair_count) {
    size_t *accepted_indices;
    size_t accepted_count = 0U;
    size_t i;
    size_t j;
    if (source_pair_count != NULL) {
        *source_pair_count = 0U;
    }
    if (target_pair_count != NULL) {
        *target_pair_count = 0U;
    }
    accepted_indices = (size_t *)calloc(attempts->count == 0U ? 1U : attempts->count,
                                        sizeof(size_t));
    if (accepted_indices == NULL) {
        return;
    }
    for (i = 0U; i < attempts->count; ++i) {
        if (attempts->items[i].accepted) {
            accepted_indices[accepted_count++] = i;
        }
    }
    for (i = 0U; i < attempts->count; ++i) {
        NNGDArcPipelineAttempt *attempt = &attempts->items[i];
        attempt->contrast_accepted_index = SIZE_MAX;
        if (attempt->accepted || attempt->contrast_accepted_attempt_id == NULL ||
            attempt->contrast_accepted_attempt_id[0] == '\0') {
            continue;
        }
        for (j = 0U; j < accepted_count; ++j) {
            const NNGDArcPipelineAttempt *candidate = &attempts->items[accepted_indices[j]];
            if (strcmp(candidate->attempt_id, attempt->contrast_accepted_attempt_id) == 0) {
                attempt->contrast_accepted_index = accepted_indices[j];
                if (strcmp(attempt->split_name, "source") == 0 && source_pair_count != NULL) {
                    ++*source_pair_count;
                } else if (strcmp(attempt->split_name, "target") == 0 &&
                           target_pair_count != NULL) {
                    ++*target_pair_count;
                }
                break;
            }
        }
    }
    free(accepted_indices);
}

static int nn_gd_build_arc_vocab(const NNGDArcPipelineAttemptList *attempts,
                                 NNGDStringList *vocab) {
    NNGDStringTable table = {0};
    int status = -1;
    size_t i;
    size_t j;
    for (i = 0U; i < attempts->count; ++i) {
        const NNGDArcPipelineAttempt *attempt = &attempts->items[i];
        for (j = 0U; j < attempt->tokens.count; ++j) {
            if (nn_gd_string_table_add_unique(vocab,
                                              &table,
                                              attempt->tokens.items[j]) != 0) {
                goto done;
            }
        }
    }
    nn_gd_string_list_sort(vocab);
    status = 0;

done:
    nn_gd_string_table_free(&table);
    return status;
}

static int nn_gd_index_arc_attempt_tokens(NNGDArcPipelineAttemptList *attempts,
                                          const NNGDStringList *vocab) {
    NNGDStringTable index = {0};
    int status = -1;
    size_t i;
    if (attempts == NULL || vocab == NULL) {
        return -1;
    }
    if (nn_gd_string_table_rehash(vocab, &index, vocab->count) != 0) {
        return -1;
    }
    for (i = 0U; i < attempts->count; ++i) {
        NNGDArcPipelineAttempt *attempt = &attempts->items[i];
        size_t j;
        free(attempt->token_indices);
        attempt->token_indices = NULL;
        attempt->token_index_count = 0U;
        attempt->token_count = attempt->tokens.count;
        if (attempt->tokens.count == 0U) {
            continue;
        }
        attempt->token_indices =
            (size_t *)calloc(attempt->tokens.count, sizeof(size_t));
        if (attempt->token_indices == NULL) {
            goto done;
        }
        for (j = 0U; j < attempt->tokens.count; ++j) {
            long token_index = nn_gd_string_table_find(&index, vocab, attempt->tokens.items[j]);
            if (token_index >= 0L) {
                attempt->token_indices[attempt->token_index_count++] =
                    (size_t)token_index;
            }
        }
        nn_gd_string_list_free(&attempt->tokens);
    }
    status = 0;

done:
    nn_gd_string_table_free(&index);
    return status;
}

static double nn_gd_arc_attempt_logit(const NNGDArcPipelineAttempt *attempt,
                                      const NNGDStringList *vocab,
                                      const double *weights,
                                      double bias) {
    size_t i;
    double z = bias;
    (void)vocab;
    for (i = 0U; i < attempt->token_index_count; ++i) {
        z += weights[attempt->token_indices[i]];
    }
    return z;
}

static double nn_gd_clamp01(double value) {
    if (value < 0.0) {
        return 0.0;
    }
    if (value > 1.0) {
        return 1.0;
    }
    return value;
}

static int nn_gd_str_contains(const char *haystack, const char *needle) {
    return haystack != NULL && needle != NULL && needle[0] != '\0' &&
           strstr(haystack, needle) != NULL;
}

static int nn_gd_arc_attempt_is_sequence_block_renderer(
    const NNGDArcPipelineAttempt *attempt) {
    if (attempt == NULL) {
        return 0;
    }
    return nn_gd_str_contains(attempt->visual_program, "repeat") ||
           nn_gd_str_contains(attempt->visual_program, "count") ||
           nn_gd_str_contains(attempt->visual_program, "template") ||
           nn_gd_str_contains(attempt->typed_stages, "repeat") ||
           nn_gd_str_contains(attempt->typed_stages, "count") ||
           nn_gd_str_contains(attempt->typed_stages, "template") ||
           strcmp(attempt->op_name, "macrocell_substitute") == 0 ||
           strcmp(attempt->op_name, "marker_count_motif_tile") == 0 ||
           strcmp(attempt->op_name, "component_count_line") == 0 ||
           strcmp(attempt->op_name, "visual_combinator") == 0;
}

static int nn_gd_arc_attempt_is_component_selector_layout(
    const NNGDArcPipelineAttempt *attempt) {
    if (attempt == NULL) {
        return 0;
    }
    return nn_gd_str_contains(attempt->primitive_contract, "selector:components") ||
           nn_gd_str_contains(attempt->primitive_contract, "selector:object") ||
           nn_gd_str_contains(attempt->normalization_frame, "object") ||
           nn_gd_str_contains(attempt->normalization_frame, "component") ||
           nn_gd_str_contains(attempt->typed_stages, "select_components") ||
           strcmp(attempt->op_name, "visual_combinator") == 0 ||
           strcmp(attempt->op_name, "object_mask_extract") == 0 ||
           strcmp(attempt->op_name, "object_color_tile") == 0 ||
           strcmp(attempt->op_name, "component_color_summary") == 0;
}

static int nn_gd_arc_aux_head_applies(const NNGDArcPipelineAttempt *attempt,
                                      NNGDArcAuxHeadKind kind) {
    if (attempt == NULL) {
        return 0;
    }
    if (kind == NN_GD_ARC_HEAD_SEQUENCE_BLOCK_RENDERER) {
        return nn_gd_arc_attempt_is_sequence_block_renderer(attempt);
    }
    if (kind == NN_GD_ARC_HEAD_COMPONENT_SELECTOR_LAYOUT) {
        return nn_gd_arc_attempt_is_component_selector_layout(attempt);
    }
    return 1;
}

static double nn_gd_arc_renderer_label(const NNGDArcPipelineAttempt *attempt) {
    double score;
    if (attempt == NULL) {
        return 0.0;
    }
    if (attempt->accepted) {
        return 1.0;
    }
    score = nn_gd_clamp01(attempt->verifier_score);
    if (strcmp(attempt->failure_stage, "renderer_cells") == 0) {
        return score;
    }
    if (strcmp(attempt->failure_stage, "transform_or_color") == 0) {
        return 0.65 * score;
    }
    if (strcmp(attempt->failure_stage, "renderer_shape") == 0) {
        return 0.35 * score;
    }
    return 0.0;
}

static double nn_gd_arc_component_label(const NNGDArcPipelineAttempt *attempt) {
    double score;
    if (attempt == NULL) {
        return 0.0;
    }
    if (attempt->accepted) {
        return 1.0;
    }
    score = nn_gd_clamp01(attempt->verifier_score);
    if (strcmp(attempt->failure_stage, "selector_or_transform") == 0 ||
        strcmp(attempt->failure_stage, "unsupported_shape") == 0 ||
        attempt->verifier_apply_failed_pairs > 0L) {
        return 0.0;
    }
    if (strcmp(attempt->failure_stage, "renderer_cells") == 0) {
        return nn_gd_clamp01(0.45 + 0.55 * score);
    }
    if (strcmp(attempt->failure_stage, "transform_or_color") == 0) {
        return 0.45 * score;
    }
    if (strcmp(attempt->failure_stage, "renderer_shape") == 0) {
        return 0.25 * score;
    }
    return 0.25 * score;
}

static double nn_gd_arc_component_selector_layout_label(
    const NNGDArcPipelineAttempt *attempt) {
    double score;
    if (attempt == NULL) {
        return 0.0;
    }
    if (attempt->accepted) {
        return 1.0;
    }
    score = nn_gd_clamp01(attempt->verifier_score);
    if (strcmp(attempt->failure_stage, "selector_or_transform") == 0 ||
        attempt->verifier_apply_failed_pairs > 0L ||
        attempt->verifier_checked_cells <= 0L) {
        return 0.0;
    }
    if (strcmp(attempt->failure_stage, "renderer_shape") == 0) {
        return 0.20 * score;
    }
    if (strcmp(attempt->failure_stage, "transform_or_color") == 0) {
        return 0.55 * score;
    }
    if (strcmp(attempt->failure_stage, "renderer_cells") == 0) {
        return nn_gd_clamp01(0.50 + 0.50 * score);
    }
    return 0.35 * score;
}

static double nn_gd_arc_aux_label(const NNGDArcPipelineAttempt *attempt,
                                  NNGDArcAuxHeadKind kind) {
    switch (kind) {
        case NN_GD_ARC_HEAD_RENDERER:
            return nn_gd_arc_renderer_label(attempt);
        case NN_GD_ARC_HEAD_SEQUENCE_BLOCK_RENDERER:
            return nn_gd_arc_renderer_label(attempt);
        case NN_GD_ARC_HEAD_COMPONENT_SHAPE_COLOR:
            return nn_gd_arc_component_label(attempt);
        case NN_GD_ARC_HEAD_COMPONENT_SELECTOR_LAYOUT:
            return nn_gd_arc_component_selector_layout_label(attempt);
        default:
            return 0.0;
    }
}

static int nn_gd_train_arc_aux_head(const NNGDArcPipelineAttemptList *attempts,
                                    const NNGDStringList *vocab,
                                    NNGDArcAuxHead *head,
                                    unsigned epochs,
                                    double learning_rate) {
    unsigned epoch;
    size_t i;
    if (attempts == NULL || vocab == NULL || head == NULL || head->weights == NULL) {
        return -1;
    }
    for (epoch = 0U; epoch < epochs; ++epoch) {
        double loss = 0.0;
        size_t rows = 0U;
        for (i = 0U; i < attempts->count; ++i) {
            const NNGDArcPipelineAttempt *attempt = &attempts->items[i];
            double label;
            double pred;
            double error;
            double active;
            double scale;
            size_t j;
            if (strcmp(attempt->split_name, "source") != 0 ||
                !nn_gd_arc_aux_head_applies(attempt, head->kind)) {
                continue;
            }
            label = nn_gd_arc_aux_label(attempt, head->kind);
            pred = nn_gd_sigmoid(
                nn_gd_arc_attempt_logit(attempt, vocab, head->weights, head->bias));
            error = pred - label;
            active = attempt->token_count == 0U ? 1.0 : (double)attempt->token_count;
            scale = learning_rate * error / active;
            head->bias -= scale;
            for (j = 0U; j < attempt->token_index_count; ++j) {
                head->weights[attempt->token_indices[j]] -= scale;
            }
            loss += error * error;
            ++rows;
            if (epoch == 0U) {
                ++head->train_rows;
                if (label >= 0.95) {
                    ++head->positive_rows;
                }
                head->label_sum += label;
            }
        }
        head->final_loss = rows == 0U ? 0.0 : loss / (double)rows;
    }
    return 0;
}

static double nn_gd_arc_aux_head_score_applied(const NNGDArcPipelineAttempt *attempt,
                                               const NNGDStringList *vocab,
                                               const NNGDArcAuxHead *head,
                                               int applies) {
    if (attempt == NULL || vocab == NULL || head == NULL || head->weights == NULL ||
        !applies) {
        return 0.5;
    }
    return nn_gd_sigmoid(
        nn_gd_arc_attempt_logit(attempt, vocab, head->weights, head->bias));
}

static int nn_gd_compare_arc_scored_attempts(const void *a, const void *b) {
    const NNGDScoredArcAttempt *sa = (const NNGDScoredArcAttempt *)a;
    const NNGDScoredArcAttempt *sb = (const NNGDScoredArcAttempt *)b;
    const NNGDArcPipelineAttempt *aa =
        &nn_gd_sort_arc_attempts[sa->attempt_index];
    const NNGDArcPipelineAttempt *ab =
        &nn_gd_sort_arc_attempts[sb->attempt_index];
    int cmp = strcmp(aa->family, ab->family);
    if (cmp != 0) {
        return cmp;
    }
    cmp = strcmp(aa->task_id, ab->task_id);
    if (cmp != 0) {
        return cmp;
    }
    cmp = strcmp(aa->split, ab->split);
    if (cmp != 0) {
        return cmp;
    }
    if (sa->score < sb->score) {
        return 1;
    }
    if (sa->score > sb->score) {
        return -1;
    }
    return strcmp(aa->attempt_id, ab->attempt_id);
}

static uint64_t nn_gd_arc_pipeline_proposal_hash(const char *attempt_id) {
    uint64_t hash = UINT64_C(1469598103934665603);
    hash = nn_gd_fnv1a_update(hash, "arc-visual-pipeline-neural");
    hash ^= 0U;
    hash *= UINT64_C(1099511628211);
    hash = nn_gd_fnv1a_update(hash, attempt_id);
    return hash;
}

typedef struct {
    char *name;
    long count;
    long accepted;
    long evaluated_tasks;
    long top1_hits;
    long top3_hits;
    long retained_hits;
    double score_sum;
} NNGDArcNamedMetric;

typedef struct {
    NNGDArcNamedMetric *items;
    size_t count;
    size_t capacity;
} NNGDArcMetricList;

static int nn_gd_arc_metric_list_reserve(NNGDArcMetricList *list, size_t needed) {
    NNGDArcNamedMetric *items;
    size_t capacity;
    if (needed <= list->capacity) {
        return 0;
    }
    capacity = list->capacity == 0U ? 8U : list->capacity * 2U;
    while (capacity < needed) {
        capacity *= 2U;
    }
    items = (NNGDArcNamedMetric *)realloc(list->items,
                                          capacity * sizeof(NNGDArcNamedMetric));
    if (items == NULL) {
        return -1;
    }
    list->items = items;
    list->capacity = capacity;
    return 0;
}

static NNGDArcNamedMetric *nn_gd_arc_metric_get(NNGDArcMetricList *list,
                                                const char *name) {
    size_t i;
    if (name == NULL || name[0] == '\0') {
        name = "(none)";
    }
    for (i = 0U; i < list->count; ++i) {
        if (strcmp(list->items[i].name, name) == 0) {
            return &list->items[i];
        }
    }
    if (nn_gd_arc_metric_list_reserve(list, list->count + 1U) != 0) {
        return NULL;
    }
    memset(&list->items[list->count], 0, sizeof(list->items[list->count]));
    list->items[list->count].name = nn_gd_strdup(name);
    if (list->items[list->count].name == NULL) {
        return NULL;
    }
    return &list->items[list->count++];
}

static void nn_gd_arc_metric_list_free(NNGDArcMetricList *list) {
    size_t i;
    for (i = 0U; i < list->count; ++i) {
        free(list->items[i].name);
    }
    free(list->items);
    list->items = NULL;
    list->count = 0U;
    list->capacity = 0U;
}

static void nn_gd_write_arc_op_recovery(FILE *fp, const NNGDArcMetricList *metrics) {
    size_t i;
    fputc('{', fp);
    for (i = 0U; i < metrics->count; ++i) {
        const NNGDArcNamedMetric *metric = &metrics->items[i];
        if (i > 0U) {
            fputc(',', fp);
        }
        nn_gd_json_write_string(fp, metric->name);
        fprintf(fp,
                ":{\"evaluated_tasks\":%ld,\"top1_hits\":%ld,"
                "\"top3_hits\":%ld,\"retained_hits\":%ld,"
                "\"top1_rate\":%.8f,\"top3_rate\":%.8f,"
                "\"retained_rate\":%.8f}",
                metric->evaluated_tasks,
                metric->top1_hits,
                metric->top3_hits,
                metric->retained_hits,
                metric->evaluated_tasks == 0L
                    ? 0.0
                    : (double)metric->top1_hits / (double)metric->evaluated_tasks,
                metric->evaluated_tasks == 0L
                    ? 0.0
                    : (double)metric->top3_hits / (double)metric->evaluated_tasks,
                metric->evaluated_tasks == 0L
                    ? 0.0
                    : (double)metric->retained_hits / (double)metric->evaluated_tasks);
    }
    fputc('}', fp);
}

static void nn_gd_write_arc_reason_calibration(FILE *fp,
                                               const NNGDArcMetricList *metrics) {
    size_t i;
    fputc('{', fp);
    for (i = 0U; i < metrics->count; ++i) {
        const NNGDArcNamedMetric *metric = &metrics->items[i];
        if (i > 0U) {
            fputc(',', fp);
        }
        nn_gd_json_write_string(fp, metric->name);
        fprintf(fp,
                ":{\"retained\":%ld,\"accepted\":%ld,\"mean_score\":%.8f,"
                "\"acceptance_rate\":%.8f}",
                metric->count,
                metric->accepted,
                metric->count == 0L ? 0.0 : metric->score_sum / (double)metric->count,
                metric->count == 0L ? 0.0 : (double)metric->accepted / (double)metric->count);
    }
    fputc('}', fp);
}

static int nn_gd_arc_group_has_accepted_op(const NNGDScoredArcAttempt *scores,
                                           size_t start,
                                           size_t end,
                                           const NNGDArcPipelineAttemptList *attempts,
                                           const char *op_name,
                                           size_t limit) {
    size_t i;
    size_t capped = start + limit;
    if (capped > end) {
        capped = end;
    }
    for (i = start; i < capped; ++i) {
        const NNGDArcPipelineAttempt *attempt =
            &attempts->items[scores[i].attempt_index];
        if (attempt->accepted && strcmp(attempt->op_name, op_name) == 0) {
            return 1;
        }
    }
    return 0;
}

int nn_grammar_discovery_write_neural_sketch_proposals(
    const char *sketches_path,
    const char *examples_path,
    const char *proposals_output_path,
    const char *checkpoint_input_path,
    const char *checkpoint_output_path,
    unsigned epochs,
    double learning_rate) {
    NNGDSketchList sketches = {0};
    NNGDLabelList labels = {0};
    NNGDStringList vocab = {0};
    NNGDMemoryState memory = {0};
    double *weights = NULL;
    double bias = 0.0;
    FILE *output = NULL;
    int status = -1;
    unsigned epoch;
    size_t i;

    if (nn_gd_read_sketches(sketches_path, &sketches) != 0 ||
        nn_gd_read_labels(examples_path, &sketches, &labels) != 0 ||
        nn_gd_build_vocab(&sketches, &vocab) != 0) {
        goto done;
    }
    weights = (double *)calloc(vocab.count == 0U ? 1U : vocab.count, sizeof(double));
    if (weights == NULL) {
        goto done;
    }
    if (nn_gd_load_neural_checkpoint(checkpoint_input_path,
                                     &vocab,
                                     weights,
                                     &bias,
                                     &memory) != 0) {
        goto done;
    }
    for (epoch = 0U; epoch < epochs; ++epoch) {
        for (i = 0; i < labels.count; ++i) {
            const NNGDLabel *label = &labels.items[i];
            const NNGDSketch *sketch = &sketches.items[label->sketch_index];
            double z = nn_gd_sketch_logit(sketch, &vocab, weights, bias);
            double pred = nn_gd_sigmoid(z);
            double error = pred - label->label;
            double active = sketch->tokens.count == 0U ? 1.0 : (double)sketch->tokens.count;
            double scale = learning_rate * error / active;
            size_t j;
            bias -= learning_rate * error;
            for (j = 0; j < sketch->tokens.count; ++j) {
                long index = nn_gd_token_index(&vocab, sketch->tokens.items[j]);
                if (index >= 0L) {
                    weights[index] -= scale;
                }
            }
        }
    }
    memory.epochs_seen_after = memory.epochs_seen_before + epochs;
    memory.training_examples_seen_after =
        memory.training_examples_seen_before + labels.count;
    if (nn_gd_save_neural_checkpoint(checkpoint_output_path,
                                     &vocab,
                                     weights,
                                     bias,
                                     &memory) != 0) {
        goto done;
    }
    output = fopen(proposals_output_path, "w");
    if (output == NULL) {
        goto done;
    }
    if (nn_gd_write_neural_rows(output,
                                &sketches,
                                &labels,
                                &vocab,
                                weights,
                                bias,
                                epochs,
                                learning_rate,
                                &memory) != 0) {
        goto done;
    }
    status = 0;

done:
    if (output != NULL) {
        fclose(output);
    }
    free(weights);
    nn_gd_string_list_free(&vocab);
    nn_gd_label_list_free(&labels);
    nn_gd_sketch_list_free(&sketches);
    return status;
}

int nn_grammar_discovery_write_closed_loop_sketch_proposals(
    const char *sketches_path,
    const char *examples_path,
    const char *static_proposals_path,
    const char *proposals_output_path,
    const char *summary_output_path) {
    NNGDSketchList sketches = {0};
    NNGDLabelList labels = {0};
    double *static_scores = NULL;
    int *static_seen = NULL;
    size_t static_proposal_count = 0U;
    int status = -1;

    if (nn_gd_read_sketches(sketches_path, &sketches) != 0 ||
        nn_gd_read_labels(examples_path, &sketches, &labels) != 0) {
        goto done;
    }
    static_scores = (double *)calloc(sketches.count == 0U ? 1U : sketches.count,
                                     sizeof(double));
    static_seen = (int *)calloc(sketches.count == 0U ? 1U : sketches.count,
                                sizeof(int));
    if (static_scores == NULL || static_seen == NULL) {
        goto done;
    }
    if (nn_gd_read_static_sketch_scores(static_proposals_path,
                                        &sketches,
                                        static_scores,
                                        static_seen,
                                        &static_proposal_count) != 0) {
        goto done;
    }
    if (nn_gd_write_closed_loop_sketch_rows(proposals_output_path,
                                            &sketches,
                                            &labels,
                                            static_scores,
                                            static_seen,
                                            static_proposal_count,
                                            summary_output_path) != 0) {
        goto done;
    }
    status = 0;

done:
    free(static_scores);
    free(static_seen);
    nn_gd_label_list_free(&labels);
    nn_gd_sketch_list_free(&sketches);
    return status;
}

static int nn_gd_write_arc_visual_pipeline_ranker_indexed(
    NNGDArcPipelineAttemptList *attempts,
    const NNGDStringList *vocab,
    const char *proposals_output_path,
    const char *summary_output_path,
    unsigned epochs,
    double learning_rate,
    unsigned max_per_task) {
    NNGDScoredArcAttempt *target_scores = NULL;
    double *weights = NULL;
    double bias = 0.0;
    NNGDArcAuxHead aux_heads[NNGD_ARC_AUX_HEAD_COUNT] = {
        {"renderer_correctness", NN_GD_ARC_HEAD_RENDERER, NULL, 0.0, 0U, 0U, 0.0, 0.0, 0.0, 0U},
        {"sequence_block_renderer", NN_GD_ARC_HEAD_SEQUENCE_BLOCK_RENDERER, NULL, 0.0, 0U, 0U, 0.0, 0.0, 0.0, 0U},
        {"component_shape_color", NN_GD_ARC_HEAD_COMPONENT_SHAPE_COLOR, NULL, 0.0, 0U, 0U, 0.0, 0.0, 0.0, 0U},
        {"component_selector_layout", NN_GD_ARC_HEAD_COMPONENT_SELECTOR_LAYOUT, NULL, 0.0, 0U, 0U, 0.0, 0.0, 0.0, 0U},
    };
    FILE *proposals = NULL;
    FILE *summary = NULL;
    int status = -1;
    size_t source_count = 0U;
    size_t source_accepted = 0U;
    size_t source_rejected = 0U;
    size_t target_count = 0U;
    size_t target_accepted = 0U;
    size_t source_contrast_pair_count = 0U;
    size_t target_contrast_pair_count = 0U;
    size_t pairwise_training_updates = 0U;
    size_t pairwise_last_epoch_violations = 0U;
    double positive_weight = 1.0;
    unsigned epoch;
    unsigned aux_epochs;
    size_t i;
    size_t scored_count = 0U;
    long target_task_count = 0L;
    long evaluated_tasks = 0L;
    long no_accepted_target_task_count = 0L;
    long top1_hits = 0L;
    long top3_hits = 0L;
    long retained_hits = 0L;
    long proposal_count = 0L;
    NNGDArcMetricList op_recovery = {0};
    NNGDArcMetricList reason_calibration = {0};
    size_t head_index;

    if (attempts == NULL || vocab == NULL) {
        goto done;
    }
    if (max_per_task == 0U) {
        max_per_task = 1U;
    }
    aux_epochs = epochs > 16U ? 16U : epochs;
    nn_gd_resolve_arc_contrast_pairs(attempts,
                                     &source_contrast_pair_count,
                                     &target_contrast_pair_count);
    for (i = 0U; i < attempts->count; ++i) {
        const NNGDArcPipelineAttempt *attempt = &attempts->items[i];
        if (strcmp(attempt->split_name, "source") == 0) {
            ++source_count;
            if (attempt->accepted) {
                ++source_accepted;
            } else {
                ++source_rejected;
            }
        } else if (strcmp(attempt->split_name, "target") == 0) {
            ++target_count;
            if (attempt->accepted) {
                ++target_accepted;
            }
        }
    }
    if (source_accepted > 0U && source_rejected > source_accepted) {
        positive_weight = (double)source_rejected / (double)source_accepted;
        if (positive_weight > 64.0) {
            positive_weight = 64.0;
        }
    }
    weights = (double *)calloc(vocab->count == 0U ? 1U : vocab->count, sizeof(double));
    target_scores = (NNGDScoredArcAttempt *)calloc(target_count == 0U ? 1U : target_count,
                                                   sizeof(NNGDScoredArcAttempt));
    if (weights == NULL || target_scores == NULL) {
        goto done;
    }
    for (head_index = 0U;
         head_index < NNGD_ARC_AUX_HEAD_COUNT;
         ++head_index) {
        aux_heads[head_index].weights =
            (double *)calloc(vocab->count == 0U ? 1U : vocab->count, sizeof(double));
        if (aux_heads[head_index].weights == NULL) {
            goto done;
        }
    }
    for (epoch = 0U; epoch < epochs; ++epoch) {
        for (i = 0U; i < attempts->count; ++i) {
            const NNGDArcPipelineAttempt *attempt = &attempts->items[i];
            double label;
            double z;
            double pred;
            double error;
            double active;
            double row_weight;
            double scale;
            size_t j;
            if (strcmp(attempt->split_name, "source") != 0) {
                continue;
            }
            label = attempt->accepted ? 1.0 : attempt->verifier_score;
            if (label < 0.0) {
                label = 0.0;
            } else if (label > 1.0) {
                label = 1.0;
            }
            z = nn_gd_arc_attempt_logit(attempt, vocab, weights, bias);
            pred = nn_gd_sigmoid(z);
            row_weight = attempt->accepted ? positive_weight : 1.0 + label;
            error = (pred - label) * row_weight;
            active = attempt->token_count == 0U ? 1.0 : (double)attempt->token_count;
            scale = learning_rate * error / active;
            bias -= scale;
            for (j = 0U; j < attempt->token_index_count; ++j) {
                weights[attempt->token_indices[j]] -= scale;
            }
        }
        pairwise_last_epoch_violations = 0U;
        for (i = 0U; i < attempts->count; ++i) {
            const NNGDArcPipelineAttempt *negative = &attempts->items[i];
            const NNGDArcPipelineAttempt *positive;
            double pos_z;
            double neg_z;
            double pred;
            double pair_scale;
            double pos_active;
            double neg_active;
            size_t j;
            if (strcmp(negative->split_name, "source") != 0 ||
                negative->accepted ||
                negative->contrast_accepted_index == SIZE_MAX ||
                negative->contrast_accepted_index >= attempts->count) {
                continue;
            }
            positive = &attempts->items[negative->contrast_accepted_index];
            if (!positive->accepted || strcmp(positive->split_name, "source") != 0) {
                continue;
            }
            pos_z = nn_gd_arc_attempt_logit(positive, vocab, weights, bias);
            neg_z = nn_gd_arc_attempt_logit(negative, vocab, weights, bias);
            if (pos_z <= neg_z) {
                ++pairwise_last_epoch_violations;
            }
            pred = nn_gd_sigmoid(pos_z - neg_z);
            pair_scale = 0.5 * learning_rate * (pred - 1.0);
            pos_active =
                positive->token_count == 0U ? 1.0 : (double)positive->token_count;
            neg_active =
                negative->token_count == 0U ? 1.0 : (double)negative->token_count;
            for (j = 0U; j < positive->token_index_count; ++j) {
                weights[positive->token_indices[j]] -= pair_scale / pos_active;
            }
            for (j = 0U; j < negative->token_index_count; ++j) {
                weights[negative->token_indices[j]] += pair_scale / neg_active;
            }
            ++pairwise_training_updates;
        }
    }
    for (head_index = 0U;
         head_index < NNGD_ARC_AUX_HEAD_COUNT;
         ++head_index) {
        if (nn_gd_train_arc_aux_head(attempts,
                                     vocab,
                                     &aux_heads[head_index],
                                     aux_epochs,
                                     learning_rate) != 0) {
            goto done;
        }
    }
    for (i = 0U; i < attempts->count; ++i) {
        NNGDArcPipelineAttempt *attempt = &attempts->items[i];
        double aux_scores[NNGD_ARC_AUX_HEAD_COUNT];
        int aux_applies[NNGD_ARC_AUX_HEAD_COUNT];
        if (strcmp(attempt->split_name, "target") != 0) {
            continue;
        }
        for (head_index = 0U; head_index < NNGD_ARC_AUX_HEAD_COUNT; ++head_index) {
            aux_applies[head_index] =
                nn_gd_arc_aux_head_applies(attempt, aux_heads[head_index].kind);
            aux_scores[head_index] =
                nn_gd_arc_aux_head_score_applied(attempt,
                                                 vocab,
                                                 &aux_heads[head_index],
                                                 aux_applies[head_index]);
        }
        target_scores[scored_count].attempt_index = i;
        target_scores[scored_count].model_score =
            nn_gd_sigmoid(nn_gd_arc_attempt_logit(attempt, vocab, weights, bias));
        target_scores[scored_count].renderer_score = aux_scores[0];
        target_scores[scored_count].sequence_block_score = aux_scores[1];
        target_scores[scored_count].component_score = aux_scores[2];
        target_scores[scored_count].component_selector_layout_score = aux_scores[3];
        target_scores[scored_count].score =
            0.646 * target_scores[scored_count].model_score +
            0.350 * attempt->verifier_score +
            0.001 * target_scores[scored_count].renderer_score +
            0.001 * target_scores[scored_count].component_score +
            0.001 * target_scores[scored_count].component_selector_layout_score +
            0.001 * target_scores[scored_count].sequence_block_score;
        for (head_index = 0U;
             head_index < NNGD_ARC_AUX_HEAD_COUNT;
             ++head_index) {
            if (aux_applies[head_index]) {
                aux_heads[head_index].target_score_sum += aux_scores[head_index];
                ++aux_heads[head_index].target_rows;
            }
        }
        ++scored_count;
    }
    nn_gd_sort_arc_attempts = attempts->items;
    qsort(target_scores,
          scored_count,
          sizeof(NNGDScoredArcAttempt),
          nn_gd_compare_arc_scored_attempts);
    nn_gd_sort_arc_attempts = NULL;

    proposals = fopen(proposals_output_path, "w");
    summary = fopen(summary_output_path, "w");
    if (proposals == NULL || summary == NULL) {
        goto done;
    }
    for (i = 0U; i < scored_count;) {
        size_t start = i;
        size_t end;
        size_t rank;
        int group_has_accepted = 0;
        int group_top1_hit = 0;
        int group_top3_hit = 0;
        int group_retained_hit = 0;
        NNGDStringList accepted_ops = {0};
        const NNGDArcPipelineAttempt *first_attempt =
            &attempts->items[target_scores[start].attempt_index];
        for (end = start; end < scored_count; ++end) {
            const NNGDArcPipelineAttempt *attempt =
                &attempts->items[target_scores[end].attempt_index];
            if (strcmp(first_attempt->family, attempt->family) != 0 ||
                strcmp(first_attempt->task_id, attempt->task_id) != 0 ||
                strcmp(first_attempt->split, attempt->split) != 0) {
                break;
            }
            if (attempt->accepted) {
                group_has_accepted = 1;
                if (nn_gd_string_list_add_unique(&accepted_ops, attempt->op_name) != 0) {
                    nn_gd_string_list_free(&accepted_ops);
                    goto done;
                }
            }
        }
        for (rank = 0U; rank < max_per_task && start + rank < end; ++rank) {
            const NNGDScoredArcAttempt *score = &target_scores[start + rank];
            const NNGDArcPipelineAttempt *attempt =
                &attempts->items[score->attempt_index];
            NNGDArcNamedMetric *reason_metric =
                nn_gd_arc_metric_get(&reason_calibration, attempt->rejection_reason);
            uint64_t proposal_hash = nn_gd_arc_pipeline_proposal_hash(attempt->attempt_id);
            if (reason_metric == NULL) {
                nn_gd_string_list_free(&accepted_ops);
                goto done;
            }
            ++reason_metric->count;
            reason_metric->accepted += attempt->accepted ? 1L : 0L;
            reason_metric->score_sum += score->score;
            ++proposal_count;
            if (attempt->accepted) {
                group_retained_hit = 1;
                if (rank == 0U) {
                    group_top1_hit = 1;
                }
                if (rank < 3U) {
                    group_top3_hit = 1;
                }
            }
            fputs("{\"schema\":\"crlp.neural_grammar_discovery.arc_visual_pipeline_proposal.v1\","
                  "\"report_version\":1,\"proposal_id\":\"avpn:",
                  proposals);
            fprintf(proposals, "%016" PRIx64, proposal_hash);
            fputs("\",\"proposer\":\"neural_visual_pipeline_ranker\",\"family\":", proposals);
            nn_gd_json_write_string(proposals, attempt->family);
            fputs(",\"task_id\":", proposals);
            nn_gd_json_write_string(proposals, attempt->task_id);
            fputs(",\"split\":", proposals);
            nn_gd_json_write_string(proposals, attempt->split);
            fputs(",\"target_attempt_id\":", proposals);
            nn_gd_json_write_string(proposals, attempt->attempt_id);
            fputs(",\"library_id\":", proposals);
            nn_gd_json_write_string(proposals, attempt->library_id);
            fputs(",\"pipeline_kind\":", proposals);
            nn_gd_json_write_string(proposals, attempt->pipeline_kind);
            fputs(",\"op_name\":", proposals);
            nn_gd_json_write_string(proposals, attempt->op_name);
            fputs(",\"composition_kind\":", proposals);
            nn_gd_json_write_string(proposals, attempt->composition_kind);
            fputs(",\"normalization_frame\":", proposals);
            nn_gd_json_write_string(proposals, attempt->normalization_frame);
            fputs(",\"typed_stages\":", proposals);
            nn_gd_json_write_string(proposals, attempt->typed_stages);
            fputs(",\"primitive_contract\":", proposals);
            nn_gd_json_write_string(proposals, attempt->primitive_contract);
            fprintf(proposals,
                    ",\"score\":%.8f,\"target_accepted\":%s,"
                    "\"target_rejection_reason\":",
                    score->score,
                    attempt->accepted ? "true" : "false");
            nn_gd_json_write_string(proposals, attempt->rejection_reason);
            fputs(",\"failure_stage\":", proposals);
            nn_gd_json_write_string(proposals, attempt->failure_stage);
            fputs(",\"failure_detail\":", proposals);
            nn_gd_json_write_string(proposals, attempt->failure_detail);
            fprintf(proposals,
                    ",\"rank_features\":{\"epochs\":%u,\"learning_rate\":%.17g,"
                    "\"vocab_size\":%zu,\"source_attempt_count\":%zu,"
                    "\"source_accepted_count\":%zu,\"source_rejected_count\":%zu,"
                    "\"positive_weight\":%.8f,\"rank_within_task\":%zu,"
                    "\"tokens\":%zu,\"source_contrast_pairs\":%zu,"
                    "\"target_contrast_pairs\":%zu,"
                    "\"contrast_slot_distance\":%ld,"
                    "\"contrast_rejected_sibling_count\":%ld,"
                    "\"model_score\":%.8f,\"verifier_proxy_score\":%.8f,"
                    "\"renderer_model_score\":%.8f,"
                    "\"sequence_block_renderer_score\":%.8f,"
                    "\"component_shape_color_score\":%.8f,"
                    "\"component_selector_layout_score\":%.8f,"
                    "\"verifier_train_pairs\":%ld,"
                    "\"verifier_checked_cells\":%ld,"
                    "\"verifier_mismatches\":%ld,"
                    "\"verifier_unsupported_pairs\":%ld,"
                    "\"verifier_apply_failed_pairs\":%ld,"
                    "\"verifier_shape_mismatch_pairs\":%ld,"
                    "\"verifier_missing_output_pairs\":%ld,"
                    "\"verifier_first_failing_pair\":%ld,"
                    "\"verifier_mismatch_ratio\":%.8f,"
                    "\"proposal_count\":%ld,\"support_tasks\":%ld,"
                    "\"utility_mean\":%.8f},\"rank\":%zu}\n",
                    epochs,
                    learning_rate,
                    vocab->count,
                    source_count,
                    source_accepted,
                    source_rejected,
                    positive_weight,
                    rank + 1U,
                    attempt->token_count,
                    source_contrast_pair_count,
                    target_contrast_pair_count,
                    attempt->contrast_slot_distance,
                    attempt->contrast_rejected_sibling_count,
                    score->model_score,
                    attempt->verifier_score,
                    score->renderer_score,
                    score->sequence_block_score,
                    score->component_score,
                    score->component_selector_layout_score,
                    attempt->verifier_train_pairs,
                    attempt->verifier_checked_cells,
                    attempt->verifier_mismatches,
                    attempt->verifier_unsupported_pairs,
                    attempt->verifier_apply_failed_pairs,
                    attempt->verifier_shape_mismatch_pairs,
                    attempt->verifier_missing_output_pairs,
                    attempt->verifier_first_failing_pair,
                    attempt->verifier_mismatch_ratio,
                    attempt->proposal_count,
                    attempt->support_tasks,
                    attempt->utility_mean,
                    rank + 1U);
        }
        ++target_task_count;
        if (group_has_accepted) {
            size_t op_index;
            ++evaluated_tasks;
            top1_hits += group_top1_hit ? 1L : 0L;
            top3_hits += group_top3_hit ? 1L : 0L;
            retained_hits += group_retained_hit ? 1L : 0L;
            for (op_index = 0U; op_index < accepted_ops.count; ++op_index) {
                const char *op_name = accepted_ops.items[op_index];
                NNGDArcNamedMetric *metric = nn_gd_arc_metric_get(&op_recovery, op_name);
                if (metric == NULL) {
                    nn_gd_string_list_free(&accepted_ops);
                    goto done;
                }
                ++metric->evaluated_tasks;
                metric->top1_hits +=
                    nn_gd_arc_group_has_accepted_op(target_scores,
                                                    start,
                                                    end,
                                                    attempts,
                                                    op_name,
                                                    1U)
                        ? 1L
                        : 0L;
                metric->top3_hits +=
                    nn_gd_arc_group_has_accepted_op(target_scores,
                                                    start,
                                                    end,
                                                    attempts,
                                                    op_name,
                                                    3U)
                        ? 1L
                        : 0L;
                metric->retained_hits +=
                    nn_gd_arc_group_has_accepted_op(target_scores,
                                                    start,
                                                    end,
                                                    attempts,
                                                    op_name,
                                                    max_per_task)
                        ? 1L
                        : 0L;
            }
        } else {
            ++no_accepted_target_task_count;
        }
        nn_gd_string_list_free(&accepted_ops);
        i = end;
    }
    fputs("{\"schema\":\"crlp.neural_grammar_discovery.arc_visual_pipeline_neural_summary.v1\","
          "\"report_version\":1,\"proposer\":\"neural_visual_pipeline_ranker\",",
          summary);
    fprintf(summary,
            "\"source_attempt_count\":%zu,\"source_accepted_count\":%zu,"
            "\"source_rejected_count\":%zu,\"target_attempt_count\":%zu,"
            "\"target_accepted_count\":%zu,\"proposal_count\":%ld,"
            "\"target_task_count\":%ld,"
            "\"accepted_target_task_count\":%ld,"
            "\"no_accepted_target_task_count\":%ld,"
            "\"search_space_coverage_rate\":%.8f,"
            "\"evaluated_tasks\":%ld,"
            "\"top1_hits\":%ld,\"top3_hits\":%ld,\"retained_hits\":%ld,"
            "\"top1_rate\":%.8f,\"top3_rate\":%.8f,\"retained_rate\":%.8f,"
            "\"top1_all_task_rate\":%.8f,"
            "\"top3_all_task_rate\":%.8f,"
            "\"retained_all_task_rate\":%.8f,"
            "\"epochs\":%u,\"learning_rate\":%.17g,\"vocab_size\":%zu,"
            "\"positive_weight\":%.8f,\"source_contrast_pair_count\":%zu,"
            "\"target_contrast_pair_count\":%zu,"
            "\"pairwise_training_updates\":%zu,"
            "\"pairwise_last_epoch_violations\":%zu,"
            "\"auxiliary_head_epochs\":%u,"
            "\"rerank_model_weight\":0.646,\"rerank_verifier_proxy_weight\":0.35,"
            "\"rerank_renderer_weight\":0.001,"
            "\"rerank_component_shape_color_weight\":0.001,"
            "\"rerank_component_selector_layout_weight\":0.001,"
            "\"rerank_sequence_block_weight\":0.001,"
            "\"training_mode\":\"source_to_target_visual_pipeline_attempts_with_dense_verifier_targets_and_auxiliary_renderer_component_heads\",",
            source_count,
            source_accepted,
            source_rejected,
            target_count,
            target_accepted,
            proposal_count,
            target_task_count,
            evaluated_tasks,
            no_accepted_target_task_count,
            target_task_count == 0L
                ? 0.0
                : (double)evaluated_tasks / (double)target_task_count,
            evaluated_tasks,
            top1_hits,
            top3_hits,
            retained_hits,
            evaluated_tasks == 0L ? 0.0 : (double)top1_hits / (double)evaluated_tasks,
            evaluated_tasks == 0L ? 0.0 : (double)top3_hits / (double)evaluated_tasks,
            evaluated_tasks == 0L ? 0.0 : (double)retained_hits / (double)evaluated_tasks,
            target_task_count == 0L ? 0.0 : (double)top1_hits / (double)target_task_count,
            target_task_count == 0L ? 0.0 : (double)top3_hits / (double)target_task_count,
            target_task_count == 0L
                ? 0.0
                : (double)retained_hits / (double)target_task_count,
            epochs,
            learning_rate,
            vocab->count,
            positive_weight,
            source_contrast_pair_count,
            target_contrast_pair_count,
            pairwise_training_updates,
            pairwise_last_epoch_violations,
            aux_epochs);
    fputs("\"auxiliary_heads\":{", summary);
    for (head_index = 0U;
         head_index < NNGD_ARC_AUX_HEAD_COUNT;
         ++head_index) {
        const NNGDArcAuxHead *head = &aux_heads[head_index];
        if (head_index > 0U) {
            fputc(',', summary);
        }
        nn_gd_json_write_string(summary, head->name);
        fprintf(summary,
                ":{\"train_rows\":%zu,\"positive_rows\":%zu,"
                "\"mean_label\":%.8f,\"final_loss\":%.8f,"
                "\"target_rows\":%zu,\"target_mean_score\":%.8f}",
                head->train_rows,
                head->positive_rows,
                head->train_rows == 0U ? 0.0 : head->label_sum / (double)head->train_rows,
                head->final_loss,
                head->target_rows,
                head->target_rows == 0U
                    ? 0.0
                    : head->target_score_sum / (double)head->target_rows);
    }
    fputs("},", summary);
    fputs("\"by_proposer\":{\"neural_visual_pipeline_ranker\":{", summary);
    fprintf(summary,
            "\"target_task_count\":%ld,"
            "\"accepted_target_task_count\":%ld,"
            "\"no_accepted_target_task_count\":%ld,"
            "\"search_space_coverage_rate\":%.8f,"
            "\"evaluated_tasks\":%ld,\"top1_hits\":%ld,\"top3_hits\":%ld,"
            "\"retained_hits\":%ld,\"top1_rate\":%.8f,\"top3_rate\":%.8f,"
            "\"retained_rate\":%.8f,\"top1_all_task_rate\":%.8f,"
            "\"top3_all_task_rate\":%.8f,\"retained_all_task_rate\":%.8f}},"
            "\"by_accepted_op\":",
            target_task_count,
            evaluated_tasks,
            no_accepted_target_task_count,
            target_task_count == 0L
                ? 0.0
                : (double)evaluated_tasks / (double)target_task_count,
            evaluated_tasks,
            top1_hits,
            top3_hits,
            retained_hits,
            evaluated_tasks == 0L ? 0.0 : (double)top1_hits / (double)evaluated_tasks,
            evaluated_tasks == 0L ? 0.0 : (double)top3_hits / (double)evaluated_tasks,
            evaluated_tasks == 0L ? 0.0 : (double)retained_hits / (double)evaluated_tasks,
            target_task_count == 0L ? 0.0 : (double)top1_hits / (double)target_task_count,
            target_task_count == 0L ? 0.0 : (double)top3_hits / (double)target_task_count,
            target_task_count == 0L
                ? 0.0
                : (double)retained_hits / (double)target_task_count);
    nn_gd_write_arc_op_recovery(summary, &op_recovery);
    fputs(",\"calibration_by_rejection_reason\":", summary);
    nn_gd_write_arc_reason_calibration(summary, &reason_calibration);
    fputs("}\n", summary);
    if (ferror(proposals) || ferror(summary)) {
        goto done;
    }
    status = 0;

done:
    if (proposals != NULL) {
        fclose(proposals);
    }
    if (summary != NULL) {
        fclose(summary);
    }
    nn_gd_arc_metric_list_free(&reason_calibration);
    nn_gd_arc_metric_list_free(&op_recovery);
    free(target_scores);
    for (head_index = 0U;
         head_index < NNGD_ARC_AUX_HEAD_COUNT;
         ++head_index) {
        free(aux_heads[head_index].weights);
    }
    free(weights);
    return status;
}

static int nn_gd_write_arc_visual_pipeline_ranker_loaded(
    NNGDArcPipelineAttemptList *attempts,
    const char *proposals_output_path,
    const char *summary_output_path,
    unsigned epochs,
    double learning_rate,
    unsigned max_per_task) {
    NNGDStringList vocab = {0};
    int status = -1;
    if (nn_gd_build_arc_vocab(attempts, &vocab) != 0 ||
        nn_gd_index_arc_attempt_tokens(attempts, &vocab) != 0) {
        goto done;
    }
    status = nn_gd_write_arc_visual_pipeline_ranker_indexed(attempts,
                                                            &vocab,
                                                            proposals_output_path,
                                                            summary_output_path,
                                                            epochs,
                                                            learning_rate,
                                                            max_per_task);

done:
    nn_gd_string_list_free(&vocab);
    return status;
}

static int nn_gd_prepare_arc_ranker_from_rows(
    const char *attempt_rows_path,
    const char *cache_path,
    NNGDArcPipelineAttemptList *attempts,
    NNGDStringList *vocab) {
    const char *source_paths[1];
    source_paths[0] = attempt_rows_path;
    if (nn_gd_read_arc_ranker_cache(cache_path,
                                    NN_GD_ARC_RANKER_CACHE_TSV,
                                    source_paths,
                                    1U,
                                    NULL,
                                    0U,
                                    attempts,
                                    vocab) == 0) {
        return 0;
    }
    if (nn_gd_read_arc_pipeline_attempts(attempt_rows_path, attempts) != 0 ||
        nn_gd_build_arc_vocab(attempts, vocab) != 0 ||
        nn_gd_index_arc_attempt_tokens(attempts, vocab) != 0) {
        return -1;
    }
    (void)nn_gd_write_arc_ranker_cache(cache_path,
                                       NN_GD_ARC_RANKER_CACHE_TSV,
                                       source_paths,
                                       1U,
                                       NULL,
                                       0U,
                                       attempts,
                                       vocab);
    return 0;
}

static int nn_gd_prepare_arc_ranker_from_jsonl(
    const char *const *source_paths,
    size_t source_count,
    const char *const *target_paths,
    size_t target_count,
    const char *cache_path,
    NNGDArcPipelineAttemptList *attempts,
    NNGDStringList *vocab) {
    size_t i;
    if (source_paths == NULL || target_paths == NULL || source_count == 0U ||
        target_count == 0U) {
        return -1;
    }
    if (nn_gd_read_arc_ranker_cache(cache_path,
                                    NN_GD_ARC_RANKER_CACHE_JSONL,
                                    source_paths,
                                    source_count,
                                    target_paths,
                                    target_count,
                                    attempts,
                                    vocab) == 0) {
        return 0;
    }
    for (i = 0U; i < source_count; ++i) {
        if (nn_gd_read_arc_pipeline_attempts_jsonl(source_paths[i],
                                                   "source",
                                                   attempts) != 0) {
            return -1;
        }
    }
    for (i = 0U; i < target_count; ++i) {
        if (nn_gd_read_arc_pipeline_attempts_jsonl(target_paths[i],
                                                   "target",
                                                   attempts) != 0) {
            return -1;
        }
    }
    if (nn_gd_build_arc_vocab(attempts, vocab) != 0 ||
        nn_gd_index_arc_attempt_tokens(attempts, vocab) != 0) {
        return -1;
    }
    (void)nn_gd_write_arc_ranker_cache(cache_path,
                                       NN_GD_ARC_RANKER_CACHE_JSONL,
                                       source_paths,
                                       source_count,
                                       target_paths,
                                       target_count,
                                       attempts,
                                       vocab);
    return 0;
}

int nn_grammar_discovery_write_arc_visual_pipeline_ranker(
    const char *attempt_rows_path,
    const char *proposals_output_path,
    const char *summary_output_path,
    unsigned epochs,
    double learning_rate,
    unsigned max_per_task) {
    NNGDArcPipelineAttemptList attempts = {0};
    int status = -1;
    if (nn_gd_read_arc_pipeline_attempts(attempt_rows_path, &attempts) != 0) {
        goto done;
    }
    status = nn_gd_write_arc_visual_pipeline_ranker_loaded(&attempts,
                                                           proposals_output_path,
                                                           summary_output_path,
                                                           epochs,
                                                           learning_rate,
                                                           max_per_task);

done:
    nn_gd_arc_attempt_list_free(&attempts);
    return status;
}

int nn_grammar_discovery_write_arc_visual_pipeline_ranker_with_cache(
    const char *attempt_rows_path,
    const char *cache_path,
    const char *proposals_output_path,
    const char *summary_output_path,
    unsigned epochs,
    double learning_rate,
    unsigned max_per_task) {
    NNGDArcPipelineAttemptList attempts = {0};
    NNGDStringList vocab = {0};
    int status = -1;
    if (nn_gd_prepare_arc_ranker_from_rows(attempt_rows_path,
                                           cache_path,
                                           &attempts,
                                           &vocab) != 0) {
        goto done;
    }
    status = nn_gd_write_arc_visual_pipeline_ranker_indexed(&attempts,
                                                            &vocab,
                                                            proposals_output_path,
                                                            summary_output_path,
                                                            epochs,
                                                            learning_rate,
                                                            max_per_task);

done:
    nn_gd_string_list_free(&vocab);
    nn_gd_arc_attempt_list_free(&attempts);
    return status;
}

int nn_grammar_discovery_write_arc_visual_pipeline_ranker_from_jsonl(
    const char *const *source_paths,
    size_t source_count,
    const char *const *target_paths,
    size_t target_count,
    const char *proposals_output_path,
    const char *summary_output_path,
    unsigned epochs,
    double learning_rate,
    unsigned max_per_task) {
    NNGDArcPipelineAttemptList attempts = {0};
    int status = -1;
    size_t i;
    if (source_paths == NULL || target_paths == NULL || source_count == 0U ||
        target_count == 0U) {
        return -1;
    }
    for (i = 0U; i < source_count; ++i) {
        if (nn_gd_read_arc_pipeline_attempts_jsonl(source_paths[i],
                                                   "source",
                                                   &attempts) != 0) {
            goto done;
        }
    }
    for (i = 0U; i < target_count; ++i) {
        if (nn_gd_read_arc_pipeline_attempts_jsonl(target_paths[i],
                                                   "target",
                                                   &attempts) != 0) {
            goto done;
        }
    }
    status = nn_gd_write_arc_visual_pipeline_ranker_loaded(&attempts,
                                                           proposals_output_path,
                                                           summary_output_path,
                                                           epochs,
                                                           learning_rate,
                                                           max_per_task);

done:
    nn_gd_arc_attempt_list_free(&attempts);
    return status;
}

int nn_grammar_discovery_write_arc_visual_pipeline_ranker_from_jsonl_with_cache(
    const char *const *source_paths,
    size_t source_count,
    const char *const *target_paths,
    size_t target_count,
    const char *cache_path,
    const char *proposals_output_path,
    const char *summary_output_path,
    unsigned epochs,
    double learning_rate,
    unsigned max_per_task) {
    NNGDArcPipelineAttemptList attempts = {0};
    NNGDStringList vocab = {0};
    int status = -1;
    if (nn_gd_prepare_arc_ranker_from_jsonl(source_paths,
                                            source_count,
                                            target_paths,
                                            target_count,
                                            cache_path,
                                            &attempts,
                                            &vocab) != 0) {
        goto done;
    }
    status = nn_gd_write_arc_visual_pipeline_ranker_indexed(&attempts,
                                                            &vocab,
                                                            proposals_output_path,
                                                            summary_output_path,
                                                            epochs,
                                                            learning_rate,
                                                            max_per_task);

done:
    nn_gd_string_list_free(&vocab);
    nn_gd_arc_attempt_list_free(&attempts);
    return status;
}
