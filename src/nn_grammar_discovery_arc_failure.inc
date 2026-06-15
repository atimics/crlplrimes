typedef struct {
    char *proposal_id;
    char *target_attempt_id;
    char *target_rejection_reason;
    char *failure_stage;
    char *failure_detail;
    char *pipeline_kind;
    char *op_name;
    long rank;
    double score;
    int target_accepted;
    double model_score;
    double verifier_proxy_score;
    double renderer_model_score;
    double component_shape_color_score;
    double component_selector_layout_score;
    double sequence_block_renderer_score;
    double verifier_mismatch_ratio;
    long verifier_unsupported_pairs;
    long verifier_apply_failed_pairs;
    long verifier_shape_mismatch_pairs;
} NNGDArcFailureProposal;

typedef struct {
    NNGDArcFailureProposal *items;
    size_t count;
    size_t capacity;
} NNGDArcFailureProposalList;

typedef struct {
    char *family;
    char *task_id;
    char *split;
    long candidate_count;
    long accepted_candidate_count;
    NNGDArcHoleCounterList rejection_reasons;
    NNGDArcHoleCounterList failure_stages;
    NNGDArcHoleCounterList failure_details;
    NNGDArcHoleCounterList accepted_ops;
    NNGDArcHoleCounterList candidate_ops;
    NNGDArcHoleCounterList candidate_pipelines;
    NNGDArcFailureProposalList proposals;
} NNGDArcFailureTask;

typedef struct {
    NNGDArcFailureTask *items;
    size_t count;
    size_t capacity;
} NNGDArcFailureTaskList;

static int nn_gd_mkdirs(const char *dir) {
    char buffer[4096];
    size_t len;
    char *p;
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

static int nn_gd_mkdir_parent(const char *path) {
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
    if (slash == NULL) {
        return 0;
    }
    if (slash == buffer) {
        return 0;
    }
    *slash = '\0';
    return nn_gd_mkdirs(buffer);
}

static int nn_gd_arc_failure_task_key_compare_values(
    const char *family_a,
    const char *task_id_a,
    const char *split_a,
    const char *family_b,
    const char *task_id_b,
    const char *split_b) {
    int cmp = strcmp(family_a == NULL ? "" : family_a,
                     family_b == NULL ? "" : family_b);
    if (cmp != 0) {
        return cmp;
    }
    cmp = strcmp(task_id_a == NULL ? "" : task_id_a,
                 task_id_b == NULL ? "" : task_id_b);
    if (cmp != 0) {
        return cmp;
    }
    return strcmp(split_a == NULL ? "" : split_a,
                  split_b == NULL ? "" : split_b);
}

static int nn_gd_arc_failure_task_compare(const void *a, const void *b) {
    const NNGDArcFailureTask *ta = (const NNGDArcFailureTask *)a;
    const NNGDArcFailureTask *tb = (const NNGDArcFailureTask *)b;
    return nn_gd_arc_failure_task_key_compare_values(ta->family,
                                                     ta->task_id,
                                                     ta->split,
                                                     tb->family,
                                                     tb->task_id,
                                                     tb->split);
}

static int nn_gd_arc_failure_task_list_reserve(NNGDArcFailureTaskList *list,
                                               size_t needed) {
    NNGDArcFailureTask *items;
    size_t capacity;
    if (needed <= list->capacity) {
        return 0;
    }
    capacity = list->capacity == 0U ? 64U : list->capacity * 2U;
    while (capacity < needed) {
        capacity *= 2U;
    }
    items = (NNGDArcFailureTask *)realloc(list->items,
                                          capacity * sizeof(*list->items));
    if (items == NULL) {
        return -1;
    }
    list->items = items;
    list->capacity = capacity;
    return 0;
}

static NNGDArcFailureTask *nn_gd_arc_failure_task_get(
    NNGDArcFailureTaskList *list,
    const char *family,
    const char *task_id,
    const char *split) {
    size_t i;
    NNGDArcFailureTask *task;
    for (i = 0U; i < list->count; ++i) {
        task = &list->items[i];
        if (strcmp(task->family, family == NULL ? "" : family) == 0 &&
            strcmp(task->task_id, task_id == NULL ? "" : task_id) == 0 &&
            strcmp(task->split, split == NULL ? "" : split) == 0) {
            return task;
        }
    }
    if (nn_gd_arc_failure_task_list_reserve(list, list->count + 1U) != 0) {
        return NULL;
    }
    task = &list->items[list->count];
    memset(task, 0, sizeof(*task));
    task->family = nn_gd_strdup(family);
    task->task_id = nn_gd_strdup(task_id);
    task->split = nn_gd_strdup(split);
    if (task->family == NULL || task->task_id == NULL || task->split == NULL) {
        free(task->family);
        free(task->task_id);
        free(task->split);
        memset(task, 0, sizeof(*task));
        return NULL;
    }
    ++list->count;
    return task;
}

static void nn_gd_arc_failure_proposal_free(NNGDArcFailureProposal *proposal) {
    free(proposal->proposal_id);
    free(proposal->target_attempt_id);
    free(proposal->target_rejection_reason);
    free(proposal->failure_stage);
    free(proposal->failure_detail);
    free(proposal->pipeline_kind);
    free(proposal->op_name);
    memset(proposal, 0, sizeof(*proposal));
}

static void nn_gd_arc_failure_proposal_list_free(
    NNGDArcFailureProposalList *list) {
    size_t i;
    for (i = 0U; i < list->count; ++i) {
        nn_gd_arc_failure_proposal_free(&list->items[i]);
    }
    free(list->items);
    memset(list, 0, sizeof(*list));
}

static void nn_gd_arc_failure_task_free(NNGDArcFailureTask *task) {
    free(task->family);
    free(task->task_id);
    free(task->split);
    nn_gd_arc_hole_counter_list_free(&task->rejection_reasons);
    nn_gd_arc_hole_counter_list_free(&task->failure_stages);
    nn_gd_arc_hole_counter_list_free(&task->failure_details);
    nn_gd_arc_hole_counter_list_free(&task->accepted_ops);
    nn_gd_arc_hole_counter_list_free(&task->candidate_ops);
    nn_gd_arc_hole_counter_list_free(&task->candidate_pipelines);
    nn_gd_arc_failure_proposal_list_free(&task->proposals);
    memset(task, 0, sizeof(*task));
}

static void nn_gd_arc_failure_task_list_free(NNGDArcFailureTaskList *list) {
    size_t i;
    for (i = 0U; i < list->count; ++i) {
        nn_gd_arc_failure_task_free(&list->items[i]);
    }
    free(list->items);
    memset(list, 0, sizeof(*list));
}

static int nn_gd_arc_failure_proposal_list_reserve(
    NNGDArcFailureProposalList *list,
    size_t needed) {
    NNGDArcFailureProposal *items;
    size_t capacity;
    if (needed <= list->capacity) {
        return 0;
    }
    capacity = list->capacity == 0U ? 8U : list->capacity * 2U;
    while (capacity < needed) {
        capacity *= 2U;
    }
    items = (NNGDArcFailureProposal *)realloc(list->items,
                                              capacity * sizeof(*list->items));
    if (items == NULL) {
        return -1;
    }
    list->items = items;
    list->capacity = capacity;
    return 0;
}

static int nn_gd_json_index_bool_or(const NNGDJsonIndex *index,
                                    const char *key,
                                    int default_value) {
    const char *cursor = nn_gd_json_skip_space(nn_gd_json_index_find(index, key));
    if (cursor == NULL || strncmp(cursor, "null", 4U) == 0) {
        return default_value;
    }
    if (strncmp(cursor, "true", 4U) == 0 || strncmp(cursor, "1", 1U) == 0) {
        return 1;
    }
    if (strncmp(cursor, "false", 5U) == 0 || strncmp(cursor, "0", 1U) == 0) {
        return 0;
    }
    return default_value;
}

static int nn_gd_arc_failure_proposal_add(NNGDArcFailureTask *task,
                                          const NNGDJsonIndex *index) {
    NNGDArcFailureProposal *proposal;
    if (nn_gd_arc_failure_proposal_list_reserve(&task->proposals,
                                                task->proposals.count + 1U) != 0) {
        return -1;
    }
    proposal = &task->proposals.items[task->proposals.count];
    memset(proposal, 0, sizeof(*proposal));
    proposal->proposal_id = nn_gd_json_dup_index_string(index, "proposal_id");
    proposal->target_attempt_id =
        nn_gd_json_dup_index_string(index, "target_attempt_id");
    proposal->target_rejection_reason =
        nn_gd_json_dup_index_string(index, "target_rejection_reason");
    proposal->failure_stage = nn_gd_json_dup_index_string(index, "failure_stage");
    proposal->failure_detail = nn_gd_json_dup_index_string(index, "failure_detail");
    proposal->pipeline_kind = nn_gd_json_dup_index_string(index, "pipeline_kind");
    proposal->op_name = nn_gd_json_dup_index_string(index, "op_name");
    proposal->rank = nn_gd_json_index_long_or(index, "rank", 1000000L);
    proposal->score = nn_gd_json_index_double_or_zero(index, "score");
    proposal->target_accepted =
        nn_gd_json_index_bool_or(index, "target_accepted", 0);
    proposal->model_score = nn_gd_json_index_double_or_zero(index, "model_score");
    proposal->verifier_proxy_score =
        nn_gd_json_index_double_or_zero(index, "verifier_proxy_score");
    proposal->renderer_model_score =
        nn_gd_json_index_double_or_zero(index, "renderer_model_score");
    proposal->component_shape_color_score =
        nn_gd_json_index_double_or_zero(index, "component_shape_color_score");
    proposal->component_selector_layout_score =
        nn_gd_json_index_double_or_zero(index, "component_selector_layout_score");
    proposal->sequence_block_renderer_score =
        nn_gd_json_index_double_or_zero(index, "sequence_block_renderer_score");
    proposal->verifier_mismatch_ratio =
        nn_gd_json_index_double_or_zero(index, "verifier_mismatch_ratio");
    proposal->verifier_unsupported_pairs =
        nn_gd_json_index_long_or(index, "verifier_unsupported_pairs", 0L);
    proposal->verifier_apply_failed_pairs =
        nn_gd_json_index_long_or(index, "verifier_apply_failed_pairs", 0L);
    proposal->verifier_shape_mismatch_pairs =
        nn_gd_json_index_long_or(index, "verifier_shape_mismatch_pairs", 0L);
    if (proposal->proposal_id == NULL || proposal->target_attempt_id == NULL ||
        proposal->target_rejection_reason == NULL ||
        proposal->failure_stage == NULL || proposal->failure_detail == NULL ||
        proposal->pipeline_kind == NULL || proposal->op_name == NULL) {
        nn_gd_arc_failure_proposal_free(proposal);
        return -1;
    }
    ++task->proposals.count;
    return 0;
}

static int nn_gd_compare_arc_failure_proposals(const void *a, const void *b) {
    const NNGDArcFailureProposal *pa = (const NNGDArcFailureProposal *)a;
    const NNGDArcFailureProposal *pb = (const NNGDArcFailureProposal *)b;
    if (pa->rank < pb->rank) {
        return -1;
    }
    if (pa->rank > pb->rank) {
        return 1;
    }
    if (pa->score < pb->score) {
        return 1;
    }
    if (pa->score > pb->score) {
        return -1;
    }
    return strcmp(pa->proposal_id, pb->proposal_id);
}

static void nn_gd_arc_failure_sort(NNGDArcFailureTaskList *tasks) {
    size_t i;
    if (tasks->count > 1U) {
        qsort(tasks->items,
              tasks->count,
              sizeof(tasks->items[0]),
              nn_gd_arc_failure_task_compare);
    }
    for (i = 0U; i < tasks->count; ++i) {
        NNGDArcFailureProposalList *proposals = &tasks->items[i].proposals;
        if (proposals->count > 1U) {
            qsort(proposals->items,
                  proposals->count,
                  sizeof(proposals->items[0]),
                  nn_gd_compare_arc_failure_proposals);
        }
    }
}

static const char *nn_gd_arc_failure_counter_top(
    const NNGDArcHoleCounterList *list,
    const char *fallback,
    size_t *count_out) {
    size_t i;
    size_t best = SIZE_MAX;
    if (count_out != NULL) {
        *count_out = 0U;
    }
    for (i = 0U; i < list->count; ++i) {
        if (best == SIZE_MAX ||
            list->items[i].count > list->items[best].count ||
            (list->items[i].count == list->items[best].count &&
             strcmp(list->items[i].key, list->items[best].key) < 0)) {
            best = i;
        }
    }
    if (best == SIZE_MAX) {
        return fallback == NULL ? "" : fallback;
    }
    if (count_out != NULL) {
        *count_out = list->items[best].count;
    }
    return list->items[best].key;
}

static int nn_gd_arc_failure_load_attempts(const char *path,
                                           NNGDArcFailureTaskList *tasks) {
    FILE *fp = fopen(path, "r");
    char line[NN_GD_LINE_MAX];
    char *fields[NN_GD_MAX_FIELDS];
    if (fp == NULL) {
        return -1;
    }
    if (fgets(line, sizeof(line), fp) == NULL) {
        fclose(fp);
        return 0;
    }
    while (fgets(line, sizeof(line), fp) != NULL) {
        size_t field_count = nn_gd_split_tabs(line, fields, NN_GD_MAX_FIELDS);
        const char *split_name = nn_gd_field(fields, field_count, 0U);
        const char *family;
        const char *task_id;
        const char *split;
        const char *reason;
        const char *stage;
        const char *detail;
        const char *op_name;
        const char *pipeline_kind;
        NNGDArcFailureTask *task;
        int accepted;
        if (strcmp(split_name, "target") != 0) {
            continue;
        }
        family = nn_gd_field(fields, field_count, 1U);
        task_id = nn_gd_field(fields, field_count, 2U);
        split = nn_gd_field(fields, field_count, 3U);
        task = nn_gd_arc_failure_task_get(tasks, family, task_id, split);
        if (task == NULL) {
            fclose(fp);
            return -1;
        }
        accepted = nn_gd_parse_bool(nn_gd_field(fields, field_count, 16U));
        reason = nn_gd_field(fields, field_count, 17U);
        stage = nn_gd_field(fields, field_count, 18U);
        detail = nn_gd_field(fields, field_count, 19U);
        pipeline_kind = nn_gd_field(fields, field_count, 7U);
        op_name = nn_gd_field(fields, field_count, 8U);
        ++task->candidate_count;
        if (accepted) {
            ++task->accepted_candidate_count;
        }
        if (nn_gd_arc_hole_counter_add(&task->rejection_reasons,
                                       reason[0] == '\0' ? "unknown" : reason) != 0 ||
            nn_gd_arc_hole_counter_add(&task->failure_stages,
                                       stage[0] == '\0' ? "unknown" : stage) != 0 ||
            nn_gd_arc_hole_counter_add(&task->failure_details,
                                       detail[0] == '\0' ? "unknown" : detail) != 0 ||
            nn_gd_arc_hole_counter_add(&task->candidate_ops,
                                       op_name[0] == '\0' ? "unknown" : op_name) != 0 ||
            nn_gd_arc_hole_counter_add(&task->candidate_pipelines,
                                       pipeline_kind[0] == '\0'
                                           ? "unknown"
                                           : pipeline_kind) != 0) {
            fclose(fp);
            return -1;
        }
        if (accepted &&
            nn_gd_arc_hole_counter_add(&task->accepted_ops,
                                       op_name[0] == '\0' ? "unknown" : op_name) != 0) {
            fclose(fp);
            return -1;
        }
    }
    if (ferror(fp)) {
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
}

static int nn_gd_arc_failure_load_proposals(const char *path,
                                            NNGDArcFailureTaskList *tasks) {
    FILE *fp = fopen(path, "r");
    char *line;
    int status = 0;
    if (fp == NULL) {
        return -1;
    }
    while ((line = nn_gd_read_line_dynamic(fp)) != NULL) {
        NNGDJsonIndex index;
        char *schema;
        char *family;
        char *task_id;
        char *split;
        NNGDArcFailureTask *task;
        if (line[0] != '{') {
            free(line);
            continue;
        }
        nn_gd_json_index_line(line, &index);
        schema = nn_gd_json_dup_index_string(&index, "schema");
        if (schema == NULL) {
            free(line);
            status = -1;
            break;
        }
        if (strcmp(schema,
                   "crlp.neural_grammar_discovery.arc_visual_pipeline_proposal.v1") != 0) {
            free(schema);
            free(line);
            continue;
        }
        free(schema);
        family = nn_gd_json_dup_index_string(&index, "family");
        task_id = nn_gd_json_dup_index_string(&index, "task_id");
        split = nn_gd_json_dup_index_string(&index, "split");
        if (family == NULL || task_id == NULL || split == NULL) {
            free(family);
            free(task_id);
            free(split);
            free(line);
            status = -1;
            break;
        }
        task = nn_gd_arc_failure_task_get(tasks, family, task_id, split);
        free(family);
        free(task_id);
        free(split);
        if (task == NULL || nn_gd_arc_failure_proposal_add(task, &index) != 0) {
            free(line);
            status = -1;
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

static long nn_gd_arc_failure_first_accepted_rank(
    const NNGDArcFailureTask *task,
    long *accepted_retained_count,
    double *best_accepted_score,
    int *has_best_accepted_score) {
    size_t i;
    long best_rank = 0L;
    if (accepted_retained_count != NULL) {
        *accepted_retained_count = 0L;
    }
    if (has_best_accepted_score != NULL) {
        *has_best_accepted_score = 0;
    }
    if (best_accepted_score != NULL) {
        *best_accepted_score = 0.0;
    }
    for (i = 0U; i < task->proposals.count; ++i) {
        const NNGDArcFailureProposal *proposal = &task->proposals.items[i];
        if (!proposal->target_accepted) {
            continue;
        }
        if (accepted_retained_count != NULL) {
            ++*accepted_retained_count;
        }
        if (best_rank == 0L || proposal->rank < best_rank) {
            best_rank = proposal->rank;
        }
        if (has_best_accepted_score != NULL && best_accepted_score != NULL &&
            (!*has_best_accepted_score || proposal->score > *best_accepted_score)) {
            *best_accepted_score = proposal->score;
            *has_best_accepted_score = 1;
        }
    }
    return best_rank;
}

static void nn_gd_arc_failure_classify(const NNGDArcFailureTask *task,
                                       long first_accepted_rank,
                                       const char **bucket,
                                       const char **diagnosis,
                                       const char **next_step_hint) {
    if (task->proposals.count == 0U) {
        *bucket = "no_retained_proposals";
        *diagnosis = "retention";
        *next_step_hint = "ranker emitted no retained proposals for this target task";
        return;
    }
    if (task->accepted_candidate_count == 0L) {
        *bucket = "candidate_generation_gap";
        *diagnosis = "candidate_generation";
        *next_step_hint = "no accepted candidate exists in the target attempt search space";
        return;
    }
    if (first_accepted_rank == 0L) {
        *bucket = "retention_gap";
        *diagnosis = "retention";
        *next_step_hint =
            "accepted candidates exist, but none survived the retained proposal cap";
        return;
    }
    if (first_accepted_rank == 1L) {
        *bucket = "solved_top1";
        *diagnosis = "solved";
        *next_step_hint = "accepted candidate is ranked first among retained proposals";
        return;
    }
    *bucket = "ranking_gap";
    *diagnosis = "visual_feature_ranking";
    *next_step_hint = "accepted candidate was retained but ranked below a rejected proposal";
}

static void nn_gd_arc_failure_write_nullable_long(FILE *fp, int has_value, long value) {
    if (!has_value) {
        fputs("null", fp);
    } else {
        fprintf(fp, "%ld", value);
    }
}

static void nn_gd_arc_failure_write_nullable_double(FILE *fp,
                                                    int has_value,
                                                    double value) {
    if (!has_value) {
        fputs("null", fp);
    } else {
        fprintf(fp, "%.8f", value);
    }
}

static int nn_gd_arc_failure_write_task_row(FILE *fp,
                                            NNGDArcFailureTask *task) {
    const NNGDArcFailureProposal *top =
        task->proposals.count == 0U ? NULL : &task->proposals.items[0];
    const NNGDArcFailureProposal *second =
        task->proposals.count < 2U ? NULL : &task->proposals.items[1];
    NNGDArcHoleCounterList retained_reasons = {0};
    NNGDArcHoleCounterList retained_stages = {0};
    NNGDArcHoleCounterList retained_details = {0};
    long retained_accepted_count = 0L;
    long first_accepted_rank;
    double best_accepted_score = 0.0;
    int has_best_accepted_score = 0;
    const char *bucket;
    const char *diagnosis;
    const char *next_step_hint;
    const char *candidate_reason;
    const char *candidate_stage;
    const char *candidate_detail;
    const char *retained_reason;
    const char *retained_stage;
    const char *retained_detail;
    const char *candidate_op;
    const char *candidate_pipeline;
    size_t candidate_reason_count = 0U;
    size_t candidate_stage_count = 0U;
    size_t candidate_detail_count = 0U;
    size_t retained_reason_count = 0U;
    size_t retained_stage_count = 0U;
    size_t retained_detail_count = 0U;
    size_t candidate_op_count = 0U;
    size_t candidate_pipeline_count = 0U;
    size_t i;
    int status = -1;
    for (i = 0U; i < task->proposals.count; ++i) {
        const NNGDArcFailureProposal *proposal = &task->proposals.items[i];
        if (nn_gd_arc_hole_counter_add(&retained_reasons,
                                       proposal->target_rejection_reason) != 0 ||
            nn_gd_arc_hole_counter_add(&retained_stages,
                                       proposal->failure_stage) != 0 ||
            nn_gd_arc_hole_counter_add(&retained_details,
                                       proposal->failure_detail) != 0) {
            goto done;
        }
    }
    first_accepted_rank = nn_gd_arc_failure_first_accepted_rank(
        task,
        &retained_accepted_count,
        &best_accepted_score,
        &has_best_accepted_score);
    nn_gd_arc_failure_classify(task,
                               first_accepted_rank,
                               &bucket,
                               &diagnosis,
                               &next_step_hint);
    candidate_reason = nn_gd_arc_failure_counter_top(&task->rejection_reasons,
                                                     "",
                                                     &candidate_reason_count);
    candidate_stage = nn_gd_arc_failure_counter_top(&task->failure_stages,
                                                    "",
                                                    &candidate_stage_count);
    candidate_detail = nn_gd_arc_failure_counter_top(&task->failure_details,
                                                     "",
                                                     &candidate_detail_count);
    retained_reason = nn_gd_arc_failure_counter_top(&retained_reasons,
                                                    "",
                                                    &retained_reason_count);
    retained_stage = nn_gd_arc_failure_counter_top(&retained_stages,
                                                   "",
                                                   &retained_stage_count);
    retained_detail = nn_gd_arc_failure_counter_top(&retained_details,
                                                    "",
                                                    &retained_detail_count);
    candidate_op = nn_gd_arc_failure_counter_top(&task->candidate_ops,
                                                 "",
                                                 &candidate_op_count);
    candidate_pipeline = nn_gd_arc_failure_counter_top(&task->candidate_pipelines,
                                                       "",
                                                       &candidate_pipeline_count);
    fputs("{\"schema\":\"crlp.arc_ranker_failure_task.v1\","
          "\"report_version\":1,\"family\":",
          fp);
    nn_gd_json_write_string(fp, task->family);
    fputs(",\"task_id\":", fp);
    nn_gd_json_write_string(fp, task->task_id);
    fputs(",\"split\":", fp);
    nn_gd_json_write_string(fp, task->split);
    fputs(",\"failure_bucket\":", fp);
    nn_gd_json_write_string(fp, bucket);
    fputs(",\"ranker_diagnosis\":", fp);
    nn_gd_json_write_string(fp, diagnosis);
    fputs(",\"next_step_hint\":", fp);
    nn_gd_json_write_string(fp, next_step_hint);
    fprintf(fp,
            ",\"candidate_count\":%ld,\"accepted_candidate_count\":%ld,"
            "\"rejected_candidate_count\":%ld,\"retained_count\":%zu,"
            "\"retained_accepted_count\":%ld,\"first_accepted_rank\":",
            task->candidate_count,
            task->accepted_candidate_count,
            task->candidate_count - task->accepted_candidate_count,
            task->proposals.count,
            retained_accepted_count);
    nn_gd_arc_failure_write_nullable_long(fp, first_accepted_rank != 0L,
                                          first_accepted_rank);
    fputs(",\"top_score\":", fp);
    nn_gd_arc_failure_write_nullable_double(fp, top != NULL, top == NULL ? 0.0 : top->score);
    fputs(",\"second_score\":", fp);
    nn_gd_arc_failure_write_nullable_double(fp,
                                            second != NULL,
                                            second == NULL ? 0.0 : second->score);
    fputs(",\"top_score_margin\":", fp);
    nn_gd_arc_failure_write_nullable_double(fp,
                                            top != NULL && second != NULL,
                                            top == NULL || second == NULL
                                                ? 0.0
                                                : top->score - second->score);
    fputs(",\"best_accepted_score\":", fp);
    nn_gd_arc_failure_write_nullable_double(fp,
                                            has_best_accepted_score,
                                            best_accepted_score);
    fputs(",\"top_vs_best_accepted_score_delta\":", fp);
    nn_gd_arc_failure_write_nullable_double(fp,
                                            top != NULL && has_best_accepted_score,
                                            top == NULL
                                                ? 0.0
                                                : top->score - best_accepted_score);
    fputs(",\"top_proposal_id\":", fp);
    nn_gd_json_write_string(fp, top == NULL ? "" : top->proposal_id);
    fputs(",\"top_attempt_id\":", fp);
    nn_gd_json_write_string(fp, top == NULL ? "" : top->target_attempt_id);
    fputs(",\"top_rank\":", fp);
    nn_gd_arc_failure_write_nullable_long(fp, top != NULL, top == NULL ? 0L : top->rank);
    fprintf(fp, ",\"top_accepted\":%s,\"top_rejection_reason\":",
            top != NULL && top->target_accepted ? "true" : "false");
    nn_gd_json_write_string(fp, top == NULL ? "" : top->target_rejection_reason);
    fputs(",\"top_failure_stage\":", fp);
    nn_gd_json_write_string(fp, top == NULL ? "" : top->failure_stage);
    fputs(",\"top_failure_detail\":", fp);
    nn_gd_json_write_string(fp, top == NULL ? "" : top->failure_detail);
    fputs(",\"top_pipeline_kind\":", fp);
    nn_gd_json_write_string(fp, top == NULL ? "" : top->pipeline_kind);
    fputs(",\"top_op_name\":", fp);
    nn_gd_json_write_string(fp, top == NULL ? "" : top->op_name);
    fputs(",\"top_model_score\":", fp);
    nn_gd_arc_failure_write_nullable_double(fp,
                                            top != NULL,
                                            top == NULL ? 0.0 : top->model_score);
    fputs(",\"top_verifier_proxy_score\":", fp);
    nn_gd_arc_failure_write_nullable_double(
        fp,
        top != NULL,
        top == NULL ? 0.0 : top->verifier_proxy_score);
    fputs(",\"top_renderer_model_score\":", fp);
    nn_gd_arc_failure_write_nullable_double(
        fp,
        top != NULL,
        top == NULL ? 0.0 : top->renderer_model_score);
    fputs(",\"top_component_shape_color_score\":", fp);
    nn_gd_arc_failure_write_nullable_double(
        fp,
        top != NULL,
        top == NULL ? 0.0 : top->component_shape_color_score);
    fputs(",\"top_component_selector_layout_score\":", fp);
    nn_gd_arc_failure_write_nullable_double(
        fp,
        top != NULL,
        top == NULL ? 0.0 : top->component_selector_layout_score);
    fputs(",\"top_sequence_block_renderer_score\":", fp);
    nn_gd_arc_failure_write_nullable_double(
        fp,
        top != NULL,
        top == NULL ? 0.0 : top->sequence_block_renderer_score);
    fputs(",\"top_verifier_mismatch_ratio\":", fp);
    nn_gd_arc_failure_write_nullable_double(
        fp,
        top != NULL,
        top == NULL ? 0.0 : top->verifier_mismatch_ratio);
    fputs(",\"top_verifier_unsupported_pairs\":", fp);
    nn_gd_arc_failure_write_nullable_long(
        fp,
        top != NULL,
        top == NULL ? 0L : top->verifier_unsupported_pairs);
    fputs(",\"top_verifier_apply_failed_pairs\":", fp);
    nn_gd_arc_failure_write_nullable_long(
        fp,
        top != NULL,
        top == NULL ? 0L : top->verifier_apply_failed_pairs);
    fputs(",\"top_verifier_shape_mismatch_pairs\":", fp);
    nn_gd_arc_failure_write_nullable_long(
        fp,
        top != NULL,
        top == NULL ? 0L : top->verifier_shape_mismatch_pairs);
    fputs(",\"dominant_candidate_rejection_reason\":", fp);
    nn_gd_json_write_string(fp, candidate_reason);
    fprintf(fp,
            ",\"dominant_candidate_rejection_reason_count\":%zu,"
            "\"dominant_candidate_failure_stage\":",
            candidate_reason_count);
    nn_gd_json_write_string(fp, candidate_stage);
    fprintf(fp,
            ",\"dominant_candidate_failure_stage_count\":%zu,"
            "\"dominant_candidate_failure_detail\":",
            candidate_stage_count);
    nn_gd_json_write_string(fp, candidate_detail);
    fprintf(fp,
            ",\"dominant_candidate_failure_detail_count\":%zu,"
            "\"dominant_retained_rejection_reason\":",
            candidate_detail_count);
    nn_gd_json_write_string(fp, retained_reason);
    fprintf(fp,
            ",\"dominant_retained_rejection_reason_count\":%zu,"
            "\"dominant_retained_failure_stage\":",
            retained_reason_count);
    nn_gd_json_write_string(fp, retained_stage);
    fprintf(fp,
            ",\"dominant_retained_failure_stage_count\":%zu,"
            "\"dominant_retained_failure_detail\":",
            retained_stage_count);
    nn_gd_json_write_string(fp, retained_detail);
    fprintf(fp,
            ",\"dominant_retained_failure_detail_count\":%zu,"
            "\"dominant_candidate_op\":",
            retained_detail_count);
    nn_gd_json_write_string(fp, candidate_op);
    fprintf(fp,
            ",\"dominant_candidate_op_count\":%zu,"
            "\"dominant_candidate_pipeline\":",
            candidate_op_count);
    nn_gd_json_write_string(fp, candidate_pipeline);
    fprintf(fp, ",\"dominant_candidate_pipeline_count\":%zu,\"accepted_ops\":",
            candidate_pipeline_count);
    if (nn_gd_arc_hole_write_counter_object(fp, &task->accepted_ops) != 0) {
        goto done;
    }
    fputs("}\n", fp);
    status = ferror(fp) ? -1 : 0;

done:
    nn_gd_arc_hole_counter_list_free(&retained_reasons);
    nn_gd_arc_hole_counter_list_free(&retained_stages);
    nn_gd_arc_hole_counter_list_free(&retained_details);
    return status;
}

static int nn_gd_compare_arc_failure_counter_desc(const void *a, const void *b) {
    const NNGDArcHoleCounter *ca = (const NNGDArcHoleCounter *)a;
    const NNGDArcHoleCounter *cb = (const NNGDArcHoleCounter *)b;
    if (ca->count < cb->count) {
        return 1;
    }
    if (ca->count > cb->count) {
        return -1;
    }
    return strcmp(ca->key, cb->key);
}

static void nn_gd_arc_failure_write_counter_table(FILE *fp,
                                                  NNGDArcHoleCounterList *counter,
                                                  const char *left_header) {
    size_t i;
    if (counter->count > 1U) {
        qsort(counter->items,
              counter->count,
              sizeof(counter->items[0]),
              nn_gd_compare_arc_failure_counter_desc);
    }
    fprintf(fp, "| %s | Tasks |\n|---|---:|\n", left_header);
    for (i = 0U; i < counter->count; ++i) {
        fprintf(fp, "| %s | %zu |\n", counter->items[i].key, counter->items[i].count);
    }
}

static const NNGDArcFailureTask *nn_gd_sort_arc_failure_tasks = NULL;

static int nn_gd_compare_arc_failure_missed_index(const void *a, const void *b) {
    size_t ia = *(const size_t *)a;
    size_t ib = *(const size_t *)b;
    const NNGDArcFailureTask *ta = &nn_gd_sort_arc_failure_tasks[ia];
    const NNGDArcFailureTask *tb = &nn_gd_sort_arc_failure_tasks[ib];
    const char *bucket_a;
    const char *diagnosis_a;
    const char *hint_a;
    const char *bucket_b;
    const char *diagnosis_b;
    const char *hint_b;
    long first_a = nn_gd_arc_failure_first_accepted_rank(ta, NULL, NULL, NULL);
    long first_b = nn_gd_arc_failure_first_accepted_rank(tb, NULL, NULL, NULL);
    int cmp;
    nn_gd_arc_failure_classify(ta, first_a, &bucket_a, &diagnosis_a, &hint_a);
    nn_gd_arc_failure_classify(tb, first_b, &bucket_b, &diagnosis_b, &hint_b);
    cmp = strcmp(bucket_a, bucket_b);
    (void)diagnosis_a;
    (void)hint_a;
    (void)diagnosis_b;
    (void)hint_b;
    if (cmp != 0) {
        return cmp;
    }
    if (ta->candidate_count < tb->candidate_count) {
        return 1;
    }
    if (ta->candidate_count > tb->candidate_count) {
        return -1;
    }
    return strcmp(ta->task_id, tb->task_id);
}

static void nn_gd_markdown_cell(FILE *fp, const char *value) {
    const char *p;
    if (value == NULL) {
        return;
    }
    for (p = value; *p != '\0'; ++p) {
        fputc(*p == '|' ? '/' : *p, fp);
    }
}

static int nn_gd_arc_failure_write_markdown(
    const char *path,
    NNGDArcFailureTaskList *tasks,
    const char *summary_proposal_count,
    const char *summary_top1_all_task_rate) {
    FILE *fp;
    NNGDArcHoleCounterList bucket_counts = {0};
    NNGDArcHoleCounterList diagnosis_counts = {0};
    NNGDArcHoleCounterList missed_reason_counts = {0};
    size_t *missed_indices = NULL;
    size_t missed_count = 0U;
    size_t solved_count = 0U;
    size_t i;
    int status = -1;
    if (nn_gd_mkdir_parent(path) != 0) {
        return -1;
    }
    fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    missed_indices = (size_t *)calloc(tasks->count == 0U ? 1U : tasks->count,
                                      sizeof(size_t));
    if (missed_indices == NULL) {
        goto done;
    }
    for (i = 0U; i < tasks->count; ++i) {
        NNGDArcFailureTask *task = &tasks->items[i];
        const char *bucket;
        const char *diagnosis;
        const char *hint;
        const char *candidate_reason;
        long first = nn_gd_arc_failure_first_accepted_rank(task, NULL, NULL, NULL);
        nn_gd_arc_failure_classify(task, first, &bucket, &diagnosis, &hint);
        (void)hint;
        if (nn_gd_arc_hole_counter_add(&bucket_counts, bucket) != 0 ||
            nn_gd_arc_hole_counter_add(&diagnosis_counts, diagnosis) != 0) {
            goto done;
        }
        if (strcmp(bucket, "solved_top1") != 0) {
            candidate_reason =
                nn_gd_arc_failure_counter_top(&task->rejection_reasons, "", NULL);
            if (nn_gd_arc_hole_counter_add(&missed_reason_counts,
                                           candidate_reason[0] == '\0'
                                               ? "unknown"
                                               : candidate_reason) != 0) {
                goto done;
            }
            missed_indices[missed_count++] = i;
        } else {
            ++solved_count;
        }
    }
    fprintf(fp,
            "# ARC Ranker Failure Report\n\n"
            "Target tasks: `%zu`\n"
            "Solved top-1: `%zu`\n"
            "Missed: `%zu`\n",
            tasks->count,
            solved_count,
            missed_count);
    if (summary_proposal_count != NULL && summary_proposal_count[0] != '\0') {
        fprintf(fp, "Ranker proposals: `%s`\n", summary_proposal_count);
    }
    if (summary_top1_all_task_rate != NULL &&
        summary_top1_all_task_rate[0] != '\0') {
        fprintf(fp, "Top-1 all-task rate: `%s`\n", summary_top1_all_task_rate);
    }
    fputs("\n## Buckets\n\n", fp);
    nn_gd_arc_failure_write_counter_table(fp, &bucket_counts, "Bucket");
    fputs("\n## Diagnosis\n\n", fp);
    nn_gd_arc_failure_write_counter_table(fp, &diagnosis_counts, "Diagnosis");
    fputs("\n## Dominant Missed Candidate Rejection Reasons\n\n", fp);
    nn_gd_arc_failure_write_counter_table(fp, &missed_reason_counts, "Reason");
    fputs("\n## Missed Tasks\n\n"
          "| Task | Bucket | Candidates | Accepted candidates | Retained | "
          "First accepted rank | Top margin | Top reason | Candidate reason | Top op |\n"
          "|---|---|---:|---:|---:|---:|---:|---|---|---|\n",
          fp);
    if (missed_count > 1U) {
        nn_gd_sort_arc_failure_tasks = tasks->items;
        qsort(missed_indices,
              missed_count,
              sizeof(missed_indices[0]),
              nn_gd_compare_arc_failure_missed_index);
        nn_gd_sort_arc_failure_tasks = NULL;
    }
    for (i = 0U; i < missed_count; ++i) {
        NNGDArcFailureTask *task = &tasks->items[missed_indices[i]];
        const NNGDArcFailureProposal *top =
            task->proposals.count == 0U ? NULL : &task->proposals.items[0];
        const NNGDArcFailureProposal *second =
            task->proposals.count < 2U ? NULL : &task->proposals.items[1];
        const char *bucket;
        const char *diagnosis;
        const char *hint;
        const char *candidate_reason =
            nn_gd_arc_failure_counter_top(&task->rejection_reasons, "", NULL);
        long first = nn_gd_arc_failure_first_accepted_rank(task, NULL, NULL, NULL);
        nn_gd_arc_failure_classify(task, first, &bucket, &diagnosis, &hint);
        (void)diagnosis;
        (void)hint;
        fputs("| ", fp);
        nn_gd_markdown_cell(fp, task->task_id);
        fputs(" | ", fp);
        nn_gd_markdown_cell(fp, bucket);
        fprintf(fp,
                " | %ld | %ld | %zu | ",
                task->candidate_count,
                task->accepted_candidate_count,
                task->proposals.count);
        if (first == 0L) {
            fputs(" | ", fp);
        } else {
            fprintf(fp, "%ld | ", first);
        }
        if (top != NULL && second != NULL) {
            fprintf(fp, "%.4f", top->score - second->score);
        }
        fputs(" | ", fp);
        nn_gd_markdown_cell(fp, top == NULL ? "" : top->target_rejection_reason);
        fputs(" | ", fp);
        nn_gd_markdown_cell(fp, candidate_reason);
        fputs(" | ", fp);
        nn_gd_markdown_cell(fp, top == NULL ? "" : top->op_name);
        fputs(" |\n", fp);
    }
    status = ferror(fp) ? -1 : 0;

done:
    free(missed_indices);
    nn_gd_arc_hole_counter_list_free(&bucket_counts);
    nn_gd_arc_hole_counter_list_free(&diagnosis_counts);
    nn_gd_arc_hole_counter_list_free(&missed_reason_counts);
    if (fclose(fp) != 0) {
        status = -1;
    }
    return status;
}

static int nn_gd_arc_failure_write_jsonl(const char *path,
                                         NNGDArcFailureTaskList *tasks) {
    FILE *fp;
    size_t i;
    int status = -1;
    if (nn_gd_mkdir_parent(path) != 0) {
        return -1;
    }
    fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    for (i = 0U; i < tasks->count; ++i) {
        if (nn_gd_arc_failure_write_task_row(fp, &tasks->items[i]) != 0) {
            goto done;
        }
    }
    status = ferror(fp) ? -1 : 0;

done:
    if (fclose(fp) != 0) {
        status = -1;
    }
    return status;
}

int nn_grammar_discovery_write_arc_ranker_failure_report(
    const char *attempt_rows_path,
    const char *proposals_path,
    const char *summary_path,
    const char *output_path,
    const char *markdown_path) {
    NNGDArcFailureTaskList tasks = {0};
    char *expected_tasks_value = NULL;
    char *proposal_count_value = NULL;
    char *top1_all_task_rate_value = NULL;
    long expected_tasks;
    int status = -1;
    size_t i;
    NNGDArcHoleCounterList bucket_counts = {0};
    if (attempt_rows_path == NULL || proposals_path == NULL ||
        output_path == NULL || markdown_path == NULL ||
        nn_gd_arc_failure_load_attempts(attempt_rows_path, &tasks) != 0 ||
        nn_gd_arc_failure_load_proposals(proposals_path, &tasks) != 0) {
        goto done;
    }
    nn_gd_arc_failure_sort(&tasks);
    expected_tasks_value =
        summary_path == NULL ? nn_gd_strdup("") :
                               nn_gd_json_dup_first_row_scalar(summary_path,
                                                               "target_task_count");
    proposal_count_value =
        summary_path == NULL ? nn_gd_strdup("") :
                               nn_gd_json_dup_first_row_scalar(summary_path,
                                                               "proposal_count");
    top1_all_task_rate_value =
        summary_path == NULL ? nn_gd_strdup("") :
                               nn_gd_json_dup_first_row_scalar(summary_path,
                                                               "top1_all_task_rate");
    if (expected_tasks_value == NULL || proposal_count_value == NULL ||
        top1_all_task_rate_value == NULL) {
        goto done;
    }
    expected_tasks = nn_gd_parse_long(expected_tasks_value);
    if (expected_tasks != 0L && expected_tasks != (long)tasks.count) {
        goto done;
    }
    if (nn_gd_arc_failure_write_jsonl(output_path, &tasks) != 0 ||
        nn_gd_arc_failure_write_markdown(markdown_path,
                                         &tasks,
                                         proposal_count_value,
                                         top1_all_task_rate_value) != 0) {
        goto done;
    }
    for (i = 0U; i < tasks.count; ++i) {
        const char *bucket;
        const char *diagnosis;
        const char *hint;
        long first = nn_gd_arc_failure_first_accepted_rank(&tasks.items[i],
                                                           NULL,
                                                           NULL,
                                                           NULL);
        nn_gd_arc_failure_classify(&tasks.items[i],
                                   first,
                                   &bucket,
                                   &diagnosis,
                                   &hint);
        (void)diagnosis;
        (void)hint;
        if (nn_gd_arc_hole_counter_add(&bucket_counts, bucket) != 0) {
            goto done;
        }
    }
    {
        size_t solved = 0U;
        size_t generation = 0U;
        size_t retention = 0U;
        size_t ranking = 0U;
        for (i = 0U; i < bucket_counts.count; ++i) {
            if (strcmp(bucket_counts.items[i].key, "solved_top1") == 0) {
                solved = bucket_counts.items[i].count;
            } else if (strcmp(bucket_counts.items[i].key,
                              "candidate_generation_gap") == 0) {
                generation = bucket_counts.items[i].count;
            } else if (strcmp(bucket_counts.items[i].key, "retention_gap") == 0) {
                retention = bucket_counts.items[i].count;
            } else if (strcmp(bucket_counts.items[i].key, "ranking_gap") == 0) {
                ranking = bucket_counts.items[i].count;
            }
        }
        printf("arc-ranker-failure-report tasks=%zu solved_top1=%zu "
               "candidate_generation_gap=%zu retention_gap=%zu ranking_gap=%zu "
               "output=%s markdown=%s\n",
               tasks.count,
               solved,
               generation,
               retention,
               ranking,
               output_path,
               markdown_path);
    }
    status = 0;

done:
    free(expected_tasks_value);
    free(proposal_count_value);
    free(top1_all_task_rate_value);
    nn_gd_arc_hole_counter_list_free(&bucket_counts);
    nn_gd_arc_failure_task_list_free(&tasks);
    return status;
}
