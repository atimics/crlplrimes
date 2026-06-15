typedef struct {
    char *key;
    size_t count;
} NNGDHealthCounter;

typedef struct {
    NNGDHealthCounter *items;
    size_t count;
    size_t capacity;
} NNGDHealthCounterList;

typedef struct {
    char *name;
    NNGDStringList numeric_targets;
    NNGDStringList passing_gates;
    NNGDStringList failing_required_gates;
    NNGDStringList failing_optional_gates;
    NNGDStringList coverage_sources;
    NNGDStringList evidence_sources;
    NNGDStringList diagnostic_sources;
    NNGDStringList claim_ids;
    NNGDStringList required_claim_ids;
    NNGDHealthCounterList numeric_status_counts;
    NNGDHealthCounterList claim_status_counts;
    NNGDHealthCounterList triage_kind_counts;
    NNGDHealthCounterList triage_priority_counts;
    NNGDHealthCounterList coverage_hole_counts;
    NNGDHealthCounterList diagnostic_failure_class_counts;
    NNGDHealthCounterList diagnostic_next_step_counts;
    size_t numeric_certificate_rows;
    size_t numeric_committed_rows;
    size_t numeric_rejected_rows;
    size_t numeric_failed_statuses;
    size_t verifier_certificate_rows;
    size_t coverage_rows;
    size_t evidence_rows;
    size_t diagnostic_rows;
    size_t engine_evidence_rows;
    size_t protocol_candidate_rows;
    size_t protocol_outcome_rows;
    size_t operator_rows;
    size_t promoted_operator_rows;
} NNGDHealthExperiment;

typedef struct {
    NNGDHealthExperiment *items;
    size_t count;
    size_t capacity;
} NNGDHealthExperimentList;

typedef struct {
    const char *claim_id;
    const char *owners[3];
    size_t owner_count;
} NNGDHealthClaimOwner;

typedef struct {
    const NNGDHealthExperiment *experiment;
    char *failure_class;
    char *next_step;
    size_t failure_class_count;
    size_t next_step_count;
    int priority_score;
    const char *priority;
} NNGDHealthWorkItem;

typedef struct {
    NNGDHealthWorkItem *items;
    size_t count;
    size_t capacity;
} NNGDHealthWorkItemList;

static int nn_gd_health_counter_reserve(NNGDHealthCounterList *list,
                                        size_t needed) {
    if (needed <= list->capacity) {
        return 0;
    }
    size_t next = list->capacity == 0U ? 8U : list->capacity * 2U;
    while (next < needed) {
        next *= 2U;
    }
    NNGDHealthCounter *items =
        (NNGDHealthCounter *)realloc(list->items, next * sizeof(*items));
    if (items == NULL) {
        return -1;
    }
    list->items = items;
    list->capacity = next;
    return 0;
}

static int nn_gd_health_counter_add(NNGDHealthCounterList *list,
                                    const char *key,
                                    size_t amount) {
    size_t i;
    if (key == NULL || key[0] == '\0' || amount == 0U) {
        return 0;
    }
    for (i = 0U; i < list->count; ++i) {
        if (strcmp(list->items[i].key, key) == 0) {
            list->items[i].count += amount;
            return 0;
        }
    }
    if (nn_gd_health_counter_reserve(list, list->count + 1U) != 0) {
        return -1;
    }
    list->items[list->count].key = nn_gd_strdup(key);
    if (list->items[list->count].key == NULL) {
        return -1;
    }
    list->items[list->count].count = amount;
    ++list->count;
    return 0;
}

static size_t nn_gd_health_counter_get(const NNGDHealthCounterList *list,
                                       const char *key) {
    size_t i;
    if (list == NULL || key == NULL) {
        return 0U;
    }
    for (i = 0U; i < list->count; ++i) {
        if (strcmp(list->items[i].key, key) == 0) {
            return list->items[i].count;
        }
    }
    return 0U;
}

static void nn_gd_health_counter_free(NNGDHealthCounterList *list) {
    size_t i;
    if (list == NULL) {
        return;
    }
    for (i = 0U; i < list->count; ++i) {
        free(list->items[i].key);
    }
    free(list->items);
    memset(list, 0, sizeof(*list));
}

static int nn_gd_health_experiments_reserve(NNGDHealthExperimentList *list,
                                            size_t needed) {
    if (needed <= list->capacity) {
        return 0;
    }
    size_t next = list->capacity == 0U ? 16U : list->capacity * 2U;
    while (next < needed) {
        next *= 2U;
    }
    NNGDHealthExperiment *items =
        (NNGDHealthExperiment *)realloc(list->items, next * sizeof(*items));
    if (items == NULL) {
        return -1;
    }
    list->items = items;
    list->capacity = next;
    return 0;
}

static NNGDHealthExperiment *nn_gd_health_experiment(
    NNGDHealthExperimentList *list,
    const char *name) {
    size_t i;
    if (list == NULL || name == NULL || name[0] == '\0') {
        return NULL;
    }
    for (i = 0U; i < list->count; ++i) {
        if (strcmp(list->items[i].name, name) == 0) {
            return &list->items[i];
        }
    }
    if (nn_gd_health_experiments_reserve(list, list->count + 1U) != 0) {
        return NULL;
    }
    memset(&list->items[list->count], 0, sizeof(list->items[list->count]));
    list->items[list->count].name = nn_gd_strdup(name);
    if (list->items[list->count].name == NULL) {
        return NULL;
    }
    return &list->items[list->count++];
}

static void nn_gd_health_experiment_free(NNGDHealthExperiment *experiment) {
    if (experiment == NULL) {
        return;
    }
    free(experiment->name);
    nn_gd_string_list_free(&experiment->numeric_targets);
    nn_gd_string_list_free(&experiment->passing_gates);
    nn_gd_string_list_free(&experiment->failing_required_gates);
    nn_gd_string_list_free(&experiment->failing_optional_gates);
    nn_gd_string_list_free(&experiment->coverage_sources);
    nn_gd_string_list_free(&experiment->evidence_sources);
    nn_gd_string_list_free(&experiment->diagnostic_sources);
    nn_gd_string_list_free(&experiment->claim_ids);
    nn_gd_string_list_free(&experiment->required_claim_ids);
    nn_gd_health_counter_free(&experiment->numeric_status_counts);
    nn_gd_health_counter_free(&experiment->claim_status_counts);
    nn_gd_health_counter_free(&experiment->triage_kind_counts);
    nn_gd_health_counter_free(&experiment->triage_priority_counts);
    nn_gd_health_counter_free(&experiment->coverage_hole_counts);
    nn_gd_health_counter_free(&experiment->diagnostic_failure_class_counts);
    nn_gd_health_counter_free(&experiment->diagnostic_next_step_counts);
}

static void nn_gd_health_experiment_list_free(NNGDHealthExperimentList *list) {
    size_t i;
    if (list == NULL) {
        return;
    }
    for (i = 0U; i < list->count; ++i) {
        nn_gd_health_experiment_free(&list->items[i]);
    }
    free(list->items);
    memset(list, 0, sizeof(*list));
}

static char *nn_gd_health_experiment_from_target(const char *target) {
    static const struct {
        const char *target;
        const char *experiment;
    } map[] = {
        {"action-score", "action_score"},
        {"prime-gap-trace", "prime_gap"},
        {"prime-rule-search-smoke", "prime_gap"},
        {"delayed-claim-eval", "delayed_claim"},
        {"grid-path-eval", "grid_path"},
        {"multitask-domain", "multitask_domain"},
        {"symbolic-domains", "symbolic_domains"},
        {"program-synthesis", "program_synthesis"},
        {"program-synthesis-baselines", "program_synthesis"},
        {"modular-gate-search-smoke", "modular_gate_search"},
        {"arc-agi-induction-smoke", "arc_agi"},
        {"haiku-lm-smoke", "haiku_lm"},
        {"haiku-word-lm-smoke", "haiku_word_lm"},
        {"haiku-understanding-smoke", "haiku_understanding"},
        {"haiku-coherence-hard-negatives", "haiku_coherence_hard_negatives"},
        {"signal-flight", "signal_flight"},
        {"signal-mining-smoke", "signal_mining"},
    };
    size_t i;
    char *value;
    if (target == NULL || target[0] == '\0') {
        return nn_gd_strdup("unknown");
    }
    for (i = 0U; i < sizeof(map) / sizeof(map[0]); ++i) {
        if (strcmp(target, map[i].target) == 0) {
            return nn_gd_strdup(map[i].experiment);
        }
    }
    value = nn_gd_strdup(target);
    if (value != NULL) {
        char *p;
        for (p = value; *p != '\0'; ++p) {
            if (*p == '-') {
                *p = '_';
            }
        }
    }
    return value;
}

static int nn_gd_health_add_scoped_gate(NNGDStringList *list,
                                        const char *scope,
                                        const char *gate) {
    char *value;
    int written;
    size_t needed;
    if (scope == NULL || scope[0] == '\0' || strcmp(scope, "unknown") == 0) {
        return nn_gd_string_list_add_unique(list, gate);
    }
    needed = strlen(scope) + strlen(gate) + 2U;
    value = (char *)malloc(needed);
    if (value == NULL) {
        return -1;
    }
    written = snprintf(value, needed, "%s:%s", scope, gate);
    if (written < 0 || (size_t)written >= needed) {
        free(value);
        return -1;
    }
    int status = nn_gd_string_list_add_unique(list, value);
    free(value);
    return status;
}

static int nn_gd_health_add_triage(NNGDHealthExperiment *experiment,
                                   const char *priority,
                                   const char *kind) {
    if (nn_gd_health_counter_add(&experiment->triage_priority_counts,
                                 priority,
                                 1U) != 0 ||
        nn_gd_health_counter_add(&experiment->triage_kind_counts,
                                 kind,
                                 1U) != 0) {
        return -1;
    }
    return 0;
}

static const char *nn_gd_health_optional_kind(const char *experiment,
                                             const char *scope,
                                             const char *gate,
                                             const char **priority_out) {
    if (priority_out != NULL) {
        *priority_out = "P2";
    }
    if (strcmp(experiment, "arc_agi") == 0 &&
        strncmp(gate, "learned_visual_program_", 23U) == 0) {
        if (priority_out != NULL) {
            *priority_out = "P0";
        }
        return "learned_visual_program_replacement_gate";
    }
    if (strcmp(experiment, "arc_agi") == 0 && scope != NULL &&
        strncmp(scope, "public_", 7U) == 0) {
        if (priority_out != NULL) {
            *priority_out = "P0";
        }
        return "public_arc_optional_gate";
    }
    if (strcmp(experiment, "haiku_understanding") == 0) {
        if (priority_out != NULL) {
            *priority_out = "P1";
        }
        return "haiku_understanding_optional_gate";
    }
    if (strcmp(experiment, "haiku_coherence_hard_negatives") == 0) {
        if (priority_out != NULL) {
            if (strcmp(gate, "neural_ranker_places_solution_near_top") == 0 ||
                strcmp(gate, "topic_coherence_family_survives") == 0 ||
                strcmp(gate, "plausible_topic_coherence_survives") == 0) {
                *priority_out = "P1";
            } else {
                *priority_out = "INFO";
            }
        }
        return "haiku_coherence_optional_gate";
    }
    if (strcmp(experiment, "haiku_word_lm") == 0) {
        if (priority_out != NULL) {
            *priority_out = "INFO";
        }
        return "diagnostic_generation_gate";
    }
    return "optional_gate";
}

static int nn_gd_health_add_optional_gate(NNGDHealthExperiment *experiment,
                                          const char *experiment_name,
                                          const char *scope,
                                          const char *gate) {
    const char *priority = "P2";
    const char *kind = nn_gd_health_optional_kind(experiment_name,
                                                 scope,
                                                 gate,
                                                 &priority);
    if (nn_gd_health_add_scoped_gate(&experiment->failing_optional_gates,
                                     scope,
                                     gate) != 0 ||
        nn_gd_health_add_triage(experiment, priority, kind) != 0) {
        return -1;
    }
    return 0;
}

static const char *nn_gd_health_priority_for_kind(const char *kind) {
    if (strcmp(kind, "learned_visual_program_replacement_gate") == 0 ||
        strcmp(kind, "public_arc_optional_gate") == 0 ||
        strcmp(kind, "arc_visual_replacement_gap_summary") == 0 ||
        strcmp(kind, "handcrafted_transform_dependency") == 0 ||
        strcmp(kind, "learned_visual_program_coverage_gap") == 0 ||
        strcmp(kind, "coverage_budget_violation") == 0) {
        return "P0";
    }
    if (strcmp(kind, "coherence_ranker_gap") == 0 ||
        strcmp(kind, "weak_corruption_family") == 0 ||
        strcmp(kind, "hard_negative_ranker_gap") == 0 ||
        strcmp(kind, "weak_hard_negative_family") == 0 ||
        strcmp(kind, "ranker_or_retention_gap") == 0 ||
        strcmp(kind, "engine_evidence_gap") == 0 ||
        strcmp(kind, "haiku_understanding_optional_gate") == 0 ||
        strcmp(kind, "tiny_net_weak_signal") == 0 ||
        strcmp(kind, "tiny_net_seed_variance_dominates_width") == 0) {
        return "P1";
    }
    if (strcmp(kind, "paper_claim_hygiene") == 0 ||
        strcmp(kind, "underpowered_claim") == 0 ||
        strcmp(kind, "optional_gate") == 0 ||
        strcmp(kind, "missing_plausible_negative_family") == 0 ||
        strcmp(kind, "tiny_net_missing_coverage") == 0 ||
        strcmp(kind, "tiny_net_not_plateaued") == 0 ||
        strcmp(kind, "tiny_net_capacity_helped_but_unsettled") == 0) {
        return "P2";
    }
    if (strcmp(kind, "diagnostic_generation_gate") == 0 ||
        strcmp(kind, "haiku_coherence_optional_gate") == 0 ||
        strcmp(kind, "tiny_net_not_width_limited") == 0 ||
        strcmp(kind, "tiny_net_audit_ready") == 0) {
        return "INFO";
    }
    return "P3";
}

static int nn_gd_health_add_coverage_hole(NNGDHealthExperiment *experiment,
                                          const char *experiment_name,
                                          const char *kind) {
    const char *triage_kind = kind;
    if (strcmp(experiment_name, "arc_agi") == 0 &&
        strcmp(kind, "no_train_consistent_candidate") == 0) {
        triage_kind = "learned_visual_program_coverage_gap";
    } else if (strcmp(experiment_name, "haiku_understanding") == 0 &&
               strcmp(kind, "ranker_accuracy_below_gate") == 0) {
        triage_kind = "coherence_ranker_gap";
    } else if (strcmp(experiment_name, "haiku_coherence_hard_negatives") == 0 &&
               strcmp(kind, "hard_negative_ranker_below_gate") == 0) {
        triage_kind = "hard_negative_ranker_gap";
    }
    if (nn_gd_health_counter_add(&experiment->coverage_hole_counts,
                                 kind,
                                 1U) != 0 ||
        nn_gd_health_add_triage(experiment,
                                nn_gd_health_priority_for_kind(triage_kind),
                                triage_kind) != 0) {
        return -1;
    }
    return 0;
}

static const char *nn_gd_health_json_array_end(const char *start) {
    const char *p;
    unsigned depth = 0U;
    if (start == NULL || *start != '[') {
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
        if (*p == '[') {
            ++depth;
        } else if (*p == ']') {
            --depth;
            if (depth == 0U) {
                return p + 1;
            }
        }
    }
    return NULL;
}

static int nn_gd_health_range_contains(const char *start,
                                       const char *end,
                                       const char *needle) {
    const char *cursor = start;
    if (start == NULL || end == NULL || needle == NULL) {
        return 0;
    }
    while (cursor < end) {
        const char *found = strstr(cursor, needle);
        if (found == NULL || found >= end) {
            return 0;
        }
        if (found + strlen(needle) <= end) {
            return 1;
        }
        cursor = found + 1;
    }
    return 0;
}

static const char *nn_gd_health_json_find_top_key_value(const char *line,
                                                       const char *key) {
    const char *p;
    size_t key_len;
    unsigned depth = 0U;
    if (line == NULL || key == NULL) {
        return NULL;
    }
    key_len = strlen(key);
    for (p = line; *p != '\0'; ++p) {
        if (*p == '"') {
            const char *key_start = p + 1;
            const char *after = nn_gd_json_skip_string_literal(p);
            const char *colon;
            if (after == NULL) {
                return NULL;
            }
            colon = nn_gd_json_skip_space(after);
            if (depth == 1U && colon != NULL && *colon == ':' &&
                (size_t)(after - key_start - 1) == key_len &&
                strncmp(key_start, key, key_len) == 0) {
                return colon + 1;
            }
            p = after - 1;
            continue;
        }
        if (*p == '{' || *p == '[') {
            ++depth;
        } else if (*p == '}' || *p == ']') {
            if (depth == 0U) {
                return NULL;
            }
            --depth;
        }
    }
    return NULL;
}

static char *nn_gd_health_json_dup_top_string(const char *line, const char *key) {
    return nn_gd_json_dup_string_value(
        nn_gd_health_json_find_top_key_value(line, key));
}

static long nn_gd_health_json_top_long_or(const char *line,
                                          const char *key,
                                          long default_value) {
    const char *cursor =
        nn_gd_json_skip_space(nn_gd_health_json_find_top_key_value(line, key));
    char *end = NULL;
    long value;
    if (cursor == NULL || strncmp(cursor, "null", 4U) == 0) {
        return default_value;
    }
    value = strtol(cursor, &end, 10);
    return end == cursor ? default_value : value;
}

static int nn_gd_health_parse_coverage_gates(NNGDHealthExperiment *experiment,
                                             const char *experiment_name,
                                             const char *line,
                                             const char *scope) {
    const char *gates =
        nn_gd_json_skip_space(nn_gd_health_json_find_top_key_value(line, "gates"));
    const char *end;
    const char *cursor;
    if (gates == NULL || *gates != '{') {
        return 0;
    }
    end = nn_gd_arc_hole_json_object_end(gates);
    if (end == NULL) {
        return -1;
    }
    cursor = gates + 1;
    while (cursor < end) {
        char *gate_name;
        const char *value;
        const char *value_end;
        const char *passed_cursor;
        const char *required_cursor;
        int passed;
        int required;
        cursor = nn_gd_json_skip_space(cursor);
        if (cursor == NULL || cursor >= end || *cursor == '}') {
            break;
        }
        if (*cursor == ',') {
            ++cursor;
            continue;
        }
        if (*cursor != '"') {
            break;
        }
        gate_name = nn_gd_json_dup_string_value_advance(&cursor);
        if (gate_name == NULL) {
            return -1;
        }
        cursor = nn_gd_json_skip_space(cursor);
        if (cursor == NULL || *cursor != ':') {
            free(gate_name);
            return -1;
        }
        value = nn_gd_json_skip_space(cursor + 1);
        if (value == NULL || *value != '{') {
            free(gate_name);
            return -1;
        }
        value_end = nn_gd_arc_hole_json_object_end(value);
        if (value_end == NULL) {
            free(gate_name);
            return -1;
        }
        passed_cursor =
            nn_gd_json_skip_space(nn_gd_json_find_key_value_range(value,
                                                                  value_end,
                                                                  "passed"));
        required_cursor =
            nn_gd_json_skip_space(nn_gd_json_find_key_value_range(value,
                                                                  value_end,
                                                                  "required"));
        passed = passed_cursor != NULL && strncmp(passed_cursor, "true", 4U) == 0;
        required = required_cursor != NULL &&
                   strncmp(required_cursor, "true", 4U) == 0;
        if (passed) {
            if (nn_gd_health_add_scoped_gate(&experiment->passing_gates,
                                             scope,
                                             gate_name) != 0) {
                free(gate_name);
                return -1;
            }
        } else if (required) {
            if (nn_gd_health_add_scoped_gate(&experiment->failing_required_gates,
                                             scope,
                                             gate_name) != 0) {
                free(gate_name);
                return -1;
            }
        } else if (nn_gd_health_add_optional_gate(experiment,
                                                  experiment_name,
                                                  scope,
                                                  gate_name) != 0) {
            free(gate_name);
            return -1;
        }
        free(gate_name);
        cursor = value_end;
    }
    return 0;
}

static int nn_gd_health_parse_coverage_holes(NNGDHealthExperiment *experiment,
                                             const char *experiment_name,
                                             const char *line) {
    const char *holes =
        nn_gd_json_skip_space(
            nn_gd_health_json_find_top_key_value(line, "coverage_holes"));
    const char *end;
    const char *cursor;
    if (holes == NULL || *holes != '[') {
        return 0;
    }
    end = nn_gd_health_json_array_end(holes);
    if (end == NULL) {
        return -1;
    }
    cursor = holes;
    while (cursor < end) {
        const char *found = strstr(cursor, "\"kind\":");
        char *kind;
        if (found == NULL || found >= end) {
            break;
        }
        kind = nn_gd_json_dup_string_value(found + strlen("\"kind\":"));
        if (kind == NULL) {
            return -1;
        }
        if (kind[0] != '\0' &&
            nn_gd_health_add_coverage_hole(experiment,
                                           experiment_name,
                                           kind) != 0) {
            free(kind);
            return -1;
        }
        free(kind);
        cursor = found + strlen("\"kind\":");
    }
    return 0;
}

static int nn_gd_health_parse_budget_violations(NNGDHealthExperiment *experiment,
                                                const char *line) {
    const char *violations =
        nn_gd_json_skip_space(
            nn_gd_health_json_find_top_key_value(line, "budget_violations"));
    if (violations == NULL || *violations != '[') {
        return 0;
    }
    ++violations;
    violations = nn_gd_json_skip_space(violations);
    if (violations != NULL && *violations != ']') {
        if (nn_gd_health_counter_add(&experiment->coverage_hole_counts,
                                     "coverage_budget_violation",
                                     1U) != 0 ||
            nn_gd_health_add_triage(experiment,
                                    "P0",
                                    "coverage_budget_violation") != 0) {
            return -1;
        }
    }
    return 0;
}

static int nn_gd_health_read_numeric_summary(NNGDHealthExperimentList *experiments,
                                             const char *path) {
    FILE *fp;
    char *line;
    if (path == NULL || path[0] == '\0') {
        return 0;
    }
    fp = fopen(path, "r");
    if (fp == NULL) {
        return 0;
    }
    while ((line = nn_gd_read_line_dynamic(fp)) != NULL) {
        NNGDJsonIndex index;
        char *schema;
        nn_gd_json_index_line(line, &index);
        schema = nn_gd_json_dup_index_string(&index, "schema");
        if (schema != NULL &&
            strcmp(schema, "crlp.numeric_certificate_experiment_summary.v1") == 0) {
            char *target = nn_gd_json_dup_index_string(&index, "target");
            char *status = nn_gd_json_dup_index_string(&index, "status");
            char *experiment_name = nn_gd_health_experiment_from_target(target);
            NNGDHealthExperiment *experiment =
                nn_gd_health_experiment(experiments, experiment_name);
            long certificate_rows =
                nn_gd_json_index_long_or(&index, "certificate_rows", 0L);
            long committed_rows =
                nn_gd_json_index_long_or(&index, "committed_rows", 0L);
            long rejected_rows =
                nn_gd_json_index_long_or(&index, "rejected_rows", 0L);
            if (experiment == NULL ||
                nn_gd_string_list_add_unique(&experiment->numeric_targets,
                                             target) != 0 ||
                nn_gd_health_counter_add(&experiment->numeric_status_counts,
                                         status,
                                         1U) != 0) {
                free(target);
                free(status);
                free(experiment_name);
                free(schema);
                free(line);
                fclose(fp);
                return -1;
            }
            experiment->numeric_certificate_rows +=
                certificate_rows < 0L ? 0U : (size_t)certificate_rows;
            experiment->numeric_committed_rows +=
                committed_rows < 0L ? 0U : (size_t)committed_rows;
            experiment->numeric_rejected_rows +=
                rejected_rows < 0L ? 0U : (size_t)rejected_rows;
            if (strcmp(status, "failed") == 0) {
                ++experiment->numeric_failed_statuses;
            }
            free(target);
            free(status);
            free(experiment_name);
        }
        free(schema);
        free(line);
    }
    fclose(fp);
    return 0;
}

static int nn_gd_health_read_coverage(NNGDHealthExperimentList *experiments,
                                      const char *path) {
    FILE *fp;
    char *line;
    if (path == NULL || path[0] == '\0') {
        return 0;
    }
    fp = fopen(path, "r");
    if (fp == NULL) {
        return 0;
    }
    while ((line = nn_gd_read_line_dynamic(fp)) != NULL) {
        char *schema;
        int is_coverage_row;
        schema = nn_gd_health_json_dup_top_string(line, "schema");
        is_coverage_row =
            (schema != NULL &&
             strcmp(schema, "crlp.learned_grammar_coverage.v1") == 0) ||
            strstr(line, "\"schema\":\"crlp.learned_grammar_coverage.v1\"") != NULL;
        if (is_coverage_row) {
            char *experiment_name =
                nn_gd_health_json_dup_top_string(line, "experiment");
            char *scope = nn_gd_health_json_dup_top_string(line, "scope");
            NNGDHealthExperiment *experiment =
                nn_gd_health_experiment(experiments, experiment_name);
            long certificates =
                nn_gd_health_json_top_long_or(line, "certificate_rows", 0L);
            if (experiment_name != NULL && experiment_name[0] == '\0') {
                free(experiment_name);
                experiment_name = nn_gd_json_dup_key_string(line, "experiment");
                experiment = nn_gd_health_experiment(experiments, experiment_name);
            }
            if (scope != NULL && scope[0] == '\0') {
                free(scope);
                scope = nn_gd_json_dup_key_string(line, "scope");
            }
            if (certificates == 0L) {
                certificates = (long)nn_gd_json_get_double_or_zero(line,
                                                                   "certificate_rows");
            }
            if (experiment == NULL ||
                nn_gd_string_list_add_unique(&experiment->coverage_sources,
                                             path) != 0) {
                free(experiment_name);
                free(scope);
                free(schema);
                free(line);
                fclose(fp);
                return -1;
            }
            ++experiment->coverage_rows;
            experiment->verifier_certificate_rows +=
                certificates < 0L ? 0U : (size_t)certificates;
            if (nn_gd_health_parse_coverage_gates(experiment,
                                                  experiment_name,
                                                  line,
                                                  scope) != 0 ||
                nn_gd_health_parse_coverage_holes(experiment,
                                                  experiment_name,
                                                  line) != 0 ||
                nn_gd_health_parse_budget_violations(experiment, line) != 0) {
                free(experiment_name);
                free(scope);
                free(schema);
                free(line);
                fclose(fp);
                return -1;
            }
            free(experiment_name);
            free(scope);
        }
        free(schema);
        free(line);
    }
    fclose(fp);
    return 0;
}

static int nn_gd_health_read_claims(NNGDHealthExperimentList *experiments,
                                    const char *path) {
    static const NNGDHealthClaimOwner owners[] = {
        {"C1.prime_gate_delta", {"prime_gap", NULL, NULL}, 1U},
        {"C1.prime_gate_delta_ci", {"prime_gap", NULL, NULL}, 1U},
        {"C2.prime_proposer_fixed_budget", {"prime_gap", NULL, NULL}, 1U},
        {"C1.arc_gate_scorer_invariance", {"arc_agi", NULL, NULL}, 1U},
        {"C1.arc_public_evidence_surface", {"arc_agi", NULL, NULL}, 1U},
        {"C3.arc_ood_calibration_tradeoff", {"arc_agi", NULL, NULL}, 1U},
        {"C4.arc_ood_shared_vs_single", {"arc_agi", "multitask_domain", NULL}, 2U},
        {"C4.delayed_claim_sibling_transfer", {"delayed_claim", NULL, NULL}, 1U},
        {"C4.multitask_capacity_tradeoff", {"multitask_domain", NULL, NULL}, 1U},
        {"C4.adapter_heads_shared_trunk", {"multitask_domain", NULL, NULL}, 1U},
        {"C5.rejected_candidate_reason_learning", {"modular_gate_search", NULL, NULL}, 1U},
        {"C5.hypothesis_library_promotion", {"arc_agi", NULL, NULL}, 1U},
        {"C5.arc_verified_self_play", {"arc_agi", NULL, NULL}, 1U},
        {"C7.tiny_net_plateau_stop_rule", {"multitask_domain", NULL, NULL}, 1U},
        {"C8.trace_determinism_audit", {"prime_gap", NULL, NULL}, 1U},
        {"C9.signal_flight_replay_control", {"signal_flight", NULL, NULL}, 1U},
        {"C9.signal_live_neural_active", {"signal_flight", NULL, NULL}, 1U},
    };
    size_t len = 0U;
    char *content;
    const char *claims_start;
    const char *claims_end;
    size_t i;
    if (path == NULL || path[0] == '\0') {
        return 0;
    }
    content = nn_gd_read_file_dynamic_path(path, &len);
    if (content == NULL) {
        return 0;
    }
    claims_start = strstr(content, "\"claims\"");
    claims_end = claims_start == NULL ? NULL : strstr(claims_start, "\"limitations\"");
    if (claims_start == NULL) {
        free(content);
        return 0;
    }
    if (claims_end == NULL) {
        claims_end = content + len;
    }
    for (i = 0U; i < sizeof(owners) / sizeof(owners[0]); ++i) {
        const char *found = strstr(claims_start, owners[i].claim_id);
        const char *next_id;
        const char *object_end;
        const char *status = "unknown";
        int required;
        size_t owner_index;
        if (found == NULL || found >= claims_end) {
            continue;
        }
        next_id = strstr(found + strlen(owners[i].claim_id), "\"id\"");
        object_end = next_id != NULL && next_id < claims_end ? next_id : claims_end;
        if (nn_gd_health_range_contains(found, object_end, "\"status\":\"supported\"")) {
            status = "supported";
        } else if (nn_gd_health_range_contains(found,
                                               object_end,
                                               "\"status\":\"contradicted\"")) {
            status = "contradicted";
        } else if (nn_gd_health_range_contains(found,
                                               object_end,
                                               "\"status\":\"underpowered\"")) {
            status = "underpowered";
        } else if (nn_gd_health_range_contains(found, object_end, "\"status\":\"missing\"")) {
            status = "missing";
        } else if (nn_gd_health_range_contains(found, object_end, "\"status\":\"stale\"")) {
            status = "stale";
        }
        required = nn_gd_health_range_contains(found,
                                               object_end,
                                               "\"required_for_submission\":true");
        for (owner_index = 0U; owner_index < owners[i].owner_count; ++owner_index) {
            NNGDHealthExperiment *experiment =
                nn_gd_health_experiment(experiments, owners[i].owners[owner_index]);
            if (experiment == NULL ||
                nn_gd_string_list_add_unique(&experiment->claim_ids,
                                             owners[i].claim_id) != 0 ||
                nn_gd_health_counter_add(&experiment->claim_status_counts,
                                         status,
                                         1U) != 0) {
                free(content);
                return -1;
            }
            if (required &&
                nn_gd_string_list_add_unique(&experiment->required_claim_ids,
                                             owners[i].claim_id) != 0) {
                free(content);
                return -1;
            }
            if (strcmp(status, "contradicted") == 0 &&
                nn_gd_health_add_triage(experiment,
                                        "P2",
                                        "paper_claim_hygiene") != 0) {
                free(content);
                return -1;
            }
            if (strcmp(status, "underpowered") == 0 &&
                nn_gd_health_add_triage(experiment,
                                        "P2",
                                        "underpowered_claim") != 0) {
                free(content);
                return -1;
            }
        }
    }
    free(content);
    return 0;
}

static int nn_gd_health_add_evidence_gate(NNGDHealthExperiment *experiment,
                                          const char *gate,
                                          int passed,
                                          int required);

static int nn_gd_health_engine_evidence_required(const char *name) {
    static const char *required[] = {
        "arc_cell_color",
        "bitwise_macro_step",
        "delayed_claim",
        "grid_path",
        "haiku_adjacency",
        "prime_gap_candidate",
        "program_synthesis_ops",
        "signal_wasd",
    };
    size_t i;
    if (name == NULL) {
        return 0;
    }
    for (i = 0U; i < sizeof(required) / sizeof(required[0]); ++i) {
        if (strcmp(name, required[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

static int nn_gd_health_seed_engine_evidence_expectations(
    NNGDHealthExperimentList *experiments) {
    static const char *required[] = {
        "arc_cell_color",
        "bitwise_macro_step",
        "delayed_claim",
        "grid_path",
        "haiku_adjacency",
        "prime_gap_candidate",
        "program_synthesis_ops",
        "signal_wasd",
    };
    size_t i;
    for (i = 0U; i < sizeof(required) / sizeof(required[0]); ++i) {
        if (nn_gd_health_experiment(experiments, required[i]) == NULL) {
            return -1;
        }
    }
    return 0;
}

static int nn_gd_health_read_engine_evidence(NNGDHealthExperimentList *experiments,
                                             const char *path) {
    FILE *fp;
    char *line;
    if (path == NULL || path[0] == '\0') {
        return 0;
    }
    if (nn_gd_health_seed_engine_evidence_expectations(experiments) != 0) {
        return -1;
    }
    fp = fopen(path, "r");
    if (fp == NULL) {
        return 0;
    }
    while ((line = nn_gd_read_line_dynamic(fp)) != NULL) {
        NNGDJsonIndex index;
        char *schema;
        char *experiment_name;
        char *rule_search_schema;
        char *replay_schema;
        NNGDHealthExperiment *experiment;
        int complete_candidate_groups;
        int has_verifier_targets;
        int has_checkpoint_identity;
        int smoke_promotion_ready;
        nn_gd_json_index_line(line, &index);
        schema = nn_gd_json_dup_index_string(&index, "schema");
        if (schema == NULL ||
            strcmp(schema, "crlp.engine_evidence.v1") != 0) {
            free(schema);
            free(line);
            continue;
        }
        experiment_name = nn_gd_json_dup_index_string(&index, "experiment");
        rule_search_schema =
            nn_gd_json_dup_index_string(&index, "rule_search_schema");
        replay_schema = nn_gd_json_dup_index_string(&index, "replay_schema");
        experiment = nn_gd_health_experiment(experiments, experiment_name);
        if (experiment == NULL ||
            nn_gd_string_list_add_unique(&experiment->evidence_sources,
                                         path) != 0) {
            free(replay_schema);
            free(rule_search_schema);
            free(experiment_name);
            free(schema);
            free(line);
            fclose(fp);
            return -1;
        }
        ++experiment->evidence_rows;
        ++experiment->engine_evidence_rows;
        complete_candidate_groups =
            nn_gd_json_get_bool(line, "complete_candidate_groups");
        has_verifier_targets =
            nn_gd_json_get_bool(line, "has_verifier_targets");
        has_checkpoint_identity =
            nn_gd_json_get_bool(line, "has_checkpoint_identity");
        smoke_promotion_ready =
            nn_gd_json_get_bool(line, "smoke_promotion_ready");
        if (nn_gd_health_add_evidence_gate(experiment,
                                           "dataset_engine_evidence_surface",
                                           1,
                                           1) != 0 ||
            nn_gd_health_add_evidence_gate(
                experiment,
                "dataset_engine_candidate_group_contract",
                complete_candidate_groups,
                1) != 0 ||
            nn_gd_health_add_evidence_gate(
                experiment,
                "dataset_engine_verifier_target_contract",
                has_verifier_targets,
                1) != 0 ||
            nn_gd_health_add_evidence_gate(
                experiment,
                "dataset_engine_checkpoint_identity",
                has_checkpoint_identity,
                1) != 0 ||
            nn_gd_health_add_evidence_gate(
                experiment,
                "dataset_engine_promoted_smoke",
                smoke_promotion_ready,
                1) != 0 ||
            nn_gd_health_add_evidence_gate(
                experiment,
                "dataset_engine_rule_search_schema_declared",
                rule_search_schema != NULL &&
                    strcmp(rule_search_schema, "none") != 0,
                0) != 0 ||
            nn_gd_health_add_evidence_gate(
                experiment,
                "dataset_engine_replay_schema_declared",
                replay_schema != NULL && strcmp(replay_schema, "none") != 0,
                0) != 0) {
            free(replay_schema);
            free(rule_search_schema);
            free(experiment_name);
            free(schema);
            free(line);
            fclose(fp);
            return -1;
        }
        if (!smoke_promotion_ready &&
            nn_gd_health_add_triage(experiment,
                                    "P1",
                                    "engine_smoke_promotion_gap") != 0) {
            free(replay_schema);
            free(rule_search_schema);
            free(experiment_name);
            free(schema);
            free(line);
            fclose(fp);
            return -1;
        }
        free(replay_schema);
        free(rule_search_schema);
        free(experiment_name);
        free(schema);
        free(line);
    }
    fclose(fp);
    return 0;
}

static int nn_gd_health_add_evidence_gate(NNGDHealthExperiment *experiment,
                                          const char *gate,
                                          int passed,
                                          int required) {
    if (passed) {
        return nn_gd_string_list_add_unique(&experiment->passing_gates, gate);
    }
    if (required) {
        return nn_gd_string_list_add_unique(&experiment->failing_required_gates, gate);
    }
    if (nn_gd_string_list_add_unique(&experiment->failing_optional_gates, gate) != 0 ||
        nn_gd_health_add_triage(experiment, "P3", "evidence_surface_gap") != 0) {
        return -1;
    }
    return 0;
}

static int nn_gd_health_ingest_signal_file(NNGDHealthExperiment *experiment,
                                           const char *path,
                                           const char *gate,
                                           int any_signal_rows) {
    size_t rows = nn_gd_ngd_count_rows(path);
    if (rows > 0U) {
        experiment->evidence_rows += rows;
        if (nn_gd_string_list_add_unique(&experiment->evidence_sources, path) != 0 ||
            nn_gd_health_add_evidence_gate(experiment, gate, 1, 0) != 0) {
            return -1;
        }
    } else if (any_signal_rows &&
               nn_gd_health_add_evidence_gate(experiment, gate, 0, 0) != 0) {
        return -1;
    }
    return 0;
}

static int nn_gd_health_read_signal(NNGDHealthExperimentList *experiments,
                                    const char *summary_path,
                                    const char *rollout_path,
                                    const char *live_active_path,
                                    const char *shadow_replay_path) {
    size_t summary_rows = nn_gd_ngd_count_rows(summary_path);
    size_t rollout_rows = nn_gd_ngd_count_rows(rollout_path);
    size_t live_rows = nn_gd_ngd_count_rows(live_active_path);
    size_t shadow_rows = nn_gd_ngd_count_rows(shadow_replay_path);
    int any = summary_rows > 0U || rollout_rows > 0U ||
              live_rows > 0U || shadow_rows > 0U;
    NNGDHealthExperiment *experiment;
    if (!any) {
        return 0;
    }
    experiment = nn_gd_health_experiment(experiments, "signal_flight");
    if (experiment == NULL) {
        return -1;
    }
    return nn_gd_health_ingest_signal_file(
               experiment,
               summary_path,
               "signal_flight_counterfactual_replay_surface",
               any) ||
           nn_gd_health_ingest_signal_file(experiment,
                                           rollout_path,
                                           "signal_flight_rollout_surface",
                                           any) ||
           nn_gd_health_ingest_signal_file(experiment,
                                           live_active_path,
                                           "signal_live_neural_active_surface",
                                           any) ||
           nn_gd_health_ingest_signal_file(experiment,
                                           shadow_replay_path,
                                           "signal_shadow_replay_surface",
                                           any)
               ? -1
               : 0;
}

static int nn_gd_health_read_neural_grammar(NNGDHealthExperimentList *experiments,
                                            const char *path) {
    FILE *fp;
    char *line;
    NNGDHealthExperiment *experiment;
    double attempts;
    double heldout_attempts;
    double neural_proposals;
    double frontier;
    double repair;
    if (path == NULL || path[0] == '\0' || nn_gd_ngd_count_rows(path) == 0U) {
        return 0;
    }
    fp = fopen(path, "r");
    if (fp == NULL) {
        return 0;
    }
    line = nn_gd_read_line_dynamic(fp);
    fclose(fp);
    if (line == NULL) {
        return 0;
    }
    experiment = nn_gd_health_experiment(experiments, "arc_agi");
    if (experiment == NULL ||
        nn_gd_string_list_add_unique(&experiment->evidence_sources, path) != 0) {
        free(line);
        return -1;
    }
    experiment->evidence_rows += 1U;
    attempts = nn_gd_json_get_double_or_zero(line, "arc_visual_pipeline_attempt_count");
    heldout_attempts =
        nn_gd_json_get_double_or_zero(line, "arc_visual_pipeline_heldout_attempt_count");
    neural_proposals =
        nn_gd_json_get_double_or_zero(line, "arc_visual_pipeline_neural_proposal_count");
    frontier = nn_gd_json_get_double_or_zero(
        line,
        "arc_visual_pipeline_coverage_frontier_count");
    repair = nn_gd_json_get_double_or_zero(line,
                                           "arc_visual_pipeline_repair_sketch_count");
    if (nn_gd_health_add_evidence_gate(experiment,
                                       "learned_visual_program_attempt_surface",
                                       attempts + heldout_attempts > 0.0,
                                       0) != 0 ||
        nn_gd_health_add_evidence_gate(experiment,
                                       "learned_visual_program_ranker_surface",
                                       neural_proposals > 0.0,
                                       0) != 0 ||
        nn_gd_health_add_evidence_gate(experiment,
                                       "learned_visual_program_frontier_surface",
                                       frontier > 0.0 || repair > 0.0,
                                       0) != 0) {
        free(line);
        return -1;
    }
    free(line);
    return 0;
}

static int nn_gd_health_read_protocol_spine(NNGDHealthExperimentList *experiments,
                                            const char *path) {
    FILE *fp;
    char *line;
    if (path == NULL || path[0] == '\0') {
        return 0;
    }
    fp = fopen(path, "r");
    if (fp == NULL) {
        return 0;
    }
    while ((line = nn_gd_read_line_dynamic(fp)) != NULL) {
        NNGDJsonIndex index;
        char *schema;
        char *domain;
        NNGDHealthExperiment *experiment;
        nn_gd_json_index_line(line, &index);
        schema = nn_gd_json_dup_index_string(&index, "schema");
        if (schema == NULL ||
            (strcmp(schema, "crlp.protocol.candidate_row.v1") != 0 &&
             strcmp(schema, "crlp.protocol.verifier_outcome.v1") != 0)) {
            free(schema);
            free(line);
            continue;
        }
        domain = nn_gd_json_dup_index_string(&index, "domain");
        experiment = nn_gd_health_experiment(experiments, domain);
        if (experiment == NULL ||
            nn_gd_string_list_add_unique(&experiment->evidence_sources, path) != 0) {
            free(domain);
            free(schema);
            free(line);
            fclose(fp);
            return -1;
        }
        ++experiment->evidence_rows;
        if (strcmp(schema, "crlp.protocol.candidate_row.v1") == 0) {
            ++experiment->protocol_candidate_rows;
        } else {
            ++experiment->protocol_outcome_rows;
        }
        free(domain);
        free(schema);
        free(line);
    }
    fclose(fp);
    return 0;
}

static int nn_gd_health_read_operator_promotions(
    NNGDHealthExperimentList *experiments,
    const char *path) {
    FILE *fp;
    char *line;
    NNGDHealthExperiment *experiment;
    if (path == NULL || path[0] == '\0') {
        return 0;
    }
    fp = fopen(path, "r");
    if (fp == NULL) {
        return 0;
    }
    experiment = nn_gd_health_experiment(experiments, "protocol_operator_promotion");
    if (experiment == NULL) {
        fclose(fp);
        return -1;
    }
    while ((line = nn_gd_read_line_dynamic(fp)) != NULL) {
        NNGDJsonIndex index;
        char *schema;
        nn_gd_json_index_line(line, &index);
        schema = nn_gd_json_dup_index_string(&index, "schema");
        if (schema != NULL &&
            strcmp(schema, "crlp.protocol.operator_promotion.v1") == 0) {
            ++experiment->operator_rows;
            ++experiment->evidence_rows;
            if (nn_gd_string_list_add_unique(&experiment->evidence_sources,
                                             path) != 0) {
                free(schema);
                free(line);
                fclose(fp);
                return -1;
            }
            if (nn_gd_json_get_bool(line, "promoted")) {
                ++experiment->promoted_operator_rows;
            }
        }
        free(schema);
        free(line);
    }
    fclose(fp);
    return 0;
}

static const char *nn_gd_failure_class_for_tiny_diagnosis(
    const char *diagnosis) {
    if (diagnosis == NULL || diagnosis[0] == '\0') {
        return "diagnostic_unclassified";
    }
    if (strcmp(diagnosis, "missing_coverage") == 0) {
        return "instrumentation_gap";
    }
    if (strcmp(diagnosis, "weak_signal") == 0) {
        return "representation_or_label_gap";
    }
    if (strcmp(diagnosis, "seed_variance_dominates_width") == 0) {
        return "seed_variance";
    }
    if (strcmp(diagnosis, "not_plateaued") == 0) {
        return "optimizer_unsettled";
    }
    if (strcmp(diagnosis, "not_width_limited") == 0) {
        return "protocol_or_sampling_gap";
    }
    if (strcmp(diagnosis, "capacity_helped_but_unsettled") == 0) {
        return "capacity_unsettled";
    }
    if (strcmp(diagnosis, "audit_ready") == 0) {
        return "claim_review_ready";
    }
    return "diagnostic_unclassified";
}

static int nn_gd_health_read_tiny_net_diagnostics(
    NNGDHealthExperimentList *experiments,
    const char *path) {
    FILE *fp;
    char *line;
    NNGDHealthExperiment *experiment;
    if (path == NULL || path[0] == '\0') {
        return 0;
    }
    fp = fopen(path, "r");
    if (fp == NULL) {
        return 0;
    }
    experiment = nn_gd_health_experiment(experiments, "multitask_domain");
    if (experiment == NULL) {
        fclose(fp);
        return -1;
    }
    while ((line = nn_gd_read_line_dynamic(fp)) != NULL) {
        NNGDJsonIndex index;
        char *schema;
        char *diagnosis;
        char *next_step;
        int audit_ready;
        nn_gd_json_index_line(line, &index);
        schema = nn_gd_json_dup_index_string(&index, "schema");
        if (schema == NULL ||
            strcmp(schema, "crlp.multitask.tiny_net_diagnostic.v1") != 0) {
            free(schema);
            free(line);
            continue;
        }
        diagnosis = nn_gd_json_dup_index_string(&index, "diagnosis");
        next_step = nn_gd_json_dup_index_string(&index, "next_step");
        audit_ready = nn_gd_json_index_bool_or(&index, "audit_ready", 0);
        if (diagnosis == NULL || next_step == NULL ||
            nn_gd_string_list_add_unique(&experiment->evidence_sources,
                                         path) != 0 ||
            nn_gd_string_list_add_unique(&experiment->diagnostic_sources,
                                         path) != 0) {
            free(next_step);
            free(diagnosis);
            free(schema);
            free(line);
            fclose(fp);
            return -1;
        }
        ++experiment->evidence_rows;
        ++experiment->diagnostic_rows;
        if (!audit_ready && strcmp(diagnosis, "audit_ready") != 0) {
            char triage_kind[160];
            int written =
                snprintf(triage_kind,
                         sizeof(triage_kind),
                         "tiny_net_%s",
                         diagnosis);
            if (written <= 0 || (size_t)written >= sizeof(triage_kind) ||
                nn_gd_health_counter_add(
                    &experiment->diagnostic_failure_class_counts,
                    nn_gd_failure_class_for_tiny_diagnosis(diagnosis),
                    1U) != 0 ||
                nn_gd_health_counter_add(
                    &experiment->diagnostic_next_step_counts,
                    next_step,
                    1U) != 0 ||
                nn_gd_health_add_triage(
                    experiment,
                    nn_gd_health_priority_for_kind(triage_kind),
                    triage_kind) != 0) {
                free(next_step);
                free(diagnosis);
                free(schema);
                free(line);
                fclose(fp);
                return -1;
            }
        }
        free(next_step);
        free(diagnosis);
        free(schema);
        free(line);
    }
    fclose(fp);
    return 0;
}

static int nn_gd_health_compare_experiments(const void *a, const void *b) {
    const NNGDHealthExperiment *ea = (const NNGDHealthExperiment *)a;
    const NNGDHealthExperiment *eb = (const NNGDHealthExperiment *)b;
    return strcmp(ea->name, eb->name);
}

static int nn_gd_health_finalize(NNGDHealthExperimentList *experiments) {
    size_t i;
    qsort(experiments->items,
          experiments->count,
          sizeof(experiments->items[0]),
          nn_gd_health_compare_experiments);
    for (i = 0U; i < experiments->count; ++i) {
        NNGDHealthExperiment *experiment = &experiments->items[i];
        nn_gd_string_list_sort(&experiment->numeric_targets);
        nn_gd_string_list_sort(&experiment->passing_gates);
        nn_gd_string_list_sort(&experiment->failing_required_gates);
        nn_gd_string_list_sort(&experiment->failing_optional_gates);
        nn_gd_string_list_sort(&experiment->coverage_sources);
        nn_gd_string_list_sort(&experiment->evidence_sources);
        nn_gd_string_list_sort(&experiment->claim_ids);
        nn_gd_string_list_sort(&experiment->required_claim_ids);
        if (experiment->protocol_candidate_rows > 0U ||
            experiment->protocol_outcome_rows > 0U) {
            if (nn_gd_health_add_evidence_gate(
                    experiment,
                    "protocol_candidate_outcome_surface",
                    experiment->protocol_candidate_rows > 0U &&
                        experiment->protocol_outcome_rows > 0U,
                    0) != 0) {
                return -1;
            }
        }
        if (experiment->operator_rows > 0U) {
            if (nn_gd_health_add_evidence_gate(
                    experiment,
                    "cross_domain_operator_promotion_surface",
                    experiment->promoted_operator_rows > 0U,
                    0) != 0) {
                return -1;
            }
        }
        if (nn_gd_health_engine_evidence_required(experiment->name) &&
            experiment->engine_evidence_rows == 0U) {
            if (nn_gd_string_list_add_unique(
                    &experiment->failing_required_gates,
                    "dataset_engine_evidence_surface") != 0 ||
                nn_gd_health_add_triage(experiment,
                                        "P1",
                                        "engine_evidence_gap") != 0) {
                return -1;
            }
        }
        if (experiment->claim_ids.count > 0U &&
            experiment->numeric_targets.count == 0U &&
            experiment->coverage_rows == 0U &&
            experiment->evidence_rows == 0U) {
            if (nn_gd_health_counter_add(&experiment->coverage_hole_counts,
                                         "missing_health_evidence_surface",
                                         1U) != 0 ||
                nn_gd_health_add_triage(experiment,
                                        "P3",
                                        "instrumentation_gap") != 0) {
                return -1;
            }
        }
    }
    return 0;
}

static void nn_gd_health_write_counter_object(FILE *fp,
                                              const NNGDHealthCounterList *list) {
    size_t i;
    fputc('{', fp);
    for (i = 0U; i < list->count; ++i) {
        if (i > 0U) {
            fputc(',', fp);
        }
        nn_gd_json_write_string(fp, list->items[i].key);
        fprintf(fp, ":%zu", list->items[i].count);
    }
    fputc('}', fp);
}

static void nn_gd_health_write_hole_array(FILE *fp,
                                          const NNGDHealthCounterList *holes) {
    size_t i;
    fputc('[', fp);
    for (i = 0U; i < holes->count; ++i) {
        if (i > 0U) {
            fputc(',', fp);
        }
        fputs("{\"kind\":", fp);
        nn_gd_json_write_string(fp, holes->items[i].key);
        fprintf(fp, ",\"count\":%zu}", holes->items[i].count);
    }
    fputc(']', fp);
}

static void nn_gd_health_write_triage_array(FILE *fp,
                                            const NNGDHealthCounterList *triage) {
    size_t i;
    fputc('[', fp);
    for (i = 0U; i < triage->count; ++i) {
        const char *kind = triage->items[i].key;
        const char *priority = nn_gd_health_priority_for_kind(kind);
        if (i > 0U) {
            fputc(',', fp);
        }
        fputs("{\"priority\":", fp);
        nn_gd_json_write_string(fp, priority);
        fputs(",\"kind\":", fp);
        nn_gd_json_write_string(fp, kind);
        fputs(",\"source\":\"c_experiment_health\",", fp);
        fputs("\"summary\":", fp);
        nn_gd_json_write_string(fp, kind);
        fputs(",\"action\":\"inspect source evidence and keep verifier-gated promotion explicit\",",
              fp);
        fprintf(fp, "\"count\":%zu}", triage->items[i].count);
    }
    fputc(']', fp);
}

static const char *nn_gd_health_status(const NNGDHealthExperiment *experiment) {
    int fail = experiment->numeric_rejected_rows > 0U ||
               experiment->numeric_failed_statuses > 0U ||
               experiment->failing_required_gates.count > 0U;
    int warn = nn_gd_health_counter_get(&experiment->triage_priority_counts,
                                        "P0") > 0U ||
               nn_gd_health_counter_get(&experiment->triage_priority_counts,
                                        "P1") > 0U ||
               nn_gd_health_counter_get(&experiment->triage_priority_counts,
                                        "P2") > 0U;
    if (fail) {
        return "fail";
    }
    return warn ? "warn" : "pass";
}

static void nn_gd_health_write_experiment(FILE *fp,
                                          const NNGDHealthExperiment *experiment,
                                          const char *paper_claims_path) {
    size_t claim_supported =
        nn_gd_health_counter_get(&experiment->claim_status_counts, "supported");
    size_t claim_contradicted =
        nn_gd_health_counter_get(&experiment->claim_status_counts, "contradicted");
    int numeric_passed = experiment->numeric_rejected_rows == 0U &&
                         experiment->numeric_failed_statuses == 0U;
    fputs("{\"schema\":\"crlp.experiment_health.v1\",\"experiment\":", fp);
    nn_gd_json_write_string(fp, experiment->name);
    fputs(",\"status\":", fp);
    nn_gd_json_write_string(fp, nn_gd_health_status(experiment));
    fputs(",\"passing_gates\":", fp);
    nn_gd_json_write_string_list(fp, &experiment->passing_gates);
    fputs(",\"failing_required_gates\":", fp);
    nn_gd_json_write_string_list(fp, &experiment->failing_required_gates);
    fputs(",\"failing_optional_gates\":", fp);
    nn_gd_json_write_string_list(fp, &experiment->failing_optional_gates);
    fprintf(fp,
            ",\"certificate_rows\":{\"numeric\":%zu,\"verifier\":%zu}",
            experiment->numeric_certificate_rows,
            experiment->verifier_certificate_rows);
    fputs(",\"learning_signal\":{\"numeric_targets\":", fp);
    nn_gd_json_write_string_list(fp, &experiment->numeric_targets);
    fprintf(fp,
            ",\"numeric_committed_rows\":%zu,\"coverage_rows\":%zu,"
            "\"evidence_rows\":%zu,\"diagnostic_rows\":%zu,"
            "\"engine_evidence_rows\":%zu,"
            "\"protocol_candidate_rows\":%zu,"
            "\"protocol_outcome_rows\":%zu,\"promoted_operator_rows\":%zu}",
            experiment->numeric_committed_rows,
            experiment->coverage_rows,
            experiment->evidence_rows,
            experiment->diagnostic_rows,
            experiment->engine_evidence_rows,
            experiment->protocol_candidate_rows,
            experiment->protocol_outcome_rows,
            experiment->promoted_operator_rows);
    fputs(",\"coverage_holes\":", fp);
    nn_gd_health_write_hole_array(fp, &experiment->coverage_hole_counts);
    fputs(",\"triage\":", fp);
    nn_gd_health_write_triage_array(fp, &experiment->triage_kind_counts);
    fputs(",\"triage_summary\":{\"by_kind\":", fp);
    nn_gd_health_write_counter_object(fp, &experiment->triage_kind_counts);
    fputs(",\"by_priority\":", fp);
    nn_gd_health_write_counter_object(fp, &experiment->triage_priority_counts);
    fputs("},\"verifier_delta\":{}", fp);
    fputs(",\"numeric_certificate\":{\"passed\":", fp);
    fputs(numeric_passed ? "true" : "false", fp);
    fprintf(fp, ",\"rejected_rows\":%zu,\"rejection_reasons\":{},\"statuses\":",
            experiment->numeric_rejected_rows);
    nn_gd_health_write_counter_object(fp, &experiment->numeric_status_counts);
    fputs("}", fp);
    fputs(",\"paper_claim_support\":{\"coverage_scope\":{},\"status_counts\":", fp);
    nn_gd_health_write_counter_object(fp, &experiment->claim_status_counts);
    fputs(",\"claim_ids\":", fp);
    nn_gd_json_write_string_list(fp, &experiment->claim_ids);
    fputs(",\"required_claim_ids\":", fp);
    nn_gd_json_write_string_list(fp, &experiment->required_claim_ids);
    fputs(",\"supports_paper_claim\":", fp);
    fputs(claim_supported > 0U && claim_contradicted == 0U ? "true" : "false", fp);
    fputs("}", fp);
    fputs(",\"sources\":{\"coverage\":", fp);
    nn_gd_json_write_string_list(fp, &experiment->coverage_sources);
    fputs(",\"evidence\":", fp);
    nn_gd_json_write_string_list(fp, &experiment->evidence_sources);
    fputs(",\"diagnostics\":", fp);
    nn_gd_json_write_string_list(fp, &experiment->diagnostic_sources);
    fputs(",\"numeric_targets\":", fp);
    nn_gd_json_write_string_list(fp, &experiment->numeric_targets);
    fputs(",\"paper_claims\":", fp);
    if (paper_claims_path == NULL || paper_claims_path[0] == '\0') {
        fputs("null", fp);
    } else {
        nn_gd_json_write_string(fp, paper_claims_path);
    }
    fputs("}}\n", fp);
}

static int nn_gd_health_write_report(NNGDHealthExperimentList *experiments,
                                     const char *paper_claims_path,
                                     const char *output_path) {
    FILE *fp = nn_gd_ngd_open_write(output_path);
    size_t i;
    size_t pass_count = 0U;
    size_t warn_count = 0U;
    size_t fail_count = 0U;
    NNGDStringList failing = {0};
    NNGDStringList warning = {0};
    NNGDHealthCounterList all_triage_kinds = {0};
    NNGDHealthCounterList all_triage_priorities = {0};
    if (fp == NULL) {
        return -1;
    }
    for (i = 0U; i < experiments->count; ++i) {
        const char *status = nn_gd_health_status(&experiments->items[i]);
        size_t j;
        if (strcmp(status, "fail") == 0) {
            ++fail_count;
            if (nn_gd_string_list_add_unique(&failing,
                                             experiments->items[i].name) != 0) {
                fclose(fp);
                return -1;
            }
        } else if (strcmp(status, "warn") == 0) {
            ++warn_count;
            if (nn_gd_string_list_add_unique(&warning,
                                             experiments->items[i].name) != 0) {
                fclose(fp);
                return -1;
            }
        } else {
            ++pass_count;
        }
        for (j = 0U; j < experiments->items[i].triage_kind_counts.count; ++j) {
            if (nn_gd_health_counter_add(
                    &all_triage_kinds,
                    experiments->items[i].triage_kind_counts.items[j].key,
                    experiments->items[i].triage_kind_counts.items[j].count) != 0) {
                fclose(fp);
                return -1;
            }
        }
        for (j = 0U; j < experiments->items[i].triage_priority_counts.count; ++j) {
            if (nn_gd_health_counter_add(
                    &all_triage_priorities,
                    experiments->items[i].triage_priority_counts.items[j].key,
                    experiments->items[i].triage_priority_counts.items[j].count) != 0) {
                fclose(fp);
                return -1;
            }
        }
        nn_gd_health_write_experiment(fp, &experiments->items[i], paper_claims_path);
    }
    nn_gd_string_list_sort(&failing);
    nn_gd_string_list_sort(&warning);
    fputs("{\"schema\":\"crlp.experiment_health_summary.v1\",", fp);
    fprintf(fp, "\"experiments\":%zu,\"status_counts\":{", experiments->count);
    fprintf(fp, "\"fail\":%zu,\"pass\":%zu,\"warn\":%zu},",
            fail_count,
            pass_count,
            warn_count);
    fputs("\"triage_priority_counts\":", fp);
    nn_gd_health_write_counter_object(fp, &all_triage_priorities);
    fputs(",\"triage_kind_counts\":", fp);
    nn_gd_health_write_counter_object(fp, &all_triage_kinds);
    fputs(",\"failing_experiments\":", fp);
    nn_gd_json_write_string_list(fp, &failing);
    fputs(",\"warning_experiments\":", fp);
    nn_gd_json_write_string_list(fp, &warning);
    fputs("}\n", fp);
    if (ferror(fp)) {
        fclose(fp);
        nn_gd_string_list_free(&failing);
        nn_gd_string_list_free(&warning);
        nn_gd_health_counter_free(&all_triage_kinds);
        nn_gd_health_counter_free(&all_triage_priorities);
        return -1;
    }
    fclose(fp);
    nn_gd_string_list_free(&failing);
    nn_gd_string_list_free(&warning);
    nn_gd_health_counter_free(&all_triage_kinds);
    nn_gd_health_counter_free(&all_triage_priorities);
    return 0;
}

static int nn_gd_health_counter_merge(NNGDHealthCounterList *dst,
                                      const NNGDHealthCounterList *src) {
    size_t i;
    if (dst == NULL || src == NULL) {
        return 0;
    }
    for (i = 0U; i < src->count; ++i) {
        if (nn_gd_health_counter_add(dst,
                                     src->items[i].key,
                                     src->items[i].count) != 0) {
            return -1;
        }
    }
    return 0;
}

static const char *nn_gd_failure_class_for_triage_kind(const char *kind) {
    if (kind == NULL || kind[0] == '\0') {
        return "diagnostic_unclassified";
    }
    if (strncmp(kind, "tiny_net_", 9U) == 0) {
        return nn_gd_failure_class_for_tiny_diagnosis(kind + 9U);
    }
    if (strstr(kind, "engine") != NULL) {
        return "engine_surface_gap";
    }
    if (strstr(kind, "visual") != NULL ||
        strstr(kind, "arc_") != NULL ||
        strstr(kind, "coverage") != NULL) {
        return "coverage_or_representation_gap";
    }
    if (strstr(kind, "coherence") != NULL ||
        strstr(kind, "ranker") != NULL ||
        strstr(kind, "corruption") != NULL) {
        return "representation_or_label_gap";
    }
    if (strstr(kind, "paper_claim") != NULL ||
        strcmp(kind, "underpowered_claim") == 0) {
        return "paper_claim_gap";
    }
    if (strstr(kind, "optional_gate") != NULL ||
        strstr(kind, "protocol") != NULL) {
        return "protocol_or_sampling_gap";
    }
    if (strcmp(kind, "instrumentation_gap") == 0) {
        return "instrumentation_gap";
    }
    return "diagnostic_or_protocol_gap";
}

static const char *nn_gd_failure_class_for_required_gate(const char *gate) {
    if (gate == NULL || gate[0] == '\0') {
        return "required_gate_failure";
    }
    if (strstr(gate, "dataset_engine") != NULL ||
        strstr(gate, "engine") != NULL) {
        return "engine_surface_gap";
    }
    if (strstr(gate, "protocol") != NULL ||
        strstr(gate, "signal") != NULL) {
        return "protocol_or_sampling_gap";
    }
    if (strstr(gate, "coverage") != NULL ||
        strstr(gate, "visual") != NULL) {
        return "coverage_or_representation_gap";
    }
    return "required_gate_failure";
}

static const char *nn_gd_action_for_triage_kind(const char *kind) {
    if (kind == NULL || kind[0] == '\0') {
        return "inspect_unclassified_diagnostic";
    }
    if (strstr(kind, "engine") != NULL) {
        return "repair_dataset_engine_surface";
    }
    if (strstr(kind, "visual") != NULL ||
        strstr(kind, "arc_") != NULL ||
        strstr(kind, "coverage") != NULL) {
        return "inspect_coverage_and_representation_holes";
    }
    if (strstr(kind, "coherence") != NULL ||
        strstr(kind, "ranker") != NULL ||
        strstr(kind, "corruption") != NULL) {
        return "inspect_feature_or_label_encoding";
    }
    if (strstr(kind, "paper_claim") != NULL ||
        strcmp(kind, "underpowered_claim") == 0) {
        return "collect_or_refresh_claim_evidence";
    }
    if (strstr(kind, "protocol") != NULL ||
        strstr(kind, "optional_gate") != NULL) {
        return "add_protocol_surface_or_sampling_ablation";
    }
    return "inspect_source_evidence";
}

static int nn_gd_health_build_failure_atlas_counts(
    const NNGDHealthExperiment *experiment,
    NNGDHealthCounterList *classes,
    NNGDHealthCounterList *next_steps) {
    size_t i;
    size_t claim_underpowered;
    size_t claim_missing;
    size_t claim_stale;
    size_t claim_contradicted;
    if (nn_gd_health_counter_merge(
            classes,
            &experiment->diagnostic_failure_class_counts) != 0 ||
        nn_gd_health_counter_merge(
            next_steps,
            &experiment->diagnostic_next_step_counts) != 0) {
        return -1;
    }
    for (i = 0U; i < experiment->triage_kind_counts.count; ++i) {
        const char *kind = experiment->triage_kind_counts.items[i].key;
        size_t count = experiment->triage_kind_counts.items[i].count;
        if (strncmp(kind, "tiny_net_", 9U) == 0 ||
            strcmp(nn_gd_health_priority_for_kind(kind), "INFO") == 0) {
            continue;
        }
        if (nn_gd_health_counter_add(
                classes,
                nn_gd_failure_class_for_triage_kind(kind),
                count) != 0 ||
            nn_gd_health_counter_add(
                next_steps,
                nn_gd_action_for_triage_kind(kind),
                count) != 0) {
            return -1;
        }
    }
    for (i = 0U; i < experiment->coverage_hole_counts.count; ++i) {
        size_t count = experiment->coverage_hole_counts.items[i].count;
        if (nn_gd_health_counter_add(classes, "coverage_gap", count) != 0 ||
            nn_gd_health_counter_add(next_steps,
                                     "inspect_coverage_holes",
                                     count) != 0) {
            return -1;
        }
    }
    for (i = 0U; i < experiment->failing_required_gates.count; ++i) {
        if (nn_gd_health_counter_add(
                classes,
                nn_gd_failure_class_for_required_gate(
                    experiment->failing_required_gates.items[i]),
                1U) != 0 ||
            nn_gd_health_counter_add(next_steps,
                                     "restore_required_health_gate",
                                     1U) != 0) {
            return -1;
        }
    }
    if (experiment->failing_optional_gates.count > 0U &&
        strcmp(nn_gd_health_status(experiment), "pass") != 0) {
        if (nn_gd_health_counter_add(classes,
                                     "optional_evidence_gap",
                                     experiment->failing_optional_gates.count) != 0 ||
            nn_gd_health_counter_add(next_steps,
                                     "decide_or_repair_optional_gate",
                                     experiment->failing_optional_gates.count) != 0) {
            return -1;
        }
    }
    if (experiment->numeric_rejected_rows > 0U ||
        experiment->numeric_failed_statuses > 0U) {
        size_t count = experiment->numeric_rejected_rows +
                       experiment->numeric_failed_statuses;
        if (nn_gd_health_counter_add(classes,
                                     "numeric_certificate_gap",
                                     count) != 0 ||
            nn_gd_health_counter_add(next_steps,
                                     "inspect_numeric_certificate_rejections",
                                     count) != 0) {
            return -1;
        }
    }
    claim_underpowered =
        nn_gd_health_counter_get(&experiment->claim_status_counts, "underpowered");
    claim_missing =
        nn_gd_health_counter_get(&experiment->claim_status_counts, "missing");
    claim_stale =
        nn_gd_health_counter_get(&experiment->claim_status_counts, "stale");
    claim_contradicted =
        nn_gd_health_counter_get(&experiment->claim_status_counts, "contradicted");
    if (claim_underpowered + claim_missing + claim_stale > 0U) {
        if (nn_gd_health_counter_add(classes,
                                     "paper_claim_gap",
                                     claim_underpowered + claim_missing +
                                         claim_stale) != 0 ||
            nn_gd_health_counter_add(next_steps,
                                     "collect_or_refresh_claim_evidence",
                                     claim_underpowered + claim_missing +
                                         claim_stale) != 0) {
            return -1;
        }
    }
    if (claim_contradicted > 0U) {
        if (nn_gd_health_counter_add(classes,
                                     "paper_claim_contradiction",
                                     claim_contradicted) != 0 ||
            nn_gd_health_counter_add(next_steps,
                                     "revise_claim_or_rerun_evidence",
                                     claim_contradicted) != 0) {
            return -1;
        }
    }
    return 0;
}

static const char *nn_gd_health_counter_top_key(
    const NNGDHealthCounterList *list) {
    const char *best = NULL;
    size_t best_count = 0U;
    size_t i;
    if (list == NULL) {
        return NULL;
    }
    for (i = 0U; i < list->count; ++i) {
        if (list->items[i].count > best_count) {
            best = list->items[i].key;
            best_count = list->items[i].count;
        }
    }
    return best;
}

static int nn_gd_health_write_failure_atlas_jsonl(
    NNGDHealthExperimentList *experiments,
    const char *health_report_path,
    const char *failure_atlas_path) {
    FILE *fp;
    NNGDHealthCounterList all_classes = {0};
    NNGDHealthCounterList all_next_steps = {0};
    size_t i;
    if (failure_atlas_path == NULL || failure_atlas_path[0] == '\0') {
        return 0;
    }
    fp = nn_gd_ngd_open_write(failure_atlas_path);
    if (fp == NULL) {
        return -1;
    }
    for (i = 0U; i < experiments->count; ++i) {
        const NNGDHealthExperiment *experiment = &experiments->items[i];
        NNGDHealthCounterList classes = {0};
        NNGDHealthCounterList next_steps = {0};
        const char *top_failure_class;
        size_t claim_supported =
            nn_gd_health_counter_get(&experiment->claim_status_counts,
                                     "supported");
        size_t claim_contradicted =
            nn_gd_health_counter_get(&experiment->claim_status_counts,
                                     "contradicted");
        if (nn_gd_health_build_failure_atlas_counts(experiment,
                                                    &classes,
                                                    &next_steps) != 0 ||
            nn_gd_health_counter_merge(&all_classes, &classes) != 0 ||
            nn_gd_health_counter_merge(&all_next_steps, &next_steps) != 0) {
            nn_gd_health_counter_free(&classes);
            nn_gd_health_counter_free(&next_steps);
            fclose(fp);
            nn_gd_health_counter_free(&all_classes);
            nn_gd_health_counter_free(&all_next_steps);
            return -1;
        }
        top_failure_class = nn_gd_health_counter_top_key(&classes);
        fputs("{\"schema\":\"crlp.experiment_failure_atlas.v1\",", fp);
        fputs("\"experiment\":", fp);
        nn_gd_json_write_string(fp, experiment->name);
        fputs(",\"health_status\":", fp);
        nn_gd_json_write_string(fp, nn_gd_health_status(experiment));
        fputs(",\"top_failure_class\":", fp);
        if (top_failure_class == NULL) {
            fputs("null", fp);
        } else {
            nn_gd_json_write_string(fp, top_failure_class);
        }
        fputs(",\"failure_class_counts\":", fp);
        nn_gd_health_write_counter_object(fp, &classes);
        fputs(",\"next_step_counts\":", fp);
        nn_gd_health_write_counter_object(fp, &next_steps);
        fputs(",\"diagnostic_failure_class_counts\":", fp);
        nn_gd_health_write_counter_object(
            fp,
            &experiment->diagnostic_failure_class_counts);
        fputs(",\"triage_kind_counts\":", fp);
        nn_gd_health_write_counter_object(fp, &experiment->triage_kind_counts);
        fputs(",\"coverage_hole_counts\":", fp);
        nn_gd_health_write_counter_object(fp, &experiment->coverage_hole_counts);
        fprintf(fp,
                ",\"rows\":{\"coverage\":%zu,\"evidence\":%zu,"
                "\"diagnostic\":%zu,\"verifier_certificate\":%zu,"
                "\"numeric_certificate\":%zu}",
                experiment->coverage_rows,
                experiment->evidence_rows,
                experiment->diagnostic_rows,
                experiment->verifier_certificate_rows,
                experiment->numeric_certificate_rows);
        fputs(",\"paper_claim_status_counts\":", fp);
        nn_gd_health_write_counter_object(fp, &experiment->claim_status_counts);
        fputs(",\"supports_paper_claim\":", fp);
        fputs(claim_supported > 0U && claim_contradicted == 0U ? "true" : "false",
              fp);
        fputs(",\"promotion_boundary\":", fp);
        nn_gd_json_write_string(
            fp,
            "diagnostic ranking only; paper promotion still requires verifier/certificate support");
        fputs(",\"sources\":{\"experiment_health\":", fp);
        if (health_report_path == NULL || health_report_path[0] == '\0') {
            fputs("null", fp);
        } else {
            nn_gd_json_write_string(fp, health_report_path);
        }
        fputs(",\"diagnostics\":", fp);
        nn_gd_json_write_string_list(fp, &experiment->diagnostic_sources);
        fputs(",\"evidence\":", fp);
        nn_gd_json_write_string_list(fp, &experiment->evidence_sources);
        fputs("}}\n", fp);
        nn_gd_health_counter_free(&classes);
        nn_gd_health_counter_free(&next_steps);
    }
    fputs("{\"schema\":\"crlp.experiment_failure_atlas_summary.v1\",", fp);
    fprintf(fp, "\"experiments\":%zu,", experiments->count);
    fputs("\"failure_class_counts\":", fp);
    nn_gd_health_write_counter_object(fp, &all_classes);
    fputs(",\"next_step_counts\":", fp);
    nn_gd_health_write_counter_object(fp, &all_next_steps);
    fputs(",\"promotion_boundary\":", fp);
    nn_gd_json_write_string(
        fp,
        "summary counts rank diagnostic work; claim status remains owned by generated claims and verifiers");
    fputs("}\n", fp);
    if (ferror(fp)) {
        fclose(fp);
        nn_gd_health_counter_free(&all_classes);
        nn_gd_health_counter_free(&all_next_steps);
        return -1;
    }
    fclose(fp);
    nn_gd_health_counter_free(&all_classes);
    nn_gd_health_counter_free(&all_next_steps);
    return 0;
}

static void nn_gd_health_write_markdown_counter(FILE *fp,
                                                const NNGDHealthCounterList *list) {
    size_t i;
    if (list == NULL || list->count == 0U) {
        fputs("-", fp);
        return;
    }
    for (i = 0U; i < list->count; ++i) {
        if (i > 0U) {
            fputs("<br>", fp);
        }
        fprintf(fp, "%s=%zu", list->items[i].key, list->items[i].count);
    }
}

static int nn_gd_health_write_failure_atlas_markdown(
    NNGDHealthExperimentList *experiments,
    const char *failure_atlas_jsonl_path,
    const char *markdown_path) {
    FILE *fp;
    size_t i;
    if (markdown_path == NULL || markdown_path[0] == '\0') {
        return 0;
    }
    fp = nn_gd_ngd_open_write(markdown_path);
    if (fp == NULL) {
        return -1;
    }
    fputs("# Experiment Failure Atlas\n\n", fp);
    fputs("Diagnostic ranking surface. These rows prioritize feature, protocol, "
          "capacity, and evidence work; they do not promote paper claims without "
          "the existing verifier/certificate gates.\n\n",
          fp);
    fputs("JSONL: `", fp);
    if (failure_atlas_jsonl_path != NULL) {
        fputs(failure_atlas_jsonl_path, fp);
    }
    fputs("`\n\n", fp);
    fputs("| Experiment | Health | Top failure class | Failure classes | Next steps |\n",
          fp);
    fputs("| --- | --- | --- | --- | --- |\n", fp);
    for (i = 0U; i < experiments->count; ++i) {
        const NNGDHealthExperiment *experiment = &experiments->items[i];
        NNGDHealthCounterList classes = {0};
        NNGDHealthCounterList next_steps = {0};
        const char *top_failure_class;
        if (nn_gd_health_build_failure_atlas_counts(experiment,
                                                    &classes,
                                                    &next_steps) != 0) {
            nn_gd_health_counter_free(&classes);
            nn_gd_health_counter_free(&next_steps);
            fclose(fp);
            return -1;
        }
        top_failure_class = nn_gd_health_counter_top_key(&classes);
        fprintf(fp,
                "| `%s` | `%s` | `%s` | ",
                experiment->name,
                nn_gd_health_status(experiment),
                top_failure_class == NULL ? "-" : top_failure_class);
        nn_gd_health_write_markdown_counter(fp, &classes);
        fputs(" | ", fp);
        nn_gd_health_write_markdown_counter(fp, &next_steps);
        fputs(" |\n", fp);
        nn_gd_health_counter_free(&classes);
        nn_gd_health_counter_free(&next_steps);
    }
    if (ferror(fp)) {
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
}

static int nn_gd_health_work_items_reserve(NNGDHealthWorkItemList *list,
                                           size_t needed) {
    size_t next;
    NNGDHealthWorkItem *items;
    if (needed <= list->capacity) {
        return 0;
    }
    next = list->capacity == 0U ? 16U : list->capacity * 2U;
    while (next < needed) {
        next *= 2U;
    }
    items = (NNGDHealthWorkItem *)realloc(list->items, next * sizeof(*items));
    if (items == NULL) {
        return -1;
    }
    list->items = items;
    list->capacity = next;
    return 0;
}

static void nn_gd_health_work_items_free(NNGDHealthWorkItemList *list) {
    size_t i;
    if (list == NULL) {
        return;
    }
    for (i = 0U; i < list->count; ++i) {
        free(list->items[i].failure_class);
        free(list->items[i].next_step);
    }
    free(list->items);
    memset(list, 0, sizeof(*list));
}

static int nn_gd_health_work_priority_bonus(const char *value) {
    if (value == NULL) {
        return 0;
    }
    if (strcmp(value, "paper_claim_contradiction") == 0 ||
        strcmp(value, "revise_claim_or_rerun_evidence") == 0) {
        return 350;
    }
    if (strcmp(value, "required_gate_failure") == 0 ||
        strcmp(value, "restore_required_health_gate") == 0 ||
        strcmp(value, "coverage_or_representation_gap") == 0 ||
        strcmp(value, "inspect_coverage_and_representation_holes") == 0) {
        return 250;
    }
    if (strcmp(value, "coverage_gap") == 0 ||
        strcmp(value, "inspect_coverage_holes") == 0 ||
        strcmp(value, "paper_claim_gap") == 0 ||
        strcmp(value, "collect_or_refresh_claim_evidence") == 0) {
        return 180;
    }
    if (strcmp(value, "optimizer_unsettled") == 0 ||
        strcmp(value, "extend_or_stabilize_training") == 0 ||
        strcmp(value, "capacity_unsettled") == 0 ||
        strcmp(value, "run_stronger_baseline_ladder") == 0 ||
        strcmp(value, "seed_variance") == 0 ||
        strcmp(value, "increase_restarts_before_width_claim") == 0) {
        return 120;
    }
    return 0;
}

static const char *nn_gd_health_failure_class_for_next_step(
    const char *next_step) {
    if (next_step == NULL) {
        return NULL;
    }
    if (strcmp(next_step, "revise_claim_or_rerun_evidence") == 0) {
        return "paper_claim_contradiction";
    }
    if (strcmp(next_step, "collect_or_refresh_claim_evidence") == 0) {
        return "paper_claim_gap";
    }
    if (strcmp(next_step, "restore_required_health_gate") == 0) {
        return "required_gate_failure";
    }
    if (strcmp(next_step, "inspect_coverage_and_representation_holes") == 0) {
        return "coverage_or_representation_gap";
    }
    if (strcmp(next_step, "inspect_coverage_holes") == 0) {
        return "coverage_gap";
    }
    if (strcmp(next_step, "decide_or_repair_optional_gate") == 0) {
        return "optional_evidence_gap";
    }
    if (strcmp(next_step, "inspect_feature_or_label_encoding") == 0) {
        return "representation_or_label_gap";
    }
    if (strcmp(next_step, "add_protocol_surface_or_sampling_ablation") == 0 ||
        strcmp(next_step, "try_protocol_artifacts_or_sampling") == 0) {
        return "protocol_or_sampling_gap";
    }
    if (strcmp(next_step, "extend_or_stabilize_training") == 0) {
        return "optimizer_unsettled";
    }
    if (strcmp(next_step, "run_stronger_baseline_ladder") == 0) {
        return "capacity_unsettled";
    }
    if (strcmp(next_step, "increase_restarts_before_width_claim") == 0) {
        return "seed_variance";
    }
    return NULL;
}

static const char *nn_gd_health_work_priority_for_score(int score) {
    if (score >= 900) {
        return "P0";
    }
    if (score >= 750) {
        return "P1";
    }
    if (score >= 600) {
        return "P2";
    }
    return "P3";
}

static int nn_gd_health_work_item_compare(const void *a, const void *b) {
    const NNGDHealthWorkItem *ia = (const NNGDHealthWorkItem *)a;
    const NNGDHealthWorkItem *ib = (const NNGDHealthWorkItem *)b;
    int cmp;
    if (ia->priority_score != ib->priority_score) {
        return ib->priority_score - ia->priority_score;
    }
    cmp = strcmp(ia->priority, ib->priority);
    if (cmp != 0) {
        return cmp;
    }
    cmp = strcmp(ia->experiment->name, ib->experiment->name);
    if (cmp != 0) {
        return cmp;
    }
    return strcmp(ia->next_step, ib->next_step);
}

static int nn_gd_health_work_items_add(NNGDHealthWorkItemList *list,
                                       const NNGDHealthExperiment *experiment,
                                       const char *failure_class,
                                       size_t failure_class_count,
                                       const char *next_step,
                                       size_t next_step_count) {
    NNGDHealthWorkItem *item;
    int score;
    if (list == NULL || experiment == NULL || failure_class == NULL ||
        next_step == NULL) {
        return -1;
    }
    if (nn_gd_health_work_items_reserve(list, list->count + 1U) != 0) {
        return -1;
    }
    item = &list->items[list->count];
    memset(item, 0, sizeof(*item));
    item->experiment = experiment;
    item->failure_class = nn_gd_strdup(failure_class);
    item->next_step = nn_gd_strdup(next_step);
    if (item->failure_class == NULL || item->next_step == NULL) {
        free(item->failure_class);
        free(item->next_step);
        memset(item, 0, sizeof(*item));
        return -1;
    }
    item->failure_class_count = failure_class_count;
    item->next_step_count = next_step_count;
    score = 10 * (int)next_step_count + 6 * (int)failure_class_count;
    if (strcmp(nn_gd_health_status(experiment), "fail") == 0) {
        score += 500;
    } else if (strcmp(nn_gd_health_status(experiment), "warn") == 0) {
        score += 300;
    }
    score += nn_gd_health_work_priority_bonus(failure_class);
    score += nn_gd_health_work_priority_bonus(next_step);
    item->priority_score = score;
    item->priority = nn_gd_health_work_priority_for_score(score);
    ++list->count;
    return 0;
}

static int nn_gd_health_build_work_items(NNGDHealthExperimentList *experiments,
                                         NNGDHealthWorkItemList *items) {
    size_t i;
    if (experiments == NULL || items == NULL) {
        return -1;
    }
    for (i = 0U; i < experiments->count; ++i) {
        const NNGDHealthExperiment *experiment = &experiments->items[i];
        NNGDHealthCounterList classes = {0};
        NNGDHealthCounterList next_steps = {0};
        const char *top_failure_class;
        size_t top_failure_count;
        size_t j;
        if (nn_gd_health_build_failure_atlas_counts(experiment,
                                                    &classes,
                                                    &next_steps) != 0) {
            nn_gd_health_counter_free(&classes);
            nn_gd_health_counter_free(&next_steps);
            return -1;
        }
        top_failure_class = nn_gd_health_counter_top_key(&classes);
        top_failure_count =
            top_failure_class == NULL
                ? 0U
                : nn_gd_health_counter_get(&classes, top_failure_class);
        if (top_failure_class != NULL) {
            for (j = 0U; j < next_steps.count; ++j) {
                const char *step_failure_class =
                    nn_gd_health_failure_class_for_next_step(
                        next_steps.items[j].key);
                size_t step_failure_count;
                if (step_failure_class == NULL ||
                    nn_gd_health_counter_get(&classes,
                                             step_failure_class) == 0U) {
                    step_failure_class = top_failure_class;
                }
                step_failure_count =
                    nn_gd_health_counter_get(&classes, step_failure_class);
                if (nn_gd_health_work_items_add(items,
                                                experiment,
                                                step_failure_class,
                                                step_failure_count == 0U
                                                    ? top_failure_count
                                                    : step_failure_count,
                                                next_steps.items[j].key,
                                                next_steps.items[j].count) != 0) {
                    nn_gd_health_counter_free(&classes);
                    nn_gd_health_counter_free(&next_steps);
                    return -1;
                }
            }
        }
        nn_gd_health_counter_free(&classes);
        nn_gd_health_counter_free(&next_steps);
    }
    qsort(items->items,
          items->count,
          sizeof(items->items[0]),
          nn_gd_health_work_item_compare);
    return 0;
}

static int nn_gd_health_write_failure_work_queue_jsonl(
    NNGDHealthExperimentList *experiments,
    const char *failure_atlas_jsonl_path,
    const char *work_queue_path) {
    FILE *fp;
    NNGDHealthWorkItemList items = {0};
    size_t i;
    if (work_queue_path == NULL || work_queue_path[0] == '\0') {
        return 0;
    }
    if (nn_gd_health_build_work_items(experiments, &items) != 0) {
        nn_gd_health_work_items_free(&items);
        return -1;
    }
    fp = nn_gd_ngd_open_write(work_queue_path);
    if (fp == NULL) {
        nn_gd_health_work_items_free(&items);
        return -1;
    }
    for (i = 0U; i < items.count; ++i) {
        const NNGDHealthWorkItem *item = &items.items[i];
        fputs("{\"schema\":\"crlp.experiment_work_item.v1\",", fp);
        fprintf(fp, "\"rank\":%zu,", i + 1U);
        fputs("\"priority\":", fp);
        nn_gd_json_write_string(fp, item->priority);
        fprintf(fp, ",\"priority_score\":%d", item->priority_score);
        fputs(",\"experiment\":", fp);
        nn_gd_json_write_string(fp, item->experiment->name);
        fputs(",\"health_status\":", fp);
        nn_gd_json_write_string(fp, nn_gd_health_status(item->experiment));
        fputs(",\"failure_class\":", fp);
        nn_gd_json_write_string(fp, item->failure_class);
        fprintf(fp, ",\"failure_class_count\":%zu", item->failure_class_count);
        fputs(",\"next_step\":", fp);
        nn_gd_json_write_string(fp, item->next_step);
        fprintf(fp, ",\"next_step_count\":%zu", item->next_step_count);
        fputs(",\"source\":\"c_experiment_failure_atlas\",", fp);
        fputs("\"source_atlas\":", fp);
        if (failure_atlas_jsonl_path == NULL ||
            failure_atlas_jsonl_path[0] == '\0') {
            fputs("null", fp);
        } else {
            nn_gd_json_write_string(fp, failure_atlas_jsonl_path);
        }
        fputs(",\"promotion_boundary\":", fp);
        nn_gd_json_write_string(
            fp,
            "work item priority is diagnostic; claim promotion requires verifier/certificate support");
        fputs("}\n", fp);
    }
    fputs("{\"schema\":\"crlp.experiment_work_queue_summary.v1\",", fp);
    fprintf(fp, "\"work_items\":%zu", items.count);
    fputs(",\"source_atlas\":", fp);
    if (failure_atlas_jsonl_path == NULL ||
        failure_atlas_jsonl_path[0] == '\0') {
        fputs("null", fp);
    } else {
        nn_gd_json_write_string(fp, failure_atlas_jsonl_path);
    }
    fputs("}\n", fp);
    if (ferror(fp)) {
        fclose(fp);
        nn_gd_health_work_items_free(&items);
        return -1;
    }
    fclose(fp);
    nn_gd_health_work_items_free(&items);
    return 0;
}

static int nn_gd_health_write_failure_work_queue_markdown(
    NNGDHealthExperimentList *experiments,
    const char *work_queue_jsonl_path,
    const char *markdown_path) {
    FILE *fp;
    NNGDHealthWorkItemList items = {0};
    size_t i;
    if (markdown_path == NULL || markdown_path[0] == '\0') {
        return 0;
    }
    if (nn_gd_health_build_work_items(experiments, &items) != 0) {
        nn_gd_health_work_items_free(&items);
        return -1;
    }
    fp = nn_gd_ngd_open_write(markdown_path);
    if (fp == NULL) {
        nn_gd_health_work_items_free(&items);
        return -1;
    }
    fputs("# Experiment Work Queue\n\n", fp);
    fputs("Prioritized diagnostic queue generated from the experiment failure "
          "atlas. These rows rank follow-up work only; they do not change claim "
          "support.\n\n",
          fp);
    fputs("JSONL: `", fp);
    if (work_queue_jsonl_path != NULL) {
        fputs(work_queue_jsonl_path, fp);
    }
    fputs("`\n\n", fp);
    fputs("| Rank | Priority | Experiment | Failure class | Next step | Count |\n",
          fp);
    fputs("| ---: | --- | --- | --- | --- | ---: |\n", fp);
    for (i = 0U; i < items.count; ++i) {
        const NNGDHealthWorkItem *item = &items.items[i];
        fprintf(fp,
                "| %zu | `%s` | `%s` | `%s` | `%s` | %zu |\n",
                i + 1U,
                item->priority,
                item->experiment->name,
                item->failure_class,
                item->next_step,
                item->next_step_count);
    }
    if (ferror(fp)) {
        fclose(fp);
        nn_gd_health_work_items_free(&items);
        return -1;
    }
    fclose(fp);
    nn_gd_health_work_items_free(&items);
    return 0;
}

int nn_grammar_discovery_write_experiment_health(
    const char *coverage_path,
    const char *numeric_summary_path,
    const char *numeric_report_path,
    const char *paper_claims_path,
    const char *signal_flight_summary_path,
    const char *signal_flight_rollout_path,
    const char *signal_live_active_path,
    const char *signal_shadow_replay_path,
    const char *neural_grammar_summary_path,
    const char *protocol_spine_path,
    const char *operator_promotions_path,
    const char *engine_evidence_path,
    const char *tiny_net_diagnostics_path,
    const char *failure_atlas_path,
    const char *failure_atlas_markdown_path,
    const char *failure_work_queue_path,
    const char *failure_work_queue_markdown_path,
    const char *output_path) {
    NNGDHealthExperimentList experiments = {0};
    int status = -1;
    (void)numeric_report_path;
    if (coverage_path == NULL || coverage_path[0] == '\0' ||
        output_path == NULL || output_path[0] == '\0') {
        return -1;
    }
    if (nn_gd_health_read_coverage(&experiments, coverage_path) != 0 ||
        nn_gd_health_read_numeric_summary(&experiments, numeric_summary_path) != 0 ||
        nn_gd_health_read_claims(&experiments, paper_claims_path) != 0 ||
        nn_gd_health_read_signal(&experiments,
                                 signal_flight_summary_path,
                                 signal_flight_rollout_path,
                                 signal_live_active_path,
                                 signal_shadow_replay_path) != 0 ||
        nn_gd_health_read_neural_grammar(&experiments,
                                         neural_grammar_summary_path) != 0 ||
        nn_gd_health_read_protocol_spine(&experiments, protocol_spine_path) != 0 ||
        nn_gd_health_read_operator_promotions(&experiments,
                                              operator_promotions_path) != 0 ||
        nn_gd_health_read_engine_evidence(&experiments, engine_evidence_path) != 0 ||
        nn_gd_health_read_tiny_net_diagnostics(&experiments,
                                               tiny_net_diagnostics_path) != 0 ||
        nn_gd_health_finalize(&experiments) != 0 ||
        nn_gd_health_write_report(&experiments,
                                  paper_claims_path,
                                  output_path) != 0 ||
        nn_gd_health_write_failure_atlas_jsonl(&experiments,
                                               output_path,
                                               failure_atlas_path) != 0 ||
        nn_gd_health_write_failure_atlas_markdown(
            &experiments,
            failure_atlas_path,
            failure_atlas_markdown_path) != 0 ||
        nn_gd_health_write_failure_work_queue_jsonl(
            &experiments,
            failure_atlas_path,
            failure_work_queue_path) != 0 ||
        nn_gd_health_write_failure_work_queue_markdown(
            &experiments,
            failure_work_queue_path,
            failure_work_queue_markdown_path) != 0) {
        goto done;
    }
    printf("wrote %zu experiment health rows to %s\n",
           experiments.count,
           output_path);
    if (failure_atlas_path != NULL && failure_atlas_path[0] != '\0') {
        printf("wrote experiment failure atlas to %s\n", failure_atlas_path);
    }
    if (failure_work_queue_path != NULL &&
        failure_work_queue_path[0] != '\0') {
        printf("wrote experiment work queue to %s\n", failure_work_queue_path);
    }
    status = 0;

done:
    nn_gd_health_experiment_list_free(&experiments);
    return status;
}
