static int arc_program_learned_save_hypothesis_library_prior(const ARCProgramLearnedState *state, const char *path) {
    if (state == NULL || path == NULL || path[0] == '\0') return 0;
    FILE *fp = fopen(path, "w");
    if (fp == NULL) { fprintf(stderr, "failed to open hypothesis library prior for writing: %s\n", path); return 0; }
    fprintf(fp, "schema_version\t1\n");
    size_t bc = 0; const ARCHypothesisTemplateBinding *bindings = arc_hypothesis_template_registry(&bc);
    for (size_t i = 0; i < bc; ++i) {
        uint32_t tid = bindings[i].template_id;
        size_t proposals = 0, certified = 0;
        arc_hypothesis_template_prior_counts(state, tid, &proposals, &certified);
        if (proposals > 0) fprintf(fp, "crlp.hypothesis_library.template_prior.v1\t%u\t-\t-\t-\t-\t-\t-\t%zu\t%zu\t-\t-\t1.0\n", tid, proposals, certified);
    }
    fclose(fp); fprintf(stderr, "saved hypothesis library prior: %s\n", path);
    return 1;
}
static int arc_program_learned_load_arc_generative_prior(
    ARCProgramLearnedState *state,
    const char *path)
{
    if (state == NULL || path == NULL || path[0] == '\0') {
        return 0;
    }
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "failed to open ARC generative prior: %s\n", path);
        return 0;
    }
    if (state->generative_priors == NULL) {
        state->generative_prior_capacity = ARC_GENERATIVE_PRIOR_CAPACITY;
        state->generative_priors =
            calloc(state->generative_prior_capacity, sizeof(*state->generative_priors));
        if (state->generative_priors == NULL) {
            state->generative_prior_capacity = 0;
            fclose(fp);
            return 0;
        }
    }
    state->generative_prior_count = 0;
    char line[2048];
    size_t line_no = 0;
    while (fgets(line, sizeof(line), fp) != NULL) {
        ++line_no;
        if (line[0] == '\0' || line[0] == '\n' || line[0] == '#') {
            continue;
        }
        if (strncmp(line, "schema_version\t", strlen("schema_version\t")) == 0) {
            continue;
        }
        char *fields[19];
        size_t field_count =
            arc_split_tab_fields(line, fields, sizeof(fields) / sizeof(fields[0]));
        if ((field_count != 15 && field_count != 19) ||
            strcmp(fields[0], "crlp.arc_generative_prior.v1") != 0) {
            fprintf(stderr,
                    "invalid ARC generative prior row at %s:%zu\n",
                    path,
                    line_no);
            fclose(fp);
            return 0;
        }
        if (state->generative_prior_count >= state->generative_prior_capacity) {
            continue;
        }
        size_t op_id = 0;
        size_t input_width = 0;
        size_t input_height = 0;
        size_t output_width = 0;
        size_t output_height = 0;
        size_t shape_scale_x = 0;
        size_t shape_scale_y = 0;
        size_t input_component_count = 0;
        size_t output_component_count = 0;
        size_t accepted_count = 0;
        size_t rejected_count = 0;
        size_t near_miss_rejected_count = 0;
        size_t hard_rejected_count = 0;
        double mean_verifier_score = 0.0;
        double rejection_penalty = 0.0;
        double score = 0.0;
        if (!nn_parse_size_nonnegative_text(fields[1], &op_id) ||
            !nn_parse_size_nonnegative_text(fields[4], &input_width) ||
            !nn_parse_size_nonnegative_text(fields[5], &input_height) ||
            !nn_parse_size_nonnegative_text(fields[6], &output_width) ||
            !nn_parse_size_nonnegative_text(fields[7], &output_height) ||
            !nn_parse_size_nonnegative_text(fields[8], &shape_scale_x) ||
            !nn_parse_size_nonnegative_text(fields[9], &shape_scale_y) ||
            !nn_parse_size_nonnegative_text(fields[10], &input_component_count) ||
            !nn_parse_size_nonnegative_text(fields[11], &output_component_count) ||
            !nn_parse_size_nonnegative_text(fields[12], &accepted_count) ||
            !nn_parse_size_nonnegative_text(fields[13], &rejected_count) ||
            op_id > 255u) {
            fprintf(stderr,
                    "invalid ARC generative prior fields at %s:%zu\n",
                    path,
                    line_no);
            fclose(fp);
            return 0;
        }
        if (field_count == 19) {
            if (!nn_parse_size_nonnegative_text(fields[14], &near_miss_rejected_count) ||
                !nn_parse_size_nonnegative_text(fields[15], &hard_rejected_count) ||
                !nn_parse_double_text(fields[16], &mean_verifier_score) ||
                !nn_parse_double_text(fields[17], &rejection_penalty) ||
                !nn_parse_double_text(fields[18], &score)) {
                fprintf(stderr,
                        "invalid ARC generative prior rejection fields at %s:%zu\n",
                        path,
                        line_no);
                fclose(fp);
                return 0;
            }
        } else if (!nn_parse_double_text(fields[14], &score)) {
            fprintf(stderr,
                    "invalid ARC generative prior score at %s:%zu\n",
                    path,
                    line_no);
            fclose(fp);
            return 0;
        }
        ARCGenerativePriorRow *row =
            &state->generative_priors[state->generative_prior_count++];
        memset(row, 0, sizeof(*row));
        row->op = (int)op_id;
        row->input_width = input_width;
        row->input_height = input_height;
        row->output_width = output_width;
        row->output_height = output_height;
        row->shape_scale_x = shape_scale_x;
        row->shape_scale_y = shape_scale_y;
        row->input_component_count = input_component_count;
        row->output_component_count = output_component_count;
        row->accepted_count = accepted_count;
        row->rejected_count = rejected_count;
        row->near_miss_rejected_count = near_miss_rejected_count;
        row->hard_rejected_count = hard_rejected_count;
        row->mean_verifier_score = mean_verifier_score;
        row->rejection_penalty = rejection_penalty;
        row->score = score;
    }
    fclose(fp);
    return 1;
}

static void arc_hypothesis_template_prior_counts(const ARCProgramLearnedState *state,
                                                 uint32_t template_id,
                                                 size_t *proposal_count_out,
                                                 size_t *certified_count_out)
{
    if (proposal_count_out != NULL) {
        *proposal_count_out = 0;
    }
    if (certified_count_out != NULL) {
        *certified_count_out = 0;
    }
    if (state == NULL) {
        return;
    }
    size_t template_bucket = arc_hypothesis_template_bucket(template_id);
    if (proposal_count_out != NULL) {
        *proposal_count_out = state->hypothesis_template_proposal_count[template_bucket];
    }
    if (certified_count_out != NULL) {
        *certified_count_out = state->hypothesis_template_certified_count[template_bucket];
    }
}

static double arc_hypothesis_template_log_odds_for_id(const ARCProgramLearnedState *state,
                                                      uint32_t template_id,
                                                      size_t *proposal_count_out)
{
    size_t proposals = 0;
    size_t certified = 0;
    arc_hypothesis_template_prior_counts(state, template_id, &proposals, &certified);
    if (proposal_count_out != NULL) {
        *proposal_count_out = proposals;
    }
    if (proposals == 0) {
        return 0.0;
    }
    double positives = (double)certified + 1.0;
    double negatives = (double)(proposals - certified) + 1.0;
    return log(positives / negatives);
}

static double arc_hypothesis_template_log_odds(const ARCProgramLearnedState *state,
                                               const NNHypothesis *hypothesis,
                                               size_t *proposal_count_out)
{
    if (proposal_count_out != NULL) {
        *proposal_count_out = 0;
    }
    if (state == NULL || hypothesis == NULL || hypothesis->template_ref == NULL) {
        return 0.0;
    }
    return arc_hypothesis_template_log_odds_for_id(state,
                                                   hypothesis->template_ref->template_id,
                                                   proposal_count_out);
}

static size_t arc_hypothesis_template_probe_count(const ARCProgramLearnedState *state,
                                                  uint32_t template_id)
{
    if (state == NULL) {
        return 0;
    }
    size_t template_bucket = arc_hypothesis_template_bucket(template_id);
    return state->hypothesis_template_probe_count[template_bucket];
}

static double arc_hypothesis_template_probe_score(const ARCProgramLearnedState *state,
                                                  const NNHypothesis *hypothesis)
{
    if (state == NULL || hypothesis == NULL || hypothesis->template_ref == NULL) {
        return 0.0;
    }
    size_t count = arc_hypothesis_template_probe_count(
        state,
        hypothesis->template_ref->template_id);
    if (count == 0) {
        return 0.0;
    }
    return (double)ARC_HYPOTHESIS_TEMPLATE_PROBE_WEIGHT * log1p((double)count);
}

static void arc_hypothesis_prior_observe(ARCProgramLearnedState *state,
                                         const NNHypothesis *hypothesis,
                                         int certified)
{
    if (state == NULL || hypothesis == NULL || hypothesis->template_ref == NULL) {
        return;
    }
    size_t template_bucket =
        arc_hypothesis_template_bucket(hypothesis->template_ref->template_id);
    ++state->hypothesis_template_proposal_count[template_bucket];
    if (certified) {
        ++state->hypothesis_template_certified_count[template_bucket];
    }
    for (size_t i = 0; i < hypothesis->slot_count; ++i) {
        const NNHypothesisSlotValue *slot = &hypothesis->slots[i];
        if (!slot->bound) {
            continue;
        }
        size_t type_bucket = arc_hypothesis_slot_type_bucket(slot->type);
        size_t value_bucket = arc_hypothesis_slot_value_bucket(hypothesis, slot);
        ++state->hypothesis_slot_proposal_count[template_bucket]
                                                [type_bucket]
                                                [value_bucket];
        if (certified) {
            ++state->hypothesis_slot_certified_count[template_bucket]
                                                   [type_bucket]
                                                   [value_bucket];
        }
    }
}

static double arc_hypothesis_slot_log_odds(const ARCProgramLearnedState *state,
                                           const NNHypothesis *hypothesis,
                                           size_t *proposal_count_out)
{
    if (proposal_count_out != NULL) {
        *proposal_count_out = 0;
    }
    if (state == NULL || hypothesis == NULL || hypothesis->template_ref == NULL) {
        return 0.0;
    }
    size_t template_bucket =
        arc_hypothesis_template_bucket(hypothesis->template_ref->template_id);
    double score = 0.0;
    size_t proposal_count = 0;
    for (size_t i = 0; i < hypothesis->slot_count; ++i) {
        const NNHypothesisSlotValue *slot = &hypothesis->slots[i];
        if (!slot->bound) {
            continue;
        }
        size_t type_bucket = arc_hypothesis_slot_type_bucket(slot->type);
        size_t value_bucket = arc_hypothesis_slot_value_bucket(hypothesis, slot);
        size_t proposals = state->hypothesis_slot_proposal_count[template_bucket]
                                                            [type_bucket]
                                                            [value_bucket];
        size_t certified = state->hypothesis_slot_certified_count[template_bucket]
                                                            [type_bucket]
                                                            [value_bucket];
        if (proposals == 0) {
            continue;
        }
        proposal_count += proposals;
        double positives = (double)certified + 1.0;
        double negatives = (double)(proposals - certified) + 1.0;
        score += log(positives / negatives);
    }
    if (proposal_count_out != NULL) {
        *proposal_count_out = proposal_count;
    }
    return score;
}

static double arc_hypothesis_slot_probe_score(const ARCProgramLearnedState *state,
                                              const NNHypothesis *hypothesis,
                                              size_t *proposal_count_out)
{
    if (proposal_count_out != NULL) {
        *proposal_count_out = 0;
    }
    if (state == NULL || hypothesis == NULL || hypothesis->template_ref == NULL) {
        return 0.0;
    }
    size_t template_bucket =
        arc_hypothesis_template_bucket(hypothesis->template_ref->template_id);
    double score = 0.0;
    size_t proposal_count = 0;
    for (size_t i = 0; i < hypothesis->slot_count; ++i) {
        const NNHypothesisSlotValue *slot = &hypothesis->slots[i];
        if (!slot->bound) {
            continue;
        }
        size_t type_bucket = arc_hypothesis_slot_type_bucket(slot->type);
        size_t value_bucket = arc_hypothesis_slot_value_bucket(hypothesis, slot);
        size_t proposals = state->hypothesis_slot_probe_count[template_bucket]
                                                             [type_bucket]
                                                             [value_bucket];
        if (proposals == 0) {
            continue;
        }
        proposal_count += proposals;
        score += log1p((double)proposals);
    }
    if (proposal_count_out != NULL) {
        *proposal_count_out = proposal_count;
    }
    return (double)ARC_HYPOTHESIS_SLOT_PROBE_WEIGHT * score;
}

static double arc_hypothesis_slot_probe_score_for_template(
    const ARCProgramLearnedState *state,
    uint32_t template_id)
{
    if (state == NULL) {
        return 0.0;
    }
    size_t template_bucket = arc_hypothesis_template_bucket(template_id);
    double score = 0.0;
    for (size_t type_bucket = 0;
         type_bucket < ARC_HYPOTHESIS_SLOT_TYPE_BUCKETS;
         ++type_bucket) {
        for (size_t value_bucket = 0;
             value_bucket < ARC_HYPOTHESIS_SLOT_VALUE_BUCKETS;
             ++value_bucket) {
            size_t proposals =
                state->hypothesis_slot_probe_count[template_bucket]
                                                  [type_bucket]
                                                  [value_bucket];
            if (proposals > 0) {
                score += log1p((double)proposals);
            }
        }
    }
    return (double)ARC_HYPOTHESIS_SLOT_PROBE_WEIGHT * score;
}

static double arc_learned_family_score(const ARCProgramLearnedState *state,
                                       const ARCTransformCandidate *candidate)
{
    size_t family = arc_candidate_family(candidate);
    if (state == NULL) {
        return 0.0;
    }
    if (family >= sizeof(state->family_score) / sizeof(state->family_score[0])) {
        return 0.0;
    }
    if (arc_candidate_is_object_relation(candidate)) {
        return 20.0 * state->family_score[family];
    }
    if (arc_candidate_is_selected_object_program(candidate)) {
        return 10.0 * state->family_score[family];
    }
    return state->family_score[family];
}

static double arc_learned_relation_score(const ARCProgramLearnedState *state,
                                         const ARCTransformCandidate *candidate)
{
    size_t relation_kind = arc_object_relation_kind(candidate);
    if (state == NULL ||
        relation_kind == 0 ||
        relation_kind >= ARC_OBJECT_RELATION_KIND_COUNT ||
        candidate->input_color >= ARC_COLOR_COUNT ||
        candidate->output_color >= ARC_COLOR_COUNT ||
        arc_candidate_selector_kind(candidate) >= ARC_OBJECT_SELECTOR_KIND_COUNT ||
        arc_candidate_output_selector_kind(candidate) >= ARC_OBJECT_SELECTOR_KIND_COUNT ||
        arc_candidate_render_mode(candidate) >= ARC_OBJECT_RENDER_KIND_COUNT) {
        return 0.0;
    }
    return 2.0 * state->relation_score[relation_kind] +
           state->relation_color_score[relation_kind]
                                      [candidate->input_color]
                                      [candidate->output_color] +
           0.5 * state->relation_background_score[relation_kind]
                                                [candidate->background_color] +
           0.5 * state->relation_selector_score[relation_kind]
                                              [arc_candidate_selector_kind(candidate)] +
           0.25 * state->relation_output_selector_score[relation_kind]
                                                     [arc_candidate_output_selector_kind(candidate)] +
           0.25 * state->relation_render_score[relation_kind]
                                            [arc_candidate_render_mode(candidate)];
}

static double arc_learned_object_score(const ARCProgramLearnedState *state,
                                       const ARCTransformCandidate *candidate)
{
    if (state == NULL ||
        !arc_candidate_is_selected_object_program(candidate) ||
        arc_candidate_selector_kind(candidate) >= ARC_OBJECT_SELECTOR_KIND_COUNT ||
        candidate->input_color >= ARC_COLOR_COUNT ||
        candidate->output_color >= ARC_COLOR_COUNT ||
        arc_candidate_render_mode(candidate) >= ARC_OBJECT_RENDER_KIND_COUNT) {
        return 0.0;
    }
    return 40.0 * state->object_selector_score[arc_candidate_selector_kind(candidate)] +
           20.0 * state->object_color_score[candidate->input_color][candidate->output_color] +
           10.0 * state->object_render_score[arc_candidate_render_mode(candidate)];
}

static double arc_learned_program_score(ARCProgramLearnedState *state,
                                        const ARCTransformCandidate *candidate,
                                        const ARCProgramTaskFeatures *task_features)
{
    if (state == NULL || state->net == NULL || state->trained_tasks == 0) {
        return arc_learned_family_score(state, candidate) +
               arc_learned_relation_score(state, candidate) +
               arc_learned_object_score(state, candidate);
    }
    if (arc_candidate_is_composed(candidate)) {
        ARCTransformCandidate first;
        ARCTransformCandidate second;
        arc_candidate_from_step(&candidate->steps[0], &first);
        arc_candidate_from_step(&candidate->steps[1], &second);
        double first_score = arc_learned_program_score(state, &first, task_features);
        double second_score = arc_learned_program_score(state, &second, task_features);
        return -40.0 + 0.5 * (first_score + second_score);
    }
    nn_real features[ARC_PROGRAM_FEATURE_COUNT];
    nn_real output[1];
    if (!arc_encode_program_features(candidate,
                                     task_features,
                                     features,
                                     ARC_PROGRAM_FEATURE_COUNT)) {
        return arc_learned_family_score(state, candidate) +
               arc_learned_relation_score(state, candidate) +
               arc_learned_object_score(state, candidate);
    }
    nn_forward(state->net, features, output);
    return (double)output[0] +
           arc_learned_relation_score(state, candidate) +
           arc_learned_object_score(state, candidate);
}

static int arc_program_eval_is_exact_success(const ARCProgramEval *eval)
{
    return eval->train_consistent &&
           eval->test_supported &&
           eval->test_cells > 0 &&
           eval->test_correct_cells == eval->test_cells;
}

static double arc_trace_prior_log_odds(size_t seen, size_t success)
{
    if (seen == 0) {
        return 0.0;
    }
    if (success > seen) {
        success = seen;
    }
    return log1p((double)success) - 0.05 * log1p((double)(seen - success));
}

static double arc_trace_prior_slot_score(size_t seen, size_t success, double weight)
{
    return weight * arc_trace_prior_log_odds(seen, success);
}

static double arc_program_task_feature_similarity(const double *sum,
                                                  size_t count,
                                                  const ARCProgramTaskFeatures *features)
{
    if (sum == NULL || features == NULL || count == 0) {
        return 0.0;
    }
    nn_real task_values[ARC_PROGRAM_TASK_FEATURE_COUNT];
    arc_program_task_features_write(features, task_values);
    double dot = 0.0;
    double lhs_norm = 0.0;
    double rhs_norm = 0.0;
    double denom = (double)count;
    for (size_t i = 0; i < ARC_PROGRAM_TASK_FEATURE_COUNT; ++i) {
        double lhs = sum[i] / denom;
        double rhs = (double)task_values[i];
        dot += lhs * rhs;
        lhs_norm += lhs * lhs;
        rhs_norm += rhs * rhs;
    }
    if (lhs_norm <= 0.0 || rhs_norm <= 0.0) {
        return 0.0;
    }
    return dot / sqrt(lhs_norm * rhs_norm);
}

static double arc_proposal_trace_prior_score(const ARCProposalTracePrior *prior,
                                             const ARCTransformCandidate *candidate)
{
    if (prior == NULL || candidate == NULL || prior->observed_rows == 0) {
        return 0.0;
    }

    double score = 0.0;
    size_t family = arc_candidate_family(candidate);
    if (family < sizeof(prior->family_seen) / sizeof(prior->family_seen[0])) {
        score += 16.0 * arc_trace_prior_log_odds(prior->family_seen[family],
                                                 prior->family_success[family]);
    }
    if (candidate->op < ARC_RULE_COUNT) {
        score += 8.0 * arc_trace_prior_log_odds(prior->op_seen[candidate->op],
                                                prior->op_success[candidate->op]);
    }
    if (candidate->input_color < ARC_COLOR_COUNT) {
        score += arc_trace_prior_slot_score(
            prior->input_color_seen[candidate->input_color],
            prior->input_color_success[candidate->input_color],
            1.0);
    }
    if (candidate->output_color < ARC_COLOR_COUNT) {
        score += arc_trace_prior_slot_score(
            prior->output_color_seen[candidate->output_color],
            prior->output_color_success[candidate->output_color],
            2.0);
    }
    if (candidate->background_color < ARC_COLOR_COUNT) {
        score += arc_trace_prior_slot_score(
            prior->background_color_seen[candidate->background_color],
            prior->background_color_success[candidate->background_color],
            0.5);
    }
    if (candidate->selector_kind < ARC_OBJECT_SELECTOR_KIND_COUNT) {
        score += arc_trace_prior_slot_score(
            prior->selector_seen[candidate->selector_kind],
            prior->selector_success[candidate->selector_kind],
            1.0);
    }
    if (candidate->output_selector_kind < ARC_OBJECT_SELECTOR_KIND_COUNT) {
        score += arc_trace_prior_slot_score(
            prior->output_selector_seen[candidate->output_selector_kind],
            prior->output_selector_success[candidate->output_selector_kind],
            1.0);
    }
    if (candidate->render_mode < ARC_OBJECT_RENDER_KIND_COUNT) {
        score += arc_trace_prior_slot_score(
            prior->render_seen[candidate->render_mode],
            prior->render_success[candidate->render_mode],
            0.75);
    }
    size_t scale = candidate->scale < sizeof(prior->scale_seen) / sizeof(prior->scale_seen[0])
                       ? candidate->scale
                       : 0;
    score += arc_trace_prior_slot_score(prior->scale_seen[scale],
                                        prior->scale_success[scale],
                                        0.75);
    return score;
}

static double arc_proposal_trace_task_prior_score(
    const ARCProposalTracePrior *prior,
    const ARCTransformCandidate *candidate,
    const ARCProgramTaskFeatures *task_features)
{
    if (prior == NULL || candidate == NULL || task_features == NULL ||
        prior->observed_rows == 0) {
        return 0.0;
    }
    double score = arc_proposal_trace_prior_score(prior, candidate);
    size_t family = arc_candidate_family(candidate);
    if (family < sizeof(prior->family_seen) / sizeof(prior->family_seen[0])) {
        double similarity = arc_program_task_feature_similarity(
            prior->family_task_feature_sum[family],
            prior->family_success[family],
            task_features);
        score += 3.0 * log1p((double)prior->family_success[family]) *
                 (similarity - 0.92);
    }
    if (candidate->op < ARC_RULE_COUNT) {
        double similarity = arc_program_task_feature_similarity(
            prior->op_task_feature_sum[candidate->op],
            prior->op_success[candidate->op],
            task_features);
        score += 2.0 * log1p((double)prior->op_success[candidate->op]) *
                 (similarity - 0.92);
    }
    return score;
}

static double arc_proposal_trace_linear_raw(const ARCProposalTracePrior *prior,
                                            const ARCTransformCandidate *candidate,
                                            const ARCProgramTaskFeatures *task_features)
{
    if (prior == NULL || candidate == NULL || task_features == NULL ||
        prior->linear_updates == 0) {
        return 0.0;
    }
    nn_real features[ARC_PROGRAM_FEATURE_COUNT];
    if (!arc_encode_program_features(candidate,
                                     task_features,
                                     features,
                                     ARC_PROGRAM_FEATURE_COUNT)) {
        return 0.0;
    }
    double raw = prior->linear_bias;
    for (size_t i = 0; i < ARC_PROGRAM_FEATURE_COUNT; ++i) {
        raw += prior->linear_weights[i] * (double)features[i];
    }
    if (raw > 8.0) {
        raw = 8.0;
    } else if (raw < -8.0) {
        raw = -8.0;
    }
    return raw;
}

static double arc_proposal_trace_linear_score(const ARCProposalTracePrior *prior,
                                              const ARCTransformCandidate *candidate,
                                              const ARCProgramTaskFeatures *task_features)
{
    double score = arc_proposal_trace_task_prior_score(prior, candidate, task_features);
    double raw = arc_proposal_trace_linear_raw(prior, candidate, task_features);
    return score + 0.75 * (raw > 0.0 ? raw : 0.0);
}

static double arc_proposal_trace_mlp_raw(ARCProposalTracePrior *prior,
                                         const ARCTransformCandidate *candidate,
                                         const ARCProgramTaskFeatures *task_features)
{
    if (prior == NULL || prior->mlp == NULL || candidate == NULL ||
        task_features == NULL || prior->mlp_updates == 0) {
        return 0.0;
    }
    nn_real features[ARC_PROGRAM_FEATURE_COUNT];
    nn_real output[1];
    if (!arc_encode_program_features(candidate,
                                     task_features,
                                     features,
                                     ARC_PROGRAM_FEATURE_COUNT)) {
        return 0.0;
    }
    nn_forward(prior->mlp, features, output);
    double raw = (double)output[0];
    if (raw > 8.0) {
        raw = 8.0;
    } else if (raw < -8.0) {
        raw = -8.0;
    }
    return raw;
}

static double arc_proposal_trace_mlp_score(ARCProposalTracePrior *prior,
                                           const ARCTransformCandidate *candidate,
                                           const ARCProgramTaskFeatures *task_features)
{
    double score = arc_proposal_trace_task_prior_score(prior, candidate, task_features);
    double raw = arc_proposal_trace_mlp_raw(prior, candidate, task_features);
    return score + 1.5 * (raw > 0.0 ? raw : 0.0);
}

static void arc_proposal_trace_linear_update(ARCProposalTracePrior *prior,
                                             const ARCTransformCandidate *candidate,
                                             const ARCProgramEval *eval,
                                             const ARCProgramTaskFeatures *task_features)
{
    if (prior == NULL || candidate == NULL || eval == NULL || task_features == NULL) {
        return;
    }
    nn_real features[ARC_PROGRAM_FEATURE_COUNT];
    if (!arc_encode_program_features(candidate,
                                     task_features,
                                     features,
                                     ARC_PROGRAM_FEATURE_COUNT)) {
        return;
    }
    double target = eval->train_consistent ? 1.0 : 0.0;
    double raw = prior->linear_bias;
    for (size_t i = 0; i < ARC_PROGRAM_FEATURE_COUNT; ++i) {
        raw += prior->linear_weights[i] * (double)features[i];
    }
    if (raw > 30.0) {
        raw = 30.0;
    } else if (raw < -30.0) {
        raw = -30.0;
    }
    double prediction = 1.0 / (1.0 + exp(-raw));
    double sample_weight = target > 0.5 ? 8.0 : 1.0;
    double rate = 0.025 * sample_weight;
    double gradient = rate * (target - prediction);
    prior->linear_bias += gradient;
    for (size_t i = 0; i < ARC_PROGRAM_FEATURE_COUNT; ++i) {
        prior->linear_weights[i] =
            0.99999 * prior->linear_weights[i] + gradient * (double)features[i];
    }
    ++prior->linear_updates;
}

static void arc_proposal_trace_mlp_update_task(ARCProposalTracePrior *prior,
                                               const ARCTransformCandidate *candidates,
                                               const ARCProgramEval *evals,
                                               size_t candidate_count,
                                               const ARCProgramTaskFeatures *task_features,
                                               size_t rejected_cap)
{
    if (prior == NULL || prior->mlp == NULL || candidates == NULL || evals == NULL ||
        task_features == NULL || candidate_count == 0) {
        return;
    }
    size_t emitted_indices[ARC_PROPOSAL_TRACE_NEGATIVE_CAP + ARC_RULE_CANDIDATE_COUNT];
    size_t emitted_count = 0;
    size_t rejected_emitted = 0;
    for (size_t i = 0; i < candidate_count; ++i) {
        if (!arc_proposal_trace_should_emit(&evals[i], rejected_emitted, rejected_cap)) {
            continue;
        }
        if (!evals[i].train_consistent &&
            !arc_program_eval_is_exact_success(&evals[i])) {
            ++rejected_emitted;
        }
        if (emitted_count < sizeof(emitted_indices) / sizeof(emitted_indices[0])) {
            emitted_indices[emitted_count++] = i;
        }
    }
    if (emitted_count < 2) {
        return;
    }
    nn_real *features = calloc(emitted_count * ARC_PROGRAM_FEATURE_COUNT, sizeof(*features));
    if (features == NULL) {
        return;
    }
    size_t positives = 0;
    for (size_t row = 0; row < emitted_count; ++row) {
        size_t index = emitted_indices[row];
        if (!arc_encode_program_features(&candidates[index],
                                         task_features,
                                         &features[row * ARC_PROGRAM_FEATURE_COUNT],
                                         ARC_PROGRAM_FEATURE_COUNT)) {
            free(features);
            return;
        }
        positives += evals[index].train_consistent ? 1u : 0u;
    }
    if (positives == 0) {
        free(features);
        return;
    }
    nn_real weight = (nn_real)(1.0 / (double)positives);
    for (size_t epoch = 0; epoch < 3u; ++epoch) {
        for (size_t row = 0; row < emitted_count; ++row) {
            size_t index = emitted_indices[row];
            if (!evals[index].train_consistent) {
                continue;
            }
            nn_real loss = nn_train_softmax_scorer_weighted(prior->mlp,
                                                            features,
                                                            emitted_count,
                                                            row,
                                                            weight,
                                                            (nn_real)0.012);
            if (!isfinite((double)loss)) {
                free(features);
                return;
            }
            ++prior->mlp_updates;
        }
    }
    free(features);
}

static void arc_build_program_ranks(const ARCTask *task,
                                    const ARCTransformCandidate *candidates,
                                    const ARCProgramEval *evals,
                                    size_t candidate_count,
                                    size_t budget,
                                    ARCProgramStrategy strategy,
                                    ARCProgramLearnedState *learned,
                                    ARCProposalTracePrior *trace_prior,
                                    const ARCProgramTaskFeatures *task_features,
                                    ARCProgramRank *ranks);


static void arc_proposal_trace_prior_observe_candidate(ARCProposalTracePrior *prior,
                                                       const ARCTransformCandidate *candidate,
                                                       const ARCProgramEval *eval,
                                                       const ARCProgramTaskFeatures *task_features)
{
    if (prior == NULL || candidate == NULL || eval == NULL) {
        return;
    }
    int success = eval->train_consistent ? 1 : 0;
    nn_real task_values[ARC_PROGRAM_TASK_FEATURE_COUNT];
    if (success && task_features != NULL) {
        arc_program_task_features_write(task_features, task_values);
    }
    arc_proposal_trace_linear_update(prior, candidate, eval, task_features);
    size_t family = arc_candidate_family(candidate);
    if (family < sizeof(prior->family_seen) / sizeof(prior->family_seen[0])) {
        ++prior->family_seen[family];
        prior->family_success[family] += success ? 1u : 0u;
        if (success && task_features != NULL) {
            for (size_t i = 0; i < ARC_PROGRAM_TASK_FEATURE_COUNT; ++i) {
                prior->family_task_feature_sum[family][i] += (double)task_values[i];
            }
        }
    }
    if (candidate->op < ARC_RULE_COUNT) {
        ++prior->op_seen[candidate->op];
        prior->op_success[candidate->op] += success ? 1u : 0u;
        if (success && task_features != NULL) {
            for (size_t i = 0; i < ARC_PROGRAM_TASK_FEATURE_COUNT; ++i) {
                prior->op_task_feature_sum[candidate->op][i] += (double)task_values[i];
            }
        }
    }
    if (candidate->input_color < ARC_COLOR_COUNT) {
        ++prior->input_color_seen[candidate->input_color];
        prior->input_color_success[candidate->input_color] += success ? 1u : 0u;
    }
    if (candidate->output_color < ARC_COLOR_COUNT) {
        ++prior->output_color_seen[candidate->output_color];
        prior->output_color_success[candidate->output_color] += success ? 1u : 0u;
    }
    if (candidate->background_color < ARC_COLOR_COUNT) {
        ++prior->background_color_seen[candidate->background_color];
        prior->background_color_success[candidate->background_color] += success ? 1u : 0u;
    }
    if (candidate->selector_kind < ARC_OBJECT_SELECTOR_KIND_COUNT) {
        ++prior->selector_seen[candidate->selector_kind];
        prior->selector_success[candidate->selector_kind] += success ? 1u : 0u;
    }
    if (candidate->output_selector_kind < ARC_OBJECT_SELECTOR_KIND_COUNT) {
        ++prior->output_selector_seen[candidate->output_selector_kind];
        prior->output_selector_success[candidate->output_selector_kind] += success ? 1u : 0u;
    }
    if (candidate->render_mode < ARC_OBJECT_RENDER_KIND_COUNT) {
        ++prior->render_seen[candidate->render_mode];
        prior->render_success[candidate->render_mode] += success ? 1u : 0u;
    }
    size_t scale = candidate->scale < sizeof(prior->scale_seen) / sizeof(prior->scale_seen[0])
                       ? candidate->scale
                       : 0;
    ++prior->scale_seen[scale];
    prior->scale_success[scale] += success ? 1u : 0u;
    ++prior->observed_rows;
    prior->positive_rows += success ? 1u : 0u;
}

static void arc_proposal_trace_prior_observe_task(ARCProposalTracePrior *prior,
                                                  const ARCTask *task,
                                                  const ARCTransformCandidate *candidates,
                                                  const ARCProgramEval *evals,
                                                  size_t candidate_count,
                                                  const size_t *budgets,
                                                  size_t budget_count,
                                                  size_t rejected_cap)
{
    if (prior == NULL || task == NULL || candidates == NULL || evals == NULL) {
        return;
    }
    (void)budgets;
    (void)budget_count;
    ARCProgramTaskFeatures task_features;
    arc_program_task_features(task, &task_features);
    arc_proposal_trace_mlp_update_task(prior,
                                       candidates,
                                       evals,
                                       candidate_count,
                                       &task_features,
                                       rejected_cap);
    size_t rejected_emitted = 0;
    for (size_t i = 0; i < candidate_count; ++i) {
        if (!arc_proposal_trace_should_emit(&evals[i], rejected_emitted, rejected_cap)) {
            continue;
        }
        if (!evals[i].train_consistent &&
            !arc_program_eval_is_exact_success(&evals[i])) {
            ++rejected_emitted;
        }
        arc_proposal_trace_prior_observe_candidate(prior,
                                                   &candidates[i],
                                                   &evals[i],
                                                   &task_features);
    }
    ++prior->observed_tasks;
}

static int arc_program_learned_train_hypothesis_prior(ARCProgramLearnedState *state,
                                                      const ARCTransformCandidate *candidates,
                                                      const ARCProgramEval *evals,
                                                      size_t candidate_count,
                                                      nn_real learning_rate)
{
    if (state == NULL || candidates == NULL || evals == NULL || candidate_count == 0) {
        return 0;
    }

    size_t sample_count = 0;
    for (size_t i = 0; i < candidate_count; ++i) {
        sample_count += arc_candidate_is_hypothesis_induced(&candidates[i]) ? 1u : 0u;
    }
    if (sample_count == 0) {
        return 1;
    }

    int train_neural = state->hypothesis_neural_prior_enabled &&
                       state->hypothesis_prior_net != NULL &&
                       learning_rate > (nn_real)0;
    nn_real *features = NULL;
    nn_real *targets = NULL;
    if (train_neural) {
        features = calloc(sample_count * ARC_HYPOTHESIS_PRIOR_FEATURE_COUNT,
                          sizeof(*features));
        targets = calloc(sample_count, sizeof(*targets));
        if (features == NULL || targets == NULL) {
            free(features);
            free(targets);
            return 0;
        }
    }

    size_t row = 0;
    for (size_t i = 0; i < candidate_count; ++i) {
        if (!arc_candidate_is_hypothesis_induced(&candidates[i])) {
            continue;
        }
        NNHypothesis hypothesis;
        if (!arc_hypothesis_from_candidate(&candidates[i], &hypothesis)) {
            continue;
        }
        int certified = evals[i].train_consistent ? 1 : 0;
        arc_hypothesis_prior_observe(state, &hypothesis, certified);
        if (!train_neural) {
            continue;
        }
        size_t encoded_count = 0;
        if (!nn_hypothesis_encode_features(
                &hypothesis,
                &features[row * ARC_HYPOTHESIS_PRIOR_FEATURE_COUNT],
                ARC_HYPOTHESIS_PRIOR_FEATURE_COUNT,
                &encoded_count) ||
            encoded_count != ARC_HYPOTHESIS_PRIOR_FEATURE_COUNT) {
            continue;
        }
        targets[row] = certified ? (nn_real)1 : (nn_real)0;
        ++row;
    }

    if (train_neural && row != 0) {
        nn_real loss = nn_train_batched(state->hypothesis_prior_net,
                                        features,
                                        targets,
                                        row,
                                        row,
                                        learning_rate,
                                        ARC_HYPOTHESIS_PRIOR_EPOCHS,
                                        0);
        if (!isfinite((double)loss)) {
            free(features);
            free(targets);
            return 0;
        }
        state->hypothesis_prior_loss = (double)loss;
        state->hypothesis_prior_samples += row;
        state->hypothesis_prior_updates += ARC_HYPOTHESIS_PRIOR_EPOCHS;
    }

    free(features);
    free(targets);
    return 1;
}

static int arc_program_learned_train_task(ARCProgramLearnedState *state,
                                          const ARCTask *task,
                                          const ARCTransformCandidate *candidates,
                                          const ARCProgramEval *evals,
                                          size_t candidate_count,
                                          nn_real learning_rate)
{
    if (state == NULL || state->net == NULL || task == NULL || candidates == NULL ||
        evals == NULL || candidate_count == 0 || learning_rate <= (nn_real)0) {
        return 0;
    }

    size_t exact_count = 0;
    for (size_t i = 0; i < candidate_count; ++i) {
        if (arc_candidate_is_composed(&candidates[i])) {
            continue;
        }
        exact_count += arc_program_eval_is_exact_success(&evals[i]) ? 1u : 0u;
    }
    if (exact_count == 0) {
        return 1;
    }

    ARCProgramTaskFeatures task_features;
    arc_program_task_features(task, &task_features);
    nn_real *features =
        calloc(candidate_count * ARC_PROGRAM_FEATURE_COUNT, sizeof(*features));
    if (features == NULL) {
        return 0;
    }
    if (!arc_build_program_feature_matrix(candidates,
                                          candidate_count,
                                          &task_features,
                                          features,
                                          ARC_PROGRAM_FEATURE_COUNT)) {
        free(features);
        return 0;
    }

    nn_real weight = (nn_real)(1.0 / (double)exact_count);
    for (size_t epoch = 0; epoch < ARC_PROGRAM_TRAIN_EPOCHS; ++epoch) {
        for (size_t i = 0; i < candidate_count; ++i) {
            if (arc_candidate_is_composed(&candidates[i])) {
                continue;
            }
            if (!arc_program_eval_is_exact_success(&evals[i])) {
                continue;
            }
            nn_real loss = nn_train_softmax_scorer_weighted(state->net,
                                                            features,
                                                            candidate_count,
                                                            i,
                                                            weight,
                                                            learning_rate);
            if (!isfinite((double)loss)) {
                free(features);
                return 0;
            }
            ++state->train_updates;
        }
    }
    ++state->trained_tasks;
    free(features);
    return 1;
}

static void arc_build_program_ranks(const ARCTask *task,
                                    const ARCTransformCandidate *candidates,
                                    const ARCProgramEval *evals,
                                    size_t candidate_count,
                                    size_t budget,
                                    ARCProgramStrategy strategy,
                                    ARCProgramLearnedState *learned,
                                    ARCProposalTracePrior *trace_prior,
                                    const ARCProgramTaskFeatures *task_features,
                                    ARCProgramRank *ranks)
{
    (void)budget;
    uint64_t task_hash = arc_task_hash(task);
    for (size_t i = 0; i < candidate_count; ++i) {
        ranks[i].candidate_index = i;
        ranks[i].primary = 0.0;
        ranks[i].secondary = arc_task_conditioned_program_score(task, &candidates[i]);
        ranks[i].tie_breaker = arc_hash_mix(task_hash, evals[i].program_hash ^ (uint64_t)i);
        switch (strategy) {
        case ARC_PROGRAM_STRATEGY_GRAMMAR:
            ranks[i].primary = -(double)i;
            ranks[i].secondary = 0.0;
            ranks[i].tie_breaker = i;
            break;
        case ARC_PROGRAM_STRATEGY_STRUCTURAL:
            ranks[i].primary = ranks[i].secondary;
            break;
        case ARC_PROGRAM_STRATEGY_RANDOM:
            ranks[i].primary = (double)(ranks[i].tie_breaker & UINT64_C(0xffffffff));
            ranks[i].secondary = 0.0;
            break;
        case ARC_PROGRAM_STRATEGY_LEARNED:
            ranks[i].primary =
                ranks[i].secondary +
                arc_learned_program_score(learned, &candidates[i], task_features);
            break;
        case ARC_PROGRAM_STRATEGY_TRACE_PRIOR:
            ranks[i].primary =
                ranks[i].secondary +
                arc_proposal_trace_prior_score(trace_prior, &candidates[i]);
            break;
        case ARC_PROGRAM_STRATEGY_TRACE_TASK_PRIOR:
            ranks[i].primary =
                ranks[i].secondary +
                arc_proposal_trace_task_prior_score(trace_prior,
                                                    &candidates[i],
                                                    task_features);
            break;
        case ARC_PROGRAM_STRATEGY_TRACE_LINEAR:
            ranks[i].primary =
                ranks[i].secondary +
                arc_proposal_trace_linear_score(trace_prior,
                                                &candidates[i],
                                                task_features);
            break;
        case ARC_PROGRAM_STRATEGY_TRACE_MLP:
            ranks[i].primary =
                ranks[i].secondary +
                arc_proposal_trace_mlp_score(trace_prior,
                                             &candidates[i],
                                             task_features);
            break;
        case ARC_PROGRAM_STRATEGY_ORACLE:
            ranks[i].primary = evals[i].train_consistent ? (double)evals[i].test_correct_cells : -1.0;
            ranks[i].secondary = evals[i].test_cells != 0 &&
                                         evals[i].test_correct_cells == evals[i].test_cells
                                     ? 1.0
                                     : 0.0;
            break;
        }
    }

    for (size_t i = 1; i < candidate_count; ++i) {
        ARCProgramRank key = ranks[i];
        size_t j = i;
        while (j > 0) {
            ARCProgramRank prev = ranks[j - 1u];
            int before = key.primary > prev.primary ||
                         (key.primary == prev.primary && key.secondary > prev.secondary) ||
                         (key.primary == prev.primary && key.secondary == prev.secondary &&
                          key.tie_breaker < prev.tie_breaker);
            if (!before) {
                break;
            }
            ranks[j] = ranks[j - 1u];
            --j;
        }
        ranks[j] = key;
    }
}

static int arc_candidate_supports_train_inputs(const ARCTask *task,
                                               const ARCTransformCandidate *candidate)
{
    ARCGrid produced;
    for (size_t pair = 0; pair < task->train_count; ++pair) {
        if (!task->train[pair].has_output ||
            !arc_apply_candidate(candidate, &task->train[pair].input, &produced)) {
            return 0;
        }
    }
    return task->train_count > 0;
}

static int arc_composed_candidate_exists(const ARCTransformCandidate *candidates,
                                         size_t count,
                                         const ARCTransformCandidate *candidate)
{
    if (!arc_candidate_is_composed(candidate)) {
        return 0;
    }
    for (size_t i = 0; i < count; ++i) {
        if (!arc_candidate_is_composed(&candidates[i])) {
            continue;
        }
        if (candidates[i].steps[0].op == candidate->steps[0].op &&
            candidates[i].steps[1].op == candidate->steps[1].op &&
            strcmp(candidates[i].steps[0].name, candidate->steps[0].name) == 0 &&
            strcmp(candidates[i].steps[1].name, candidate->steps[1].name) == 0) {
            return 1;
        }
    }
    return 0;
}

static size_t arc_add_learned_compositions(const ARCTask *task,
                                           ARCProgramLearnedState *learned,
                                           ARCTransformCandidate *candidates,
                                           size_t capacity,
                                           size_t base_count)
{
    if (task == NULL || learned == NULL || learned->trained_tasks == 0 ||
        candidates == NULL || base_count == 0 || base_count >= capacity) {
        return base_count;
    }

    ARCProgramEval evals[ARC_RULE_CANDIDATE_COUNT];
    ARCProgramRank ranks[ARC_RULE_CANDIDATE_COUNT];
    ARCProgramTaskFeatures task_features;
    arc_program_task_features(task, &task_features);
    for (size_t i = 0; i < base_count; ++i) {
        NNProgram program;
        memset(&evals[i], 0, sizeof(evals[i]));
        if (arc_candidate_to_program(&candidates[i], &program)) {
            evals[i].program_hash = program.canonical_hash;
        }
    }
    arc_build_program_ranks(task,
                            candidates,
                            evals,
                            base_count,
                            0,
                            ARC_PROGRAM_STRATEGY_LEARNED,
                            learned,
                            NULL,
                            &task_features,
                            ranks);

    size_t top_indices[ARC_LEARNED_COMPOSE_TOP];
    size_t top_count = 0;
    for (size_t rank = 0; rank < base_count && top_count < ARC_LEARNED_COMPOSE_TOP; ++rank) {
        size_t index = ranks[rank].candidate_index;
        if (arc_candidate_is_composed(&candidates[index])) {
            continue;
        }
        top_indices[top_count++] = index;
    }

    size_t count = base_count;
    size_t added = 0;
    for (size_t first = 0; first < top_count; ++first) {
        for (size_t second = 0; second < top_count; ++second) {
            if (count >= capacity || added >= ARC_LEARNED_COMPOSE_LIMIT) {
                return count;
            }
            const ARCTransformCandidate *first_candidate = &candidates[top_indices[first]];
            const ARCTransformCandidate *second_candidate = &candidates[top_indices[second]];
            if (first_candidate->op == ARC_RULE_GRID_IDENTITY &&
                second_candidate->op == ARC_RULE_GRID_IDENTITY) {
                continue;
            }
            ARCTransformCandidate composed;
            if (!arc_init_composed_candidate(&composed, first_candidate, second_candidate) ||
                arc_composed_candidate_exists(candidates, count, &composed) ||
                !arc_candidate_supports_train_inputs(task, &composed)) {
                continue;
            }
            arc_add_candidate(candidates, capacity, &count, &composed);
            ++added;
        }
    }
    return count;
}

static int arc_candidate_name_exists(const ARCTransformCandidate *candidates,
                                     size_t count,
                                     const char *name)
{
    if (candidates == NULL || name == NULL) {
        return 0;
    }
    for (size_t i = 0; i < count; ++i) {
        if (strcmp(candidates[i].name, name) == 0) {
            return 1;
        }
    }
    return 0;
}

static int arc_hypothesis_proposal_name_exists(const ARCHypothesisProposal *proposals,
                                               size_t count,
                                               const char *name)
{
    if (proposals == NULL || name == NULL) {
        return 0;
    }
    for (size_t i = 0; i < count; ++i) {
        if (strcmp(proposals[i].candidate.name, name) == 0) {
            return 1;
        }
    }
    return 0;
}

static double arc_hypothesis_structural_proposal_score(const ARCProgramLearnedState *learned,
                                                       const ARCTransformCandidate *candidate,
                                                       const ARCObject *input_object,
                                                       const ARCObject *output_object,
                                                       const ARCProgramTaskFeatures *task_features)
{
    (void)task_features;
    if (candidate == NULL) {
        return -1.0e9;
    }
    if (input_object == NULL || output_object == NULL) {
        return arc_structural_program_score(candidate);
    }

    double score = 64.0;
    score += arc_objects_same_shape(input_object, output_object) ? 16.0 : -32.0;
    score -= 0.5 * (double)(abs(candidate->dx) + abs(candidate->dy));
    score += candidate->input_color == candidate->output_color ? 0.25 : 1.0;
    switch (arc_candidate_selector_kind(candidate)) {
    case ARC_OBJECT_SELECTOR_UNIQUE_COLOR:
        score += 6.0;
        break;
    case ARC_OBJECT_SELECTOR_LARGEST_COLOR:
    case ARC_OBJECT_SELECTOR_SMALLEST_COLOR:
        score += 4.0;
        break;
    case ARC_OBJECT_SELECTOR_INTERIOR_COLOR:
    case ARC_OBJECT_SELECTOR_TOUCHING_BORDER_COLOR:
        score += 3.0;
        break;
    case ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR:
        score += 2.0;
        break;
    default:
        break;
    }
    if (learned != NULL) {
        score += arc_learned_family_score(learned, candidate);
        score += 8.0 * arc_learned_relation_score(learned, candidate);
    }
    return score;
}

static double arc_hypothesis_slot_proposal_score(ARCProgramLearnedState *learned,
                                                 const NNHypothesis *hypothesis,
                                                 const ARCTransformCandidate *candidate,
                                                 const ARCObject *input_object,
                                                 const ARCObject *output_object,
                                                 const ARCProgramTaskFeatures *task_features,
                                                 int train_consistent_pre_cap)
{
    double structural = 0.0;
    if (learned == NULL || learned->hypothesis_structural_score_enabled) {
        structural = arc_hypothesis_structural_proposal_score(learned,
                                                              candidate,
                                                              input_object,
                                                              output_object,
                                                              task_features);
    }
    if (learned == NULL || hypothesis == NULL) {
        return structural;
    }
    double verifier_score =
        learned->hypothesis_train_consistency_prior_enabled && train_consistent_pre_cap
            ? (double)ARC_HYPOTHESIS_TRAIN_CONSISTENCY_PRIOR
            : 0.0;

    size_t proposal_support = 0;
    size_t template_support = 0;
    double template_log_odds = 0.0;
    if (learned->hypothesis_slot_prior_enabled) {
        template_log_odds =
            arc_hypothesis_template_log_odds(learned, hypothesis, &template_support);
    }
    double slot_log_odds = 0.0;
    if (learned->hypothesis_slot_prior_enabled) {
        slot_log_odds =
            arc_hypothesis_slot_log_odds(learned, hypothesis, &proposal_support);
    }
    int template_ready = learned->hypothesis_slot_prior_enabled &&
                         template_support >= ARC_HYPOTHESIS_TEMPLATE_PRIOR_MIN_PROPOSALS;
    int slot_ready = learned->hypothesis_slot_prior_enabled &&
                     proposal_support >= ARC_HYPOTHESIS_PRIOR_MIN_PROPOSALS;
    int neural_ready = learned->hypothesis_neural_prior_enabled &&
                       learned->hypothesis_prior_net != NULL &&
                       learned->hypothesis_prior_samples >=
                           ARC_HYPOTHESIS_PRIOR_MIN_PROPOSALS;
    size_t template_probe_support =
        hypothesis->template_ref == NULL
            ? 0
            : arc_hypothesis_template_probe_count(
                  learned,
                  hypothesis->template_ref->template_id);
    size_t slot_probe_support = 0;
    double slot_probe_score =
        arc_hypothesis_slot_probe_score(learned, hypothesis, &slot_probe_support);
    int probe_ready = template_probe_support > 0 || slot_probe_support > 0;
    if (!template_ready && !slot_ready && !neural_ready && !probe_ready) {
        return structural + verifier_score;
    }

    nn_real features[ARC_HYPOTHESIS_PRIOR_FEATURE_COUNT];
    size_t feature_count = 0;
    nn_real output[1] = {0};
    double neural_score = 0.0;
    if (neural_ready &&
        nn_hypothesis_encode_features(hypothesis,
                                      features,
                                      ARC_HYPOTHESIS_PRIOR_FEATURE_COUNT,
                                      &feature_count) &&
        feature_count == ARC_HYPOTHESIS_PRIOR_FEATURE_COUNT) {
        nn_forward(learned->hypothesis_prior_net, features, output);
        neural_score = (double)output[0];
    }

    double cold_start = 0.0;
    uint8_t selector = arc_candidate_selector_kind(candidate);
    if (learned->hypothesis_cold_start_enabled &&
        selector < ARC_OBJECT_SELECTOR_KIND_COUNT) {
        cold_start += learned->object_selector_score[selector];
        for (size_t relation = 1; relation < ARC_OBJECT_RELATION_KIND_COUNT; ++relation) {
            cold_start += 0.25 * learned->relation_selector_score[relation][selector];
        }
    }

    return structural + verifier_score +
           (template_ready ? (double)ARC_HYPOTHESIS_TEMPLATE_PRIOR_WEIGHT *
                                 template_log_odds
                           : 0.0) +
           arc_hypothesis_template_probe_score(learned, hypothesis) +
           slot_probe_score +
           (slot_ready ? (double)ARC_HYPOTHESIS_LOG_ODDS_WEIGHT * slot_log_odds : 0.0) +
           (neural_ready ? (double)ARC_HYPOTHESIS_NEURAL_WEIGHT * neural_score : 0.0) +
           (double)ARC_HYPOTHESIS_COLD_START_WEIGHT * cold_start;
}
