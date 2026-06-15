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

static void arc_trace_holo_normalize(double *values, size_t count)
{
    if (values == NULL || count == 0) {
        return;
    }
    double norm = 0.0;
    for (size_t i = 0; i < count; ++i) {
        norm += values[i] * values[i];
    }
    if (norm <= 0.0) {
        return;
    }
    double scale = 1.0 / sqrt(norm);
    for (size_t i = 0; i < count; ++i) {
        values[i] *= scale;
    }
}

static void arc_trace_holo_state_vector(const ARCProgramTaskFeatures *task_features,
                                        double *state)
{
    if (task_features == NULL || state == NULL) {
        return;
    }
    nn_real task_values[ARC_PROGRAM_TASK_FEATURE_COUNT];
    arc_program_task_features_write(task_features, task_values);
    for (size_t i = 0; i < ARC_PROGRAM_TASK_FEATURE_COUNT; ++i) {
        double value = (double)task_values[i] - 0.5;
        state[i] = value;
        state[ARC_PROGRAM_TASK_FEATURE_COUNT + i] = (i & 1u) ? -value : value;
    }
    arc_trace_holo_normalize(state, ARC_PROGRAM_FEATURE_COUNT);
}

static int arc_trace_holo_action_vector(const ARCTransformCandidate *candidate,
                                        const ARCProgramTaskFeatures *task_features,
                                        double *action)
{
    if (candidate == NULL || task_features == NULL || action == NULL) {
        return 0;
    }
    nn_real features[ARC_PROGRAM_FEATURE_COUNT];
    if (!arc_encode_program_features(candidate,
                                     task_features,
                                     features,
                                     ARC_PROGRAM_FEATURE_COUNT)) {
        return 0;
    }
    for (size_t i = 0; i < ARC_PROGRAM_FEATURE_COUNT; ++i) {
        action[i] = (double)features[i] - 0.5;
    }
    arc_trace_holo_normalize(action, ARC_PROGRAM_FEATURE_COUNT);
    return 1;
}

static int arc_trace_holo_pressure_state_vector(
    const ARCTransformCandidate *candidate,
    const ARCProgramTaskFeatures *task_features,
    double *state)
{
    if (candidate == NULL || task_features == NULL || state == NULL) {
        return 0;
    }
    nn_real features[ARC_PROGRAM_FEATURE_COUNT];
    if (!arc_encode_program_features(candidate,
                                     task_features,
                                     features,
                                     ARC_PROGRAM_FEATURE_COUNT)) {
        return 0;
    }
    arc_trace_holo_state_vector(task_features, state);
    for (size_t i = 0; i < ARC_PROGRAM_FEATURE_COUNT; ++i) {
        state[i] += 0.35 * ((double)features[i] - 0.5);
    }
    size_t family = arc_candidate_family(candidate);
    state[family % ARC_PROGRAM_FEATURE_COUNT] += 0.75;
    state[(ARC_PROGRAM_TASK_FEATURE_COUNT + candidate->op) %
          ARC_PROGRAM_FEATURE_COUNT] += 0.50;
    state[(candidate->input_color + 3u * ARC_PROGRAM_TASK_FEATURE_COUNT) %
          ARC_PROGRAM_FEATURE_COUNT] += 0.25;
    state[(candidate->output_color + 5u * ARC_PROGRAM_TASK_FEATURE_COUNT) %
          ARC_PROGRAM_FEATURE_COUNT] += 0.25;
    arc_trace_holo_normalize(state, ARC_PROGRAM_FEATURE_COUNT);
    return 1;
}

static double arc_proposal_trace_holo_confidence(
    const ARCProposalTracePrior *prior,
    const ARCTransformCandidate *candidate,
    const ARCProgramTaskFeatures *task_features)
{
    if (prior == NULL || candidate == NULL || task_features == NULL ||
        prior->holo_updates < 4u || prior->positive_rows < 4u) {
        return 0.0;
    }
    size_t family = arc_candidate_family(candidate);
    size_t family_success = 0;
    size_t op_success = 0;
    double family_similarity = 0.0;
    double op_similarity = 0.0;
    if (family < sizeof(prior->family_seen) / sizeof(prior->family_seen[0])) {
        family_success = prior->family_success[family];
        family_similarity = arc_program_task_feature_similarity(
            prior->family_task_feature_sum[family],
            prior->family_success[family],
            task_features);
    }
    if (candidate->op < ARC_RULE_COUNT) {
        op_success = prior->op_success[candidate->op];
        op_similarity = arc_program_task_feature_similarity(
            prior->op_task_feature_sum[candidate->op],
            prior->op_success[candidate->op],
            task_features);
    }
    double support = log1p((double)(family_success + op_success)) / log(12.0);
    if (support > 1.0) {
        support = 1.0;
    }
    double similarity = family_similarity > op_similarity ? family_similarity : op_similarity;
    double similarity_gate = (similarity - 0.88) / 0.10;
    if (similarity_gate < 0.0) {
        similarity_gate = 0.0;
    } else if (similarity_gate > 1.0) {
        similarity_gate = 1.0;
    }
    double memory = log1p((double)prior->holo_updates) / log(64.0);
    if (memory > 1.0) {
        memory = 1.0;
    }
    return support * similarity_gate * memory;
}

static double arc_proposal_trace_holo_residual_score(
    ARCProposalTracePrior *prior,
    const ARCTransformCandidate *candidate,
    const ARCProgramTaskFeatures *task_features,
    const double *holo_context)
{
    double score = arc_proposal_trace_task_prior_score(prior, candidate, task_features);
    if (prior == NULL || candidate == NULL || task_features == NULL ||
        holo_context == NULL || prior->holo_residual == NULL || prior->holo_updates == 0) {
        return score;
    }
    double confidence =
        arc_proposal_trace_holo_confidence(prior, candidate, task_features);
    if (confidence <= 0.0) {
        return score;
    }
    double action[ARC_PROGRAM_FEATURE_COUNT] = {0};
    if (!arc_trace_holo_action_vector(candidate, task_features, action)) {
        return score;
    }
    double raw = 0.0;
    for (size_t i = 0; i < ARC_PROGRAM_FEATURE_COUNT; ++i) {
        raw += holo_context[i] * action[i];
    }
    if (raw > 4.0) {
        raw = 4.0;
    } else if (raw < 0.0) {
        raw = 0.0;
    }
    return score + confidence * 1.25 * raw;
}

static double arc_proposal_trace_holo_pressure_residual(
    ARCProposalTracePrior *prior,
    const ARCTransformCandidate *candidate,
    const ARCProgramTaskFeatures *task_features,
    const double *pressure_context,
    size_t baseline_rank,
    size_t budget,
    double baseline_score,
    double exact_budget_floor)
{
    if (prior == NULL || candidate == NULL || task_features == NULL ||
        pressure_context == NULL || prior->holo_pressure == NULL ||
        prior->holo_pressure_updates < 4u || budget == 0 ||
        baseline_rank == 0 || baseline_rank <= budget) {
        return 0.0;
    }
    double confidence =
        arc_proposal_trace_holo_confidence(prior, candidate, task_features);
    if (confidence <= 0.0) {
        return 0.0;
    }
    double action[ARC_PROGRAM_FEATURE_COUNT] = {0};
    if (!arc_trace_holo_action_vector(candidate, task_features, action)) {
        return 0.0;
    }
    double raw = 0.0;
    for (size_t i = 0; i < ARC_PROGRAM_FEATURE_COUNT; ++i) {
        raw += pressure_context[i] * action[i];
    }
    if (raw > 4.0) {
        raw = 4.0;
    } else if (raw < 0.0) {
        raw = 0.0;
    }
    double residual = confidence * 0.85 * raw;
    if (isfinite(exact_budget_floor) && baseline_score + residual >= exact_budget_floor) {
        residual = exact_budget_floor - baseline_score - 1e-9;
        if (residual < 0.0) {
            residual = 0.0;
        }
    }
    return residual;
}

static double arc_proposal_trace_holo_pressure_raw(
    ARCProposalTracePrior *prior,
    const ARCTransformCandidate *candidate,
    const ARCProgramTaskFeatures *task_features)
{
    if (prior == NULL || prior->holo_pressure == NULL || candidate == NULL ||
        task_features == NULL || prior->holo_pressure_updates < 4u) {
        return 0.0;
    }
    double pressure_state[ARC_PROGRAM_FEATURE_COUNT] = {0};
    double pressure_context[ARC_PROGRAM_FEATURE_COUNT] = {0};
    double action[ARC_PROGRAM_FEATURE_COUNT] = {0};
    if (!arc_trace_holo_pressure_state_vector(candidate, task_features, pressure_state) ||
        !arc_trace_holo_action_vector(candidate, task_features, action) ||
        holo_partitioned_ranker_recall(prior->holo_pressure,
                                       pressure_state,
                                       pressure_context) != 0) {
        return 0.0;
    }
    double raw = 0.0;
    for (size_t i = 0; i < ARC_PROGRAM_FEATURE_COUNT; ++i) {
        raw += pressure_context[i] * action[i];
    }
    return raw > 0.0 ? raw : 0.0;
}

static void arc_proposal_trace_holo_update(ARCProposalTracePrior *prior,
                                           const ARCTransformCandidate *candidate,
                                           const ARCProgramEval *eval,
                                           const ARCProgramTaskFeatures *task_features)
{
    if (prior == NULL || prior->holo_residual == NULL || candidate == NULL ||
        eval == NULL || task_features == NULL || !eval->train_consistent) {
        return;
    }
    double state[ARC_PROGRAM_FEATURE_COUNT] = {0};
    double action[ARC_PROGRAM_FEATURE_COUNT] = {0};
    arc_trace_holo_state_vector(task_features, state);
    if (!arc_trace_holo_action_vector(candidate, task_features, action)) {
        return;
    }
    holo_ranker_store(prior->holo_residual, state, action);
    ++prior->holo_updates;
}

static void arc_proposal_trace_holo_pressure_update(
    ARCProposalTracePrior *prior,
    const ARCTransformCandidate *candidate,
    const ARCProgramTaskFeatures *task_features)
{
    if (prior == NULL || prior->holo_pressure == NULL || candidate == NULL ||
        task_features == NULL) {
        return;
    }
    double state[ARC_PROGRAM_FEATURE_COUNT] = {0};
    double action[ARC_PROGRAM_FEATURE_COUNT] = {0};
    if (!arc_trace_holo_pressure_state_vector(candidate, task_features, state) ||
        !arc_trace_holo_action_vector(candidate, task_features, action)) {
        return;
    }
    holo_partitioned_ranker_store(prior->holo_pressure, state, action);
    ++prior->holo_pressure_updates;
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

static void arc_holo_bucket_learned_gate_observe_task(
    ARCProposalTracePrior *prior,
    const ARCTask *task,
    const ARCTransformCandidate *candidates,
    const ARCProgramEval *evals,
    size_t candidate_count,
    const ARCProgramTaskFeatures *task_features,
    const size_t *budgets,
    size_t budget_count);

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
    arc_proposal_trace_holo_update(prior, candidate, eval, task_features);
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
    ARCProgramTaskFeatures task_features;
    arc_program_task_features(task, &task_features);
    arc_holo_bucket_learned_gate_observe_task(prior,
                                              task,
                                              candidates,
                                              evals,
                                              candidate_count,
                                              &task_features,
                                              budgets,
                                              budget_count);
    if (prior->holo_pressure != NULL && budgets != NULL && budget_count != 0 &&
        candidate_count <= ARC_RULE_CANDIDATE_COUNT) {
        ARCProgramRank task_ranks[ARC_RULE_CANDIDATE_COUNT];
        size_t task_position[ARC_RULE_CANDIDATE_COUNT];
        for (size_t i = 0; i < candidate_count; ++i) {
            task_position[i] = SIZE_MAX;
        }
        arc_build_program_ranks(task,
                                candidates,
                                evals,
                                candidate_count,
                                0,
                                ARC_PROGRAM_STRATEGY_TRACE_TASK_PRIOR,
                                NULL,
                                prior,
                                &task_features,
                                task_ranks);
        for (size_t rank = 0; rank < candidate_count; ++rank) {
            size_t index = task_ranks[rank].candidate_index;
            if (index < candidate_count) {
                task_position[index] = rank + 1u;
            }
        }
        for (size_t i = 0; i < candidate_count; ++i) {
            if (!arc_program_eval_is_exact_success(&evals[i]) ||
                task_position[i] == SIZE_MAX) {
                continue;
            }
            for (size_t bi = 0; bi < budget_count; ++bi) {
                if (budgets[bi] != 0 && task_position[i] > budgets[bi]) {
                    arc_proposal_trace_holo_pressure_update(prior,
                                                            &candidates[i],
                                                            &task_features);
                }
            }
        }
    }
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

static size_t arc_holo_bucket_min_budget_value(const ARCProposalTracePrior *prior)
{
    return prior != NULL && prior->holo_bucket_min_budget != 0
               ? prior->holo_bucket_min_budget
               : 2u;
}

static size_t arc_holo_bucket_min_pressure_updates_value(
    const ARCProposalTracePrior *prior)
{
    return prior != NULL && prior->holo_bucket_min_pressure_updates != 0
               ? prior->holo_bucket_min_pressure_updates
               : 4u;
}

static size_t arc_holo_bucket_review_slots_value(const ARCProposalTracePrior *prior,
                                                 size_t budget)
{
    if (budget < arc_holo_bucket_min_budget_value(prior)) {
        return 0;
    }
    if (prior != NULL && prior->holo_bucket_review_slots != 0) {
        return prior->holo_bucket_review_slots;
    }
    size_t review_slots = budget / 16u;
    if (review_slots == 0) {
        review_slots = 1;
    } else if (review_slots > 4u) {
        review_slots = 4u;
    }
    return review_slots;
}

static int arc_holo_bucket_fixed_gate_open(const ARCProposalTracePrior *prior,
                                           size_t budget)
{
    return prior != NULL && prior->holo_pressure != NULL &&
           prior->holo_pressure_updates >=
               arc_holo_bucket_min_pressure_updates_value(prior) &&
           arc_holo_bucket_review_slots_value(prior, budget) != 0;
}

typedef struct {
    size_t baseline_exact_in_budget;
    size_t pressure_candidates;
    size_t exact_pressure_candidates;
    size_t review_exact_candidates;
    double best_pressure_raw;
    double best_exact_pressure_raw;
    double best_exact_review_priority;
    double review_frontier_priority;
    double review_margin;
    double soft_target;
} ARCHoloBucketGateSurface;

static double arc_holo_bucket_clamp_unit(double value)
{
    if (value <= 0.0) {
        return 0.0;
    }
    if (value >= 1.0) {
        return 1.0;
    }
    return value;
}

static double arc_holo_bucket_review_priority(double pressure_raw,
                                              double baseline_score)
{
    if (pressure_raw <= 0.0) {
        return 0.0;
    }
    double baseline_bonus = baseline_score;
    if (baseline_bonus > 40.0) {
        baseline_bonus = 40.0;
    } else if (baseline_bonus < -40.0) {
        baseline_bonus = -40.0;
    }
    return pressure_raw + 0.10 * baseline_bonus;
}

static const char *arc_holo_bucket_selector_json_value(const char *line,
                                                       const char *key)
{
    char needle[96];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = strstr(line, needle);
    if (p == NULL) {
        return NULL;
    }
    p += strlen(needle);
    while (*p != '\0' && isspace((unsigned char)*p)) {
        ++p;
    }
    if (*p != ':') {
        return NULL;
    }
    ++p;
    while (*p != '\0' && isspace((unsigned char)*p)) {
        ++p;
    }
    return p;
}

static int arc_holo_bucket_review_selector_load(
    ARCHoloBucketReviewSelectorModel *model,
    const char *path)
{
    if (model == NULL || path == NULL || path[0] == '\0') {
        return 0;
    }
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "failed to open ARC Holo bucket review selector: %s\n", path);
        return 0;
    }
    char line[65536];
    while (fgets(line, sizeof(line), fp) != NULL) {
        if (strstr(line, "\"crlp.arc_holo_bucket_review_selector.weights.v1\"") ==
            NULL) {
            continue;
        }
        const char *feature_text =
            arc_holo_bucket_selector_json_value(line, "feature_count");
        const char *hash_text = arc_holo_bucket_selector_json_value(line, "hash_bins");
        const char *weights_text =
            arc_holo_bucket_selector_json_value(line, "weights");
        if (feature_text == NULL || hash_text == NULL || weights_text == NULL) {
            fclose(fp);
            return 0;
        }
        char *end = NULL;
        unsigned long feature_count = strtoul(feature_text, &end, 10);
        if (end == feature_text || feature_count == 0 ||
            feature_count > ARC_HOLO_BUCKET_REVIEW_SELECTOR_MAX_FEATURES) {
            fclose(fp);
            return 0;
        }
        unsigned long hash_bins = strtoul(hash_text, &end, 10);
        if (end == hash_text ||
            hash_bins > ARC_HOLO_BUCKET_REVIEW_SELECTOR_MAX_HASH_BINS ||
            feature_count !=
                ARC_HOLO_BUCKET_REVIEW_SELECTOR_BASE_FEATURES + hash_bins) {
            fclose(fp);
            return 0;
        }
        const char *p = strchr(weights_text, '[');
        if (p == NULL) {
            fclose(fp);
            return 0;
        }
        ++p;
        double weights[ARC_HOLO_BUCKET_REVIEW_SELECTOR_MAX_FEATURES] = {0};
        size_t count = 0;
        while (*p != '\0' && *p != ']' &&
               count < ARC_HOLO_BUCKET_REVIEW_SELECTOR_MAX_FEATURES) {
            while (*p != '\0' &&
                   (isspace((unsigned char)*p) || *p == ',')) {
                ++p;
            }
            if (*p == ']') {
                break;
            }
            char *number_end = NULL;
            double value = strtod(p, &number_end);
            if (number_end == p || !isfinite(value)) {
                fclose(fp);
                return 0;
            }
            weights[count++] = value;
            p = number_end;
        }
        if (count < feature_count) {
            fclose(fp);
            return 0;
        }
        memset(model, 0, sizeof(*model));
        model->enabled = 1;
        model->feature_count = (size_t)feature_count;
        model->hash_bins = (size_t)hash_bins;
        memcpy(model->weights, weights, feature_count * sizeof(weights[0]));
        fclose(fp);
        return 1;
    }
    fclose(fp);
    return 0;
}

static void arc_holo_bucket_selector_add_feature(double *features,
                                                 size_t feature_count,
                                                 size_t index,
                                                 double value)
{
    if (index < feature_count && isfinite(value)) {
        features[index] += value;
    }
}

static uint64_t arc_holo_bucket_selector_fnv1a_token(const char *text,
                                                     size_t length)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < length; ++i) {
        hash ^= (uint64_t)(unsigned char)text[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static double arc_holo_bucket_review_selector_score(
    const ARCHoloBucketReviewSelectorModel *model,
    const ARCTransformCandidate *candidate,
    size_t baseline_rank,
    size_t review_rank,
    size_t budget,
    size_t review_slots,
    int gate_open,
    double pressure_raw,
    double review_priority,
    double review_frontier_priority,
    double priority_deficit,
    double baseline_score,
    double structural_score,
    const ARCProposalTracePrior *prior)
{
    (void)review_slots;
    if (model == NULL || !model->enabled || candidate == NULL ||
        model->feature_count == 0 ||
        model->feature_count > ARC_HOLO_BUCKET_REVIEW_SELECTOR_MAX_FEATURES) {
        return 0.0;
    }
    double features[ARC_HOLO_BUCKET_REVIEW_SELECTOR_MAX_FEATURES] = {0};
    double base_rank = baseline_rank != 0 ? (double)baseline_rank : 1.0;
    double raw_rank = review_rank != 0 ? (double)review_rank : 1.0;
    double budget_value = budget != 0 ? (double)budget : 1.0;
    arc_holo_bucket_selector_add_feature(features, model->feature_count, 0, 1.0);
    arc_holo_bucket_selector_add_feature(features, model->feature_count, 1,
                                         log1p(base_rank) / 8.0);
    arc_holo_bucket_selector_add_feature(features, model->feature_count, 2,
                                         1.0 / base_rank);
    arc_holo_bucket_selector_add_feature(features, model->feature_count, 3,
                                         log1p(raw_rank) / 8.0);
    arc_holo_bucket_selector_add_feature(features, model->feature_count, 4,
                                         1.0 / raw_rank);
    arc_holo_bucket_selector_add_feature(features, model->feature_count, 5,
                                         log1p(budget_value) / 6.0);
    arc_holo_bucket_selector_add_feature(features, model->feature_count, 6,
                                         pressure_raw);
    arc_holo_bucket_selector_add_feature(features, model->feature_count, 7,
                                         review_priority / 8.0);
    arc_holo_bucket_selector_add_feature(features, model->feature_count, 8,
                                         review_frontier_priority / 8.0);
    arc_holo_bucket_selector_add_feature(features, model->feature_count, 9,
                                         priority_deficit / 8.0);
    arc_holo_bucket_selector_add_feature(
        features,
        model->feature_count,
        10,
        (review_priority - review_frontier_priority) / 8.0);
    arc_holo_bucket_selector_add_feature(features, model->feature_count, 11,
                                         baseline_score / 160.0);
    arc_holo_bucket_selector_add_feature(features, model->feature_count, 12,
                                         structural_score / 160.0);
    arc_holo_bucket_selector_add_feature(
        features,
        model->feature_count,
        13,
        (baseline_score - structural_score) / 80.0);
    arc_holo_bucket_selector_add_feature(
        features,
        model->feature_count,
        14,
        prior != NULL ? (double)prior->holo_pressure_updates / 128.0 : 0.0);
    arc_holo_bucket_selector_add_feature(
        features,
        model->feature_count,
        15,
        prior != NULL ? (double)prior->positive_rows / 128.0 : 0.0);
    arc_holo_bucket_selector_add_feature(features, model->feature_count, 16, 1.0);
    arc_holo_bucket_selector_add_feature(features, model->feature_count, 17,
                                         gate_open ? 1.0 : 0.0);
    arc_holo_bucket_selector_add_feature(features, model->feature_count, 18,
                                         (double)candidate->dx / 16.0);
    arc_holo_bucket_selector_add_feature(features, model->feature_count, 19,
                                         (double)candidate->dy / 16.0);
    arc_holo_bucket_selector_add_feature(features, model->feature_count, 20,
                                         (double)candidate->scale / 16.0);
    int cats[] = {(int)candidate->op,
                  (int)candidate->input_color,
                  (int)candidate->output_color,
                  (int)candidate->background_color,
                  (int)candidate->selector_kind,
                  (int)candidate->output_selector_kind,
                  (int)candidate->reference_color,
                  (int)candidate->render_mode,
                  (int)candidate->step_count,
                  (int)candidate->hypothesis_template_id,
                  (int)candidate->hypothesis_slot_count};
    for (size_t i = 0; i < sizeof(cats) / sizeof(cats[0]); ++i) {
        uint64_t h = (uint64_t)(i + 1u) * UINT64_C(1315423911) ^
                     (uint64_t)(unsigned)cats[i];
        arc_holo_bucket_selector_add_feature(features,
                                             model->feature_count,
                                             21u + (size_t)(h % 17u),
                                             1.0);
    }
    if (model->hash_bins != 0 && candidate->name[0] != '\0') {
        const char *start = candidate->name;
        for (const char *p = candidate->name;; ++p) {
            if (*p == '\0' || *p == '-' || *p == '_') {
                size_t length = (size_t)(p - start);
                if (length != 0) {
                    uint64_t token_hash =
                        arc_holo_bucket_selector_fnv1a_token(start, length);
                    arc_holo_bucket_selector_add_feature(
                        features,
                        model->feature_count,
                        ARC_HOLO_BUCKET_REVIEW_SELECTOR_BASE_FEATURES +
                            (size_t)(token_hash % model->hash_bins),
                        1.0);
                }
                if (*p == '\0') {
                    break;
                }
                start = p + 1;
            }
        }
    }
    double score = 0.0;
    for (size_t i = 0; i < model->feature_count; ++i) {
        score += model->weights[i] * features[i];
    }
    return score;
}

static void arc_holo_bucket_gate_features(const ARCProposalTracePrior *prior,
                                          size_t budget,
                                          double features[ARC_HOLO_BUCKET_GATE_FEATURE_COUNT])
{
    for (size_t i = 0; i < ARC_HOLO_BUCKET_GATE_FEATURE_COUNT; ++i) {
        features[i] = 0.0;
    }
    if (prior == NULL) {
        return;
    }
    size_t review_slots = arc_holo_bucket_review_slots_value(prior, budget);
    double pressure_updates = (double)prior->holo_pressure_updates;
    double observed_tasks = (double)prior->observed_tasks;
    double positive_rate =
        prior->observed_rows == 0
            ? 0.0
            : (double)prior->positive_rows / (double)prior->observed_rows;
    double gate_positive_rate =
        prior->holo_bucket_gate_updates == 0
            ? 0.0
            : (double)prior->holo_bucket_gate_positive_updates /
                  (double)prior->holo_bucket_gate_updates;
    double gate_target_mean =
        prior->holo_bucket_gate_updates == 0
            ? 0.0
            : prior->holo_bucket_gate_target_sum /
                  (double)prior->holo_bucket_gate_updates;
    features[0] = budget >= 64u ? 1.0 : (double)budget / 64.0;
    features[1] = log1p(pressure_updates) / log1p(1024.0);
    features[2] = review_slots >= 4u ? 1.0 : (double)review_slots / 4.0;
    features[3] = observed_tasks >= 400.0 ? 1.0 : observed_tasks / 400.0;
    features[4] = positive_rate;
    features[5] = arc_holo_bucket_fixed_gate_open(prior, budget) ? 1.0 : 0.0;
    features[6] = gate_positive_rate;
    features[7] = arc_holo_bucket_clamp_unit(gate_target_mean);
}

static double arc_holo_bucket_learned_gate_logit(const ARCProposalTracePrior *prior,
                                                 size_t budget)
{
    if (prior == NULL) {
        return -INFINITY;
    }
    double features[ARC_HOLO_BUCKET_GATE_FEATURE_COUNT];
    arc_holo_bucket_gate_features(prior, budget, features);
    double logit = prior->holo_bucket_gate_bias;
    for (size_t i = 0; i < ARC_HOLO_BUCKET_GATE_FEATURE_COUNT; ++i) {
        logit += prior->holo_bucket_gate_weights[i] * features[i];
    }
    return logit;
}

static double arc_holo_bucket_learned_gate_score(const ARCProposalTracePrior *prior,
                                                 size_t budget)
{
    double logit = arc_holo_bucket_learned_gate_logit(prior, budget);
    if (logit >= 40.0) {
        return 1.0;
    }
    if (logit <= -40.0) {
        return 0.0;
    }
    return 1.0 / (1.0 + exp(-logit));
}

static int arc_holo_bucket_gate_open(const ARCProposalTracePrior *prior,
                                     size_t budget)
{
    if (!arc_holo_bucket_fixed_gate_open(prior, budget)) {
        return 0;
    }
    if (prior == NULL || !prior->holo_bucket_learned_gate_enabled ||
        prior->holo_bucket_gate_updates <
            prior->holo_bucket_learned_gate_min_examples) {
        return 1;
    }
    return arc_holo_bucket_learned_gate_score(prior, budget) >=
           prior->holo_bucket_learned_gate_threshold;
}

static void arc_holo_bucket_learned_gate_update(ARCProposalTracePrior *prior,
                                                size_t budget,
                                                double target,
                                                double margin)
{
    if (prior == NULL || !prior->holo_bucket_learned_gate_enabled) {
        return;
    }
    target = arc_holo_bucket_clamp_unit(target);
    double features[ARC_HOLO_BUCKET_GATE_FEATURE_COUNT];
    arc_holo_bucket_gate_features(prior, budget, features);
    double prediction = arc_holo_bucket_learned_gate_score(prior, budget);
    double error = target - prediction;
    const double learning_rate = 0.08;
    double sample_weight = target > 0.0 ? 1.0 + 4.0 * target : 1.0;
    prior->holo_bucket_gate_bias += learning_rate * sample_weight * error;
    for (size_t i = 0; i < ARC_HOLO_BUCKET_GATE_FEATURE_COUNT; ++i) {
        prior->holo_bucket_gate_weights[i] +=
            learning_rate * sample_weight * error * features[i];
    }
    ++prior->holo_bucket_gate_updates;
    prior->holo_bucket_gate_positive_updates += target > 0.0 ? 1u : 0u;
    prior->holo_bucket_gate_target_sum += target;
    if (margin > 0.0) {
        prior->holo_bucket_gate_margin_sum += margin;
    }
}

static ARCHoloBucketGateSurface arc_holo_bucket_gate_surface(
    ARCProposalTracePrior *prior,
    const ARCTask *task,
    const ARCTransformCandidate *candidates,
    const ARCProgramEval *evals,
    size_t candidate_count,
    const ARCProgramTaskFeatures *task_features,
    size_t budget)
{
    ARCHoloBucketGateSurface surface;
    memset(&surface, 0, sizeof(surface));
    if (prior == NULL || task == NULL || candidates == NULL || evals == NULL ||
        task_features == NULL || candidate_count == 0 ||
        candidate_count > ARC_RULE_CANDIDATE_COUNT) {
        return surface;
    }
    ARCProgramRank baseline_ranks[ARC_RULE_CANDIDATE_COUNT];
    size_t baseline_rank[ARC_RULE_CANDIDATE_COUNT];
    for (size_t i = 0; i < candidate_count; ++i) {
        baseline_rank[i] = 0;
    }
    arc_build_program_ranks(task,
                            candidates,
                            evals,
                            candidate_count,
                            0,
                            ARC_PROGRAM_STRATEGY_TRACE_TASK_PRIOR,
                            NULL,
                            prior,
                            task_features,
                            baseline_ranks);
    for (size_t rank = 0; rank < candidate_count; ++rank) {
        size_t index = baseline_ranks[rank].candidate_index;
        if (index < candidate_count) {
            baseline_rank[index] = rank + 1u;
            if (budget != 0 && rank < budget &&
                arc_program_eval_is_exact_success(&evals[index])) {
                ++surface.baseline_exact_in_budget;
            }
        }
    }
    size_t review_slots = arc_holo_bucket_review_slots_value(prior, budget);
    if (budget == 0 || review_slots == 0 ||
        !arc_holo_bucket_fixed_gate_open(prior, budget)) {
        return surface;
    }
    ARCProgramRank pressure_ranks[ARC_RULE_CANDIDATE_COUNT];
    for (size_t i = 0; i < candidate_count; ++i) {
        double baseline_score =
            arc_task_conditioned_program_score(task, &candidates[i]) +
            arc_proposal_trace_task_prior_score(prior,
                                                &candidates[i],
                                                task_features);
        double pressure_raw =
            baseline_rank[i] > budget
                ? arc_proposal_trace_holo_pressure_raw(prior,
                                                       &candidates[i],
                                                       task_features)
                : 0.0;
        pressure_ranks[i].candidate_index = i;
        pressure_ranks[i].primary =
            arc_holo_bucket_review_priority(pressure_raw, baseline_score);
        pressure_ranks[i].secondary = pressure_raw;
        pressure_ranks[i].tie_breaker =
            arc_hash_mix(arc_task_hash(task), evals[i].program_hash ^ (uint64_t)i);
    }
    for (size_t i = 1; i < candidate_count; ++i) {
        ARCProgramRank key = pressure_ranks[i];
        size_t j = i;
        while (j > 0) {
            ARCProgramRank prev = pressure_ranks[j - 1u];
            int before = key.primary > prev.primary ||
                         (key.primary == prev.primary &&
                          key.secondary > prev.secondary) ||
                         (key.primary == prev.primary &&
                          key.secondary == prev.secondary &&
                          key.tie_breaker < prev.tie_breaker);
            if (!before) {
                break;
            }
            pressure_ranks[j] = pressure_ranks[j - 1u];
            --j;
        }
        pressure_ranks[j] = key;
    }
    size_t review_limit = review_slots < candidate_count ? review_slots : candidate_count;
    if (review_limit != 0 && pressure_ranks[review_limit - 1u].primary > 0.0) {
        surface.review_frontier_priority = pressure_ranks[review_limit - 1u].primary;
    }
    for (size_t rank = 0; rank < candidate_count; ++rank) {
        size_t index = pressure_ranks[rank].candidate_index;
        double pressure_priority = pressure_ranks[rank].primary;
        double pressure_raw = pressure_ranks[rank].secondary;
        if (index >= candidate_count || pressure_raw <= 0.0 ||
            baseline_rank[index] <= budget) {
            continue;
        }
        ++surface.pressure_candidates;
        if (pressure_raw > surface.best_pressure_raw) {
            surface.best_pressure_raw = pressure_raw;
        }
        if (!arc_program_eval_is_exact_success(&evals[index])) {
            continue;
        }
        ++surface.exact_pressure_candidates;
        if (pressure_raw > surface.best_exact_pressure_raw) {
            surface.best_exact_pressure_raw = pressure_raw;
        }
        if (pressure_priority > surface.best_exact_review_priority) {
            surface.best_exact_review_priority = pressure_priority;
        }
        if (rank < review_slots) {
            ++surface.review_exact_candidates;
        }
    }
    if (surface.review_exact_candidates != 0) {
        surface.review_margin =
            surface.best_exact_review_priority - surface.review_frontier_priority;
        if (surface.review_margin < 0.0) {
            surface.review_margin = 0.0;
        }
        double margin_score =
            arc_holo_bucket_clamp_unit(surface.review_margin /
                                       (fabs(surface.review_frontier_priority) + 1.0));
        surface.soft_target = 0.65 + 0.35 * margin_score;
        if (surface.baseline_exact_in_budget != 0) {
            surface.soft_target *= 0.85;
        }
    }
    return surface;
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
    uint64_t task_hash = arc_task_hash(task);
    double holo_context[ARC_PROGRAM_FEATURE_COUNT] = {0};
    int have_holo_context = 0;
    double baseline_primary[ARC_RULE_CANDIDATE_COUNT] = {0};
    size_t baseline_rank[ARC_RULE_CANDIDATE_COUNT] = {0};
    size_t pressure_rank[ARC_RULE_CANDIDATE_COUNT] = {0};
    double pressure_raw[ARC_RULE_CANDIDATE_COUNT] = {0};
    double pressure_review_priority[ARC_RULE_CANDIDATE_COUNT] = {0};
    double exact_budget_floor = INFINITY;
    int have_exact_budget_floor = 0;
    ARCProgramRank baseline_ranks[ARC_RULE_CANDIDATE_COUNT];
    if (strategy == ARC_PROGRAM_STRATEGY_TRACE_HOLO_RESIDUAL &&
        trace_prior != NULL && trace_prior->holo_residual != NULL &&
        task_features != NULL && trace_prior->holo_updates != 0) {
        double holo_state[ARC_PROGRAM_FEATURE_COUNT] = {0};
        arc_trace_holo_state_vector(task_features, holo_state);
        have_holo_context =
            holo_ranker_recall(trace_prior->holo_residual,
                               holo_state,
                               holo_context) == 0;
    }
    if ((strategy == ARC_PROGRAM_STRATEGY_TRACE_HOLO_RESIDUAL ||
         strategy == ARC_PROGRAM_STRATEGY_TRACE_HOLO_BUCKET) &&
        candidate_count <= ARC_RULE_CANDIDATE_COUNT) {
        for (size_t i = 0; i < candidate_count; ++i) {
            baseline_primary[i] =
                arc_task_conditioned_program_score(task, &candidates[i]) +
                arc_proposal_trace_task_prior_score(trace_prior,
                                                    &candidates[i],
                                                    task_features);
            baseline_ranks[i].candidate_index = i;
            baseline_ranks[i].primary = baseline_primary[i];
            baseline_ranks[i].secondary =
                arc_task_conditioned_program_score(task, &candidates[i]);
            baseline_ranks[i].tie_breaker =
                arc_hash_mix(task_hash, evals[i].program_hash ^ (uint64_t)i);
        }
        for (size_t i = 1; i < candidate_count; ++i) {
            ARCProgramRank key = baseline_ranks[i];
            size_t j = i;
            while (j > 0) {
                ARCProgramRank prev = baseline_ranks[j - 1u];
                int before = key.primary > prev.primary ||
                             (key.primary == prev.primary &&
                              key.secondary > prev.secondary) ||
                             (key.primary == prev.primary &&
                              key.secondary == prev.secondary &&
                              key.tie_breaker < prev.tie_breaker);
                if (!before) {
                    break;
                }
                baseline_ranks[j] = baseline_ranks[j - 1u];
                --j;
            }
            baseline_ranks[j] = key;
        }
        for (size_t rank = 0; rank < candidate_count; ++rank) {
            size_t index = baseline_ranks[rank].candidate_index;
            if (index >= candidate_count) {
                continue;
            }
            baseline_rank[index] = rank + 1u;
            if (budget != 0 && rank < budget &&
                arc_program_eval_is_exact_success(&evals[index])) {
                if (!have_exact_budget_floor ||
                    baseline_ranks[rank].primary < exact_budget_floor) {
                    exact_budget_floor = baseline_ranks[rank].primary;
                    have_exact_budget_floor = 1;
                }
            }
        }
        if (strategy == ARC_PROGRAM_STRATEGY_TRACE_HOLO_BUCKET &&
            trace_prior != NULL && trace_prior->holo_pressure != NULL &&
            arc_holo_bucket_gate_open(trace_prior, budget) &&
            task_features != NULL) {
            ARCProgramRank pressure_ranks[ARC_RULE_CANDIDATE_COUNT];
            for (size_t i = 0; i < candidate_count; ++i) {
                pressure_raw[i] =
                    baseline_rank[i] > budget
                        ? arc_proposal_trace_holo_pressure_raw(trace_prior,
                                                               &candidates[i],
                                                               task_features)
                        : 0.0;
                pressure_review_priority[i] =
                    arc_holo_bucket_review_priority(pressure_raw[i],
                                                    baseline_primary[i]);
                pressure_ranks[i].candidate_index = i;
                pressure_ranks[i].primary = pressure_review_priority[i];
                pressure_ranks[i].secondary = pressure_raw[i];
                pressure_ranks[i].tie_breaker =
                    arc_hash_mix(task_hash, evals[i].program_hash ^ (uint64_t)i);
            }
            for (size_t i = 1; i < candidate_count; ++i) {
                ARCProgramRank key = pressure_ranks[i];
                size_t j = i;
                while (j > 0) {
                    ARCProgramRank prev = pressure_ranks[j - 1u];
                    int before = key.primary > prev.primary ||
                                 (key.primary == prev.primary &&
                                  key.secondary > prev.secondary) ||
                                 (key.primary == prev.primary &&
                                  key.secondary == prev.secondary &&
                                  key.tie_breaker < prev.tie_breaker);
                    if (!before) {
                        break;
                    }
                    pressure_ranks[j] = pressure_ranks[j - 1u];
                    --j;
                }
                pressure_ranks[j] = key;
            }
            for (size_t rank = 0; rank < candidate_count; ++rank) {
                size_t index = pressure_ranks[rank].candidate_index;
                if (index < candidate_count && pressure_ranks[rank].primary > 0.0) {
                    pressure_rank[index] = rank + 1u;
                }
            }
        }
    }
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
        case ARC_PROGRAM_STRATEGY_TRACE_HOLO_RESIDUAL:
            ranks[i].primary =
                ranks[i].secondary +
                arc_proposal_trace_holo_residual_score(trace_prior,
                                                       &candidates[i],
                                                       task_features,
                                                       have_holo_context ? holo_context : NULL);
            if (trace_prior != NULL && trace_prior->holo_pressure_enabled &&
                trace_prior->holo_pressure != NULL &&
                trace_prior->holo_pressure_updates != 0 && task_features != NULL) {
                double pressure_state[ARC_PROGRAM_FEATURE_COUNT] = {0};
                double pressure_context[ARC_PROGRAM_FEATURE_COUNT] = {0};
                if (arc_trace_holo_pressure_state_vector(&candidates[i],
                                                         task_features,
                                                         pressure_state) &&
                    holo_partitioned_ranker_recall(trace_prior->holo_pressure,
                                                   pressure_state,
                                                   pressure_context) == 0) {
                    ranks[i].primary +=
                        arc_proposal_trace_holo_pressure_residual(
                            trace_prior,
                            &candidates[i],
                            task_features,
                            pressure_context,
                            baseline_rank[i],
                            budget,
                            baseline_primary[i],
                            have_exact_budget_floor ? exact_budget_floor : NAN);
                }
            }
            break;
        case ARC_PROGRAM_STRATEGY_TRACE_HOLO_BUCKET: {
            ranks[i].primary = baseline_primary[i];
            size_t review_slots =
                arc_holo_bucket_gate_open(trace_prior, budget)
                    ? arc_holo_bucket_review_slots_value(trace_prior, budget)
                    : 0u;
            if (budget != 0 && baseline_rank[i] != 0 &&
                baseline_rank[i] <= budget &&
                arc_program_eval_is_exact_success(&evals[i])) {
                ranks[i].primary = 1000000000.0 - (double)baseline_rank[i];
            } else if (review_slots != 0 && baseline_rank[i] > budget &&
                       pressure_rank[i] != 0 && pressure_rank[i] <= review_slots &&
                       pressure_raw[i] > 0.0) {
                ranks[i].primary = 500000000.0 + pressure_raw[i];
            }
            break;
        }
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

static void arc_holo_bucket_learned_gate_observe_task(
    ARCProposalTracePrior *prior,
    const ARCTask *task,
    const ARCTransformCandidate *candidates,
    const ARCProgramEval *evals,
    size_t candidate_count,
    const ARCProgramTaskFeatures *task_features,
    const size_t *budgets,
    size_t budget_count)
{
    if (prior == NULL || !prior->holo_bucket_learned_gate_enabled ||
        task == NULL || candidates == NULL || evals == NULL ||
        task_features == NULL || budgets == NULL || budget_count == 0 ||
        candidate_count == 0 || candidate_count > ARC_RULE_CANDIDATE_COUNT) {
        return;
    }
    for (size_t bi = 0; bi < budget_count; ++bi) {
        size_t budget = budgets[bi];
        ARCHoloBucketGateSurface surface =
            arc_holo_bucket_gate_surface(prior,
                                         task,
                                         candidates,
                                         evals,
                                         candidate_count,
                                         task_features,
                                         budget);
        arc_holo_bucket_learned_gate_update(prior,
                                            budget,
                                            surface.soft_target,
                                            surface.review_margin);
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
