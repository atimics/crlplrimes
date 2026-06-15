void nn_numeric_guard_config_init(NNNumericGuardConfig *config)
{
    if (config == NULL) {
        return;
    }

    config->require_finite = 1;
    config->max_shrink_steps = 4u;
    config->shrink_factor = (nn_real)0.5;
    config->max_abs_parameter = (nn_real)0;
    config->max_abs_parameter_delta = (nn_real)0;
    config->max_abs_logit = (nn_real)0;
    config->max_abs_logit_delta = (nn_real)0;
    config->min_selected_margin = (nn_real)0;
}

void nn_training_step_certificate_init(NNTrainingStepCertificate *certificate)
{
    if (certificate == NULL) {
        return;
    }

    memset(certificate, 0, sizeof(*certificate));
    snprintf(certificate->schema,
             sizeof(certificate->schema),
             "%s",
             NN_NUMERIC_STEP_CERTIFICATE_SCHEMA);
    snprintf(certificate->step_kind, sizeof(certificate->step_kind), "%s", "candidate_scorer");
    certificate->report_version = 1u;
    certificate->rejection_reason = NN_NUMERIC_STEP_REJECT_NONE;
    certificate->optimizer_loss = (nn_real)NAN;
    certificate->loss_before = (nn_real)NAN;
    certificate->loss_after = (nn_real)NAN;
    certificate->selected_margin_before = (nn_real)NAN;
    certificate->selected_margin_after = (nn_real)NAN;
}

const char *nn_numeric_step_reject_reason_name(NNNumericStepRejectReason reason)
{
    switch (reason) {
    case NN_NUMERIC_STEP_REJECT_NONE:
        return "none";
    case NN_NUMERIC_STEP_REJECT_INVALID_ARGUMENT:
        return "invalid_argument";
    case NN_NUMERIC_STEP_REJECT_NONFINITE:
        return "nonfinite";
    case NN_NUMERIC_STEP_REJECT_PARAMETER_BOUND:
        return "parameter_bound";
    case NN_NUMERIC_STEP_REJECT_PARAMETER_DELTA_BOUND:
        return "parameter_delta_bound";
    case NN_NUMERIC_STEP_REJECT_LOGIT_BOUND:
        return "logit_bound";
    case NN_NUMERIC_STEP_REJECT_LOGIT_DELTA_BOUND:
        return "logit_delta_bound";
    case NN_NUMERIC_STEP_REJECT_MARGIN_BOUND:
        return "margin_bound";
    case NN_NUMERIC_STEP_REJECT_ALLOC_FAILED:
        return "alloc_failed";
    default:
        return "unknown";
    }
}

static nn_real nn_abs_real(nn_real value)
{
    return value < (nn_real)0 ? -value : value;
}

static int nn_parameter_metrics(const NeuralNet *net, nn_real *max_abs)
{
    if (max_abs != NULL) {
        *max_abs = (nn_real)0;
    }
    if (net == NULL) {
        return 0;
    }

    int finite = 1;
    nn_real observed = (nn_real)0;
    for (size_t layer = 1; layer < net->layer_count; ++layer) {
        size_t previous = net->sizes[layer - 1];
        size_t current = net->sizes[layer];
        size_t weight_count = previous * current;
        for (size_t i = 0; i < weight_count; ++i) {
            nn_real value = net->weights[layer][i];
            if (!isfinite((double)value)) {
                finite = 0;
            } else if (nn_abs_real(value) > observed) {
                observed = nn_abs_real(value);
            }
        }
        for (size_t i = 0; i < current; ++i) {
            nn_real value = net->biases[layer][i];
            if (!isfinite((double)value)) {
                finite = 0;
            } else if (nn_abs_real(value) > observed) {
                observed = nn_abs_real(value);
            }
        }
    }
    if (max_abs != NULL) {
        *max_abs = observed;
    }
    return finite;
}

static nn_real nn_parameter_max_abs_delta(const NeuralNet *before, const NeuralNet *after)
{
    if (before == NULL || after == NULL || before->layer_count != after->layer_count) {
        return (nn_real)NAN;
    }

    nn_real observed = (nn_real)0;
    for (size_t layer = 1; layer < before->layer_count; ++layer) {
        if (before->sizes[layer] != after->sizes[layer] ||
            before->sizes[layer - 1] != after->sizes[layer - 1]) {
            return (nn_real)NAN;
        }
        size_t previous = before->sizes[layer - 1];
        size_t current = before->sizes[layer];
        size_t weight_count = previous * current;
        for (size_t i = 0; i < weight_count; ++i) {
            nn_real delta = nn_abs_real(after->weights[layer][i] - before->weights[layer][i]);
            if (!isfinite((double)delta)) {
                return (nn_real)NAN;
            }
            if (delta > observed) {
                observed = delta;
            }
        }
        for (size_t i = 0; i < current; ++i) {
            nn_real delta = nn_abs_real(after->biases[layer][i] - before->biases[layer][i]);
            if (!isfinite((double)delta)) {
                return (nn_real)NAN;
            }
            if (delta > observed) {
                observed = delta;
            }
        }
    }
    return observed;
}

static int nn_copy_parameters(NeuralNet *dst, const NeuralNet *src)
{
    if (dst == NULL || src == NULL || dst->layer_count != src->layer_count) {
        return 0;
    }

    for (size_t layer = 1; layer < src->layer_count; ++layer) {
        if (dst->sizes[layer] != src->sizes[layer] ||
            dst->sizes[layer - 1] != src->sizes[layer - 1]) {
            return 0;
        }
        size_t previous = src->sizes[layer - 1];
        size_t current = src->sizes[layer];
        memcpy(dst->biases[layer], src->biases[layer], current * sizeof(*dst->biases[layer]));
        memcpy(dst->weights[layer],
               src->weights[layer],
               current * previous * sizeof(*dst->weights[layer]));
    }
    dst->rng_state = src->rng_state;
    invalidate_batch_cache(dst);
    return 1;
}

static int nn_score_metrics(NeuralNet *net,
                            const nn_real *candidate_inputs,
                            size_t candidate_count,
                            size_t selected_index,
                            nn_real *scores,
                            nn_real *max_abs_logit,
                            nn_real *selected_margin,
                            nn_real *loss,
                            nn_real weight)
{
    if (max_abs_logit != NULL) {
        *max_abs_logit = (nn_real)0;
    }
    if (selected_margin != NULL) {
        *selected_margin = (nn_real)NAN;
    }
    if (loss != NULL) {
        *loss = (nn_real)NAN;
    }
    if (net == NULL || candidate_inputs == NULL || candidate_count == 0 ||
        selected_index >= candidate_count || scores == NULL || nn_output_size(net) != 1) {
        return 0;
    }

    nn_forward_batch(net, candidate_inputs, scores, candidate_count);
    int finite = 1;
    nn_real observed = (nn_real)0;
    for (size_t i = 0; i < candidate_count; ++i) {
        if (!isfinite((double)scores[i])) {
            finite = 0;
            continue;
        }
        nn_real abs_score = nn_abs_real(scores[i]);
        if (abs_score > observed) {
            observed = abs_score;
        }
    }
    if (max_abs_logit != NULL) {
        *max_abs_logit = observed;
    }
    if (selected_margin != NULL) {
        *selected_margin = nn_action_score_margin(scores, candidate_count, selected_index);
    }
    if (loss != NULL) {
        *loss = weight * softmax_cross_entropy_index(scores, candidate_count, selected_index);
    }
    return finite && selected_margin != NULL && isfinite((double)*selected_margin) &&
           loss != NULL && isfinite((double)*loss);
}

static int nn_dataset_metrics(NeuralNet *net,
                              const nn_real *inputs,
                              const nn_real *targets,
                              size_t sample_count,
                              nn_real *outputs,
                              nn_real *loss,
                              nn_real *max_abs_output)
{
    if (loss != NULL) {
        *loss = (nn_real)NAN;
    }
    if (max_abs_output != NULL) {
        *max_abs_output = (nn_real)0;
    }
    if (net == NULL || inputs == NULL || targets == NULL || sample_count == 0 ||
        outputs == NULL || loss == NULL || max_abs_output == NULL) {
        return 0;
    }

    size_t output_width = nn_output_size(net);
    if (output_width == 0) {
        return 0;
    }

    nn_forward_batch(net, inputs, outputs, sample_count);
    int finite = 1;
    nn_real total_loss = (nn_real)0;
    nn_real observed = (nn_real)0;
    for (size_t sample = 0; sample < sample_count; ++sample) {
        const nn_real *row_output = &outputs[sample * output_width];
        const nn_real *row_target = &targets[sample * output_width];
        for (size_t i = 0; i < output_width; ++i) {
            if (!isfinite((double)row_output[i]) || !isfinite((double)row_target[i])) {
                finite = 0;
            } else if (nn_abs_real(row_output[i]) > observed) {
                observed = nn_abs_real(row_output[i]);
            }
        }

        if (net->loss == NN_LOSS_SOFTMAX_CROSS_ENTROPY) {
            total_loss += nn_softmax_cross_entropy(row_output, row_target, output_width);
        } else {
            for (size_t i = 0; i < output_width; ++i) {
                nn_real error = row_output[i] - row_target[i];
                total_loss += (nn_real)0.5 * error * error;
            }
        }
    }

    *loss = total_loss / (nn_real)sample_count;
    *max_abs_output = observed;
    return finite && isfinite((double)*loss);
}

static nn_real nn_policy_regularized_objective(const nn_real *scores,
                                               size_t candidate_count,
                                               size_t selected_index,
                                               nn_real policy_weight,
                                               const nn_real *reference_probabilities,
                                               nn_real reference_weight,
                                               nn_real entropy_weight)
{
    if (scores == NULL || candidate_count == 0 || selected_index >= candidate_count ||
        reference_weight < (nn_real)0 || entropy_weight < (nn_real)0 ||
        (reference_weight != (nn_real)0 && reference_probabilities == NULL)) {
        return (nn_real)NAN;
    }

    nn_real loss = (nn_real)0;
    if (policy_weight != (nn_real)0) {
        loss += policy_weight * softmax_cross_entropy_index(scores,
                                                            candidate_count,
                                                            selected_index);
    }
    if (reference_weight != (nn_real)0) {
        loss += reference_weight *
                nn_softmax_cross_entropy(scores, reference_probabilities, candidate_count);
    }
    if (entropy_weight != (nn_real)0) {
        nn_real max_score = scores[0];
        for (size_t i = 1; i < candidate_count; ++i) {
            if (scores[i] > max_score) {
                max_score = scores[i];
            }
        }
        nn_real denominator = (nn_real)0;
        for (size_t i = 0; i < candidate_count; ++i) {
            denominator += NN_EXP(scores[i] - max_score);
        }
        if (denominator <= (nn_real)0 || !isfinite((double)denominator)) {
            return (nn_real)NAN;
        }
        nn_real entropy_loss = (nn_real)0;
        for (size_t i = 0; i < candidate_count; ++i) {
            nn_real probability = NN_EXP(scores[i] - max_score) / denominator;
            entropy_loss += probability * safe_log_probability(probability);
        }
        loss += entropy_weight * entropy_loss;
    }
    return loss;
}

static nn_real nn_score_max_abs_delta(const nn_real *before,
                                      const nn_real *after,
                                      size_t count)
{
    if (before == NULL || after == NULL) {
        return (nn_real)NAN;
    }
    nn_real observed = (nn_real)0;
    for (size_t i = 0; i < count; ++i) {
        nn_real delta = nn_abs_real(after[i] - before[i]);
        if (!isfinite((double)delta)) {
            return (nn_real)NAN;
        }
        if (delta > observed) {
            observed = delta;
        }
    }
    return observed;
}

static int nn_guard_bound_active(nn_real value)
{
    return isfinite((double)value) && value > (nn_real)0;
}

static NNNumericStepRejectReason nn_numeric_guard_check(
    const NNNumericGuardConfig *guard,
    const NNTrainingStepCertificate *certificate)
{
    if (guard == NULL || certificate == NULL) {
        return NN_NUMERIC_STEP_REJECT_INVALID_ARGUMENT;
    }
    if (guard->require_finite &&
        (!certificate->finite_before || !certificate->finite_after ||
         !isfinite((double)certificate->loss_before) ||
         !isfinite((double)certificate->loss_after) ||
         !isfinite((double)certificate->optimizer_loss) ||
         !isfinite((double)certificate->max_abs_parameter_delta) ||
         !isfinite((double)certificate->max_abs_logit_delta) ||
         !isfinite((double)certificate->selected_margin_before) ||
         !isfinite((double)certificate->selected_margin_after))) {
        return NN_NUMERIC_STEP_REJECT_NONFINITE;
    }
    if (nn_guard_bound_active(guard->max_abs_parameter) &&
        certificate->max_abs_parameter_after > guard->max_abs_parameter) {
        return NN_NUMERIC_STEP_REJECT_PARAMETER_BOUND;
    }
    if (nn_guard_bound_active(guard->max_abs_parameter_delta) &&
        certificate->max_abs_parameter_delta > guard->max_abs_parameter_delta) {
        return NN_NUMERIC_STEP_REJECT_PARAMETER_DELTA_BOUND;
    }
    if (nn_guard_bound_active(guard->max_abs_logit) &&
        certificate->max_abs_logit_after > guard->max_abs_logit) {
        return NN_NUMERIC_STEP_REJECT_LOGIT_BOUND;
    }
    if (nn_guard_bound_active(guard->max_abs_logit_delta) &&
        certificate->max_abs_logit_delta > guard->max_abs_logit_delta) {
        return NN_NUMERIC_STEP_REJECT_LOGIT_DELTA_BOUND;
    }
    if (nn_guard_bound_active(guard->min_selected_margin) &&
        certificate->selected_margin_after < guard->min_selected_margin) {
        return NN_NUMERIC_STEP_REJECT_MARGIN_BOUND;
    }
    return NN_NUMERIC_STEP_REJECT_NONE;
}

static int nn_write_json_real(FILE *fp, nn_real value)
{
    if (fp == NULL) {
        return 0;
    }
    if (!isfinite((double)value)) {
        return fputs("null", fp) != EOF;
    }
    return fprintf(fp, "%.17g", (double)value) >= 0;
}

int nn_training_step_certificate_write_jsonl(FILE *fp,
                                             const NNTrainingStepCertificate *certificate)
{
    if (fp == NULL || certificate == NULL) {
        return 0;
    }

    if (fputs("{\"schema\":", fp) == EOF ||
        !nn_write_json_string(fp,
                              certificate->schema[0] == '\0'
                                  ? NN_NUMERIC_STEP_CERTIFICATE_SCHEMA
                                  : certificate->schema) ||
        fputs(",\"step_kind\":", fp) == EOF ||
        !nn_write_json_string(fp,
                              certificate->step_kind[0] == '\0'
                                  ? "candidate_scorer"
                                  : certificate->step_kind) ||
        fprintf(fp,
                ",\"report_version\":%u,\"step_committed\":%s,"
                "\"rejection_reason\":",
                certificate->report_version,
                certificate->step_committed ? "true" : "false") < 0 ||
        !nn_write_json_string(fp,
                              nn_numeric_step_reject_reason_name(
                                  certificate->rejection_reason)) ||
        fprintf(fp,
                ",\"shrink_steps\":%zu,\"parameter_count\":%zu,"
                "\"candidate_count\":%zu,\"selected_index\":%zu,"
                "\"finite_before\":%s,\"finite_after\":%s",
                certificate->shrink_steps,
                certificate->parameter_count,
                certificate->candidate_count,
                certificate->selected_index,
                certificate->finite_before ? "true" : "false",
                certificate->finite_after ? "true" : "false") < 0) {
        return 0;
    }

    const struct {
        const char *name;
        nn_real value;
    } fields[] = {
        {"attempted_learning_rate", certificate->attempted_learning_rate},
        {"applied_learning_rate", certificate->applied_learning_rate},
        {"optimizer_loss", certificate->optimizer_loss},
        {"loss_before", certificate->loss_before},
        {"loss_after", certificate->loss_after},
        {"max_abs_parameter_before", certificate->max_abs_parameter_before},
        {"max_abs_parameter_after", certificate->max_abs_parameter_after},
        {"max_abs_parameter_delta", certificate->max_abs_parameter_delta},
        {"max_abs_logit_before", certificate->max_abs_logit_before},
        {"max_abs_logit_after", certificate->max_abs_logit_after},
        {"max_abs_logit_delta", certificate->max_abs_logit_delta},
        {"selected_margin_before", certificate->selected_margin_before},
        {"selected_margin_after", certificate->selected_margin_after},
    };
    for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); ++i) {
        if (fprintf(fp, ",\"%s\":", fields[i].name) < 0 ||
            !nn_write_json_real(fp, fields[i].value)) {
            return 0;
        }
    }
    return fputs("}\n", fp) != EOF;
}

static const char *nn_numeric_certificate_path(void)
{
    const char *path = getenv("CRLP_NUMERIC_CERT_PATH");
    if (path == NULL || path[0] == '\0') {
        return NULL;
    }
    return path;
}

static int nn_env_parse_real(const char *name, nn_real *out)
{
    const char *text = getenv(name);
    if (text == NULL || text[0] == '\0' || out == NULL) {
        return 0;
    }

    errno = 0;
    char *end = NULL;
    double parsed = strtod(text, &end);
    if (end == text || errno == ERANGE || !isfinite(parsed)) {
        return 0;
    }
    while (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r') {
        ++end;
    }
    if (*end != '\0') {
        return 0;
    }
    *out = (nn_real)parsed;
    return 1;
}

static int nn_env_parse_size(const char *name, size_t *out)
{
    const char *text = getenv(name);
    if (text == NULL || text[0] == '\0' || out == NULL) {
        return 0;
    }

    errno = 0;
    char *end = NULL;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (end == text || errno == ERANGE || parsed > (unsigned long long)SIZE_MAX) {
        return 0;
    }
    while (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r') {
        ++end;
    }
    if (*end != '\0') {
        return 0;
    }
    *out = (size_t)parsed;
    return 1;
}

static void nn_numeric_guard_config_sanitize(NNNumericGuardConfig *config)
{
    if (config == NULL) {
        return;
    }
    if (config->shrink_factor <= (nn_real)0 || config->shrink_factor >= (nn_real)1 ||
        !isfinite((double)config->shrink_factor)) {
        config->shrink_factor = (nn_real)0.5;
    }
    if (config->max_abs_parameter < (nn_real)0 ||
        !isfinite((double)config->max_abs_parameter)) {
        config->max_abs_parameter = (nn_real)0;
    }
    if (config->max_abs_parameter_delta < (nn_real)0 ||
        !isfinite((double)config->max_abs_parameter_delta)) {
        config->max_abs_parameter_delta = (nn_real)0;
    }
    if (config->max_abs_logit < (nn_real)0 || !isfinite((double)config->max_abs_logit)) {
        config->max_abs_logit = (nn_real)0;
    }
    if (config->max_abs_logit_delta < (nn_real)0 ||
        !isfinite((double)config->max_abs_logit_delta)) {
        config->max_abs_logit_delta = (nn_real)0;
    }
    if (config->min_selected_margin < (nn_real)0 ||
        !isfinite((double)config->min_selected_margin)) {
        config->min_selected_margin = (nn_real)0;
    }
}

static void nn_numeric_guard_config_from_env(NNNumericGuardConfig *config)
{
    if (config == NULL) {
        return;
    }
    nn_numeric_guard_config_init(config);

    nn_real real_value = (nn_real)0;
    size_t size_value = 0u;
    if (nn_env_parse_size("CRLP_NUMERIC_MAX_SHRINK_STEPS", &size_value)) {
        config->max_shrink_steps = size_value;
    }
    if (nn_env_parse_real("CRLP_NUMERIC_SHRINK_FACTOR", &real_value)) {
        config->shrink_factor = real_value;
    }
    if (nn_env_parse_real("CRLP_NUMERIC_MAX_ABS_PARAMETER", &real_value)) {
        config->max_abs_parameter = real_value;
    }
    if (nn_env_parse_real("CRLP_NUMERIC_MAX_ABS_PARAMETER_DELTA", &real_value)) {
        config->max_abs_parameter_delta = real_value;
    }
    if (nn_env_parse_real("CRLP_NUMERIC_MAX_ABS_LOGIT", &real_value)) {
        config->max_abs_logit = real_value;
    }
    if (nn_env_parse_real("CRLP_NUMERIC_MAX_ABS_LOGIT_DELTA", &real_value)) {
        config->max_abs_logit_delta = real_value;
    }
    if (nn_env_parse_real("CRLP_NUMERIC_MIN_SELECTED_MARGIN", &real_value)) {
        config->min_selected_margin = real_value;
    }
    if (nn_env_parse_real("CRLP_NUMERIC_REQUIRE_FINITE", &real_value)) {
        config->require_finite = real_value != (nn_real)0;
    }
    nn_numeric_guard_config_sanitize(config);
}

static int nn_numeric_certificate_write_env(const NNTrainingStepCertificate *certificate)
{
    const char *path = nn_numeric_certificate_path();
    if (path == NULL) {
        return 1;
    }

    FILE *fp = fopen(path, "a");
    if (fp == NULL) {
        return 0;
    }
    int ok = nn_training_step_certificate_write_jsonl(fp, certificate);
    if (fclose(fp) != 0) {
        ok = 0;
    }
    return ok;
}
