#include "nn_symbolic.h"
#include "nn_train.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static void nn_copy_text(char *dst, size_t dst_size, const char *src)
{
    if (dst == NULL || dst_size == 0) {
        return;
    }
    if (src == NULL) {
        dst[0] = '\0';
        return;
    }
    snprintf(dst, dst_size, "%s", src);
}

static int nn_text_is_present(const char *text)
{
    return text != NULL && text[0] != '\0';
}

static size_t nn_popcount_u64(uint64_t value)
{
#if defined(__GNUC__) || defined(__clang__)
    return (size_t)__builtin_popcountll((unsigned long long)value);
#else
    size_t count = 0;
    while (value != 0) {
        value &= value - 1u;
        ++count;
    }
    return count;
#endif
}

void nn_certificate_clear(NNCertificate *certificate)
{
    if (certificate == NULL) {
        return;
    }
    memset(certificate, 0, sizeof(*certificate));
    certificate->status = NN_CERTIFICATE_UNCHECKED;
}

void nn_certificate_accept(NNCertificate *certificate,
                           const char *proof_kind,
                           const char *reason,
                           size_t checked_cases,
                           size_t support)
{
    if (certificate == NULL) {
        return;
    }
    nn_certificate_clear(certificate);
    certificate->status = NN_CERTIFICATE_CERTIFIED;
    certificate->deterministic = 1;
    certificate->checked_cases = checked_cases;
    certificate->support = support;
    certificate->false_rejections = 0;
    nn_copy_text(certificate->proof_kind, sizeof(certificate->proof_kind), proof_kind);
    nn_copy_text(certificate->reason, sizeof(certificate->reason), reason);
}

void nn_certificate_reject(NNCertificate *certificate,
                           const char *proof_kind,
                           const char *reason,
                           size_t checked_cases,
                           size_t false_rejections)
{
    if (certificate == NULL) {
        return;
    }
    nn_certificate_clear(certificate);
    certificate->status = NN_CERTIFICATE_REJECTED;
    certificate->deterministic = 1;
    certificate->checked_cases = checked_cases;
    certificate->false_rejections = false_rejections;
    nn_copy_text(certificate->proof_kind, sizeof(certificate->proof_kind), proof_kind);
    nn_copy_text(certificate->reason, sizeof(certificate->reason), reason);
}

const char *nn_certificate_status_name(NNCertificateStatus status)
{
    switch (status) {
    case NN_CERTIFICATE_UNCHECKED:
        return "unchecked";
    case NN_CERTIFICATE_REJECTED:
        return "rejected";
    case NN_CERTIFICATE_CERTIFIED:
        return "certified";
    }
    return "unknown";
}

void nn_rule_clear(NNRule *rule)
{
    if (rule == NULL) {
        return;
    }
    memset(rule, 0, sizeof(*rule));
    nn_certificate_clear(&rule->certificate);
}

void nn_rule_init(NNRule *rule,
                  const char *domain,
                  const char *name,
                  NNRuleKind kind,
                  uint32_t template_id)
{
    if (rule == NULL) {
        return;
    }
    nn_rule_clear(rule);
    nn_copy_text(rule->domain, sizeof(rule->domain), domain);
    nn_copy_text(rule->name, sizeof(rule->name), name);
    rule->kind = kind;
    rule->template_id = template_id;
}

int nn_rule_add_param(NNRule *rule, const char *name, double value, int64_t integer)
{
    if (rule == NULL || name == NULL || rule->param_count >= NN_RULE_MAX_PARAMS) {
        return 0;
    }
    NNRuleParam *param = &rule->params[rule->param_count++];
    nn_copy_text(param->name, sizeof(param->name), name);
    param->value = value;
    param->integer = integer;
    return 1;
}

int nn_rule_get_param(const NNRule *rule, const char *name, double *value_out)
{
    if (rule == NULL || name == NULL || value_out == NULL) {
        return 0;
    }
    for (size_t i = 0; i < rule->param_count; ++i) {
        if (strcmp(rule->params[i].name, name) == 0) {
            *value_out = rule->params[i].value;
            return 1;
        }
    }
    return 0;
}

int nn_rule_get_param_i64(const NNRule *rule, const char *name, int64_t *value_out)
{
    if (rule == NULL || name == NULL || value_out == NULL) {
        return 0;
    }
    for (size_t i = 0; i < rule->param_count; ++i) {
        if (strcmp(rule->params[i].name, name) == 0) {
            *value_out = rule->params[i].integer;
            return 1;
        }
    }
    return 0;
}

int nn_rule_add_feature(NNRule *rule, nn_real value)
{
    if (rule == NULL || rule->structural_feature_count >= NN_RULE_MAX_FEATURES) {
        return 0;
    }
    rule->structural_features[rule->structural_feature_count++] = value;
    return 1;
}

NNFeatureAudit nn_symbolic_domain_rule_feature_audit(const NNSymbolicDomain *domain)
{
    if (domain == NULL) {
        NNFeatureAudit audit = {0};
        audit.scorer_input_is_policy_compliant = 0;
        audit.has_warning = 1;
        return audit;
    }
    return nn_feature_schema_audit(domain->rule_feature_schema,
                                   domain->rule_feature_schema_count,
                                   domain->leakage_policy);
}

NNFeatureAudit nn_symbolic_domain_scorer_feature_audit(const NNSymbolicDomain *domain)
{
    if (domain == NULL) {
        NNFeatureAudit audit = {0};
        audit.scorer_input_is_policy_compliant = 0;
        audit.has_warning = 1;
        return audit;
    }
    return nn_feature_schema_audit(domain->scorer_feature_schema,
                                   domain->scorer_feature_count,
                                   domain->leakage_policy);
}

int nn_symbolic_domain_is_valid(const NNSymbolicDomain *domain)
{
    if (domain == NULL || !nn_text_is_present(domain->name) ||
        !nn_text_is_present(domain->description) ||
        !nn_text_is_present(domain->symbols) ||
        !nn_text_is_present(domain->axioms) ||
        !nn_text_is_present(domain->grammar) ||
        domain->max_rule_count == 0 || domain->max_action_count == 0 ||
        domain->rule_feature_count > NN_RULE_MAX_FEATURES ||
        !nn_leakage_policy_is_valid(domain->leakage_policy) ||
        domain->enumerate_rules == NULL || domain->check_certificate == NULL) {
        return 0;
    }
    if (domain->rule_feature_count == 0) {
        if (domain->rule_feature_schema != NULL || domain->rule_feature_schema_count != 0) {
            return 0;
        }
    } else if (domain->rule_feature_schema_count != domain->rule_feature_count ||
               !nn_feature_schema_is_valid(domain->rule_feature_schema,
                                           domain->rule_feature_schema_count)) {
        return 0;
    }
    if (domain->scorer_feature_schema == NULL) {
        if (domain->scorer_feature_count != 0) {
            return 0;
        }
    } else if (!nn_feature_schema_is_valid(domain->scorer_feature_schema,
                                           domain->scorer_feature_count)) {
        return 0;
    }
    if (domain->scorer_feature_schema != NULL &&
        domain->leakage_policy == NN_LEAKAGE_GATE_ALLOWED) {
        return 0;
    }

    NNFeatureAudit rule_audit = nn_symbolic_domain_rule_feature_audit(domain);
    if (!rule_audit.scorer_input_is_policy_compliant) {
        return 0;
    }
    if (domain->scorer_feature_schema != NULL) {
        NNFeatureAudit scorer_audit = nn_symbolic_domain_scorer_feature_audit(domain);
        if (!scorer_audit.scorer_input_is_policy_compliant) {
            return 0;
        }
    }
    return 1;
}

size_t nn_symbolic_domain_enumerate_rules(const NNSymbolicDomain *domain,
                                          NNRule *rules,
                                          size_t capacity)
{
    if (!nn_symbolic_domain_is_valid(domain)) {
        return 0;
    }
    return domain->enumerate_rules(domain, rules, capacity);
}

int nn_symbolic_domain_encode_rule(const NNSymbolicDomain *domain,
                                   const NNRule *rule,
                                   nn_real *features,
                                   size_t capacity,
                                   size_t *feature_count_out)
{
    if (!nn_symbolic_domain_is_valid(domain) || rule == NULL || features == NULL) {
        return 0;
    }
    if (domain->encode_rule != NULL) {
        size_t encoded_count = 0;
        int ok = domain->encode_rule(domain, rule, features, capacity, &encoded_count);
        if (!ok || encoded_count != domain->rule_feature_count) {
            return 0;
        }
        if (feature_count_out != NULL) {
            *feature_count_out = encoded_count;
        }
        return 1;
    }
    if (capacity < rule->structural_feature_count) {
        return 0;
    }
    if (rule->structural_feature_count != domain->rule_feature_count) {
        return 0;
    }
    if (rule->structural_feature_count != 0) {
        memcpy(features,
               rule->structural_features,
               rule->structural_feature_count * sizeof(*features));
    }
    if (feature_count_out != NULL) {
        *feature_count_out = rule->structural_feature_count;
    }
    return 1;
}

int nn_symbolic_domain_check_certificate(const NNSymbolicDomain *domain,
                                         NNRule *rule,
                                         NNCertificate *certificate_out)
{
    if (!nn_symbolic_domain_is_valid(domain) || rule == NULL) {
        return 0;
    }
    NNCertificate certificate;
    nn_certificate_clear(&certificate);
    int ok = domain->check_certificate(domain, rule, &certificate);
    if (!ok) {
        return 0;
    }
    rule->certificate = certificate;
    rule->accepted = certificate.status == NN_CERTIFICATE_CERTIFIED;
    if (certificate_out != NULL) {
        *certificate_out = certificate;
    }
    return 1;
}

int nn_symbolic_domain_rule_rejects(const NNSymbolicDomain *domain,
                                    const NNRule *rule,
                                    const void *state,
                                    size_t action,
                                    int *rejects)
{
    if (!nn_symbolic_domain_is_valid(domain) || rule == NULL || rejects == NULL ||
        domain->rule_rejects == NULL) {
        return 0;
    }
    return domain->rule_rejects(domain, rule, state, action, rejects);
}

int nn_symbolic_domain_verify_action(const NNSymbolicDomain *domain,
                                     const void *state,
                                     size_t action,
                                     NNVerifyResult *result)
{
    if (!nn_symbolic_domain_is_valid(domain) || result == NULL ||
        domain->verify_action == NULL) {
        return 0;
    }
    return domain->verify_action(domain, state, action, result);
}

int nn_rule_coverage_init(NNRuleCoverage *coverage,
                          size_t rule_count,
                          size_t event_count)
{
    if (coverage == NULL) {
        return 0;
    }
    memset(coverage, 0, sizeof(*coverage));
    coverage->rule_count = rule_count;
    coverage->event_count = event_count;
    coverage->word_count = (event_count + 63u) / 64u;
    if (rule_count == 0 || event_count == 0 || coverage->word_count == 0) {
        return 1;
    }
    if (coverage->word_count > SIZE_MAX / rule_count ||
        rule_count * coverage->word_count > SIZE_MAX / sizeof(*coverage->bits)) {
        memset(coverage, 0, sizeof(*coverage));
        return 0;
    }
    coverage->bits = calloc(rule_count * coverage->word_count, sizeof(*coverage->bits));
    if (coverage->bits == NULL) {
        memset(coverage, 0, sizeof(*coverage));
        return 0;
    }
    return 1;
}

void nn_rule_coverage_free(NNRuleCoverage *coverage)
{
    if (coverage == NULL) {
        return;
    }
    free(coverage->bits);
    memset(coverage, 0, sizeof(*coverage));
}

void nn_rule_coverage_clear(NNRuleCoverage *coverage)
{
    if (coverage == NULL || coverage->bits == NULL) {
        return;
    }
    memset(coverage->bits,
           0,
           coverage->rule_count * coverage->word_count * sizeof(*coverage->bits));
}

int nn_rule_coverage_set(NNRuleCoverage *coverage, size_t rule_index, size_t event_index)
{
    if (coverage == NULL || coverage->bits == NULL ||
        rule_index >= coverage->rule_count || event_index >= coverage->event_count) {
        return 0;
    }
    size_t word = event_index / 64u;
    uint64_t mask = UINT64_C(1) << (event_index % 64u);
    coverage->bits[rule_index * coverage->word_count + word] |= mask;
    return 1;
}

int nn_rule_coverage_get(const NNRuleCoverage *coverage, size_t rule_index, size_t event_index)
{
    if (coverage == NULL || coverage->bits == NULL ||
        rule_index >= coverage->rule_count || event_index >= coverage->event_count) {
        return 0;
    }
    size_t word = event_index / 64u;
    uint64_t mask = UINT64_C(1) << (event_index % 64u);
    return (coverage->bits[rule_index * coverage->word_count + word] & mask) != 0;
}

size_t nn_rule_coverage_count_covered(const uint64_t *covered, size_t word_count)
{
    if (covered == NULL) {
        return 0;
    }
    size_t count = 0;
    for (size_t word = 0; word < word_count; ++word) {
        count += nn_popcount_u64(covered[word]);
    }
    return count;
}

size_t nn_rule_coverage_gain(const NNRuleCoverage *coverage,
                             size_t rule_index,
                             const uint64_t *covered)
{
    if (coverage == NULL || coverage->bits == NULL || covered == NULL ||
        rule_index >= coverage->rule_count) {
        return 0;
    }
    const uint64_t *rule_bits = &coverage->bits[rule_index * coverage->word_count];
    size_t gain = 0;
    for (size_t word = 0; word < coverage->word_count; ++word) {
        gain += nn_popcount_u64(rule_bits[word] & ~covered[word]);
    }
    return gain;
}

int nn_rule_coverage_add_rule_to_residual(const NNRuleCoverage *coverage,
                                          size_t rule_index,
                                          uint64_t *covered)
{
    if (coverage == NULL || coverage->bits == NULL || covered == NULL ||
        rule_index >= coverage->rule_count) {
        return 0;
    }
    const uint64_t *rule_bits = &coverage->bits[rule_index * coverage->word_count];
    for (size_t word = 0; word < coverage->word_count; ++word) {
        covered[word] |= rule_bits[word];
    }
    return 1;
}

void nn_residual_summary_clear(NNResidualSummary *summary)
{
    if (summary == NULL) {
        return;
    }
    memset(summary, 0, sizeof(*summary));
}

void nn_residual_summary_from_coverage(NNResidualSummary *summary,
                                       size_t accepted_count,
                                       size_t max_rules,
                                       const NNRuleCoverage *coverage,
                                       const uint64_t *covered)
{
    if (summary == NULL) {
        return;
    }
    nn_residual_summary_clear(summary);
    summary->accepted_count = accepted_count;
    summary->max_rules = max_rules;
    if (coverage != NULL) {
        summary->event_count = coverage->event_count;
        summary->covered_events =
            nn_rule_coverage_count_covered(covered, coverage->word_count);
    }
}

size_t nn_residual_summary_write_features(const NNResidualSummary *summary,
                                          nn_real *features,
                                          size_t capacity)
{
    if (summary == NULL || features == NULL || capacity < 4u) {
        return 0;
    }
    double max_rules = summary->max_rules == 0 ? 1.0 : (double)summary->max_rules;
    double residual_events = summary->event_count <= summary->covered_events
                                 ? 0.0
                                 : (double)(summary->event_count - summary->covered_events);
    double residual_fraction = summary->event_count == 0
                                   ? 0.0
                                   : residual_events / (double)summary->event_count;
    features[0] = (nn_real)((double)summary->accepted_count / max_rules);
    features[1] = (nn_real)residual_fraction;
    features[2] = (nn_real)((double)summary->covered_events /
                            (double)(summary->event_count == 0 ? 1u : summary->event_count));
    features[3] = (nn_real)summary->event_count;
    return 4u;
}

static uint64_t nn_symbolic_mix_u64(uint64_t value)
{
    value ^= value >> 33;
    value *= UINT64_C(0xff51afd7ed558ccd);
    value ^= value >> 33;
    value *= UINT64_C(0xc4ceb9fe1a85ec53);
    value ^= value >> 33;
    return value;
}

static int nn_append_closed_loop_sample(nn_real *inputs,
                                        nn_real *targets,
                                        size_t capacity,
                                        size_t feature_count,
                                        size_t *sample_count,
                                        size_t rule_index,
                                        const size_t *accepted_indices,
                                        size_t accepted_count,
                                        const NNRuleCoverage *coverage,
                                        const uint64_t *covered,
                                        NNResidualRuleFeatureFn encode_features,
                                        void *context,
                                        nn_real target)
{
    if (*sample_count >= capacity) {
        return 0;
    }
    if (!encode_features(rule_index,
                         accepted_indices,
                         accepted_count,
                         coverage,
                         covered,
                         context,
                         &inputs[*sample_count * feature_count],
                         feature_count)) {
        return 0;
    }
    targets[*sample_count] = target;
    ++*sample_count;
    return 1;
}

static void nn_compact_accepted_indices(size_t *accepted_indices,
                                        size_t *accepted_count,
                                        size_t candidate_index,
                                        NNResidualRuleKeepAcceptedFn keep_accepted,
                                        void *context)
{
    if (accepted_indices == NULL || accepted_count == NULL || keep_accepted == NULL) {
        return;
    }

    size_t write = 0;
    for (size_t read = 0; read < *accepted_count; ++read) {
        size_t accepted_index = accepted_indices[read];
        if (!keep_accepted(accepted_index, candidate_index, context)) {
            continue;
        }
        accepted_indices[write++] = accepted_index;
    }
    *accepted_count = write;
}

static int nn_closed_loop_rule_policy_config_is_valid(const NNClosedLoopRulePolicyConfig *config)
{
    if (config == NULL || config->feature_count == 0 || config->max_rules == 0 ||
        config->max_train_rules_per_step == 0 ||
        config->feature_schema_count != config->feature_count ||
        !nn_leakage_policy_is_valid(config->leakage_policy) ||
        !nn_feature_schema_is_valid(config->feature_schema, config->feature_schema_count)) {
        return 0;
    }
    NNFeatureAudit audit =
        nn_feature_schema_audit(config->feature_schema,
                                config->feature_schema_count,
                                config->leakage_policy);
    return audit.scorer_input_is_policy_compliant;
}

void nn_closed_loop_rule_policy_config_defaults(NNClosedLoopRulePolicyConfig *config,
                                                size_t feature_count)
{
    if (config == NULL) {
        return;
    }
    memset(config, 0, sizeof(*config));
    config->max_rules = 64;
    config->feature_count = feature_count;
    config->max_train_rules_per_step = 4096;
    config->epochs = 120;
    config->batch_size = 256;
    config->hidden1 = 32;
    config->hidden2 = 16;
    config->leakage_policy = NN_LEAKAGE_GATE_ALLOWED;
    config->feature_schema = NULL;
    config->feature_schema_count = 0;
    config->learning_rate = (nn_real)0.02;
    config->seed = 119417u;
    config->verbose = 0;
}

void nn_closed_loop_rule_policy_clear(NNClosedLoopRulePolicy *policy)
{
    if (policy == NULL) {
        return;
    }
    memset(policy, 0, sizeof(*policy));
}

void nn_closed_loop_rule_policy_free(NNClosedLoopRulePolicy *policy)
{
    if (policy == NULL) {
        return;
    }
    nn_free(policy->net);
    nn_closed_loop_rule_policy_clear(policy);
}

int nn_closed_loop_rule_policy_train(const NNRuleCoverage *coverage,
                                     const NNClosedLoopRulePolicyConfig *config,
                                     NNResidualRuleFeatureFn encode_features,
                                     NNResidualRuleCandidateFn candidate_allowed,
                                     NNResidualRuleKeepAcceptedFn keep_accepted,
                                     void *context,
                                     NNClosedLoopRulePolicy *policy)
{
    if (policy == NULL || coverage == NULL || config == NULL || encode_features == NULL ||
        candidate_allowed == NULL) {
        return 0;
    }
    nn_closed_loop_rule_policy_clear(policy);
    if (coverage->rule_count == 0 || coverage->event_count == 0 ||
        coverage->word_count == 0 ||
        !nn_closed_loop_rule_policy_config_is_valid(config)) {
        return 0;
    }

    size_t eligible_count = 0;
    for (size_t i = 0; i < coverage->rule_count; ++i) {
        if (candidate_allowed(i, NULL, 0, context)) {
            ++eligible_count;
        }
    }
    if (eligible_count == 0) {
        return 0;
    }

    size_t max_steps = config->max_rules < eligible_count ? config->max_rules : eligible_count;
    size_t per_step_limit = eligible_count < config->max_train_rules_per_step
                                ? eligible_count
                                : config->max_train_rules_per_step;
    if (max_steps == 0 || per_step_limit == 0 ||
        max_steps > SIZE_MAX / per_step_limit) {
        return 0;
    }
    size_t capacity = max_steps * per_step_limit;
    if (capacity > SIZE_MAX / config->feature_count ||
        capacity * config->feature_count > SIZE_MAX / sizeof(nn_real)) {
        return 0;
    }

    nn_real *inputs = calloc(capacity * config->feature_count, sizeof(*inputs));
    nn_real *targets = calloc(capacity, sizeof(*targets));
    uint8_t *selected = calloc(coverage->rule_count, sizeof(*selected));
    uint64_t *covered = calloc(coverage->word_count, sizeof(*covered));
    size_t *gains = calloc(coverage->rule_count, sizeof(*gains));
    size_t *accepted_indices = calloc(max_steps, sizeof(*accepted_indices));
    if (inputs == NULL || targets == NULL || selected == NULL || covered == NULL ||
        gains == NULL || accepted_indices == NULL) {
        free(inputs);
        free(targets);
        free(selected);
        free(covered);
        free(gains);
        free(accepted_indices);
        return 0;
    }

    size_t accepted_count = 0;
    size_t sample_count = 0;
    for (size_t step = 0; step < max_steps; ++step) {
        size_t best_index = (size_t)-1;
        size_t best_gain = 0;
        memset(gains, 0, coverage->rule_count * sizeof(*gains));

        for (size_t i = 0; i < coverage->rule_count; ++i) {
            if (selected[i] ||
                !candidate_allowed(i, accepted_indices, accepted_count, context)) {
                continue;
            }
            size_t gain = nn_rule_coverage_gain(coverage, i, covered);
            gains[i] = gain;
            if (gain > best_gain) {
                best_gain = gain;
                best_index = i;
            }
        }
        if (best_index == (size_t)-1 || best_gain == 0) {
            break;
        }

        size_t kept = 0;
        if (!nn_append_closed_loop_sample(inputs,
                                          targets,
                                          capacity,
                                          config->feature_count,
                                          &sample_count,
                                          best_index,
                                          accepted_indices,
                                          accepted_count,
                                          coverage,
                                          covered,
                                          encode_features,
                                          context,
                                          (nn_real)1)) {
            break;
        }
        ++kept;

        for (size_t i = 0; i < coverage->rule_count && kept < per_step_limit; ++i) {
            if (i == best_index || gains[i] == 0 || selected[i] ||
                !candidate_allowed(i, accepted_indices, accepted_count, context)) {
                continue;
            }
            nn_real target = (nn_real)((double)gains[i] / (double)best_gain);
            if (!nn_append_closed_loop_sample(inputs,
                                              targets,
                                              capacity,
                                              config->feature_count,
                                              &sample_count,
                                              i,
                                              accepted_indices,
                                              accepted_count,
                                              coverage,
                                              covered,
                                              encode_features,
                                              context,
                                              target)) {
                break;
            }
            ++kept;
        }

        for (size_t i = 0; i < coverage->rule_count && kept < per_step_limit; ++i) {
            if (i == best_index || gains[i] != 0 || selected[i] ||
                !candidate_allowed(i, accepted_indices, accepted_count, context)) {
                continue;
            }
            uint64_t hash = nn_symbolic_mix_u64((uint64_t)i ^
                                                ((uint64_t)step << 32) ^
                                                (uint64_t)config->seed);
            if ((hash % 17u) != 0u && kept + 1u < per_step_limit) {
                continue;
            }
            if (!nn_append_closed_loop_sample(inputs,
                                              targets,
                                              capacity,
                                              config->feature_count,
                                              &sample_count,
                                              i,
                                              accepted_indices,
                                              accepted_count,
                                              coverage,
                                              covered,
                                              encode_features,
                                              context,
                                              (nn_real)0)) {
                break;
            }
            ++kept;
        }

        selected[best_index] = 1;
        nn_compact_accepted_indices(accepted_indices,
                                    &accepted_count,
                                    best_index,
                                    keep_accepted,
                                    context);
        if (accepted_count < max_steps) {
            accepted_indices[accepted_count++] = best_index;
        }
        nn_rule_coverage_add_rule_to_residual(coverage, best_index, covered);
    }

    if (sample_count == 0) {
        free(inputs);
        free(targets);
        free(selected);
        free(covered);
        free(gains);
        free(accepted_indices);
        return 0;
    }

    size_t layers[4];
    layers[0] = config->feature_count;
    layers[1] = config->hidden1 == 0 ? 1u : config->hidden1;
    size_t layer_count = 0;
    if (config->hidden2 == 0) {
        layers[2] = 1u;
        layer_count = 3u;
    } else {
        layers[2] = config->hidden2;
        layers[3] = 1u;
        layer_count = 4u;
    }

    NNConfig nn_config;
    nn_config.layer_sizes = layers;
    nn_config.layer_count = layer_count;
    nn_config.seed = config->seed;
    nn_config.hidden_activation = NN_ACTIVATION_RELU;
    nn_config.output_activation = NN_ACTIVATION_LINEAR;
    nn_config.loss = NN_LOSS_MSE;

    NeuralNet *net = nn_create_config(&nn_config);
    if (net == NULL) {
        free(inputs);
        free(targets);
        free(selected);
        free(covered);
        free(gains);
        free(accepted_indices);
        return 0;
    }

    nn_real loss = nn_train_batched(net,
                                    inputs,
                                    targets,
                                    sample_count,
                                    config->batch_size == 0 ? sample_count : config->batch_size,
                                    config->learning_rate,
                                    config->epochs,
                                    config->verbose);
    if (isnan((double)loss)) {
        nn_free(net);
        free(inputs);
        free(targets);
        free(selected);
        free(covered);
        free(gains);
        free(accepted_indices);
        return 0;
    }

    policy->net = net;
    policy->loss = (double)loss;
    policy->sample_count = sample_count;

    free(inputs);
    free(targets);
    free(selected);
    free(covered);
    free(gains);
    free(accepted_indices);
    return 1;
}

int nn_closed_loop_rule_policy_select(const NNRuleCoverage *coverage,
                                      const NNClosedLoopRulePolicyConfig *config,
                                      NNResidualRuleFeatureFn encode_features,
                                      NNResidualRuleCandidateFn candidate_allowed,
                                      NNResidualRuleKeepAcceptedFn keep_accepted,
                                      void *context,
                                      NNClosedLoopRulePolicy *policy,
                                      nn_real *scores,
                                      size_t *accepted_indices,
                                      size_t accepted_capacity,
                                      size_t *accepted_count_out,
                                      size_t *proposal_calls_out)
{
    if (coverage == NULL || config == NULL || encode_features == NULL ||
        candidate_allowed == NULL || policy == NULL || policy->net == NULL ||
        accepted_indices == NULL || accepted_capacity == 0 ||
        !nn_closed_loop_rule_policy_config_is_valid(config)) {
        return 0;
    }
    if (accepted_count_out != NULL) {
        *accepted_count_out = 0;
    }
    if (proposal_calls_out != NULL) {
        *proposal_calls_out = 0;
    }

    uint8_t *selected = calloc(coverage->rule_count, sizeof(*selected));
    uint64_t *covered = calloc(coverage->word_count, sizeof(*covered));
    nn_real *features = calloc(config->feature_count, sizeof(*features));
    if (selected == NULL || covered == NULL || features == NULL) {
        free(selected);
        free(covered);
        free(features);
        return 0;
    }

    size_t accepted_count = 0;
    size_t proposal_calls = 0;
    size_t max_calls = config->max_rules < accepted_capacity ? config->max_rules : accepted_capacity;
    nn_real output[1];

    for (size_t call = 0; call < max_calls; ++call) {
        size_t best_index = (size_t)-1;
        nn_real best_score = (nn_real)-INFINITY;

        for (size_t i = 0; i < coverage->rule_count; ++i) {
            if (selected[i] ||
                !candidate_allowed(i, accepted_indices, accepted_count, context)) {
                continue;
            }
            if (!encode_features(i,
                                 accepted_indices,
                                 accepted_count,
                                 coverage,
                                 covered,
                                 context,
                                 features,
                                 config->feature_count)) {
                free(selected);
                free(covered);
                free(features);
                return 0;
            }
            nn_forward(policy->net, features, output);
            if (scores != NULL) {
                scores[i] = output[0];
            }
            if (best_index == (size_t)-1 || output[0] > best_score) {
                best_score = output[0];
                best_index = i;
            }
        }
        if (best_index == (size_t)-1) {
            break;
        }

        selected[best_index] = 1;
        ++proposal_calls;
        size_t gain = nn_rule_coverage_gain(coverage, best_index, covered);
        if (gain == 0) {
            continue;
        }

        nn_compact_accepted_indices(accepted_indices,
                                    &accepted_count,
                                    best_index,
                                    keep_accepted,
                                    context);
        if (accepted_count < accepted_capacity) {
            accepted_indices[accepted_count++] = best_index;
        }
        nn_rule_coverage_add_rule_to_residual(coverage, best_index, covered);
    }

    if (accepted_count_out != NULL) {
        *accepted_count_out = accepted_count;
    }
    if (proposal_calls_out != NULL) {
        *proposal_calls_out = proposal_calls;
    }
    free(selected);
    free(covered);
    free(features);
    return 1;
}
