#include "nn_experiment.h"

#include <stdint.h>

static int text_is_present(const char *text)
{
    return text != NULL && text[0] != '\0';
}

static int feature_provenance_is_valid(NNFeatureProvenance provenance)
{
    return provenance == NN_FEATURE_STATE_OBSERVATION ||
           provenance == NN_FEATURE_STRUCTURAL_PRIOR ||
           provenance == NN_FEATURE_VERIFIER_DERIVED ||
           provenance == NN_FEATURE_DERIVED_FROM_GATE;
}

int nn_leakage_policy_is_valid(NNLeakagePolicy policy)
{
    return policy == NN_LEAKAGE_FORBIDDEN ||
           policy == NN_LEAKAGE_PRIORS_ALLOWED ||
           policy == NN_LEAKAGE_LEGACY ||
           policy == NN_LEAKAGE_GATE_ALLOWED;
}

int nn_grounding_authority_is_valid(NNGroundingAuthority authority)
{
    return authority == NN_GROUNDING_EXACT_CHECK ||
           authority == NN_GROUNDING_DETERMINISTIC_REPLAY ||
           authority == NN_GROUNDING_CORPUS_PROXY ||
           authority == NN_GROUNDING_HUMAN_REVIEW;
}

int nn_feature_schema_is_valid(const NNFeatureSchema *features, size_t feature_count)
{
    if (feature_count == 0) {
        return features == NULL;
    }
    if (features == NULL) {
        return 0;
    }

    for (size_t i = 0; i < feature_count; ++i) {
        if (!text_is_present(features[i].name) ||
            !feature_provenance_is_valid(features[i].provenance)) {
            return 0;
        }
    }
    return 1;
}

const char *nn_feature_provenance_name(NNFeatureProvenance provenance)
{
    switch (provenance) {
    case NN_FEATURE_STATE_OBSERVATION:
        return "state_observation";
    case NN_FEATURE_STRUCTURAL_PRIOR:
        return "structural_prior";
    case NN_FEATURE_VERIFIER_DERIVED:
        return "verifier_derived";
    case NN_FEATURE_DERIVED_FROM_GATE:
        return "derived_from_gate";
    default:
        return "unknown";
    }
}

const char *nn_leakage_policy_name(NNLeakagePolicy policy)
{
    switch (policy) {
    case NN_LEAKAGE_FORBIDDEN:
        return "forbidden";
    case NN_LEAKAGE_PRIORS_ALLOWED:
        return "priors_allowed";
    case NN_LEAKAGE_LEGACY:
        return "legacy";
    case NN_LEAKAGE_GATE_ALLOWED:
        return "gate_allowed";
    default:
        return "unknown";
    }
}

const char *nn_proposer_mode_name(NNProposerMode mode)
{
    switch (mode) {
    case NN_PROPOSER_SELECTION:
        return "selection";
    case NN_PROPOSER_DISCOVERY:
        return "discovery";
    default:
        return "unknown";
    }
}

const char *nn_grounding_authority_name(NNGroundingAuthority authority)
{
    switch (authority) {
    case NN_GROUNDING_EXACT_CHECK:
        return "exact_check";
    case NN_GROUNDING_DETERMINISTIC_REPLAY:
        return "deterministic_replay";
    case NN_GROUNDING_CORPUS_PROXY:
        return "corpus_proxy";
    case NN_GROUNDING_HUMAN_REVIEW:
        return "human_review";
    default:
        return "unknown";
    }
}

NNFeatureAudit nn_feature_schema_audit(const NNFeatureSchema *features,
                                       size_t feature_count,
                                       NNLeakagePolicy policy)
{
    NNFeatureAudit audit = {0};
    audit.scorer_input_is_policy_compliant = 0;
    audit.has_warning = 1;

    if (!nn_feature_schema_is_valid(features, feature_count) ||
        !nn_leakage_policy_is_valid(policy)) {
        return audit;
    }

    audit.total_features = feature_count;
    for (size_t i = 0; i < feature_count; ++i) {
        const NNFeatureSchema *feature = &features[i];
        int enabled = feature->enabled_by_default != 0;
        if (enabled) {
            ++audit.enabled_features;
        }

        switch (feature->provenance) {
        case NN_FEATURE_STRUCTURAL_PRIOR:
            ++audit.structural_prior_features;
            if (enabled) {
                ++audit.enabled_structural_prior_features;
            }
            break;
        case NN_FEATURE_VERIFIER_DERIVED:
            ++audit.verifier_derived_features;
            if (enabled) {
                ++audit.enabled_verifier_derived_features;
            }
            break;
        case NN_FEATURE_DERIVED_FROM_GATE:
            ++audit.gate_derived_features;
            if (enabled) {
                ++audit.enabled_gate_derived_features;
            }
            break;
        case NN_FEATURE_STATE_OBSERVATION:
        default:
            break;
        }
    }

    switch (policy) {
    case NN_LEAKAGE_FORBIDDEN:
        audit.scorer_input_is_policy_compliant =
            audit.enabled_structural_prior_features == 0 &&
            audit.enabled_verifier_derived_features == 0 &&
            audit.enabled_gate_derived_features == 0;
        break;
    case NN_LEAKAGE_PRIORS_ALLOWED:
        audit.scorer_input_is_policy_compliant =
            audit.enabled_verifier_derived_features == 0 &&
            audit.enabled_gate_derived_features == 0;
        break;
    case NN_LEAKAGE_LEGACY:
        audit.scorer_input_is_policy_compliant =
            audit.enabled_verifier_derived_features == 0;
        break;
    case NN_LEAKAGE_GATE_ALLOWED:
        audit.scorer_input_is_policy_compliant =
            audit.enabled_verifier_derived_features == 0;
        break;
    default:
        audit.scorer_input_is_policy_compliant = 0;
        break;
    }

    audit.has_warning = audit.enabled_structural_prior_features > 0 ||
                        audit.enabled_verifier_derived_features > 0 ||
                        audit.enabled_gate_derived_features > 0 ||
                        !audit.scorer_input_is_policy_compliant;
    return audit;
}

NNFeatureAudit nn_experiment_feature_audit(const NNExperimentSpec *spec)
{
    if (spec == NULL) {
        NNFeatureAudit audit = {0};
        audit.scorer_input_is_policy_compliant = 0;
        audit.has_warning = 1;
        return audit;
    }
    return nn_feature_schema_audit(spec->features,
                                   spec->feature_count,
                                   spec->leakage_policy);
}

static int experiment_leakage_policy_is_valid(NNLeakagePolicy policy)
{
    return policy == NN_LEAKAGE_FORBIDDEN ||
           policy == NN_LEAKAGE_PRIORS_ALLOWED ||
           policy == NN_LEAKAGE_LEGACY ||
           policy == NN_LEAKAGE_GATE_ALLOWED;
}

int nn_experiment_spec_is_valid(const NNExperimentSpec *spec)
{
    NNFeatureAudit audit = nn_experiment_feature_audit(spec);
    return spec != NULL && text_is_present(spec->task_name) && text_is_present(spec->role) &&
           text_is_present(spec->state_space) && text_is_present(spec->state_distribution) &&
           text_is_present(spec->action_space) &&
           text_is_present(spec->candidate_enumerator) &&
           text_is_present(spec->observable_features) &&
           experiment_leakage_policy_is_valid(spec->leakage_policy) &&
           text_is_present(spec->neural_scorer) &&
           nn_grounding_authority_is_valid(spec->grounding_authority) &&
           text_is_present(spec->grounding_operator) &&
           text_is_present(spec->exact_verifier) &&
           text_is_present(spec->verifier_feedback) &&
           text_is_present(spec->candidate_grammar) &&
           text_is_present(spec->outcome_row_schema) &&
           text_is_present(spec->claim_scope) &&
           text_is_present(spec->known_non_claims) && text_is_present(spec->rule_grammar) &&
           text_is_present(spec->certificate_checker) &&
           text_is_present(spec->gate_executor) && text_is_present(spec->proposer_inputs) &&
           text_is_present(spec->proposer_targets) &&
           nn_proposer_mode_name(spec->proposer_mode)[0] != 'u' &&
           text_is_present(spec->baseline_orderings) &&
           text_is_present(spec->generalization_axis) &&
           text_is_present(spec->trace_schema) && text_is_present(spec->replay_protocol) &&
           text_is_present(spec->rl_protocol) && spec->feature_count > 0 &&
           nn_feature_schema_is_valid(spec->features, spec->feature_count) &&
           audit.scorer_input_is_policy_compliant;
}

static int write_json_escaped(FILE *fp, const char *text)
{
    if (fp == NULL || text == NULL || fputc('"', fp) == EOF) {
        return 0;
    }

    for (const unsigned char *p = (const unsigned char *)text; *p != '\0'; ++p) {
        switch (*p) {
        case '"':
            if (fputs("\\\"", fp) == EOF) {
                return 0;
            }
            break;
        case '\\':
            if (fputs("\\\\", fp) == EOF) {
                return 0;
            }
            break;
        case '\b':
            if (fputs("\\b", fp) == EOF) {
                return 0;
            }
            break;
        case '\f':
            if (fputs("\\f", fp) == EOF) {
                return 0;
            }
            break;
        case '\n':
            if (fputs("\\n", fp) == EOF) {
                return 0;
            }
            break;
        case '\r':
            if (fputs("\\r", fp) == EOF) {
                return 0;
            }
            break;
        case '\t':
            if (fputs("\\t", fp) == EOF) {
                return 0;
            }
            break;
        default:
            if (*p < 0x20u) {
                if (fprintf(fp, "\\u%04x", (unsigned int)*p) < 0) {
                    return 0;
                }
            } else if (fputc(*p, fp) == EOF) {
                return 0;
            }
            break;
        }
    }

    return fputc('"', fp) != EOF;
}

static int write_json_field(FILE *fp, const char *key, const char *value, int *first)
{
    if (fp == NULL || key == NULL || value == NULL || first == NULL) {
        return 0;
    }

    if (!*first && fputc(',', fp) == EOF) {
        return 0;
    }
    *first = 0;
    return write_json_escaped(fp, key) && fputc(':', fp) != EOF &&
           write_json_escaped(fp, value);
}

static int write_json_size_field(FILE *fp, const char *key, size_t value, int *first)
{
    if (fp == NULL || key == NULL || first == NULL) {
        return 0;
    }

    if (!*first && fputc(',', fp) == EOF) {
        return 0;
    }
    *first = 0;
    return write_json_escaped(fp, key) && fprintf(fp, ":%zu", value) >= 0;
}

static int write_json_bool_field(FILE *fp, const char *key, int value, int *first)
{
    if (fp == NULL || key == NULL || first == NULL) {
        return 0;
    }

    if (!*first && fputc(',', fp) == EOF) {
        return 0;
    }
    *first = 0;
    return write_json_escaped(fp, key) && fputc(':', fp) != EOF &&
           fputs(value ? "true" : "false", fp) >= 0;
}

static int write_feature_audit_json(FILE *fp, const NNFeatureAudit *audit)
{
    if (fp == NULL || audit == NULL || fputs(",\"feature_audit\":{", fp) == EOF) {
        return 0;
    }

    int first = 1;
    if (!write_json_size_field(fp, "total_features", audit->total_features, &first) ||
        !write_json_size_field(fp, "enabled_features", audit->enabled_features, &first) ||
        !write_json_size_field(fp,
                               "structural_prior_features",
                               audit->structural_prior_features,
                               &first) ||
        !write_json_size_field(fp,
                               "verifier_derived_features",
                               audit->verifier_derived_features,
                               &first) ||
        !write_json_size_field(fp,
                               "gate_derived_features",
                               audit->gate_derived_features,
                               &first) ||
        !write_json_size_field(fp,
                               "enabled_structural_prior_features",
                               audit->enabled_structural_prior_features,
                               &first) ||
        !write_json_size_field(fp,
                               "enabled_verifier_derived_features",
                               audit->enabled_verifier_derived_features,
                               &first) ||
        !write_json_size_field(fp,
                               "enabled_gate_derived_features",
                               audit->enabled_gate_derived_features,
                               &first) ||
        !write_json_bool_field(fp,
                               "scorer_input_is_policy_compliant",
                               audit->scorer_input_is_policy_compliant,
                               &first) ||
        !write_json_bool_field(fp, "has_warning", audit->has_warning, &first)) {
        return 0;
    }

    return fputc('}', fp) != EOF;
}

static int write_feature_schema_json(FILE *fp, const NNExperimentSpec *spec)
{
    if (fp == NULL || spec == NULL || spec->features == NULL ||
        fputs(",\"feature_schema\":[", fp) == EOF) {
        return 0;
    }

    for (size_t i = 0; i < spec->feature_count; ++i) {
        const NNFeatureSchema *feature = &spec->features[i];
        int first = 1;
        if (i > 0 && fputc(',', fp) == EOF) {
            return 0;
        }
        if (fputc('{', fp) == EOF ||
            !write_json_field(fp, "name", feature->name, &first) ||
            !write_json_field(fp,
                              "provenance",
                              nn_feature_provenance_name(feature->provenance),
                              &first) ||
            !write_json_bool_field(fp, "enabled_by_default", feature->enabled_by_default, &first) ||
            fputc('}', fp) == EOF) {
            return 0;
        }
    }

    return fputc(']', fp) != EOF;
}

int nn_experiment_write_schema_json(FILE *fp, const NNExperimentSpec *spec)
{
    if (fp == NULL || !nn_experiment_spec_is_valid(spec)) {
        return 0;
    }
    NNFeatureAudit audit = nn_experiment_feature_audit(spec);

    int first = 1;
    if (fputc('{', fp) == EOF) {
        return 0;
    }
    if (!write_json_field(fp, "schema", NN_EXPERIMENT_SCHEMA, &first)) {
        return 0;
    }
    if (fprintf(fp, ",\"schema_version\":%u", NN_EXPERIMENT_SCHEMA_VERSION) < 0) {
        return 0;
    }
    if (!write_json_field(fp, "task", spec->task_name, &first) ||
        !write_json_field(fp, "role", spec->role, &first) ||
        !write_json_field(fp, "state_space", spec->state_space, &first) ||
        !write_json_field(fp, "state_distribution", spec->state_distribution, &first) ||
        !write_json_field(fp, "action_space", spec->action_space, &first) ||
        !write_json_field(fp, "candidate_enumerator", spec->candidate_enumerator, &first) ||
        !write_json_field(fp, "observable_features", spec->observable_features, &first) ||
        !write_json_field(fp,
                          "leakage_policy",
                          nn_leakage_policy_name(spec->leakage_policy),
                          &first) ||
        !write_json_field(fp, "neural_scorer", spec->neural_scorer, &first) ||
        !write_json_field(fp,
                          "grounding_authority",
                          nn_grounding_authority_name(spec->grounding_authority),
                          &first) ||
        !write_json_field(fp, "grounding_operator", spec->grounding_operator, &first) ||
        !write_json_field(fp, "exact_verifier", spec->exact_verifier, &first) ||
        !write_json_field(fp, "verifier_feedback", spec->verifier_feedback, &first) ||
        !write_json_field(fp, "candidate_grammar", spec->candidate_grammar, &first) ||
        !write_json_field(fp, "outcome_row_schema", spec->outcome_row_schema, &first) ||
        !write_json_field(fp, "claim_scope", spec->claim_scope, &first) ||
        !write_json_field(fp, "known_non_claims", spec->known_non_claims, &first) ||
        !write_json_field(fp, "rule_grammar", spec->rule_grammar, &first) ||
        !write_json_field(fp, "certificate_checker", spec->certificate_checker, &first) ||
        !write_json_field(fp, "gate_executor", spec->gate_executor, &first) ||
        !write_json_field(fp, "proposer_inputs", spec->proposer_inputs, &first) ||
        !write_json_field(fp, "proposer_targets", spec->proposer_targets, &first) ||
        !write_json_field(fp, "proposer_mode", nn_proposer_mode_name(spec->proposer_mode), &first) ||
        !write_json_field(fp, "baseline_orderings", spec->baseline_orderings, &first) ||
        !write_json_field(fp, "generalization_axis", spec->generalization_axis, &first) ||
        !write_json_field(fp, "trace_schema", spec->trace_schema, &first) ||
        !write_json_field(fp, "replay_protocol", spec->replay_protocol, &first) ||
        !write_json_field(fp, "rl_protocol", spec->rl_protocol, &first)) {
        return 0;
    }
    if (!write_feature_audit_json(fp, &audit) || !write_feature_schema_json(fp, spec)) {
        return 0;
    }

    return fputs("}\n", fp) >= 0;
}
