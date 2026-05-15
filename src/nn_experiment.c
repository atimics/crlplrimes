#include "nn_experiment.h"

#include <stdint.h>
#include <string.h>

static int text_is_present(const char *text)
{
    return text != NULL && text[0] != '\0';
}

int nn_experiment_spec_is_valid(const NNExperimentSpec *spec)
{
    return spec != NULL && text_is_present(spec->task_name) && text_is_present(spec->role) &&
           text_is_present(spec->state_space) && text_is_present(spec->state_distribution) &&
           text_is_present(spec->action_space) &&
           text_is_present(spec->candidate_enumerator) &&
           text_is_present(spec->observable_features) &&
           text_is_present(spec->leakage_policy) && text_is_present(spec->neural_scorer) &&
           text_is_present(spec->exact_verifier) &&
           text_is_present(spec->verifier_feedback) && text_is_present(spec->rule_grammar) &&
           text_is_present(spec->certificate_checker) &&
           text_is_present(spec->gate_executor) && text_is_present(spec->proposer_inputs) &&
           text_is_present(spec->proposer_targets) &&
           text_is_present(spec->baseline_orderings) &&
           text_is_present(spec->generalization_axis) &&
           text_is_present(spec->trace_schema) && text_is_present(spec->replay_protocol) &&
           text_is_present(spec->rl_protocol);
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

int nn_experiment_write_schema_json(FILE *fp, const NNExperimentSpec *spec)
{
    if (fp == NULL || !nn_experiment_spec_is_valid(spec)) {
        return 0;
    }

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
        !write_json_field(fp, "leakage_policy", spec->leakage_policy, &first) ||
        !write_json_field(fp, "neural_scorer", spec->neural_scorer, &first) ||
        !write_json_field(fp, "exact_verifier", spec->exact_verifier, &first) ||
        !write_json_field(fp, "verifier_feedback", spec->verifier_feedback, &first) ||
        !write_json_field(fp, "rule_grammar", spec->rule_grammar, &first) ||
        !write_json_field(fp, "certificate_checker", spec->certificate_checker, &first) ||
        !write_json_field(fp, "gate_executor", spec->gate_executor, &first) ||
        !write_json_field(fp, "proposer_inputs", spec->proposer_inputs, &first) ||
        !write_json_field(fp, "proposer_targets", spec->proposer_targets, &first) ||
        !write_json_field(fp, "baseline_orderings", spec->baseline_orderings, &first) ||
        !write_json_field(fp, "generalization_axis", spec->generalization_axis, &first) ||
        !write_json_field(fp, "trace_schema", spec->trace_schema, &first) ||
        !write_json_field(fp, "replay_protocol", spec->replay_protocol, &first) ||
        !write_json_field(fp, "rl_protocol", spec->rl_protocol, &first)) {
        return 0;
    }

    return fputs("}\n", fp) >= 0;
}
