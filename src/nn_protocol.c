#include "nn_protocol.h"
#include "nn_text.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static const char *nn_protocol_text_or_empty(const char *text)
{
    return text != NULL ? text : "";
}

static const char *nn_protocol_rule_kind_name(NNRuleKind kind)
{
    switch (kind) {
    case NN_RULE_KIND_REJECT_ACTION:
        return "reject_action_rule";
    case NN_RULE_KIND_TRANSFORM:
        return "transform_rule";
    case NN_RULE_KIND_THRESHOLD:
        return "threshold_rule";
    case NN_RULE_KIND_ARC_TRANSFORM:
        return "arc_transform_rule";
    case NN_RULE_KIND_CUSTOM:
        return "custom_rule";
    }
    return "unknown_rule";
}

static void nn_protocol_json_string(FILE *fp, const char *value)
{
    (void)nn_write_json_string(fp, value);
}

static int nn_protocol_has_text(const char *text)
{
    return text != NULL && text[0] != '\0';
}

static void nn_protocol_write_optional_string(FILE *fp, const char *key, const char *value)
{
    if (!nn_protocol_has_text(value)) {
        return;
    }
    fprintf(fp, ",\"%s\":", key);
    nn_protocol_json_string(fp, value);
}

static int nn_protocol_json_get_size(const char *line, const char *key, size_t *out_value)
{
    uint64_t parsed = 0;
    if (!nn_trace_json_get_u64(line, key, &parsed)) {
        return 0;
    }
    *out_value = (size_t)parsed;
    return 1;
}

static int nn_protocol_json_get_u32(const char *line, const char *key, uint32_t *out_value)
{
    uint64_t parsed = 0;
    if (!nn_trace_json_get_u64(line, key, &parsed) || parsed > UINT32_MAX) {
        return 0;
    }
    *out_value = (uint32_t)parsed;
    return 1;
}

static int nn_protocol_json_get_double(const char *line, const char *key, double *out_value)
{
    nn_real parsed = (nn_real)0;
    if (!nn_trace_json_get_real(line, key, &parsed)) {
        return 0;
    }
    *out_value = (double)parsed;
    return 1;
}

const char *nn_protocol_status_name(NNVerifierOutcomeStatus status)
{
    switch (status) {
    case NN_VERIFIER_ACCEPTED:
        return "accepted";
    case NN_VERIFIER_UNSAFE:
        return "unsafe";
    case NN_VERIFIER_NO_SUPPORT:
        return "no_support";
    case NN_VERIFIER_SAFE_DUPLICATE:
        return "safe_duplicate";
    case NN_VERIFIER_REJECTED:
        return "rejected";
    }
    return "rejected";
}

int nn_protocol_status_from_name(const char *name, NNVerifierOutcomeStatus *out_status)
{
    if (name == NULL || out_status == NULL) {
        return 0;
    }
    if (strcmp(name, "accepted") == 0) {
        *out_status = NN_VERIFIER_ACCEPTED;
        return 1;
    }
    if (strcmp(name, "unsafe") == 0) {
        *out_status = NN_VERIFIER_UNSAFE;
        return 1;
    }
    if (strcmp(name, "no_support") == 0) {
        *out_status = NN_VERIFIER_NO_SUPPORT;
        return 1;
    }
    if (strcmp(name, "safe_duplicate") == 0) {
        *out_status = NN_VERIFIER_SAFE_DUPLICATE;
        return 1;
    }
    if (strcmp(name, "rejected") == 0) {
        *out_status = NN_VERIFIER_REJECTED;
        return 1;
    }
    return 0;
}

static int nn_protocol_status_is_valid(NNVerifierOutcomeStatus status)
{
    switch (status) {
    case NN_VERIFIER_ACCEPTED:
    case NN_VERIFIER_UNSAFE:
    case NN_VERIFIER_NO_SUPPORT:
    case NN_VERIFIER_SAFE_DUPLICATE:
    case NN_VERIFIER_REJECTED:
        return 1;
    }
    return 0;
}

int nn_protocol_status_is_safe(NNVerifierOutcomeStatus status)
{
    return status == NN_VERIFIER_ACCEPTED ||
           status == NN_VERIFIER_NO_SUPPORT ||
           status == NN_VERIFIER_SAFE_DUPLICATE;
}

int nn_protocol_candidate_validate(const NNCandidateRow *row)
{
    if (row == NULL || row->domain == NULL || row->domain[0] == '\0' ||
        row->candidate_kind == NULL || row->candidate_kind[0] == '\0' ||
        row->features == NULL || row->feature_count == 0) {
        return 0;
    }
    if (row->frame != NULL &&
        row->frame->length > 0u &&
        row->frame->index >= row->frame->length) {
        return 0;
    }
    for (size_t i = 0; i < row->feature_count; ++i) {
        if (!isfinite((double)row->features[i])) {
            return 0;
        }
    }
    return 1;
}

int nn_protocol_candidate_batch_validate(const NNCandidateRow *rows,
                                         size_t count,
                                         size_t expected_feature_count)
{
    if (rows == NULL || count == 0 || expected_feature_count == 0) {
        return 0;
    }
    for (size_t i = 0; i < count; ++i) {
        if (!nn_protocol_candidate_validate(&rows[i]) ||
            rows[i].feature_count != expected_feature_count) {
            return 0;
        }
        for (size_t j = i + 1; j < count; ++j) {
            if (rows[i].stable_id == rows[j].stable_id) {
                return 0;
            }
        }
    }
    return 1;
}

int nn_protocol_outcome_validate(const NNVerifierOutcome *outcome)
{
    if (outcome == NULL || !nn_protocol_status_is_valid(outcome->status) ||
        !isfinite(outcome->utility) ||
        !isfinite(outcome->cost) || outcome->cost < 0.0) {
        return 0;
    }
    switch (outcome->status) {
    case NN_VERIFIER_ACCEPTED:
    case NN_VERIFIER_NO_SUPPORT:
    case NN_VERIFIER_SAFE_DUPLICATE:
        if (outcome->false_rejections != 0u) {
            return 0;
        }
        break;
    case NN_VERIFIER_UNSAFE:
    case NN_VERIFIER_REJECTED:
        break;
    default:
        return 0;
    }
    return 1;
}

int nn_protocol_feedback_signal_validate(const NNVerifierFeedbackSignal *signal)
{
    if (signal == NULL ||
        !nn_protocol_has_text(signal->domain) ||
        !nn_protocol_has_text(signal->grounding_authority) ||
        !nn_protocol_status_is_valid(signal->outcome_status) ||
        !nn_protocol_has_text(signal->reason_code) ||
        !nn_protocol_has_text(signal->training_signal) ||
        !nn_protocol_has_text(signal->target_role) ||
        !isfinite(signal->utility) ||
        !isfinite(signal->cost) ||
        signal->cost < 0.0) {
        return 0;
    }
    if (nn_protocol_status_is_safe(signal->outcome_status) &&
        signal->false_rejections != 0u) {
        return 0;
    }
    return 1;
}

int nn_protocol_ranker_decision_validate(const NNRankerDecision *decision)
{
    if (decision == NULL || decision->candidate_count == 0 ||
        decision->candidate_ids == NULL || decision->scores == NULL ||
        decision->selected_index >= decision->candidate_count ||
        decision->target_index >= decision->candidate_count ||
        !isfinite((double)decision->selected_score_margin) ||
        !isfinite((double)decision->rank_score_epsilon) ||
        decision->rank_score_epsilon < (nn_real)0) {
        return 0;
    }
    double probability_sum = 0.0;
    for (size_t i = 0; i < decision->candidate_count; ++i) {
        if (!isfinite((double)decision->scores[i])) {
            return 0;
        }
        for (size_t j = i + 1; j < decision->candidate_count; ++j) {
            if (decision->candidate_ids[i] == decision->candidate_ids[j]) {
                return 0;
            }
        }
        if (decision->probabilities != NULL) {
            double probability = (double)decision->probabilities[i];
            if (!isfinite(probability) || probability < 0.0 || probability > 1.0) {
                return 0;
            }
            probability_sum += probability;
        }
    }
    if (decision->probabilities != NULL && fabs(probability_sum - 1.0) > 1.0e-3) {
        return 0;
    }
    return 1;
}

int nn_protocol_candidate_from_action(const char *domain,
                                      const char *candidate_kind,
                                      const NNActionCandidate *action,
                                      size_t feature_count,
                                      const char *provenance,
                                      const char *example_text,
                                      NNCandidateRow *out_row)
{
    if (action == NULL || out_row == NULL || feature_count == 0 ||
        action->features == NULL) {
        return 0;
    }
    out_row->stable_id = action->action_id;
    out_row->domain = nn_protocol_text_or_empty(domain);
    out_row->candidate_kind = nn_protocol_text_or_empty(candidate_kind);
    out_row->features = action->features;
    out_row->feature_count = feature_count;
    out_row->provenance = nn_protocol_text_or_empty(provenance);
    out_row->example_text = nn_protocol_text_or_empty(example_text);
    out_row->source_schema = "";
    out_row->source_artifact = "";
    out_row->state_hash = "";
    out_row->frame = NULL;
    return 1;
}

int nn_protocol_candidate_from_rule(const NNRule *rule, NNCandidateRow *out_row)
{
    if (rule == NULL || out_row == NULL || rule->structural_feature_count == 0) {
        return 0;
    }
    out_row->stable_id = rule->template_id;
    out_row->domain = rule->domain;
    out_row->candidate_kind = nn_protocol_rule_kind_name(rule->kind);
    out_row->features = rule->structural_features;
    out_row->feature_count = rule->structural_feature_count;
    out_row->provenance = "certificate_rule";
    out_row->example_text = rule->name;
    out_row->source_schema = "";
    out_row->source_artifact = "";
    out_row->state_hash = "";
    out_row->frame = NULL;
    return 1;
}

int nn_protocol_outcome_from_certificate(const NNCertificate *certificate,
                                         double utility,
                                         double cost,
                                         NNVerifierOutcome *outcome)
{
    if (certificate == NULL || outcome == NULL) {
        return 0;
    }
    outcome->utility = utility;
    outcome->support = certificate->support;
    outcome->false_rejections = certificate->false_rejections;
    outcome->cost = cost;
    outcome->reason_code = certificate->reason;
    outcome->proof_kind = certificate->proof_kind;
    outcome->evidence = NULL;
    if (certificate->status == NN_CERTIFICATE_CERTIFIED) {
        outcome->status = certificate->support > 0u ?
            NN_VERIFIER_ACCEPTED :
            NN_VERIFIER_NO_SUPPORT;
    } else if (certificate->status == NN_CERTIFICATE_REJECTED) {
        outcome->status = certificate->false_rejections > 0u ?
            NN_VERIFIER_UNSAFE :
            NN_VERIFIER_REJECTED;
    } else {
        outcome->status = NN_VERIFIER_REJECTED;
    }
    return 1;
}

int nn_protocol_outcome_safe_duplicate(size_t support,
                                       double utility,
                                       const char *reason_code,
                                       NNVerifierOutcome *outcome)
{
    if (outcome == NULL) {
        return 0;
    }
    outcome->status = NN_VERIFIER_SAFE_DUPLICATE;
    outcome->utility = utility;
    outcome->support = support;
    outcome->false_rejections = 0u;
    outcome->cost = 0.0;
    outcome->reason_code = nn_protocol_text_or_empty(reason_code);
    outcome->proof_kind = "safe_duplicate";
    outcome->evidence = NULL;
    return 1;
}

int nn_protocol_feedback_signal_from_outcome(const char *domain,
                                             uint32_t stable_id,
                                             const char *grounding_authority,
                                             const NNVerifierOutcome *outcome,
                                             const char *training_signal,
                                             const char *target_role,
                                             NNVerifierFeedbackSignal *signal)
{
    const NNProtocolEvidence *evidence;
    if (signal == NULL || !nn_protocol_outcome_validate(outcome)) {
        return 0;
    }
    evidence = outcome->evidence;
    signal->domain = nn_protocol_text_or_empty(domain);
    signal->stable_id = stable_id;
    signal->grounding_authority = nn_protocol_text_or_empty(grounding_authority);
    signal->outcome_status = outcome->status;
    signal->reason_code = nn_protocol_text_or_empty(outcome->reason_code);
    signal->residual_class = evidence != NULL ?
        nn_protocol_text_or_empty(evidence->residual_class) :
        "";
    signal->utility = outcome->utility;
    signal->cost = outcome->cost;
    signal->support = outcome->support;
    signal->false_rejections = outcome->false_rejections;
    signal->training_signal = nn_protocol_text_or_empty(training_signal);
    signal->target_role = nn_protocol_text_or_empty(target_role);
    signal->source_schema = evidence != NULL ?
        nn_protocol_text_or_empty(evidence->source_schema) :
        NN_PROTOCOL_VERIFIER_SCHEMA;
    signal->source_artifact = evidence != NULL ?
        nn_protocol_text_or_empty(evidence->source_artifact) :
        "";
    return nn_protocol_feedback_signal_validate(signal);
}

int nn_protocol_score_candidates(NeuralNet *net,
                                 const NNCandidateRow *rows,
                                 size_t count,
                                 nn_real *scores,
                                 nn_real *probabilities)
{
    if (net == NULL || rows == NULL || count == 0 || scores == NULL) {
        return 0;
    }
    const size_t feature_count = rows[0].feature_count;
    if (feature_count == 0 ||
        !nn_protocol_candidate_batch_validate(rows, count, feature_count) ||
        feature_count != nn_input_size(net)) {
        return 0;
    }
    nn_real *inputs = (nn_real *)calloc(count * feature_count, sizeof(*inputs));
    if (inputs == NULL) {
        return 0;
    }
    for (size_t i = 0; i < count; ++i) {
        if (rows[i].features == NULL || rows[i].feature_count != feature_count) {
            free(inputs);
            return 0;
        }
        memcpy(&inputs[i * feature_count],
               rows[i].features,
               feature_count * sizeof(*inputs));
    }
    int ok = nn_score_action_rows(net, inputs, count, scores, probabilities);
    free(inputs);
    return ok;
}

int nn_candidate_row_write_json(FILE *fp, const NNCandidateRow *row)
{
    if (fp == NULL || !nn_protocol_candidate_validate(row)) {
        return 0;
    }
    fputs("{\"schema\":", fp);
    nn_protocol_json_string(fp, NN_PROTOCOL_CANDIDATE_SCHEMA);
    fprintf(fp, ",\"stable_id\":%u", row->stable_id);
    fputs(",\"domain\":", fp);
    nn_protocol_json_string(fp, row->domain);
    fputs(",\"candidate_kind\":", fp);
    nn_protocol_json_string(fp, row->candidate_kind);
    fprintf(fp, ",\"feature_count\":%zu", row->feature_count);
    fputs(",\"provenance\":", fp);
    nn_protocol_json_string(fp, row->provenance);
    fputs(",\"example_text\":", fp);
    nn_protocol_json_string(fp, row->example_text);
    nn_protocol_write_optional_string(fp, "source_schema", row->source_schema);
    nn_protocol_write_optional_string(fp, "source_artifact", row->source_artifact);
    nn_protocol_write_optional_string(fp, "state_hash", row->state_hash);
    if (row->frame != NULL) {
        nn_protocol_write_optional_string(fp,
                                          "frame_sequence_id",
                                          row->frame->sequence_id);
        if (row->frame->length > 0u) {
            fprintf(fp,
                    ",\"frame_index\":%zu,\"frame_length\":%zu",
                    row->frame->index,
                    row->frame->length);
        }
        nn_protocol_write_optional_string(fp, "frame_position", row->frame->position);
        nn_protocol_write_optional_string(fp, "frame_previous", row->frame->previous);
        nn_protocol_write_optional_string(fp, "frame_next", row->frame->next);
        nn_protocol_write_optional_string(fp,
                                          "composition_context",
                                          row->frame->composition_context);
    }
    fputs("}\n", fp);
    return ferror(fp) == 0;
}

int nn_candidate_row_parse_jsonl(const char *line, NNParsedCandidateRow *out_row)
{
    if (line == NULL || out_row == NULL) {
        return 0;
    }
    memset(out_row, 0, sizeof(*out_row));
    if (!nn_trace_json_get_string(line, "schema", out_row->schema, sizeof(out_row->schema)) ||
        strcmp(out_row->schema, NN_PROTOCOL_CANDIDATE_SCHEMA) != 0 ||
        !nn_protocol_json_get_u32(line, "stable_id", &out_row->stable_id) ||
        !nn_trace_json_get_string(line, "domain", out_row->domain, sizeof(out_row->domain)) ||
        !nn_trace_json_get_string(line,
                                  "candidate_kind",
                                  out_row->candidate_kind,
                                  sizeof(out_row->candidate_kind)) ||
        !nn_protocol_json_get_size(line, "feature_count", &out_row->feature_count) ||
        !nn_trace_json_get_string(line,
                                  "provenance",
                                  out_row->provenance,
                                  sizeof(out_row->provenance)) ||
        !nn_trace_json_get_string(line,
                                  "example_text",
                                  out_row->example_text,
                                  sizeof(out_row->example_text))) {
        return 0;
    }
    (void)nn_trace_json_get_string(line,
                                   "source_schema",
                                   out_row->source_schema,
                                   sizeof(out_row->source_schema));
    (void)nn_trace_json_get_string(line,
                                   "source_artifact",
                                   out_row->source_artifact,
                                   sizeof(out_row->source_artifact));
    (void)nn_trace_json_get_string(line,
                                   "state_hash",
                                   out_row->state_hash,
                                   sizeof(out_row->state_hash));
    (void)nn_trace_json_get_string(line,
                                   "frame_sequence_id",
                                   out_row->frame_sequence_id,
                                   sizeof(out_row->frame_sequence_id));
    (void)nn_protocol_json_get_size(line, "frame_index", &out_row->frame_index);
    (void)nn_protocol_json_get_size(line, "frame_length", &out_row->frame_length);
    (void)nn_trace_json_get_string(line,
                                   "frame_position",
                                   out_row->frame_position,
                                   sizeof(out_row->frame_position));
    (void)nn_trace_json_get_string(line,
                                   "frame_previous",
                                   out_row->frame_previous,
                                   sizeof(out_row->frame_previous));
    (void)nn_trace_json_get_string(line,
                                   "frame_next",
                                   out_row->frame_next,
                                   sizeof(out_row->frame_next));
    (void)nn_trace_json_get_string(line,
                                   "composition_context",
                                   out_row->composition_context,
                                   sizeof(out_row->composition_context));
    return 1;
}

int nn_verifier_outcome_write_json(FILE *fp,
                                   const char *domain,
                                   uint32_t stable_id,
                                   const NNVerifierOutcome *outcome)
{
    if (fp == NULL || !nn_protocol_outcome_validate(outcome)) {
        return 0;
    }
    fputs("{\"schema\":", fp);
    nn_protocol_json_string(fp, NN_PROTOCOL_VERIFIER_SCHEMA);
    fputs(",\"domain\":", fp);
    nn_protocol_json_string(fp, domain);
    fprintf(fp, ",\"stable_id\":%u", stable_id);
    fputs(",\"status\":", fp);
    nn_protocol_json_string(fp, nn_protocol_status_name(outcome->status));
    fprintf(fp,
            ",\"utility\":%.17g,\"support\":%zu,\"false_rejections\":%zu,\"cost\":%.17g",
            outcome->utility,
            outcome->support,
            outcome->false_rejections,
            outcome->cost);
    fputs(",\"reason_code\":", fp);
    nn_protocol_json_string(fp, outcome->reason_code);
    fputs(",\"proof_kind\":", fp);
    nn_protocol_json_string(fp, outcome->proof_kind);
    if (outcome->evidence != NULL) {
        const NNProtocolEvidence *evidence = outcome->evidence;
        nn_protocol_write_optional_string(fp, "source_schema", evidence->source_schema);
        nn_protocol_write_optional_string(fp, "source_artifact", evidence->source_artifact);
        nn_protocol_write_optional_string(fp, "state_hash", evidence->state_hash);
        nn_protocol_write_optional_string(fp, "parent_hash", evidence->parent_hash);
        nn_protocol_write_optional_string(fp, "trace_root", evidence->trace_root);
        nn_protocol_write_optional_string(fp, "authority", evidence->authority);
        nn_protocol_write_optional_string(fp, "signature", evidence->signature);
        nn_protocol_write_optional_string(fp, "replay_seed", evidence->replay_seed);
        nn_protocol_write_optional_string(fp, "failure_stage", evidence->failure_stage);
        nn_protocol_write_optional_string(fp, "residual_class", evidence->residual_class);
        nn_protocol_write_optional_string(fp,
                                          "domain_payload_schema",
                                          evidence->domain_payload_schema);
    }
    fputs("}\n", fp);
    return ferror(fp) == 0;
}

int nn_verifier_outcome_parse_jsonl(const char *line,
                                    NNParsedVerifierOutcome *outcome)
{
    char status[32];
    if (line == NULL || outcome == NULL) {
        return 0;
    }
    memset(outcome, 0, sizeof(*outcome));
    if (!nn_trace_json_get_string(line, "schema", outcome->schema, sizeof(outcome->schema)) ||
        strcmp(outcome->schema, NN_PROTOCOL_VERIFIER_SCHEMA) != 0 ||
        !nn_trace_json_get_string(line, "domain", outcome->domain, sizeof(outcome->domain)) ||
        !nn_protocol_json_get_u32(line, "stable_id", &outcome->stable_id) ||
        !nn_trace_json_get_string(line, "status", status, sizeof(status)) ||
        !nn_protocol_status_from_name(status, &outcome->status) ||
        !nn_protocol_json_get_double(line, "utility", &outcome->utility) ||
        !nn_protocol_json_get_size(line, "support", &outcome->support) ||
        !nn_protocol_json_get_size(line, "false_rejections", &outcome->false_rejections) ||
        !nn_protocol_json_get_double(line, "cost", &outcome->cost) ||
        !nn_trace_json_get_string(line,
                                  "reason_code",
                                  outcome->reason_code,
                                  sizeof(outcome->reason_code)) ||
        !nn_trace_json_get_string(line,
                                  "proof_kind",
                                  outcome->proof_kind,
                                  sizeof(outcome->proof_kind))) {
        return 0;
    }
    (void)nn_trace_json_get_string(line,
                                   "source_schema",
                                   outcome->source_schema,
                                   sizeof(outcome->source_schema));
    (void)nn_trace_json_get_string(line,
                                   "source_artifact",
                                   outcome->source_artifact,
                                   sizeof(outcome->source_artifact));
    (void)nn_trace_json_get_string(line,
                                   "state_hash",
                                   outcome->state_hash,
                                   sizeof(outcome->state_hash));
    (void)nn_trace_json_get_string(line,
                                   "parent_hash",
                                   outcome->parent_hash,
                                   sizeof(outcome->parent_hash));
    (void)nn_trace_json_get_string(line,
                                   "trace_root",
                                   outcome->trace_root,
                                   sizeof(outcome->trace_root));
    (void)nn_trace_json_get_string(line,
                                   "authority",
                                   outcome->authority,
                                   sizeof(outcome->authority));
    (void)nn_trace_json_get_string(line,
                                   "signature",
                                   outcome->signature,
                                   sizeof(outcome->signature));
    (void)nn_trace_json_get_string(line,
                                   "replay_seed",
                                   outcome->replay_seed,
                                   sizeof(outcome->replay_seed));
    (void)nn_trace_json_get_string(line,
                                   "failure_stage",
                                   outcome->failure_stage,
                                   sizeof(outcome->failure_stage));
    (void)nn_trace_json_get_string(line,
                                   "residual_class",
                                   outcome->residual_class,
                                   sizeof(outcome->residual_class));
    (void)nn_trace_json_get_string(line,
                                   "domain_payload_schema",
                                   outcome->domain_payload_schema,
                                   sizeof(outcome->domain_payload_schema));
    return 1;
}

int nn_verifier_feedback_signal_write_json(FILE *fp,
                                           const NNVerifierFeedbackSignal *signal)
{
    if (fp == NULL || !nn_protocol_feedback_signal_validate(signal)) {
        return 0;
    }
    fputs("{\"schema\":", fp);
    nn_protocol_json_string(fp, NN_PROTOCOL_FEEDBACK_SCHEMA);
    fputs(",\"domain\":", fp);
    nn_protocol_json_string(fp, signal->domain);
    fprintf(fp, ",\"stable_id\":%u", signal->stable_id);
    fputs(",\"grounding_authority\":", fp);
    nn_protocol_json_string(fp, signal->grounding_authority);
    fputs(",\"outcome_status\":", fp);
    nn_protocol_json_string(fp, nn_protocol_status_name(signal->outcome_status));
    fputs(",\"reason_code\":", fp);
    nn_protocol_json_string(fp, signal->reason_code);
    nn_protocol_write_optional_string(fp, "residual_class", signal->residual_class);
    fprintf(fp,
            ",\"utility\":%.17g,\"support\":%zu,\"false_rejections\":%zu,\"cost\":%.17g",
            signal->utility,
            signal->support,
            signal->false_rejections,
            signal->cost);
    fputs(",\"training_signal\":", fp);
    nn_protocol_json_string(fp, signal->training_signal);
    fputs(",\"target_role\":", fp);
    nn_protocol_json_string(fp, signal->target_role);
    nn_protocol_write_optional_string(fp, "source_schema", signal->source_schema);
    nn_protocol_write_optional_string(fp, "source_artifact", signal->source_artifact);
    fputs("}\n", fp);
    return ferror(fp) == 0;
}

int nn_ranker_decision_write_json(FILE *fp, const NNRankerDecision *decision)
{
    if (fp == NULL || !nn_protocol_ranker_decision_validate(decision)) {
        return 0;
    }
    fputs("{\"schema\":", fp);
    nn_protocol_json_string(fp, NN_PROTOCOL_RANKER_SCHEMA);
    fputs(",\"task_name\":", fp);
    nn_protocol_json_string(fp, decision->task_name);
    fputs(",\"model_hash_hex\":", fp);
    nn_protocol_json_string(fp, decision->model_hash_hex);
    fprintf(fp,
            ",\"candidate_count\":%zu,\"selected_index\":%zu,\"target_index\":%zu",
            decision->candidate_count,
            decision->selected_index,
            decision->target_index);
    fprintf(fp,
            ",\"selected_action_id\":%u,\"target_action_id\":%u",
            decision->candidate_ids[decision->selected_index],
            decision->candidate_ids[decision->target_index]);
    fprintf(fp,
            ",\"selected_score_margin\":%.9g,\"rank_score_epsilon\":%.9g",
            (double)decision->selected_score_margin,
            (double)decision->rank_score_epsilon);
    fputs(",\"ranking_policy\":", fp);
    nn_protocol_json_string(fp,
                            decision->ranking_policy != NULL ?
                            decision->ranking_policy :
                            NN_RANKING_POLICY);
    fputs(",\"candidate_ids\":[", fp);
    for (size_t i = 0; i < decision->candidate_count; ++i) {
        fprintf(fp, "%s%u", i == 0 ? "" : ",", decision->candidate_ids[i]);
    }
    fputs("],\"scores\":[", fp);
    for (size_t i = 0; i < decision->candidate_count; ++i) {
        fprintf(fp, "%s%.9g", i == 0 ? "" : ",", (double)decision->scores[i]);
    }
    fputc(']', fp);
    if (decision->probabilities != NULL) {
        fputs(",\"probabilities\":[", fp);
        for (size_t i = 0; i < decision->candidate_count; ++i) {
            fprintf(fp,
                    "%s%.9g",
                    i == 0 ? "" : ",",
                    (double)decision->probabilities[i]);
        }
        fputc(']', fp);
    }
    fputs("}\n", fp);
    return ferror(fp) == 0;
}

int nn_discovered_rule_write_json(FILE *fp, const NNDiscoveredRuleRecord *record)
{
    if (fp == NULL || record == NULL || record->candidate == NULL ||
        record->outcome == NULL) {
        return 0;
    }
    fputs("{\"schema\":", fp);
    nn_protocol_json_string(fp, NN_PROTOCOL_RULE_SCHEMA);
    fputs(",\"domain\":", fp);
    nn_protocol_json_string(fp, record->candidate->domain);
    fprintf(fp, ",\"stable_id\":%u", record->candidate->stable_id);
    fputs(",\"candidate_kind\":", fp);
    nn_protocol_json_string(fp, record->candidate->candidate_kind);
    fputs(",\"rule_name\":", fp);
    nn_protocol_json_string(fp, record->rule_name);
    fprintf(fp, ",\"template_id\":%u", record->template_id);
    fputs(",\"status\":", fp);
    nn_protocol_json_string(fp, nn_protocol_status_name(record->outcome->status));
    fprintf(fp,
            ",\"utility\":%.17g,\"support\":%zu,\"false_rejections\":%zu,\"cost\":%.17g",
            record->outcome->utility,
            record->outcome->support,
            record->outcome->false_rejections,
            record->outcome->cost);
    fputs(",\"reason_code\":", fp);
    nn_protocol_json_string(fp, record->outcome->reason_code);
    fputs("}\n", fp);
    return ferror(fp) == 0;
}

int nn_library_promotion_write_json(FILE *fp, const NNLibraryPromotionRecord *record)
{
    if (fp == NULL || record == NULL) {
        return 0;
    }
    fputs("{\"schema\":", fp);
    nn_protocol_json_string(fp, NN_PROTOCOL_LIBRARY_SCHEMA);
    fputs(",\"domain\":", fp);
    nn_protocol_json_string(fp, record->domain);
    fputs(",\"template_name\":", fp);
    nn_protocol_json_string(fp, record->template_name);
    fprintf(fp, ",\"template_id\":%u", record->template_id);
    fprintf(fp,
            ",\"promoted_count\":%zu,\"support_tasks\":%zu,\"covered_events\":%zu",
            record->promoted_count,
            record->support_tasks,
            record->covered_events);
    fprintf(fp,
            ",\"false_rejections\":%zu,\"utility\":%.17g,\"heldout_rows\":%zu",
            record->false_rejections,
            record->utility,
            record->heldout_rows);
    fprintf(fp,
            ",\"zero_false_rejections\":%s",
            record->zero_false_rejections ? "true" : "false");
    fputs(",\"promotion_task_hash\":", fp);
    nn_protocol_json_string(fp, record->promotion_task_hash);
    fputs(",\"heldout_result_hash\":", fp);
    nn_protocol_json_string(fp, record->heldout_result_hash);
    fputs(",\"reason_code\":", fp);
    nn_protocol_json_string(fp, record->reason_code);
    fputs("}\n", fp);
    return ferror(fp) == 0;
}

int nn_operator_promotion_write_json(FILE *fp, const NNOperatorPromotionRecord *record)
{
    if (fp == NULL || record == NULL) {
        return 0;
    }
    fputs("{\"schema\":", fp);
    nn_protocol_json_string(fp, NN_PROTOCOL_OPERATOR_SCHEMA);
    fputs(",\"operator_id\":", fp);
    nn_protocol_json_string(fp, record->operator_id);
    fputs(",\"operator_name\":", fp);
    nn_protocol_json_string(fp, record->operator_name);
    fputs(",\"residual_family\":", fp);
    nn_protocol_json_string(fp, record->residual_family);
    fputs(",\"source_residual_class\":", fp);
    nn_protocol_json_string(fp, record->source_residual_class);
    fprintf(fp,
            ",\"support_rows\":%zu,\"support_domains\":%zu",
            record->support_rows,
            record->support_domains);
    fprintf(fp,
            ",\"accepted_rows\":%zu,\"unsafe_rows\":%zu,\"false_rejections\":%zu",
            record->accepted_rows,
            record->unsafe_rows,
            record->false_rejections);
    fprintf(fp,
            ",\"utility_mean\":%.17g,\"cost_mean\":%.17g",
            record->utility_mean,
            record->cost_mean);
    fprintf(fp, ",\"promoted\":%s", record->promoted ? "true" : "false");
    fputs(",\"promotion_policy\":", fp);
    nn_protocol_json_string(fp, record->promotion_policy);
    fputs(",\"evidence_hash\":", fp);
    nn_protocol_json_string(fp, record->evidence_hash);
    fputs(",\"reason_code\":", fp);
    nn_protocol_json_string(fp, record->reason_code);
    fputs("}\n", fp);
    return ferror(fp) == 0;
}
