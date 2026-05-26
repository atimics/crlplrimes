#ifndef NN_PROTOCOL_H
#define NN_PROTOCOL_H

#include "nn_core.h"
#include "nn_symbolic.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define NN_PROTOCOL_CANDIDATE_SCHEMA "crlp.protocol.candidate_row.v1"
#define NN_PROTOCOL_VERIFIER_SCHEMA "crlp.protocol.verifier_outcome.v1"
#define NN_PROTOCOL_FEEDBACK_SCHEMA "crlp.protocol.verifier_feedback_signal.v1"
#define NN_PROTOCOL_RANKER_SCHEMA "crlp.protocol.ranker_decision.v1"
#define NN_PROTOCOL_RULE_SCHEMA "crlp.protocol.discovered_rule.v1"
#define NN_PROTOCOL_LIBRARY_SCHEMA "crlp.protocol.library_promotion.v1"
#define NN_PROTOCOL_OPERATOR_SCHEMA "crlp.protocol.operator_promotion.v1"

typedef struct {
    const char *sequence_id;
    size_t index;
    size_t length;
    const char *position;
    const char *previous;
    const char *next;
    const char *composition_context;
} NNProtocolFrame;

typedef struct {
    const char *source_schema;
    const char *source_artifact;
    const char *state_hash;
    const char *parent_hash;
    const char *trace_root;
    const char *authority;
    const char *signature;
    const char *replay_seed;
    const char *failure_stage;
    const char *residual_class;
    const char *domain_payload_schema;
} NNProtocolEvidence;

typedef enum {
    NN_VERIFIER_ACCEPTED = 0,
    NN_VERIFIER_UNSAFE = 1,
    NN_VERIFIER_NO_SUPPORT = 2,
    NN_VERIFIER_SAFE_DUPLICATE = 3,
    NN_VERIFIER_REJECTED = 4,
} NNVerifierOutcomeStatus;

typedef struct {
    uint32_t stable_id;
    const char *domain;
    const char *candidate_kind;
    const nn_real *features;
    size_t feature_count;
    const char *provenance;
    const char *example_text;
    const char *source_schema;
    const char *source_artifact;
    const char *state_hash;
    const NNProtocolFrame *frame;
} NNCandidateRow;

typedef struct {
    NNVerifierOutcomeStatus status;
    double utility;
    size_t support;
    size_t false_rejections;
    double cost;
    const char *reason_code;
    const char *proof_kind;
    const NNProtocolEvidence *evidence;
} NNVerifierOutcome;

typedef struct {
    const char *domain;
    uint32_t stable_id;
    const char *grounding_authority;
    NNVerifierOutcomeStatus outcome_status;
    const char *reason_code;
    const char *residual_class;
    double utility;
    double cost;
    size_t support;
    size_t false_rejections;
    const char *training_signal;
    const char *target_role;
    const char *source_schema;
    const char *source_artifact;
} NNVerifierFeedbackSignal;

typedef struct {
    const char *task_name;
    const char *model_hash_hex;
    size_t candidate_count;
    const uint32_t *candidate_ids;
    const nn_real *scores;
    const nn_real *probabilities;
    size_t selected_index;
    size_t target_index;
    nn_real selected_score_margin;
    nn_real rank_score_epsilon;
    const char *ranking_policy;
} NNRankerDecision;

typedef struct {
    const NNCandidateRow *candidate;
    const NNVerifierOutcome *outcome;
    const char *rule_name;
    uint32_t template_id;
} NNDiscoveredRuleRecord;

typedef struct {
    const char *domain;
    const char *template_name;
    uint32_t template_id;
    size_t promoted_count;
    size_t support_tasks;
    size_t covered_events;
    size_t false_rejections;
    double utility;
    size_t heldout_rows;
    int zero_false_rejections;
    const char *promotion_task_hash;
    const char *heldout_result_hash;
    const char *reason_code;
} NNLibraryPromotionRecord;

typedef struct {
    const char *operator_id;
    const char *operator_name;
    const char *residual_family;
    const char *source_residual_class;
    size_t support_rows;
    size_t support_domains;
    size_t accepted_rows;
    size_t unsafe_rows;
    size_t false_rejections;
    double utility_mean;
    double cost_mean;
    int promoted;
    const char *promotion_policy;
    const char *evidence_hash;
    const char *reason_code;
} NNOperatorPromotionRecord;

typedef struct {
    char schema[64];
    uint32_t stable_id;
    char domain[64];
    char candidate_kind[64];
    size_t feature_count;
    char provenance[128];
    char example_text[256];
    char source_schema[128];
    char source_artifact[256];
    char state_hash[128];
    char frame_sequence_id[128];
    size_t frame_index;
    size_t frame_length;
    char frame_position[64];
    char frame_previous[256];
    char frame_next[256];
    char composition_context[256];
} NNParsedCandidateRow;

typedef struct {
    char schema[64];
    char domain[64];
    uint32_t stable_id;
    NNVerifierOutcomeStatus status;
    double utility;
    size_t support;
    size_t false_rejections;
    double cost;
    char reason_code[128];
    char proof_kind[128];
    char source_schema[128];
    char source_artifact[256];
    char state_hash[128];
    char parent_hash[128];
    char trace_root[128];
    char authority[128];
    char signature[160];
    char replay_seed[128];
    char failure_stage[128];
    char residual_class[128];
    char domain_payload_schema[128];
} NNParsedVerifierOutcome;

const char *nn_protocol_status_name(NNVerifierOutcomeStatus status);
int nn_protocol_status_from_name(const char *name, NNVerifierOutcomeStatus *out_status);
int nn_protocol_status_is_safe(NNVerifierOutcomeStatus status);
int nn_protocol_candidate_validate(const NNCandidateRow *row);
int nn_protocol_candidate_batch_validate(const NNCandidateRow *rows,
                                         size_t count,
                                         size_t expected_feature_count);
int nn_protocol_outcome_validate(const NNVerifierOutcome *outcome);
int nn_protocol_feedback_signal_validate(const NNVerifierFeedbackSignal *signal);
int nn_protocol_ranker_decision_validate(const NNRankerDecision *decision);

int nn_protocol_candidate_from_action(const char *domain,
                                      const char *candidate_kind,
                                      const NNActionCandidate *action,
                                      size_t feature_count,
                                      const char *provenance,
                                      const char *example_text,
                                      NNCandidateRow *out_row);
int nn_protocol_candidate_from_rule(const NNRule *rule, NNCandidateRow *out_row);
int nn_protocol_outcome_from_certificate(const NNCertificate *certificate,
                                         double utility,
                                         double cost,
                                         NNVerifierOutcome *outcome);
int nn_protocol_outcome_safe_duplicate(size_t support,
                                       double utility,
                                       const char *reason_code,
                                       NNVerifierOutcome *outcome);
int nn_protocol_feedback_signal_from_outcome(const char *domain,
                                             uint32_t stable_id,
                                             const char *grounding_authority,
                                             const NNVerifierOutcome *outcome,
                                             const char *training_signal,
                                             const char *target_role,
                                             NNVerifierFeedbackSignal *signal);
int nn_protocol_score_candidates(NeuralNet *net,
                                 const NNCandidateRow *rows,
                                 size_t count,
                                 nn_real *scores,
                                 nn_real *probabilities);

int nn_candidate_row_write_json(FILE *fp, const NNCandidateRow *row);
int nn_candidate_row_parse_jsonl(const char *line, NNParsedCandidateRow *out_row);
int nn_verifier_outcome_write_json(FILE *fp,
                                   const char *domain,
                                   uint32_t stable_id,
                                   const NNVerifierOutcome *outcome);
int nn_verifier_outcome_parse_jsonl(const char *line,
                                    NNParsedVerifierOutcome *outcome);
int nn_verifier_feedback_signal_write_json(FILE *fp,
                                           const NNVerifierFeedbackSignal *signal);
int nn_ranker_decision_write_json(FILE *fp, const NNRankerDecision *decision);
int nn_discovered_rule_write_json(FILE *fp, const NNDiscoveredRuleRecord *record);
int nn_library_promotion_write_json(FILE *fp, const NNLibraryPromotionRecord *record);
int nn_operator_promotion_write_json(FILE *fp, const NNOperatorPromotionRecord *record);

#endif
